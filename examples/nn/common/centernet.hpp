// CenterNet (Objects as Points): a ResNet encoder, three learned upsampling
// layers and stride-four heatmap/size/offset heads, with Gaussian targets,
// the penalty-reduced focal loss and peak decoding (port of tflow's
// common/nn/centernet.h and common/ml/centernet.h).
// https://arxiv.org/abs/1904.07850
// https://github.com/xingyizhou/CenterNet/blob/master/src/lib/utils/image.py
//
// Model: the ResNet classifier is omitted. Each 4x4 stride-two transposed
// convolution has BN/ReLU. Three independent 3x3/ReLU -> 1x1 heads predict
// class heatmaps, (height,width) and (dy,dx). Heatmap bias starts at -2.19;
// head kernels use normal stddev .001. No pretrained weights are loaded.
// Defaults: ResNet-18, upsample widths 256/128/64 and hidden head width 64.
// Input dimensions must be divisible by 32. Grayscale is repeated to RGB
// for the shared backbone.
//
// Targets/loss: heatmap centres are floor(pixel centre / 4), with a
// size-adaptive Gaussian using the authors' radius formula and
// sigma=diameter/6. Same-class kernels combine by maximum. Gaussian shoulders
// reduce negative penalties by (1-target)^4; positive/negative prediction
// powers are two. Offset L1 and .1*size L1 apply only at retained centres.
// All terms normalize by retained objects. Colliding centre cells keep the
// larger object and are reported. Decoding takes per-class 3x3 local maxima
// and the best 100 peaks, without box NMS.
#pragma once

#include "common/detection_scenes.hpp"
#include "common/resnet.hpp"

namespace centernet {

namespace nn = datamunge::nn;
using detector::Box;
using detector::DetectionScenes;

inline nn::Tensor CenterHead(nn::ParameterStore& p, const std::string& name, const nn::Tensor& x, std::int64_t hidden,
                             std::int64_t out, double bias = 0) {
  nn::ConvLayerOptions first;
  first.init       = nn::init::normal(.001);
  first.activation = nn::Activation::Relu;
  const auto h     = nn::conv2d(p, name + "/conv", x, hidden, 3, first);
  const auto w     = p.get(name + "/output/W", {1, 1, hidden, out}, nn::init::normal(.001));
  const auto b     = p.get(name + "/output/b", {out}, nn::init::constant(bias));
  return nn::bias_add(nn::conv2d(h, w), b);
}
/// Sigmoid probabilities kept only at 3x3 per-class local maxima.
inline nn::Tensor CenterPeaks(const nn::Tensor& logits) {
  const auto probability = nn::sigmoid(logits);
  return probability * nn::cast(nn::equal(probability, nn::max_pool2d(probability, 3, 1, nn::Padding::Same)), probability.dtype());
}

struct CenterNetConfig {
  int                classes = 80, backbone_depth = 18, width_divisor = 1, head_width = 64;
  std::array<int, 3> upsample_widths = {256, 128, 64};
  bool               grayscale = false;
  bool               update_stats = true;  // training mode: update the BN moving averages
};
struct CenterNetOutputs {
  nn::Tensor features, heatmap, size, offset, peaks;
};
inline CenterNetOutputs CenterNet(nn::ParameterStore& p, nn::ParameterStore& state, const nn::Tensor& image,
                                  const CenterNetConfig& cfg = {}, bool training = false) {
  if (cfg.classes < 1 || cfg.head_width < 1) throw std::invalid_argument("invalid CenterNet class/head widths");
  for (int width : cfg.upsample_widths)
    if (width < 1) throw std::invalid_argument("CenterNet upsample widths must be positive");
  resnet::ResNetConfig backbone;
  backbone.depth              = cfg.backbone_depth;
  backbone.width_divisor      = cfg.width_divisor;
  backbone.include_classifier = false;
  backbone.update_stats       = cfg.update_stats;
  const auto net = resnet::ResNet(p, state, cfg.grayscale ? nn::tile(image, {1, 1, 1, 3}) : image, backbone, training);
  nn::Tensor h   = net.stages[3];
  nn::ConvLayerOptions options;
  options.stride = 2;
  options.bias   = false;
  options.init   = nn::init::he_uniform();
  nn::BatchNormOptions bn;
  bn.momentum     = .9;
  bn.eps          = 1e-5;
  bn.update_stats = cfg.update_stats;
  for (int i = 0; i < 3; ++i) {
    const auto name = "upsample" + std::to_string(i + 1);
    h = nn::relu(nn::batch_norm(p, state, name + "/bn", nn::conv2d_transpose(p, name, h, cfg.upsample_widths[std::size_t(i)], 4, options),
                                training, bn));
  }
  CenterNetOutputs result;
  result.features = h;
  result.heatmap  = CenterHead(p, "heatmap", h, cfg.head_width, cfg.classes, -2.19);
  result.size     = CenterHead(p, "size", h, cfg.head_width, 2);
  result.offset   = CenterHead(p, "offset", h, cfg.head_width, 2);
  result.peaks    = CenterPeaks(result.heatmap);
  return result;
}

// ---- targets, loss and decoding -----------------------------------------------------

inline double CenterGaussianRadius(double h, double w, double overlap = .7) {
  if (!(h > 0 && w > 0 && overlap > 0 && overlap <= 1) || !std::isfinite(h + w))
    throw std::invalid_argument("invalid Gaussian radius dimensions/overlap");
  const double b1 = h + w, c1 = h * w * (1 - overlap) / (1 + overlap), r1 = (b1 + std::sqrt(b1 * b1 - 4 * c1)) / 2;
  const double b2 = 2 * (h + w), c2 = (1 - overlap) * h * w, r2 = (b2 + std::sqrt(b2 * b2 - 16 * c2)) / 2;
  const double b3 = -2 * overlap * (h + w), c3 = (overlap - 1) * h * w,
               r3 = (b3 + std::sqrt(b3 * b3 - 16 * overlap * c3)) / 2;
  return std::min({r1, r2, r3});
}

/// A dense float NHWC host buffer.
struct Grid {
  std::int64_t       n = 0, h = 0, w = 0, c = 0;
  std::vector<float> v;
  Grid() = default;
  Grid(std::int64_t n_, std::int64_t h_, std::int64_t w_, std::int64_t c_)
      : n(n_), h(h_), w(w_), c(c_), v(std::size_t(n_ * h_ * w_ * c_), 0.f) {}
  float&       at(std::int64_t b, std::int64_t y, std::int64_t x, std::int64_t k) { return v[std::size_t(((b * h + y) * w + x) * c + k)]; }
  [[nodiscard]] nn::Tensor tensor() const { return nn::from_vector(v, {n, h, w, c}); }
};

inline void DrawCenterGaussian(Grid& heatmap, int batch, int row, int col, int label, int radius) {
  if (radius < 0 || batch < 0 || batch >= heatmap.n || row < 0 || row >= heatmap.h || col < 0 || col >= heatmap.w ||
      label < 0 || label >= heatmap.c)
    throw std::invalid_argument("invalid Gaussian heatmap/centre");
  const double sigma = (2 * radius + 1) / 6.;
  for (int dy = -radius; dy <= radius; ++dy)
    for (int dx = -radius; dx <= radius; ++dx) {
      const int y = row + dy, x = col + dx;
      if (y < 0 || y >= heatmap.h || x < 0 || x >= heatmap.w) continue;
      auto& value = heatmap.at(batch, y, x, label);
      value       = std::max(value, float(std::exp(-(dy * dy + dx * dx) / (2 * sigma * sigma))));
    }
}

/// Per-element penalty-reduced focal loss of heatmap logits against Gaussian targets.
inline nn::Tensor CenterHeatmapFocal(const nn::Tensor& logits, const nn::Tensor& target) {
  const auto positive = nn::square(nn::sigmoid(-logits)) * nn::softplus(-logits);
  const auto negative = nn::pow(1.0 - target, 4.0) * nn::square(nn::sigmoid(logits)) * nn::softplus(logits);
  return nn::where(nn::equal(target, 1.0), positive, negative);
}

struct CenterTargets {
  Grid heatmap, size, offset, mask;  // mask has one channel
  int  collisions = 0;
};
/// (focal + .1 size L1 + offset L1) / max(retained centres, 1).
inline nn::Tensor CenterLoss(const nn::Tensor& heatmap, const nn::Tensor& size, const nn::Tensor& offset,
                             const nn::Tensor& target_heatmap, const nn::Tensor& target_size,
                             const nn::Tensor& target_offset, const nn::Tensor& mask) {
  const auto count = nn::maximum(nn::sum(mask), nn::scalar(1.0));
  const auto heat  = nn::sum(CenterHeatmapFocal(heatmap, target_heatmap));
  const auto l1s   = nn::sum(nn::abs(size - target_size) * mask);
  const auto l1o   = nn::sum(nn::abs(offset - target_offset) * mask);
  return (heat + l1s * .1 + l1o) / count;
}
inline nn::Tensor CenterLoss(const CenterNetOutputs& net, const CenterTargets& t) {
  return CenterLoss(net.heatmap, net.size, net.offset, t.heatmap.tensor(), t.size.tensor(), t.offset.tensor(), t.mask.tensor());
}

inline CenterTargets MakeCenterTargets(const DetectionScenes& scenes, const std::vector<int>& rows, int classes) {
  if (scenes.height % 32 || scenes.width % 32)
    throw std::invalid_argument("CenterNet images must have dimensions divisible by 32");
  const std::int64_t batch = std::int64_t(rows.size());
  const int          h = scenes.height / 4, w = scenes.width / 4;
  CenterTargets      t{Grid(batch, h, w, classes), Grid(batch, h, w, 2), Grid(batch, h, w, 2), Grid(batch, h, w, 1), 0};
  for (std::int64_t i = 0; i < batch; ++i) {
    std::vector<int>   selected(std::size_t(h * w), -1);
    std::vector<float> area(std::size_t(h * w), 0);
    const auto&        boxes = scenes.boxes[std::size_t(rows[std::size_t(i)])];
    for (std::size_t g = 0; g < boxes.size(); ++g) {
      const auto& b = boxes[g];
      const float a = detector::BoxArea(b);
      if (!(a > 0)) throw std::invalid_argument("CenterNet targets need positive box area");
      const int r = std::clamp(int((b[0] + b[2]) / 8), 0, h - 1), c = std::clamp(int((b[1] + b[3]) / 8), 0, w - 1);
      const auto cell = std::size_t(r * w + c);
      if (selected[cell] >= 0) {
        ++t.collisions;
        if (a <= area[cell]) continue;
      }
      selected[cell] = int(g);
      area[cell]     = a;
    }
    for (int cell = 0; cell < h * w; ++cell)
      if (selected[std::size_t(cell)] >= 0) {
        const int   g = selected[std::size_t(cell)], r = cell / w, c = cell % w;
        const int   label = scenes.labels[std::size_t(rows[std::size_t(i)])][std::size_t(g)];
        const auto& b      = boxes[std::size_t(g)];
        const float height = (b[2] - b[0]) / 4, width = (b[3] - b[1]) / 4, y = (b[0] + b[2]) / 8, x = (b[1] + b[3]) / 8;
        DrawCenterGaussian(t.heatmap, int(i), r, c, label, int(std::floor(CenterGaussianRadius(height, width))));
        t.size.at(i, r, c, 0)   = height;
        t.size.at(i, r, c, 1)   = width;
        t.offset.at(i, r, c, 0) = y - float(r);
        t.offset.at(i, r, c, 1) = x - float(c);
        t.mask.at(i, r, c, 0)   = 1;
      }
  }
  return t;
}

/// Boxes of the best 100 peaks above .01 of each image (peaks/size/offset of `rows`).
inline std::vector<detector::Detection> DecodeCenter(const nn::Tensor& peaks, const nn::Tensor& size,
                                                     const nn::Tensor& offset, const DetectionScenes& scenes,
                                                     const std::vector<int>& rows) {
  const auto P = peaks.to_vector<float>(), S = size.to_vector<float>(), O = offset.to_vector<float>();
  const int  height = int(peaks.shape()[1]), width = int(peaks.shape()[2]), classes = int(peaks.shape()[3]),
            count = height * width * classes;
  std::vector<detector::Detection> detections;
  for (std::size_t i = 0; i < rows.size(); ++i) {
    std::vector<std::pair<float, int>> found;
    for (int k = 0; k < count; ++k) {
      const float v = P[i * std::size_t(count) + std::size_t(k)];
      if (v > .01f) found.push_back({v, k});
    }
    const std::size_t keep = std::min<std::size_t>(100, found.size());
    std::partial_sort(found.begin(), found.begin() + std::ptrdiff_t(keep), found.end(), std::greater<std::pair<float, int>>());
    for (std::size_t k = 0; k < keep; ++k) {
      const int   index = found[k].second, label = index % classes, pixel = index / classes, r = pixel / width,
                c = pixel % width;
      const auto  at = [&](const std::vector<float>& v, int ch) { return v[((i * std::size_t(height) + std::size_t(r)) * std::size_t(width) + std::size_t(c)) * 2 + std::size_t(ch)]; };
      const float y = 4 * (float(r) + at(O, 0)), x = 4 * (float(c) + at(O, 1));
      const float h = 4 * at(S, 0), w = 4 * at(S, 1);
      if (!(h > 0 && w > 0)) continue;
      const Box box = detector::ClipBox({y - h / 2, x - w / 2, y + h / 2, x + w / 2}, float(scenes.height), float(scenes.width));
      if (detector::BoxArea(box) > 0) detections.push_back({rows[i], label, found[k].first, box});
    }
  }
  return detections;
}
/// Inference-mode detections in chunks of 32 scenes.
inline std::vector<detector::Detection> DetectCenter(nn::ParameterStore& p, nn::ParameterStore& state,
                                                     const CenterNetConfig& cfg, const DetectionScenes& scenes,
                                                     const std::vector<int>& rows) {
  nn::NoGradGuard                  guard;
  std::vector<detector::Detection> detections;
  for (std::size_t start = 0; start < rows.size(); start += 32) {
    const std::vector<int> chunk(rows.begin() + std::ptrdiff_t(start), rows.begin() + std::ptrdiff_t(std::min(rows.size(), start + 32)));
    const auto net   = CenterNet(p, state, detector::SceneImages(scenes, chunk), cfg, false);
    const auto found = DecodeCenter(net.peaks, net.size, net.offset, scenes, chunk);
    detections.insert(detections.end(), found.begin(), found.end());
  }
  return detections;
}

}  // namespace centernet
