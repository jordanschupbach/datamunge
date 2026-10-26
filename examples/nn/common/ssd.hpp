// SSD: VGG-16 SSD300 and the compact digit-scene variant (port of tflow's
// common/nn/ssd.h), default boxes, the multibox loss with per-image 3:1 hard
// negative mining, targets and batched decoding (port of common/ml/ssd.h).
// https://arxiv.org/abs/1512.02325
// https://github.com/pytorch/vision/blob/main/torchvision/models/detection/ssd.py
//
// Full models use conv4_3 with learned L2 rescaling (initial scale 20),
// atrous fc6 (dilation 6), fc7 and four extra blocks. The six grids are
// 38/19/10/5/3/1 for a 300-pixel image, with 4/6/6/6/4/4 boxes per cell.
// The compact option uses a small 64-pixel grayscale backbone and grids
// 16/8/4/2/1/1 (an explicit training-demonstration adaptation). Kernels use
// He-uniform initialization; no pretrained weights are loaded. fc6 uses the
// library's dilated convolution (tflow expands the kernel with zeros, which
// is the same operation).
//
// Background is the last class. Matching (detector::MatchSsd) first greedily
// gives every target a distinct default box, then assigns the remaining boxes
// with IoU >= 0.5. Hard negatives are ranked per image by softmax
// cross-entropy (positives last, ties by lower index as TF TopK); at most
// three per positive are kept. The loss is summed over the batch and divided
// by the total positives (zero for an all-background batch). The selection
// is computed on the host from detached cross-entropies, as TF's TopK
// selection is not differentiated either. Deltas use variances 0.1/0.1/0.2/0.2;
// inference boxes are clipped before class-wise NMS (.45).
#pragma once

#include "common/detection_scenes.hpp"
#include "common/vgg.hpp"

namespace ssd {

namespace nn = datamunge::nn;
using detector::Box;
using detector::DetectionScenes;

inline std::vector<Box> SsdDefaultBoxes(const std::vector<int>& cells, const std::vector<float>& sizes,
                                        const std::vector<float>& steps, const std::vector<int>& per_cell) {
  if (cells.empty() || sizes.size() != cells.size() + 1 || steps.size() != cells.size() || per_cell.size() != cells.size())
    throw std::invalid_argument("invalid SSD default-box specification");
  std::vector<Box> result;
  for (std::size_t l = 0; l < cells.size(); ++l) {
    if (cells[l] < 1 || !(sizes[l] > 0) || !(sizes[l + 1] > 0) || !(steps[l] > 0) || (per_cell[l] != 4 && per_cell[l] != 6))
      throw std::invalid_argument("invalid SSD default-box dimensions");
    const float                         mean = std::sqrt(sizes[l] * sizes[l + 1]);
    std::vector<std::pair<float, float>> shapes = {{sizes[l], sizes[l]}, {mean, mean}};
    for (int ratio = 2; ratio <= (per_cell[l] == 6 ? 3 : 2); ++ratio) {
      const float h = sizes[l] / std::sqrt(float(ratio)), w = sizes[l] * std::sqrt(float(ratio));
      shapes.push_back({h, w});
      shapes.push_back({w, h});
    }
    for (int r = 0; r < cells[l]; ++r)
      for (int c = 0; c < cells[l]; ++c)
        for (const auto& [h, w] : shapes) {
          const float y = (r + .5f) * steps[l], x = (c + .5f) * steps[l];
          result.push_back({y - h / 2, x - w / 2, y + h / 2, x + w / 2});
        }
  }
  return result;
}

/// x / ||x||_channels times a learned per-channel scale "<name>/scale".
inline nn::Tensor SsdNormalize(nn::ParameterStore& p, const std::string& name, const nn::Tensor& x,
                               double initial_scale = 20) {
  const std::int64_t channels = x.shape()[3];
  const auto norm = nn::sqrt(nn::maximum(nn::sum(nn::square(x), {3}, true), nn::scalar(1e-12)));
  return x / norm * p.get(name + "/scale", {channels}, nn::init::constant(initial_scale));
}

struct SsdOutputs {
  std::vector<nn::Tensor> features;
  std::vector<int>        cells, channels, boxes_per_cell;
  nn::Tensor              logits, probabilities, deltas;  // [N, A, classes], [N, A, classes], [N, A, 4]
  int                     anchors = 0;
};

inline SsdOutputs SSD(nn::ParameterStore& p, const nn::Tensor& image, int classes = 21, bool compact = false) {
  if (classes < 2) throw std::invalid_argument("SSD requires foreground and background classes");
  SsdOutputs result;
  result.boxes_per_cell = {4, 6, 6, 6, 4, 4};
  nn::ConvLayerOptions cfg;
  cfg.init       = nn::init::he_uniform();
  cfg.activation = nn::Activation::Relu;
  nn::Tensor h   = image;
  if (compact) {
    result.cells    = {16, 8, 4, 2, 1, 1};
    result.channels = {64, 128, 128, 128, 128, 128};
    cfg.stride      = 2;
    h               = nn::conv2d(p, "backbone/conv1", h, 32, 3, cfg);
    h               = nn::conv2d(p, "backbone/conv2", h, 64, 3, cfg);
    result.features.push_back(SsdNormalize(p, "normalize", h, 10));
    for (int i = 1; i < 6; ++i) {
      cfg.stride = i == 5 ? 1 : 2;
      h          = nn::conv2d(p, "backbone/feature" + std::to_string(i + 1), h, 128, 3, cfg);
      result.features.push_back(h);
    }
  } else {
    result.cells    = {38, 19, 10, 5, 3, 1};
    result.channels = {512, 1024, 512, 256, 256, 256};
    const int  widths[] = {64, 128, 256, 512, 512};
    const auto depths   = vgg::VggBlockDepths(16);
    for (int block = 0; block < 5; ++block) {
      for (int layer = 0; layer < depths[std::size_t(block)]; ++layer)
        h = nn::conv2d(p, "backbone/block" + std::to_string(block + 1) + "/conv" + std::to_string(layer + 1), h,
                       widths[block], 3, cfg);
      if (block == 3) result.features.push_back(SsdNormalize(p, "normalize", h));
      if (block < 4) h = nn::max_pool2d(h, 2, 2, block == 2 ? nn::Padding::Same : nn::Padding::Valid);
    }
    h = nn::max_pool2d(h, 3, 1, nn::Padding::Same);
    auto fc6  = cfg;
    fc6.dilation = 6;
    h         = nn::conv2d(p, "backbone/fc6", h, 1024, 3, fc6);
    h         = nn::conv2d(p, "backbone/fc7", h, 1024, 1, cfg);
    result.features.push_back(h);
    for (int i = 0; i < 4; ++i) {
      const std::string name = "extra" + std::to_string(i + 1);
      const int         hidden = i ? 128 : 256, out = i ? 256 : 512;
      auto              reduce = cfg, spatial = cfg;
      h               = nn::conv2d(p, name + "/reduce", h, hidden, 1, reduce);
      spatial.stride  = i < 2 ? 2 : 1;
      spatial.padding = nn::Padding::Valid;
      if (i < 2) h = nn::pad(h, {{0, 0}, {1, 1}, {1, 1}, {0, 0}});
      h = nn::conv2d(p, name + "/spatial", h, out, 3, spatial);
      result.features.push_back(h);
    }
  }
  nn::ConvLayerOptions linear;
  linear.init = nn::init::he_uniform();
  std::vector<nn::Tensor> scores, offsets;
  for (std::size_t i = 0; i < 6; ++i) {
    const int count = result.cells[i] * result.cells[i] * result.boxes_per_cell[i];
    result.anchors += count;
    const auto name = "head" + std::to_string(i + 1);
    scores.push_back(nn::reshape(nn::conv2d(p, name + "/scores", result.features[i], result.boxes_per_cell[i] * classes, 3, linear),
                                 {-1, count, classes}));
    offsets.push_back(nn::reshape(nn::conv2d(p, name + "/boxes", result.features[i], result.boxes_per_cell[i] * 4, 3, linear),
                                  {-1, count, 4}));
  }
  result.logits        = nn::concat(scores, 1);
  result.probabilities = nn::softmax(result.logits, -1);
  result.deltas        = nn::concat(offsets, 1);
  return result;
}

struct SsdLossOutputs {
  nn::Tensor             loss;
  std::vector<char>      selected_negatives;  // [N * A]
};
/// Multibox loss; labels [N * A] (background = classes - 1), targets [N * A * 4].
inline SsdLossOutputs SsdLoss(const nn::Tensor& logits, const nn::Tensor& deltas, const std::vector<int>& labels,
                              const std::vector<float>& targets, int classes) {
  const std::int64_t n = logits.shape()[0], anchors = logits.shape()[1];
  if (std::int64_t(labels.size()) != n * anchors || std::int64_t(targets.size()) != n * anchors * 4)
    throw std::invalid_argument("SSD target dimensions do not match the predictions");
  std::vector<float> one_hot(labels.size() * std::size_t(classes), 0), mask(labels.size(), 0);
  for (std::size_t k = 0; k < labels.size(); ++k) {
    one_hot[k * std::size_t(classes) + std::size_t(labels[k])] = 1;
    mask[k]                                                   = labels[k] != classes - 1;
  }
  const auto log_probabilities = nn::reshape(nn::log_softmax(nn::reshape(logits, {-1, classes}), -1), {n, anchors, classes});
  const auto cross_entropy     = -nn::sum(nn::from_vector(one_hot, {n, anchors, classes}) * log_probabilities, {2});
  SsdLossOutputs out;
  out.selected_negatives.assign(labels.size(), 0);
  {
    const auto ce = cross_entropy.detach().to_vector<float>();
    for (std::int64_t i = 0; i < n; ++i) {
      std::vector<int> order(static_cast<std::size_t>(anchors));
      std::iota(order.begin(), order.end(), 0);
      auto value = [&](int a) { return mask[std::size_t(i * anchors + a)] ? -1e9f : ce[std::size_t(i * anchors + a)]; };
      std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return value(a) > value(b); });
      std::int64_t positives = 0;
      for (std::int64_t a = 0; a < anchors; ++a) positives += mask[std::size_t(i * anchors + a)] > 0;
      const std::int64_t wanted = std::min(3 * positives, anchors - positives);
      for (std::int64_t rank = 0; rank < wanted; ++rank) {
        const auto k = std::size_t(i * anchors + order[std::size_t(rank)]);
        if (!mask[k]) out.selected_negatives[k] = 1;
      }
    }
  }
  std::vector<float> weight(labels.size());
  for (std::size_t k = 0; k < labels.size(); ++k) weight[k] = mask[k] + float(out.selected_negatives[k]);
  const auto m              = nn::from_vector(mask, {n, anchors});
  const auto classification = nn::sum(cross_entropy * nn::from_vector(weight, {n, anchors}));
  const auto regression     = nn::sum(nn::sum(detector::SmoothL1(deltas - nn::from_vector(targets, {n, anchors, 4})), {2}) * m);
  out.loss = (classification + regression) / nn::maximum(nn::sum(m), nn::scalar(1.0));
  return out;
}

struct SsdTargets {
  std::vector<int>   labels;
  std::vector<float> targets;
};
inline SsdTargets MakeSsdTargets(const std::vector<Box>& anchors, const DetectionScenes& scenes, const std::vector<int>& rows,
                                 int background) {
  const std::size_t count = anchors.size();
  SsdTargets        t;
  t.labels.assign(rows.size() * count, background);
  t.targets.assign(rows.size() * count * 4, 0);
  for (std::size_t i = 0; i < rows.size(); ++i) {
    const auto& truth    = scenes.boxes[std::size_t(rows[i])];
    const auto  matching = detector::MatchSsd(anchors, truth);
    for (std::size_t a = 0; a < count; ++a)
      if (matching[a] >= 0) {
        t.labels[i * count + a] = scenes.labels[std::size_t(rows[i])][std::size_t(matching[a])];
        const auto delta        = detector::EncodeBox(anchors[a], truth[std::size_t(matching[a])]);
        for (std::size_t j = 0; j < 4; ++j) t.targets[(i * count + a) * 4 + j] = delta[j];
      }
  }
  return t;
}

/// Top 200 default boxes above .05 per image and foreground class, decoded and clipped; class-wise NMS .45.
template <typename Forward>
std::vector<detector::Detection> DetectSsd(const Forward& forward, const std::vector<Box>& anchors,
                                           const DetectionScenes& scenes, const std::vector<int>& rows, int classes) {
  nn::NoGradGuard                  guard;
  std::vector<detector::Detection> result;
  const std::size_t                A = anchors.size();
  for (std::size_t start = 0; start < rows.size(); start += 32) {
    const std::vector<int> chunk(rows.begin() + std::ptrdiff_t(start), rows.begin() + std::ptrdiff_t(std::min(rows.size(), start + 32)));
    const SsdOutputs       net = forward(detector::SceneImages(scenes, chunk));
    const auto             P = net.probabilities.to_vector<float>(), D = net.deltas.to_vector<float>();
    for (std::size_t i = 0; i < chunk.size(); ++i)
      for (int c = 0; c < classes - 1; ++c) {
        std::vector<std::pair<float, int>> scored;
        for (std::size_t a = 0; a < A; ++a) {
          const float score = P[(i * A + a) * std::size_t(classes) + std::size_t(c)];
          if (score > .05f) scored.push_back({score, int(a)});
        }
        const std::size_t count = std::min<std::size_t>(200, scored.size());
        std::partial_sort(scored.begin(), scored.begin() + std::ptrdiff_t(count), scored.end(), std::greater<std::pair<float, int>>());
        for (std::size_t k = 0; k < count; ++k) {
          const auto a = std::size_t(scored[k].second);
          const auto b = detector::ClipBox(detector::DecodeBox(anchors[a], D.data() + (i * A + a) * 4), float(scenes.height),
                                           float(scenes.width));
          if (detector::BoxArea(b) > 0) result.push_back({chunk[i], c, scored[k].first, b});
        }
      }
  }
  return detector::ClassNms(result, .45f);
}

}  // namespace ssd
