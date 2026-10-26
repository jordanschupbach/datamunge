// SegNet: VGG16 encoder and index-guided sparse unpooling decoder (port of
// tflow's common/nn/segnet.h).
// https://arxiv.org/abs/1511.00561
// https://github.com/alexgkendall/SegNet-Tutorial
// Thirteen encoder convolutions and mirrored decoder convolutions use
// BN/ReLU, except the final 3x3 class predictor. Store actual pre-pool shapes,
// so odd dimensions return exactly; pooling uses nonoverlapping 2x2/VALID.
// Argmax indices include batch offsets (TF MaxPoolWithArgmax with
// include_batch_in_index): ((n H + y) W + x) C + c, first maximum in scan order.
// Values use ordinary max_pool2d, whose derivative is the library's; discrete
// argmax indices are detached host data.
//
// TF -> datamunge: TF builds the inverse map with ScatterNd and gathers; here
// the inverse map and the winner mask are computed on the host from the
// detached indices and the unpool is a differentiable nn::gather times the
// mask (the same values and derivatives). datamunge has no max-pool-with-argmax
// or unpool op; the indices come from one host pass over the pooled input.
// Full defaults: RGB, 11 classes, widths 64/128/256/512/512. Compact uses widths/8.
#pragma once

#include <cstdint>
#include <datamunge/nn/nn.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace segnet {

namespace nn = datamunge::nn;

struct IndexedPool {
  nn::Tensor                values;   // [N, H/2, W/2, C]
  std::vector<std::int64_t> indices;  // flat input index of every pooled value (row-major over values)
  nn::Shape                 shape;    // pre-pool input shape
};

inline IndexedPool PoolWithIndices(const nn::Tensor& input) {
  if (input.dim() != 4) throw std::invalid_argument("indexed pooling expects NHWC input");
  IndexedPool r;
  r.shape                 = input.shape();
  const std::int64_t n = r.shape[0], h = r.shape[1], w = r.shape[2], c = r.shape[3], ph = h / 2, pw = w / 2;
  if (ph < 1 || pw < 1) throw std::invalid_argument("indexed pooling needs at least a 2x2 grid");
  r.values = nn::max_pool2d(input, 2, 2, nn::Padding::Valid);
  const auto x = input.detach().to(nn::DType::Float64).to_vector<double>();
  r.indices.resize(std::size_t(n * ph * pw * c));
  for (std::int64_t b = 0; b < n; ++b)
    for (std::int64_t y = 0; y < ph; ++y)
      for (std::int64_t xx = 0; xx < pw; ++xx)
        for (std::int64_t k = 0; k < c; ++k) {
          std::int64_t best = -1;
          double       top  = -INFINITY;
          for (int dy = 0; dy < 2; ++dy)
            for (int dx = 0; dx < 2; ++dx) {
              const std::int64_t index = ((b * h + 2 * y + dy) * w + 2 * xx + dx) * c + k;
              if (best < 0 || x[std::size_t(index)] > top) {
                top  = x[std::size_t(index)];
                best = index;
              }
            }
          r.indices[std::size_t(((b * ph + y) * pw + xx) * c + k)] = best;
        }
  return r;
}

/// Sparse unpool of values [N, H/2, W/2, C] to record.shape at the recorded winners (zeros elsewhere).
inline nn::Tensor IndexUnpool(const nn::Tensor& values, const IndexedPool& record) {
  std::int64_t total = 1;
  for (auto d : record.shape) total *= d;
  if (values.numel() != std::int64_t(record.indices.size()))
    throw std::invalid_argument("unpool values do not match the recorded pooling");
  std::vector<std::int64_t> inverse(std::size_t(total), 0);
  std::vector<float>        mask(std::size_t(total), 0.f);
  for (std::size_t slot = 0; slot < record.indices.size(); ++slot) {
    inverse[std::size_t(record.indices[slot])] = std::int64_t(slot);
    mask[std::size_t(record.indices[slot])]    = 1.f;
  }
  const nn::Tensor gathered = nn::gather(nn::reshape(values, {-1}), nn::from_vector(inverse, {total}), 0);
  return nn::reshape(gathered * nn::from_vector(mask, {total}).to(values.dtype()), record.shape);
}

struct SegNetConfig {
  int classes = 11, width_divisor = 1;
};
struct SegNetOutputs {
  nn::Tensor               logits;
  std::vector<IndexedPool> pooling;
  std::vector<nn::Tensor>  unpooled;
};

inline SegNetOutputs SegNet(nn::ParameterStore& p, nn::ParameterStore& state, const nn::Tensor& images,
                            const SegNetConfig& cfg = {}, bool training = false) {
  if (cfg.classes < 2 || cfg.width_divisor < 1 || 64 % cfg.width_divisor)
    throw std::invalid_argument("invalid SegNet configuration");
  SegNetOutputs      result;
  const std::int64_t widths[]  = {64 / cfg.width_divisor, 128 / cfg.width_divisor, 256 / cfg.width_divisor,
                                  512 / cfg.width_divisor, 512 / cfg.width_divisor};
  const int          repeats[] = {2, 2, 3, 3, 3};
  auto conv = [&](const std::string& name, const nn::Tensor& x, std::int64_t out, bool normalized = true) {
    nn::ConvLayerOptions options;  // 3x3 SAME with bias, He uniform
    options.init       = nn::init::he_uniform();
    const nn::Tensor y = nn::conv2d(p, name, x, out, 3, options);
    if (!normalized) return y;
    nn::BatchNormOptions bn;
    bn.momentum = .9;
    bn.eps      = 1e-5;
    return nn::relu(nn::batch_norm(p, state, name + "/bn", y, training, bn));
  };
  nn::Tensor h = images;
  for (int stage = 0; stage < 5; ++stage) {
    for (int block = 0; block < repeats[stage]; ++block)
      h = conv("encoder" + std::to_string(stage) + "/conv" + std::to_string(block), h, widths[stage]);
    result.pooling.push_back(PoolWithIndices(h));
    h = result.pooling.back().values;
  }
  for (int stage = 4; stage >= 0; --stage) {
    h = IndexUnpool(h, result.pooling[std::size_t(stage)]);
    result.unpooled.push_back(h);
    for (int block = 0; block < repeats[stage]; ++block) {
      const bool         last = block == repeats[stage] - 1;
      const std::int64_t out  = last ? (stage ? widths[stage - 1] : cfg.classes) : widths[stage];
      h = conv("decoder" + std::to_string(stage) + "/conv" + std::to_string(block), h, out, !(stage == 0 && last));
    }
  }
  result.logits = h;
  return result;
}

}  // namespace segnet
