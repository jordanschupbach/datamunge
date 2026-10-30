#pragma once
/// @file segnet_models.hpp
/// @brief Tensor indexed2x2 max-pooling, exact sparse unpooling and SegNet encoder/decoder.
#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/ops.hpp"
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Own pooled values,detached winner indices and exact pre-pool shape.
///
/// Indices include batch/channel offsets into the flattened NHWC input.
/// They are Int64 tensors,not a host numerical array.
///
/// @throws std::invalid_argument For invalid geometry,dtypes,indices or configuration.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto record = n::models::SegNetPoolWithIndices(n::ones({2, 5, 7, 2}));
/// @endcode
/// @par References
/// - Badrinarayanan,Kendall and Cipolla(2015), SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation, https://arxiv.org/abs/1511.00561; original implementation https://github.com/alexgkendall/SegNet-Tutorial.
/// @see SegNet()
struct SegNetIndexedPool {
  nn::Tensor values; ///< [N,H/2,W/2,C] ordinary differentiable2x2 VALID max-pool output.
  nn::Tensor
      indices; ///< Same-shape detached Int64 flattened NHWC winners,first row-major maximum on ties.
  nn::Shape shape; ///< Original NHWC geometry,including odd spatial dimensions.
};
/// @brief Pool nonoverlapping2x2 cells and compute first-maximum winner indices using tensors.
///
/// Trailing odd row/column are dropped by VALID pooling,but saved shape lets
/// unpool restore their zero positions. Winner offsets include batch/channel.
/// Tie selection is row-major first among top-left,top-right,bottom-left,bottom-right.
/// Values use the library max-pool gradient. No fitting,randomness or mutation.
///
/// @param input Finite Float32/64 positive NHWC tensor,H/W>=2.
/// @return Differentiable max-pool values,detached Int64 winners and original shape.
/// @throws std::invalid_argument For invalid geometry,dtypes,indices or configuration.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto record = n::models::SegNetPoolWithIndices(n::ones({2, 5, 7, 2}));
/// @endcode
/// @par References
/// - Badrinarayanan,Kendall and Cipolla(2015), SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation, https://arxiv.org/abs/1511.00561; original implementation https://github.com/alexgkendall/SegNet-Tutorial.
/// @see SegNetIndexUnpool()
inline SegNetIndexedPool SegNetPoolWithIndices(const nn::Tensor &input) {
  if (input.dim() != 4 || input.shape()[0] < 1 || input.shape()[1] < 2 ||
      input.shape()[2] < 2 || input.shape()[3] < 1 ||
      (input.dtype() != nn::DType::Float32 && input.dtype() != nn::DType::Float64) ||
      !nn::all(nn::less(nn::abs(input), std::numeric_limits<double>::infinity())).item<bool>())
    throw std::invalid_argument("invalid indexed pooling input");
  const auto &s = input.shape();
  const auto N = s[0], H = s[1], W = s[2], C = s[3], PH = H / 2, PW = W / 2;
  const auto cells =
                 nn::transpose(nn::slice(input.detach(), {0, 0, 0, 0}, {N, PH * 2, PW * 2, C})
                                   .reshape({N, PH, 2, PW, 2, C}),
                               {0, 1, 3, 5, 2, 4})
                     .reshape({N, PH, PW, C, 4}),
             winner = nn::argmax(cells, 4);
  const auto type = nn::DType::Int64;
  const auto row_delta =
                 nn::gather(nn::from_vector(std::vector<std::int64_t>{0, 0, 1, 1}), winner),
             col_delta =
                 nn::gather(nn::from_vector(std::vector<std::int64_t>{0, 1, 0, 1}), winner);
  const auto batch = nn::arange(0, double(N), 1, type).reshape({N, 1, 1, 1}),
             row = nn::arange(0, double(PH), 1, type).reshape({1, PH, 1, 1}) *
                       nn::full({}, 2., type) +
                   row_delta,
             col = nn::arange(0, double(PW), 1, type).reshape({1, 1, PW, 1}) *
                       nn::full({}, 2., type) +
                   col_delta,
             channel = nn::arange(0, double(C), 1, type).reshape({1, 1, 1, C}),
             indices = ((batch * nn::full({}, double(H), type) + row) *
                            nn::full({}, double(W), type) +
                        col) *
                           nn::full({}, double(C), type) +
                       channel;
  return {nn::max_pool2d(input, 2, 2, nn::Padding::Valid), indices, s};
}
/// @brief Scatter pooled values to recorded winner positions and fill every other position with zero.
///
/// Tensor segment-sum performs sparse scattering. Saved odd trailing edges
/// are zero,including for negative values. No fitting,random draws or mutation.
/// Manually constructed records must have unique indices; duplicates reject.
/// record.values supplies no numerical input to unpool.
///
/// @param values Float32/64[N,H/2,W/2,C] values,matching the saved pooling geometry.
/// @param record Original positive NHWC shape and aligned unique,in-range Int64 winner indices.
/// @return Same-dtype sparse NHWC output with gradients to values only.
/// @throws std::invalid_argument For invalid geometry,dtypes,indices or configuration.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto record = n::models::SegNetPoolWithIndices(n::ones({2, 5, 7, 2}));
/// const auto restored = n::models::SegNetIndexUnpool(record.values, record);
/// @endcode
/// @par References
/// - Badrinarayanan,Kendall and Cipolla(2015), SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation, https://arxiv.org/abs/1511.00561; original implementation https://github.com/alexgkendall/SegNet-Tutorial.
/// @see SegNetPoolWithIndices()
inline nn::Tensor SegNetIndexUnpool(const nn::Tensor &values,
                                    const SegNetIndexedPool &record) {
  const auto &s = record.shape;
  if (s.size() != 4 || s[0] < 1 || s[1] < 2 || s[2] < 2 || s[3] < 1 ||
      values.shape() != nn::Shape{s[0], s[1] / 2, s[2] / 2, s[3]} ||
      record.indices.shape() != values.shape() || record.indices.dtype() != nn::DType::Int64 ||
      (values.dtype() != nn::DType::Float32 && values.dtype() != nn::DType::Float64))
    throw std::invalid_argument("invalid unpool record geometry/dtype");
  const auto total = nn::shape_numel(s);
  const auto indices = record.indices.reshape({-1});
  if (!nn::all(
           nn::logical_and(nn::greater_equal(indices, 0.), nn::less(indices, double(total))))
           .item<bool>())
    throw std::invalid_argument("invalid unpool winner range");
  const auto sorted = nn::gather(indices, nn::argsort(indices));
  if (indices.numel() > 1 &&
      nn::any(nn::equal(nn::slice_axis(sorted, 0, 0, indices.numel() - 1),
                        nn::slice_axis(sorted, 0, 1, indices.numel())))
          .item<bool>())
    throw std::invalid_argument("unpool winner indices must be unique");
  return nn::unsorted_segment_sum(values.reshape({-1}), indices, total).reshape(s);
}
/// @brief Configure full or width-reduced SegNet with learned BN affine parameters.
///
/// Full RGB/11-class widths64/128/256/512/512; compact divisor8.
/// No pretrained weights or optimizer; BN uses epsilon1e-5/momentum.9.
///
/// @throws std::invalid_argument For invalid geometry,dtypes,indices or configuration.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::SegNetConfig c;
/// c.classes = 4;
/// c.width_divisor = 8;
/// const auto net = n::models::SegNet(p, state, n::ones({1, 32, 32, 3}), c);
/// @endcode
/// @par References
/// - Badrinarayanan,Kendall and Cipolla(2015), SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation, https://arxiv.org/abs/1511.00561; original implementation https://github.com/alexgkendall/SegNet-Tutorial.
/// @see SegNet()
struct SegNetConfig {
  int classes = 11;      ///< At least two linear output classes.
  int width_divisor = 1; ///< Positive divisor of64,reduces every encoder/decoder width.
};
/// @brief Own SegNet logits,saved pooling records and sparse decoder features.
///
/// Pooling records retain values/indices and pre-pool shapes; index bytes and
/// feature bytes can be measured independently. No independent learned state.
///
/// @throws std::invalid_argument For invalid geometry,dtypes,indices or configuration.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::SegNetConfig c;
/// c.classes = 4;
/// c.width_divisor = 8;
/// const auto net = n::models::SegNet(p, state, n::ones({1, 32, 32, 3}), c);
/// @endcode
/// @par References
/// - Badrinarayanan,Kendall and Cipolla(2015), SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation, https://arxiv.org/abs/1511.00561; original implementation https://github.com/alexgkendall/SegNet-Tutorial.
/// @see SegNet()
struct SegNetOutputs {
  nn::Tensor logits; ///< Linear NHWC logits,exact original H/W including odd dimensions.
  std::vector<SegNetIndexedPool> pooling; ///< Five encoder records,shallow to deep.
  std::vector<nn::Tensor> unpooled;       ///< Five sparse decoder maps,deep to shallow.
};
/// @brief Build a VGG16 encoder and mirrored index-guided sparse decoder.
///
/// Thirteen encoder and mirrored decoder convolutions use BN/ReLU except
/// the final biased3x3 class predictor. Original geometry includes odd rows via
/// saved pre-pool shapes. Full learned/state counts29449355/15872,102 learned
/// tensors. No fitting or pretrained weights. Warm every leaf before optimizer
/// creation; checkpoint learned and moving state together. Keep leaves unchanged
/// until backward and avoid concurrent store mutation.
///
/// @param p Same-dtype learned store,creates/reuses convolution/bias/BN affine leaves.
/// @param state Same-dtype moving BN moments; training forwards update eagerly.
/// @param images Positive Float32/64 RGB NHWC,side dimensions>=32.
/// @param cfg Valid classes and channel divisor.
/// @param training False(default) moving moments,true batch statistics and moving updates.
/// @return Linear class logits aligned to original pixels and five pooling/unpooling records.
/// @throws std::invalid_argument For invalid geometry,dtypes,indices or configuration.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::SegNetConfig c;
/// c.classes = 4;
/// c.width_divisor = 8;
/// const auto net = n::models::SegNet(p, state, n::ones({1, 32, 32, 3}), c);
/// @endcode
/// @par References
/// - Badrinarayanan,Kendall and Cipolla(2015), SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation, https://arxiv.org/abs/1511.00561; original implementation https://github.com/alexgkendall/SegNet-Tutorial.
/// @see SegNetPoolWithIndices(), SegNetIndexUnpool()
inline SegNetOutputs SegNet(nn::ParameterStore &p, nn::ParameterStore &state,
                            const nn::Tensor &images, const SegNetConfig &cfg = {},
                            bool training = false) {
  if (cfg.classes < 2 || cfg.width_divisor < 1 || 64 % cfg.width_divisor ||
      images.dim() != 4 || images.shape()[0] < 1 || images.shape()[1] < 32 ||
      images.shape()[2] < 32 || images.shape()[3] != 3 || images.dtype() != p.dtype() ||
      state.dtype() != p.dtype())
    throw std::invalid_argument("invalid SegNet configuration");
  SegNetOutputs result;
  const std::int64_t widths[] = {64 / cfg.width_divisor, 128 / cfg.width_divisor,
                                 256 / cfg.width_divisor, 512 / cfg.width_divisor,
                                 512 / cfg.width_divisor};
  const int repeats[] = {2, 2, 3, 3, 3};
  /// @brief Borrow stores/mode to create convolution and optional BN/ReLU during this forward.
  auto conv = [&](const std::string &name, const nn::Tensor &x, std::int64_t out,
                  bool normalized = true) {
    nn::ConvLayerOptions options; // 3x3 SAME with bias, He uniform
    options.init = nn::init::he_uniform();
    const nn::Tensor y = nn::conv2d(p, name, x, out, 3, options);
    if (!normalized)
      return y;
    nn::BatchNormOptions bn;
    bn.momentum = .9;
    bn.eps = 1e-5;
    return nn::relu(nn::batch_norm(p, state, name + "/bn", y, training, bn));
  };
  nn::Tensor h = images;
  for (int stage = 0; stage < 5; ++stage) {
    for (int block = 0; block < repeats[stage]; ++block)
      h = conv("encoder" + std::to_string(stage) + "/conv" + std::to_string(block), h,
               widths[stage]);
    result.pooling.push_back(SegNetPoolWithIndices(h));
    h = result.pooling.back().values;
  }
  for (int stage = 4; stage >= 0; --stage) {
    h = SegNetIndexUnpool(h, result.pooling[std::size_t(stage)]);
    result.unpooled.push_back(h);
    for (int block = 0; block < repeats[stage]; ++block) {
      const bool last = block == repeats[stage] - 1;
      const std::int64_t out =
          last ? (stage ? widths[stage - 1] : cfg.classes) : widths[stage];
      h = conv("decoder" + std::to_string(stage) + "/conv" + std::to_string(block), h, out,
               !(stage == 0 && last));
    }
  }
  result.logits = h;
  return result;
}

} // namespace datamunge::nn::models
