#pragma once
/// @file partial_inpainting_data.hpp
/// @brief Tensor irregular-mask geometry,observed-only context reference,metrics and actual preview tiles.
#include "datamunge/nn/inpainting_data.hpp"
#include <random>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Own original target RGB,masked observations and multichannel validity.
///
/// Targets never enter model forward or the visible-only reference. Tensor
/// handles retain storage;fixed masks depend only on seed and source row metadata.
///
/// @throws std::invalid_argument For invalid source rows,geometry,unsupported visible context or failed mask rejection sampling.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto d = m::PartialPairs(n::zeros({2, 32, 32, 3}, n::DType::Int32), {1}, 107);
/// @endcode
/// @par References
/// - Liu et al.(2018),Image Inpainting for Irregular Holes Using Partial Convolutions, https://arxiv.org/abs/1804.07723. Custom deterministic CIFAR32 teaching masks use one rectangle and two swept disks,12..55 percent holes,not a published mask benchmark.
/// @see PartialPairs(), PartialMean(), ScoreTensorPartial()
struct PartialData {
  nn::Tensor truth; ///< Original Float32 unit RGB[N,32,32,3].
  nn::Tensor input; ///< Observed RGB with holes exactly0,same shape.
  nn::Tensor mask;  ///< Float32 binary validity replicated across three RGB channels.
};
/// @brief Generate source-identical irregular validity using tensor geometry.
///
/// MT19937 seed is uint32(seed+65537*source_row). Host code draws rectangle
/// and two stroke geometry parameters only. Tensor coordinates compute the
/// rectangle,two65-position swept disks,unions and hole counts. Retry at most
/// 100 candidates until123..563 of1024 pixels are holes. Geometry never reads
/// image values or labels;reordering/repeating rows reproduces each mask.
///
/// @param source_row Original nonnegative image-row identity,not current batch position.
/// @param seed Split-specific32-bit seed.
/// @return Float32[1,32,32,3] identical binary validity across RGB.
/// @throws std::invalid_argument For invalid source rows,geometry,unsupported visible context or failed mask rejection sampling.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto mask = m::PartialMask(7, 107);
/// @endcode
/// @par References
/// - Liu et al.(2018),Image Inpainting for Irregular Holes Using Partial Convolutions, https://arxiv.org/abs/1804.07723. Custom deterministic CIFAR32 teaching masks use one rectangle and two swept disks,12..55 percent holes,not a published mask benchmark.
/// @see PartialPairs(), PartialMean(), ScoreTensorPartial()
inline nn::Tensor PartialMask(std::size_t source_row, std::uint32_t seed) {
  std::mt19937 rng(std::uint32_t(seed + 65537 * source_row));
  /// @brief Draw one integer mask-geometry parameter with the source RNG policy.
  /// @param low Inclusive lower metadata bound.
  /// @param high Inclusive upper metadata bound.
  /// @return Uniform integer geometry metadata.
  auto integer = [&](int low, int high) {
    return std::uniform_int_distribution<int>(low, high)(rng);
  };
  const auto y = nn::arange(0, 32, 1, nn::DType::Float64).reshape({32, 1}),
             x = nn::arange(0, 32, 1, nn::DType::Float64).reshape({1, 32}),
             steps = nn::arange(0, 65, 1, nn::DType::Float64).reshape({65, 1, 1}) / 64.;
  for (int attempt = 0; attempt < 100; ++attempt) {
    const int y0 = integer(0, 25), x0 = integer(0, 25), rh = integer(5, 12),
              rw = integer(5, 12);
    auto holes = nn::logical_and(
        nn::logical_and(nn::greater_equal(y, double(y0)), nn::less(y, double(y0 + rh))),
        nn::logical_and(nn::greater_equal(x, double(x0)), nn::less(x, double(x0 + rw))));
    for (int stroke = 0; stroke < 2; ++stroke) {
      const int ay = integer(0, 31), ax = integer(0, 31), by = integer(0, 31),
                bx = integer(0, 31), radius = integer(1, 3);
      const auto cy = double(ay) + double(by - ay) * steps,
                 cx = double(ax) + double(bx - ax) * steps,
                 distance = nn::square(y.reshape({1, 32, 1}) - cy) +
                            nn::square(x.reshape({1, 1, 32}) - cx);
      const auto swept = nn::greater(
          nn::sum(nn::less_equal(distance, double(radius * radius)).to(nn::DType::Int32), {0}),
          0.);
      holes = nn::logical_or(holes, swept);
    }
    const auto count = nn::count_nonzero(holes).item<std::int64_t>();
    if (count >= 123 && count <= 563)
      return nn::ones({1, 32, 32, 3}) *
             nn::logical_not(holes).to(nn::DType::Float32).reshape({1, 32, 32, 1});
  }
  throw std::invalid_argument("irregular mask12..55 percent rejection limit");
}
/// @brief Pair original CIFAR targets with fixed source-row irregular observations.
///
/// Tensor raw-byte normalization uses canonical ContextPairs truth only.
/// Masks are geometry-only;holes become exact0 through tensor multiplication.
/// No generated replacement data,external image processing or hidden fit.
///
/// @param images Original Int32[N,32,32,3] encoded RGB bytes.
/// @param rows Nonempty bounded original row metadata.
/// @param seed Split-specific mask seed.
/// @return Aligned Float32 truth,input and multichannel mask.
/// @throws std::invalid_argument For invalid source rows,geometry,unsupported visible context or failed mask rejection sampling.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto d = m::PartialPairs(n::zeros({2, 32, 32, 3}, n::DType::Int32), {1}, 107);
/// @endcode
/// @par References
/// - Liu et al.(2018),Image Inpainting for Irregular Holes Using Partial Convolutions, https://arxiv.org/abs/1804.07723. Custom deterministic CIFAR32 teaching masks use one rectangle and two swept disks,12..55 percent holes,not a published mask benchmark.
/// @see PartialPairs(), PartialMean(), ScoreTensorPartial()
inline PartialData PartialPairs(const nn::Tensor &images, const std::vector<std::size_t> &rows,
                                std::uint32_t seed) {
  if (rows.empty())
    throw std::invalid_argument("nonempty partial image rows required");
  const auto truth = ContextPairs(images, rows).truth;
  std::vector<nn::Tensor> masks;
  for (auto row : rows)
    masks.push_back(PartialMask(row, seed));
  const auto mask = nn::concat(masks, 0);
  return {truth, truth * mask, mask};
}
/// @brief Gather original targets,masked observations and validity together.
///
/// No new mask sampling,numeric host copies or learned preprocessing.
///
/// @param data Aligned32x32 source pair tensors.
/// @param rows Bounded source row metadata.
/// @return Aligned tensors in requested row order.
/// @throws std::invalid_argument For invalid source rows,geometry,unsupported visible context or failed mask rejection sampling.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto d = m::PartialPairs(n::zeros({2, 32, 32, 3}, n::DType::Int32), {1, 0}, 107);
/// const auto b = m::PartialBatch(d, {1});
/// @endcode
/// @par References
/// - Liu et al.(2018),Image Inpainting for Irregular Holes Using Partial Convolutions, https://arxiv.org/abs/1804.07723. Custom deterministic CIFAR32 teaching masks use one rectangle and two swept disks,12..55 percent holes,not a published mask benchmark.
/// @see PartialPairs(), PartialMean(), ScoreTensorPartial()
inline PartialData PartialBatch(const PartialData &data,
                                const std::vector<std::size_t> &rows) {
  return {InpaintBatch(data.truth, rows), InpaintBatch(data.input, rows),
          InpaintBatch(data.mask, rows)};
}
/// @brief Preserve known pixels and fill holes with that image’s observed mean RGB.
///
/// A deterministic per-observation statistic,not a fitted training model.
/// Float64 tensor sums over visible pixels divide by per-channel support,then
/// cast once to Float32. Hidden arbitrary/NaN fill is selected out BEFORE sum;
/// changing unobserved target values cannot affect this reference.
///
/// @param data Supplies observed input and numeric validity only;truth values are never read.
/// @return Float32 matching-shape visible-mean reconstruction.
/// @throws std::invalid_argument For invalid source rows,geometry,unsupported visible context or failed mask rejection sampling.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto d = m::PartialPairs(n::zeros({2, 32, 32, 3}, n::DType::Int32), {1}, 107);
/// const auto mean = m::PartialMean(d);
/// @endcode
/// @par References
/// - Liu et al.(2018),Image Inpainting for Irregular Holes Using Partial Convolutions, https://arxiv.org/abs/1804.07723. Custom deterministic CIFAR32 teaching masks use one rectangle and two swept disks,12..55 percent holes,not a published mask benchmark.
/// @see PartialPairs(), PartialMean(), ScoreTensorPartial()
inline nn::Tensor PartialMean(const PartialData &data) {
  if (data.input.dim() != 4 || data.mask.shape() != data.input.shape() ||
      data.input.size(0) < 1 || data.input.size(3) != 3)
    throw std::invalid_argument("aligned observed partial RGB/mask required");
  const auto known = nn::greater(data.mask.to(nn::DType::Float64), 0.),
             weights = known.to(nn::DType::Float64), counts = nn::sum(weights, {1, 2}, true);
  if (!nn::all(nn::greater(counts, 0.)).item<bool>())
    throw std::invalid_argument("observed-only mean requires visible support");
  const auto safe = nn::where(known, data.input, 0.).to(nn::DType::Float64),
             mean = (nn::sum(safe, {1, 2}, true) / counts).to(nn::DType::Float32);
  return nn::where(known, data.input, nn::ones_like(data.input) * mean);
}
/// @brief Score all irregular-hole predictions with canonical tensor inpainting metrics.
///
/// Mask first channel supplies pixel support;expected RGB masks agree.
/// Each image’s hole error divides by true missing RGB count. Mean PSNR
/// differs from PSNR of pooled MSE. No training objective or target alignment.
///
/// @param data Original truth and RGB-replicated validity.
/// @param output Finite raw32x32 RGB reconstruction matching truth.
/// @param clip Whether to clip estimates to[0,1] for this metric only,default false.
/// @return Detached Float64 hole MSE,mean per-image hole/full PSNR and known max error.
/// @throws std::invalid_argument For invalid source rows,geometry,unsupported visible context or failed mask rejection sampling.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto d = m::PartialPairs(n::zeros({2, 32, 32, 3}, n::DType::Int32), {1}, 107);
/// const auto s = m::ScoreTensorPartial(d, d.truth);
/// @endcode
/// @par References
/// - Liu et al.(2018),Image Inpainting for Irregular Holes Using Partial Convolutions, https://arxiv.org/abs/1804.07723. Custom deterministic CIFAR32 teaching masks use one rectangle and two swept disks,12..55 percent holes,not a published mask benchmark.
/// @see PartialPairs(), PartialMean(), ScoreTensorPartial()
inline TensorInpaintMetrics ScoreTensorPartial(const PartialData &data,
                                               const nn::Tensor &output, bool clip = false) {
  InpaintData adapter;
  adapter.truth = data.truth;
  if (data.mask.dim() != 4 || data.mask.shape() != data.truth.shape())
    throw std::invalid_argument("aligned partial score mask required");
  adapter.valid = nn::slice_axis(data.mask, 3, 0, 1).contiguous();
  return ScoreTensorInpaint(adapter, clip ? nn::clip(output, 0., 1.) : output);
}
/// @brief Present canonical tensor irregular-hole reconstruction scores as host scalars.
///
/// All numerical work delegates to ScoreTensorPartial.
///
/// @param data Original truth and RGB-replicated validity.
/// @param output Finite raw32x32 RGB reconstruction.
/// @param clip Metric-only clipping,default false.
/// @return Legacy host presentation metrics.
/// @throws std::invalid_argument For invalid source rows,geometry,unsupported visible context or failed mask rejection sampling.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto d = m::PartialPairs(n::zeros({2, 32, 32, 3}, n::DType::Int32), {1}, 107);
/// const auto s = m::PartialScore(d, d.truth);
/// @endcode
/// @par References
/// - Liu et al.(2018),Image Inpainting for Irregular Holes Using Partial Convolutions, https://arxiv.org/abs/1804.07723. Custom deterministic CIFAR32 teaching masks use one rectangle and two swept disks,12..55 percent holes,not a published mask benchmark.
/// @see PartialPairs(), PartialMean(), ScoreTensorPartial()
inline InpaintMetrics PartialScore(const PartialData &data, const nn::Tensor &output,
                                   bool clip = false) {
  const auto s = ScoreTensorPartial(data, output, clip);
  return {s.hole_mse.item(), s.hole_psnr.item(), s.full_psnr.item(), s.known_error.item()};
}
/// @brief Build all observed/reference/prediction/truth/error tiles with source display units.
///
/// Error matches source Float32 channel differences/division,then Float64
/// channel accumulation. Display clips only after computing raw RGB-mean
/// absolute error. Float64 half-up byte quantization matches std::lround.
/// Reported numerical scores remain separate from display saturation.
///
/// @param data Original truth,observed input and masks.
/// @param predicted Raw or composite reconstruction.
/// @param mean Supplied observed-only mean reconstruction.
/// @return Float32[N,32,160,3] observed|mean|prediction|truth|4x red RGB-mean error.
/// @throws std::invalid_argument For invalid source rows,geometry,unsupported visible context or failed mask rejection sampling.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto d = m::PartialPairs(n::zeros({2, 32, 32, 3}, n::DType::Int32), {1}, 107);
/// const auto mean = m::PartialMean(d);
/// const auto tiles = m::PartialPreviewTiles(d, mean, mean);
/// @endcode
/// @par References
/// - Liu et al.(2018),Image Inpainting for Irregular Holes Using Partial Convolutions, https://arxiv.org/abs/1804.07723. Custom deterministic CIFAR32 teaching masks use one rectangle and two swept disks,12..55 percent holes,not a published mask benchmark.
/// @see PartialPairs(), PartialMean(), ScoreTensorPartial()
inline nn::Tensor PartialPreviewTiles(const PartialData &data, const nn::Tensor &predicted,
                                      const nn::Tensor &mean) {
  if (data.truth.dim() != 4 ||
      data.truth.shape() != nn::Shape{data.truth.size(0), 32, 32, 3} ||
      predicted.shape() != data.truth.shape() || mean.shape() != data.truth.shape() ||
      data.input.shape() != data.truth.shape())
    throw std::invalid_argument("aligned32px partial preview required");
  const auto error = nn::sum((nn::abs(predicted - data.truth) / 3.).to(nn::DType::Float64),
                             {3}, true) *
                     4.,
             zero = nn::zeros_like(error);
  const auto tiles =
      nn::concat({data.input.to(nn::DType::Float64), mean.to(nn::DType::Float64),
                  predicted.to(nn::DType::Float64), data.truth.to(nn::DType::Float64),
                  nn::concat({error, zero, zero}, 3)},
                 2);
  return (nn::floor(nn::clip(tiles, 0., 1.) * 255. + .5) / 255.).to(nn::DType::Float32);
}
} // namespace datamunge::nn::models
