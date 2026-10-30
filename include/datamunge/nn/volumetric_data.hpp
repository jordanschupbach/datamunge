#pragma once
/// @file volumetric_data.hpp
/// @brief Tensor ellipsoid volumes,sparse axial annotations,binary metrics and orthogonal slice previews.
#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/ops.hpp"
#include "datamunge/nn/random.hpp"
#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Own detached noisy volume images and dense binary voxel labels.
///
/// No learned parameters,optimizer or borrowed source buffers. Generator
/// outputs Float32 images and Int64 labels.
///
/// @throws std::invalid_argument For invalid geometry,dtypes,indices,probabilities or binary labels.
/// @par Example
/// @code
/// const auto data = dm::nn::models::MakeTensorEllipsoidVolumes(3, 16, 11);
/// @endcode
/// @par References
/// - Synthetic oblique ellipsoid/noise and orthogonal-preview conventions from examples/nn/common/volumetric.hpp. Sparse-slice learning motivation: Çiçek et al.(2016), https://arxiv.org/abs/1606.06650; squared-Dice motivation: Milletari et al.(2016), https://arxiv.org/abs/1606.04797. Tensor RNG intentionally differs from the original std::mt19937 sampler.
/// @see MakeTensorEllipsoidVolumes()
struct TensorVolumeData {
  nn::Tensor images; ///< Float32[N,S,S,S,1] image intensity,not restricted to[0,1].
  nn::Tensor labels; ///< Int64[N,S,S,S] dense binary ellipsoid truth.
};
/// @brief Generate noisy oblique ellipsoids using only tensor random sampling and voxel arithmetic.
///
/// Centres uniform[.38S,.62S],radii[.16S,.28S],tilt[-.2,.2]. Foreground
/// means.8,background-.8,independent Gaussian noise SD.12 and weak.05*x/S
/// ramp. Normalized coordinates obey dz²+dy²+(dx-tilt*dz)²<=1. Data are
/// synthetic teaching volumes,no medical/anatomical claim. Tensor draw order
/// differs from the old host implementation;separate seeds give independent
/// reproducible splits. Work/storageO(count*side³),no fitting/input mutation.
///
/// @param count Positive volume count.
/// @param side Spatial side at least16,divisible by16,default16.
/// @param seed Deterministic tensor-generator seed,default11.
/// @return Detached Float32 images and Int64 dense binary labels.
/// @throws std::invalid_argument For invalid geometry,dtypes,indices,probabilities or binary labels.
/// @par Example
/// @code
/// const auto data = dm::nn::models::MakeTensorEllipsoidVolumes(3, 16, 11);
/// @endcode
/// @par References
/// - Synthetic oblique ellipsoid/noise and orthogonal-preview conventions from examples/nn/common/volumetric.hpp. Sparse-slice learning motivation: Çiçek et al.(2016), https://arxiv.org/abs/1606.06650; squared-Dice motivation: Milletari et al.(2016), https://arxiv.org/abs/1606.04797. Tensor RNG intentionally differs from the original std::mt19937 sampler.
/// @see MakeTensorEllipsoidVolumes()
inline TensorVolumeData MakeTensorEllipsoidVolumes(int count, int side = 16,
                                                   std::uint64_t seed = 11) {
  if (count < 1 || side < 16 || side % 16)
    throw std::invalid_argument("volume count must be positive and side divisible by16");
  nn::NoGradGuard guard;
  nn::Generator rng(seed);
  const auto centers = nn::unstack(nn::uniform({count, 3}, .38 * side, .62 * side, rng), 1),
             radii = nn::unstack(nn::uniform({count, 3}, .16 * side, .28 * side, rng), 1);
  const auto tilt = nn::uniform({count, 1, 1, 1}, -.2, .2, rng);
  const auto coordinates = nn::arange(0, double(side));
  const auto z = coordinates.reshape({1, side, 1, 1}),
             y = coordinates.reshape({1, 1, side, 1}),
             x = coordinates.reshape({1, 1, 1, side});
  const auto dz = (z + .5 - centers[0].reshape({count, 1, 1, 1})) /
                  radii[0].reshape({count, 1, 1, 1}),
             dy = (y + .5 - centers[1].reshape({count, 1, 1, 1})) /
                  radii[1].reshape({count, 1, 1, 1}),
             dx = (x + .5 - centers[2].reshape({count, 1, 1, 1})) /
                      radii[2].reshape({count, 1, 1, 1}) -
                  tilt * dz;
  const auto foreground = nn::less_equal(nn::square(dz) + nn::square(dy) + nn::square(dx), 1.);
  const auto images = nn::where(foreground, nn::full({}, .8), nn::full({}, -.8)) +
                      nn::normal({count, side, side, side}, 0., .12, rng) +
                      .05 * (x / double(side) - .5);
  return {images.reshape({count, side, side, side, 1}), foreground.to(nn::DType::Int64)};
}
/// @brief Copy selected volumes and optionally retain exactly one in four axial annotation slices.
///
/// Side is a positive multiple of16. Sparse mode annotates exactly25% of
/// voxels without changing image intensities or row pairing. No fitting,
/// source mutation or RNG.
///
/// @param data Valid floating[N,S,S,S,1] images and Int64[N,S,S,S] binary labels.
/// @param rows Source volume IDs in requested order,duplicates/empty allowed.
/// @param sparse False(default) keeps dense labels;true retains z%4==2 and fills others with255.
/// @return Detached copied image/label tensors for rows;images always keep all voxels.
/// @throws std::invalid_argument For invalid geometry,dtypes,indices,probabilities or binary labels.
/// @par Example
/// @code
/// const auto data = dm::nn::models::MakeTensorEllipsoidVolumes(3, 16, 11);
/// const auto batch = dm::nn::models::TensorVolumeBatch(data, {2, 0}, true);
/// @endcode
/// @par References
/// - Synthetic oblique ellipsoid/noise and orthogonal-preview conventions from examples/nn/common/volumetric.hpp. Sparse-slice learning motivation: Çiçek et al.(2016), https://arxiv.org/abs/1606.06650; squared-Dice motivation: Milletari et al.(2016), https://arxiv.org/abs/1606.04797. Tensor RNG intentionally differs from the original std::mt19937 sampler.
/// @see SparseVolumeCE(), VolumeSlicePreview()
inline std::pair<nn::Tensor, nn::Tensor>
TensorVolumeBatch(const TensorVolumeData &data, const std::vector<std::int64_t> &rows,
                  bool sparse = false) {
  if (data.images.dim() != 5 || data.images.shape()[0] < 1 || data.images.shape()[1] < 16 ||
      data.images.shape()[1] % 16 || data.images.shape()[2] != data.images.shape()[1] ||
      data.images.shape()[3] != data.images.shape()[1] || data.images.shape()[4] != 1 ||
      (data.images.dtype() != nn::DType::Float32 &&
       data.images.dtype() != nn::DType::Float64) ||
      data.labels.shape() != nn::Shape{data.images.shape()[0], data.images.shape()[1],
                                       data.images.shape()[2], data.images.shape()[3]} ||
      data.labels.dtype() != nn::DType::Int64 ||
      !nn::all(nn::logical_and(nn::greater_equal(data.labels, 0.), nn::less(data.labels, 2.)))
           .item<bool>())
    throw std::invalid_argument("invalid tensor volume data");
  for (const auto row : rows)
    if (row < 0 || row >= data.images.shape()[0])
      throw std::invalid_argument("volume row out of range");
  nn::NoGradGuard guard;
  const auto indices = nn::from_vector(rows);
  const auto images = nn::gather(data.images, indices);
  auto labels = nn::gather(data.labels, indices);
  if (sparse) {
    const auto S = data.images.shape()[1];
    const auto z = nn::arange(0, double(S), 1., nn::DType::Int64);
    const auto remainder = z - nn::floor(z.to(nn::DType::Float64) / 4.).to(nn::DType::Int64) *
                                   nn::full({}, 4., nn::DType::Int64);
    labels = nn::where(nn::equal(remainder, 2.).reshape({1, S, 1, 1}), labels,
                       nn::full({}, 255., nn::DType::Int64));
  }
  return {images, labels};
}
/// @brief Own mean per-volume foreground overlap,global voxel accuracy and exact binary confusion counts.
///
/// Metric tensors are detached. Dice/IoU average volumes equally;accuracy
/// weights all voxels. No fitting or hidden inference.
///
/// @throws std::invalid_argument For invalid geometry,dtypes,indices,probabilities or binary labels.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto truth = n::from_vector(std::vector<std::int64_t>{0, 1, 1, 0}, {1, 1, 2, 2});
/// const auto score = n::models::VolumePredictionMetrics(truth, truth);
/// @endcode
/// @par References
/// - Synthetic oblique ellipsoid/noise and orthogonal-preview conventions from examples/nn/common/volumetric.hpp. Sparse-slice learning motivation: Çiçek et al.(2016), https://arxiv.org/abs/1606.06650; squared-Dice motivation: Milletari et al.(2016), https://arxiv.org/abs/1606.04797. Tensor RNG intentionally differs from the original std::mt19937 sampler.
/// @see VolumePredictionMetrics()
struct TensorVolumeMetrics {
  nn::Tensor dice;     ///< Float64 scalar,mean per-volume2TP/(2TP+FP+FN),both-empty score1.
  nn::Tensor iou;      ///< Float64 scalar,mean per-volumeTP/(TP+FP+FN),both-empty score1.
  nn::Tensor accuracy; ///< Float64 scalar,(globalTP+TN)/all voxels.
  nn::Tensor tp;       ///< Int64 scalar global true foreground count.
  nn::Tensor fp;       ///< Int64 scalar global false foreground count.
  nn::Tensor fn;       ///< Int64 scalar global missed foreground count.
  nn::Tensor tn;       ///< Int64 scalar global true background count.
};
/// @brief Score supplied dense binary volume predictions with tensor confusion and overlap reductions.
///
/// No model forward or optimizer is hidden. Threshold/argmax choices belong
/// to the caller. Both-empty foreground gives Dice/IoU1. No void labels are
/// accepted for held-out dense scoring. WorkO(NDHW),no source mutation.
///
/// @param predictions Int32/64[N,D,H,W] binary labels,all axes positive.
/// @param labels Same-shape Int32/64 dense binary truth.
/// @return Detached mean-volume Dice/IoU,global voxel accuracy and Int64 confusion counts.
/// @throws std::invalid_argument For invalid geometry,dtypes,indices,probabilities or binary labels.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto truth = n::from_vector(std::vector<std::int64_t>{0, 1, 1, 0}, {1, 1, 2, 2});
/// const auto score = n::models::VolumePredictionMetrics(truth, truth);
/// @endcode
/// @par References
/// - Synthetic oblique ellipsoid/noise and orthogonal-preview conventions from examples/nn/common/volumetric.hpp. Sparse-slice learning motivation: Çiçek et al.(2016), https://arxiv.org/abs/1606.06650; squared-Dice motivation: Milletari et al.(2016), https://arxiv.org/abs/1606.04797. Tensor RNG intentionally differs from the original std::mt19937 sampler.
/// @see TensorVolumeBatch(), SquaredVolumeDice()
inline TensorVolumeMetrics VolumePredictionMetrics(const nn::Tensor &predictions,
                                                   const nn::Tensor &labels) {
  if (predictions.dim() != 4 || predictions.shape()[0] < 1 || predictions.shape()[1] < 1 ||
      predictions.shape()[2] < 1 || predictions.shape()[3] < 1 ||
      predictions.shape() != labels.shape() ||
      (predictions.dtype() != nn::DType::Int32 && predictions.dtype() != nn::DType::Int64) ||
      (labels.dtype() != nn::DType::Int32 && labels.dtype() != nn::DType::Int64) ||
      !nn::all(nn::logical_and(nn::greater_equal(predictions, 0.), nn::less(predictions, 2.)))
           .item<bool>() ||
      !nn::all(nn::logical_and(nn::greater_equal(labels, 0.), nn::less(labels, 2.)))
           .item<bool>())
    throw std::invalid_argument("invalid dense binary volume metrics");
  nn::NoGradGuard guard;
  const auto p = nn::equal(predictions, 1.), t = nn::equal(labels, 1.);
  const auto tp = nn::sum(nn::logical_and(p, t).to(nn::DType::Int64), {1, 2, 3}),
             fp = nn::sum(nn::logical_and(p, nn::logical_not(t)).to(nn::DType::Int64),
                          {1, 2, 3}),
             fn = nn::sum(nn::logical_and(nn::logical_not(p), t).to(nn::DType::Int64),
                          {1, 2, 3}),
             tn = nn::sum(
                 nn::logical_and(nn::logical_not(p), nn::logical_not(t)).to(nn::DType::Int64),
                 {1, 2, 3});
  const auto dtp = tp.to(nn::DType::Float64), dfp = fp.to(nn::DType::Float64),
             dfn = fn.to(nn::DType::Float64), dice_den = 2. * dtp + dfp + dfn,
             iou_den = dtp + dfp + dfn;
  const auto dice = nn::mean(
                 nn::where(nn::greater(dice_den, 0.),
                           2. * dtp / nn::maximum(dice_den, nn::ones({}, nn::DType::Float64)),
                           nn::ones({}, nn::DType::Float64))),
             iou = nn::mean(
                 nn::where(nn::greater(iou_den, 0.),
                           dtp / nn::maximum(iou_den, nn::ones({}, nn::DType::Float64)),
                           nn::ones({}, nn::DType::Float64))),
             accuracy = nn::sum((tp + tn).to(nn::DType::Float64)) / double(labels.numel());
  return {dice, iou, accuracy, nn::sum(tp), nn::sum(fp), nn::sum(fn), nn::sum(tn)};
}
/// @brief Arrange axial,coronal and sagittal centre slices as input/truth/optional-prediction RGB rows.
///
/// Input display clips[-1.2,1.2] and floors byte intensity. Binary truth/pred
/// use dark background(20,20,20)/255 and green foreground(60,220,60)/255.
/// No interpolation,model inference,fitting or input mutation. Native spatial
/// slice geometry remains explicit;the caller sets image-grid display scale.
///
/// @param data Valid TensorVolumeData with binary dense labels.
/// @param predictions Optional Int32/64[N,S,S,S] binary predictions;undefined(default) omits prediction panel.
/// @param count Positive maximum number of first volumes,default2.
/// @return Float32[take*3,S,S*(predictions?3:2),3] RGB slice rows in volume/axial-coronal-sagittal order.
/// @throws std::invalid_argument For invalid geometry,dtypes,indices,probabilities or binary labels.
/// @par Example
/// @code
/// const auto data = dm::nn::models::MakeTensorEllipsoidVolumes(3, 16, 11);
/// const auto slices = dm::nn::models::VolumeSlicePreview(data);
/// @endcode
/// @par References
/// - Synthetic oblique ellipsoid/noise and orthogonal-preview conventions from examples/nn/common/volumetric.hpp. Sparse-slice learning motivation: Çiçek et al.(2016), https://arxiv.org/abs/1606.06650; squared-Dice motivation: Milletari et al.(2016), https://arxiv.org/abs/1606.04797. Tensor RNG intentionally differs from the original std::mt19937 sampler.
/// @see TensorVolumeBatch(), VolumePredictionMetrics()
inline nn::Tensor VolumeSlicePreview(const TensorVolumeData &data,
                                     const nn::Tensor &predictions = {}, int count = 2) {
  if (count < 1)
    throw std::invalid_argument("positive volume preview count required");
  TensorVolumeBatch(data, {});
  const auto N = data.images.shape()[0], S = data.images.shape()[1],
             take = std::min<std::int64_t>(N, count);
  if (predictions.defined() &&
      (predictions.shape() != data.labels.shape() ||
       (predictions.dtype() != nn::DType::Int32 && predictions.dtype() != nn::DType::Int64) ||
       !nn::all(nn::logical_and(nn::greater_equal(predictions, 0.), nn::less(predictions, 2.)))
            .item<bool>()))
    throw std::invalid_argument("invalid volume preview predictions");
  nn::NoGradGuard guard;
  const auto palette = nn::from_vector(std::vector<float>{20 / 255.f, 20 / 255.f, 20 / 255.f,
                                                          60 / 255.f, 220 / 255.f, 60 / 255.f},
                                       {2, 3});
  std::vector<nn::Tensor> rows;
  for (std::int64_t volume = 0; volume < take; ++volume) {
    const auto image = nn::slice_axis(data.images, 0, volume, volume + 1).reshape({S, S, S});
    const auto truth = nn::slice_axis(data.labels, 0, volume, volume + 1).reshape({S, S, S});
    nn::Tensor prediction;
    if (predictions.defined())
      prediction = nn::slice_axis(predictions, 0, volume, volume + 1).reshape({S, S, S});
    for (int plane = 0; plane < 3; ++plane) {
      const auto intensity =
          nn::floor(
              nn::clip((nn::slice_axis(image, plane, S / 2, S / 2 + 1).reshape({S, S}) + 1.2) /
                           2.4 * 255.,
                       0., 255.)) /
          255.;
      const auto input =
          intensity.to(nn::DType::Float32).reshape({1, S, S, 1}) * nn::ones({1, 1, 1, 3});
      const auto target =
          nn::gather(palette, nn::slice_axis(truth, plane, S / 2, S / 2 + 1).reshape({S * S}))
              .reshape({1, S, S, 3});
      if (predictions.defined()) {
        const auto found =
            nn::gather(palette,
                       nn::slice_axis(prediction, plane, S / 2, S / 2 + 1).reshape({S * S}))
                .reshape({1, S, S, 3});
        rows.push_back(nn::concat({input, target, found}, 2));
      } else
        rows.push_back(nn::concat({input, target}, 2));
    }
  }
  return nn::concat(rows, 0);
}
} // namespace datamunge::nn::models
