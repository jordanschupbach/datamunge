#pragma once
/// @file paf_evaluation.hpp
/// @brief Tensor field metrics,fit-only reference oracle and genuine full-image PAF previews.
#include "datamunge/nn/image_io.hpp"
#include "datamunge/nn/paf_decode.hpp"
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Compute supported-image field MSE with Double tensor arithmetic.
///
/// Detach inputs for evaluation only. Select missing values before subtracting;
/// images with any support have equal weight,independent of support size.
///
/// @param prediction Floating[N,H,W,C] raw field.
/// @param truth Matching field targets,unknown entries may be NaN.
/// @param mask Matching numeric support;Double mask>.5 selects observations.
/// @return Double scalar average of per-image supported means;all-empty gives0.
/// @throws std::invalid_argument For incompatible geometry or empty batch.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto x = n::ones({2, 2, 2, 3});
/// const auto v = m::PafFieldMSE(x, n::zeros_like(x), x);
/// @endcode
/// @par References
/// - Cao et al.(2017),Realtime Multi-Person2D Pose Estimation using Part Affinity Fields, https://arxiv.org/abs/1611.08050. Custom tensor display and mean-field parity oracle for the small teaching split,not COCO AP.
/// @see PafMaskedMSE()

inline nn::Tensor PafFieldMSE(const nn::Tensor &prediction, const nn::Tensor &truth,
                              const nn::Tensor &mask) {
  nn::NoGradGuard guard;
  if (prediction.dim() != 4 || prediction.size(0) < 1 || prediction.shape() != truth.shape() ||
      truth.shape() != mask.shape())
    throw std::invalid_argument("aligned nonempty NHWC fields required");
  const auto active = nn::greater(mask.to(nn::DType::Float64), .5),
             difference = nn::where(active, prediction.to(nn::DType::Float64), 0.) -
                          nn::where(active, truth.to(nn::DType::Float64), 0.),
             count = nn::sum(active.to(nn::DType::Float64), {1, 2, 3}),
             present = nn::greater(count, 0.);
  return nn::sum(nn::sum(nn::square(difference), {1, 2, 3}) /
                 nn::maximum(count, nn::ones_like(count))) /
         nn::maximum(nn::sum(present.to(nn::DType::Float64)),
                     nn::full({}, 1., nn::DType::Float64));
}
/// @brief Return a tensor masked fitting-field mean solely as an optimization parity oracle.
///
/// Demonstrations fit the corresponding trainable Double field with explicit
/// SGD updates. This closed-form helper verifies that fit;it is not the fitting API.
/// Missing NaNs select out before Double reduction,then source Float32 rounding.
///
/// @param fitting Floating[N,H,W,C] fitting targets,never validation/test labels.
/// @param mask Matching numeric support;Double mask>.5 selects observations.
/// @param count Positive number of repeated output rows.
/// @return Detached Float32[count,H,W,C],unsupported coordinates0.
/// @throws std::invalid_argument For incompatible or empty geometry or nonpositive count.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto f = n::ones({2, 2, 2, 3});
/// const auto mean = m::PafMeanField(f, f, 4);
/// @endcode
/// @par References
/// - Cao et al.(2017),Realtime Multi-Person2D Pose Estimation using Part Affinity Fields, https://arxiv.org/abs/1611.08050. Custom tensor display and mean-field parity oracle for the small teaching split,not COCO AP.
/// @see PafFieldMSE()

inline nn::Tensor PafMeanField(const nn::Tensor &fitting, const nn::Tensor &mask,
                               std::int64_t count) {
  nn::NoGradGuard guard;
  if (fitting.dim() != 4 || fitting.size(0) < 1 || fitting.shape() != mask.shape() ||
      count < 1)
    throw std::invalid_argument("aligned fitting fields and positive repeat count required");
  const auto active = nn::greater(mask.to(nn::DType::Float64), .5),
             total = nn::sum(nn::where(active, fitting.to(nn::DType::Float64), 0.), {0}, true),
             present = nn::sum(active.to(nn::DType::Float64), {0}, true),
             mean =
                 (total / nn::maximum(present, nn::ones_like(present))).to(nn::DType::Float32);
  return mean + nn::zeros({count, fitting.size(1), fitting.size(2), fitting.size(3)});
}
/// @brief Private tensor limb rasterization helpers.
namespace paf_detail {
/// @brief Rasterize finite continuous segments using tensor distance<=1 pixel.
/// @param segments Double[L,2,2] endpoints in128px input geometry,L>0.
/// @return Bool[128,128,1] union of one-pixel-radius continuous segments.
inline nn::Tensor Lines(const nn::Tensor &segments) {
  const auto v = segments.to(nn::DType::Float64);
  if (v.dim() != 3 || v.size(0) < 1 || v.size(1) != 2 || v.size(2) != 2 ||
      !nn::all(nn::is_finite(v)).item<bool>())
    throw std::invalid_argument("finite limb endpoints required");
  const auto a = nn::select(v, 1, 0), b = nn::select(v, 1, 1),
             ax = nn::select(a, 1, 0).reshape({-1, 1, 1}),
             ay = nn::select(a, 1, 1).reshape({-1, 1, 1}),
             dx = (nn::select(b, 1, 0) - nn::select(a, 1, 0)).reshape({-1, 1, 1}),
             dy = (nn::select(b, 1, 1) - nn::select(a, 1, 1)).reshape({-1, 1, 1}),
             x = nn::arange(0, 128, 1, nn::DType::Float64).reshape({1, 1, 128}),
             y = nn::arange(0, 128, 1, nn::DType::Float64).reshape({1, 128, 1}),
             t = nn::clip(
                 ((x - ax) * dx + (y - ay) * dy) /
                     nn::maximum(dx * dx + dy * dy, nn::full({}, 1e-30, nn::DType::Float64)),
                 0., 1.);
  return nn::expand_dims(
      nn::any(nn::less_equal(nn::square(x - ax - t * dx) + nn::square(y - ay - t * dy), 1.),
              {0}),
      2);
}
} // namespace paf_detail
/// @brief Render actual RGB,annotated/grouped skeletons,confidence and directed-field magnitudes.
///
/// Only visible annotated limbs and assembled predicted links render;no isolated
/// joint dots. Display uses continuous segment distance<=1,an explicit geometry
/// update from legacy sampled3x3 line dots. Numeric targets/decoding/scores are
/// unchanged. Max confidence starts at0;max Double limb norm clips only for
/// display. Nearest8x fields and half-up byte quantization use tensors.
/// Host loops handle topology and serialization only;all coordinate/pixel math
/// is tensor based. Unknown annotation coordinates are excluded before geometry.
///
/// @param data Original Float32[N,128,128,3] images with COCO18 annotations/counts.
/// @param decoded One grouped decoding per image,indices refer to its own peaks/links.
/// @param heat Finite[N,16,16,19] unbounded final confidence.
/// @param paf Finite[N,16,16,38] signed final field.
/// @return Float32[N,128,640,3] input|green GT|red prediction|red confidence|blue magnitude.
/// @throws std::invalid_argument For misaligned geometry,invalid topology or nonfinite rendered values.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// m::PafData d{n::zeros({1, 128, 128, 3}), n::zeros({1, 1, 18, 3}), n::ones({1, 1, 4}),
///              n::ones({1}, n::DType::Int32), n::zeros({1}, n::DType::Int32)};
/// const auto tiles = m::PafPreviewTiles(d, {m::PafDecoded{}}, n::zeros({1, 16, 16, 19}),
///                                       n::zeros({1, 16, 16, 38}));
/// @endcode
/// @par References
/// - Cao et al.(2017),Realtime Multi-Person2D Pose Estimation using Part Affinity Fields, https://arxiv.org/abs/1611.08050. Custom tensor display and mean-field parity oracle for the small teaching split,not COCO AP.
/// @see DecodePafs(), ScorePafs()

inline nn::Tensor PafPreviewTiles(const PafData &data, const std::vector<PafDecoded> &decoded,
                                  const nn::Tensor &heat, const nn::Tensor &paf) {
  nn::NoGradGuard guard;
  if (data.rgb.dim() != 4 || data.rgb.size(0) < 1 ||
      data.rgb.shape() != nn::Shape{data.rgb.size(0), 128, 128, 3} || data.joints.dim() != 4 ||
      data.joints.shape() != nn::Shape{data.rgb.size(0), data.joints.size(1), 18, 3} ||
      data.counts.shape() != nn::Shape{data.rgb.size(0)} ||
      decoded.size() != std::size_t(data.rgb.size(0)) ||
      heat.shape() != nn::Shape{data.rgb.size(0), 16, 16, 19} ||
      paf.shape() != nn::Shape{data.rgb.size(0), 16, 16, 38} ||
      !nn::all(nn::is_finite(heat)).item<bool>() ||
      !nn::all(nn::is_finite(paf)).item<bool>() ||
      !nn::all(nn::is_finite(data.rgb)).item<bool>())
    throw std::invalid_argument("aligned finite full-image PAF previews required");
  const auto counts = data.counts.to_vector<std::int32_t>();
  std::vector<nn::Tensor> tiles;
  const auto green = nn::tensor({0., 1., 0.}), red = nn::tensor({1., 0., 0.});
  for (std::int64_t n = 0; n < data.rgb.size(0); ++n) {
    if (counts[n] < 0 || counts[n] > data.joints.size(1))
      throw std::invalid_argument("bounded preview person counts required");
    std::vector<nn::Tensor> gt;
    std::vector<float> metadata;
    for (int p = 0; p < counts[n]; ++p)
      for (const auto &edge : kPoseEdges) {
        const auto joints = nn::select(nn::select(data.joints, 0, n), 0, p),
                   a = nn::select(joints, 0, edge[0]), b = nn::select(joints, 0, edge[1]);
        if (nn::logical_and(nn::greater(nn::select(a, 0, 2), 0.),
                            nn::greater(nn::select(b, 0, 2), 0.))
                .item<bool>())
          gt.push_back(nn::stack({nn::slice_axis(a, 0, 0, 2), nn::slice_axis(b, 0, 0, 2)}, 0));
      }
    const auto &d = decoded[n];
    for (const auto &person : d.people)
      for (int id : person.links) {
        if (id < 0 || id >= int(d.links.size()))
          throw std::invalid_argument("bounded displayed link required");
        const auto &link = d.links[id];
        if (link.a < 0 || link.b < 0 || link.a >= int(d.peaks.size()) ||
            link.b >= int(d.peaks.size()))
          throw std::invalid_argument("bounded displayed peak required");
        const auto &a = d.peaks[link.a];
        const auto &b = d.peaks[link.b];
        metadata.insert(metadata.end(), {a.x, a.y, b.x, b.y});
      }
    const auto gm = gt.empty() ? nn::zeros({128, 128, 1}, nn::DType::Bool)
                               : paf_detail::Lines(nn::stack(gt, 0));
    const auto pm =
        metadata.empty()
            ? nn::zeros({128, 128, 1}, nn::DType::Bool)
            : paf_detail::Lines(
                  (nn::from_vector(metadata, {std::int64_t(metadata.size() / 4), 2, 2})
                       .to(nn::DType::Float64) +
                   .5) *
                      8. -
                  .5);
    const auto rgb =
        nn::floor(nn::clip(nn::select(data.rgb, 0, n).to(nn::DType::Float64), 0., 1.) * 255. +
                  .5) /
        255.;
    const auto h = nn::expand_dims(nn::select(heat, 0, n), 0),
               f = nn::select(paf, 0, n).to(nn::DType::Float64).reshape({1, 16, 16, 19, 2}),
               peak = nn::resize_nearest(
                   nn::clip(nn::max(nn::slice_axis(h, 3, 0, 18), {3}, true), 0., 1.),
                   {128, 128}),
               magnitude = nn::resize_nearest(
                   nn::clip(nn::max(nn::sqrt(nn::sum(nn::square(f), {4})), {3}, true), 0., 1.),
                   {128, 128}),
               zero = nn::zeros_like(peak);
    const auto hp = nn::select(nn::concat({peak, zero, zero}, 3), 0, 0).to(nn::DType::Float64),
               fp = nn::select(nn::concat({zero.to(nn::DType::Float64),
                                           zero.to(nn::DType::Float64), magnitude},
                                          3),
                               0, 0);
    tiles.push_back(
        nn::concat({rgb, nn::where(gm, green.to(nn::DType::Float64), rgb),
                    nn::where(pm, red.to(nn::DType::Float64), rgb),
                    nn::floor(hp * 255. + .5) / 255., nn::floor(fp * 255. + .5) / 255.},
                   1)
            .to(nn::DType::Float32));
  }
  return nn::stack(tiles, 0);
}
} // namespace datamunge::nn::models
