#pragma once
/// @file flow_visualization.hpp
/// @brief Tensor HSV direction colors and observed/GT/prediction/warp optical-flow previews.
#include "datamunge/nn/flow_data.hpp"
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Map signed dx/dy fields to the source teaching HSV direction colors.
///
/// Hue follows atan2(dy,dx);positive image y points down. Saturation is
/// min(1,magnitude/4),value1. This preserves the source HSV mapping,not the
/// official Middlebury color wheel. All numeric work uses tensors;no fitting.
///
/// @param field Finite numeric[...,2] dx/dy,in input-image pixel units.
/// @return Float64[...,3] RGB in[0,1],white at zero displacement.
/// @throws std::invalid_argument For invalid geometry or nonfinite flow/color inputs.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto colors = n::models::FlowColors(n::tensor({4., 0., 0., 4., -4., 0.}, {3, 2}));
/// @endcode
/// @par References
/// - Middlebury public optical-flow data and evaluation, https://vision.middlebury.edu/flow/data/. The source teaching preview uses an HSV direction wheel with saturation min(1,magnitude/4px),not the official Middlebury color-wheel implementation.
/// @see FlowWarp(), FlowColor(), FlowPreviewTiles()
inline nn::Tensor FlowColors(const nn::Tensor &field) {
  if (field.dim() < 1 || field.shape().back() != 2 || field.numel() < 1 ||
      !nn::all(nn::is_finite(field)).item<bool>())
    throw std::invalid_argument("finite dx/dy color field required");
  const auto x = field.to(nn::DType::Float64);
  const auto axis = x.dim() - 1;
  const auto u = nn::slice_axis(x, axis, 0, 1), v = nn::slice_axis(x, axis, 1, 2);
  const auto saturation = nn::minimum(nn::ones_like(u), nn::sqrt(u * u + v * v) / 4.);
  const auto hue = (nn::atan2(v, u) / 6.2831853071795864769 + 1.) * 6.,
             integer = nn::floor(hue), part = hue - integer,
             sector = integer - 6. * nn::floor(integer / 6.);
  const auto p = 1. - saturation, q = 1. - saturation * part,
             t = 1. - saturation * (1. - part), one = nn::ones_like(u);
  const auto r =
      nn::where(nn::equal(sector, 0.), one,
                nn::where(nn::equal(sector, 1.), q,
                          nn::where(nn::equal(sector, 2.), p,
                                    nn::where(nn::equal(sector, 3.), p,
                                              nn::where(nn::equal(sector, 4.), t, one)))));
  const auto g =
      nn::where(nn::equal(sector, 0.), t,
                nn::where(nn::equal(sector, 1.), one,
                          nn::where(nn::equal(sector, 2.), one,
                                    nn::where(nn::equal(sector, 3.), q,
                                              nn::where(nn::equal(sector, 4.), p, p)))));
  const auto b =
      nn::where(nn::equal(sector, 0.), p,
                nn::where(nn::equal(sector, 1.), p,
                          nn::where(nn::equal(sector, 2.), t,
                                    nn::where(nn::equal(sector, 3.), one,
                                              nn::where(nn::equal(sector, 4.), one, q)))));
  return nn::concat({r, g, b}, axis);
}
/// @brief Present one tensor HSV flow color as three host scalars.
///
/// Compatibility ingress/egress only;no host trigonometry or color arithmetic.
///
/// @param u Finite input-pixel horizontal displacement.
/// @param v Finite input-pixel vertical displacement.
/// @return Three Double RGB values from FlowColors.
/// @throws std::invalid_argument For invalid geometry or nonfinite flow/color inputs.
/// @par Example
/// @code
/// const auto rgb = dm::nn::models::FlowColor(4., 0.); // red
/// @endcode
/// @par References
/// - Middlebury public optical-flow data and evaluation, https://vision.middlebury.edu/flow/data/. The source teaching preview uses an HSV direction wheel with saturation min(1,magnitude/4px),not the official Middlebury color-wheel implementation.
/// @see FlowWarp(), FlowColor(), FlowPreviewTiles()
inline std::array<double, 3> FlowColor(double u, double v) {
  const auto color =
      FlowColors(nn::from_vector(std::vector<double>{u, v}, {2})).to_vector<double>();
  return {color[0], color[1], color[2]};
}
/// @brief Render observed pair,true/predicted colors,EPE and warped second image.
///
/// Columns:first,second,GT HSV,predicted HSV,EPE/5 in red,warped second.
/// Invalid GT colors/errors are black. Float64 quantization preserves source
/// lround(255*x). Display clipping never changes numeric score arrays. All pixel
/// computation is tensor based;save_image_grid handles layout/scaling.
///
/// @param data Aligned RGB pair,true input-pixel flow and binary GT mask.
/// @param prediction Matching finite dx/dy estimate,in input pixels.
/// @param warped Matching RGB second frame sampled into first-frame coordinates.
/// @return Float32[N,H,6W,3] byte-quantized display tiles in[0,1].
/// @throws std::invalid_argument For invalid geometry or nonfinite flow/color inputs.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const n::models::FlowData d{n::zeros({1, 3, 4, 3}),         n::ones({1, 3, 4, 3}),
///                             n::zeros({1, 3, 4, 2}),         n::ones({1, 3, 4, 1}),
///                             n::zeros({1}, n::DType::Int32), n::zeros({1}, n::DType::Int32)};
/// const auto tiles = n::models::FlowPreviewTiles(d, d.flow, d.second);
/// @endcode
/// @par References
/// - Middlebury public optical-flow data and evaluation, https://vision.middlebury.edu/flow/data/. The source teaching preview uses an HSV direction wheel with saturation min(1,magnitude/4px),not the official Middlebury color-wheel implementation.
/// @see FlowWarp(), FlowColor(), FlowPreviewTiles()
inline nn::Tensor FlowPreviewTiles(const FlowData &data, const nn::Tensor &prediction,
                                   const nn::Tensor &warped) {
  if (data.first.dim() != 4 || data.first.size(0) < 1 || data.first.size(1) < 1 ||
      data.first.size(2) < 1 || data.first.size(3) != 3 ||
      data.second.shape() != data.first.shape() || warped.shape() != data.first.shape() ||
      data.flow.shape() !=
          nn::Shape{data.first.size(0), data.first.size(1), data.first.size(2), 2} ||
      prediction.shape() != data.flow.shape() ||
      data.valid.shape() !=
          nn::Shape{data.first.size(0), data.first.size(1), data.first.size(2), 1})
    throw std::invalid_argument("aligned RGB/flow preview required");
  const auto p = prediction.to(nn::DType::Float64), truth = data.flow.to(nn::DType::Float64),
             mask = nn::greater(data.valid, 0.);
  const auto gt = nn::where(mask, FlowColors(truth), 0.);
  const auto color = FlowColors(p);
  const auto error =
      nn::where(mask, nn::sqrt(nn::sum(nn::square(p - truth), {3}, true)) / 5., 0.) *
      nn::from_vector(std::vector<double>{1., 0., 0.}, {1, 1, 1, 3});
  const auto tiles =
      nn::concat({data.first.to(nn::DType::Float64), data.second.to(nn::DType::Float64), gt,
                  color, error, warped.to(nn::DType::Float64)},
                 2);
  return (nn::floor(nn::clip(tiles, 0., 1.) * 255. + .5) / 255.).to(nn::DType::Float32);
}
} // namespace datamunge::nn::models
