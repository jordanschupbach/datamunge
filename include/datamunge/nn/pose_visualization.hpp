#pragma once
/// @file pose_visualization.hpp
/// @brief Tensor skeleton overlays and raw-heatmap preview tiles for actual pose observations.
#include "datamunge/nn/pose_data.hpp"
#include <array>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Private metadata and distance-based skeleton rasterization helpers.
namespace pose_detail {
/// @brief COCO17 skeleton topology as18 zero-based anatomical edge pairs.
inline constexpr std::array<std::int64_t, 36> edges = {
    5,  7,  7,  9,  6,  8,  8, 10, 5, 6, 5, 11, 6, 12, 11, 12, 11, 13,
    13, 15, 12, 14, 14, 16, 0, 1,  1, 3, 0, 2,  2, 4,  0,  5,  0,  6};
/// @brief Rasterize point squares and continuous one-pixel-radius limb segments with tensors.
///
/// Rendering uses rounded3x3 point squares and geometric distance<=1 pixel
/// to continuous segments,not the original legacy sampled-line rasterizer.
/// This explicit display policy changes edge antialias geometry only;numeric
/// predictions,GT coordinates,targets and scores remain untouched.
/// @param joints Numeric[N,17,3] x/y/visibility-or-peak,finite wherever active.
/// @param annotated If true,visibility>0 gates joints and both edge endpoints.
/// @return Bool[N,128,128,1] skeleton coverage;no numeric host pixel loops.
inline nn::Tensor Skeleton(const nn::Tensor &joints, bool annotated) {
  if (joints.dim() != 3 || joints.size(0) < 1 ||
      joints.shape() != nn::Shape{joints.size(0), 17, 3})
    throw std::invalid_argument("COCO17 skeleton geometry required");
  const auto raw = joints.to(nn::DType::Float64),
             visible = annotated ? nn::greater(nn::slice_axis(raw, 2, 2, 3), 0.)
                                 : nn::ones({raw.size(0), 17, 1}, nn::DType::Bool),
             safe = nn::where(visible, nn::slice_axis(raw, 2, 0, 2), 0.);
  if (!nn::all(nn::is_finite(safe)).item<bool>())
    throw std::invalid_argument("finite active skeleton coordinates required");
  const auto x = nn::arange(0, 128, 1, nn::DType::Float64).reshape({1, 1, 1, 128}),
             y = nn::arange(0, 128, 1, nn::DType::Float64).reshape({1, 1, 128, 1});
  const auto px = nn::select(safe, 2, 0).reshape({raw.size(0), 17, 1, 1}),
             py = nn::select(safe, 2, 1).reshape({raw.size(0), 17, 1, 1});
  const auto rx = nn::where(nn::greater_equal(px, 0.), nn::floor(px + .5),
                            -nn::floor(-px + .5)),
             ry = nn::where(nn::greater_equal(py, 0.), nn::floor(py + .5),
                            -nn::floor(-py + .5));
  const auto dots = nn::logical_and(visible.reshape({raw.size(0), 17, 1, 1}),
                                    nn::logical_and(nn::less_equal(nn::abs(x - rx), 1.),
                                                    nn::less_equal(nn::abs(y - ry), 1.)));
  const auto indices = nn::from_vector(std::vector<std::int64_t>(edges.begin(), edges.end()),
                                       {18, 2}),
             first = nn::select(indices, 1, 0), second = nn::select(indices, 1, 1),
             a = nn::gather(safe, first, 1), b = nn::gather(safe, second, 1),
             both = nn::logical_and(nn::gather(visible, first, 1),
                                    nn::gather(visible, second, 1));
  const auto ax = nn::select(a, 2, 0).reshape({raw.size(0), 18, 1, 1}),
             ay = nn::select(a, 2, 1).reshape({raw.size(0), 18, 1, 1}),
             dx = (nn::select(b, 2, 0) - nn::select(a, 2, 0)).reshape({raw.size(0), 18, 1, 1}),
             dy = (nn::select(b, 2, 1) - nn::select(a, 2, 1)).reshape({raw.size(0), 18, 1, 1}),
             length = dx * dx + dy * dy;
  const auto t = nn::clip(((x - ax) * dx + (y - ay) * dy) /
                              nn::maximum(length, nn::full({}, 1e-30, nn::DType::Float64)),
                          0., 1.),
             distance = nn::square(x - (ax + t * dx)) + nn::square(y - (ay + t * dy));
  const auto lines =
      nn::logical_and(both.reshape({raw.size(0), 18, 1, 1}), nn::less_equal(distance, 1.));
  return nn::expand_dims(
      nn::logical_or(nn::greater(nn::sum(dots.to(nn::DType::Int32), {1}), 0.),
                     nn::greater(nn::sum(lines.to(nn::DType::Int32), {1}), 0.)),
      3);
}
} // namespace pose_detail
/// @brief Build actual RGB,annotated skeleton,predicted skeleton and heatmap preview tiles.
///
/// Actual input pixels retain source half-up byte display. GT visibility1/2
/// both render;unlabeled coordinates are ignored before geometry. Predicted
/// peaks are unthresholded,matching the source decoder contract. Maximum raw
/// heat clips only for red=v/blue=1-v display,nearest-upsampled4x. Skeleton
/// lines use continuous distance<=1 pixel with rounded3x3 point squares,an
/// explicit tensor rasterization update from legacy sampled lines. No host
/// pixel loops,SVG output,training or target-derived coordinate alignment.
///
/// @param data Original Float32[N,128,128,3] observations and[N,17,3] annotations.
/// @param decoded Finite[N,17,3] predicted x/y/peak;all predicted joints render regardless peak.
/// @param heatmap Finite[N,32,32,17] raw linear predictions.
/// @return Float32[N,128,512,3] observed|green GT|red prediction|red/blue heat maximum.
/// @throws std::invalid_argument For incompatible geometry or nonfinite active skeleton coordinates.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// m::PoseData d;
/// d.rgb = n::zeros({1, 128, 128, 3});
/// d.joints = n::zeros({1, 17, 3});
/// const auto h = n::zeros({1, 32, 32, 17});
/// const auto p = m::DecodePoseHeatmaps(h);
/// const auto tiles = m::PosePreviewTiles(d, p, h);
/// @endcode
/// @par References
/// - Lin et al.(2014),Microsoft COCO:Common Objects in Context, https://arxiv.org/abs/1405.0312. COCO17 skeleton topology follows the source example. Tensor display uses rounded3x3 joints and continuous segment-distance<=1 rasterization rather than the legacy sampled-line renderer;this does not affect predictions or metrics.
/// @see DecodePoseHeatmaps(), ScoreTensorPose()

inline nn::Tensor PosePreviewTiles(const PoseData &data, const nn::Tensor &decoded,
                                   const nn::Tensor &heatmap) {
  nn::NoGradGuard guard;
  if (data.rgb.dim() != 4 || data.rgb.size(0) < 1 ||
      data.rgb.shape() != nn::Shape{data.rgb.size(0), 128, 128, 3} ||
      data.joints.shape() != nn::Shape{data.rgb.size(0), 17, 3} ||
      decoded.shape() != data.joints.shape() ||
      heatmap.shape() != nn::Shape{data.rgb.size(0), 32, 32, 17})
    throw std::invalid_argument("aligned128px COCO17 previews required");
  const auto rgb = nn::floor(nn::clip(data.rgb.to(nn::DType::Float64), 0., 1.) * 255. + .5) /
                   255.,
             gt = pose_detail::Skeleton(data.joints, true),
             pred = pose_detail::Skeleton(decoded, false),
             green = nn::from_vector(std::vector<double>{0., 1., 0.}, {3}),
             red = nn::from_vector(std::vector<double>{1., 0., 0.}, {3});
  const auto maximum = nn::clip(nn::max(heatmap.to(nn::DType::Float32), {3}, true), 0., 1.),
             v = nn::resize_nearest(maximum, {128, 128}), zero = nn::zeros_like(v);
  const auto heat = nn::concat({v, zero, 1. - v}, 3).to(nn::DType::Float64);
  return nn::concat({rgb, nn::where(gt, green, rgb), nn::where(pred, red, rgb),
                     nn::floor(255. * heat + .5) / 255.},
                    2)
      .to(nn::DType::Float32);
}
} // namespace datamunge::nn::models
