#pragma once
/// @file pose_data.hpp
/// @brief Tensor COCO17 pose data,masked Gaussian supervision,deterministic decoding and joint-weighted PCK.
#include "datamunge/nn/checkpoint.hpp"
#include "datamunge/nn/hourglass_models.hpp"
#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/training.hpp"
#include <filesystem>
#include <map>
#include <set>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Own original COCO17 oracle person observations,targets and provenance.
///
/// Tensor handles own/share storage. IDs,annotations and boxes never enter the
/// RGB-only network forward. Fitting mirrors keep original source identities.
///
/// @throws std::invalid_argument For invalid geometry,dtype,configuration,eligible coordinates or missing annotated support.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const m::PoseData d{};
/// @endcode
/// @par References
/// - Newell,Yang and Deng(2016),Stacked Hourglass Networks for Human Pose Estimation, https://arxiv.org/abs/1603.06937;official small COCO8-Pose source, https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8-pose.yaml. Custom source-image-disjoint oracle crops and PCK@.2 maximum-box-side are teaching diagnostics,not COCO keypoint AP/MPII benchmark accuracy.
/// @see PoseHeatmaps(), PoseHeatmapMSE(), DecodePoseHeatmaps(), ScoreTensorPose()
struct PoseData {
  nn::Tensor rgb;       ///< Float32[N,128,128,3] original observed RGB unit values.
  nn::Tensor joints;    ///< Float32[N,17,3] crop x/y pixels and original visibility0/1/2.
  nn::Tensor box_size;  ///< Float32[N,2] original person box extent in crop pixel units.
  nn::Tensor image_ids; ///< Int32[N] original source-image identity.
  nn::Tensor
      person_ids; ///< Int32[N] original source-person identity,repeated by fitting mirrors.
};
/// @brief Own source-image-disjoint custom COCO8-Pose person-crop splits.
///
/// Multiple people from the same photograph are dependent observations.
/// Oracle boxes are preparation inputs,not detected boxes or detector accuracy.
///
/// @throws std::invalid_argument For invalid geometry,dtype,configuration,eligible coordinates or missing annotated support.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const m::PoseSplits d{};
/// @endcode
/// @par References
/// - Newell,Yang and Deng(2016),Stacked Hourglass Networks for Human Pose Estimation, https://arxiv.org/abs/1603.06937;official small COCO8-Pose source, https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8-pose.yaml. Custom source-image-disjoint oracle crops and PCK@.2 maximum-box-side are teaching diagnostics,not COCO keypoint AP/MPII benchmark accuracy.
/// @see PoseHeatmaps(), PoseHeatmapMSE(), DecodePoseHeatmaps(), ScoreTensorPose()
struct PoseSplits {
  PoseData fit; ///< Twelve fitting crops,six people/three source images including mirrors.
  PoseData validation; ///< One person from a separate source image.
  PoseData test;       ///< Fourteen people from four held-out source images.
};
/// @brief Read a named pose tensor archive with unique member names.
///
/// No normalization,target generation or fitting. Missing/unreadable archives
/// may throw I/O errors;duplicate names are rejected.
///
/// @param file Readable NPZ path.
/// @return Owning name-to-tensor map.
/// @throws std::invalid_argument For invalid geometry,dtype,configuration,eligible coordinates or missing annotated support.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// try {
///   const auto a = m::PoseArrays("missing-pose-doc-example.npz");
/// } catch (const std::exception &) {
/// }
/// @endcode
/// @par References
/// - Newell,Yang and Deng(2016),Stacked Hourglass Networks for Human Pose Estimation, https://arxiv.org/abs/1603.06937;official small COCO8-Pose source, https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8-pose.yaml. Custom source-image-disjoint oracle crops and PCK@.2 maximum-box-side are teaching diagnostics,not COCO keypoint AP/MPII benchmark accuracy.
/// @see PoseHeatmaps(), PoseHeatmapMSE(), DecodePoseHeatmaps(), ScoreTensorPose()
inline std::map<std::string, nn::Tensor> PoseArrays(const std::filesystem::path &file) {
  std::map<std::string, nn::Tensor> arrays;
  for (const auto &[name, value] : nn::load_npz(file.string()))
    if (!arrays.emplace(name, value).second)
      throw std::invalid_argument("duplicate pose archive member");
  return arrays;
}
/// @brief Gather arbitrary-dtype pose data or provenance by leading-row metadata.
///
/// Host loops validate/serialize index metadata only;no numeric image loops.
///
/// @param input Tensor with leading observation axis.
/// @param rows Bounded source rows;duplicates,reordering and empty output allowed.
/// @return Contiguous gathered tensor,preserving dtype and graph.
/// @throws std::invalid_argument For invalid geometry,dtype,configuration,eligible coordinates or missing annotated support.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto b = m::PoseBatch(n::tensor({7, 13, 19}, {3}, n::DType::Int32), {2, 0});
/// @endcode
/// @par References
/// - Newell,Yang and Deng(2016),Stacked Hourglass Networks for Human Pose Estimation, https://arxiv.org/abs/1603.06937;official small COCO8-Pose source, https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8-pose.yaml. Custom source-image-disjoint oracle crops and PCK@.2 maximum-box-side are teaching diagnostics,not COCO keypoint AP/MPII benchmark accuracy.
/// @see PoseHeatmaps(), PoseHeatmapMSE(), DecodePoseHeatmaps(), ScoreTensorPose()
inline nn::Tensor PoseBatch(const nn::Tensor &input, const std::vector<std::size_t> &rows) {
  if (input.dim() < 1)
    throw std::invalid_argument("pose row axis required");
  std::vector<std::int64_t> ids;
  for (auto row : rows) {
    if (row >= static_cast<std::size_t>(input.size(0)))
      throw std::invalid_argument("pose row bounds");
    ids.push_back(static_cast<std::int64_t>(row));
  }
  return nn::take_rows(input, ids).contiguous();
}
/// @brief Gather observed crops,joints,box scale and original IDs together.
///
/// No geometric transformation,annotation generation or fitting.
///
/// @param input Aligned person-crop tensors.
/// @param rows Bounded requested row metadata.
/// @return PoseData in caller-specified order.
/// @throws std::invalid_argument For invalid geometry,dtype,configuration,eligible coordinates or missing annotated support.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// m::PoseData d{n::zeros({2, 128, 128, 3}), n::zeros({2, 17, 3}), n::ones({2, 2}),
///               n::zeros({2}, n::DType::Int32), n::zeros({2}, n::DType::Int32)};
/// const auto b = m::PoseBatch(d, {1});
/// @endcode
/// @par References
/// - Newell,Yang and Deng(2016),Stacked Hourglass Networks for Human Pose Estimation, https://arxiv.org/abs/1603.06937;official small COCO8-Pose source, https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8-pose.yaml. Custom source-image-disjoint oracle crops and PCK@.2 maximum-box-side are teaching diagnostics,not COCO keypoint AP/MPII benchmark accuracy.
/// @see PoseHeatmaps(), PoseHeatmapMSE(), DecodePoseHeatmaps(), ScoreTensorPose()
inline PoseData PoseBatch(const PoseData &input, const std::vector<std::size_t> &rows) {
  return {PoseBatch(input.rgb, rows), PoseBatch(input.joints, rows),
          PoseBatch(input.box_size, rows), PoseBatch(input.image_ids, rows),
          PoseBatch(input.person_ids, rows)};
}
/// @brief Load validated12/1/14 original person crops with source-image-disjoint splits.
///
/// Checks all shapes/dtypes,RGB range,visibility0/1/2,finite labeled positions
/// within0..127,nonempty annotated support and positive finite original-box
/// normalization. Image IDs may repeat inside a split for people/mirrors,but
/// never cross splits. Numeric validation cannot establish annotation provenance.
///
/// @param directory Prepared poses.npz and source provenance location.
/// @return Fitting/validation/test crop tensors.
/// @throws std::invalid_argument For invalid geometry,dtype,configuration,eligible coordinates or missing annotated support.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// try {
///   const auto d = m::LoadPoseCrops("missing-pose-doc-example");
/// } catch (const std::exception &) {
/// }
/// @endcode
/// @par References
/// - Newell,Yang and Deng(2016),Stacked Hourglass Networks for Human Pose Estimation, https://arxiv.org/abs/1603.06937;official small COCO8-Pose source, https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8-pose.yaml. Custom source-image-disjoint oracle crops and PCK@.2 maximum-box-side are teaching diagnostics,not COCO keypoint AP/MPII benchmark accuracy.
/// @see PoseHeatmaps(), PoseHeatmapMSE(), DecodePoseHeatmaps(), ScoreTensorPose()
inline PoseSplits LoadPoseCrops(const std::filesystem::path &directory) {
  const auto arrays = PoseArrays(directory / "poses.npz");
  std::set<int> previous;
  /// @brief Read one exact-size split and reject overlapping original source images.
  /// @param split Named archive prefix.
  /// @param count Required crop count.
  /// @return Validated RGB/person target/provenance handles.
  auto read = [&](const std::string &split, std::int64_t count) {
    PoseData d{arrays.at(split + "_rgb"), arrays.at(split + "_joints"),
               arrays.at(split + "_box_size"), arrays.at(split + "_image_ids"),
               arrays.at(split + "_person_ids")};
    if (d.rgb.shape() != nn::Shape{count, 128, 128, 3} ||
        d.joints.shape() != nn::Shape{count, 17, 3} ||
        d.box_size.shape() != nn::Shape{count, 2} || d.image_ids.shape() != nn::Shape{count} ||
        d.person_ids.shape() != nn::Shape{count})
      throw std::invalid_argument("pose128px COCO17 geometry invalid");
    for (const auto *t : {&d.rgb, &d.joints, &d.box_size})
      if (t->dtype() != nn::DType::Float32)
        throw std::invalid_argument("Float32 pose pixels/coordinates required");
    for (const auto *t : {&d.image_ids, &d.person_ids})
      if (t->dtype() != nn::DType::Int32)
        throw std::invalid_argument("Int32 pose IDs required");
    const auto v = nn::slice_axis(d.joints, 2, 2, 3), xy = nn::slice_axis(d.joints, 2, 0, 2),
               known = nn::greater(v, 0.);
    if (!nn::all(nn::logical_or(nn::equal(v, 0.),
                                nn::logical_or(nn::equal(v, 1.), nn::equal(v, 2.))))
             .item<bool>() ||
        !nn::all(nn::where(known,
                           nn::logical_and(nn::is_finite(xy),
                                           nn::logical_and(nn::greater_equal(xy, 0.),
                                                           nn::less_equal(xy, 127.))),
                           nn::ones_like(xy).to(nn::DType::Bool)))
             .item<bool>() ||
        !nn::all(nn::greater(nn::sum(known.to(nn::DType::Int32), {1, 2}), 0.)).item<bool>())
      throw std::invalid_argument(
          "COCO visibility and bounded annotated crop joints required");
    if (!nn::all(nn::logical_and(nn::is_finite(d.box_size), nn::greater(d.box_size, 0.)))
             .item<bool>() ||
        !nn::all(nn::logical_and(nn::greater_equal(d.rgb, 0.), nn::less_equal(d.rgb, 1.)))
             .item<bool>())
      throw std::invalid_argument("positive finite crop box sizes and unit RGB required");
    std::set<int> ids;
    for (auto id : d.image_ids.to_vector<int>()) {
      if (id < 0 || previous.contains(id))
        throw std::invalid_argument("pose source images cross splits");
      ids.insert(id);
    }
    for (auto id : d.person_ids.to_vector<int>())
      if (id < 0)
        throw std::invalid_argument("negative source-person ID");
    previous.insert(ids.begin(), ids.end());
    return d;
  };
  auto fit = read("fit", 12), validation = read("validation", 1), test = read("test", 14);
  return {fit, validation, test};
}
/// @brief Own fixed per-joint heatmaps and their annotation support.
///
/// Visible and occluded labels both supervise;missing/out-of-image joints
/// produce zero channels. Target construction is detached from optimization.
///
/// @throws std::invalid_argument For invalid geometry,dtype,configuration,eligible coordinates or missing annotated support.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto t = m::PoseHeatmaps(n::tensor({41.5, 81.5, 2.}, {1, 1, 3}));
/// @endcode
/// @par References
/// - Newell,Yang and Deng(2016),Stacked Hourglass Networks for Human Pose Estimation, https://arxiv.org/abs/1603.06937;official small COCO8-Pose source, https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8-pose.yaml. Custom source-image-disjoint oracle crops and PCK@.2 maximum-box-side are teaching diagnostics,not COCO keypoint AP/MPII benchmark accuracy.
/// @see PoseHeatmaps(), PoseHeatmapMSE(), DecodePoseHeatmaps(), ScoreTensorPose()
struct PoseTargets {
  nn::Tensor heatmap; ///< Float32[N,side,side,J] unnormalized truncated Gaussians.
  nn::Tensor mask;    ///< Float32[N,1,1,J] binary eligible original annotation support.
};
/// @brief Prepare detached half-pixel-centered,truncated Gaussian joint targets.
///
/// Uses source Float32 annotation ingress then Double centers(x+.5)*side/input-.5.
/// A square window truncates each axis at3*sigma,including boundaries.
/// Sigma radius/variance follow source Float32 coefficient rounding. Invalid
/// NaN coordinates select safe values BEFORE arithmetic and yield zero channels.
/// Target generation has no trainable leaves or optimizer.
///
/// @param joints Numeric[N,J,3] input-pixel x/y and original visibility.
/// @param side Positive target side,default32.
/// @param input_size Positive observed square side,default128.
/// @param sigma Positive finite Float32 standard deviation in target pixels,default1.
/// @return Unnormalized Gaussian channels and per-image/per-joint annotation mask.
/// @throws std::invalid_argument For invalid geometry,dtype,configuration,eligible coordinates or missing annotated support.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto t = m::PoseHeatmaps(n::tensor({41.5, 81.5, 2.}, {1, 1, 3}));
/// @endcode
/// @par References
/// - Newell,Yang and Deng(2016),Stacked Hourglass Networks for Human Pose Estimation, https://arxiv.org/abs/1603.06937;official small COCO8-Pose source, https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8-pose.yaml. Custom source-image-disjoint oracle crops and PCK@.2 maximum-box-side are teaching diagnostics,not COCO keypoint AP/MPII benchmark accuracy.
/// @see PoseHeatmaps(), PoseHeatmapMSE(), DecodePoseHeatmaps(), ScoreTensorPose()
inline PoseTargets PoseHeatmaps(const nn::Tensor &joints, int side = 32, int input_size = 128,
                                float sigma = 1.f) {
  nn::NoGradGuard guard;
  if (joints.dim() != 3 || joints.size(0) < 1 || joints.size(1) < 1 || joints.size(2) != 3 ||
      side < 1 || input_size < 1 || !std::isfinite(sigma) || sigma <= 0 ||
      !std::isfinite(2.f * sigma * sigma) || 2.f * sigma * sigma <= 0)
    throw std::invalid_argument("valid Gaussian pose target configuration required");
  const auto raw = joints.to(nn::DType::Float32), xy = nn::slice_axis(raw, 2, 0, 2),
             v = nn::slice_axis(raw, 2, 2, 3);
  const auto eligible = nn::logical_and(
      nn::greater(v, 0.),
      nn::all(nn::logical_and(nn::is_finite(xy),
                              nn::logical_and(nn::greater_equal(xy, 0.),
                                              nn::less_equal(xy, double(input_size - 1)))),
              {2}, true));
  const auto safe = nn::where(eligible, xy, 0.).to(nn::DType::Float64),
             centers = (safe + .5) * double(side) / double(input_size) - .5;
  const auto cx = nn::slice_axis(centers, 2, 0, 1).reshape({raw.size(0), 1, 1, raw.size(1)}),
             cy = nn::slice_axis(centers, 2, 1, 2).reshape({raw.size(0), 1, 1, raw.size(1)}),
             x = nn::arange(0, side, 1, nn::DType::Float64).reshape({1, 1, side, 1}),
             y = nn::arange(0, side, 1, nn::DType::Float64).reshape({1, side, 1, 1});
  const auto dx = x - cx, dy = y - cy,
             mask = eligible.reshape({raw.size(0), 1, 1, raw.size(1)});
  const auto window =
      nn::logical_and(mask, nn::logical_and(nn::less_equal(nn::abs(dx), double(3.f * sigma)),
                                            nn::less_equal(nn::abs(dy), double(3.f * sigma))));
  return {nn::where(window, nn::exp(-(dx * dx + dy * dy) / double(2.f * sigma * sigma)), 0.)
              .to(nn::DType::Float32),
          mask.to(nn::DType::Float32)};
}
/// @brief Compute equal-image masked joint-channel MSE with safe missing values.
///
/// Selects inactive predictions/targets to0 BEFORE subtraction,so missing NaNs
/// have zero derivatives. Images without annotations are excluded;all-empty
/// input returns graph-connected0. Eligible NaNs propagate as invalid losses.
/// No target generation,BN mutation or optimizer step is hidden.
///
/// @param prediction Floating[N,side,side,J] raw linear heatmaps.
/// @param target Matching Gaussian targets.
/// @param mask Numeric[N,1,1,J],active when>.5,detached.
/// @param side Positive shared heatmap side.
/// @return Scalar mean of each supported image’s per-annotated-channel/per-pixel MSE.
/// @throws std::invalid_argument For invalid geometry,dtype,configuration,eligible coordinates or missing annotated support.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto x = n::ones({2, 4, 4, 3});
/// const auto loss = m::PoseHeatmapMSE(x, n::zeros_like(x), n::ones({2, 1, 1, 3}), 4);
/// @endcode
/// @par References
/// - Newell,Yang and Deng(2016),Stacked Hourglass Networks for Human Pose Estimation, https://arxiv.org/abs/1603.06937;official small COCO8-Pose source, https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8-pose.yaml. Custom source-image-disjoint oracle crops and PCK@.2 maximum-box-side are teaching diagnostics,not COCO keypoint AP/MPII benchmark accuracy.
/// @see PoseHeatmaps(), PoseHeatmapMSE(), DecodePoseHeatmaps(), ScoreTensorPose()
inline nn::Tensor PoseHeatmapMSE(const nn::Tensor &prediction, const nn::Tensor &target,
                                 const nn::Tensor &mask, int side) {
  if (prediction.dim() != 4 || prediction.size(0) < 1 || side < 1 ||
      prediction.shape() != target.shape() || prediction.size(1) != side ||
      prediction.size(2) != side ||
      mask.shape() != nn::Shape{prediction.size(0), 1, 1, prediction.size(3)})
    throw std::invalid_argument("aligned pose head/target/annotation mask required");
  const auto active = nn::greater(mask.detach().to(nn::DType::Float64), .5),
             p = nn::where(active, prediction, 0.), t = nn::where(active, target, 0.),
             count = nn::sum(active.to(prediction.dtype()), {1, 2, 3}),
             present = nn::greater(count, 0.).to(prediction.dtype());
  const auto error = nn::sum(nn::square(p - t), {1, 2, 3}) /
                     nn::maximum(count * double(side * side), nn::ones_like(count));
  return nn::sum(error * present) /
         nn::maximum(nn::sum(present), nn::scalar_like(1., prediction));
}
/// @brief Decode the first row-major maximum of each linear joint heatmap.
///
/// Ties select the first spatial position. Source Float32 inverse half-pixel
/// centers map128/32 first index to1.5. No sigmoid,threshold,GT input or
/// coordinate fitting;even negative peaks decode a position.
///
/// @param heatmaps Finite numeric[N,H,H,J] raw heatmaps.
/// @param input_size Positive observed square side,default128.
/// @return Detached Float32[N,J,3] input-pixel x/y and unbounded peak.
/// @throws std::invalid_argument For invalid geometry,dtype,configuration,eligible coordinates or missing annotated support.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto p = m::DecodePoseHeatmaps(n::zeros({1, 32, 32, 17}));
/// @endcode
/// @par References
/// - Newell,Yang and Deng(2016),Stacked Hourglass Networks for Human Pose Estimation, https://arxiv.org/abs/1603.06937;official small COCO8-Pose source, https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8-pose.yaml. Custom source-image-disjoint oracle crops and PCK@.2 maximum-box-side are teaching diagnostics,not COCO keypoint AP/MPII benchmark accuracy.
/// @see PoseHeatmaps(), PoseHeatmapMSE(), DecodePoseHeatmaps(), ScoreTensorPose()
inline nn::Tensor DecodePoseHeatmaps(const nn::Tensor &heatmaps, int input_size = 128) {
  nn::NoGradGuard guard;
  if (heatmaps.dim() != 4 || heatmaps.size(0) < 1 || heatmaps.size(1) < 1 ||
      heatmaps.size(1) != heatmaps.size(2) || heatmaps.size(3) < 1 || input_size < 1 ||
      !nn::all(nn::is_finite(heatmaps)).item<bool>())
    throw std::invalid_argument("finite square pose heatmaps required");
  const auto h = heatmaps.to(nn::DType::Float32),
             flat = nn::transpose(h, {0, 3, 1, 2})
                        .reshape({h.size(0), h.size(3), h.size(1) * h.size(1)}),
             index = nn::argmax(flat, 2).to(nn::DType::Float32),
             row = nn::floor(index / double(h.size(1))),
             column = index - row * double(h.size(1));
  const auto x = (column + .5) * double(input_size) / double(h.size(1)) - .5,
             y = (row + .5) * double(input_size) / double(h.size(1)) - .5,
             peak = nn::max(flat, {2});
  return nn::stack({x, y, peak}, 2);
}
/// @brief Own joint-weighted pose diagnostics. Detached tensors.
///
/// All annotated joints have equal weight across images. This is not
/// per-image averaging or COCO OKS/AP;predicted confidence is not a metric gate.
///
/// @throws std::invalid_argument For invalid geometry,dtype,configuration,eligible coordinates or missing annotated support.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// m::PoseData d;
/// d.joints = n::tensor({0., 0., 1.}, {1, 1, 3});
/// d.box_size = n::tensor({25., 20.}, {1, 2});
/// const auto score = m::ScoreTensorPose(n::tensor({3., 4., 0.}, {1, 1, 3}), d);
/// @endcode
/// @par References
/// - Newell,Yang and Deng(2016),Stacked Hourglass Networks for Human Pose Estimation, https://arxiv.org/abs/1603.06937;official small COCO8-Pose source, https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8-pose.yaml. Custom source-image-disjoint oracle crops and PCK@.2 maximum-box-side are teaching diagnostics,not COCO keypoint AP/MPII benchmark accuracy.
/// @see PoseHeatmaps(), PoseHeatmapMSE(), DecodePoseHeatmaps(), ScoreTensorPose()
struct TensorPoseMetrics {
  nn::Tensor pck;              ///< Inclusive PCK@.2 of maximum original-box side.
  nn::Tensor normalized_error; ///< Mean annotated-joint pixel error/max-box-side.
  nn::Tensor pixels;           ///< Mean annotated-joint Euclidean error,crop pixels.
  nn::Tensor joints;           ///< Eligible annotated-joint count.
};
/// @brief Own joint-weighted pose diagnostics. Host presentation.
///
/// All annotated joints have equal weight across images. This is not
/// per-image averaging or COCO OKS/AP;predicted confidence is not a metric gate.
///
/// @throws std::invalid_argument For invalid geometry,dtype,configuration,eligible coordinates or missing annotated support.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// m::PoseData d;
/// d.joints = n::tensor({0., 0., 1.}, {1, 1, 3});
/// d.box_size = n::tensor({25., 20.}, {1, 2});
/// const auto score = m::ScorePose(n::tensor({3., 4., 0.}, {1, 1, 3}), d);
/// @endcode
/// @par References
/// - Newell,Yang and Deng(2016),Stacked Hourglass Networks for Human Pose Estimation, https://arxiv.org/abs/1603.06937;official small COCO8-Pose source, https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8-pose.yaml. Custom source-image-disjoint oracle crops and PCK@.2 maximum-box-side are teaching diagnostics,not COCO keypoint AP/MPII benchmark accuracy.
/// @see PoseHeatmaps(), PoseHeatmapMSE(), DecodePoseHeatmaps(), ScoreTensorPose()
struct PoseMetrics {
  double pck = 0;              ///< Inclusive PCK@.2 of maximum original-box side.
  double normalized_error = 0; ///< Mean annotated-joint pixel error/max-box-side.
  double pixels = 0;           ///< Mean annotated-joint Euclidean error,crop pixels.
  int joints = 0;              ///< Eligible annotated-joint count.
};
/// @brief Score joint-weighted inclusive PCK and normalized/pixel errors with tensor reductions.
///
/// Source Float32 coordinate ingress precedes Double differences and Euclidean
/// distance. Visibility1 and2 both count;missing joint NaNs select safe values
/// before differences. Count all annotated joints equally,not each crop equally.
/// PCK boundary distance<=.2*max(boxwidth,boxheight) is inclusive. Total empty
/// annotations throw. No GT alignment,confidence filtering or fitted thresholds.
///
/// @param prediction Numeric[N,J,3] decoded x/y/peak;only coordinates are scored.
/// @param data Original Float32 joints and positive original-box extents in crop units.
/// @return Detached Float64 PCK,NME,mean pixels and Int64 joint count.
/// @throws std::invalid_argument For invalid geometry,dtype,configuration,eligible coordinates or missing annotated support.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// m::PoseData d;
/// d.joints = n::tensor({0., 0., 1.}, {1, 1, 3});
/// d.box_size = n::tensor({25., 20.}, {1, 2});
/// const auto score = m::ScoreTensorPose(n::tensor({3., 4., 0.}, {1, 1, 3}), d);
/// @endcode
/// @par References
/// - Newell,Yang and Deng(2016),Stacked Hourglass Networks for Human Pose Estimation, https://arxiv.org/abs/1603.06937;official small COCO8-Pose source, https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8-pose.yaml. Custom source-image-disjoint oracle crops and PCK@.2 maximum-box-side are teaching diagnostics,not COCO keypoint AP/MPII benchmark accuracy.
/// @see PoseHeatmaps(), PoseHeatmapMSE(), DecodePoseHeatmaps(), ScoreTensorPose()
inline TensorPoseMetrics ScoreTensorPose(const nn::Tensor &prediction, const PoseData &data) {
  nn::NoGradGuard guard;
  if (prediction.dim() != 3 || prediction.shape() != data.joints.shape() ||
      prediction.size(0) < 1 || prediction.size(1) < 1 || prediction.size(2) != 3 ||
      data.box_size.shape() != nn::Shape{prediction.size(0), 2})
    throw std::invalid_argument("aligned decoded pose/box geometry required");
  const auto truth = data.joints.to(nn::DType::Float32), p = prediction.to(nn::DType::Float32),
             known = nn::greater(nn::select(truth, 2, 2), 0.),
             pxy = nn::where(known.reshape({p.size(0), p.size(1), 1}),
                             nn::slice_axis(p, 2, 0, 2), 0.)
                       .to(nn::DType::Float64),
             txy = nn::where(known.reshape({p.size(0), p.size(1), 1}),
                             nn::slice_axis(truth, 2, 0, 2), 0.)
                       .to(nn::DType::Float64),
             scale = nn::max(data.box_size.to(nn::DType::Float32).to(nn::DType::Float64), {1})
                         .reshape({p.size(0), 1}),
             distance = nn::sqrt(nn::sum(nn::square(pxy - txy), {2})),
             count = nn::count_nonzero(known).to(nn::DType::Int64);
  if (count.item<std::int64_t>() < 1 || !nn::all(nn::is_finite(distance)).item<bool>() ||
      !nn::all(nn::logical_and(nn::is_finite(scale), nn::greater(scale, 0.))).item<bool>())
    throw std::invalid_argument(
        "annotated finite pose errors and positive box scale required");
  const auto weight = known.to(nn::DType::Float64), denominator = count.to(nn::DType::Float64);
  return {nn::sum(nn::less_equal(distance, .2 * scale).to(nn::DType::Float64) * weight) /
              denominator,
          nn::sum(distance / scale * weight) / denominator,
          nn::sum(distance * weight) / denominator, count};
}
/// @brief Present tensor pose diagnostics as legacy host scalars.
///
/// All numeric work delegates to ScoreTensorPose.
///
/// @param prediction Decoded coordinate/peak tensor.
/// @param data Original annotations and box extents.
/// @return Host PCK,NME,mean pixel error and joint count.
/// @throws std::invalid_argument For invalid geometry,dtype,configuration,eligible coordinates or missing annotated support.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// m::PoseData d;
/// d.joints = n::tensor({0., 0., 1.}, {1, 1, 3});
/// d.box_size = n::tensor({25., 20.}, {1, 2});
/// const auto score = m::ScorePose(n::tensor({3., 4., 0.}, {1, 1, 3}), d);
/// @endcode
/// @par References
/// - Newell,Yang and Deng(2016),Stacked Hourglass Networks for Human Pose Estimation, https://arxiv.org/abs/1603.06937;official small COCO8-Pose source, https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8-pose.yaml. Custom source-image-disjoint oracle crops and PCK@.2 maximum-box-side are teaching diagnostics,not COCO keypoint AP/MPII benchmark accuracy.
/// @see PoseHeatmaps(), PoseHeatmapMSE(), DecodePoseHeatmaps(), ScoreTensorPose()
inline PoseMetrics ScorePose(const nn::Tensor &prediction, const PoseData &data) {
  const auto s = ScoreTensorPose(prediction, data);
  return {s.pck.item(), s.normalized_error.item(), s.pixels.item(), s.joints.item<int>()};
}
/// @brief Compute fitting-only per-joint coordinate means as a closed-form parity oracle.
///
/// Float64 tensor sums select missing NaNs out before reduction. This is an
/// oracle only:reports must visibly fit trainable coordinate leaves with a
/// tensor optimizer and compare parity rather than hide the demonstrated fit.
///
/// @param fitting Original Float32[N,17,3] annotated training crops only.
/// @return Float32[17,3] x/y means and confidence1;missing joints fall back63.5.
/// @throws std::invalid_argument For invalid geometry,dtype,configuration,eligible coordinates or missing annotated support.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// m::PoseData d;
/// d.joints = n::ones({2, 17, 3});
/// const auto mean = m::FitPoseMean(d);
/// @endcode
/// @par References
/// - Newell,Yang and Deng(2016),Stacked Hourglass Networks for Human Pose Estimation, https://arxiv.org/abs/1603.06937;official small COCO8-Pose source, https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8-pose.yaml. Custom source-image-disjoint oracle crops and PCK@.2 maximum-box-side are teaching diagnostics,not COCO keypoint AP/MPII benchmark accuracy.
/// @see PoseHeatmaps(), PoseHeatmapMSE(), DecodePoseHeatmaps(), ScoreTensorPose()
inline nn::Tensor FitPoseMean(const PoseData &fitting) {
  nn::NoGradGuard guard;
  if (fitting.joints.dim() != 3 || fitting.joints.size(0) < 1 ||
      fitting.joints.shape() != nn::Shape{fitting.joints.size(0), 17, 3})
    throw std::invalid_argument("nonempty COCO17 fitting joints required");
  const auto raw = fitting.joints.to(nn::DType::Float32),
             known = nn::greater(nn::slice_axis(raw, 2, 2, 3), 0.),
             weight = known.to(nn::DType::Float64),
             xy = nn::where(known, nn::slice_axis(raw, 2, 0, 2), 0.).to(nn::DType::Float64),
             count = nn::sum(weight, {0});
  const auto mean =
      nn::where(nn::greater(count, 0.),
                nn::sum(xy, {0}) / nn::maximum(count, nn::ones_like(count)), 63.5)
          .to(nn::DType::Float32);
  return nn::concat({mean, nn::ones({17, 1})}, 1);
}
/// @brief Broadcast supplied fitting-only coordinate predictions to an observed batch shape.
///
/// No fitting or target-derived scaling. Fitting priors must come from the
/// training split and an explicit optimizer loop in the report.
///
/// @param data Supplies[N,17,3] target geometry only,values are never read.
/// @param mean Supplied[17,3] fixed fitting estimate or declared center reference.
/// @return Float32[N,17,3] constant predictions.
/// @throws std::invalid_argument For invalid geometry,dtype,configuration,eligible coordinates or missing annotated support.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// m::PoseData d;
/// d.joints = n::zeros({2, 17, 3});
/// const auto p = m::ConstantPose(d, n::ones({17, 3}));
/// @endcode
/// @par References
/// - Newell,Yang and Deng(2016),Stacked Hourglass Networks for Human Pose Estimation, https://arxiv.org/abs/1603.06937;official small COCO8-Pose source, https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8-pose.yaml. Custom source-image-disjoint oracle crops and PCK@.2 maximum-box-side are teaching diagnostics,not COCO keypoint AP/MPII benchmark accuracy.
/// @see PoseHeatmaps(), PoseHeatmapMSE(), DecodePoseHeatmaps(), ScoreTensorPose()
inline nn::Tensor ConstantPose(const PoseData &data, const nn::Tensor &mean) {
  if (data.joints.dim() != 3 || data.joints.size(1) != 17 || data.joints.size(2) != 3 ||
      mean.shape() != nn::Shape{17, 3})
    throw std::invalid_argument("COCO17 constant pose geometry required");
  return nn::ones(data.joints.shape(), nn::DType::Float32) * mean.to(nn::DType::Float32);
}
} // namespace datamunge::nn::models
