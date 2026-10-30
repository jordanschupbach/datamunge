#pragma once
/// @file paf_data.hpp
/// @brief Tensor original full-image COCO18 data,masked multi-person heatmaps and directed limb fields.
#include "datamunge/nn/pose_data.hpp"
#include <array>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Store the19 directed COCO18 limb endpoint pairs.
///
/// Zero-based COCO18 order includes derived neck1. Each edge corresponds to
/// interleaved x/y PAF channels2*edge and2*edge+1. Integer topology metadata.
///
/// @throws std::invalid_argument For invalid geometry,dtype,source IDs,person counts or target configuration.

/// @par References
/// - Cao,Simon,Wei and Sheikh(2017),Realtime Multi-Person2D Pose Estimation using Part Affinity Fields, https://arxiv.org/abs/1611.08050;official COCO8-Pose source, https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8-pose.yaml. Custom128px source-image-disjoint teaching split,derived neck and conservative unknown-box masks,not COCO crowd masks or benchmark AP.
/// @see Coco18(), LoadPafImages(), PreparePafTargets(), PafMaskedMSE()
inline constexpr int kPoseEdges[19][2] = {{1, 2},   {1, 5},   {2, 3},  {3, 4},  {5, 6},
                                          {6, 7},   {1, 8},   {8, 9},  {9, 10}, {1, 11},
                                          {11, 12}, {12, 13}, {1, 0},  {0, 14}, {14, 16},
                                          {0, 15},  {15, 17}, {2, 16}, {5, 17}};

/// @brief Own full-image observations,multi-person annotations and provenance.
///
/// RGB is the only network input. Targets/boxes/counts/IDs are supervision
/// and evaluation metadata. Multiple annotated people share one photograph.
///
/// @throws std::invalid_argument For invalid geometry,dtype,source IDs,person counts or target configuration.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// m::PafData d{};
/// @endcode
/// @par References
/// - Cao,Simon,Wei and Sheikh(2017),Realtime Multi-Person2D Pose Estimation using Part Affinity Fields, https://arxiv.org/abs/1611.08050;official COCO8-Pose source, https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8-pose.yaml. Custom128px source-image-disjoint teaching split,derived neck and conservative unknown-box masks,not COCO crowd masks or benchmark AP.
/// @see Coco18(), LoadPafImages(), PreparePafTargets(), PafMaskedMSE()
struct PafData {
  nn::Tensor rgb; ///< Float32[N,128,128,3] observed original full-image RGB.
  nn::Tensor
      joints; ///< Float32[N,P,18,3] input-pixel x/y and original visibility/derived neck.
  nn::Tensor boxes;     ///< Float32[N,P,4] input-pixel center x/y,width/height.
  nn::Tensor counts;    ///< Int32[N] original people per image,1..P.
  nn::Tensor image_ids; ///< Int32[N] original source photograph IDs.
};
/// @brief Own custom source-image-disjoint full-image pose splits.
///
/// Only fitting images are mirrored. Source IDs repeat for fitting mirrors
/// but never cross splits. Every original person annotation is retained.
///
/// @throws std::invalid_argument For invalid geometry,dtype,source IDs,person counts or target configuration.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// m::PafSplits d{};
/// @endcode
/// @par References
/// - Cao,Simon,Wei and Sheikh(2017),Realtime Multi-Person2D Pose Estimation using Part Affinity Fields, https://arxiv.org/abs/1611.08050;official COCO8-Pose source, https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8-pose.yaml. Custom128px source-image-disjoint teaching split,derived neck and conservative unknown-box masks,not COCO crowd masks or benchmark AP.
/// @see Coco18(), LoadPafImages(), PreparePafTargets(), PafMaskedMSE()
struct PafSplits {
  PafData fit;        ///< Six rows,three fitting photographs plus mirrors.
  PafData validation; ///< One original separate validation photograph.
  PafData test;       ///< Four original held-out photographs.
};
/// @brief Gather full-image RGB and aligned annotations/provenance together.
///
/// No target generation,mask mutation or numeric host image copies.
///
/// @param data Aligned full-image pose tensors.
/// @param rows Bounded leading-row metadata.
/// @return Requested full-image rows,preserving dtype and graph.
/// @throws std::invalid_argument For invalid geometry,dtype,source IDs,person counts or target configuration.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// m::PafData d{n::zeros({2, 128, 128, 3}), n::zeros({2, 8, 18, 3}), n::ones({2, 8, 4}),
///              n::ones({2}, n::DType::Int32), n::zeros({2}, n::DType::Int32)};
/// const auto b = m::PafBatch(d, {1});
/// @endcode
/// @par References
/// - Cao,Simon,Wei and Sheikh(2017),Realtime Multi-Person2D Pose Estimation using Part Affinity Fields, https://arxiv.org/abs/1611.08050;official COCO8-Pose source, https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8-pose.yaml. Custom128px source-image-disjoint teaching split,derived neck and conservative unknown-box masks,not COCO crowd masks or benchmark AP.
/// @see Coco18(), LoadPafImages(), PreparePafTargets(), PafMaskedMSE()
inline PafData PafBatch(const PafData &data, const std::vector<std::size_t> &rows) {
  return {PoseBatch(data.rgb, rows), PoseBatch(data.joints, rows), PoseBatch(data.boxes, rows),
          PoseBatch(data.counts, rows), PoseBatch(data.image_ids, rows)};
}
/// @brief Map original COCO17 joint identities to COCO18 and derive an annotated neck.
///
/// Neck is the Float32 midpoint of labeled left/right shoulders and minimum
/// visibility. It exists only when BOTH shoulder labels exist. Tensor gather,
/// visibility selection and arithmetic retain source Float32 rounding;no
/// external annotations or fitted values create this derived joint.
///
/// @param input Numeric[N,P,17,3] x/y/visibility,original COCO order.
/// @return Float32[N,P,18,3] OpenPose order with missing joints zeroed.
/// @throws std::invalid_argument For invalid geometry,dtype,source IDs,person counts or target configuration.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto j = m::Coco18(n::ones({1, 2, 17, 3}));
/// @endcode
/// @par References
/// - Cao,Simon,Wei and Sheikh(2017),Realtime Multi-Person2D Pose Estimation using Part Affinity Fields, https://arxiv.org/abs/1611.08050;official COCO8-Pose source, https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8-pose.yaml. Custom128px source-image-disjoint teaching split,derived neck and conservative unknown-box masks,not COCO crowd masks or benchmark AP.
/// @see Coco18(), LoadPafImages(), PreparePafTargets(), PafMaskedMSE()
inline nn::Tensor Coco18(const nn::Tensor &input) {
  if (input.dim() != 4 || input.size(0) < 1 || input.size(1) < 1 || input.size(2) != 17 ||
      input.size(3) != 3)
    throw std::invalid_argument("COCO17 multi-person annotations required");
  const auto raw = input.to(nn::DType::Float32);
  const auto indices = nn::from_vector(
      std::vector<std::int64_t>{0, 5, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3},
      {18});
  auto out = nn::gather(raw, indices, 2);
  out = nn::where(nn::greater(nn::slice_axis(out, 3, 2, 3), 0.), out, 0.);
  const auto a = nn::select(raw, 2, 5), b = nn::select(raw, 2, 6),
             va = nn::slice_axis(a, 2, 2, 3), vb = nn::slice_axis(b, 2, 2, 3),
             both = nn::logical_and(nn::greater(va, 0.), nn::greater(vb, 0.));
  const auto neck =
      nn::where(both,
                nn::concat({.5 * (nn::slice_axis(a, 2, 0, 2) + nn::slice_axis(b, 2, 0, 2)),
                            nn::minimum(va, vb)},
                           2),
                0.);
  return nn::concat(
      {nn::slice_axis(out, 2, 0, 1), nn::expand_dims(neck, 2), nn::slice_axis(out, 2, 2, 18)},
      2);
}
/// @brief Load original full-image6/1/4 COCO8-Pose splits without person-box network inputs.
///
/// Validates all original RGB ranges,count/ID/dtype/shape contracts and active
/// visibility/finite coordinates/positive boxes. Source IDs may repeat for
/// fitting mirrors but never cross splits. Derives neck with Coco18;it is not
/// a new observed label. Numeric checks do not prove dataset provenance.
///
/// @param root Prepared poses.npz and pinned original-source manifest directory.
/// @return Validated full-image observations and padded COCO18 annotations.
/// @throws std::invalid_argument For invalid geometry,dtype,source IDs,person counts or target configuration.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// try {
///   const auto d = m::LoadPafImages("missing-paf-doc-example");
/// } catch (const std::exception &) {
/// }
/// @endcode
/// @par References
/// - Cao,Simon,Wei and Sheikh(2017),Realtime Multi-Person2D Pose Estimation using Part Affinity Fields, https://arxiv.org/abs/1611.08050;official COCO8-Pose source, https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8-pose.yaml. Custom128px source-image-disjoint teaching split,derived neck and conservative unknown-box masks,not COCO crowd masks or benchmark AP.
/// @see Coco18(), LoadPafImages(), PreparePafTargets(), PafMaskedMSE()
inline PafSplits LoadPafImages(const std::filesystem::path &root) {
  const auto arrays = PoseArrays(root / "poses.npz");
  std::set<int> previous;
  /// @brief Load one full-image split and extend the disjoint source-image registry.
  /// @param split Exact named archive prefix.
  /// @param count Required original/mirrored observation count.
  /// @return Validated full-image RGB and COCO18 target handles.
  auto read = [&](const std::string &split, std::int64_t count) {
    PafData d{arrays.at(split + "_full_rgb"), Coco18(arrays.at(split + "_full_joints")),
              arrays.at(split + "_full_boxes"), arrays.at(split + "_full_counts"),
              arrays.at(split + "_full_image_ids")};
    if (d.rgb.shape() != nn::Shape{count, 128, 128, 3} ||
        d.joints.shape() != nn::Shape{count, 8, 18, 3} ||
        d.boxes.shape() != nn::Shape{count, 8, 4} || d.counts.shape() != nn::Shape{count} ||
        d.image_ids.shape() != nn::Shape{count})
      throw std::invalid_argument("full-image128px/eight-person pose dimensions required");
    for (const auto *t : {&d.rgb, &d.joints, &d.boxes})
      if (t->dtype() != nn::DType::Float32)
        throw std::invalid_argument("Float32 full-image pose values required");
    for (const auto *t : {&d.counts, &d.image_ids})
      if (t->dtype() != nn::DType::Int32)
        throw std::invalid_argument("Int32 pose counts/IDs required");
    if (!nn::all(nn::logical_and(nn::greater(d.counts, 0.), nn::less_equal(d.counts, 8.)))
             .item<bool>() ||
        !nn::all(nn::logical_and(nn::greater_equal(d.rgb, 0.), nn::less_equal(d.rgb, 1.)))
             .item<bool>())
      throw std::invalid_argument("bounded people and unit RGB required");
    const auto person = nn::arange(0, 8).reshape({1, 8, 1}),
               present = nn::less(person, d.counts.reshape({count, 1, 1})),
               v = nn::slice_axis(d.joints, 3, 2, 3), xy = nn::slice_axis(d.joints, 3, 0, 2);
    const auto good_visibility =
        nn::logical_or(nn::equal(v, 0.), nn::logical_or(nn::equal(v, 1.), nn::equal(v, 2.)));
    if (!nn::all(nn::where(nn::expand_dims(present, 2), good_visibility,
                           nn::ones_like(v).to(nn::DType::Bool)))
             .item<bool>() ||
        !nn::all(nn::where(nn::logical_and(nn::expand_dims(present, 2), nn::greater(v, 0.)),
                           nn::is_finite(xy), nn::ones_like(xy).to(nn::DType::Bool)))
             .item<bool>() ||
        !nn::all(nn::where(present, nn::is_finite(d.boxes),
                           nn::ones_like(d.boxes).to(nn::DType::Bool)))
             .item<bool>() ||
        !nn::all(nn::where(present, nn::greater(nn::slice_axis(d.boxes, 2, 2, 4), 0.),
                           nn::ones({count, 8, 2}, nn::DType::Bool)))
             .item<bool>())
      throw std::invalid_argument("finite original joint/box annotations required");
    std::set<int> ids;
    for (auto id : d.image_ids.to_vector<int>()) {
      if (id < 0 || previous.contains(id))
        throw std::invalid_argument("source photographs cross full-image pose splits");
      ids.insert(id);
    }
    previous.insert(ids.begin(), ids.end());
    return d;
  };
  auto fit = read("fit", 6), validation = read("validation", 1), test = read("test", 4);
  return {fit, validation, test};
}
/// @brief Own fixed multi-person confidence and directed-limb targets with conservative support.
///
/// Target generation is detached. Original annotations/boxes are supervision
/// inputs only. Background confidence is1-max(joints);background support is
/// the intersection of all joint supports.
///
/// @throws std::invalid_argument For invalid geometry,dtype,source IDs,person counts or target configuration.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// m::PafData d{n::zeros({1, 128, 128, 3}), n::ones({1, 1, 18, 3}),
///              n::tensor({64., 64., 80., 80.}, {1, 1, 4}), n::ones({1}, n::DType::Int32),
///              n::zeros({1}, n::DType::Int32)};
/// const auto t = m::PreparePafTargets(d);
/// @endcode
/// @par References
/// - Cao,Simon,Wei and Sheikh(2017),Realtime Multi-Person2D Pose Estimation using Part Affinity Fields, https://arxiv.org/abs/1611.08050;official COCO8-Pose source, https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8-pose.yaml. Custom128px source-image-disjoint teaching split,derived neck and conservative unknown-box masks,not COCO crowd masks or benchmark AP.
/// @see Coco18(), LoadPafImages(), PreparePafTargets(), PafMaskedMSE()
struct PafTargets {
  nn::Tensor heatmap; ///< Float32[N,S,S,19] max-combined joint Gaussians plus background.
  nn::Tensor paf;     ///< Float32[N,S,S,38] averaged directed unit limb strips.
  nn::Tensor
      heatmask; ///< Float32 matching heatmap support,unknown boxes zero supervised channels.
  nn::Tensor
      pafmask; ///< Float32 matching PAF support,unknown endpoints zero both limb channels.
};
/// @brief Construct detached max-Gaussian confidence and averaged directed-limb strip targets.
///
/// Tensor coordinates use half-pixel source-to-field centers and truncated
/// unnormalized Gaussians,max-combined across people. Directed unit vectors
/// occupy strips with0<=along<=length and abs(across)<=thickness;overlapping
/// people average their vectors. Coincident endpoints add no field. A missing/
/// invalid joint or limb disables supervision only inside that person’s box.
/// These conservative unknown rectangles are not COCO crowd segmentation.
/// Person iteration preserves source Float32 accumulation order;all pixel/
/// Gaussian/strip/mask/overlap arithmetic is tensor-based. No optimizer or fit.
///
/// @param data Original multi-person COCO18 coordinates,boxes and person counts.
/// @param side Positive field side,default16.
/// @param input_size Positive source square side,default128.
/// @param sigma Positive finite Gaussian width in field pixels,default1.
/// @param thickness Positive finite limb strip half-width in field pixels,default1.
/// @return Four Float32 target/support tensors,confidence19 and PAF38 channels.
/// @throws std::invalid_argument For invalid geometry,dtype,source IDs,person counts or target configuration.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// m::PafData d{n::zeros({1, 128, 128, 3}), n::ones({1, 1, 18, 3}),
///              n::tensor({64., 64., 80., 80.}, {1, 1, 4}), n::ones({1}, n::DType::Int32),
///              n::zeros({1}, n::DType::Int32)};
/// const auto t = m::PreparePafTargets(d);
/// @endcode
/// @par References
/// - Cao,Simon,Wei and Sheikh(2017),Realtime Multi-Person2D Pose Estimation using Part Affinity Fields, https://arxiv.org/abs/1611.08050;official COCO8-Pose source, https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8-pose.yaml. Custom128px source-image-disjoint teaching split,derived neck and conservative unknown-box masks,not COCO crowd masks or benchmark AP.
/// @see Coco18(), LoadPafImages(), PreparePafTargets(), PafMaskedMSE()
inline PafTargets PreparePafTargets(const PafData &data, int side = 16, int input_size = 128,
                                    float sigma = 1.f, float thickness = 1.f) {
  nn::NoGradGuard guard;
  if (side < 1 || input_size < 1 || !std::isfinite(sigma) || sigma <= 0 ||
      !std::isfinite(thickness) || thickness <= 0 || data.joints.dim() != 4 ||
      data.joints.size(0) < 1 || data.joints.size(1) < 1 || data.joints.size(2) != 18 ||
      data.joints.size(3) != 3 ||
      data.boxes.shape() != nn::Shape{data.joints.size(0), data.joints.size(1), 4} ||
      data.counts.shape() != nn::Shape{data.joints.size(0)})
    throw std::invalid_argument("valid multi-person PAF target configuration required");
  const auto batch = data.joints.size(0);
  auto heat = nn::zeros({batch, side, side, 18}),
       fields = nn::zeros({batch, side, side, 19, 2}),
       counts = nn::zeros({batch, side, side, 19}),
       heatmask = nn::ones({batch, side, side, 18}, nn::DType::Bool),
       pafmask = nn::ones({batch, side, side, 19}, nn::DType::Bool);
  const auto x = nn::arange(0, side, 1, nn::DType::Float64).reshape({1, 1, side, 1}),
             y = nn::arange(0, side, 1, nn::DType::Float64).reshape({1, side, 1, 1}),
             ix = (x + .5) * double(input_size) / double(side) - .5,
             iy = (y + .5) * double(input_size) / double(side) - .5;
  std::vector<std::int64_t> edges;
  for (const auto &e : kPoseEdges) {
    edges.push_back(e[0]);
    edges.push_back(e[1]);
  }
  const auto topology = nn::from_vector(edges, {19, 2}), ea = nn::select(topology, 1, 0),
             eb = nn::select(topology, 1, 1);
  for (std::int64_t person = 0; person < data.joints.size(1); ++person) {
    const auto original = nn::select(data.joints, 1, person).to(nn::DType::Float32),
               raw_xy = nn::slice_axis(original, 2, 0, 2),
               v = nn::slice_axis(original, 2, 2, 3);
    const auto present = nn::greater(data.counts, double(person)).reshape({batch, 1}),
               eligible = nn::logical_and(
                   nn::greater(nn::select(v, 2, 0), 0.),
                   nn::all(
                       nn::logical_and(
                           nn::is_finite(raw_xy),
                           nn::logical_and(nn::greater_equal(raw_xy, 0.),
                                           nn::less_equal(raw_xy, double(input_size - 1)))),
                       {2})),
               annotated = nn::logical_and(eligible, present);
    const auto safe =
                   nn::where(nn::expand_dims(annotated, 2), raw_xy, 0.).to(nn::DType::Float64),
               points = (safe + .5) * double(side) / double(input_size) - .5,
               px = nn::select(points, 2, 0).reshape({batch, 1, 1, 18}),
               py = nn::select(points, 2, 1).reshape({batch, 1, 1, 18}), dx = x - px,
               dy = y - py;
    const auto box = nn::select(data.boxes, 1, person).to(nn::DType::Float64),
               bx = nn::select(box, 1, 0).reshape({batch, 1, 1, 1}),
               by = nn::select(box, 1, 1).reshape({batch, 1, 1, 1}),
               bw = nn::select(box, 1, 2).reshape({batch, 1, 1, 1}),
               bh = nn::select(box, 1, 3).reshape({batch, 1, 1, 1}),
               inside = nn::logical_and(nn::less_equal(nn::abs(ix - bx), bw / 2.),
                                        nn::less_equal(nn::abs(iy - by), bh / 2.));
    const auto known = annotated.reshape({batch, 1, 1, 18}),
               unknown =
                   nn::logical_and(present.reshape({batch, 1, 1, 1}), nn::logical_not(known));
    heatmask = nn::logical_and(heatmask, nn::logical_not(nn::logical_and(inside, unknown)));
    const auto window = nn::logical_and(
        known, nn::logical_and(nn::less_equal(nn::abs(dx), double(3.f * sigma)),
                               nn::less_equal(nn::abs(dy), double(3.f * sigma))));
    heat = nn::maximum(
        heat,
        nn::where(window, nn::exp(-(dx * dx + dy * dy) / double(2.f * sigma * sigma)), 0.)
            .to(nn::DType::Float32));
    const auto a = nn::gather(points, ea, 1), b = nn::gather(points, eb, 1),
               both =
                   nn::logical_and(nn::gather(annotated, ea, 1), nn::gather(annotated, eb, 1)),
               edge_unknown = nn::logical_and(present, nn::logical_not(both));
    pafmask = nn::logical_and(pafmask, nn::logical_not(nn::logical_and(
                                           inside, edge_unknown.reshape({batch, 1, 1, 19}))));
    const auto vx = (nn::select(b, 2, 0) - nn::select(a, 2, 0)).reshape({batch, 1, 1, 19}),
               vy = (nn::select(b, 2, 1) - nn::select(a, 2, 1)).reshape({batch, 1, 1, 19}),
               length = nn::sqrt(vx * vx + vy * vy), nonzero = nn::greater_equal(length, 1e-9),
               safe_length = nn::where(nonzero, length, 1.), ux = vx / safe_length,
               uy = vy / safe_length, rx = x - nn::select(a, 2, 0).reshape({batch, 1, 1, 19}),
               ry = y - nn::select(a, 2, 1).reshape({batch, 1, 1, 19}),
               along = rx * ux + ry * uy, across = nn::abs(rx * uy - ry * ux);
    const auto strip = nn::logical_and(
        both.reshape({batch, 1, 1, 19}),
        nn::logical_and(nonzero,
                        nn::logical_and(nn::logical_and(nn::greater_equal(along, 0.),
                                                        nn::less_equal(along, length)),
                                        nn::less_equal(across, double(thickness)))));
    const auto direction = nn::stack({ux, uy}, 4).to(nn::DType::Float32);
    fields = fields + nn::where(nn::expand_dims(strip, 4), direction, 0.);
    counts = counts + strip.to(nn::DType::Float32);
  }
  const auto normalized =
                 fields / nn::expand_dims(nn::maximum(counts, nn::ones_like(counts)), 4),
             background = 1. - nn::max(heat, {3}, true),
             background_mask = nn::all(heatmask, {3}, true);
  return {nn::concat({heat, background}, 3), normalized.reshape({batch, side, side, 38}),
          nn::concat({heatmask, background_mask}, 3).to(nn::DType::Float32),
          nn::stack({pafmask, pafmask}, 4)
              .reshape({batch, side, side, 38})
              .to(nn::DType::Float32)};
}
/// @brief Compute safe equal-supported-image masked field MSE.
///
/// Inactive predictions/targets select0 BEFORE subtraction,so missing NaNs
/// yield zero derivatives. Bool/numeric masks share the Double threshold.
/// All-empty support returns graph-connected0;no target generation or fitting.
///
/// @param prediction Floating[N,H,W,C] generated field values.
/// @param truth Matching fixed target field.
/// @param mask Matching numeric target support,active when>.5.
/// @return Scalar per-image masked MSE averaged over supported images.
/// @throws std::invalid_argument For invalid geometry,dtype,source IDs,person counts or target configuration.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto p = n::ones({2, 3, 4, 5});
/// const auto l = m::PafMaskedMSE(p, n::zeros_like(p), n::ones_like(p));
/// @endcode
/// @par References
/// - Cao,Simon,Wei and Sheikh(2017),Realtime Multi-Person2D Pose Estimation using Part Affinity Fields, https://arxiv.org/abs/1611.08050;official COCO8-Pose source, https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8-pose.yaml. Custom128px source-image-disjoint teaching split,derived neck and conservative unknown-box masks,not COCO crowd masks or benchmark AP.
/// @see Coco18(), LoadPafImages(), PreparePafTargets(), PafMaskedMSE()
inline nn::Tensor PafMaskedMSE(const nn::Tensor &prediction, const nn::Tensor &truth,
                               const nn::Tensor &mask) {
  if (prediction.dim() != 4 || prediction.size(0) < 1 || prediction.shape() != truth.shape() ||
      truth.shape() != mask.shape())
    throw std::invalid_argument("aligned masked NHWC PAF fields required");
  const auto active = nn::greater(mask.detach().to(nn::DType::Float64), .5),
             p = nn::where(active, prediction, 0.), t = nn::where(active, truth, 0.),
             count = nn::sum(active.to(prediction.dtype()), {1, 2, 3}),
             present = nn::greater(count, 0.).to(prediction.dtype()),
             mean = nn::sum(nn::square(p - t), {1, 2, 3}) /
                    nn::maximum(count, nn::ones_like(count));
  return nn::sum(mean * present) /
         nn::maximum(nn::sum(present), nn::scalar_like(1., prediction));
}
} // namespace datamunge::nn::models
