#pragma once
/// @file panoptic_models.hpp
/// @brief Tensor panoptic overlap resolution and strict-IoU PQ/SQ/RQ with void/crowd handling.
#include "datamunge/nn/instance_scene_models.hpp"
#include <map>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Describe one nonvoid panoptic segment category,thing/stuff role and crowd flag.
///
/// Plain metadata;segment IDs belong to a TensorPanopticMap catalog. Crowd
/// truth is excluded from matching and may ignore same-class predictions.
///
/// @throws std::invalid_argument For invalid settings,tensor shapes/dtypes,unknown IDs,categories or empty metadata segments.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::TensorPanopticMap map{n::from_vector(std::vector<std::int64_t>{1, 1, 0}),
///                                  {{1, {0, true, false}}}};
/// const auto areas = n::models::PanopticAreas(map);
/// @endcode
/// @par References
/// - Kirillov et al.(2019),Panoptic Segmentation, https://arxiv.org/abs/1801.00868; official evaluator https://github.com/cocodataset/panopticapi/blob/master/panopticapi/evaluation.py; overlap resolution follows Detectron2 Panoptic FPN https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/meta_arch/panoptic_fpn.py.
/// @see PanopticStatistics()
struct PanopticSegment {
  int label = 0;      ///< Nonnegative category ID.
  bool thing = false; ///< True for countable objects,false for stuff regions.
  bool crowd = false; ///< True for crowd truth excluded from PQ matching.
};
/// @brief Own integer segment pixels and a catalog for every present nonvoid segment.
///
/// Id0 is void and has no catalog entry. Positive IDs need metadata and
/// nonempty pixel area;category IDs are independent of segment IDs. Tensor
/// handles own storage;numeric APIs detach and never mutate the map.
///
/// @throws std::invalid_argument For invalid settings,tensor shapes/dtypes,unknown IDs,categories or empty metadata segments.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::TensorPanopticMap map{n::from_vector(std::vector<std::int64_t>{1, 1, 0}),
///                                  {{1, {0, true, false}}}};
/// const auto areas = n::models::PanopticAreas(map);
/// @endcode
/// @par References
/// - Kirillov et al.(2019),Panoptic Segmentation, https://arxiv.org/abs/1801.00868; official evaluator https://github.com/cocodataset/panopticapi/blob/master/panopticapi/evaluation.py; overlap resolution follows Detectron2 Panoptic FPN https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/meta_arch/panoptic_fpn.py.
/// @see PanopticStatistics()
struct TensorPanopticMap {
  nn::Tensor ids; ///< Int64 segment grid,any shape including empty,0 denotes void.
  std::map<int, PanopticSegment>
      segments; ///< Positive present segment IDs mapped to category/thing/crowd metadata.
};
/// @brief Validate a panoptic catalog and count each segment area with tensor reductions.
///
/// Every catalog ID must appear and every positive pixel needs metadata.
/// Host loops traverse only segment metadata. WorkO((M+1)*pixels),no fitting
/// or input mutation. Empty grid with empty metadata returns one zero.
///
/// @param map Int64 nonnegative segment grid with complete nonempty positive-ID metadata.
/// @return Detached Int64[M+1] areas:entry0 for void,then catalog IDs in ascending order.
/// @throws std::invalid_argument For invalid settings,tensor shapes/dtypes,unknown IDs,categories or empty metadata segments.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::TensorPanopticMap map{n::from_vector(std::vector<std::int64_t>{1, 1, 0}),
///                                  {{1, {0, true, false}}}};
/// const auto areas = n::models::PanopticAreas(map);
/// @endcode
/// @par References
/// - Kirillov et al.(2019),Panoptic Segmentation, https://arxiv.org/abs/1801.00868; official evaluator https://github.com/cocodataset/panopticapi/blob/master/panopticapi/evaluation.py; overlap resolution follows Detectron2 Panoptic FPN https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/meta_arch/panoptic_fpn.py.
/// @see TensorPanopticMap, PanopticStatistics()
inline nn::Tensor PanopticAreas(const TensorPanopticMap &map) {
  if (!map.ids.defined() || map.ids.dtype() != nn::DType::Int64 ||
      !nn::all(nn::greater_equal(map.ids, 0.)).item<bool>())
    throw std::invalid_argument("invalid panoptic ID grid");
  nn::NoGradGuard guard;
  auto known = nn::equal(map.ids, 0.);
  std::vector<nn::Tensor> areas{nn::count_nonzero(known)};
  for (const auto &[id, segment] : map.segments) {
    if (id <= 0 || segment.label < 0)
      throw std::invalid_argument("invalid panoptic segment metadata");
    const auto mask = nn::equal(map.ids, double(id));
    const auto area = nn::count_nonzero(mask);
    if (!area.item<std::int64_t>())
      throw std::invalid_argument("empty panoptic segment metadata");
    areas.push_back(area);
    known = nn::logical_or(known, mask);
  }
  if (!nn::all(known).item<bool>())
    throw std::invalid_argument("panoptic pixel lacks metadata");
  return nn::stack(areas, 0);
}
/// @brief Merge scored thing masks with semantic stuff predictions into a nonoverlapping panoptic map.
///
/// Instances sort descending score,stable input ties. Reject empty/fully
/// claimed supports or fraction>overlap;accepted masks lose previously claimed
/// pixels. Stuff classes fill remaining pixels in semantic-class order if
/// area>=stuff_area. IDs increase in acceptance order. Tensor kernels perform
/// all sorting,overlap,area,thresholding and painting. Host loops carry discrete
/// object/class metadata only. No model optimization or input mutation.
///
/// @param semantic Int64[H,W] valid semantic class IDs.
/// @param instances Same-image scored Bool[D,H,W] supports with floating scores and Int64 category labels;box coordinates/image IDs unused.
/// @param semantic_categories Metadata mapping semantic class to panoptic stuff category;negative means no stuff claim.
/// @param score Finite inclusive instance cutoff in[0,1],default.5.
/// @param overlap Finite maximum already-claimed fraction in[0,1],default.5;equality accepted.
/// @param stuff_area Positive minimum unclaimed pixels for a stuff category,default32.
/// @return Detached Int64 grid and nonempty segment catalog,with unclaimed pixels left void0.
/// @throws std::invalid_argument For invalid settings,tensor shapes/dtypes,unknown IDs,categories or empty metadata segments.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto semantic = n::from_vector(std::vector<std::int64_t>{1, 1, 2, 2}, {2, 2});
/// n::models::TensorMaskDetections masks{
///     {n::zeros({1, 4}), n::tensor({.9}), n::zeros({1}, n::DType::Int64),
///      n::zeros({1}, n::DType::Int64)},
///     n::tensor({1., 0., 0., 0.}, {1, 2, 2}).to(n::DType::Bool)};
/// const auto map = n::models::MergeTensorPanoptic(semantic, masks, {-1, 10, 11}, .5, .5, 1);
/// @endcode
/// @par References
/// - Kirillov et al.(2019),Panoptic Segmentation, https://arxiv.org/abs/1801.00868; official evaluator https://github.com/cocodataset/panopticapi/blob/master/panopticapi/evaluation.py; overlap resolution follows Detectron2 Panoptic FPN https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/meta_arch/panoptic_fpn.py.
/// @see PanopticAreas(), PanopticStatistics()
inline TensorPanopticMap MergeTensorPanoptic(const nn::Tensor &semantic,
                                             const TensorMaskDetections &instances,
                                             const std::vector<int> &semantic_categories,
                                             double score = .5, double overlap = .5,
                                             int stuff_area = 32) {
  const auto D = instances.objects.scores.numel();
  if (semantic.dim() != 2 || semantic.dtype() != nn::DType::Int64 ||
      semantic_categories.empty() || !std::isfinite(score) || score < 0 || score > 1 ||
      !std::isfinite(overlap) || overlap < 0 || overlap > 1 || stuff_area < 1 ||
      instances.objects.scores.shape() != nn::Shape{D} ||
      (instances.objects.scores.dtype() != nn::DType::Float32 &&
       instances.objects.scores.dtype() != nn::DType::Float64) ||
      instances.objects.labels.shape() != nn::Shape{D} ||
      instances.objects.labels.dtype() != nn::DType::Int64 ||
      instances.masks.shape() != nn::Shape{D, semantic.shape()[0], semantic.shape()[1]} ||
      instances.masks.dtype() != nn::DType::Bool ||
      !nn::all(nn::logical_and(nn::greater_equal(semantic, 0.),
                               nn::less(semantic, double(semantic_categories.size()))))
           .item<bool>() ||
      !nn::all(nn::greater_equal(instances.objects.labels, 0.)).item<bool>() ||
      !nn::all(
           nn::less_equal(instances.objects.labels, double(std::numeric_limits<int>::max())))
           .item<bool>() ||
      !nn::all(nn::less(nn::abs(instances.objects.scores),
                        std::numeric_limits<double>::infinity()))
           .item<bool>())
    throw std::invalid_argument("invalid panoptic merge inputs/settings");
  nn::NoGradGuard guard;
  TensorPanopticMap out{nn::zeros(semantic.shape(), nn::DType::Int64), {}};
  int id = 0;
  const auto order = nn::argsort(instances.objects.scores, true);
  for (const auto k : order.to_vector<std::int64_t>()) {
    if (nn::slice_axis(instances.objects.scores, 0, k, k + 1).item() < score)
      continue;
    const auto mask = nn::slice_axis(instances.masks, 0, k, k + 1).reshape(semantic.shape());
    const auto area = nn::count_nonzero(mask),
               claimed = nn::count_nonzero(nn::logical_and(mask, nn::not_equal(out.ids, 0.)));
    if (!area.item<std::int64_t>() ||
        claimed.item<std::int64_t>() == area.item<std::int64_t>() ||
        nn::greater(claimed.to(nn::DType::Float64) / area.to(nn::DType::Float64), overlap)
            .item<bool>())
      continue;
    const auto free = nn::logical_and(mask, nn::equal(out.ids, 0.));
    ++id;
    out.segments[id] = {
        int(nn::slice_axis(instances.objects.labels, 0, k, k + 1).item<std::int64_t>()), true,
        false};
    out.ids = nn::where(free, nn::full({}, double(id), nn::DType::Int64), out.ids);
  }
  for (std::size_t c = 0; c < semantic_categories.size(); ++c) {
    if (semantic_categories[c] < 0)
      continue;
    const auto free = nn::logical_and(nn::equal(semantic, double(c)), nn::equal(out.ids, 0.));
    if (nn::count_nonzero(free).item<std::int64_t>() < stuff_area)
      continue;
    ++id;
    out.segments[id] = {semantic_categories[c], false, false};
    out.ids = nn::where(free, nn::full({}, double(id), nn::DType::Int64), out.ids);
  }
  return out;
}
/// @brief Compute per-category matched IoU and TP/FP/FN with strict overlap and void/crowd rules.
///
/// Match only same-category noncrowd truths at IoU>.5;truth void pixels in
/// a prediction are excluded from its match union. Disjoint segment grids
/// make strict>.5 matches unique. Unmatched noncrowd truth counts FN. An
/// unmatched prediction is ignored when more than half its area lies in
/// truth void or same-category crowd,else FP. Category/thing flags remain
/// metadata;all areas,intersections,matching,counting and sums use tensors.
/// Dense segment-mask workO((T+M)*P+T*M*P),temporaryO((T+M)*P+T*M).
/// No fit,input mutation or gradients. Counts are exactly represented below2^53.
///
/// @param truth Valid panoptic truth,nonvoid segments may include crowd.
/// @param prediction Valid predicted map with exactly matching grid shape.
/// @param classes Positive number of categories;all metadata labels must be in range.
/// @return Detached Float64[classes,4] columns summed matched IoU,TP,FP,FN.
/// @throws std::invalid_argument For invalid settings,tensor shapes/dtypes,unknown IDs,categories or empty metadata segments.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::TensorPanopticMap truth{n::from_vector(std::vector<std::int64_t>{1, 1, 1, 2, 2, 2}),
///                                    {{1, {0, true, false}}, {2, {0, true, false}}}},
///     prediction{n::from_vector(std::vector<std::int64_t>{3, 3, 3, 4, 0, 0}),
///                {{3, {0, true, false}}, {4, {0, true, false}}}};
/// const auto stats = n::models::PanopticStatistics(truth, prediction, 1);
/// @endcode
/// @par References
/// - Kirillov et al.(2019),Panoptic Segmentation, https://arxiv.org/abs/1801.00868; official evaluator https://github.com/cocodataset/panopticapi/blob/master/panopticapi/evaluation.py; overlap resolution follows Detectron2 Panoptic FPN https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/meta_arch/panoptic_fpn.py.
/// @see PanopticQuality(), PanopticAreas()
inline nn::Tensor PanopticStatistics(const TensorPanopticMap &truth,
                                     const TensorPanopticMap &prediction, int classes) {
  if (classes < 1 || truth.ids.shape() != prediction.ids.shape())
    throw std::invalid_argument("panoptic grids/classes differ");
  const auto ta = PanopticAreas(truth).to(nn::DType::Float64),
             pa = PanopticAreas(prediction).to(nn::DType::Float64);
  nn::NoGradGuard guard;
  const auto P = truth.ids.numel(), T = std::int64_t(truth.segments.size()),
             M = std::int64_t(prediction.segments.size());
  std::vector<nn::Tensor> tm, pm;
  std::vector<std::int64_t> tl, pl;
  std::vector<bool> tc;
  for (const auto &[id, segment] : truth.segments) {
    if (segment.label >= classes)
      throw std::invalid_argument("unknown panoptic truth category");
    tm.push_back(nn::equal(truth.ids, double(id)).to(nn::DType::Float64).reshape({1, P}));
    tl.push_back(segment.label);
    tc.push_back(segment.crowd);
  }
  for (const auto &[id, segment] : prediction.segments) {
    if (segment.label >= classes)
      throw std::invalid_argument("unknown panoptic predicted category");
    pm.push_back(nn::equal(prediction.ids, double(id)).to(nn::DType::Float64).reshape({1, P}));
    pl.push_back(segment.label);
  }
  const auto truth_masks =
                 tm.empty() ? nn::zeros({0, P}, nn::DType::Float64) : nn::concat(tm, 0),
             pred_masks =
                 pm.empty() ? nn::zeros({0, P}, nn::DType::Float64) : nn::concat(pm, 0);
  const auto labels_t = nn::from_vector(tl), labels_p = nn::from_vector(pl),
             crowd = nn::from_vector(tc);
  const auto intersections = nn::matmul(truth_masks, nn::transpose(pred_masks, {1, 0}));
  const auto void_overlap = nn::sum(
      pred_masks * nn::equal(truth.ids, 0.).to(nn::DType::Float64).reshape({1, P}), {1});
  const auto areas_t = nn::slice_axis(ta, 0, 1, T + 1),
             areas_p = nn::slice_axis(pa, 0, 1, M + 1);
  const auto united = areas_t.reshape({T, 1}) + areas_p.reshape({1, M}) - intersections -
                      void_overlap.reshape({1, M});
  const auto iou = intersections / nn::maximum(united, nn::ones({}, nn::DType::Float64));
  const auto same = nn::equal(labels_t.reshape({T, 1}), labels_p.reshape({1, M}));
  const auto match = nn::logical_and(
      nn::logical_and(same, nn::logical_not(crowd).reshape({T, 1})), nn::greater(iou, .5));
  const auto matched_t = nn::greater(nn::sum(match.to(nn::DType::Int64), {1}), 0.),
             matched_p = nn::greater(nn::sum(match.to(nn::DType::Int64), {0}), 0.);
  if (nn::any(nn::greater(nn::sum(match.to(nn::DType::Int64), {1}), 1.)).item<bool>() ||
      nn::any(nn::greater(nn::sum(match.to(nn::DType::Int64), {0}), 1.)).item<bool>())
    throw std::invalid_argument("nonunique strict panoptic matches");
  const auto ignored =
      void_overlap +
      nn::sum(intersections *
                  nn::logical_and(same, crowd.reshape({T, 1})).to(nn::DType::Float64),
              {0});
  const auto fp = nn::logical_and(
                 nn::logical_not(matched_p),
                 nn::less_equal(
                     ignored / nn::maximum(areas_p, nn::ones({}, nn::DType::Float64)), .5)),
             fn = nn::logical_and(nn::logical_not(crowd), nn::logical_not(matched_t));
  const auto one_t = nn::one_hot(labels_t, classes).to(nn::DType::Float64),
             one_p = nn::one_hot(labels_p, classes).to(nn::DType::Float64);
  return nn::stack(
      {nn::sum(one_t * nn::sum(iou * match.to(nn::DType::Float64), {1}).reshape({T, 1}), {0}),
       nn::sum(one_t * matched_t.to(nn::DType::Float64).reshape({T, 1}), {0}),
       nn::sum(one_p * fp.to(nn::DType::Float64).reshape({M, 1}), {0}),
       nn::sum(one_t * fn.to(nn::DType::Float64).reshape({T, 1}), {0})},
      1);
}
/// @brief Own category-macro panoptic,segmentation and recognition quality plus active class count.
///
/// Each scalar is detached;class-average SQ/RQ need not multiply to macro PQ.
///
/// @throws std::invalid_argument For invalid settings,tensor shapes/dtypes,unknown IDs,categories or empty metadata segments.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::TensorPanopticMap truth{n::from_vector(std::vector<std::int64_t>{1, 1, 1, 2, 2, 2}),
///                                    {{1, {0, true, false}}, {2, {0, true, false}}}},
///     prediction{n::from_vector(std::vector<std::int64_t>{3, 3, 3, 4, 0, 0}),
///                {{3, {0, true, false}}, {4, {0, true, false}}}};
/// const auto stats = n::models::PanopticStatistics(truth, prediction, 1);
/// const auto quality = n::models::PanopticQuality(stats, n::ones({1}, n::DType::Bool));
/// @endcode
/// @par References
/// - Kirillov et al.(2019),Panoptic Segmentation, https://arxiv.org/abs/1801.00868; official evaluator https://github.com/cocodataset/panopticapi/blob/master/panopticapi/evaluation.py; overlap resolution follows Detectron2 Panoptic FPN https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/meta_arch/panoptic_fpn.py.
/// @see PanopticQuality()
struct TensorPanopticQuality {
  nn::Tensor pq;      ///< Float64 mean per-class IoU/(TP+.5FP+.5FN),zero if no active class.
  nn::Tensor sq;      ///< Float64 mean per-class IoU/max(TP,1),zero for no matches.
  nn::Tensor rq;      ///< Float64 mean per-class TP/(TP+.5FP+.5FN).
  nn::Tensor classes; ///< Int64 scalar number of selected classes with TP/FP/FN.
};
/// @brief Macro-average category PQ/SQ/RQ after dataset counts have been accumulated.
///
/// Per-category denominator TP+.5FP+.5FN determines active classes. Sum
/// statistics over images before calling;averaging per-image scores would
/// change dataset category weighting. No fitting or mutation.
///
/// @param statistics Float64[C,4] nonnegative finite IoU/TP/FP/FN counts from PanopticStatistics.
/// @param things Bool[C] category metadata,true for things,false stuff.
/// @param subset -1(default) all,1 things,0 stuff.
/// @return Detached scalar PQ/SQ/RQ and Int64 active class count;all zero if no active category.
/// @throws std::invalid_argument For invalid settings,tensor shapes/dtypes,unknown IDs,categories or empty metadata segments.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::TensorPanopticMap truth{n::from_vector(std::vector<std::int64_t>{1, 1, 1, 2, 2, 2}),
///                                    {{1, {0, true, false}}, {2, {0, true, false}}}},
///     prediction{n::from_vector(std::vector<std::int64_t>{3, 3, 3, 4, 0, 0}),
///                {{3, {0, true, false}}, {4, {0, true, false}}}};
/// const auto stats = n::models::PanopticStatistics(truth, prediction, 1);
/// const auto quality = n::models::PanopticQuality(stats, n::ones({1}, n::DType::Bool));
/// @endcode
/// @par References
/// - Kirillov et al.(2019),Panoptic Segmentation, https://arxiv.org/abs/1801.00868; official evaluator https://github.com/cocodataset/panopticapi/blob/master/panopticapi/evaluation.py; overlap resolution follows Detectron2 Panoptic FPN https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/meta_arch/panoptic_fpn.py.
/// @see PanopticStatistics()
inline TensorPanopticQuality PanopticQuality(const nn::Tensor &statistics,
                                             const nn::Tensor &things, int subset = -1) {
  if (statistics.dim() != 2 || statistics.shape()[1] != 4 ||
      statistics.dtype() != nn::DType::Float64 ||
      things.shape() != nn::Shape{statistics.shape()[0]} ||
      things.dtype() != nn::DType::Bool || subset < -1 || subset > 1 ||
      !nn::all(nn::logical_and(nn::greater_equal(statistics, 0.),
                               nn::less(statistics, std::numeric_limits<double>::infinity())))
           .item<bool>())
    throw std::invalid_argument("invalid panoptic quality input");
  nn::NoGradGuard guard;
  const auto v = nn::unstack(statistics, 1);
  const auto denominator = v[1] + .5 * v[2] + .5 * v[3];
  auto included = nn::greater(denominator, 0.);
  if (subset >= 0)
    included = nn::logical_and(included, subset ? things : nn::logical_not(things));
  const auto count = nn::count_nonzero(included);
  const auto weight =
      included.to(nn::DType::Float64) /
      nn::maximum(count.to(nn::DType::Float64), nn::ones({}, nn::DType::Float64));
  const auto safe = nn::maximum(denominator, nn::ones({}, nn::DType::Float64) * .5);
  return {nn::sum(weight * v[0] / safe),
          nn::sum(weight * v[0] / nn::maximum(v[1], nn::ones({}, nn::DType::Float64))),
          nn::sum(weight * v[1] / safe), count};
}
} // namespace datamunge::nn::models
