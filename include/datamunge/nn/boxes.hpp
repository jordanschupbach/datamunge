#pragma once

/// @file boxes.hpp
/// @brief Bounding-box utilities for detection: areas, IoU (and GIoU, DIoU,
///        CIoU), Faster R-CNN box coding, anchors, anchor matching and
///        non-maximum suppression with TensorFlow semantics.
///
/// Boxes follow TensorFlow's convention: the last axis holds
/// [y1, x1, y2, x2] (ymin, xmin, ymax, xmax), in pixels or normalised
/// coordinates, with y1 <= y2 and x1 <= x2 unless stated otherwise. A box is
/// the half-open region [y1, y2) x [x1, x2), so its area is
/// (y2 - y1) * (x2 - x1), with no "+1" pixel convention.
///
/// Tensor functions are compositions of differentiable ops (so IoU-based
/// losses train). Selection and matching (NMS, `match_boxes`) return integer
/// indices and are not differentiable. Host-side helpers on `BoxT<T>` arrays
/// serve the per-box loops of detection post-processing and evaluation.

#include <array>
#include <cstdint>
#include <limits>
#include <vector>

#include "datamunge/nn/tensor.hpp"

namespace datamunge::nn {

// ---- tensor box math -------------------------------------------------------------------

/// Area of boxes [..., 4] -> [...]; inverted extents count as zero.
Tensor box_area(const Tensor& boxes);

/// Pairwise IoU of a [..., N, 4] and b [..., M, 4] -> [..., N, M] (leading
/// axes broadcast). Pairs whose union is empty have IoU 0.
Tensor box_iou(const Tensor& a, const Tensor& b);

/// Elementwise IoU of matching boxes a [..., 4] and b [..., 4] (broadcasting) -> [...].
Tensor box_iou_aligned(const Tensor& a, const Tensor& b);

/// Elementwise generalized IoU (Rezatofighi et al. 2019): IoU minus the
/// fraction of the smallest enclosing box not covered by the union. In [-1, 1].
/// Denominators are kept >= `eps`.
Tensor generalized_iou(const Tensor& a, const Tensor& b, double eps = 1e-7);

/// Elementwise distance IoU (Zheng et al. 2020): IoU minus the squared centre
/// distance over the squared diagonal of the enclosing box (+ eps).
Tensor distance_iou(const Tensor& a, const Tensor& b, double eps = 1e-7);

/// Elementwise complete IoU: DIoU minus alpha * v, where v = 4/pi^2 (atan(w_b/h_b)
/// - atan(w_a/h_a))^2 measures aspect disagreement and alpha = v / (1 - IoU + v + eps)
/// is held constant (stop-gradient), as in the reference loss.
Tensor complete_iou(const Tensor& a, const Tensor& b, double eps = 1e-7);

/// [y1, x1, y2, x2] -> centre form [cy, cx, h, w].
Tensor boxes_to_center(const Tensor& boxes);
/// Centre form [cy, cx, h, w] -> [y1, x1, y2, x2].
Tensor boxes_from_center(const Tensor& centers);

/// Clamps y to [0, height] and x to [0, width].
Tensor clip_boxes(const Tensor& boxes, double height, double width);

/// Variances of the Faster R-CNN / SSD box coding (dy, dx, dh, dw).
using BoxVariances = std::array<double, 4>;
/// The common SSD/Faster R-CNN variances 0.1, 0.1, 0.2, 0.2.
constexpr BoxVariances kDefaultBoxVariances{0.1, 0.1, 0.2, 0.2};

/// Encodes `boxes` relative to `anchors` (both [..., 4], broadcasting) as
/// deltas (dy, dx, dh, dw) = ((cy - acy) / ah, (cx - acx) / aw, log(h / ah),
/// log(w / aw)), each divided by its variance (Faster R-CNN).
Tensor encode_boxes(const Tensor& anchors, const Tensor& boxes, const BoxVariances& variances = kDefaultBoxVariances);

/// Inverse of `encode_boxes`. The log-scale terms dh * var and dw * var are
/// clamped above at `max_log_scale` so a wild prediction cannot overflow.
Tensor decode_boxes(const Tensor& anchors, const Tensor& deltas, const BoxVariances& variances = kDefaultBoxVariances,
                    double max_log_scale = 4.0);

/// Anchors centred on a `rows` x `cols` grid with cell size `stride`: for each
/// cell (row-major), each size, each ratio (= height / width), the box of
/// height size*sqrt(ratio) and width size/sqrt(ratio) centred at
/// ((row + 0.5) stride, (col + 0.5) stride). Returns float32 [rows*cols*S*R, 4].
Tensor anchor_grid(std::int64_t rows, std::int64_t cols, double stride, const std::vector<double>& sizes,
                   const std::vector<double>& ratios);

// ---- matching and suppression ------------------------------------------------------

/// `match_boxes` result for anchors below the negative threshold.
constexpr std::int64_t kUnmatched = -1;
/// `match_boxes` result for anchors between the thresholds.
constexpr std::int64_t kIgnored = -2;

/// Argmax matching on an IoU matrix [A, G] (anchors x targets), as in Faster
/// R-CNN's RPN: an anchor whose best IoU is >= `positive` gets that target's
/// index, below `negative` gets kUnmatched, otherwise kIgnored. Anchors with
/// no positive overlap are unmatched. With `force_best`, each target also
/// claims the anchor it overlaps most (first on ties, if the IoU is > 0;
/// later targets win conflicts). Returns int64 [A].
Tensor match_boxes(const Tensor& iou, double positive, double negative, bool force_best = true);

/// Result of `non_max_suppression`.
struct NmsResult {
  /// int64 [K]: indices of the kept boxes, in selection order (highest score first).
  Tensor indices;
  /// [K]: their scores (decayed scores for soft-NMS), in the scores' dtype.
  Tensor scores;
  /// Number of valid entries (K, or fewer when padded).
  std::int64_t valid{0};
};

/// Greedy non-maximum suppression with TensorFlow NonMaxSuppressionV3/V4/V5
/// semantics. boxes [N, 4] (corners in either order: each box is normalised
/// with min/max as TF does), scores [N]. Candidates need score >
/// `score_threshold`; they are visited by descending score, ties by lower
/// index. A candidate is dropped when its IoU with a kept box is >
/// `iou_threshold` (IoU 0 for zero-area boxes). With `soft_nms_sigma` > 0 the
/// Gaussian soft-NMS of V5 decays scores by exp(-iou^2 / (2 sigma)) instead
/// (boxes with IoU > threshold are still removed), re-queueing decayed
/// candidates while their score stays above the threshold. At most
/// `max_output_size` boxes are kept; `pad_to_max_output_size` pads the result
/// with zeros to that length (V4). IoU is computed in the boxes' precision.
NmsResult non_max_suppression(const Tensor& boxes, const Tensor& scores, std::int64_t max_output_size,
                              double iou_threshold = 0.5,
                              double score_threshold = -std::numeric_limits<double>::infinity(),
                              double soft_nms_sigma = 0.0, bool pad_to_max_output_size = false);

/// NMS run independently within each group (e.g. per class, or per image and
/// class): boxes [N, 4], scores [N], integer group_ids [N]. The survivors of all
/// groups are merged by descending score (ties by lower index) and the first
/// `max_output_size` (all when negative) returned as int64 indices.
Tensor batched_non_max_suppression(const Tensor& boxes, const Tensor& scores, const Tensor& group_ids,
                                   double iou_threshold = 0.5, std::int64_t max_output_size = -1,
                                   double score_threshold = -std::numeric_limits<double>::infinity());

// ---- host-side boxes ---------------------------------------------------------------------

/// One box [y1, x1, y2, x2] on the host.
template <typename T>
using BoxT = std::array<T, 4>;
/// Float boxes, the usual host-side type (TF boxes are float32).
using Box = BoxT<float>;

/// Area with inverted extents counted as zero.
template <typename T>
T box_area(const BoxT<T>& b) {
  const T h = b[2] - b[0], w = b[3] - b[1];
  return (h > 0 ? h : T(0)) * (w > 0 ? w : T(0));
}

/// IoU of two boxes; 0 when they do not overlap.
template <typename T>
T box_iou(const BoxT<T>& a, const BoxT<T>& b) {
  const T h = (a[2] < b[2] ? a[2] : b[2]) - (a[0] > b[0] ? a[0] : b[0]);
  const T w = (a[3] < b[3] ? a[3] : b[3]) - (a[1] > b[1] ? a[1] : b[1]);
  if (!(h > 0) || !(w > 0)) return T(0);
  const T inter = h * w;
  return inter / (box_area(a) + box_area(b) - inter);
}

/// Host form of `encode_boxes` for one anchor and box (computed in T).
BoxT<float> encode_box(const Box& anchor, const Box& box, const BoxVariances& variances = kDefaultBoxVariances);
/// Host form of `decode_boxes` for one anchor and delta (dy, dx, dh, dw).
Box decode_box(const Box& anchor, const BoxT<float>& delta, const BoxVariances& variances = kDefaultBoxVariances,
               float max_log_scale = 4.0f);

/// Host form of `non_max_suppression` (hard NMS unless soft_nms_sigma > 0):
/// kept indices in selection order; `kept_scores` (optional) receives their scores.
std::vector<std::int64_t> non_max_suppression(const std::vector<Box>& boxes, const std::vector<float>& scores,
                                              std::int64_t max_output_size, float iou_threshold = 0.5f,
                                              float score_threshold = -std::numeric_limits<float>::infinity(),
                                              float soft_nms_sigma = 0.0f,
                                              std::vector<float>* kept_scores = nullptr);

/// Host form of `match_boxes` on anchors and targets (IoU from `box_iou`).
std::vector<std::int64_t> match_boxes(const std::vector<Box>& anchors, const std::vector<Box>& targets, float positive,
                                      float negative, bool force_best = true);

/// Rows of a [K, 4] tensor as host boxes, and back (float32 [K, 4]).
std::vector<Box> boxes_to_vector(const Tensor& boxes);
Tensor           boxes_from_vector(const std::vector<Box>& boxes);

}  // namespace datamunge::nn
