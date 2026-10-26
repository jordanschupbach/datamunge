// Box utilities: IoU family, box coding, anchors, matching and NMS.
//
// Tensor box math is composed of differentiable ops. NMS and matching run on
// the host over packed copies (they produce indices, not gradients); NMS
// reproduces TensorFlow's NonMaxSuppressionV5 kernel step by step, including
// its priority-queue order and soft-NMS re-queueing.

#include "datamunge/nn/boxes.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <queue>
#include <stdexcept>
#include <string>

#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/ops.hpp"
#include "detail.hpp"

namespace datamunge::nn {

namespace {

/// The four coordinate planes of boxes [..., 4].
struct Corners {
  Tensor y1, x1, y2, x2;
};

Corners corners(const Tensor& boxes, const char* what) {
  if (boxes.dim() < 1 || boxes.shape().back() != 4)
    throw std::invalid_argument(std::string(what) + ": boxes must be [..., 4], got " + shape_str(boxes.shape()));
  const Tensor b = is_floating(boxes.dtype()) ? boxes : cast(boxes, DType::Float32);
  return {select(b, -1, 0), select(b, -1, 1), select(b, -1, 2), select(b, -1, 3)};
}

Tensor area_of(const Corners& c) { return relu(c.y2 - c.y1) * relu(c.x2 - c.x1); }

Tensor intersection_of(const Corners& a, const Corners& b) {
  return relu(minimum(a.y2, b.y2) - maximum(a.y1, b.y1)) * relu(minimum(a.x2, b.x2) - maximum(a.x1, b.x1));
}

/// num / den where den > 0, else 0 (with finite gradients everywhere).
Tensor safe_divide(const Tensor& num, const Tensor& den) {
  const Tensor positive = greater(den, 0.0);
  return where(positive, num / where(positive, den, 1.0), 0.0);
}

/// Pieces shared by the IoU variants of aligned box pairs.
struct PairTerms {
  Tensor iou, union_area;
  Corners a, b;
};

PairTerms pair_terms(const Tensor& a, const Tensor& b, double eps, const char* what) {
  PairTerms t{Tensor(), Tensor(), corners(a, what), corners(b, what)};
  const Tensor inter = intersection_of(t.a, t.b);
  t.union_area       = area_of(t.a) + area_of(t.b) - inter;
  t.iou              = inter / maximum(t.union_area, scalar_like(eps, t.union_area));
  return t;
}

/// Height and width of the smallest box enclosing both.
std::pair<Tensor, Tensor> enclosing(const Corners& a, const Corners& b) {
  return {maximum(a.y2, b.y2) - minimum(a.y1, b.y1), maximum(a.x2, b.x2) - minimum(a.x1, b.x1)};
}

/// TF NonMaxSuppression's IoU: corners normalised with min/max, 0 for empty boxes.
template <typename T>
T nms_iou(const T* a, const T* b) {
  const T ay1 = std::min(a[0], a[2]), ax1 = std::min(a[1], a[3]), ay2 = std::max(a[0], a[2]), ax2 = std::max(a[1], a[3]);
  const T by1 = std::min(b[0], b[2]), bx1 = std::min(b[1], b[3]), by2 = std::max(b[0], b[2]), bx2 = std::max(b[1], b[3]);
  const T area_a = (ay2 - ay1) * (ax2 - ax1), area_b = (by2 - by1) * (bx2 - bx1);
  if (area_a <= 0 || area_b <= 0) return T(0);
  const T inter = std::max<T>(std::min(ay2, by2) - std::max(ay1, by1), 0) *
                  std::max<T>(std::min(ax2, bx2) - std::max(ax1, bx1), 0);
  return inter / (area_a + area_b - inter);
}

void check_nms_arguments(std::int64_t n_boxes, std::int64_t n_scores, std::int64_t max_output_size, double iou,
                         double sigma) {
  if (n_boxes != n_scores)
    throw std::invalid_argument("non_max_suppression: " + std::to_string(n_boxes) + " boxes but " +
                                std::to_string(n_scores) + " scores");
  if (max_output_size < 0) throw std::invalid_argument("non_max_suppression: max_output_size must be >= 0");
  if (!(iou >= 0 && iou <= 1)) throw std::invalid_argument("non_max_suppression: iou_threshold must be in [0, 1]");
  if (!(sigma >= 0)) throw std::invalid_argument("non_max_suppression: soft_nms_sigma must be >= 0");
}

/// TensorFlow's DoNonMaxSuppressionOp (V5): kept indices and their scores.
template <typename T>
std::vector<std::int64_t> nms_kernel(const T* boxes, const T* scores, std::int64_t n, std::int64_t max_output_size,
                                     T iou_threshold, T score_threshold, T sigma, std::vector<T>* kept_scores) {
  struct Candidate {
    std::int64_t index;
    T            score;
    std::int64_t suppress_begin;
  };
  const auto lower = [](const Candidate& a, const Candidate& b) {
    return (a.score == b.score && a.index > b.index) || a.score < b.score;
  };
  std::priority_queue<Candidate, std::vector<Candidate>, decltype(lower)> queue(lower);
  for (std::int64_t i = 0; i < n; ++i)
    if (scores[i] > score_threshold) queue.push({i, scores[i], 0});
  const bool soft  = sigma > 0;
  const T    scale = soft ? T(-0.5) / sigma : T(0);
  const auto weight = [&](T iou) { return iou <= iou_threshold ? std::exp(scale * iou * iou) : T(0); };
  const std::int64_t        limit = std::min(max_output_size, n);
  std::vector<std::int64_t> selected;
  std::vector<T>            selected_scores;
  while (static_cast<std::int64_t>(selected.size()) < limit && !queue.empty()) {
    Candidate next     = queue.top();
    const T   original = next.score;
    queue.pop();
    bool hard_suppressed = false;
    for (std::int64_t j = static_cast<std::int64_t>(selected.size()) - 1; j >= next.suppress_begin; --j) {
      const T iou = nms_iou(boxes + 4 * next.index, boxes + 4 * selected[static_cast<std::size_t>(j)]);
      next.score *= weight(iou);
      if (!soft && iou > iou_threshold) {
        hard_suppressed = true;
        break;
      }
      if (next.score <= score_threshold) break;
    }
    next.suppress_begin = static_cast<std::int64_t>(selected.size());
    if (hard_suppressed) continue;
    if (next.score == original) {
      selected.push_back(next.index);
      selected_scores.push_back(next.score);
    } else if (next.score > score_threshold) {
      queue.push(next);
    }
  }
  if (kept_scores) *kept_scores = std::move(selected_scores);
  return selected;
}

/// Matching on an IoU matrix [A, G] (row-major), thresholds in T's precision.
template <typename T>
std::vector<std::int64_t> match_kernel(const T* iou, std::int64_t anchors, std::int64_t targets, T positive,
                                       T negative, bool force_best) {
  std::vector<std::int64_t> match(static_cast<std::size_t>(anchors), kUnmatched);
  for (std::int64_t a = 0; a < anchors; ++a) {
    T            best  = 0;
    std::int64_t which = kUnmatched;
    for (std::int64_t g = 0; g < targets; ++g)
      if (iou[a * targets + g] > best) best = iou[a * targets + g], which = g;
    match[static_cast<std::size_t>(a)] = best >= positive ? which : best < negative ? kUnmatched : kIgnored;
  }
  if (force_best)
    for (std::int64_t g = 0; g < targets; ++g) {
      T            top = 0;
      std::int64_t arg = -1;
      for (std::int64_t a = 0; a < anchors; ++a)
        if (iou[a * targets + g] > top) top = iou[a * targets + g], arg = a;
      if (arg >= 0) match[static_cast<std::size_t>(arg)] = g;
    }
  return match;
}

}  // namespace

// ---- tensor box math ---------------------------------------------------------------------

Tensor box_area(const Tensor& boxes) { return area_of(corners(boxes, "box_area")); }

Tensor box_iou(const Tensor& a, const Tensor& b) {
  if (a.dim() < 2 || b.dim() < 2)
    throw std::invalid_argument("box_iou: expected [..., N, 4] and [..., M, 4], got " + shape_str(a.shape()) +
                                " and " + shape_str(b.shape()));
  const Tensor pa = expand_dims(a, -2), pb = expand_dims(b, -3);  // [..., N, 1, 4], [..., 1, M, 4]
  return box_iou_aligned(pa, pb);
}

Tensor box_iou_aligned(const Tensor& a, const Tensor& b) {
  const Corners ca = corners(a, "box_iou"), cb = corners(b, "box_iou");
  const Tensor  inter = intersection_of(ca, cb);
  return safe_divide(inter, area_of(ca) + area_of(cb) - inter);
}

Tensor generalized_iou(const Tensor& a, const Tensor& b, double eps) {
  const PairTerms t    = pair_terms(a, b, eps, "generalized_iou");
  const auto [eh, ew]  = enclosing(t.a, t.b);
  const Tensor enclose = eh * ew;
  return t.iou - (enclose - t.union_area) / maximum(enclose, scalar_like(eps, enclose));
}

Tensor distance_iou(const Tensor& a, const Tensor& b, double eps) {
  const PairTerms t   = pair_terms(a, b, eps, "distance_iou");
  const auto [eh, ew] = enclosing(t.a, t.b);
  const Tensor dy = (t.a.y1 + t.a.y2 - t.b.y1 - t.b.y2) * 0.5, dx = (t.a.x1 + t.a.x2 - t.b.x1 - t.b.x2) * 0.5;
  return t.iou - (square(dy) + square(dx)) / (square(eh) + square(ew) + eps);
}

Tensor complete_iou(const Tensor& a, const Tensor& b, double eps) {
  const Tensor    diou = distance_iou(a, b, eps);
  const PairTerms t    = pair_terms(a, b, eps, "complete_iou");
  const auto aspect = [eps](const Corners& c) {
    const Tensor h = c.y2 - c.y1;
    return atan((c.x2 - c.x1) / maximum(h, scalar_like(eps, h)));
  };
  const double pi    = std::acos(-1.0);
  const Tensor v     = square(aspect(t.b) - aspect(t.a)) * (4.0 / (pi * pi));
  const Tensor alpha = stop_gradient(v / (1.0 - t.iou + v + eps));
  return diou - alpha * v;
}

Tensor boxes_to_center(const Tensor& boxes) {
  const Corners c = corners(boxes, "boxes_to_center");
  return stack({(c.y1 + c.y2) * 0.5, (c.x1 + c.x2) * 0.5, c.y2 - c.y1, c.x2 - c.x1}, -1);
}

Tensor boxes_from_center(const Tensor& centers) {
  const Corners c = corners(centers, "boxes_from_center");  // planes cy, cx, h, w
  return stack({c.y1 - c.y2 * 0.5, c.x1 - c.x2 * 0.5, c.y1 + c.y2 * 0.5, c.x1 + c.x2 * 0.5}, -1);
}

Tensor clip_boxes(const Tensor& boxes, double height, double width) {
  const Corners c = corners(boxes, "clip_boxes");
  return stack({clip(c.y1, 0, height), clip(c.x1, 0, width), clip(c.y2, 0, height), clip(c.x2, 0, width)}, -1);
}

Tensor encode_boxes(const Tensor& anchors, const Tensor& boxes, const BoxVariances& v) {
  const Corners a = corners(anchors, "encode_boxes"), b = corners(boxes, "encode_boxes");
  const Tensor  ah = a.y2 - a.y1, aw = a.x2 - a.x1, h = b.y2 - b.y1, w = b.x2 - b.x1;
  return stack({((b.y1 + b.y2) / 2.0 - (a.y1 + a.y2) / 2.0) / ah / v[0],
                ((b.x1 + b.x2) / 2.0 - (a.x1 + a.x2) / 2.0) / aw / v[1], log(h / ah) / v[2], log(w / aw) / v[3]},
               -1);
}

Tensor decode_boxes(const Tensor& anchors, const Tensor& deltas, const BoxVariances& v, double max_log_scale) {
  const Corners a = corners(anchors, "decode_boxes"), d = corners(deltas, "decode_boxes");  // d: dy, dx, dh, dw
  const Tensor  ah = a.y2 - a.y1, aw = a.x2 - a.x1;
  const Tensor  cy = (a.y1 + a.y2) / 2.0 + d.y1 * v[0] * ah, cx = (a.x1 + a.x2) / 2.0 + d.x1 * v[1] * aw;
  const Tensor  h  = ah * exp(minimum(d.y2 * v[2], scalar_like(max_log_scale, d.y2)));
  const Tensor  w  = aw * exp(minimum(d.x2 * v[3], scalar_like(max_log_scale, d.x2)));
  return stack({cy - h / 2.0, cx - w / 2.0, cy + h / 2.0, cx + w / 2.0}, -1);
}

Tensor anchor_grid(std::int64_t rows, std::int64_t cols, double stride, const std::vector<double>& sizes,
                   const std::vector<double>& ratios) {
  if (rows < 0 || cols < 0 || !(stride > 0)) throw std::invalid_argument("anchor_grid: invalid grid");
  for (double s : sizes)
    if (!(s > 0)) throw std::invalid_argument("anchor_grid: sizes must be positive");
  for (double r : ratios)
    if (!(r > 0)) throw std::invalid_argument("anchor_grid: ratios must be positive");
  std::vector<float> out;
  const float        st = static_cast<float>(stride);
  for (std::int64_t i = 0; i < rows; ++i)
    for (std::int64_t j = 0; j < cols; ++j)
      for (double size : sizes)
        for (double ratio : ratios) {
          const float cy = (static_cast<float>(i) + .5f) * st, cx = (static_cast<float>(j) + .5f) * st;
          const float r = static_cast<float>(ratio), s = static_cast<float>(size);
          const float h = s * std::sqrt(r), w = s / std::sqrt(r);
          out.insert(out.end(), {cy - h / 2, cx - w / 2, cy + h / 2, cx + w / 2});
        }
  return from_vector(out, {static_cast<std::int64_t>(out.size() / 4), 4});
}

// ---- matching and suppression ---------------------------------------------------------

Tensor match_boxes(const Tensor& iou, double positive, double negative, bool force_best) {
  if (iou.dim() != 2) throw std::invalid_argument("match_boxes: iou must be [A, G], got " + shape_str(iou.shape()));
  const Tensor m = is_floating(iou.dtype()) ? detail::nograd(iou).contiguous() : cast(iou, DType::Float32);
  return detail::dispatch_float(m.dtype(), "match_boxes", [&](auto tag) {
    using T = typename decltype(tag)::type;
    return from_vector(match_kernel<T>(m.data_ptr<T>(), m.shape()[0], m.shape()[1], static_cast<T>(positive),
                                       static_cast<T>(negative), force_best),
                       {m.shape()[0]});
  });
}

NmsResult non_max_suppression(const Tensor& boxes, const Tensor& scores, std::int64_t max_output_size,
                              double iou_threshold, double score_threshold, double soft_nms_sigma,
                              bool pad_to_max_output_size) {
  if (boxes.dim() != 2 || boxes.shape()[1] != 4)
    throw std::invalid_argument("non_max_suppression: boxes must be [N, 4], got " + shape_str(boxes.shape()));
  if (scores.dim() != 1) throw std::invalid_argument("non_max_suppression: scores must be [N]");
  check_nms_arguments(boxes.shape()[0], scores.shape()[0], max_output_size, iou_threshold, soft_nms_sigma);
  NoGradGuard  guard;
  const DType  dtype = is_floating(scores.dtype()) ? scores.dtype() : DType::Float32;
  const Tensor b     = boxes.to(dtype).contiguous().detach(), s = scores.to(dtype).contiguous().detach();
  return detail::dispatch_float(dtype, "non_max_suppression", [&](auto tag) {
    using T = typename decltype(tag)::type;
    std::vector<T> kept_scores;
    auto kept = nms_kernel<T>(b.data_ptr<T>(), s.data_ptr<T>(), boxes.shape()[0], max_output_size,
                              static_cast<T>(iou_threshold), static_cast<T>(score_threshold),
                              static_cast<T>(soft_nms_sigma), &kept_scores);
    NmsResult r;
    r.valid = static_cast<std::int64_t>(kept.size());
    if (pad_to_max_output_size) {
      kept.resize(static_cast<std::size_t>(max_output_size), 0);
      kept_scores.resize(static_cast<std::size_t>(max_output_size), T(0));
    }
    r.indices = from_vector(kept, {static_cast<std::int64_t>(kept.size())});
    r.scores  = from_vector(kept_scores, {static_cast<std::int64_t>(kept_scores.size())});
    return r;
  });
}

Tensor batched_non_max_suppression(const Tensor& boxes, const Tensor& scores, const Tensor& group_ids,
                                   double iou_threshold, std::int64_t max_output_size, double score_threshold) {
  if (boxes.dim() != 2 || boxes.shape()[1] != 4 || scores.dim() != 1 || group_ids.dim() != 1 ||
      group_ids.shape()[0] != boxes.shape()[0] || !is_integral(group_ids.dtype()))
    throw std::invalid_argument("batched_non_max_suppression: expected boxes [N, 4], scores [N], integer group_ids [N]");
  const std::int64_t n = boxes.shape()[0];
  check_nms_arguments(n, scores.shape()[0], 0, iou_threshold, 0.0);
  NoGradGuard  guard;
  const DType  dtype = is_floating(scores.dtype()) ? scores.dtype() : DType::Float32;
  const Tensor b     = boxes.to(dtype).contiguous().detach(), s = scores.to(dtype).contiguous().detach();
  const auto   ids   = group_ids.to_vector<std::int64_t>();
  return detail::dispatch_float(dtype, "batched_non_max_suppression", [&](auto tag) {
    using T       = typename decltype(tag)::type;
    const T* pb   = b.data_ptr<T>();
    const T* ps   = s.data_ptr<T>();
    std::map<std::int64_t, std::vector<std::int64_t>> groups;
    for (std::int64_t i = 0; i < n; ++i) groups[ids[static_cast<std::size_t>(i)]].push_back(i);
    std::vector<std::int64_t> kept;
    std::vector<T>            gb, gs;
    for (const auto& [id, members] : groups) {
      gb.clear();
      gs.clear();
      for (auto i : members) {
        gb.insert(gb.end(), pb + 4 * i, pb + 4 * i + 4);
        gs.push_back(ps[i]);
      }
      const auto m = static_cast<std::int64_t>(members.size());
      for (auto k : nms_kernel<T>(gb.data(), gs.data(), m, m, static_cast<T>(iou_threshold),
                                  static_cast<T>(score_threshold), T(0), nullptr))
        kept.push_back(members[static_cast<std::size_t>(k)]);
    }
    std::sort(kept.begin(), kept.end(), [&](std::int64_t i, std::int64_t j) {
      return ps[i] > ps[j] || (ps[i] == ps[j] && i < j);
    });
    if (max_output_size >= 0 && static_cast<std::int64_t>(kept.size()) > max_output_size)
      kept.resize(static_cast<std::size_t>(max_output_size));
    return from_vector(kept, {static_cast<std::int64_t>(kept.size())});
  });
}

// ---- host-side boxes ---------------------------------------------------------------------

BoxT<float> encode_box(const Box& anchor, const Box& box, const BoxVariances& variances) {
  const float v[4] = {static_cast<float>(variances[0]), static_cast<float>(variances[1]),
                      static_cast<float>(variances[2]), static_cast<float>(variances[3])};
  const float ah = anchor[2] - anchor[0], aw = anchor[3] - anchor[1], h = box[2] - box[0], w = box[3] - box[1];
  return {((box[0] + box[2]) / 2 - (anchor[0] + anchor[2]) / 2) / ah / v[0],
          ((box[1] + box[3]) / 2 - (anchor[1] + anchor[3]) / 2) / aw / v[1], std::log(h / ah) / v[2],
          std::log(w / aw) / v[3]};
}

Box decode_box(const Box& anchor, const BoxT<float>& delta, const BoxVariances& variances, float max_log_scale) {
  const float v[4] = {static_cast<float>(variances[0]), static_cast<float>(variances[1]),
                      static_cast<float>(variances[2]), static_cast<float>(variances[3])};
  const float ah = anchor[2] - anchor[0], aw = anchor[3] - anchor[1];
  const float cy = (anchor[0] + anchor[2]) / 2 + delta[0] * v[0] * ah;
  const float cx = (anchor[1] + anchor[3]) / 2 + delta[1] * v[1] * aw;
  const float h = ah * std::exp(std::min(delta[2] * v[2], max_log_scale));
  const float w = aw * std::exp(std::min(delta[3] * v[3], max_log_scale));
  return {cy - h / 2, cx - w / 2, cy + h / 2, cx + w / 2};
}

std::vector<std::int64_t> non_max_suppression(const std::vector<Box>& boxes, const std::vector<float>& scores,
                                              std::int64_t max_output_size, float iou_threshold, float score_threshold,
                                              float soft_nms_sigma, std::vector<float>* kept_scores) {
  check_nms_arguments(static_cast<std::int64_t>(boxes.size()), static_cast<std::int64_t>(scores.size()),
                      max_output_size, iou_threshold, soft_nms_sigma);
  return nms_kernel<float>(boxes.empty() ? nullptr : boxes[0].data(), scores.data(),
                           static_cast<std::int64_t>(boxes.size()), max_output_size, iou_threshold, score_threshold,
                           soft_nms_sigma, kept_scores);
}

std::vector<std::int64_t> match_boxes(const std::vector<Box>& anchors, const std::vector<Box>& targets, float positive,
                                      float negative, bool force_best) {
  std::vector<float> iou;
  iou.reserve(anchors.size() * targets.size());
  for (const auto& a : anchors)
    for (const auto& t : targets) iou.push_back(box_iou(a, t));
  return match_kernel<float>(iou.data(), static_cast<std::int64_t>(anchors.size()),
                             static_cast<std::int64_t>(targets.size()), positive, negative, force_best);
}

std::vector<Box> boxes_to_vector(const Tensor& boxes) {
  if (boxes.dim() != 2 || boxes.shape()[1] != 4)
    throw std::invalid_argument("boxes_to_vector: expected [K, 4], got " + shape_str(boxes.shape()));
  const auto       v = boxes.to_vector<float>();
  std::vector<Box> out(v.size() / 4);
  for (std::size_t i = 0; i < out.size(); ++i) out[i] = {v[4 * i], v[4 * i + 1], v[4 * i + 2], v[4 * i + 3]};
  return out;
}

Tensor boxes_from_vector(const std::vector<Box>& boxes) {
  std::vector<float> v;
  v.reserve(boxes.size() * 4);
  for (const auto& b : boxes) v.insert(v.end(), b.begin(), b.end());
  return from_vector(v, {static_cast<std::int64_t>(boxes.size()), 4});
}

}  // namespace datamunge::nn
