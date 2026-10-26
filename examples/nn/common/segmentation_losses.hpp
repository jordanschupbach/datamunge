// Soft Dice/Tversky, Lovasz-Softmax and signed-distance boundary losses (port
// of tflow's common/ml/segmentation_losses.h).
// https://arxiv.org/abs/1706.05721
// https://arxiv.org/abs/1705.08790
// https://arxiv.org/abs/1812.07032
// NHWC probabilities, integer NHW labels; region losses ignore labels outside
// [0, classes) (e.g. void 255). Dice = Tversky(alpha = beta = .5). Tversky
// defaults FP .3/FN .7; epsilon is added to numerator/denominator. Lovasz uses
// detached error ordering and Jaccard increments, averaging only present
// classes. first_class = 1 omits background. Boundary loss is
// mean(probability * signed distance), so can be negative; it complements a
// region loss. Distance maps use the authors' discrete interior convention
// -(distance_to_background - 1), positive distance outside. Empty/full
// foreground classes have zero boundary maps (no internal boundary).
//
// TF -> datamunge: labels are host data, so the per-class indicator/valid
// masks are built on the host; Lovasz's detached sort and Jaccard increments
// are computed on the host and enter as constant per-pixel weights
// (sum_k errors[order[k]] inc[k] = sum_i errors[i] w[i]), the same value and
// subgradient as TF's TopK/Gather/Cumsum graph.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <datamunge/nn/nn.hpp>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace seg {

namespace nn = datamunge::nn;

namespace detail {
inline std::vector<std::int32_t> FlatLabels(const nn::Tensor& labels) {
  return labels.to(nn::DType::Int32).to_vector<std::int32_t>();
}
inline nn::Tensor Column(const nn::Tensor& probabilities, int classes, int c) {
  return nn::reshape(nn::slice_axis(nn::reshape(probabilities, {-1, classes}), 1, c, c + 1), {-1});
}
}  // namespace detail

inline nn::Tensor SoftTverskyLoss(const nn::Tensor& probabilities, const nn::Tensor& labels, int classes,
                                  double alpha = .3, double beta = .7, int first_class = 1, double epsilon = 1e-6) {
  if (classes < 2 || first_class < 0 || first_class >= classes || !(alpha >= 0 && beta >= 0 && alpha + beta > 0 && epsilon > 0) ||
      !std::isfinite(alpha + beta + epsilon))
    throw std::invalid_argument("invalid Tversky configuration");
  const auto         truth = detail::FlatLabels(labels);
  const std::int64_t n     = std::int64_t(truth.size());
  std::vector<float> valid(truth.size());
  for (std::size_t i = 0; i < truth.size(); ++i) valid[i] = truth[i] >= 0 && truth[i] < classes ? 1.f : 0.f;
  const nn::Tensor        valid_t = nn::from_vector(valid, {n}).to(probabilities.dtype());
  std::vector<nn::Tensor> terms;
  for (int c = first_class; c < classes; ++c) {
    std::vector<float> y(truth.size());
    for (std::size_t i = 0; i < truth.size(); ++i) y[i] = truth[i] == c ? 1.f : 0.f;
    const nn::Tensor yt = nn::from_vector(y, {n}).to(probabilities.dtype());
    const nn::Tensor p  = detail::Column(probabilities, classes, c) * valid_t;
    const nn::Tensor tp = nn::sum(p * yt), fp = nn::sum(p * (1.0 - yt)), fn = nn::sum((valid_t - p) * yt);
    terms.push_back(1.0 - (tp + epsilon) / (tp + fp * alpha + fn * beta + epsilon));
  }
  nn::Tensor sum = terms[0];
  for (std::size_t i = 1; i < terms.size(); ++i) sum = sum + terms[i];
  return sum / double(terms.size());
}

inline nn::Tensor SoftDiceLoss(const nn::Tensor& probabilities, const nn::Tensor& labels, int classes,
                               int first_class = 1, double epsilon = 1e-6) {
  return SoftTverskyLoss(probabilities, labels, classes, .5, .5, first_class, epsilon);
}

inline nn::Tensor LovaszSoftmaxLoss(const nn::Tensor& probabilities, const nn::Tensor& labels, int classes,
                                    int first_class = 0) {
  if (classes < 2 || first_class < 0 || first_class >= classes)
    throw std::invalid_argument("invalid Lovasz class configuration");
  const auto         truth = detail::FlatLabels(labels);
  const std::size_t  n     = truth.size();
  std::vector<float> valid(n);
  for (std::size_t i = 0; i < n; ++i) valid[i] = truth[i] >= 0 && truth[i] < classes ? 1.f : 0.f;
  const nn::Tensor valid_t = nn::from_vector(valid, {std::int64_t(n)}).to(probabilities.dtype());
  nn::Tensor       total   = nn::scalar_like(0.0, probabilities);
  int              present = 0;
  for (int c = first_class; c < classes; ++c) {
    std::vector<float> fg(n);
    float              count = 0;
    for (std::size_t i = 0; i < n; ++i) {
      fg[i] = truth[i] == c ? 1.f : 0.f;
      count += fg[i];
    }
    const nn::Tensor fg_t   = nn::from_vector(fg, {std::int64_t(n)}).to(probabilities.dtype());
    const nn::Tensor errors = nn::abs(fg_t - detail::Column(probabilities, classes, c)) * valid_t;
    if (count <= 0) continue;  // included = 0: an absent class contributes nothing
    ++present;
    // Detached descending order (TF TopK: ties keep the lower index first).
    const auto               e = errors.detach().to(nn::DType::Float32).to_vector<float>();
    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), std::size_t(0));
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return e[a] > e[b]; });
    std::vector<float> weights(n, 0.f);
    float              cum_fg = 0, cum_bg = 0, previous = 0;
    for (std::size_t k = 0; k < n; ++k) {
      cum_fg += fg[order[k]];
      cum_bg += 1 - fg[order[k]];
      const float intersection = count - cum_fg, combined = count + cum_bg;
      const float jaccard      = 1 - intersection / std::max(combined, 1e-6f);
      weights[order[k]]        = jaccard - previous;
      previous                 = jaccard;
    }
    total = total + nn::sum(errors * nn::from_vector(weights, {std::int64_t(n)}).to(probabilities.dtype()));
  }
  return total / double(std::max(present, 1));
}

/// mean(foreground_probabilities * signed_distances), distances detached.
inline nn::Tensor BoundaryLoss(const nn::Tensor& foreground_probabilities, const nn::Tensor& signed_distances) {
  return nn::mean(foreground_probabilities * signed_distances.detach());
}

/// Lower envelopes of parabolas give exact squared Euclidean distance in O(n).
inline std::vector<double> SquaredDistance1D(const std::vector<double>& f) {
  const int           n = int(f.size());
  std::vector<double> result(f.size(), INFINITY), breaks(f.size() + 1);
  std::vector<int>    sites(f.size());
  int                 last = -1;
  for (int q = 0; q < n; ++q)
    if (std::isfinite(f[std::size_t(q)])) {
      double crossing = -INFINITY;
      while (last >= 0) {
        const int v = sites[std::size_t(last)];
        crossing    = ((f[std::size_t(q)] + double(q) * q) - (f[std::size_t(v)] + double(v) * v)) / (2 * (q - v));
        if (crossing > breaks[std::size_t(last)]) break;
        --last;
      }
      ++last;
      sites[std::size_t(last)]      = q;
      breaks[std::size_t(last)]     = last ? crossing : -INFINITY;
      breaks[std::size_t(last + 1)] = INFINITY;
    }
  if (last < 0) return result;
  int current = 0;
  for (int q = 0; q < n; ++q) {
    while (current < last && breaks[std::size_t(current + 1)] < q) ++current;
    const double delta     = q - sites[std::size_t(current)];
    result[std::size_t(q)] = delta * delta + f[std::size_t(sites[std::size_t(current)])];
  }
  return result;
}

inline std::vector<double> SquaredDistance2D(const std::vector<bool>& seeds, int height, int width) {
  if (height < 1 || width < 1 || seeds.size() != std::size_t(height) * std::size_t(width))
    throw std::invalid_argument("invalid distance grid");
  std::vector<double> intermediate(seeds.size()), result(seeds.size());
  for (int r = 0; r < height; ++r) {
    std::vector<double> f(static_cast<std::size_t>(width));
    for (int c = 0; c < width; ++c) f[std::size_t(c)] = seeds[std::size_t(r * width + c)] ? 0 : INFINITY;
    const auto distance = SquaredDistance1D(f);
    for (int c = 0; c < width; ++c) intermediate[std::size_t(r * width + c)] = distance[std::size_t(c)];
  }
  for (int c = 0; c < width; ++c) {
    std::vector<double> f(static_cast<std::size_t>(height));
    for (int r = 0; r < height; ++r) f[std::size_t(r)] = intermediate[std::size_t(r * width + c)];
    const auto distance = SquaredDistance1D(f);
    for (int r = 0; r < height; ++r) result[std::size_t(r * width + c)] = distance[std::size_t(r)];
  }
  return result;
}

/// Signed boundary distances [B, H, W, classes - 1] (float32) of NHW integer labels.
inline nn::Tensor SignedBoundaryDistances(const nn::Tensor& labels, int classes) {
  if (labels.dim() != 3 || classes < 2) throw std::invalid_argument("distance maps require NHW labels and classes>=2");
  const auto         values = detail::FlatLabels(labels);
  const std::int64_t batch = labels.shape()[0];
  const int          h = int(labels.shape()[1]), w = int(labels.shape()[2]);
  std::vector<float> result(std::size_t(batch) * h * w * (classes - 1), 0.f);
  for (std::int64_t n = 0; n < batch; ++n)
    for (int c = 1; c < classes; ++c) {
      std::vector<bool> foreground(std::size_t(h * w)), background(std::size_t(h * w));
      int               positives = 0;
      for (int i = 0; i < h * w; ++i) {
        foreground[std::size_t(i)] = values[std::size_t(n * h * w + i)] == c;
        background[std::size_t(i)] = !foreground[std::size_t(i)];
        positives += foreground[std::size_t(i)];
      }
      if (!positives || positives == h * w) continue;
      const auto to_foreground = SquaredDistance2D(foreground, h, w), to_background = SquaredDistance2D(background, h, w);
      for (int i = 0; i < h * w; ++i) {
        const int label = values[std::size_t(n * h * w + i)];
        if (label < 0 || label >= classes) continue;
        result[std::size_t((n * h * w + i) * (classes - 1) + c - 1)] =
            float(foreground[std::size_t(i)] ? -(std::sqrt(to_background[std::size_t(i)]) - 1)
                                             : std::sqrt(to_foreground[std::size_t(i)]));
      }
    }
  return nn::from_vector(result, {batch, h, w, classes - 1});
}

}  // namespace seg
