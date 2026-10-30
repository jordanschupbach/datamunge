#pragma once
/// @file segmentation_losses.hpp
/// @brief Tensor soft Dice/Tversky, Lovasz sorted subgradients and detached boundary distances.
#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/ops.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Internal shape checks and column views for tensor segmentation losses.
namespace segmentation_loss_detail {
/// @brief Validate floating probabilities and aligned integer label geometry.
/// @param p Positive-size floating [...,C] probabilities.
/// @param y Integer shape equal to the leading axes of p.
/// @param classes Positive expected trailing class count.
/// @throws std::invalid_argument For incompatible shapes/dtypes or invalid probabilities.
inline void Validate(const nn::Tensor &p, const nn::Tensor &y, int classes) {
  auto shape = p.shape();
  if (shape.size() < 2 || p.numel() < 1 || classes < 2 || shape.back() != classes ||
      (p.dtype() != nn::DType::Float32 && p.dtype() != nn::DType::Float64) ||
      (y.dtype() != nn::DType::Int32 && y.dtype() != nn::DType::Int64))
    throw std::invalid_argument("invalid segmentation loss geometry/dtype");
  shape.pop_back();
  if (y.shape() != shape ||
      !nn::all(nn::logical_and(nn::greater_equal(p, 0.), nn::less_equal(p, 1.))).item<bool>())
    throw std::invalid_argument("invalid segmentation probabilities/labels");
}
/// @brief View one class as a flat differentiable column.
/// @param p Floating [...,classes] tensor.
/// @param classes Trailing dimension.
/// @param c In-range class index.
/// @return Flat same-dtype class view.
inline nn::Tensor Column(const nn::Tensor &p, int classes, int c) {
  return nn::slice_axis(p.reshape({-1, classes}), 1, c, c + 1).reshape({-1});
}
} // namespace segmentation_loss_detail
/// @brief Average soft Tversky overlap loss across selected classes.
///
/// All images/pixels pool before each class overlap. Void pixels have zero
/// value/gradient. Absent classes remain in this average. Empty-valid batches
/// produce graph-connected zero. No fitting, mutation or random draws.
///
/// @param probabilities Floating [...,classes] finite probabilities in[0,1]; class sums need not equal1.
/// @param labels Aligned Int32/64 leading-shape labels; values outside[0,classes) are void.
/// @param classes At least two classes, including background0.
/// @param alpha Finite nonnegative false-positive weight, default.3.
/// @param beta Finite nonnegative false-negative weight, default.7; alpha+beta>0.
/// @param first_class First included class, default1 omits background.
/// @param epsilon Finite positive smoothing in numerator and denominator, default1e-6.
/// @return Scalar mean(1-(TP+epsilon)/(TP+alpha*FP+beta*FN+epsilon)).
/// @throws std::invalid_argument For invalid shapes/dtypes, nonfinite or out-of-range probabilities, or invalid loss settings.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto p = n::tensor({.2, .8, .7, .3, .6, .4, .9, .1}, {1, 1, 4, 2});
/// const auto y = n::from_vector(std::vector<std::int64_t>{1, 0, 1, 0}, {1, 1, 4});
/// const auto loss = n::models::SoftTverskyLoss(p, y, 2);
/// @endcode
/// @par References
/// - Salehi et al.(2017), Tversky loss function for image segmentation using3D fully convolutional deep networks, https://arxiv.org/abs/1706.05721; Berman et al.(2018), The Lovasz-Softmax loss, https://arxiv.org/abs/1705.08790; Kervadec et al.(2019), Boundary loss for highly unbalanced segmentation, https://arxiv.org/abs/1812.07032.
/// @see SoftDiceLoss(), LovaszSoftmaxLoss()
inline nn::Tensor SoftTverskyLoss(const nn::Tensor &probabilities, const nn::Tensor &labels,
                                  int classes, double alpha = .3, double beta = .7,
                                  int first_class = 1, double epsilon = 1e-6) {
  segmentation_loss_detail::Validate(probabilities, labels, classes);
  if (first_class < 0 || first_class >= classes ||
      !(alpha >= 0 && beta >= 0 && alpha + beta > 0 && epsilon > 0) ||
      !std::isfinite(alpha + beta + epsilon))
    throw std::invalid_argument("invalid Tversky settings");
  const auto flat = labels.reshape({-1}),
             valid =
                 nn::logical_and(nn::greater_equal(flat, 0.), nn::less(flat, double(classes)))
                     .to(probabilities.dtype());
  auto total = nn::sum(probabilities) * 0.;
  for (int c = first_class; c < classes; ++c) {
    const auto y = nn::equal(flat, double(c)).to(probabilities.dtype()),
               p = segmentation_loss_detail::Column(probabilities, classes, c) * valid,
               tp = nn::sum(p * y), fp = nn::sum(p * (1. - y)), fn = nn::sum((valid - p) * y);
    total = total + 1. - (tp + epsilon) / (tp + alpha * fp + beta * fn + epsilon);
  }
  return total / double(classes - first_class);
}
/// @brief Average soft Dice overlap loss using equal Tversky FP/FN weights.
///
/// Pools all pixels across the batch. No fitting, mutation or random draws.
///
/// @param probabilities Floating [...,classes] finite probabilities in[0,1].
/// @param labels Aligned Int32/64 labels; outside class range ignored.
/// @param classes At least two, background0.
/// @param first_class First included class, default1.
/// @param epsilon Finite positive numerator/denominator smoothing, default1e-6.
/// @return Scalar Tversky loss with alpha=beta=.5.
/// @throws std::invalid_argument For invalid shapes/dtypes, nonfinite or out-of-range probabilities, or invalid loss settings.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto p = n::tensor({.2, .8, .7, .3, .6, .4, .9, .1}, {1, 1, 4, 2});
/// const auto y = n::from_vector(std::vector<std::int64_t>{1, 0, 1, 0}, {1, 1, 4});
/// const auto loss = n::models::SoftDiceLoss(p, y, 2);
/// @endcode
/// @par References
/// - Salehi et al.(2017), Tversky loss function for image segmentation using3D fully convolutional deep networks, https://arxiv.org/abs/1706.05721; Berman et al.(2018), The Lovasz-Softmax loss, https://arxiv.org/abs/1705.08790; Kervadec et al.(2019), Boundary loss for highly unbalanced segmentation, https://arxiv.org/abs/1812.07032.
/// @see SoftTverskyLoss()
inline nn::Tensor SoftDiceLoss(const nn::Tensor &probabilities, const nn::Tensor &labels,
                               int classes, int first_class = 1, double epsilon = 1e-6) {
  return SoftTverskyLoss(probabilities, labels, classes, .5, .5, first_class, epsilon);
}
/// @brief Average Lovasz-Softmax sorted Jaccard surrogates over present selected classes.
///
/// Present-class filtering skips absent classes. Detached errors cast Float32
/// before stable descending sorting, preserving the source lower-index tie rule
/// even for Float64 inputs; values/derivatives use original dtype. Cumsums and
/// Jaccard differences use tensors. All-void/absent gives graph-connected zero.
/// No hidden fitting, model mutation or random draws.
///
/// @param probabilities Floating [...,classes] finite probabilities in[0,1].
/// @param labels Aligned Int32/64 labels; outside class range ignored.
/// @param classes At least two, background0.
/// @param first_class First included class, default0 includes background.
/// @return Scalar mean of sorted absolute class errors times detached Jaccard increments.
/// @throws std::invalid_argument For invalid shapes/dtypes, nonfinite or out-of-range probabilities, or invalid loss settings.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto p = n::tensor({.2, .8, .7, .3, .6, .4, .9, .1}, {1, 1, 4, 2});
/// const auto y = n::from_vector(std::vector<std::int64_t>{1, 0, 1, 0}, {1, 1, 4});
/// const auto loss = n::models::LovaszSoftmaxLoss(p, y, 2, 1);
/// @endcode
/// @par References
/// - Salehi et al.(2017), Tversky loss function for image segmentation using3D fully convolutional deep networks, https://arxiv.org/abs/1706.05721; Berman et al.(2018), The Lovasz-Softmax loss, https://arxiv.org/abs/1705.08790; Kervadec et al.(2019), Boundary loss for highly unbalanced segmentation, https://arxiv.org/abs/1812.07032.
/// @see SoftDiceLoss()
inline nn::Tensor LovaszSoftmaxLoss(const nn::Tensor &probabilities, const nn::Tensor &labels,
                                    int classes, int first_class = 0) {
  segmentation_loss_detail::Validate(probabilities, labels, classes);
  if (first_class < 0 || first_class >= classes)
    throw std::invalid_argument("invalid Lovasz classes");
  const auto type = probabilities.dtype();
  const auto truth = labels.reshape({-1}),
             valid = nn::logical_and(nn::greater_equal(truth, 0.),
                                     nn::less(truth, double(classes)))
                         .to(type);
  const auto N = truth.numel();
  auto total = nn::sum(probabilities) * 0.;
  int present = 0;
  for (int c = first_class; c < classes; ++c) {
    const auto fg = nn::equal(truth, double(c)).to(type), count = nn::sum(fg);
    if (count.item() == 0)
      continue;
    ++present;
    const auto errors =
                   nn::abs(fg - segmentation_loss_detail::Column(probabilities, classes, c)) *
                   valid,
               order = nn::argsort(errors.detach().to(nn::DType::Float32), true),
               sorted_fg = nn::gather(fg, order),
               intersection = count - nn::cumsum(sorted_fg, 0),
               combined = count + nn::cumsum(1. - sorted_fg, 0),
               jaccard = 1. - intersection / nn::maximum(combined, nn::full({}, 1e-6, type)),
               previous =
                   nn::concat({nn::zeros({1}, type), nn::slice_axis(jaccard, 0, 0, N - 1)}, 0),
               increments = (jaccard - previous).detach();
    total = total + nn::sum(nn::gather(errors, order) * increments);
  }
  return total / double(std::max(present, 1));
}
/// @brief Average foreground probability times a detached signed boundary distance map.
///
/// Use alongside a region loss; this term alone does not enforce region overlap.
/// Gradient with respect to probabilities is distance/numel; no distance gradient.
/// No fitting or input mutation.
///
/// @param foreground_probabilities Nonempty Float32/64 probabilities.
/// @param signed_distances Same-shape/same-dtype finite signed map; detached internally.
/// @return Scalar mean(probability*distance), which may be negative.
/// @throws std::invalid_argument For invalid shapes/dtypes, nonfinite or out-of-range probabilities, or invalid loss settings.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto p = n::tensor({.2, .8, .7, .3, .6, .4, .9, .1}, {1, 1, 4, 2});
/// const auto y = n::from_vector(std::vector<std::int64_t>{1, 0, 1, 0}, {1, 1, 4});
/// const auto loss = n::models::BoundaryLoss(n::slice_axis(p, 3, 1, 2), n::zeros({1, 1, 4, 1}));
/// @endcode
/// @par References
/// - Salehi et al.(2017), Tversky loss function for image segmentation using3D fully convolutional deep networks, https://arxiv.org/abs/1706.05721; Berman et al.(2018), The Lovasz-Softmax loss, https://arxiv.org/abs/1705.08790; Kervadec et al.(2019), Boundary loss for highly unbalanced segmentation, https://arxiv.org/abs/1812.07032.
/// @see SignedBoundaryDistances()
inline nn::Tensor BoundaryLoss(const nn::Tensor &foreground_probabilities,
                               const nn::Tensor &signed_distances) {
  if (foreground_probabilities.numel() < 1 ||
      foreground_probabilities.shape() != signed_distances.shape() ||
      foreground_probabilities.dtype() != signed_distances.dtype() ||
      (foreground_probabilities.dtype() != nn::DType::Float32 &&
       foreground_probabilities.dtype() != nn::DType::Float64) ||
      !nn::all(nn::less(nn::abs(signed_distances), std::numeric_limits<double>::infinity()))
           .item<bool>())
    throw std::invalid_argument("invalid boundary loss geometry/dtype/distances");
  return nn::mean(foreground_probabilities * signed_distances.detach());
}
/// @brief Compute an exact detached one-dimensional squared-distance envelope using tensor reductions.
///
/// This vectorized tensor adaptation costs O(L²) work/memory, unlike the legacy
/// linear-time host lower-envelope algorithm; values are the same. Empty L
/// returns an empty vector. It is preprocessing with no optimizer or gradients.
///
/// @param cost Floating[L] finite costs or positive infinity; NaN/negative infinity rejected.
/// @return Float64[L] min_j(cost[j]+(i-j)²), infinity when every site is infinite.
/// @throws std::invalid_argument For invalid shapes/dtypes, nonfinite or out-of-range probabilities, or invalid loss settings.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto d = n::models::SquaredDistance1D(n::tensor({0., INFINITY, INFINITY}));
/// @endcode
/// @par References
/// - Salehi et al.(2017), Tversky loss function for image segmentation using3D fully convolutional deep networks, https://arxiv.org/abs/1706.05721; Berman et al.(2018), The Lovasz-Softmax loss, https://arxiv.org/abs/1705.08790; Kervadec et al.(2019), Boundary loss for highly unbalanced segmentation, https://arxiv.org/abs/1812.07032.
/// @see SquaredDistance2D()
inline nn::Tensor SquaredDistance1D(const nn::Tensor &cost) {
  if (cost.dim() != 1 ||
      (cost.dtype() != nn::DType::Float32 && cost.dtype() != nn::DType::Float64) ||
      !nn::all(nn::greater(cost, -std::numeric_limits<double>::infinity())).item<bool>())
    throw std::invalid_argument("invalid distance costs");
  nn::NoGradGuard guard;
  const auto L = cost.numel();
  if (!L)
    return nn::zeros({0}, nn::DType::Float64);
  const auto x = nn::arange(0, double(L)).to(nn::DType::Float64);
  return nn::min(cost.to(nn::DType::Float64).reshape({1, L}) +
                     nn::square(x.reshape({L, 1}) - x.reshape({1, L})),
                 {1});
}
/// @brief Compute exact detached squared Euclidean nearest-seed distances on a rectangular grid.
///
/// Two separable tensor reductions cost O(H*W*(H+W)) work and
/// O(H*W*max(H,W)) temporary memory. This preserves exact values while
/// replacing the source linear-time host algorithm with tensor arithmetic.
/// No input mutation, fitting or gradients.
///
/// @param seeds Nonempty Bool[H,W] true at seed sites.
/// @return Float64[H,W] nearest squared distance, infinity if no seed exists.
/// @throws std::invalid_argument For invalid shapes/dtypes, nonfinite or out-of-range probabilities, or invalid loss settings.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto seeds = n::equal(n::arange(0, 12).reshape({3, 4}), 5.);
/// const auto d = n::models::SquaredDistance2D(seeds);
/// @endcode
/// @par References
/// - Salehi et al.(2017), Tversky loss function for image segmentation using3D fully convolutional deep networks, https://arxiv.org/abs/1706.05721; Berman et al.(2018), The Lovasz-Softmax loss, https://arxiv.org/abs/1705.08790; Kervadec et al.(2019), Boundary loss for highly unbalanced segmentation, https://arxiv.org/abs/1812.07032.
/// @see SignedBoundaryDistances()
inline nn::Tensor SquaredDistance2D(const nn::Tensor &seeds) {
  if (seeds.dim() != 2 || seeds.shape()[0] < 1 || seeds.shape()[1] < 1 ||
      seeds.dtype() != nn::DType::Bool)
    throw std::invalid_argument("distance seeds require Bool[H,W]");
  nn::NoGradGuard guard;
  const auto H = seeds.shape()[0], W = seeds.shape()[1];
  const auto type = nn::DType::Float64;
  const auto x = nn::arange(0, double(W)).to(type), y = nn::arange(0, double(H)).to(type),
             cost = nn::where(seeds, nn::zeros({}, type),
                              nn::full({}, std::numeric_limits<double>::infinity(), type)),
             horizontal = nn::min(cost.reshape({H, 1, W}) +
                                      nn::square(x.reshape({1, W, 1}) - x.reshape({1, 1, W})),
                                  {2});
  return nn::min(horizontal.reshape({1, H, W}) +
                     nn::square(y.reshape({H, 1, 1}) - y.reshape({1, H, 1})),
                 {1});
}
/// @brief Build detached signed boundary maps for each foreground class.
///
/// Outside uses distance to foreground; inside uses -(distance to background-1),
/// so boundary foreground pixels are zero. Absent/full foreground classes have
/// zero maps because no internal boundary exists. Void pixels are background
/// for distance topology but their final map entries are zero. Uses exact tensor
/// separable distances with quadratic axis work, not the legacy linear host EDT.
/// No gradients, random draws, fitted state or mutation.
///
/// @param labels Nonempty Int32/64[N,H,W] masks; out-of-range labels are void.
/// @param classes At least two classes, background0.
/// @return Float32[N,H,W,classes-1] maps with zero void pixels.
/// @throws std::invalid_argument For invalid shapes/dtypes, nonfinite or out-of-range probabilities, or invalid loss settings.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto y = n::zeros({1, 5, 7}, n::DType::Int64);
/// const auto d = n::models::SignedBoundaryDistances(y, 2);
/// @endcode
/// @par References
/// - Salehi et al.(2017), Tversky loss function for image segmentation using3D fully convolutional deep networks, https://arxiv.org/abs/1706.05721; Berman et al.(2018), The Lovasz-Softmax loss, https://arxiv.org/abs/1705.08790; Kervadec et al.(2019), Boundary loss for highly unbalanced segmentation, https://arxiv.org/abs/1812.07032.
/// @see BoundaryLoss(), SquaredDistance2D()
inline nn::Tensor SignedBoundaryDistances(const nn::Tensor &labels, int classes) {
  if (labels.dim() != 3 || labels.shape()[0] < 1 || labels.shape()[1] < 1 ||
      labels.shape()[2] < 1 || classes < 2 ||
      (labels.dtype() != nn::DType::Int32 && labels.dtype() != nn::DType::Int64))
    throw std::invalid_argument("signed distances require integer NHW labels");
  nn::NoGradGuard guard;
  const auto N = labels.shape()[0], H = labels.shape()[1], W = labels.shape()[2];
  std::vector<nn::Tensor> batches;
  for (std::int64_t row = 0; row < N; ++row) {
    const auto y = nn::slice_axis(labels, 0, row, row + 1).reshape({H, W}),
               valid = nn::logical_and(nn::greater_equal(y, 0.), nn::less(y, double(classes)));
    std::vector<nn::Tensor> maps;
    for (int c = 1; c < classes; ++c) {
      const auto fg = nn::equal(y, double(c));
      const auto count = nn::count_nonzero(fg).item<std::int64_t>();
      auto d = nn::zeros({H, W}, nn::DType::Float64);
      if (count > 0 && count < H * W)
        d = nn::where(valid,
                      nn::where(fg, 1. - nn::sqrt(SquaredDistance2D(nn::logical_not(fg))),
                                nn::sqrt(SquaredDistance2D(fg))),
                      nn::zeros({}, nn::DType::Float64));
      maps.push_back(d.to(nn::DType::Float32));
    }
    batches.push_back(nn::stack(maps, 2));
  }
  return nn::stack(batches, 0);
}
} // namespace datamunge::nn::models
