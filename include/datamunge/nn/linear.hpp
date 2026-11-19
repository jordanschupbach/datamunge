// datamunge::nn port of tflow_examples src/common/ml/linear.h.
// Linear least squares and proximal regularization built with nn ops.
// Cholesky and thin QR require a full-column-rank design with rows >= columns.
// Neither forms an explicit inverse. QR avoids squaring the condition number.
// ElasticNetStep minimizes ||Xw-y||^2/(2N) + l1*sum(mask*|w|)
// + l2*sum(mask*w^2)/2. A zero mask entry leaves an intercept unpenalized.
// The proximal step assumes nonnegative penalties/mask and a
// rate <= 1/(||X||_2^2/N + l2*max(mask)). It returns the next value, not an
// assignment, so callers control parameter lifetime and update ordering.
#pragma once

/// @file linear.hpp
/// @brief Tensor least-squares, Bayesian Gaussian regression and proximal elastic-net steps.

#include <cmath>
#include <datamunge/nn/autograd.hpp>
#include <datamunge/nn/linalg.hpp>
#include <datamunge/nn/losses.hpp>
#include <datamunge/nn/ops.hpp>
#include <numbers>
#include <stdexcept>

/// @brief Small tensor-based statistical workflows and validation utilities.
namespace datamunge::nn::ml {
namespace nn = datamunge::nn;

/// @brief Solve full-column-rank least squares by Cholesky normal equations.
///
/// No explicit inverse is formed. QR avoids squaring the condition number.
/// Rank deficiency is not reliably diagnosed by a dedicated rank test; callers must
/// check their design. Decomposition operations retain their usual autograd behavior.
/// @param x Floating design [N,p], N >= p, full column rank.
/// @param y Floating response [N,r], same dtype; multiple right-hand sides allowed.
/// @return Coefficient tensor [p,r]; inputs unchanged.
/// @throws std::invalid_argument On incompatible tensor shapes; std::runtime_error on a failed
/// factorisation.
///
/// @par Example
/// @code
/// namespace n=dm::nn; namespace m=n::ml;
/// const auto x=n::tensor({1,0,1,1,1,2},{3,2},n::DType::Float64);
/// const auto y=n::tensor({1,3,5},{3,1},n::DType::Float64);
/// const auto w=m::LeastSquaresCholesky(x,y); // [1,2]
/// @endcode
///
/// @par References
/// - LAPACK Users Guide, https://www.netlib.org/lapack/lug/node27.html
/// @see LeastSquaresQr(), LeastSquaresCholesky()
inline nn::Tensor LeastSquaresCholesky(const nn::Tensor& x, const nn::Tensor& y) {
  const auto gram  = nn::matmul(x, x, true, false);
  const auto rhs   = nn::matmul(x, y, true, false);
  const auto lower = nn::cholesky(gram);
  const auto z     = nn::triangular_solve(lower, rhs);
  return nn::triangular_solve(lower, z, /*lower=*/true, /*adjoint=*/true);
}

/// @brief Solve full-column-rank least squares by thin QR.
///
/// No explicit inverse is formed. QR avoids squaring the condition number.
/// Rank deficiency is not reliably diagnosed by a dedicated rank test; callers must
/// check their design. Decomposition operations retain their usual autograd behavior.
/// @param x Floating design [N,p], N >= p, full column rank.
/// @param y Floating response [N,r], same dtype; multiple right-hand sides allowed.
/// @return Coefficient tensor [p,r]; inputs unchanged.
/// @throws std::invalid_argument On incompatible tensor shapes; std::runtime_error on a failed
/// factorisation.
///
/// @par Example
/// @code
/// namespace n=dm::nn; namespace m=n::ml;
/// const auto x=n::tensor({1,0,1,1,1,2},{3,2},n::DType::Float64);
/// const auto y=n::tensor({1,3,5},{3,1},n::DType::Float64);
/// const auto w=m::LeastSquaresQr(x,y); // [1,2]
/// @endcode
///
/// @par References
/// - LAPACK Users Guide, https://www.netlib.org/lapack/lug/node27.html
/// @see LeastSquaresQr(), LeastSquaresCholesky()
inline nn::Tensor LeastSquaresQr(const nn::Tensor& x, const nn::Tensor& y) {
  const auto qr  = nn::qr(x);
  const auto rhs = nn::matmul(qr.q, y, true, false);
  return nn::triangular_solve(qr.r, rhs, /*lower=*/false);
}

/// @brief Exact Gaussian posterior for a linear model with known observation variance.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto fit = n::ml::FitBayesianLinear(n::ones({1, 1}), n::tensor({2.}), 1.);
/// const auto coefficient = fit.mean.item(); // 1
/// @endcode
/// @see FitBayesianLinear(), PredictBayesianLinear()
struct BayesianLinearFit {
  nn::Tensor mean;               ///< Detached Float64 posterior coefficient mean [P,1].
  nn::Tensor covariance;         ///< Detached Float64 coefficient covariance [P,P].
  nn::Tensor precision_cholesky; ///< Detached Float64 lower Cholesky factor of posterior precision.
  double observation_variance{}; ///< Known response noise variance.
  double log_evidence{};          ///< Exact log marginal density of the fitting responses.
};

/// @brief Marginal response predictions from a Gaussian linear posterior.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto x = n::ones({1, 1});
/// const auto fit = n::ml::FitBayesianLinear(x, n::tensor({2.}), 1.);
/// const auto predicted = n::ml::PredictBayesianLinear(fit, x);
/// const auto variance = predicted.variance.item(); // 1.5
/// @endcode
/// @see PredictBayesianLinear()
struct BayesianLinearPrediction {
  nn::Tensor mean;     ///< Detached Float64 mean [Q] for each query.
  nn::Tensor variance; ///< Detached Float64 marginal variance [Q], including observation noise.
};

/// @brief Condition a zero-mean isotropic Gaussian coefficient prior using a Gaussian likelihood.
///
/// Uses Float64 Cholesky solves for posterior precision and exact model evidence.
/// The proper prior regularizes rank-deficient designs. The log evidence evaluates
/// the residual sum of squares at the posterior mean plus its prior quadratic.
/// Time is O(N P² + P³) for N observations and P columns. Tensors are detached.
/// @param x Nonempty finite CPU design [N,P].
/// @param y Finite CPU response [N] or [N,1].
/// @param observation_variance Known positive finite response noise variance with finite reciprocal.
/// @param prior_variance Positive finite coefficient variance with finite reciprocal, default 1.
/// @return Posterior coefficient moments, factor and log evidence.
/// @throws std::invalid_argument For invalid shapes, devices, values or variances.
/// @throws std::runtime_error For failed factorization or nonfinite calculations.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto fit = n::ml::FitBayesianLinear(n::ones({1, 1}), n::tensor({2.}), 1.);
/// const auto posterior_variance = fit.covariance.item(); // 0.5
/// @endcode
/// @par References
/// - Rasmussen and Williams (2006), Gaussian Processes for Machine Learning, Chapter 2.
///   https://gaussianprocess.org/gpml/chapters/RW2.pdf
/// @see BayesianLinearFit, PredictBayesianLinear()
inline BayesianLinearFit FitBayesianLinear(const nn::Tensor& x, const nn::Tensor& y,
                                           double observation_variance,
                                           double prior_variance = 1.) {
  const bool valid_y = y.defined() && (y.dim() == 1 || (y.dim() == 2 && y.size(1) == 1));
  if (!x.defined() || x.dim() != 2 || x.numel() == 0 || !valid_y || x.size(0) != y.size(0) ||
      x.device().backend != compute::Backend::CPU || y.device().backend != compute::Backend::CPU ||
      !std::isfinite(observation_variance) || observation_variance <= 0. ||
      !std::isfinite(prior_variance) || prior_variance <= 0. ||
      !std::isfinite(1. / observation_variance) || !std::isfinite(1. / prior_variance))
    throw std::invalid_argument("invalid Bayesian linear regression design, response or variance");
  nn::NoGradGuard guard;
  const auto design = x.to(nn::DType::Float64);
  const auto response = y.to(nn::DType::Float64).reshape({x.size(0), 1});
  if (!nn::all(nn::is_finite(design)).item<bool>() ||
      !nn::all(nn::is_finite(response)).item<bool>())
    throw std::invalid_argument("Bayesian linear regression inputs must be finite");

  const auto identity = nn::eye(x.size(1), nn::DType::Float64);
  const auto precision = identity / prior_variance +
                         nn::matmul(design, design, true, false) / observation_variance;
  if (!nn::all(nn::is_finite(precision)).item<bool>())
    throw std::runtime_error("Bayesian linear posterior precision overflowed");
  const auto lower = nn::cholesky(precision);
  const auto rhs = nn::matmul(design, response, true, false) / observation_variance;
  const auto mean = nn::cholesky_solve(lower, rhs);
  const auto covariance = nn::cholesky_solve(lower, identity);
  const double quadratic =
      nn::sum(nn::square(response - nn::matmul(design, mean))).item() / observation_variance +
      nn::sum(nn::square(mean)).item() / prior_variance;
  const double log_determinant =
      2. * nn::sum(nn::log(nn::matrix_diag_part(lower))).item() +
      x.size(1) * std::log(prior_variance);
  const double log_evidence = -.5 * (quadratic + log_determinant +
      x.size(0) * (std::log(2. * std::numbers::pi) + std::log(observation_variance)));
  if (!std::isfinite(log_evidence) || !nn::all(nn::is_finite(covariance)).item<bool>())
    throw std::runtime_error("Bayesian linear posterior or log evidence overflowed");
  return {mean, covariance, lower, observation_variance, log_evidence};
}

/// @brief Predict responses using a fitted Gaussian linear model.
///
/// A triangular solve measures coefficient uncertainty in each query direction.
/// The observation variance is added for new responses. The result contains the
/// marginal variances, with no joint predictive covariance.
/// @param fit Valid posterior returned by FitBayesianLinear().
/// @param queries Nonempty finite CPU design [Q,P] with the fitting column order.
/// @return Detached Float64 means and variances, each shaped [Q].
/// @throws std::invalid_argument For invalid query shapes, devices or values.
/// @throws std::runtime_error If prediction overflows.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto x = n::ones({1, 1});
/// const auto fit = n::ml::FitBayesianLinear(x, n::tensor({2.}), 1.);
/// const auto at_training_point = n::ml::PredictBayesianLinear(fit, x).mean.item(); // 1
/// @endcode
/// @see BayesianLinearPrediction, FitBayesianLinear()
inline BayesianLinearPrediction PredictBayesianLinear(const BayesianLinearFit& fit,
                                                       const nn::Tensor& queries) {
  if (!queries.defined() || queries.dim() != 2 || queries.numel() == 0 ||
      queries.size(1) != fit.mean.size(0) || queries.device().backend != compute::Backend::CPU)
    throw std::invalid_argument("invalid Bayesian linear regression query design");
  nn::NoGradGuard guard;
  const auto design = queries.to(nn::DType::Float64);
  if (!nn::all(nn::is_finite(design)).item<bool>())
    throw std::invalid_argument("Bayesian linear regression queries must be finite");
  const auto mean = nn::matmul(design, fit.mean).reshape({queries.size(0)});
  const auto projected = nn::triangular_solve(fit.precision_cholesky, nn::transpose(design));
  const auto variance = fit.observation_variance + nn::sum(nn::square(projected), {0});
  if (!nn::all(nn::is_finite(mean)).item<bool>() || !nn::all(nn::is_finite(variance)).item<bool>())
    throw std::runtime_error("Bayesian linear prediction overflowed");
  return {mean, variance};
}

/// @brief Gaussian approximation around a Bayesian logistic-regression posterior mode.
struct BayesianLogisticLaplaceFit {
  nn::Tensor mode;       ///< Detached Float64 coefficient mode [features, 1].
  nn::Tensor gradient;   ///< Gradient of the negative log posterior at mode.
  nn::Tensor precision;  ///< Hessian at the returned mode, [features, features].
  nn::Tensor covariance; ///< Inverse Hessian at the returned mode.
};

/// @brief Fit binary logistic regression under an isotropic zero-mean normal prior.
/// Newton solves use the exact posterior Hessian and Cholesky factorization.
/// Backtracking reduces the step if the negative log posterior would increase.
/// The final gradient allows callers to check stationarity.
/// @param x Finite CPU design [rows, features], including any desired intercept column.
/// @param y CPU binary observations [rows] or [rows, 1].
/// @param prior_variance Positive finite variance of each independent coefficient.
/// @param max_iterations Maximum number of Newton updates.
/// @return Detached coefficient mode and local posterior curvature in Float64.
/// @throws std::invalid_argument Invalid shapes, observations or settings.
/// @throws std::runtime_error Failed numerical optimization.
/// @code
/// namespace n = dm::nn;
/// auto fit = n::ml::FitBayesianLogisticLaplace(n::ones({2, 1}), n::tensor({0., 1.}, {2}));
/// @endcode
inline BayesianLogisticLaplaceFit FitBayesianLogisticLaplace(
    const nn::Tensor& x, const nn::Tensor& y, double prior_variance = 1.,
    std::size_t max_iterations = 25) {
  const bool valid_y = y.defined() && (y.dim() == 1 || (y.dim() == 2 && y.size(1) == 1));
  if (!x.defined() || x.dim() != 2 || x.numel() <= 0 || !valid_y || y.size(0) != x.size(0) ||
      x.device().backend != compute::Backend::CPU || y.device().backend != compute::Backend::CPU ||
      !std::isfinite(prior_variance) || prior_variance <= 0. || !std::isfinite(1. / prior_variance) ||
      max_iterations == 0) {
    throw std::invalid_argument("invalid Bayesian logistic regression data or settings");
  }
  nn::NoGradGuard guard;
  const auto design = nn::cast(x, nn::DType::Float64);
  const auto response = nn::cast(y, nn::DType::Float64).reshape({x.size(0), 1});
  if (!nn::all(nn::is_finite(design)).item<bool>() ||
      !nn::all(nn::logical_or(nn::equal(response, 0.), nn::equal(response, 1.))).item<bool>()) {
    throw std::invalid_argument("Bayesian logistic regression requires a finite design and binary responses");
  }

  const auto identity = nn::eye(x.size(1), nn::DType::Float64);
  const auto prior = identity / prior_variance;
  const auto objective = [&](const nn::Tensor& w) {
    return nn::sigmoid_cross_entropy(nn::matmul(design, w), response, nn::Reduction::Sum).item() +
           .5 * nn::sum(nn::square(w)).item() / prior_variance;
  };
  const auto derivatives = [&](const nn::Tensor& w) {
    const auto p = nn::sigmoid(nn::matmul(design, w));
    const auto gradient = w / prior_variance + nn::matmul(design, p - response, true, false);
    const auto hessian = prior + nn::matmul(design, design * p * (1. - p), true, false);
    return std::pair{gradient, hessian};
  };
  auto mode = nn::zeros({x.size(1), 1}, nn::DType::Float64);
  for (std::size_t i = 0; i < max_iterations; ++i) {
    const auto [gradient, hessian] = derivatives(mode);
    if (nn::max(nn::abs(gradient)).item() < 1e-10)
      break;
    const auto direction = nn::cholesky_solve(nn::cholesky(hessian), gradient);
    const double current = objective(mode);
    double step = 1.;
    auto candidate = mode - direction;
    while (!(objective(candidate) <= current)) {
      step *= .5;
      if (step < 1e-12)
        throw std::runtime_error("Bayesian logistic Newton backtracking failed");
      candidate = mode - step * direction;
    }
    mode = candidate;
  }
  const auto [gradient, hessian] = derivatives(mode);
  return {mode, gradient, hessian, nn::cholesky_solve(nn::cholesky(hessian), identity)};
}

/// @brief Return one proximal-gradient elastic-net iterate without assigning it.
///
/// The smooth gradient is X-transpose residual / N + l2*mask*w. The
/// threshold is rate*l1*mask. Callers ensure nonnegative mask and a stable
/// rate; these spectral/mask preconditions are not measured by the helper.
/// @param x Floating design [N,p], N positive.
/// @param y Response [N,r].
/// @param w Current coefficients [p,r].
/// @param mask Nonnegative broadcastable penalty weights; zero excludes an intercept.
/// @param l1 Finite nonnegative absolute penalty.
/// @param l2 Finite nonnegative squared penalty.
/// @param rate Finite positive step, at most reciprocal smooth Lipschitz constant for descent.
/// @return Soft-thresholded next coefficient tensor.
/// @throws std::invalid_argument On invalid penalties/rate or incompatible tensor operations.
///
/// @par Example
/// @code
/// namespace n=dm::nn; namespace m=n::ml;
/// const auto x=n::tensor({1,0,1,1,1,2},{3,2});
/// const auto
/// next=m::ElasticNetStep(x,n::ones({3,1}),n::zeros({2,1}),n::tensor({0,1},{2,1}),.1f,.2f,.1f);
/// @endcode
///
/// @par References
/// - Parikh and Boyd (2014), Proximal Algorithms,
/// https://web.stanford.edu/~boyd/papers/prox_algs.html
/// @see LeastSquaresQr()
inline nn::Tensor ElasticNetStep(
    const nn::Tensor& x,
    const nn::Tensor& y,
    const nn::Tensor& w,
    const nn::Tensor& mask,
    float             l1,
    float             l2,
    float             rate) {
  if (l1 < 0 || l2 < 0 || rate <= 0 || !std::isfinite(l1 + l2 + rate))
    throw std::invalid_argument("invalid elastic-net step parameters");
  const double n         = static_cast<double>(x.size(0));
  const auto   residual  = nn::matmul(x, w) - y;
  const auto   gradient  = nn::matmul(x, residual, true, false) / n + l2 * (mask * w);
  const auto   z         = w - rate * gradient;
  const auto   threshold = (static_cast<double>(l1) * rate) * mask;
  return nn::sign(z) * nn::maximum(nn::abs(z) - threshold, nn::scalar_like(0.0, z));
}

} // namespace datamunge::nn::ml
