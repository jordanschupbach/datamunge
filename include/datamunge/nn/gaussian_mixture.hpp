// datamunge::nn port of tflow_examples src/common/ml/gaussian_mixture.h.
// Full-covariance Gaussian-mixture EM with float64 nn operations.
// E step: batched Cholesky solves, log determinants and log-sum-exp. M step:
// responsibility-weighted means/covariances and normalized mixture masses.
// Clamp covariance eigenvalues to a positive floor (constrained covariance
// MLE), rather than adding a ridge to every eigenvalue. Reject a collapsed
// component with negligible effective count; callers can restart the fit.
// Small datasets only: the M step materializes an [N,K,D,D] temporary.
#pragma once

/// @file gaussian_mixture.hpp
/// @brief Full-covariance Gaussian-mixture expectation-maximization.

#include <cmath>
#include <datamunge/nn/autograd.hpp>
#include <datamunge/nn/linalg.hpp>
#include <datamunge/nn/ops.hpp>
#include <stdexcept>

namespace datamunge::nn::ml {
namespace nn = datamunge::nn;

/// @brief Owned tensor handles for full-covariance mixture parameters.
///
/// Float64 weights [K], means [K,D] and positive-definite covariances [K,D,D].
/// Handles may alias caller tensors; Step does not mutate them.
///
/// @par Example
/// @code
/// namespace n=dm::nn; namespace m=n::ml;
/// m::GaussianMixtureState
/// s{n::ones({1},n::DType::Float64),n::zeros({1,1},n::DType::Float64),n::ones({1,1,1},n::DType::Float64)};
/// @endcode
/// @see GaussianMixture
struct GaussianMixtureState {
  nn::Tensor weights;     ///< Positive Float64 [K] masses, summing to one.
  nn::Tensor means;       ///< Float64 [K,D] centres.
  nn::Tensor covariances; ///< Float64 positive-definite [K,D,D] matrices.
};
/// @brief E-step diagnostics at input state and the subsequent M-step parameters.
///
/// @par Example
/// @code
/// namespace n=dm::nn; namespace m=n::ml;
/// m::GaussianMixtureState
/// s{n::ones({1},n::DType::Float64),n::zeros({1,1},n::DType::Float64),n::ones({1,1,1},n::DType::Float64)};
/// const auto step=m::GaussianMixture(1,1).Step(n::tensor({-1,1},{2,1},n::DType::Float64),s);
/// @endcode
/// @see GaussianMixture::Step()
struct GaussianMixtureStep {
  GaussianMixtureState next;               ///< Parameters after one constrained M step.
  nn::Tensor           responsibilities;   ///< Float64 [N,K] posterior membership at input state.
  nn::Tensor           log_density;        ///< Float64 [N] mixture log density at input state.
  double               log_likelihood = 0; ///< Sum of log_density at the input state.
};

/// @brief Small-data full-covariance Gaussian-mixture EM with an eigenvalue floor.
///
/// Uses Cholesky densities, log-sum-exp responsibilities and responsibility-weighted
/// MLE updates. Covariance eigenvalues are clamped, not ridge-shifted. Step
/// materialises [N,K,D,D], so memory is O(NKD²); covariance decomposition costs O(KD³).
/// This is an EM step API, not automatic initialization or global optimization.
///
/// @par Example
/// @code
/// namespace n=dm::nn; namespace m=n::ml;
/// m::GaussianMixtureState
/// s{n::ones({1},n::DType::Float64),n::zeros({1,1},n::DType::Float64),n::ones({1,1,1},n::DType::Float64)};
/// const auto step=m::GaussianMixture(1,1).Step(n::tensor({-1,1},{2,1},n::DType::Float64),s);
/// @endcode
///
/// @par References
/// - Dempster, Laird and Rubin (1977), Maximum likelihood from incomplete data via the EM
/// algorithm, doi:10.1111/j.2517-6161.1977.tb01600.x
/// @see GaussianMixtureState
class GaussianMixture {
 public:
  /// @brief Validate fixed dimensions and covariance constraint.
  /// @param components Positive K.
  /// @param dimensions Positive D.
  /// @param covariance_floor Positive finite minimum covariance eigenvalue, default 1e-4.
  /// @throws std::invalid_argument On invalid dimensions/floor.
  ///
  /// @par Example
  /// @code
  /// namespace n=dm::nn; namespace m=n::ml;
  /// const m::GaussianMixture model(2,1,.001);
  /// @endcode
  /// @see Step()
  GaussianMixture(int components, int dimensions, double covariance_floor = 1e-4)
      : k_(components), d_(dimensions), floor_(covariance_floor) {
    if (k_ <= 0 || d_ <= 0 || !std::isfinite(covariance_floor) || covariance_floor <= 0)
      throw std::invalid_argument("positive GMM dimensions and covariance floor required");
  }

  /// @brief Evaluate current likelihood/responsibilities and construct one constrained M step.
  ///
  /// No gradients are recorded. Source state is unchanged. The reported likelihood
  /// belongs to state, not next. Collapsed components cause failure; callers may restart.
  /// @param x Nonempty Float64 [N,D] data, finite by caller precondition.
  /// @param state Matching Float64 state, positive weights summing to one and positive-definite
  /// covariances.
  /// @return Current responsibilities [N,K], log density [N], their sum, and next parameters.
  /// @throws std::invalid_argument On bad shapes/weights; std::runtime_error on failed
  /// factorisation or effective count below 1e-8.
  ///
  /// @par Example
  /// @code
  /// namespace n=dm::nn; namespace m=n::ml;
  /// m::GaussianMixtureState
  /// s{n::ones({1},n::DType::Float64),n::zeros({1,1},n::DType::Float64),n::ones({1,1,1},n::DType::Float64)};
  /// const auto step=m::GaussianMixture(1,1).Step(n::tensor({-1,1},{2,1},n::DType::Float64),s);
  /// @endcode
  /// @see GaussianMixtureState
  GaussianMixtureStep Step(const nn::Tensor& x, const GaussianMixtureState& state) const {
    using nn::DType;
    if (x.dtype() != DType::Float64 || x.dim() != 2 || x.size(0) < 1 || x.size(1) != d_ ||
        state.weights.dtype() != DType::Float64 || state.weights.shape() != nn::Shape{k_} ||
        state.means.dtype() != DType::Float64 || state.means.shape() != nn::Shape{k_, d_} ||
        state.covariances.dtype() != DType::Float64 ||
        state.covariances.shape() != nn::Shape{k_, d_, d_})
      throw std::invalid_argument("GMM requires matching float64 data/state shapes");
    double sum = 0;
    for (double weight : state.weights.to_vector<double>()) {
      if (!std::isfinite(weight) || weight <= 0)
        throw std::invalid_argument("mixture weights must be positive");
      sum += weight;
    }
    if (std::abs(sum - 1) > 1e-10)
      throw std::invalid_argument("mixture weights must sum to one");
    nn::NoGradGuard guard;
    const auto      delta = nn::expand_dims(x, 1) - nn::expand_dims(state.means, 0);    // [N, K, D]
    const auto      chol  = nn::cholesky(state.covariances);                            // [K, D, D]
    const auto whitened  = nn::triangular_solve(chol, nn::transpose(delta, {1, 2, 0})); // [K, D, N]
    const auto quadratic = nn::transpose(nn::sum(nn::square(whitened), {1}), {1, 0});   // [N, K]
    const auto logdet    = 2.0 * nn::sum(nn::log(nn::matrix_diag_part(chol)), {1});     // [K]
    const auto log_joint =
        nn::log(state.weights) + -.5 * (quadratic + (logdet + d_ * std::log(2 * std::acos(-1.))));
    const auto maximum        = nn::max(log_joint, {1}, true);
    const auto log_normalizer = maximum + nn::log(nn::sum(nn::exp(log_joint - maximum), {1}, true));
    GaussianMixtureStep out;
    out.responsibilities = nn::exp(log_joint - log_normalizer);
    out.log_density      = nn::reshape(log_normalizer, {-1});
    out.log_likelihood   = nn::sum(out.log_density, {0}).item();
    const auto counts    = nn::sum(out.responsibilities, {0});
    const auto safe      = nn::maximum(counts, nn::scalar_like(1e-100, counts));
    out.next.weights     = counts / nn::sum(counts, {0});
    out.next.means = nn::matmul(out.responsibilities, x, true, false) / nn::expand_dims(safe, 1);
    const auto centered   = nn::expand_dims(x, 1) - nn::expand_dims(out.next.means, 0);
    const auto outer      = nn::expand_dims(centered, 3) * nn::expand_dims(centered, 2);
    const auto weighted   = outer * nn::expand_dims(nn::expand_dims(out.responsibilities, 2), 3);
    const auto covariance = nn::sum(weighted, {0}) / nn::reshape(safe, {k_, 1, 1});
    const auto eigen      = nn::self_adjoint_eig(covariance);
    const auto clipped    = nn::maximum(eigen.e, nn::scalar_like(floor_, eigen.e));
    out.next.covariances  = nn::matmul(eigen.v * nn::expand_dims(clipped, 1), eigen.v, false, true);
    for (double c : counts.to_vector<double>())
      if (!std::isfinite(c) || c < 1e-8)
        throw std::runtime_error("GMM component collapsed; restart with different initialization");
    return out;
  }

 private:
  std::int64_t k_;     ///< Fixed component count.
  std::int64_t d_;     ///< Fixed observation width.
  double       floor_; ///< Positive minimum updated covariance eigenvalue.
};

} // namespace datamunge::nn::ml
