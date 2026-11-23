// datamunge::nn port of tflow_examples src/common/ml/gaussian_mixture.h.
// Gaussian-mixture EM with float64 nn operations.
// E step: batched Cholesky solves, log determinants and log-sum-exp. M step:
// responsibility-weighted means/covariances and normalized mixture masses.
// Clamp covariance eigenvalues to a positive floor (constrained covariance
// MLE), rather than adding a ridge to every eigenvalue. Reject a collapsed
// component with negligible effective count; callers can restart the fit.
// Small datasets only: the M step materializes an [N,K,D,D] temporary.
#pragma once

/// @file gaussian_mixture.hpp
/// @brief Gaussian-mixture expectation maximization and selection by BIC.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <datamunge/nn/autograd.hpp>
#include <datamunge/nn/linalg.hpp>
#include <datamunge/nn/ops.hpp>
#include <limits>
#include <stdexcept>
#include <vector>

namespace datamunge::nn::ml {
namespace nn = datamunge::nn;

/// @brief Covariance constraint applied in the Gaussian mixture M step.
enum class GaussianMixtureCovariance {
  Tied,      ///< One shared unrestricted covariance.
  Diagonal, ///< A different diagonal covariance per component.
  Full      ///< A different unrestricted covariance per component.
};

/// @brief Owned tensor handles for Gaussian mixture parameters.
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

/// @brief Small-data Gaussian-mixture EM with an eigenvalue floor.
///
/// Uses Cholesky densities, log-sum-exp responsibilities and responsibility-weighted
/// MLE updates. Covariance eigenvalues are clamped, not ridge-shifted. Step
/// materialises [N,K,D,D] for unrestricted covariances. The diagonal update
/// uses [N,K,D].
/// @see select_gaussian_mixture()
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
  /// @param covariance Covariance constraint used by the M step, full by default.
  /// @throws std::invalid_argument On invalid dimensions, floor or constraint.
  ///
  /// @par Example
  /// @code
  /// namespace n=dm::nn; namespace m=n::ml;
  /// const m::GaussianMixture model(2,1,.001);
  /// @endcode
  /// @see Step()
  GaussianMixture(int components, int dimensions, double covariance_floor = 1e-4,
      GaussianMixtureCovariance covariance = GaussianMixtureCovariance::Full)
      : k_(components), d_(dimensions), floor_(covariance_floor), covariance_(covariance) {
    if (k_ <= 0 || d_ <= 0 || !std::isfinite(covariance_floor) || covariance_floor <= 0)
      throw std::invalid_argument("positive GMM dimensions and covariance floor required");
    if (covariance_ != GaussianMixtureCovariance::Tied &&
        covariance_ != GaussianMixtureCovariance::Diagonal &&
        covariance_ != GaussianMixtureCovariance::Full)
      throw std::invalid_argument("unknown GMM covariance constraint");
  }

  /// @brief Evaluate current likelihood/responsibilities and construct one constrained M step.
  ///
  /// No gradients are recorded. Source state is unchanged. The reported likelihood
  /// belongs to state, not next. Collapsed components cause failure; callers may restart.
  /// @param x Nonempty Float64 [N,D] data, finite by caller precondition.
  /// @param state Matching Float64 state, positive weights summing to one and positive-definite
  /// covariances. For EM within one model, initialize its covariances to match the constraint.
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
    for (double c : counts.to_vector<double>())
      if (!std::isfinite(c) || c < 1e-8)
        throw std::runtime_error("GMM component collapsed; restart with different initialization");
    const auto total = nn::sum(counts, {0});
    out.next.weights = counts / total;
    out.next.means = nn::matmul(out.responsibilities, x, true, false) / nn::expand_dims(counts, 1);
    const auto centered   = nn::expand_dims(x, 1) - nn::expand_dims(out.next.means, 0);
    const auto weights = nn::expand_dims(out.responsibilities, 2);
    if (covariance_ == GaussianMixtureCovariance::Diagonal) {
      const auto variance = nn::sum(nn::square(centered) * weights, {0}) / nn::expand_dims(counts, 1);
      out.next.covariances = nn::matrix_diag(nn::maximum(variance, nn::scalar_like(floor_, variance)));
      return out;
    }
    const auto outer      = nn::expand_dims(centered, 3) * nn::expand_dims(centered, 2);
    const auto scatter    = nn::sum(outer * nn::expand_dims(weights, 3), {0});
    const bool tied = covariance_ == GaussianMixtureCovariance::Tied;
    const auto covariance = tied ? nn::sum(scatter, {0}, true) / total
                                 : scatter / nn::reshape(counts, {k_, 1, 1});
    const auto eigen      = nn::self_adjoint_eig(covariance);
    const auto clipped    = nn::maximum(eigen.e, nn::scalar_like(floor_, eigen.e));
    out.next.covariances  = nn::matmul(eigen.v * nn::expand_dims(clipped, 1), eigen.v, false, true);
    if (tied)
      out.next.covariances = nn::repeat(out.next.covariances, k_, 0);
    return out;
  }

 private:
  std::int64_t k_;     ///< Fixed component count.
  std::int64_t d_;     ///< Fixed observation width.
  double       floor_; ///< Positive minimum updated covariance eigenvalue.
  GaussianMixtureCovariance covariance_; ///< Covariance structure applied at each M step.
};

/// @brief Multistart controls shared by each model in a Gaussian mixture comparison.
/// @par Example
/// @code
/// const dm::nn::ml::GaussianMixtureSelectionOptions options{.starts=4, .seed=7};
/// std::println("{}", options.starts); // 4
/// @endcode
struct GaussianMixtureSelectionOptions {
  std::size_t starts{8}; ///< Number of attempts per model with more than one component.
  std::size_t max_iterations{250}; ///< Maximum accepted EM updates per start.
  double relative_tolerance{1e-8}; ///< Stop when improvement is below this fraction of one plus the absolute log likelihood.
  double covariance_floor{1e-4}; ///< Minimum covariance eigenvalue in observation units.
  std::uint64_t seed{1}; ///< Local random seed for model initialization.
};

/// @brief Fit and score summary for one covariance structure and component count.
/// @par Example
/// @code
/// const dm::nn::ml::GaussianMixtureCandidate unfinished;
/// std::println("{}", std::isfinite(unfinished.bic)); // false
/// @endcode
struct GaussianMixtureCandidate {
  int components{}; ///< Fitted number of components.
  GaussianMixtureCovariance covariance{}; ///< Within-component covariance constraint.
  std::size_t parameters{}; ///< Number of free weights, means and covariance coefficients.
  double log_likelihood{-std::numeric_limits<double>::infinity()}; ///< Observed-data log likelihood at the best start.
  double bic{std::numeric_limits<double>::infinity()}; ///< Lower-is-better BIC, positive infinity when every start fails.
  bool converged{}; ///< Whether the best start met the tolerance.
};

/// @brief Candidate table and winning Gaussian mixture with matching posterior probabilities.
/// @par Example
/// @code
/// namespace n=dm::nn; namespace m=n::ml;
/// const auto x=n::tensor({-3,-2,-1,1,2,3},{6,1},n::DType::Float64);
/// const m::GaussianMixtureSelection fit=m::select_gaussian_mixture(x,{1});
/// std::println("{}", fit.candidates[fit.best_index].components); // 1
/// @endcode
struct GaussianMixtureSelection {
  std::vector<GaussianMixtureCandidate> candidates; ///< Scores in requested count order, tied, diagonal, then full.
  std::size_t best_index{}; ///< Index of the smallest finite BIC, using the first model for ties.
  GaussianMixtureState model; ///< Winning weights, means and covariance matrices.
  nn::Tensor responsibilities; ///< Model posterior probabilities for the input rows.
};

/// @brief Selects a Gaussian mixture by BIC over component counts and covariance structures.
/// Uses the best finite likelihood from seeded starts for each candidate.
/// Finite iteration-limited fits are scored and identified by their convergence flag.
/// @param x Finite CPU Float64 [N,D] observations.
/// @param component_counts Distinct component counts from one through N.
/// @param options Shared fitting controls, including the covariance eigenvalue floor.
/// @return Candidate table and selected fit minimizing -2 log likelihood plus parameter count times log(N).
/// @throws std::invalid_argument For invalid observations, component counts or options.
/// @throws std::length_error If the covariance or EM workspace dimensions exceed storage capacity.
/// @throws std::runtime_error If no candidate produces a finite score.
/// @par Example
/// @code
/// namespace n=dm::nn; namespace m=n::ml;
/// const auto x=n::tensor({-3,-2,-1,1,2,3},{6,1},n::DType::Float64);
/// const auto fit=m::select_gaussian_mixture(x,{1,2});
/// std::println("{}", fit.candidates.size()); // 6
/// @endcode
/// @par References
/// Scrucca et al. (2016), https://journal.r-project.org/articles/RJ-2016-021/
[[nodiscard]] GaussianMixtureSelection select_gaussian_mixture(const nn::Tensor& x,
    const std::vector<int>& component_counts, const GaussianMixtureSelectionOptions& options = {});

} // namespace datamunge::nn::ml
