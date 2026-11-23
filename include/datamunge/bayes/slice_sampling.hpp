#pragma once

/// @file slice_sampling.hpp
/// @brief Gradient-free slice updates for scalar targets and Gaussian latent blocks.

#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/random/random.hpp>

#include <cstddef>
#include <functional>
#include <vector>

namespace datamunge::bayes {

/// @brief Search limits for a scalar stepping-out slice update.
/// @par Example
/// @code
/// const dm::bayes::SliceOptions settings{.width = 0.5};
/// std::println("{}", settings.width); // 0.5
/// @endcode
/// @see univariate_slice_step()
struct SliceOptions {
    double width{1.0};                    ///< Positive finite initial interval width.
    std::size_t max_intervals{100};        ///< Total interval-width budget, including the initial interval.
    std::size_t max_proposals{1000};       ///< Maximum number of proposals during shrinkage.
};

/// @brief Updates one scalar state by stepping out and shrinking a slice interval.
///
/// Randomly positions an interval of the given width and randomly allocates its
/// finite expansion budget between the two sides. Rejected proposals shrink
/// the interval toward the current state. The callback is evaluated at the
/// current state, at the tested interval ends, and at the proposals.
/// @param log_density Log density up to an additive constant. Negative infinity denotes zero density.
/// @param current Finite state with finite log density.
/// @param rng Caller-owned generator advanced by the update.
/// @param options Initial width and positive interval and proposal budgets.
/// @return The accepted scalar state. Repeated calls form a Markov chain.
/// @throws std::invalid_argument Empty callback, invalid options or invalid current state.
/// @throws std::domain_error Callback returned NaN or positive infinity.
/// @throws std::runtime_error The bracket has no representable width, or the proposal budget is exhausted.
/// @par Example
/// @code
/// dm::random::SplitMix64 rng(42);
/// const auto draw = dm::bayes::univariate_slice_step([](double x) { return -0.5 * x * x; }, 0, rng);
/// std::println("{}", std::isfinite(draw)); // true
/// @endcode
/// @par References
/// - Neal (2003), Figures 3 and 5, https://doi.org/10.1214/aos/1056562461
/// @see SliceOptions, EllipticalSliceSampler
double univariate_slice_step(const std::function<double(double)>& log_density, double current,
                             random::RandomGenerator& rng, SliceOptions options = {});

/// @brief Caches a Gaussian prior factor for repeated elliptical slice updates.
///
/// Proposals rotate the centered current state toward a fresh draw from the
/// centered prior. The callback evaluates only the log likelihood. The Gaussian
/// prior is preserved by the rotation.
/// @par Example
/// @code
/// const dm::bayes::EllipticalSliceSampler sampler({0}, dm::linalg::DenseMatrix<double>(1, 1, 1.0));
/// dm::random::SplitMix64 rng(42);
/// std::println("{}", sampler.step([](const auto&) { return 0.0; }, {0}, rng).size()); // 1
/// @endcode
/// @par References
/// - Murray, Adams and MacKay (2010), Figure 2, https://proceedings.mlr.press/v9/murray10a.html
/// @see univariate_slice_step()
class EllipticalSliceSampler {
public:
    /// @brief Validates and factors a Gaussian prior covariance once.
    /// @param prior_mean Nonempty finite prior location.
    /// @param prior_covariance Finite symmetric positive definite matrix matching the mean dimension.
    /// @throws std::invalid_argument Invalid mean, covariance shape, symmetry or definiteness.
    /// @par Example
    /// @code
    /// const dm::bayes::EllipticalSliceSampler sampler({2}, dm::linalg::DenseMatrix<double>(1, 1, 0.25));
    /// @endcode
    /// @see step()
    EllipticalSliceSampler(std::vector<double> prior_mean, const linalg::DenseMatrix<double>& prior_covariance);

    /// @brief Updates a latent block by shrinking a randomly positioned angular interval.
    /// @param log_likelihood Log likelihood of the uncentered block. Negative infinity excludes a proposal.
    /// @param current Finite state of the prior dimension with finite log likelihood.
    /// @param rng Caller-owned generator advanced by the update.
    /// @param max_proposals Positive limit on the number of candidate angles evaluated.
    /// @return The accepted latent block in input coordinate order.
    /// @throws std::invalid_argument Empty callback, invalid current state or zero proposal limit.
    /// @throws std::domain_error Callback returned NaN or positive infinity.
    /// @throws std::runtime_error The angular bracket collapses or the proposal budget is exhausted.
    /// @par Example
    /// @code
    /// const dm::bayes::EllipticalSliceSampler sampler({0}, dm::linalg::DenseMatrix<double>(1, 1, 1.0));
    /// dm::random::SplitMix64 rng(42);
    /// const auto draw = sampler.step([](const auto& x) { return -x.front() * x.front(); }, {0}, rng);
    /// std::println("{}", std::isfinite(draw.front())); // true
    /// @endcode
    /// @see EllipticalSliceSampler()
    std::vector<double> step(const std::function<double(const std::vector<double>&)>& log_likelihood,
                             const std::vector<double>& current, random::RandomGenerator& rng,
                             std::size_t max_proposals = 1000) const;

private:
    std::vector<double> prior_mean_;         ///< Owned Gaussian center.
    linalg::DenseMatrix<double> prior_factor_; ///< Cached lower triangular Cholesky factor.
};

} // namespace datamunge::bayes
