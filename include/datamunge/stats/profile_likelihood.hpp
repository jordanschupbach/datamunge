#pragma once

/// @file profile_likelihood.hpp
/// @brief Confidence intervals by inversion of a scalar profile-likelihood ratio.

#include <cstddef>
#include <functional>
#include <limits>

namespace datamunge::stats {

/// @brief Why a profile-likelihood endpoint search terminated.
/// @par Example
/// @code
/// const auto status = dm::stats::ProfileEndpointStatus::Crossing;
/// std::println("{}", status == dm::stats::ProfileEndpointStatus::Crossing); // true
/// @endcode
/// @see profile_likelihood_interval()
enum class ProfileEndpointStatus {
    Crossing,       ///< The likelihood-ratio contour was located within tolerance.
    ParameterBound, ///< The finite parameter bound remained inside the confidence set.
    SearchLimit,    ///< A crossing could not be resolved within the numerical or evaluation limits.
};

/// @brief Search controls for a scalar profile-likelihood interval.
/// @par Example
/// @code
/// const dm::stats::ProfileLikelihoodOptions options{.initial_step = .25};
/// std::println("{:.2f}", options.initial_step); // 0.25
/// @endcode
/// @see profile_likelihood_interval()
struct ProfileLikelihoodOptions {
    double confidence_level{.95}; ///< Coverage level for the asymptotic chi-square cutoff.
    double lower_bound{-std::numeric_limits<double>::infinity()}; ///< Inclusive parameter lower bound.
    double upper_bound{std::numeric_limits<double>::infinity()};  ///< Inclusive parameter upper bound.
    double initial_step{1};       ///< Positive first outward step in parameter units.
    double tolerance{1e-8};      ///< Relative bracket width, scaled by the initial step and endpoint magnitudes.
    std::size_t max_expansions{64};  ///< Maximum number of outward evaluations per side.
    std::size_t max_bisections{128}; ///< Maximum number of bracket refinements per side.
};

/// @brief A profile-likelihood interval endpoint or last accepted point if the search limit is reached.
/// @par Example
/// @code
/// const dm::stats::ProfileLikelihoodEndpoint endpoint{2, dm::stats::ProfileEndpointStatus::ParameterBound};
/// std::println("{:.0f}", endpoint.value); // 2
/// @endcode
/// @see ProfileLikelihoodInterval
struct ProfileLikelihoodEndpoint {
    double value;                 ///< Endpoint in the original parameter units.
    ProfileEndpointStatus status; ///< Evidence supporting the returned value.
};

/// @brief A two-sided interval and the likelihood-ratio cutoff used to calculate it.
/// @par Example
/// @code
/// const auto interval = dm::stats::profile_likelihood_interval(0, [](double x) { return -x*x/2; });
/// std::println("{:.2f}", interval.upper.value); // 1.96
/// @endcode
/// @see profile_likelihood_interval()
struct ProfileLikelihoodInterval {
    ProfileLikelihoodEndpoint lower; ///< Lower endpoint, with termination status.
    ProfileLikelihoodEndpoint upper; ///< Upper endpoint, with termination status.
    double likelihood_ratio_cutoff;  ///< Quantile of chi-square with one degree of freedom.
};

/// @brief Inverts the likelihood-ratio statistic along a scalar profile.
///
/// The callback fits or eliminates nuisance parameters at each supplied
/// value of the parameter of interest and returns the maximized log
/// likelihood. Starting at the maximum, the implementation takes growing
/// steps outwards and bisects the first bracket crossing the chi-square
/// threshold on each side. It targets the component around the supplied
/// maximum. Choose the starting step and bounds for the scale and shape of
/// the profile. The search assumes continuous, outward-decreasing likelihood
/// up to each crossing and uses the regular asymptotic chi-square calibration.
/// @param estimate Finite maximizer of the profile within the supplied bounds.
/// @param profile Function of the fixed parameter. Negative infinity denotes zero likelihood.
/// @param options Confidence level, domain, and search controls.
/// @return Endpoint values and statuses, with the likelihood-ratio cutoff.
/// @throws std::invalid_argument Invalid controls, estimate, or empty callback.
/// @throws std::domain_error Nonfinite log likelihood at the estimate, NaN or positive infinity
/// at a candidate, or a visited candidate with larger likelihood than the supplied maximum.
/// @par Example
/// @code
/// const auto interval = dm::stats::profile_likelihood_interval(2, [](double x) {
///     const double offset = x - 2;
///     return -.5 * offset * offset;
/// });
/// std::println("{:.2f}", interval.lower.value); // 0.04
/// @endcode
/// @par References
/// - Venzon and Moolgavkar (1988), https://doi.org/10.2307/2347496
[[nodiscard]] ProfileLikelihoodInterval profile_likelihood_interval(
    double estimate, const std::function<double(double)>& profile, const ProfileLikelihoodOptions& options = {});

} // namespace datamunge::stats
