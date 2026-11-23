/// @file profile_likelihood.cpp
/// @brief Bracketing and inversion of scalar likelihood-ratio contours.

#include <datamunge/stats/profile_likelihood.hpp>

#include <datamunge/random/distributions.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace datamunge::stats {

/// @brief Scalar searches in the two directions away from a profile maximum.
namespace {

/// @brief Expands towards one domain bound, then bisects a bracket that straddles the contour.
ProfileLikelihoodEndpoint find_endpoint(double estimate, double bound, int direction,
                                        const std::function<long double(double)>& score,
                                        const ProfileLikelihoodOptions& options) {
    using Status = ProfileEndpointStatus;
    if (estimate == bound)
        return {bound, Status::ParameterBound};
    const double limit = std::isfinite(bound) ? bound : std::copysign(std::numeric_limits<double>::max(), bound);
    double inside = estimate;
    long double step = options.initial_step;
    for (std::size_t i = 0; i < options.max_expansions; ++i) {
        const auto proposed = static_cast<long double>(inside) + direction * step;
        double outside = static_cast<double>(direction > 0 ? std::min<long double>(proposed, limit)
                                                          : std::max<long double>(proposed, limit));
        if (outside == inside)
            outside = std::nextafter(inside, limit);
        if (outside == inside)
            break;
        const auto value = score(outside);
        if (value == 0)
            return {outside, Status::Crossing};
        if (value > 0) {
            inside = outside;
            if (inside == bound)
                return {inside, Status::ParameterBound};
            step *= 2;
            continue;
        }
        for (std::size_t j = 0;; ++j) {
            const auto middle = std::midpoint(inside, outside);
            const auto scale = std::max({options.initial_step, std::abs(inside), std::abs(outside)});
            const auto width = std::abs(static_cast<long double>(inside) / scale -
                                        static_cast<long double>(outside) / scale);
            if (width <= options.tolerance || middle == inside || middle == outside)
                return {middle, Status::Crossing};
            if (j == options.max_bisections)
                break;
            const auto mid_value = score(middle);
            if (mid_value == 0)
                return {middle, Status::Crossing};
            if (mid_value > 0)
                inside = middle;
            else
                outside = middle;
        }
        break;
    }
    return {inside, Status::SearchLimit};
}

} // namespace

ProfileLikelihoodInterval profile_likelihood_interval(double estimate, const std::function<double(double)>& profile,
                                                      const ProfileLikelihoodOptions& options) {
    const auto positive = [](double x) { return x > 0 && std::isfinite(x); };
    if (!profile || !std::isfinite(estimate) || !(options.confidence_level > 0 && options.confidence_level < 1) ||
        !(options.lower_bound < options.upper_bound) ||
        estimate < options.lower_bound || estimate > options.upper_bound ||
        !positive(options.initial_step) || !positive(options.tolerance) ||
        options.max_expansions == 0 || options.max_bisections == 0)
        throw std::invalid_argument("profile_likelihood_interval: invalid estimate or search controls");
    const auto maximum = profile(estimate);
    if (!std::isfinite(maximum))
        throw std::domain_error("profile_likelihood_interval: log likelihood at estimate must be finite");
    const auto cutoff = random::gamma_quantile(options.confidence_level, .5, 2);
    if (cutoff == 0)
        return {{estimate, ProfileEndpointStatus::Crossing}, {estimate, ProfileEndpointStatus::Crossing}, cutoff};
    const auto score = [&](double value) {
        const auto likelihood = profile(value);
        if (std::isnan(likelihood) || likelihood == std::numeric_limits<double>::infinity())
            throw std::domain_error("profile_likelihood_interval: invalid profile log likelihood");
        const auto drop = static_cast<long double>(maximum) - likelihood;
        const auto rounding = 32 * std::numeric_limits<double>::epsilon() * std::max(1.0, std::abs(maximum));
        if (drop < -rounding)
            throw std::domain_error("profile_likelihood_interval: profile exceeds likelihood at estimate");
        return cutoff / 2.0L - std::max(0.0L, drop);
    };
    return {find_endpoint(estimate, options.lower_bound, -1, score, options),
            find_endpoint(estimate, options.upper_bound, 1, score, options), cutoff};
}

} // namespace datamunge::stats
