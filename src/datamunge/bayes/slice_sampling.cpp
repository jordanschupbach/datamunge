/// @file slice_sampling.cpp
/// @brief Scalar interval and Gaussian ellipse slice kernels.

#include <datamunge/bayes/slice_sampling.hpp>

#include <datamunge/linalg/cholesky.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <utility>

namespace datamunge::bayes {

/// @brief Numeric checks shared by the scalar and elliptical updates.
namespace {

/// @brief Largest representable state coordinate or scalar interval endpoint.
constexpr double state_limit = std::numeric_limits<double>::max();

/// @brief Checks log callback results while accepting negative infinity as zero density.
double checked_log_value(double value) {
    if (std::isnan(value) || value == std::numeric_limits<double>::infinity())
        throw std::domain_error("slice sampling: log callback returned NaN or positive infinity");
    return value;
}

/// @brief Clips extended-precision scalar interval endpoints to representable states.
double finite_endpoint(long double endpoint) {
    return static_cast<double>(std::clamp(endpoint, -static_cast<long double>(state_limit),
                                          static_cast<long double>(state_limit)));
}

/// @brief Tests membership using a log-density difference, retaining small slices at large offsets.
bool in_slice(double value, double current_log, double log_drop) {
    return std::isfinite(value) && static_cast<long double>(current_log) - value <= log_drop;
}

} // namespace

double univariate_slice_step(const std::function<double(double)>& log_density, double current,
                             random::RandomGenerator& rng, SliceOptions options) {
    if (!log_density || !std::isfinite(current) || !std::isfinite(options.width) ||
        options.width <= 0 || options.max_intervals == 0 || options.max_proposals == 0)
        throw std::invalid_argument("univariate_slice_step: invalid callback, current state or options");
    const auto evaluate = [&](double x) { return checked_log_value(log_density(x)); };
    const double current_log = evaluate(current);
    if (!std::isfinite(current_log))
        throw std::invalid_argument("univariate_slice_step: current state must have positive density");

    const double log_drop = rng.exponential(1.0);
    const long double offset = static_cast<long double>(options.width) * rng.uniform01();
    double lower = finite_endpoint(static_cast<long double>(current) - offset);
    double upper = finite_endpoint(static_cast<long double>(current) + (options.width - offset));
    const auto left_budget = options.max_intervals > 1 ? rng.uniform_u64(0, options.max_intervals - 1) : 0;
    for (std::size_t i = 0; i < left_budget; ++i) {
        const double next = finite_endpoint(static_cast<long double>(lower) - options.width);
        if (next == lower || !in_slice(evaluate(lower), current_log, log_drop))
            break;
        lower = next;
    }
    const std::size_t right_budget = options.max_intervals - 1 - left_budget;
    for (std::size_t i = 0; i < right_budget; ++i) {
        const double next = finite_endpoint(static_cast<long double>(upper) + options.width);
        if (next == upper || !in_slice(evaluate(upper), current_log, log_drop))
            break;
        upper = next;
    }

    for (std::size_t i = 0; i < options.max_proposals && lower < upper; ++i) {
        const double candidate = rng.uniform(lower, upper);
        if (in_slice(evaluate(candidate), current_log, log_drop))
            return candidate;
        if (candidate < current)
            lower = candidate;
        else
            upper = candidate;
    }
    throw std::runtime_error("univariate_slice_step: bracket collapsed or proposal limit exhausted");
}

EllipticalSliceSampler::EllipticalSliceSampler(std::vector<double> prior_mean,
                                               const linalg::DenseMatrix<double>& prior_covariance)
    : prior_mean_(std::move(prior_mean)) {
    const auto n = prior_mean_.size();
    if (n == 0 || prior_covariance.rows() != n || prior_covariance.cols() != n)
        throw std::invalid_argument("EllipticalSliceSampler: covariance shape must match a nonempty mean");
    for (std::size_t i = 0; i < n; ++i) {
        if (!std::isfinite(prior_mean_[i]))
            throw std::invalid_argument("EllipticalSliceSampler: mean must be finite");
        for (std::size_t j = 0; j <= i; ++j) {
            if (!std::isfinite(prior_covariance(i, j)) || prior_covariance(i, j) != prior_covariance(j, i))
                throw std::invalid_argument("EllipticalSliceSampler: covariance must be finite and symmetric");
        }
    }
    auto factor = linalg::cholesky(prior_covariance);
    if (!factor.ok || !std::ranges::all_of(factor.L.data(), [](double x) { return std::isfinite(x); }))
        throw std::invalid_argument("EllipticalSliceSampler: covariance must be positive definite");
    prior_factor_ = std::move(factor.L);
}

std::vector<double> EllipticalSliceSampler::step(
    const std::function<double(const std::vector<double>&)>& log_likelihood,
    const std::vector<double>& current, random::RandomGenerator& rng, std::size_t max_proposals) const {
    const auto n = prior_mean_.size();
    if (!log_likelihood || n == 0 || current.size() != n || max_proposals == 0 ||
        !std::ranges::all_of(current, [](double x) { return std::isfinite(x); }))
        throw std::invalid_argument("EllipticalSliceSampler::step: invalid callback, state or proposal limit");
    const double current_log = checked_log_value(log_likelihood(current));
    if (!std::isfinite(current_log))
        throw std::invalid_argument("EllipticalSliceSampler::step: current state must have positive likelihood");

    std::vector<double> normal(n);
    for (auto& value : normal)
        value = rng.normal();
    std::vector<long double> direction(n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j <= i; ++j)
            direction[i] += static_cast<long double>(prior_factor_(i, j)) * normal[j];

    constexpr double full_turn = 2.0 * std::numbers::pi;
    const double log_drop = rng.exponential(1.0);
    double angle = rng.uniform(0, full_turn);
    double lower = angle - full_turn;
    double upper = angle;
    std::vector<double> candidate(n);
    for (std::size_t i = 0; i < max_proposals; ++i) {
        const double sine = std::sin(angle);
        const double cosine = std::cos(angle);
        bool representable = true;
        for (std::size_t j = 0; j < n; ++j) {
            const auto coordinate = static_cast<long double>(prior_mean_[j]);
            const long double value = coordinate + (static_cast<long double>(current[j]) - coordinate) * cosine
                                      + direction[j] * sine;
            if (!std::isfinite(value) || std::abs(value) > state_limit) {
                representable = false;
                break;
            }
            candidate[j] = static_cast<double>(value);
        }
        if (representable && in_slice(checked_log_value(log_likelihood(candidate)), current_log, log_drop))
            return candidate;
        if (angle < 0)
            lower = angle;
        else
            upper = angle;
        if (i + 1 == max_proposals || !(lower < upper))
            break;
        angle = rng.uniform(lower, upper);
    }
    throw std::runtime_error("EllipticalSliceSampler::step: bracket collapsed or proposal limit exhausted");
}

} // namespace datamunge::bayes
