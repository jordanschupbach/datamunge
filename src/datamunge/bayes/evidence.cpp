/// @file evidence.cpp
/// @brief Optimal bridge iteration and variance-corrected power-posterior quadrature.

#include <datamunge/bayes/evidence.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace datamunge::bayes {

/// @brief Log-domain weights and range-checked quadrature arithmetic.
namespace {

/// @brief Log weight of a draw outside the shared support.
constexpr long double log_zero = -std::numeric_limits<long double>::infinity();

/// @brief Narrows a reported diagnostic to finite double range.
double evidence_value(long double value) {
    if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<double>::max())
        throw std::overflow_error("marginal likelihood: result exceeds double range");
    return static_cast<double>(value);
}

/// @brief Log of the sum of two weights when the first log weight is finite.
long double log_sum(long double finite_log_weight, long double other_log_weight) {
    return std::max(finite_log_weight, other_log_weight)
         + std::log1p(std::exp(-std::abs(finite_log_weight - other_log_weight)));
}

/// @brief Accumulates a first and second weight moment under a running log scale.
struct BridgeWeights {
    long double maximum{log_zero}; ///< Largest nonzero log weight seen so far.
    long double sum{};             ///< Sum of weights after shifting by the maximum.
    long double squares{};         ///< Sum of squared shifted weights.

    /// @brief Incorporates a finite log contribution or a zero contribution.
    void add(long double log_weight) {
        if (log_weight == log_zero)
            return;
        if (log_weight > maximum) {
            const long double scale = std::exp(maximum - log_weight);
            sum *= scale;
            squares *= scale * scale;
            maximum = log_weight;
        }
        const long double weight = std::exp(log_weight - maximum);
        sum += weight;
        squares += weight * weight;
    }

    /// @brief Log mean on the full sample, including zero contributions.
    long double log_mean(std::size_t count) const {
        return maximum + std::log(sum / static_cast<long double>(count));
    }

    /// @brief Kish effective count on the retained weight scale.
    long double effective_count() const {
        return sum * sum / squares;
    }
};

/// @brief Validates and collects log joint-to-proposal ratios for one sample.
std::vector<long double> log_ratios(const std::vector<BridgeLogDensities>& sample, bool from_posterior) {
    std::vector<long double> ratios;
    ratios.reserve(sample.size());
    bool overlaps = false;
    for (const auto& draw : sample) {
        const double own = from_posterior ? draw.log_joint : draw.log_proposal;
        const double cross = from_posterior ? draw.log_proposal : draw.log_joint;
        if (!std::isfinite(own) || std::isnan(cross) || cross == std::numeric_limits<double>::infinity())
            throw std::invalid_argument("bridge_sampling: invalid log density on a draw");
        const long double ratio = static_cast<long double>(draw.log_joint) - draw.log_proposal;
        if (std::isfinite(cross) && !std::isfinite(ratio))
            throw std::overflow_error("bridge_sampling: log ratio exceeds working range");
        overlaps |= std::isfinite(cross);
        ratios.push_back(ratio);
    }
    if (!overlaps)
        throw std::domain_error("bridge_sampling: each sample needs observed support overlap");
    return ratios;
}

} // namespace

BridgeSamplingResult bridge_sampling(const std::vector<BridgeLogDensities>& posterior,
                                     const std::vector<BridgeLogDensities>& proposal,
                                     double tolerance, std::size_t max_iterations) {
    if (posterior.empty() || proposal.empty() || !std::isfinite(tolerance) || tolerance <= 0 || max_iterations == 0)
        throw std::invalid_argument("bridge_sampling: samples and iteration controls must be valid");
    const auto target_ratios = log_ratios(posterior, true);
    const auto proposal_ratios = log_ratios(proposal, false);
    const long double total = static_cast<long double>(posterior.size()) + proposal.size();
    const long double log_target_fraction = std::log(posterior.size() / total);
    const long double log_proposal_fraction = std::log(proposal.size() / total);

    std::vector<long double> finite_ratios;
    for (const auto& sample : {&target_ratios, &proposal_ratios})
        for (const auto ratio : *sample)
            if (std::isfinite(ratio))
                finite_ratios.push_back(ratio);
    auto middle = finite_ratios.begin() + static_cast<std::ptrdiff_t>(finite_ratios.size() / 2);
    std::nth_element(finite_ratios.begin(), middle, finite_ratios.end());
    long double estimate = *middle;
    const auto weights = [&](long double at) {
        BridgeWeights numerator, denominator;
        for (const auto ratio : proposal_ratios)
            numerator.add(-log_sum(log_target_fraction, log_proposal_fraction + (at - ratio)));
        for (const auto ratio : target_ratios)
            denominator.add(-log_sum(log_proposal_fraction, log_target_fraction + (ratio - at)));
        return std::pair{numerator, denominator};
    };

    long double change = 0;
    std::size_t iterations = 0;
    do {
        const auto [numerator, denominator] = weights(estimate);
        change = numerator.log_mean(proposal.size()) - denominator.log_mean(posterior.size());
        estimate += change;
        if (!std::isfinite(estimate))
            throw std::overflow_error("bridge_sampling: iterative estimate exceeds working range");
        ++iterations;
    } while (iterations < max_iterations && std::abs(change) > tolerance);

    const auto [numerator, denominator] = weights(estimate);
    const long double overlap = std::midpoint(std::exp(numerator.log_mean(proposal.size())),
                                              std::exp(denominator.log_mean(posterior.size())));
    return {evidence_value(estimate), evidence_value(overlap), evidence_value(denominator.effective_count()),
            evidence_value(numerator.effective_count()), evidence_value(std::abs(change)),
            iterations, std::abs(change) <= tolerance};
}

ThermodynamicIntegrationResult thermodynamic_integration(const std::vector<PowerPosteriorRung>& ladder) {
    if (ladder.size() < 2 || ladder.front().inverse_temperature != 0 || ladder.back().inverse_temperature != 1)
        throw std::invalid_argument("thermodynamic_integration: ladder must span zero to one");
    long double trapezoidal = 0;
    long double correction = 0;
    for (std::size_t i = 0; i < ladder.size(); ++i) {
        const auto& rung = ladder[i];
        if (!std::isfinite(rung.mean_log_likelihood) || !std::isfinite(rung.variance_log_likelihood)
            || rung.variance_log_likelihood < 0)
            throw std::invalid_argument("thermodynamic_integration: invalid log-likelihood moments");
        if (i == 0)
            continue;
        const auto& previous = ladder[i - 1];
        const long double step = static_cast<long double>(rung.inverse_temperature) - previous.inverse_temperature;
        if (!(step > 0))
            throw std::invalid_argument("thermodynamic_integration: temperatures must strictly increase");
        trapezoidal += step * std::midpoint(static_cast<long double>(previous.mean_log_likelihood),
                                            static_cast<long double>(rung.mean_log_likelihood));
        correction += (step * step / 12.0L)
                    * (static_cast<long double>(rung.variance_log_likelihood) - previous.variance_log_likelihood);
    }
    return {evidence_value(trapezoidal - correction), evidence_value(trapezoidal),
            evidence_value(std::abs(correction))};
}

ThermodynamicIntegrationResult thermodynamic_integration(
    const std::vector<double>& inverse_temperatures, const std::vector<std::vector<double>>& log_likelihood_draws) {
    if (inverse_temperatures.size() != log_likelihood_draws.size())
        throw std::invalid_argument("thermodynamic_integration: provide draws for every temperature");
    std::vector<PowerPosteriorRung> ladder;
    ladder.reserve(inverse_temperatures.size());
    for (std::size_t i = 0; i < inverse_temperatures.size(); ++i) {
        const auto& draws = log_likelihood_draws[i];
        if (draws.size() < 2 || !std::ranges::all_of(draws, [](double x) { return std::isfinite(x); }))
            throw std::invalid_argument("thermodynamic_integration: each rung needs at least two finite draws");
        const long double anchor = draws.front();
        long double mean = 0;
        long double sum_squared_deviations = 0;
        long double count = 0;
        for (const double draw : draws) {
            const long double shifted = static_cast<long double>(draw) - anchor;
            const long double delta = shifted - mean;
            mean += delta / ++count;
            sum_squared_deviations += delta * (shifted - mean);
        }
        ladder.push_back({inverse_temperatures[i], evidence_value(anchor + mean),
                          evidence_value(sum_squared_deviations / (count - 1))});
    }
    return thermodynamic_integration(ladder);
}

} // namespace datamunge::bayes
