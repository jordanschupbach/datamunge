/// @file survival.cpp
/// @brief Kaplan–Meier, Prentice–Wilcoxon and Cox calculations on common tied-time risk sets.
#include <datamunge/stats/survival.hpp>

#include <datamunge/stats/detail/test_common.hpp>
#include <datamunge/linalg/cholesky.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <utility>
#include <stdexcept>

namespace datamunge::stats {
/// @brief Risk-set traversal shared by the product-limit and rank-score methods.
namespace {
/// @brief Visits distinct follow-up times in ascending order, keeping source indices.
/// @tparam Visit Callable accepting time, risk, events and begin/end source-index iterators.
/// @param data Valid nonempty follow-up observations.
/// @param visit Function called once for each distinct time.
/// @throws std::invalid_argument If records are empty or contain invalid times.
template <class Visit>
void risk_sets(const std::vector<SurvivalObservation>& data, Visit visit) {
    if (data.empty()) throw std::invalid_argument("survival: no observations");
    for (const auto& row : data)
        if (!std::isfinite(row.time) || row.time < 0)
            throw std::invalid_argument("survival: times must be finite and nonnegative");
    std::vector<std::size_t> order(data.size());
    std::iota(order.begin(), order.end(), 0);
    std::ranges::sort(order, {}, [&](std::size_t i) { return data[i].time; });
    for (std::size_t start = 0; start < order.size();) {
        const double time = data[order[start]].time;
        auto end = start;
        std::size_t events = 0;
        while (end < order.size() && data[order[end]].time == time) events += data[order[end++]].event;
        visit(time, data.size() - start, events, order.begin() + start, order.begin() + end);
        start = end;
    }
}
} // namespace

KaplanMeierEstimate kaplan_meier(const std::vector<SurvivalObservation>& observations) {
    KaplanMeierEstimate result;
    long double survival = 1, greenwood = 0;
    risk_sets(observations, [&](double time, std::size_t risk, std::size_t events, auto first, auto end) {
        survival *= static_cast<long double>(risk - events) / risk;
        if (events < risk) greenwood += static_cast<long double>(events) / risk / (risk - events);
        const auto censored = static_cast<std::size_t>(end - first) - events;
        result.points.push_back({time, risk, events, censored, static_cast<double>(survival),
                                 static_cast<double>(survival * std::sqrt(greenwood))});
        if (!result.median_time && survival <= 0.5L) result.median_time = time;
    });
    return result;
}

graphics::RPlot kaplan_meier_plot(const std::vector<KaplanMeierEstimate>& estimates,
                                  const std::vector<std::string>& labels) {
    if (estimates.empty() || (!labels.empty() && labels.size() != estimates.size()))
        throw std::invalid_argument("survival plot: empty estimates or label count mismatch");
    constexpr std::array<graphics::RGBA, 4> colors{{{36, 105, 172}, {182, 83, 29}, {35, 125, 89}, {122, 73, 154}}};
    auto result = graphics::RPlot::create().title("Kaplan–Meier survival estimates")
        .x_label("Follow-up time").y_label("Survival probability").y_limits(0, 1.17);
    for (std::size_t i = 0; i < estimates.size(); ++i) {
        if (estimates[i].points.empty()) throw std::invalid_argument("survival plot: empty curve");
        std::vector<double> x{0}, y{1}, cx, cy;
        for (const auto& point : estimates[i].points) {
            x.insert(x.end(), 2, point.time);
            y.push_back(y.back());
            y.push_back(point.survival);
            if (point.censored > 0) {
                cx.push_back(point.time);
                cy.push_back(point.survival);
            }
        }
        const auto color = colors[i % colors.size()];
        result.lines(x, y, labels.empty() ? "Group " + std::to_string(i + 1) : labels[i], color);
        if (!cx.empty()) result.points(cx, cy, color, 3);
    }
    return result;
}

std::vector<double> prentice_wilcoxon_scores(const std::vector<SurvivalObservation>& observations) {
    std::vector<double> scores(observations.size());
    long double weight = 1, cumulative = 0;
    risk_sets(observations, [&](double, std::size_t risk, std::size_t events, auto first, auto end) {
        weight *= risk / (static_cast<long double>(risk) + events);
        cumulative += weight * events / risk;
        for (auto it = first; it != end; ++it)
            scores[*it] = static_cast<double>(cumulative - (observations[*it].event ? weight : 0));
    });
    return scores;
}

HypothesisTestResult prentice_wilcoxon_test(const std::vector<SurvivalObservation>& x,
                                           const std::vector<SurvivalObservation>& y,
                                           Alternative alternative) {
    detail::check_alternative(alternative);
    if (x.empty() || y.empty()) throw std::invalid_argument("Prentice-Wilcoxon: both groups are required");
    auto pooled = x;
    pooled.insert(pooled.end(), y.begin(), y.end());
    const auto scores = prentice_wilcoxon_scores(pooled);
    const auto [mean, variance] = detail::test_moments(scores);
    if (!(variance > 0)) throw std::invalid_argument("Prentice-Wilcoxon: no pooled score variation");
    const auto observed = std::accumulate(scores.begin(), scores.begin() + x.size(), 0.L);
    const auto sd = std::sqrt(variance * x.size() * y.size() / pooled.size());
    HypothesisTestResult result;
    result.statistic = static_cast<double>(observed);
    result.parameter1 = static_cast<double>(x.size());
    result.parameter2 = static_cast<double>(y.size());
    result.alternative = alternative;
    result.p_value = detail::normal_test_p(static_cast<double>((observed - x.size() * mean) / sd), alternative);
    result.method = "Prentice-Wilcoxon survival test (normal approximation)";
    return result;
}


namespace {

/// @brief Risk-set sums of weights and their first and second covariate moments.
struct CoxMoments {
    double total{}; ///< Sum of weights.
    std::vector<double> first; ///< Weighted covariate sum.
    linalg::DenseMatrix<double> second; ///< Weighted outer-product sum, lower triangle.

    /// @brief Allocates zero moments for p covariates.
    /// @param p Number of columns.
    explicit CoxMoments(std::size_t p) : first(p), second(p, p) {}

    /// @brief Adds one standardized row at the given weight.
    /// @param x Standardized covariates.
    /// @param weight Exponential risk weight relative to the current reference.
    void add(const std::vector<double>& x, double weight) {
        total += weight;
        for (std::size_t j = 0; j < x.size(); ++j) {
            const double weighted = weight * x[j];
            first[j] += weighted;
            for (std::size_t k = 0; k <= j; ++k) second(j, k) += weighted * x[k];
        }
    }

    /// @brief Rescales all moments when the risk reference increases.
    /// @param multiplier Ratio of the old to the new exponential scale.
    void rescale(double multiplier) {
        total *= multiplier;
        for (std::size_t j = 0; j < first.size(); ++j) {
            first[j] *= multiplier;
            for (std::size_t k = 0; k <= j; ++k) second(j, k) *= multiplier;
        }
    }

    /// @brief Transfers tied deaths into the risk set for the next earlier time.
    /// @param deaths Moments on the same exponential scale.
    void merge(const CoxMoments& deaths) {
        total += deaths.total;
        for (std::size_t j = 0; j < first.size(); ++j) {
            first[j] += deaths.first[j];
            for (std::size_t k = 0; k <= j; ++k) second(j, k) += deaths.second(j, k);
        }
    }
};

/// @brief Partial log likelihood and its derivatives at a trial coefficient vector.
struct CoxEvaluation {
    double likelihood{}; ///< Partial log likelihood.
    std::vector<double> score; ///< First derivative with respect to standardized coefficients.
    linalg::DenseMatrix<double> information; ///< Negative second derivative, lower triangle.
    std::vector<double> log_increments; ///< Optional baseline log hazard increments in ascending time.
};

/// @brief Adds two nonnegative quantities represented on the log scale.
/// @param a Log of the first quantity, possibly negative infinity.
/// @param b Log of the second quantity, possibly negative infinity.
/// @return Log of the sum.
double cox_log_add(double a, double b) {
    const double largest = std::max(a, b);
    if (largest == -std::numeric_limits<double>::infinity()) return largest;
    return largest + std::log1p(std::exp(std::min(a, b) - largest));
}

/// @brief Sweeps risk sets once in descending time, using locally scaled exponential weights.
/// @param observations Follow-up data.
/// @param groups Source indices grouped by ascending observed time.
/// @param x Centered and scaled covariate rows.
/// @param beta Trial coefficients on the standardized scale.
/// @param ties Tied-event approximation.
/// @param baseline Whether to record a baseline hazard increment per event time.
/// @return Partial likelihood, score, information and optional increments.
CoxEvaluation evaluate_cox(const std::vector<SurvivalObservation>& observations,
                           const std::vector<std::vector<std::size_t>>& groups,
                           const std::vector<std::vector<double>>& x, const std::vector<double>& beta,
                           CoxTies ties, bool baseline = false) {
    const auto p = beta.size();
    CoxEvaluation result{0, std::vector<double>(p), linalg::DenseMatrix<double>(p, p), {}};
    CoxMoments risk(p), deaths(p);
    std::vector<double> eta(x.size()), means(p);
    for (std::size_t i = 0; i < x.size(); ++i)
        eta[i] = std::inner_product(x[i].begin(), x[i].end(), beta.begin(), 0.0);
    if (!std::ranges::all_of(eta, [](double value) { return std::isfinite(value); })) {
        result.likelihood = -std::numeric_limits<double>::infinity();
        return result;
    }
    double reference = -std::numeric_limits<double>::infinity();
    for (auto group = groups.rbegin(); group != groups.rend(); ++group) {
        double next_reference = reference;
        for (auto i : *group) next_reference = std::max(next_reference, eta[i]);
        if (next_reference != reference) risk.rescale(std::exp(reference - next_reference));
        reference = next_reference;

        deaths.rescale(0);
        std::size_t event_count = 0;
        for (auto i : *group) {
            const double weight = std::exp(eta[i] - reference);
            if (observations[i].event) {
                deaths.add(x[i], weight);
                result.likelihood += eta[i] - reference;
                for (std::size_t j = 0; j < p; ++j) result.score[j] += x[i][j];
                ++event_count;
            } else {
                risk.add(x[i], weight);
            }
        }
        double increment = -std::numeric_limits<double>::infinity();
        const auto terms = ties == CoxTies::breslow && event_count ? std::size_t{1} : event_count;
        const double multiplicity = ties == CoxTies::breslow ? static_cast<double>(event_count) : 1.0;
        for (std::size_t k = 0; k < terms; ++k) {
            const double fraction = static_cast<double>(event_count - k) / static_cast<double>(event_count);
            const double denominator = risk.total + fraction * deaths.total;
            const double log_denominator = std::log(denominator);
            result.likelihood -= multiplicity * log_denominator;
            if (baseline) increment = cox_log_add(increment, std::log(multiplicity) - log_denominator - reference);
            for (std::size_t j = 0; j < p; ++j) {
                means[j] = (risk.first[j] + fraction * deaths.first[j]) / denominator;
                result.score[j] -= multiplicity * means[j];
                for (std::size_t q = 0; q <= j; ++q) {
                    const double second = (risk.second(j, q) + fraction * deaths.second(j, q)) / denominator;
                    result.information(j, q) += multiplicity * (second - means[j] * means[q]);
                }
            }
        }
        if (baseline && event_count) result.log_increments.push_back(increment);
        risk.merge(deaths);
    }
    std::ranges::reverse(result.log_increments);
    return result;
}

} // namespace

CoxPHFit cox_ph(const std::vector<SurvivalObservation>& observations,
                const std::vector<std::vector<double>>& covariates, CoxTies ties) {
    if (covariates.size() != observations.size() || covariates.empty() || covariates.front().empty()
        || (ties != CoxTies::efron && ties != CoxTies::breslow))
        throw std::invalid_argument("cox_ph: supply one covariate row per observation and a valid tie method");
    const auto p = covariates.front().size();
    CoxPHFit fit;
    fit.ties = ties;
    std::vector<std::vector<std::size_t>> groups;
    risk_sets(observations, [&](double time, std::size_t, std::size_t events, auto first, auto last) {
        groups.emplace_back(first, last);
        if (events) fit.event_times.push_back(time);
    });
    if (fit.event_times.empty()) throw std::invalid_argument("cox_ph: at least one event is required");

    std::vector<long double> means(p), scales(p);
    std::vector<double> minima = covariates.front(), maxima = covariates.front();
    for (const auto& row : covariates) {
        if (row.size() != p || !std::ranges::all_of(row, [](double value) { return std::isfinite(value); }))
            throw std::invalid_argument("cox_ph: covariate rows must have matching widths and finite values");
        for (std::size_t j = 0; j < p; ++j) {
            means[j] += static_cast<long double>(row[j]) / covariates.size();
            minima[j] = std::min(minima[j], row[j]);
            maxima[j] = std::max(maxima[j], row[j]);
        }
    }
    fit.reference.assign(means.begin(), means.end());
    auto x = covariates;
    for (const auto& row : covariates)
        for (std::size_t j = 0; j < p; ++j)
            scales[j] = std::max(scales[j], std::abs(static_cast<long double>(row[j]) - fit.reference[j]));
    for (std::size_t j = 0; j < p; ++j) {
        if (minima[j] == maxima[j] || !std::isfinite(scales[j]))
            throw std::runtime_error("cox_ph: a covariate is constant or cannot be scaled");
        for (auto& row : x) row[j] = static_cast<double>((static_cast<long double>(row[j]) - fit.reference[j]) / scales[j]);
    }

    std::vector<double> beta(p);
    auto current = evaluate_cox(observations, groups, x, beta, ties);
    fit.null_log_likelihood = current.likelihood;
    std::vector<double> initial_pivots(p);
    for (;;) {
        const auto factor = linalg::cholesky(current.information);
        if (!factor.ok) throw std::runtime_error("cox_ph: singular information matrix");
        for (std::size_t j = 0; j < p; ++j) {
            const double pivot = factor.L(j, j);
            if (fit.iterations == 0) initial_pivots[j] = std::sqrt(current.information(j, j));
            if (!std::isfinite(pivot) || pivot <= 1e-7 * initial_pivots[j])
                throw std::runtime_error("cox_ph: singular information or monotone likelihood");
        }
        const auto step = factor.solve(current.score);
        if (!std::ranges::all_of(step, [](double value) { return std::isfinite(value); }))
            throw std::runtime_error("cox_ph: Newton step is not finite");
        const double max_step = std::transform_reduce(step.begin(), step.end(), 0.0,
            [](double a, double b) { return std::max(a, b); }, [](double value) { return std::abs(value); });
        if (max_step <= 1e-8) {
            fit.coefficients.resize(p);
            fit.standard_errors.resize(p);
            std::vector<double> unit(p);
            for (std::size_t j = 0; j < p; ++j) {
                unit[j] = 1;
                fit.coefficients[j] = static_cast<double>(beta[j] / scales[j]);
                fit.standard_errors[j] = static_cast<double>(std::sqrt(factor.solve(unit)[j]) / scales[j]);
                unit[j] = 0;
                if (!std::isfinite(fit.coefficients[j]) || !std::isfinite(fit.standard_errors[j]))
                    throw std::runtime_error("cox_ph: estimates exceed the original scale's numeric range");
            }
            fit.log_likelihood = current.likelihood;
            fit.log_baseline_cumulative_hazard = evaluate_cox(observations, groups, x, beta, ties, true).log_increments;
            auto& hazard = fit.log_baseline_cumulative_hazard;
            for (std::size_t i = 1; i < hazard.size(); ++i) hazard[i] = cox_log_add(hazard[i - 1], hazard[i]);
            return fit;
        }
        if (fit.iterations == 50) throw std::runtime_error("cox_ph: Newton iteration did not converge");
        bool accepted = false;
        auto trial = beta;
        for (double fraction = 1; fraction >= 1.0 / 1048576; fraction *= 0.5) {
            for (std::size_t j = 0; j < p; ++j) trial[j] = beta[j] + fraction * step[j];
            if (!std::ranges::all_of(trial, [](double value) { return std::isfinite(value); })) continue;
            auto proposed = evaluate_cox(observations, groups, x, trial, ties);
            if (std::isfinite(proposed.likelihood) && proposed.likelihood >= current.likelihood -
                8 * std::numeric_limits<double>::epsilon() * (1 + std::abs(current.likelihood))) {
                beta = trial;
                current = std::move(proposed);
                accepted = true;
                break;
            }
        }
        if (!accepted) throw std::runtime_error("cox_ph: Newton step failed to improve the likelihood");
        ++fit.iterations;
    }
}

double CoxPHFit::predict_survival(double time, const std::vector<double>& covariates) const {
    if (!std::isfinite(time) || time < 0 || covariates.size() != coefficients.size()
        || !std::ranges::all_of(covariates, [](double value) { return std::isfinite(value); }))
        throw std::invalid_argument("cox_ph: supply a finite nonnegative time and one finite value per covariate");
    const auto event = std::upper_bound(event_times.begin(), event_times.end(), time);
    if (event == event_times.begin()) return 1;
    long double log_hazard = log_baseline_cumulative_hazard[static_cast<std::size_t>(event - event_times.begin() - 1)];
    for (std::size_t j = 0; j < coefficients.size(); ++j)
        log_hazard += static_cast<long double>(coefficients[j]) *
                      (static_cast<long double>(covariates[j]) - reference[j]);
    return static_cast<double>(std::exp(-std::exp(log_hazard)));
}

} // namespace datamunge::stats
