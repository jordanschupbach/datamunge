/// @file survival.cpp
/// @brief Kaplan–Meier and Prentice–Wilcoxon calculations on common tied-time risk sets.
#include <datamunge/stats/survival.hpp>

#include <datamunge/stats/detail/test_common.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
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

} // namespace datamunge::stats
