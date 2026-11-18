/// @file control_chart.cpp
/// @brief Control chart fitting, limit calculations and rendering.
#include <datamunge/stats/control_chart.hpp>
#include <datamunge/linalg/cholesky.hpp>
#include <datamunge/random/distributions.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <numbers>
#include <span>
#include <stdexcept>

namespace datamunge::stats {
/// @brief Private control-chart validation, calibration and assembly.
namespace {

/// @brief Checks options and resolves the size of the fit prefix.
/// @param size Total input length.
/// @param options Limit multiplier and optional prefix size.
/// @param minimum Minimum usable baseline length.
/// @return Numeric baseline length.
/// @throws std::invalid_argument For invalid options or insufficient data.
std::size_t baseline_length(std::size_t size, ControlChartOptions options, std::size_t minimum = 1) {
    const auto n = options.baseline_size == 0 ? size : options.baseline_size;
    if (n < minimum || n > size || !(options.sigma_multiplier > 0) || !std::isfinite(options.sigma_multiplier))
        throw std::invalid_argument("control chart: invalid baseline size or limit multiplier");
    return n;
}

/// @brief Validates a nonempty, finite series of continuous observations.
/// @param values Measurements in process order.
/// @throws std::invalid_argument For missing or nonfinite observations.
void validate_values(const std::vector<double>& values) {
    if (values.empty() || !std::ranges::all_of(values, [](double x) { return std::isfinite(x); }))
        throw std::invalid_argument("control chart: observations must be nonempty and finite");
}

/// @brief Validates process parameters for monitoring against a fixed baseline.
/// @param target Process mean.
/// @param sigma Individual standard deviation.
/// @param multiplier Limit multiplier or decision interval.
/// @throws std::invalid_argument For nonfinite parameters or nonpositive scale or multiplier.
void validate_known(double target, double sigma, double multiplier) {
    if (!std::isfinite(target) || !std::isfinite(sigma) || sigma <= 0 ||
        !std::isfinite(multiplier) || multiplier <= 0)
        throw std::invalid_argument("control chart: invalid target, scale or limit");
}

/// @brief Appends a checked statistic and limits, marking a strict limit violation.
/// @param chart Destination chart.
/// @param index One-based input position.
/// @param value Plotted statistic.
/// @param center Center line value.
/// @param lower Lower control limit.
/// @param upper Upper control limit.
/// @throws std::overflow_error If a value cannot be represented as a finite double.
void append(ControlChart& chart, std::size_t index, long double value, long double center,
            long double lower, long double upper) {
    /// Whether a statistic or limit can be converted to a finite double.
    const auto representable = [](long double x) {
        constexpr long double max = std::numeric_limits<double>::max();
        return x >= -max && x <= max;
    };
    if (!representable(value) || !representable(center) || !representable(lower) || !representable(upper))
        throw std::overflow_error("control chart: statistic or limit is not finite");
    const double y = static_cast<double>(value);
    const double lo = static_cast<double>(lower);
    const double hi = static_cast<double>(upper);
    chart.points.push_back({index, y, static_cast<double>(center), lo, hi, y < lo || y > hi});
}

/// @brief Mean and standard deviation of the range from standard Normal observations.
struct RangeMoments {
    long double mean; ///< Expected range, d2.
    long double sd;   ///< Standard deviation of the range, d3.
};

/// @brief Range moments for two observations, from a half-normal difference.
constexpr RangeMoments pair_range{1.12837916709551257L, 0.85250246642742171L};

/// @brief Normal range moments indexed by subgroup size minus two.
///
/// These are the d2 and d3 factors behind the NIST Shewhart range limits.
/// @see https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc321.htm
constexpr std::array<RangeMoments, 9> range_table{{
    pair_range, {1.692569L, 0.888368L}, {2.058751L, 0.879808L},
    {2.325929L, 0.864082L}, {2.534413L, 0.848040L}, {2.704357L, 0.833205L},
    {2.847201L, 0.819831L}, {2.970026L, 0.807834L}, {3.077505L, 0.797050L}
}};

/// @brief Selects the assumptions and units for an attribute chart.
enum class Attribute {
    Count, ///< Constant-exposure Poisson counts.
    Rate, ///< Poisson counts divided by exposure.
    Proportion, ///< Binomial proportion.
    Defectives ///< Binomial count at fixed subgroup size.
};

/// @brief Unit conversion and standard error for an attribute observation.
/// @param kind Attribute model and output units.
/// @param denominator Exposure or binomial sample size.
/// @param rate Pooled baseline rate or probability.
/// @return Standard deviation in the plotted units.
long double standard_error(Attribute kind, long double denominator, long double rate) {
    const bool binomial = kind == Attribute::Proportion || kind == Attribute::Defectives;
    const auto variance = binomial ? rate * (1 - rate) : rate;
    return kind == Attribute::Defectives
        ? std::sqrt(variance * denominator)
        : std::sqrt(variance / denominator);
}

/// @brief Fits and evaluates a Poisson, binomial or Laney attribute chart.
/// @tparam Denominator Numeric denominator type.
/// @param counts Observed counts.
/// @param denominators Inspection exposures or sample sizes.
/// @param options Fit prefix and limit multiplier.
/// @param kind Attribute model and units.
/// @param laney Whether to rescale theoretical standard errors with baseline moving ranges.
/// @param chart Initial empty chart carrying plot labels.
/// @return Owned chart filled in input order.
/// @throws std::invalid_argument For invalid denominators, counts or reference series.
/// @throws std::overflow_error For nonrepresentable outputs.
template <class Denominator>
ControlChart attribute_chart(const std::vector<std::size_t>& counts,
                             const std::vector<Denominator>& denominators,
                             ControlChartOptions options, Attribute kind, bool laney, ControlChart chart) {
    const auto n = baseline_length(counts.size(), options, laney ? 2 : 1);
    if (counts.size() != denominators.size())
        throw std::invalid_argument("control chart: counts and denominators differ in length");
    const bool binomial = kind == Attribute::Proportion || kind == Attribute::Defectives;
    long double total_count = 0, total_exposure = 0;
    for (std::size_t i = 0; i < counts.size(); ++i) {
        const long double denominator = denominators[i];
        if (!std::isfinite(denominator) || denominator <= 0 || (binomial && counts[i] > denominator))
            throw std::invalid_argument("control chart: invalid exposure, sample size or defective count");
        if (i < n) {
            total_count += counts[i];
            total_exposure += denominator;
        }
    }
    const long double rate = total_count / total_exposure;
    long double scale = 1;
    if (laney) {
        if (rate == 0 || (binomial && rate == 1))
            throw std::invalid_argument("Laney chart: baseline theoretical variance is zero");
        long double moving_sum = 0, previous = 0;
        for (std::size_t i = 0; i < n; ++i) {
            const long double d = denominators[i];
            const auto z = (counts[i] / d - rate) / standard_error(kind, d, rate);
            if (i > 0) moving_sum += std::abs(z - previous);
            previous = z;
        }
        scale = moving_sum / (n - 1) / pair_range.mean;
    }
    chart.points.reserve(counts.size());
    for (std::size_t i = 0; i < counts.size(); ++i) {
        const long double d = denominators[i];
        const bool count = kind == Attribute::Count || kind == Attribute::Defectives;
        const auto value = count ? static_cast<long double>(counts[i]) : counts[i] / d;
        const auto center = kind == Attribute::Defectives ? d * rate : rate;
        const auto delta = options.sigma_multiplier * scale * standard_error(kind, d, rate);
        const auto cap = kind == Attribute::Proportion ? 1.L : d;
        append(chart, i + 1, value, center, std::max(0.L, center - delta),
               binomial ? std::min(cap, center + delta) : center + delta);
    }
    return chart;
}

/// @brief Fits either member of the individuals and moving-range pair.
/// @param values Ordered observations.
/// @param options Baseline prefix and multiplier.
/// @param ranges Whether to chart moving ranges instead of individual observations.
/// @return Individuals or moving-range chart.
/// @throws std::invalid_argument For invalid observations or baseline.
/// @throws std::overflow_error For nonrepresentable outputs.
ControlChart individuals_or_ranges(const std::vector<double>& values, ControlChartOptions options, bool ranges) {
    validate_values(values);
    const auto n = baseline_length(values.size(), options, 2);
    long double total = values[0], sum_ranges = 0;
    for (std::size_t i = 1; i < n; ++i) {
        total += values[i];
        sum_ranges += std::abs(static_cast<long double>(values[i]) - values[i - 1]);
    }
    const auto mr = sum_ranges / (n - 1);
    const auto scale = mr / pair_range.mean;
    const auto center = ranges ? mr : total / n;
    const auto delta = options.sigma_multiplier * scale * (ranges ? pair_range.sd : 1);
    const auto lower = ranges ? std::max(0.L, center - delta) : center - delta;
    ControlChart result{ranges ? "Moving-range chart" : "Individuals chart",
                        ranges ? "Moving range" : "Measurement", {}};
    for (std::size_t i = ranges ? 1 : 0; i < values.size(); ++i) {
        const auto value = ranges ? std::abs(static_cast<long double>(values[i]) - values[i - 1]) : values[i];
        append(result, i + 1, value, center, lower, center + delta);
    }
    return result;
}

/// @brief Summary of a rational subgroup with the Normal moments of its scale statistic.
struct Subgroup {
    std::size_t size; ///< Number of observations.
    long double mean; ///< Within-subgroup arithmetic mean.
    long double spread; ///< Sample range or sample SD.
    RangeMoments factors; ///< Mean and SD of spread under unit process variance.
};

/// @brief Returns the mean and standard deviation of a sample SD at unit process scale.
/// @param n Sample size, at least two.
/// @return Normal sample SD calibration moments.
RangeMoments sd_moments(std::size_t n) {
    const long double df = n - 1;
    const auto c4 = std::min(1.L, std::sqrt(2 / df) * std::exp(std::lgamma((df + 1) / 2) - std::lgamma(df / 2)));
    return {c4, std::sqrt((1 - c4) * (1 + c4))};
}

/// @brief Calculates sample SD using deviations from a supplied mean.
/// @param values Sample containing at least two observations.
/// @param mean Sample mean.
/// @return Sample SD with divisor n minus one.
long double sample_sd(std::span<const double> values, long double mean) {
    long double squared = 0;
    for (const auto value : values) squared += (value - mean) * (value - mean);
    return std::sqrt(squared / (values.size() - 1));
}

/// @brief Computes subgroup location and spread, with Normal calibration factors.
/// @param values Finite subgroup measurements.
/// @param range Whether to use the range instead of sample SD.
/// @return Summary used for fitting and monitoring.
/// @throws std::invalid_argument For unsupported sizes or nonfinite data.
Subgroup summarize_subgroup(const std::vector<double>& values, bool range) {
    const auto n = values.size();
    if (n < 2 || (range && n > range_table.size() + 1))
        throw std::invalid_argument("control chart: unsupported subgroup size");
    validate_values(values);
    const auto mean = std::accumulate(values.begin(), values.end(), 0.L) / n;
    if (range) {
        const auto [lo, hi] = std::ranges::minmax_element(values);
        return {n, mean, static_cast<long double>(*hi) - *lo, range_table[n - 2]};
    }
    return {n, mean, sample_sd(values, mean), sd_moments(n)};
}

/// @brief Fits and evaluates an X-bar or subgroup dispersion chart.
/// @param groups Ordered rational subgroups.
/// @param options Fit prefix and multiplier.
/// @param range Whether to estimate scale using ranges.
/// @param dispersion Whether to chart spread instead of subgroup means.
/// @param result Empty chart with display labels.
/// @return Chart with subgroup-specific limits.
/// @throws std::invalid_argument For invalid subgroup data or fit options.
/// @throws std::overflow_error For nonrepresentable outputs.
ControlChart subgroup_chart(const std::vector<std::vector<double>>& groups, ControlChartOptions options,
                            bool range, bool dispersion, ControlChart result) {
    const auto n = baseline_length(groups.size(), options);
    std::vector<Subgroup> summaries;
    summaries.reserve(groups.size());
    for (const auto& group : groups) summaries.push_back(summarize_subgroup(group, range));
    long double sum = 0, count = 0, spread = 0, correction = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const auto& group = summaries[i];
        sum += group.mean * group.size;
        count += group.size;
        spread += group.spread;
        correction += group.factors.mean;
    }
    const auto target = sum / count;
    const auto sigma = spread / correction;
    for (std::size_t i = 0; i < summaries.size(); ++i) {
        const auto& group = summaries[i];
        const auto center = dispersion ? sigma * group.factors.mean : target;
        const auto se = dispersion ? sigma * group.factors.sd : sigma / std::sqrt(static_cast<long double>(group.size));
        const auto delta = options.sigma_multiplier * se;
        const auto lower = dispersion ? std::max(0.L, center - delta) : center - delta;
        append(result, i + 1, dispersion ? group.spread : group.mean, center, lower, center + delta);
    }
    return result;
}

/// @brief Converts a Normal multiplier into a one-sided tail probability.
/// @param k Positive finite sigma multiplier.
/// @return Positive Normal tail probability.
/// @throws std::invalid_argument If the tail underflows to zero.
long double tail_probability(double k) {
    const auto tail = std::erfc(static_cast<long double>(k) / std::numbers::sqrt2_v<long double>) / 2;
    if (tail == 0) throw std::invalid_argument("control chart: tail probability underflows");
    return tail;
}
} // namespace

graphics::RPlot ControlChart::plot() const {
    auto result = graphics::RPlot::create().title(title).x_label("Observation or subgroup").y_label(statistic);
    if (points.empty()) return result;
    std::vector<double> x, y, center, lower, upper, flagged_x, flagged_y;
    for (const auto& point : points) {
        x.push_back(static_cast<double>(point.index));
        y.push_back(point.value);
        center.push_back(point.center);
        lower.push_back(point.lower);
        upper.push_back(point.upper);
        if (point.outside) {
            flagged_x.push_back(x.back());
            flagged_y.push_back(y.back());
        }
    }
    if (points.size() == 1) {
        x = {x[0] - 0.35, x[0] + 0.35};
        center.push_back(center[0]);
        lower.push_back(lower[0]);
        upper.push_back(upper[0]);
    }
    result.lines(x, center, "Center", {95, 95, 95})
        .lines(x, lower, "Lower limit", {180, 90, 40})
        .lines(x, upper, "Upper limit", {150, 65, 35});
    if (points.size() == 1) x = {static_cast<double>(points[0].index)};
    if (points.size() > 1) result.lines(x, y, graphics::RGBA{36, 105, 172}, 1);
    result.points(x, y, "Statistic", {36, 105, 172}, 3);
    if (!flagged_x.empty()) result.points(flagged_x, flagged_y, "Outside limits", {180, 44, 45}, 5);
    // Keep the automatic legend above the highest plotted point or limit.
    const auto [min_value, max_value] = std::ranges::minmax_element(y);
    const double bottom = std::min({*min_value, *std::ranges::min_element(lower), *std::ranges::min_element(center)});
    const double top = std::max({*max_value, *std::ranges::max_element(upper), *std::ranges::max_element(center)});
    const long double span = static_cast<long double>(top) - bottom;
    constexpr long double max = std::numeric_limits<double>::max();
    if (span > 0) {
        result.y_limits(static_cast<double>(std::max(-max, bottom - 0.08L * span)),
                        static_cast<double>(std::min(max, top + 0.32L * span)));
    }
    return result;
}

ControlChart c_chart(const std::vector<std::size_t>& counts, ControlChartOptions options) {
    return attribute_chart(counts, std::vector<std::size_t>(counts.size(), 1), options, Attribute::Count, false,
                           {"c-chart", "Defect count", {}});
}

ControlChart u_chart(const std::vector<std::size_t>& counts, const std::vector<double>& exposure,
                     ControlChartOptions options) {
    return attribute_chart(counts, exposure, options, Attribute::Rate, false, {"u-chart", "Defects per unit", {}});
}

ControlChart p_chart(const std::vector<std::size_t>& counts, const std::vector<std::size_t>& sizes,
                     ControlChartOptions options) {
    return attribute_chart(counts, sizes, options, Attribute::Proportion, false, {"p-chart", "Defective proportion", {}});
}

ControlChart np_chart(const std::vector<std::size_t>& counts, std::size_t size, ControlChartOptions options) {
    return attribute_chart(counts, std::vector<std::size_t>(counts.size(), size), options, Attribute::Defectives, false,
                           {"np-chart", "Defective units", {}});
}

ControlChart laney_p_chart(const std::vector<std::size_t>& counts, const std::vector<std::size_t>& sizes,
                           ControlChartOptions options) {
    return attribute_chart(counts, sizes, options, Attribute::Proportion, true, {"Laney p′ chart", "Defective proportion", {}});
}

ControlChart laney_u_chart(const std::vector<std::size_t>& counts, const std::vector<double>& exposure,
                           ControlChartOptions options) {
    return attribute_chart(counts, exposure, options, Attribute::Rate, true, {"Laney u′ chart", "Defects per unit", {}});
}

ControlChart individuals_chart(const std::vector<double>& values, ControlChartOptions options) {
    return individuals_or_ranges(values, options, false);
}

ControlChart moving_range_chart(const std::vector<double>& values, ControlChartOptions options) {
    return individuals_or_ranges(values, options, true);
}

ControlChart xbar_chart(const std::vector<std::vector<double>>& groups, ControlChartOptions options) {
    return subgroup_chart(groups, options, true, false, {"X-bar chart (range estimate)", "Subgroup mean", {}});
}

ControlChart r_chart(const std::vector<std::vector<double>>& groups, ControlChartOptions options) {
    return subgroup_chart(groups, options, true, true, {"R-chart", "Subgroup range", {}});
}

ControlChart xbar_s_chart(const std::vector<std::vector<double>>& groups, ControlChartOptions options) {
    return subgroup_chart(groups, options, false, false, {"X-bar chart (SD estimate)", "Subgroup mean", {}});
}

ControlChart s_chart(const std::vector<std::vector<double>>& groups, ControlChartOptions options) {
    return subgroup_chart(groups, options, false, true, {"S-chart", "Subgroup standard deviation", {}});
}

ControlChart ewma_chart(const std::vector<double>& values, double target, double sigma, double lambda, double k) {
    validate_values(values);
    validate_known(target, sigma, k);
    if (!(lambda > 0 && lambda <= 1)) throw std::invalid_argument("EWMA chart: lambda must be in (0, 1]");
    const long double weight = lambda;
    long double z = target, variance_factor = 0;
    ControlChart result{"EWMA chart", "Exponentially weighted mean", {}};
    for (std::size_t i = 0; i < values.size(); ++i) {
        z = std::lerp(z, static_cast<long double>(values[i]), weight);
        variance_factor = weight * weight + (1 - weight) * (1 - weight) * variance_factor;
        const auto delta = static_cast<long double>(k) * sigma * std::sqrt(variance_factor);
        append(result, i + 1, z, target, target - delta, target + delta);
    }
    return result;
}

ControlChart moving_average_chart(const std::vector<double>& values, std::size_t window,
                                  double target, double sigma, double k) {
    validate_values(values);
    validate_known(target, sigma, k);
    if (window == 0 || window > values.size()) throw std::invalid_argument("moving average chart: invalid window");
    const auto delta = static_cast<long double>(k) * sigma / std::sqrt(static_cast<long double>(window));
    ControlChart result{"Moving average chart", "Trailing mean", {}};
    long double sum = 0;
    for (std::size_t i = 0; i < values.size(); ++i) {
        sum += values[i];
        if (i >= window) sum -= values[i - window];
        if (i >= window - 1) append(result, i + 1, sum / window, target, target - delta, target + delta);
    }
    return result;
}

ControlChart moving_sd_chart(const std::vector<double>& values, std::size_t window, double sigma, double k) {
    validate_values(values);
    validate_known(0, sigma, k);
    if (window < 2 || window > values.size()) throw std::invalid_argument("moving SD chart: invalid window");
    const auto moments = sd_moments(window);
    const auto center = sigma * moments.mean;
    const auto delta = static_cast<long double>(k) * sigma * moments.sd;
    ControlChart result{"Moving SD chart", "Trailing sample standard deviation", {}};
    const std::span<const double> data(values);
    for (std::size_t i = window; i <= data.size(); ++i) {
        const auto sample = data.subspan(i - window, window);
        const auto mean = std::accumulate(sample.begin(), sample.end(), 0.L) / window;
        append(result, i, sample_sd(sample, mean), center, std::max(0.L, center - delta), center + delta);
    }
    return result;
}

std::pair<ControlChart, ControlChart> cusum_chart(const std::vector<double>& values, double target, double sigma,
                                                 double reference, double decision) {
    validate_values(values);
    validate_known(target, sigma, decision);
    if (!std::isfinite(reference) || reference < 0) throw std::invalid_argument("CUSUM chart: invalid reference");
    ControlChart up{"Upward CUSUM", "Standardized cumulative sum", {}};
    ControlChart down{"Downward CUSUM", "Standardized cumulative sum", {}};
    long double positive = 0, negative = 0;
    for (std::size_t i = 0; i < values.size(); ++i) {
        const auto z = (static_cast<long double>(values[i]) - target) / sigma;
        positive = std::max(0.L, positive + z - reference);
        negative = std::max(0.L, negative - z - reference);
        append(up, i + 1, positive, 0, 0, decision);
        append(down, i + 1, negative, 0, 0, decision);
    }
    return {std::move(up), std::move(down)};
}

ControlChart g_chart(const std::vector<std::size_t>& gaps, ControlChartOptions options) {
    const auto n = baseline_length(gaps.size(), options);
    const auto mean = std::accumulate(gaps.begin(), gaps.begin() + n, 0.L) / n;
    const auto tail = tail_probability(options.sigma_multiplier);
    long double center = 0, lower = 0, upper = 0;
    if (mean > 0) {
        const auto log_survival = -std::log1p(1 / mean);
        center = std::max(0.L, std::ceil(std::log(0.5L) / log_survival) - 1);
        lower = std::max(0.L, std::floor(std::log1p(-tail) / log_survival));
        upper = std::max(0.L, std::ceil(std::log(tail) / log_survival) - 1);
    }
    ControlChart result{"g-chart", "Non-events between events", {}};
    for (std::size_t i = 0; i < gaps.size(); ++i) append(result, i + 1, gaps[i], center, lower, upper);
    return result;
}

ControlChart t_chart(const std::vector<double>& intervals, ControlChartOptions options) {
    const auto n = baseline_length(intervals.size(), options);
    if (!std::ranges::all_of(intervals, [](double x) { return std::isfinite(x) && x > 0; }))
        throw std::invalid_argument("t-chart: durations must be positive and finite");
    const auto mean = std::accumulate(intervals.begin(), intervals.begin() + n, 0.L) / n;
    const auto tail = tail_probability(options.sigma_multiplier);
    const auto center = mean * std::log(2.L);
    const auto lower = -mean * std::log1p(-tail);
    const auto upper = -mean * std::log(tail);
    ControlChart result{"t-chart", "Time between events", {}};
    for (std::size_t i = 0; i < intervals.size(); ++i) append(result, i + 1, intervals[i], center, lower, upper);
    return result;
}

ControlChart hotelling_t2_chart(const std::vector<std::vector<double>>& observations,
                                const std::vector<double>& target,
                                const linalg::DenseMatrix<double>& covariance, double alpha) {
    validate_values(target);
    const auto dim = target.size();
    if (observations.empty() || covariance.rows() != dim || covariance.cols() != dim ||
        !(alpha > 0 && alpha < 1 && 1 - alpha < 1))
        throw std::invalid_argument("Hotelling chart: invalid data, covariance shape or alpha");
    linalg::DenseMatrix<long double> normal_covariance(dim, dim);
    for (std::size_t i = 0; i < dim; ++i) {
        for (std::size_t j = 0; j < dim; ++j) {
            if (!std::isfinite(covariance(i, j)) || covariance(i, j) != covariance(j, i))
                throw std::invalid_argument("Hotelling chart: covariance must be finite and symmetric");
            normal_covariance(i, j) = covariance(i, j);
        }
    }
    const auto factor = linalg::cholesky(normal_covariance);
    if (!factor.ok) throw std::invalid_argument("Hotelling chart: covariance must be positive definite");
    const auto upper = random::chi_squared_quantile(1 - alpha, static_cast<double>(dim));
    ControlChart result{"Hotelling T² chart", "Squared Mahalanobis distance", {}};
    std::vector<long double> whitened(dim);
    for (std::size_t i = 0; i < observations.size(); ++i) {
        const auto& row = observations[i];
        if (row.size() != dim) throw std::invalid_argument("Hotelling chart: observation dimension mismatch");
        validate_values(row);
        long double distance = 0;
        for (std::size_t j = 0; j < dim; ++j) {
            // Solve Lz = x - target. The statistic is the squared norm of z.
            long double residual = static_cast<long double>(row[j]) - target[j];
            for (std::size_t k = 0; k < j; ++k) residual -= factor.L(j, k) * whitened[k];
            whitened[j] = residual / factor.L(j, j);
            distance += whitened[j] * whitened[j];
        }
        append(result, i + 1, distance, dim, 0, upper);
    }
    return result;
}

} // namespace datamunge::stats
