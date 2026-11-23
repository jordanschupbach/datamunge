/// @file segmented_regression.cpp
/// @brief Exact coordinate searches of the continuous hinge least-squares criterion.

#include <datamunge/stats/segmented_regression.hpp>

#include <datamunge/linalg/svd.hpp>
#include <datamunge/random/distributions.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <utility>

namespace datamunge::stats {

/// @brief Working-scale segmented designs and breakpoint searches.
namespace {

/// @brief Working-precision row vector.
using Vector = std::vector<long double>;

/// @brief Working-precision design matrix.
using Matrix = linalg::DenseMatrix<long double>;

/// @brief Data sorted by scaled predictor, with factors for converting back to original units.
struct Data {
    Vector x;                        ///< Normalized predictors in sorted order.
    Vector y;                        ///< Normalized responses in sorted order.
    Vector distinct;                 ///< Distinct normalized predictors in increasing order.
    std::vector<std::size_t> order;  ///< Original row index for each sorted row.
    long double origin;              ///< Smallest predictor in original units.
    long double span;                ///< Positive predictor range in original units.
    long double response_scale;      ///< Positive scale for linear coefficients and fit errors.
};

/// @brief Conditional coefficient fit at specified breakpoints.
struct LinearFit {
    Vector beta;        ///< Constant, initial slope, and hinge coefficients in normalized units.
    Vector residuals;   ///< Residual at each sorted observation in normalized units.
    long double ss;     ///< Normalized residual sum of squares.
};

/// @brief Segmentation and its conditional fit.
struct SearchResult {
    Vector knots;      ///< Fitted breakpoints in sorted, normalized units.
    LinearFit linear;  ///< Linear fit at these breakpoints.
    bool converged;    ///< Coordinate sweep met the numerical tolerance.
};

/// @brief Index and value of the breakpoint held fixed during profiling.
using FixedKnot = std::pair<std::size_t, long double>;

/// @brief Converts a finite working-precision result to the public scalar type.
double output(long double x) {
    if (!std::isfinite(x) || std::abs(x) > std::numeric_limits<double>::max())
        throw std::overflow_error("segmented_regression: result exceeds finite double range");
    return static_cast<double>(x);
}

/// @brief Forms the conditional hinge design, omitting one moving knot if needed.
Matrix design(const Data& data, const Vector& knots, std::size_t omit) {
    Matrix result(data.x.size(), 2 + knots.size() - (omit < knots.size()), 1);
    for (std::size_t i = 0; i < data.x.size(); ++i) {
        result(i, 1) = data.x[i];
        std::size_t col = 2;
        for (std::size_t j = 0; j < knots.size(); ++j)
            if (j != omit)
                result(i, col++) = std::max(0.0L, data.x[i] - knots[j]);
    }
    return result;
}

/// @brief Fits or projects a conditional hinge design without forming normal equations.
linalg::SVDDecomposition<long double> decompose(const Matrix& matrix) {
    auto factor = linalg::svd(matrix);
    if (!factor.converged || factor.rank() != matrix.cols())
        throw std::runtime_error("segmented_regression: conditional hinge design lost numerical rank");
    return factor;
}

/// @brief Projects the response onto the design and retains the residual for additional tests.
LinearFit fit_linear(const Data& data, const Matrix& matrix,
                     const linalg::SVDDecomposition<long double>& factor) {
    auto beta = factor.solve(data.y);
    Vector residuals(data.x.size());
    long double ss = 0;
    for (std::size_t i = 0; i < residuals.size(); ++i) {
        auto r = data.y[i];
        for (std::size_t j = 0; j < beta.size(); ++j)
            r = std::fma(-matrix(i, j), beta[j], r);
        residuals[i] = r;
        ss += r * r;
    }
    return {std::move(beta), std::move(residuals), ss};
}

/// @brief Feasible range for one knot with its immediate neighbors held fixed.
std::pair<long double, long double> coordinate_bounds(const Data& data, const Vector& knots,
                                                       std::size_t index, std::size_t minimum) {
    auto first = index == 0 ? data.distinct.begin()
        : std::ranges::lower_bound(data.distinct, knots[index - 1]);
    auto last = index + 1 == knots.size() ? data.distinct.end()
        : std::ranges::upper_bound(data.distinct, knots[index + 1]);
    if (last - first < static_cast<std::ptrdiff_t>(minimum))
        return {1, 0};
    return {*(first + (minimum - 1)), *(last - minimum)};
}

/// @brief Maximizes the added hinge's sum of squares over one feasible coordinate.
///
/// In an observed gap the hinge is A minus t times B, where B is the
/// right-side indicator and A is the right-side excess over the gap's
/// lower endpoint. After projection away from the nuisance design, the
/// improvement is the squared residual inner product divided by hinge
/// squared norm. It is a ratio of quadratics. Suffix sums give it and
/// its nonzero stationary point in every gap.
long double best_coordinate(const Data& data, const Vector& knots, std::size_t index, std::size_t minimum) {
    const auto [lower, upper] = coordinate_bounds(data, knots, index, minimum);
    if (lower == upper)
        return lower;
    const auto matrix = design(data, knots, index);
    const auto factor = decompose(matrix);
    const auto linear = fit_linear(data, matrix, factor);
    const auto p = matrix.cols();
    Vector q(p), qa(p);
    long double raw_aa = 0, raw_ab = 0, r = 0, alpha = 0;
    auto anchor = data.distinct.back();
    std::size_t i = data.x.size();
    long double best = knots[index], best_gain = -1;
    for (std::size_t gap = data.distinct.size() - 1; gap > 0; --gap) {
        const auto left = data.distinct[gap - 1];
        const auto right = data.distinct[gap];
        if (right < lower)
            break;
        // Translate the existing suffix to the new gap origin before adding rows.
        const auto shift = anchor - left;
        const auto previous_count = static_cast<long double>(data.x.size() - i);
        raw_aa += shift * (2 * raw_ab + shift * previous_count);
        raw_ab += shift * previous_count;
        alpha += shift * r;
        for (std::size_t j = 0; j < p; ++j)
            qa[j] += shift * q[j];
        anchor = left;
        while (i > 0 && data.x[i - 1] > left) {
            --i;
            const auto value = data.x[i] - left;
            raw_ab += value;
            raw_aa += value * value;
            r += linear.residuals[i];
            alpha += linear.residuals[i] * value;
            for (std::size_t j = 0; j < p; ++j) {
                q[j] += factor.U(i, j);
                qa[j] += factor.U(i, j) * value;
            }
        }
        if (left > upper)
            continue;
        const auto count = static_cast<long double>(data.x.size() - i);
        long double aa = raw_aa, ab = raw_ab, bb = count;
        for (std::size_t j = 0; j < p; ++j) {
            aa -= qa[j] * qa[j];
            ab -= qa[j] * q[j];
            bb -= q[j] * q[j];
        }
        const auto lo = std::max(left, lower) - left;
        const auto hi = std::min(right, upper) - left;
        /// @brief Records the best feasible hinge gain in this gap.
        const auto consider = [&](long double t) {
            if (!(t >= lo && t <= hi))
                return;
            const auto denominator = aa - 2 * t * ab + t * t * bb;
            const auto raw_norm = raw_aa - 2 * t * raw_ab + t * t * count;
            if (!(denominator > std::numeric_limits<long double>::epsilon() * std::max(0.0L, raw_norm)))
                return;
            const auto numerator = std::fma(-t, r, alpha);
            const auto score = numerator * numerator / denominator;
            if (score > best_gain) {
                best_gain = score;
                best = std::clamp(left + t, lower, upper);
            }
        };
        consider(knots[index] - left);
        consider(lo);
        consider(hi);
        const auto denominator = alpha * bb - r * ab;
        if (denominator != 0)
            consider((alpha * ab - r * aa) / denominator);
    }
    return best;
}

/// @brief Checks whether a candidate ordered segmentation has the required support.
bool feasible(const Data& data, const Vector& knots, std::size_t minimum) {
    for (std::size_t i = 0; i < knots.size(); ++i) {
        const auto [lo, hi] = coordinate_bounds(data, knots, i, minimum);
        if (!(knots[i] >= lo && knots[i] <= hi))
            return false;
    }
    return true;
}

/// @brief Fills the interior knots of one section using spaced observed ranks.
void place_knots(Vector& knots, const Vector& distinct, std::size_t offset, std::size_t parts,
                  std::size_t first, std::size_t last, std::size_t spacing, long double power) {
    const auto extra = last - first - parts * spacing;
    for (std::size_t j = 1; j < parts; ++j) {
        const auto fraction = static_cast<long double>(j) / parts;
        const auto shift = static_cast<std::size_t>(std::round(extra * std::pow(fraction, power)));
        knots[offset + j - 1] = distinct[first + j * spacing + shift];
    }
}

/// @brief Builds spaced starts and keeps the best conditional least-squares result.
SearchResult search(const Data& data, const SegmentedRegressionOptions& options,
                    std::optional<FixedKnot> fixed = {}, const Vector& warm = {}) {
    const auto count = options.breakpoints;
    const auto spacing = options.minimum_segment_size - 1;
    const auto last = data.distinct.size() - 1;
    SearchResult best{{}, {{}, {}, std::numeric_limits<long double>::infinity()}, false};
    /// @brief Runs a conditional block search from one supplied configuration.
    const auto solve = [&](Vector knots) {
        bool converged = false;
        for (std::size_t sweep = 0; sweep < options.max_sweeps; ++sweep) {
            long double motion = 0;
            for (std::size_t j = 0; j < count; ++j) {
                if (fixed && j == fixed->first)
                    continue;
                const auto next = best_coordinate(data, knots, j, options.minimum_segment_size);
                motion = std::max(motion, std::abs(next - knots[j]));
                knots[j] = next;
            }
            if (count - static_cast<std::size_t>(fixed.has_value()) <= 1 || motion <= options.tolerance) {
                converged = true;
                break;
            }
        }
        const auto matrix = design(data, knots, count);
        auto fit = fit_linear(data, matrix, decompose(matrix));
        if (fit.ss < best.linear.ss || (fit.ss == best.linear.ss && converged))
            best = {std::move(knots), std::move(fit), converged};
    };
    const auto free_count = count - static_cast<std::size_t>(fixed.has_value());
    if (fixed && free_count > 1 && !warm.empty()) {
        auto knots = warm;
        knots[fixed->first] = fixed->second;
        if (feasible(data, knots, options.minimum_segment_size))
            solve(std::move(knots));
    }
    const auto starts = free_count <= 1 ? 1 : options.starts;
    for (std::size_t i = 0; i < starts; ++i) {
        const auto power = i == 0 ? 1.0L : std::exp(std::lerp(-1.0L, 1.0L, (i - 1.0L) / std::max(1.0L, starts - 2.0L)));
        Vector knots(count);
        if (fixed) {
            const auto [index, location] = *fixed;
            const auto left = std::ranges::upper_bound(data.distinct, location) - data.distinct.begin();
            const auto right = std::ranges::lower_bound(data.distinct, location) - data.distinct.begin();
            place_knots(knots, data.distinct, 0, index + 1, 0, left - 1, spacing, power);
            place_knots(knots, data.distinct, index + 1, count - index, right, last, spacing, power);
            knots[index] = location;
        } else {
            place_knots(knots, data.distinct, 0, count + 1, 0, last, spacing, power);
        }
        solve(std::move(knots));
    }
    return best;
}

} // namespace

SegmentedRegressionResult segmented_regression(const std::vector<double>& predictor,
    const std::vector<double>& response, const SegmentedRegressionOptions& options) {
    const auto n = predictor.size();
    const auto count = options.breakpoints;
    const auto finite = [](double x) { return std::isfinite(x); };
    if (n < 5 || response.size() != n || count == 0 || count > (n - 3) / 2 ||
        options.minimum_segment_size < 2 || options.starts == 0 || options.max_sweeps == 0 ||
        !(options.tolerance > 0 && options.tolerance < 1) ||
        !(options.confidence_level > 0 && options.confidence_level < 1) ||
        !std::ranges::all_of(predictor, finite) || !std::ranges::all_of(response, finite))
        throw std::invalid_argument("segmented_regression: incompatible data or search controls");
    Data data{Vector(n), Vector(n), {}, std::vector<std::size_t>(n), 0, 0, 0};
    std::iota(data.order.begin(), data.order.end(), 0u);
    std::ranges::stable_sort(data.order, {}, [&](std::size_t i) { return predictor[i]; });
    data.origin = predictor[data.order.front()];
    data.span = static_cast<long double>(predictor[data.order.back()]) - data.origin;
    if (!(data.span > 0) || !std::isfinite(data.span))
        throw std::invalid_argument("segmented_regression: predictor range must be finite and positive");
    for (auto value : response)
        data.response_scale = std::max(data.response_scale, std::abs(static_cast<long double>(value)));
    if (data.response_scale == 0)
        data.response_scale = 1;
    for (std::size_t i = 0; i < n; ++i) {
        data.x[i] = (static_cast<long double>(predictor[data.order[i]]) - data.origin) / data.span;
        data.y[i] = static_cast<long double>(response[data.order[i]]) / data.response_scale;
        if (i == 0 || data.x[i] != data.x[i - 1])
            data.distinct.push_back(data.x[i]);
    }
    const auto spacing = options.minimum_segment_size - 1;
    if (spacing > (data.distinct.size() - 1) / (count + 1))
        throw std::invalid_argument("segmented_regression: too few distinct predictors for segment support");
    const auto solution = search(data, options);
    const auto original_x = [&](long double x) { return output(std::fma(x, data.span, data.origin)); };
    SegmentedRegressionResult result{output(data.origin), output(solution.linear.beta.front() * data.response_scale),
        {}, {}, std::vector<double>(n), output(solution.linear.ss * data.response_scale * data.response_scale),
        solution.converged, {}};
    auto slope = solution.linear.beta[1];
    result.slopes.push_back(output(slope * data.response_scale / data.span));
    for (std::size_t j = 0; j < count; ++j) {
        result.breakpoints.push_back(original_x(solution.knots[j]));
        slope += solution.linear.beta[j + 2];
        result.slopes.push_back(output(slope * data.response_scale / data.span));
    }
    for (std::size_t i = 0; i < n; ++i)
        result.fitted[data.order[i]] = output((data.y[i] - solution.linear.residuals[i]) * data.response_scale);
    if (!options.profile_intervals || !solution.converged)
        return result;
    const auto baseline = solution.linear.ss;
    const auto floor = 16 * std::numeric_limits<double>::epsilon();
    if (baseline <= n * floor * floor)
        throw std::domain_error("segmented_regression: positive residual dispersion is needed for profiling");
    for (std::size_t j = 0; j < count; ++j) {
        const double estimate = static_cast<double>(solution.knots[j]);
        const double lower = static_cast<double>(data.distinct[(j + 1) * spacing]);
        const double upper = static_cast<double>(data.distinct[data.distinct.size() - 1 - (count - j) * spacing]);
        if (lower == upper) {
            result.intervals.push_back({{original_x(lower), ProfileEndpointStatus::ParameterBound},
                {original_x(upper), ProfileEndpointStatus::ParameterBound},
                random::gamma_quantile(options.confidence_level, .5, 2)});
            continue;
        }
        const auto profile = [&](double fixed) {
            if (fixed == estimate)
                return 0.0;
            const auto location = std::clamp(static_cast<long double>(fixed),
                data.distinct[(j + 1) * spacing], data.distinct[data.distinct.size() - 1 - (count - j) * spacing]);
            const auto restricted = search(data, options, FixedKnot{j, location}, solution.knots);
            if (!restricted.converged)
                throw std::runtime_error("segmented_regression: a breakpoint nuisance fit did not converge");
            const auto change = (restricted.linear.ss - baseline) / baseline;
            if (change < 0 && change >= -64 * std::numeric_limits<double>::epsilon())
                return 0.0;
            return static_cast<double>(-.5L * n * std::log1p(change));
        };
        auto interval = profile_likelihood_interval(estimate, profile, {
            .confidence_level = options.confidence_level, .lower_bound = lower, .upper_bound = upper,
            .initial_step = std::min(.01, (upper - lower) / 8), .tolerance = options.tolerance,
        });
        interval.lower.value = original_x(interval.lower.value);
        interval.upper.value = original_x(interval.upper.value);
        result.intervals.push_back(interval);
    }
    return result;
}

} // namespace datamunge::stats
