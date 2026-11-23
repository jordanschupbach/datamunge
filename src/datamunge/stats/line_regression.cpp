/// @file line_regression.cpp
/// @brief Median slope estimators and Deming errors-in-variables regression.

#include <datamunge/stats/line_regression.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace datamunge::stats {

/// @brief Paired-value checks and extended-precision arithmetic for line estimates.
namespace {

/// @brief Rejects unequal, insufficient or nonfinite measurements.
void check_pairs(const std::vector<double>& x, const std::vector<double>& y) {
    const auto finite = [](double value) { return std::isfinite(value); };
    if (x.size() != y.size() || x.size() < 2 || !std::ranges::all_of(x, finite) ||
        !std::ranges::all_of(y, finite))
        throw std::invalid_argument("line regression: need at least two finite paired measurements");
}

/// @brief Tests whether a coefficient fits in the finite double range.
bool representable(long double value) {
    return std::isfinite(value) && std::abs(value) <= std::numeric_limits<double>::max();
}

/// @brief Returns the midpoint median of a nonempty working vector.
long double median(std::vector<long double>& values) {
    auto middle = values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2);
    std::nth_element(values.begin(), middle, values.end());
    if (values.size() % 2 != 0)
        return *middle;
    return std::midpoint(*std::max_element(values.begin(), middle), *middle);
}

/// @brief Computes the slope for a pair of observations with distinct predictors.
long double pair_slope(double xi, double yi, double xj, double yj) {
    const long double dx = static_cast<long double>(xj) - xi;
    const long double dy = static_cast<long double>(yj) - yi;
    if (std::isfinite(dx) && std::isfinite(dy))
        return dy / dx;
    return (yj / 2.0L - yi / 2.0L) / (xj / 2.0L - xi / 2.0L);
}

/// @brief Converts a median slope and a median residual intercept to public coefficients.
LineCoefficients median_line(const std::vector<double>& x, const std::vector<double>& y, long double slope) {
    if (!representable(slope))
        throw std::overflow_error("line regression: slope exceeds finite double range");
    std::vector<long double> intercepts;
    intercepts.reserve(x.size());
    for (std::size_t i = 0; i < x.size(); ++i)
        intercepts.push_back(std::fma(-slope, static_cast<long double>(x[i]), y[i]));
    const long double intercept = median(intercepts);
    if (!representable(intercept))
        throw std::overflow_error("line regression: intercept exceeds finite double range");
    return {static_cast<double>(intercept), static_cast<double>(slope)};
}

} // namespace

LineCoefficients theil_sen_regression(const std::vector<double>& x, const std::vector<double>& y) {
    check_pairs(x, y);
    std::vector<long double> slopes;
    for (std::size_t i = 0; i < x.size(); ++i)
        for (std::size_t j = i + 1; j < x.size(); ++j)
            if (x[j] != x[i])
                slopes.push_back(pair_slope(x[i], y[i], x[j], y[j]));
    if (slopes.empty())
        throw std::invalid_argument("theil_sen_regression: predictor must vary");
    return median_line(x, y, median(slopes));
}

LineCoefficients repeated_median_regression(const std::vector<double>& x, const std::vector<double>& y) {
    check_pairs(x, y);
    if (std::ranges::all_of(x, [&](double value) { return value == x.front(); }))
        throw std::invalid_argument("repeated_median_regression: predictor must vary");
    std::vector<long double> slopes;
    std::vector<long double> local_medians;
    slopes.reserve(x.size() - 1);
    local_medians.reserve(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        slopes.clear();
        for (std::size_t j = 0; j < x.size(); ++j)
            if (x[j] != x[i])
                slopes.push_back(pair_slope(x[i], y[i], x[j], y[j]));
        local_medians.push_back(median(slopes));
    }
    return median_line(x, y, median(local_medians));
}

DemingLine deming_regression(const std::vector<double>& x, const std::vector<double>& y,
                            double error_variance_ratio) {
    check_pairs(x, y);
    if (!std::isfinite(error_variance_ratio) || error_variance_ratio <= 0)
        throw std::invalid_argument("deming_regression: error variance ratio must be positive and finite");
    const long double response_scale = std::sqrt(static_cast<long double>(error_variance_ratio));
    const long double x_origin = x.front();
    const long double y_origin = y.front();
    const auto dx = [&](std::size_t i) { return static_cast<long double>(x[i]) - x_origin; };
    const auto dy = [&](std::size_t i) { return (static_cast<long double>(y[i]) - y_origin) / response_scale; };

    long double scale = 0;
    for (std::size_t i = 0; i < x.size(); ++i)
        scale = std::max({scale, std::abs(dx(i)), std::abs(dy(i))});
    if (!std::isfinite(scale))
        throw std::overflow_error("deming_regression: centered measurements exceed working range");
    if (scale == 0)
        throw std::domain_error("deming_regression: identical measurements have no principal direction");

    long double cx = 0, cy = 0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        cx += dx(i) / scale;
        cy += dy(i) / scale;
    }
    cx /= static_cast<long double>(x.size());
    cy /= static_cast<long double>(x.size());
    long double xx = 0, yy = 0, xy = 0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        const long double u = dx(i) / scale - cx;
        const long double v = dy(i) / scale - cy;
        xx += u * u;
        yy += v * v;
        xy += u * v;
    }
    const long double difference = xx - yy;
    if (difference == 0 && xy == 0)
        throw std::domain_error("deming_regression: scaled principal directions are tied");

    // Select the eigenvector expression with the well-resolved large component.
    const long double gap = std::hypot(difference, 2 * xy);
    const long double major = gap + std::abs(difference);
    const long double vx = difference >= 0 ? major : 2 * std::abs(xy);
    const long double vy = response_scale * (difference >= 0 ? 2 * xy : (xy < 0 ? -major : major));
    const long double norm = std::hypot(vx, vy);
    const long double center_x = std::fma(cx, scale, x_origin);
    const long double center_y = std::fma(cy, scale * response_scale, y_origin);
    if (!representable(center_x) || !representable(center_y) || !std::isfinite(norm))
        throw std::overflow_error("deming_regression: line geometry exceeds working range");
    DemingLine result{static_cast<double>(center_x), static_cast<double>(center_y),
                      static_cast<double>(vx / norm), static_cast<double>(vy / norm), std::nullopt};
    if (vx > 0) {
        const long double slope = vy / vx;
        const long double intercept = std::fma(-slope, center_x, center_y);
        if (representable(slope) && representable(intercept))
            result.coefficients = LineCoefficients{static_cast<double>(intercept), static_cast<double>(slope)};
    }
    return result;
}

} // namespace datamunge::stats
