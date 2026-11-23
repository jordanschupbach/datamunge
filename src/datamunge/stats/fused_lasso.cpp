/// @file fused_lasso.cpp
/// @brief Adjacent-difference shrinkage and dual interval checks for fused lasso.

#include <datamunge/stats/fused_lasso.hpp>

#include "proximal_least_squares.hpp"

namespace datamunge::stats {
/// @brief One-dimensional total variation and stationarity for ordered slopes.
namespace {
/// @brief Working-precision slope vector.
using Vector = detail::ProximalVector;

/// @brief Applies Condat's direct total-variation proximal operator in place.
void fuse(Vector& values, long double lambda) {
    if (values.size() < 2 || lambda == 0)
        return;
    // A constant fits if the cumulative centered residual stays in the dual tube.
    // Check it first to avoid subtracting large penalties in this common limit.
    const auto origin = values.front();
    long double difference = 0;
    for (const auto value : values)
        difference += (value - origin) / values.size();
    long double cumulative = 0, maximum = 0;
    for (const auto value : values) {
        cumulative += (value - origin) - difference;
        maximum = std::max(maximum, std::abs(cumulative));
    }
    if (maximum <= lambda) {
        std::ranges::fill(values, origin + difference);
        return;
    }

    // The lower and upper levels bound the next constant segment. Each
    // associated dual sum consumes one end of the admissible interval.
    const auto last = values.size() - 1;
    std::size_t first = 0, cursor = 0, low_end = 0, high_end = 0;
    auto low = values.front() - lambda, high = values.front() + lambda;
    auto dual_low = lambda, dual_high = -lambda;
    while (true) {
        bool falling;
        if (cursor == last) {
            if (dual_low >= 0 && dual_high <= 0) {
                std::fill(values.begin() + first, values.end(), low + dual_low / (cursor - first + 1));
                return;
            }
            falling = dual_low < 0;
        } else {
            const auto next = values[cursor + 1];
            const auto lower = dual_low + (next - low);
            const auto upper = dual_high + (next - high);
            if (lower >= -lambda && upper <= lambda) {
                ++cursor;
                const auto width = cursor - first + 1;
                dual_low = lower;
                dual_high = upper;
                if (lower >= lambda) {
                    low += (lower - lambda) / width;
                    dual_low = lambda;
                    low_end = cursor;
                }
                if (upper <= -lambda) {
                    high += (upper + lambda) / width;
                    dual_high = -lambda;
                    high_end = cursor;
                }
                continue;
            }
            falling = lower < -lambda;
        }

        // A bound failed. Fix the previous plateau at the last contact
        // with that bound, then start with the dual already at the jump.
        const auto stop = falling ? low_end : high_end;
        std::fill(values.begin() + first, values.begin() + stop + 1, falling ? low : high);
        if (stop == last)
            return;
        first = cursor = low_end = high_end = stop + 1;
        low = falling ? values[first] : values[first] - 2 * lambda;
        high = falling ? values[first] + 2 * lambda : values[first];
        dual_low = lambda;
        dual_high = -lambda;
    }
}

/// @brief Returns true when admissible subgradient intervals satisfy all slope equations.
bool stationary(const Vector& slopes, const Vector& gradient, long double l1, long double difference,
                long double slack) {
    long double lower = 0, upper = 0;
    for (std::size_t j = 0; j < slopes.size(); ++j) {
        const auto signed_l1 = std::copysign(l1, slopes[j]);
        lower += gradient[j] + (slopes[j] == 0 ? -l1 : signed_l1) - slack;
        upper += gradient[j] + (slopes[j] == 0 ? l1 : signed_l1) + slack;

        // If D beta is the forward difference, its contribution to the
        // current equation is u[j-1] - u[j]. Propagate admissible u[j].
        long double allowed_lower = 0, allowed_upper = 0;
        if (j + 1 < slopes.size()) {
            allowed_lower = slopes[j + 1] > slopes[j] ? difference : -difference;
            allowed_upper = slopes[j + 1] < slopes[j] ? -difference : difference;
        }
        lower = std::max(lower, allowed_lower);
        upper = std::min(upper, allowed_upper);
        if (lower > upper)
            return false;
    }
    return true;
}

/// @brief Measures the smallest uniform slack admitting a stationarity certificate.
long double kkt_error(const Vector& beta, const Vector& gradient, long double l1,
                      long double difference, long double tolerance) {
    if (stationary(beta, gradient, l1, difference, 0))
        return 0;
    long double lower = 0, upper = tolerance;
    if (!stationary(beta, gradient, l1, difference, upper)) {
        // Fix the dual on each observed jump and choose zero on free
        // coordinates. Its largest discrepancy is a feasible upper bound.
        long double previous = 0;
        for (std::size_t j = 0; j < beta.size(); ++j) {
            const auto next = j + 1 == beta.size() || beta[j + 1] == beta[j] ? 0 :
                std::copysign(difference, beta[j + 1] - beta[j]);
            const auto individual = beta[j] == 0 ? 0 : std::copysign(l1, beta[j]);
            upper = std::max(upper, std::abs(gradient[j] + individual + (previous - next)));
            previous = next;
        }
        // Add a scale-relative margin for summing many balancing terms.
        upper *= 1 + 64 * std::numeric_limits<long double>::epsilon();
    }
    for (int i = 0; i < 64; ++i) {
        const auto middle = std::midpoint(lower, upper);
        if (stationary(beta, gradient, l1, difference, middle))
            upper = middle;
        else
            lower = middle;
    }
    return upper;
}

} // namespace

std::vector<double> FusedLassoFit::predict(const linalg::DenseMatrix<double>& design) const {
    return detail::predict_penalized_least_squares(design, coefficients, intercept);
}

FusedLassoFit fused_lasso(const linalg::DenseMatrix<double>& design, const std::vector<double>& response,
    const FusedLassoOptions& options) {
    const auto n = design.rows(), d = design.cols();
    const auto finite = [](double value) { return std::isfinite(value); };
    const auto nonnegative = [&](double value) { return value >= 0 && finite(value); };
    if (n == 0 || d == 0 || response.size() != n || (!options.weights.empty() && options.weights.size() != n) ||
        !nonnegative(options.lambda_l1) || !nonnegative(options.lambda_difference) ||
        !finite(options.tolerance) || !(options.tolerance > 0) || options.max_iterations == 0 ||
        !std::ranges::all_of(design.data(), finite) || !std::ranges::all_of(response, finite) ||
        !std::ranges::all_of(options.weights, [&](double value) { return value > 0 && finite(value); }))
        throw std::invalid_argument("fused lasso: invalid data or solver controls");

    const auto problem = detail::prepare_penalized_least_squares(design, response, options.weights, false);
    auto curvature = problem.curvature;
    Vector beta(d);
    const auto initial = detail::evaluate_quadratic(problem, beta);
    long double score = 0;
    for (const auto value : initial.gradient)
        score = std::max(score, std::abs(value));
    const auto tolerance = options.tolerance * score;
    const long double l1 = options.lambda_l1, difference = options.lambda_difference;
    /// @brief Applies neighbor fusion before soft-thresholding each plateau.
    const auto prox = [&](Vector& values, long double current_curvature) {
        fuse(values, difference / current_curvature);
        for (auto& value : values)
            value = detail::soft_threshold(value, l1 / current_curvature);
    };
    /// @brief Tests stationarity to the requested slope-score tolerance.
    const auto done = [&](const Vector& values, const Vector& gradient) {
        return stationary(values, gradient, l1, difference, tolerance);
    };
    const auto solution = detail::solve_proximal(problem, beta, curvature, options.max_iterations, prox, done);
    FusedLassoFit result{0, {}, {}, 0,
        detail::proximal_output(kkt_error(beta, solution.state.gradient, l1, difference, tolerance)),
        solution.iterations, solution.converged};
    auto intercept = problem.intercept;
    auto objective = solution.state.loss;
    result.coefficients.reserve(d);
    for (std::size_t j = 0; j < d; ++j) {
        const auto value = detail::proximal_output(beta[j]);
        result.coefficients.push_back(value);
        intercept = std::fma(-beta[j], problem.centers[j], intercept);
        objective += l1 * std::abs(beta[j]);
        if (j > 0)
            objective += difference * std::abs(beta[j] - beta[j - 1]);
        if (result.segments.empty() || value != result.segments.back().level)
            result.segments.push_back({j, j + 1, value});
        else
            result.segments.back().end = j + 1;
    }
    result.intercept = detail::proximal_output(intercept);
    result.objective = detail::proximal_output(objective);
    return result;
}

} // namespace datamunge::stats
