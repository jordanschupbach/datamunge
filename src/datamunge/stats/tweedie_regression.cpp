/// @file tweedie_regression.cpp
/// @brief Compound Poisson–gamma log density and log-link likelihood profiling.

#include <datamunge/stats/tweedie_regression.hpp>

#include "likelihood_newton.hpp"
#include "../random/count_likelihood.hpp"
#include <datamunge/algorithms/golden_section_search.hpp>
#include <datamunge/linalg/svd.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace datamunge::stats {
/// @brief Working coordinates and density series for the Tweedie fit.
namespace {
/// @brief Extended-precision working matrix.
using Matrix = linalg::DenseMatrix<long double>;
/// @brief Extended-precision working vector.
using Vector = std::vector<long double>;
/// @brief Outcome of optimization and inference.
using Status = TweedieRegressionStatus;
/// @brief Internal coefficient optimizer outcomes.
enum class MeanStatus {
    Converged,       ///< Identifiable predictor step below tolerance.
    IterationLimit,  ///< Exhausted updates.
    NumericalFailure ///< Failed information or line search.
};
/// @brief Score and lower-triangular information at a conditional fit.
struct Evaluation {
    long double value; ///< Negative half unit deviance, averaged by case weight.
    Vector score;      ///< Score with constant dispersion one.
    Matrix info;       ///< Observed mean-parameter information at dispersion one.
};
/// @brief Observations and normalized case weights used in repeated profiles.
struct Problem {
    Matrix x;                               ///< Column-scaled design.
    const std::vector<double>& response;    ///< Original nonnegative responses.
    Vector log_response;                    ///< Natural logarithms for positive responses.
    Vector offset;                          ///< Fixed additive offsets.
    Vector weights;                         ///< Case weights normalized to sum to one.
};
/// @brief Mean and dispersion maximized at one choice of power.
struct Profile {
    double power{0};           ///< Fixed power for the conditional fit.
    long double log_phi{0};    ///< Profiled log dispersion.
    long double value{-std::numeric_limits<long double>::infinity()}; ///< Mean weighted log likelihood.
    Vector beta;               ///< Working-scale log-mean coefficients.
    Vector eta;                ///< Log means including exposure offsets.
    bool bracketed{false};     ///< A finite dispersion maximum found away from an unresolved search edge.
};

/// @brief Converts a working quantity to finite public precision.
double output(long double value) {
    if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<double>::max())
        throw std::overflow_error("Tweedie regression: result exceeds the finite reporting range");
    return static_cast<double>(value);
}

/// @brief Computes an exponential remainder without cancellation near zero.
long double exponential_remainder(long double x) {
    if (std::abs(x) > .003L)
        return std::expm1(x) - x;
    return x * x * (.5L + x * (1.0L/6 + x * (1.0L/24 + x * (1.0L/120 + x * (1.0L/720 + x / 5040)))));
}

/// @brief Computes conditional mean likelihood differences, score and information.
Evaluation evaluate_mean(const Problem& problem, const Vector& beta, long double power) {
    const auto d = beta.size();
    Evaluation state{0, Vector(d), Matrix(d, d)};
    const auto a = power - 1, b = 2 - power;
    const auto linear = problem.x * beta;
    for (std::size_t i = 0; i < linear.size(); ++i) {
        const auto eta = linear[i] + problem.offset[i];
        const auto mean_power = std::exp(b * eta);
        const auto y = problem.response[i];
        const auto log_y = problem.log_response[i];
        const auto scaled_y = y > 0 ? std::exp(log_y - a * eta) : 0;
        const auto weight = problem.weights[i];
        auto half_deviance = mean_power / b;
        auto score = -mean_power;
        if (y > 0) {
            const auto delta = eta - log_y;
            half_deviance = std::exp(b * log_y) * (exponential_remainder(-a * delta) / a +
                                                   exponential_remainder(b * delta) / b);
            score = std::abs(delta) < .5L ? mean_power * std::expm1(-delta) : scaled_y - mean_power;
        }
        if (!std::isfinite(half_deviance) || !std::isfinite(score))
            return {-std::numeric_limits<long double>::infinity(), {}, {}};
        const auto curvature = a * scaled_y + b * mean_power;
        state.value -= weight * half_deviance;
        for (std::size_t j = 0; j < d; ++j) {
            const auto x = problem.x(i, j);
            state.score[j] += weight * x * score;
            for (std::size_t k = 0; k <= j; ++k)
                state.info(j, k) += weight * curvature * x * problem.x(i, k);
        }
    }
    return state;
}

/// @brief Log gamma density using the response, its log, and the scale-one variate.
long double gamma_log_density(long double log_y, long double log_scaled, long double scaled, long double shape) {
    if (shape < 16)
        return shape * log_scaled - scaled - std::lgamma(shape) - log_y;
    const auto delta = (scaled - shape) / shape;
    const auto loss = delta < -.5L ? shape * (std::log(shape) - log_scaled) + scaled - shape
                                  : shape * random::detail::gamma_divergence(delta);
    return .5L * (std::log(shape) - std::log(2 * std::numbers::pi_v<long double>)) -
           random::detail::stirling_remainder(shape) - loss - log_y;
}

/// @brief Evaluates one Tweedie response using a mode-centered positive density series.
long double log_density(double y, long double log_y, long double eta, long double log_phi, long double power) {
    constexpr auto failure = -std::numeric_limits<long double>::infinity();
    const auto a = power - 1, b = 2 - power, shape = b / a;
    const auto log_rate = b * eta - log_phi - std::log(b);
    const auto rate = std::exp(log_rate);
    if (y == 0)
        return -rate;
    const auto log_scaled = log_y - log_phi - std::log(a) - a * eta;
    const auto scaled = std::exp(log_scaled);
    const auto ratio_base = -log_phi - std::log(b) + shape * (log_y - log_phi - std::log(a));
    if (!std::isfinite(rate) || !std::isfinite(scaled) || !std::isfinite(ratio_base))
        return failure;

    /// @brief Ratio of consecutive positive summands, expressed on the log scale.
    const auto ratio = [&](std::uint64_t index) {
        return ratio_base - std::log(static_cast<long double>(index) + 1) -
               random::detail::gamma_increment(shape * index, shape);
    };
    // The ratios decrease with index. Use the asymptotic mode as a starting
    // point, then locate the first non-increasing step without walking to it.
    constexpr std::uint64_t largest = std::uint64_t{1} << 50;
    const auto approximate = std::exp(b * log_y - log_phi - std::log(b));
    auto upper = static_cast<std::uint64_t>(std::clamp(approximate, 1.0L, static_cast<long double>(largest)));
    std::uint64_t lower = 1;
    while (ratio(upper) > 0 && upper < largest) {
        lower = upper + 1;
        upper = std::min(largest, 2 * upper);
    }
    if (upper == largest && ratio(upper) > 0)
        return failure;
    while (lower < upper) {
        const auto mid = std::midpoint(lower, upper);
        if (ratio(mid) > 0)
            lower = mid + 1;
        else
            upper = mid;
    }
    const auto mode = lower;
    const auto peak = random::detail::poisson_log_mass(mode, log_rate, rate) +
                      gamma_log_density(log_y, log_scaled, scaled, shape * mode);
    if (!std::isfinite(peak))
        return failure;

    long double sum = 1;
    for (const int direction : {-1, 1}) {
        auto index = mode;
        long double relative = 0;
        for (std::size_t j = 0;; ++j) {
            if (direction < 0 && index == 1)
                break;
            if (j == 20000)
                return failure;
            const auto step = direction < 0 ? -ratio(index - 1) : ratio(index);
            // Log-concavity bounds all remaining terms by this geometric tail.
            if (step < 0 && std::exp(relative + step) <= 1e-15L * sum * -std::expm1(step))
                break;
            relative += step;
            sum += std::exp(relative);
            if (direction < 0)
                --index;
            else
                ++index;
        }
    }
    return peak + std::log(sum);
}

/// @brief Average weighted likelihood including the zero mass and positive density.
long double likelihood(const Problem& problem, const Vector& eta, long double log_phi, long double power) {
    long double result = 0;
    for (std::size_t i = 0; i < eta.size(); ++i) {
        const auto value = log_density(problem.response[i], problem.log_response[i], eta[i], log_phi, power);
        if (!std::isfinite(value))
            return -std::numeric_limits<long double>::infinity();
        result += problem.weights[i] * value;
    }
    return result;
}

/// @brief Fits mean coefficients and profiles dispersion at a fixed power.
Profile at_power(const Problem& problem, const Vector& start, double power) {
    Profile result;
    result.power = power;
    /// @brief Evaluates the concave conditional mean likelihood.
    const auto evaluate = [&](const Vector& beta) { return evaluate_mean(problem, beta, power); };
    /// @brief Measures the scoring step in the log mean at all training cases.
    const auto effect = [&](const Vector& step) {
        long double maximum = 0;
        for (auto delta : problem.x * step)
            maximum = std::max(maximum, std::abs(delta));
        return detail::LikelihoodStep{maximum, false};
    };
    auto mean_fit = detail::maximize_likelihood<MeanStatus>(start, 200, 1e-10, evaluate, effect);
    if (!mean_fit || mean_fit->status != MeanStatus::Converged)
        return result;
    result.beta = std::move(mean_fit->beta);
    result.eta = problem.x * result.beta;
    long double pearson = 0, reference = 0;
    for (std::size_t i = 0; i < result.eta.size(); ++i) {
        auto& eta = result.eta[i];
        eta += problem.offset[i];
        const auto d = (problem.response[i] - std::exp(eta)) * std::exp(-.5L * power * eta);
        pearson += problem.weights[i] * d * d;
        reference += problem.weights[i] * std::exp((2 - power) * eta);
    }
    const auto center = static_cast<double>(std::clamp(std::log(std::max(pearson, std::max(reference * 1e-14L,
        static_cast<long double>(std::numeric_limits<double>::min())))), -700.0L, 700.0L));
    /// @brief Evaluates one log dispersion and preserves the best finite likelihood.
    const auto consider = [&](double log_phi) {
        const auto value = likelihood(problem, result.eta, log_phi, power);
        if (std::isfinite(value) && value > result.value) {
            result.value = value;
            result.log_phi = log_phi;
        }
        return static_cast<double>(std::max(value, -static_cast<long double>(std::numeric_limits<double>::max())));
    };
    std::vector<double> grid, values;
    // As p approaches one, the gamma components concentrate around
    // multiples of the scale. Sample dispersion more finely in that regime.
    const auto spacing = std::clamp(.35 * std::sqrt((power - 1) / (2 - power)), .02, .5);
    const auto half_grid = static_cast<int>(std::ceil(3.5 / spacing));
    for (int i = -half_grid; i <= half_grid; ++i) {
        grid.push_back(center + i * spacing);
        values.push_back(consider(grid.back()));
    }
    // Expand when a sampled edge is still rising in its outward direction.
    for (const int direction : {-1, 1}) {
        if (direction < 0) {
            std::ranges::reverse(grid);
            std::ranges::reverse(values);
        }
        for (int i = 0; i < 20 && values.back() > values[values.size() - 2]; ++i) {
            grid.push_back(grid.back() + direction);
            values.push_back(consider(grid.back()));
        }
        if (direction < 0) {
            std::ranges::reverse(grid);
            std::ranges::reverse(values);
        }
    }
    for (std::size_t i = 1; i + 1 < grid.size(); ++i)
        if (values[i] != -std::numeric_limits<double>::max() && values[i] >= values[i - 1] &&
            values[i] >= values[i + 1] && (values[i] > values[i - 1] || values[i] > values[i + 1]))
            algorithms::golden_section_search(consider, grid[i - 1], grid[i + 1], 1e-8, 80);
    const auto ahead = likelihood(problem, result.eta, result.log_phi + .01L, power);
    const auto behind = likelihood(problem, result.eta, result.log_phi - .01L, power);
    result.bracketed = std::isfinite(ahead) && std::isfinite(behind) && result.value >= ahead && result.value >= behind;
    return result;
}

/// @brief Computes conditional or joint observed information in working coordinates.
linalg::DenseMatrix<double> covariance(const Problem& problem, const Profile& fit, const Vector& scales,
    long double weight_sum, bool joint_power) {
    const auto p = fit.beta.size(), d = p + 1 + static_cast<std::size_t>(joint_power);
    const auto phi = std::exp(fit.log_phi);
    const long double power = fit.power;
    const auto beta_info = evaluate_mean(problem, fit.beta, power).info;
    Matrix hessian(d, d);
    for (std::size_t j = 0; j < p; ++j)
        for (std::size_t k = 0; k <= j; ++k)
            hessian(j, k) = beta_info(j, k) / phi;
    for (std::size_t i = 0; i < fit.eta.size(); ++i) {
        const auto eta = fit.eta[i];
        const auto mean_power = std::exp((2 - power) * eta);
        const auto score = problem.response[i] == 0 ? -mean_power :
            mean_power * std::expm1(problem.log_response[i] - eta);
        for (std::size_t j = 0; j < p; ++j) {
            const auto entry = problem.weights[i] * problem.x(i, j) * score / phi;
            hessian(p, j) += entry;
            if (joint_power)
                hessian(p + 1, j) += eta * entry;
        }
    }
    constexpr long double delta = .0002L;
    const auto power_delta = std::min(delta, std::min(power - 1, 2 - power) / 8);
    /// @brief Evaluates the fitted-mean log likelihood with shifted shape parameters.
    const auto at = [&](long double v, long double u) {
        return likelihood(problem, fit.eta, fit.log_phi + v, power + u);
    };
    /// @brief Uses a fourth-order central difference for a negative second derivative.
    const auto curvature = [&](auto f, long double h) {
        return ((f(2*h) - fit.value) + (f(-2*h) - fit.value) -
                16 * ((f(h) - fit.value) + (f(-h) - fit.value))) / (12*h*h);
    };
    hessian(p, p) = curvature([&](long double v) { return at(v, 0); }, delta);
    if (joint_power) {
        hessian(p + 1, p + 1) = curvature([&](long double u) { return at(0, u); }, power_delta);
        hessian(p + 1, p) = ((at(delta, -power_delta) - at(delta, power_delta)) -
                             (at(-delta, -power_delta) - at(-delta, power_delta))) / (4 * delta * power_delta);
    }
    Vector normalization(d);
    Matrix corr(d, d);
    for (std::size_t j = 0; j < d; ++j) {
        if (!(hessian(j, j) > 0) || !std::isfinite(hessian(j, j)))
            return {};
        normalization[j] = std::sqrt(hessian(j, j));
        corr(j, j) = 1;
        for (std::size_t k = 0; k < j; ++k) {
            corr(j, k) = hessian(j, k) / normalization[j] / normalization[k];
            if (!std::isfinite(corr(j, k)))
                return {};
        }
    }
    const auto factor = linalg::cholesky(corr);
    if (!factor.ok)
        return {};
    const auto inverse = factor.solve(Matrix::identity(d));
    Vector scale(d, 1);
    for (std::size_t j = 0; j < p; ++j)
        scale[j] = 1 / scales[j];
    scale[p] = phi;
    linalg::DenseMatrix<double> result(d, d);
    for (std::size_t j = 0; j < d; ++j)
        for (std::size_t k = 0; k <= j; ++k)
            result(j, k) = result(k, j) = output(inverse(j, k) * (scale[j] / normalization[j]) *
                                                (scale[k] / normalization[k]) / weight_sum);
    return result;
}

} // namespace

std::vector<double> TweedieRegressionResult::predict(const linalg::DenseMatrix<double>& design,
    const std::vector<double>& offset) const {
    const auto finite = [](double v) { return std::isfinite(v); };
    if (design.cols() != coefficients.size() || (!offset.empty() && offset.size() != design.rows()) ||
        !std::ranges::all_of(design.data(), finite) || !std::ranges::all_of(coefficients, finite) ||
        !std::ranges::all_of(offset, finite))
        throw std::invalid_argument("Tweedie regression: incompatible prediction inputs");
    std::vector<double> result(design.rows());
    for (std::size_t i = 0; i < design.rows(); ++i) {
        long double eta = offset.empty() ? 0 : offset[i];
        for (std::size_t j = 0; j < design.cols(); ++j)
            eta = std::fma(static_cast<long double>(design(i, j)), coefficients[j], eta);
        result[i] = output(std::exp(eta));
        if (!(result[i] > 0))
            throw std::overflow_error("Tweedie regression: predicted mean rounds to zero");
    }
    return result;
}

TweedieRegressionResult tweedie_regression(const linalg::DenseMatrix<double>& design,
    const std::vector<double>& response, const TweedieRegressionOptions& options) {
    const auto n = design.rows(), d = design.cols();
    const auto finite = [](double v) { return std::isfinite(v); };
    if (d == 0 || n <= d || response.size() != n ||
        !(options.minimum_power > 1 && options.minimum_power <= options.maximum_power && options.maximum_power < 2) ||
        (!options.offset.empty() && options.offset.size() != n) || (!options.weights.empty() && options.weights.size() != n) ||
        !std::ranges::all_of(design.data(), finite) || !std::ranges::all_of(options.offset, finite) ||
        !std::ranges::all_of(response, [&](double y) { return y >= 0 && finite(y); }) ||
        std::ranges::all_of(response, [](double y) { return y == 0; }) ||
        !std::ranges::all_of(options.weights, [&](double w) { return w > 0 && finite(w); }))
        throw std::invalid_argument("Tweedie regression: invalid response, design, or options");
    const auto limit = std::numeric_limits<std::size_t>::max();
    if (d > limit - 2 || d + 2 > limit / (d + 2))
        throw std::length_error("Tweedie regression: coefficient matrix exceeds capacity");
    Problem problem{Matrix(n, d), response, Vector(n), Vector(n), Vector(n, 1)};
    Vector scales(d);
    for (std::size_t j = 0; j < d; ++j) {
        for (std::size_t i = 0; i < n; ++i)
            scales[j] = std::max(scales[j], std::abs(static_cast<long double>(design(i, j))));
        if (scales[j] == 0)
            throw std::invalid_argument("Tweedie regression: zero design column");
        for (std::size_t i = 0; i < n; ++i)
            problem.x(i, j) = design(i, j) / scales[j];
    }
    const auto factor = linalg::svd(problem.x);
    if (!factor.converged || factor.rank() != d)
        throw std::invalid_argument("Tweedie regression: rank-deficient design");
    if (!options.offset.empty())
        std::ranges::copy(options.offset, problem.offset.begin());
    if (!options.weights.empty())
        std::ranges::copy(options.weights, problem.weights.begin());
    const auto weight_sum = std::accumulate(problem.weights.begin(), problem.weights.end(), 0.0L);
    if (!std::isfinite(weight_sum))
        throw std::overflow_error("Tweedie regression: weight sum exceeds the working range");
    long double mean = 0;
    for (std::size_t i = 0; i < n; ++i) {
        problem.weights[i] /= weight_sum;
        problem.log_response[i] = response[i] > 0 ? std::log(static_cast<long double>(response[i])) : 0;
        mean += problem.weights[i] * response[i];
    }
    Vector rough(n);
    for (std::size_t i = 0; i < n; ++i)
        rough[i] = std::log(response[i] + .1L * mean) - problem.offset[i];
    const auto start = factor.solve(rough);

    const bool fixed = options.minimum_power == options.maximum_power;
    const int intervals = fixed ? 0 : 32;
    std::vector<TweedieProfilePoint> profile;
    Profile best;
    /// @brief Preserves the best candidate and optionally records a power grid point.
    const auto consider = [&](double power, bool save) {
        auto trial = at_power(problem, start, power);
        const auto value = trial.value;
        if (save && std::isfinite(value))
            profile.push_back({power, output(std::exp(trial.log_phi)), output(weight_sum * value)});
        if (value > best.value)
            best = std::move(trial);
        return static_cast<double>(std::max(value, -static_cast<long double>(std::numeric_limits<double>::max())));
    };
    std::vector<double> grid, values;
    for (int i = 0; i <= intervals; ++i) {
        const auto phase = fixed ? 0. : std::numbers::pi * i / intervals;
        const auto fraction = .5 * (1 - std::cos(phase));
        grid.push_back(std::lerp(options.minimum_power, options.maximum_power, fraction));
        values.push_back(consider(grid.back(), true));
    }
    for (int i = 0; i <= intervals && !fixed; ++i) {
        const auto left = std::max(0, i - 1), right = std::min(intervals, i + 1);
        if (values[i] != -std::numeric_limits<double>::max() && values[i] >= values[left] &&
            values[i] >= values[right] && (values[i] > values[left] || values[i] > values[right]))
            algorithms::golden_section_search([&](double power) { return consider(power, false); },
                grid[left], grid[right], 1e-8, 80);
    }
    if (best.beta.empty())
        throw std::runtime_error("Tweedie regression: no usable finite profile point");

    const auto power_tolerance = std::min(1e-8, (options.maximum_power - options.minimum_power) / 16);
    const bool boundary = !fixed && std::min(best.power - options.minimum_power,
        options.maximum_power - best.power) <= power_tolerance;
    TweedieRegressionResult result{{}, output(std::exp(best.log_phi)), best.power, {}, output(weight_sum * best.value),
                                    {}, std::move(profile), Status::NumericalFailure};
    if (!(result.dispersion > 0))
        throw std::overflow_error("Tweedie regression: dispersion rounds to zero");
    if (best.bracketed) {
        result.covariance = covariance(problem, best, scales, weight_sum, !fixed && !boundary);
        if (result.covariance.rows() > 0)
            result.status = boundary ? Status::PowerBoundary : Status::Converged;
    }
    for (std::size_t j = 0; j < d; ++j)
        result.coefficients.push_back(output(best.beta[j] / scales[j]));
    result.fitted_mean = result.predict(design, options.offset);
    if (std::ranges::none_of(result.profile, [&](const auto& pt) { return pt.power == result.power; }))
        result.profile.push_back({result.power, result.dispersion, result.log_likelihood});
    std::ranges::sort(result.profile, {}, &TweedieProfilePoint::power);
    return result;
}

} // namespace datamunge::stats
