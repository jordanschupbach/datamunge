/// @file beta_regression.cpp
/// @brief Fisher scoring for beta regression with independent mean and precision designs.

#include <datamunge/stats/beta_regression.hpp>

#include "likelihood_newton.hpp"
#include "../random/beta_kernels.hpp"
#include <datamunge/linalg/svd.hpp>

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>

namespace datamunge::stats {
/// @brief Working coordinates and expected information for beta maximum likelihood.
namespace {
/// @brief Extended-precision matrix used for scaled predictors and information.
using Matrix = linalg::DenseMatrix<long double>;
/// @brief Extended-precision vector used for working parameters.
using Vector = std::vector<long double>;

/// @brief Views a matrix row without copying its entries.
template <typename T>
std::span<const T> row(const linalg::DenseMatrix<T>& design, std::size_t i) {
    return std::span<const T>(design.data()).subspan(i * design.cols(), design.cols());
}

/// @brief Beta mean and its complement, computed independently in both tails.
struct Mean {
    long double value;      ///< Inverse-logit probability.
    long double complement; ///< One minus the inverse-logit probability.
};

/// @brief Beta response values and the logarithms needed for its density and score.
struct Response {
    long double value;          ///< Proportion on the open unit interval.
    long double log_value;      ///< Natural logarithm of the proportion.
    long double log_complement; ///< Natural logarithm of its complement.
};

/// @brief Weighted beta log likelihood, score, and lower triangle of Fisher information.
struct Evaluation {
    long double value; ///< Total log likelihood divided by total weight.
    Vector score;       ///< Score in the scaled mean-then-precision coefficient order.
    Matrix info;        ///< Expected information in the same order.
};

/// @brief Calculates the two complementary probabilities without cancellation.
Mean inverse_logit(long double eta) {
    const auto small = std::exp(-std::abs(eta));
    const auto scale = 1 + small;
    return eta >= 0 ? Mean{1 / scale, small / scale} : Mean{small / scale, 1 / scale};
}

/// @brief Computes digamma(x) minus log(x) for positive shapes.
long double digamma_remainder(long double x) {
    auto t = x;
    long double adjustment = 0;
    for (; t < 16; ++t)
        adjustment -= 1 / t;
    const auto u = 1 / t, u2 = u * u;
    const auto correction = -u / 2 - u2 * (1.0L / 12 - u2 * (1.0L / 120 -
                             u2 * (1.0L / 252 - u2 * (1.0L / 240 - u2 * (1.0L / 132)))));
    return t == x ? correction : adjustment + (std::log(t) - std::log(x)) + correction;
}

/// @brief Computes x squared times trigamma(x), minus x, retaining its large-shape limit.
long double trigamma_remainder(long double x) {
    auto t = x;
    long double adjustment = 0;
    for (; t < 16; ++t) {
        const auto ratio = x / t;
        adjustment += ratio * ratio;
    }
    const auto u = 1 / t, u2 = u * u;
    const auto ratio = x / t;
    const auto correction = .5L + u * (1.0L / 6 - u2 * (1.0L / 30 - u2 *
                                (1.0L / 42 - u2 * (1.0L / 30 - u2 * (5.0L / 66)))));
    return adjustment + ratio * (ratio * correction - (t - x));
}

/// @brief Converts a finite working value to double.
double to_double(long double value) {
    if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<double>::max())
        throw std::overflow_error("beta regression: result outside the finite double range");
    return static_cast<double>(value);
}

/// @brief Copies a design after scaling its columns by their largest absolute values.
Vector scale_design(const linalg::DenseMatrix<double>& design, Matrix& scaled) {
    Vector scales(design.cols());
    for (std::size_t j = 0; j < design.cols(); ++j) {
        for (std::size_t i = 0; i < design.rows(); ++i)
            scales[j] = std::max(scales[j], std::abs(static_cast<long double>(design(i, j))));
        if (scales[j] == 0)
            throw std::invalid_argument("beta regression: design contains a zero column");
        for (std::size_t i = 0; i < design.rows(); ++i)
            scaled(i, j) = design(i, j) / scales[j];
    }
    return scales;
}

/// @brief Joint weighted objective and Fisher information for both linear predictors.
Evaluation evaluate(const Matrix& x, const Matrix& z, const std::vector<Response>& responses,
                    const Vector& weights, const Vector& coefficients) {
    const auto p = x.cols(), q = z.cols(), d = p + q;
    const Evaluation invalid{-std::numeric_limits<long double>::infinity(), {}, {}};
    Evaluation result{0, Vector(d), Matrix(d, d)};
    for (std::size_t i = 0; i < responses.size(); ++i) {
        const auto xr = row(x, i), zr = row(z, i);
        const auto eta = std::inner_product(xr.begin(), xr.end(), coefficients.begin(), 0.0L);
        const auto log_phi = std::inner_product(zr.begin(), zr.end(), coefficients.begin() + p, 0.0L);
        const auto [mu, nu] = inverse_logit(eta);
        const auto phi = std::exp(log_phi);
        const auto a = phi * mu, b = phi * nu;
        if (!(a > 0 && b > 0) || phi > std::numeric_limits<double>::max() || !std::isfinite(phi))
            return invalid;
        const auto [y, log_y, log_ny] = responses[i];
        const auto weight = weights[i];
        const auto mean_scale = a * nu;
        const auto log_mu = -std::max(0.0L, -eta) - std::log1p(std::exp(-std::abs(eta)));
        const auto log_nu = log_mu - eta;
        const auto ca = digamma_remainder(a), cb = digamma_remainder(b), cp = digamma_remainder(phi);
        long double kernel, precision_score;
        if (a >= 16 && b >= 16) {
            const auto deviance = random::detail::beta_deviance(a, b, (1 - y) / y, std::fma(phi, y, -a));
            kernel = random::detail::beta_kernel_log(a, b, deviance);
            precision_score = -deviance + a * (cp - ca) + b * (cp - cb);
        } else {
            kernel = a * log_y + b * log_ny - random::detail::count_log_beta(a, b);
            precision_score = a * (log_y - log_mu + cp - ca) + b * (log_ny - log_nu + cp - cb);
        }
        const auto mean_score = mean_scale * (log_y - log_ny - eta + cb - ca);
        const auto ta = trigamma_remainder(a), tb = trigamma_remainder(b), tp = trigamma_remainder(phi);
        const auto xx = weight * (mean_scale + nu * nu * ta + mu * mu * tb);
        const auto xz = weight * (nu * ta - mu * tb);
        const auto zz = weight * (ta + tb - tp);
        if (!std::isfinite(kernel) || !std::isfinite(mean_score) || !std::isfinite(precision_score))
            return invalid;
        result.value += weight * (kernel - log_y - log_ny);
        for (std::size_t j = 0; j < p; ++j) {
            result.score[j] += weight * mean_score * xr[j];
            for (std::size_t k = 0; k <= j; ++k)
                result.info(j, k) += xx * xr[j] * xr[k];
        }
        for (std::size_t j = 0; j < q; ++j) {
            result.score[p + j] += weight * precision_score * zr[j];
            for (std::size_t k = 0; k < p; ++k)
                result.info(p + j, k) += xz * zr[j] * xr[k];
            for (std::size_t k = 0; k <= j; ++k)
                result.info(p + j, p + k) += zz * zr[j] * zr[k];
        }
    }
    return result;
}

} // namespace

BetaRegressionPrediction BetaRegressionResult::predict(const linalg::DenseMatrix<double>& mean_design,
    const linalg::DenseMatrix<double>& precision_design) const {
    const auto finite = [](double v) { return std::isfinite(v); };
    const auto n = mean_design.rows();
    if (mean_design.cols() != mean_coefficients.size() || precision_design.cols() != precision_coefficients.size() ||
        precision_design.rows() != n || !std::ranges::all_of(mean_design.data(), finite) ||
        !std::ranges::all_of(precision_design.data(), finite) || !std::ranges::all_of(mean_coefficients, finite) ||
        !std::ranges::all_of(precision_coefficients, finite))
        throw std::invalid_argument("beta regression: incompatible prediction inputs");
    BetaRegressionPrediction predictions{std::vector<double>(n), std::vector<double>(n)};
    for (std::size_t i = 0; i < n; ++i) {
        const auto mean_row = row(mean_design, i), precision_row = row(precision_design, i);
        /// @brief Widens a design-by-coefficient product.
        const auto product = [](double a, double b) { return static_cast<long double>(a) * b; };
        const auto eta = std::inner_product(mean_row.begin(), mean_row.end(), mean_coefficients.begin(),
                                            0.0L, std::plus<>{}, product);
        const auto log_phi = std::inner_product(precision_row.begin(), precision_row.end(),
            precision_coefficients.begin(), 0.0L, std::plus<>{}, product);
        const auto mean = to_double(inverse_logit(eta).value);
        const auto precision = to_double(std::exp(log_phi));
        if (!(mean > 0 && mean < 1 && precision > 0))
            throw std::overflow_error("beta regression: predicted parameter rounded to the domain boundary");
        predictions.mean[i] = mean;
        predictions.precision[i] = precision;
    }
    return predictions;
}

BetaRegressionResult beta_regression(const linalg::DenseMatrix<double>& mean_design,
    const linalg::DenseMatrix<double>& precision_design, const std::vector<double>& response,
    const BetaRegressionOptions& options) {
    const auto n = mean_design.rows(), p = mean_design.cols(), q = precision_design.cols();
    const auto finite = [](double v) { return std::isfinite(v); };
    if (p == 0 || q == 0 || n < std::max(p, q) || precision_design.rows() != n || response.size() != n ||
        (!options.weights.empty() && options.weights.size() != n) || options.max_iterations == 0 ||
        !(options.tolerance > 0 && options.tolerance < 1) || !std::ranges::all_of(mean_design.data(), finite) ||
        !std::ranges::all_of(precision_design.data(), finite) ||
        !std::ranges::all_of(response, [](double y) { return y > 0 && y < 1; }) ||
        !std::ranges::all_of(options.weights, [&](double w) { return w > 0 && finite(w); }))
        throw std::invalid_argument("beta regression: invalid design, response, weight or control");
    const auto limit = std::numeric_limits<std::size_t>::max();
    if (p > limit - q || p + q > limit / (p + q))
        throw std::length_error("beta regression: coefficient matrix exceeds capacity");
    Matrix x(n, p), z(n, q);
    const auto xs = scale_design(mean_design, x), zs = scale_design(precision_design, z);
    const auto xf = linalg::svd(x), zf = linalg::svd(z);
    if (!xf.converged || !zf.converged || xf.rank() != p || zf.rank() != q)
        throw std::invalid_argument("beta regression: rank-deficient design");

    Vector weights(n, 1);
    if (!options.weights.empty())
        std::ranges::copy(options.weights, weights.begin());
    const auto weight_sum = std::accumulate(weights.begin(), weights.end(), 0.0L);
    if (!std::isfinite(weight_sum))
        throw std::overflow_error("beta regression: weight sum exceeds working range");
    std::vector<Response> data;
    data.reserve(n);
    Vector logits(n);
    for (std::size_t i = 0; i < n; ++i) {
        const long double y = response[i];
        data.push_back({y, std::log(y), std::log1p(-y)});
        logits[i] = data.back().log_value - data.back().log_complement;
        weights[i] /= weight_sum;
    }
    auto start = xf.solve(logits);
    const auto fitted_logits = x * start;
    const auto precision_intercept = zf.solve(Vector(n, 1));
    const auto precision_direction = z * precision_intercept;
    long double max_error = 0, max_logit = 1, relative_variance = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const auto [mu, nu] = inverse_logit(fitted_logits[i]);
        max_error = std::max(max_error, std::abs(logits[i] - fitted_logits[i]));
        max_logit = std::max(max_logit, std::abs(logits[i]));
        const auto difference = data[i].value - mu;
        relative_variance += weights[i] * difference * difference / (mu * nu);
    }
    if (max_error <= 64 * std::numeric_limits<long double>::epsilon() * max_logit &&
        std::ranges::all_of(precision_direction, [](long double value) { return value > 0; }))
        throw std::invalid_argument("beta regression: a perfect logit fit with unbounded precision");
    const auto initial_precision = std::log(std::clamp(1 / relative_variance - 1, 1.0L, 10000.0L));
    for (const auto b : precision_intercept)
        start.push_back(b * initial_precision);

    /// @brief Evaluates likelihood, score, and expected information.
    const auto likelihood = [&](const Vector& beta) { return evaluate(x, z, data, weights, beta); };
    /// @brief Measures the largest change across the mean and precision predictors.
    const auto effect = [&](const Vector& step) {
        long double greatest = 0;
        for (std::size_t i = 0; i < n; ++i) {
            const auto xr = row(x, i), zr = row(z, i);
            greatest = std::max(greatest, std::abs(std::inner_product(xr.begin(), xr.end(), step.begin(), 0.0L)));
            greatest = std::max(greatest, std::abs(std::inner_product(zr.begin(), zr.end(), step.begin() + p, 0.0L)));
        }
        return detail::LikelihoodStep{greatest, false};
    };
    using Status = BetaRegressionStatus;
    const auto fit = detail::maximize_likelihood<Status>(start, options.max_iterations, options.tolerance,
                                                        likelihood, effect);
    const auto& coefficients = fit ? fit->beta : start;
    auto scales = xs;
    scales.insert(scales.end(), zs.begin(), zs.end());
    BetaRegressionResult result{{}, {}, {}, {}, to_double(weight_sum * (fit ? fit->value : likelihood(start).value)),
                                fit ? fit->iterations : 0, fit ? fit->status : Status::NumericalFailure};
    for (std::size_t j = 0; j < p; ++j)
        result.mean_coefficients.push_back(to_double(coefficients[j] / scales[j]));
    for (std::size_t j = 0; j < q; ++j)
        result.precision_coefficients.push_back(to_double(coefficients[p + j] / scales[p + j]));
    if (result.status == Status::Converged) {
        const auto d = p + q;
        const auto inverse = fit->factor.solve(Matrix::identity(d));
        result.covariance = linalg::DenseMatrix<double>(d, d);
        for (std::size_t i = 0; i < d; ++i)
            for (std::size_t j = 0; j <= i; ++j)
                result.covariance(i, j) = result.covariance(j, i) =
                    to_double(inverse(i, j) / scales[i] / scales[j] / weight_sum);
    }
    result.fitted = result.predict(mean_design, precision_design);
    return result;
}

} // namespace datamunge::stats
