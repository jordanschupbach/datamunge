/// @file gls.cpp
/// @brief Whitening, least squares, and profiled structured covariance likelihoods.

#include <datamunge/stats/gls.hpp>

#include <datamunge/linalg/cholesky.hpp>
#include <datamunge/linalg/svd.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numbers>
#include <stdexcept>
#include <utility>

namespace datamunge::stats {

/// @brief Shared preconditioning and covariance profile arithmetic.
namespace {

/// @brief Dense working-precision matrix.
using Matrix = linalg::DenseMatrix<long double>;

/// @brief Independent blocks in within-label row order.
using Blocks = std::vector<std::vector<std::size_t>>;

/// @brief Design and response with bounded columns, plus scales needed for inference.
struct ScaledProblem {
    Matrix values;                          ///< Design followed by response.
    std::vector<long double> predictor_scale; ///< One scale per design column.
    long double response_scale;             ///< Response scale.
};

/// @brief Whitened least-squares solution in scaled coefficient units.
struct WhitenedFit {
    std::vector<long double> coefficients; ///< Coefficients on scaled design and response.
    Matrix inverse_information;            ///< Optional inverse information on scaled design.
    long double ss;                        ///< Whitened residual sum of squares for the scaled response.
};

/// @brief Narrows a finite working result.
double output(long double value) {
    if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<double>::max())
        throw std::overflow_error("GLS: result exceeds finite double range");
    return static_cast<double>(value);
}

/// @brief Scales columns of the augmented input matrix before whitening.
ScaledProblem scale_problem(const linalg::DenseMatrix<double>& design, const std::vector<double>& y) {
    const auto n = design.rows();
    const auto p = design.cols();
    const auto finite = [](double x) { return std::isfinite(x); };
    if (p == 0 || n < p || y.size() != n || !std::ranges::all_of(y, finite) ||
        !std::ranges::all_of(design.data(), finite))
        throw std::invalid_argument("GLS: invalid design or response");
    ScaledProblem problem{Matrix(n, p + 1), std::vector<long double>(p), 0};
    for (std::size_t i = 0; i < n; ++i) {
        problem.response_scale = std::max(problem.response_scale, std::abs(static_cast<long double>(y[i])));
        for (std::size_t j = 0; j < p; ++j)
            problem.predictor_scale[j] = std::max(problem.predictor_scale[j],
                std::abs(static_cast<long double>(design(i, j))));
    }
    if (std::ranges::find(problem.predictor_scale, 0.0L) != problem.predictor_scale.end())
        throw std::invalid_argument("GLS: design is rank deficient");
    if (problem.response_scale == 0)
        problem.response_scale = 1;
    for (std::size_t i = 0; i < n; ++i) {
        problem.values(i, p) = y[i] / problem.response_scale;
        for (std::size_t j = 0; j < p; ++j)
            problem.values(i, j) = design(i, j) / problem.predictor_scale[j];
    }
    return problem;
}

/// @brief Solves a bounded-column whitened problem and optionally forms inverse information.
WhitenedFit solve_whitened(const Matrix& augmented, bool covariance, bool original_design = false) {
    const auto n = augmented.rows();
    const auto p = augmented.cols() - 1;
    Matrix design(n, p);
    std::vector<long double> weights(p), response(n);
    for (std::size_t j = 0; j < p; ++j) {
        for (std::size_t i = 0; i < n; ++i)
            weights[j] = std::max(weights[j], std::abs(augmented(i, j)));
        if (!(weights[j] > 0) || !std::isfinite(weights[j]))
            throw std::runtime_error("GLS: invalid whitened design column");
        for (std::size_t i = 0; i < n; ++i)
            design(i, j) = augmented(i, j) / weights[j];
    }
    for (std::size_t i = 0; i < n; ++i)
        response[i] = augmented(i, p);
    const auto factor = linalg::svd(design);
    const auto cutoff = static_cast<long double>(std::max(n, p)) * std::numeric_limits<double>::epsilon();
    if (factor.rank(cutoff) != p) {
        if (original_design)
            throw std::invalid_argument("GLS: design is rank deficient");
        throw std::runtime_error("GLS: whitened design lost numerical rank");
    }
    if (!factor.converged)
        throw std::runtime_error("GLS: whitened design SVD reached its sweep limit");
    auto coefficients = factor.solve(response, cutoff);
    long double ss = 0;
    for (std::size_t i = 0; i < n; ++i) {
        long double residual = response[i];
        for (std::size_t j = 0; j < p; ++j)
            residual = std::fma(-design(i, j), coefficients[j], residual);
        ss += residual * residual;
    }
    for (std::size_t j = 0; j < p; ++j)
        coefficients[j] /= weights[j];
    Matrix inverse;
    if (covariance) {
        inverse = Matrix(p, p);
        for (std::size_t i = 0; i < p; ++i)
            for (std::size_t j = 0; j <= i; ++j) {
                for (std::size_t k = 0; k < p; ++k)
                    inverse(i, j) += (factor.V(i, k) / factor.s[k]) * (factor.V(j, k) / factor.s[k]);
                inverse(i, j) = inverse(j, i) = inverse(i, j) / weights[i] / weights[j];
            }
    }
    return {std::move(coefficients), std::move(inverse), ss};
}

/// @brief Returns ordinary-scale fits and model-based coefficient covariance.
GeneralizedLeastSquaresResult finish_fit(const ScaledProblem& problem, const WhitenedFit& fit,
                                        long double variance, long double log_likelihood) {
    const auto n = problem.values.rows();
    const auto p = fit.coefficients.size();
    GeneralizedLeastSquaresResult result{{}, linalg::DenseMatrix<double>(p, p), {}, {}, output(log_likelihood)};
    for (std::size_t j = 0; j < p; ++j) {
        result.coefficients.push_back(output(fit.coefficients[j] * problem.response_scale / problem.predictor_scale[j]));
        for (std::size_t k = 0; k <= j; ++k)
            result.covariance(k, j) = result.covariance(j, k) = output(fit.inverse_information(j, k) * variance
                / problem.predictor_scale[j] / problem.predictor_scale[k]);
    }
    for (std::size_t i = 0; i < n; ++i) {
        long double fitted = 0;
        for (std::size_t j = 0; j < p; ++j)
            fitted += problem.values(i, j) * fit.coefficients[j];
        result.fitted.push_back(output(fitted * problem.response_scale));
        result.residuals.push_back(output((problem.values(i, p) - fitted) * problem.response_scale));
    }
    return result;
}

/// @brief Groups rows by label, preserving order inside each independent block.
Blocks group_rows(std::size_t n, const std::vector<std::int64_t>& ids) {
    if (!ids.empty() && ids.size() != n)
        throw std::invalid_argument("GLS: group labels must match the response");
    std::map<std::int64_t, std::vector<std::size_t>> groups;
    for (std::size_t i = 0; i < n; ++i)
        groups[ids.empty() ? 0 : ids[i]].push_back(i);
    Blocks result;
    for (auto& [id, rows] : groups)
        result.push_back(std::move(rows));
    return result;
}

/// @brief Applies the structured inverse square root and returns the log determinant.
long double whiten_blocks(const Matrix& value, Matrix& whitened, const Blocks& blocks, GlsCorrelation structure,
                          long double rho) {
    long double logdet = 0;
    for (const auto& block : blocks) {
        if (block.size() == 1)
            continue;
        const auto count = static_cast<long double>(block.size() - 1);
        if (structure == GlsCorrelation::AR1) {
            const auto innovation_variance = (1 - rho) * (1 + rho);
            const auto innovation_scale = std::sqrt(innovation_variance);
            logdet += count * std::log(innovation_variance);
            for (std::size_t i = 1; i < block.size(); ++i)
                for (std::size_t j = 0; j < value.cols(); ++j)
                    whitened(block[i], j) = std::fma(-rho, value(block[i - 1], j), value(block[i], j)) / innovation_scale;
        } else {
            const auto mean_variance = std::fma(count, rho, 1.0L);
            logdet += count * std::log1p(-rho) + std::log(mean_variance);
            const auto mean_scale = std::sqrt(mean_variance);
            const auto contrast_scale = std::sqrt(1 - rho);
            for (std::size_t j = 0; j < value.cols(); ++j) {
                long double mean = 0;
                for (auto i : block)
                    mean += value(i, j);
                mean /= static_cast<long double>(block.size());
                for (auto i : block)
                    whitened(i, j) = (value(i, j) - mean) / contrast_scale + mean / mean_scale;
            }
        }
    }
    return logdet;
}

/// @brief Gaussian likelihood profiled over the marginal variance.
long double profile_likelihood(std::size_t n, long double ss, long double scale, long double logdet) {
    return -0.5L * (n * (std::log(2 * std::numbers::pi_v<long double>) + 1
                        + std::log(ss / n) + 2 * std::log(scale)) + logdet);
}

} // namespace

GeneralizedLeastSquaresResult generalized_least_squares(
    const linalg::DenseMatrix<double>& design, const std::vector<double>& response,
    const linalg::DenseMatrix<double>& residual_covariance) {
    const auto problem = scale_problem(design, response);
    const auto n = design.rows();
    if (residual_covariance.rows() != n || residual_covariance.cols() != n)
        throw std::invalid_argument("GLS: residual covariance must match the observation count");
    Matrix covariance(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j <= i; ++j) {
            const auto value = residual_covariance(i, j);
            if (!std::isfinite(value) || value != residual_covariance(j, i))
                throw std::invalid_argument("GLS: covariance must be finite and symmetric");
            covariance(i, j) = value;
        }
    const auto chol = linalg::cholesky(covariance);
    if (!chol.ok || !std::ranges::all_of(chol.L.data(), [](long double x) { return std::isfinite(x); }))
        throw std::invalid_argument("GLS: residual covariance must be positive definite");
    Matrix whitened = problem.values;
    long double logdet = 0;
    for (std::size_t i = 0; i < n; ++i) {
        logdet += 2 * std::log(chol.L(i, i));
        for (std::size_t j = 0; j < whitened.cols(); ++j) {
            for (std::size_t k = 0; k < i; ++k)
                whitened(i, j) -= chol.L(i, k) * whitened(k, j);
            whitened(i, j) /= chol.L(i, i);
        }
    }
    const auto fit = solve_whitened(whitened, true, true);
    const auto loglik = -0.5L * (n * std::log(2 * std::numbers::pi_v<long double>) + logdet
                               + fit.ss * problem.response_scale * problem.response_scale);
    return finish_fit(problem, fit, 1, loglik);
}

StructuredGlsResult structured_gls(const linalg::DenseMatrix<double>& design, const std::vector<double>& response,
                                   GlsCorrelation structure, const StructuredGlsOptions& options) {
    const auto n = design.rows();
    if (n <= design.cols() || (structure != GlsCorrelation::AR1 && structure != GlsCorrelation::CompoundSymmetry) ||
        !std::isfinite(options.tolerance) || options.tolerance <= 0 || options.max_iterations == 0 ||
        !(options.boundary_margin > 0 && options.boundary_margin < 0.5))
        throw std::invalid_argument("GLS: invalid structured covariance inputs or search controls");
    const auto problem = scale_problem(design, response);
    const auto blocks = group_rows(n, options.groups);
    const auto largest = std::ranges::max_element(blocks, {}, [](const auto& block) { return block.size(); })->size();
    if (largest < 2)
        throw std::domain_error("GLS: correlation requires repeated observations within a block");
    const auto iid = solve_whitened(problem.values, false, true);
    const long double precision = 16 * std::numeric_limits<double>::epsilon();
    if (!(iid.ss > n * precision * precision))
        throw std::domain_error("GLS: exact fit has no positive variance estimate");
    const long double lower = structure == GlsCorrelation::AR1 ? -1 : -1.0L / (largest - 1);
    const long double margin = (1 - lower) * options.boundary_margin;
    const long double lo = std::max(lower + margin,
        static_cast<long double>(std::nextafter(static_cast<double>(lower), 1.0)));
    const long double hi = std::min(1 - margin, static_cast<long double>(std::nextafter(1.0, 0.0)));
    Matrix whitened = problem.values;
    long double best = 0, best_ll = -std::numeric_limits<long double>::infinity();
    const auto evaluate = [&](long double rho) {
        const auto logdet = whiten_blocks(problem.values, whitened, blocks, structure, rho);
        const auto fit = solve_whitened(whitened, false);
        const auto value = profile_likelihood(n, fit.ss, problem.response_scale, logdet);
        if (value > best_ll) {
            best_ll = value;
            best = rho;
        }
        return value;
    };
    constexpr std::size_t intervals = 64;
    std::vector<long double> grid(intervals + 1), values(intervals + 1);
    for (std::size_t i = 0; i <= intervals; ++i) {
        const long double angle = std::numbers::pi_v<long double> * i / intervals;
        grid[i] = std::lerp(lo, hi, (1 - std::cos(angle)) / 2);
        values[i] = evaluate(grid[i]);
    }
    bool converged = true;
    const long double golden = (std::sqrt(5.0L) - 1) / 2;
    for (std::size_t i = 0; i <= intervals; ++i) {
        if ((i > 0 && values[i] < values[i - 1]) || (i < intervals && values[i] < values[i + 1]))
            continue;
        auto a = grid[i == 0 ? 0 : i - 1];
        auto b = grid[std::min(i + 1, intervals)];
        auto x = std::lerp(a, b, 1 - golden);
        auto y = std::lerp(a, b, golden);
        auto fx = evaluate(x), fy = evaluate(y);
        for (std::size_t j = 0; j < options.max_iterations && b - a > options.tolerance; ++j) {
            if (fx < fy) {
                a = x;
                x = y;
                fx = fy;
                y = std::lerp(a, b, golden);
                fy = evaluate(y);
            } else {
                b = y;
                y = x;
                fy = fx;
                x = std::lerp(a, b, 1 - golden);
                fx = evaluate(x);
            }
        }
        converged &= b - a <= options.tolerance;
    }
    const auto logdet = whiten_blocks(problem.values, whitened, blocks, structure, best);
    const auto fit = solve_whitened(whitened, true);
    const auto variance = fit.ss / n * problem.response_scale * problem.response_scale;
    const double reported_variance = output(variance);
    if (!(reported_variance > 0))
        throw std::overflow_error("GLS: positive residual variance falls below double range");
    const bool boundary = std::min(best - lo, hi - best) <= options.tolerance;
    return {finish_fit(problem, fit, variance, profile_likelihood(n, fit.ss, problem.response_scale, logdet)),
            output(best), reported_variance, converged, boundary};
}

} // namespace datamunge::stats
