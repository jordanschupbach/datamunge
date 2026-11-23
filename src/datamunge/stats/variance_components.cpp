/// @file variance_components.cpp
/// @brief Dense marginal Gaussian ML and REML for several independent variance components.

#include <datamunge/stats/variance_components.hpp>

#include "likelihood_bfgs.hpp"
#include <datamunge/linalg/cholesky.hpp>
#include <datamunge/linalg/svd.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace datamunge::stats {
/// @brief Scaled marginal likelihood and nonnegative variance-ratio optimization.
namespace {
/// @brief Extended precision matrix.
using Matrix = linalg::DenseMatrix<long double>;
/// @brief Extended precision vector.
using Vector = std::vector<long double>;
/// @brief Cholesky factor for marginal covariance and fixed-effect information.
using Factor = linalg::CholeskyDecomposition<long double>;
/// @brief Variance optimization status.
using Status = VarianceComponentsStatus;

/// @brief Scaled fixed design, response, and normalized covariance components.
struct Problem {
    Matrix x;                    ///< Column-scaled fixed design.
    Vector y;                    ///< Scaled observations.
    std::vector<Matrix> kernels; ///< Random covariance contributions, each with mean diagonal one.
    bool reml;                   ///< Includes the fixed-design determinant and uses residual degrees of freedom.
};

/// @brief Conditional fixed-effect fit and score at one choice of variance ratios.
struct Profile {
    long double value{-std::numeric_limits<long double>::infinity()}; ///< Profile log likelihood per row.
    Vector score;            ///< Derivative in the scaled log1p-ratio coordinates.
    Vector beta;             ///< Working fixed-effect estimates.
    Vector precision_error;  ///< Shape covariance inverse times the fitted residual.
    Factor fixed_factor;     ///< Factor of the unscaled fixed-effect precision.
    long double residual{0}; ///< Working residual variance.
};

/// @brief Best finite iterate from one deterministic starting point.
struct Candidate {
    Vector parameters; ///< Nonnegative log1p of the normalized variance ratios.
    Profile profile;   ///< Conditional likelihood and estimates.
    Status status;     ///< Numerical stopping reason.
};

/// @brief Converts a working result to finite double precision.
double output(long double x) {
    if (!std::isfinite(x) || std::abs(x) > std::numeric_limits<double>::max())
        throw std::overflow_error("variance components: result exceeds the finite reporting range");
    return static_cast<double>(x);
}

/// @brief Inner product in working coordinates.
long double dot(const Vector& a, const Vector& b) {
    return std::inner_product(a.begin(), a.end(), b.begin(), 0.0L);
}

/// @brief Forms Z times its transpose while keeping exact symmetry in storage.
Matrix kernel(const Matrix& z) {
    Matrix result(z.rows(), z.rows());
    for (std::size_t i = 0; i < z.rows(); ++i)
        for (std::size_t j = 0; j <= i; ++j) {
            long double value = 0;
            for (std::size_t k = 0; k < z.cols(); ++k)
                value += z(i, k) * z(j, k);
            result(i, j) = result(j, i) = value;
        }
    return result;
}

/// @brief Log determinant from a positive diagonal with a safeguard against lost pivots.
long double log_determinant(const Factor& factor) {
    if (!factor.ok)
        return -std::numeric_limits<long double>::infinity();
    long double result = 0, smallest = std::numeric_limits<long double>::infinity(), largest = 0;
    for (std::size_t j = 0; j < factor.L.rows(); ++j) {
        const auto d = factor.L(j, j);
        if (!(d > 0) || !std::isfinite(d))
            return -std::numeric_limits<long double>::infinity();
        smallest = std::min(smallest, d);
        largest = std::max(largest, d);
        result += 2 * std::log(d);
    }
    return smallest > largest * 8 * std::sqrt(std::numeric_limits<long double>::epsilon()) ?
        result : -std::numeric_limits<long double>::infinity();
}

/// @brief Rejects linearly dependent covariance contributions on the relevant observation space.
void check_identifiable(const Problem& problem, const Matrix& orthogonal_x, const std::vector<Matrix>& random) {
    const auto n = problem.x.rows();
    std::vector<Matrix> projected;
    if (problem.reml) {
        const auto xt = orthogonal_x.transpose();
        projected.push_back(Matrix::identity(n) - orthogonal_x * xt);
        for (const auto& z : random)
            projected.push_back(kernel(z - orthogonal_x * (xt * z)));
    } else {
        projected.push_back(Matrix::identity(n));
    }
    const auto d = random.size() + 1;
    /// @brief Returns a covariance contribution in the full or restricted space.
    const auto at = [&](std::size_t j) -> const Matrix& {
        return problem.reml || j == 0 ? projected[j] : problem.kernels[j - 1];
    };
    Vector norm(d);
    Matrix gram(d, d);
    for (std::size_t i = 0; i < d; ++i) {
        norm[i] = std::sqrt(dot(at(i).data(), at(i).data()));
        if (!(norm[i] > 1e-12L * std::sqrt(static_cast<long double>(n))))
            throw std::invalid_argument("variance components: a block has no identifiable variance");
        gram(i, i) = 1;
        for (std::size_t j = 0; j < i; ++j)
            gram(i, j) = dot(at(i).data(), at(j).data()) / norm[i] / norm[j];
    }
    if (!std::isfinite(log_determinant(linalg::cholesky(gram))))
        throw std::invalid_argument("variance components: dependent covariance blocks");
}

/// @brief Computes the profiled criterion and its exact variance-ratio score.
Profile evaluate(const Problem& problem, const Vector& parameters) {
    const auto n = problem.x.rows(), p = problem.x.cols();
    Matrix shape = Matrix::identity(n);
    for (std::size_t k = 0; k < parameters.size(); ++k) {
        const auto ratio = std::expm1(parameters[k]);
        if (!std::isfinite(ratio))
            return {};
        for (std::size_t j = 0; j < n; ++j)
            for (std::size_t i = 0; i <= j; ++i)
                shape(j, i) += ratio * problem.kernels[k](j, i);
    }
    const auto factor = linalg::cholesky(shape);
    const auto log_shape = log_determinant(factor);
    if (!std::isfinite(log_shape))
        return {};
    const auto precision_x = factor.solve(problem.x);
    const auto xt = problem.x.transpose();
    auto fixed_factor = linalg::cholesky(xt * precision_x);
    const auto log_fixed = log_determinant(fixed_factor);
    if (!std::isfinite(log_fixed))
        return {};
    const auto beta = fixed_factor.solve(xt * factor.solve(problem.y));
    const auto fitted = problem.x * beta;
    auto error = problem.y;
    for (std::size_t i = 0; i < n; ++i)
        error[i] -= fitted[i];
    auto precision_error = factor.solve(error);
    const auto degrees = problem.reml ? n - p : n;
    const auto residual = dot(error, precision_error) / degrees;
    if (!(residual > 0) || !std::isfinite(residual))
        return {};
    const auto value = -.5L * (degrees * (1 + std::log(2 * std::numbers::pi_v<long double> * residual)) +
        log_shape + (problem.reml ? log_fixed : 0)) / n;
    auto precision = factor.solve(Matrix::identity(n));
    if (problem.reml)
        precision = precision - precision_x * fixed_factor.solve(precision_x.transpose());

    Vector score(parameters.size());
    for (std::size_t j = 0; j < parameters.size(); ++j) {
        const auto& k = problem.kernels[j];
        const auto explained = dot(precision_error, k * precision_error) / residual;
        const auto trace = dot(precision.data(), k.data());
        score[j] = .5L * (explained - trace) * std::exp(parameters[j]) / n;
    }
    if (!std::isfinite(value) ||
        !std::ranges::all_of(score, [](long double x) { return std::isfinite(x); }))
        return {};
    return {value, std::move(score), beta, std::move(precision_error), std::move(fixed_factor), residual};
}

/// @brief Projects the score by suppressing outward components at the nonnegative boundary.
Vector projected_score(const Vector& parameters, const Vector& score) {
    auto result = score;
    for (std::size_t j = 0; j < score.size(); ++j)
        if (parameters[j] == 0 && score[j] < 0)
            result[j] = 0;
    return result;
}

/// @brief Maximum component magnitude used for the projected-score stopping criterion.
long double norm(const Vector& vector) {
    long double maximum = 0;
    for (const auto x : vector)
        maximum = std::max(maximum, std::abs(x));
    return maximum;
}

/// @brief Maximizes one start using BFGS with a feasible projected Armijo search.
Candidate maximize(const Problem& problem, Vector parameters, const VarianceComponentsOptions& options) {
    const auto d = parameters.size();
    auto metric = Matrix::identity(d);
    auto profile = evaluate(problem, parameters);
    if (profile.score.empty())
        return {std::move(parameters), std::move(profile), Status::NumericalFailure};
    for (std::size_t iter = 0;; ++iter) {
        const auto projected = projected_score(parameters, profile.score);
        if (norm(projected) <= options.tolerance)
            return {std::move(parameters), std::move(profile), Status::Converged};
        if (iter == options.max_iterations)
            return {std::move(parameters), std::move(profile), Status::IterationLimit};
        bool accepted = false;
        for (int restart = 0; restart < 2 && !accepted; ++restart) {
            if (restart)
                metric = Matrix::identity(d);
            auto direction = metric * projected;
            for (std::size_t j = 0; j < d; ++j)
                if (parameters[j] == 0 && profile.score[j] <= 0)
                    direction[j] = 0;
            const auto maximum_step = norm(direction);
            if (!(maximum_step > 0) || !std::isfinite(maximum_step))
                continue;
            auto step = std::min(1.0L, 3 / maximum_step);
            const auto rounding = 64 * std::numeric_limits<long double>::epsilon() *
                std::max(1.0L, std::abs(profile.value));
            for (int attempt = 0; attempt < 72; ++attempt, step *= .5L) {
                auto trial = parameters;
                long double predicted = 0;
                for (std::size_t j = 0; j < d; ++j) {
                    trial[j] = std::max(0.0L, parameters[j] + step * direction[j]);
                    predicted += profile.score[j] * (trial[j] - parameters[j]);
                }
                if (trial == parameters)
                    break;
                if (!(predicted > 0))
                    continue;
                auto next = evaluate(problem, trial);
                if (next.score.empty() || next.value - profile.value < 1e-4L * predicted - rounding)
                    continue;
                if (predicted <= rounding && norm(projected_score(trial, next.score)) >= norm(projected))
                    continue;
                detail::update_likelihood_metric(metric, parameters, trial, profile.score, next.score);
                parameters = std::move(trial);
                profile = std::move(next);
                accepted = true;
                break;
            }
        }
        if (!accepted)
            return {std::move(parameters), std::move(profile), Status::NumericalFailure};
    }
}

} // namespace

std::vector<double> VarianceComponentsFit::predict(const linalg::DenseMatrix<double>& fixed_design,
    const std::vector<linalg::DenseMatrix<double>>& random_designs) const {
    const auto n = fixed_design.rows();
    const auto finite = [](double x) { return std::isfinite(x); };
    if (fixed_design.cols() != coefficients.size() ||
        !std::ranges::all_of(fixed_design.data(), finite) || !std::ranges::all_of(coefficients, finite) ||
        (!random_designs.empty() && random_designs.size() != effects.size()))
        throw std::invalid_argument("variance components: incompatible prediction design");
    for (std::size_t b = 0; b < random_designs.size(); ++b)
        if (random_designs[b].rows() != n || random_designs[b].cols() != effects[b].size() ||
            !std::ranges::all_of(random_designs[b].data(), finite) || !std::ranges::all_of(effects[b], finite))
            throw std::invalid_argument("variance components: incompatible random prediction design");
    std::vector<double> result(n);
    for (std::size_t i = 0; i < n; ++i) {
        long double value = 0;
        for (std::size_t j = 0; j < coefficients.size(); ++j)
            value += static_cast<long double>(fixed_design(i, j)) * coefficients[j];
        for (std::size_t b = 0; b < random_designs.size(); ++b)
            for (std::size_t j = 0; j < effects[b].size(); ++j)
                value += static_cast<long double>(random_designs[b](i, j)) * effects[b][j];
        result[i] = output(value);
    }
    return result;
}

VarianceComponentsFit fit_variance_components(const linalg::DenseMatrix<double>& fixed_design,
    const std::vector<double>& response, const std::vector<linalg::DenseMatrix<double>>& random_designs,
    const VarianceComponentsOptions& options) {
    const auto n = fixed_design.rows(), p = fixed_design.cols();
    const auto finite = [](double x) { return std::isfinite(x); };
    if (n <= p || p == 0 || response.size() != n || random_designs.empty() ||
        !(options.tolerance > 0 && options.tolerance < 1) || options.max_iterations == 0 ||
        !std::ranges::all_of(fixed_design.data(), finite) || !std::ranges::all_of(response, finite))
        throw std::invalid_argument("variance components: invalid fixed design, response or controls");
    const auto capacity = Vector{}.max_size();
    const auto blocks = random_designs.size() + 1;
    if (n > capacity / n || blocks > capacity / blocks)
        throw std::length_error("variance components: dense covariance or information exceeds storage capacity");
    Problem problem{Matrix(n, p), Vector(n), {}, options.reml};
    Vector x_scale(p);
    long double y_scale = 0;
    for (const auto y : response)
        y_scale = std::max(y_scale, std::abs(static_cast<long double>(y)));
    if (y_scale == 0)
        throw std::invalid_argument("variance components: no residual variation");
    for (std::size_t i = 0; i < n; ++i)
        problem.y[i] = response[i] / y_scale;
    for (std::size_t j = 0; j < p; ++j) {
        for (std::size_t i = 0; i < n; ++i)
            x_scale[j] = std::max(x_scale[j], std::abs(static_cast<long double>(fixed_design(i, j))));
        if (x_scale[j] == 0)
            throw std::invalid_argument("variance components: zero fixed design column");
        for (std::size_t i = 0; i < n; ++i)
            problem.x(i, j) = fixed_design(i, j) / x_scale[j];
    }
    const auto svd = linalg::svd(problem.x);
    if (!svd.converged || svd.rank() != p)
        throw std::invalid_argument("variance components: aliased fixed design");
    const auto fitted = problem.x * svd.solve(problem.y);
    long double error = 0;
    for (std::size_t i = 0; i < n; ++i)
        error = std::max(error, std::abs(problem.y[i] - fitted[i]));
    if (error <= 64 * std::numeric_limits<long double>::epsilon())
        throw std::invalid_argument("variance components: no residual variation");

    std::vector<Matrix> random;
    Vector z_scale;
    for (const auto& input : random_designs) {
        if (input.rows() != n || input.cols() == 0 || !std::ranges::all_of(input.data(), finite))
            throw std::invalid_argument("variance components: invalid random design");
        long double rms = 0;
        for (const auto x : input.data())
            rms = std::hypot(rms, static_cast<long double>(x));
        rms /= std::sqrt(static_cast<long double>(n));
        if (!(rms > 0) || !std::isfinite(rms))
            throw std::invalid_argument("variance components: unusable random design scale");
        auto& z = random.emplace_back(n, input.cols());
        for (std::size_t i = 0; i < input.data().size(); ++i)
            z.data()[i] = input.data()[i] / rms;
        z_scale.push_back(rms);
        problem.kernels.push_back(kernel(z));
    }
    check_identifiable(problem, svd.U, random);

    Candidate best{{}, {}, Status::NumericalFailure};
    for (const long double start : {0.L, std::log(2.L), std::log(11.L)}) {
        auto trial = maximize(problem, Vector(random.size(), start), options);
        const auto difference = trial.profile.value - best.profile.value;
        if (difference > 1e-13L || (std::abs(difference) <= 1e-13L &&
                trial.status == Status::Converged && best.status != Status::Converged))
            best = std::move(trial);
    }
    if (best.parameters.empty())
        throw std::runtime_error("variance components: no usable likelihood fit");
    const auto& profile = best.profile;
    const auto error_variance = profile.residual * y_scale * y_scale;
    auto log_likelihood = n * profile.value - (options.reml ? n - p : n) * std::log(y_scale);
    if (options.reml)
        for (const auto scale : x_scale)
            log_likelihood -= std::log(scale);
    VarianceComponentsFit result{{}, linalg::DenseMatrix<double>(p, p), {}, output(error_variance),
                                  {}, {}, output(log_likelihood), best.status, options.reml};
    if (!(result.residual_variance > 0))
        throw std::overflow_error("variance components: residual variance rounds to zero");
    for (std::size_t j = 0; j < p; ++j)
        result.coefficients.push_back(output(profile.beta[j] * (y_scale / x_scale[j])));
    const auto covariance = profile.fixed_factor.solve(Matrix::identity(p));
    for (std::size_t i = 0; i < p; ++i)
        for (std::size_t j = 0; j <= i; ++j)
            result.covariance(i, j) = result.covariance(j, i) =
                output(covariance(i, j) * profile.residual * (y_scale / x_scale[i]) * (y_scale / x_scale[j]));
    for (std::size_t j = 0; j < random.size(); ++j) {
        const auto ratio = std::expm1(best.parameters[j]);
        const auto scale = y_scale / z_scale[j];
        result.variances.push_back(output(profile.residual * ratio * scale * scale));
        auto& effects = result.effects.emplace_back();
        for (const auto value : random[j].transpose() * profile.precision_error)
            effects.push_back(output(ratio * scale * value));
        if (ratio == 0 && result.status == Status::Converged)
            result.status = Status::Boundary;
    }
    result.fitted = result.predict(fixed_design, random_designs);
    return result;
}

} // namespace datamunge::stats
