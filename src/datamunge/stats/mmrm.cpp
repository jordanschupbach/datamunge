/// @file mmrm.cpp
/// @brief Profile maximum likelihood for Gaussian outcomes observed at a subset of visits.

#include <datamunge/stats/mmrm.hpp>

#include "likelihood_bfgs.hpp"
#include <datamunge/linalg/cholesky.hpp>
#include <datamunge/linalg/svd.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numbers>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace datamunge::stats {
/// @brief Observation patterns and working coordinates for the Gaussian likelihood.
namespace {

/// @brief Extended-precision likelihood matrix.
using Matrix = linalg::DenseMatrix<long double>;
/// @brief Extended-precision likelihood vector.
using Vector = std::vector<long double>;
/// @brief Subject rows or positions in the full visit covariance.
using Indices = std::vector<std::size_t>;
/// @brief Symmetric positive-definite matrix factorization.
using Factor = linalg::CholeskyDecomposition<long double>;

/// @brief Observations for a single subject in ascending visit order.
struct Subject {
    Matrix design;   ///< Fixed predictors after column and visit scaling.
    Vector response; ///< Observed outcomes after scaling.
};

/// @brief Subjects sharing exactly the same observed visits.
struct Pattern {
    Indices visits;                ///< Positions in the full visit covariance.
    std::vector<Subject> subjects; ///< Data blocks sharing one covariance factor.
};

/// @brief Gaussian likelihood in rescaled response and predictor coordinates.
struct Problem {
    std::vector<Pattern> patterns; ///< Observed visit patterns.
    std::size_t observations;     ///< Total number of outcomes.
    std::size_t predictors;       ///< Fixed design column count.
    std::size_t visits;           ///< Total number of visit levels.
    bool reml;                    ///< Include the fixed information determinant.
};

/// @brief Likelihood and conditional mean estimates for one covariance.
struct Profile {
    long double value{-std::numeric_limits<long double>::infinity()}; ///< Log likelihood per observed row.
    Vector score;            ///< Gradient in log-Cholesky parameters, empty on numerical failure.
    Vector coefficients;     ///< GLS estimates in the working coordinate system.
    Matrix fixed_covariance; ///< Inverse fixed-effect information.
    Matrix visit_covariance; ///< Full covariance on the working response scale.
};

/// @brief Returns ascending unique visit labels.
Indices visit_levels(Indices levels) {
    std::ranges::sort(levels);
    levels.erase(std::unique(levels.begin(), levels.end()), levels.end());
    return levels;
}

/// @brief Maps each label to its position in the sorted visit levels.
Indices visit_positions(const Indices& visit, const Indices& levels) {
    Indices positions;
    positions.reserve(visit.size());
    for (const auto label : visit)
        positions.push_back(std::ranges::lower_bound(levels, label) - levels.begin());
    return positions;
}

/// @brief Converts the estimate back to finite double precision.
double output(long double value) {
    if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<double>::max())
        throw std::overflow_error("mmrm: result exceeds the finite reporting range");
    return static_cast<double>(value);
}

/// @brief Inner product of working vectors of equal length.
long double dot(const Vector& a, const Vector& b) {
    return std::inner_product(a.begin(), a.end(), b.begin(), 0.0L);
}

/// @brief Log determinant from a positive Cholesky factor with numerically resolved pivots.
long double log_determinant(const Factor& factor) {
    if (!factor.ok)
        return -std::numeric_limits<long double>::infinity();
    long double result = 0, smallest = std::numeric_limits<long double>::infinity(), largest = 0;
    for (std::size_t i = 0; i < factor.L.rows(); ++i) {
        const auto d = factor.L(i, i);
        if (!(d > 0) || !std::isfinite(d))
            return -std::numeric_limits<long double>::infinity();
        smallest = std::min(smallest, d);
        largest = std::max(largest, d);
        result += 2 * std::log(d);
    }
    return smallest > largest * 8 * std::sqrt(std::numeric_limits<long double>::epsilon()) ?
        result : -std::numeric_limits<long double>::infinity();
}

/// @brief Evaluates the observed-data Gaussian likelihood and its analytic covariance score.
Profile evaluate(const Problem& problem, const Vector& parameters) {
    const auto [n, p, q] = std::tuple{problem.observations, problem.predictors, problem.visits};
    Matrix lower(q, q);
    std::size_t parameter = 0;
    for (std::size_t i = 0; i < q; ++i) {
        for (std::size_t j = 0; j < i; ++j)
            lower(i, j) = parameters[parameter++];
        lower(i, i) = std::exp(parameters[parameter++]);
    }
    Profile result;
    result.visit_covariance = lower * lower.transpose();
    if (!std::isfinite(log_determinant(linalg::cholesky(result.visit_covariance))))
        return {};

    /// @brief Covariance inverse and its actions on each observed design block.
    struct Precision {
        Matrix inverse;        ///< Shared inverse for the observed visits.
        std::vector<Matrix> x; ///< Inverse covariance times each subject's design.
    };
    std::vector<Precision> precision;
    precision.reserve(problem.patterns.size());
    Matrix gram(p, p);
    Vector rhs(p);
    long double determinant = 0;
    for (const auto& pattern : problem.patterns) {
        const auto m = pattern.visits.size();
        Matrix covariance(m, m);
        for (std::size_t i = 0; i < m; ++i)
            for (std::size_t j = 0; j < m; ++j)
                covariance(i, j) = result.visit_covariance(pattern.visits[i], pattern.visits[j]);
        const auto factor = linalg::cholesky(covariance);
        const auto log_det = log_determinant(factor);
        if (!std::isfinite(log_det))
            return {};
        determinant += pattern.subjects.size() * log_det;
        auto& block = precision.emplace_back(Precision{factor.solve(Matrix::identity(m)), {}});
        block.x.reserve(pattern.subjects.size());
        for (const auto& subject : pattern.subjects) {
            const auto& weighted_x = block.x.emplace_back(block.inverse * subject.design);
            for (std::size_t j = 0; j < p; ++j) {
                for (std::size_t i = 0; i < m; ++i)
                    rhs[j] += weighted_x(i, j) * subject.response[i];
                for (std::size_t k = 0; k <= j; ++k)
                    for (std::size_t i = 0; i < m; ++i)
                        gram(j, k) += subject.design(i, j) * weighted_x(i, k);
            }
        }
    }
    for (std::size_t j = 0; j < p; ++j)
        for (std::size_t k = 0; k < j; ++k)
            gram(k, j) = gram(j, k);
    const auto fixed_factor = linalg::cholesky(gram);
    const auto fixed_determinant = log_determinant(fixed_factor);
    if (!std::isfinite(fixed_determinant))
        return {};
    result.coefficients = fixed_factor.solve(rhs);
    result.fixed_covariance = fixed_factor.solve(Matrix::identity(p));

    Matrix derivative(q, q);
    long double quadratic = 0;
    for (std::size_t b = 0; b < problem.patterns.size(); ++b) {
        const auto& pattern = problem.patterns[b];
        const auto& block = precision[b];
        const auto m = pattern.visits.size();
        for (std::size_t i = 0; i < m; ++i)
            for (std::size_t j = 0; j < m; ++j)
                derivative(pattern.visits[i], pattern.visits[j]) -= pattern.subjects.size() * block.inverse(i, j);
        for (std::size_t g = 0; g < pattern.subjects.size(); ++g) {
            const auto& subject = pattern.subjects[g];
            const auto fitted = subject.design * result.coefficients;
            auto error = subject.response;
            for (std::size_t i = 0; i < m; ++i)
                error[i] -= fitted[i];
            const auto weighted_error = block.inverse * error;
            quadratic += dot(error, weighted_error);
            for (std::size_t i = 0; i < m; ++i)
                for (std::size_t j = 0; j < m; ++j)
                    derivative(pattern.visits[i], pattern.visits[j]) += weighted_error[i] * weighted_error[j];
            if (problem.reml) {
                const auto adjustment = (block.x[g] * result.fixed_covariance) * block.x[g].transpose();
                for (std::size_t i = 0; i < m; ++i)
                    for (std::size_t j = 0; j < m; ++j)
                        derivative(pattern.visits[i], pattern.visits[j]) += adjustment(i, j);
            }
        }
    }
    const auto df = problem.reml ? n - p : n;
    result.value = -.5L * (determinant + quadratic + df * std::log(2 * std::numbers::pi_v<long double>) +
                          (problem.reml ? fixed_determinant : 0)) / n;
    const auto lower_score = derivative * lower;
    result.score.reserve(parameters.size());
    for (std::size_t i = 0; i < q; ++i) {
        for (std::size_t j = 0; j < i; ++j)
            result.score.push_back(lower_score(i, j) / n);
        result.score.push_back(lower_score(i, i) * lower(i, i) / n);
    }
    if (!std::isfinite(result.value) || !std::ranges::all_of(result.score, [](auto x) { return std::isfinite(x); }))
        return {};
    return result;
}

/// @brief Packs a positive-definite residual correlation as log-Cholesky coordinates.
Vector initial_correlation(Matrix correlation, std::size_t parameters) {
    // Pairwise residual correlations need not be jointly positive definite.
    auto factor = linalg::cholesky(correlation);
    while (!std::isfinite(log_determinant(factor))) {
        for (std::size_t i = 0; i < correlation.rows(); ++i)
            for (std::size_t j = 0; j < i; ++j) {
                correlation(i, j) *= .5L;
                correlation(j, i) = correlation(i, j);
            }
        factor = linalg::cholesky(correlation);
    }
    Vector result;
    result.reserve(parameters);
    for (std::size_t i = 0; i < correlation.rows(); ++i) {
        for (std::size_t j = 0; j < i; ++j)
            result.push_back(factor.L(i, j));
        result.push_back(std::log(factor.L(i, i)));
    }
    return result;
}

} // namespace

std::vector<double> MMRMFit::predict(const linalg::DenseMatrix<double>& design) const {
    const auto finite = [](double x) { return std::isfinite(x); };
    if (design.cols() != coefficients.size() || !std::ranges::all_of(design.data(), finite) ||
        !std::ranges::all_of(coefficients, finite))
        throw std::invalid_argument("mmrm: incompatible prediction design");
    std::vector<double> predictions(design.rows());
    for (std::size_t i = 0; i < design.rows(); ++i) {
        long double mean = 0;
        for (std::size_t j = 0; j < design.cols(); ++j)
            mean += static_cast<long double>(design(i, j)) * coefficients[j];
        predictions[i] = output(mean);
    }
    return predictions;
}

MMRMFit fit_mmrm(const linalg::DenseMatrix<double>& design, const std::vector<double>& response,
    const std::vector<std::size_t>& subject, const std::vector<std::size_t>& visit, const MMRMOptions& options) {
    const auto n = design.rows(), p = design.cols();
    const auto finite = [](double x) { return std::isfinite(x); };
    if (n <= p || p == 0 || response.size() != n || subject.size() != n || visit.size() != n ||
        !(options.tolerance > 0 && options.tolerance < 1) || options.max_iterations == 0 ||
        !std::ranges::all_of(design.data(), finite) || !std::ranges::all_of(response, finite))
        throw std::invalid_argument("mmrm: invalid fixed design, observations, labels or controls");

    const auto levels = visit_levels(visit);
    const auto positions = visit_positions(visit, levels);
    const auto q = levels.size();
    const auto capacity = Vector{}.max_size();
    if (q > capacity / q || n > capacity / p || p > capacity / p)
        throw std::length_error("mmrm: covariance or design exceeds storage capacity");
    const auto parameters = q * (q + 1) / 2;
    if (parameters > capacity / parameters)
        throw std::length_error("mmrm: likelihood information exceeds storage capacity");

    std::map<std::size_t, Indices> rows;
    for (std::size_t i = 0; i < n; ++i)
        rows[subject[i]].push_back(i);
    if (rows.size() < 2)
        throw std::invalid_argument("mmrm: at least two subjects are needed");
    std::map<Indices, std::vector<Indices>> grouped;
    linalg::DenseMatrix<std::size_t> pairs(q, q);
    for (auto& [id, indices] : rows) {
        std::ranges::sort(indices, {}, [&](auto row) { return positions[row]; });
        Indices pattern;
        pattern.reserve(indices.size());
        for (const auto i : indices)
            pattern.push_back(positions[i]);
        if (std::ranges::adjacent_find(pattern) != pattern.end())
            throw std::invalid_argument("mmrm: each subject and visit require a unique row");
        for (std::size_t i = 0; i < pattern.size(); ++i)
            for (std::size_t j = 0; j <= i; ++j)
                ++pairs(pattern[i], pattern[j]);
        grouped[pattern].push_back(std::move(indices));
    }
    for (std::size_t i = 0; i < q; ++i)
        for (std::size_t j = 0; j < i; ++j)
            if (pairs(i, j) == 0)
                throw std::invalid_argument("mmrm: each pair of visits requires joint observations");

    Matrix x(n, p);
    Vector y(n), x_scale(p), visit_scale(q), visit_magnitude(q);
    long double y_scale = 0;
    for (const auto value : response)
        y_scale = std::max(y_scale, std::abs(static_cast<long double>(value)));
    if (y_scale == 0)
        throw std::invalid_argument("mmrm: no residual variation");
    for (std::size_t i = 0; i < n; ++i)
        y[i] = response[i] / y_scale;
    for (std::size_t j = 0; j < p; ++j) {
        for (std::size_t i = 0; i < n; ++i)
            x_scale[j] = std::max(x_scale[j], std::abs(static_cast<long double>(design(i, j))));
        if (x_scale[j] == 0)
            throw std::invalid_argument("mmrm: zero fixed design column");
        for (std::size_t i = 0; i < n; ++i)
            x(i, j) = design(i, j) / x_scale[j];
    }
    const auto svd = linalg::svd(x);
    if (!svd.converged || svd.rank() != p)
        throw std::invalid_argument("mmrm: aliased fixed design");
    // Translating the response by X times an initial coefficient vector
    // preserves the covariance likelihood and reduces cancellation in GLS.
    const auto initial_beta = svd.solve(y);
    auto residual = x * initial_beta;
    for (std::size_t i = 0; i < n; ++i) {
        auto& magnitude = visit_magnitude[positions[i]];
        magnitude = std::max({magnitude, std::abs(y[i]), std::abs(residual[i])});
        residual[i] = y[i] - residual[i];
        auto& scale = visit_scale[positions[i]];
        scale = std::hypot(scale, residual[i]);
    }
    for (std::size_t v = 0; v < q; ++v) {
        visit_scale[v] /= std::sqrt(static_cast<long double>(pairs(v, v)));
        if (visit_scale[v] <= 64 * std::numeric_limits<long double>::epsilon() * visit_magnitude[v])
            throw std::invalid_argument("mmrm: a visit has no residual variation");
    }
    Vector working_scale(p);
    for (std::size_t j = 0; j < p; ++j)
        for (std::size_t i = 0; i < n; ++i)
            working_scale[j] = std::max(working_scale[j], std::abs(x(i, j)) / visit_scale[positions[i]]);

    Problem problem{{}, n, p, q, options.reml};
    Matrix correlation = Matrix::identity(q);
    for (const auto& [positions_at_visit, subjects] : grouped) {
        auto& pattern = problem.patterns.emplace_back(Pattern{positions_at_visit, {}});
        const auto m = pattern.visits.size();
        for (const auto& indices : subjects) {
            auto& block = pattern.subjects.emplace_back(Subject{Matrix(m, p), Vector(m)});
            for (std::size_t j = 0; j < m; ++j) {
                const auto row = indices[j], v = pattern.visits[j];
                const auto scale = visit_scale[v];
                block.response[j] = residual[row] / scale;
                for (std::size_t k = 0; k < p; ++k)
                    block.design(j, k) = x(row, k) / (scale * working_scale[k]);
                for (std::size_t k = 0; k < j; ++k) {
                    const auto w = pattern.visits[k];
                    correlation(v, w) += (residual[row] / scale) * (residual[indices[k]] / visit_scale[w]);
                }
            }
        }
    }
    for (std::size_t i = 0; i < q; ++i)
        for (std::size_t j = 0; j < i; ++j)
            correlation(j, i) = correlation(i, j) = .75L * std::clamp(correlation(i, j) / pairs(i, j), -.99L, .99L);

    /// @brief Profiles fixed effects at each candidate covariance.
    const auto profile = [&](const Vector& beta) { return evaluate(problem, beta); };
    auto best = detail::maximize_likelihood<MMRMStatus>(Vector(parameters), options.max_iterations, options.tolerance, profile);
    const auto start = initial_correlation(std::move(correlation), parameters);
    if (std::ranges::any_of(start, [](long double value) { return value != 0; })) {
        auto alternative = detail::maximize_likelihood<MMRMStatus>(start, options.max_iterations, options.tolerance, profile);
        const auto rounding = 64 * std::numeric_limits<long double>::epsilon() *
            std::max(1.0L, std::abs(best.state.value));
        if (best.state.score.empty() || alternative.state.value > best.state.value + rounding ||
            (alternative.status == MMRMStatus::Converged && best.status != MMRMStatus::Converged &&
             alternative.state.value >= best.state.value - rounding))
            best = std::move(alternative);
    }
    if (best.state.score.empty())
        throw std::runtime_error("mmrm: no usable positive-definite fit was found");

    MMRMFit result{{}, linalg::DenseMatrix<double>(p, p), levels, linalg::DenseMatrix<double>(q, q), {},
        0, best.status};
    result.coefficients.resize(p);
    const auto log_y_scale = std::log(y_scale);
    auto log_likelihood = n * best.state.value;
    for (std::size_t v = 0; v < q; ++v)
        log_likelihood -= pairs(v, v) * (log_y_scale + std::log(visit_scale[v]));
    for (std::size_t j = 0; j < p; ++j) {
        const auto scale = (y_scale / x_scale[j]) / working_scale[j];
        result.coefficients[j] = output((initial_beta[j] + best.state.coefficients[j] / working_scale[j]) *
            (y_scale / x_scale[j]));
        for (std::size_t k = 0; k <= j; ++k)
            result.covariance(j, k) = result.covariance(k, j) =
                output(best.state.fixed_covariance(j, k) * scale * ((y_scale / x_scale[k]) / working_scale[k]));
        if (options.reml)
            log_likelihood -= std::log(x_scale[j]) + std::log(working_scale[j]) - log_y_scale;
    }
    for (std::size_t i = 0; i < q; ++i)
        for (std::size_t j = 0; j <= i; ++j)
            result.visit_covariance(i, j) = result.visit_covariance(j, i) =
                output(best.state.visit_covariance(i, j) * (y_scale * visit_scale[i]) * (y_scale * visit_scale[j]));
    result.fitted = result.predict(design);
    result.log_likelihood = output(log_likelihood);
    return result;
}

MMRMFit fit_mmrm(const std::vector<double>& response, const std::vector<std::size_t>& subject,
    const std::vector<std::size_t>& visit, const MMRMOptions& options) {
    if (response.empty() || response.size() != visit.size() || response.size() != subject.size())
        throw std::invalid_argument("mmrm: response and labels must have the same nonzero length");
    const auto levels = visit_levels(visit);
    if (response.size() > std::vector<double>{}.max_size() / levels.size())
        throw std::length_error("mmrm: fixed design exceeds storage capacity");
    const auto positions = visit_positions(visit, levels);
    linalg::DenseMatrix<double> design(response.size(), levels.size());
    for (std::size_t i = 0; i < response.size(); ++i)
        design(i, positions[i]) = 1;
    return fit_mmrm(design, response, subject, visit, options);
}

} // namespace datamunge::stats
