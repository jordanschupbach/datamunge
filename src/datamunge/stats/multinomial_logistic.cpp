/// @file multinomial_logistic.cpp
/// @brief Softmax likelihood, joint information, and safeguarded Newton fitting.

#include <datamunge/stats/multinomial_logistic.hpp>

#include "likelihood_newton.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace datamunge::stats {

/// @brief Likelihood and optimization details for both response forms.
namespace {

/// @brief Working-precision vector.
using Vector = std::vector<long double>;

/// @brief Working-precision matrix.
using Matrix = linalg::DenseMatrix<long double>;

/// @brief Conditional multinomial likelihood and its joint derivatives.
struct Evaluation {
    long double value; ///< Log-likelihood kernel.
    Vector score;      ///< First derivative in working coefficient order.
    Matrix info;       ///< Lower triangle of joint information.
};

/// @brief Narrows a finite reported quantity.
double output(long double value) {
    if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<double>::max())
        throw std::overflow_error("multinomial_logistic: result exceeds finite double range");
    return static_cast<double>(value);
}

/// @brief Identifies all coefficient-bearing classes in probability column order.
std::vector<std::size_t> nonbaseline_classes(std::size_t count, std::size_t baseline) {
    std::vector<std::size_t> result;
    for (std::size_t c = 0; c < count; ++c)
        if (c != baseline)
            result.push_back(c);
    return result;
}

/// @brief Computes shifted logits, masses, and the sum of the masses other than the largest.
long double softmax(Vector& eta, Vector& mass, std::size_t& largest) {
    largest = static_cast<std::size_t>(std::ranges::max_element(eta) - eta.begin());
    const auto shift = eta[largest];
    long double rest = 0;
    for (std::size_t c = 0; c < eta.size(); ++c) {
        eta[c] -= shift;
        mass[c] = std::exp(eta[c]);
        if (c != largest)
            rest += mass[c];
    }
    return rest;
}

/// @brief Computes logits of a single row in class-column order.
void logits(const Matrix& design, std::size_t row, const Vector& beta, const std::vector<std::size_t>& nonbaseline,
            Vector& eta) {
    const auto p = design.cols();
    std::ranges::fill(eta, 0.0L);
    for (std::size_t a = 0; a < nonbaseline.size(); ++a) {
        long double value = 0;
        for (std::size_t j = 0; j < p; ++j)
            value = std::fma(design(row, j), beta[a * p + j], value);
        eta[nonbaseline[a]] = value;
    }
}

/// @brief Evaluates the softmax likelihood, score, and full cross-class information.
Evaluation evaluate(const Matrix& x, const linalg::DenseMatrix<double>& counts, const Vector& totals,
                    const Vector& weights, const std::vector<std::size_t>& nonbaseline, const Vector& beta) {
    const auto p = x.cols(), d = beta.size();
    Evaluation result{0, Vector(d), Matrix(d, d)};
    Vector eta(counts.cols()), mass(counts.cols());
    for (std::size_t i = 0; i < x.rows(); ++i) {
        const auto total = totals[i];
        if (total == 0)
            continue;
        logits(x, i, beta, nonbaseline, eta);
        std::size_t largest;
        const auto rest = softmax(eta, mass, largest);
        const auto norm = 1 + rest;
        const auto log_norm = std::log1p(rest);
        long double observed = 0, other_count = 0;
        for (std::size_t c = 0; c < counts.cols(); ++c) {
            if (counts(i, c) != 0)
                observed += counts(i, c) * eta[c];
            if (c != largest)
                other_count += counts(i, c);
        }
        result.value += weights[i] * (observed - total * log_norm);
        for (std::size_t a = 0; a < nonbaseline.size(); ++a) {
            const auto c = nonbaseline[a];
            const auto prob = mass[c] / norm;
            const auto opposite_mass = c == largest ? rest : norm - mass[c];
            const auto residual = c == largest ? counts(i, c) * (rest / norm) - other_count * prob
                                               : counts(i, c) - total * prob;
            for (std::size_t j = 0; j < p; ++j)
                result.score[a * p + j] += weights[i] * residual * x(i, j);
            for (std::size_t b = 0; b <= a; ++b) {
                const auto class_covariance = prob * (a == b ? opposite_mass / norm : -mass[nonbaseline[b]] / norm);
                const auto factor = weights[i] * total * class_covariance;
                for (std::size_t j = 0; j < p; ++j)
                    for (std::size_t k = 0; k < (a == b ? j + 1 : p); ++k)
                        result.info(a * p + j, b * p + k) += (factor * x(i, j)) * x(i, k);
            }
        }
    }
    return result;
}

/// @brief Tests whether a change in coefficients supports a likelihood recession.
detail::LikelihoodStep step_effect(const Matrix& x, const linalg::DenseMatrix<double>& counts, const Vector& totals,
                       const std::vector<std::size_t>& nonbaseline, const Vector& step) {
    long double maximum = 0, largest_violation = 0, largest_gap = 0;
    Vector eta(counts.cols());
    for (std::size_t i = 0; i < x.rows(); ++i) {
        if (totals[i] == 0)
            continue;
        logits(x, i, step, nonbaseline, eta);
        const auto [lo, hi] = std::ranges::minmax_element(eta);
        maximum = std::max({maximum, std::abs(*lo), std::abs(*hi)});
        for (std::size_t c = 0; c < counts.cols(); ++c) {
            if (counts(i, c) > 0)
                largest_violation = std::max(largest_violation, *hi - eta[c]);
        }
        largest_gap = std::max(largest_gap, *hi - *lo);
    }
    const auto rounding = 64 * std::numeric_limits<long double>::epsilon() * maximum;
    return {maximum, largest_violation <= rounding && largest_gap > rounding};
}

/// @brief Fits either the one-hot or count representation and assembles the public result.
MultinomialLogisticResult fit_multinomial(
    const linalg::DenseMatrix<double>& design, const linalg::DenseMatrix<double>& counts,
    std::vector<std::int64_t> classes, const MultinomialLogisticOptions& options) {
    using Status = MultinomialLogisticStatus;
    const auto n = design.rows(), p = design.cols(), classes_count = classes.size();
    if (p == 0 || n < p || counts.rows() != n || classes_count < 2 || counts.cols() != classes_count ||
        (!options.weights.empty() && options.weights.size() != n) || options.max_iterations == 0 ||
        !(options.tolerance > 0 && options.tolerance < 1) ||
        !std::ranges::all_of(design.data(), [](double x) { return std::isfinite(x); }))
        throw std::invalid_argument("multinomial_logistic: invalid data shape or fit controls");
    const auto baseline_label = options.baseline.value_or(classes.front());
    const auto baseline_at = std::ranges::find(classes, baseline_label);
    if (baseline_at == classes.end())
        throw std::invalid_argument("multinomial_logistic: baseline must be an observed class label");
    const auto others = nonbaseline_classes(classes_count, static_cast<std::size_t>(baseline_at - classes.begin()));
    const auto maximum_dimension = std::numeric_limits<std::size_t>::max();
    if (p > maximum_dimension / others.size() || p * others.size() > maximum_dimension / (p * others.size()))
        throw std::length_error("multinomial_logistic: information matrix dimensions exceed capacity");
    const auto d = p * others.size();
    Vector totals(n), weights(n, 1), scales(p);
    std::vector<bool> observed(classes_count);
    for (std::size_t i = 0; i < n; ++i) {
        if (!options.weights.empty())
            weights[i] = options.weights[i];
        if (!(weights[i] > 0) || !std::isfinite(weights[i]))
            throw std::invalid_argument("multinomial_logistic: weights must be finite and positive");
        for (std::size_t c = 0; c < classes_count; ++c) {
            const auto count = counts(i, c);
            if (!(count >= 0) || !std::isfinite(count) || count != std::floor(count))
                throw std::invalid_argument("multinomial_logistic: counts must be nonnegative integers");
            observed[c] = observed[c] || count > 0;
            totals[i] += count;
        }
        if (totals[i] > 0)
            for (std::size_t j = 0; j < p; ++j)
                scales[j] = std::max(scales[j], std::abs(static_cast<long double>(design(i, j))));
    }
    if (std::ranges::find(observed, false) != observed.end() || std::ranges::find(scales, 0.0L) != scales.end())
        throw std::invalid_argument("multinomial_logistic: every class must be observed and design columns must be nonzero");
    Matrix x(n, p);
    for (std::size_t i = 0; i < n; ++i)
        if (totals[i] > 0)
            for (std::size_t j = 0; j < p; ++j)
                x(i, j) = design(i, j) / scales[j];

    const auto fit = detail::maximize_likelihood<Status>(d, options.max_iterations, options.tolerance,
        [&](const Vector& beta) { return evaluate(x, counts, totals, weights, others, beta); },
        [&](const Vector& step) { return step_effect(x, counts, totals, others, step); });
    if (!fit)
        throw std::invalid_argument("multinomial_logistic: design is numerically rank deficient");

    MultinomialLogisticResult result{std::move(classes), baseline_label, linalg::DenseMatrix<double>(others.size(), p),
                                     {}, {}, output(fit->value), fit->iterations, fit->status};
    for (std::size_t a = 0; a < others.size(); ++a)
        for (std::size_t j = 0; j < p; ++j)
            result.coefficients(a, j) = output(fit->beta[a * p + j] / scales[j]);
    if (fit->status == Status::Converged) {
        const auto inverse = fit->factor.solve(Matrix::identity(d));
        result.covariance = linalg::DenseMatrix<double>(d, d);
        for (std::size_t j = 0; j < d; ++j)
            for (std::size_t k = 0; k <= j; ++k)
                result.covariance(j, k) = result.covariance(k, j) = output(inverse(j, k) / scales[j % p] / scales[k % p]);
    }
    result.fitted_probabilities = result.predict(design);
    return result;
}

} // namespace

linalg::DenseMatrix<double> MultinomialLogisticResult::predict(const linalg::DenseMatrix<double>& design) const {
    const auto reference = std::ranges::find(classes, baseline);
    const auto finite = [](double x) { return std::isfinite(x); };
    if (classes.size() < 2 || reference == classes.end() || coefficients.rows() != classes.size() - 1 ||
        coefficients.cols() != design.cols() || !std::ranges::all_of(coefficients.data(), finite) ||
        !std::ranges::all_of(design.data(), finite))
        throw std::invalid_argument("multinomial_logistic: incompatible prediction design or coefficient layout");
    const auto others = nonbaseline_classes(classes.size(), static_cast<std::size_t>(reference - classes.begin()));
    linalg::DenseMatrix<double> result(design.rows(), classes.size());
    Vector eta(classes.size()), mass(classes.size());
    for (std::size_t i = 0; i < design.rows(); ++i) {
        for (std::size_t a = 0; a < others.size(); ++a) {
            long double value = 0;
            for (std::size_t j = 0; j < design.cols(); ++j)
                value = std::fma(static_cast<long double>(design(i, j)), coefficients(a, j), value);
            if (!std::isfinite(value))
                throw std::overflow_error("multinomial_logistic: query logit exceeds finite working range");
            eta[others[a]] = value;
        }
        eta[static_cast<std::size_t>(reference - classes.begin())] = 0;
        std::size_t largest;
        const auto norm = 1 + softmax(eta, mass, largest);
        for (std::size_t c = 0; c < classes.size(); ++c)
            result(i, c) = static_cast<double>(mass[c] / norm);
    }
    return result;
}

MultinomialLogisticResult multinomial_logistic(const linalg::DenseMatrix<double>& design,
    const std::vector<std::int64_t>& labels, const MultinomialLogisticOptions& options) {
    if (design.rows() != labels.size())
        throw std::invalid_argument("multinomial_logistic: class labels must match design rows");
    auto classes = labels;
    std::ranges::sort(classes);
    classes.erase(std::unique(classes.begin(), classes.end()), classes.end());
    if (classes.size() < 2)
        throw std::invalid_argument("multinomial_logistic: at least two classes are required");
    if (labels.size() > std::numeric_limits<std::size_t>::max() / classes.size())
        throw std::length_error("multinomial_logistic: class indicator dimensions exceed capacity");
    linalg::DenseMatrix<double> counts(labels.size(), classes.size());
    for (std::size_t i = 0; i < labels.size(); ++i)
        counts(i, std::ranges::lower_bound(classes, labels[i]) - classes.begin()) = 1;
    return fit_multinomial(design, counts, std::move(classes), options);
}

MultinomialLogisticResult multinomial_logistic(const linalg::DenseMatrix<double>& design,
    const linalg::DenseMatrix<double>& counts, const MultinomialLogisticOptions& options) {
    std::vector<std::int64_t> classes(counts.cols());
    std::iota(classes.begin(), classes.end(), std::int64_t{0});
    return fit_multinomial(design, counts, std::move(classes), options);
}

} // namespace datamunge::stats
