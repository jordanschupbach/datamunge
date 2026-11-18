/// @file effects.cpp
/// @brief Treatment effect point estimation.
#include <datamunge/causal/effects.hpp>
#include <datamunge/linalg/least_squares.hpp>
#include <datamunge/stats/glm.hpp>
#include "detail.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace datamunge::causal {
/// @brief Implementation helpers local to this translation unit.
namespace {
/// @brief Double-valued regression design.
using Matrix = linalg::DenseMatrix<double>;

/// @brief Validates the length and finiteness of a numeric column.
void check_column(const std::vector<double>& values, std::size_t n) {
    if (values.size() != n || !std::ranges::all_of(values, [](double x) { return std::isfinite(x); }))
        throw std::invalid_argument("Effect estimator columns must have matching lengths and finite values");
}

/// @brief Validates binary indicators and counts the two arms.
auto arm_counts(const std::vector<int>& treatment, const std::vector<double>& outcome) {
    check_column(outcome, treatment.size());
    std::array<std::size_t, 2> counts{};
    for (int arm : treatment) {
        if (arm != 0 && arm != 1) throw std::invalid_argument("Treatment must be 0 or 1");
        ++counts[arm];
    }
    if (counts[0] == 0 || counts[1] == 0) throw std::invalid_argument("Both treatment arms need observations");
    return counts;
}

/// @brief Checks matrix dimensions and finite covariate values.
void check_matrix(const Matrix& matrix, std::size_t rows) {
    if (matrix.rows() != rows && (matrix.rows() != 0 || matrix.cols() != 0))
        throw std::invalid_argument("Effect estimator matrix row count mismatch");
    detail::finite_matrix(matrix);
}

/// @brief Fits a full-rank design using the library least-squares solver.
auto regress(const Matrix& design, const std::vector<double>& y) {
    const auto fit = linalg::least_squares(design, y, linalg::LeastSquaresMethod::SVD);
    if (fit.rank != design.cols()) throw std::runtime_error("Rank-deficient effect regression");
    return fit;
}
} // namespace

BinaryATE binary_ate(const std::vector<double>& y, const std::vector<int>& a, const std::vector<double>& e,
                     const std::vector<double>& m0, const std::vector<double>& m1) {
    const auto counts = arm_counts(a, y);
    check_column(e, a.size());
    check_column(m0, a.size());
    check_column(m1, a.size());
    std::array<long double, 2> sums{}, weights{}, weighted_sums{};
    long double g = 0, augmented = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (!(e[i] > 0 && e[i] < 1)) throw std::invalid_argument("Propensities must lie strictly between zero and one");
        const int arm = a[i];
        const long double weight = 1.0L / (arm ? e[i] : 1.0 - e[i]);
        const long double contrast = static_cast<long double>(m1[i]) - m0[i];
        sums[arm] += y[i];
        weights[arm] += weight;
        weighted_sums[arm] += weight * y[i];
        g += contrast;
        augmented += contrast + (arm ? 1 : -1) * weight * (static_cast<long double>(y[i]) - (arm ? m1[i] : m0[i]));
    }
    const auto [lo, hi] = std::ranges::minmax(e);
    return {static_cast<double>(sums[1] / counts[1] - sums[0] / counts[0]),
            static_cast<double>(g / a.size()),
            static_cast<double>(weighted_sums[1] / weights[1] - weighted_sums[0] / weights[0]),
            static_cast<double>(augmented / a.size()), lo, hi};
}

BinaryATE fit_binary_ate(const std::vector<double>& y, const std::vector<int>& a, const linalg::DenseMatrix<double>& x) {
    const auto counts = arm_counts(a, y);
    check_matrix(x, a.size());
    const auto n = a.size(), p = x.cols() + 1;
    std::vector<double> propensity(n, static_cast<double>(counts[1]) / static_cast<double>(n));
    Matrix design(n, p, 1.0);
    if (p > 1) {
        dstruct::DataFrame frame;
        frame.add_column("treated", std::vector<double>(a.begin(), a.end()));
        std::string formula = "treated ~ 1";
        for (std::size_t j = 1; j < p; ++j) {
            const auto name = "covariate" + std::to_string(j);
            std::vector<double> column(n);
            for (std::size_t i = 0; i < n; ++i) column[i] = design(i, j) = x(i, j - 1);
            frame.add_column(name, column);
            formula += " + " + name;
        }
        const stats::GLM model(frame, formula, {.family = stats::GLMFamily::Binomial});
        if (!model.converged() || model.boundary()) throw std::runtime_error("Unstable logistic propensity fit");
        propensity = model.fitted_values();
    }
    std::array<std::vector<double>, 2> means;
    for (int arm = 0; arm < 2; ++arm) {
        if (counts[arm] < p) throw std::invalid_argument("Too few observations for an arm-specific outcome model");
        Matrix arm_design(counts[arm], p);
        std::vector<double> response;
        response.reserve(counts[arm]);
        for (std::size_t i = 0; i < n; ++i) {
            if (a[i] != arm) continue;
            for (std::size_t j = 0; j < p; ++j) arm_design(response.size(), j) = design(i, j);
            response.push_back(y[i]);
        }
        const auto fit = regress(arm_design, response);
        means[arm].resize(n);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < p; ++j) means[arm][i] += design(i, j) * fit.coefficients[j];
    }
    return binary_ate(y, a, propensity, means[0], means[1]);
}

double difference_in_differences(const std::vector<double>& before, const std::vector<double>& after,
                                 const std::vector<int>& a) {
    const auto counts = arm_counts(a, before);
    check_column(after, a.size());
    std::array<long double, 2> change{};
    for (std::size_t i = 0; i < a.size(); ++i) change[a[i]] += static_cast<long double>(after[i]) - before[i];
    return static_cast<double>(change[1] / counts[1] - change[0] / counts[0]);
}

InstrumentalEffect two_stage_least_squares(const std::vector<double>& y, const std::vector<double>& x,
                                          const linalg::DenseMatrix<double>& instruments,
                                          const linalg::DenseMatrix<double>& controls) {
    const auto n = y.size(), p = controls.cols() + 1, q = instruments.cols();
    check_column(y, n);
    check_column(x, n);
    check_matrix(instruments, n);
    check_matrix(controls, n);
    if (q == 0 || p >= n || q >= n - p)
        throw std::invalid_argument("2SLS needs excluded instruments and first-stage residual degrees of freedom");
    Matrix restricted(n, p, 1.0), first(n, p + q, 1.0), second(n, p + 1, 1.0);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 1; j < p; ++j) restricted(i, j) = first(i, j) = second(i, j) = controls(i, j - 1);
        for (std::size_t j = 0; j < q; ++j) first(i, p + j) = instruments(i, j);
    }
    const auto full = regress(first, x);
    const auto reduced = regress(restricted, x);
    for (std::size_t i = 0; i < n; ++i) second(i, p) = x[i] - full.residuals[i];
    const double effect = regress(second, y).coefficients[p];
    const double ratio = reduced.residual_norm / full.residual_norm;
    const double f = full.residual_norm == 0 ? std::numeric_limits<double>::infinity()
        : std::max(0.0, (ratio - 1) * (ratio + 1)) * static_cast<double>(n - p - q) / static_cast<double>(q);
    return {effect, f};
}
} // namespace datamunge::causal
