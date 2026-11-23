/// @file repeated_measures_anova.cpp
/// @brief Univariate repeated-measures F tests for within-subject contrast spaces.

#include <datamunge/stats/repeated_measures_anova.hpp>

#include "linear_model_spaces.hpp"
#include <datamunge/linalg/svd.hpp>
#include <datamunge/stats/detail/test_common.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace datamunge::stats {
/// @brief Orthonormal bases and covariance trace calculations for repeated effects.
namespace {

/// @brief Input and result matrix type.
using PublicMatrix = linalg::DenseMatrix<double>;
/// @brief Working matrix type for projecting responses and model designs.
using Matrix = linalg::DenseMatrix<long double>;

/// @brief Checks that a dense workspace fits before multiplying dimensions.
void check_storage(std::size_t rows, std::size_t columns) {
    if (columns && rows > std::vector<long double>{}.max_size() / columns)
        throw std::length_error("repeated-measures ANOVA: dense workspace exceeds storage capacity");
}

/// @brief Checks that all matrix entries are finite.
bool finite(const PublicMatrix& matrix) {
    return std::ranges::all_of(matrix.data(), [](double x) { return std::isfinite(x); });
}

/// @brief Squared Frobenius norm.
long double squared_norm(const Matrix& matrix) {
    return std::inner_product(matrix.data().begin(), matrix.data().end(), matrix.data().begin(), 0.0L);
}

/// @brief Constructs orthonormal columns spanning valid within-subject contrasts.
Matrix contrast_basis(const PublicMatrix& contrasts) {
    check_storage(contrasts.cols(), contrasts.rows());
    Matrix transposed(contrasts.cols(), contrasts.rows());
    for (std::size_t r = 0; r < contrasts.rows(); ++r) {
        long double peak = 0;
        for (std::size_t j = 0; j < contrasts.cols(); ++j)
            peak = std::max(peak, std::abs(static_cast<long double>(contrasts(r,j))));
        if (peak == 0)
            throw std::invalid_argument("repeated-measures ANOVA: zero within-subject contrast");
        long double sum = 0, magnitude = 0;
        for (std::size_t j = 0; j < contrasts.cols(); ++j) {
            const auto value = contrasts(r,j) / peak;
            transposed(j,r) = value;
            sum += value;
            magnitude += std::abs(value);
        }
        if (std::abs(sum) > 64 * std::numeric_limits<double>::epsilon() * magnitude)
            throw std::invalid_argument("repeated-measures ANOVA: within-subject contrast must sum to zero");
        for (std::size_t j = 0; j < contrasts.cols(); ++j)
            transposed(j,r) -= sum / contrasts.cols();
    }
    auto svd = linalg::svd(transposed);
    if (!svd.converged)
        throw std::runtime_error("repeated-measures ANOVA: within-subject contrast factorization failed");
    if (svd.rank(contrasts.cols() * std::numeric_limits<double>::epsilon()) != contrasts.rows())
        throw std::invalid_argument("repeated-measures ANOVA: aliased within-subject contrasts");
    return std::move(svd.U);
}

/// @brief Returns the F reference distribution after multiplying its degrees of freedom.
RepeatedMeasuresFTest corrected(double f, double df1, double df2, long double epsilon) {
    const auto scale = static_cast<double>(epsilon);
    const auto numerator = df1 * scale, denominator = df2 * scale;
    return {scale, numerator, denominator, detail::f_test_survival(f, numerator, denominator)};
}

/// @brief Projects observations onto the tested effects and uses the residual cross-product for epsilon.
RepeatedMeasuresANOVAResult test_effect(const PublicMatrix& response,
    const Matrix& full, const Matrix& reduced, const Matrix& contrasts) {
    const auto n = response.rows(), visits = response.cols(), q = contrasts.cols();
    check_storage(n, q);
    check_storage(q, q);
    const auto full_t = full.transpose();
    detail::nested_coordinates(full, reduced, "repeated-measures ANOVA");

    // Remove the subject intercept at a visit used by each contrast.
    // This keeps a large offset at an unrelated visit out of the projection.
    Matrix projected(n, q);
    long double scale = 0;
    for (std::size_t j = 0; j < q; ++j) {
        std::size_t reference = 0;
        for (std::size_t v = 1; v < visits; ++v)
            if (std::abs(contrasts(v,j)) > std::abs(contrasts(reference,j)))
                reference = v;
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t v = 0; v < visits; ++v)
                if (contrasts(v,j) != 0)
                    projected(i,j) += (static_cast<long double>(response(i,v)) - response(i,reference)) * contrasts(v,j);
            scale = std::max(scale, std::abs(projected(i,j)));
        }
    }
    if (scale == 0)
        throw std::invalid_argument("repeated-measures ANOVA: zero residual contrast variation");
    for (auto& entry : projected.data())
        entry /= scale;
    const auto fitted = full * (full_t * projected);
    const auto residual = projected - fitted;
    const auto error = residual.transpose() * residual;
    const auto trace = squared_norm(residual);
    const auto epsilon_precision = 64 * std::numeric_limits<long double>::epsilon() * std::max(n, visits);
    if (!(trace > epsilon_precision * epsilon_precision * squared_norm(projected)))
        throw std::invalid_argument("repeated-measures ANOVA: zero residual contrast variation");
    const auto hypothesis = fitted - reduced * (reduced.transpose() * fitted);
    const auto residual_df = n - full.cols();
    const auto effect_df = full.cols() - reduced.cols();
    const auto f = static_cast<double>((squared_norm(hypothesis) / trace) *
        (static_cast<long double>(residual_df) / effect_df));
    const auto df1 = static_cast<double>(effect_df) * q, df2 = static_cast<double>(residual_df) * q;
    const auto lower = 1.0L / q;
    const auto gg = std::clamp((trace * trace) / (q * squared_norm(error)), lower, 1.0L);
    RepeatedMeasuresANOVAResult result{f, corrected(f, df1, df2, 1), corrected(f, df1, df2, gg), std::nullopt};
    if (q == 1) {
        result.huynh_feldt = result.uncorrected;
    } else if (residual_df > 1) {
        // The residual degrees of freedom account for every fitted subject-level column.
        const auto k_epsilon = gg * q;
        const auto denominator = q * (residual_df - k_epsilon);
        const auto hf = denominator > 0 ? ((residual_df + 1) * k_epsilon - 2) / denominator : 1;
        result.huynh_feldt = corrected(f, df1, df2, std::clamp(hf, lower, 1.0L));
    }
    return result;
}

} // namespace

RepeatedMeasuresANOVAResult repeated_measures_anova(const linalg::DenseMatrix<double>& response) {
    const auto n = response.rows(), v = response.cols();
    if (n < 2 || v < 2 || !finite(response))
        throw std::invalid_argument("repeated-measures ANOVA: requires two subjects, two visits and finite responses");
    check_storage(v, v - 1);
    Matrix helmert(v, v - 1);
    for (std::size_t j = 0; j + 1 < v; ++j) {
        const auto k = static_cast<long double>(j + 1);
        const auto unit = 1 / std::sqrt(k * (k + 1));
        for (std::size_t i = 0; i <= j; ++i)
            helmert(i,j) = unit;
        helmert(j+1,j) = -k * unit;
    }
    return test_effect(response, Matrix(n,1,1 / std::sqrt(static_cast<long double>(n))), Matrix(n,0), helmert);
}

RepeatedMeasuresANOVAResult repeated_measures_anova(const linalg::DenseMatrix<double>& response,
    const linalg::DenseMatrix<double>& full_subject_design, const linalg::DenseMatrix<double>& reduced_subject_design,
    const linalg::DenseMatrix<double>& within_contrasts) {
    const auto n = response.rows(), visits = response.cols();
    const auto p = full_subject_design.cols(), r = reduced_subject_design.cols();
    if (full_subject_design.rows() != n || reduced_subject_design.rows() != n || r >= p || p >= n ||
        within_contrasts.cols() != visits || within_contrasts.rows() == 0 || within_contrasts.rows() >= visits ||
        !finite(response) || !finite(full_subject_design) || !finite(reduced_subject_design) || !finite(within_contrasts))
        throw std::invalid_argument("repeated-measures ANOVA: incompatible shapes or nonfinite data");
    return test_effect(response, detail::orthonormal_design(full_subject_design, "repeated-measures ANOVA"),
        detail::orthonormal_design(reduced_subject_design, "repeated-measures ANOVA"), contrast_basis(within_contrasts));
}

} // namespace datamunge::stats
