/// @file lm_marginal_anova.cpp
/// @brief Marginal term tests with the full linear model's error estimate.

#include <datamunge/stats/lm.hpp>

#include <datamunge/linalg/cholesky.hpp>
#include <datamunge/stats/detail/test_common.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace datamunge::stats {

/// @brief Coefficient-space projections for Type II and III sums of squares.
namespace {

/// @brief Computes the extra sum of squares for the final coefficients in the joint hypothesis.
///
/// Cholesky whitening first projects out the restriction on any omitted terms.
/// The squared whitened coordinates corresponding to the tested term then give
/// the conditional Wald quadratic form. The coefficient scale is removed before
/// factoring to accommodate columns measured in different units.
double term_sum_sq(const std::vector<double>& coefficients, const linalg::DenseMatrix<double>& covariance,
                   const std::vector<std::size_t>& columns, std::size_t omitted, long double variance_scale) {
    const auto count = columns.size();
    linalg::DenseMatrix<long double> correlation(count, count);
    std::vector<long double> scale(count), transformed(count);
    for (std::size_t i = 0; i < count; ++i) {
        scale[i] = std::sqrt(static_cast<long double>(covariance(columns[i], columns[i])));
        if (!(scale[i] > 0) || !std::isfinite(scale[i]))
            throw std::runtime_error("LM::marginal_anova: degenerate coefficient information");
        correlation(i, i) = 1;
        for (std::size_t j = 0; j < i; ++j)
            correlation(i, j) = covariance(columns[i], columns[j]) / scale[i] / scale[j];
    }
    const auto factor = linalg::cholesky(correlation);
    if (!factor.ok)
        throw std::runtime_error("LM::marginal_anova: hypothesis lost numerical estimability");
    long double sum_sq = 0;
    for (std::size_t i = 0; i < count; ++i) {
        auto value = static_cast<long double>(coefficients[columns[i]]) / scale[i];
        for (std::size_t j = 0; j < i; ++j)
            value = std::fma(-factor.L(i, j), transformed[j], value);
        transformed[i] = value / factor.L(i, i);
        if (i >= omitted)
            sum_sq += transformed[i] * transformed[i];
    }
    sum_sq *= variance_scale;
    if (!std::isfinite(sum_sq) || sum_sq > std::numeric_limits<double>::max())
        throw std::overflow_error("LM::marginal_anova: nonfinite term sum of squares");
    return static_cast<double>(sum_sq);
}

} // namespace

std::vector<AnovaRow> LM::marginal_anova(MarginalAnovaType type) const {
    if (type != MarginalAnovaType::II && type != MarginalAnovaType::III)
        throw std::invalid_argument("LM::marginal_anova: unknown ANOVA type");
    const auto& terms = design_.term_groups;
    if (terms.empty())
        return {};

    const std::size_t intercept = design_.has_intercept ? 1 : 0;
    std::vector<AnovaRow> result;
    for (const auto& term : terms) {
        std::vector<std::size_t> columns;
        const auto append = [&](const TermGroup& group) {
            for (const auto column : group.column_indices)
                columns.push_back(column + intercept);
        };
        if (type == MarginalAnovaType::II)
            for (const auto& candidate : terms)
                if (candidate.factors.size() > term.factors.size() &&
                    std::ranges::includes(candidate.factors, term.factors))
                    append(candidate);
        const auto omitted = columns.size();
        append(term);
        const auto df = term.column_indices.size();
        const auto sum_sq = sigma2_ > 0
            ? term_sum_sq(coefficients_, covariance_, columns, omitted, sigma2_)
            : term_sum_sq(coefficients_, xtx_inverse_, columns, omitted, 1);
        const auto mean_sq = sum_sq / static_cast<double>(df);
        const auto statistic = mean_sq / sigma2_;
        const auto p = std::isnan(statistic) ? statistic
            : detail::f_test_survival(statistic, static_cast<double>(df), static_cast<double>(degrees_of_freedom_));
        result.push_back({term.label, df, sum_sq, mean_sq, statistic, p});
    }
    result.push_back({"Residuals", degrees_of_freedom_, rss(), sigma2_});
    return result;
}

} // namespace datamunge::stats
