/// @file lm_covariance.cpp
/// @brief Heteroscedastic, serial-correlation and cluster covariance for fitted linear models.

#include <datamunge/stats/lm.hpp>

#include "cluster_covariance.hpp"

#include <bit>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace datamunge::stats {

/// @brief Accumulation and conversion for coefficient-space sandwich sums.
namespace {

/// @brief Working precision for signed sums of cross-products.
using CovarianceSum = linalg::DenseMatrix<long double>;

/// @brief Adds the symmetric part of a weighted outer product to the lower triangle.
void add_cross_product(CovarianceSum& sum, const std::vector<long double>& a,
                       const std::vector<long double>& b, long double multiplier) {
    for (std::size_t i = 0; i < a.size(); ++i)
        for (std::size_t j = 0; j <= i; ++j)
            sum(i, j) += multiplier * std::midpoint(a[i] * b[j], b[i] * a[j]);
}

/// @brief Mirrors the covariance triangle and checks its finite output range.
linalg::DenseMatrix<double> finish_covariance(const CovarianceSum& sum, long double multiplier = 1) {
    linalg::DenseMatrix<double> result(sum.rows(), sum.cols());
    for (std::size_t i = 0; i < sum.rows(); ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            const long double value = multiplier * sum(i, j);
            if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<double>::max())
                throw std::overflow_error("LM covariance: entry exceeds finite double range");
            result(i, j) = result(j, i) = static_cast<double>(value);
        }
    }
    return result;
}

/// @brief Working score totals for an observed joint cluster.
struct ClusterScore {
    std::vector<long double> values; ///< Sum of coefficient-space scores.
    std::size_t observations{};      ///< Number of contributing fitted rows.
};

} // namespace

std::vector<std::vector<long double>> LM::sandwich_scores(HcCovariance estimator) const {
    const auto p = rank();
    const bool leverage_adjusted = estimator == HcCovariance::HC2 || estimator == HcCovariance::HC3;
    std::vector<std::vector<long double>> scores(observations(), std::vector<long double>(p));
    for (std::size_t i = 0; i < observations(); ++i) {
        auto& score = scores[i];
        long double leverage = 0;
        if (sparse_design_matrix_) {
            const auto& design = *sparse_design_matrix_;
            const auto first = design.row_ptr[i];
            const auto last = design.row_ptr[i + 1];
            for (std::size_t j = 0; j < p; ++j)
                for (std::size_t k = first; k < last; ++k)
                    score[j] += static_cast<long double>(xtx_inverse_(j, design.col_idx[k])) * design.values[k];
            if (leverage_adjusted)
                for (std::size_t k = first; k < last; ++k)
                    leverage += score[design.col_idx[k]] * design.values[k];
        } else {
            for (std::size_t j = 0; j < p; ++j)
                for (std::size_t k = 0; k < p; ++k)
                    score[j] += static_cast<long double>(xtx_inverse_(j, k)) * design_matrix_(i, k);
            if (leverage_adjusted)
                for (std::size_t j = 0; j < p; ++j)
                    leverage += score[j] * design_matrix_(i, j);
        }
        long double residual = residuals_[i];
        if (!weights_.empty())
            residual *= std::sqrt(weights_[i]);
        if (leverage_adjusted) {
            const long double remaining = 1 - leverage;
            if (!(remaining > 16 * std::numeric_limits<double>::epsilon()))
                throw std::domain_error("LM covariance: leverage equals one to working precision");
            residual /= estimator == HcCovariance::HC2 ? std::sqrt(remaining) : remaining;
        }
        for (auto& value : score)
            value *= residual;
    }
    return scores;
}

linalg::DenseMatrix<double> LM::heteroscedastic_covariance(HcCovariance estimator) const {
    if (estimator != HcCovariance::HC0 && estimator != HcCovariance::HC1 &&
        estimator != HcCovariance::HC2 && estimator != HcCovariance::HC3)
        throw std::invalid_argument("LM covariance: unrecognized HC estimator");
    const auto scores = sandwich_scores(estimator);
    CovarianceSum sum(rank(), rank());
    for (const auto& score : scores)
        add_cross_product(sum, score, score, 1);
    const long double multiplier = estimator == HcCovariance::HC1
        ? static_cast<long double>(observations()) / degrees_of_freedom_ : 1;
    return finish_covariance(sum, multiplier);
}

linalg::DenseMatrix<double> LM::newey_west_covariance(std::size_t max_lag, bool adjust_degrees_of_freedom) const {
    if (max_lag >= observations())
        throw std::invalid_argument("LM Newey-West covariance: lag must be less than fitted row count");
    if (max_lag == 0)
        return heteroscedastic_covariance(adjust_degrees_of_freedom ? HcCovariance::HC1 : HcCovariance::HC0);
    const auto scores = sandwich_scores(HcCovariance::HC0);
    CovarianceSum sum(rank(), rank());
    for (const auto& score : scores)
        add_cross_product(sum, score, score, 1);
    for (std::size_t lag = 1; lag <= max_lag; ++lag) {
        const long double weight = 1 - static_cast<long double>(lag) / (max_lag + 1);
        for (std::size_t i = lag; i < scores.size(); ++i)
            add_cross_product(sum, scores[i], scores[i - lag], 2 * weight);
    }
    const long double multiplier = adjust_degrees_of_freedom
        ? static_cast<long double>(observations()) / degrees_of_freedom_ : 1;
    return finish_covariance(sum, multiplier);
}

ClusterCovarianceResult LM::cluster_robust_covariance(
    const std::vector<std::vector<std::int64_t>>& labels, bool finite_sample_adjustment) const {
    return detail::cluster_covariance_from_scores(sandwich_scores(HcCovariance::HC0), labels, degrees_of_freedom_,
        finite_sample_adjustment);
}

ClusterCovarianceResult detail::cluster_covariance_from_scores(
    const std::vector<std::vector<long double>>& scores, const std::vector<std::vector<std::int64_t>>& labels,
    std::size_t residual_df, bool finite_sample_adjustment) {
    if (labels.empty())
        throw std::invalid_argument("LM cluster covariance: specify at least one clustering dimension");
    for (const auto& dimension : labels)
        if (dimension.size() != scores.size())
            throw std::invalid_argument("LM cluster covariance: labels must match the fitted rows");
    if (labels.size() >= std::numeric_limits<std::size_t>::digits)
        throw std::length_error("LM cluster covariance: too many subsets for the index range");

    const auto rank = scores.front().size();
    ClusterCovarianceResult result;
    CovarianceSum sum(rank, rank);
    const std::size_t subsets = std::size_t{1} << labels.size();
    for (std::size_t mask = 1; mask < subsets; ++mask) {
        auto& component = result.components.emplace_back();
        for (std::size_t axis = 0; axis < labels.size(); ++axis)
            if ((mask >> axis) & 1)
                component.dimensions.push_back(axis);

        std::map<std::vector<std::int64_t>, ClusterScore> groups;
        std::vector<std::int64_t> key(component.dimensions.size());
        for (std::size_t i = 0; i < scores.size(); ++i) {
            for (std::size_t j = 0; j < key.size(); ++j)
                key[j] = labels[component.dimensions[j]][i];
            auto [group, added] = groups.try_emplace(key);
            auto& total = group->second;
            if (added)
                total.values.resize(rank);
            ++total.observations;
            for (std::size_t j = 0; j < rank; ++j)
                total.values[j] += scores[i][j];
        }

        long double multiplier = std::popcount(mask) % 2 == 0 ? -1 : 1;
        if (finite_sample_adjustment) {
            if (groups.size() < 2)
                throw std::domain_error("LM cluster covariance: adjusted components need at least two groups");
            multiplier *= static_cast<long double>(groups.size()) / (groups.size() - 1)
                        * static_cast<long double>(scores.size() - 1) / residual_df;
        }
        component.multiplier = static_cast<double>(multiplier);
        component.groups.reserve(groups.size());
        for (const auto& [group_labels, total] : groups) {
            add_cross_product(sum, total.values, total.values, multiplier);
            component.groups.push_back({group_labels, total.observations});
        }
    }
    result.covariance = finish_covariance(sum);
    return result;
}

} // namespace datamunge::stats
