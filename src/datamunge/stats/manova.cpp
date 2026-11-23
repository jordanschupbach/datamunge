/// @file manova.cpp
/// @brief Hypothesis and error cross-products and four multivariate F tests.

#include <datamunge/stats/manova.hpp>

#include "linear_model_spaces.hpp"
#include <datamunge/linalg/svd.hpp>
#include <datamunge/stats/detail/test_common.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>

namespace datamunge::stats {
/// @brief Projection, covariance scaling and reference calculations.
namespace {

/// @brief Working response, hypothesis or error matrix.
using Matrix = linalg::DenseMatrix<long double>;
/// @brief Working response scales or generalized eigenvalues.
using Vector = std::vector<long double>;
/// @brief Public sample and cross-product matrix.
using InputMatrix = linalg::DenseMatrix<double>;

/// @brief Checks that all entries of a supplied matrix are finite.
bool finite(const InputMatrix& matrix) {
    return std::ranges::all_of(matrix.data(), [](auto value) { return std::isfinite(value); });
}

/// @brief Response data scaled independently by outcome.
struct ScaledResponse {
    Matrix values; ///< Normalized observations optionally shifted by a common baseline.
    Vector scales; ///< Multipliers for restoring response units.
};

/// @brief Converts response data to working precision with stable per-outcome scaling.
ScaledResponse scale_response(const InputMatrix& response, bool remove_baseline) {
    const auto n = response.rows(), p = response.cols();
    if (n > Vector{}.max_size() / p)
        throw std::length_error("MANOVA: response workspace exceeds storage capacity");
    ScaledResponse result{Matrix(n,p), Vector(p)};
    for (std::size_t j = 0; j < p; ++j) {
        const auto reference = remove_baseline ? static_cast<long double>(response(0,j)) : 0;
        for (std::size_t i = 0; i < n; ++i) {
            result.values(i,j) = response(i,j) - reference;
            result.scales[j] = std::max(result.scales[j], std::abs(result.values(i,j)));
        }
        if (result.scales[j] == 0)
            throw std::domain_error("MANOVA: zero residual outcome variance");
        for (std::size_t i = 0; i < n; ++i)
            result.values(i,j) /= result.scales[j];
    }
    return result;
}

/// @brief Detects the intercept in a reduced model so common offsets can be removed.
bool includes_intercept(const Matrix& reduced) {
    if (reduced.cols() == 0)
        return false;
    const auto projection = reduced * (reduced.transpose() * Vector(reduced.rows(),1));
    const auto tolerance = 64 * std::numeric_limits<long double>::epsilon() *
        std::sqrt(static_cast<long double>(reduced.rows()));
    return std::ranges::all_of(projection, [&](auto value) { return std::abs(1 - value) <= tolerance; });
}

/// @brief Restores a symmetric cross-product matrix in the original response units.
InputMatrix cross_product(const Matrix& values, const Vector& scales) {
    const auto p = values.cols();
    InputMatrix result(p,p);
    for (std::size_t j = 0; j < p; ++j) {
        for (std::size_t k = 0; k <= j; ++k) {
            long double entry = 0;
            for (std::size_t r = 0; r < values.rows(); ++r)
                entry += values(r,j) * values(r,k);
            entry *= scales[j] * scales[k];
            if (!std::isfinite(entry) || std::abs(entry) > std::numeric_limits<double>::max())
                throw std::overflow_error("MANOVA: cross-product exceeds the finite reporting range");
            result(j,k) = result(k,j) = static_cast<double>(entry);
        }
    }
    return result;
}

/// @brief Converts a nonnegative criterion or F value with a positive-infinity limit.
double statistic(long double value) {
    return value > std::numeric_limits<double>::max() ?
        std::numeric_limits<double>::infinity() : static_cast<double>(value);
}

/// @brief Builds a criterion and its F reference where both degrees of freedom are positive.
MANOVACriterion reference(long double criterion, long double f, long double df1, long double df2) {
    MANOVACriterion result{statistic(criterion),std::nullopt};
    if (df1 > 0 && df2 > 0) {
        const auto value = statistic(f), numerator = static_cast<double>(df1), denominator = static_cast<double>(df2);
        result.reference = MANOVAFReference{value,numerator,denominator,
            detail::f_test_survival(value,numerator,denominator)};
    }
    return result;
}

/// @brief Evaluates the four criteria using the usual F transformations.
void fill_criteria(MANOVAResult& result, const Vector& roots, std::size_t p) {
    const auto s = static_cast<long double>(roots.size());
    const auto dimension = static_cast<long double>(p), effect = static_cast<long double>(result.hypothesis_df);
    const auto error = static_cast<long double>(result.error_df);
    const auto larger = std::max(dimension,effect);
    long double trace = 0, log_wilks = 0, pillai = 0, pillai_complement = 0;
    for (const auto value : roots) {
        const auto denominator = 1 + value;
        trace += value;
        log_wilks -= std::log1p(value);
        pillai += value / denominator;
        pillai_complement += 1 / denominator;
    }
    const auto df1 = s*larger, pillai_df2 = s*(error-dimension+s);
    result.pillai = reference(pillai, (pillai_df2/df1)*(pillai/pillai_complement),df1,pillai_df2);

    const auto pd = dimension*effect, denominator = dimension*dimension + effect*effect - 5;
    const auto rao = denominator > 0 ? std::sqrt((pd*pd-4)/denominator) : 1.0L;
    const auto wilks_df2 = (error-(dimension-effect+1)/2)*rao-(pd-2)/2;
    result.wilks = reference(std::exp(log_wilks), std::expm1(-log_wilks/rao)*wilks_df2/pd,pd,wilks_df2);

    const auto hl_df2 = 2 + s*(error-dimension-1);
    result.hotelling_lawley = reference(trace, (hl_df2*trace)/(s*df1),df1,hl_df2);

    const auto roy_df2 = error-larger+effect;
    result.roy = reference(roots.front(),roy_df2*roots.front()/larger,larger,roy_df2);
}

/// @brief Forms cross-products and uses two singular value decompositions to obtain the covariance ratio roots.
MANOVAResult finalize(Matrix error, Matrix hypothesis, Vector scales, std::size_t effect_df, std::size_t error_df) {
    const auto n = error.rows(), p = error.cols();
    const auto negligible = 16 * std::numeric_limits<long double>::epsilon() * std::sqrt(static_cast<long double>(n));
    for (std::size_t j = 0; j < p; ++j) {
        long double scale = 0;
        for (std::size_t i = 0; i < n; ++i)
            scale = std::max(scale, std::abs(error(i,j)));
        if (scale <= negligible)
            throw std::domain_error("MANOVA: full model leaves no residual variation in an outcome");
        for (std::size_t i = 0; i < n; ++i)
            error(i,j) /= scale;
        for (std::size_t i = 0; i < hypothesis.rows(); ++i)
            hypothesis(i,j) /= scale;
        scales[j] *= scale;
    }
    const auto decomposition = linalg::svd(error);
    if (!decomposition.converged)
        throw std::runtime_error("MANOVA: residual factorization failed");
    if (decomposition.rank(n * std::numeric_limits<double>::epsilon()) != p)
        throw std::domain_error("MANOVA: singular full-model residual covariance");

    auto weighted = hypothesis * decomposition.V;
    for (std::size_t j = 0; j < p; ++j)
        for (std::size_t i = 0; i < weighted.rows(); ++i)
            weighted(i,j) /= decomposition.s[j];
    const auto effect = linalg::svd(weighted);
    if (!effect.converged)
        throw std::runtime_error("MANOVA: hypothesis factorization failed");
    MANOVAResult result{};
    result.hypothesis = cross_product(hypothesis,scales);
    result.error = cross_product(error,scales);
    result.hypothesis_df = effect_df;
    result.error_df = error_df;
    Vector roots(std::min(p,effect_df));
    for (std::size_t k = 0; k < roots.size(); ++k) {
        roots[k] = effect.s[k]*effect.s[k];
        result.roots.push_back(statistic(roots[k]));
    }
    fill_criteria(result,roots,p);
    return result;
}

} // namespace

MANOVAResult manova(const linalg::DenseMatrix<double>& response, const linalg::DenseMatrix<double>& full_design,
    const linalg::DenseMatrix<double>& reduced_design) {
    const auto n = response.rows(), p = response.cols(), full_rank = full_design.cols(), reduced_rank = reduced_design.cols();
    if (p == 0 || full_design.rows() != n || reduced_design.rows() != n || full_rank >= n ||
        reduced_rank >= full_rank || n-full_rank < p || !finite(response) || !finite(full_design) || !finite(reduced_design))
        throw std::invalid_argument("MANOVA: incompatible models, nonfinite input or insufficient residual degrees of freedom");
    const auto full = detail::orthonormal_design(full_design,"MANOVA");
    const auto reduced = detail::orthonormal_design(reduced_design,"MANOVA");
    const auto nested = detail::nested_coordinates(full,reduced,"MANOVA");
    auto scaled = scale_response(response, includes_intercept(reduced));
    const auto fitted_full = full.transpose() * scaled.values;
    auto error = scaled.values - full*fitted_full;
    auto hypothesis = fitted_full - nested*(nested.transpose()*fitted_full);
    return finalize(std::move(error),std::move(hypothesis),std::move(scaled.scales),full_rank-reduced_rank,n-full_rank);
}

MANOVAResult one_way_manova(const linalg::DenseMatrix<double>& response, const std::vector<std::int64_t>& group) {
    const auto n = response.rows(), p = response.cols();
    if (p == 0 || n != group.size() || !finite(response))
        throw std::invalid_argument("MANOVA: incompatible groups or nonfinite response");
    std::map<std::int64_t, std::size_t> index;
    std::vector<std::size_t> counts, positions;
    positions.reserve(n);
    for (const auto label : group) {
        const auto [row, created] = index.try_emplace(label,index.size());
        if (created)
            counts.push_back(0);
        positions.push_back(row->second);
        ++counts[row->second];
    }
    const auto levels = counts.size();
    if (levels < 2 || n-levels < p)
        throw std::invalid_argument("MANOVA: insufficient groups or residual degrees of freedom");
    auto scaled = scale_response(response,true);
    Matrix means(levels,p);
    Vector grand_mean(p);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < p; ++j) {
            means(positions[i],j) += scaled.values(i,j);
            grand_mean[j] += scaled.values(i,j);
        }
    for (auto& value : grand_mean)
        value /= n;
    for (std::size_t k = 0; k < levels; ++k)
        for (std::size_t j = 0; j < p; ++j)
            means(k,j) /= counts[k];
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < p; ++j)
            scaled.values(i,j) -= means(positions[i],j);
    for (std::size_t k = 0; k < levels; ++k) {
        const auto size = std::sqrt(static_cast<long double>(counts[k]));
        for (std::size_t j = 0; j < p; ++j)
            means(k,j) = size*(means(k,j)-grand_mean[j]);
    }
    return finalize(std::move(scaled.values),std::move(means),std::move(scaled.scales),levels-1,n-levels);
}

} // namespace datamunge::stats
