/// @file hausman_test.cpp
/// @brief Covariance-difference specification test on matched, estimable coefficients.

#include <datamunge/stats/hausman_test.hpp>

#include <datamunge/linalg/eigen.hpp>
#include <datamunge/stats/detail/test_common.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace datamunge::stats {
/// @brief Validation and stable standardized covariance calculations.
namespace {

/// @brief Working covariance matrix in extended precision.
using Matrix = linalg::DenseMatrix<long double>;
/// @brief Extended-precision coefficient coordinates.
using Vector = std::vector<long double>;
/// @brief Name-to-column index over a supplied model.
using TermIndex = std::map<std::string_view, std::size_t>;
/// @brief Corresponding fixed and random model coordinates.
using Matches = std::vector<std::pair<std::size_t, std::size_t>>;

/// @brief Validates model shapes, names and covariance symmetry.
TermIndex index_terms(const HausmanEstimates& fit) {
    const auto count = fit.coefficients.size();
    const auto finite = [](double value) { return std::isfinite(value); };
    if (count == 0 || fit.terms.size() != count || fit.covariance.rows() != count || fit.covariance.cols() != count ||
        !std::ranges::all_of(fit.coefficients, finite) || !std::ranges::all_of(fit.covariance.data(), finite))
        throw std::invalid_argument("Hausman test: invalid model coefficients or covariance");
    TermIndex index;
    for (std::size_t i = 0; i < count; ++i) {
        if (fit.terms[i].empty() || !index.emplace(fit.terms[i], i).second)
            throw std::invalid_argument("Hausman test: terms must be nonempty and unique within a model");
        if (fit.covariance(i,i) < 0)
            throw std::domain_error("Hausman test: a model covariance has a negative variance");
    }
    for (std::size_t i = 0; i < count; ++i)
        for (std::size_t j = 0; j < i; ++j) {
            const long double a = fit.covariance(i,j), b = fit.covariance(j,i);
            const long double diagonal_scale =
                std::sqrt(static_cast<long double>(fit.covariance(i,i))) * std::sqrt(static_cast<long double>(fit.covariance(j,j)));
            if (std::abs(a - b) > 64 * std::numeric_limits<double>::epsilon() *
                std::max({std::abs(a), std::abs(b), diagonal_scale}))
                throw std::invalid_argument("Hausman test: a model covariance is not symmetric");
        }
    return index;
}

/// @brief Extracts and symmetrizes one shared covariance entry.
long double covariance_entry(const HausmanEstimates& model, std::size_t i, std::size_t j) {
    return std::midpoint(static_cast<long double>(model.covariance(i,j)), static_cast<long double>(model.covariance(j,i)));
}

/// @brief Checks a model covariance on the shared subspace and returns its largest eigenvalue.
long double psd_scale(const Matrix& covariance, long double precision) {
    const auto eigen = linalg::symmetric_eigen(covariance);
    const auto scale = std::max(1.0L, eigen.eigenvalues.front());
    if (eigen.eigenvalues.back() < -precision * scale)
        throw std::domain_error("Hausman test: a model covariance is indefinite on the common terms");
    return scale;
}

/// @brief Converts a returned difference after checking its finite double range.
double output(long double value) {
    if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<double>::max())
        throw std::overflow_error("Hausman test: difference exceeds the finite reporting range");
    return static_cast<double>(value);
}

} // namespace

HausmanTestResult hausman_test(const HausmanEstimates& fixed_effects,
    const HausmanEstimates& random_effects, double relative_tolerance) {
    if (!(relative_tolerance > 0 && relative_tolerance < 1))
        throw std::invalid_argument("Hausman test: relative tolerance must lie strictly between zero and one");
    index_terms(fixed_effects);
    const auto random_index = index_terms(random_effects);
    Matches shared;
    HausmanTestResult result{};
    for (std::size_t i = 0; i < fixed_effects.terms.size(); ++i) {
        const auto& term = fixed_effects.terms[i];
        const auto match = random_index.find(term);
        if (match != random_index.end()) {
            result.terms.push_back(term);
            shared.emplace_back(i, match->second);
        }
    }
    const auto n = shared.size();
    if (n == 0)
        throw std::invalid_argument("Hausman test: models have no common terms");
    if (n > Vector{}.max_size() / n)
        throw std::length_error("Hausman test: covariance workspace exceeds storage capacity");

    Matrix fixed(n,n), random(n,n), difference(n,n);
    Vector scales(n), standardized_difference(n), untested(n);
    result.difference.resize(n);
    result.covariance_difference = linalg::DenseMatrix<double>(n,n);
    result.untested_difference.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        const auto [f, r] = shared[i];
        const auto variance = std::max(fixed_effects.covariance(f,f), random_effects.covariance(r,r));
        scales[i] = variance > 0 ? std::sqrt(static_cast<long double>(variance)) : 1;
        const auto delta = static_cast<long double>(fixed_effects.coefficients[f]) - random_effects.coefficients[r];
        result.difference[i] = output(delta);
        standardized_difference[i] = delta / scales[i];
    }
    for (std::size_t i = 0; i < n; ++i) {
        const auto [f, r] = shared[i];
        for (std::size_t j = 0; j <= i; ++j) {
            const auto [other_f, other_r] = shared[j];
            const auto a = covariance_entry(fixed_effects, f, other_f);
            const auto b = covariance_entry(random_effects, r, other_r);
            const auto scale = scales[i] * scales[j];
            fixed(i,j) = fixed(j,i) = a / scale;
            random(i,j) = random(j,i) = b / scale;
            difference(i,j) = difference(j,i) = (a - b) / scale;
        }
    }

    const auto precision = 64.0L * n * std::numeric_limits<double>::epsilon();
    const auto scale = std::max(psd_scale(fixed, precision), psd_scale(random, precision));
    const auto cutoff = std::max(precision, static_cast<long double>(relative_tolerance)) * scale;
    const auto eigen = linalg::symmetric_eigen(difference);
    if (eigen.eigenvalues.back() < -cutoff)
        throw std::domain_error("Hausman test: covariance difference is indefinite");
    long double statistic = 0;
    for (std::size_t mode = 0; mode < n; ++mode) {
        const auto value = eigen.eigenvalues[mode];
        long double projection = 0;
        for (std::size_t i = 0; i < n; ++i)
            projection += eigen.eigenvectors(i,mode) * standardized_difference[i];
        if (value > cutoff) {
            statistic += projection * projection / value;
            ++result.df;
        } else {
            for (std::size_t i = 0; i < n; ++i)
                untested[i] += projection * eigen.eigenvectors(i,mode);
        }
    }
    if (result.df == 0)
        throw std::domain_error("Hausman test: covariance difference has zero numerical rank");

    for (std::size_t i = 0; i < n; ++i) {
        result.untested_difference[i] = output(untested[i] * scales[i]);
        for (std::size_t j = 0; j <= i; ++j)
            result.covariance_difference(i,j) = result.covariance_difference(j,i) =
                output(difference(i,j) * scales[i] * scales[j]);
    }
    result.statistic = statistic > std::numeric_limits<double>::max() ?
        std::numeric_limits<double>::infinity() : static_cast<double>(statistic);
    result.p_value = detail::chi_squared_test_survival(result.statistic, result.df);
    return result;
}

} // namespace datamunge::stats
