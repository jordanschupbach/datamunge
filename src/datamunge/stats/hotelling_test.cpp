/// @file hotelling_test.cpp
/// @brief Stable covariance-weighted multivariate mean comparisons.

#include <datamunge/stats/hotelling_test.hpp>

#include <datamunge/linalg/svd.hpp>
#include <datamunge/stats/detail/test_common.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace datamunge::stats {
/// @brief Centering and scaled Mahalanobis calculations shared by the three tests.
namespace {

/// @brief Extended precision for differences of observations and their means.
using Vector = std::vector<long double>;
/// @brief Centered observations before forming the covariance-weighted statistic.
using Matrix = linalg::DenseMatrix<long double>;
/// @brief Public input sample matrix.
using InputMatrix = linalg::DenseMatrix<double>;

/// @brief Checks the finite sample matrix and the requested mean-test target.
void check_sample(const InputMatrix& input, const std::vector<double>& target) {
    const auto finite = [](double value) { return std::isfinite(value); };
    if (input.rows() < 2 || input.cols() == 0 ||
        (!target.empty() && target.size() != input.cols()) ||
        !std::ranges::all_of(input.data(), finite) || !std::ranges::all_of(target, finite))
        throw std::invalid_argument("Hotelling T-squared: invalid sample or null vector");
}

/// @brief Allocates a centered sample only after checking its dimensions.
Matrix workspace(std::size_t rows, std::size_t cols, std::size_t covariance_df) {
    if (covariance_df < cols)
        throw std::invalid_argument("Hotelling T-squared: insufficient covariance degrees of freedom");
    if (rows > Vector{}.max_size() / cols)
        throw std::length_error("Hotelling T-squared: centered matrix exceeds storage capacity");
    return Matrix(rows, cols);
}

/// @brief Centers one block using deviations from its first observation and returns its mean offset.
template<class Deviation>
Vector center(Matrix& storage, std::size_t first, std::size_t count, const Deviation& deviation) {
    Vector offset(storage.cols());
    for (std::size_t j = 0; j < storage.cols(); ++j) {
        for (std::size_t i = 0; i < count; ++i) {
            storage(first+i,j) = deviation(i,j);
            offset[j] += storage(first+i,j);
        }
        offset[j] /= count;
        for (std::size_t i = 0; i < count; ++i)
            storage(first+i,j) -= offset[j];
    }
    return offset;
}

/// @brief Computes the statistic using singular values of the centered sample instead of its squared condition number.
HotellingTestResult compare(Matrix centered, const Vector& reference, const Vector& offset,
    const std::vector<double>& null_difference, std::size_t covariance_df, long double effective_n) {
    const auto p = centered.cols(), den_df = covariance_df - p + 1;
    HotellingTestResult result{0,0,p,den_df,0,std::vector<double>(p)};
    Vector difference(p);
    for (std::size_t j = 0; j < p; ++j) {
        const auto estimate = reference[j] + offset[j];
        if (!std::isfinite(estimate) || std::abs(estimate) > std::numeric_limits<double>::max())
            throw std::overflow_error("Hotelling T-squared: mean difference exceeds the finite reporting range");
        result.mean_difference[j] = static_cast<double>(estimate);
        long double scale = 0;
        for (std::size_t i = 0; i < centered.rows(); ++i)
            scale = std::max(scale, std::abs(centered(i,j)));
        if (scale == 0)
            throw std::domain_error("Hotelling T-squared: covariance has a constant coordinate");
        for (std::size_t i = 0; i < centered.rows(); ++i)
            centered(i,j) /= scale;
        const auto target = null_difference.empty() ? 0.0 : null_difference[j];
        difference[j] = ((reference[j] - target) + offset[j]) / scale;
    }
    const auto decomposition = linalg::svd(centered);
    const auto tolerance = std::max(centered.rows(), p) * std::numeric_limits<double>::epsilon();
    if (!decomposition.converged)
        throw std::runtime_error("Hotelling T-squared: centered sample factorization failed");
    if (decomposition.rank(tolerance) != p)
        throw std::domain_error("Hotelling T-squared: singular sample covariance");

    long double quadratic = 0;
    for (std::size_t k = 0; k < p; ++k) {
        long double projection = 0;
        for (std::size_t j = 0; j < p; ++j)
            projection += decomposition.V(j,k) * difference[j];
        const auto standardized = projection / decomposition.s[k];
        quadratic += standardized * standardized;
    }
    const auto t_squared = (effective_n * covariance_df) * quadratic;
    const auto f = (effective_n * den_df / p) * quadratic;
    /// @brief Saturates a nonnegative test statistic at positive infinity on overflow.
    const auto statistic = [](long double value) {
        return value > std::numeric_limits<double>::max() ?
            std::numeric_limits<double>::infinity() : static_cast<double>(value);
    };
    result.t_squared = statistic(t_squared);
    result.f_statistic = statistic(f);
    result.p_value = detail::f_test_survival(result.f_statistic, p, den_df);
    return result;
}

} // namespace

HotellingTestResult hotelling_t2_one_sample(const linalg::DenseMatrix<double>& observations,
    const std::vector<double>& null_mean) {
    check_sample(observations, null_mean);
    const auto n = observations.rows(), p = observations.cols();
    auto centered = workspace(n,p,n-1);
    Vector reference(p);
    for (std::size_t j = 0; j < p; ++j)
        reference[j] = observations(0,j);
    const auto mean_offset = center(centered,0,n,[&](auto i, auto j) {
        return observations(i,j) - reference[j];
    });
    return compare(std::move(centered), reference, mean_offset, null_mean, n-1, n);
}

HotellingTestResult hotelling_t2_paired(const linalg::DenseMatrix<double>& first,
    const linalg::DenseMatrix<double>& second, const std::vector<double>& null_difference) {
    check_sample(first, null_difference);
    check_sample(second, null_difference);
    const auto n = first.rows(), p = first.cols();
    if (second.rows() != n || second.cols() != p)
        throw std::invalid_argument("Hotelling T-squared: incompatible paired samples");
    auto centered = workspace(n,p,n-1);
    Vector reference(p);
    for (std::size_t j = 0; j < p; ++j)
        reference[j] = static_cast<long double>(first(0,j)) - second(0,j);
    const auto mean_offset = center(centered,0,n,[&](auto i, auto j) {
        return (static_cast<long double>(first(i,j)) - first(0,j)) -
               (static_cast<long double>(second(i,j)) - second(0,j));
    });
    return compare(std::move(centered), reference, mean_offset, null_difference, n-1, n);
}

HotellingTestResult hotelling_t2_two_sample(const linalg::DenseMatrix<double>& first,
    const linalg::DenseMatrix<double>& second, const std::vector<double>& null_difference) {
    check_sample(first, null_difference);
    check_sample(second, null_difference);
    const auto n1 = first.rows(), n2 = second.rows(), p = first.cols();
    if (second.cols() != p)
        throw std::invalid_argument("Hotelling T-squared: incompatible independent samples");
    if (n1 > std::numeric_limits<std::size_t>::max() - n2)
        throw std::length_error("Hotelling T-squared: total sample size exceeds the index range");
    const auto n = n1+n2;
    auto centered = workspace(n,p,n-2);
    Vector reference(p);
    for (std::size_t j = 0; j < p; ++j)
        reference[j] = static_cast<long double>(first(0,j)) - second(0,j);
    auto mean_offset = center(centered,0,n1,[&](auto i, auto j) {
        return static_cast<long double>(first(i,j)) - first(0,j);
    });
    const auto other = center(centered,n1,n2,[&](auto i, auto j) {
        return static_cast<long double>(second(i,j)) - second(0,j);
    });
    for (std::size_t j = 0; j < p; ++j)
        mean_offset[j] -= other[j];
    const auto effective_n = static_cast<long double>(n1) * n2 / n;
    return compare(std::move(centered), reference, mean_offset, null_difference, n-2, effective_n);
}

} // namespace datamunge::stats
