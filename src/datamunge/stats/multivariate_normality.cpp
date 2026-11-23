/// @file multivariate_normality.cpp
/// @brief Mardia and Henze–Zirkler tests on whitened centered observations.

#include <datamunge/stats/multivariate_normality.hpp>

#include <datamunge/linalg/svd.hpp>
#include <datamunge/stats/detail/test_common.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace datamunge::stats {
/// @brief Common data whitening and pairwise moments.
namespace {

/// @brief Working precision for centered values and whitened scores.
using Matrix = linalg::DenseMatrix<long double>;

/// @brief Whitens using empirical covariance with divisor equal to the sample size.
Matrix whiten(const linalg::DenseMatrix<double>& observations) {
    const auto n = observations.rows(), p = observations.cols();
    if (p == 0 || n <= p || n-p < 2 ||
        !std::ranges::all_of(observations.data(), [](auto value) { return std::isfinite(value); }))
        throw std::invalid_argument("multivariate normality: require finite data and at least two more rows than columns");
    if (n > std::vector<long double>{}.max_size()/p)
        throw std::length_error("multivariate normality: dense workspace exceeds storage capacity");

    Matrix centered(n,p);
    for (std::size_t j = 0; j < p; ++j) {
        const auto baseline = static_cast<long double>(observations(0,j));
        long double mean = 0, peak = 0;
        for (std::size_t i = 0; i < n; ++i) {
            centered(i,j) = observations(i,j)-baseline;
            mean += centered(i,j);
        }
        mean /= n;
        for (std::size_t i = 0; i < n; ++i) {
            centered(i,j) -= mean;
            peak = std::max(peak,std::abs(centered(i,j)));
        }
        if (peak == 0)
            throw std::domain_error("multivariate normality: constant feature");
        for (std::size_t i = 0; i < n; ++i)
            centered(i,j) /= peak;
    }
    auto factor = linalg::svd(centered);
    if (!factor.converged)
        throw std::runtime_error("multivariate normality: centered factorization failed");
    if (factor.rank(n*std::numeric_limits<double>::epsilon()) != p)
        throw std::domain_error("multivariate normality: singular covariance");
    const auto factor_n = std::sqrt(static_cast<long double>(n));
    for (auto& value : factor.U.data())
        value *= factor_n;
    return factor.U;
}

/// @brief Dot product of two whitened rows.
long double dot(const Matrix& z, std::size_t i, std::size_t j) {
    long double value = 0;
    for (std::size_t k = 0; k < z.cols(); ++k)
        value += z(i,k)*z(j,k);
    return value;
}

/// @brief Squared distance of two whitened rows, evaluated without subtracting inner products.
long double distance(const Matrix& z, std::size_t i, std::size_t j) {
    long double value = 0;
    for (std::size_t k = 0; k < z.cols(); ++k) {
        const auto difference = z(i,k)-z(j,k);
        value += difference*difference;
    }
    return value;
}

} // namespace

MardiaTestResult mardia_test(const linalg::DenseMatrix<double>& observations) {
    const auto z = whiten(observations);
    const auto n = static_cast<long double>(z.rows()), p = static_cast<long double>(z.cols());
    long double skewness = 0, kurtosis = 0;
    for (std::size_t i = 0; i < z.rows(); ++i) {
        const auto squared_length = dot(z,i,i);
        skewness += squared_length*squared_length*squared_length;
        kurtosis += squared_length*squared_length;
        for (std::size_t j = 0; j < i; ++j) {
            const auto product = dot(z,i,j);
            skewness += 2*product*product*product;
        }
    }
    skewness = std::max(0.0L, skewness/n/n);
    kurtosis /= n;
    const auto correction = n < 20 ? (p+1)*(n+1)*(n+3)/(n*((n+1)*(p+1)-6)) : 1.0L;
    const auto df = static_cast<double>(p*(p+1)*(p+2)/6);
    const auto skew = static_cast<double>(n*correction*skewness/6);
    const auto kurt = static_cast<double>((kurtosis-p*(p+2))*std::sqrt(n/(8*p*(p+2))));
    return {static_cast<double>(skewness),static_cast<double>(kurtosis),skew,df,
        detail::chi_squared_test_survival(skew,df),kurt,detail::normal_test_p(kurt,Alternative::TwoSided)};
}

HenzeZirklerTestResult henze_zirkler_test(const linalg::DenseMatrix<double>& observations) {
    const auto z = whiten(observations);
    const auto n = static_cast<long double>(z.rows()), p = static_cast<long double>(z.cols());
    const auto b2 = .5L*std::pow(n*(2*p+1)/4,2/(p+4));
    const auto a = 1+2*b2, wb = (1+b2)*(1+3*b2);
    const auto d = b2/(2*(1+b2));
    long double pairs = n, single = 0;
    for (std::size_t i = 0; i < z.rows(); ++i) {
        single += std::exp(-d*dot(z,i,i));
        for (std::size_t j = 0; j < i; ++j)
            pairs += 2*std::exp(-b2*distance(z,i,j)/2);
    }
    const auto statistic = std::max(0.0L,pairs/n-2*std::pow(1+b2,-p/2)*single+n*std::pow(a,-p/2));

    const auto u = b2/a, v = b2*(b2/wb);
    const auto mean = 1-std::pow(a,-p/2)*(1+p*u+p*(p+2)*u*u/2);
    const auto variance = 2*std::pow(1+4*b2,-p/2) +
        2*std::pow(a,-p)*(1+2*p*u*u+3*p*(p+2)*u*u*u*u/4) -
        4*std::pow(wb,-p/2)*(1+3*p*v/2+p*(p+2)*v*v/2);
    if (!(mean > 0 && variance > 0))
        throw std::runtime_error("Henze-Zirkler: lognormal reference moments could not be resolved");
    const auto variance_log = std::log1p(variance/mean/mean);
    const auto mean_log = std::log(mean)-variance_log/2, sd_log = std::sqrt(variance_log);
    const auto z_statistic = static_cast<double>((std::log(statistic)-mean_log)/sd_log);
    return {static_cast<double>(statistic),static_cast<double>(std::sqrt(b2)),static_cast<double>(mean_log),
        static_cast<double>(sd_log),detail::normal_test_p(z_statistic,Alternative::Greater)};
}

} // namespace datamunge::stats
