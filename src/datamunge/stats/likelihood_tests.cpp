/// @file likelihood_tests.cpp
/// @brief Pseudo-inverse quadratic forms and chi-square tails for the three classical likelihood tests.

#include <datamunge/stats/likelihood_tests.hpp>

#include <datamunge/linalg/eigen.hpp>
#include <datamunge/random/distributions.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace datamunge::stats {
/// @brief Rank-revealing quadratic forms and chi-square tails.
namespace {

/// @brief Dense double matrix used by the helpers.
using Matrix = linalg::DenseMatrix<double>;

/// @brief \f$x^\top A^{+}x\f$ and the numerical rank of a symmetric positive semidefinite A.
/// @param a Symmetric matrix.
/// @param x Vector.
/// @param rank Set to the number of eigenvalues above \f$10^{-10}\f$ times the largest.
/// @return Quadratic form with the Moore–Penrose inverse.
double pseudo_quadratic(const Matrix& a, const std::vector<double>& x, std::size_t& rank) {
    const auto eig = linalg::symmetric_eigen(a);
    const double top = eig.eigenvalues.empty() ? 0 : std::max(0.0,eig.eigenvalues.front());
    double q = 0;
    rank = 0;
    for (std::size_t k = 0; k < eig.eigenvalues.size(); ++k) {
        if (!(top > 0) || eig.eigenvalues[k] <= 1e-10*top)
            continue;
        ++rank;
        double projection = 0;
        for (std::size_t i = 0; i < x.size(); ++i)
            projection += eig.eigenvectors(i,k)*x[i];
        q += projection*projection/eig.eigenvalues[k];
    }
    return q;
}

/// @brief Upper chi-square tail.
/// @param statistic Nonnegative statistic.
/// @param df Degrees of freedom.
/// @return \f$P(\chi^2_{df}\ge\) statistic\f$)\f$.
double upper_tail(double statistic, std::size_t df) {
    return std::clamp(1-random::chi_squared_cdf(statistic,static_cast<double>(df)),0.0,1.0);
}

} // namespace

RestrictionTestResult wald_test(const MaximumLikelihoodFit& fit, const linalg::DenseMatrix<double>& restrictions,
                                const std::vector<double>& values, bool robust) {
    const std::size_t p = fit.estimate.size(), q = restrictions.rows();
    const Matrix& v = robust ? fit.robust_covariance : fit.covariance;
    if (p == 0 || q == 0 || restrictions.cols() != p || values.size() != q || v.rows() != p || v.cols() != p)
        throw std::invalid_argument("wald_test: need a fitted model, a q x p restriction matrix and q values");
    if (!std::ranges::all_of(restrictions.data(),[](double x) { return std::isfinite(x); }) ||
        !std::ranges::all_of(values,[](double x) { return std::isfinite(x); }) ||
        !std::ranges::all_of(v.data(),[](double x) { return std::isfinite(x); }))
        throw std::invalid_argument("wald_test: restrictions, values and covariance must be finite");
    auto discrepancy = restrictions*fit.estimate;
    for (std::size_t k = 0; k < q; ++k)
        discrepancy[k] -= values[k];
    Matrix middle = restrictions*v*restrictions.transpose();
    for (std::size_t r = 0; r < q; ++r)
        for (std::size_t c = 0; c < r; ++c)
            middle(r,c) = middle(c,r) = .5*(middle(r,c)+middle(c,r));
    RestrictionTestResult result;
    result.statistic = pseudo_quadratic(middle,discrepancy,result.rank);
    if (result.rank == 0)
        throw std::invalid_argument("wald_test: restrictions have rank zero");
    result.p_value = upper_tail(result.statistic,result.rank);
    result.method = robust ? "Wald test (sandwich covariance)" : "Wald test (observed information)";
    return result;
}

RestrictionTestResult score_test(const MaximumLikelihoodFit& at_restricted, std::size_t rank) {
    const std::size_t p = at_restricted.estimate.size();
    const auto& scores = at_restricted.score_contributions;
    if (rank == 0 || rank > p || scores.cols() != p || at_restricted.observed_information.rows() != p)
        throw std::invalid_argument("score_test: rank must be in 1..p for an evaluated likelihood");
    if (!std::ranges::all_of(at_restricted.covariance.data(),[](double x) { return std::isfinite(x); }))
        throw std::invalid_argument("score_test: observed information is singular");
    std::vector<double> u(p,0.0);
    for (std::size_t i = 0; i < scores.rows(); ++i)
        for (std::size_t j = 0; j < p; ++j)
            u[j] += scores(i,j);
    const auto ju = at_restricted.covariance*u;
    RestrictionTestResult result;
    for (std::size_t j = 0; j < p; ++j)
        result.statistic += u[j]*ju[j];
    result.statistic = std::max(0.0,result.statistic);
    result.rank = rank;
    result.p_value = upper_tail(result.statistic,rank);
    result.method = "Rao score test (observed information)";
    return result;
}

RestrictionTestResult likelihood_ratio_test(double full_log_likelihood, double restricted_log_likelihood,
                                            std::size_t rank) {
    if (!std::isfinite(full_log_likelihood) || !std::isfinite(restricted_log_likelihood) || rank == 0)
        throw std::invalid_argument("likelihood_ratio_test: need finite log likelihoods and a positive rank");
    const double difference = full_log_likelihood-restricted_log_likelihood;
    if (difference < -1e-6*(1+std::abs(full_log_likelihood)))
        throw std::invalid_argument("likelihood_ratio_test: restricted log likelihood exceeds the full one");
    RestrictionTestResult result;
    result.statistic = std::max(0.0,2*difference);
    result.rank = rank;
    result.p_value = upper_tail(result.statistic,rank);
    result.method = "Likelihood-ratio test";
    return result;
}

} // namespace datamunge::stats
