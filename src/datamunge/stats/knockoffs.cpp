/// @file knockoffs.cpp
/// @brief Gaussian knockoff construction, SDP diagonal and knockoff(+) selection.

#include <datamunge/stats/knockoffs.hpp>

#include <datamunge/linalg/cholesky.hpp>
#include <datamunge/linalg/eigen.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

namespace datamunge::stats {
/// @brief Covariance checks, eigenvalue floors and the SDP barrier solver.
namespace {

/// @brief Dense double matrix used throughout the knockoff helpers.
using Matrix = linalg::DenseMatrix<double>;

/// @brief Smallest eigenvalue below which a matrix is treated as not positive definite (R knockoff's 10 * 1e-9).
constexpr double positive_definite_floor = 1e-8;

/// @brief Throws unless the covariance is a finite, symmetric, positive definite square matrix.
/// @param covariance Candidate covariance.
/// @throws std::invalid_argument On any violation.
void check_covariance(const Matrix& covariance) {
    const auto p = covariance.rows();
    if (p == 0 || covariance.cols() != p ||
        !std::ranges::all_of(covariance.data(),[](double v) { return std::isfinite(v); }))
        throw std::invalid_argument("knockoffs: covariance must be a nonempty finite square matrix");
    for (std::size_t i = 0; i < p; ++i)
        for (std::size_t j = 0; j < i; ++j)
            if (std::abs(covariance(i,j)-covariance(j,i)) >
                1e-10*std::max({1.0,std::abs(covariance(i,j)),std::abs(covariance(j,i))}))
                throw std::invalid_argument("knockoffs: covariance must be symmetric");
    if (!linalg::cholesky(covariance).ok)
        throw std::invalid_argument("knockoffs: covariance must be positive definite");
}

/// @brief Smallest eigenvalue of a symmetric matrix.
/// @param a Symmetric matrix.
/// @return Its least eigenvalue.
double minimum_eigenvalue(const Matrix& a) {
    return linalg::symmetric_eigen(a).eigenvalues.back();
}

/// @brief \f$2G-\operatorname{diag}(s)\f$.
/// @param correlation Correlation matrix \f$G\f$.
/// @param s Diagonal on the correlation scale.
/// @return The gap matrix.
Matrix gap_matrix(const Matrix& correlation, const std::vector<double>& s) {
    auto gap = correlation*2.0;
    for (std::size_t j = 0; j < s.size(); ++j)
        gap(j,j) -= s[j];
    return gap;
}

/// @brief Shrinks s by the first factor \f$1-10^{-8+k}\f$ that leaves \f$2G-\operatorname{diag}(s)\f$ positive definite.
/// @param correlation Correlation matrix.
/// @param s Diagonal on the correlation scale, updated in place.
void shrink_to_positive_definite(const Matrix& correlation, std::vector<double>& s) {
    for (double eps = 1e-8; eps <= 1; eps *= 10) {
        auto trial = s;
        for (auto& v : trial)
            v = std::max(0.0,v*(1-eps));
        if (minimum_eigenvalue(gap_matrix(correlation,trial)) > positive_definite_floor) {
            s = std::move(trial);
            return;
        }
    }
    std::ranges::fill(s,0.0);
}

/// @brief Inverse of a symmetric positive definite matrix through its Cholesky factor.
/// @param a SPD matrix.
/// @param ok Set to false if the factorization fails.
/// @return \f$A^{-1}\f$ (unspecified when ok is false).
Matrix spd_inverse(const Matrix& a, bool& ok) {
    const auto factor = linalg::cholesky(a);
    ok = factor.ok;
    if (!ok)
        return {};
    return factor.solve(Matrix::identity(a.rows()));
}

/// @brief Maximizes sum(s) over 0<=s<=1 with 2G-diag(s) PSD by a primal log-barrier Newton path.
///
/// The barrier objective is \f$t\sum s+\log\det(2G-D)+\sum\log s+\sum\log(1-s)\f$; its Hessian
/// is \f$-(M\circ M)-\operatorname{diag}(s^{-2}+(1-s)^{-2})\f$ with \f$M=(2G-D)^{-1}\f$. The path
/// stops when the barrier duality bound \f$3p/t\f$ is below \f$10^{-10}\f$.
/// @param correlation Positive definite correlation matrix.
/// @return The near-optimal diagonal on the correlation scale.
std::vector<double> solve_sdp(const Matrix& correlation) {
    const auto p = correlation.rows();
    std::vector<double> s(p,.5*std::min(1.0,2*minimum_eigenvalue(correlation)));
    const auto barrier = [&](const std::vector<double>& v,double t,bool& feasible) {
        feasible = std::ranges::all_of(v,[](double x) { return x > 0 && x < 1; });
        if (!feasible)
            return 0.0;
        const auto factor = linalg::cholesky(gap_matrix(correlation,v));
        if (!factor.ok) {
            feasible = false;
            return 0.0;
        }
        double value = 0;
        for (std::size_t j = 0; j < p; ++j)
            value += t*v[j]+2*std::log(factor.L(j,j))+std::log(v[j])+std::log1p(-v[j]);
        return value;
    };
    for (double t = 1; 3.0*static_cast<double>(p)/t > 1e-10; t *= 8) {
        for (int iteration = 0; iteration < 200; ++iteration) {
            bool ok = false;
            const auto inverse = spd_inverse(gap_matrix(correlation,s),ok);
            if (!ok)
                throw std::runtime_error("knockoffs: SDP iterate left the feasible region");
            std::vector<double> gradient(p);
            Matrix negative_hessian(p,p);
            for (std::size_t i = 0; i < p; ++i) {
                gradient[i] = t-inverse(i,i)+1/s[i]-1/(1-s[i]);
                for (std::size_t j = 0; j < p; ++j)
                    negative_hessian(i,j) = inverse(i,j)*inverse(i,j);
                negative_hessian(i,i) += 1/(s[i]*s[i])+1/((1-s[i])*(1-s[i]));
            }
            const auto factor = linalg::cholesky(negative_hessian);
            if (!factor.ok)
                throw std::runtime_error("knockoffs: SDP Newton system is singular");
            const auto step = factor.solve(gradient);
            double decrement = 0;
            for (std::size_t j = 0; j < p; ++j)
                decrement += gradient[j]*step[j];
            if (decrement/2 < 1e-12)
                break;
            bool feasible = false;
            const double current = barrier(s,t,feasible);
            double length = 1;
            std::vector<double> trial(p);
            for (; length > 1e-14; length /= 2) {
                for (std::size_t j = 0; j < p; ++j)
                    trial[j] = s[j]+length*step[j];
                const double value = barrier(trial,t,feasible);
                if (feasible && value >= current+.25*length*decrement)
                    break;
            }
            if (length <= 1e-14)
                break;
            s = trial;
        }
    }
    return s;
}

/// @brief Throws unless every entry of the matrix is finite.
/// @param x Matrix to check.
/// @param what Name used in the error.
void check_finite(const Matrix& x, const char* what) {
    if (!std::ranges::all_of(x.data(),[](double v) { return std::isfinite(v); }))
        throw std::invalid_argument(std::string("knockoffs: ")+what+" must be finite");
}

} // namespace

std::vector<double> knockoff_diagonal(const linalg::DenseMatrix<double>& covariance, KnockoffScale scale) {
    check_covariance(covariance);
    const auto p = covariance.rows();
    linalg::DenseMatrix<double> correlation(p,p);
    for (std::size_t i = 0; i < p; ++i)
        for (std::size_t j = 0; j < p; ++j)
            correlation(i,j) = covariance(i,j)/std::sqrt(covariance(i,i)*covariance(j,j));
    std::vector<double> s;
    if (scale == KnockoffScale::equicorrelated) {
        s.assign(p,std::min(1.0,2*minimum_eigenvalue(correlation)));
    } else {
        s = solve_sdp(correlation);
        for (auto& v : s)
            v = std::clamp(v,0.0,1.0);
    }
    shrink_to_positive_definite(correlation,s);
    for (std::size_t j = 0; j < p; ++j)
        s[j] *= covariance(j,j);
    return s;
}

GaussianKnockoffs gaussian_knockoffs(const linalg::DenseMatrix<double>& predictors, const std::vector<double>& mean,
                                     const linalg::DenseMatrix<double>& covariance, const GaussianKnockoffOptions& options) {
    const auto n = predictors.rows(), p = predictors.cols();
    if (p == 0 || mean.size() != p || covariance.rows() != p ||
        !std::ranges::all_of(mean,[](double v) { return std::isfinite(v); }))
        throw std::invalid_argument("knockoffs: predictors, mean and covariance dimensions disagree");
    check_finite(predictors,"predictors");
    check_covariance(covariance);

    GaussianKnockoffs result;
    if (options.diagonal.empty()) {
        result.diagonal = knockoff_diagonal(covariance,options.scale);
    } else {
        if (options.diagonal.size() != p ||
            !std::ranges::all_of(options.diagonal,[](double v) { return v >= 0 && std::isfinite(v); }))
            throw std::invalid_argument("knockoffs: diagonal must hold p finite nonnegative values");
        result.diagonal = options.diagonal;
    }
    const auto& s = result.diagonal;

    // Sigma^{-1} D, then the conditional covariance 2D - D Sigma^{-1} D.
    const auto factor = linalg::cholesky(covariance);
    linalg::DenseMatrix<double> d(p,p);
    for (std::size_t j = 0; j < p; ++j)
        d(j,j) = s[j];
    const auto inverse_d = factor.solve(d);
    result.conditional_covariance = linalg::DenseMatrix<double>(p,p);
    for (std::size_t i = 0; i < p; ++i)
        for (std::size_t j = 0; j < p; ++j)
            result.conditional_covariance(i,j) = (i == j ? 2*s[i] : 0.0)-s[i]*inverse_d(i,j);
    for (std::size_t i = 0; i < p; ++i)
        for (std::size_t j = 0; j < i; ++j) {
            const double average = (result.conditional_covariance(i,j)+result.conditional_covariance(j,i))/2;
            result.conditional_covariance(i,j) = result.conditional_covariance(j,i) = average;
        }

    // Symmetric square root C with C^T C equal to the conditional covariance.
    const auto eigen = linalg::symmetric_eigen(result.conditional_covariance);
    double scale = 0;
    for (std::size_t j = 0; j < p; ++j)
        scale = std::max(scale,std::abs(s[j]));
    const double tolerance = 1e-9*std::max(1.0,scale);
    if (eigen.eigenvalues.back() < -tolerance)
        throw std::invalid_argument("knockoffs: diagonal makes the conditional covariance indefinite");
    linalg::DenseMatrix<double> root(p,p);
    for (std::size_t k = 0; k < p; ++k) {
        const double value = std::sqrt(std::max(0.0,eigen.eigenvalues[k]));
        for (std::size_t j = 0; j < p; ++j)
            root(k,j) = value*eigen.eigenvectors(j,k);
    }

    std::mt19937_64 random(options.seed);
    std::normal_distribution<double> normal;
    result.knockoffs = linalg::DenseMatrix<double>(n,p);
    std::vector<double> centered(p), z(p);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < p; ++j) {
            centered[j] = predictors(i,j)-mean[j];
            z[j] = normal(random);
        }
        for (std::size_t j = 0; j < p; ++j) {
            double value = predictors(i,j);
            for (std::size_t k = 0; k < p; ++k)
                value += -centered[k]*inverse_d(k,j)+z[k]*root(k,j);
            result.knockoffs(i,j) = value;
        }
    }
    return result;
}

double knockoff_threshold(const std::vector<double>& statistics, double fdr, bool plus) {
    if (!(fdr > 0 && fdr < 1) ||
        !std::ranges::all_of(statistics,[](double v) { return std::isfinite(v); }))
        throw std::invalid_argument("knockoffs: statistics must be finite and fdr in (0, 1)");
    std::vector<double> candidates{0};
    for (const double w : statistics)
        candidates.push_back(std::abs(w));
    std::ranges::sort(candidates);
    const double offset = plus ? 1 : 0;
    for (const double t : candidates) {
        const auto negative = std::ranges::count_if(statistics,[t](double w) { return w <= -t; });
        const auto positive = std::ranges::count_if(statistics,[t](double w) { return w >= t; });
        if ((offset+static_cast<double>(negative))/std::max<double>(1,static_cast<double>(positive)) <= fdr)
            return t;
    }
    return std::numeric_limits<double>::infinity();
}

KnockoffFilterResult knockoff_filter(const linalg::DenseMatrix<double>& predictors, const std::vector<double>& response,
                                     const std::vector<double>& mean, const linalg::DenseMatrix<double>& covariance,
                                     const KnockoffStatistic& statistic, const KnockoffFilterOptions& options) {
    if (!statistic || response.size() != predictors.rows() ||
        !std::ranges::all_of(response,[](double v) { return std::isfinite(v); }))
        throw std::invalid_argument("knockoffs: statistic is empty or response is invalid");
    if (!(options.fdr > 0 && options.fdr < 1))
        throw std::invalid_argument("knockoffs: fdr must lie in (0, 1)");
    KnockoffFilterResult result;
    result.draw = gaussian_knockoffs(predictors,mean,covariance,options.knockoffs);
    result.statistics = statistic(predictors,result.draw.knockoffs,response);
    if (result.statistics.size() != predictors.cols())
        throw std::invalid_argument("knockoffs: statistic must return one value per predictor");
    result.threshold = knockoff_threshold(result.statistics,options.fdr,options.plus);
    for (std::size_t j = 0; j < result.statistics.size(); ++j)
        if (result.statistics[j] >= result.threshold)
            result.selected.push_back(j);
    return result;
}

} // namespace datamunge::stats
