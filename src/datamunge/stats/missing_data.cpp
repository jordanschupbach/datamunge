/// @file missing_data.cpp
/// @brief Pattern-grouped EM for the multivariate normal and Little's MCAR statistic.

#include <datamunge/stats/missing_data.hpp>

#include <datamunge/linalg/cholesky.hpp>
#include <datamunge/random/distributions.hpp>

#include <algorithm>
#include <cmath>
#include <map>
#include <numbers>
#include <stdexcept>

namespace datamunge::stats {

/// @brief Internal helpers for incomplete-data estimation.
namespace {

using Matrix = linalg::DenseMatrix<double>; ///< Dense matrix of doubles.

/// @brief Submatrix of a covariance on the given row and column index sets.
/// @param s Matrix.
/// @param r Row indices.
/// @param c Column indices.
/// @return The block.
Matrix block(const Matrix& s, const std::vector<std::size_t>& r, const std::vector<std::size_t>& c) {
    Matrix b(r.size(),c.size());
    for (std::size_t i = 0; i < r.size(); ++i)
        for (std::size_t j = 0; j < c.size(); ++j)
            b(i,j) = s(r[i],c[j]);
    return b;
}

/// @brief Observed and missing variable indices of a pattern.
/// @param observed Observation flags.
/// @param obs Receives observed indices.
/// @param mis Receives missing indices.
void split(const std::vector<bool>& observed, std::vector<std::size_t>& obs, std::vector<std::size_t>& mis) {
    obs.clear();
    mis.clear();
    for (std::size_t j = 0; j < observed.size(); ++j)
        (observed[j] ? obs : mis).push_back(j);
}

/// @brief Cholesky factor of an observed block, failing loudly.
/// @param s Block.
/// @return The factorization.
/// @throws std::runtime_error If the block is not positive definite.
linalg::CholeskyDecomposition<double> factor(const Matrix& s) {
    auto f = linalg::cholesky(s);
    if (!f.ok)
        throw std::runtime_error("em_multivariate_normal: singular covariance of observed variables");
    return f;
}

/// @brief Observed-data log-likelihood at given parameters.
/// @param data Data.
/// @param patterns Patterns.
/// @param mu Mean.
/// @param sigma Covariance.
/// @return Log-likelihood.
double log_likelihood(const Matrix& data, const std::vector<MissingPattern>& patterns, const std::vector<double>& mu,
                      const Matrix& sigma) {
    double ll = 0;
    std::vector<std::size_t> obs, mis;
    for (const auto& pat : patterns) {
        split(pat.observed,obs,mis);
        if (obs.empty())
            continue;
        const auto f = factor(block(sigma,obs,obs));
        double logdet = 0;
        for (std::size_t a = 0; a < obs.size(); ++a)
            logdet += 2*std::log(f.L(a,a));
        std::vector<double> d(obs.size());
        for (const std::size_t i : pat.rows) {
            for (std::size_t a = 0; a < obs.size(); ++a)
                d[a] = data(i,obs[a])-mu[obs[a]];
            const auto s = f.solve(d);
            double q = 0;
            for (std::size_t a = 0; a < obs.size(); ++a)
                q += d[a]*s[a];
            ll -= 0.5*(static_cast<double>(obs.size())*std::log(2*std::numbers::pi)+logdet+q);
        }
    }
    return ll;
}

} // namespace

std::vector<MissingPattern> missing_patterns(const linalg::DenseMatrix<double>& data) {
    if (data.rows() == 0 || data.cols() == 0)
        throw std::invalid_argument("missing_patterns: empty data");
    std::map<std::vector<bool>,std::size_t> index;
    std::vector<MissingPattern> patterns;
    for (std::size_t i = 0; i < data.rows(); ++i) {
        std::vector<bool> observed(data.cols());
        for (std::size_t j = 0; j < data.cols(); ++j) {
            if (std::isinf(data(i,j)))
                throw std::invalid_argument("missing_patterns: values must be finite or NaN");
            observed[j] = !std::isnan(data(i,j));
        }
        const auto [it,fresh] = index.emplace(observed,patterns.size());
        if (fresh)
            patterns.push_back({observed,{}});
        patterns[it->second].rows.push_back(i);
    }
    std::ranges::stable_sort(patterns,[](const MissingPattern& a, const MissingPattern& b) {
        return std::ranges::count(a.observed,true) > std::ranges::count(b.observed,true);
    });
    return patterns;
}

MvnEmResult em_multivariate_normal(const linalg::DenseMatrix<double>& data, const MvnEmOptions& options) {
    const std::size_t p = data.cols();
    const auto patterns = missing_patterns(data);
    MvnEmResult r;
    for (const auto& pat : patterns)
        if (std::ranges::count(pat.observed,true) > 0) {
            r.cases += pat.rows.size();
            ++r.patterns;
        }
    if (r.cases < 2)
        throw std::invalid_argument("em_multivariate_normal: need at least two cases with observed values");
    // Available-case start.
    std::vector<double> mu(p,0.0), count(p,0.0);
    Matrix sigma(p,p);
    for (std::size_t i = 0; i < data.rows(); ++i)
        for (std::size_t j = 0; j < p; ++j)
            if (!std::isnan(data(i,j))) {
                mu[j] += data(i,j);
                count[j] += 1;
            }
    for (std::size_t j = 0; j < p; ++j) {
        if (count[j] == 0)
            throw std::invalid_argument("em_multivariate_normal: a variable has no observed values");
        mu[j] /= count[j];
    }
    for (std::size_t i = 0; i < data.rows(); ++i)
        for (std::size_t j = 0; j < p; ++j)
            if (!std::isnan(data(i,j)))
                sigma(j,j) += (data(i,j)-mu[j])*(data(i,j)-mu[j])/count[j];
    for (std::size_t j = 0; j < p; ++j)
        if (!(sigma(j,j) > 0))
            sigma(j,j) = 1; // a single observed value: any positive start
    const double n = static_cast<double>(r.cases);
    std::vector<std::size_t> obs, mis;
    double previous_change = 0;
    for (;;) {
        std::vector<double> t1(p,0.0);
        Matrix t2(p,p);
        for (const auto& pat : patterns) {
            split(pat.observed,obs,mis);
            if (obs.empty())
                continue;
            const auto f = factor(block(sigma,obs,obs));
            // Regression of missing on observed: B = S_MO S_OO^{-1}; conditional covariance S_MM - B S_OM.
            Matrix B(mis.size(),obs.size()), C(mis.size(),mis.size());
            std::vector<double> col(obs.size());
            for (std::size_t m = 0; m < mis.size(); ++m) {
                for (std::size_t a = 0; a < obs.size(); ++a)
                    col[a] = sigma(obs[a],mis[m]);
                const auto s = f.solve(col);
                for (std::size_t a = 0; a < obs.size(); ++a)
                    B(m,a) = s[a];
            }
            for (std::size_t m = 0; m < mis.size(); ++m)
                for (std::size_t q = 0; q < mis.size(); ++q) {
                    double v = sigma(mis[m],mis[q]);
                    for (std::size_t a = 0; a < obs.size(); ++a)
                        v -= B(m,a)*sigma(obs[a],mis[q]);
                    C(m,q) = v;
                }
            std::vector<double> xhat(p);
            for (const std::size_t i : pat.rows) {
                for (const std::size_t j : obs)
                    xhat[j] = data(i,j);
                for (std::size_t m = 0; m < mis.size(); ++m) {
                    double v = mu[mis[m]];
                    for (std::size_t a = 0; a < obs.size(); ++a)
                        v += B(m,a)*(data(i,obs[a])-mu[obs[a]]);
                    xhat[mis[m]] = v;
                }
                for (std::size_t j = 0; j < p; ++j) {
                    t1[j] += xhat[j];
                    for (std::size_t k = 0; k <= j; ++k)
                        t2(j,k) += xhat[j]*xhat[k];
                }
            }
            const double rows = static_cast<double>(pat.rows.size());
            for (std::size_t m = 0; m < mis.size(); ++m)
                for (std::size_t q = 0; q < mis.size(); ++q)
                    if (mis[q] <= mis[m])
                        t2(mis[m],mis[q]) += rows*C(m,q);
        }
        std::vector<double> next_mu(p);
        Matrix next_sigma(p,p);
        for (std::size_t j = 0; j < p; ++j)
            next_mu[j] = t1[j]/n;
        for (std::size_t j = 0; j < p; ++j)
            for (std::size_t k = 0; k <= j; ++k)
                next_sigma(j,k) = next_sigma(k,j) = t2(j,k)/n-next_mu[j]*next_mu[k];
        double change = 0;
        bool small = true;
        for (std::size_t j = 0; j < p; ++j) {
            const double d = std::abs(next_mu[j]-mu[j]);
            change = std::max(change,d);
            small = small && d <= options.tolerance*(1+std::abs(mu[j]));
            for (std::size_t k = 0; k <= j; ++k) {
                const double e = std::abs(next_sigma(j,k)-sigma(j,k));
                change = std::max(change,e);
                small = small && e <= options.tolerance*(1+std::abs(sigma(j,k)));
            }
        }
        mu = std::move(next_mu);
        sigma = std::move(next_sigma);
        ++r.iterations;
        if (previous_change > 0 && change > 0)
            r.convergence_rate = change/previous_change;
        previous_change = change;
        if (small) {
            r.converged = true;
            break;
        }
        if (r.iterations >= options.max_iterations)
            break;
    }
    r.mean = mu;
    r.covariance = sigma;
    r.log_likelihood = log_likelihood(data,patterns,mu,sigma);
    return r;
}

LittleMcarTest little_mcar_test(const linalg::DenseMatrix<double>& data, const MvnEmOptions& options) {
    LittleMcarTest t;
    t.fit = em_multivariate_normal(data,options);
    const auto patterns = missing_patterns(data);
    std::vector<std::size_t> obs, mis;
    double observed_total = 0;
    for (const auto& pat : patterns) {
        split(pat.observed,obs,mis);
        if (obs.empty())
            continue;
        ++t.patterns;
        observed_total += static_cast<double>(obs.size());
        const auto f = factor(block(t.fit.covariance,obs,obs));
        std::vector<double> d(obs.size(),0.0);
        for (const std::size_t i : pat.rows)
            for (std::size_t a = 0; a < obs.size(); ++a)
                d[a] += data(i,obs[a])/static_cast<double>(pat.rows.size());
        for (std::size_t a = 0; a < obs.size(); ++a)
            d[a] -= t.fit.mean[obs[a]];
        const auto s = f.solve(d);
        double q = 0;
        for (std::size_t a = 0; a < obs.size(); ++a)
            q += d[a]*s[a];
        t.statistic += static_cast<double>(pat.rows.size())*q;
    }
    t.df = observed_total-static_cast<double>(data.cols());
    if (!(t.df > 0))
        throw std::invalid_argument("little_mcar_test: need at least two missing-data patterns");
    t.p_value = 1-random::chi_squared_cdf(t.statistic,t.df);
    return t;
}

MvnRegression mvn_regression(const std::vector<double>& mean, const linalg::DenseMatrix<double>& covariance,
                             std::size_t response, const std::vector<std::size_t>& predictors) {
    const std::size_t p = mean.size();
    if (covariance.rows() != p || covariance.cols() != p || response >= p)
        throw std::invalid_argument("mvn_regression: mean, covariance and response index must agree");
    std::vector<bool> used(p,false);
    used[response] = true;
    for (const std::size_t j : predictors) {
        if (j >= p || used[j])
            throw std::invalid_argument("mvn_regression: predictors must be distinct, in range and not the response");
        used[j] = true;
    }
    MvnRegression r;
    std::vector<double> sxy(predictors.size());
    for (std::size_t a = 0; a < predictors.size(); ++a)
        sxy[a] = covariance(predictors[a],response);
    if (!predictors.empty()) {
        const auto f = linalg::cholesky(block(covariance,predictors,predictors));
        if (!f.ok)
            throw std::runtime_error("mvn_regression: singular predictor covariance");
        r.slopes = f.solve(sxy);
    }
    r.intercept = mean[response];
    r.residual_variance = covariance(response,response);
    for (std::size_t a = 0; a < predictors.size(); ++a) {
        r.intercept -= r.slopes[a]*mean[predictors[a]];
        r.residual_variance -= r.slopes[a]*sxy[a];
    }
    r.r_squared = covariance(response,response) > 0 ? 1-r.residual_variance/covariance(response,response) : 0.0;
    return r;
}

} // namespace datamunge::stats
