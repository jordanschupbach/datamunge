/// @file survey_glm.cpp
/// @brief IRLS for survey-weighted canonical GLMs with a linearized sandwich covariance.

#include <datamunge/stats/survey_glm.hpp>

#include <datamunge/linalg/cholesky.hpp>
#include <datamunge/random/distributions.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace datamunge::stats {

/// @brief Internal helpers for survey GLMs.
namespace {

/// @brief Mean from the linear predictor for a canonical link.
/// @param family Family.
/// @param eta Linear predictor.
/// @return Mean.
double mean_of(SurveyGlmFamily family, double eta) {
    switch (family) {
    case SurveyGlmFamily::binomial:
        return eta >= 0 ? 1/(1+std::exp(-eta)) : std::exp(eta)/(1+std::exp(eta));
    case SurveyGlmFamily::poisson:
        return std::exp(eta);
    default:
        return eta;
    }
}

/// @brief Variance function, which for a canonical link is also \f$d\mu/d\eta\f$.
/// @param family Family.
/// @param mu Mean.
/// @return \f$V(\mu)\f$.
double variance_of(SurveyGlmFamily family, double mu) {
    switch (family) {
    case SurveyGlmFamily::binomial:
        return mu*(1-mu);
    case SurveyGlmFamily::poisson:
        return mu;
    default:
        return 1;
    }
}

/// @brief Unit deviance contribution.
/// @param family Family.
/// @param y Response.
/// @param mu Mean.
/// @return Deviance contribution.
double deviance_of(SurveyGlmFamily family, double y, double mu) {
    const auto xlogy = [](double a, double b) { return a > 0 ? a*std::log(a/b) : 0.0; };
    switch (family) {
    case SurveyGlmFamily::binomial:
        return 2*(xlogy(y,mu)+xlogy(1-y,1-mu));
    case SurveyGlmFamily::poisson:
        return 2*(xlogy(y,mu)-(y-mu));
    default:
        return (y-mu)*(y-mu);
    }
}

/// @brief Inverse of a symmetric positive-definite matrix.
/// @param a Matrix.
/// @return Its inverse.
/// @throws std::runtime_error If a is not positive definite.
linalg::DenseMatrix<double> inverse(const linalg::DenseMatrix<double>& a) {
    const auto factor = linalg::cholesky(a);
    if (!factor.ok)
        throw std::runtime_error("fit_survey_glm: singular weighted information matrix");
    const std::size_t p = a.rows();
    linalg::DenseMatrix<double> inv(p,p);
    std::vector<double> e(p);
    for (std::size_t j = 0; j < p; ++j) {
        std::fill(e.begin(),e.end(),0.0);
        e[j] = 1;
        const auto col = factor.solve(e);
        for (std::size_t i = 0; i < p; ++i)
            inv(i,j) = col[i];
    }
    return inv;
}

} // namespace

SurveyGlmFit fit_survey_glm(const SurveyDesign& design, const linalg::DenseMatrix<double>& x,
                            const std::vector<double>& y, const SurveyGlmOptions& options) {
    const std::size_t n = y.size(), p = x.cols();
    if (x.rows() != n || design.weights.size() != n || p == 0)
        throw std::invalid_argument("fit_survey_glm: need one model-matrix row and one weight per response");
    if (!options.domain.empty() && options.domain.size() != n)
        throw std::invalid_argument("fit_survey_glm: domain must have one entry per unit");
    const auto family = options.family;
    if (family != SurveyGlmFamily::gaussian && family != SurveyGlmFamily::binomial && family != SurveyGlmFamily::poisson)
        throw std::invalid_argument("fit_survey_glm: unknown family");
    std::vector<double> w(n);
    std::size_t used = 0;
    double weight_sum = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const bool in = options.domain.empty() || options.domain[i];
        w[i] = in ? design.weights[i] : 0.0;
        if (!in)
            continue;
        ++used;
        weight_sum += w[i];
        if (!std::isfinite(y[i]) || (family == SurveyGlmFamily::binomial && (y[i] < 0 || y[i] > 1)) ||
            (family == SurveyGlmFamily::poisson && y[i] < 0))
            throw std::invalid_argument("fit_survey_glm: responses must be finite and in the family's support");
        for (std::size_t a = 0; a < p; ++a)
            if (!std::isfinite(x(i,a)))
                throw std::invalid_argument("fit_survey_glm: model matrix must be finite");
    }
    if (used <= p)
        throw std::invalid_argument("fit_survey_glm: need more domain units than coefficients");

    // Starting values as R's glm: mu = (y + 0.5) / 2 for binomial, y + 0.1 for Poisson, y for gaussian.
    std::vector<double> eta(n,0.0), mu(n,0.0);
    for (std::size_t i = 0; i < n; ++i) {
        if (w[i] == 0)
            continue;
        switch (family) {
        case SurveyGlmFamily::binomial:
            mu[i] = (y[i]+0.5)/2;
            eta[i] = std::log(mu[i]/(1-mu[i]));
            break;
        case SurveyGlmFamily::poisson:
            mu[i] = y[i]+0.1;
            eta[i] = std::log(mu[i]);
            break;
        default:
            mu[i] = y[i];
            eta[i] = y[i];
        }
    }
    SurveyGlmFit fit;
    std::vector<double> beta(p,0.0);
    double deviance = std::numeric_limits<double>::infinity();
    linalg::DenseMatrix<double> information(p,p);
    for (;;) {
        information = linalg::DenseMatrix<double>(p,p);
        std::vector<double> rhs(p,0.0);
        for (std::size_t i = 0; i < n; ++i) {
            if (w[i] == 0)
                continue;
            const double v = std::max(variance_of(family,mu[i]),1e-300);
            const double z = eta[i]+(y[i]-mu[i])/v, ww = w[i]*v;
            for (std::size_t a = 0; a < p; ++a) {
                rhs[a] += ww*x(i,a)*z;
                for (std::size_t b = 0; b <= a; ++b)
                    information(a,b) += ww*x(i,a)*x(i,b);
            }
        }
        for (std::size_t a = 0; a < p; ++a)
            for (std::size_t b = 0; b < a; ++b)
                information(b,a) = information(a,b);
        const auto factor = linalg::cholesky(information);
        if (!factor.ok)
            throw std::runtime_error("fit_survey_glm: singular weighted information matrix");
        beta = factor.solve(rhs);
        double next = 0;
        for (std::size_t i = 0; i < n; ++i) {
            double e = 0;
            for (std::size_t a = 0; a < p; ++a)
                e += x(i,a)*beta[a];
            eta[i] = e;
            mu[i] = mean_of(family,e);
            if (w[i] > 0)
                next += w[i]*deviance_of(family,y[i],mu[i]);
        }
        ++fit.iterations;
        const bool done = std::abs(next-deviance) < options.tolerance*(std::abs(next)+0.1);
        deviance = next;
        if (done || family == SurveyGlmFamily::gaussian) {
            fit.converged = true;
            break;
        }
        if (fit.iterations >= options.max_iterations)
            break;
    }
    // Information and scores at the final estimates.
    information = linalg::DenseMatrix<double>(p,p);
    linalg::DenseMatrix<double> scores(n,p);
    double pearson = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (w[i] == 0)
            continue;
        const double v = variance_of(family,mu[i]);
        pearson += w[i]/weight_sum*static_cast<double>(used)*(y[i]-mu[i])*(y[i]-mu[i])/std::max(v,1e-300);
        for (std::size_t a = 0; a < p; ++a) {
            scores(i,a) = w[i]*x(i,a)*(y[i]-mu[i]);
            for (std::size_t b = 0; b < p; ++b)
                information(a,b) += w[i]*v*x(i,a)*x(i,b);
        }
    }
    const auto ainv = inverse(information);
    const auto meat = survey_score_covariance(design,scores);
    fit.covariance = linalg::DenseMatrix<double>(p,p);
    for (std::size_t a = 0; a < p; ++a)
        for (std::size_t b = 0; b < p; ++b) {
            double s = 0;
            for (std::size_t c = 0; c < p; ++c)
                for (std::size_t d = 0; d < p; ++d)
                    s += ainv(a,c)*meat(c,d)*ainv(d,b);
            fit.covariance(a,b) = s;
        }
    fit.df_residual = survey_degrees_of_freedom(design,options.domain)+1-static_cast<double>(p);
    // Model-based: weights rescaled to mean 1 over the domain, dispersion from Pearson residuals for gaussian.
    const double scale = static_cast<double>(used)/weight_sum;
    const double dispersion = family == SurveyGlmFamily::gaussian ? pearson/static_cast<double>(used-p) : 1.0;
    const double tq = fit.df_residual > 0 ? random::student_t_quantile(0.5+design.confidence/2,fit.df_residual)
                                          : std::numeric_limits<double>::quiet_NaN();
    fit.coefficients = beta;
    for (std::size_t a = 0; a < p; ++a) {
        const double se = std::sqrt(std::max(0.0,fit.covariance(a,a)));
        const double model = std::sqrt(ainv(a,a)/scale*dispersion);
        fit.standard_errors.push_back(se);
        fit.model_standard_errors.push_back(model);
        fit.design_effects.push_back(se*se/(model*model));
        const double t = beta[a]/se;
        fit.t_values.push_back(t);
        fit.p_values.push_back(fit.df_residual > 0 ? 2*(1-random::student_t_cdf(std::abs(t),fit.df_residual))
                                                   : std::numeric_limits<double>::quiet_NaN());
        fit.lower.push_back(beta[a]-tq*se);
        fit.upper.push_back(beta[a]+tq*se);
    }
    fit.fitted = mu;
    return fit;
}

SurveyGlmWald survey_glm_wald_test(const SurveyGlmFit& fit, const std::vector<std::size_t>& indices) {
    const std::size_t p = fit.coefficients.size(), k = indices.size();
    if (k == 0 || fit.covariance.rows() != p || fit.covariance.cols() != p)
        throw std::invalid_argument("survey_glm_wald_test: need a fitted model and at least one coefficient");
    std::vector<bool> seen(p,false);
    for (const std::size_t j : indices) {
        if (j >= p || seen[j])
            throw std::invalid_argument("survey_glm_wald_test: indices must be distinct and in range");
        seen[j] = true;
    }
    if (!(fit.df_residual > 0))
        throw std::invalid_argument("survey_glm_wald_test: no residual degrees of freedom");
    linalg::DenseMatrix<double> v(k,k);
    std::vector<double> b(k);
    for (std::size_t a = 0; a < k; ++a) {
        b[a] = fit.coefficients[indices[a]];
        for (std::size_t c = 0; c < k; ++c)
            v(a,c) = fit.covariance(indices[a],indices[c]);
    }
    const auto factor = linalg::cholesky(v);
    if (!factor.ok)
        throw std::runtime_error("survey_glm_wald_test: singular covariance block");
    const auto s = factor.solve(b);
    SurveyGlmWald out;
    for (std::size_t a = 0; a < k; ++a)
        out.chi_square += b[a]*s[a];
    out.df1 = static_cast<double>(k);
    out.df2 = fit.df_residual;
    out.statistic = out.chi_square/out.df1;
    out.p_value = 1-random::f_cdf(out.statistic,out.df1,out.df2);
    return out;
}

} // namespace datamunge::stats
