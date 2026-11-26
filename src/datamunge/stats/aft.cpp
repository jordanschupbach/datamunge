/// @file aft.cpp
/// @brief Log-location-scale likelihoods with analytic derivatives and Newton fitting.

#include <datamunge/stats/aft.hpp>

#include <datamunge/linalg/cholesky.hpp>
#include <datamunge/random/distributions.hpp>
#include <datamunge/stats/detail/test_common.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace datamunge::stats {

/// @brief Internal helpers for accelerated failure time fitting.
namespace {

/// @brief Dense double matrix.
using Matrix = linalg::DenseMatrix<double>;

/// @brief log(1 + e^w) without overflow.
/// @param w Argument.
/// @return The softplus.
double softplus(double w) { return w > 0 ? w+std::log1p(std::exp(-w)) : std::log1p(std::exp(w)); }

/// @brief Standard error-distribution quantities at one point.
struct Point {
    double log_f{};  ///< Log density.
    double f{};      ///< Density.
    double df{};     ///< Density derivative.
    double S{};      ///< Survival function.
    double log_S{};  ///< Log survival.
    double log_F{};  ///< Log distribution function.
    double g{};      ///< d log f / dw.
    double dg{};     ///< d^2 log f / dw^2.
    double h{};      ///< d log S / dw.
    double dh{};     ///< d^2 log S / dw^2.
    double k{};      ///< d log F / dw.
    double dk{};     ///< d^2 log F / dw^2.
};

/// @brief Evaluates the standardized error distribution at w.
/// @param d Distribution.
/// @param w Standardized log time.
/// @return Density, tails and log-derivatives.
Point at(AftDistribution d, double w) {
    Point p;
    switch (d) {
    case AftDistribution::weibull:
    case AftDistribution::exponential: {
        const double e = std::exp(w);
        p.log_f = w-e;
        p.f = std::exp(p.log_f);
        p.df = p.f*(1-e);
        p.log_S = -e;
        p.S = std::exp(-e);
        const double F = -std::expm1(-e);
        p.log_F = std::log(F);
        p.g = 1-e;
        p.dg = -e;
        p.h = -e;
        p.dh = -e;
        p.k = p.f/F;
        p.dk = (p.df*F-p.f*p.f)/(F*F);
        break;
    }
    case AftDistribution::loglogistic: {
        const double F = 1/(1+std::exp(-w)), S = 1/(1+std::exp(w));
        p.log_S = -softplus(w);
        p.log_F = -softplus(-w);
        p.log_f = p.log_S+p.log_F;
        p.f = F*S;
        p.df = p.f*(S-F);
        p.S = S;
        p.g = S-F;
        p.dg = -2*p.f;
        p.h = -F;
        p.dh = -p.f;
        p.k = S;
        p.dk = -p.f;
        break;
    }
    case AftDistribution::lognormal: {
        p.log_f = -w*w/2-0.5*std::log(2*std::numbers::pi);
        p.f = std::exp(p.log_f);
        p.df = -w*p.f;
        p.S = 0.5*std::erfc(w/std::numbers::sqrt2);
        const double F = 0.5*std::erfc(-w/std::numbers::sqrt2);
        p.log_S = std::log(p.S);
        p.log_F = std::log(F);
        p.g = -w;
        p.dg = -1;
        p.h = -p.f/p.S;
        p.dh = -w*p.h-p.h*p.h;
        p.k = p.f/F;
        p.dk = -w*p.k-p.k*p.k;
        break;
    }
    }
    return p;
}

/// @brief One observation's log likelihood and derivatives in (eta, log sigma).
struct Contribution {
    double value{}; ///< Log likelihood.
    double e{};     ///< d/d eta.
    double s{};     ///< d/d log sigma.
    double ee{};    ///< d^2/d eta^2.
    double es{};    ///< d^2/d eta d log sigma.
    double ss{};    ///< d^2/d (log sigma)^2.
};

/// @brief Log-likelihood contribution of one censored time.
/// @param d Distribution.
/// @param t Time record.
/// @param eta Linear predictor.
/// @param sigma Scale.
/// @return Value and derivatives.
Contribution contribution(AftDistribution d, const CensoredTime& t, double eta, double sigma) {
    Contribution c;
    const auto standard = [&](double time) { return (std::log(time)-eta)/sigma; };
    const auto from = [&](double v, double g, double dg, double z) {
        c.value = v;
        c.e = -g/sigma;
        c.s = -g*z;
        c.ee = dg/(sigma*sigma);
        c.es = (dg*z+g)/sigma;
        c.ss = dg*z*z+g*z;
    };
    if (t.lower == t.upper) {
        const double z = standard(t.lower);
        const Point p = at(d,z);
        from(p.log_f-std::log(sigma)-std::log(t.lower),p.g,p.dg,z);
        c.s -= 1;
    } else if (std::isinf(t.upper)) {
        const double z = standard(t.lower);
        const Point p = at(d,z);
        from(p.log_S,p.h,p.dh,z);
    } else if (t.lower == 0) {
        const double z = standard(t.upper);
        const Point p = at(d,z);
        from(p.log_F,p.k,p.dk,z);
    } else {
        const double zl = standard(t.lower), zu = standard(t.upper);
        const Point l = at(d,zl), u = at(d,zu);
        // Derivatives of S(z) in (eta, s): S_e = f/sigma, S_s = f z, and second derivatives.
        const auto pieces = [&](const Point& p, double z) {
            return std::array<double,5>{p.f/sigma,p.f*z,-p.df/(sigma*sigma),-(p.df*z+p.f)/sigma,
                                        -(p.df*z*z+p.f*z)};
        };
        const auto a = pieces(l,zl), b = pieces(u,zu);
        const double D = l.S-u.S;
        if (!(D > 0))
            throw std::runtime_error("fit_aft: interval probability underflowed");
        const double De = a[0]-b[0], Ds = a[1]-b[1];
        c.value = std::log(D);
        c.e = De/D;
        c.s = Ds/D;
        c.ee = (a[2]-b[2])/D-c.e*c.e;
        c.es = (a[3]-b[3])/D-c.e*c.s;
        c.ss = (a[4]-b[4])/D-c.s*c.s;
    }
    return c;
}

/// @brief Total log likelihood with gradient and Hessian in (beta, log sigma).
struct Evaluation {
    double value{};            ///< Log likelihood.
    std::vector<double> grad;  ///< Gradient.
    Matrix hess;               ///< Hessian.
};

/// @brief Evaluates the log likelihood.
/// @param d Distribution.
/// @param times Records.
/// @param x Design with intercept.
/// @param theta Coefficients, then log sigma unless fixed.
/// @param fixed_scale True for the exponential.
/// @return Value, gradient and Hessian.
Evaluation evaluate(AftDistribution d, const std::vector<CensoredTime>& times, const Matrix& x,
                    const std::vector<double>& theta, bool fixed_scale) {
    const std::size_t n = times.size(), q = x.cols(), m = theta.size();
    const double sigma = fixed_scale ? 1.0 : std::exp(theta[q]);
    Evaluation e{0,std::vector<double>(m,0.0),Matrix(m,m)};
    for (std::size_t i = 0; i < n; ++i) {
        double eta = 0;
        for (std::size_t a = 0; a < q; ++a)
            eta += x(i,a)*theta[a];
        const auto c = contribution(d,times[i],eta,sigma);
        e.value += c.value;
        for (std::size_t a = 0; a < q; ++a) {
            e.grad[a] += c.e*x(i,a);
            for (std::size_t b = 0; b <= a; ++b)
                e.hess(a,b) += c.ee*x(i,a)*x(i,b);
            if (!fixed_scale)
                e.hess(q,a) += c.es*x(i,a);
        }
        if (!fixed_scale) {
            e.grad[q] += c.s;
            e.hess(q,q) += c.ss;
        }
    }
    for (std::size_t a = 0; a < m; ++a)
        for (std::size_t b = 0; b < a; ++b)
            e.hess(b,a) = e.hess(a,b);
    return e;
}

/// @brief Inverts a symmetric positive definite matrix.
/// @param a Matrix.
/// @return Inverse.
/// @throws std::runtime_error If not positive definite.
Matrix spd_inverse(const Matrix& a) {
    const auto factor = linalg::cholesky(a);
    if (!factor.ok)
        throw std::runtime_error("fit_aft: information matrix is singular");
    const std::size_t p = a.rows();
    Matrix inverse(p,p);
    for (std::size_t j = 0; j < p; ++j) {
        std::vector<double> unit(p,0.0);
        unit[j] = 1;
        const auto column = factor.solve(unit);
        for (std::size_t i = 0; i < p; ++i)
            inverse(i,j) = column[i];
    }
    return inverse;
}

/// @brief Maximizes the likelihood by damped Newton iteration.
/// @param d Distribution.
/// @param times Records.
/// @param x Design with intercept.
/// @param options Iteration control.
/// @param iterations Receives the number of iterations.
/// @param converged Receives convergence.
/// @return Estimate and the evaluation there.
std::pair<std::vector<double>,Evaluation> maximize(AftDistribution d, const std::vector<CensoredTime>& times,
                                                  const Matrix& x, const AftOptions& options,
                                                  std::size_t& iterations, bool& converged) {
    const std::size_t n = times.size(), q = x.cols();
    const bool fixed = d == AftDistribution::exponential;
    // Start: least squares of a representative log time on the design.
    std::vector<double> y(n);
    for (std::size_t i = 0; i < n; ++i) {
        const auto& t = times[i];
        y[i] = std::log(t.lower == 0 ? t.upper : std::isinf(t.upper) ? t.lower : std::sqrt(t.lower*t.upper));
    }
    Matrix xtx(q,q);
    std::vector<double> xty(q,0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t a = 0; a < q; ++a) {
            xty[a] += x(i,a)*y[i];
            for (std::size_t b = 0; b < q; ++b)
                xtx(a,b) += x(i,a)*x(i,b);
        }
    const auto ls = linalg::cholesky(xtx);
    if (!ls.ok)
        throw std::invalid_argument("fit_aft: covariates are collinear");
    std::vector<double> theta = ls.solve(xty);
    double rss = 0;
    for (std::size_t i = 0; i < n; ++i) {
        double r = y[i];
        for (std::size_t a = 0; a < q; ++a)
            r -= x(i,a)*theta[a];
        rss += r*r;
    }
    if (!fixed)
        theta.push_back(std::log(std::max(std::sqrt(rss/static_cast<double>(n)),1e-3)));
    auto current = evaluate(d,times,x,theta,fixed);
    if (!std::isfinite(current.value))
        throw std::runtime_error("fit_aft: starting values give a nonfinite likelihood");
    const std::size_t m = theta.size();
    iterations = 0;
    converged = false;
    while (iterations < options.max_iterations) {
        Matrix information(m,m);
        for (std::size_t a = 0; a < m; ++a)
            for (std::size_t b = 0; b < m; ++b)
                information(a,b) = -current.hess(a,b);
        double ridge = 0;
        auto factor = linalg::cholesky(information);
        while (!factor.ok) {
            double trace = 0;
            for (std::size_t a = 0; a < m; ++a)
                trace += std::abs(information(a,a));
            ridge = ridge == 0 ? 1e-6*(trace/static_cast<double>(m)+1) : ridge*10;
            Matrix damped = information;
            for (std::size_t a = 0; a < m; ++a)
                damped(a,a) += ridge;
            factor = linalg::cholesky(damped);
        }
        const auto step = factor.solve(current.grad);
        double scale = 1;
        std::vector<double> candidate(m);
        Evaluation trial;
        for (int halving = 0;; ++halving) {
            for (std::size_t a = 0; a < m; ++a)
                candidate[a] = theta[a]+scale*step[a];
            bool ok = true;
            try {
                trial = evaluate(d,times,x,candidate,fixed);
            } catch (const std::runtime_error&) {
                ok = false;
            }
            if (ok && std::isfinite(trial.value) && trial.value >= current.value-1e-12*std::abs(current.value))
                break;
            if (halving == 40)
                throw std::runtime_error("fit_aft: step halving failed to improve the likelihood");
            scale /= 2;
        }
        const double change = std::abs(trial.value-current.value)/(std::abs(trial.value)+1e-10);
        theta = candidate;
        current = std::move(trial);
        ++iterations;
        if (change <= options.tolerance && ridge == 0) {
            converged = true;
            break;
        }
    }
    return {theta,current};
}

/// @brief Standardized p-quantile of the error distribution.
/// @param d Distribution.
/// @param p Probability.
/// @return Quantile.
double error_quantile(AftDistribution d, double p) {
    switch (d) {
    case AftDistribution::lognormal:
        return random::normal_quantile(p);
    case AftDistribution::loglogistic:
        return std::log(p/(1-p));
    default:
        return std::log(-std::log1p(-p));
    }
}

} // namespace

double AftFit::predict_quantile(const std::vector<double>& covariates, double p) const {
    if (covariates.size()+1 != coefficients.size() || !(p > 0 && p < 1))
        throw std::invalid_argument("AftFit::predict_quantile: invalid covariates or probability");
    double eta = coefficients[0];
    for (std::size_t a = 0; a < covariates.size(); ++a)
        eta += coefficients[a+1]*covariates[a];
    return std::exp(eta+scale*error_quantile(distribution,p));
}

double AftFit::predict_survival(double time, const std::vector<double>& covariates) const {
    if (covariates.size()+1 != coefficients.size() || !(time > 0))
        throw std::invalid_argument("AftFit::predict_survival: invalid covariates or time");
    double eta = coefficients[0];
    for (std::size_t a = 0; a < covariates.size(); ++a)
        eta += coefficients[a+1]*covariates[a];
    return at(distribution,(std::log(time)-eta)/scale).S;
}

AftFit fit_aft(const std::vector<CensoredTime>& times, const linalg::DenseMatrix<double>& covariates,
               const AftOptions& options) {
    const std::size_t n = times.size(), p = covariates.cols();
    if (n == 0 || covariates.rows() != n)
        throw std::invalid_argument("fit_aft: need one covariate row per time");
    if (!(options.confidence > 0 && options.confidence < 1))
        throw std::invalid_argument("fit_aft: confidence must be in (0, 1)");
    bool informative = false;
    for (const auto& t : times) {
        const bool valid = std::isfinite(t.lower) && t.lower >= 0 && t.upper >= t.lower && !std::isnan(t.upper) &&
                           (t.lower > 0 || (std::isfinite(t.upper) && t.upper > 0));
        if (!valid)
            throw std::invalid_argument("fit_aft: invalid censored time");
        informative = informative || std::isfinite(t.upper);
    }
    if (!informative)
        throw std::invalid_argument("fit_aft: every observation is right-censored");
    Matrix x(n,p+1);
    for (std::size_t i = 0; i < n; ++i) {
        x(i,0) = 1;
        for (std::size_t a = 0; a < p; ++a) {
            if (!std::isfinite(covariates(i,a)))
                throw std::invalid_argument("fit_aft: covariates must be finite");
            x(i,a+1) = covariates(i,a);
        }
    }
    AftFit fit;
    fit.distribution = options.distribution;
    const bool fixed = options.distribution == AftDistribution::exponential;
    auto [theta,e] = maximize(options.distribution,times,x,options,fit.iterations,fit.converged);
    const std::size_t m = theta.size();
    Matrix information(m,m);
    for (std::size_t a = 0; a < m; ++a)
        for (std::size_t b = 0; b < m; ++b)
            information(a,b) = -e.hess(a,b);
    fit.covariance = spd_inverse(information);
    fit.coefficients.assign(theta.begin(),theta.begin()+static_cast<std::ptrdiff_t>(p+1));
    for (std::size_t a = 0; a <= p; ++a)
        fit.standard_errors.push_back(std::sqrt(fit.covariance(a,a)));
    if (!fixed) {
        fit.scale = std::exp(theta[p+1]);
        fit.log_scale_standard_error = std::sqrt(fit.covariance(p+1,p+1));
    }
    fit.log_likelihood = e.value;
    const double z = random::normal_quantile(.5+options.confidence/2);
    for (std::size_t a = 1; a <= p; ++a) {
        fit.time_ratios.push_back(std::exp(fit.coefficients[a]));
        fit.time_ratio_lower.push_back(std::exp(fit.coefficients[a]-z*fit.standard_errors[a]));
        fit.time_ratio_upper.push_back(std::exp(fit.coefficients[a]+z*fit.standard_errors[a]));
    }
    fit.degrees_of_freedom = p;
    if (p == 0) {
        fit.null_log_likelihood = fit.log_likelihood;
    } else {
        std::size_t iterations = 0;
        bool converged = false;
        Matrix ones(n,1,1.0);
        fit.null_log_likelihood = maximize(options.distribution,times,ones,options,iterations,converged).second.value;
        fit.chi_square = std::max(0.0,2*(fit.log_likelihood-fit.null_log_likelihood));
        fit.p_value = detail::chi_squared_test_survival(fit.chi_square,static_cast<double>(p));
    }
    return fit;
}

} // namespace datamunge::stats
