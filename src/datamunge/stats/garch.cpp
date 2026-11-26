/// @file garch.cpp
/// @brief ARCH-LM auxiliary regression, GARCH likelihood, constrained optimization and sandwich covariance.

#include <datamunge/stats/garch.hpp>

#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/linalg/eigen.hpp>
#include <datamunge/optim/lbfgs.hpp>
#include <datamunge/random/distributions.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>

namespace datamunge::stats {
/// @brief Linear algebra, likelihood and the optimizer objective.
namespace {

/// @brief Matrix alias used throughout.
using Matrix = linalg::DenseMatrix<double>;

/// @brief Inverse of a symmetric positive definite matrix.
/// @param a Matrix.
/// @return \f$a^{-1}\f$.
/// @throws std::invalid_argument When a is not numerically positive definite.
Matrix spd_inverse(const Matrix& a) {
    const std::size_t k = a.rows();
    const auto eig = linalg::symmetric_eigen(a);
    if (!(eig.eigenvalues.back() > 1e-14*std::abs(eig.eigenvalues.front())))
        throw std::invalid_argument("garch: singular information matrix");
    Matrix inv(k,k);
    for (std::size_t e = 0; e < k; ++e)
        for (std::size_t i = 0; i < k; ++i)
            for (std::size_t j = 0; j < k; ++j)
                inv(i,j) += eig.eigenvectors(i,e)*eig.eigenvectors(j,e)/eig.eigenvalues[e];
    return inv;
}

/// @brief Model layout and data shared by the likelihood routines.
struct Problem {
    const std::vector<double>* y; ///< Data.
    std::size_t p;                ///< ARCH order.
    std::size_t q;                ///< GARCH order.
    bool has_mean;                ///< Whether mu is a parameter.
    double backcast;              ///< Presample value.

    /// @brief Number of natural parameters.
    /// @return mu? + 1 + p + q.
    [[nodiscard]] std::size_t size() const { return static_cast<std::size_t>(has_mean)+1+p+q; }
};

/// @brief Per-observation Gaussian log-likelihood terms at natural parameters.
/// @param prob Problem.
/// @param theta mu?, omega, alpha, beta.
/// @param terms Set to the n contributions.
/// @param variance When not null, set to the conditional variances.
/// @return Total log likelihood (\f$-\infty\f$ if a variance is not positive).
double loglik_terms(const Problem& prob, const std::vector<double>& theta, std::vector<double>& terms,
                    std::vector<double>* variance = nullptr) {
    const auto& y = *prob.y;
    const std::size_t n = y.size(), off = prob.has_mean ? 1 : 0;
    const double mu = prob.has_mean ? theta[0] : 0.0, omega = theta[off];
    std::vector<double> e2(n), s2(n);
    terms.resize(n);
    double total = 0;
    for (std::size_t t = 0; t < n; ++t) {
        double v = omega;
        for (std::size_t i = 1; i <= prob.p; ++i)
            v += theta[off+i]*(t >= i ? e2[t-i] : prob.backcast);
        for (std::size_t j = 1; j <= prob.q; ++j)
            v += theta[off+prob.p+j]*(t >= j ? s2[t-j] : prob.backcast);
        if (!(v > 0) || !std::isfinite(v))
            return -std::numeric_limits<double>::infinity();
        const double e = y[t]-mu;
        e2[t] = e*e;
        s2[t] = v;
        terms[t] = -.5*(std::log(2*std::numbers::pi)+std::log(v)+e2[t]/v);
        total += terms[t];
    }
    if (variance)
        *variance = s2;
    return total;
}

/// @brief Negative mean log likelihood in unconstrained coordinates.
///
/// Coordinates: mu (scaled by the sample standard deviation), log omega, and logits z with
/// \f$(\alpha,\beta)=e^{z}/(1+\sum e^{z})\f$, so the coefficients are positive with sum below one.
class Objective final : public optim::DifferentiableFunction {
public:
    /// @brief Stores the problem and mean scale.
    /// @param prob Problem.
    /// @param mean_start Mean start value.
    /// @param scale Mean scale.
    Objective(const Problem& prob, double mean_start, double scale) : prob_(prob), mean0_(mean_start), scale_(scale) {}

    /// @brief Maps coordinates to natural parameters.
    /// @param x Coordinates.
    /// @return mu?, omega, alpha, beta.
    [[nodiscard]] std::vector<double> natural(const std::vector<double>& x) const {
        std::vector<double> theta;
        std::size_t at = 0;
        if (prob_.has_mean)
            theta.push_back(mean0_+scale_*x[at++]);
        theta.push_back(std::exp(x[at++]));
        double peak = 0;
        for (std::size_t i = at; i < x.size(); ++i)
            peak = std::max(peak,x[i]);
        double denom = std::exp(-peak);
        for (std::size_t i = at; i < x.size(); ++i)
            denom += std::exp(x[i]-peak);
        for (std::size_t i = at; i < x.size(); ++i)
            theta.push_back(std::exp(x[i]-peak)/denom);
        return theta;
    }

    /// @brief Inverse of natural() for interior parameters.
    /// @param theta Natural parameters.
    /// @return Coordinates.
    [[nodiscard]] std::vector<double> coordinates(const std::vector<double>& theta) const {
        std::vector<double> x;
        std::size_t at = 0;
        if (prob_.has_mean)
            x.push_back((theta[at++]-mean0_)/scale_);
        x.push_back(std::log(theta[at++]));
        double slack = 1;
        for (std::size_t i = at; i < theta.size(); ++i)
            slack -= theta[i];
        for (std::size_t i = at; i < theta.size(); ++i)
            x.push_back(std::log(theta[i]/slack));
        return x;
    }

    /// @brief Negative mean log likelihood.
    /// @param x Coordinates.
    /// @return \f$-\ell/n\f$, or a large value outside the domain.
    double evaluate(const std::vector<double>& x) override {
        std::vector<double> terms;
        const double ll = loglik_terms(prob_,natural(x),terms);
        return std::isfinite(ll) ? -ll/static_cast<double>(prob_.y->size()) : 1e10;
    }

    /// @brief Central-difference gradient.
    /// @param x Coordinates.
    /// @return Gradient.
    std::vector<double> gradient(const std::vector<double>& x) override {
        std::vector<double> g(x.size()), z = x;
        for (std::size_t i = 0; i < x.size(); ++i) {
            const double h = 1e-6*std::max(1.0,std::abs(x[i]));
            z[i] = x[i]+h;
            const double up = evaluate(z);
            z[i] = x[i]-h;
            const double down = evaluate(z);
            z[i] = x[i];
            g[i] = (up-down)/(2*h);
        }
        return g;
    }

private:
    const Problem& prob_; ///< Problem.
    double mean0_;        ///< Mean start.
    double scale_;        ///< Mean scale.
};

/// @brief Central-difference Hessian of the total log likelihood and per-observation scores.
/// @param prob Problem.
/// @param theta Natural parameters.
/// @param hessian Set to the Hessian of the total log likelihood.
/// @param scores Set to n by k score contributions.
void derivatives(const Problem& prob, const std::vector<double>& theta, Matrix& hessian, Matrix& scores) {
    const std::size_t k = theta.size(), n = prob.y->size();
    // Steps eps^(1/4) max(|theta|, 0.1) for second differences and eps^(1/3) max(|theta|, 0.1) for scores.
    std::vector<double> h(k), hs(k);
    for (std::size_t i = 0; i < k; ++i) {
        h[i] = 1.2e-4*std::max(std::abs(theta[i]),.1);
        hs[i] = 6e-6*std::max(std::abs(theta[i]),.1);
    }
    std::vector<double> terms, up, down;
    const auto total = [&](std::vector<double> t) { return loglik_terms(prob,t,terms); };
    hessian = Matrix(k,k);
    scores = Matrix(n,k);
    const double f0 = total(theta);
    for (std::size_t i = 0; i < k; ++i) {
        auto a = theta, b = theta;
        a[i] += hs[i];
        b[i] -= hs[i];
        (void)loglik_terms(prob,a,up);
        (void)loglik_terms(prob,b,down);
        for (std::size_t t = 0; t < n; ++t)
            scores(t,i) = (up[t]-down[t])/(2*hs[i]);
        a = theta;
        b = theta;
        a[i] += h[i];
        b[i] -= h[i];
        for (std::size_t j = i; j < k; ++j) {
            double v;
            if (i == j) {
                v = (total(a)-2*f0+total(b))/(h[i]*h[i]);
            } else {
                auto pp = theta, pm = theta, mp = theta, mm = theta;
                pp[i] += h[i]; pp[j] += h[j];
                pm[i] += h[i]; pm[j] -= h[j];
                mp[i] -= h[i]; mp[j] += h[j];
                mm[i] -= h[i]; mm[j] -= h[j];
                v = (total(pp)-total(pm)-total(mp)+total(mm))/(4*h[i]*h[j]);
            }
            hessian(i,j) = hessian(j,i) = v;
        }
    }
}

/// @brief Whether natural parameters satisfy positivity and stationarity.
/// @param prob Problem.
/// @param theta Natural parameters.
/// @return Feasibility.
bool feasible(const Problem& prob, const std::vector<double>& theta) {
    const std::size_t off = prob.has_mean ? 1 : 0;
    if (!(theta[off] > 0))
        return false;
    double s = 0;
    for (std::size_t i = off+1; i < theta.size(); ++i) {
        if (!(theta[i] >= 0))
            return false;
        s += theta[i];
    }
    return s < 1;
}

} // namespace

ArchLmTest arch_lm_test(const std::vector<double>& residuals, std::size_t lags, std::size_t ddof) {
    const std::size_t n = residuals.size();
    if (lags == 0)
        throw std::invalid_argument("arch_lm_test: need at least one lag");
    if (n < 2*lags+3)
        throw std::invalid_argument("arch_lm_test: too few observations for the lags");
    if (!std::ranges::all_of(residuals,[](double v) { return std::isfinite(v); }))
        throw std::invalid_argument("arch_lm_test: residuals must be finite");
    const std::size_t m = n-lags, k = lags+1;
    Matrix gram(k,k);
    std::vector<double> xty(k,0.0), yv(m);
    std::vector<std::vector<double>> rows(m,std::vector<double>(k));
    for (std::size_t t = 0; t < m; ++t) {
        const double e = residuals[t+lags];
        yv[t] = e*e;
        rows[t][0] = 1;
        for (std::size_t l = 1; l <= lags; ++l) {
            const double f = residuals[t+lags-l];
            rows[t][l] = f*f;
        }
        for (std::size_t a = 0; a < k; ++a) {
            xty[a] += rows[t][a]*yv[t];
            for (std::size_t b = 0; b < k; ++b)
                gram(a,b) += rows[t][a]*rows[t][b];
        }
    }
    const Matrix inv = spd_inverse(gram);
    std::vector<double> coef(k,0.0);
    for (std::size_t a = 0; a < k; ++a)
        for (std::size_t b = 0; b < k; ++b)
            coef[a] += inv(a,b)*xty[b];
    double mean = 0;
    for (const double v : yv)
        mean += v/static_cast<double>(m);
    double ssr = 0, tss = 0;
    for (std::size_t t = 0; t < m; ++t) {
        double f = 0;
        for (std::size_t a = 0; a < k; ++a)
            f += rows[t][a]*coef[a];
        ssr += (yv[t]-f)*(yv[t]-f);
        tss += (yv[t]-mean)*(yv[t]-mean);
    }
    if (!(tss > 0))
        throw std::invalid_argument("arch_lm_test: squared residuals are constant");
    const double r2 = 1-ssr/tss, L = static_cast<double>(lags), md = static_cast<double>(m);
    ArchLmTest r;
    r.lags = lags;
    r.observations = m;
    r.lm = (md-static_cast<double>(ddof))*r2;
    r.lm_p_value = random::detail::gamma_probability(.5L*static_cast<long double>(r.lm),.5*L,false);
    const double df2 = md-L-1;
    r.f = (r2/L)/((1-r2)/df2);
    r.f_p_value = random::detail::regularized_incomplete_beta(.5*df2,.5*L,df2/(df2+L*r.f));
    return r;
}

GarchFit fit_garch(const std::vector<double>& y, const GarchOptions& options) {
    const std::size_t n = y.size(), p = options.arch_order, q = options.garch_order;
    const bool has_mean = options.mean == GarchMean::constant;
    if (p == 0)
        throw std::invalid_argument("fit_garch: arch_order must be positive");
    if (!std::ranges::all_of(y,[](double v) { return std::isfinite(v); }))
        throw std::invalid_argument("fit_garch: data must be finite");
    const std::size_t k = static_cast<std::size_t>(has_mean)+1+p+q;
    if (n < 10+k)
        throw std::invalid_argument("fit_garch: too few observations");
    double ybar = 0;
    for (const double v : y)
        ybar += v;
    ybar /= static_cast<double>(n);
    const double center = has_mean ? ybar : 0.0;
    double var = 0;
    for (const double v : y)
        var += (v-center)*(v-center)/static_cast<double>(n);
    if (!(var > 1e-14*(center*center+1e-300)))
        throw std::invalid_argument("fit_garch: data have no variation");
    // arch's backcast: exponentially weighted mean of the first min(75, n) squared starting residuals.
    const std::size_t tau = std::min<std::size_t>(75,n);
    double wsum = 0, back = 0, w = 1;
    for (std::size_t t = 0; t < tau; ++t) {
        back += w*(y[t]-center)*(y[t]-center);
        wsum += w;
        w *= .94;
    }
    const Problem prob{&y,p,q,has_mean,back/wsum};
    Objective objective(prob,center,std::sqrt(var/static_cast<double>(n)));

    // Several starting persistence splits; keep the best optimum.
    std::vector<double> best_theta;
    double best = std::numeric_limits<double>::infinity();
    bool converged = false;
    for (const auto& [a0,b0] : std::vector<std::pair<double,double>>{{.05,.9},{.1,.8},{.2,.6},{.3,.3}}) {
        const double sa = q == 0 ? std::min(.5,a0+b0) : a0, sb = q == 0 ? 0.0 : b0;
        std::vector<double> theta;
        if (has_mean)
            theta.push_back(center);
        theta.push_back(var*(1-sa-sb));
        for (std::size_t i = 0; i < p; ++i)
            theta.push_back(sa/static_cast<double>(p));
        for (std::size_t j = 0; j < q; ++j)
            theta.push_back(sb/static_cast<double>(q));
        auto x = objective.coordinates(theta);
        optim::LBFGSOptions controls;
        controls.max_iterations = options.max_iterations;
        controls.tolerance = 1e-10;
        const double value = optim::LBFGS(controls).optimize(objective,x);
        if (value < best) {
            best = value;
            best_theta = objective.natural(x);
            double g = 0;
            for (const double v : objective.gradient(x))
                g = std::max(g,std::abs(v));
            converged = g < 1e-5;
        }
    }
    // Newton refinement on the natural parameters while the step stays feasible and improves.
    std::vector<double> terms;
    double ll = loglik_terms(prob,best_theta,terms);
    for (int it = 0; it < 20; ++it) {
        Matrix hess, scores;
        derivatives(prob,best_theta,hess,scores);
        Matrix neg(k,k);
        for (std::size_t i = 0; i < k; ++i)
            for (std::size_t j = 0; j < k; ++j)
                neg(i,j) = -hess(i,j);
        Matrix inv;
        try {
            inv = spd_inverse(neg);
        } catch (const std::invalid_argument&) {
            break;
        }
        std::vector<double> grad(k,0.0), step(k,0.0);
        for (std::size_t t = 0; t < n; ++t)
            for (std::size_t i = 0; i < k; ++i)
                grad[i] += scores(t,i);
        for (std::size_t i = 0; i < k; ++i)
            for (std::size_t j = 0; j < k; ++j)
                step[i] += inv(i,j)*grad[j];
        bool improved = false;
        for (double s = 1; s > 1e-3; s /= 2) {
            auto trial = best_theta;
            for (std::size_t i = 0; i < k; ++i)
                trial[i] += s*step[i];
            if (!feasible(prob,trial))
                continue;
            const double v = loglik_terms(prob,trial,terms);
            if (v > ll) {
                improved = v-ll > 1e-12;
                ll = v;
                best_theta = trial;
                break;
            }
        }
        if (!improved)
            break;
        converged = true;
    }

    GarchFit fit;
    fit.parameters = best_theta;
    std::size_t at = 0;
    fit.mu = has_mean ? best_theta[at++] : 0.0;
    fit.omega = best_theta[at++];
    for (std::size_t i = 0; i < p; ++i)
        fit.alpha.push_back(best_theta[at++]);
    for (std::size_t j = 0; j < q; ++j)
        fit.beta.push_back(best_theta[at++]);
    fit.persistence = 0;
    for (const double v : fit.alpha)
        fit.persistence += v;
    for (const double v : fit.beta)
        fit.persistence += v;
    fit.unconditional_variance = fit.omega/(1-fit.persistence);
    fit.backcast = prob.backcast;
    fit.log_likelihood = loglik_terms(prob,best_theta,terms,&fit.conditional_variance);
    const double kd = static_cast<double>(k), nd = static_cast<double>(n);
    fit.aic = -2*fit.log_likelihood+2*kd;
    fit.bic = -2*fit.log_likelihood+kd*std::log(nd);
    for (std::size_t t = 0; t < n; ++t) {
        fit.residuals.push_back(y[t]-fit.mu);
        fit.standardized_residuals.push_back(fit.residuals[t]/std::sqrt(fit.conditional_variance[t]));
    }
    fit.converged = converged;
    // Covariances: classical H^{-1}/n and sandwich H^{-1} J H^{-1}/n with H, J per observation (as arch).
    Matrix hess, scores;
    derivatives(prob,best_theta,hess,scores);
    Matrix h(k,k);
    for (std::size_t i = 0; i < k; ++i)
        for (std::size_t j = 0; j < k; ++j)
            h(i,j) = -hess(i,j)/nd;
    fit.standard_errors.assign(k,std::numeric_limits<double>::quiet_NaN());
    fit.classical_standard_errors = fit.standard_errors;
    try {
        const Matrix hi = spd_inverse(h);
        std::vector<double> mean(k,0.0);
        for (std::size_t t = 0; t < n; ++t)
            for (std::size_t i = 0; i < k; ++i)
                mean[i] += scores(t,i)/nd;
        Matrix j(k,k);
        for (std::size_t t = 0; t < n; ++t)
            for (std::size_t a = 0; a < k; ++a)
                for (std::size_t b = 0; b < k; ++b)
                    j(a,b) += (scores(t,a)-mean[a])*(scores(t,b)-mean[b])/(nd-1);
        for (std::size_t i = 0; i < k; ++i) {
            fit.classical_standard_errors[i] = std::sqrt(hi(i,i)/nd);
            double s = 0;
            for (std::size_t a = 0; a < k; ++a)
                for (std::size_t b = 0; b < k; ++b)
                    s += hi(i,a)*j(a,b)*hi(b,i);
            fit.standard_errors[i] = std::sqrt(s/nd);
        }
    } catch (const std::invalid_argument&) {
        // Boundary or flat optimum: standard errors stay NaN.
    }
    return fit;
}

std::vector<double> forecast_garch_variance(const GarchFit& fit, std::size_t horizon) {
    if (horizon == 0)
        throw std::invalid_argument("forecast_garch_variance: horizon must be positive");
    const std::size_t p = fit.alpha.size(), q = fit.beta.size(), n = fit.residuals.size();
    // e2 and s2 hold known values followed by forecasts (E[e^2] = sigma^2 in the future).
    std::vector<double> e2, s2;
    for (std::size_t t = 0; t < n; ++t) {
        e2.push_back(fit.residuals[t]*fit.residuals[t]);
        s2.push_back(fit.conditional_variance[t]);
    }
    std::vector<double> out;
    for (std::size_t h = 0; h < horizon; ++h) {
        const std::size_t now = n+h;
        double v = fit.omega;
        for (std::size_t i = 1; i <= p; ++i)
            v += fit.alpha[i-1]*(now >= i ? e2[now-i] : fit.backcast);
        for (std::size_t j = 1; j <= q; ++j)
            v += fit.beta[j-1]*(now >= j ? s2[now-j] : fit.backcast);
        out.push_back(v);
        e2.push_back(v);
        s2.push_back(v);
    }
    return out;
}

} // namespace datamunge::stats
