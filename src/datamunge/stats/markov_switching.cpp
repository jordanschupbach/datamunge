/// @file markov_switching.cpp
/// @brief Hamilton filter, Kim smoother, EM start, L-BFGS refinement and Hessian standard errors.

#include <datamunge/stats/markov_switching.hpp>

#include <datamunge/linalg/lu.hpp>
#include <datamunge/optim/lbfgs.hpp>

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace datamunge::stats {
/// @brief Model state, filtering and parameter transformations.
namespace {

/// @brief Matrix alias.
using Matrix = linalg::DenseMatrix<double>;

/// @brief Parameters in natural form.
struct Params {
    Matrix p;                  ///< k by k transition, rows = source.
    Matrix beta;               ///< k by p coefficients.
    std::vector<double> var;   ///< Variances (k entries).
};

/// @brief Filter output.
struct Filtered {
    double loglik{};                  ///< Log likelihood.
    std::vector<std::vector<double>> predicted; ///< \f$P(s_t\mid y_{1:t-1})\f$.
    std::vector<std::vector<double>> filtered;  ///< \f$P(s_t\mid y_{1:t})\f$.
};

/// @brief Ergodic distribution of a row-stochastic matrix by solving \f$\pi^\top(I-P)=0,\ \sum\pi=1\f$ (least squares, as statsmodels).
/// @param p Transition matrix.
/// @return Steady-state probabilities (bounded below by 1e-20).
std::vector<double> steady_state(const Matrix& p) {
    const std::size_t k = p.rows();
    // A = [(I - P)^T; 1^T], solve normal equations A^T A pi = A^T e_{last}.
    Matrix a(k+1,k);
    for (std::size_t i = 0; i < k; ++i)
        for (std::size_t j = 0; j < k; ++j)
            a(i,j) = (i == j ? 1.0 : 0.0)-p(j,i);
    for (std::size_t j = 0; j < k; ++j)
        a(k,j) = 1;
    Matrix ata(k,k);
    std::vector<double> rhs(k);
    for (std::size_t i = 0; i < k; ++i) {
        rhs[i] = a(k,i);
        for (std::size_t j = 0; j < k; ++j)
            for (std::size_t r = 0; r <= k; ++r)
                ata(i,j) += a(r,i)*a(r,j);
    }
    const auto inv = linalg::lu(ata).inverse();
    std::vector<double> pi(k,0.0);
    for (std::size_t i = 0; i < k; ++i)
        for (std::size_t j = 0; j < k; ++j)
            pi[i] += inv(i,j)*rhs[j];
    for (double& v : pi)
        v = std::max(v,1e-20);
    return pi;
}

/// @brief Hamilton filter.
/// @param y Response.
/// @param x Design (with constant if requested).
/// @param q Parameters.
/// @param keep Whether to store probabilities.
/// @return Log likelihood and probabilities.
Filtered hamilton(const std::vector<double>& y, const Matrix& x, const Params& q, bool keep) {
    const std::size_t n = y.size(), k = q.p.rows(), p = x.cols();
    Filtered out;
    std::vector<double> prob = steady_state(q.p), pred(k), joint(k);
    for (std::size_t t = 0; t < n; ++t) {
        // One-step prediction P' prob (at t = 0 this returns the steady state itself).
        for (std::size_t j = 0; j < k; ++j) {
            pred[j] = 0;
            for (std::size_t i = 0; i < k; ++i)
                pred[j] += q.p(i,j)*prob[i];
        }
        double total = 0;
        for (std::size_t j = 0; j < k; ++j) {
            double mean = 0;
            for (std::size_t c = 0; c < p; ++c)
                mean += x(t,c)*q.beta(j,c);
            const double r = y[t]-mean;
            const double density = std::exp(-.5*r*r/q.var[j])/std::sqrt(2*std::numbers::pi*q.var[j]);
            joint[j] = pred[j]*density;
            total += joint[j];
        }
        out.loglik += std::log(total);
        for (std::size_t j = 0; j < k; ++j)
            prob[j] = joint[j]/total;
        if (keep) {
            out.predicted.push_back(pred);
            out.filtered.push_back(prob);
        }
    }
    return out;
}

/// @brief Kim smoother.
/// @param f Filter output with stored probabilities.
/// @param p Transition matrix.
/// @return Smoothed probabilities.
std::vector<std::vector<double>> kim(const Filtered& f, const Matrix& p) {
    const std::size_t n = f.filtered.size(), k = p.rows();
    std::vector<std::vector<double>> s(n);
    s[n-1] = f.filtered[n-1];
    for (std::size_t t = n-1; t-- > 0;) {
        s[t].assign(k,0.0);
        for (std::size_t i = 0; i < k; ++i) {
            double sum = 0;
            for (std::size_t j = 0; j < k; ++j)
                sum += p(i,j)*s[t+1][j]/std::max(f.predicted[t+1][j],1e-300);
            s[t][i] = f.filtered[t][i]*sum;
        }
    }
    return s;
}

/// @brief Packs natural parameters in statsmodels order.
/// @param q Parameters.
/// @param shared Whether the variance is shared.
/// @return Vector.
std::vector<double> pack(const Params& q, bool shared) {
    const std::size_t k = q.p.rows(), p = q.beta.cols();
    std::vector<double> v;
    for (std::size_t j = 0; j+1 < k; ++j)
        for (std::size_t i = 0; i < k; ++i)
            v.push_back(q.p(i,j));
    for (std::size_t c = 0; c < p; ++c)
        for (std::size_t s = 0; s < k; ++s)
            v.push_back(q.beta(s,c));
    if (shared)
        v.push_back(q.var[0]);
    else
        for (std::size_t s = 0; s < k; ++s)
            v.push_back(q.var[s]);
    return v;
}

/// @brief Unpacks natural parameters.
/// @param v Vector in statsmodels order.
/// @param k Regimes.
/// @param p Regressors.
/// @param shared Shared variance.
/// @return Parameters.
Params unpack(const std::vector<double>& v, std::size_t k, std::size_t p, bool shared) {
    Params q{Matrix(k,k),Matrix(k,p),std::vector<double>(k)};
    std::size_t idx = 0;
    for (std::size_t j = 0; j+1 < k; ++j)
        for (std::size_t i = 0; i < k; ++i)
            q.p(i,j) = v[idx++];
    for (std::size_t i = 0; i < k; ++i) {
        double sum = 0;
        for (std::size_t j = 0; j+1 < k; ++j)
            sum += q.p(i,j);
        q.p(i,k-1) = 1-sum;
    }
    for (std::size_t c = 0; c < p; ++c)
        for (std::size_t s = 0; s < k; ++s)
            q.beta(s,c) = v[idx++];
    for (std::size_t s = 0; s < k; ++s)
        q.var[s] = shared ? v[idx] : v[idx+s];
    return q;
}

/// @brief Negative log likelihood in unconstrained coordinates (softmax rows, log variances).
class Objective final : public optim::DifferentiableFunction {
public:
    /// @brief Stores the data.
    /// @param y Response.
    /// @param x Design.
    /// @param k Regimes.
    /// @param shared Shared variance.
    Objective(const std::vector<double>& y, const Matrix& x, std::size_t k, bool shared)
        : y_(y), x_(x), k_(k), shared_(shared) {}

    /// @brief Natural parameters from unconstrained coordinates.
    /// @param u Coordinates: k(k-1) logits (source-major), kp coefficients (regime-major), then log variances.
    /// @return Parameters.
    [[nodiscard]] Params natural(const std::vector<double>& u) const {
        const std::size_t k = k_, p = x_.cols();
        Params q{Matrix(k,k),Matrix(k,p),std::vector<double>(k)};
        std::size_t idx = 0;
        for (std::size_t i = 0; i < k; ++i) {
            double top = 0;
            for (std::size_t j = 0; j+1 < k; ++j)
                top = std::max(top,u[idx+j]);
            double sum = std::exp(-top);
            for (std::size_t j = 0; j+1 < k; ++j)
                sum += std::exp(u[idx+j]-top);
            for (std::size_t j = 0; j+1 < k; ++j)
                q.p(i,j) = std::exp(u[idx+j]-top)/sum;
            q.p(i,k-1) = std::exp(-top)/sum;
            idx += k-1;
        }
        for (std::size_t s = 0; s < k; ++s)
            for (std::size_t c = 0; c < p; ++c)
                q.beta(s,c) = u[idx++];
        for (std::size_t s = 0; s < k; ++s)
            q.var[s] = std::exp(shared_ ? u[idx] : u[idx+s]);
        return q;
    }

    /// @brief Unconstrained coordinates from natural parameters.
    /// @param q Parameters.
    /// @return Coordinates.
    [[nodiscard]] std::vector<double> coordinates(const Params& q) const {
        const std::size_t k = k_, p = x_.cols();
        std::vector<double> u;
        for (std::size_t i = 0; i < k; ++i)
            for (std::size_t j = 0; j+1 < k; ++j)
                u.push_back(std::log(std::max(q.p(i,j),1e-12))-std::log(std::max(q.p(i,k-1),1e-12)));
        for (std::size_t s = 0; s < k; ++s)
            for (std::size_t c = 0; c < p; ++c)
                u.push_back(q.beta(s,c));
        if (shared_)
            u.push_back(std::log(q.var[0]));
        else
            for (std::size_t s = 0; s < k; ++s)
                u.push_back(std::log(q.var[s]));
        return u;
    }

    /// @brief Negative log likelihood.
    /// @param u Coordinates.
    /// @return Value (large when not finite).
    double evaluate(const std::vector<double>& u) override {
        const double v = -hamilton(y_,x_,natural(u),false).loglik;
        return std::isfinite(v) ? v : 1e300;
    }

    /// @brief Central-difference gradient.
    /// @param u Coordinates.
    /// @return Gradient.
    std::vector<double> gradient(const std::vector<double>& u) override {
        std::vector<double> g(u.size()), z = u;
        for (std::size_t i = 0; i < u.size(); ++i) {
            const double h = 1e-6*std::max(1.0,std::abs(u[i]));
            z[i] = u[i]+h;
            const double up = evaluate(z);
            z[i] = u[i]-h;
            const double down = evaluate(z);
            z[i] = u[i];
            g[i] = (up-down)/(2*h);
        }
        return g;
    }

private:
    const std::vector<double>& y_; ///< Response.
    const Matrix& x_;              ///< Design.
    std::size_t k_;                ///< Regimes.
    bool shared_;                  ///< Shared variance.
};

/// @brief One EM step from smoothed probabilities.
/// @param y Response.
/// @param x Design.
/// @param q Current parameters.
/// @param shared Shared variance.
/// @return Updated parameters and the log likelihood at the current ones.
std::pair<Params,double> em_step(const std::vector<double>& y, const Matrix& x, const Params& q, bool shared) {
    const std::size_t n = y.size(), k = q.p.rows(), p = x.cols();
    const auto f = hamilton(y,x,q,true);
    const auto s = kim(f,q.p);
    Params next = q;
    // Transitions from smoothed joint probabilities.
    Matrix counts(k,k);
    for (std::size_t t = 1; t < n; ++t)
        for (std::size_t i = 0; i < k; ++i)
            for (std::size_t j = 0; j < k; ++j)
                counts(i,j) += f.filtered[t-1][i]*q.p(i,j)*s[t][j]/std::max(f.predicted[t][j],1e-300);
    for (std::size_t i = 0; i < k; ++i) {
        double row = 0;
        for (std::size_t j = 0; j < k; ++j)
            row += counts(i,j);
        for (std::size_t j = 0; j < k; ++j)
            next.p(i,j) = row > 0 ? counts(i,j)/row : 1.0/static_cast<double>(k);
    }
    // Weighted least squares per regime.
    double pooled = 0;
    for (std::size_t r = 0; r < k; ++r) {
        Matrix xtx(p,p);
        std::vector<double> xty(p,0.0);
        double weight = 0;
        for (std::size_t t = 0; t < n; ++t) {
            weight += s[t][r];
            for (std::size_t a = 0; a < p; ++a) {
                xty[a] += s[t][r]*x(t,a)*y[t];
                for (std::size_t b = 0; b < p; ++b)
                    xtx(a,b) += s[t][r]*x(t,a)*x(t,b);
            }
        }
        if (p > 0) {
            const auto inv = linalg::lu(xtx).inverse();
            for (std::size_t a = 0; a < p; ++a) {
                next.beta(r,a) = 0;
                for (std::size_t b = 0; b < p; ++b)
                    next.beta(r,a) += inv(a,b)*xty[b];
            }
        }
        double ss = 0;
        for (std::size_t t = 0; t < n; ++t) {
            double mean = 0;
            for (std::size_t a = 0; a < p; ++a)
                mean += x(t,a)*next.beta(r,a);
            ss += s[t][r]*(y[t]-mean)*(y[t]-mean);
        }
        pooled += ss;
        next.var[r] = ss/std::max(weight,1e-300);
    }
    if (shared)
        std::ranges::fill(next.var,pooled/static_cast<double>(n));
    for (double& v : next.var)
        v = std::max(v,1e-10);
    return {next,f.loglik};
}

} // namespace

MarkovSwitchingFit fit_markov_switching(const std::vector<double>& y, const linalg::DenseMatrix<double>& x,
                                        const MarkovSwitchingOptions& options) {
    const std::size_t n = y.size(), k = options.regimes;
    if (k < 2)
        throw std::invalid_argument("fit_markov_switching: need at least two regimes");
    if (x.rows() != n)
        throw std::invalid_argument("fit_markov_switching: x must have one row per observation");
    const std::size_t p = x.cols()+(options.intercept ? 1 : 0);
    if (p == 0)
        throw std::invalid_argument("fit_markov_switching: need a constant or at least one regressor");
    const bool shared = !options.switching_variance;
    const std::size_t q_count = k*(k-1)+k*p+(shared ? 1 : k);
    if (n < std::max<std::size_t>(10,2*q_count))
        throw std::invalid_argument("fit_markov_switching: too few observations for the parameters");
    Matrix design(n,p);
    for (std::size_t t = 0; t < n; ++t) {
        if (!std::isfinite(y[t]))
            throw std::invalid_argument("fit_markov_switching: y must be finite");
        std::size_t c = 0;
        if (options.intercept)
            design(t,c++) = 1;
        for (std::size_t j = 0; j < x.cols(); ++j) {
            if (!std::isfinite(x(t,j)))
                throw std::invalid_argument("fit_markov_switching: x must be finite");
            design(t,c++) = x(t,j);
        }
    }

    // statsmodels' start: equal transitions, beta_s = OLS * s/k, variances linspace(v/10, v).
    Matrix xtx(p,p);
    std::vector<double> xty(p,0.0), ols(p,0.0);
    for (std::size_t t = 0; t < n; ++t)
        for (std::size_t a = 0; a < p; ++a) {
            xty[a] += design(t,a)*y[t];
            for (std::size_t b = 0; b < p; ++b)
                xtx(a,b) += design(t,a)*design(t,b);
        }
    const auto inv = linalg::lu(xtx).inverse();
    for (std::size_t a = 0; a < p; ++a)
        for (std::size_t b = 0; b < p; ++b)
            ols[a] += inv(a,b)*xty[b];
    double rbar = 0, rss = 0;
    std::vector<double> resid(n);
    for (std::size_t t = 0; t < n; ++t) {
        double m = 0;
        for (std::size_t a = 0; a < p; ++a)
            m += design(t,a)*ols[a];
        resid[t] = y[t]-m;
        rbar += resid[t];
    }
    rbar /= static_cast<double>(n);
    for (const double r : resid)
        rss += (r-rbar)*(r-rbar);
    const double variance = rss/static_cast<double>(n);
    if (!(variance > 0))
        throw std::invalid_argument("fit_markov_switching: the regression fits exactly");
    Params q{Matrix(k,k,1.0/static_cast<double>(k)),Matrix(k,p),std::vector<double>(k)};
    for (std::size_t s = 0; s < k; ++s) {
        for (std::size_t a = 0; a < p; ++a)
            q.beta(s,a) = ols[a]*static_cast<double>(s)/static_cast<double>(k);
        q.var[s] = shared ? variance : variance/10+(variance-variance/10)*static_cast<double>(s)/static_cast<double>(k-1);
    }

    double previous = -INFINITY;
    for (std::size_t it = 0; it < options.em_iterations; ++it) {
        auto [next, ll] = em_step(y,design,q,shared);
        q = next;
        if (std::abs(ll-previous) < 1e-10*std::max(1.0,std::abs(ll)))
            break;
        previous = ll;
    }

    Objective objective(y,design,k,shared);
    auto u = objective.coordinates(q);
    optim::LBFGSOptions controls;
    controls.max_iterations = options.max_iterations;
    controls.tolerance = 1e-10;
    (void)optim::LBFGS(controls).optimize(objective,u);
    // Safeguarded Newton steps with a finite-difference Hessian of the gradient.
    for (int it = 0; it < 5; ++it) {
        const auto g = objective.gradient(u);
        const std::size_t d = u.size();
        Matrix h(d,d);
        auto z = u;
        for (std::size_t j = 0; j < d; ++j) {
            const double step = 1e-4*std::max(1.0,std::abs(u[j]));
            z[j] = u[j]+step;
            const auto gp = objective.gradient(z);
            z[j] = u[j]-step;
            const auto gm = objective.gradient(z);
            z[j] = u[j];
            for (std::size_t i = 0; i < d; ++i)
                h(i,j) = (gp[i]-gm[i])/(2*step);
        }
        for (std::size_t i = 0; i < d; ++i)
            for (std::size_t j = 0; j < i; ++j)
                h(i,j) = h(j,i) = .5*(h(i,j)+h(j,i));
        std::vector<double> delta;
        try {
            delta = linalg::lu(h).solve(g);
        } catch (const std::exception&) {
            break;
        }
        const double f0 = objective.evaluate(u);
        bool improved = false;
        for (double scale = 1; scale > 1e-4; scale /= 2) {
            for (std::size_t i = 0; i < d; ++i)
                z[i] = u[i]-scale*delta[i];
            if (objective.evaluate(z) <= f0) {
                u = z;
                improved = true;
                break;
            }
        }
        if (!improved)
            break;
    }
    q = objective.natural(u);
    double largest = 0;
    for (const double g : objective.gradient(u))
        largest = std::max(largest,std::abs(g));

    MarkovSwitchingFit fit;
    fit.converged = largest < 1e-5;
    fit.transition = q.p;
    fit.coefficients = q.beta;
    fit.variances = q.var;
    fit.steady_state = steady_state(q.p);
    for (std::size_t s = 0; s < k; ++s)
        fit.expected_durations.push_back(1/(1-q.p(s,s)));
    fit.parameters = pack(q,shared);
    const auto f = hamilton(y,design,q,true);
    fit.log_likelihood = f.loglik;
    const double m = static_cast<double>(fit.parameters.size());
    fit.aic = -2*f.loglik+2*m;
    fit.bic = -2*f.loglik+m*std::log(static_cast<double>(n));
    const auto smoothed = kim(f,q.p);
    fit.filtered_probabilities = Matrix(n,k);
    fit.smoothed_probabilities = Matrix(n,k);
    for (std::size_t t = 0; t < n; ++t)
        for (std::size_t s = 0; s < k; ++s) {
            fit.filtered_probabilities(t,s) = f.filtered[t][s];
            fit.smoothed_probabilities(t,s) = smoothed[t][s];
        }

    // Numerical Hessian of the log likelihood in the natural (statsmodels) parameters.
    const std::size_t d = fit.parameters.size();
    const auto ll = [&](const std::vector<double>& v) {
        return hamilton(y,design,unpack(v,k,p,shared),false).loglik;
    };
    std::vector<double> h(d);
    for (std::size_t i = 0; i < d; ++i)
        h[i] = 1e-4*std::max(std::abs(fit.parameters[i]),1e-2);
    Matrix hess(d,d);
    const double f0 = ll(fit.parameters);
    auto v = fit.parameters;
    for (std::size_t i = 0; i < d; ++i) {
        v[i] = fit.parameters[i]+h[i];
        const double fp = ll(v);
        v[i] = fit.parameters[i]-h[i];
        const double fm = ll(v);
        v[i] = fit.parameters[i];
        hess(i,i) = (fp-2*f0+fm)/(h[i]*h[i]);
        for (std::size_t j = 0; j < i; ++j) {
            double sum = 0;
            for (const int a : {1, -1})
                for (const int b : {1, -1}) {
                    v[i] = fit.parameters[i]+a*h[i];
                    v[j] = fit.parameters[j]+b*h[j];
                    sum += a*b*ll(v);
                }
            v[i] = fit.parameters[i];
            v[j] = fit.parameters[j];
            hess(i,j) = hess(j,i) = sum/(4*h[i]*h[j]);
        }
    }
    fit.standard_errors.assign(d,NAN);
    try {
        const auto cov = linalg::lu(hess).inverse();
        for (std::size_t i = 0; i < d; ++i)
            fit.standard_errors[i] = -cov(i,i) > 0 ? std::sqrt(-cov(i,i)) : NAN;
    } catch (const std::exception&) {
    }
    return fit;
}

} // namespace datamunge::stats
