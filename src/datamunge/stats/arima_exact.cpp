/// @file arima_exact.cpp
/// @brief State space form, Kalman filter, PACF reparameterization and L-BFGS fitting for exact ARIMA likelihoods.

#include <datamunge/stats/arima_exact.hpp>

#include <datamunge/linalg/eigen.hpp>
#include <datamunge/optim/function_types.hpp>
#include <datamunge/optim/lbfgs.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>

namespace datamunge::stats {
/// @brief State space construction, filtering and parameter transforms.
namespace {

/// @brief Dense double matrix used for state covariances.
using Matrix = linalg::DenseMatrix<double>;

/// @brief Diffuse prior variance of the differencing states, as R's `kappa`.
constexpr double kappa = 1e6;

/// @brief ARIMA model in the state space form of R's `makeARIMA`.
struct StateSpace {
    std::vector<double> phi;    ///< Expanded AR coefficients.
    std::vector<double> theta;  ///< Expanded MA coefficients.
    std::vector<double> delta;  ///< Differencing coefficients: \f$y_t=\sum\delta_iy_{t-i}+w_t\f$.
    std::size_t r{};            ///< ARMA block dimension.
    std::size_t rd{};           ///< Full state dimension.
    Matrix initial;             ///< \f$P_{1|0}\f$.
};

/// @brief Product of two polynomials given by coefficient vectors with leading 1 omitted.
/// @param a First polynomial \f$1+\sum a_iB^i\f$.
/// @param b Second polynomial \f$1+\sum b_iB^i\f$.
/// @return Coefficients of the product, leading 1 omitted.
std::vector<double> multiply(const std::vector<double>& a, const std::vector<double>& b) {
    std::vector<double> full_a{1}, full_b{1};
    full_a.insert(full_a.end(),a.begin(),a.end());
    full_b.insert(full_b.end(),b.begin(),b.end());
    std::vector<double> out(full_a.size()+full_b.size()-1,0);
    for (std::size_t i = 0; i < full_a.size(); ++i)
        for (std::size_t j = 0; j < full_b.size(); ++j)
            out[i+j] += full_a[i]*full_b[j];
    return {out.begin()+1,out.end()};
}

/// @brief Places seasonal coefficients at multiples of the period.
/// @param c Seasonal coefficients.
/// @param s Period.
/// @param sign Multiplier applied to each coefficient.
/// @return Coefficient vector of length \f$s\cdot\mathrm{len}(c)\f$.
std::vector<double> spread(const std::vector<double>& c, std::size_t s, double sign) {
    std::vector<double> out(c.size()*s,0);
    for (std::size_t i = 0; i < c.size(); ++i)
        out[(i+1)*s-1] = sign*c[i];
    return out;
}

/// @brief Validates orders and coefficient lengths.
/// @param order Orders.
/// @param c Coefficients.
/// @throws std::invalid_argument On mismatch or an invalid period.
void check_order(const ArimaOrder& order, const ArimaCoefficients* c) {
    const bool seasonal = order.seasonal_p || order.seasonal_d || order.seasonal_q;
    if (seasonal && order.period < 2)
        throw std::invalid_argument("arima: seasonal terms need a period of at least 2");
    if (c && (c->ar.size() != order.p || c->ma.size() != order.q || c->seasonal_ar.size() != order.seasonal_p ||
              c->seasonal_ma.size() != order.seasonal_q))
        throw std::invalid_argument("arima: coefficient lengths must match the orders");
    if (c)
        for (const auto* v : {&c->ar,&c->ma,&c->seasonal_ar,&c->seasonal_ma})
            for (const double x : *v)
                if (!std::isfinite(x))
                    throw std::invalid_argument("arima: coefficients must be finite");
}

/// @brief Whether \f$1-\sum\phi_iB^i\f$ has all roots outside the unit circle (step-down recursion).
/// @param phi AR coefficients.
/// @return True when stationary.
bool stationary(std::vector<double> phi) {
    for (std::size_t j = phi.size(); j-- > 0;) {
        const double a = phi[j];
        if (!(std::abs(a) < 1))
            return false;
        std::vector<double> work(j);
        for (std::size_t k = 0; k < j; ++k)
            work[k] = (phi[k]+a*phi[j-k-1])/(1-a*a);
        for (std::size_t k = 0; k < j; ++k)
            phi[k] = work[k];
    }
    return true;
}

/// @brief Applies the transition matrix T to a state vector.
/// @param m Model.
/// @param x State of length rd.
/// @return \f$Tx\f$.
std::vector<double> transition(const StateSpace& m, const std::vector<double>& x) {
    const std::size_t r = m.r, d = m.delta.size();
    std::vector<double> out(m.rd);
    for (std::size_t i = 0; i < r; ++i) {
        double v = i+1 < r ? x[i+1] : 0.0;
        if (i < m.phi.size())
            v += m.phi[i]*x[0];
        out[i] = v;
    }
    if (d > 0) {
        double v = x[0];
        for (std::size_t i = 0; i < d; ++i)
            v += m.delta[i]*x[r+i];
        out[r] = v;
        for (std::size_t i = r+1; i < m.rd; ++i)
            out[i] = x[i-1];
    }
    return out;
}

/// @brief \f$TPT^\top+V\f$ with V the ARMA disturbance covariance \f$RR^\top\f$.
/// @param m Model.
/// @param p Covariance.
/// @return Predicted covariance.
Matrix predict_covariance(const StateSpace& m, const Matrix& p) {
    const std::size_t n = m.rd;
    Matrix tp(n,n), out(n,n);
    std::vector<double> column(n);
    for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t i = 0; i < n; ++i)
            column[i] = p(i,j);
        const auto c = transition(m,column);
        for (std::size_t i = 0; i < n; ++i)
            tp(i,j) = c[i];
    }
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j)
            column[j] = tp(i,j);
        const auto c = transition(m,column);
        for (std::size_t j = 0; j < n; ++j)
            out(i,j) = c[j];
    }
    std::vector<double> rvec(m.r,0);
    rvec[0] = 1;
    for (std::size_t j = 0; j < m.theta.size() && j+1 < m.r; ++j)
        rvec[j+1] = m.theta[j];
    for (std::size_t i = 0; i < m.r; ++i)
        for (std::size_t j = 0; j < m.r; ++j)
            out(i,j) += rvec[i]*rvec[j];
    return out;
}

/// @brief Builds the state space model and its initial covariance.
/// @param order Orders.
/// @param c Coefficients.
/// @return Model with \f$P_{1|0}\f$.
/// @throws std::invalid_argument When the AR part is not stationary.
StateSpace build(const ArimaOrder& order, const ArimaCoefficients& c) {
    StateSpace m;
    // AR: (1 - sum phi B)(1 - sum Phi B^s) = 1 - sum a B; in "leading 1 omitted" form with negated signs.
    std::vector<double> neg_ar(c.ar.size());
    for (std::size_t i = 0; i < c.ar.size(); ++i)
        neg_ar[i] = -c.ar[i];
    const auto ar_poly = multiply(neg_ar,spread(c.seasonal_ar,order.period,-1));
    m.phi.resize(ar_poly.size());
    for (std::size_t i = 0; i < ar_poly.size(); ++i)
        m.phi[i] = -ar_poly[i];
    m.theta = multiply(c.ma,spread(c.seasonal_ma,order.period,1));
    if (!stationary(c.ar) || !stationary(c.seasonal_ar))
        throw std::invalid_argument("arima: non-stationary AR part");
    std::vector<double> diff_poly;
    for (std::size_t i = 0; i < order.d; ++i)
        diff_poly = multiply(diff_poly,{-1});
    std::vector<double> seasonal_diff(order.period,0);
    if (order.period)
        seasonal_diff.back() = -1;
    for (std::size_t i = 0; i < order.seasonal_d; ++i)
        diff_poly = multiply(diff_poly,seasonal_diff);
    m.delta.resize(diff_poly.size());
    for (std::size_t i = 0; i < diff_poly.size(); ++i)
        m.delta[i] = -diff_poly[i];
    m.r = std::max(m.phi.size(),m.theta.size()+1);
    m.rd = m.r+m.delta.size();

    // Stationary covariance of the ARMA block by doubling: Q = sum_k A^k V A'^k.
    const std::size_t r = m.r;
    Matrix a(r,r), q(r,r);
    for (std::size_t i = 0; i < r; ++i) {
        if (i < m.phi.size())
            a(i,0) = m.phi[i];
        if (i+1 < r)
            a(i,i+1) = 1;
    }
    std::vector<double> rvec(r,0);
    rvec[0] = 1;
    for (std::size_t j = 0; j < m.theta.size(); ++j)
        rvec[j+1] = m.theta[j];
    for (std::size_t i = 0; i < r; ++i)
        for (std::size_t j = 0; j < r; ++j)
            q(i,j) = rvec[i]*rvec[j];
    for (int it = 0; it < 200; ++it) {
        Matrix aq(r,r), aqa(r,r), aa(r,r);
        for (std::size_t i = 0; i < r; ++i)
            for (std::size_t k = 0; k < r; ++k) {
                const double v = a(i,k);
                if (v == 0)
                    continue;
                for (std::size_t j = 0; j < r; ++j) {
                    aq(i,j) += v*q(k,j);
                    aa(i,j) += v*a(k,j);
                }
            }
        double change = 0, size = 0;
        for (std::size_t i = 0; i < r; ++i)
            for (std::size_t j = 0; j < r; ++j) {
                double v = 0;
                for (std::size_t k = 0; k < r; ++k)
                    v += aq(i,k)*a(j,k);
                aqa(i,j) = v;
                change = std::max(change,std::abs(v));
                size = std::max(size,std::abs(q(i,j)));
            }
        for (std::size_t i = 0; i < r; ++i)
            for (std::size_t j = 0; j < r; ++j)
                q(i,j) += aqa(i,j);
        a = aa;
        if (change <= 1e-16*size)
            break;
    }
    m.initial = Matrix(m.rd,m.rd);
    for (std::size_t i = 0; i < r; ++i)
        for (std::size_t j = 0; j < r; ++j)
            m.initial(i,j) = q(i,j);
    for (std::size_t i = r; i < m.rd; ++i)
        m.initial(i,i) = kappa;
    return m;
}

/// @brief Kalman filter output.
struct FilterResult {
    double ssq{};                   ///< \f$\sum v_t^2/F_t\f$.
    double sumlog{};                ///< \f$\sum\log F_t\f$.
    std::size_t nu{};               ///< Terms in the sums.
    std::size_t nonmissing{};       ///< Nonmissing observations.
    std::vector<double> residuals;  ///< \f$v_t/\sqrt{F_t}\f$, NaN where missing.
    std::vector<double> state;      ///< Final filtered state.
    Matrix covariance;              ///< Final filtered covariance.
};

/// @brief Runs the filter of R's `ARIMA_Like`.
/// @param y Series with NaN for missing values.
/// @param offset Regression mean subtracted from y at each time, or empty for none.
/// @param m Model.
/// @return Sums, residuals and the final state.
FilterResult filter(const std::vector<double>& y, const std::vector<double>& offset, const StateSpace& m) {
    const std::size_t n = m.rd, r = m.r, d = m.delta.size();
    FilterResult out;
    out.residuals.assign(y.size(),std::numeric_limits<double>::quiet_NaN());
    std::vector<double> a(n,0);
    Matrix p;
    std::vector<double> zp(n);
    for (std::size_t t = 0; t < y.size(); ++t) {
        auto anew = transition(m,a);
        Matrix pnew = t == 0 ? m.initial : predict_covariance(m,p);
        if (std::isnan(y[t])) {
            a = std::move(anew);
            p = std::move(pnew);
            continue;
        }
        ++out.nonmissing;
        double resid = y[t]-(offset.empty() ? 0.0 : offset[t])-anew[0];
        for (std::size_t i = 0; i < d; ++i)
            resid -= m.delta[i]*anew[r+i];
        // M = Pnew Z with Z = (1, 0, ..., 0, delta).
        for (std::size_t i = 0; i < n; ++i) {
            double v = pnew(i,0);
            for (std::size_t j = 0; j < d; ++j)
                v += pnew(i,r+j)*m.delta[j];
            zp[i] = v;
        }
        double gain = zp[0];
        for (std::size_t j = 0; j < d; ++j)
            gain += m.delta[j]*zp[r+j];
        if (gain < 1e4) {
            ++out.nu;
            out.ssq += resid*resid/gain;
            out.sumlog += std::log(gain);
        }
        out.residuals[t] = resid/std::sqrt(gain);
        for (std::size_t i = 0; i < n; ++i)
            anew[i] += zp[i]*resid/gain;
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
                pnew(i,j) -= zp[i]*zp[j]/gain;
        a = std::move(anew);
        p = std::move(pnew);
    }
    out.state = a;
    out.covariance = y.empty() ? Matrix(n,n) : p;
    return out;
}

/// @brief Profile log likelihood from filter sums, following R's `arima`.
/// @param f Filter output.
/// @param used Nonmissing observations minus the differencing length.
/// @return Log likelihood, or minus infinity when undefined.
double profile_log_likelihood(const FilterResult& f, std::size_t used) {
    if (f.nu == 0 || !(f.ssq > 0))
        return -std::numeric_limits<double>::infinity();
    const double s2 = f.ssq/static_cast<double>(f.nu);
    const double value = .5*std::log(s2)+.5*f.sumlog/static_cast<double>(f.nu);
    const double nn = static_cast<double>(used);
    return -.5*(2*nn*value+nn+nn*std::log(2*std::numbers::pi));
}

/// @brief Maps unconstrained values to a stationary AR polynomial (R's `partrans`).
/// @param raw Unconstrained values.
/// @return Coefficients \f$\phi\f$ with all roots of \f$1-\sum\phi_iB^i\f$ outside the unit circle.
std::vector<double> partrans(const std::vector<double>& raw) {
    const std::size_t p = raw.size();
    std::vector<double> work(p), out(p);
    for (std::size_t j = 0; j < p; ++j)
        work[j] = out[j] = std::tanh(raw[j]);
    for (std::size_t j = 1; j < p; ++j) {
        const double a = out[j];
        for (std::size_t k = 0; k < j; ++k)
            work[k] -= a*out[j-k-1];
        for (std::size_t k = 0; k < j; ++k)
            out[k] = work[k];
    }
    return out;
}

/// @brief Regression part of the model: design columns, search origin and scale.
struct Regression {
    linalg::DenseMatrix<double> design; ///< n by m: a column of ones for the mean, then the regressors.
    std::vector<double> start;          ///< OLS start values.
    std::vector<double> scale;          ///< Search scale of each coefficient.

    /// @brief Mean at each time for coefficients beta.
    /// @param beta Coefficients, one per design column.
    /// @return \f$X\beta\f$, or empty when there are no columns.
    [[nodiscard]] std::vector<double> offset(const std::vector<double>& beta) const {
        if (beta.empty())
            return {};
        std::vector<double> out(design.rows(),0);
        for (std::size_t t = 0; t < design.rows(); ++t)
            for (std::size_t j = 0; j < beta.size(); ++j)
                out[t] += design(t,j)*beta[j];
        return out;
    }
};

/// @brief Applies the differencing filter \f$z_t=x_t-\sum\delta_ix_{t-i}\f$.
/// @param x Series.
/// @param delta Differencing coefficients.
/// @return Differenced series of length \f$n-\mathrm{len}(\delta)\f$ (NaN propagates).
std::vector<double> difference(const std::vector<double>& x, const std::vector<double>& delta) {
    std::vector<double> out;
    for (std::size_t t = delta.size(); t < x.size(); ++t) {
        double v = x[t];
        for (std::size_t i = 0; i < delta.size(); ++i)
            v -= delta[i]*x[t-i-1];
        out.push_back(v);
    }
    return out;
}

/// @brief OLS start values and scales from the differenced series, as R's `arima`.
/// @param y Series.
/// @param design Regression design (may have no columns).
/// @param delta Differencing coefficients.
/// @return Regression with start values and scales.
Regression start_regression(const std::vector<double>& y, const linalg::DenseMatrix<double>& design,
                            const std::vector<double>& delta) {
    Regression reg;
    reg.design = design;
    const std::size_t m = design.cols();
    reg.start.assign(m,0);
    reg.scale.assign(m,1);
    if (m == 0)
        return reg;
    const auto dy = difference(y,delta);
    std::vector<std::vector<double>> dx(m);
    for (std::size_t j = 0; j < m; ++j) {
        std::vector<double> column(design.rows());
        for (std::size_t t = 0; t < design.rows(); ++t)
            column[t] = design(t,j);
        dx[j] = difference(column,delta);
    }
    std::vector<std::size_t> rows;
    for (std::size_t t = 0; t < dy.size(); ++t)
        if (!std::isnan(dy[t]))
            rows.push_back(t);
    if (rows.size() <= m)
        return reg;
    Matrix gram(m,m);
    std::vector<double> xty(m,0);
    for (const auto t : rows)
        for (std::size_t a = 0; a < m; ++a) {
            xty[a] += dx[a][t]*dy[t];
            for (std::size_t b = 0; b < m; ++b)
                gram(a,b) += dx[a][t]*dx[b][t];
        }
    const auto eig = linalg::symmetric_eigen(gram);
    if (!(eig.eigenvalues.back() > 1e-12*eig.eigenvalues.front()))
        return reg; // regressors collinear after differencing: start at zero
    Matrix inverse(m,m);
    for (std::size_t k = 0; k < m; ++k)
        for (std::size_t a = 0; a < m; ++a)
            for (std::size_t b = 0; b < m; ++b)
                inverse(a,b) += eig.eigenvectors(a,k)*eig.eigenvectors(b,k)/eig.eigenvalues[k];
    for (std::size_t a = 0; a < m; ++a)
        for (std::size_t b = 0; b < m; ++b)
            reg.start[a] += inverse(a,b)*xty[b];
    double rss = 0;
    for (const auto t : rows) {
        double fit = 0;
        for (std::size_t a = 0; a < m; ++a)
            fit += dx[a][t]*reg.start[a];
        rss += (dy[t]-fit)*(dy[t]-fit);
    }
    const double s2 = rss/static_cast<double>(rows.size()-m);
    for (std::size_t a = 0; a < m; ++a) {
        const double sd = std::sqrt(s2*inverse(a,a)*static_cast<double>(rows.size()));
        if (sd > 0 && std::isfinite(sd))
            reg.scale[a] = sd;
    }
    return reg;
}

/// @brief Objective for L-BFGS: negative profile log likelihood per observation in search coordinates.
class Objective final : public optim::DifferentiableFunction {
public:
    /// @brief Stores the data and parameter layout.
    /// @param y Series.
    /// @param order Orders.
    /// @param regression Regression design, start values and scales.
    /// @param used Observation count for the likelihood.
    Objective(const std::vector<double>& y, const ArimaOrder& order, const Regression& regression, std::size_t used)
        : y_(y), order_(order), regression_(regression), used_(used) {}

    /// @brief Coefficients and regression coefficients from search coordinates.
    /// @param x Search coordinates.
    /// @param c Set to the ARMA coefficients.
    /// @param beta Set to the regression coefficients.
    void decode(const std::vector<double>& x, ArimaCoefficients& c, std::vector<double>& beta) const {
        std::size_t at = 0;
        const auto take = [&](std::size_t k) {
            std::vector<double> v(x.begin()+static_cast<std::ptrdiff_t>(at),x.begin()+static_cast<std::ptrdiff_t>(at+k));
            at += k;
            return v;
        };
        c.ar = partrans(take(order_.p));
        c.ma = partrans(take(order_.q));
        for (auto& v : c.ma)
            v = -v;
        c.seasonal_ar = partrans(take(order_.seasonal_p));
        c.seasonal_ma = partrans(take(order_.seasonal_q));
        for (auto& v : c.seasonal_ma)
            v = -v;
        beta.resize(regression_.start.size());
        for (std::size_t j = 0; j < beta.size(); ++j)
            beta[j] = regression_.start[j]+regression_.scale[j]*x[at+j];
    }

    /// @brief Negative log likelihood per observation, as R's `armafn` scale.
    /// @param x Search coordinates.
    /// @return \f$-\ell/n\f$, or a large value where the likelihood is undefined.
    double evaluate(const std::vector<double>& x) override {
        ArimaCoefficients c;
        std::vector<double> beta;
        decode(x,c,beta);
        double value;
        try {
            value = -profile_log_likelihood(filter(y_,regression_.offset(beta),build(order_,c)),used_)/
                    static_cast<double>(used_);
        } catch (const std::invalid_argument&) {
            value = std::numeric_limits<double>::infinity();
        }
        return std::isfinite(value) ? value : 1e300;
    }

    /// @brief Central-difference gradient.
    /// @param x Search coordinates.
    /// @return Gradient of evaluate().
    std::vector<double> gradient(const std::vector<double>& x) override {
        std::vector<double> g(x.size()), z = x;
        for (std::size_t i = 0; i < x.size(); ++i) {
            const double h = 1e-5;
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
    const std::vector<double>& y_; ///< Series.
    ArimaOrder order_;             ///< Orders.
    const Regression& regression_; ///< Regression part.
    std::size_t used_;             ///< Likelihood observation count.
};

/// @brief Splits a parameter vector in R's order into ARMA and regression coefficients.
/// @param order Orders.
/// @param v Parameters: ar, ma, sar, sma, then regression coefficients.
/// @param c Set to the coefficients.
/// @param beta Set to the regression coefficients.
void unpack(const ArimaOrder& order, const std::vector<double>& v, ArimaCoefficients& c, std::vector<double>& beta) {
    std::size_t at = 0;
    const auto take = [&](std::size_t k) {
        std::vector<double> out(v.begin()+static_cast<std::ptrdiff_t>(at),v.begin()+static_cast<std::ptrdiff_t>(at+k));
        at += k;
        return out;
    };
    c.ar = take(order.p);
    c.ma = take(order.q);
    c.seasonal_ar = take(order.seasonal_p);
    c.seasonal_ma = take(order.seasonal_q);
    beta.assign(v.begin()+static_cast<std::ptrdiff_t>(at),v.end());
}

} // namespace

ArimaLikelihood arima_likelihood(const std::vector<double>& y, const ArimaOrder& order,
                                 const ArimaCoefficients& coefficients, double mean) {
    check_order(order,&coefficients);
    if (!std::isfinite(mean) || std::ranges::any_of(y,[](double v) { return std::isinf(v); }))
        throw std::invalid_argument("arima_likelihood: values must be finite or NaN, and the mean finite");
    const auto model = build(order,coefficients);
    const auto f = filter(y,std::vector<double>(y.size(),mean),model);
    if (f.nonmissing <= model.delta.size() || f.nu == 0)
        throw std::invalid_argument("arima_likelihood: too few observations");
    ArimaLikelihood out;
    out.observations_used = f.nonmissing-model.delta.size();
    out.log_likelihood = profile_log_likelihood(f,out.observations_used);
    out.sigma2 = f.ssq/static_cast<double>(f.nu);
    out.residuals = f.residuals;
    return out;
}

ExactArimaFit fit_arima_exact(const std::vector<double>& y, const ExactArimaOptions& options) {
    const auto& order = options.order;
    check_order(order,nullptr);
    if (std::ranges::any_of(y,[](double v) { return std::isinf(v); }))
        throw std::invalid_argument("fit_arima_exact: values must be finite or NaN");
    const auto& xreg = options.regressors;
    const std::size_t nx = xreg.rows() == 0 && xreg.cols() == 0 ? 0 : xreg.cols();
    if (nx > 0 && xreg.rows() != y.size())
        throw std::invalid_argument("fit_arima_exact: regressors need one row per observation");
    for (std::size_t t = 0; t < xreg.rows(); ++t)
        for (std::size_t j = 0; j < nx; ++j)
            if (!std::isfinite(xreg(t,j)))
                throw std::invalid_argument("fit_arima_exact: regressors must be finite");
    if (!options.regressor_names.empty() && options.regressor_names.size() != nx)
        throw std::invalid_argument("fit_arima_exact: need one name per regressor");
    const bool fit_mean = options.include_mean && order.d == 0 && order.seasonal_d == 0;
    const std::size_t m = nx+(fit_mean ? 1 : 0);
    const std::size_t arma = order.p+order.q+order.seasonal_p+order.seasonal_q;
    const std::size_t k = arma+m;
    std::size_t nonmissing = 0;
    for (const double v : y)
        nonmissing += !std::isnan(v);
    const std::size_t differencing = order.d+order.seasonal_d*order.period;
    if (nonmissing <= differencing+k+1)
        throw std::invalid_argument("fit_arima_exact: too few observations for the model");
    const std::size_t used = nonmissing-differencing;

    linalg::DenseMatrix<double> design(y.size(),m);
    for (std::size_t t = 0; t < y.size(); ++t) {
        std::size_t c = 0;
        if (fit_mean)
            design(t,c++) = 1;
        for (std::size_t j = 0; j < nx; ++j)
            design(t,c++) = xreg(t,j);
    }
    const auto model0 = build(order,{std::vector<double>(order.p,0),std::vector<double>(order.q,0),
                                      std::vector<double>(order.seasonal_p,0),std::vector<double>(order.seasonal_q,0)});
    const auto regression = start_regression(y,design,model0.delta);

    Objective objective(y,order,regression,used);
    std::vector<double> x(k,0.0);
    optim::LBFGSOptions controls;
    controls.max_iterations = options.max_iterations;
    controls.tolerance = 1e-9;
    (void)optim::LBFGS(controls).optimize(objective,x);
    const auto g = objective.gradient(x);
    double largest = 0;
    for (const double v : g)
        largest = std::max(largest,std::abs(v));

    ExactArimaFit fit;
    fit.order = order;
    std::vector<double> beta;
    objective.decode(x,fit.coefficients,beta);
    fit.mean = fit_mean ? beta[0] : 0.0;
    fit.regression.assign(beta.begin()+(fit_mean ? 1 : 0),beta.end());
    fit.converged = largest <= 1e-4;
    for (const auto* part : {&fit.coefficients.ar,&fit.coefficients.ma,&fit.coefficients.seasonal_ar,
                             &fit.coefficients.seasonal_ma})
        fit.estimates.insert(fit.estimates.end(),part->begin(),part->end());
    fit.estimates.insert(fit.estimates.end(),beta.begin(),beta.end());
    for (std::size_t i = 1; i <= order.p; ++i)
        fit.names.push_back("ar"+std::to_string(i));
    for (std::size_t i = 1; i <= order.q; ++i)
        fit.names.push_back("ma"+std::to_string(i));
    for (std::size_t i = 1; i <= order.seasonal_p; ++i)
        fit.names.push_back("sar"+std::to_string(i));
    for (std::size_t i = 1; i <= order.seasonal_q; ++i)
        fit.names.push_back("sma"+std::to_string(i));
    if (fit_mean)
        fit.names.push_back("intercept");
    for (std::size_t j = 0; j < nx; ++j)
        fit.names.push_back(options.regressor_names.empty() ? "x"+std::to_string(j+1) : options.regressor_names[j]);

    const auto model = build(order,fit.coefficients);
    const auto f = filter(y,regression.offset(beta),model);
    fit.observations_used = used;
    fit.log_likelihood = profile_log_likelihood(f,used);
    fit.sigma2 = f.ssq/static_cast<double>(f.nu);
    fit.aic = -2*fit.log_likelihood+2*static_cast<double>(k+1);
    fit.residuals = f.residuals;
    fit.final_state = f.state;
    fit.final_state_covariance = f.covariance;

    // Observed information on the original scale by central differences of -loglik.
    const auto negative = [&](const std::vector<double>& theta) {
        ArimaCoefficients c;
        std::vector<double> b;
        unpack(order,theta,c,b);
        try {
            return -profile_log_likelihood(filter(y,regression.offset(b),build(order,c)),used);
        } catch (const std::invalid_argument&) {
            return std::numeric_limits<double>::quiet_NaN();
        }
    };
    Matrix hessian(k,k);
    std::vector<double> step(k,1e-4);
    for (std::size_t j = 0; j < m; ++j)
        step[arma+j] = 1e-4*regression.scale[j];
    const double f0 = negative(fit.estimates);
    for (std::size_t i = 0; i < k; ++i)
        for (std::size_t j = i; j < k; ++j) {
            auto t = fit.estimates;
            double value;
            if (i == j) {
                t[i] += step[i];
                const double up = negative(t);
                t[i] -= 2*step[i];
                const double down = negative(t);
                value = (up-2*f0+down)/(step[i]*step[i]);
            } else {
                double s[4];
                int idx = 0;
                for (const double a : {1.0,-1.0})
                    for (const double b : {1.0,-1.0}) {
                        t = fit.estimates;
                        t[i] += a*step[i];
                        t[j] += b*step[j];
                        s[idx++] = negative(t);
                    }
                value = (s[0]-s[1]-s[2]+s[3])/(4*step[i]*step[j]);
            }
            hessian(i,j) = hessian(j,i) = value;
        }
    fit.covariance = Matrix(k,k);
    fit.standard_errors.assign(k,std::numeric_limits<double>::quiet_NaN());
    if (k > 0) {
        bool finite = true;
        for (std::size_t i = 0; i < k; ++i)
            for (std::size_t j = 0; j < k; ++j)
                finite = finite && std::isfinite(hessian(i,j));
        if (finite) {
            const auto eig = linalg::symmetric_eigen(hessian);
            if (eig.eigenvalues.back() > 0) {
                for (std::size_t e = 0; e < k; ++e)
                    for (std::size_t i = 0; i < k; ++i)
                        for (std::size_t j = 0; j < k; ++j)
                            fit.covariance(i,j) += eig.eigenvectors(i,e)*eig.eigenvectors(j,e)/eig.eigenvalues[e];
                for (std::size_t i = 0; i < k; ++i)
                    fit.standard_errors[i] = std::sqrt(fit.covariance(i,i));
            }
        }
    }
    return fit;
}

ArimaForecast forecast_arima_exact(const ExactArimaFit& fit, std::size_t horizon,
                                   const linalg::DenseMatrix<double>& future_regressors) {
    const auto model = build(fit.order,fit.coefficients);
    if (fit.final_state.size() != model.rd || fit.final_state_covariance.rows() != model.rd)
        throw std::invalid_argument("forecast_arima_exact: fit has no filter state");
    const std::size_t nx = fit.regression.size();
    if (nx > 0 && (future_regressors.rows() != horizon || future_regressors.cols() != nx))
        throw std::invalid_argument("forecast_arima_exact: need horizon rows of future regressors, one column each");
    const std::size_t r = model.r, d = model.delta.size();
    auto a = fit.final_state;
    Matrix p = fit.final_state_covariance;
    ArimaForecast out;
    for (std::size_t h = 0; h < horizon; ++h) {
        a = transition(model,a);
        p = predict_covariance(model,p);
        double mean = a[0], var = p(0,0);
        for (std::size_t i = 0; i < d; ++i) {
            mean += model.delta[i]*a[r+i];
            var += 2*model.delta[i]*p(0,r+i);
            for (std::size_t j = 0; j < d; ++j)
                var += model.delta[i]*model.delta[j]*p(r+i,r+j);
        }
        mean += fit.mean;
        for (std::size_t j = 0; j < nx; ++j)
            mean += future_regressors(h,j)*fit.regression[j];
        out.mean.push_back(mean);
        out.standard_error.push_back(std::sqrt(fit.sigma2*var));
    }
    return out;
}

} // namespace datamunge::stats
