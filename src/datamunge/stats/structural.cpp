/// @file structural.cpp
/// @brief State-space form of structural models, R's Kalman filter and smoother, and the variance search.

#include <datamunge/stats/structural.hpp>

#include <datamunge/optim/lbfgs.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>

namespace datamunge::stats {
/// @brief System matrices, filter passes and the optimizer objective.
namespace {

/// @brief Matrix alias used throughout.
using Matrix = linalg::DenseMatrix<double>;

/// @brief Time-invariant state-space system \f$y_t=Z^\top\alpha_t+\varepsilon_t,\ \alpha_{t+1}=T\alpha_t+\eta_t\f$.
struct System {
    std::vector<double> z; ///< Observation vector Z.
    Matrix t;              ///< Transition matrix T.
    Matrix v;              ///< State disturbance covariance V.
    double h{};            ///< Observation variance.
    std::vector<double> a0; ///< Initial state.
    Matrix p0;             ///< Initial covariance.
};

/// @brief Number of free variances (level, slope, seasonal) for a model.
/// @param model Model type.
/// @return 1, 2 or 3.
std::size_t state_variances(StructuralModel model) {
    return model == StructuralModel::local_level ? 1 : (model == StructuralModel::local_linear_trend ? 2 : 3);
}

/// @brief Builds the system of R's makeLevel / makeTrend / makeBSM with the given variances.
/// @param model Model type.
/// @param period Seasonal period.
/// @param first First observation.
/// @param prior Value of every entry of \f$P_0\f$.
/// @param variances State variances followed by the observation variance.
/// @return System.
System build(StructuralModel model, std::size_t period, double first, double prior, const std::vector<double>& variances) {
    const std::size_t m = model == StructuralModel::local_level ? 1
                          : model == StructuralModel::local_linear_trend ? 2 : period+1;
    System s;
    s.z.assign(m,0.0);
    s.z[0] = 1;
    s.t = Matrix(m,m);
    s.t(0,0) = 1;
    if (m >= 2) {
        s.t(0,1) = 1;
        s.t(1,1) = 1;
    }
    if (model == StructuralModel::basic_seasonal) {
        s.z[2] = 1;
        for (std::size_t j = 2; j < m; ++j)
            s.t(2,j) = -1;
        for (std::size_t j = 2; j+1 < m; ++j)
            s.t(j+1,j) = 1;
    }
    s.v = Matrix(m,m);
    const std::size_t k = state_variances(model);
    for (std::size_t i = 0; i < k; ++i)
        s.v(i,i) = variances[i];
    s.h = variances[k];
    s.a0.assign(m,0.0);
    s.a0[0] = first;
    s.p0 = Matrix(m,m,prior);
    return s;
}

/// @brief Output of one filter pass.
struct Pass {
    double ssq{};     ///< \f$\sum v_t^2/F_t\f$.
    double sumlog{};  ///< \f$\sum\ln F_t\f$.
    std::size_t used{}; ///< Observed values.
    std::vector<double> std_resid;      ///< \f$v_t/\sqrt{F_t}\f$.
    std::vector<std::vector<double>> filtered; ///< \f$a_{t|t}\f$.
    std::vector<std::vector<double>> predicted; ///< \f$a_{t|t-1}\f$.
    std::vector<Matrix> predicted_cov;  ///< \f$P_{t|t-1}\f$.
    std::vector<std::vector<double>> m; ///< \f$M_t=P_{t|t-1}Z\f$ (zero when missing).
    std::vector<double> gain;           ///< \f$F_t\f$ (NaN when missing).
    std::vector<double> innovation;     ///< \f$v_t\f$ (NaN when missing).
    std::vector<double> a;              ///< Final filtered state.
    Matrix p;                           ///< Final filtered covariance.
};

/// @brief R's KalmanLike/KalmanRun/KalmanSmooth forward recursion (prediction at every step).
/// @param y Series (NaN missing).
/// @param s System.
/// @param keep Whether to store per-step quantities.
/// @return Pass results.
Pass filter(const std::vector<double>& y, const System& s, bool keep) {
    const std::size_t n = y.size(), m = s.z.size();
    Pass out;
    std::vector<double> a = s.a0, anew(m), M(m);
    Matrix P = s.p0, Pnew(m,m), mm(m,m);
    for (std::size_t l = 0; l < n; ++l) {
        for (std::size_t i = 0; i < m; ++i) {
            double tmp = 0;
            for (std::size_t k = 0; k < m; ++k)
                tmp += s.t(i,k)*a[k];
            anew[i] = tmp;
        }
        for (std::size_t i = 0; i < m; ++i)
            for (std::size_t j = 0; j < m; ++j) {
                double tmp = 0;
                for (std::size_t k = 0; k < m; ++k)
                    tmp += s.t(i,k)*P(k,j);
                mm(i,j) = tmp;
            }
        for (std::size_t i = 0; i < m; ++i)
            for (std::size_t j = 0; j < m; ++j) {
                double tmp = s.v(i,j);
                for (std::size_t k = 0; k < m; ++k)
                    tmp += mm(i,k)*s.t(j,k);
                Pnew(i,j) = tmp;
            }
        if (keep) {
            out.predicted.push_back(anew);
            out.predicted_cov.push_back(Pnew);
        }
        if (!std::isnan(y[l])) {
            ++out.used;
            double resid = y[l];
            for (std::size_t i = 0; i < m; ++i)
                resid -= s.z[i]*anew[i];
            double gain = s.h;
            for (std::size_t i = 0; i < m; ++i) {
                double tmp = 0;
                for (std::size_t k = 0; k < m; ++k)
                    tmp += Pnew(i,k)*s.z[k];
                M[i] = tmp;
                gain += s.z[i]*M[i];
            }
            out.ssq += resid*resid/gain;
            out.sumlog += std::log(gain);
            for (std::size_t i = 0; i < m; ++i)
                a[i] = anew[i]+M[i]*resid/gain;
            for (std::size_t i = 0; i < m; ++i)
                for (std::size_t j = 0; j < m; ++j)
                    P(i,j) = Pnew(i,j)-M[i]*M[j]/gain;
            if (keep) {
                out.std_resid.push_back(resid/std::sqrt(gain));
                out.m.push_back(M);
                out.gain.push_back(gain);
                out.innovation.push_back(resid);
            }
        } else {
            a = anew;
            P = Pnew;
            if (keep) {
                out.std_resid.push_back(std::numeric_limits<double>::quiet_NaN());
                out.m.emplace_back(m,0.0);
                out.gain.push_back(std::numeric_limits<double>::quiet_NaN());
                out.innovation.push_back(std::numeric_limits<double>::quiet_NaN());
            }
        }
        if (keep)
            out.filtered.push_back(a);
    }
    out.a = a;
    out.p = P;
    return out;
}

/// @brief R's KalmanSmooth backward pass (state smoother \f$\hat\alpha_t=a_{t|t-1}+P_{t|t-1}r_{t-1}\f$).
/// @param pass Forward pass with stored quantities.
/// @param s System.
/// @return Smoothed states.
std::vector<std::vector<double>> smooth(const Pass& pass, const System& s) {
    const std::size_t n = pass.gain.size(), m = s.z.size();
    std::vector<std::vector<double>> at = pass.predicted, r(n,std::vector<double>(m,0.0));
    Matrix mm(m,m), L(m,m);
    for (std::size_t l = n; l-- > 0;) {
        double gn = 0;
        if (!std::isnan(pass.gain[l])) {
            gn = 1/pass.gain[l];
            for (std::size_t i = 0; i < m; ++i)
                r[l][i] = s.z[i]*pass.innovation[l]*gn;
        }
        if (l+1 < n) {
            for (std::size_t i = 0; i < m; ++i)
                for (std::size_t j = 0; j < m; ++j)
                    mm(i,j) = (i == j ? 1.0 : 0.0)-pass.m[l][i]*s.z[j]*gn;
            for (std::size_t i = 0; i < m; ++i)
                for (std::size_t j = 0; j < m; ++j) {
                    double tmp = 0;
                    for (std::size_t k = 0; k < m; ++k)
                        tmp += s.t(i,k)*mm(k,j);
                    L(i,j) = tmp;
                }
            for (std::size_t i = 0; i < m; ++i) {
                double tmp = 0;
                for (std::size_t j = 0; j < m; ++j)
                    tmp += L(j,i)*r[l+1][j];
                r[l][i] += tmp;
            }
        }
        for (std::size_t i = 0; i < m; ++i) {
            double tmp = 0;
            for (std::size_t j = 0; j < m; ++j)
                tmp += pass.predicted_cov[l](i,j)*r[l][j];
            at[l][i] += tmp;
        }
    }
    return at;
}

/// @brief Objective in R's scale: \f$\tfrac12(\sum v^2/F+\sum\ln F)/n_{\text{obs}}\f$ over \f$\sigma_i^2=x_i^2\,\operatorname{var}(y)/100\f$.
class Objective final : public optim::DifferentiableFunction {
public:
    /// @brief Stores the data and model.
    /// @param y Series.
    /// @param model Model.
    /// @param period Period.
    /// @param scale var(y)/100.
    Objective(const std::vector<double>& y, StructuralModel model, std::size_t period, double scale)
        : y_(y), model_(model), period_(period), scale_(scale) {}

    /// @brief Variances from coordinates.
    /// @param x Coordinates.
    /// @return Variances.
    [[nodiscard]] std::vector<double> variances(const std::vector<double>& x) const {
        std::vector<double> v;
        for (const double c : x)
            v.push_back(c*c*scale_);
        return v;
    }

    /// @brief Objective value.
    /// @param x Coordinates.
    /// @return Scaled negative log likelihood (1000 when all variances vanish, as R).
    double evaluate(const std::vector<double>& x) override {
        const auto v = variances(x);
        if (std::ranges::all_of(v,[](double d) { return d == 0; }))
            return 1000;
        const auto pass = filter(y_,build(model_,period_,y_[0],1e6*scale_,v),false);
        const double value = .5*(pass.ssq+pass.sumlog)/static_cast<double>(pass.used);
        return std::isfinite(value) ? value : 1e10;
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
    const std::vector<double>& y_; ///< Series.
    StructuralModel model_;        ///< Model.
    std::size_t period_;           ///< Period.
    double scale_;                 ///< var(y)/100.
};

/// @brief Component columns of the state: level, slope, current seasonal (as present).
/// @param model Model.
/// @return Column count.
std::size_t components(StructuralModel model) {
    return state_variances(model);
}

} // namespace

StructuralFit fit_structural(const std::vector<double>& y, const StructuralOptions& options) {
    if (y.empty() || std::isnan(y[0]))
        throw std::invalid_argument("fit_structural: the first value must be observed");
    std::size_t observed = 0;
    double mean = 0;
    for (const double v : y) {
        if (std::isinf(v))
            throw std::invalid_argument("fit_structural: values must be finite or NaN");
        if (!std::isnan(v)) {
            ++observed;
            mean += v;
        }
    }
    if (observed < 3)
        throw std::invalid_argument("fit_structural: need at least three observed values");
    mean /= static_cast<double>(observed);
    double var = 0;
    for (const double v : y)
        if (!std::isnan(v))
            var += (v-mean)*(v-mean);
    var /= static_cast<double>(observed-1);
    if (!(var > 0))
        throw std::invalid_argument("fit_structural: series is constant");
    const std::size_t period = options.model == StructuralModel::basic_seasonal ? options.period : 0;
    if (options.model == StructuralModel::basic_seasonal && period < 2)
        throw std::invalid_argument("fit_structural: the basic structural model needs period >= 2");
    const double scale = var/100;
    const std::size_t k = state_variances(options.model)+1;
    Objective objective(y,options.model,period,scale);
    std::vector<double> x(k,1.0);
    optim::LBFGSOptions controls;
    controls.max_iterations = options.max_iterations;
    controls.tolerance = 1e-10;
    (void)optim::LBFGS(controls).optimize(objective,x);
    double largest = 0;
    for (const double g : objective.gradient(x))
        largest = std::max(largest,std::abs(g));

    StructuralFit fit;
    fit.model = options.model;
    fit.period = period;
    fit.converged = largest < 1e-5;
    const auto v = objective.variances(x);
    fit.level_variance = v[0];
    if (k >= 3)
        fit.slope_variance = v[1];
    if (k >= 4)
        fit.seasonal_variance = v[2];
    fit.irregular_variance = v[k-1];
    const auto system = build(options.model,period,y[0],1e6*scale,v);
    const auto pass = filter(y,system,true);
    fit.log_likelihood = -.5*(pass.ssq+pass.sumlog)-.5*static_cast<double>(observed)*std::log(2*std::numbers::pi);
    fit.standardized_residuals = pass.std_resid;
    const auto smoothed = smooth(pass,system);
    const std::size_t c = components(options.model), n = y.size();
    fit.filtered = Matrix(n,c);
    fit.smoothed = Matrix(n,c);
    for (std::size_t t = 0; t < n; ++t)
        for (std::size_t j = 0; j < c; ++j) {
            fit.filtered(t,j) = pass.filtered[t][j];
            fit.smoothed(t,j) = smoothed[t][j];
        }
    fit.final_state = pass.a;
    fit.final_covariance = pass.p;
    return fit;
}

StructuralForecast forecast_structural(const StructuralFit& fit, std::size_t horizon) {
    if (horizon == 0)
        throw std::invalid_argument("forecast_structural: horizon must be positive");
    std::vector<double> variances;
    variances.push_back(fit.level_variance);
    if (fit.model != StructuralModel::local_level)
        variances.push_back(fit.slope_variance);
    if (fit.model == StructuralModel::basic_seasonal)
        variances.push_back(fit.seasonal_variance);
    variances.push_back(fit.irregular_variance);
    const auto s = build(fit.model,fit.period,0,0,variances);
    const std::size_t m = s.z.size();
    std::vector<double> a = fit.final_state, anew(m);
    Matrix P = fit.final_covariance, mm(m,m), Pnew(m,m);
    StructuralForecast f;
    for (std::size_t h = 0; h < horizon; ++h) {
        double fc = 0;
        for (std::size_t i = 0; i < m; ++i) {
            double tmp = 0;
            for (std::size_t k = 0; k < m; ++k)
                tmp += s.t(i,k)*a[k];
            anew[i] = tmp;
            fc += tmp*s.z[i];
        }
        a = anew;
        for (std::size_t i = 0; i < m; ++i)
            for (std::size_t j = 0; j < m; ++j) {
                double tmp = 0;
                for (std::size_t k = 0; k < m; ++k)
                    tmp += s.t(i,k)*P(k,j);
                mm(i,j) = tmp;
            }
        double var = s.h;
        for (std::size_t i = 0; i < m; ++i)
            for (std::size_t j = 0; j < m; ++j) {
                double tmp = s.v(i,j);
                for (std::size_t k = 0; k < m; ++k)
                    tmp += mm(i,k)*s.t(j,k);
                Pnew(i,j) = tmp;
            }
        P = Pnew;
        for (std::size_t i = 0; i < m; ++i)
            for (std::size_t j = 0; j < m; ++j)
                var += s.z[i]*s.z[j]*P(i,j);
        f.mean.push_back(fc);
        f.standard_error.push_back(std::sqrt(var));
    }
    return f;
}

} // namespace datamunge::stats
