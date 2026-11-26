/// @file unit_root.cpp
/// @brief Dickey–Fuller regressions, lag selection, Phillips–Perron corrections and MacKinnon tables.

#include <datamunge/stats/unit_root.hpp>

#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/linalg/eigen.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>

namespace datamunge::stats {
/// @brief Least squares, tables and shared validation.
namespace {

/// @brief Least squares fit with coefficient standard errors.
struct OlsFit {
    std::vector<double> coefficients; ///< Estimates.
    std::vector<double> std_errors;   ///< Classical standard errors with divisor n - k.
    double rss{};                     ///< Residual sum of squares.
    std::vector<double> residuals;    ///< Residuals.
};

/// @brief Fits y on the columns of x.
/// @param x Rows of regressors.
/// @param y Response.
/// @return Coefficients, standard errors, RSS and residuals.
/// @throws std::invalid_argument When the design is singular or has no residual degrees of freedom.
OlsFit ols(const std::vector<std::vector<double>>& x, const std::vector<double>& y) {
    const std::size_t n = y.size(), k = x.empty() ? 0 : x[0].size();
    if (n <= k)
        throw std::invalid_argument("unit root test: too few observations for the regression");
    linalg::DenseMatrix<double> gram(k,k);
    std::vector<double> xty(k,0);
    for (std::size_t t = 0; t < n; ++t)
        for (std::size_t a = 0; a < k; ++a) {
            xty[a] += x[t][a]*y[t];
            for (std::size_t b = 0; b < k; ++b)
                gram(a,b) += x[t][a]*x[t][b];
        }
    const auto eig = linalg::symmetric_eigen(gram);
    if (!(eig.eigenvalues.back() > 1e-12*eig.eigenvalues.front()))
        throw std::invalid_argument("unit root test: singular regression (constant series?)");
    linalg::DenseMatrix<double> inverse(k,k);
    for (std::size_t e = 0; e < k; ++e)
        for (std::size_t a = 0; a < k; ++a)
            for (std::size_t b = 0; b < k; ++b)
                inverse(a,b) += eig.eigenvectors(a,e)*eig.eigenvectors(b,e)/eig.eigenvalues[e];
    OlsFit fit;
    fit.coefficients.assign(k,0);
    for (std::size_t a = 0; a < k; ++a)
        for (std::size_t b = 0; b < k; ++b)
            fit.coefficients[a] += inverse(a,b)*xty[b];
    fit.residuals.resize(n);
    for (std::size_t t = 0; t < n; ++t) {
        double f = 0;
        for (std::size_t a = 0; a < k; ++a)
            f += x[t][a]*fit.coefficients[a];
        fit.residuals[t] = y[t]-f;
        fit.rss += fit.residuals[t]*fit.residuals[t];
    }
    const double s2 = fit.rss/static_cast<double>(n-k);
    for (std::size_t a = 0; a < k; ++a)
        fit.std_errors.push_back(std::sqrt(s2*inverse(a,a)));
    return fit;
}

/// @brief Number of deterministic columns.
/// @param trend Trend option.
/// @return 0, 1 or 2.
std::size_t deterministic_columns(UnitRootTrend trend) {
    return trend == UnitRootTrend::none ? 0 : (trend == UnitRootTrend::constant ? 1 : 2);
}

/// @brief Appends deterministic columns for regression row t (trend counts from 1).
/// @param row Row to extend.
/// @param trend Trend option.
/// @param t Zero-based row index within the regression sample.
void add_deterministic(std::vector<double>& row, UnitRootTrend trend, std::size_t t) {
    if (trend != UnitRootTrend::none)
        row.push_back(1);
    if (trend == UnitRootTrend::linear)
        row.push_back(static_cast<double>(t+1));
}

/// @brief Validates the series.
/// @param y Series.
/// @throws std::invalid_argument For nonfinite values or fewer than four observations.
void check_series(const std::vector<double>& y) {
    if (y.size() < 4 || !std::ranges::all_of(y,[](double v) { return std::isfinite(v); }))
        throw std::invalid_argument("unit root test: need at least four finite observations");
}

/// @brief Builds the ADF regression with p lagged differences on observations from `first` on.
/// @param dy Differences \f$\Delta y_t=y_{t+1}-y_t\f$.
/// @param y Levels.
/// @param p Lagged differences.
/// @param first First index into dy used as the response (at least p).
/// @param trend Trend option.
/// @param x Set to the regressor rows: y_{t-1}, deterministic terms, lagged differences.
/// @param response Set to the responses.
void adf_design(const std::vector<double>& dy, const std::vector<double>& y, std::size_t p, std::size_t first,
                UnitRootTrend trend, std::vector<std::vector<double>>& x, std::vector<double>& response) {
    x.clear();
    response.clear();
    for (std::size_t t = first; t < dy.size(); ++t) {
        std::vector<double> row{y[t]};
        add_deterministic(row,trend,t-first);
        for (std::size_t i = 1; i <= p; ++i)
            row.push_back(dy[t-i]);
        x.push_back(std::move(row));
        response.push_back(dy[t]);
    }
}

} // namespace

double unit_root_p_value(double tau, UnitRootTrend trend) {
    // MacKinnon (1994) N = 1 coefficients, as tabulated (scaled) in statsmodels.
    struct Table { double max, min, star; double small[3]; double large[4]; };
    static constexpr Table tables[3] = {
        {std::numeric_limits<double>::infinity(),-19.04,-1.04,{.6344,1.2378,.032496},{.4797,.93557,-.06999,.033066}},
        {2.74,-18.83,-1.61,{2.1659,1.4412,.038269},{1.7339,.93202,-.12745,-.010368}},
        {.7,-16.18,-2.89,{3.2512,1.6047,.049588},{2.5261,.61654,-.37956,-.060285}},
    };
    const auto& c = tables[static_cast<int>(trend)];
    if (std::isnan(tau))
        return std::numeric_limits<double>::quiet_NaN();
    if (tau > c.max)
        return 1;
    if (tau < c.min)
        return 0;
    double z;
    if (tau <= c.star)
        z = c.small[0]+tau*(c.small[1]+tau*c.small[2]);
    else
        z = c.large[0]+tau*(c.large[1]+tau*(c.large[2]+tau*c.large[3]));
    return .5*std::erfc(-z/std::numbers::sqrt2);
}

UnitRootCriticalValues unit_root_critical_values(UnitRootTrend trend, std::size_t observations) {
    if (observations == 0)
        throw std::invalid_argument("unit_root_critical_values: need a positive sample size");
    // MacKinnon (2010), Table 2, N = 1: rows 1%, 5%, 10%; columns beta_inf, beta_1, beta_2, beta_3.
    static constexpr double table[3][3][4] = {
        {{-2.56574,-2.2358,-3.627,0},{-1.941,-.2686,-3.365,31.223},{-1.61682,.2656,-2.714,25.364}},
        {{-3.43035,-6.5393,-16.786,-79.433},{-2.86154,-2.8903,-4.234,-40.04},{-2.56677,-1.5384,-2.809,0}},
        {{-3.95877,-9.0531,-28.428,-134.155},{-3.41049,-4.3904,-9.036,-45.374},{-3.12705,-2.5856,-3.925,-22.38}},
    };
    const auto& rows = table[static_cast<int>(trend)];
    const double inv = 1/static_cast<double>(observations);
    const auto value = [&](const double* b) { return b[0]+inv*(b[1]+inv*(b[2]+inv*b[3])); };
    return {value(rows[0]),value(rows[1]),value(rows[2])};
}

UnitRootTest augmented_dickey_fuller(const std::vector<double>& y, const AdfOptions& options) {
    check_series(y);
    const std::size_t n = y.size(), k = deterministic_columns(options.trend);
    std::vector<double> dy(n-1);
    for (std::size_t t = 0; t+1 < n; ++t)
        dy[t] = y[t+1]-y[t];
    std::size_t max_lags;
    if (options.lags) {
        max_lags = *options.lags;
    } else {
        max_lags = static_cast<std::size_t>(std::ceil(12*std::pow(static_cast<double>(n)/100,.25)));
        const std::ptrdiff_t cap = static_cast<std::ptrdiff_t>(n/2)-static_cast<std::ptrdiff_t>(k)-1;
        max_lags = static_cast<std::size_t>(std::max<std::ptrdiff_t>(0,std::min<std::ptrdiff_t>(cap,
                                                                     static_cast<std::ptrdiff_t>(max_lags))));
    }
    if (max_lags+k+3 > n)
        throw std::invalid_argument("augmented_dickey_fuller: series too short for the lags");
    std::vector<std::vector<double>> x;
    std::vector<double> response;
    std::size_t p = max_lags;
    if (options.selection != LagSelection::fixed) {
        double best = std::numeric_limits<double>::infinity();
        for (std::size_t lag = 0; lag <= max_lags; ++lag) {
            adf_design(dy,y,lag,max_lags,options.trend,x,response);
            const auto fit = ols(x,response);
            const double m = static_cast<double>(response.size()), params = static_cast<double>(x[0].size());
            const double llf = -.5*m*(std::log(2*std::numbers::pi)+std::log(fit.rss/m)+1);
            const double ic = -2*llf+(options.selection == LagSelection::aic ? 2*params : params*std::log(m));
            if (ic < best) {
                best = ic;
                p = lag;
            }
        }
    }
    adf_design(dy,y,p,p,options.trend,x,response);
    const auto fit = ols(x,response);
    UnitRootTest r;
    r.coefficient = fit.coefficients[0];
    r.statistic = fit.coefficients[0]/fit.std_errors[0];
    r.p_value = unit_root_p_value(r.statistic,options.trend);
    r.z_alpha = std::numeric_limits<double>::quiet_NaN();
    r.lags = p;
    r.observations = response.size();
    r.critical = unit_root_critical_values(options.trend,r.observations);
    return r;
}

UnitRootTest phillips_perron(const std::vector<double>& y, const PhillipsPerronOptions& options) {
    check_series(y);
    const std::size_t n = y.size();
    const std::size_t lags = options.lags ? *options.lags
                                          : static_cast<std::size_t>(4*std::pow(static_cast<double>(n)/100,.25));
    std::vector<std::vector<double>> x;
    std::vector<double> response;
    for (std::size_t t = 1; t < n; ++t) {
        std::vector<double> row{y[t-1]};
        add_deterministic(row,options.trend,t-1);
        x.push_back(std::move(row));
        response.push_back(y[t]);
    }
    if (lags >= response.size())
        throw std::invalid_argument("phillips_perron: more Newey-West lags than observations");
    const auto fit = ols(x,response);
    const double m = static_cast<double>(response.size()), k = static_cast<double>(x[0].size());
    const auto& u = fit.residuals;
    const double gamma0 = fit.rss/m;
    double lambda2 = gamma0;
    for (std::size_t j = 1; j <= lags; ++j) {
        double g = 0;
        for (std::size_t t = j; t < u.size(); ++t)
            g += u[t]*u[t-j];
        lambda2 += 2*(1-static_cast<double>(j)/static_cast<double>(lags+1))*g/m;
    }
    const double s2 = fit.rss/(m-k), s = std::sqrt(s2);
    const double sigma = fit.std_errors[0], rho = fit.coefficients[0];
    const double lambda = std::sqrt(lambda2);
    UnitRootTest r;
    r.coefficient = rho-1;
    r.statistic = std::sqrt(gamma0/lambda2)*((rho-1)/sigma)-.5*((lambda2-gamma0)/lambda)*(m*sigma/s);
    r.z_alpha = m*(rho-1)-.5*(m*m*sigma*sigma/s2)*(lambda2-gamma0);
    r.p_value = unit_root_p_value(r.statistic,options.trend);
    r.lags = lags;
    r.observations = response.size();
    r.critical = unit_root_critical_values(options.trend,r.observations);
    return r;
}

KpssTest kpss_test(const std::vector<double>& y, const KpssOptions& options) {
    check_series(y);
    if (options.trend == UnitRootTrend::none)
        throw std::invalid_argument("kpss_test: trend must be constant or linear");
    const std::size_t n = y.size();
    std::vector<std::vector<double>> x;
    for (std::size_t t = 0; t < n; ++t) {
        std::vector<double> row;
        add_deterministic(row,options.trend,t);
        x.push_back(std::move(row));
    }
    const auto fit = ols(x,y);
    const auto& e = fit.residuals;
    const double nn = static_cast<double>(n);
    if (!(fit.rss > 0))
        throw std::invalid_argument("kpss_test: residuals are identically zero");
    const auto product = [&](std::size_t j) {
        double s = 0;
        for (std::size_t t = j; t < n; ++t)
            s += e[t]*e[t-j];
        return s;
    };
    std::size_t lags;
    if (options.lags) {
        lags = *options.lags;
    } else {
        // Hobijn, Franses and Ooms (1998) automatic bandwidth, as statsmodels.
        const auto covlags = static_cast<std::size_t>(std::pow(nn,2.0/9));
        double s0 = fit.rss/nn, s1 = 0;
        for (std::size_t i = 1; i <= covlags && i < n; ++i) {
            const double p = product(i)/(nn/2);
            s0 += p;
            s1 += static_cast<double>(i)*p;
        }
        const double ratio = s1/s0;
        lags = static_cast<std::size_t>(std::max(0.0,1.1447*std::pow(ratio*ratio,1.0/3)*std::pow(nn,1.0/3)));
    }
    lags = std::min(lags,n-1);
    double lambda2 = fit.rss;
    for (std::size_t j = 1; j <= lags; ++j)
        lambda2 += 2*product(j)*(1-static_cast<double>(j)/static_cast<double>(lags+1));
    lambda2 /= nn;
    double partial = 0, eta = 0;
    for (const double v : e) {
        partial += v;
        eta += partial*partial;
    }
    KpssTest r;
    r.statistic = eta/(nn*nn)/lambda2;
    r.lags = lags;
    r.critical = options.trend == UnitRootTrend::constant ? KpssCriticalValues{.347,.463,.574,.739}
                                                          : KpssCriticalValues{.119,.146,.176,.216};
    const double crit[4] = {r.critical.ten_percent,r.critical.five_percent,r.critical.two_and_a_half_percent,
                            r.critical.one_percent};
    const double level[4] = {.10,.05,.025,.01};
    if (r.statistic <= crit[0]) {
        r.p_value = .10;
        r.p_value_truncated = r.statistic < crit[0];
    } else if (r.statistic >= crit[3]) {
        r.p_value = .01;
        r.p_value_truncated = r.statistic > crit[3];
    } else {
        for (int i = 0; i < 3; ++i)
            if (r.statistic <= crit[i+1]) {
                const double w = (r.statistic-crit[i])/(crit[i+1]-crit[i]);
                r.p_value = level[i]+w*(level[i+1]-level[i]);
                break;
            }
    }
    return r;
}

} // namespace datamunge::stats
