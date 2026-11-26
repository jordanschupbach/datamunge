/// @file var.cpp
/// @brief Least-squares VAR estimation, information criteria, companion eigenvalues and MSE forecasts.

#include <datamunge/stats/var.hpp>

#include <datamunge/linalg/eigen.hpp>
#include <datamunge/nn/linalg.hpp>
#include <datamunge/random/distributions.hpp>

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace datamunge::stats {
/// @brief Design construction and symmetric-matrix helpers.
namespace {

/// @brief Matrix alias used throughout.
using Matrix = linalg::DenseMatrix<double>;

/// @brief Number of deterministic columns.
/// @param trend Trend option.
/// @return 0, 1 or 2.
std::size_t deterministic_count(VarTrend trend) {
    return trend == VarTrend::none ? 0 : (trend == VarTrend::constant ? 1 : 2);
}

/// @brief Inverse and log determinant of a symmetric positive definite matrix.
/// @param a Matrix.
/// @param logdet Set to \f$\ln|a|\f$.
/// @return \f$a^{-1}\f$.
/// @throws std::invalid_argument When a is numerically singular.
Matrix spd_inverse(const Matrix& a, double& logdet) {
    const std::size_t k = a.rows();
    Matrix inverse(k,k);
    logdet = 0;
    if (k == 0)
        return inverse;
    const auto eig = linalg::symmetric_eigen(a);
    const double top = *std::ranges::max_element(eig.eigenvalues);
    for (const double v : eig.eigenvalues)
        if (!(v > 1e-12*top))
            throw std::invalid_argument("VAR: singular design or residual covariance");
    for (std::size_t e = 0; e < k; ++e) {
        logdet += std::log(eig.eigenvalues[e]);
        for (std::size_t i = 0; i < k; ++i)
            for (std::size_t j = 0; j < k; ++j)
                inverse(i,j) += eig.eigenvectors(i,e)*eig.eigenvectors(j,e)/eig.eigenvalues[e];
    }
    return inverse;
}

/// @brief Fits a VAR(p) to rows offset+p .. n-1, using rows from `offset` as presample.
/// @param y Observations.
/// @param p Lag order.
/// @param offset Leading rows skipped (for a common sample in order selection).
/// @param trend Deterministic terms.
/// @return Fitted model (stability not yet computed).
VarModel estimate(const Matrix& y, std::size_t p, std::size_t offset, VarTrend trend) {
    const std::size_t n = y.rows(), K = y.cols(), d = deterministic_count(trend);
    if (K == 0)
        throw std::invalid_argument("fit_var: need at least one variable");
    if (n < offset+p+1)
        throw std::invalid_argument("fit_var: too few observations for the lag order");
    const std::size_t T = n-offset-p, m = d+K*p;
    if (T <= m)
        throw std::invalid_argument("fit_var: too few observations for the number of coefficients");
    Matrix z(T,m);
    for (std::size_t t = 0; t < T; ++t) {
        std::size_t c = 0;
        if (d >= 1)
            z(t,c++) = 1;
        if (d == 2)
            z(t,c++) = static_cast<double>(offset+p+t+1);
        for (std::size_t l = 1; l <= p; ++l)
            for (std::size_t j = 0; j < K; ++j)
                z(t,c++) = y(offset+p+t-l,j);
    }
    Matrix gram(m,m), zty(m,K);
    for (std::size_t t = 0; t < T; ++t)
        for (std::size_t a = 0; a < m; ++a) {
            for (std::size_t k = 0; k < K; ++k)
                zty(a,k) += z(t,a)*y(offset+p+t,k);
            for (std::size_t b = 0; b < m; ++b)
                gram(a,b) += z(t,a)*z(t,b);
        }
    double unused = 0;
    const Matrix gram_inverse = spd_inverse(gram,unused);
    VarModel v;
    v.lags = p;
    v.variables = K;
    v.observations = T;
    v.trend = trend;
    v.parameters = Matrix(m,K);
    for (std::size_t a = 0; a < m; ++a)
        for (std::size_t k = 0; k < K; ++k)
            for (std::size_t b = 0; b < m; ++b)
                v.parameters(a,k) += gram_inverse(a,b)*zty(b,k);
    v.residuals = Matrix(T,K);
    v.sigma_mle = Matrix(K,K);
    for (std::size_t t = 0; t < T; ++t) {
        for (std::size_t k = 0; k < K; ++k) {
            double f = 0;
            for (std::size_t a = 0; a < m; ++a)
                f += z(t,a)*v.parameters(a,k);
            v.residuals(t,k) = y(offset+p+t,k)-f;
        }
        for (std::size_t i = 0; i < K; ++i)
            for (std::size_t j = 0; j < K; ++j)
                v.sigma_mle(i,j) += v.residuals(t,i)*v.residuals(t,j);
    }
    const double Td = static_cast<double>(T), df = static_cast<double>(T-m);
    v.sigma = Matrix(K,K);
    for (std::size_t i = 0; i < K; ++i)
        for (std::size_t j = 0; j < K; ++j) {
            v.sigma(i,j) = v.sigma_mle(i,j)/df;
            v.sigma_mle(i,j) /= Td;
        }
    v.design_inverse = gram_inverse;
    v.standard_errors = Matrix(m,K);
    for (std::size_t a = 0; a < m; ++a)
        for (std::size_t k = 0; k < K; ++k)
            v.standard_errors(a,k) = std::sqrt(gram_inverse(a,a)*v.sigma(k,k));
    for (std::size_t l = 0; l < p; ++l) {
        Matrix a(K,K);
        for (std::size_t i = 0; i < K; ++i)
            for (std::size_t j = 0; j < K; ++j)
                a(i,j) = v.parameters(d+l*K+j,i);
        v.lag_coefficients.push_back(std::move(a));
    }
    double logdet = 0;
    (void)spd_inverse(v.sigma_mle,logdet);
    const double Kd = static_cast<double>(K);
    v.log_likelihood = -.5*Td*Kd*std::log(2*std::numbers::pi)-.5*Td*logdet-.5*Td*Kd;
    const double free = static_cast<double>(p)*Kd*Kd+Kd*static_cast<double>(d);
    v.aic = logdet+2*free/Td;
    v.bic = logdet+std::log(Td)*free/Td;
    v.hqic = logdet+2*std::log(std::log(Td))*free/Td;
    v.fpe = std::pow((Td+static_cast<double>(m))/df,Kd)*std::exp(logdet);
    v.history = Matrix(p,K);
    for (std::size_t l = 0; l < p; ++l)
        for (std::size_t k = 0; k < K; ++k)
            v.history(l,k) = y(n-p+l,k);
    return v;
}

/// @brief Validates the data matrix.
/// @param y Observations.
/// @throws std::invalid_argument For nonfinite values.
void check_data(const Matrix& y) {
    for (std::size_t i = 0; i < y.rows(); ++i)
        for (std::size_t j = 0; j < y.cols(); ++j)
            if (!std::isfinite(y(i,j)))
                throw std::invalid_argument("fit_var: observations must be finite");
}

} // namespace

VarModel fit_var(const linalg::DenseMatrix<double>& y, const VarOptions& options) {
    check_data(y);
    auto v = estimate(y,options.lags,0,options.trend);
    const std::size_t K = v.variables, p = v.lags, kp = K*p;
    if (p == 0) {
        v.stable = true;
        return v;
    }
    std::vector<double> companion(kp*kp,0.0);
    for (std::size_t l = 0; l < p; ++l)
        for (std::size_t i = 0; i < K; ++i)
            for (std::size_t j = 0; j < K; ++j)
                companion[i*kp+l*K+j] = v.lag_coefficients[l](i,j);
    for (std::size_t i = K; i < kp; ++i)
        companion[i*kp+i-K] = 1;
    const auto n = static_cast<std::int64_t>(kp);
    const auto values = nn::eigvals(nn::from_vector<double>(companion,{n,n}));
    const auto re = values.real.to_vector<double>(), im = values.imag.to_vector<double>();
    for (std::size_t i = 0; i < re.size(); ++i)
        v.companion_moduli.push_back(std::hypot(re[i],im[i]));
    std::ranges::sort(v.companion_moduli,std::greater<>());
    v.stable = v.companion_moduli.front() < 1;
    return v;
}

VarOrderSelection select_var_order(const linalg::DenseMatrix<double>& y, std::size_t max_lags, VarTrend trend) {
    check_data(y);
    VarOrderSelection s;
    for (std::size_t p = 0; p <= max_lags; ++p) {
        const auto v = estimate(y,p,max_lags-p,trend);
        s.aic.push_back(v.aic);
        s.bic.push_back(v.bic);
        s.hqic.push_back(v.hqic);
        s.fpe.push_back(v.fpe);
    }
    const auto best = [](const std::vector<double>& c) {
        return static_cast<std::size_t>(std::ranges::min_element(c)-c.begin());
    };
    s.aic_order = best(s.aic);
    s.bic_order = best(s.bic);
    s.hqic_order = best(s.hqic);
    s.fpe_order = best(s.fpe);
    return s;
}

GrangerTest granger_causality(const VarModel& model, const std::vector<std::size_t>& caused,
                              const std::vector<std::size_t>& causing, GrangerStatistic kind) {
    const std::size_t K = model.variables, p = model.lags, d = deterministic_count(model.trend);
    if (p == 0)
        throw std::invalid_argument("granger_causality: the VAR has no lags");
    if (caused.empty() || causing.empty())
        throw std::invalid_argument("granger_causality: need caused and causing series");
    std::vector<int> role(K,0);
    for (const auto i : caused) {
        if (i >= K || role[i] != 0)
            throw std::invalid_argument("granger_causality: invalid or repeated caused index");
        role[i] = 1;
    }
    for (const auto j : causing) {
        if (j >= K || role[j] != 0)
            throw std::invalid_argument("granger_causality: invalid, repeated or overlapping causing index");
        role[j] = 2;
    }
    // Restricted coefficients as (regressor row, equation) pairs.
    std::vector<std::pair<std::size_t,std::size_t>> cells;
    for (const auto i : caused)
        for (std::size_t l = 0; l < p; ++l)
            for (const auto j : causing)
                cells.emplace_back(d+l*K+j,i);
    const std::size_t r = cells.size();
    Matrix v(r,r);
    std::vector<double> b(r);
    for (std::size_t a = 0; a < r; ++a) {
        b[a] = model.parameters(cells[a].first,cells[a].second);
        for (std::size_t c = 0; c < r; ++c)
            v(a,c) = model.design_inverse(cells[a].first,cells[c].first)*model.sigma(cells[a].second,cells[c].second);
    }
    double unused = 0;
    const Matrix inverse = spd_inverse(v,unused);
    double wald = 0;
    for (std::size_t a = 0; a < r; ++a)
        for (std::size_t c = 0; c < r; ++c)
            wald += b[a]*inverse(a,c)*b[c];
    GrangerTest g;
    g.restrictions = r;
    const double rd = static_cast<double>(r);
    if (kind == GrangerStatistic::wald) {
        g.statistic = wald;
        g.p_value = random::detail::gamma_probability(.5L*static_cast<long double>(wald),.5*rd,false);
    } else {
        const double df2 = static_cast<double>(K*(model.observations-d-K*p));
        g.statistic = wald/rd;
        g.denominator_df = df2;
        g.p_value = random::detail::regularized_incomplete_beta(.5*df2,.5*rd,df2/(df2+rd*g.statistic));
    }
    return g;
}

VarForecast forecast_var(const VarModel& model, std::size_t horizon) {
    if (horizon == 0)
        throw std::invalid_argument("forecast_var: horizon must be positive");
    const std::size_t K = model.variables, p = model.lags, d = deterministic_count(model.trend);
    VarForecast f;
    f.mean = Matrix(horizon,K);
    std::vector<std::vector<double>> path;
    for (std::size_t l = 0; l < p; ++l) {
        std::vector<double> row(K);
        for (std::size_t k = 0; k < K; ++k)
            row[k] = model.history(l,k);
        path.push_back(std::move(row));
    }
    for (std::size_t h = 0; h < horizon; ++h) {
        std::vector<double> next(K,0.0);
        for (std::size_t k = 0; k < K; ++k) {
            if (d >= 1)
                next[k] += model.parameters(0,k);
            if (d == 2)
                next[k] += model.parameters(1,k)*static_cast<double>(model.observations+p+h+1);
            for (std::size_t l = 1; l <= p; ++l)
                for (std::size_t j = 0; j < K; ++j)
                    next[k] += model.lag_coefficients[l-1](k,j)*path[path.size()-l][j];
            f.mean(h,k) = next[k];
        }
        path.push_back(std::move(next));
    }
    // Moving-average coefficients Phi_0 = I, Phi_i = sum_{l=1}^{min(i,p)} Phi_{i-l} A_l.
    std::vector<Matrix> phi;
    Matrix identity(K,K);
    for (std::size_t i = 0; i < K; ++i)
        identity(i,i) = 1;
    phi.push_back(identity);
    Matrix mse(K,K);
    for (std::size_t h = 0; h < horizon; ++h) {
        if (h > 0) {
            Matrix next(K,K);
            for (std::size_t l = 1; l <= std::min(h,p); ++l)
                for (std::size_t i = 0; i < K; ++i)
                    for (std::size_t j = 0; j < K; ++j)
                        for (std::size_t r = 0; r < K; ++r)
                            next(i,j) += phi[h-l](i,r)*model.lag_coefficients[l-1](r,j);
            phi.push_back(std::move(next));
        }
        const Matrix& q = phi[h];
        for (std::size_t i = 0; i < K; ++i)
            for (std::size_t j = 0; j < K; ++j) {
                double s = 0;
                for (std::size_t a = 0; a < K; ++a)
                    for (std::size_t b = 0; b < K; ++b)
                        s += q(i,a)*model.sigma(a,b)*q(j,b);
                mse(i,j) += s;
            }
        f.covariance.push_back(mse);
    }
    return f;
}

} // namespace datamunge::stats
