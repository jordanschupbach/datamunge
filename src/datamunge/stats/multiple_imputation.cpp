/// @file multiple_imputation.cpp
/// @brief Chained-equation imputation with mice-style posterior draws, and Rubin/Barnard–Rubin pooling.

#include <datamunge/stats/multiple_imputation.hpp>

#include <datamunge/linalg/cholesky.hpp>
#include <datamunge/random/distributions.hpp>
#include <datamunge/random/variates.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace datamunge::stats {

/// @brief Internal helpers for multiple imputation.
namespace {

using Matrix = linalg::DenseMatrix<double>; ///< Dense matrix of doubles.

/// @brief Posterior draw of a linear regression, as mice's `.norm.draw`.
struct NormDraw {
    std::vector<double> coef;  ///< Least-squares coefficients.
    std::vector<double> beta;  ///< Drawn coefficients.
    double sigma{};            ///< Drawn residual standard deviation.
};

/// @brief Inverts a symmetric positive-definite matrix, adding a relative ridge if needed.
/// @param a Matrix (modified in place when the ridge is added).
/// @param ridge Relative ridge.
/// @return The inverse.
/// @throws std::runtime_error If the matrix stays singular after the ridge.
Matrix spd_inverse(Matrix a, double ridge) {
    auto f = linalg::cholesky(a);
    if (!f.ok) {
        for (std::size_t j = 0; j < a.rows(); ++j)
            a(j,j) *= 1+ridge;
        f = linalg::cholesky(a);
        if (!f.ok)
            throw std::runtime_error("mice_impute: singular regression (a column is constant among the observed cases)");
    }
    const std::size_t p = a.rows();
    Matrix inv(p,p);
    std::vector<double> e(p);
    for (std::size_t j = 0; j < p; ++j) {
        std::fill(e.begin(),e.end(),0.0);
        e[j] = 1;
        const auto col = f.solve(e);
        for (std::size_t i = 0; i < p; ++i)
            inv(i,j) = col[i];
    }
    return inv;
}

/// @brief Fits y on X over the given rows and draws parameters from their posterior.
/// @param x Design (with intercept), all rows.
/// @param y Response, all rows.
/// @param rows Rows with observed response.
/// @param ridge Relative ridge for singular X'X.
/// @param rng Random generator.
/// @return Least-squares and drawn parameters.
NormDraw norm_draw(const Matrix& x, const std::vector<double>& y, const std::vector<std::size_t>& rows, double ridge,
                   random::RandomGenerator& rng) {
    const std::size_t p = x.cols();
    Matrix xtx(p,p);
    std::vector<double> xty(p,0.0);
    for (const std::size_t i : rows)
        for (std::size_t a = 0; a < p; ++a) {
            xty[a] += x(i,a)*y[i];
            for (std::size_t b = 0; b < p; ++b)
                xtx(a,b) += x(i,a)*x(i,b);
        }
    const Matrix v = spd_inverse(xtx,ridge);
    NormDraw d;
    d.coef.assign(p,0.0);
    for (std::size_t a = 0; a < p; ++a)
        for (std::size_t b = 0; b < p; ++b)
            d.coef[a] += v(a,b)*xty[b];
    double rss = 0;
    for (const std::size_t i : rows) {
        double fit = 0;
        for (std::size_t a = 0; a < p; ++a)
            fit += x(i,a)*d.coef[a];
        rss += (y[i]-fit)*(y[i]-fit);
    }
    const double df = std::max(static_cast<double>(rows.size())-static_cast<double>(p),1.0);
    d.sigma = std::sqrt(rss/random::chi_squared_variate(rng,df));
    const auto l = linalg::cholesky(v);
    if (!l.ok)
        throw std::runtime_error("mice_impute: covariance of the regression coefficients is not positive definite");
    std::vector<double> z(p);
    for (double& v_ : z)
        v_ = rng.normal();
    d.beta = d.coef;
    for (std::size_t a = 0; a < p; ++a)
        for (std::size_t b = 0; b <= a; ++b)
            d.beta[a] += l.L(a,b)*z[b]*d.sigma;
    return d;
}

} // namespace

MiceResult mice_impute(const linalg::DenseMatrix<double>& data, random::RandomGenerator& rng, const MiceOptions& options) {
    const std::size_t n = data.rows(), p = data.cols();
    if (n == 0 || p == 0)
        throw std::invalid_argument("mice_impute: empty data");
    if (options.imputations == 0 || options.iterations == 0 || options.donors == 0)
        throw std::invalid_argument("mice_impute: imputations, iterations and donors must be positive");
    if ((!options.methods.empty() && options.methods.size() != p) || (!options.shift.empty() && options.shift.size() != p))
        throw std::invalid_argument("mice_impute: methods and shift need one entry per column");
    if (!options.shift_rows.empty() && options.shift_rows.size() != n)
        throw std::invalid_argument("mice_impute: shift_rows needs one entry per row");
    std::vector<std::vector<std::size_t>> observed(p), missing(p);
    for (std::size_t j = 0; j < p; ++j) {
        for (std::size_t i = 0; i < n; ++i) {
            if (std::isinf(data(i,j)))
                throw std::invalid_argument("mice_impute: values must be finite or NaN");
            (std::isnan(data(i,j)) ? missing[j] : observed[j]).push_back(i);
        }
        if (observed[j].empty())
            throw std::invalid_argument("mice_impute: a column has no observed values");
        if (!options.shift.empty() && !std::isfinite(options.shift[j]))
            throw std::invalid_argument("mice_impute: shifts must be finite");
    }
    MiceResult result;
    for (std::size_t j = 0; j < p; ++j)
        if (!missing[j].empty())
            result.incomplete_columns.push_back(j);
    for (std::size_t m = 0; m < options.imputations; ++m) {
        Matrix x = data;
        for (const std::size_t j : result.incomplete_columns)
            for (const std::size_t i : missing[j])
                x(i,j) = data(observed[j][rng.uniform_u64(0,observed[j].size()-1)],j);
        Matrix chain(options.iterations,p,std::numeric_limits<double>::quiet_NaN());
        for (std::size_t it = 0; it < options.iterations; ++it) {
            for (const std::size_t j : result.incomplete_columns) {
                Matrix design(n,p);
                std::vector<double> y(n);
                for (std::size_t i = 0; i < n; ++i) {
                    design(i,0) = 1;
                    std::size_t c = 1;
                    for (std::size_t k = 0; k < p; ++k)
                        if (k != j)
                            design(i,c++) = x(i,k);
                    y[i] = x(i,j);
                }
                const auto draw = norm_draw(design,y,observed[j],options.ridge,rng);
                const auto method = options.methods.empty() ? ImputationMethod::pmm : options.methods[j];
                const double column_shift = options.shift.empty() ? 0.0 : options.shift[j];
                const auto shift_of = [&](std::size_t i) {
                    return options.shift_rows.empty() || options.shift_rows[i] ? column_shift : 0.0;
                };
                const auto predict = [&](std::size_t i, const std::vector<double>& b) {
                    double s = 0;
                    for (std::size_t a = 0; a < p; ++a)
                        s += design(i,a)*b[a];
                    return s;
                };
                if (method == ImputationMethod::norm) {
                    for (const std::size_t i : missing[j])
                        x(i,j) = predict(i,draw.beta)+draw.sigma*rng.normal()+shift_of(i);
                } else {
                    // Type-1 matching: observed cases use the least-squares fit, missing ones the draw.
                    std::vector<std::size_t> donors(observed[j]);
                    for (std::size_t k = donors.size(); k > 1; --k) // random order breaks ties at random
                        std::swap(donors[k-1],donors[rng.uniform_u64(0,k-1)]);
                    std::vector<double> yhat(donors.size());
                    for (std::size_t k = 0; k < donors.size(); ++k)
                        yhat[k] = predict(donors[k],draw.coef);
                    const std::size_t pool = std::min(options.donors,donors.size());
                    std::vector<std::size_t> order(donors.size());
                    for (const std::size_t i : missing[j]) {
                        const double target = predict(i,draw.beta);
                        std::iota(order.begin(),order.end(),std::size_t{0});
                        std::nth_element(order.begin(),order.begin()+static_cast<std::ptrdiff_t>(pool-1),order.end(),
                                         [&](std::size_t a, std::size_t b) {
                                             return std::abs(yhat[a]-target) < std::abs(yhat[b]-target);
                                         });
                        const std::size_t pick = order[rng.uniform_u64(0,pool-1)];
                        x(i,j) = y[donors[pick]]+shift_of(i);
                    }
                }
                double mean = 0;
                for (const std::size_t i : missing[j])
                    mean += x(i,j)/static_cast<double>(missing[j].size());
                chain(it,j) = mean;
            }
        }
        result.completed.push_back(std::move(x));
        result.chain_means.push_back(std::move(chain));
    }
    return result;
}

RubinPooling pool_rubin(const std::vector<std::vector<double>>& estimates, const std::vector<linalg::DenseMatrix<double>>& covariances,
                        double complete_df, double confidence) {
    const std::size_t m = estimates.size();
    if (m < 2 || covariances.size() != m)
        throw std::invalid_argument("pool_rubin: need at least two imputations with one covariance each");
    if (!(complete_df > 0) || !(confidence > 0 && confidence < 1))
        throw std::invalid_argument("pool_rubin: complete-data df must be positive and confidence in (0, 1)");
    const std::size_t k = estimates[0].size();
    if (k == 0)
        throw std::invalid_argument("pool_rubin: need at least one parameter");
    for (std::size_t i = 0; i < m; ++i) {
        if (estimates[i].size() != k || covariances[i].rows() != k || covariances[i].cols() != k)
            throw std::invalid_argument("pool_rubin: estimates and covariances must have matching sizes");
        for (std::size_t a = 0; a < k; ++a) {
            if (!std::isfinite(estimates[i][a]))
                throw std::invalid_argument("pool_rubin: estimates must be finite");
            for (std::size_t b = 0; b < k; ++b)
                if (!std::isfinite(covariances[i](a,b)))
                    throw std::invalid_argument("pool_rubin: covariances must be finite");
        }
    }
    const double md = static_cast<double>(m);
    RubinPooling r;
    r.imputations = m;
    r.complete_df = complete_df;
    r.estimate.assign(k,0.0);
    r.within = Matrix(k,k);
    r.between = Matrix(k,k);
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t a = 0; a < k; ++a) {
            r.estimate[a] += estimates[i][a]/md;
            for (std::size_t b = 0; b < k; ++b)
                r.within(a,b) += covariances[i](a,b)/md;
        }
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t a = 0; a < k; ++a)
            for (std::size_t b = 0; b < k; ++b)
                r.between(a,b) += (estimates[i][a]-r.estimate[a])*(estimates[i][b]-r.estimate[b])/(md-1);
    for (std::size_t a = 0; a < k; ++a) {
        PooledEstimate e;
        e.estimate = r.estimate[a];
        e.within = r.within(a,a);
        e.between = r.between(a,a);
        e.total = e.within+(1+1/md)*e.between;
        e.standard_error = std::sqrt(e.total);
        e.riv = (1+1/md)*e.between/e.within;
        e.lambda = (1+1/md)*e.between/e.total;
        const double old = (md-1)/(e.lambda*e.lambda);
        if (std::isinf(complete_df)) {
            e.df = old;
        } else {
            const double tmp = (1-e.lambda)*(1+complete_df)*complete_df;
            e.df = (md-1)*tmp/((complete_df+3)*(md-1)+e.lambda*e.lambda*tmp);
        }
        e.fmi = (e.riv+2/(e.df+3))/(e.riv+1);
        e.t_value = e.estimate/e.standard_error;
        const double df = std::min(e.df,1e10);
        e.p_value = 2*(1-random::student_t_cdf(std::abs(e.t_value),df));
        const double q = random::student_t_quantile(0.5+confidence/2,df);
        e.lower = e.estimate-q*e.standard_error;
        e.upper = e.estimate+q*e.standard_error;
        r.parameters.push_back(e);
    }
    return r;
}

PooledWaldTest pooled_wald_test(const RubinPooling& pooling, const std::vector<std::size_t>& indices) {
    const std::size_t p = pooling.estimate.size(), k = indices.size();
    if (k == 0 || pooling.imputations < 2)
        throw std::invalid_argument("pooled_wald_test: need a pooling result and at least one parameter");
    std::vector<bool> seen(p,false);
    for (const std::size_t j : indices) {
        if (j >= p || seen[j])
            throw std::invalid_argument("pooled_wald_test: indices must be distinct and in range");
        seen[j] = true;
    }
    Matrix u(k,k), b(k,k);
    std::vector<double> q(k);
    for (std::size_t a = 0; a < k; ++a) {
        q[a] = pooling.estimate[indices[a]];
        for (std::size_t c = 0; c < k; ++c) {
            u(a,c) = pooling.within(indices[a],indices[c]);
            b(a,c) = pooling.between(indices[a],indices[c]);
        }
    }
    const auto f = linalg::cholesky(u);
    if (!f.ok)
        throw std::runtime_error("pooled_wald_test: singular within-imputation covariance");
    const double md = static_cast<double>(pooling.imputations), kd = static_cast<double>(k);
    double trace = 0;
    std::vector<double> col(k);
    for (std::size_t c = 0; c < k; ++c) { // tr(B U^{-1}) = sum_c (U^{-1} B)_{cc}
        for (std::size_t a = 0; a < k; ++a)
            col[a] = b(a,c);
        trace += f.solve(col)[c];
    }
    PooledWaldTest t;
    t.riv = (1+1/md)*trace/kd;
    const auto s = f.solve(q);
    double quad = 0;
    for (std::size_t a = 0; a < k; ++a)
        quad += q[a]*s[a];
    t.statistic = quad/(1+t.riv)/kd;
    t.df1 = kd;
    const double tt = kd*(md-1), r = t.riv;
    if (std::isfinite(pooling.complete_df)) {
        const double a = r*tt/(tt-2), vstar = (pooling.complete_df+1)/(pooling.complete_df+3)*pooling.complete_df;
        const double c0 = 1/(tt-4), c1 = vstar-2*(1+a), c2 = vstar-4*(1+a);
        const double z = 1/c2+c0*(a*a*c1/((1+a)*(1+a)*c2))+c0*(8*a*a*c1/((1+a)*c2*c2)+4*a*a/((1+a)*c2))+
                         c0*(4*a*a/(c2*c1)+16*a*a*c1/(c2*c2*c2))+c0*(8*a*a/(c2*c2));
        t.df2 = 4+1/z;
    } else if (tt > 4) {
        t.df2 = 4+(tt-4)*std::pow(1+(1-2/tt)/r,2);
    } else {
        t.df2 = tt*(1+1/kd)*std::pow(1+1/r,2)/2;
    }
    t.p_value = 1-random::f_cdf(t.statistic,t.df1,std::min(t.df2,1e10));
    return t;
}

TippingPointResult tipping_point_analysis(const linalg::DenseMatrix<double>& data, const ImputedAnalysis& analysis,
                                          const TippingPointOptions& options) {
    if (options.deltas.empty() || options.column >= data.cols() || !analysis)
        throw std::invalid_argument("tipping_point_analysis: need deltas, a valid column and an analysis");
    if (!options.shifted_rows.empty() && options.shifted_rows.size() != data.rows())
        throw std::invalid_argument("tipping_point_analysis: shifted_rows needs one entry per row");
    if (!(options.alpha > 0 && options.alpha < 1))
        throw std::invalid_argument("tipping_point_analysis: alpha must lie in (0, 1)");
    TippingPointResult out;
    for (const double delta : options.deltas) {
        if (!std::isfinite(delta))
            throw std::invalid_argument("tipping_point_analysis: deltas must be finite");
        MiceOptions mo = options.mice;
        mo.shift.assign(data.cols(),0.0);
        mo.shift[options.column] = delta;
        mo.shift_rows = options.shifted_rows;
        random::SplitMix64 rng(options.seed);
        const auto imp = mice_impute(data,rng,mo);
        std::vector<std::vector<double>> est;
        std::vector<Matrix> cov;
        for (const auto& c : imp.completed) {
            auto [e,v] = analysis(c);
            if (options.parameter >= e.size())
                throw std::invalid_argument("tipping_point_analysis: parameter index out of range");
            est.push_back(std::move(e));
            cov.push_back(std::move(v));
        }
        const auto pooled = pool_rubin(est,cov,options.complete_df);
        out.rows.push_back({delta,pooled.parameters[options.parameter]});
    }
    const bool significant = out.rows[0].result.p_value < options.alpha;
    const double sign = out.rows[0].result.estimate;
    for (std::size_t k = 1; k < out.rows.size(); ++k) {
        const auto& a = out.rows[k-1];
        const auto& b = out.rows[k];
        if (!out.tipping_delta && (b.result.p_value < options.alpha) != significant) {
            const double pa = a.result.p_value, pb = b.result.p_value;
            const double f = pb != pa ? (options.alpha-pa)/(pb-pa) : 0.0;
            out.tipping_delta = a.delta+f*(b.delta-a.delta);
        }
        if (!out.null_delta && sign != 0 && b.result.estimate*sign <= 0) {
            const double ea = a.result.estimate, eb = b.result.estimate;
            const double f = eb != ea ? ea/(ea-eb) : 0.0;
            out.null_delta = a.delta+f*(b.delta-a.delta);
        }
    }
    return out;
}

} // namespace datamunge::stats
