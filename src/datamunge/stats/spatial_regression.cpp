/// @file spatial_regression.cpp
/// @brief Concentrated likelihoods, log-determinants and asymptotic covariances of SAR and SEM models.
#include <datamunge/stats/spatial_regression.hpp>

#include <datamunge/linalg/cholesky.hpp>
#include <datamunge/linalg/eigen.hpp>
#include <datamunge/linalg/lu.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <numbers>
#include <stdexcept>
#include <utility>

namespace datamunge::stats {

/// @brief Implementation helpers for spatial regression.
namespace {
/// @brief Dense matrix used throughout the implementation.
using Matrix = linalg::DenseMatrix<double>;

/// @brief Checks the inputs shared by both models.
/// @param y Response.
/// @param x Design.
/// @param w Weights.
void validate(const std::vector<double>& y, const Matrix& x, const Matrix& w) {
    const std::size_t n = y.size();
    if (n == 0 || x.rows() != n || w.rows() != n || w.cols() != n || x.cols() == 0 || n <= x.cols() + 1)
        throw std::invalid_argument("spatial regression: need n > p + 1 and matching y, X and W");
    const auto finite = [](double v) { return std::isfinite(v); };
    if (!std::ranges::all_of(y, finite) || !std::ranges::all_of(x.data(), finite) ||
        !std::ranges::all_of(w.data(), finite))
        throw std::invalid_argument("spatial regression: inputs must be finite");
    for (std::size_t i = 0; i < n; ++i)
        if (w(i, i) != 0)
            throw std::invalid_argument("spatial regression: the weight diagonal must be zero");
}

/// @brief Ordinary least squares by the normal equations.
/// @param x Design.
/// @param y Response.
/// @return Coefficients and residuals.
std::pair<std::vector<double>, std::vector<double>> ols(const Matrix& x, const std::vector<double>& y) {
    const std::size_t n = x.rows(), p = x.cols();
    Matrix xtx(p, p);
    std::vector<double> xty(p, 0.0);
    for (std::size_t a = 0; a < p; ++a) {
        for (std::size_t b = 0; b <= a; ++b) {
            double s = 0;
            for (std::size_t i = 0; i < n; ++i)
                s += x(i, a) * x(i, b);
            xtx(a, b) = xtx(b, a) = s;
        }
        for (std::size_t i = 0; i < n; ++i)
            xty[a] += x(i, a) * y[i];
    }
    const auto c = linalg::cholesky(xtx);
    if (!c.ok)
        throw std::runtime_error("spatial regression: the design matrix is rank deficient");
    std::vector<double> beta = c.solve(xty), r(y);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t a = 0; a < p; ++a)
            r[i] -= x(i, a) * beta[a];
    return {beta, r};
}

/// @brief Matrix-vector product W v.
/// @param w Matrix.
/// @param v Vector.
/// @return The product.
std::vector<double> times(const Matrix& w, const std::vector<double>& v) {
    std::vector<double> out(w.rows(), 0.0);
    for (std::size_t i = 0; i < w.rows(); ++i)
        for (std::size_t j = 0; j < w.cols(); ++j)
            out[i] += w(i, j) * v[j];
    return out;
}

/// @brief log|I - r W| by eigenvalues (when W is symmetrisable) or LU.
struct LogDeterminant {
    const Matrix* w{};               ///< Weights.
    std::vector<double> eigenvalues; ///< Real eigenvalues (empty when LU is used).

    /// @brief Evaluates log|I - r W|.
    /// @param r Spatial parameter.
    /// @return The log absolute determinant.
    double operator()(double r) const {
        if (!eigenvalues.empty()) {
            double s = 0;
            for (const double e : eigenvalues)
                s += std::log(std::abs(1 - r * e));
            return s;
        }
        return lu_value(r);
    }
    /// @brief Evaluates log|I - r W| by LU.
    /// @param r Spatial parameter.
    /// @return The log absolute determinant.
    double lu_value(double r) const {
        const std::size_t n = w->rows();
        Matrix a(n, n);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
                a(i, j) = (i == j ? 1.0 : 0.0) - r * (*w)(i, j);
        const auto f = linalg::lu(a);
        if (f.singular)
            return -std::numeric_limits<double>::infinity();
        double s = 0;
        for (std::size_t i = 0; i < n; ++i)
            s += std::log(std::abs(f.LU(i, i)));
        return s;
    }
};

/// @brief Prepares the log-determinant, using eigenvalues of the symmetrised W when that is exact.
/// @param w Weights.
/// @return The evaluator.
LogDeterminant make_log_determinant(const Matrix& w) {
    LogDeterminant ld{&w, {}};
    const std::size_t n = w.rows();
    Matrix s(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            const double a = w(i, j), b = w(j, i);
            if ((a == 0) != (b == 0) || a * b < 0)
                return ld; // not similar to a symmetric matrix: use LU
            s(i, j) = a == 0 ? 0.0 : std::copysign(std::sqrt(a * b), a);
        }
    std::vector<double> e = linalg::symmetric_eigen(s).eigenvalues;
    const double scale = std::max(std::abs(e.front()), std::abs(e.back()));
    if (!(scale > 0))
        return ld;
    LogDeterminant trial{&w, e};
    const double r = 0.5 / scale;
    if (std::abs(trial(r) - ld.lu_value(r)) < 1e-8 * static_cast<double>(n) &&
        std::abs(trial(-r) - ld.lu_value(-r)) < 1e-8 * static_cast<double>(n))
        return trial;
    return ld;
}

/// @brief Maximises a function on an open interval by a grid and golden-section refinement.
/// @param f Function.
/// @param lo Lower bound.
/// @param hi Upper bound.
/// @return The maximiser.
double maximise(const std::function<double(double)>& f, double lo, double hi) {
    constexpr int grid = 200;
    std::vector<double> x(grid + 1);
    std::size_t best = 1;
    double fb = -std::numeric_limits<double>::infinity();
    for (int k = 1; k < grid; ++k) {
        x[k] = lo + (hi - lo) * k / grid;
        const double v = f(x[k]);
        if (v > fb)
            fb = v, best = static_cast<std::size_t>(k);
    }
    double a = best == 1 ? lo + (hi - lo) * 1e-9 : x[best - 1];
    double b = best == grid - 1 ? hi - (hi - lo) * 1e-9 : x[best + 1];
    const double g = (std::sqrt(5.0) - 1) / 2;
    double x1 = b - g * (b - a), x2 = a + g * (b - a), f1 = f(x1), f2 = f(x2);
    while (b - a > 1e-12 * std::max(1.0, std::abs(a) + std::abs(b))) {
        if (f1 >= f2) {
            b = x2, x2 = x1, f2 = f1;
            x1 = b - g * (b - a);
            f1 = f(x1);
        } else {
            a = x1, x1 = x2, f1 = f2;
            x2 = a + g * (b - a);
            f2 = f(x2);
        }
    }
    return f1 >= f2 ? x1 : x2;
}

/// @brief Inverse of I - r W by LU.
/// @param w Weights.
/// @param r Spatial parameter.
/// @return The inverse.
Matrix inverse_filter(const Matrix& w, double r) {
    const std::size_t n = w.rows();
    Matrix a(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            a(i, j) = (i == j ? 1.0 : 0.0) - r * w(i, j);
    const auto f = linalg::lu(a);
    if (f.singular)
        throw std::runtime_error("spatial regression: I - rho W is singular");
    Matrix inv(n, n);
    std::vector<double> e(n, 0.0);
    for (std::size_t j = 0; j < n; ++j) {
        e[j] = 1;
        const auto col = f.solve(e);
        e[j] = 0;
        for (std::size_t i = 0; i < n; ++i)
            inv(i, j) = col[i];
    }
    return inv;
}

/// @brief Product of two square matrices.
/// @param a Left factor.
/// @param b Right factor.
/// @return a b.
Matrix product(const Matrix& a, const Matrix& b) {
    const std::size_t n = a.rows(), m = b.cols(), k = a.cols();
    Matrix out(n, m);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t l = 0; l < k; ++l) {
            const double v = a(i, l);
            if (v == 0)
                continue;
            for (std::size_t j = 0; j < m; ++j)
                out(i, j) += v * b(l, j);
        }
    return out;
}

/// @brief Traces tr(A), tr(A A) and tr(A' A).
/// @param a Square matrix.
/// @return The three traces.
std::array<double, 3> traces(const Matrix& a) {
    const std::size_t n = a.rows();
    double t = 0, taa = 0, tata = 0;
    for (std::size_t i = 0; i < n; ++i) {
        t += a(i, i);
        for (std::size_t j = 0; j < n; ++j) {
            taa += a(i, j) * a(j, i);
            tata += a(i, j) * a(i, j);
        }
    }
    return {t, taa, tata};
}

/// @brief Inverse of a symmetric positive definite matrix.
/// @param a Matrix.
/// @return The inverse.
Matrix spd_inverse(const Matrix& a) {
    const auto c = linalg::cholesky(a);
    if (!c.ok)
        throw std::runtime_error("spatial regression: the information matrix is singular");
    const std::size_t n = a.rows();
    Matrix inv(n, n);
    std::vector<double> e(n, 0.0);
    for (std::size_t j = 0; j < n; ++j) {
        e[j] = 1;
        const auto col = c.solve(e);
        e[j] = 0;
        for (std::size_t i = 0; i < n; ++i)
            inv(i, j) = col[i];
    }
    return inv;
}

/// @brief Two-sided normal p-value.
/// @param z Statistic.
/// @return The p-value.
double two_sided(double z) { return std::erfc(std::abs(z) / std::numbers::sqrt2); }

/// @brief Fills the search interval, likelihood summaries and Wald statistics shared by both models.
/// @param r Result to complete.
/// @param n Observations.
/// @param ols_sse OLS residual sum of squares.
void finish(SpatialRegressionResult& r, std::size_t n, double ols_sse) {
    const double nd = static_cast<double>(n);
    r.ols_log_likelihood = -nd / 2 * (std::log(2 * std::numbers::pi) + std::log(ols_sse / nd) + 1);
    r.lr_statistic = std::max(0.0, 2 * (r.log_likelihood - r.ols_log_likelihood));
    r.lr_p_value = std::erfc(std::sqrt(r.lr_statistic / 2));
    r.aic = -2 * r.log_likelihood + 2 * static_cast<double>(r.coefficients.size() + 2);
    r.spatial_z = r.spatial_parameter / r.spatial_se;
    r.spatial_p_value = two_sided(r.spatial_z);
    for (std::size_t k = 0; k < r.coefficients.size(); ++k) {
        r.z.push_back(r.coefficients[k] / r.standard_errors[k]);
        r.p_values.push_back(two_sided(r.z.back()));
    }
}

/// @brief Resolves the search interval from options and eigenvalues.
/// @param ld Log-determinant (with eigenvalues when available).
/// @param options User bounds.
/// @return Lower and upper bounds.
std::pair<double, double> interval(const LogDeterminant& ld, const SpatialRegressionOptions& options) {
    double lo = -1, hi = 1;
    if (!ld.eigenvalues.empty()) {
        const double emax = ld.eigenvalues.front(), emin = ld.eigenvalues.back();
        if (emin < 0)
            lo = 1 / emin;
        if (emax > 0)
            hi = 1 / emax;
    }
    if (!std::isnan(options.lower))
        lo = options.lower;
    if (!std::isnan(options.upper))
        hi = options.upper;
    if (!(lo < hi) || !std::isfinite(lo) || !std::isfinite(hi))
        throw std::invalid_argument("spatial regression: the search interval is empty");
    return {lo, hi};
}
} // namespace

SpatialRegressionResult spatial_lag_regression(const std::vector<double>& y, const linalg::DenseMatrix<double>& x,
                                               const linalg::DenseMatrix<double>& weights,
                                               const SpatialRegressionOptions& options) {
    validate(y, x, weights);
    const std::size_t n = y.size(), p = x.cols();
    const double nd = static_cast<double>(n);
    const LogDeterminant ld = make_log_determinant(weights);
    const auto [lo, hi] = interval(ld, options);
    const std::vector<double> wy = times(weights, y);
    const auto [b0, e0] = ols(x, y);
    const auto [bl, el] = ols(x, wy);
    const auto sse = [&](double r) {
        double s = 0;
        for (std::size_t i = 0; i < n; ++i)
            s += (e0[i] - r * el[i]) * (e0[i] - r * el[i]);
        return s;
    };
    const auto loglik = [&](double r) {
        return -nd / 2 * (std::log(2 * std::numbers::pi) + std::log(sse(r) / nd) + 1) + ld(r);
    };
    SpatialRegressionResult out;
    out.lower = lo;
    out.upper = hi;
    const double rho = maximise(loglik, lo, hi);
    out.spatial_parameter = rho;
    out.log_likelihood = loglik(rho);
    out.sigma2 = sse(rho) / nd;
    for (std::size_t k = 0; k < p; ++k)
        out.coefficients.push_back(b0[k] - rho * bl[k]);
    // Information matrix of (beta, rho, sigma^2).
    const Matrix ainv = inverse_filter(weights, rho);
    const Matrix wa = product(weights, ainv);
    const auto [tr, trwawa, trwatwa] = traces(wa);
    std::vector<double> xb(n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t k = 0; k < p; ++k)
            xb[i] += x(i, k) * out.coefficients[k];
    const std::vector<double> waxb = times(wa, xb);
    const double s2 = out.sigma2;
    Matrix info(p + 2, p + 2);
    for (std::size_t a = 0; a < p; ++a) {
        for (std::size_t b = 0; b < p; ++b) {
            double v = 0;
            for (std::size_t i = 0; i < n; ++i)
                v += x(i, a) * x(i, b);
            info(a, b) = v / s2;
        }
        double v = 0;
        for (std::size_t i = 0; i < n; ++i)
            v += x(i, a) * waxb[i];
        info(a, p) = info(p, a) = v / s2;
    }
    double q = 0;
    for (const double v : waxb)
        q += v * v;
    info(p, p) = trwawa + trwatwa + q / s2;
    info(p, p + 1) = info(p + 1, p) = tr / s2;
    info(p + 1, p + 1) = nd / (2 * s2 * s2);
    const Matrix cov = spd_inverse(info);
    for (std::size_t k = 0; k < p; ++k)
        out.standard_errors.push_back(std::sqrt(cov(k, k)));
    out.spatial_se = std::sqrt(cov(p, p));
    // Fitted values y - innovations (as spatialreg) and impacts.
    double trace_inv = 0, total_inv = 0;
    for (std::size_t i = 0; i < n; ++i) {
        trace_inv += ainv(i, i);
        for (std::size_t j = 0; j < n; ++j)
            total_inv += ainv(i, j);
    }
    for (std::size_t i = 0; i < n; ++i) {
        out.fitted.push_back(rho * wy[i] + xb[i]);
        out.residuals.push_back(y[i] - out.fitted.back());
    }
    for (std::size_t k = 0; k < p; ++k) {
        SpatialImpact im;
        im.direct = out.coefficients[k] * trace_inv / nd;
        im.total = out.coefficients[k] * total_inv / nd;
        im.indirect = im.total - im.direct;
        out.impacts.push_back(im);
    }
    double ols_sse = 0;
    for (const double v : e0)
        ols_sse += v * v;
    finish(out, n, ols_sse);
    return out;
}

SpatialRegressionResult spatial_error_regression(const std::vector<double>& y, const linalg::DenseMatrix<double>& x,
                                                 const linalg::DenseMatrix<double>& weights,
                                                 const SpatialRegressionOptions& options) {
    validate(y, x, weights);
    const std::size_t n = y.size(), p = x.cols();
    const double nd = static_cast<double>(n);
    const LogDeterminant ld = make_log_determinant(weights);
    const auto [lo, hi] = interval(ld, options);
    const std::vector<double> wy = times(weights, y);
    Matrix wx(n, p);
    for (std::size_t k = 0; k < p; ++k) {
        std::vector<double> col(n);
        for (std::size_t i = 0; i < n; ++i)
            col[i] = x(i, k);
        const auto wc = times(weights, col);
        for (std::size_t i = 0; i < n; ++i)
            wx(i, k) = wc[i];
    }
    const auto filtered = [&](double l) {
        Matrix xl(n, p);
        std::vector<double> yl(n);
        for (std::size_t i = 0; i < n; ++i) {
            yl[i] = y[i] - l * wy[i];
            for (std::size_t k = 0; k < p; ++k)
                xl(i, k) = x(i, k) - l * wx(i, k);
        }
        return std::pair{xl, yl};
    };
    const auto loglik = [&](double l) {
        const auto [xl, yl] = filtered(l);
        const auto [b, e] = ols(xl, yl);
        double s = 0;
        for (const double v : e)
            s += v * v;
        return -nd / 2 * (std::log(2 * std::numbers::pi) + std::log(s / nd) + 1) + ld(l);
    };
    SpatialRegressionResult out;
    out.lower = lo;
    out.upper = hi;
    const double lambda = maximise(loglik, lo, hi);
    out.spatial_parameter = lambda;
    out.log_likelihood = loglik(lambda);
    const auto [xl, yl] = filtered(lambda);
    const auto [beta, innov] = ols(xl, yl);
    out.coefficients = beta;
    double s = 0;
    for (const double v : innov)
        s += v * v;
    out.sigma2 = s / nd;
    Matrix xtx(p, p);
    for (std::size_t a = 0; a < p; ++a)
        for (std::size_t b = 0; b < p; ++b) {
            double v = 0;
            for (std::size_t i = 0; i < n; ++i)
                v += xl(i, a) * xl(i, b);
            xtx(a, b) = v / out.sigma2;
        }
    const Matrix cov = spd_inverse(xtx);
    for (std::size_t k = 0; k < p; ++k)
        out.standard_errors.push_back(std::sqrt(cov(k, k)));
    const Matrix wb = product(weights, inverse_filter(weights, lambda));
    const auto [tr, trwbwb, trwbtwb] = traces(wb);
    const double s2 = out.sigma2;
    Matrix info(2, 2);
    info(0, 0) = trwbwb + trwbtwb;
    info(0, 1) = info(1, 0) = tr / s2;
    info(1, 1) = nd / (2 * s2 * s2);
    out.spatial_se = std::sqrt(spd_inverse(info)(0, 0));
    // Residuals are the innovations (I - lambda W)(y - X beta), as in spatialreg.
    for (std::size_t i = 0; i < n; ++i) {
        out.residuals.push_back(innov[i]);
        out.fitted.push_back(y[i] - innov[i]);
    }
    const auto [b0, e0] = ols(x, y);
    double ols_sse = 0;
    for (const double v : e0)
        ols_sse += v * v;
    finish(out, n, ols_sse);
    return out;
}

} // namespace datamunge::stats
