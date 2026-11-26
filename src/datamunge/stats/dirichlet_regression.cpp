/// @file dirichlet_regression.cpp
/// @brief Fisher scoring for the mean/precision Dirichlet regression.
#include <datamunge/stats/dirichlet_regression.hpp>

#include <datamunge/linalg/cholesky.hpp>
#include <datamunge/random/distributions.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace datamunge::stats {

/// @brief Implementation helpers for Dirichlet regression.
namespace {
/// @brief Dense matrix alias.
using Matrix = linalg::DenseMatrix<double>;
/// @brief Quiet NaN.
constexpr double nan = std::numeric_limits<double>::quiet_NaN();

/// @brief Problem data shared by the likelihood routines.
struct Problem {
    const Matrix& x;          ///< Mean design.
    const Matrix& z;          ///< Precision design.
    Matrix log_y;             ///< Logs of the (closed, compressed) response.
    std::size_t base;         ///< Reference part.

    /// @brief Number of parts.
    /// @return D.
    std::size_t parts() const { return log_y.cols(); }
    /// @brief Mean-design columns.
    /// @return p.
    std::size_t p() const { return x.cols(); }
    /// @brief Precision-design columns.
    /// @return q.
    std::size_t q() const { return z.cols(); }
    /// @brief Number of free parameters.
    /// @return p(D - 1) + q.
    std::size_t size() const { return p() * (parts() - 1) + q(); }
    /// @brief Position of part j's coefficient block (j != base).
    /// @param j Part.
    /// @return Block index in 0..D-2.
    std::size_t block(std::size_t j) const { return j < base ? j : j - 1; }
};

/// @brief Means and precision of one row at parameters theta.
/// @param pr Problem.
/// @param theta Parameters.
/// @param i Row.
/// @param mu Output means.
/// @return Precision.
double row_parameters(const Problem& pr, const std::vector<double>& theta, std::size_t i, std::vector<double>& mu) {
    const std::size_t d = pr.parts(), p = pr.p();
    mu.assign(d, 0.0);
    double m = 0;
    for (std::size_t j = 0; j < d; ++j) {
        if (j == pr.base)
            continue;
        double eta = 0;
        for (std::size_t c = 0; c < p; ++c)
            eta += pr.x(i, c) * theta[pr.block(j) * p + c];
        mu[j] = eta;
        m = std::max(m, eta);
    }
    double s = 0;
    for (std::size_t j = 0; j < d; ++j)
        s += (mu[j] = std::exp(mu[j] - m));
    for (auto& v : mu)
        v /= s;
    double lp = 0;
    for (std::size_t c = 0; c < pr.q(); ++c)
        lp += pr.z(i, c) * theta[p * (d - 1) + c];
    return std::exp(lp);
}

/// @brief Log-likelihood, and optionally score and expected information.
/// @param pr Problem.
/// @param theta Parameters.
/// @param score Output score (if non-null).
/// @param info Output expected information (if non-null).
/// @return Log-likelihood (-inf if a parameter is not finite).
double evaluate(const Problem& pr, const std::vector<double>& theta, std::vector<double>* score, Matrix* info) {
    const std::size_t n = pr.log_y.rows(), d = pr.parts(), p = pr.p(), q = pr.q(), k = pr.size();
    if (score)
        score->assign(k, 0.0);
    if (info)
        *info = Matrix(k, k, 0.0);
    double ll = 0;
    std::vector<double> mu, g(d), dpsi(d);
    // Linear-predictor coordinates: the D - 1 mean predictors, then log phi.
    Matrix jac(d, d, 0.0), m_info(d, d, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        const double phi = row_parameters(pr, theta, i, mu);
        if (!std::isfinite(phi) || !(phi > 0))
            return -std::numeric_limits<double>::infinity();
        ll += std::lgamma(phi);
        const double psi_phi = random::digamma(phi);
        for (std::size_t j = 0; j < d; ++j) {
            const double a = mu[j] * phi;
            if (!(a > 0))
                return -std::numeric_limits<double>::infinity();
            ll += -std::lgamma(a) + (a - 1) * pr.log_y(i, j);
            g[j] = psi_phi - random::digamma(a) + pr.log_y(i, j);
            dpsi[j] = random::trigamma(a);
        }
        if (!score && !info)
            continue;
        // jac(j, r): d alpha_j / d (eta_r for r < D - 1, log phi for r = D - 1).
        for (std::size_t j = 0; j < d; ++j) {
            for (std::size_t kk = 0; kk < d; ++kk) {
                if (kk == pr.base)
                    continue;
                jac(j, pr.block(kk)) = phi * mu[j] * ((j == kk) - mu[kk]);
            }
            jac(j, d - 1) = mu[j] * phi;
        }
        // Score in predictor coordinates.
        std::vector<double> u(d, 0.0);
        for (std::size_t r = 0; r < d; ++r)
            for (std::size_t j = 0; j < d; ++j)
                u[r] += jac(j, r) * g[j];
        const auto design = [&](std::size_t r, std::size_t c) { return r < d - 1 ? pr.x(i, c) : pr.z(i, c); };
        const auto offset = [&](std::size_t r) { return r < d - 1 ? r * p : p * (d - 1); };
        const auto width = [&](std::size_t r) { return r < d - 1 ? p : q; };
        if (score)
            for (std::size_t r = 0; r < d; ++r)
                for (std::size_t c = 0; c < width(r); ++c)
                    (*score)[offset(r) + c] += u[r] * design(r, c);
        if (info) {
            // E[-d2 l / d alpha d alpha'] = diag(trigamma(alpha)) - trigamma(phi) 11'.
            const double tp = random::trigamma(phi);
            for (std::size_t r = 0; r < d; ++r)
                for (std::size_t s = r; s < d; ++s) {
                    double v = 0, sr = 0, ss = 0;
                    for (std::size_t j = 0; j < d; ++j) {
                        v += jac(j, r) * dpsi[j] * jac(j, s);
                        sr += jac(j, r), ss += jac(j, s);
                    }
                    m_info(r, s) = m_info(s, r) = v - tp * sr * ss;
                }
            for (std::size_t r = 0; r < d; ++r)
                for (std::size_t s = 0; s < d; ++s)
                    for (std::size_t a = 0; a < width(r); ++a)
                        for (std::size_t b = 0; b < width(s); ++b)
                            (*info)(offset(r) + a, offset(s) + b) += m_info(r, s) * design(r, a) * design(s, b);
        }
    }
    return ll;
}

/// @brief Least-squares coefficients of y on the columns of a (normal equations).
/// @param a Design.
/// @param y Response.
/// @return Coefficients (zeros if singular).
std::vector<double> least_squares(const Matrix& a, const std::vector<double>& y) {
    const std::size_t n = a.rows(), p = a.cols();
    Matrix ata(p, p, 0.0);
    std::vector<double> aty(p, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t r = 0; r < p; ++r) {
            aty[r] += a(i, r) * y[i];
            for (std::size_t s = 0; s < p; ++s)
                ata(r, s) += a(i, r) * a(i, s);
        }
    const auto chol = linalg::cholesky(ata);
    return chol.ok ? chol.solve(aty) : std::vector<double>(p, 0.0);
}

/// @brief Inverse of a symmetric positive-definite matrix (empty if not positive definite).
/// @param m Matrix.
/// @return Inverse.
Matrix spd_inverse(const Matrix& m) {
    const auto chol = linalg::cholesky(m);
    if (!chol.ok)
        return {};
    const std::size_t k = m.rows();
    Matrix inv(k, k, 0.0);
    for (std::size_t c = 0; c < k; ++c) {
        std::vector<double> e(k, 0.0);
        e[c] = 1;
        const auto col = chol.solve(e);
        for (std::size_t r = 0; r < k; ++r)
            inv(r, c) = col[r];
    }
    return inv;
}

/// @brief Predictions for a parameter vector.
/// @param pr Problem (designs as given).
/// @param theta Parameters.
/// @return Predictions.
DirichletPrediction predictions(const Problem& pr, const std::vector<double>& theta) {
    const std::size_t n = pr.x.rows(), d = pr.parts();
    DirichletPrediction out{Matrix(n, d, 0.0), std::vector<double>(n), Matrix(n, d, 0.0)};
    std::vector<double> mu;
    for (std::size_t i = 0; i < n; ++i) {
        out.precision[i] = row_parameters(pr, theta, i, mu);
        for (std::size_t j = 0; j < d; ++j)
            out.mean(i, j) = mu[j], out.alpha(i, j) = mu[j] * out.precision[i];
    }
    return out;
}
} // namespace

DirichletPrediction DirichletRegression::predict(const linalg::DenseMatrix<double>& mean_design,
                                                 const linalg::DenseMatrix<double>& precision_design) const {
    if (mean_design.cols() != mean_coefficients.rows() ||
        precision_design.cols() != precision_coefficients.size() || mean_design.rows() != precision_design.rows())
        throw std::invalid_argument("DirichletRegression::predict: design shapes do not match the fit");
    const std::size_t d = mean_coefficients.cols(), p = mean_coefficients.rows();
    std::vector<double> theta;
    for (std::size_t j = 0; j < d; ++j)
        if (j != base)
            for (std::size_t c = 0; c < p; ++c)
                theta.push_back(mean_coefficients(c, j));
    theta.insert(theta.end(), precision_coefficients.begin(), precision_coefficients.end());
    Problem pr{mean_design, precision_design, Matrix(mean_design.rows(), d, 0.0), base};
    return predictions(pr, theta);
}

DirichletRegression dirichlet_regression(const linalg::DenseMatrix<double>& mean_design,
                                         const linalg::DenseMatrix<double>& precision_design,
                                         const linalg::DenseMatrix<double>& response,
                                         const DirichletRegressionOptions& options) {
    const std::size_t n = response.rows(), d = response.cols();
    if (d < 2 || n == 0 || mean_design.rows() != n || precision_design.rows() != n || mean_design.cols() == 0 ||
        precision_design.cols() == 0)
        throw std::invalid_argument("dirichlet_regression: designs and response must share rows; D >= 2");
    if (options.base >= d)
        throw std::invalid_argument("dirichlet_regression: base part out of range");
    for (const Matrix* m : {&mean_design, &precision_design, &response})
        for (std::size_t i = 0; i < m->rows(); ++i)
            for (std::size_t j = 0; j < m->cols(); ++j)
                if (!std::isfinite((*m)(i, j)))
                    throw std::invalid_argument("dirichlet_regression: input must be finite");
    // Close, and compress the boundary if needed.
    Matrix y(n, d, 0.0);
    bool boundary = false;
    for (std::size_t i = 0; i < n; ++i) {
        double s = 0;
        for (std::size_t j = 0; j < d; ++j) {
            if (response(i, j) < 0)
                throw std::invalid_argument("dirichlet_regression: response must be nonnegative");
            s += response(i, j);
        }
        if (!(s > 0))
            throw std::invalid_argument("dirichlet_regression: every response row needs a positive sum");
        for (std::size_t j = 0; j < d; ++j) {
            y(i, j) = response(i, j) / s;
            boundary = boundary || y(i, j) <= 0 || y(i, j) >= 1;
        }
    }
    DirichletRegression fit;
    fit.base = options.base, fit.observations = n;
    if (boundary) {
        if (!options.compress_boundaries)
            throw std::invalid_argument("dirichlet_regression: response has 0 or 1 values and compression is off");
        const double nn = static_cast<double>(n);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < d; ++j)
                y(i, j) = (y(i, j) * (nn - 1) + 1 / static_cast<double>(d)) / nn;
        fit.compressed = true;
    }
    Problem pr{mean_design, precision_design, Matrix(n, d, 0.0), options.base};
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < d; ++j)
            pr.log_y(i, j) = std::log(y(i, j));
    const std::size_t p = pr.p(), q = pr.q(), k = pr.size();
    fit.parameters = k;

    // Starting values: least squares of the alr logs, and the method-of-moments precision.
    std::vector<double> theta(k, 0.0);
    for (std::size_t j = 0; j < d; ++j) {
        if (j == options.base)
            continue;
        std::vector<double> r(n);
        for (std::size_t i = 0; i < n; ++i)
            r[i] = pr.log_y(i, j) - pr.log_y(i, options.base);
        const auto b = least_squares(mean_design, r);
        std::copy(b.begin(), b.end(), theta.begin() + static_cast<std::ptrdiff_t>(pr.block(j) * p));
    }
    {
        std::vector<double> mu;
        double phi_sum = 0;
        for (std::size_t j = 0; j < d; ++j) {
            double m = 0, v = 0;
            for (std::size_t i = 0; i < n; ++i)
                m += y(i, j) / static_cast<double>(n);
            for (std::size_t i = 0; i < n; ++i)
                v += (y(i, j) - m) * (y(i, j) - m) / static_cast<double>(std::max<std::size_t>(n - 1, 1));
            phi_sum += v > 0 ? std::max(m * (1 - m) / v - 1, 0.5) : 10.0;
        }
        const auto g = least_squares(precision_design, std::vector<double>(n, std::log(phi_sum / static_cast<double>(d))));
        std::copy(g.begin(), g.end(), theta.begin() + static_cast<std::ptrdiff_t>(p * (d - 1)));
        if (!std::isfinite(evaluate(pr, theta, nullptr, nullptr)))
            std::fill(theta.begin(), theta.end(), 0.0);
    }

    // Fisher scoring with step halving.
    std::vector<double> score;
    Matrix info;
    double ll = evaluate(pr, theta, &score, &info);
    for (fit.iterations = 0; fit.iterations < options.max_iterations; ++fit.iterations) {
        const auto chol = linalg::cholesky(info);
        if (!chol.ok)
            break;
        const auto step = chol.solve(score);
        double t = 1, next_ll = -std::numeric_limits<double>::infinity();
        std::vector<double> next(k);
        for (int h = 0; h < 50; ++h, t /= 2) {
            for (std::size_t r = 0; r < k; ++r)
                next[r] = theta[r] + t * step[r];
            next_ll = evaluate(pr, next, nullptr, nullptr);
            if (next_ll >= ll - 1e-12 * std::abs(ll))
                break;
        }
        if (!std::isfinite(next_ll) || next_ll < ll - 1e-12 * std::abs(ll))
            break;
        double largest = 0;
        for (std::size_t r = 0; r < k; ++r)
            largest = std::max(largest, std::abs(next[r] - theta[r]));
        theta = next;
        ll = evaluate(pr, theta, &score, &info);
        if (largest < options.tolerance) {
            fit.converged = true;
            ++fit.iterations;
            break;
        }
    }
    fit.log_likelihood = ll;
    fit.aic = -2 * ll + 2 * static_cast<double>(k);
    fit.bic = -2 * ll + std::log(static_cast<double>(n)) * static_cast<double>(k);

    // Observed information by central differences of the analytic score; each step moves its linear predictor
    // by about 1e-5 whatever the scale of the design column.
    Matrix observed(k, k, 0.0);
    for (std::size_t c = 0; c < k; ++c) {
        const Matrix& des = c < p * (d - 1) ? mean_design : precision_design;
        const std::size_t col = c < p * (d - 1) ? c % p : c - p * (d - 1);
        double scale = 0;
        for (std::size_t i = 0; i < n; ++i)
            scale = std::max(scale, std::abs(des(i, col)));
        const double h = 1e-5 * std::max(std::abs(theta[c]), 1 / std::max(scale, 1e-300));
        auto a = theta, b = theta;
        a[c] += h, b[c] -= h;
        std::vector<double> sa, sb;
        evaluate(pr, a, &sa, nullptr);
        evaluate(pr, b, &sb, nullptr);
        for (std::size_t r = 0; r < k; ++r)
            observed(r, c) = -(sa[r] - sb[r]) / (2 * h);
    }
    for (std::size_t r = 0; r < k; ++r)
        for (std::size_t c = r + 1; c < k; ++c)
            observed(r, c) = observed(c, r) = (observed(r, c) + observed(c, r)) / 2;
    fit.covariance = spd_inverse(observed);

    fit.mean_coefficients = Matrix(p, d, 0.0);
    fit.mean_standard_errors = Matrix(p, d, nan);
    for (std::size_t j = 0; j < d; ++j) {
        if (j == options.base)
            continue;
        for (std::size_t c = 0; c < p; ++c) {
            const std::size_t r = pr.block(j) * p + c;
            fit.mean_coefficients(c, j) = theta[r];
            if (fit.covariance.rows() == k)
                fit.mean_standard_errors(c, j) = std::sqrt(fit.covariance(r, r));
        }
    }
    fit.precision_coefficients.assign(theta.begin() + static_cast<std::ptrdiff_t>(p * (d - 1)), theta.end());
    fit.precision_standard_errors.assign(q, nan);
    if (fit.covariance.rows() == k)
        for (std::size_t c = 0; c < q; ++c)
            fit.precision_standard_errors[c] = std::sqrt(fit.covariance(p * (d - 1) + c, p * (d - 1) + c));
    fit.fitted = predictions(pr, theta);
    fit.standardized_residuals = Matrix(n, d, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < d; ++j) {
            const double m = fit.fitted.mean(i, j);
            fit.standardized_residuals(i, j) = (y(i, j) - m) / std::sqrt(m * (1 - m) / (1 + fit.fitted.precision[i]));
        }
    return fit;
}

} // namespace datamunge::stats
