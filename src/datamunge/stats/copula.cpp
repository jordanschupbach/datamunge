/// @file copula.cpp
/// @brief Elliptical copula densities, C-vine correlation parametrization, L-BFGS fitting and simulation.
#include <datamunge/stats/copula.hpp>

#include <datamunge/linalg/cholesky.hpp>
#include <datamunge/linalg/eigen.hpp>
#include <datamunge/optim/lbfgs.hpp>
#include <datamunge/random/distributions.hpp>
#include <datamunge/random/random.hpp>
#include <datamunge/stats/correlation_test.hpp>

#include <algorithm>
#include <cmath>
#include <numbers>
#include <numeric>
#include <stdexcept>

namespace datamunge::stats {

/// @brief Implementation helpers for copulas.
namespace {
/// @brief Dense matrix alias.
using Matrix = linalg::DenseMatrix<double>;

/// @brief Smallest t degrees of freedom considered when estimating nu.
constexpr double min_df = 0.5;

/// @brief Largest t degrees of freedom considered when estimating nu (beyond it the t copula is
/// indistinguishable from the Gaussian for practical sample sizes).
constexpr double max_df = 1000.0;

/// @brief Inverse and log-determinant of a correlation matrix.
struct Inverse {
    Matrix inverse;     ///< \f$R^{-1}\f$.
    double log_det{};   ///< \f$\log|R|\f$.
    bool ok{};          ///< Whether R was positive definite.
};

/// @brief Inverts a correlation matrix by Cholesky.
/// @param r Matrix.
/// @return Inverse and log-determinant.
Inverse invert(const Matrix& r) {
    const std::size_t d = r.rows();
    Inverse out;
    const auto chol = linalg::cholesky(r);
    if (!chol.ok)
        return out;
    out.inverse = Matrix(d, d, 0.0);
    for (std::size_t i = 0; i < d; ++i) {
        out.log_det += 2 * std::log(chol.L(i, i));
        std::vector<double> e(d, 0.0);
        e[i] = 1;
        const auto col = chol.solve(e);
        for (std::size_t k = 0; k < d; ++k)
            out.inverse(k, i) = col[k];
    }
    out.ok = true;
    return out;
}

/// @brief Log density of the univariate t distribution.
/// @param x Point.
/// @param nu Degrees of freedom.
/// @return Log density.
double log_t1(double x, double nu) {
    return std::lgamma((nu + 1) / 2) - std::lgamma(nu / 2) - 0.5 * std::log(nu * std::numbers::pi) -
           (nu + 1) / 2 * std::log1p(x * x / nu);
}

/// @brief Copula scores: normal or t quantiles of the uniforms.
/// @param u Uniforms (n x d).
/// @param family Family.
/// @param nu Degrees of freedom.
/// @return Scores.
Matrix scores(const Matrix& u, CopulaFamily family, double nu) {
    Matrix x(u.rows(), u.cols(), 0.0);
    for (std::size_t i = 0; i < u.rows(); ++i)
        for (std::size_t j = 0; j < u.cols(); ++j)
            x(i, j) = family == CopulaFamily::gaussian ? random::normal_quantile(u(i, j))
                                                       : random::student_t_quantile(u(i, j), nu);
    return x;
}

/// @brief Total copula log-likelihood given scores.
/// @param x Scores.
/// @param inv Inverse correlation.
/// @param family Family.
/// @param nu Degrees of freedom.
/// @return Log-likelihood.
double log_likelihood(const Matrix& x, const Inverse& inv, CopulaFamily family, double nu) {
    const std::size_t n = x.rows(), d = x.cols();
    const double dd = static_cast<double>(d);
    double ll = -0.5 * static_cast<double>(n) * inv.log_det;
    const double cst = family == CopulaFamily::gaussian
                           ? 0.0
                           : std::lgamma((nu + dd) / 2) - std::lgamma(nu / 2) - dd / 2 * std::log(nu * std::numbers::pi);
    for (std::size_t i = 0; i < n; ++i) {
        double q = 0, ss = 0, marg = 0;
        for (std::size_t a = 0; a < d; ++a) {
            ss += x(i, a) * x(i, a);
            for (std::size_t b = 0; b < d; ++b)
                q += x(i, a) * inv.inverse(a, b) * x(i, b);
            if (family == CopulaFamily::student_t)
                marg += log_t1(x(i, a), nu);
        }
        ll += family == CopulaFamily::gaussian ? -0.5 * (q - ss) : cst - (nu + dd) / 2 * std::log1p(q / nu) - marg;
    }
    return ll;
}

/// @brief Correlation matrix from C-vine partial correlations tanh(z) (any z gives a valid matrix).
/// @param z d(d-1)/2 unconstrained values, ordered (0,1), (0,2), ..., (1,2), ...
/// @param d Dimension.
/// @return Correlation matrix.
Matrix vine_to_correlation(const std::vector<double>& z, std::size_t d) {
    // Partial correlation P(k, i) of variables k and i given 0..k-1; build the Cholesky factor row by row.
    Matrix p(d, d, 0.0), l(d, d, 0.0);
    std::size_t at = 0;
    for (std::size_t k = 0; k < d; ++k)
        for (std::size_t i = k + 1; i < d; ++i)
            p(k, i) = std::tanh(z[at++]);
    l(0, 0) = 1;
    for (std::size_t i = 1; i < d; ++i) {
        double rem = 1;
        for (std::size_t k = 0; k < i; ++k) {
            l(i, k) = p(k, i) * std::sqrt(rem);
            rem *= 1 - p(k, i) * p(k, i);
        }
        l(i, i) = std::sqrt(rem);
    }
    Matrix r(d, d, 0.0);
    for (std::size_t i = 0; i < d; ++i)
        for (std::size_t j = 0; j < d; ++j) {
            double s = 0;
            for (std::size_t k = 0; k <= std::min(i, j); ++k)
                s += l(i, k) * l(j, k);
            r(i, j) = s;
        }
    for (std::size_t i = 0; i < d; ++i)
        r(i, i) = 1;
    return r;
}

/// @brief Inverse of vine_to_correlation() for a positive-definite correlation matrix.
/// @param r Correlation matrix.
/// @return Unconstrained values.
std::vector<double> correlation_to_vine(const Matrix& r) {
    const std::size_t d = r.rows();
    const auto chol = linalg::cholesky(r);
    std::vector<double> z;
    for (std::size_t k = 0; k < d; ++k)
        for (std::size_t i = k + 1; i < d; ++i) {
            double rem = 1;
            for (std::size_t m = 0; m < k; ++m)
                rem -= chol.L(i, m) * chol.L(i, m);
            const double pc = std::clamp(chol.L(i, k) / std::sqrt(rem), -0.999999, 0.999999);
            z.push_back(std::atanh(pc));
        }
    return z;
}

/// @brief Negative mean log-likelihood in (vine z..., log nu) coordinates with central-difference gradients.
class Objective final : public optim::DifferentiableFunction {
public:
    /// @brief Stores the problem.
    /// @param u Uniforms.
    /// @param family Family.
    /// @param fit_r Whether R is free (else fixed at `r`).
    /// @param r Fixed correlation (when !fit_r).
    /// @param nu Fixed nu (NaN when free, t only).
    Objective(const Matrix& u, CopulaFamily family, bool fit_r, Matrix r, double nu)
        : u_(u), family_(family), fit_r_(fit_r), r_(std::move(r)), nu_(nu) {
        if (family_ == CopulaFamily::gaussian || !std::isnan(nu_))
            x_ = scores(u_, family_, family_ == CopulaFamily::gaussian ? 0.0 : nu_);
    }

    /// @brief Splits coordinates into (R, nu).
    /// @param c Coordinates.
    /// @param r Output correlation.
    /// @param nu Output degrees of freedom.
    void split(const std::vector<double>& c, Matrix& r, double& nu) const {
        const std::size_t d = u_.cols(), m = fit_r_ ? d * (d - 1) / 2 : 0;
        r = fit_r_ ? vine_to_correlation(std::vector<double>(c.begin(), c.begin() + static_cast<std::ptrdiff_t>(m)), d)
                   : r_;
        // nu = exp(c) confined to [min_df, max_df]; flat outside so the optimizer stops at a bound.
        nu = family_ == CopulaFamily::gaussian ? 0.0
             : std::isnan(nu_)                 ? std::exp(std::clamp(c[m], std::log(min_df), std::log(max_df)))
                                               : nu_;
    }

    /// @brief Negative mean log-likelihood.
    /// @param c Coordinates.
    /// @return Objective.
    double evaluate(const std::vector<double>& c) override {
        if (!std::ranges::all_of(c, [](double v) { return std::isfinite(v); }))
            return 1e300;
        Matrix r;
        double nu = 0;
        split(c, r, nu);
        const auto inv = invert(r);
        if (!inv.ok)
            return 1e300;
        const Matrix& x = (family_ == CopulaFamily::student_t && std::isnan(nu_)) ? cached(nu) : x_;
        const double v = -log_likelihood(x, inv, family_, nu) / static_cast<double>(u_.rows());
        if (v < best_value)
            best_value = v, best = c;
        return v;
    }

    double best_value{INFINITY}; ///< Smallest objective evaluated.
    std::vector<double> best;    ///< Coordinates of best_value.

    /// @brief Central-difference gradient.
    /// @param c Coordinates.
    /// @return Gradient.
    std::vector<double> gradient(const std::vector<double>& c) override {
        std::vector<double> g(c.size());
        for (std::size_t i = 0; i < c.size(); ++i) {
            const double h = 1e-6 * std::max(1.0, std::abs(c[i]));
            auto p = c, m = c;
            p[i] += h, m[i] -= h;
            g[i] = (evaluate(p) - evaluate(m)) / (2 * h);
        }
        return g;
    }

private:
    /// @brief t scores for a given nu, recomputed only when nu changes.
    /// @param nu Degrees of freedom.
    /// @return Scores.
    const Matrix& cached(double nu) {
        if (nu != cached_nu_) {
            x_ = scores(u_, family_, nu);
            cached_nu_ = nu;
        }
        return x_;
    }

    const Matrix& u_;       ///< Uniforms.
    CopulaFamily family_;   ///< Family.
    bool fit_r_;            ///< Whether R is free.
    Matrix r_;              ///< Fixed R.
    double nu_;             ///< Fixed nu or NaN.
    Matrix x_;              ///< Scores.
    double cached_nu_{std::numeric_limits<double>::quiet_NaN()}; ///< nu of the cached scores.
};

/// @brief Validates a correlation matrix and degrees of freedom.
/// @param r Matrix.
/// @param family Family.
/// @param df Degrees of freedom.
/// @param fn Caller name.
void validate(const Matrix& r, CopulaFamily family, double df, const char* fn) {
    if (r.rows() != r.cols() || r.rows() < 1)
        throw std::invalid_argument(std::string(fn) + ": correlation must be square");
    if (family == CopulaFamily::student_t && !(df > 0))
        throw std::invalid_argument(std::string(fn) + ": df must be positive");
}
} // namespace

linalg::DenseMatrix<double> pseudo_observations(const linalg::DenseMatrix<double>& x) {
    const std::size_t n = x.rows(), d = x.cols();
    if (n < 2)
        throw std::invalid_argument("pseudo_observations: need at least two rows");
    Matrix u(n, d, 0.0);
    std::vector<std::size_t> idx(n);
    for (std::size_t j = 0; j < d; ++j) {
        for (std::size_t i = 0; i < n; ++i)
            if (!std::isfinite(x(i, j)))
                throw std::invalid_argument("pseudo_observations: values must be finite");
        std::iota(idx.begin(), idx.end(), std::size_t{0});
        std::ranges::stable_sort(idx, [&](std::size_t a, std::size_t b) { return x(a, j) < x(b, j); });
        for (std::size_t i = 0; i < n;) {
            std::size_t k = i;
            while (k + 1 < n && x(idx[k + 1], j) == x(idx[i], j))
                ++k;
            const double rank = (static_cast<double>(i) + static_cast<double>(k)) / 2 + 1;
            for (std::size_t m = i; m <= k; ++m)
                u(idx[m], j) = rank / (static_cast<double>(n) + 1);
            i = k + 1;
        }
    }
    return u;
}

double copula_log_density(const std::vector<double>& u, const linalg::DenseMatrix<double>& correlation,
                          CopulaFamily family, double df) {
    validate(correlation, family, df, "copula_log_density");
    if (u.size() != correlation.rows())
        throw std::invalid_argument("copula_log_density: point and matrix dimensions differ");
    Matrix m(1, u.size(), 0.0);
    for (std::size_t j = 0; j < u.size(); ++j) {
        if (!(u[j] > 0 && u[j] < 1))
            throw std::invalid_argument("copula_log_density: u must lie in (0, 1)");
        m(0, j) = u[j];
    }
    const auto inv = invert(correlation);
    if (!inv.ok)
        throw std::invalid_argument("copula_log_density: correlation must be positive definite");
    return log_likelihood(scores(m, family, df), inv, family, df);
}

CopulaFit fit_copula(const linalg::DenseMatrix<double>& u, const CopulaOptions& options) {
    const std::size_t n = u.rows(), d = u.cols();
    if (d < 2 || n <= d)
        throw std::invalid_argument("fit_copula: need d >= 2 columns and n > d rows");
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < d; ++j)
            if (!(u(i, j) > 0 && u(i, j) < 1))
                throw std::invalid_argument("fit_copula: values must lie in (0, 1)");
    const bool t = options.family == CopulaFamily::student_t;
    if (t && !std::isnan(options.df) && !(options.df > 0))
        throw std::invalid_argument("fit_copula: fixed df must be positive");
    CopulaFit fit;
    fit.family = options.family, fit.n = n;
    // Kendall inversion: the estimate (method of moments) and the MPL starting value.
    Matrix r(d, d, 1.0);
    for (std::size_t a = 0; a < d; ++a)
        for (std::size_t b = a + 1; b < d; ++b) {
            std::vector<double> x(n), y(n);
            for (std::size_t i = 0; i < n; ++i)
                x[i] = u(i, a), y[i] = u(i, b);
            r(a, b) = r(b, a) = std::sin(std::numbers::pi / 2 * kendall_tau(x, y).tau_b);
        }
    const auto eig = linalg::jacobi_eigen(r);
    if (eig.eigenvalues.back() < 1e-8) {
        // Clip eigenvalues and rescale to a unit diagonal.
        Matrix s(d, d, 0.0);
        for (std::size_t a = 0; a < d; ++a)
            for (std::size_t b = 0; b < d; ++b)
                for (std::size_t k = 0; k < d; ++k)
                    s(a, b) += eig.eigenvectors(a, k) * std::max(eig.eigenvalues[k], 1e-6) * eig.eigenvectors(b, k);
        for (std::size_t a = 0; a < d; ++a)
            for (std::size_t b = 0; b < d; ++b)
                r(a, b) = a == b ? 1.0 : s(a, b) / std::sqrt(s(a, a) * s(b, b));
        fit.projected = true;
    }
    const bool fit_r = options.method == CopulaMethod::maximum_pseudo_likelihood;
    const bool fit_nu = t && std::isnan(options.df);
    double nu = t ? (fit_nu ? 5.0 : options.df) : std::numeric_limits<double>::infinity();
    if (fit_r || fit_nu) {
        Objective obj(u, options.family, fit_r, r, fit_nu ? std::numeric_limits<double>::quiet_NaN() : nu);
        std::vector<double> c = fit_r ? correlation_to_vine(r) : std::vector<double>{};
        if (fit_nu) {
            // Start nu from a coarse profile over a grid.
            double best = INFINITY;
            for (const double g : {2.0, 3.0, 4.0, 6.0, 10.0, 20.0, 50.0}) {
                auto trial = c;
                trial.push_back(std::log(g));
                const double v = obj.evaluate(trial);
                if (v < best)
                    best = v, nu = g;
            }
            c.push_back(std::log(nu));
        }
        optim::LBFGSOptions controls;
        controls.max_iterations = options.max_iterations;
        controls.tolerance = options.tolerance;
        optim::LBFGS(controls).optimize(obj, c);
        const double final_value = obj.evaluate(c);
        obj.split(final_value <= obj.best_value ? c : obj.best, r, nu);
        if (!t)
            nu = std::numeric_limits<double>::infinity();
    }
    fit.correlation = r, fit.df = nu;
    const auto inv = invert(r);
    if (!inv.ok)
        throw std::invalid_argument("fit_copula: estimated correlation is not positive definite");
    fit.log_likelihood = log_likelihood(scores(u, options.family, t ? nu : 0.0), inv, options.family, nu);
    fit.parameters = d * (d - 1) / 2 + (fit_nu ? 1 : 0);
    const double k = static_cast<double>(fit.parameters);
    fit.aic = -2 * fit.log_likelihood + 2 * k;
    fit.bic = -2 * fit.log_likelihood + k * std::log(static_cast<double>(n));
    fit.tail_dependence = Matrix(d, d, 1.0);
    for (std::size_t a = 0; a < d; ++a)
        for (std::size_t b = 0; b < d; ++b)
            if (a != b)
                fit.tail_dependence(a, b) = t ? t_copula_tail_dependence(r(a, b), nu) : 0.0;
    return fit;
}

double t_copula_tail_dependence(double rho, double df) {
    if (!(rho > -1 && rho <= 1) || !(df > 0))
        throw std::invalid_argument("t_copula_tail_dependence: need rho in (-1, 1] and df > 0");
    if (std::isinf(df))
        return rho == 1 ? 1.0 : 0.0;
    return 2 * random::student_t_cdf(-std::sqrt((df + 1) * (1 - rho) / (1 + rho)), df + 1);
}

linalg::DenseMatrix<double> simulate_copula(const linalg::DenseMatrix<double>& correlation, CopulaFamily family,
                                            double df, std::size_t n, std::uint64_t seed) {
    validate(correlation, family, df, "simulate_copula");
    const std::size_t d = correlation.rows();
    const auto chol = linalg::cholesky(correlation);
    if (!chol.ok)
        throw std::invalid_argument("simulate_copula: correlation must be positive definite");
    random::SplitMix64 rng(seed);
    Matrix u(n, d, 0.0);
    std::vector<double> z(d);
    for (std::size_t i = 0; i < n; ++i) {
        for (auto& v : z)
            v = rng.normal();
        double w = 1;
        if (family == CopulaFamily::student_t) {
            // Chi-square(df) / df via a gamma draw (Marsaglia-Tsang, boosted for shape < 1).
            const double shape = df / 2;
            const auto gamma = [&](double a) {
                const double dd = (a < 1 ? a + 1 : a) - 1.0 / 3, c = 1 / std::sqrt(9 * dd);
                double g = 0;
                while (true) {
                    const double x = rng.normal(), v = std::pow(1 + c * x, 3);
                    if (v > 0 && std::log(rng.uniform01()) < 0.5 * x * x + dd - dd * v + dd * std::log(v)) {
                        g = dd * v;
                        break;
                    }
                }
                return a < 1 ? g * std::pow(rng.uniform01(), 1 / a) : g;
            };
            w = std::sqrt(df / (2 * gamma(shape)));
        }
        for (std::size_t j = 0; j < d; ++j) {
            double x = 0;
            for (std::size_t k = 0; k <= j; ++k)
                x += chol.L(j, k) * z[k];
            u(i, j) = family == CopulaFamily::gaussian ? random::normal_cdf(x) : random::student_t_cdf(w * x, df);
        }
    }
    return u;
}

} // namespace datamunge::stats
