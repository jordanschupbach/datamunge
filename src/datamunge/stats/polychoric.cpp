/// @file polychoric.cpp
/// @brief Genz bivariate normal CDF, cell probabilities with analytic derivatives, Fisher scoring and tests.
#include <datamunge/stats/polychoric.hpp>

#include <datamunge/linalg/cholesky.hpp>
#include <datamunge/random/distributions.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <numbers>
#include <stdexcept>

namespace datamunge::stats {

/// @brief Implementation helpers for polychoric correlations.
namespace {
/// @brief Dense matrix alias.
using Matrix = linalg::DenseMatrix<double>;

/// @brief Positive infinity.
constexpr double inf = std::numeric_limits<double>::infinity();

/// @brief Largest admissible |rho| during fitting (as `polycor`'s maxcor).
constexpr double max_rho = 0.9999;

/// @brief Standard normal distribution function (exact at infinities).
/// @param x Argument.
/// @return \f$\Phi(x)\f$.
double phi_cdf(double x) { return 0.5 * std::erfc(-x / std::numbers::sqrt2); }

/// @brief Standard normal density (0 at infinities).
/// @param x Argument.
/// @return \f$\phi(x)\f$.
double phi_pdf(double x) { return std::isinf(x) ? 0.0 : std::exp(-0.5 * x * x) / std::sqrt(2 * std::numbers::pi); }

/// @brief Genz's BVNU: \f$P(X>h,Y>k)\f$ for a standard bivariate normal with correlation r.
/// @param h Lower limit for X.
/// @param k Lower limit for Y.
/// @param r Correlation.
/// @return Upper-orthant probability.
double bvnu(double h, double k, double r) {
    if (h == inf || k == inf)
        return 0.0;
    if (h == -inf)
        return k == -inf ? 1.0 : phi_cdf(-k);
    if (k == -inf)
        return phi_cdf(-h);
    if (r == 0)
        return phi_cdf(-h) * phi_cdf(-k);
    static constexpr std::array<double, 3> w6{0.1713244923791705, 0.3607615730481384, 0.4679139345726904};
    static constexpr std::array<double, 3> x6{0.9324695142031522, 0.6612093864662647, 0.2386191860831970};
    static constexpr std::array<double, 6> w12{.04717533638651177, 0.1069393259953183, 0.1600783285433464,
                                               0.2031674267230659,  0.2334925365383547, 0.2491470458134029};
    static constexpr std::array<double, 6> x12{0.9815606342467191, 0.9041172563704750, 0.7699026741943050,
                                               0.5873179542866171, 0.3678314989981802, 0.1252334085114692};
    static constexpr std::array<double, 10> w20{.01761400713915212, .04060142980038694, .06267204833410906,
                                                .08327674157670475, 0.1019301198172404, 0.1181945319615184,
                                                0.1316886384491766, 0.1420961093183821, 0.1491729864726037,
                                                0.1527533871307259};
    static constexpr std::array<double, 10> x20{0.9931285991850949, 0.9639719272779138, 0.9122344282513259,
                                                0.8391169718222188, 0.7463319064601508, 0.6360536807265150,
                                                0.5108670019508271, 0.3737060887154196, 0.2277858511416451,
                                                0.07652652113349733};
    const double* w;
    const double* x;
    std::size_t m;
    if (std::abs(r) < 0.3)
        w = w6.data(), x = x6.data(), m = 3;
    else if (std::abs(r) < 0.75)
        w = w12.data(), x = x12.data(), m = 6;
    else
        w = w20.data(), x = x20.data(), m = 10;
    const double tp = 2 * std::numbers::pi;
    double hk = h * k, bvn = 0;
    if (std::abs(r) < 0.925) {
        const double hs = (h * h + k * k) / 2, asr = std::asin(r) / 2;
        for (std::size_t i = 0; i < m; ++i)
            for (const double s : {-1.0, 1.0}) {
                const double sn = std::sin(asr * (1 + s * x[i]));
                bvn += w[i] * std::exp((sn * hk - hs) / (1 - sn * sn));
            }
        return std::clamp(bvn * asr / tp + phi_cdf(-h) * phi_cdf(-k), 0.0, 1.0);
    }
    // |r| >= 0.925: TVPACK BVND expansion around the degenerate (r = +-1) case.
    if (r < 0)
        k = -k, hk = -hk;
    if (std::abs(r) < 1) {
        const double as = (1 - r) * (1 + r);
        double a = std::sqrt(as);
        const double bs = (h - k) * (h - k);
        const double c = (4 - hk) / 8, d = (12 - hk) / 16;
        bvn = a * std::exp(-(bs / as + hk) / 2) * (1 - c * (bs - as) * (1 - d * bs / 5) / 3 + c * d * as * as / 5);
        if (hk > -160) {
            const double b = std::sqrt(bs);
            bvn -= std::exp(-hk / 2) * std::sqrt(tp) * phi_cdf(-b / a) * b * (1 - c * bs * (1 - d * bs / 5) / 3);
        }
        a /= 2;
        for (std::size_t i = 0; i < m; ++i)
            for (const double s : {-1.0, 1.0}) {
                const double xs = std::pow(a * (1 + s * x[i]), 2), rs = std::sqrt(1 - xs);
                bvn += a * w[i] * std::exp(-(bs / xs + hk) / 2) *
                       (std::exp(-hk * (1 - rs) / (2 * (1 + rs))) / rs - (1 + c * xs * (1 + d * xs)));
            }
        bvn = -bvn / tp;
    }
    if (r > 0)
        bvn += phi_cdf(-std::max(h, k));
    else
        bvn = -bvn + std::max(0.0, phi_cdf(-h) - phi_cdf(-k));
    return std::clamp(bvn, 0.0, 1.0);
}

/// @brief Bivariate normal density \f$\phi_2(h,k;\rho)\f$ (0 if either argument is infinite).
/// @param h First coordinate.
/// @param k Second coordinate.
/// @param rho Correlation.
/// @return Density.
double bvn_pdf(double h, double k, double rho) {
    if (std::isinf(h) || std::isinf(k))
        return 0.0;
    const double s = 1 - rho * rho;
    return std::exp(-(h * h - 2 * rho * h * k + k * k) / (2 * s)) / (2 * std::numbers::pi * std::sqrt(s));
}

/// @brief \f$\partial F(h,k)/\partial h=\phi(h)\Phi((k-\rho h)/\sqrt{1-\rho^2})\f$.
/// @param h Differentiated limit.
/// @param k Other limit.
/// @param rho Correlation.
/// @return Partial derivative.
double bvn_dh(double h, double k, double rho) {
    if (std::isinf(h) || k == -inf)
        return 0.0;
    if (k == inf)
        return phi_pdf(h);
    return phi_pdf(h) * phi_cdf((k - rho * h) / std::sqrt(1 - rho * rho));
}

/// @brief A cleaned table and its likelihood machinery.
struct Model {
    std::vector<std::vector<double>> n; ///< Counts (r x c, no empty margins).
    std::size_t r{};                    ///< Rows.
    std::size_t c{};                    ///< Columns.
    double total{};                     ///< Total count.

    /// @brief Cell probabilities and derivatives w.r.t. (rho, row thresholds, column thresholds).
    /// @param rho Correlation.
    /// @param a Row thresholds (r - 1).
    /// @param b Column thresholds (c - 1).
    /// @param p Output probabilities (row-major).
    /// @param dp Output derivatives (row-major cells, each with 1 + (r - 1) + (c - 1) entries).
    void cells(double rho, const std::vector<double>& a, const std::vector<double>& b, std::vector<double>& p,
               std::vector<double>& dp) const {
        const std::size_t q = r + c - 1;
        std::vector<double> ea{-inf}, eb{-inf};
        ea.insert(ea.end(), a.begin(), a.end()), ea.push_back(inf);
        eb.insert(eb.end(), b.begin(), b.end()), eb.push_back(inf);
        // Corner values F, dF/drho, dF/dh, dF/dk on the (r + 1) x (c + 1) grid.
        std::vector<double> f((r + 1) * (c + 1)), fr(f.size()), fh(f.size()), fk(f.size());
        for (std::size_t i = 0; i <= r; ++i)
            for (std::size_t j = 0; j <= c; ++j) {
                const std::size_t g = i * (c + 1) + j;
                f[g] = bivariate_normal_cdf(ea[i], eb[j], rho);
                fr[g] = bvn_pdf(ea[i], eb[j], rho);
                fh[g] = bvn_dh(ea[i], eb[j], rho);
                fk[g] = bvn_dh(eb[j], ea[i], rho);
            }
        p.assign(r * c, 0.0);
        dp.assign(r * c * q, 0.0);
        for (std::size_t i = 0; i < r; ++i)
            for (std::size_t j = 0; j < c; ++j) {
                const std::size_t g00 = i * (c + 1) + j, g01 = g00 + 1, g10 = g00 + c + 1, g11 = g10 + 1;
                const std::size_t cell = i * c + j;
                p[cell] = std::max(f[g11] - f[g01] - f[g10] + f[g00], 1e-300);
                double* d = &dp[cell * q];
                d[0] = fr[g11] - fr[g01] - fr[g10] + fr[g00];
                if (i + 1 < r) // upper row limit is threshold a_i (index 1 + i)
                    d[1 + i] += fh[g11] - fh[g10];
                if (i > 0) // lower row limit is threshold a_{i-1}
                    d[i] -= fh[g01] - fh[g00];
                if (j + 1 < c)
                    d[r + j] += fk[g11] - fk[g01];
                if (j > 0)
                    d[r + j - 1] -= fk[g10] - fk[g00];
            }
    }

    /// @brief Log-likelihood, score and expected information for the first `q` parameters.
    /// @param theta Parameters (rho, row thresholds, column thresholds).
    /// @param q Number of leading parameters free (1: two-step).
    /// @param score Output score.
    /// @param info Output expected (Fisher) information, q x q.
    /// @return Log-likelihood.
    double evaluate(const std::vector<double>& theta, std::size_t q, std::vector<double>& score,
                    Matrix& info) const {
        const std::vector<double> a(theta.begin() + 1, theta.begin() + static_cast<std::ptrdiff_t>(r));
        const std::vector<double> b(theta.begin() + static_cast<std::ptrdiff_t>(r), theta.end());
        std::vector<double> p, dp;
        cells(theta[0], a, b, p, dp);
        const std::size_t all = r + c - 1;
        double ll = 0;
        score.assign(q, 0.0);
        info = Matrix(q, q, 0.0);
        for (std::size_t cell = 0; cell < r * c; ++cell) {
            const double cnt = n[cell / c][cell % c];
            if (cnt > 0)
                ll += cnt * std::log(p[cell]);
            const double* d = &dp[cell * all];
            for (std::size_t u = 0; u < q; ++u) {
                score[u] += cnt * d[u] / p[cell];
                for (std::size_t v = 0; v < q; ++v)
                    info(u, v) += total * d[u] * d[v] / p[cell];
            }
        }
        return ll;
    }

    /// @brief Whether thresholds are strictly increasing and |rho| admissible.
    /// @param theta Parameters.
    /// @return Validity.
    bool admissible(const std::vector<double>& theta) const {
        if (!(std::abs(theta[0]) <= max_rho))
            return false;
        for (std::size_t i = 2; i < r; ++i)
            if (!(theta[i] > theta[i - 1]))
                return false;
        for (std::size_t j = r + 1; j < r + c - 1; ++j)
            if (!(theta[j] > theta[j - 1]))
                return false;
        return true;
    }
};

/// @brief Solves a small symmetric positive-definite system.
/// @param m Matrix.
/// @param v Right-hand side.
/// @return Solution (empty if not positive definite).
std::vector<double> spd_solve(const Matrix& m, const std::vector<double>& v) {
    const auto chol = linalg::cholesky(m);
    return chol.ok ? chol.solve(v) : std::vector<double>{};
}

/// @brief Fisher scoring with step halving on the first q parameters.
/// @param model Model.
/// @param theta Parameters (updated).
/// @param q Free parameters.
/// @param o Options.
/// @param it Output iterations.
/// @return Whether converged.
bool fit(const Model& model, std::vector<double>& theta, std::size_t q, const PolychoricOptions& o,
         std::size_t& it) {
    std::vector<double> score;
    Matrix info;
    double ll = model.evaluate(theta, q, score, info);
    for (it = 0; it < o.max_iterations; ++it) {
        const auto step = spd_solve(info, score);
        if (step.empty())
            return false;
        double size = 0;
        for (const double s : step)
            size = std::max(size, std::abs(s));
        if (size < o.tolerance)
            return true;
        double t = 1;
        bool moved = false;
        for (int halve = 0; halve < 60; ++halve, t /= 2) {
            auto trial = theta;
            for (std::size_t u = 0; u < q; ++u)
                trial[u] += t * step[u];
            trial[0] = std::clamp(trial[0], -max_rho, max_rho);
            if (!model.admissible(trial))
                continue;
            std::vector<double> s2;
            Matrix i2;
            const double l2 = model.evaluate(trial, q, s2, i2);
            if (l2 >= ll - 1e-12 * std::abs(ll)) {
                theta = trial, ll = l2, score = s2, info = i2, moved = true;
                break;
            }
        }
        if (!moved)
            return std::abs(theta[0]) >= max_rho; // pinned at the boundary
    }
    return false;
}

/// @brief Observed information for the first q parameters by central differences of the analytic score.
/// @param model Model.
/// @param theta Estimate.
/// @param q Free parameters.
/// @return Negative Hessian of the log-likelihood.
Matrix observed_information(const Model& model, const std::vector<double>& theta, std::size_t q) {
    Matrix h(q, q, 0.0);
    std::vector<double> sp, sm;
    Matrix dummy;
    for (std::size_t v = 0; v < q; ++v) {
        const double e = 1e-6 * std::max(1.0, std::abs(theta[v]));
        auto tp = theta, tm = theta;
        tp[v] += e, tm[v] -= e;
        model.evaluate(tp, q, sp, dummy);
        model.evaluate(tm, q, sm, dummy);
        for (std::size_t u = 0; u < q; ++u)
            h(u, v) = -(sp[u] - sm[u]) / (2 * e);
    }
    for (std::size_t u = 0; u < q; ++u)
        for (std::size_t v = 0; v < u; ++v)
            h(u, v) = h(v, u) = (h(u, v) + h(v, u)) / 2;
    return h;
}
} // namespace

double bivariate_normal_cdf(double h, double k, double rho) {
    if (std::isnan(h) || std::isnan(k) || !(rho >= -1 && rho <= 1))
        throw std::invalid_argument("bivariate_normal_cdf: arguments must be numbers and |rho| <= 1");
    if (rho == 1)
        return phi_cdf(std::min(h, k));
    if (rho == -1)
        return std::max(0.0, phi_cdf(h) - phi_cdf(-k));
    return bvnu(-h, -k, rho);
}

PolychoricCorrelation polychoric_correlation(const std::vector<std::vector<double>>& table,
                                             const PolychoricOptions& options) {
    if (table.empty() || !(options.confidence > 0 && options.confidence < 1))
        throw std::invalid_argument("polychoric_correlation: need a table and confidence in (0, 1)");
    const std::size_t cols = table[0].size();
    for (const auto& row : table) {
        if (row.size() != cols)
            throw std::invalid_argument("polychoric_correlation: table rows must have equal length");
        for (const double v : row)
            if (!std::isfinite(v) || v < 0)
                throw std::invalid_argument("polychoric_correlation: counts must be finite and nonnegative");
    }
    // Drop empty rows and columns.
    std::vector<std::size_t> keep_r, keep_c;
    for (std::size_t i = 0; i < table.size(); ++i) {
        double s = 0;
        for (const double v : table[i])
            s += v;
        if (s > 0)
            keep_r.push_back(i);
    }
    for (std::size_t j = 0; j < cols; ++j) {
        double s = 0;
        for (const auto& row : table)
            s += row[j];
        if (s > 0)
            keep_c.push_back(j);
    }
    if (keep_r.size() < 2 || keep_c.size() < 2)
        throw std::invalid_argument("polychoric_correlation: need at least two nonempty rows and columns");
    Model model;
    model.r = keep_r.size(), model.c = keep_c.size();
    model.n.assign(model.r, std::vector<double>(model.c, 0.0));
    for (std::size_t i = 0; i < model.r; ++i)
        for (std::size_t j = 0; j < model.c; ++j) {
            model.n[i][j] = table[keep_r[i]][keep_c[j]];
            model.total += model.n[i][j];
        }
    const std::size_t r = model.r, c = model.c, q_all = r + c - 1;
    // Two-step thresholds from cumulative marginal proportions.
    std::vector<double> theta(q_all, 0.0);
    double cum = 0;
    for (std::size_t i = 0; i + 1 < r; ++i) {
        for (const double v : model.n[i])
            cum += v;
        theta[1 + i] = random::normal_quantile(cum / model.total);
    }
    cum = 0;
    for (std::size_t j = 0; j + 1 < c; ++j) {
        for (std::size_t i = 0; i < r; ++i)
            cum += model.n[i][j];
        theta[r + j] = random::normal_quantile(cum / model.total);
    }
    PolychoricCorrelation out;
    std::size_t it = 0;
    out.converged = fit(model, theta, 1, options, it);
    out.iterations = it;
    const bool ml = options.estimator == PolychoricEstimator::maximum_likelihood;
    if (ml) {
        out.converged = fit(model, theta, q_all, options, it);
        out.iterations += it;
    }
    const std::size_t q = ml ? q_all : 1;
    const Matrix obs = observed_information(model, theta, q);
    std::vector<double> se(q, std::numeric_limits<double>::quiet_NaN());
    for (std::size_t u = 0; u < q; ++u) {
        std::vector<double> e(q, 0.0);
        e[u] = 1;
        const auto col = spd_solve(obs, e);
        if (!col.empty())
            se[u] = std::sqrt(col[u]);
    }
    out.rho = theta[0];
    out.standard_error = se[0];
    const double z = random::normal_quantile(1 - (1 - options.confidence) / 2);
    const double fz = std::atanh(out.rho), dz = z * out.standard_error / (1 - out.rho * out.rho);
    out.lower = std::tanh(fz - dz), out.upper = std::tanh(fz + dz);
    out.row_thresholds.assign(theta.begin() + 1, theta.begin() + static_cast<std::ptrdiff_t>(r));
    out.column_thresholds.assign(theta.begin() + static_cast<std::ptrdiff_t>(r), theta.end());
    if (ml) {
        out.row_threshold_se.assign(se.begin() + 1, se.begin() + static_cast<std::ptrdiff_t>(r));
        out.column_threshold_se.assign(se.begin() + static_cast<std::ptrdiff_t>(r), se.end());
    }
    std::vector<double> score;
    Matrix info;
    out.log_likelihood = model.evaluate(theta, 1, score, info);
    double saturated = 0;
    for (const auto& row : model.n)
        for (const double v : row)
            if (v > 0)
                saturated += v * std::log(v / model.total);
    out.chi_square = std::max(0.0, 2 * (saturated - out.log_likelihood));
    out.df = static_cast<double>(r * c) - static_cast<double>(r) - static_cast<double>(c);
    out.p_value = out.df > 0 ? random::chi_squared_survival(out.chi_square, out.df)
                             : std::numeric_limits<double>::quiet_NaN();
    out.n = model.total;
    return out;
}

PolychoricCorrelation polychoric_correlation(const std::vector<int>& x, const std::vector<int>& y,
                                             const PolychoricOptions& options) {
    if (x.size() != y.size())
        throw std::invalid_argument("polychoric_correlation: x and y need the same length");
    std::map<int, std::size_t> rx, ry;
    for (const int v : x)
        rx.emplace(v, 0);
    for (const int v : y)
        ry.emplace(v, 0);
    if (rx.size() < 2 || ry.size() < 2)
        throw std::invalid_argument("polychoric_correlation: each variable needs at least two categories");
    std::size_t k = 0;
    for (auto& [v, idx] : rx)
        idx = k++;
    k = 0;
    for (auto& [v, idx] : ry)
        idx = k++;
    std::vector<std::vector<double>> table(rx.size(), std::vector<double>(ry.size(), 0.0));
    for (std::size_t i = 0; i < x.size(); ++i)
        table[rx[x[i]]][ry[y[i]]] += 1;
    return polychoric_correlation(table, options);
}

PolychoricCorrelation tetrachoric_correlation(double a, double b, double c, double d,
                                              const PolychoricOptions& options) {
    return polychoric_correlation({{a, b}, {c, d}}, options);
}

linalg::DenseMatrix<double> polychoric_matrix(const std::vector<std::vector<int>>& columns) {
    const std::size_t p = columns.size();
    if (p < 2)
        throw std::invalid_argument("polychoric_matrix: need at least two variables");
    Matrix m(p, p, 1.0);
    for (std::size_t i = 0; i < p; ++i)
        for (std::size_t j = i + 1; j < p; ++j)
            m(i, j) = m(j, i) = polychoric_correlation(columns[i], columns[j]).rho;
    return m;
}

} // namespace datamunge::stats
