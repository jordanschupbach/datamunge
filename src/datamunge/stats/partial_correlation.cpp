/// @file partial_correlation.cpp
/// @brief Residual-based partial and semipartial correlations, matrix forms, tests and intervals.
#include <datamunge/stats/partial_correlation.hpp>

#include <datamunge/linalg/cholesky.hpp>
#include <datamunge/random/distributions.hpp>
#include <datamunge/random/random.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

namespace datamunge::stats {

/// @brief Implementation helpers for partial correlations.
namespace {
/// @brief Dense matrix alias.
using Matrix = linalg::DenseMatrix<double>;

/// @brief Average ranks (1-based) with ties sharing their mean rank.
/// @param v Values.
/// @return Ranks.
std::vector<double> ranks(const std::vector<double>& v) {
    std::vector<std::size_t> idx(v.size());
    std::iota(idx.begin(), idx.end(), std::size_t{0});
    std::ranges::stable_sort(idx, [&](std::size_t a, std::size_t b) { return v[a] < v[b]; });
    std::vector<double> r(v.size());
    for (std::size_t i = 0; i < idx.size();) {
        std::size_t j = i;
        while (j + 1 < idx.size() && v[idx[j + 1]] == v[idx[i]])
            ++j;
        const double mean = (static_cast<double>(i) + static_cast<double>(j)) / 2 + 1;
        for (std::size_t k = i; k <= j; ++k)
            r[idx[k]] = mean;
        i = j + 1;
    }
    return r;
}

/// @brief Residuals of v after least-squares regression on an intercept and the columns of z (rows selected).
/// @param v Response.
/// @param z Covariates.
/// @param rows Row indices (repeats allowed, for the bootstrap).
/// @return Residuals in row order.
std::vector<double> residualize(const std::vector<double>& v, const std::vector<std::vector<double>>& z,
                                const std::vector<std::size_t>& rows) {
    const std::size_t k = z.size() + 1, n = rows.size();
    // Centre everything (absorbs the intercept), then solve the normal equations by Cholesky.
    std::vector<double> mz(z.size(), 0.0);
    double mv = 0;
    for (const std::size_t r : rows) {
        mv += v[r] / static_cast<double>(n);
        for (std::size_t j = 0; j < z.size(); ++j)
            mz[j] += z[j][r] / static_cast<double>(n);
    }
    std::vector<double> res(n);
    for (std::size_t i = 0; i < n; ++i)
        res[i] = v[rows[i]] - mv;
    if (k == 1)
        return res;
    Matrix g(z.size(), z.size(), 0.0);
    std::vector<double> b(z.size(), 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t a = 0; a < z.size(); ++a) {
            const double za = z[a][rows[i]] - mz[a];
            b[a] += za * res[i];
            for (std::size_t c = 0; c <= a; ++c)
                g(a, c) += za * (z[c][rows[i]] - mz[c]);
        }
    for (std::size_t a = 0; a < z.size(); ++a)
        for (std::size_t c = 0; c < a; ++c)
            g(c, a) = g(a, c);
    const auto chol = linalg::cholesky(g);
    bool ok = chol.ok;
    for (std::size_t a = 0; ok && a < z.size(); ++a) // relative pivot test catches exact collinearity
        ok = chol.L(a, a) * chol.L(a, a) > 1e-10 * g(a, a);
    if (!ok)
        throw std::invalid_argument("partial correlation: covariates are collinear");
    const auto beta = chol.solve(b);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t a = 0; a < z.size(); ++a)
            res[i] -= beta[a] * (z[a][rows[i]] - mz[a]);
    return res;
}

/// @brief Pearson correlation of two (already centred or not) vectors.
/// @param a First.
/// @param b Second.
/// @return r (NaN without variation).
double pearson(const std::vector<double>& a, const std::vector<double>& b) {
    const double n = static_cast<double>(a.size());
    double ma = 0, mb = 0;
    for (std::size_t i = 0; i < a.size(); ++i)
        ma += a[i] / n, mb += b[i] / n;
    double sab = 0, saa = 0, sbb = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        sab += (a[i] - ma) * (b[i] - mb);
        saa += (a[i] - ma) * (a[i] - ma);
        sbb += (b[i] - mb) * (b[i] - mb);
    }
    return sab / std::sqrt(saa * sbb);
}

/// @brief Fills test and Fisher interval for an estimate.
/// @param r Output (estimate, n and covariates set).
/// @param confidence Level.
void finish(AdjustedCorrelation& r, double confidence) {
    const double n = static_cast<double>(r.n), k = static_cast<double>(r.covariates);
    r.df = n - 2 - k;
    const double e = std::clamp(r.estimate, -1.0, 1.0);
    r.statistic = std::abs(e) < 1 ? e * std::sqrt(r.df / (1 - e * e)) : std::copysign(INFINITY, e);
    r.p_value = std::isfinite(r.statistic) ? 2 * random::student_t_cdf(-std::abs(r.statistic), r.df) : 0.0;
    const double z = random::normal_quantile(1 - (1 - confidence) / 2) / std::sqrt(n - 3 - k);
    const double fz = std::atanh(std::clamp(e, -1 + 1e-15, 1 - 1e-15));
    r.lower = std::tanh(fz - z), r.upper = std::tanh(fz + z);
}

/// @brief Shared implementation of the partial and semipartial estimators.
/// @param x First variable.
/// @param y Second variable.
/// @param z Covariates.
/// @param o Options.
/// @param semi Whether to adjust y only.
/// @param fn Caller name.
/// @return Result.
AdjustedCorrelation adjusted(const std::vector<double>& x, const std::vector<double>& y,
                             const std::vector<std::vector<double>>& z, const PartialOptions& o, bool semi,
                             const char* fn) {
    const std::size_t n = x.size(), k = z.size();
    if (y.size() != n || std::ranges::any_of(z, [n](const auto& c) { return c.size() != n; }))
        throw std::invalid_argument(std::string(fn) + ": all variables need the same length");
    if (n <= k + 3)
        throw std::invalid_argument(std::string(fn) + ": need more than k + 3 observations");
    if (!(o.confidence > 0 && o.confidence < 1))
        throw std::invalid_argument(std::string(fn) + ": confidence must be in (0, 1)");
    const auto finite = [](const std::vector<double>& v) {
        return std::ranges::all_of(v, [](double t) { return std::isfinite(t); });
    };
    if (!finite(x) || !finite(y) || !std::ranges::all_of(z, finite))
        throw std::invalid_argument(std::string(fn) + ": values must be finite");
    std::vector<double> xs = x, ys = y;
    std::vector<std::vector<double>> zs = z;
    if (o.method == PartialMethod::spearman) {
        xs = ranks(x), ys = ranks(y);
        for (auto& c : zs)
            c = ranks(c);
    }
    const auto estimate = [&](const std::vector<std::size_t>& rows) {
        std::vector<double> xr(rows.size());
        if (semi)
            for (std::size_t i = 0; i < rows.size(); ++i)
                xr[i] = xs[rows[i]];
        else
            xr = residualize(xs, zs, rows);
        return pearson(xr, residualize(ys, zs, rows));
    };
    std::vector<std::size_t> rows(n);
    std::iota(rows.begin(), rows.end(), std::size_t{0});
    AdjustedCorrelation r;
    r.n = n, r.covariates = k;
    // The test of y's unique contribution is the t test of its coefficient in x ~ Z + y, which is the
    // partial-correlation t test for both estimators.
    r.estimate = pearson(residualize(xs, zs, rows), residualize(ys, zs, rows));
    if (!std::isfinite(r.estimate))
        throw std::invalid_argument(std::string(fn) + ": a variable has no residual variation");
    finish(r, o.confidence);
    if (semi) {
        r.estimate = estimate(rows);
        // A Fisher-z interval is not valid for the semipartial correlation; use the bootstrap.
        r.lower = r.upper = std::numeric_limits<double>::quiet_NaN();
    }
    if (o.bootstrap > 0) {
        random::SplitMix64 rng(o.seed);
        std::vector<double> boot;
        for (std::size_t b = 0; b < o.bootstrap; ++b) {
            for (auto& v : rows)
                v = static_cast<std::size_t>(rng.next_u64() % n);
            try {
                const double e = estimate(rows);
                if (std::isfinite(e))
                    boot.push_back(e);
            } catch (const std::invalid_argument&) { // collinear resample
            }
        }
        if (!boot.empty()) {
            std::ranges::sort(boot);
            const auto q = [&](double p) {
                const double h = (static_cast<double>(boot.size()) - 1) * p;
                const auto lo = static_cast<std::size_t>(std::floor(h));
                const std::size_t hi = std::min(lo + 1, boot.size() - 1);
                return boot[lo] + (h - static_cast<double>(lo)) * (boot[hi] - boot[lo]);
            };
            const double a = (1 - o.confidence) / 2;
            r.bootstrap_lower = q(a), r.bootstrap_upper = q(1 - a);
        }
    }
    return r;
}
} // namespace

AdjustedCorrelation partial_correlation(const std::vector<double>& x, const std::vector<double>& y,
                                        const std::vector<std::vector<double>>& covariates,
                                        const PartialOptions& options) {
    return adjusted(x, y, covariates, options, false, "partial_correlation");
}

AdjustedCorrelation semipartial_correlation(const std::vector<double>& x, const std::vector<double>& y,
                                            const std::vector<std::vector<double>>& covariates,
                                            const PartialOptions& options) {
    return adjusted(x, y, covariates, options, true, "semipartial_correlation");
}

PartialCorrelationMatrix partial_correlation_matrix(const linalg::DenseMatrix<double>& covariance, std::size_t n) {
    const std::size_t p = covariance.rows();
    if (covariance.cols() != p || p < 2 || n <= p)
        throw std::invalid_argument("partial_correlation_matrix: need a p x p matrix (p >= 2) and n > p");
    const auto chol = linalg::cholesky(covariance);
    if (!chol.ok)
        throw std::invalid_argument("partial_correlation_matrix: matrix must be positive definite");
    Matrix inv(p, p, 0.0);
    for (std::size_t i = 0; i < p; ++i) {
        std::vector<double> e(p, 0.0);
        e[i] = 1;
        const auto col = chol.solve(e);
        for (std::size_t k = 0; k < p; ++k)
            inv(k, i) = col[k];
    }
    PartialCorrelationMatrix m;
    m.n = n;
    m.partial = Matrix(p, p, 1.0);
    m.semipartial = Matrix(p, p, 1.0);
    m.partial_p_value = Matrix(p, p, 0.0);
    m.semipartial_p_value = Matrix(p, p, 0.0);
    const double df = static_cast<double>(n) - static_cast<double>(p);
    const auto p_of = [df](double r) {
        const double t = r * std::sqrt(df / (1 - r * r));
        return 2 * random::student_t_cdf(-std::abs(t), df);
    };
    for (std::size_t i = 0; i < p; ++i)
        for (std::size_t j = 0; j < p; ++j) {
            if (i == j)
                continue;
            const double pc = -inv(i, j) / std::sqrt(inv(i, i) * inv(j, j));
            // ppcor::spcor: x_i with x_j residualized on the other variables.
            const double sp = -inv(i, j) / std::sqrt(inv(i, i) * inv(j, j)) / std::sqrt(covariance(i, i)) /
                              std::sqrt(std::abs(inv(i, i) - inv(i, j) * inv(i, j) / inv(j, j)));
            m.partial(i, j) = pc, m.semipartial(i, j) = sp;
            // Both are tested by the coefficient t test, i.e. the partial-correlation test.
            m.partial_p_value(i, j) = m.semipartial_p_value(i, j) = p_of(pc);
        }
    return m;
}

} // namespace datamunge::stats
