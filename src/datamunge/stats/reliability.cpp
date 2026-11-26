/// @file reliability.cpp
/// @brief Cronbach's alpha, its intervals and item analysis; one-factor ML by EM; McDonald's omega.
#include <datamunge/stats/reliability.hpp>

#include <datamunge/linalg/cholesky.hpp>
#include <datamunge/random/distributions.hpp>
#include <datamunge/random/random.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

namespace datamunge::stats {

/// @brief Implementation helpers for reliability.
namespace {
/// @brief Dense matrix alias.
using Matrix = linalg::DenseMatrix<double>;

/// @brief Validates an item matrix.
/// @param x Data.
/// @param min_items Minimum number of columns.
/// @param fn Caller name.
void check_data(const Matrix& x, std::size_t min_items, const char* fn) {
    if (x.cols() < min_items || x.rows() < 3)
        throw std::invalid_argument(std::string(fn) + ": too few items or respondents");
    for (std::size_t i = 0; i < x.rows(); ++i)
        for (std::size_t j = 0; j < x.cols(); ++j)
            if (!std::isfinite(x(i, j)))
                throw std::invalid_argument(std::string(fn) + ": data must be finite");
}

/// @brief Sample covariance (n - 1 divisor) of the selected rows.
/// @param x Data.
/// @param rows Row indices (repeats allowed).
/// @return p-by-p covariance.
Matrix covariance(const Matrix& x, const std::vector<std::size_t>& rows) {
    const std::size_t p = x.cols(), n = rows.size();
    std::vector<double> mean(p, 0.0);
    for (const std::size_t r : rows)
        for (std::size_t j = 0; j < p; ++j)
            mean[j] += x(r, j) / static_cast<double>(n);
    Matrix c(p, p, 0.0);
    for (const std::size_t r : rows)
        for (std::size_t a = 0; a < p; ++a)
            for (std::size_t b = 0; b <= a; ++b)
                c(a, b) += (x(r, a) - mean[a]) * (x(r, b) - mean[b]);
    for (std::size_t a = 0; a < p; ++a)
        for (std::size_t b = 0; b <= a; ++b)
            c(b, a) = c(a, b) = c(a, b) / static_cast<double>(n - 1);
    return c;
}

/// @brief Raw alpha of a covariance matrix restricted to the items not equal to `skip`.
/// @param c Covariance.
/// @param skip Item to leave out (p for none).
/// @return Alpha.
double alpha_of(const Matrix& c, std::size_t skip) {
    const std::size_t p = c.rows();
    double total = 0, trace = 0, k = 0;
    for (std::size_t a = 0; a < p; ++a) {
        if (a == skip)
            continue;
        k += 1;
        trace += c(a, a);
        for (std::size_t b = 0; b < p; ++b)
            if (b != skip)
                total += c(a, b);
    }
    return k / (k - 1) * (1 - trace / total);
}

/// @brief Omega from a covariance matrix (fits the one-factor model).
/// @param c Covariance.
/// @param n Sample size.
/// @param fit Receives the model.
/// @return (model-implied omega, observed-variance omega).
std::pair<double, double> omega_of(const Matrix& c, std::size_t n, OneFactorFit& fit) {
    fit = one_factor_fit(c, n);
    double sl = 0, su = 0, total = 0;
    for (std::size_t j = 0; j < c.rows(); ++j) {
        sl += fit.loadings[j], su += fit.uniquenesses[j];
        for (std::size_t k = 0; k < c.rows(); ++k)
            total += c(j, k);
    }
    // Loadings are on the ML (divisor n) scale; put the observed total on the same scale.
    total *= (static_cast<double>(n) - 1) / static_cast<double>(n);
    return {sl * sl / (sl * sl + su), sl * sl / total};
}

/// @brief Empirical quantile (type 7) of a sorted sample.
/// @param s Sorted values.
/// @param q Probability.
/// @return Quantile.
double quantile_sorted(const std::vector<double>& s, double q) {
    const double h = (static_cast<double>(s.size()) - 1) * q;
    const auto lo = static_cast<std::size_t>(std::floor(h));
    const std::size_t hi = std::min(lo + 1, s.size() - 1);
    return s[lo] + (h - static_cast<double>(lo)) * (s[hi] - s[lo]);
}
} // namespace

AlphaEstimate cronbach_alpha(const linalg::DenseMatrix<double>& data, double confidence) {
    check_data(data, 2, "cronbach_alpha");
    if (!(confidence > 0 && confidence < 1))
        throw std::invalid_argument("cronbach_alpha: confidence must be in (0, 1)");
    const std::size_t n = data.rows(), p = data.cols();
    std::vector<std::size_t> rows(n);
    for (std::size_t i = 0; i < n; ++i)
        rows[i] = i;
    const Matrix c = covariance(data, rows);
    for (std::size_t j = 0; j < p; ++j)
        if (!(c(j, j) > 0))
            throw std::invalid_argument("cronbach_alpha: an item has zero variance");
    AlphaEstimate a;
    a.n = n, a.items = p;
    const double pd = static_cast<double>(p);
    a.alpha = alpha_of(c, p);
    double sum_r = 0, total = 0, trace = 0, trace_c2 = 0, sum_c2 = 0;
    for (std::size_t j = 0; j < p; ++j) {
        trace += c(j, j);
        for (std::size_t k = 0; k < p; ++k) {
            sum_r += c(j, k) / std::sqrt(c(j, j) * c(k, k));
            total += c(j, k);
            double row = 0; // (C C)_{jk}
            for (std::size_t m = 0; m < p; ++m)
                row += c(j, m) * c(m, k);
            sum_c2 += row;
            if (j == k)
                trace_c2 += row;
        }
    }
    a.standardized_alpha = pd / (pd - 1) * (1 - pd / sum_r);
    a.average_r = (sum_r - pd) / (pd * (pd - 1));
    const double q = 2 * pd * pd / ((pd - 1) * (pd - 1) * total * total * total) *
                     (total * (trace_c2 + trace * trace) - 2 * trace * sum_c2);
    a.standard_error = std::sqrt(q / static_cast<double>(n));
    const double tail = (1 - confidence) / 2, nd = static_cast<double>(n);
    const double z = random::normal_quantile(1 - tail);
    a.asymptotic_lower = a.alpha - z * a.standard_error;
    a.asymptotic_upper = a.alpha + z * a.standard_error;
    a.feldt_lower = 1 - (1 - a.alpha) * random::f_quantile(1 - tail, nd - 1, (nd - 1) * (pd - 1));
    a.feldt_upper = 1 - (1 - a.alpha) * random::f_quantile(tail, nd - 1, (nd - 1) * (pd - 1));
    for (std::size_t j = 0; j < p; ++j) {
        ItemStatistics s;
        for (std::size_t i = 0; i < n; ++i)
            s.mean += data(i, j) / nd;
        s.sd = std::sqrt(c(j, j));
        // cov(item, rest) and var(rest) from the covariance matrix.
        double cov_rest = 0, var_rest = 0;
        for (std::size_t k = 0; k < p; ++k) {
            if (k == j)
                continue;
            cov_rest += c(j, k);
            for (std::size_t m = 0; m < p; ++m)
                if (m != j)
                    var_rest += c(k, m);
        }
        s.item_rest = cov_rest / std::sqrt(c(j, j) * var_rest);
        s.alpha_if_deleted = p > 2 ? alpha_of(c, j) : std::numeric_limits<double>::quiet_NaN();
        a.item_statistics.push_back(s);
    }
    return a;
}

OneFactorFit one_factor_fit(const linalg::DenseMatrix<double>& covariance, std::size_t n,
                            std::size_t max_iterations, double tolerance) {
    const std::size_t p = covariance.rows();
    if (covariance.cols() != p || p < 3 || n < 2)
        throw std::invalid_argument("one_factor_fit: need a p x p covariance with p >= 3 and n >= 2");
    const auto chol = linalg::cholesky(covariance);
    if (!chol.ok)
        throw std::invalid_argument("one_factor_fit: covariance must be positive definite");
    // ML uses S with divisor n (as lavaan); loadings scale accordingly.
    const double scale = (static_cast<double>(n) - 1) / static_cast<double>(n);
    Matrix s(p, p);
    for (std::size_t a = 0; a < p; ++a)
        for (std::size_t b = 0; b < p; ++b)
            s(a, b) = covariance(a, b) * scale;
    // Start: principal-axis-like values from average correlations.
    std::vector<double> lambda(p), psi(p), floor(p);
    for (std::size_t j = 0; j < p; ++j) {
        double r = 0;
        for (std::size_t k = 0; k < p; ++k)
            if (k != j)
                r += std::abs(s(j, k)) / std::sqrt(s(j, j) * s(k, k));
        r = std::clamp(r / static_cast<double>(p - 1), 0.05, 0.95);
        lambda[j] = std::sqrt(r * s(j, j));
        psi[j] = (1 - r) * s(j, j);
        floor[j] = 1e-8 * s(j, j);
    }
    OneFactorFit fit;
    std::vector<double> beta(p), sb(p);
    for (fit.iterations = 1; fit.iterations <= max_iterations; ++fit.iterations) {
        // beta = lambda' Sigma^{-1} via Woodbury.
        double g = 0;
        for (std::size_t j = 0; j < p; ++j)
            g += lambda[j] * lambda[j] / psi[j];
        for (std::size_t j = 0; j < p; ++j)
            beta[j] = lambda[j] / psi[j] / (1 + g);
        double bl = 0, bsb = 0;
        for (std::size_t j = 0; j < p; ++j) {
            sb[j] = 0;
            for (std::size_t k = 0; k < p; ++k)
                sb[j] += s(j, k) * beta[k];
            bl += beta[j] * lambda[j];
        }
        for (std::size_t j = 0; j < p; ++j)
            bsb += beta[j] * sb[j];
        const double czz = 1 - bl + bsb;
        double change = 0;
        for (std::size_t j = 0; j < p; ++j) {
            const double l = sb[j] / czz;
            const double u = std::max(floor[j], s(j, j) - l * sb[j]);
            change = std::max({change, std::abs(l - lambda[j]) / std::sqrt(s(j, j)),
                               std::abs(u - psi[j]) / s(j, j)});
            lambda[j] = l, psi[j] = u;
        }
        if (change < tolerance) {
            fit.converged = true;
            break;
        }
    }
    fit.iterations = std::min(fit.iterations, max_iterations);
    double sum = 0;
    for (const double l : lambda)
        sum += l;
    if (sum < 0)
        for (double& l : lambda)
            l = -l;
    // F = log|Sigma| - log|S| + tr(S Sigma^{-1}) - p.
    double g = 0, log_det_sigma = 0, log_det_s = 0, tr = 0;
    for (std::size_t j = 0; j < p; ++j) {
        g += lambda[j] * lambda[j] / psi[j];
        log_det_sigma += std::log(psi[j]);
        log_det_s += 2 * std::log(chol.L(j, j) * std::sqrt(scale));
    }
    log_det_sigma += std::log1p(g);
    for (std::size_t a = 0; a < p; ++a)
        for (std::size_t b = 0; b < p; ++b) {
            const double inv = (a == b ? 1 / psi[a] : 0.0) -
                               lambda[a] / psi[a] * lambda[b] / psi[b] / (1 + g);
            tr += s(a, b) * inv;
        }
    const double pd = static_cast<double>(p);
    fit.chi_square = std::max(0.0, static_cast<double>(n) * (log_det_sigma - log_det_s + tr - pd));
    fit.df = pd * (pd + 1) / 2 - 2 * pd;
    fit.p_value = fit.df > 0 ? random::chi_squared_survival(fit.chi_square, fit.df)
                             : std::numeric_limits<double>::quiet_NaN();
    for (std::size_t j = 0; j < p; ++j)
        fit.heywood = fit.heywood || psi[j] <= floor[j] * (1 + 1e-9);
    fit.loadings = lambda;
    fit.uniquenesses = psi;
    return fit;
}

OmegaEstimate mcdonald_omega(const linalg::DenseMatrix<double>& data, const OmegaOptions& options) {
    check_data(data, 3, "mcdonald_omega");
    if (!(options.confidence > 0 && options.confidence < 1))
        throw std::invalid_argument("mcdonald_omega: confidence must be in (0, 1)");
    const std::size_t n = data.rows();
    std::vector<std::size_t> rows(n);
    for (std::size_t i = 0; i < n; ++i)
        rows[i] = i;
    OmegaEstimate o;
    o.n = n;
    const Matrix c = covariance(data, rows);
    std::tie(o.omega, o.omega_observed) = omega_of(c, n, o.fit);
    o.alpha = alpha_of(c, c.rows());
    const double nan = std::numeric_limits<double>::quiet_NaN();
    o.omega_lower = o.omega_upper = o.alpha_lower = o.alpha_upper = nan;
    if (options.replicates == 0)
        return o;
    random::SplitMix64 rng(options.seed);
    std::vector<double> om, al;
    std::vector<std::size_t> boot(n);
    for (std::size_t r = 0; r < options.replicates; ++r) {
        for (auto& b : boot)
            b = static_cast<std::size_t>(rng.next_u64() % n);
        const Matrix cb = covariance(data, boot);
        bool ok = linalg::cholesky(cb).ok;
        for (std::size_t j = 0; ok && j < cb.rows(); ++j)
            ok = cb(j, j) > 0;
        if (!ok)
            continue; // degenerate resample
        OneFactorFit f;
        om.push_back(omega_of(cb, n, f).first);
        al.push_back(alpha_of(cb, cb.rows()));
    }
    o.replicates = om.size();
    if (om.empty())
        return o;
    std::ranges::sort(om), std::ranges::sort(al);
    const double tail = (1 - options.confidence) / 2;
    o.omega_lower = quantile_sorted(om, tail), o.omega_upper = quantile_sorted(om, 1 - tail);
    o.alpha_lower = quantile_sorted(al, tail), o.alpha_upper = quantile_sorted(al, 1 - tail);
    return o;
}

} // namespace datamunge::stats
