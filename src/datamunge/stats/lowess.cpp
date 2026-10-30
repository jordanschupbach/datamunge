/// @file lowess.cpp
/// @brief R's `lowess()` smoother, following the algorithm of R's clowess.

#include <datamunge/stats/lowess.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace datamunge::stats {

/// @brief Helpers private to this file.
namespace {

/// @brief One local fit at @p xs over points nleft..nright (0-based, inclusive): R's `lowest`.
/// @return Whether any point had positive weight; the fitted value goes to @p ys.
bool lowest(const std::vector<double>& x, const std::vector<double>& y, double xs, double& ys, std::size_t nleft,
            std::size_t nright, std::vector<double>& w, bool use_rw, const std::vector<double>& rw) {
    const std::size_t n     = x.size();
    const double      range = x[n - 1] - x[0];
    const double      h     = std::max(xs - x[nleft], x[nright] - xs);
    const double      h9 = 0.999 * h, h1 = 0.001 * h;
    double            a  = 0.0;
    std::size_t       j  = nleft;
    while (j < n) {
        w[j]           = 0.0;
        const double r = std::abs(x[j] - xs);
        if (r <= h9) {
            w[j] = r <= h1 ? 1.0 : std::pow(1.0 - std::pow(r / h, 3), 3);
            if (use_rw) w[j] *= rw[j];
            a += w[j];
        } else if (x[j] > xs) {
            break;
        }
        ++j;
    }
    const std::size_t nrt = j - 1;
    if (a <= 0.0) return false;
    for (j = nleft; j <= nrt; ++j) w[j] /= a;
    if (h > 0.0) {
        a = 0.0;
        for (j = nleft; j <= nrt; ++j) a += w[j] * x[j];
        double b = xs - a, c = 0.0;
        for (j = nleft; j <= nrt; ++j) c += w[j] * (x[j] - a) * (x[j] - a);
        if (std::sqrt(c) > 0.001 * range) {
            b /= c;
            for (j = nleft; j <= nrt; ++j) w[j] *= b * (x[j] - a) + 1.0;
        }
    }
    ys = 0.0;
    for (j = nleft; j <= nrt; ++j) ys += w[j] * y[j];
    return true;
}

} // namespace

LowessFit lowess(const std::vector<double>& x_in, const std::vector<double>& y_in, double f, int iter, double delta) {
    if (x_in.size() != y_in.size()) throw std::invalid_argument("lowess: x and y differ in length");
    if (!(f > 0.0)) throw std::invalid_argument("lowess: f must be positive");
    const std::size_t        n = x_in.size();
    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return x_in[a] < x_in[b]; });
    LowessFit out;
    std::vector<double> y(n);
    out.x.resize(n);
    for (std::size_t i = 0; i < n; ++i) out.x[i] = x_in[order[i]], y[i] = y_in[order[i]];
    const auto& x = out.x;
    out.y.assign(n, 0.0);
    if (n == 0) return out;
    if (n < 2) {
        out.y[0] = y[0];
        return out;
    }
    if (delta < 0) delta = 0.01 * (x[n - 1] - x[0]);
    auto&       ys = out.y;
    std::vector<double> rw(n, 1.0), res(n, 0.0);
    const std::size_t ns = std::max<std::size_t>(std::min<std::size_t>(static_cast<std::size_t>(f * n + 1e-7), n), 2);

    for (int it = 1; it <= iter + 1; ++it) {
        std::size_t nleft = 0, nright = ns - 1, last = 0, i = 0;
        bool        first = true; // R's last = 0 means "none fitted yet" in 1-based indexing
        for (;;) {
            if (nright < n - 1) {
                const double d1 = x[i] - x[nleft], d2 = x[nright + 1] - x[i];
                if (d1 > d2) {
                    ++nleft, ++nright;
                    continue;
                }
            }
            if (!lowest(x, y, x[i], ys[i], nleft, nright, res, it > 1, rw)) ys[i] = y[i];
            if (!first && last + 1 < i) {
                const double denom = x[i] - x[last];
                for (std::size_t j = last + 1; j < i; ++j) {
                    const double alpha = (x[j] - x[last]) / denom;
                    ys[j]              = alpha * ys[i] + (1.0 - alpha) * ys[last];
                }
            }
            last              = i;
            first             = false;
            const double cut  = x[last] + delta;
            for (i = last + 1; i < n; ++i) {
                if (x[i] > cut) break;
                if (x[i] == x[last]) {
                    ys[i] = ys[last];
                    last  = i;
                }
            }
            i = std::max(last + 1, i - 1);
            if (last >= n - 1) break;
        }
        for (std::size_t k = 0; k < n; ++k) res[k] = y[k] - ys[k];
        double sc = 0.0;
        for (const double r : res) sc += std::abs(r);
        sc /= static_cast<double>(n);
        if (it > iter) break;
        for (std::size_t k = 0; k < n; ++k) rw[k] = std::abs(res[k]);
        std::vector<double> sorted = rw;
        const std::size_t   m1     = n / 2;
        std::nth_element(sorted.begin(), sorted.begin() + m1, sorted.end());
        double cmad;
        if (n % 2 == 0) {
            const double hi = sorted[m1];
            std::nth_element(sorted.begin(), sorted.begin() + (m1 - 1), sorted.end());
            cmad = 3.0 * (hi + sorted[m1 - 1]);
        } else {
            cmad = 6.0 * sorted[m1];
        }
        if (cmad < 1e-7 * sc) break;
        const double c9 = 0.999 * cmad, c1 = 0.001 * cmad;
        for (std::size_t k = 0; k < n; ++k) {
            const double r = std::abs(res[k]);
            rw[k]          = r <= c1 ? 1.0 : r <= c9 ? std::pow(1.0 - (r / cmad) * (r / cmad), 2) : 0.0;
        }
    }
    return out;
}

} // namespace datamunge::stats
