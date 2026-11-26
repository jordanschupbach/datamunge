/// @file rdd.cpp
/// @brief Local polynomial regression discontinuity estimates and MSE-optimal bandwidths.
#include <datamunge/causal/rdd.hpp>
#include <datamunge/random/distributions.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace datamunge::causal {
/// @brief Implementation helpers local to this translation unit.
namespace {
/// @brief A dense row-major matrix with a fixed column count.
struct Table {
    std::size_t rows{}; ///< Row count.
    std::size_t cols{}; ///< Column count.
    std::vector<double> v; ///< Row-major entries.
    Table() = default; ///< Creates an empty matrix.

    /// @brief Creates a zero matrix.
    /// @param r Row count.
    /// @param c Column count.
    Table(std::size_t r, std::size_t c) : rows(r), cols(c), v(r * c, 0.0) {}
    /// @brief Entry access.
    /// @param i Row.
    /// @param j Column.
    /// @return Reference to the entry.
    double& operator()(std::size_t i, std::size_t j) { return v[i * cols + j]; }
    /// @brief Read-only entry access.
    /// @param i Row.
    /// @param j Column.
    /// @return The entry.
    double operator()(std::size_t i, std::size_t j) const { return v[i * cols + j]; }
};

/// @brief Observations on one side of the cutoff, sorted by the running variable.
struct Side {
    std::vector<double> x;           ///< Running variable (ascending).
    Table d;                         ///< Responses: outcome, then treatment in fuzzy designs.
    std::vector<std::size_t> dups;   ///< Size of each observation's tie group.
    std::vector<std::size_t> dupsid; ///< 1-based position of each observation within its tie group.
};

/// @brief Kernel weight `K((x - c) / h) / h`, as in `rdrobust_kweight`.
/// @param x Running-variable value.
/// @param c Cutoff.
/// @param h Bandwidth.
/// @param kernel Kernel.
/// @return The weight (zero outside the bandwidth).
double kernel_weight(double x, double c, double h, RdKernel kernel) {
    const double u = (x - c) / h, a = std::abs(u);
    if (a > 1)
        return 0.0;
    switch (kernel) {
    case RdKernel::epanechnikov:
        return 0.75 * (1 - u * u) / h;
    case RdKernel::uniform:
        return 0.5 / h;
    case RdKernel::triangular:
        break;
    }
    return (1 - a) / h;
}

/// @brief Rule-of-thumb constant of the pilot bandwidth for a kernel.
/// @param kernel Kernel.
/// @return 2.576, 2.34 or 1.843.
double pilot_constant(RdKernel kernel) {
    switch (kernel) {
    case RdKernel::epanechnikov:
        return 2.34;
    case RdKernel::uniform:
        return 1.843;
    case RdKernel::triangular:
        break;
    }
    return 2.576;
}

/// @brief Inverse of a small symmetric positive definite matrix by Gauss-Jordan elimination.
/// @param a Square matrix.
/// @return Its inverse.
/// @throws std::runtime_error When the matrix is numerically singular.
Table invert(Table a) {
    const std::size_t k = a.rows;
    Table inv(k, k);
    for (std::size_t i = 0; i < k; ++i)
        inv(i, i) = 1;
    double scale = 0;
    for (const double e : a.v)
        scale = std::max(scale, std::abs(e));
    for (std::size_t col = 0; col < k; ++col) {
        std::size_t piv = col;
        for (std::size_t r = col + 1; r < k; ++r)
            if (std::abs(a(r, col)) > std::abs(a(piv, col)))
                piv = r;
        if (!(std::abs(a(piv, col)) > 1e-14 * scale))
            throw std::runtime_error("regression_discontinuity: singular local polynomial fit "
                                     "(too few distinct running-variable values in a bandwidth)");
        for (std::size_t j = 0; j < k; ++j) {
            std::swap(a(col, j), a(piv, j));
            std::swap(inv(col, j), inv(piv, j));
        }
        const double p = a(col, col);
        for (std::size_t j = 0; j < k; ++j) {
            a(col, j) /= p;
            inv(col, j) /= p;
        }
        for (std::size_t r = 0; r < k; ++r) {
            if (r == col)
                continue;
            const double f = a(r, col);
            for (std::size_t j = 0; j < k; ++j) {
                a(r, j) -= f * a(col, j);
                inv(r, j) -= f * inv(col, j);
            }
        }
    }
    return inv;
}

/// @brief A weighted polynomial fit on a subset of one side.
struct LocalFit {
    std::vector<std::size_t> rows; ///< Indices into the side.
    Table r;                       ///< Polynomial design (x - c)^j.
    std::vector<double> w;         ///< Kernel weights of the rows.
    Table inv_gram;                ///< (R' W R)^{-1}.
    Table beta;                    ///< Coefficients, one column per response.
};

/// @brief Rows of a side with positive kernel weight at bandwidth h.
/// @param s Side.
/// @param c Cutoff.
/// @param h Bandwidth.
/// @param kernel Kernel.
/// @return Ascending indices.
std::vector<std::size_t> rows_within(const Side& s, double c, double h, RdKernel kernel) {
    std::vector<std::size_t> rows;
    for (std::size_t i = 0; i < s.x.size(); ++i)
        if (kernel_weight(s.x[i], c, h, kernel) > 0)
            rows.push_back(i);
    return rows;
}

/// @brief Polynomial design matrix (x - c)^j, j = 0..order, over the given rows.
/// @param s Side.
/// @param rows Rows.
/// @param c Cutoff.
/// @param order Polynomial order.
/// @return The design.
Table design(const Side& s, const std::vector<std::size_t>& rows, double c, int order) {
    Table r(rows.size(), static_cast<std::size_t>(order) + 1);
    for (std::size_t i = 0; i < rows.size(); ++i) {
        double p = 1;
        for (std::size_t j = 0; j < r.cols; ++j, p *= s.x[rows[i]] - c)
            r(i, j) = p;
    }
    return r;
}

/// @brief Inverse weighted Gram matrix (R' W R)^{-1}.
/// @param r Design.
/// @param w Weights.
/// @return The inverse.
Table inverse_gram(const Table& r, const std::vector<double>& w) {
    Table g(r.cols, r.cols);
    for (std::size_t i = 0; i < r.rows; ++i)
        for (std::size_t a = 0; a < r.cols; ++a)
            for (std::size_t b = 0; b < r.cols; ++b)
                g(a, b) += w[i] * r(i, a) * r(i, b);
    return invert(g);
}

/// @brief Product inv * sum_i X_i D_i' where X is a weighted design (rows already multiplied).
/// @param inv k-by-k matrix.
/// @param x Weighted design over the rows.
/// @param s Side holding the responses.
/// @param rows Rows of the side.
/// @return k-by-responses coefficients.
Table solve_weighted(const Table& inv, const Table& x, const Side& s, const std::vector<std::size_t>& rows) {
    Table xd(x.cols, s.d.cols), out(x.cols, s.d.cols);
    for (std::size_t i = 0; i < rows.size(); ++i)
        for (std::size_t a = 0; a < x.cols; ++a)
            for (std::size_t m = 0; m < s.d.cols; ++m)
                xd(a, m) += x(i, a) * s.d(rows[i], m);
    for (std::size_t a = 0; a < x.cols; ++a)
        for (std::size_t m = 0; m < s.d.cols; ++m)
            for (std::size_t b = 0; b < x.cols; ++b)
                out(a, m) += inv(a, b) * xd(b, m);
    return out;
}

/// @brief Rows of the design multiplied by the kernel weights.
/// @param r Design.
/// @param w Weights.
/// @return The weighted design.
Table weighted(const Table& r, const std::vector<double>& w) {
    Table out = r;
    for (std::size_t i = 0; i < r.rows; ++i)
        for (std::size_t j = 0; j < r.cols; ++j)
            out(i, j) *= w[i];
    return out;
}

/// @brief Weighted polynomial fit of every response over rows with positive weight at h.
/// @param s Side.
/// @param c Cutoff.
/// @param h Bandwidth.
/// @param order Polynomial order.
/// @param kernel Kernel.
/// @return The fit.
LocalFit fit_local(const Side& s, double c, double h, int order, RdKernel kernel) {
    LocalFit f;
    f.rows = rows_within(s, c, h, kernel);
    f.r = design(s, f.rows, c, order);
    for (const std::size_t i : f.rows)
        f.w.push_back(kernel_weight(s.x[i], c, h, kernel));
    f.inv_gram = inverse_gram(f.r, f.w);
    f.beta = solve_weighted(f.inv_gram, weighted(f.r, f.w), s, f.rows);
    return f;
}

/// @brief Residuals for the variance estimator, one column per response (`rdrobust_res`).
/// @param s Side.
/// @param rows Rows of the side (ascending).
/// @param r Design over the rows.
/// @param w Weights of the rows.
/// @param inv_gram Inverse weighted Gram matrix of the fit.
/// @param beta Coefficients of the fit.
/// @param variance Variance estimator.
/// @param matches Nearest-neighbour matches.
/// @return Rows-by-responses residuals.
Table residuals(const Side& s, const std::vector<std::size_t>& rows, const Table& r, const std::vector<double>& w,
                const Table& inv_gram, const Table& beta, RdVariance variance, int matches) {
    const std::size_t n = rows.size(), dim = s.d.cols;
    Table res(n, dim);
    if (variance == RdVariance::nearest_neighbor) {
        const std::size_t limit = std::min<std::size_t>(static_cast<std::size_t>(matches), n - 1);
        for (std::size_t pos = 0; pos < n; ++pos) {
            const std::size_t i = rows[pos];
            std::size_t rpos = s.dups[i] - s.dupsid[i], lpos = s.dupsid[i] - 1;
            while (lpos + rpos < limit) {
                const bool no_left = pos <= lpos, no_right = pos + rpos + 1 >= n;
                if (no_left)
                    rpos += s.dups[rows[pos + rpos + 1]];
                else if (no_right)
                    lpos += s.dups[rows[pos - lpos - 1]];
                else {
                    const double dl = s.x[i] - s.x[rows[pos - lpos - 1]];
                    const double dr = s.x[rows[pos + rpos + 1]] - s.x[i];
                    if (dl > dr)
                        rpos += s.dups[rows[pos + rpos + 1]];
                    else if (dl < dr)
                        lpos += s.dups[rows[pos - lpos - 1]];
                    else {
                        rpos += s.dups[rows[pos + rpos + 1]];
                        lpos += s.dups[rows[pos - lpos - 1]];
                    }
                }
            }
            const std::size_t lo = pos - std::min(pos, lpos), hi = std::min(n - 1, pos + rpos);
            const double ji = static_cast<double>(hi - lo);
            for (std::size_t m = 0; m < dim; ++m) {
                double sum = 0;
                for (std::size_t k = lo; k <= hi; ++k)
                    sum += s.d(rows[k], m);
                sum -= s.d(i, m);
                res(pos, m) = std::sqrt(ji / (ji + 1)) * (s.d(i, m) - sum / ji);
            }
        }
        return res;
    }
    const double k = static_cast<double>(r.cols);
    for (std::size_t pos = 0; pos < n; ++pos) {
        double scale = 1;
        if (variance == RdVariance::hc1)
            scale = std::sqrt(static_cast<double>(n) / (static_cast<double>(n) - k));
        else if (variance == RdVariance::hc2 || variance == RdVariance::hc3) {
            double hii = 0;
            for (std::size_t a = 0; a < r.cols; ++a)
                for (std::size_t b = 0; b < r.cols; ++b)
                    hii += r(pos, a) * inv_gram(a, b) * r(pos, b);
            hii *= w[pos];
            scale = variance == RdVariance::hc2 ? 1 / std::sqrt(1 - hii) : 1 / (1 - hii);
        }
        for (std::size_t m = 0; m < dim; ++m) {
            double fitted = 0;
            for (std::size_t a = 0; a < r.cols; ++a)
                fitted += r(pos, a) * beta(a, m);
            res(pos, m) = scale * (s.d(rows[pos], m) - fitted);
        }
    }
    return res;
}

/// @brief Sandwich entry [j, j] of inv * (sum_i (s' e_i)^2 X_i X_i') * inv.
/// @param inv Inverse Gram matrix.
/// @param x Score design (weighted design or bias-corrected Q).
/// @param res Residuals.
/// @param combo Linear combination of the response columns.
/// @param j Coefficient index.
/// @return The variance entry.
double sandwich(const Table& inv, const Table& x, const Table& res, const std::vector<double>& combo, std::size_t j) {
    // Only row j of inv is needed: var = sum_i (s'e_i)^2 (inv_j . X_i)^2.
    double v = 0;
    for (std::size_t i = 0; i < x.rows; ++i) {
        double e = 0, a = 0;
        for (std::size_t m = 0; m < res.cols; ++m)
            e += combo[m] * res(i, m);
        for (std::size_t b = 0; b < x.cols; ++b)
            a += inv(j, b) * x(i, b);
        v += e * e * a * a;
    }
    return v;
}

/// @brief Variance, bias and regularisation constants of one bandwidth step (`rdrobust_bw`).
struct BwConstants {
    double v{};    ///< Variance constant.
    double b{};    ///< Bias constant (signed).
    double r{};    ///< Regularisation term.
    double rate{}; ///< Exponent 1 / (2o + 3).
};

/// @brief Constants of the MSE-optimal bandwidth for a derivative of order nu with a fit of order o.
/// @param s Side.
/// @param c Cutoff.
/// @param o Order of the fit whose MSE is optimised.
/// @param nu Derivative estimated.
/// @param ob Order of the fit estimating the bias derivative.
/// @param hv Pilot bandwidth for the variance and bias constants.
/// @param hb Bandwidth of the bias-derivative fit.
/// @param scale Regularisation scale (0 disables it).
/// @param o_opt Options (kernel, variance estimator, matches).
/// @return The constants.
BwConstants bw_constants(const Side& s, double c, int o, int nu, int ob, double hv, double hb, double scale,
                         const RdOptions& o_opt) {
    const LocalFit fv = fit_local(s, c, hv, o, o_opt.kernel);
    const std::size_t inu = static_cast<std::size_t>(nu);
    std::vector<double> combo{1.0};
    if (s.d.cols == 2) {
        // The nu! factors scale both weights equally and cancel in the bandwidth ratios.
        const double ty = fv.beta(inu, 0), tt = fv.beta(inu, 1);
        combo = {1 / tt, -ty / (tt * tt)};
    }
    const Table xv = weighted(fv.r, fv.w);
    const Table rv = residuals(s, fv.rows, fv.r, fv.w, fv.inv_gram, fv.beta, o_opt.variance, o_opt.matches);
    const double vv = sandwich(fv.inv_gram, xv, rv, combo, inu);
    // Leading bias constant: h^nu [ (R'WR)^{-1} R'W u^{o+1} ]_nu with u = (x - c) / hv.
    std::vector<double> mom(xv.cols, 0.0);
    for (std::size_t i = 0; i < xv.rows; ++i) {
        const double u = std::pow((s.x[fv.rows[i]] - c) / hv, o + 1);
        for (std::size_t a = 0; a < xv.cols; ++a)
            mom[a] += xv(i, a) * u;
    }
    double bconst = 0;
    for (std::size_t a = 0; a < xv.cols; ++a)
        bconst += fv.inv_gram(inu, a) * mom[a];
    bconst *= std::pow(hv, nu);
    const LocalFit fb = fit_local(s, c, hb, ob, o_opt.kernel);
    const std::size_t ib = static_cast<std::size_t>(o) + 1;
    double bwreg = 0;
    if (scale > 0) {
        const Table rb = residuals(s, fb.rows, fb.r, fb.w, fb.inv_gram, fb.beta, o_opt.variance, o_opt.matches);
        bwreg = 3 * bconst * bconst * sandwich(fb.inv_gram, weighted(fb.r, fb.w), rb, combo, ib);
    }
    double lead = 0;
    for (std::size_t m = 0; m < combo.size(); ++m)
        lead += combo[m] * fb.beta(ib, m);
    BwConstants out;
    out.b = std::sqrt(2.0 * (o + 1 - nu)) * bconst * lead;
    out.v = (2.0 * nu + 1) * std::pow(hv, 2 * nu + 1) * vv;
    out.r = scale * 2.0 * (o + 1 - nu) * bwreg;
    out.rate = 1.0 / (2.0 * o + 3);
    return out;
}

/// @brief R's type-2 sample quantile (inverse empirical CDF, averaging at jumps).
/// @param sorted Ascending sample.
/// @param p Probability.
/// @return The quantile.
double quantile_type2(const std::vector<double>& sorted, double p) {
    const double np = static_cast<double>(sorted.size()) * p;
    const double j = std::floor(np + 4 * std::numeric_limits<double>::epsilon() * np);
    const auto at = [&](double k) {
        const auto idx = static_cast<std::size_t>(std::clamp(k, 1.0, static_cast<double>(sorted.size())));
        return sorted[idx - 1];
    };
    if (std::abs(np - j) <= 4 * std::numeric_limits<double>::epsilon() * np)
        return 0.5 * (at(j) + at(j + 1));
    return at(j + 1);
}

/// @brief Validated, sorted data split at the cutoff.
struct Prepared {
    Side left;  ///< Observations below the cutoff.
    Side right; ///< Observations at or above the cutoff.
    std::vector<double> x; ///< All running-variable values, ascending.
    int p{};    ///< Point-estimate order.
    int q{};    ///< Bias-correction order.
};

/// @brief Validates the inputs and splits them at the cutoff, sorted by the running variable.
/// @param outcome Outcomes.
/// @param running Running variable.
/// @param options Options.
/// @param treatment Treatment (fuzzy) or empty.
/// @return The prepared sides.
/// @throws std::invalid_argument For invalid inputs or options.
Prepared prepare(const std::vector<double>& outcome, const std::vector<double>& running, const RdOptions& options,
                 const std::vector<double>& treatment) {
    const std::size_t n = running.size();
    if (outcome.size() != n || (!treatment.empty() && treatment.size() != n))
        throw std::invalid_argument("regression_discontinuity: input sizes differ");
    for (std::size_t i = 0; i < n; ++i)
        if (!std::isfinite(outcome[i]) || !std::isfinite(running[i]) ||
            (!treatment.empty() && !std::isfinite(treatment[i])))
            throw std::invalid_argument("regression_discontinuity: inputs must be finite");
    Prepared out;
    out.p = options.order;
    out.q = options.bias_order < 0 ? options.order + 1 : options.bias_order;
    if (out.p < 0 || out.q <= out.p || options.derivative < 0 || options.derivative > out.p)
        throw std::invalid_argument("regression_discontinuity: need 0 <= derivative <= order < bias_order");
    if (options.matches < 1)
        throw std::invalid_argument("regression_discontinuity: matches must be at least 1");
    if (!(options.confidence > 0 && options.confidence < 1))
        throw std::invalid_argument("regression_discontinuity: confidence must be in (0, 1)");
    if (!std::isfinite(options.cutoff))
        throw std::invalid_argument("regression_discontinuity: cutoff must be finite");
    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::ranges::stable_sort(order, [&](std::size_t a, std::size_t b) { return running[a] < running[b]; });
    const std::size_t dim = treatment.empty() ? 1 : 2;
    std::vector<std::size_t> li, ri;
    for (const std::size_t i : order) {
        out.x.push_back(running[i]);
        (running[i] < options.cutoff ? li : ri).push_back(i);
    }
    const auto fill = [&](Side& s, const std::vector<std::size_t>& idx) {
        s.d = Table(idx.size(), dim);
        for (std::size_t k = 0; k < idx.size(); ++k) {
            s.x.push_back(running[idx[k]]);
            s.d(k, 0) = outcome[idx[k]];
            if (dim == 2)
                s.d(k, 1) = treatment[idx[k]];
        }
        s.dups.assign(idx.size(), 1);
        s.dupsid.assign(idx.size(), 1);
        for (std::size_t a = 0; a < idx.size();) {
            std::size_t b = a;
            while (b < idx.size() && s.x[b] == s.x[a])
                ++b;
            for (std::size_t k = a; k < b; ++k) {
                s.dups[k] = b - a;
                s.dupsid[k] = k - a + 1;
            }
            a = b;
        }
    };
    fill(out.left, li);
    fill(out.right, ri);
    const auto need = static_cast<std::size_t>(out.q) + 2;
    if (out.left.x.size() < need || out.right.x.size() < need)
        throw std::invalid_argument("regression_discontinuity: too few observations on a side of the cutoff");
    return out;
}

/// @brief Number of distinct values in an ascending vector.
/// @param x Ascending values.
/// @return The count.
std::size_t distinct(const std::vector<double>& x) {
    std::size_t m = 0;
    for (std::size_t i = 0; i < x.size(); ++i)
        m += i == 0 || x[i] != x[i - 1];
    return m;
}

/// @brief Variance of a column (zero means no variation).
/// @param t Table.
/// @param m Column.
/// @return The sum of squared deviations.
double column_spread(const Table& t, std::size_t m) {
    double mean = 0, ss = 0;
    for (std::size_t i = 0; i < t.rows; ++i)
        mean += t(i, m) / static_cast<double>(t.rows);
    for (std::size_t i = 0; i < t.rows; ++i)
        ss += (t(i, m) - mean) * (t(i, m) - mean);
    return ss;
}

/// @brief Copy of a side keeping only the outcome column.
/// @param s Side.
/// @return The sharp side.
Side outcome_only(const Side& s) {
    Side out = s;
    out.d = Table(s.d.rows, 1);
    for (std::size_t i = 0; i < s.d.rows; ++i)
        out.d(i, 0) = s.d(i, 0);
    return out;
}

/// @brief The mserd bandwidth selector on prepared data.
/// @param prep Prepared sides.
/// @param options Options.
/// @return Bandwidths.
RdBandwidths select_bandwidths(const Prepared& prep, const RdOptions& options) {
    const double c = options.cutoff;
    Side left = prep.left, right = prep.right;
    if (left.d.cols == 2 &&
        (options.sharp_bandwidth || column_spread(left.d, 1) == 0 || column_spread(right.d, 1) == 0)) {
        left = outcome_only(left);
        right = outcome_only(right);
    }
    const std::vector<double>& x = prep.x;
    const auto n = static_cast<double>(x.size());
    double mean = 0, ss = 0;
    for (const double v : x)
        mean += v / n;
    for (const double v : x)
        ss += (v - mean) * (v - mean);
    const double sd = std::sqrt(ss / (n - 1));
    const double iqr = quantile_type2(x, 0.75) - quantile_type2(x, 0.25);
    const double spread = std::min(sd, iqr / 1.349);
    RdBandwidths out;
    out.distinct_left = distinct(left.x);
    out.distinct_right = distinct(right.x);
    double count = n;
    double bw_min = 0;
    if (options.mass_point_adjust) {
        count = static_cast<double>(out.distinct_left + out.distinct_right);
        const double mass_l = 1 - static_cast<double>(out.distinct_left) / static_cast<double>(left.x.size());
        const double mass_r = 1 - static_cast<double>(out.distinct_right) / static_cast<double>(right.x.size());
        out.mass_points = mass_l >= 0.2 || mass_r >= 0.2;
        if (out.mass_points) {
            // Require ten distinct values per side: distance to the 10th closest distinct value.
            std::vector<double> ul, ur;
            for (std::size_t i = left.x.size(); i-- > 0;)
                if (ul.empty() || left.x[i] != ul.back())
                    ul.push_back(left.x[i]);
            for (const double v : right.x)
                if (ur.empty() || v != ur.back())
                    ur.push_back(v);
            const double ml = std::abs(ul[std::min<std::size_t>(10, ul.size()) - 1] - c) + 1e-8;
            const double mr = std::abs(ur[std::min<std::size_t>(10, ur.size()) - 1] - c) + 1e-8;
            bw_min = std::max(ml, mr);
        }
    }
    const double bw_max = std::max(std::abs(c - x.front()), std::abs(c - x.back()));
    double pilot = pilot_constant(options.kernel) * spread * std::pow(count, -0.2);
    pilot = std::max(std::min(pilot, bw_max), bw_min);
    const double range_l = std::abs(c - left.x.front()), range_r = std::abs(c - right.x.back());
    const int p = prep.p, q = prep.q;
    const auto cl = bw_constants(left, c, q + 1, q + 1, q + 2, pilot, range_l, 0, options);
    const auto cr = bw_constants(right, c, q + 1, q + 1, q + 2, pilot, range_r, 0, options);
    double d = std::pow((cl.v + cr.v) / ((cr.b - cl.b) * (cr.b - cl.b)), cl.rate);
    d = std::max(std::min(d, bw_max), bw_min);
    const auto bl = bw_constants(left, c, q, p + 1, q + 1, pilot, d, 1, options);
    const auto br = bw_constants(right, c, q, p + 1, q + 1, pilot, d, 1, options);
    double b = std::pow((bl.v + br.v) / ((br.b - bl.b) * (br.b - bl.b) + (br.r + bl.r)), bl.rate);
    b = std::min(b, bw_max);
    const auto hl = bw_constants(left, c, p, options.derivative, q, pilot, b, 1, options);
    const auto hr = bw_constants(right, c, p, options.derivative, q, pilot, b, 1, options);
    double h = std::pow((hl.v + hr.v) / ((hr.b - hl.b) * (hr.b - hl.b) + (hr.r + hl.r)), hl.rate);
    h = std::min(h, bw_max);
    out.h = h;
    out.b = b;
    out.pilot = pilot;
    out.d = d;
    return out;
}

/// @brief Fitted quantities of one side used for the estimate and its variances.
struct SideEstimate {
    Table beta_p;   ///< Order-p coefficients within h.
    Table beta_bc;  ///< Bias-corrected order-p coefficients.
    Table inv_p;    ///< Inverse weighted Gram of the order-p fit.
    Table score_p;  ///< Weighted order-p design (conventional scores).
    Table score_q;  ///< Bias-corrected scores Q.
    Table res_h;    ///< Residuals for the conventional variance.
    Table res_b;    ///< Residuals for the robust variance.
    std::size_t effective{}; ///< Rows with positive weight at h.
};

/// @brief Conventional and bias-corrected local polynomial fits on one side (`rdrobust` core).
/// @param s Side.
/// @param c Cutoff.
/// @param h Main bandwidth.
/// @param b Bias bandwidth.
/// @param p Point-estimate order.
/// @param q Bias-correction order.
/// @param options Kernel, variance and matches.
/// @return The side's fitted quantities.
SideEstimate estimate_side(const Side& s, double c, double h, double b, int p, int q, const RdOptions& options) {
    SideEstimate out;
    out.effective = rows_within(s, c, h, options.kernel).size();
    const auto rows = rows_within(s, c, h > b ? h : b, options.kernel);
    const Table rq = design(s, rows, c, q);
    const Table rp = design(s, rows, c, p);
    std::vector<double> wh, wb;
    for (const std::size_t i : rows) {
        wh.push_back(kernel_weight(s.x[i], c, h, options.kernel));
        wb.push_back(kernel_weight(s.x[i], c, b, options.kernel));
    }
    const std::size_t kp = rp.cols;
    const Table inv_q = inverse_gram(rq, wb);
    out.inv_p = inverse_gram(rp, wh);
    out.score_p = weighted(rp, wh);
    // L = sum_i W_h R_p u^{p+1}; Q_i = W_h R_p,i - h^{p+1} L [ (R_q'W_b R_q)^{-1} R_q,i ]_{p+1} W_b,i.
    std::vector<double> l(kp, 0.0);
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const double u = std::pow((s.x[rows[i]] - c) / h, p + 1);
        for (std::size_t a = 0; a < kp; ++a)
            l[a] += out.score_p(i, a) * u;
    }
    const double hp = std::pow(h, p + 1);
    out.score_q = out.score_p;
    for (std::size_t i = 0; i < rows.size(); ++i) {
        double g = 0;
        for (std::size_t a = 0; a < rq.cols; ++a)
            g += inv_q(kp, a) * rq(i, a);
        for (std::size_t a = 0; a < kp; ++a)
            out.score_q(i, a) -= hp * l[a] * g * wb[i];
    }
    out.beta_p = solve_weighted(out.inv_p, out.score_p, s, rows);
    out.beta_bc = solve_weighted(out.inv_p, out.score_q, s, rows);
    const Table beta_q = solve_weighted(inv_q, weighted(rq, wb), s, rows);
    out.res_h = residuals(s, rows, rp, wh, out.inv_p, out.beta_p, options.variance, options.matches);
    out.res_b = options.variance == RdVariance::nearest_neighbor
                    ? out.res_h
                    : residuals(s, rows, rq, wb, inv_q, beta_q, options.variance, options.matches);
    return out;
}

/// @brief Builds an estimate with normal inference.
/// @param est Point estimate.
/// @param se Standard error.
/// @param z Critical value.
/// @return The estimate.
RdEstimate make_estimate(double est, double se, double z) {
    const double t = est / se;
    return {est, se, t, 2 * random::normal_cdf(-std::abs(t)), est - z * se, est + z * se};
}
} // namespace

RdBandwidths rd_bandwidths(const std::vector<double>& outcome, const std::vector<double>& running,
                           const RdOptions& options, const std::vector<double>& treatment) {
    return select_bandwidths(prepare(outcome, running, options, treatment), options);
}

RdResult regression_discontinuity(const std::vector<double>& outcome, const std::vector<double>& running,
                                  const RdOptions& options, const std::vector<double>& treatment) {
    const Prepared prep = prepare(outcome, running, options, treatment);
    const bool given_h = !std::isnan(options.bandwidth), given_b = !std::isnan(options.bias_bandwidth);
    if ((given_h && !(options.bandwidth > 0 && std::isfinite(options.bandwidth))) ||
        (given_b && !(options.bias_bandwidth > 0 && std::isfinite(options.bias_bandwidth))))
        throw std::invalid_argument("regression_discontinuity: bandwidths must be positive");
    double h = options.bandwidth, b = options.bias_bandwidth;
    if (!given_h) {
        const auto bw = select_bandwidths(prep, options);
        h = bw.h;
        if (!given_b)
            b = bw.b;
    } else if (!given_b)
        b = h;
    const double c = options.cutoff;
    const auto el = estimate_side(prep.left, c, h, b, prep.p, prep.q, options);
    const auto er = estimate_side(prep.right, c, h, b, prep.p, prep.q, options);
    const auto j = static_cast<std::size_t>(options.derivative);
    double fact = 1;
    for (int k = 2; k <= options.derivative; ++k)
        fact *= k;
    const auto jump = [&](const Table SideEstimate::*m, std::size_t col) {
        return fact * ((er.*m)(j, col) - (el.*m)(j, col));
    };
    const auto se = [&](const Table SideEstimate::*score, const Table SideEstimate::*res,
                        const std::vector<double>& combo) {
        return fact * std::sqrt(sandwich(el.inv_p, el.*score, el.*res, combo, j) +
                                sandwich(er.inv_p, er.*score, er.*res, combo, j));
    };
    const double z = random::normal_quantile(0.5 + options.confidence / 2);
    RdResult out;
    out.fuzzy = !treatment.empty();
    const double y_cl = jump(&SideEstimate::beta_p, 0), y_bc = jump(&SideEstimate::beta_bc, 0);
    double tau_cl = y_cl, tau_bc = y_bc;
    std::vector<double> combo{1.0};
    if (out.fuzzy) {
        const double t_cl = jump(&SideEstimate::beta_p, 1), t_bc = jump(&SideEstimate::beta_bc, 1);
        if (t_cl == 0)
            throw std::runtime_error("regression_discontinuity: the treatment does not jump at the cutoff");
        tau_cl = y_cl / t_cl;
        combo = {1 / t_cl, -y_cl / (t_cl * t_cl)};
        tau_bc = tau_cl - (combo[0] * (y_cl - y_bc) + combo[1] * (t_cl - t_bc));
        const std::vector<double> first{0.0, 1.0};
        const double t_se_cl = se(&SideEstimate::score_p, &SideEstimate::res_h, first);
        const double t_se_rb = se(&SideEstimate::score_q, &SideEstimate::res_b, first);
        out.first_stage_conventional = make_estimate(t_cl, t_se_cl, z);
        out.first_stage_robust = make_estimate(t_bc, t_se_rb, z);
    }
    const double se_cl = se(&SideEstimate::score_p, &SideEstimate::res_h, combo);
    const double se_rb = se(&SideEstimate::score_q, &SideEstimate::res_b, combo);
    out.conventional = make_estimate(tau_cl, se_cl, z);
    out.bias_corrected = make_estimate(tau_bc, se_cl, z);
    out.robust = make_estimate(tau_bc, se_rb, z);
    out.h_left = out.h_right = h;
    out.b_left = out.b_right = b;
    out.left_limit = fact * el.beta_p(j, 0);
    out.right_limit = fact * er.beta_p(j, 0);
    out.n_left = prep.left.x.size();
    out.n_right = prep.right.x.size();
    out.effective_left = el.effective;
    out.effective_right = er.effective;
    return out;
}

} // namespace datamunge::causal
