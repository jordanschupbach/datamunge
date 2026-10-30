/// @file quantile_regression.cpp
/// @brief QuantileRegression: ADMM and exact simplex solvers, dual solution, standard errors
///        (IID, NID, bootstrap) and the quantile process.

#include <datamunge/stats/quantile_regression.hpp>

#include <datamunge/linalg/lu.hpp>
#include <datamunge/random/distributions.hpp>
#include <datamunge/random/random.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace datamunge::stats {

/// @brief Solvers and helpers private to this file.
namespace {

/// @brief The dense matrix type used throughout.
using Matrix = linalg::DenseMatrix<double>;

/// @brief Euclidean norm of a vector.
/// @param values  The vector.
/// @return \f$\|v\|_2\f$.
double l2_norm(const std::vector<double>& values) {
    double sum = 0.0;
    for (const double value : values) sum += value * value;
    return std::sqrt(sum);
}

/// @brief The smallest value whose cumulative weight reaches tau times the total (a weighted
///        lower quantile, the intercept-only check-loss minimiser).
/// @param values  (value, weight) pairs; consumed.
/// @param tau  Quantile level.
/// @return The weighted quantile.
double weighted_quantile(std::vector<std::pair<double, double>> values, double tau) {
    std::sort(values.begin(), values.end(),
              [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });
    double total_weight = 0.0;
    for (const auto& [_, weight] : values) total_weight += weight;
    const double target = tau * total_weight;
    double       cumulative = 0.0;
    for (const auto& [value, weight] : values) {
        cumulative += weight;
        if (cumulative >= target) return value;
    }
    return values.back().first;
}

/// @brief The weighted check loss of a coefficient vector.
/// @param x  Design (n x p).
/// @param y  Responses.
/// @param w  Weights.
/// @param beta  Coefficients.
/// @param tau  Quantile level.
/// @return \f$\sum_iw_i\rho_\tau(y_i-x_i^\top\beta)\f$.
double loss_of(const Matrix& x, const std::vector<double>& y, const std::vector<double>& w,
               const std::vector<double>& beta, double tau) {
    const auto fit = x * beta;
    double     total = 0.0;
    for (std::size_t i = 0; i < y.size(); ++i) total += w[i] * check_loss(y[i] - fit[i], tau);
    return total;
}

/// @brief A thin QR factorisation \f$X=QR\f$ (Q n x p with orthonormal columns, R p x p upper
///        triangular) by modified Gram-Schmidt with one reorthogonalisation pass, O(n p^2).
///
/// linalg::qr() forms the full n x n Q, which costs O(n^2) memory; ADMM only needs the thin factor.
class ThinQR {
 public:
    /// @brief Factorises x.
    /// @param x  The n x p design, n >= p.
    /// @throws std::runtime_error  If x is numerically rank-deficient.
    explicit ThinQR(const Matrix& x) : n_(x.rows()), p_(x.cols()), q_(x.cols()), r_(x.cols(), x.cols(), 0.0) {
        for (std::size_t j = 0; j < p_; ++j) {
            std::vector<double> v(n_);
            for (std::size_t i = 0; i < n_; ++i) v[i] = x(i, j);
            const double size = l2_norm(v);
            for (int pass = 0; pass < 2; ++pass)
                for (std::size_t k = 0; k < j; ++k) {
                    double dot = 0.0;
                    for (std::size_t i = 0; i < n_; ++i) dot += q_[k][i] * v[i];
                    for (std::size_t i = 0; i < n_; ++i) v[i] -= dot * q_[k][i];
                    r_(k, j) += dot;
                }
            const double rest = l2_norm(v);
            if (!(rest > 1e-10 * size))
                throw std::runtime_error("QuantileRegression: design matrix is rank-deficient");
            r_(j, j) = rest;
            for (double& value : v) value /= rest;
            q_[j] = std::move(v);
        }
    }

    /// @brief The least-squares solution of \f$X\beta\approx b\f$: \f$R\beta=Q^\top b\f$.
    /// @param b  Right-hand side of length n.
    /// @return \f$\beta\f$.
    [[nodiscard]] std::vector<double> solve(const std::vector<double>& b) const {
        std::vector<double> c(p_, 0.0);
        for (std::size_t k = 0; k < p_; ++k)
            for (std::size_t i = 0; i < n_; ++i) c[k] += q_[k][i] * b[i];
        for (std::size_t k = p_; k-- > 0;) {
            for (std::size_t j = k + 1; j < p_; ++j) c[k] -= r_(k, j) * c[j];
            c[k] /= r_(k, k);
        }
        return c;
    }

 private:
    std::size_t                      n_;  ///< Rows.
    std::size_t                      p_;  ///< Columns.
    std::vector<std::vector<double>> q_;  ///< Columns of Q.
    Matrix                           r_;  ///< R.
};

/// @brief What either solver returns.
struct SolverResult {
    std::vector<double>      beta;        ///< Coefficients.
    std::vector<std::size_t> basis;       ///< Interpolated rows, ascending.
    std::vector<double>      dual;        ///< Dual solution a.
    std::size_t              iterations{0}; ///< Iterations or pivots.
    bool                     converged{false}; ///< Stopping rule met.
    double                   primal_norm{0.0}; ///< ADMM primal residual norm.
    double                   dual_norm{0.0};   ///< ADMM dual residual norm.
};

/// @brief Scaled ADMM for \f$y-X\beta-r=0\f$: a QR least-squares step, an asymmetric soft threshold, a dual step.
/// @param x  Full-rank design.
/// @param y  Responses.
/// @param w  Weights.
/// @param tau  Quantile level.
/// @param options  Penalty, tolerances and iteration cap.
/// @return The fit.
SolverResult admm_fit(const Matrix& x, const std::vector<double>& y, const std::vector<double>& w,
                      double tau, const QuantileRegressionOptions& options) {
    const std::size_t n = x.rows(), p = x.cols();
    const ThinQR decomposition(x);

    SolverResult out;
    out.beta = decomposition.solve(y); // OLS is an effective warm start.
    std::vector<double> r(n, 0.0), previous_r(n, 0.0), u(n, 0.0), rhs(n, 0.0), xb(n, 0.0),
        primal(n, 0.0), dual_argument(n, 0.0);
    double rho = options.rho;

    for (std::size_t iteration = 1; iteration <= options.max_iterations; ++iteration) {
        for (std::size_t i = 0; i < n; ++i) rhs[i] = y[i] - r[i] + u[i];
        out.beta   = decomposition.solve(rhs);
        xb         = x * out.beta;
        previous_r = r;

        for (std::size_t i = 0; i < n; ++i) {
            const double value           = y[i] - xb[i] + u[i];
            const double upper_threshold = w[i] * tau / rho;
            const double lower_threshold = w[i] * (1.0 - tau) / rho;
            if (value > upper_threshold)
                r[i] = value - upper_threshold;
            else if (value < -lower_threshold)
                r[i] = value + lower_threshold;
            else
                r[i] = 0.0;
            primal[i]        = y[i] - xb[i] - r[i];
            u[i]            += primal[i];
            dual_argument[i] = r[i] - previous_r[i];
        }

        out.primal_norm = l2_norm(primal);
        out.dual_norm   = rho * l2_norm(x.transpose() * dual_argument);
        const double primal_scale = std::max({l2_norm(y), l2_norm(xb), l2_norm(r)});
        const double primal_tolerance = std::sqrt(static_cast<double>(n)) * options.absolute_tolerance
                                        + options.relative_tolerance * primal_scale;
        const double dual_tolerance = std::sqrt(static_cast<double>(p)) * options.absolute_tolerance
                                      + options.relative_tolerance * rho * l2_norm(x.transpose() * u);
        out.iterations = iteration;
        if (out.primal_norm <= primal_tolerance && out.dual_norm <= dual_tolerance) {
            out.converged = true;
            break;
        }
        // Residual balancing changes only the augmented-Lagrangian penalty, not the optimum.
        // Rescale the scaled dual variable so the unscaled multiplier rho * u is unchanged.
        if (options.adaptive_rho && iteration % 10 == 0) {
            constexpr double balance = 10.0;
            if (out.primal_norm > balance * out.dual_norm && rho < 1e8) {
                rho *= 2.0;
                for (double& value : u) value *= 0.5;
            } else if (out.dual_norm > balance * out.primal_norm && rho > 1e-8) {
                rho *= 0.5;
                for (double& value : u) value *= 2.0;
            }
        }
    }
    out.dual.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        out.dual[i] = rho * u[i] / w[i];
        if (r[i] == 0.0) out.basis.push_back(i);
    }
    return out;
}

/// @brief Picks p linearly independent rows, preferring those with the smallest absolute OLS
///        residual, by incremental Gram-Schmidt.
/// @param x  Full-rank design.
/// @param y  Responses.
/// @return p row indices.
std::vector<std::size_t> starting_basis(const Matrix& x, const std::vector<double>& y) {
    const std::size_t n = x.rows(), p = x.cols();
    // Least squares by the normal equations: only an ordering heuristic, so their conditioning
    // does not matter, and they cost O(n p^2) where a full QR of the n x p design is O(n^2 p).
    Matrix              gram(p, p, 0.0);
    std::vector<double> moment(p, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t a = 0; a < p; ++a) {
            moment[a] += x(i, a) * y[i];
            for (std::size_t b = 0; b < p; ++b) gram(a, b) += x(i, a) * x(i, b);
        }
    const auto normal = linalg::lu(gram);
    if (normal.singular) throw std::runtime_error("QuantileRegression: design matrix is rank-deficient");
    const auto fit = x * normal.solve(moment);
    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return std::fabs(y[a] - fit[a]) < std::fabs(y[b] - fit[b]);
    });
    std::vector<std::vector<double>> q;
    std::vector<std::size_t>         basis;
    for (const std::size_t i : order) {
        std::vector<double> v = x.row(i);
        const double        size = l2_norm(v);
        for (int pass = 0; pass < 2; ++pass)
            for (const auto& e : q) {
                double dot = 0.0;
                for (std::size_t k = 0; k < p; ++k) dot += v[k] * e[k];
                for (std::size_t k = 0; k < p; ++k) v[k] -= dot * e[k];
            }
        const double rest = l2_norm(v);
        if (!(rest > 1e-8 * size)) continue;
        for (double& value : v) value /= rest;
        q.push_back(std::move(v));
        basis.push_back(i);
        if (basis.size() == p) break;
    }
    if (basis.size() < p) throw std::runtime_error("QuantileRegression: design matrix is rank-deficient");
    return basis;
}

/// @brief Exact vertex descent on the quantile-regression linear program.
///
/// At a vertex h (p interpolated rows) each edge frees one basis row j: along
/// \f$\beta+td_j\f$ with \f$X_hd_j=e_j\f$ the loss is convex and piecewise linear in t, so its minimiser is a
/// breakpoint \f$t_i=r_i/(x_i^\top d_j)\f$ found by one weighted-median-like sweep. The best edge is
/// taken until none lowers the loss.
/// @param x  Full-rank design.
/// @param original_y  Responses.
/// @param w  Weights.
/// @param tau  Quantile level.
/// @param max_iterations  Pivot cap.
/// @return The fit, with the exact dual solution.
SolverResult simplex_fit(const Matrix& x, const std::vector<double>& original_y, const std::vector<double>& w,
                         double tau, std::size_t max_iterations) {
    const std::size_t n = x.rows(), p = x.cols();
    SolverResult      out;
    // Ties (rounded responses, duplicated rows) make vertices degenerate: more than p rows on the
    // fit, where edge descent can stall short of the optimum. A tiny fixed perturbation of y
    // removes them; the optimal basis of the perturbed problem is then checked against the
    // original data below.
    double scale = 1.0;
    for (const double value : original_y) scale = std::max(scale, std::fabs(value));
    std::vector<double> y = original_y;
    {
        random::SplitMix64 jitter(0x9e3779b97f4a7c15ULL);
        for (double& value : y)
            value += 1e-9 * scale * (static_cast<double>(jitter.next_u64() >> 11) * 0x1.0p-53 - 0.5);
    }
    std::vector<std::size_t> h = starting_basis(x, y);
    std::vector<char>        in_basis(n, 0);
    std::vector<double>      r(n), a(n);
    struct Breakpoint {
        double      t;
        double      jump;
        std::size_t row;
    };
    std::vector<Breakpoint> points;
    std::vector<double>     slopes;
    points.reserve(n);
    bool stalled = false;
    Matrix basis_inverse;
    for (;;) {
        Matrix xh(p, p);
        std::vector<double> yh(p);
        for (std::size_t k = 0; k < p; ++k) {
            for (std::size_t c = 0; c < p; ++c) xh(k, c) = x(h[k], c);
            yh[k] = y[h[k]];
        }
        const auto factor = linalg::lu(xh);
        if (factor.singular) throw std::runtime_error("QuantileRegression: singular simplex basis");
        out.beta      = factor.solve(yh);
        basis_inverse = factor.inverse();
        std::fill(in_basis.begin(), in_basis.end(), 0);
        for (const std::size_t i : h) in_basis[i] = 1;
        const auto fit = x * out.beta;
        double     f0  = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            r[i] = in_basis[i] ? 0.0 : y[i] - fit[i];
            f0  += w[i] * check_loss(r[i], tau);
        }
        if (stalled || out.iterations >= max_iterations) break;

        double      best_decrease = 0.0;
        std::size_t best_j = p, best_row = n;
        for (std::size_t j = 0; j < p; ++j) {
            double norm_d = 0.0;
            for (std::size_t c = 0; c < p; ++c) norm_d += basis_inverse(c, j) * basis_inverse(c, j);
            norm_d = std::sqrt(norm_d);
            points.clear();
            double slope = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                if (in_basis[i] && i != h[j]) {
                    a[i] = 0.0;
                    continue;
                }
                double value = 0.0, size = 0.0;
                for (std::size_t c = 0; c < p; ++c) {
                    value += x(i, c) * basis_inverse(c, j);
                    size  += x(i, c) * x(i, c);
                }
                if (i == h[j]) value = 1.0;
                if (std::fabs(value) <= 1e-13 * std::sqrt(size) * norm_d) value = 0.0;
                a[i] = value;
                if (value == 0.0) continue;
                const double jump = w[i] * std::fabs(value);
                slope -= jump * (value > 0.0 ? tau : 1.0 - tau);
                points.push_back({r[i] / value, jump, i});
            }
            std::sort(points.begin(), points.end(),
                      [](const Breakpoint& l, const Breakpoint& q) { return l.t < q.t; });
            // Walk the slope up from its value at -infinity; the minimiser is the breakpoint where
            // it turns non-negative. The loss change is the integral of the slope from 0 to there,
            // which is exact to rounding (a difference of two sums is not).
            std::size_t       k = 0;
            std::vector<double>& before = slopes;
            before.resize(points.size());
            for (; k < points.size(); ++k) {
                before[k] = slope;
                slope += points[k].jump;
                if (slope >= 0.0) break;
            }
            if (k == points.size()) k = points.size() - 1;
            const Breakpoint& stop = points[k];
            if (stop.row == h[j] || stop.t == 0.0) continue;
            double change = 0.0;
            if (stop.t > 0.0) {
                for (std::size_t m = 0; m <= k; ++m)
                    if (points[m].t > 0.0) {
                        const double lo = m > 0 ? std::max(points[m - 1].t, 0.0) : 0.0;
                        change += before[m] * (points[m].t - lo);
                    }
            } else {
                // Slopes after breakpoints k, k+1, ... up to 0: recompute forward from before[k].
                double after = before[k] + points[k].jump;
                for (std::size_t m = k; m < points.size() && points[m].t < 0.0; ++m) {
                    if (m > k) after += points[m].jump;
                    const double hi = m + 1 < points.size() ? std::min(points[m + 1].t, 0.0) : 0.0;
                    change -= after * (hi - points[m].t);
                }
            }
            const double decrease = -change;
            if (decrease > best_decrease) {
                best_decrease = decrease;
                best_j        = j;
                best_row      = stop.row;
            }
        }
        if (best_j == p || !(best_decrease > 1e-15 * std::max(1.0, f0))) {
            stalled = true;
            continue; // recompute nothing new; exits at the top of the loop
        }
        h[best_j] = best_row;
        ++out.iterations;
    }

    // Dual: a_i = tau above the (perturbed) fit and tau - 1 below; the basis values solve
    // X_h^T (w_h a_h) = g := -sum_{i not in h} w_i a_i x_i. The vertex is optimal for the original
    // data if a_h lies in [tau - 1, tau] and every row clearly off the original fit has the sign
    // its dual value claims (rows the original fit passes through accept any value in the box).
    out.dual.assign(n, 0.0);
    std::vector<double> g(p, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        if (in_basis[i]) continue;
        out.dual[i] = r[i] > 0.0 ? tau : tau - 1.0;
        for (std::size_t c = 0; c < p; ++c) g[c] -= w[i] * out.dual[i] * x(i, c);
    }
    // (X_h^T)^{-1} = (X_h^{-1})^T.
    bool in_box = true;
    for (std::size_t k = 0; k < p; ++k) {
        double value = 0.0;
        for (std::size_t c = 0; c < p; ++c) value += basis_inverse(c, k) * g[c];
        value /= w[h[k]];
        out.dual[h[k]] = value;
        in_box = in_box && value >= tau - 1.0 - 1e-9 && value <= tau + 1e-9;
    }
    std::vector<double> yh(p);
    for (std::size_t k = 0; k < p; ++k) yh[k] = original_y[h[k]];
    Matrix xh(p, p);
    for (std::size_t k = 0; k < p; ++k)
        for (std::size_t c = 0; c < p; ++c) xh(k, c) = x(h[k], c);
    out.beta       = linalg::lu(xh).solve(yh);
    const auto fit = x * out.beta;
    for (std::size_t i = 0; i < n && in_box; ++i) {
        if (in_basis[i]) continue;
        const double residual = original_y[i] - fit[i];
        if (std::fabs(residual) <= 1e-8 * scale) continue;
        in_box = (residual > 0.0) == (out.dual[i] == tau);
    }
    out.converged = stalled && in_box;
    out.basis     = h;
    std::sort(out.basis.begin(), out.basis.end());
    return out;
}

/// @brief Runs the solver the options name.
/// @param x  Design.
/// @param y  Responses.
/// @param w  Weights.
/// @param tau  Quantile level.
/// @param options  Options (method and solver settings).
/// @return The fit.
SolverResult solve(const Matrix& x, const std::vector<double>& y, const std::vector<double>& w, double tau,
                   const QuantileRegressionOptions& options) {
    return options.method == QuantileRegressionMethod::Simplex ? simplex_fit(x, y, w, tau, options.max_iterations)
                                                               : admm_fit(x, y, w, tau, options);
}

/// @brief \f$(X^\top DX)^{-1}\f$ for a diagonal D.
/// @param x  Design.
/// @param d  Diagonal entries.
/// @return The p x p inverse.
Matrix weighted_cross_inverse(const Matrix& x, const std::vector<double>& d) {
    const std::size_t n = x.rows(), p = x.cols();
    Matrix            m(p, p, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t a = 0; a < p; ++a)
            for (std::size_t b = 0; b < p; ++b) m(a, b) += d[i] * x(i, a) * x(i, b);
    const auto factor = linalg::lu(m);
    if (factor.singular) throw std::runtime_error("QuantileRegression: singular information matrix");
    return factor.inverse();
}

/// @brief Formats a number with fixed precision.
/// @param value  The number.
/// @param digits  Digits after the point.
/// @return The text.
std::string fixed(double value, int digits) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(digits) << value;
    return out.str();
}

} // namespace

double check_loss(double residual, double tau) {
    return residual >= 0.0 ? tau * residual : (tau - 1.0) * residual;
}

double hall_sheather_bandwidth(double tau, std::size_t n, double alpha) {
    if (!(tau > 0.0 && tau < 1.0) || n == 0 || !(alpha > 0.0 && alpha < 1.0))
        throw std::invalid_argument("hall_sheather_bandwidth: need 0 < tau < 1, n > 0, 0 < alpha < 1");
    const double x0 = random::normal_quantile(tau);
    const double f0 = random::normal_pdf(x0);
    return std::pow(static_cast<double>(n), -1.0 / 3.0) * std::pow(random::normal_quantile(1.0 - alpha / 2.0), 2.0 / 3.0)
           * std::cbrt(1.5 * f0 * f0 / (2.0 * x0 * x0 + 1.0));
}

double bofinger_bandwidth(double tau, std::size_t n) {
    if (!(tau > 0.0 && tau < 1.0) || n == 0)
        throw std::invalid_argument("bofinger_bandwidth: need 0 < tau < 1 and n > 0");
    const double x0 = random::normal_quantile(tau);
    const double f0 = random::normal_pdf(x0);
    return std::pow(static_cast<double>(n), -0.2)
           * std::pow(4.5 * std::pow(f0, 4.0) / std::pow(2.0 * x0 * x0 + 1.0, 2.0), 0.2);
}

QuantileRegression::QuantileRegression(const dstruct::DataFrame& data, const std::string& formula,
                                       QuantileRegressionOptions options)
    : formula_(formula), options_(std::move(options)) {
    fit(data);
}

void QuantileRegression::fit(const dstruct::DataFrame& data) {
    const double tau = options_.quantile;
    if (!(tau > 0.0 && tau < 1.0))
        throw std::invalid_argument("QuantileRegression: quantile must be strictly between 0 and 1");
    if (!(options_.rho > 0.0))
        throw std::invalid_argument("QuantileRegression: rho must be strictly positive");
    if (options_.max_iterations == 0)
        throw std::invalid_argument("QuantileRegression: max_iterations must be positive");
    if (!(options_.absolute_tolerance > 0.0) || !(options_.relative_tolerance > 0.0))
        throw std::invalid_argument("QuantileRegression: tolerances must be strictly positive");

    design_   = formula_.resolve(data);
    auto raw  = design_.build_matrix(data, /*require_response=*/true);
    observations_   = raw.X.rows();
    const auto cols = raw.X.cols();
    if (observations_ == 0 || cols == 0)
        throw std::invalid_argument("QuantileRegression: the resolved design matrix is empty");
    if (observations_ < cols)
        throw std::invalid_argument("QuantileRegression: require at least as many observations as coefficients");

    weights_.assign(observations_, 1.0);
    if (options_.weights_column) {
        if (!data.has_column(*options_.weights_column))
            throw std::invalid_argument("QuantileRegression: weights column not found: "
                                        + *options_.weights_column);
        for (std::size_t i = 0; i < observations_; ++i) {
            const auto weight = data.optional_double_at(*options_.weights_column, raw.used_row_indices[i]);
            if (!weight)
                throw std::invalid_argument(
                    "QuantileRegression: weights column has a null value in a fitted row");
            if (!(*weight > 0.0) || !std::isfinite(*weight))
                throw std::invalid_argument("QuantileRegression: weights must be finite and strictly positive");
            weights_[i] = *weight;
        }
    }

    x_         = std::move(raw.X);
    y_         = std::move(raw.y);
    used_rows_ = std::move(raw.used_row_indices);

    auto result           = solve(x_, y_, weights_, tau, options_);
    coefficients_         = std::move(result.beta);
    basis_                = std::move(result.basis);
    dual_                 = std::move(result.dual);
    iterations_           = result.iterations;
    converged_            = result.converged;
    primal_residual_norm_ = result.primal_norm;
    dual_residual_norm_   = result.dual_norm;

    fitted_ = x_ * coefficients_;
    residuals_.resize(observations_);
    objective_ = 0.0;
    for (std::size_t i = 0; i < observations_; ++i) {
        residuals_[i] = y_[i] - fitted_[i];
        objective_ += weights_[i] * check_loss(residuals_[i], tau);
    }
    if (options_.method == QuantileRegressionMethod::Simplex)
        for (const std::size_t i : basis_) residuals_[i] = 0.0, fitted_[i] = y_[i];

    std::vector<std::pair<double, double>> response_and_weight(observations_);
    for (std::size_t i = 0; i < observations_; ++i) response_and_weight[i] = {y_[i], weights_[i]};
    const double null_quantile  = weighted_quantile(std::move(response_and_weight), tau);
    double       null_objective = 0.0;
    for (std::size_t i = 0; i < observations_; ++i)
        null_objective += weights_[i] * check_loss(y_[i] - null_quantile, tau);
    pseudo_r_squared_ =
        null_objective > 0.0 ? 1.0 - objective_ / null_objective : (objective_ == 0.0 ? 1.0 : 0.0);
}

void QuantileRegression::validate_categorical_levels(const dstruct::DataFrame& newdata) const {
    design_.validate_categorical_levels(newdata, "QuantileRegression::predict");
}

std::vector<double> QuantileRegression::predict(const dstruct::DataFrame& newdata) const {
    validate_categorical_levels(newdata);
    const auto matrix = design_.build_matrix(newdata, /*require_response=*/false);
    return matrix.X * coefficients_;
}

QuantileCoefficientTable QuantileRegression::coefficient_table(QuantileStandardErrors method, double level,
                                                               std::size_t replications, std::uint64_t seed) const {
    if (!(level > 0.0 && level < 1.0))
        throw std::invalid_argument("QuantileRegression::coefficient_table: level must be in (0, 1)");
    const std::size_t n = observations_, p = coefficients_.size();
    if (n <= p) throw std::invalid_argument("QuantileRegression::coefficient_table: need more rows than coefficients");
    const double tau = options_.quantile;

    QuantileCoefficientTable table;
    table.names              = design_.coefficient_names;
    table.estimate           = coefficients_;
    table.method             = method;
    table.level              = level;
    table.degrees_of_freedom = n - p;
    std::vector<double> variance(p, 0.0);

    if (method == QuantileStandardErrors::Bootstrap) {
        if (replications < 2)
            throw std::invalid_argument("QuantileRegression::coefficient_table: need at least 2 replications");
        table.replications = replications;
        random::SplitMix64               rng(seed);
        std::vector<std::vector<double>> draws;
        Matrix                           xb(n, p);
        std::vector<double>              yb(n), wb(n);
        std::size_t                      attempts = 0;
        while (draws.size() < replications) {
            if (++attempts > 20 * replications)
                throw std::runtime_error("QuantileRegression::coefficient_table: too many rank-deficient resamples");
            for (std::size_t i = 0; i < n; ++i) {
                const auto row = static_cast<std::size_t>(rng.next_u64() % n);
                for (std::size_t c = 0; c < p; ++c) xb(i, c) = x_(row, c);
                yb[i] = y_[row];
                wb[i] = weights_[row];
            }
            try {
                draws.push_back(solve(xb, yb, wb, tau, options_).beta);
            } catch (const std::runtime_error&) {
                continue; // rank-deficient resample
            }
        }
        for (std::size_t j = 0; j < p; ++j) {
            double mean = 0.0;
            for (const auto& d : draws) mean += d[j] / static_cast<double>(draws.size());
            for (const auto& d : draws) variance[j] += (d[j] - mean) * (d[j] - mean);
            variance[j] /= static_cast<double>(draws.size() - 1);
        }
    } else {
        double h = hall_sheather_bandwidth(tau, n);
        h        = std::min(h, 0.99 * std::min(tau, 1.0 - tau));
        table.bandwidth = h;
        if (method == QuantileStandardErrors::IID) {
            // quantreg's recipe: the ceil(n h) residuals nearest zero (skipping exact zeros),
            // sorted, regressed by median regression on rank / (n - p); the slope is the sparsity.
            std::vector<std::size_t> order(n);
            std::iota(order.begin(), order.end(), 0);
            std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
                return std::fabs(residuals_[a]) < std::fabs(residuals_[b]);
            });
            double scale = 1.0;
            for (const double value : y_) scale = std::max(scale, 1.0 + std::fabs(value));
            std::size_t zeros = 0;
            for (const double r : residuals_) zeros += std::fabs(r) <= 1e-8 * scale;
            const std::size_t m = std::max<std::size_t>(p + 1, static_cast<std::size_t>(std::ceil(static_cast<double>(n) * h)));
            if (zeros + m + 1 > n)
                throw std::invalid_argument("QuantileRegression::coefficient_table: too few rows for the IID sparsity");
            std::vector<double> ordered;
            for (std::size_t k = zeros; k <= zeros + m; ++k) ordered.push_back(residuals_[order[k]]);
            std::sort(ordered.begin(), ordered.end());
            Matrix              xs(ordered.size(), 2);
            std::vector<double> ones(ordered.size(), 1.0);
            for (std::size_t k = 0; k < ordered.size(); ++k) {
                xs(k, 0) = 1.0;
                xs(k, 1) = static_cast<double>(zeros + k + 1) / static_cast<double>(n - p);
            }
            const double sparsity = simplex_fit(xs, ordered, ones, 0.5, 10000).beta[1];
            const auto   inverse  = weighted_cross_inverse(x_, weights_);
            for (std::size_t j = 0; j < p; ++j) variance[j] = tau * (1.0 - tau) * sparsity * sparsity * inverse(j, j);
        } else {
            auto low = options_, high = options_;
            low.quantile  = tau - h;
            high.quantile = tau + h;
            const auto beta_low  = solve(x_, y_, weights_, low.quantile, low).beta;
            const auto beta_high = solve(x_, y_, weights_, high.quantile, high).beta;
            std::vector<double> diff(p), density(n), fw(n);
            for (std::size_t j = 0; j < p; ++j) diff[j] = beta_high[j] - beta_low[j];
            const auto   dyhat = x_ * diff;
            const double eps   = std::sqrt(std::numeric_limits<double>::epsilon());
            for (std::size_t i = 0; i < n; ++i) {
                density[i] = std::max(0.0, 2.0 * h / (dyhat[i] - eps));
                fw[i]      = density[i] * weights_[i];
            }
            const auto hinv = weighted_cross_inverse(x_, fw);
            Matrix     j_matrix(p, p, 0.0);
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t a = 0; a < p; ++a)
                    for (std::size_t b = 0; b < p; ++b) j_matrix(a, b) += weights_[i] * x_(i, a) * x_(i, b);
            const auto cov = hinv * j_matrix * hinv;
            for (std::size_t j = 0; j < p; ++j) variance[j] = tau * (1.0 - tau) * cov(j, j);
        }
    }

    const double df = static_cast<double>(n - p);
    const double t  = random::student_t_quantile(0.5 + level / 2.0, df);
    for (std::size_t j = 0; j < p; ++j) {
        const double se = std::sqrt(variance[j]);
        table.std_error.push_back(se);
        table.t_value.push_back(coefficients_[j] / se);
        table.p_value.push_back(2.0 * (1.0 - random::student_t_cdf(std::fabs(coefficients_[j] / se), df)));
        table.lower.push_back(coefficients_[j] - t * se);
        table.upper.push_back(coefficients_[j] + t * se);
    }
    return table;
}

std::string QuantileRegression::summary() const {
    std::ostringstream out;
    out << "Call:\nquantile_regression(formula = " << formula_.text()
        << ", quantile = " << options_.quantile << ")\n\n";
    out << "Coefficients:\n";
    out << std::left << std::setw(22) << "" << std::right << std::setw(14) << "Estimate" << "\n";
    for (std::size_t i = 0; i < coefficients_.size(); ++i)
        out << std::left << std::setw(22) << design_.coefficient_names[i] << std::right << std::setw(14)
            << fixed(coefficients_[i], 6) << "\n";
    out << "\nQuantile: " << options_.quantile << "\n";
    out << "Objective (check loss): " << fixed(objective_, 6) << "\n";
    out << "Pseudo R-squared: " << fixed(pseudo_r_squared_, 6) << "\n";
    if (options_.method == QuantileRegressionMethod::Simplex)
        out << "Simplex pivots: " << iterations_ << "\n";
    else
        out << "ADMM iterations: " << iterations_ << "\n";
    out << "Converged: " << (converged_ ? "yes" : "no") << "\n";
    return out.str();
}

void QuantileRegression::print_summary(std::ostream& os) const { os << summary(); }

void QuantileRegression::print_summary() const { print_summary(std::cout); }

QuantileProcess quantile_process(const dstruct::DataFrame& data, const std::string& formula,
                                 const std::vector<double>& quantiles, QuantileRegressionOptions options,
                                 QuantileStandardErrors method, double level) {
    if (quantiles.empty()) throw std::invalid_argument("quantile_process: no quantile levels given");
    QuantileProcess process;
    process.quantiles = quantiles;
    process.level     = level;
    for (const double tau : quantiles) {
        options.quantile = tau;
        const QuantileRegression model(data, formula, options);
        const auto               table = model.coefficient_table(method, level);
        if (process.names.empty()) process.names = table.names;
        process.coefficients.push_back(table.estimate);
        process.std_error.push_back(table.std_error);
        process.lower.push_back(table.lower);
        process.upper.push_back(table.upper);
    }
    return process;
}

} // namespace datamunge::stats
