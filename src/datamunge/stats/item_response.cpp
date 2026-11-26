/// @file item_response.cpp
/// @brief Marginal maximum likelihood for Rasch and 2PL models, information and ability scores.
#include <datamunge/stats/item_response.hpp>

#include <datamunge/linalg/cholesky.hpp>
#include <datamunge/stats/gauss_hermite.hpp>

#include <algorithm>
#include <cmath>
#include <map>
#include <limits>
#include <stdexcept>
#include <string>

namespace datamunge::stats {

/// @brief Implementation helpers for item-response models.
namespace {
/// @brief Dense matrix alias.
using Matrix = linalg::DenseMatrix<double>;

/// @brief Response patterns with their frequencies.
struct Patterns {
    std::vector<std::vector<unsigned char>> x; ///< Distinct patterns.
    std::vector<double> count;                 ///< Their frequencies.
};

/// @brief Validates 0/1 data and groups identical rows.
/// @param r Responses.
/// @param fn Caller name.
/// @return Patterns.
Patterns group_patterns(const Matrix& r, const char* fn) {
    std::map<std::vector<unsigned char>, double> m;
    for (std::size_t i = 0; i < r.rows(); ++i) {
        std::vector<unsigned char> row(r.cols());
        for (std::size_t j = 0; j < r.cols(); ++j) {
            if (r(i, j) != 0.0 && r(i, j) != 1.0)
                throw std::invalid_argument(std::string(fn) + ": responses must be 0 or 1");
            row[j] = static_cast<unsigned char>(r(i, j));
        }
        m[row] += 1;
    }
    Patterns p;
    for (const auto& [row, c] : m)
        p.x.push_back(row), p.count.push_back(c);
    return p;
}

/// @brief Numerically stable log(1 + exp(x)).
/// @param x Argument.
/// @return softplus(x).
double softplus(double x) { return x > 0 ? x + std::log1p(std::exp(-x)) : std::log1p(std::exp(x)); }

/// @brief The marginal likelihood as a function of the slope–intercept parameter vector.
struct Marginal {
    const Patterns& pat;          ///< Data.
    const GaussHermiteRule& rule; ///< Quadrature.
    std::size_t p;                ///< Items.
    IrtModel model;               ///< Model.
    double fixed_a;               ///< Fixed common slope, or NaN.

    /// @brief Slope of item j under parameter vector v.
    /// @param v Parameters.
    /// @param j Item.
    /// @return a_j.
    [[nodiscard]] double slope(const std::vector<double>& v, std::size_t j) const {
        if (model == IrtModel::two_parameter)
            return v[p + j];
        return std::isnan(fixed_a) ? v[p] : fixed_a;
    }

    /// @brief Log-likelihood and (optionally) gradient.
    /// @param v Parameters.
    /// @param grad Receives the gradient when non-null.
    /// @return log L.
    double evaluate(const std::vector<double>& v, std::vector<double>* grad) const {
        const std::size_t q = rule.nodes.size();
        // Per item and node: log P and log(1 - P).
        std::vector<double> lp(p * q), lq(p * q), prob(p * q);
        for (std::size_t j = 0; j < p; ++j)
            for (std::size_t k = 0; k < q; ++k) {
                const double eta = v[j] + slope(v, j) * rule.nodes[k];
                lp[j * q + k] = -softplus(-eta);
                lq[j * q + k] = -softplus(eta);
                prob[j * q + k] = std::exp(lp[j * q + k]);
            }
        if (grad)
            grad->assign(v.size(), 0.0);
        double ll = 0;
        std::vector<double> lw(q);
        for (std::size_t m = 0; m < pat.x.size(); ++m) {
            double top = -std::numeric_limits<double>::infinity();
            for (std::size_t k = 0; k < q; ++k) {
                double s = std::log(rule.weights[k]);
                for (std::size_t j = 0; j < p; ++j)
                    s += pat.x[m][j] ? lp[j * q + k] : lq[j * q + k];
                lw[k] = s;
                top = std::max(top, s);
            }
            double total = 0;
            for (std::size_t k = 0; k < q; ++k)
                total += std::exp(lw[k] - top);
            const double log_lik = top + std::log(total);
            ll += pat.count[m] * log_lik;
            if (!grad)
                continue;
            for (std::size_t k = 0; k < q; ++k) {
                const double post = pat.count[m] * std::exp(lw[k] - log_lik);
                for (std::size_t j = 0; j < p; ++j) {
                    const double r = post * (pat.x[m][j] - prob[j * q + k]);
                    (*grad)[j] += r;
                    if (model == IrtModel::two_parameter)
                        (*grad)[p + j] += r * rule.nodes[k];
                    else if (std::isnan(fixed_a))
                        (*grad)[p] += r * rule.nodes[k];
                }
            }
        }
        return ll;
    }

};

/// @brief Finite-difference Hessian of an analytic gradient (symmetrized).
/// @tparam F Type with `double evaluate(const std::vector<double>&, std::vector<double>*) const`.
/// @param f Objective.
/// @param v Parameters.
/// @return Hessian.
template <class F>
Matrix fd_hessian(const F& f, const std::vector<double>& v) {
    const std::size_t d = v.size();
    Matrix h(d, d, 0.0);
    std::vector<double> gp, gm;
    for (std::size_t i = 0; i < d; ++i) {
        const double step = 1e-5 * std::max(1.0, std::abs(v[i]));
        auto up = v, down = v;
        up[i] += step, down[i] -= step;
        f.evaluate(up, &gp), f.evaluate(down, &gm);
        for (std::size_t k = 0; k < d; ++k)
            h(k, i) = (gp[k] - gm[k]) / (2 * step);
    }
    for (std::size_t i = 0; i < d; ++i)
        for (std::size_t k = 0; k < i; ++k)
            h(i, k) = h(k, i) = (h(i, k) + h(k, i)) / 2;
    return h;
}

/// @brief Outcome of newton_maximize().
struct NewtonResult {
    double log_likelihood{}; ///< Final objective.
    std::size_t iterations{}; ///< Iterations used.
    bool converged{};         ///< Gradient criterion met.
};

/// @brief Damped (Levenberg–Marquardt) Newton ascent with step acceptance on the objective.
/// @tparam F Objective type (see fd_hessian()).
/// @param f Objective.
/// @param v Parameters, updated in place.
/// @param n Number of respondents (scales the gradient criterion).
/// @param max_iterations Iteration limit.
/// @param tolerance Gradient tolerance per respondent.
/// @return Result.
template <class F>
NewtonResult newton_maximize(const F& f, std::vector<double>& v, std::size_t n, std::size_t max_iterations,
                             double tolerance) {
    const std::size_t d = v.size();
    NewtonResult res;
    std::vector<double> g;
    double ll = f.evaluate(v, &g), mu = 1e-6;
    for (res.iterations = 0; res.iterations < max_iterations; ++res.iterations) {
        double gmax = 0;
        for (const double x : g)
            gmax = std::max(gmax, std::abs(x));
        if (gmax / static_cast<double>(n) < tolerance) {
            res.converged = true;
            break;
        }
        const Matrix h = fd_hessian(f, v);
        bool accepted = false;
        for (int attempt = 0; attempt < 40 && !accepted; ++attempt) {
            Matrix a(d, d, 0.0);
            for (std::size_t i = 0; i < d; ++i)
                for (std::size_t k = 0; k < d; ++k)
                    a(i, k) = -h(i, k) + (i == k ? mu * std::max(1.0, -h(i, i)) : 0.0);
            const auto c = linalg::cholesky(a);
            if (!c.ok) {
                mu = std::max(mu * 10, 1e-4);
                continue;
            }
            const auto step = c.solve(g);
            auto trial = v;
            for (std::size_t i = 0; i < d; ++i)
                trial[i] += step[i];
            std::vector<double> gt;
            const double lt = f.evaluate(trial, &gt);
            if (std::isfinite(lt) && lt >= ll - 1e-10 * std::abs(ll)) {
                v = trial, g = gt, ll = lt, accepted = true;
                mu = std::max(mu / 10, 1e-12);
            } else {
                mu = std::max(mu * 10, 1e-4);
            }
        }
        if (!accepted)
            break;
    }
    res.log_likelihood = ll;
    return res;
}

/// @brief Covariance of the estimates: inverse of the negative finite-difference Hessian.
/// @tparam F Objective type.
/// @param f Objective.
/// @param v Estimates.
/// @param ok Set to false when the information is not positive definite.
/// @return Covariance.
template <class F>
Matrix estimate_covariance(const F& f, const std::vector<double>& v, bool& ok);

/// @brief Inverse of a symmetric positive-definite matrix via Cholesky.
/// @param a Matrix.
/// @param ok Set to false if not positive definite.
/// @return Inverse (unspecified when !ok).
Matrix spd_inverse(const Matrix& a, bool& ok) {
    const std::size_t d = a.rows();
    const auto c = linalg::cholesky(a);
    ok = c.ok;
    Matrix inv(d, d, 0.0);
    if (!ok)
        return inv;
    for (std::size_t i = 0; i < d; ++i) {
        std::vector<double> e(d, 0.0);
        e[i] = 1;
        const auto col = c.solve(e);
        for (std::size_t k = 0; k < d; ++k)
            inv(k, i) = col[k];
    }
    return inv;
}
/// @brief Graded response patterns, categories per item and the marginal likelihood.
struct GradedMarginal {
    std::vector<std::vector<int>> x;   ///< Distinct response patterns.
    std::vector<double> count;         ///< Their frequencies.
    std::vector<std::size_t> cats;     ///< Number of categories per item.
    std::vector<std::size_t> offset;   ///< Offset of each item's intercepts in the parameter vector.
    std::size_t intercepts{};          ///< Total number of intercepts.
    bool common{};                     ///< One shared slope.
    GaussHermiteRule rule;             ///< Quadrature.

    /// @brief Slope of item j.
    /// @param v Parameters.
    /// @param j Item.
    /// @return a_j.
    [[nodiscard]] double slope(const std::vector<double>& v, std::size_t j) const {
        return common ? v[intercepts] : v[intercepts + j];
    }

    /// @brief Index of item j's slope in the parameter vector.
    /// @param j Item.
    /// @return Index.
    [[nodiscard]] std::size_t slope_index(std::size_t j) const { return common ? intercepts : intercepts + j; }

    /// @brief Log-likelihood and optional gradient (slope–intercept form, \f$P(X\ge k)=\sigma(d_k+a\theta)\f$).
    /// @param v Parameters.
    /// @param grad Receives the gradient when non-null.
    /// @return log L, or -infinity when intercepts are out of order.
    double evaluate(const std::vector<double>& v, std::vector<double>* grad) const {
        const std::size_t p = cats.size(), q = rule.nodes.size();
        // For item j, node k, category c: log P and the derivative pieces s_c = P*_c (1 - P*_c).
        std::vector<std::vector<double>> logp(p), sv(p), cp(p);
        for (std::size_t j = 0; j < p; ++j) {
            const std::size_t kj = cats[j];
            logp[j].assign(q * kj, 0.0);
            sv[j].assign(q * (kj + 1), 0.0);
            cp[j].assign(q * kj, 0.0);
            for (std::size_t c = 1; c + 1 < kj; ++c)
                if (!(v[offset[j] + c] < v[offset[j] + c - 1]))
                    return -std::numeric_limits<double>::infinity();
            for (std::size_t k = 0; k < q; ++k) {
                std::vector<double> star(kj + 1, 0.0);
                star[0] = 1;
                for (std::size_t c = 1; c < kj; ++c) {
                    star[c] = 1 / (1 + std::exp(-(v[offset[j] + c - 1] + slope(v, j) * rule.nodes[k])));
                    sv[j][k * (kj + 1) + c] = star[c] * (1 - star[c]);
                }
                for (std::size_t c = 0; c < kj; ++c) {
                    const double pr = star[c] - star[c + 1];
                    if (!(pr > 0))
                        return -std::numeric_limits<double>::infinity();
                    cp[j][k * kj + c] = pr;
                    logp[j][k * kj + c] = std::log(pr);
                }
            }
        }
        if (grad)
            grad->assign(v.size(), 0.0);
        double ll = 0;
        std::vector<double> lw(q);
        for (std::size_t m = 0; m < x.size(); ++m) {
            double top = -std::numeric_limits<double>::infinity();
            for (std::size_t k = 0; k < q; ++k) {
                double s = std::log(rule.weights[k]);
                for (std::size_t j = 0; j < p; ++j)
                    s += logp[j][k * cats[j] + static_cast<std::size_t>(x[m][j])];
                lw[k] = s;
                top = std::max(top, s);
            }
            double total = 0;
            for (std::size_t k = 0; k < q; ++k)
                total += std::exp(lw[k] - top);
            const double log_lik = top + std::log(total);
            ll += count[m] * log_lik;
            if (!grad)
                continue;
            for (std::size_t k = 0; k < q; ++k) {
                const double post = count[m] * std::exp(lw[k] - log_lik);
                for (std::size_t j = 0; j < p; ++j) {
                    const std::size_t kj = cats[j], c = static_cast<std::size_t>(x[m][j]);
                    const double pr = cp[j][k * kj + c];
                    const double lo = sv[j][k * (kj + 1) + c], hi = sv[j][k * (kj + 1) + c + 1];
                    if (c >= 1)
                        (*grad)[offset[j] + c - 1] += post * lo / pr;
                    if (c + 1 < kj)
                        (*grad)[offset[j] + c] -= post * hi / pr;
                    (*grad)[slope_index(j)] += post * (lo - hi) * rule.nodes[k] / pr;
                }
            }
        }
        return ll;
    }
};

template <class F>
Matrix estimate_covariance(const F& f, const std::vector<double>& v, bool& ok) {
    const Matrix h = fd_hessian(f, v);
    Matrix neg(h.rows(), h.cols(), 0.0);
    for (std::size_t i = 0; i < h.rows(); ++i)
        for (std::size_t c = 0; c < h.cols(); ++c)
            neg(i, c) = -h(i, c);
    return spd_inverse(neg, ok);
}

} // namespace

IrtFit fit_irt(const linalg::DenseMatrix<double>& responses, const IrtOptions& options) {
    const std::size_t n = responses.rows(), p = responses.cols();
    if (p < 2 || n < 2)
        throw std::invalid_argument("fit_irt: need at least two items and two respondents");
    if (options.quadrature_points == 0)
        throw std::invalid_argument("fit_irt: need quadrature points");
    const bool fixed = options.model == IrtModel::rasch && !std::isnan(options.rasch_discrimination);
    if (fixed && !(options.rasch_discrimination > 0 && std::isfinite(options.rasch_discrimination)))
        throw std::invalid_argument("fit_irt: the fixed discrimination must be positive");
    const Patterns pat = group_patterns(responses, "fit_irt");
    const auto rule = normal_gauss_hermite(options.quadrature_points);
    const Marginal model{pat, rule, p, options.model, fixed ? options.rasch_discrimination : std::nan("")};
    // Starting values: intercepts from the item p-values, slopes 1.
    std::vector<double> v(p, 0.0);
    for (std::size_t j = 0; j < p; ++j) {
        double s = 0;
        for (std::size_t i = 0; i < n; ++i)
            s += responses(i, j);
        if (s == 0 || s == static_cast<double>(n))
            throw std::invalid_argument("fit_irt: every item needs both correct and incorrect responses");
        const double pj = s / static_cast<double>(n);
        v[j] = 1.16 * std::log(pj / (1 - pj)) * (fixed ? options.rasch_discrimination : 1.0);
    }
    if (options.model == IrtModel::two_parameter)
        v.resize(2 * p, 1.0);
    else if (!fixed)
        v.push_back(1.0);
    const std::size_t d = v.size();
    IrtFit fit;
    fit.model = options.model;
    fit.n = n;
    fit.parameters = d;
    fit.quadrature_points = options.quadrature_points;
    const auto res = newton_maximize(model, v, n, options.max_iterations, options.tolerance);
    fit.iterations = res.iterations;
    fit.converged = res.converged;
    const double ll = res.log_likelihood;
    fit.log_likelihood = ll;
    const double k = static_cast<double>(d);
    fit.aic = -2 * ll + 2 * k;
    fit.bic = -2 * ll + k * std::log(static_cast<double>(n));
    bool ok = false;
    const Matrix cov = estimate_covariance(model, v, ok);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (std::size_t j = 0; j < p; ++j) {
        IrtItem item;
        const double a = model.slope(v, j);
        item.discrimination = a;
        item.difficulty = -v[j] / a;
        item.difficulty_se = item.discrimination_se = nan;
        if (ok) {
            const std::size_t ai = options.model == IrtModel::two_parameter ? p + j : p;
            if (fixed) {
                item.difficulty_se = std::sqrt(cov(j, j)) / a;
            } else {
                // b = -c / a: gradient (-1/a, c/a^2).
                const double gc = -1 / a, ga = v[j] / (a * a);
                item.difficulty_se =
                    std::sqrt(gc * gc * cov(j, j) + ga * ga * cov(ai, ai) + 2 * gc * ga * cov(j, ai));
                item.discrimination_se = std::sqrt(cov(ai, ai));
            }
        }
        fit.items.push_back(item);
    }
    return fit;
}

double irt_probability(const IrtItem& item, double theta) {
    return 1 / (1 + std::exp(-item.discrimination * (theta - item.difficulty)));
}

double irt_test_information(const IrtFit& fit, double theta) {
    double info = 0;
    for (const auto& item : fit.items) {
        const double pr = irt_probability(item, theta);
        info += item.discrimination * item.discrimination * pr * (1 - pr);
    }
    return info;
}

std::vector<IrtScore> irt_scores(const IrtFit& fit, const linalg::DenseMatrix<double>& responses,
                                 std::size_t quadrature_points) {
    const std::size_t p = fit.items.size();
    if (responses.cols() != p)
        throw std::invalid_argument("irt_scores: item count differs from the fit");
    const std::size_t q = quadrature_points ? quadrature_points : std::max<std::size_t>(fit.quadrature_points, 1);
    const auto rule = normal_gauss_hermite(q);
    std::vector<IrtScore> out;
    for (std::size_t i = 0; i < responses.rows(); ++i) {
        for (std::size_t j = 0; j < p; ++j)
            if (responses(i, j) != 0.0 && responses(i, j) != 1.0)
                throw std::invalid_argument("irt_scores: responses must be 0 or 1");
        IrtScore s;
        // MAP: Newton on the log posterior, which is strictly concave.
        double t = 0, curv = 1;
        for (int it = 0; it < 200; ++it) {
            double d1 = -t;
            curv = 1;
            for (std::size_t j = 0; j < p; ++j) {
                const double pr = irt_probability(fit.items[j], t), a = fit.items[j].discrimination;
                d1 += a * (responses(i, j) - pr);
                curv += a * a * pr * (1 - pr);
            }
            const double step = std::clamp(d1 / curv, -2.0, 2.0);
            t += step;
            if (std::abs(step) < 1e-12)
                break;
        }
        curv = 1;
        for (std::size_t j = 0; j < p; ++j) {
            const double pr = irt_probability(fit.items[j], t), a = fit.items[j].discrimination;
            curv += a * a * pr * (1 - pr);
        }
        s.map = t;
        s.map_se = 1 / std::sqrt(curv);
        // EAP by quadrature (log scale).
        std::vector<double> lw(q);
        double top = -std::numeric_limits<double>::infinity();
        for (std::size_t k = 0; k < q; ++k) {
            double l = std::log(rule.weights[k]);
            for (std::size_t j = 0; j < p; ++j) {
                const double pr = irt_probability(fit.items[j], rule.nodes[k]);
                l += responses(i, j) == 1.0 ? std::log(pr) : std::log1p(-pr);
            }
            lw[k] = l;
            top = std::max(top, l);
        }
        double w = 0, m1 = 0, m2 = 0;
        for (std::size_t k = 0; k < q; ++k) {
            const double e = std::exp(lw[k] - top);
            w += e, m1 += e * rule.nodes[k], m2 += e * rule.nodes[k] * rule.nodes[k];
        }
        s.eap = m1 / w;
        s.eap_sd = std::sqrt(std::max(0.0, m2 / w - s.eap * s.eap));
        out.push_back(s);
    }
    return out;
}

GradedFit fit_graded_response(const linalg::DenseMatrix<double>& responses, const GradedOptions& options) {
    const std::size_t n = responses.rows(), p = responses.cols();
    if (p < 2 || n < 2)
        throw std::invalid_argument("fit_graded_response: need at least two items and two respondents");
    if (options.quadrature_points == 0)
        throw std::invalid_argument("fit_graded_response: need quadrature points");
    GradedMarginal model;
    model.common = options.common_discrimination;
    model.rule = normal_gauss_hermite(options.quadrature_points);
    model.cats.assign(p, 0);
    std::vector<std::vector<double>> freq(p);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < p; ++j) {
            const double v = responses(i, j);
            if (!(v >= 0) || v != std::floor(v) || v > 1000)
                throw std::invalid_argument("fit_graded_response: categories must be integers 0, 1, ...");
            const auto c = static_cast<std::size_t>(v);
            if (c >= freq[j].size())
                freq[j].resize(c + 1, 0.0);
            freq[j][c] += 1;
        }
    std::vector<double> v;
    for (std::size_t j = 0; j < p; ++j) {
        model.cats[j] = freq[j].size();
        if (model.cats[j] < 2 || std::ranges::any_of(freq[j], [](double f) { return f == 0; }))
            throw std::invalid_argument(
                "fit_graded_response: each item needs two or more categories, all observed");
        model.offset.push_back(v.size());
        double above = static_cast<double>(n);
        for (std::size_t c = 1; c < model.cats[j]; ++c) {
            above -= freq[j][c - 1];
            const double pr = above / static_cast<double>(n);
            v.push_back(1.16 * std::log(pr / (1 - pr)));
        }
    }
    model.intercepts = v.size();
    v.resize(v.size() + (model.common ? 1 : p), 1.0);
    std::map<std::vector<int>, double> pats;
    for (std::size_t i = 0; i < n; ++i) {
        std::vector<int> row(p);
        for (std::size_t j = 0; j < p; ++j)
            row[j] = static_cast<int>(responses(i, j));
        pats[row] += 1;
    }
    for (const auto& [row, c] : pats)
        model.x.push_back(row), model.count.push_back(c);
    GradedFit fit;
    fit.n = n;
    fit.parameters = v.size();
    fit.quadrature_points = options.quadrature_points;
    const auto res = newton_maximize(model, v, n, options.max_iterations, options.tolerance);
    fit.iterations = res.iterations;
    fit.converged = res.converged;
    fit.log_likelihood = res.log_likelihood;
    const double k = static_cast<double>(v.size());
    fit.aic = -2 * fit.log_likelihood + 2 * k;
    fit.bic = -2 * fit.log_likelihood + k * std::log(static_cast<double>(n));
    bool ok = false;
    const Matrix cov = estimate_covariance(model, v, ok);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (std::size_t j = 0; j < p; ++j) {
        GradedItem item;
        const double a = model.slope(v, j);
        const std::size_t ai = model.slope_index(j);
        item.discrimination = a;
        item.discrimination_se = ok ? std::sqrt(cov(ai, ai)) : nan;
        for (std::size_t c = 0; c + 1 < model.cats[j]; ++c) {
            const std::size_t di = model.offset[j] + c;
            item.thresholds.push_back(-v[di] / a);
            const double gd = -1 / a, ga = v[di] / (a * a);
            item.threshold_se.push_back(
                ok ? std::sqrt(gd * gd * cov(di, di) + ga * ga * cov(ai, ai) + 2 * gd * ga * cov(di, ai)) : nan);
        }
        fit.items.push_back(item);
    }
    return fit;
}

std::vector<double> graded_category_probabilities(const GradedItem& item, double theta) {
    const std::size_t kj = item.thresholds.size() + 1;
    std::vector<double> star(kj + 1, 0.0), out(kj);
    star[0] = 1;
    for (std::size_t c = 1; c < kj; ++c)
        star[c] = 1 / (1 + std::exp(-item.discrimination * (theta - item.thresholds[c - 1])));
    for (std::size_t c = 0; c < kj; ++c)
        out[c] = star[c] - star[c + 1];
    return out;
}

double graded_item_information(const GradedItem& item, double theta) {
    const std::size_t kj = item.thresholds.size() + 1;
    std::vector<double> s(kj + 1, 0.0);
    for (std::size_t c = 1; c < kj; ++c) {
        const double st = 1 / (1 + std::exp(-item.discrimination * (theta - item.thresholds[c - 1])));
        s[c] = item.discrimination * st * (1 - st);
    }
    const auto pr = graded_category_probabilities(item, theta);
    double info = 0;
    for (std::size_t c = 0; c < kj; ++c)
        if (pr[c] > 0)
            info += (s[c] - s[c + 1]) * (s[c] - s[c + 1]) / pr[c];
    return info;
}

double graded_test_information(const GradedFit& fit, double theta) {
    double info = 0;
    for (const auto& item : fit.items)
        info += graded_item_information(item, theta);
    return info;
}

std::vector<IrtScore> graded_scores(const GradedFit& fit, const linalg::DenseMatrix<double>& responses,
                                    std::size_t quadrature_points) {
    const std::size_t p = fit.items.size();
    if (responses.cols() != p)
        throw std::invalid_argument("graded_scores: item count differs from the fit");
    const std::size_t q = quadrature_points ? quadrature_points : std::max<std::size_t>(fit.quadrature_points, 1);
    const auto rule = normal_gauss_hermite(q);
    std::vector<IrtScore> out;
    for (std::size_t i = 0; i < responses.rows(); ++i) {
        std::vector<std::size_t> c(p);
        for (std::size_t j = 0; j < p; ++j) {
            const double v = responses(i, j);
            if (!(v >= 0) || v != std::floor(v) || v > static_cast<double>(fit.items[j].thresholds.size()))
                throw std::invalid_argument("graded_scores: category out of range");
            c[j] = static_cast<std::size_t>(v);
        }
        const auto log_post = [&](double t) {
            double l = -t * t / 2;
            for (std::size_t j = 0; j < p; ++j)
                l += std::log(graded_category_probabilities(fit.items[j], t)[c[j]]);
            return l;
        };
        // d/dtheta log P_c = (s_c - s_{c+1}) / P_c with s_c = a P*_c (1 - P*_c).
        const auto score = [&](double t) {
            double d = -t;
            for (std::size_t j = 0; j < p; ++j) {
                const auto& it = fit.items[j];
                const auto pr = graded_category_probabilities(it, t);
                auto s = [&](std::size_t k) {
                    if (k == 0 || k > it.thresholds.size())
                        return 0.0;
                    const double st = 1 / (1 + std::exp(-it.discrimination * (t - it.thresholds[k - 1])));
                    return it.discrimination * st * (1 - st);
                };
                d += (s(c[j]) - s(c[j] + 1)) / pr[c[j]];
            }
            return d;
        };
        IrtScore sc;
        double t = 0, curv = 1;
        for (int it = 0; it < 200; ++it) {
            const double h = 1e-5;
            curv = -(score(t + h) - score(t - h)) / (2 * h);
            const double step = std::clamp(score(t) / std::max(curv, 1e-8), -2.0, 2.0);
            t += step;
            if (std::abs(step) < 1e-11)
                break;
        }
        curv = -(score(t + 1e-5) - score(t - 1e-5)) / 2e-5;
        sc.map = t;
        sc.map_se = 1 / std::sqrt(curv);
        std::vector<double> lw(q);
        double top = -std::numeric_limits<double>::infinity();
        for (std::size_t k = 0; k < q; ++k) {
            lw[k] = std::log(rule.weights[k]) + log_post(rule.nodes[k]) + rule.nodes[k] * rule.nodes[k] / 2;
            top = std::max(top, lw[k]);
        }
        double w = 0, m1 = 0, m2 = 0;
        for (std::size_t k = 0; k < q; ++k) {
            const double e = std::exp(lw[k] - top);
            w += e, m1 += e * rule.nodes[k], m2 += e * rule.nodes[k] * rule.nodes[k];
        }
        sc.eap = m1 / w;
        sc.eap_sd = std::sqrt(std::max(0.0, m2 / w - sc.eap * sc.eap));
        out.push_back(sc);
    }
    return out;
}

} // namespace datamunge::stats
