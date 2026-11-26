/// @file synthetic_control.cpp
/// @brief Synthetic control weights by an exact simplex-constrained QP, V search and placebos.
#include <datamunge/causal/synthetic_control.hpp>
#include <datamunge/optim/nelder_mead.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace datamunge::causal {
/// @brief Implementation helpers local to this translation unit.
namespace {
/// @brief Dense row-major matrix of doubles.
using Matrix = linalg::DenseMatrix<double>;

/// @brief Solves a small dense linear system by Gaussian elimination with partial pivoting.
/// @param a Square matrix (copied).
/// @param b Right-hand side (copied).
/// @param x Solution on success.
/// @return False when the matrix is numerically singular.
bool solve_dense(Matrix a, std::vector<double> b, std::vector<double>& x) {
    const std::size_t n = b.size();
    double scale = 0;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            scale = std::max(scale, std::abs(a(i, j)));
    for (std::size_t c = 0; c < n; ++c) {
        std::size_t piv = c;
        for (std::size_t r = c + 1; r < n; ++r)
            if (std::abs(a(r, c)) > std::abs(a(piv, c)))
                piv = r;
        if (!(std::abs(a(piv, c)) > 1e-13 * scale))
            return false;
        for (std::size_t j = 0; j < n; ++j)
            std::swap(a(c, j), a(piv, j));
        std::swap(b[c], b[piv]);
        for (std::size_t r = c + 1; r < n; ++r) {
            const double f = a(r, c) / a(c, c);
            for (std::size_t j = c; j < n; ++j)
                a(r, j) -= f * a(c, j);
            b[r] -= f * b[c];
        }
    }
    x.assign(n, 0.0);
    for (std::size_t i = n; i-- > 0;) {
        double s = b[i];
        for (std::size_t j = i + 1; j < n; ++j)
            s -= a(i, j) * x[j];
        x[i] = s / a(i, i);
    }
    return true;
}

/// @brief Minimises 0.5 w'Hw + g'w over the probability simplex by a primal active-set method.
///
/// Starts at the best vertex, solves the equality-constrained problem on the free set through its
/// KKT system, steps back to feasibility when a weight would turn negative, and frees the donor
/// with the most negative reduced gradient until none remains. A ridge of 1e-12 times the mean
/// diagonal keeps the KKT systems nonsingular when donors are collinear.
/// @param h Positive semidefinite J-by-J matrix.
/// @param g Linear term.
/// @return The optimal weights.
std::vector<double> simplex_qp(Matrix h, const std::vector<double>& g) {
    const std::size_t m = g.size();
    double diag = 0;
    for (std::size_t j = 0; j < m; ++j)
        diag += h(j, j) / static_cast<double>(m);
    for (std::size_t j = 0; j < m; ++j)
        h(j, j) += 1e-12 * std::max(diag, 1e-300);
    const auto value_at_vertex = [&](std::size_t j) { return 0.5 * h(j, j) + g[j]; };
    std::size_t start = 0;
    for (std::size_t j = 1; j < m; ++j)
        if (value_at_vertex(j) < value_at_vertex(start))
            start = j;
    std::vector<double> w(m, 0.0);
    w[start] = 1;
    std::vector<char> free(m, 0);
    free[start] = 1;
    const auto gradient = [&](std::size_t j) {
        double s = g[j];
        for (std::size_t k = 0; k < m; ++k)
            s += h(j, k) * w[k];
        return s;
    };
    for (std::size_t iter = 0; iter < 50 * m + 100; ++iter) {
        std::vector<std::size_t> f;
        for (std::size_t j = 0; j < m; ++j)
            if (free[j])
                f.push_back(j);
        const std::size_t nf = f.size();
        Matrix kkt(nf + 1, nf + 1, 0.0);
        std::vector<double> rhs(nf + 1, 0.0), z;
        for (std::size_t a = 0; a < nf; ++a) {
            for (std::size_t b = 0; b < nf; ++b)
                kkt(a, b) = h(f[a], f[b]);
            kkt(a, nf) = kkt(nf, a) = 1;
            rhs[a] = -g[f[a]];
        }
        rhs[nf] = 1;
        if (!solve_dense(kkt, rhs, z))
            throw std::runtime_error("synthetic_control: singular donor-weight system");
        bool feasible = true;
        for (std::size_t a = 0; a < nf; ++a)
            feasible = feasible && z[a] >= 0;
        if (feasible) {
            for (std::size_t a = 0; a < nf; ++a)
                w[f[a]] = z[a];
            // Common gradient value on the free set; a fixed donor with a smaller gradient enters.
            const double level = gradient(f[0]);
            std::size_t enter = m;
            double best = -1e-12 * (1 + std::abs(level));
            for (std::size_t j = 0; j < m; ++j) {
                if (free[j])
                    continue;
                const double d = gradient(j) - level;
                if (d < best) {
                    best = d;
                    enter = j;
                }
            }
            if (enter == m)
                return w;
            free[enter] = 1;
        } else {
            double t = 1;
            std::size_t leave = m;
            for (std::size_t a = 0; a < nf; ++a)
                if (z[a] < 0 && w[f[a]] - z[a] > 0) {
                    const double s = w[f[a]] / (w[f[a]] - z[a]);
                    if (s < t) {
                        t = s;
                        leave = f[a];
                    }
                }
            for (std::size_t a = 0; a < nf; ++a)
                w[f[a]] += t * (z[a] - w[f[a]]);
            if (leave < m) {
                w[leave] = 0;
                free[leave] = 0;
            }
            for (std::size_t a = 0; a < nf; ++a)
                if (w[f[a]] <= 0) {
                    w[f[a]] = 0;
                    free[f[a]] = 0;
                }
        }
    }
    throw std::runtime_error("synthetic_control: donor-weight QP did not converge");
}

/// @brief Standardised problem shared by every V evaluation.
struct Problem {
    std::vector<double> x1; ///< Scaled treated predictors.
    Matrix x0;              ///< Scaled donor predictors (k-by-J).
    const SyntheticControlData* data{}; ///< Original data.
};

/// @brief Donor weights for a V diagonal (normalised internally).
/// @param p Problem.
/// @param v Nonnegative predictor weights.
/// @return The weights.
std::vector<double> weights_for(const Problem& p, const std::vector<double>& v) {
    const std::size_t k = p.x0.rows(), m = p.x0.cols();
    const double total = std::accumulate(v.begin(), v.end(), 0.0);
    Matrix h(m, m, 0.0);
    std::vector<double> g(m, 0.0);
    for (std::size_t r = 0; r < k; ++r) {
        const double vr = v[r] / total;
        if (vr == 0)
            continue;
        for (std::size_t a = 0; a < m; ++a) {
            g[a] -= vr * p.x0(r, a) * p.x1[r];
            for (std::size_t b = 0; b < m; ++b)
                h(a, b) += vr * p.x0(r, a) * p.x0(r, b);
        }
    }
    return simplex_qp(h, g);
}

/// @brief Mean squared outcome gap over the fit periods.
/// @param d Data.
/// @param w Donor weights.
/// @return The MSPE.
double fit_mspe(const SyntheticControlData& d, const std::vector<double>& w) {
    double s = 0;
    for (const std::size_t t : d.fit_periods) {
        double synth = 0;
        for (std::size_t j = 0; j < w.size(); ++j)
            synth += d.donor_outcomes(t, j) * w[j];
        s += (d.treated_outcome[t] - synth) * (d.treated_outcome[t] - synth);
    }
    return s / static_cast<double>(d.fit_periods.size());
}

/// @brief Nelder-Mead objective: fit MSPE as a function of unnormalised |V|.
class VObjective : public optim::ArbitraryFunction {
public:
    /// @brief Binds the problem.
    /// @param p Problem (borrowed).
    explicit VObjective(const Problem& p) : p_(p) {}
    /// @brief Evaluates the fit MSPE at V = |par| / sum |par|.
    /// @param par Unnormalised predictor weights.
    /// @return The MSPE (infinity for an all-zero V).
    double evaluate(const std::vector<double>& par) override {
        std::vector<double> v(par.size());
        std::ranges::transform(par, v.begin(), [](double x) { return std::abs(x); });
        if (std::accumulate(v.begin(), v.end(), 0.0) <= 0)
            return std::numeric_limits<double>::infinity();
        return fit_mspe(*p_.data, weights_for(p_, v));
    }

private:
    const Problem& p_; ///< Borrowed problem.
};

/// @brief Regression-based starting V (as in `Synth`): squared coefficients of the outcomes on
/// the scaled predictors across all units.
/// @param p Problem.
/// @param v Starting V on success.
/// @return False when the regression is singular (more predictors than units).
bool regression_start(const Problem& p, std::vector<double>& v) {
    const auto& d = *p.data;
    const std::size_t k = p.x0.rows(), m = p.x0.cols(), n = m + 1, q = k + 1;
    if (n < q)
        return false;
    const auto design = [&](std::size_t unit, std::size_t c) {
        if (c == 0)
            return 1.0;
        return unit == 0 ? p.x1[c - 1] : p.x0(c - 1, unit - 1);
    };
    Matrix xtx(q, q, 0.0);
    for (std::size_t u = 0; u < n; ++u)
        for (std::size_t a = 0; a < q; ++a)
            for (std::size_t b = 0; b < q; ++b)
                xtx(a, b) += design(u, a) * design(u, b);
    v.assign(k, 0.0);
    for (const std::size_t t : d.fit_periods) {
        std::vector<double> xty(q, 0.0), beta;
        for (std::size_t u = 0; u < n; ++u) {
            const double z = u == 0 ? d.treated_outcome[t] : d.donor_outcomes(t, u - 1);
            for (std::size_t a = 0; a < q; ++a)
                xty[a] += design(u, a) * z;
        }
        if (!solve_dense(xtx, xty, beta))
            return false;
        for (std::size_t r = 0; r < k; ++r)
            v[r] += beta[r + 1] * beta[r + 1];
    }
    const double total = std::accumulate(v.begin(), v.end(), 0.0);
    if (!(total > 0) || !std::isfinite(total))
        return false;
    for (double& x : v)
        x /= total;
    return true;
}

/// @brief Validates the data shapes and values.
/// @param d Data.
/// @param min_donors Minimum donor count.
/// @throws std::invalid_argument On any inconsistency.
void validate(const SyntheticControlData& d, std::size_t min_donors) {
    const std::size_t k = d.treated_predictors.size(), m = d.donor_predictors.cols();
    const std::size_t t = d.treated_outcome.size();
    if (k == 0 || d.donor_predictors.rows() != k)
        throw std::invalid_argument("synthetic_control: predictor shapes differ");
    if (m < min_donors)
        throw std::invalid_argument("synthetic_control: too few donors");
    if (d.donor_outcomes.rows() != t || d.donor_outcomes.cols() != m)
        throw std::invalid_argument("synthetic_control: outcome shapes differ");
    if (d.fit_periods.empty() || d.treatment_period == 0 || d.treatment_period >= t)
        throw std::invalid_argument("synthetic_control: need fit periods and a treatment period inside the panel");
    for (const std::size_t r : d.fit_periods)
        if (r >= d.treatment_period)
            throw std::invalid_argument("synthetic_control: fit periods must precede the treatment period");
    const auto finite = [](const auto& range) { return std::ranges::all_of(range, [](double x) { return std::isfinite(x); }); };
    if (!finite(d.treated_predictors) || !finite(d.treated_outcome) || !finite(d.donor_predictors.data()) ||
        !finite(d.donor_outcomes.data()))
        throw std::invalid_argument("synthetic_control: inputs must be finite");
}

/// @brief Fits one synthetic control on validated data.
/// @param d Data.
/// @param options Options.
/// @return The fit.
SyntheticControlFit fit(const SyntheticControlData& d, const SyntheticControlOptions& options) {
    const std::size_t k = d.treated_predictors.size(), m = d.donor_predictors.cols();
    Problem p;
    p.data = &d;
    p.x1.resize(k);
    p.x0 = Matrix(k, m, 0.0);
    for (std::size_t r = 0; r < k; ++r) {
        double mean = d.treated_predictors[r], ss = 0;
        for (std::size_t j = 0; j < m; ++j)
            mean += d.donor_predictors(r, j);
        mean /= static_cast<double>(m + 1);
        ss += (d.treated_predictors[r] - mean) * (d.treated_predictors[r] - mean);
        for (std::size_t j = 0; j < m; ++j)
            ss += (d.donor_predictors(r, j) - mean) * (d.donor_predictors(r, j) - mean);
        const double sd = std::sqrt(ss / static_cast<double>(m));
        if (!(sd > 0))
            throw std::invalid_argument("synthetic_control: a predictor is constant across units");
        p.x1[r] = d.treated_predictors[r] / sd;
        for (std::size_t j = 0; j < m; ++j)
            p.x0(r, j) = d.donor_predictors(r, j) / sd;
    }
    std::vector<double> v;
    if (!options.predictor_weights.empty()) {
        if (options.predictor_weights.size() != k ||
            !std::ranges::all_of(options.predictor_weights, [](double x) { return x >= 0 && std::isfinite(x); }) ||
            !(std::accumulate(options.predictor_weights.begin(), options.predictor_weights.end(), 0.0) > 0))
            throw std::invalid_argument("synthetic_control: predictor weights must be k nonnegative values, not all 0");
        v = options.predictor_weights;
    } else if (k == 1) {
        v = {1.0};
    } else {
        std::vector<std::vector<double>> starts{std::vector<double>(k, 1.0 / static_cast<double>(k))};
        std::vector<double> reg;
        if (regression_start(p, reg))
            starts.push_back(reg);
        VObjective objective(p);
        optim::NelderMeadOptions nm;
        nm.max_iterations = options.max_iterations;
        nm.initial_simplex_scale = 0.5 / static_cast<double>(k);
        nm.tolerance = 1e-12;
        double best = std::numeric_limits<double>::infinity();
        for (auto start : starts) {
            double value = objective.evaluate(start);
            for (std::size_t r = 0; r <= options.restarts; ++r) {
                const double next = optim::NelderMead(nm).optimize(objective, start);
                const double total = std::accumulate(start.begin(), start.end(), 0.0,
                                                     [](double s, double x) { return s + std::abs(x); });
                for (double& x : start)
                    x = std::abs(x) / total;
                const bool stalled = value - next <= 1e-12 * (1 + std::abs(value));
                value = std::min(value, next);
                if (stalled && r > 0)
                    break;
            }
            if (value < best) {
                best = value;
                v = start;
            }
        }
    }
    const double total = std::accumulate(v.begin(), v.end(), 0.0);
    SyntheticControlFit out;
    for (const double x : v)
        out.predictor_weights.push_back(x / total);
    out.weights = weights_for(p, out.predictor_weights);
    for (std::size_t r = 0; r < k; ++r) {
        double s = 0, ss = 0;
        for (std::size_t j = 0; j < m; ++j) {
            s += d.donor_predictors(r, j) * out.weights[j];
            ss += p.x0(r, j) * out.weights[j];
        }
        out.synthetic_predictors.push_back(s);
        out.predictor_loss += out.predictor_weights[r] * (p.x1[r] - ss) * (p.x1[r] - ss);
    }
    const std::size_t periods = d.treated_outcome.size();
    double pre = 0, post = 0;
    for (std::size_t t = 0; t < periods; ++t) {
        double s = 0;
        for (std::size_t j = 0; j < m; ++j)
            s += d.donor_outcomes(t, j) * out.weights[j];
        out.synthetic.push_back(s);
        out.gap.push_back(d.treated_outcome[t] - s);
        (t < d.treatment_period ? pre : post) += out.gap.back() * out.gap.back();
    }
    out.fit_mspe = fit_mspe(d, out.weights);
    out.pre_rmspe = std::sqrt(pre / static_cast<double>(d.treatment_period));
    out.post_rmspe = std::sqrt(post / static_cast<double>(periods - d.treatment_period));
    out.rmspe_ratio = out.post_rmspe / out.pre_rmspe;
    return out;
}
} // namespace

SyntheticControlFit synthetic_control(const SyntheticControlData& data, const SyntheticControlOptions& options) {
    validate(data, 2);
    return fit(data, options);
}

SyntheticControlPlacebos synthetic_control_placebos(const SyntheticControlData& data,
                                                    const SyntheticControlOptions& options,
                                                    double max_pre_mspe_ratio) {
    validate(data, 3);
    if (!(max_pre_mspe_ratio > 0))
        throw std::invalid_argument("synthetic_control_placebos: the MSPE threshold must be positive");
    SyntheticControlPlacebos out;
    out.treated = fit(data, options);
    const std::size_t k = data.treated_predictors.size(), m = data.donor_predictors.cols();
    const std::size_t periods = data.treated_outcome.size();
    const double treated_pre = out.treated.pre_rmspe * out.treated.pre_rmspe;
    std::size_t exceed = 0;
    for (std::size_t unit = 0; unit < m; ++unit) {
        SyntheticControlData pd;
        pd.fit_periods = data.fit_periods;
        pd.treatment_period = data.treatment_period;
        pd.donor_predictors = Matrix(k, m - 1, 0.0);
        pd.donor_outcomes = Matrix(periods, m - 1, 0.0);
        for (std::size_t r = 0; r < k; ++r) {
            pd.treated_predictors.push_back(data.donor_predictors(r, unit));
            for (std::size_t j = 0, c = 0; j < m; ++j)
                if (j != unit)
                    pd.donor_predictors(r, c++) = data.donor_predictors(r, j);
        }
        for (std::size_t t = 0; t < periods; ++t) {
            pd.treated_outcome.push_back(data.donor_outcomes(t, unit));
            for (std::size_t j = 0, c = 0; j < m; ++j)
                if (j != unit)
                    pd.donor_outcomes(t, c++) = data.donor_outcomes(t, j);
        }
        out.placebos.push_back(fit(pd, options));
        const auto& f = out.placebos.back();
        if (f.pre_rmspe * f.pre_rmspe <= max_pre_mspe_ratio * treated_pre) {
            out.included.push_back(unit);
            exceed += f.rmspe_ratio >= out.treated.rmspe_ratio;
        }
    }
    out.p_value = static_cast<double>(1 + exceed) / static_cast<double>(1 + out.included.size());
    return out;
}

} // namespace datamunge::causal
