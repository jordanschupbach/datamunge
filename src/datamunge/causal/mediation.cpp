/// @file mediation.cpp
/// @brief Mediator and outcome regressions, the mediation formula, and bootstrap inference.
#include <datamunge/causal/mediation.hpp>
#include <datamunge/random/random.hpp>
#include <datamunge/stats/gauss_hermite.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace datamunge::causal {
/// @brief Implementation helpers local to this translation unit.
namespace {
/// @brief Dense row-major matrix of doubles.
using Matrix = linalg::DenseMatrix<double>;

/// @brief Inverse logit computed from the smaller tail.
/// @param eta Linear predictor.
/// @return The probability.
double expit(double eta) {
    const double z = std::exp(-std::abs(eta));
    return eta >= 0 ? 1 / (1 + z) : z / (1 + z);
}

/// @brief Solves a symmetric positive definite system by Cholesky factorisation.
/// @param a Matrix (copied).
/// @param b Right-hand side.
/// @return The solution.
/// @throws std::runtime_error When the matrix is not positive definite (collinear design).
std::vector<double> spd_solve(Matrix a, std::vector<double> b) {
    const std::size_t n = b.size();
    for (std::size_t j = 0; j < n; ++j) {
        double d = a(j, j);
        for (std::size_t k = 0; k < j; ++k)
            d -= a(j, k) * a(j, k);
        if (!(d > 1e-12 * std::max(1.0, std::abs(a(j, j)))))
            throw std::runtime_error("mediation_analysis: collinear regression design");
        a(j, j) = std::sqrt(d);
        for (std::size_t i = j + 1; i < n; ++i) {
            double s = a(i, j);
            for (std::size_t k = 0; k < j; ++k)
                s -= a(i, k) * a(j, k);
            a(i, j) = s / a(j, j);
        }
    }
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t k = 0; k < i; ++k)
            b[i] -= a(i, k) * b[k];
        b[i] /= a(i, i);
    }
    for (std::size_t i = n; i-- > 0;) {
        for (std::size_t k = i + 1; k < n; ++k)
            b[i] -= a(k, i) * b[k];
        b[i] /= a(i, i);
    }
    return b;
}

/// @brief A fitted regression.
struct Fit {
    std::vector<double> beta; ///< Coefficients.
    double sigma{};           ///< Residual SD (linear models).
};

/// @brief Least squares or logistic (IRLS) regression of y on the design x.
/// @param x n-by-p design (with intercept column).
/// @param y Response.
/// @param model Linear or logistic.
/// @return The fit.
/// @throws std::runtime_error For collinearity, separation or non-convergence.
Fit regress(const Matrix& x, const std::vector<double>& y, MediationModel model) {
    const std::size_t n = x.rows(), p = x.cols();
    if (n <= p)
        throw std::invalid_argument("mediation_analysis: more parameters than observations");
    Fit fit;
    fit.beta.assign(p, 0.0);
    const auto weighted_step = [&](const std::vector<double>& w, const std::vector<double>& z) {
        Matrix xtx(p, p, 0.0);
        std::vector<double> xtz(p, 0.0);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t a = 0; a < p; ++a) {
                xtz[a] += w[i] * x(i, a) * z[i];
                for (std::size_t b = 0; b <= a; ++b)
                    xtx(a, b) += w[i] * x(i, a) * x(i, b);
            }
        for (std::size_t a = 0; a < p; ++a)
            for (std::size_t b = a + 1; b < p; ++b)
                xtx(a, b) = xtx(b, a);
        return spd_solve(xtx, xtz);
    };
    if (model == MediationModel::linear) {
        fit.beta = weighted_step(std::vector<double>(n, 1.0), y);
        double rss = 0;
        for (std::size_t i = 0; i < n; ++i) {
            double e = y[i];
            for (std::size_t a = 0; a < p; ++a)
                e -= x(i, a) * fit.beta[a];
            rss += e * e;
        }
        fit.sigma = std::sqrt(rss / static_cast<double>(n - p));
        return fit;
    }
    double deviance = std::numeric_limits<double>::infinity();
    for (int iter = 0; iter < 100; ++iter) {
        std::vector<double> w(n), z(n);
        double next = 0;
        for (std::size_t i = 0; i < n; ++i) {
            double eta = 0;
            for (std::size_t a = 0; a < p; ++a)
                eta += x(i, a) * fit.beta[a];
            const double mu = expit(eta);
            w[i] = std::max(mu * (1 - mu), 1e-12);
            z[i] = eta + (y[i] - mu) / w[i];
            next -= 2 * std::log(std::max(y[i] > 0.5 ? mu : 1 - mu, 1e-300));
        }
        if (std::abs(deviance - next) < 1e-10 * (std::abs(next) + 0.1))
            return fit;
        deviance = next;
        fit.beta = weighted_step(w, z);
        if (std::ranges::any_of(fit.beta, [](double b) { return !std::isfinite(b) || std::abs(b) > 50; }))
            throw std::runtime_error("mediation_analysis: logistic fit diverges (separation)");
    }
    throw std::runtime_error("mediation_analysis: logistic fit did not converge");
}

/// @brief Fitted models of one (re)sample.
struct Models {
    Fit mediator; ///< Mediator regression.
    Fit outcome;  ///< Outcome regression.
};

/// @brief Fits both models on the rows listed.
/// @param y Outcome.
/// @param a Treatment.
/// @param m Mediator.
/// @param c Covariates.
/// @param rows Row indices (with repetition for bootstrap samples).
/// @param o Options.
/// @return The fitted models.
Models fit_models(const std::vector<double>& y, const std::vector<double>& a, const std::vector<double>& m,
                  const Matrix& c, const std::vector<std::size_t>& rows, const MediationOptions& o) {
    const std::size_t n = rows.size(), pc = c.cols(), inter = o.interaction ? 1 : 0;
    Matrix xm(n, 2 + pc), xy(n, 3 + inter + pc);
    std::vector<double> rm(n), ry(n);
    for (std::size_t r = 0; r < n; ++r) {
        const std::size_t i = rows[r];
        xm(r, 0) = xy(r, 0) = 1;
        xm(r, 1) = xy(r, 1) = a[i];
        xy(r, 2) = m[i];
        if (o.interaction)
            xy(r, 3) = a[i] * m[i];
        for (std::size_t j = 0; j < pc; ++j)
            xm(r, 2 + j) = xy(r, 3 + inter + j) = c(i, j);
        rm[r] = m[i];
        ry[r] = y[i];
    }
    return {regress(xm, rm, o.mediator_model), regress(xy, ry, o.outcome_model)};
}

/// @brief The five effects and the proportion mediated.
using Effects = std::array<double, 6>;

/// @brief Evaluates the mediation formula over the given rows' covariates.
/// @param f Fitted models.
/// @param c Covariates.
/// @param rows Rows over which to average.
/// @param o Options.
/// @param rule Gauss-Hermite rule for normal mediators.
/// @return delta(0), delta(1), zeta(0), zeta(1), total, proportion mediated.
Effects mediation_formula(const Models& f, const Matrix& c, const std::vector<std::size_t>& rows,
                          const MediationOptions& o, const stats::GaussHermiteRule& rule) {
    const std::size_t pc = c.cols(), inter = o.interaction ? 1 : 0;
    const auto& bm = f.mediator.beta;
    const auto& by = f.outcome.beta;
    const double lv[2]{o.control_value, o.treated_value};
    double q[2][2]{}; // q[a][a'] = E[Y(a, M(a'))]
    for (const std::size_t i : rows) {
        double cm = 0, cy = 0;
        for (std::size_t j = 0; j < pc; ++j) {
            cm += bm[2 + j] * c(i, j);
            cy += by[3 + inter + j] * c(i, j);
        }
        for (int ai = 0; ai < 2; ++ai)
            for (int mi = 0; mi < 2; ++mi) {
                const double a = lv[ai], am = lv[mi];
                const double slope = by[2] + (o.interaction ? by[3] * a : 0.0);
                const double base = by[0] + by[1] * a + cy;
                const double mlin = bm[0] + bm[1] * am + cm;
                const auto mean_y = [&](double mv) {
                    const double eta = base + slope * mv;
                    return o.outcome_model == MediationModel::logistic ? expit(eta) : eta;
                };
                double v = 0;
                if (o.mediator_model == MediationModel::logistic) {
                    const double pm = expit(mlin);
                    v = pm * mean_y(1) + (1 - pm) * mean_y(0);
                } else if (o.outcome_model == MediationModel::linear) {
                    v = mean_y(mlin);
                } else {
                    for (std::size_t k = 0; k < rule.nodes.size(); ++k)
                        v += rule.weights[k] * mean_y(mlin + f.mediator.sigma * rule.nodes[k]);
                }
                q[ai][mi] += v / static_cast<double>(rows.size());
            }
    }
    const double total = q[1][1] - q[0][0];
    const double d1 = q[1][1] - q[1][0];
    return {q[0][1] - q[0][0], d1, q[1][0] - q[0][0], q[1][1] - q[0][1], total, d1 / total};
}

/// @brief Checks that a variable is 0/1.
/// @param v Values.
/// @return True when every value is 0 or 1.
bool binary(const std::vector<double>& v) {
    return std::ranges::all_of(v, [](double x) { return x == 0 || x == 1; });
}
} // namespace

MediationResult mediation_analysis(const std::vector<double>& outcome, const std::vector<double>& treatment,
                                   const std::vector<double>& mediator, const linalg::DenseMatrix<double>& covariates,
                                   const MediationOptions& options) {
    const std::size_t n = outcome.size();
    if (treatment.size() != n || mediator.size() != n || (covariates.cols() > 0 && covariates.rows() != n))
        throw std::invalid_argument("mediation_analysis: input sizes differ");
    const auto finite = [](const std::vector<double>& v) {
        return std::ranges::all_of(v, [](double x) { return std::isfinite(x); });
    };
    if (!finite(outcome) || !finite(treatment) || !finite(mediator) || !finite(covariates.data()))
        throw std::invalid_argument("mediation_analysis: inputs must be finite");
    if ((options.mediator_model == MediationModel::logistic && !binary(mediator)) ||
        (options.outcome_model == MediationModel::logistic && !binary(outcome)))
        throw std::invalid_argument("mediation_analysis: logistic models need 0/1 variables");
    if (!(options.confidence > 0 && options.confidence < 1) || options.quadrature_points == 0 ||
        options.control_value == options.treated_value || !std::isfinite(options.control_value) ||
        !std::isfinite(options.treated_value))
        throw std::invalid_argument("mediation_analysis: bad options");
    const auto rule = stats::normal_gauss_hermite(options.quadrature_points);
    std::vector<std::size_t> all(n);
    for (std::size_t i = 0; i < n; ++i)
        all[i] = i;
    const Models models = fit_models(outcome, treatment, mediator, covariates, all, options);
    const Effects point = mediation_formula(models, covariates, all, options, rule);
    MediationResult out;
    out.mediator_coefficients = models.mediator.beta;
    out.outcome_coefficients = models.outcome.beta;
    out.mediator_sigma = models.mediator.sigma;
    std::array<std::vector<double>, 6> draws;
    if (options.bootstrap > 0) {
        random::SplitMix64 rng(options.seed);
        std::vector<std::size_t> rows(n);
        for (std::size_t b = 0; b < options.bootstrap; ++b) {
            for (std::size_t i = 0; i < n; ++i)
                rows[i] = static_cast<std::size_t>(rng.uniform01() * static_cast<double>(n)) % n;
            try {
                const Models fb = fit_models(outcome, treatment, mediator, covariates, rows, options);
                const Effects e = mediation_formula(fb, covariates, rows, options, rule);
                for (std::size_t k = 0; k < 6; ++k)
                    draws[k].push_back(e[k]);
            } catch (const std::exception&) {
                ++out.bootstrap_failures;
            }
        }
    }
    const auto summarise = [&](std::size_t k) {
        MediationEffect e;
        e.estimate = point[k];
        const auto& d = draws[k];
        if (d.size() < 2) {
            e.standard_error = e.lower = e.upper = std::numeric_limits<double>::quiet_NaN();
            return e;
        }
        double mean = 0, ss = 0;
        for (const double v : d)
            mean += v / static_cast<double>(d.size());
        for (const double v : d)
            ss += (v - mean) * (v - mean);
        e.standard_error = std::sqrt(ss / static_cast<double>(d.size() - 1));
        auto sorted = d;
        std::ranges::sort(sorted);
        // Type-7 quantiles, as R's quantile() used by mediation.
        const auto quantile = [&](double p) {
            const double h = (static_cast<double>(sorted.size()) - 1) * p;
            const auto lo = static_cast<std::size_t>(std::floor(h));
            const std::size_t hi = std::min(lo + 1, sorted.size() - 1);
            return sorted[lo] + (h - static_cast<double>(lo)) * (sorted[hi] - sorted[lo]);
        };
        e.lower = quantile((1 - options.confidence) / 2);
        e.upper = quantile((1 + options.confidence) / 2);
        return e;
    };
    out.indirect_control = summarise(0);
    out.indirect_treated = summarise(1);
    out.direct_control = summarise(2);
    out.direct_treated = summarise(3);
    out.total = summarise(4);
    out.proportion_mediated = summarise(5);
    return out;
}

} // namespace datamunge::causal
