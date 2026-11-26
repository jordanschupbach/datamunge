/// @file advi.cpp
/// @brief Support transforms, reparameterisation gradients, adaptive step sizes and ELBO monitoring.
#include <datamunge/bayes/advi.hpp>
#include <datamunge/random/random.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>

namespace datamunge::bayes {
/// @brief Implementation helpers local to this translation unit.
namespace {
/// @brief Logistic function computed from the smaller tail.
/// @param x Argument.
/// @return 1 / (1 + exp(-x)).
double logistic(double x) {
    const double z = std::exp(-std::abs(x));
    return x >= 0 ? 1 / (1 + z) : z / (1 + z);
}

/// @brief Image of one unconstrained coordinate with its log-Jacobian and derivatives.
struct Mapped {
    double theta{};     ///< Constrained value.
    double dtheta{};    ///< d theta / d zeta.
    double log_jac{};   ///< log |d theta / d zeta|.
    double dlog_jac{};  ///< d log_jac / d zeta.
};

/// @brief Applies a support transform.
/// @param s Support.
/// @param z Unconstrained value.
/// @return The constrained value and Jacobian terms.
Mapped forward(const ParameterSupport& s, double z) {
    switch (s.kind) {
    case SupportKind::real:
        return {z, 1, 0, 0};
    case SupportKind::lower:
        return {s.lower + std::exp(z), std::exp(z), z, 1};
    case SupportKind::upper:
        return {s.upper - std::exp(z), -std::exp(z), z, 1};
    case SupportKind::interval: {
        const double p = logistic(z), w = s.upper - s.lower;
        // log p + log(1 - p) = -|z| - 2 log(1 + exp(-|z|)).
        const double lj = std::log(w) - std::abs(z) - 2 * std::log1p(std::exp(-std::abs(z)));
        return {s.lower + w * p, w * p * (1 - p), lj, 1 - 2 * p};
    }
    }
    return {z, 1, 0, 0};
}

/// @brief Inverse of a support transform.
/// @param s Support.
/// @param t Constrained value (inside the support).
/// @return The unconstrained value.
/// @throws std::invalid_argument When t is outside the support.
double inverse(const ParameterSupport& s, double t) {
    switch (s.kind) {
    case SupportKind::real:
        return t;
    case SupportKind::lower:
        if (!(t > s.lower))
            break;
        return std::log(t - s.lower);
    case SupportKind::upper:
        if (!(t < s.upper))
            break;
        return std::log(s.upper - t);
    case SupportKind::interval: {
        if (!(t > s.lower && t < s.upper))
            break;
        const double p = (t - s.lower) / (s.upper - s.lower);
        return std::log(p) - std::log1p(-p);
    }
    }
    throw std::invalid_argument("advi: initial value outside its support");
}

/// @brief Variational parameters.
struct Variational {
    std::vector<double> mu;    ///< Mean.
    std::vector<double> omega; ///< Log-SDs (mean-field).
    std::vector<double> chol;  ///< Row-major lower factor (full-rank).
};

/// @brief Problem state shared by the optimisation routines.
struct Problem {
    optim::DifferentiableFunction& f;           ///< Log density on the constrained scale.
    std::vector<ParameterSupport> supports;     ///< Supports.
    const AdviOptions& o;                       ///< Options.
    std::size_t d;                              ///< Dimension.

    /// @brief Transforms a draw of the variational distribution.
    /// @param v Variational parameters.
    /// @param eps Standard normal vector.
    /// @return zeta = mu + L eps.
    [[nodiscard]] std::vector<double> draw(const Variational& v, const std::vector<double>& eps) const {
        std::vector<double> z(d);
        for (std::size_t a = 0; a < d; ++a) {
            if (o.family == AdviFamily::mean_field) {
                z[a] = v.mu[a] + std::exp(v.omega[a]) * eps[a];
            } else {
                double s = v.mu[a];
                for (std::size_t b = 0; b <= a; ++b)
                    s += v.chol[a * d + b] * eps[b];
                z[a] = s;
            }
        }
        return z;
    }

    /// @brief Log joint density on the unconstrained scale and, optionally, its gradient.
    /// @param z Unconstrained point.
    /// @param grad Receives the gradient when non-null.
    /// @return log p(T(z)) + log|J(z)|.
    double log_joint(const std::vector<double>& z, std::vector<double>* grad) const {
        std::vector<double> theta(d);
        std::vector<Mapped> m(d);
        double lj = 0;
        for (std::size_t a = 0; a < d; ++a) {
            m[a] = forward(supports[a], z[a]);
            theta[a] = m[a].theta;
            lj += m[a].log_jac;
        }
        const double lp = f.evaluate(theta);
        if (grad) {
            const auto g = f.gradient(theta);
            if (g.size() != d)
                throw std::invalid_argument("advi: gradient has the wrong length");
            grad->resize(d);
            for (std::size_t a = 0; a < d; ++a)
                (*grad)[a] = g[a] * m[a].dtheta + m[a].dlog_jac;
        }
        return lp + lj;
    }

    /// @brief Entropy of the variational Gaussian.
    /// @param v Variational parameters.
    /// @return The differential entropy.
    [[nodiscard]] double entropy(const Variational& v) const {
        double s = 0.5 * static_cast<double>(d) * (1 + std::log(2 * std::numbers::pi));
        for (std::size_t a = 0; a < d; ++a)
            s += o.family == AdviFamily::mean_field ? v.omega[a] : std::log(std::abs(v.chol[a * d + a]));
        return s;
    }

    /// @brief Monte Carlo ELBO estimate.
    /// @param v Variational parameters.
    /// @param rng Random source.
    /// @return The estimate (NaN or -inf on failure).
    double elbo(const Variational& v, random::SplitMix64& rng) const {
        double s = 0;
        std::vector<double> eps(d);
        for (std::size_t k = 0; k < o.elbo_samples; ++k) {
            for (double& e : eps)
                e = rng.normal();
            s += log_joint(draw(v, eps), nullptr);
        }
        return s / static_cast<double>(o.elbo_samples) + entropy(v);
    }

    /// @brief Gradient of the ELBO with respect to all variational parameters (flattened).
    /// @param v Variational parameters.
    /// @param rng Random source.
    /// @return Gradient in the order mu, then omega or the lower factor's entries.
    std::vector<double> gradient(const Variational& v, random::SplitMix64& rng) const {
        const bool mf = o.family == AdviFamily::mean_field;
        std::vector<double> g(d + (mf ? d : d * d), 0.0), eps(d), gz;
        const double w = 1.0 / static_cast<double>(o.gradient_samples);
        for (std::size_t k = 0; k < o.gradient_samples; ++k) {
            for (double& e : eps)
                e = rng.normal();
            log_joint(draw(v, eps), &gz);
            for (std::size_t a = 0; a < d; ++a) {
                g[a] += w * gz[a];
                if (mf) {
                    g[d + a] += w * gz[a] * eps[a] * std::exp(v.omega[a]);
                } else {
                    for (std::size_t b = 0; b <= a; ++b)
                        g[d + a * d + b] += w * gz[a] * eps[b];
                }
            }
        }
        for (std::size_t a = 0; a < d; ++a) {
            if (mf)
                g[d + a] += 1;
            else
                g[d + a * d + a] += 1 / v.chol[a * d + a];
        }
        return g;
    }
};

/// @brief Adds a flattened step to the variational parameters.
/// @param v Variational parameters (updated).
/// @param step Flattened step.
/// @param family Family.
void apply(Variational& v, const std::vector<double>& step, AdviFamily family) {
    const std::size_t d = v.mu.size();
    for (std::size_t a = 0; a < d; ++a)
        v.mu[a] += step[a];
    if (family == AdviFamily::mean_field) {
        for (std::size_t a = 0; a < d; ++a)
            v.omega[a] += step[d + a];
    } else {
        for (std::size_t a = 0; a < d; ++a)
            for (std::size_t b = 0; b <= a; ++b)
                v.chol[a * d + b] += step[d + a * d + b];
    }
}

/// @brief Runs the adaptive stochastic gradient ascent.
/// @param p Problem.
/// @param v Variational parameters (updated).
/// @param eta Step-size scale.
/// @param iterations Maximum iterations.
/// @param monitor Whether to evaluate the ELBO and test convergence.
/// @param rng Random source.
/// @param result Receives the ELBO history, iterations and convergence flag when monitoring.
/// @return False if a nonfinite gradient or parameter occurred.
bool optimise(const Problem& p, Variational& v, double eta, std::size_t iterations, bool monitor,
              random::SplitMix64& rng, AdviResult* result) {
    std::vector<double> s;
    std::vector<double> rel;
    const std::size_t window =
        std::max<std::size_t>(2, static_cast<std::size_t>(0.1 * static_cast<double>(p.o.max_iterations) /
                                                         static_cast<double>(p.o.eval_every)));
    double previous = std::numeric_limits<double>::quiet_NaN();
    for (std::size_t k = 1; k <= iterations; ++k) {
        auto g = p.gradient(v, rng);
        if (!std::ranges::all_of(g, [](double x) { return std::isfinite(x); }))
            return false;
        if (s.empty()) {
            s.resize(g.size());
            for (std::size_t j = 0; j < g.size(); ++j)
                s[j] = g[j] * g[j];
        } else {
            for (std::size_t j = 0; j < g.size(); ++j)
                s[j] = 0.1 * g[j] * g[j] + 0.9 * s[j];
        }
        const double base = eta * std::pow(static_cast<double>(k), -0.5 + 1e-16);
        for (std::size_t j = 0; j < g.size(); ++j)
            g[j] *= base / (1 + std::sqrt(s[j]));
        apply(v, g, p.o.family);
        if (monitor && k % p.o.eval_every == 0) {
            const double e = p.elbo(v, rng);
            if (!std::isfinite(e))
                return false;
            result->elbo_history.push_back(e);
            result->iterations = k;
            if (std::isfinite(previous)) {
                rel.push_back(std::abs((e - previous) / e));
                if (rel.size() > window)
                    rel.erase(rel.begin());
                double mean = 0;
                for (const double r : rel)
                    mean += r / static_cast<double>(rel.size());
                auto sorted = rel;
                std::ranges::sort(sorted);
                const std::size_t m = sorted.size();
                const double median = m % 2 ? sorted[m / 2] : 0.5 * (sorted[m / 2 - 1] + sorted[m / 2]);
                if (mean < p.o.tolerance || median < p.o.tolerance) {
                    result->converged = true;
                    return true;
                }
            }
            previous = e;
        }
        if (monitor)
            result->iterations = k;
    }
    return true;
}
} // namespace

AdviResult advi(optim::DifferentiableFunction& log_density, const std::vector<double>& initial,
                const AdviOptions& options) {
    const std::size_t d = initial.size();
    if (d == 0)
        throw std::invalid_argument("advi: empty initial value");
    if (!options.supports.empty() && options.supports.size() != d)
        throw std::invalid_argument("advi: one support per parameter is required");
    if (options.gradient_samples == 0 || options.elbo_samples == 0 || options.eval_every == 0 ||
        options.max_iterations == 0 || !(options.tolerance > 0) || options.step_size < 0 ||
        options.adapt_iterations == 0)
        throw std::invalid_argument("advi: bad options");
    std::vector<ParameterSupport> supports =
        options.supports.empty() ? std::vector<ParameterSupport>(d) : options.supports;
    for (const auto& s : supports)
        if ((s.kind == SupportKind::interval && !(s.upper > s.lower)) || !std::isfinite(s.lower) ||
            !std::isfinite(s.upper))
            throw std::invalid_argument("advi: invalid support bounds");
    Variational start;
    for (std::size_t a = 0; a < d; ++a)
        start.mu.push_back(inverse(supports[a], initial[a]));
    if (options.family == AdviFamily::mean_field) {
        start.omega.assign(d, 0.0);
    } else {
        start.chol.assign(d * d, 0.0);
        for (std::size_t a = 0; a < d; ++a)
            start.chol[a * d + a] = 1;
    }
    const Problem problem{log_density, supports, options, d};
    random::SplitMix64 rng(options.seed);
    AdviResult out;
    double eta = options.step_size;
    if (eta == 0) {
        double best = -std::numeric_limits<double>::infinity();
        for (const double candidate : {100.0, 10.0, 1.0, 0.1, 0.01}) {
            Variational v = start;
            double e = -std::numeric_limits<double>::infinity();
            try {
                if (optimise(problem, v, candidate, options.adapt_iterations, false, rng, nullptr))
                    e = problem.elbo(v, rng);
            } catch (const std::exception&) {
                e = -std::numeric_limits<double>::infinity();
            }
            if (std::isfinite(e) && e > best) {
                best = e;
                eta = candidate;
            }
        }
        if (eta == 0)
            throw std::runtime_error("advi: every candidate step size diverged");
    }
    Variational v = start;
    if (!optimise(problem, v, eta, options.max_iterations, true, rng, &out))
        throw std::runtime_error("advi: the optimisation diverged; try a smaller step size");
    out.step_size = eta;
    out.mean = v.mu;
    out.cholesky.assign(d * d, 0.0);
    for (std::size_t a = 0; a < d; ++a) {
        if (options.family == AdviFamily::mean_field) {
            out.cholesky[a * d + a] = std::exp(v.omega[a]);
        } else {
            for (std::size_t b = 0; b <= a; ++b)
                out.cholesky[a * d + b] = v.chol[a * d + b];
        }
    }
    // Report a factor with a positive diagonal (the sign of a column is not identified).
    if (options.family == AdviFamily::full_rank)
        for (std::size_t b = 0; b < d; ++b)
            if (out.cholesky[b * d + b] < 0)
                for (std::size_t a = b; a < d; ++a)
                    out.cholesky[a * d + b] = -out.cholesky[a * d + b];
    out.elbo = out.elbo_history.empty() ? problem.elbo(v, rng) : out.elbo_history.back();
    std::vector<double> eps(d);
    for (std::size_t k = 0; k < options.output_draws; ++k) {
        for (double& e : eps)
            e = rng.normal();
        const auto z = problem.draw(v, eps);
        std::vector<double> theta(d);
        for (std::size_t a = 0; a < d; ++a)
            theta[a] = forward(supports[a], z[a]).theta;
        out.draws.push_back(std::move(theta));
    }
    return out;
}

} // namespace datamunge::bayes
