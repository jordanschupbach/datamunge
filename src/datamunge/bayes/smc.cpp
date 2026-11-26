/// @file smc.cpp
/// @brief Adaptive tempering, systematic resampling and random-walk moves of the SMC sampler.
#include <datamunge/bayes/smc.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace datamunge::bayes {
/// @brief Implementation helpers local to this translation unit.
namespace {
/// @brief Log of the sum of exponentials.
/// @param v Values.
/// @return log(sum(exp(v))).
double log_sum_exp(const std::vector<double>& v) {
    const double m = *std::ranges::max_element(v);
    if (!std::isfinite(m))
        return m;
    double s = 0;
    for (const double x : v)
        s += std::exp(x - m);
    return m + std::log(s);
}

/// @brief ESS of normalised log weights.
/// @param lw Normalised log weights.
/// @return 1 / sum of squared weights.
double ess_of(const std::vector<double>& lw) {
    double s = 0;
    for (const double x : lw)
        s += std::exp(2 * x);
    return 1 / s;
}

/// @brief Log incremental normaliser and conditional ESS fraction for a temperature step.
/// @param lw Normalised log weights.
/// @param ll Log likelihoods.
/// @param delta Temperature increment.
/// @param log_increment Receives log sum W_i w_i.
/// @return Conditional ESS divided by N, in (0, 1].
double conditional_ess(const std::vector<double>& lw, const std::vector<double>& ll, double delta,
                       double& log_increment) {
    const std::size_t n = lw.size();
    std::vector<double> a(n), b(n);
    for (std::size_t i = 0; i < n; ++i) {
        a[i] = lw[i] + delta * ll[i];
        b[i] = lw[i] + 2 * delta * ll[i];
    }
    log_increment = log_sum_exp(a);
    return std::exp(2 * log_increment - log_sum_exp(b));
}

/// @brief Lower Cholesky factor of a symmetric matrix with diagonal jitter on failure.
/// @param c Matrix (row-major d x d).
/// @param d Dimension.
/// @return Lower-triangular factor (row-major).
std::vector<double> cholesky(std::vector<double> c, std::size_t d) {
    double jitter = 0;
    for (int attempt = 0; attempt < 30; ++attempt) {
        std::vector<double> l(d * d, 0.0);
        bool ok = true;
        for (std::size_t j = 0; j < d && ok; ++j) {
            double s = c[j * d + j] + jitter;
            for (std::size_t k = 0; k < j; ++k)
                s -= l[j * d + k] * l[j * d + k];
            if (!(s > 0)) {
                ok = false;
                break;
            }
            l[j * d + j] = std::sqrt(s);
            for (std::size_t i = j + 1; i < d; ++i) {
                double t = c[i * d + j];
                for (std::size_t k = 0; k < j; ++k)
                    t -= l[i * d + k] * l[j * d + k];
                l[i * d + j] = t / l[j * d + j];
            }
        }
        if (ok)
            return l;
        double scale = 0;
        for (std::size_t j = 0; j < d; ++j)
            scale = std::max(scale, std::abs(c[j * d + j]));
        jitter = jitter == 0 ? 1e-10 * std::max(scale, 1e-300) : jitter * 10;
    }
    std::vector<double> l(d * d, 0.0);
    for (std::size_t j = 0; j < d; ++j)
        l[j * d + j] = 1;
    return l;
}
} // namespace

SmcResult sequential_monte_carlo(const SmcLogDensity& log_prior, const SmcLogDensity& log_likelihood,
                                 const SmcPriorSampler& prior_sampler, const SmcOptions& options) {
    const std::size_t n = options.particles;
    if (n < 2 || !(options.target_cess > 0 && options.target_cess < 1) ||
        !(options.resample_threshold >= 0 && options.resample_threshold <= 1) || !(options.initial_scale > 0) ||
        !(options.target_acceptance > 0 && options.target_acceptance < 1) || options.max_stages == 0)
        throw std::invalid_argument("sequential_monte_carlo: bad options");
    const auto& schedule = options.temperatures;
    if (!schedule.empty()) {
        if (schedule.back() != 1.0 || !(schedule.front() > 0))
            throw std::invalid_argument("sequential_monte_carlo: the schedule must rise from above 0 to 1");
        for (std::size_t j = 1; j < schedule.size(); ++j)
            if (!(schedule[j] > schedule[j - 1]))
                throw std::invalid_argument("sequential_monte_carlo: the schedule must increase");
    }
    random::SplitMix64 rng(options.seed);
    std::vector<std::vector<double>> x(n);
    std::vector<double> lp(n), ll(n), lw(n, -std::log(static_cast<double>(n)));
    for (std::size_t i = 0; i < n; ++i) {
        x[i] = prior_sampler(rng);
        if (x[i].empty() || x[i].size() != x[0].size())
            throw std::invalid_argument("sequential_monte_carlo: prior draws must share a nonzero dimension");
        lp[i] = log_prior(x[i]);
        ll[i] = log_likelihood(x[i]);
        if (!std::isfinite(ll[i]) || !std::isfinite(lp[i]))
            throw std::runtime_error("sequential_monte_carlo: prior or likelihood not finite at a prior draw");
    }
    const std::size_t d = x[0].size();
    SmcResult out;
    double phi = 0, scale = options.initial_scale;
    while (phi < 1) {
        if (out.stages.size() >= (schedule.empty() ? options.max_stages : schedule.size()))
            throw std::runtime_error("sequential_monte_carlo: too many tempering stages");
        double next = 1, inc = 0;
        if (!schedule.empty()) {
            next = schedule[out.stages.size()];
            conditional_ess(lw, ll, next - phi, inc);
        } else if (conditional_ess(lw, ll, 1 - phi, inc) < options.target_cess) {
            double lo = 0, hi = 1 - phi;
            for (int it = 0; it < 60; ++it) {
                const double mid = 0.5 * (lo + hi);
                double tmp = 0;
                (conditional_ess(lw, ll, mid, tmp) >= options.target_cess ? lo : hi) = mid;
            }
            next = phi + std::max(lo, 1e-12);
            conditional_ess(lw, ll, next - phi, inc);
        }
        // Reweight.
        for (std::size_t i = 0; i < n; ++i)
            lw[i] += (next - phi) * ll[i];
        const double norm = log_sum_exp(lw);
        for (double& v : lw)
            v -= norm;
        phi = next;
        SmcStage stage;
        stage.temperature = phi;
        stage.log_increment = inc;
        stage.ess = ess_of(lw);
        out.log_evidence += inc;
        // Systematic resampling.
        if (stage.ess < options.resample_threshold * static_cast<double>(n)) {
            stage.resampled = true;
            const double u = rng.uniform01() / static_cast<double>(n);
            std::vector<std::vector<double>> nx(n);
            std::vector<double> nlp(n), nll(n);
            double cum = std::exp(lw[0]);
            std::size_t j = 0;
            for (std::size_t i = 0; i < n; ++i) {
                const double target = u + static_cast<double>(i) / static_cast<double>(n);
                while (cum < target && j + 1 < n)
                    cum += std::exp(lw[++j]);
                nx[i] = x[j];
                nlp[i] = lp[j];
                nll[i] = ll[j];
            }
            x = std::move(nx);
            lp = std::move(nlp);
            ll = std::move(nll);
            std::ranges::fill(lw, -std::log(static_cast<double>(n)));
        }
        // Random-walk Metropolis moves targeting prior x likelihood^phi.
        std::vector<double> mean(d, 0.0), cov(d * d, 0.0);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t a = 0; a < d; ++a)
                mean[a] += std::exp(lw[i]) * x[i][a];
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t a = 0; a < d; ++a)
                for (std::size_t b = 0; b < d; ++b)
                    cov[a * d + b] += std::exp(lw[i]) * (x[i][a] - mean[a]) * (x[i][b] - mean[b]);
        if (options.proposal == SmcProposal::diagonal)
            for (std::size_t a = 0; a < d; ++a)
                for (std::size_t b = 0; b < d; ++b)
                    if (a != b)
                        cov[a * d + b] = 0;
        const std::vector<double> chol = cholesky(cov, d);
        const double step = 2.38 * scale / std::sqrt(static_cast<double>(d));
        std::size_t accepted = 0, proposed = 0;
        std::vector<double> z(d), y(d);
        for (std::size_t sweep = 0; sweep < options.mcmc_steps; ++sweep)
            for (std::size_t i = 0; i < n; ++i) {
                for (std::size_t a = 0; a < d; ++a)
                    z[a] = rng.normal();
                for (std::size_t a = 0; a < d; ++a) {
                    double s = 0;
                    for (std::size_t b = 0; b <= a; ++b)
                        s += chol[a * d + b] * z[b];
                    y[a] = x[i][a] + step * s;
                }
                ++proposed;
                const double lpy = log_prior(y);
                if (!std::isfinite(lpy))
                    continue;
                const double lly = log_likelihood(y);
                if (std::isnan(lly))
                    continue;
                const double log_ratio = lpy + phi * lly - lp[i] - phi * ll[i];
                if (std::log(rng.uniform01()) < log_ratio) {
                    x[i] = y;
                    lp[i] = lpy;
                    ll[i] = lly;
                    ++accepted;
                }
            }
        stage.acceptance_rate = proposed ? static_cast<double>(accepted) / static_cast<double>(proposed) : 0.0;
        stage.scale = scale;
        scale *= std::exp(stage.acceptance_rate - options.target_acceptance);
        out.stages.push_back(stage);
    }
    out.particles = std::move(x);
    out.log_weights = std::move(lw);
    out.log_likelihoods = std::move(ll);
    out.ess = ess_of(out.log_weights);
    return out;
}

} // namespace datamunge::bayes
