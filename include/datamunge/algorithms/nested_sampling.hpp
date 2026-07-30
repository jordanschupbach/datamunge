#pragma once

/// \file nested_sampling.hpp
/// \brief Skilling's nested sampling: estimate the Bayesian evidence (marginal
///        likelihood) and posterior from a likelihood over a bounded prior
///        (Skilling 2004).
///
/// Bayesian model comparison needs the *evidence*
/// \f$Z=\int \mathcal{L}(\theta)\,\pi(\theta)\,d\theta\f$ -- the likelihood averaged
/// over the prior -- which is a hard high-dimensional integral. *Nested sampling*
/// turns it into a one-dimensional integral over the *prior mass*
/// \f$X(\lambda)=\int_{\mathcal{L}(\theta)>\lambda}\pi(\theta)\,d\theta\f$, the prior
/// fraction with likelihood above a level \f$\lambda\f$: then
/// \f$Z=\int_0^1 \mathcal{L}(X)\,dX\f$.
///
/// The algorithm keeps \f$N\f$ *live points* drawn from the prior. Each iteration it
/// removes the point of lowest likelihood \f$\mathcal{L}_i\f$ (a "dead" point),
/// attributes to it the shrinking prior-mass shell \f$w_i \approx X_{i-1}-X_i\f$ with
/// \f$X_i\approx e^{-i/N}\f$, accumulates \f$Z \mathrel{+}= \mathcal{L}_i w_i\f$, and
/// replaces it with a fresh point drawn from the prior *subject to*
/// \f$\mathcal{L}>\mathcal{L}_i\f$ (here by a constrained random walk). As the live
/// set climbs the likelihood, the enclosed prior mass falls geometrically, sweeping
/// out \f$Z\f$. The dead points, weighted by \f$w_i\mathcal{L}_i/Z\f$, are posterior
/// samples for free.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <random>
#include <vector>

namespace datamunge::algorithms {

/// Result of a nested-sampling run.
struct NestedSamplingResult {
    double                           log_evidence       = -std::numeric_limits<double>::infinity();  ///< log Z.
    double                           log_evidence_error = 0.0;   ///< sqrt(H/N), the standard uncertainty on log Z.
    double                           information        = 0.0;   ///< Kullback-Leibler information H (nats).
    std::size_t                      iterations         = 0;
    std::vector<std::vector<double>> posterior_samples;          ///< Dead points (posterior-weighted).
    std::vector<double>              posterior_log_weights;      ///< Normalized log posterior weight per dead point.
};

namespace detail {

inline double logaddexp(double a, double b) {
    if (a == -std::numeric_limits<double>::infinity()) return b;
    if (b == -std::numeric_limits<double>::infinity()) return a;
    const double m = a > b ? a : b;
    return m + std::log1p(std::exp(-std::fabs(a - b)));
}

}  // namespace detail

/// \brief Nested sampling over a box prior \f$[\mathrm{lo},\mathrm{hi}]^d\f$ (uniform prior).
///
/// \param log_likelihood  \f$\log\mathcal{L}(\theta)\f$.
/// \param lo, hi          Per-dimension bounds of the uniform prior box.
/// \param num_live        Number of live points \f$N\f$.
/// \param max_iterations  Iteration cap.
/// \param seed            RNG seed.
/// \param walk_steps      MCMC steps used to draw each constrained replacement.
inline NestedSamplingResult nested_sampling(const std::function<double(const std::vector<double>&)>& log_likelihood,
                                            const std::vector<double>& lo, const std::vector<double>& hi,
                                            std::size_t num_live = 100, std::size_t max_iterations = 5000,
                                            std::uint64_t seed = 0, std::size_t walk_steps = 30) {
    const std::size_t d = lo.size();
    std::mt19937_64   rng(seed);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::normal_distribution<double>       gauss(0.0, 1.0);

    auto sample_prior = [&]() {
        std::vector<double> p(d);
        for (std::size_t k = 0; k < d; ++k) p[k] = lo[k] + unit(rng) * (hi[k] - lo[k]);
        return p;
    };
    auto in_box = [&](const std::vector<double>& p) {
        for (std::size_t k = 0; k < d; ++k)
            if (p[k] < lo[k] || p[k] > hi[k]) return false;
        return true;
    };

    // Initialize live points.
    std::vector<std::vector<double>> live(num_live);
    std::vector<double>              live_logl(num_live);
    for (std::size_t i = 0; i < num_live; ++i) {
        live[i]      = sample_prior();
        live_logl[i] = log_likelihood(live[i]);
    }
    std::vector<double> step(d);
    for (std::size_t k = 0; k < d; ++k) step[k] = 0.1 * (hi[k] - lo[k]);

    NestedSamplingResult r;
    double               logZ = -std::numeric_limits<double>::infinity();
    double               H    = 0.0;
    double               logwidth = std::log(1.0 - std::exp(-1.0 / static_cast<double>(num_live)));  // log(1 - e^{-1/N})
    std::vector<std::pair<std::vector<double>, double>> dead;  // (point, logL)
    std::vector<double>                                 dead_logwt;

    for (std::size_t iter = 0; iter < max_iterations; ++iter) {
        // Worst (lowest-likelihood) live point.
        std::size_t worst = 0;
        for (std::size_t i = 1; i < num_live; ++i)
            if (live_logl[i] < live_logl[worst]) worst = i;
        const double logL_worst = live_logl[worst];

        const double logwt = logwidth + logL_worst;
        const double logZnew = detail::logaddexp(logZ, logwt);
        if (logZ != -std::numeric_limits<double>::infinity())
            H = std::exp(logwt - logZnew) * logL_worst + std::exp(logZ - logZnew) * (H + logZ) - logZnew;
        logZ = logZnew;

        dead.push_back({live[worst], logL_worst});
        dead_logwt.push_back(logwt);

        // Replace the worst by a constrained random walk from a random surviving live point.
        std::size_t seedpt = worst;
        while (seedpt == worst && num_live > 1) seedpt = static_cast<std::size_t>(unit(rng) * num_live);
        std::vector<double> cur     = live[seedpt];
        double              cur_logl = live_logl[seedpt];
        std::size_t         accepts = 0;
        for (std::size_t s = 0; s < walk_steps; ++s) {
            std::vector<double> prop(d);
            for (std::size_t k = 0; k < d; ++k) prop[k] = cur[k] + step[k] * gauss(rng);
            if (!in_box(prop)) continue;
            const double pl = log_likelihood(prop);
            if (pl > logL_worst) {  // hard likelihood constraint
                cur      = prop;
                cur_logl = pl;
                ++accepts;
            }
        }
        // Adapt the step size toward ~50% acceptance.
        const double acc = static_cast<double>(accepts) / walk_steps;
        for (double& st : step) st *= (acc > 0.5 ? 1.1 : 0.9);
        live[worst]      = cur;
        live_logl[worst] = cur_logl;

        // Shrink the prior mass estimate.
        logwidth -= 1.0 / static_cast<double>(num_live);
        r.iterations = iter + 1;

        // Stop when the largest possible remaining contribution is negligible.
        double max_live = live_logl[0];
        for (std::size_t i = 1; i < num_live; ++i) max_live = std::max(max_live, live_logl[i]);
        const double log_remaining = max_live + logwidth;  // rough upper bound on remaining evidence
        if (log_remaining - logZ < std::log(1e-4)) break;
    }

    // Add the final live points, sharing the remaining prior mass equally.
    const double log_final_width = logwidth - std::log(static_cast<double>(num_live));
    for (std::size_t i = 0; i < num_live; ++i) {
        const double logwt = log_final_width + live_logl[i];
        logZ               = detail::logaddexp(logZ, logwt);
        dead.push_back({live[i], live_logl[i]});
        dead_logwt.push_back(logwt);
    }

    r.log_evidence       = logZ;
    r.information        = H > 0.0 ? H : 0.0;
    r.log_evidence_error = std::sqrt(std::fabs(H) / static_cast<double>(num_live));
    for (std::size_t i = 0; i < dead.size(); ++i) {
        r.posterior_samples.push_back(dead[i].first);
        r.posterior_log_weights.push_back(dead_logwt[i] - logZ);  // normalized posterior weight
    }
    return r;
}

}  // namespace datamunge::algorithms
