#include <datamunge/bayes/nuts.hpp>

#include <datamunge/bayes/detail/dual_averaging.hpp>
#include <datamunge/bayes/detail/leapfrog.hpp>

#include <algorithm>
#include <cmath>
#include <random>

namespace datamunge::bayes {

namespace {

using detail::hamiltonian;
using detail::leapfrog_step;
using detail::LeapfrogState;

bool no_u_turn(const std::vector<double>& theta_minus, const std::vector<double>& theta_plus,
               const std::vector<double>& r_minus, const std::vector<double>& r_plus) {
    double dot_minus = 0.0, dot_plus = 0.0;
    for (std::size_t i = 0; i < theta_minus.size(); ++i) {
        const double diff = theta_plus[i] - theta_minus[i];
        dot_minus += diff * r_minus[i];
        dot_plus += diff * r_plus[i];
    }
    return dot_minus >= 0.0 && dot_plus >= 0.0;
}

struct TreeResult {
    LeapfrogState minus;
    LeapfrogState plus;
    std::vector<double> proposal;
    std::size_t n_valid{0};
    bool valid{true};
    bool diverged{false};
    double alpha_sum{0.0};
    std::size_t n_alpha{0};
};

// Recursively doubles the trajectory in direction v from `state`, `j` levels deep (2^j
// leapfrog steps), tracking the slice variable log_u and reporting whether the (sub)tree
// stayed within the slice / hasn't diverged / hasn't made a U-turn.
TreeResult build_tree(optim::DifferentiableFunction& log_posterior, const LeapfrogState& state, const double log_u,
                      const int v, const std::size_t j, const double step_size, const double h0,
                      const double max_delta_error, std::mt19937_64& rng) {
    if (j == 0) {
        const LeapfrogState next = leapfrog_step(log_posterior, state, static_cast<double>(v) * step_size);
        const double h1 = hamiltonian(log_posterior, next.q, next.p);

        TreeResult result;
        result.minus = next;
        result.plus = next;
        result.proposal = next.q;
        result.n_valid = (log_u <= -h1) ? 1 : 0;
        result.valid = std::isfinite(h1) && (log_u < max_delta_error + (-h1));
        result.diverged = !result.valid;
        result.alpha_sum = std::isfinite(h1) ? std::min(1.0, std::exp(h0 - h1)) : 0.0;
        result.n_alpha = 1;
        return result;
    }

    TreeResult result = build_tree(log_posterior, state, log_u, v, j - 1, step_size, h0, max_delta_error, rng);

    if (result.valid) {
        TreeResult other;
        if (v == -1) {
            other = build_tree(log_posterior, result.minus, log_u, v, j - 1, step_size, h0, max_delta_error, rng);
            result.minus = other.minus;
        } else {
            other = build_tree(log_posterior, result.plus, log_u, v, j - 1, step_size, h0, max_delta_error, rng);
            result.plus = other.plus;
        }

        const std::size_t n_total = result.n_valid + other.n_valid;
        if (n_total > 0) {
            std::uniform_real_distribution<double> unif01(0.0, 1.0);
            const double accept_prob = static_cast<double>(other.n_valid) / static_cast<double>(n_total);
            if (unif01(rng) < accept_prob) result.proposal = other.proposal;
        }
        result.alpha_sum += other.alpha_sum;
        result.n_alpha += other.n_alpha;
        result.n_valid = n_total;
        result.valid = other.valid && no_u_turn(result.minus.q, result.plus.q, result.minus.p, result.plus.p);
        result.diverged = result.diverged || other.diverged;
    }

    return result;
}

} // namespace

NUTS::NUTS(NUTSOptions options) : options_(options) {}

NUTSResult NUTS::sample(optim::DifferentiableFunction& log_posterior, const std::vector<double>& initial_params) const {
    const std::size_t d = initial_params.size();
    std::mt19937_64 rng(options_.seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    std::uniform_real_distribution<double> unif01(0.0, 1.0);

    std::vector<double> q = initial_params;
    detail::DualAveraging adaptation(options_.initial_step_size, options_.target_accept_rate);
    double step_size = options_.initial_step_size;

    NUTSResult result;
    result.samples.reserve(options_.num_samples);
    double accept_stat_sum = 0.0;
    const std::size_t total_iters = options_.num_warmup + options_.num_samples;

    for (std::size_t iter = 0; iter < total_iters; ++iter) {
        std::vector<double> r0(d);
        for (double& v : r0) v = normal(rng);
        const double h0 = hamiltonian(log_posterior, q, r0);
        const double log_u = -h0 + std::log(unif01(rng));

        LeapfrogState minus{q, r0};
        LeapfrogState plus{q, r0};
        std::vector<double> proposal = q;
        std::size_t n = 1;
        bool s = true;
        std::size_t j = 0;
        double alpha_sum = 0.0;
        std::size_t n_alpha = 0;
        bool diverged = false;

        while (s && j < options_.max_tree_depth) {
            const int v = (unif01(rng) < 0.5) ? -1 : 1;
            TreeResult subtree;
            if (v == -1) {
                subtree = build_tree(log_posterior, minus, log_u, v, j, step_size, h0, options_.max_delta_error, rng);
                minus = subtree.minus;
            } else {
                subtree = build_tree(log_posterior, plus, log_u, v, j, step_size, h0, options_.max_delta_error, rng);
                plus = subtree.plus;
            }

            if (subtree.valid && subtree.n_valid > 0) {
                const double accept_prob = std::min(1.0, static_cast<double>(subtree.n_valid) / static_cast<double>(n));
                if (unif01(rng) < accept_prob) proposal = subtree.proposal;
            }
            n += subtree.n_valid;
            alpha_sum += subtree.alpha_sum;
            n_alpha += subtree.n_alpha;
            diverged = diverged || subtree.diverged;
            s = subtree.valid && no_u_turn(minus.q, plus.q, minus.p, plus.p);
            ++j;
        }

        q = proposal;
        if (diverged) ++result.num_divergences;
        const double iter_accept_stat = n_alpha > 0 ? alpha_sum / static_cast<double>(n_alpha) : 0.0;
        accept_stat_sum += iter_accept_stat;

        if (iter < options_.num_warmup) {
            adaptation.update(iter_accept_stat);
            step_size = adaptation.step_size();
            if (iter + 1 == options_.num_warmup) step_size = adaptation.finalized_step_size();
        } else {
            result.samples.push_back(q);
        }
    }

    result.accept_rate = accept_stat_sum / static_cast<double>(total_iters);
    result.final_step_size = step_size;
    return result;
}

} // namespace datamunge::bayes
