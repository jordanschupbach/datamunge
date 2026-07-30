#pragma once

/// \file hmm.hpp
/// \brief Discrete (finite-alphabet) hidden Markov model inference: the Viterbi,
///        forward-backward, and Baum-Welch algorithms.
///
/// A hidden Markov model (HMM) describes a doubly stochastic process. A hidden
/// state \f$q_t\f$ evolves as a first-order Markov chain over \f$N\f$ states with
/// initial distribution \f$\pi_i = P(q_1 = i)\f$ and transition matrix
/// \f$A_{ij} = P(q_{t+1}=j \mid q_t=i)\f$. Each state emits one of \f$M\f$ discrete
/// observation symbols according to \f$B_{ik} = P(o_t = k \mid q_t = i)\f$. Only the
/// emitted symbols \f$O = o_1\dots o_T\f$ are observed; the state path is hidden.
///
/// The three classical problems (Rabiner 1989) and their solutions here are:
///   1. Decoding: given \f$\lambda=(\pi,A,B)\f$ and \f$O\f$, find the single most
///      likely hidden path \f$\arg\max_Q P(Q\mid O,\lambda)\f$ -- Viterbi.
///   2. Evaluation / smoothing: compute \f$P(O\mid\lambda)\f$ and the per-time state
///      posteriors \f$\gamma_t(i)=P(q_t=i\mid O,\lambda)\f$ -- forward-backward.
///   3. Learning: given only observation sequences, re-estimate \f$\lambda\f$ to
///      locally maximize their likelihood -- Baum-Welch (an EM algorithm).
///
/// Numerical strategy. Viterbi works in log space so a product of \f$T\f$ small
/// probabilities never underflows. The forward-backward pass uses Rabiner's
/// per-time-step *scaling*: each column of the forward variable is normalized to
/// sum to one and the reciprocals are accumulated, giving \f$\log P(O\mid\lambda)\f$
/// as \f$-\sum_t\log c_t\f$ with no underflow. Baum-Welch drives its E-step from the
/// same scaled recursions.

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace datamunge::algorithms {

/// Parameters of a discrete HMM \f$\lambda=(\pi,A,B)\f$.
struct HiddenMarkovModel {
    std::vector<double>              initial;     ///< \f$\pi_i\f$, length \f$N\f$ (rows sum to 1).
    std::vector<std::vector<double>> transition;  ///< \f$A_{ij}\f$, \f$N\times N\f$ (each row sums to 1).
    std::vector<std::vector<double>> emission;    ///< \f$B_{ik}\f$, \f$N\times M\f$ (each row sums to 1).

    /// Number of hidden states \f$N\f$.
    std::size_t num_states() const { return initial.size(); }
    /// Number of observation symbols \f$M\f$.
    std::size_t num_symbols() const { return emission.empty() ? 0 : emission.front().size(); }
};

/// Result of Viterbi decoding.
struct ViterbiResult {
    std::vector<std::size_t> states;           ///< Most likely hidden state path, length \f$T\f$.
    double                   log_probability;  ///< \f$\log P(\text{path}, O \mid \lambda)\f$.
};

/// Result of the scaled forward-backward pass.
struct ForwardBackwardResult {
    std::vector<std::vector<double>> alpha;  ///< Scaled forward variables \f$\hat\alpha_t(i)\f$, \f$T\times N\f$.
    std::vector<std::vector<double>> beta;   ///< Scaled backward variables \f$\hat\beta_t(i)\f$, \f$T\times N\f$.
    std::vector<std::vector<double>> gamma;  ///< State posteriors \f$\gamma_t(i)=P(q_t=i\mid O)\f$, \f$T\times N\f$.
    std::vector<double>              scale;  ///< Scaling factors \f$c_t\f$, length \f$T\f$.
    double log_likelihood;                   ///< \f$\log P(O\mid\lambda) = -\sum_t \log c_t\f$.
};

/// Result of Baum-Welch training.
struct BaumWelchResult {
    HiddenMarkovModel   model;                   ///< Re-estimated parameters.
    std::vector<double> log_likelihood_history;  ///< Total log-likelihood after each iteration.
    std::size_t         iterations = 0;          ///< Number of EM iterations actually run.
};

namespace detail {

inline void check_observations(const HiddenMarkovModel& hmm, const std::vector<std::size_t>& obs) {
    const std::size_t m = hmm.num_symbols();
    for (const std::size_t o : obs)
        if (o >= m) throw std::out_of_range("hmm: observation symbol out of range");
}

}  // namespace detail

/// \brief Viterbi decoding: the single most likely hidden state sequence.
///
/// Defines \f$\delta_t(i)=\max_{q_1\dots q_{t-1}} \log P(q_1\dots q_{t-1}, q_t=i, o_1\dots o_t\mid\lambda)\f$.
/// The recursion is \f$\delta_t(j)=\big(\max_i \delta_{t-1}(i)+\log A_{ij}\big)+\log B_{j,o_t}\f$, with
/// back-pointers \f$\psi_t(j)\f$ recording the maximizing predecessor; the optimal path is recovered
/// by backtracking from \f$\arg\max_i \delta_T(i)\f$. All arithmetic is in log space.
/// Complexity \f$O(T N^2)\f$ time, \f$O(T N)\f$ space. Zero probabilities map to \f$-\infty\f$.
///
/// \param hmm  Model parameters.
/// \param obs  Observation symbols in \f$\{0,\dots,M-1\}\f$.
/// \return the most likely path and its joint log-probability. Empty for empty input.
inline ViterbiResult viterbi(const HiddenMarkovModel& hmm, const std::vector<std::size_t>& obs) {
    detail::check_observations(hmm, obs);
    const std::size_t n = hmm.num_states();
    const std::size_t t = obs.size();
    ViterbiResult result;
    if (t == 0 || n == 0) {
        result.log_probability = 0.0;
        return result;
    }

    constexpr double kNegInf = -std::numeric_limits<double>::infinity();
    auto             safe_log = [](double p) { return p > 0.0 ? std::log(p) : -std::numeric_limits<double>::infinity(); };

    std::vector<std::vector<double>>      delta(t, std::vector<double>(n, kNegInf));
    std::vector<std::vector<std::size_t>> psi(t, std::vector<std::size_t>(n, 0));

    for (std::size_t i = 0; i < n; ++i) delta[0][i] = safe_log(hmm.initial[i]) + safe_log(hmm.emission[i][obs[0]]);

    for (std::size_t step = 1; step < t; ++step)
        for (std::size_t j = 0; j < n; ++j) {
            double      best     = kNegInf;
            std::size_t best_arg = 0;
            for (std::size_t i = 0; i < n; ++i) {
                const double cand = delta[step - 1][i] + safe_log(hmm.transition[i][j]);
                if (cand > best) {
                    best     = cand;
                    best_arg = i;
                }
            }
            delta[step][j] = best + safe_log(hmm.emission[j][obs[step]]);
            psi[step][j]   = best_arg;
        }

    double      best_final = kNegInf;
    std::size_t last_state = 0;
    for (std::size_t i = 0; i < n; ++i)
        if (delta[t - 1][i] > best_final) {
            best_final = delta[t - 1][i];
            last_state = i;
        }

    result.states.assign(t, 0);
    result.states[t - 1] = last_state;
    for (std::size_t step = t - 1; step > 0; --step) result.states[step - 1] = psi[step][result.states[step]];
    result.log_probability = best_final;
    return result;
}

/// \brief Scaled forward-backward: state posteriors and the sequence likelihood.
///
/// The unscaled forward variable is \f$\alpha_t(i)=P(o_1\dots o_t, q_t=i\mid\lambda)\f$; the backward
/// variable is \f$\beta_t(i)=P(o_{t+1}\dots o_T\mid q_t=i,\lambda)\f$. To avoid underflow each forward
/// column is normalized to sum to one, storing \f$c_t = 1/\sum_j \bar\alpha_t(j)\f$; the backward pass
/// reuses the same \f$c_t\f$. Then \f$\log P(O\mid\lambda)=-\sum_t\log c_t\f$ and the smoothed posterior is
/// \f$\gamma_t(i)\propto\hat\alpha_t(i)\hat\beta_t(i)\f$ (normalized over \f$i\f$). Complexity \f$O(T N^2)\f$.
inline ForwardBackwardResult forward_backward(const HiddenMarkovModel& hmm, const std::vector<std::size_t>& obs) {
    detail::check_observations(hmm, obs);
    const std::size_t n = hmm.num_states();
    const std::size_t t = obs.size();

    ForwardBackwardResult r;
    r.log_likelihood = 0.0;
    if (t == 0 || n == 0) return r;

    r.alpha.assign(t, std::vector<double>(n, 0.0));
    r.beta.assign(t, std::vector<double>(n, 0.0));
    r.gamma.assign(t, std::vector<double>(n, 0.0));
    r.scale.assign(t, 0.0);

    // Forward pass with scaling.
    double sum0 = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        r.alpha[0][i] = hmm.initial[i] * hmm.emission[i][obs[0]];
        sum0 += r.alpha[0][i];
    }
    r.scale[0] = sum0 > 0.0 ? 1.0 / sum0 : 0.0;
    for (std::size_t i = 0; i < n; ++i) r.alpha[0][i] *= r.scale[0];

    for (std::size_t step = 1; step < t; ++step) {
        double sum = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
            double acc = 0.0;
            for (std::size_t i = 0; i < n; ++i) acc += r.alpha[step - 1][i] * hmm.transition[i][j];
            r.alpha[step][j] = acc * hmm.emission[j][obs[step]];
            sum += r.alpha[step][j];
        }
        r.scale[step] = sum > 0.0 ? 1.0 / sum : 0.0;
        for (std::size_t j = 0; j < n; ++j) r.alpha[step][j] *= r.scale[step];
    }

    // Backward pass, reusing the forward scaling factors.
    for (std::size_t i = 0; i < n; ++i) r.beta[t - 1][i] = r.scale[t - 1];
    for (std::size_t step = t - 1; step > 0; --step)
        for (std::size_t i = 0; i < n; ++i) {
            double acc = 0.0;
            for (std::size_t j = 0; j < n; ++j)
                acc += hmm.transition[i][j] * hmm.emission[j][obs[step]] * r.beta[step][j];
            r.beta[step - 1][i] = acc * r.scale[step - 1];
        }

    // Posteriors gamma_t(i), normalized per time step.
    for (std::size_t step = 0; step < t; ++step) {
        double norm = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            r.gamma[step][i] = r.alpha[step][i] * r.beta[step][i];
            norm += r.gamma[step][i];
        }
        if (norm > 0.0)
            for (std::size_t i = 0; i < n; ++i) r.gamma[step][i] /= norm;
    }

    for (std::size_t step = 0; step < t; ++step)
        r.log_likelihood -= (r.scale[step] > 0.0 ? std::log(r.scale[step]) : 0.0);
    return r;
}

/// \brief Baum-Welch (EM) parameter re-estimation from one or more observation sequences.
///
/// Starting from an initial guess \f$\lambda^{(0)}\f$, each iteration runs the scaled forward-backward
/// pass on every sequence (E-step) to form the responsibilities \f$\gamma_t(i)\f$ and pair-posteriors
/// \f$\xi_t(i,j)=P(q_t=i,q_{t+1}=j\mid O,\lambda)\f$, then re-estimates (M-step)
/// \f[
///   \pi_i = \overline{\gamma_1(i)},\quad
///   A_{ij} = \frac{\sum_t \xi_t(i,j)}{\sum_t \gamma_t(i)},\quad
///   B_{ik} = \frac{\sum_{t: o_t=k}\gamma_t(i)}{\sum_t \gamma_t(i)},
/// \f]
/// summed over all sequences. The total log-likelihood \f$\sum_s\log P(O^{(s)}\mid\lambda)\f$ is
/// non-decreasing and the loop stops when its increase drops below \p tol or \p max_iterations is hit.
/// Baum-Welch converges to a *local* optimum; the result depends on the initial model.
///
/// \param sequences       Training observation sequences (symbols in \f$\{0,\dots,M-1\}\f$).
/// \param initial_model   Starting parameters; also fixes \f$N\f$ and \f$M\f$.
/// \param max_iterations  Iteration cap.
/// \param tolerance       Stop when the log-likelihood gain is below this.
inline BaumWelchResult baum_welch(const std::vector<std::vector<std::size_t>>& sequences,
                                  HiddenMarkovModel                            initial_model,
                                  std::size_t                                  max_iterations = 100,
                                  double                                       tolerance      = 1e-6) {
    const std::size_t n = initial_model.num_states();
    const std::size_t m = initial_model.num_symbols();
    BaumWelchResult   out;
    out.model = initial_model;
    if (n == 0 || m == 0 || sequences.empty()) return out;

    double prev_ll = -std::numeric_limits<double>::infinity();
    for (std::size_t iter = 0; iter < max_iterations; ++iter) {
        std::vector<double>              pi_acc(n, 0.0);
        std::vector<std::vector<double>> a_num(n, std::vector<double>(n, 0.0));
        std::vector<double>              a_den(n, 0.0);
        std::vector<std::vector<double>> b_num(n, std::vector<double>(m, 0.0));
        std::vector<double>              b_den(n, 0.0);
        double                           total_ll = 0.0;
        std::size_t                      used     = 0;

        for (const auto& obs : sequences) {
            const std::size_t t = obs.size();
            if (t == 0) continue;
            ++used;
            const ForwardBackwardResult fb = forward_backward(out.model, obs);
            total_ll += fb.log_likelihood;

            for (std::size_t i = 0; i < n; ++i) pi_acc[i] += fb.gamma[0][i];

            for (std::size_t step = 0; step + 1 < t; ++step) {
                // xi_t(i,j) proportional to alpha_t(i) A_ij B_{j,o_{t+1}} beta_{t+1}(j); normalize over (i,j).
                double norm = 0.0;
                std::vector<std::vector<double>> xi(n, std::vector<double>(n, 0.0));
                for (std::size_t i = 0; i < n; ++i)
                    for (std::size_t j = 0; j < n; ++j) {
                        xi[i][j] = fb.alpha[step][i] * out.model.transition[i][j] *
                                   out.model.emission[j][obs[step + 1]] * fb.beta[step + 1][j];
                        norm += xi[i][j];
                    }
                if (norm > 0.0)
                    for (std::size_t i = 0; i < n; ++i)
                        for (std::size_t j = 0; j < n; ++j) {
                            const double v = xi[i][j] / norm;
                            a_num[i][j] += v;
                            a_den[i] += v;
                        }
            }
            for (std::size_t step = 0; step < t; ++step)
                for (std::size_t i = 0; i < n; ++i) {
                    b_num[i][obs[step]] += fb.gamma[step][i];
                    b_den[i] += fb.gamma[step][i];
                }
        }

        // M-step.
        if (used > 0)
            for (std::size_t i = 0; i < n; ++i) out.model.initial[i] = pi_acc[i] / static_cast<double>(used);
        for (std::size_t i = 0; i < n; ++i) {
            if (a_den[i] > 0.0)
                for (std::size_t j = 0; j < n; ++j) out.model.transition[i][j] = a_num[i][j] / a_den[i];
            if (b_den[i] > 0.0)
                for (std::size_t k = 0; k < m; ++k) out.model.emission[i][k] = b_num[i][k] / b_den[i];
        }

        out.log_likelihood_history.push_back(total_ll);
        out.iterations = iter + 1;
        if (total_ll - prev_ll < tolerance && iter > 0) break;
        prev_ll = total_ll;
    }
    return out;
}

}  // namespace datamunge::algorithms
