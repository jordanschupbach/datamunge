#pragma once

/// @file gaussian_hmm.hpp
/// @brief Hidden Markov models with Gaussian emissions (Baum-Welch fitting, posteriors, Viterbi
///        decoding) and sampling from discrete HMMs.
///
/// A Gaussian HMM has hidden states \f$q_t\in\{0,\dots,K-1\}\f$ following a Markov chain with
/// initial distribution \f$\pi\f$ and transition matrix \f$A\f$, and emits a real value
/// \f$x_t\sim N(\mu_{q_t},\sigma_{q_t}^2)\f$. Baum-Welch fits it by expectation-maximisation using
/// scaled forward-backward recursions.

#include <datamunge/algorithms/hmm.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace datamunge::algorithms {

/// Parameters of a Gaussian-emission HMM.
struct GaussianHmm {
    std::vector<double>              initial;     ///< \f$\pi_i\f$, length K.
    std::vector<std::vector<double>> transition;  ///< \f$A_{ij}\f$, K x K, rows sum to 1.
    std::vector<double>              mean;        ///< \f$\mu_i\f$, length K.
    std::vector<double>              sd;          ///< \f$\sigma_i\f$, length K.
};

/// Options for baum_welch_gaussian().
struct GaussianHmmOptions {
    std::size_t max_iterations = 500;    ///< iteration cap.
    double      tolerance      = 1e-8;   ///< stop when the log-likelihood gain is below this.
    double      min_sd         = 1e-3;   ///< floor on each standard deviation (avoids collapse onto one point).
    bool        independent    = false;  ///< force every row of A to be equal: an i.i.d. Gaussian mixture.
};

/// Result of fitting a Gaussian HMM.
struct GaussianHmmFit {
    GaussianHmm         model;                   ///< fitted parameters.
    std::vector<double> log_likelihood_history;  ///< \f$\log p(x\mid\lambda)\f$ of the model entering each iteration.
    double              log_likelihood = 0.0;    ///< log-likelihood of the final model.
    std::size_t         iterations     = 0;      ///< EM iterations run.
    std::size_t         parameters     = 0;      ///< free parameters (for AIC/BIC).
};

/// @brief A starting model for K states: means at the quantiles (i + 1/2)/K of @p x, a common
///        standard deviation, a uniform initial distribution and transitions with 0.8 on the
///        diagonal.
/// @throws std::invalid_argument if @p x is empty or K is 0.
GaussianHmm gaussian_hmm_initial(const std::vector<double>& x, std::size_t K);

/// @brief Fits a Gaussian HMM to the series @p x by Baum-Welch, starting from @p initial.
///
/// E-step: scaled forward-backward gives \f$\gamma_t(i)=P(q_t=i\mid x)\f$ and
/// \f$\xi_t(i,j)=P(q_t=i,q_{t+1}=j\mid x)\f$. M-step:
/// \f$\pi_i=\gamma_1(i)\f$, \f$A_{ij}=\sum_t\xi_t(i,j)/\sum_t\gamma_t(i)\f$ (t < T),
/// \f$\mu_i=\sum_t\gamma_t(i)x_t/\sum_t\gamma_t(i)\f$ and
/// \f$\sigma_i^2=\sum_t\gamma_t(i)(x_t-\mu_i)^2/\sum_t\gamma_t(i)\f$.
/// @throws std::invalid_argument if @p x is empty or the model shapes disagree.
GaussianHmmFit baum_welch_gaussian(const std::vector<double>& x, GaussianHmm initial,
                                   const GaussianHmmOptions& options = {});

/// @brief State posteriors \f$\gamma_t(i)\f$ (T x K) of the series under @p model; @p log_likelihood
///        receives \f$\log p(x\mid\lambda)\f$ when not null.
std::vector<std::vector<double>> gaussian_hmm_posteriors(const GaussianHmm& model, const std::vector<double>& x,
                                                         double* log_likelihood = nullptr);

/// @brief The most likely state path under @p model (Viterbi, in log space).
std::vector<std::size_t> gaussian_hmm_viterbi(const GaussianHmm& model, const std::vector<double>& x);

/// A sampled path of a discrete HMM.
struct HmmSample {
    std::vector<std::size_t> states;        ///< hidden states \f$q_1..q_T\f$.
    std::vector<std::size_t> observations;  ///< emitted symbols \f$o_1..o_T\f$.
};

/// @brief Samples a state path and observation sequence of length @p T from a discrete HMM.
HmmSample sample_hmm(const HiddenMarkovModel& model, std::size_t T, std::uint64_t seed);

}  // namespace datamunge::algorithms
