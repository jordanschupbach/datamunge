#pragma once

/// @file pps_sampling.hpp
/// @brief Fixed-size unequal-probability (probability-proportional-to-size) sampling designs and their joint inclusion probabilities.

#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/random/random.hpp>

#include <cstddef>
#include <vector>

namespace datamunge::stats {

/// @brief Fixed-size sampling design that reproduces declared first-order inclusion probabilities.
/// @par Example
/// @code
/// const auto m = dm::stats::PpsMethod::sampford;
/// std::println("{}", m == dm::stats::PpsMethod::brewer); // false
/// @endcode
enum class PpsMethod {
    systematic,            ///< Systematic sampling along the list order: one uniform start, so many pairs have \f$\pi_{kl}=0\f$.
    randomized_systematic, ///< Systematic sampling after a uniformly random permutation of the list (Goodman–Kish).
    sampford,              ///< Sampford's rejective method: all \f$\pi_{kl}>0\f$, exact joint probabilities available.
    brewer,                ///< Brewer's draw-by-draw method: one draw per sample unit, no rejection.
    pivotal                ///< Ordered pivotal method (Deville–Tillé): spreads the sample along the list order.
};

/// @brief Inclusion probabilities proportional to size for a sample of n, capping large units at 1.
///
/// Sets \f$\pi_k=n\,x_k/\sum_jx_j\f$. Units with \f$\pi_k\ge1\f$ become certainty units and the
/// remaining sample size is reallocated proportionally among the rest, repeating until no
/// probability exceeds 1, as `sampling::inclusionprobabilities`.
/// @param sizes Nonnegative size measures.
/// @param n Sample size, at most the number of units with positive size.
/// @return Inclusion probabilities summing to n.
/// @throws std::invalid_argument For negative or non-finite sizes, n = 0, or n larger than the number of positive sizes.
/// @par Example
/// @code
/// const auto pi = dm::stats::inclusion_probabilities({1, 1, 2, 16}, 2);
/// std::println("{:.2f} {:.2f} {:.2f}", pi[0], pi[2], pi[3]); // 0.25 0.50 1.00
/// @endcode
/// @see pps_sample()
[[nodiscard]] std::vector<double> inclusion_probabilities(const std::vector<double>& sizes, std::size_t n);

/// @brief Draws a fixed-size sample with the given first-order inclusion probabilities.
///
/// The probabilities must lie in [0, 1] and sum to an integer n (to within 1e-6). Units with
/// \f$\pi_k=1\f$ are always selected and units with \f$\pi_k=0\f$ never are. Every method has the
/// declared first-order probabilities exactly and returns exactly n distinct units; they differ in
/// the joint inclusion probabilities. Systematic sampling along a meaningful list order gives
/// implicit stratification but zero joint probabilities for many pairs, so no unbiased variance
/// estimator exists. Sampford, Brewer and randomized systematic sampling give all
/// \f$\pi_{kl}>0\f$. Sampford uses rejection: its expected number of attempts grows when n is large
/// relative to the number of non-certainty units with large probabilities.
/// @param inclusion First-order inclusion probabilities.
/// @param method Sampling design.
/// @param rng Random generator.
/// @return Indices of the selected units in increasing order.
/// @throws std::invalid_argument If probabilities are outside [0, 1] or do not sum to an integer.
/// @throws std::runtime_error If Sampford sampling needs more than one million attempts.
/// @par Example
/// @code
/// dm::random::SplitMix64 rng(4);
/// const auto s = dm::stats::pps_sample({0.2, 0.4, 0.6, 0.8}, dm::stats::PpsMethod::sampford, rng);
/// std::println("{}", s.size()); // 2
/// @endcode
/// @par References
/// - M. R. Sampford (1967). "On sampling without replacement with unequal probabilities of selection."
///   Biometrika 54(3/4), 499-513. doi:10.1093/biomet/54.3-4.499
/// - K. R. W. Brewer (1975). "A simple procedure for sampling pi-pswor." Australian Journal of
///   Statistics 17(3), 166-172. doi:10.1111/j.1467-842X.1975.tb00954.x
/// - J.-C. Deville and Y. Tillé (1998). "Unequal probability sampling without replacement through a
///   splitting method." Biometrika 85(1), 89-101. doi:10.1093/biomet/85.1.89
/// - Y. Tillé (2006). Sampling Algorithms. Springer.
/// @see inclusion_probabilities(), systematic_joint_inclusion(), sampford_joint_inclusion()
[[nodiscard]] std::vector<std::size_t> pps_sample(const std::vector<double>& inclusion, PpsMethod method,
                                                  random::RandomGenerator& rng);

/// @brief Exact joint inclusion probabilities of systematic sampling in the given list order.
///
/// With cumulative sums \f$V_k\f$ and start \f$u\sim U(0,1)\f$, unit k is selected when an integer
/// falls in \f$[V_{k-1}-u,V_k-u)\f$, i.e. when u lies on an arc of length \f$\pi_k\f$ of the unit
/// circle; \f$\pi_{kl}\f$ is the length of the overlap of two arcs. Matches
/// `sampling::UPsystematicpi2`.
/// @param inclusion First-order inclusion probabilities in list order, summing to an integer.
/// @return Symmetric matrix with \f$\pi_{kl}\f$ off the diagonal and \f$\pi_k\f$ on it.
/// @throws std::invalid_argument As for pps_sample().
/// @par Example
/// @code
/// const auto p = dm::stats::systematic_joint_inclusion({0.5, 0.5, 0.5, 0.5});
/// std::println("{} {}", p(0, 1), p(0, 2)); // 0 0.5
/// @endcode
/// @see pps_sample()
[[nodiscard]] linalg::DenseMatrix<double> systematic_joint_inclusion(const std::vector<double>& inclusion);

/// @brief Exact joint inclusion probabilities of Sampford's design.
///
/// With \f$r_k=\pi_k/(1-\pi_k)\f$ and \f$e_s\f$ the elementary symmetric polynomials of the
/// non-certainty units' \f$r\f$ values,
/// \f[\pi_{kl}=r_kr_l\frac{\sum_{s=0}^{n-2}(n-s-\pi_k-\pi_l)\,e_s^{(-k,-l)}}{\sum_{s=0}^{n-1}(n-s)\,e_s},\f]
/// where \f$e^{(-k,-l)}\f$ omits units k and l (Asok and Sukhatme's form of Sampford's result, as
/// `sampling::UPsampfordpi2`). The polynomials are formed by stable products rather than
/// Newton's identities. Certainty units have \f$\pi_{kl}=\pi_l\f$.
/// @param inclusion First-order inclusion probabilities summing to an integer.
/// @return Symmetric matrix with \f$\pi_{kl}\f$ off the diagonal and \f$\pi_k\f$ on it.
/// @throws std::invalid_argument As for pps_sample().
/// @throws std::runtime_error If the polynomials overflow (very large samples).
/// @par Example
/// @code
/// const auto p = dm::stats::sampford_joint_inclusion({0.2, 0.4, 0.6, 0.8});
/// std::println("{:.4f}", p(2, 3)); // 0.4277
/// @endcode
/// @par References
/// - C. Asok and B. V. Sukhatme (1976). "On Sampford's procedure of unequal probability sampling
///   without replacement." JASA 71(356), 912-918. doi:10.1080/01621459.1976.10480968
/// @see pps_sample()
[[nodiscard]] linalg::DenseMatrix<double> sampford_joint_inclusion(const std::vector<double>& inclusion);

} // namespace datamunge::stats
