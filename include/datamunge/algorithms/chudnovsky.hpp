#pragma once
/// @file chudnovsky.hpp
/// @brief Stable double Chudnovsky recurrence, actual term records and theoretical truncation diagnostics.
#include <vector>
namespace datamunge::algorithms {
/// @brief Actual stopping reason for a Chudnovsky evaluation.
/// RequestedTerms means all requested additions ran. TermUnderflow means the
/// recurrence produced a signed zero term; that addition is recorded and every
/// later requested term is skipped. Neither outcome certifies correctly rounded
/// pi or a real-error tolerance. Default storage has not run an evaluation.
/// @par Example
/// @code
/// const auto t=dm::algorithms::chudnovsky_pi_trace(4);
/// std::print("{}\n",t.stop==dm::algorithms::ChudnovskyStop::RequestedTerms); // true
/// @endcode
/// @see ChudnovskyTrace
enum class ChudnovskyStop {
  Unprocessed, ///< Default storage; no evaluation has occurred.
  RequestedTerms, ///< All requested terms, including unchanged additions, were evaluated.
  TermUnderflow ///< The actual floating term became zero; its addition is included.
};
/// @brief Owned actual term, accumulation and estimate from the executing double kernel.
/// term is a_k=(-1)^k(6k)!(A+Bk)/((3k)!(k!)^3 C^(3k)), with
/// A=13591409, B=545140134 and C=640320. The estimate is K/sum_after,
/// K=426880*sqrt(10005), in double arithmetic. ratio_to_previous is zero at
/// index zero, where no ratio was evaluated. sum_changed tests actual equality,
/// not mathematical convergence; term can remain nonzero after additions stall.
/// @par Example
/// @code
/// const auto s=dm::algorithms::chudnovsky_pi_trace(3).steps[2];
/// std::print("{} {}\n",s.index,s.sum_changed); // 2 false
/// @endcode
/// @see chudnovsky_pi_trace(), ChudnovskyTrace
struct ChudnovskyStep {
  int index{0}; ///< Actual zero-based series index.
  double ratio_to_previous{0}; ///< Actual a_k/a_(k-1) recurrence factor; zero when index is zero.
  double term{0}; ///< Actual signed coefficient term, including a final underflowed zero.
  double sum_before{0}; ///< Accumulator immediately before this addition.
  double sum_after{0}; ///< Accumulator immediately after this addition.
  double estimate{0}; ///< Actual K/sum_after, a rounded double estimate of pi.
  bool sum_changed{false}; ///< Whether this actual addition changed the double accumulator.
};
/// @brief Owned actual Chudnovsky prefix records, stop reason and operation counts.
/// Ordinary and traced calls share one kernel without replay. Positive requested
/// counts are honoured through the first underflowed term; stagnation alone does
/// not stop the loop. The first underflowed zero is included and suffix terms are
/// skipped. first_unchanged_index is -1 if every addition changed the sum.
/// No input or shared mutable state is retained. Default construction performs no
/// evaluation. Trace records add O(m) storage for m actual evaluated terms.
/// @par Example
/// @code
/// const auto t=dm::algorithms::chudnovsky_pi_trace(4);
/// std::print("{} {} {}\n",t.terms_evaluated,t.first_unchanged_index,t.ratio_evaluations); // 4 2 3
/// @endcode
/// @see chudnovsky_pi_trace(), ChudnovskyStop
struct ChudnovskyTrace {
  double value{0}; ///< Same final rounded estimate as chudnovsky_pi().
  int terms_requested{0}; ///< Positive requested term count, zero for default storage.
  int terms_evaluated{0}; ///< Actual term/addition visits, including a final underflowed zero.
  int first_unchanged_index{-1}; ///< First actual unchanged addition, or -1 if absent.
  int ratio_evaluations{0}; ///< Actual recurrence-factor calculations; none for the initial term.
  int sum_additions{0}; ///< Actual sum += term operations, including zero/unchanged terms.
  ChudnovskyStop stop{ChudnovskyStop::Unprocessed}; ///< Actual requested-count or zero-term boundary.
  std::vector<ChudnovskyStep> steps; ///< Owned actual prefix snapshots; never a replay.
};
/// @brief Compute the signed ratio a_(k+1)/a_k without forming factorials or large powers.
/// Uses bounded rational factors, tending to -1728/640320^3. The linear factor
/// A+Bk is evaluated after conversion to double, so index arithmetic does not
/// overflow at INT_MAX. Constant floating work, no allocation or mutation.
/// @param index Nonnegative current index k, including INT_MAX.
/// @return Rounded negative recurrence factor, finite and nonzero.
/// @throws std::invalid_argument If index is negative.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::chudnovsky_term_ratio(0)<0); // true
/// @endcode
/// @par References
/// - D. V. and G. V. Chudnovsky (1989), The computation of classical constants,
///   PNAS 86(21), 8178-8182, https://doi.org/10.1073/pnas.86.21.8178.
///   The recurrence follows by cancelling consecutive factorial terms.
/// @see chudnovsky_pi(), chudnovsky_digits_per_term()
double chudnovsky_term_ratio(int index);
/// @brief Return the asymptotic exact-series decimal gain log10(640320^3/1728).
/// Approximately 14.1816474627 digits per added term when arithmetic precision
/// is sufficient. This is not a measured gain or accuracy guarantee for a double
/// result. Constant floating work; no allocation, mutation or shared state.
/// @return Rounded theoretical asymptotic decimal digits per term.
/// @par Example
/// @code
/// std::print("{:.6f}\n",dm::algorithms::chudnovsky_digits_per_term()); // 14.181647
/// @endcode
/// @par References
/// - D. V. and G. V. Chudnovsky (1989), https://doi.org/10.1073/pnas.86.21.8178.
/// @see chudnovsky_term_ratio(), chudnovsky_log10_truncation_bound()
double chudnovsky_digits_per_term();
/// @brief Evaluate the log10 of a geometric bound for the exact N-term truncation error in pi.
/// Let K=426880*sqrt(10005), A=13591409, b1=120*(A+545140134)/640320^3,
/// rho=1728/640320^3 and L=A-b1. Exact arithmetic gives
/// |K/S_N-pi| <= K*b1*rho^(N-1)/L^2 for N>=1. The bound uses alternating,
/// decreasing terms and |a_(k+1)/a_k|<=rho for k>=1. Evaluation uses logarithms
/// directly, so the bound does not underflow for large positive int counts.
/// This is a rounded evaluation of a theoretical bound, not an outward-rounded
/// interval or a bound on floating-point evaluation/normalizer/division error.
/// O(1) floating work, no allocation or mutation.
/// @param terms Positive number N of mathematical terms, independent of trace early stopping.
/// @return Rounded log10 of the exact-arithmetic truncation bound; finite and negative.
/// @throws std::invalid_argument If terms is nonpositive.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::chudnovsky_log10_truncation_bound(2)<-27); // true
/// @endcode
/// @par References
/// - D. V. and G. V. Chudnovsky (1989), https://doi.org/10.1073/pnas.86.21.8178.
///   The stated geometric error bound is derived from the term ratio.
/// @see chudnovsky_pi_trace(), chudnovsky_digits_per_term()
double chudnovsky_log10_truncation_bound(int terms);
/// @brief Approximate pi with a stable double Chudnovsky term recurrence.
/// Replaces direct factorial/power evaluation while retaining the call signature.
/// Uses K/S_N with K=426880*sqrt(10005); the rounding path can change the last
/// bits of the old implementation. Requires terms>=1 (previous nonpositive
/// requests divided by zero). Every requested term is added through the first
/// underflowed zero; later terms stay zero and are skipped. Stagnation alone does
/// not stop evaluation. No factorial overflow, arbitrary-precision arithmetic or
/// binary splitting is used. On ordinary binary64, accuracy plateaus after about
/// two terms; larger counts cannot guarantee more correct digits.
/// O(m) floating work, O(1) auxiliary storage for m actual terms through underflow;
/// no allocation, input mutation or shared state. Normal floating-point rounding
/// and underflow are assumed, without fast-math reassociation or trapping modes.
/// @param terms Positive requested number of series terms.
/// @return Finite rounded estimate of pi, not a certified correctly rounded result.
/// @throws std::invalid_argument If terms is nonpositive.
/// @par Example
/// @code
/// std::print("{:.12f}\n",dm::algorithms::chudnovsky_pi(2)); // 3.141592653590
/// @endcode
/// @par References
/// - D. V. and G. V. Chudnovsky (1989), The computation of classical constants,
///   PNAS 86(21), 8178-8182, https://doi.org/10.1073/pnas.86.21.8178.
/// @see chudnovsky_pi_trace(), chudnovsky_log10_truncation_bound()
double chudnovsky_pi(int terms);
/// @brief Record each actual Chudnovsky term, addition and rounded prefix estimate without replay.
/// Same recurrence, operation order and final estimate as chudnovsky_pi(). Records
/// include nonzero terms whose sum no longer changes and the first underflowed
/// zero; no huge reservation is made from the requested count. Underflow and
/// unchanged additions do not certify real accuracy. O(m) work and owned storage
/// for m actual evaluated terms. No partial trace escapes allocation failure.
/// @param terms Positive requested count; suffix after term underflow is skipped.
/// @return Owned actual prefix records, counts, stopping reason and final estimate.
/// @throws std::invalid_argument If terms is nonpositive.
/// @throws std::length_error If actual trace storage exceeds container capacity.
/// @throws std::bad_alloc If record allocation fails.
/// @par Example
/// @code
/// const auto t=dm::algorithms::chudnovsky_pi_trace(4);
/// std::print("{} {}\n",t.steps.size(),t.sum_additions); // 4 4
/// @endcode
/// @par References
/// - D. V. and G. V. Chudnovsky (1989), https://doi.org/10.1073/pnas.86.21.8178.
/// @see chudnovsky_pi(), ChudnovskyTrace
ChudnovskyTrace chudnovsky_pi_trace(int terms);
}
