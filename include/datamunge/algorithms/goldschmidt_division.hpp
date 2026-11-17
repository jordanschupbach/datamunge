#pragma once
/// @file goldschmidt_division.hpp
/// @brief Preserved multiplicative division with actual state, work and owned update records.
#include <cstddef>
#include <vector>

namespace datamunge::algorithms {
/// @brief Actual normalized numerator and denominator, not a mathematical quotient certificate.
/// Owns two doubles; a scaled numerator may underflow to zero or overflow to infinity.
/// @par Example
/// @code
/// const dm::algorithms::GoldschmidtState s{};
/// const double d = s.denominator; // 0
/// @endcode
/// @see GoldschmidtRun, GoldschmidtStep
struct GoldschmidtState {
  double numerator{0};   ///< Actual scaled numerator, signed as the original dividend.
  double denominator{0}; ///< Actual positive denominator state for a completed run.
};

/// @brief Actual stopping condition, without a quotient-accuracy claim.
/// @par Example
/// @code
/// const auto s = dm::algorithms::GoldschmidtStop::tolerance;
/// const bool same = s == dm::algorithms::GoldschmidtStop::tolerance; // true
/// @endcode
/// @see goldschmidt_stop_name(), GoldschmidtRun
enum class GoldschmidtStop {
  tolerance,      ///< After an update, abs(D-1) < 1e-16 in actual double arithmetic.
  iteration_limit ///< The update budget was exhausted without that denominator test passing.
};

/// @brief Name a native Goldschmidt stop without allocating storage.
/// @param stop Stop to name; an unrecognized enum value is permitted.
/// @return Program-lifetime "tolerance", "iteration_limit", or "unknown".
/// @par Example
/// @code
/// std::print("{}\n",
/// dm::algorithms::goldschmidt_stop_name(dm::algorithms::GoldschmidtStop::tolerance)); // tolerance
/// @endcode
/// @see GoldschmidtStop
inline const char* goldschmidt_stop_name(GoldschmidtStop stop) noexcept {
  switch (stop) {
    case GoldschmidtStop::tolerance:
      return "tolerance";
    case GoldschmidtStop::iteration_limit:
      return "iteration_limit";
  }
  return "unknown";
}

/// @brief Owned scalar output, normalization states and selected loop-operation counts.
/// No callbacks, input references or histories are retained. The ordinary
/// double-returning API uses the same kernel with no record allocation.
/// @par Example
/// @code
/// const auto r = dm::algorithms::goldschmidt_run(3, 4, 1);
/// std::print("{} {}\n", r.updates, r.loop_multiplications); // 1 2
/// @endcode
/// @see goldschmidt_run(), GoldschmidtTrace
struct GoldschmidtRun {
  double           quotient{0};   ///< Final denominator-sign times the actual numerator state.
  double           sign{1};       ///< Denominator sign multiplier, +1 or -1.
  int              exponent{0};   ///< frexp exponent of abs(original denominator).
  GoldschmidtState initial{};     ///< Actual mantissa/ldexp normalization, before iteration.
  GoldschmidtState final_state{}; ///< Actual state after all completed updates.
  int updates{0}; ///< Completed multiplicative updates, possibly zero for an explicit zero budget.
  std::size_t loop_multiplications{0}; ///< Two actual update products per completed step; excludes
                                       ///< scaling, final sign, tests and all other work.
  GoldschmidtStop stop{
      GoldschmidtStop::iteration_limit}; ///< Actual stop; the default is an unevaluated record.
};

/// @brief One complete actual before/factor/after update captured during execution.
/// All values own primitive storage. The same rounded factor feeds both products.
/// @par Example
/// @code
/// const dm::algorithms::GoldschmidtStep s{};
/// const int i = s.update; // 0
/// @endcode
/// @see GoldschmidtTrace
struct GoldschmidtStep {
  int              update{0}; ///< One-based completed-update number.
  GoldschmidtState before{};  ///< Actual state read by this step.
  double           factor{0}; ///< Actual rounded 2-before.denominator.
  GoldschmidtState after{};   ///< Actual results of numerator*=factor and denominator*=factor.
};

/// @brief Owned run plus one whole record per actual update, without replay.
/// Initial normalization lives in run.initial and is not a history record.
/// Copying duplicates the vector. Ordinary state/output, allocation overhead,
/// inputs and the initial/final scalar records are excluded from the quota.
/// O(t) retained history; quota units are records, not bytes or peak RAM.
/// @par Example
/// @code
/// const auto t = dm::algorithms::goldschmidt_trace(3, 4, 2);
/// const auto n = t.steps.size(); // 2
/// @endcode
/// @see goldschmidt_trace()
struct GoldschmidtTrace {
  GoldschmidtRun run{}; ///< Actual final output, states, counts and stop.
  std::vector<GoldschmidtStep>
      steps; ///< Owned chronological complete updates, with no partial return on failure.
};

/// @brief Approximate finite-input n/d by the preserved Goldschmidt loop.
/// Uses frexp(abs(d)) and ldexp(n,-exponent), then at most 60 updates with
/// f=2-D, N*=f, D*=f, testing abs(D-1)<1e-16 AFTER each update. Returns the
/// actual final signed numerator, not a substituted hardware division. Pure
/// apart from ordinary floating-point flags/errno; uses the caller's rounding
/// environment. The products have independent data dependencies after f exists;
/// this implementation launches no threads and promises no hardware speedup.
/// O(t) selected scalar work, O(1) auxiliary state. No correctly-rounded or
/// all-magnitude accuracy guarantee: scaling can lose a subnormal numerator,
/// overflow can yield infinity, and a denominator can stall until the cap.
/// @param numerator Finite signed dividend, including signed zero.
/// @param denominator Finite nonzero signed divisor; either signed zero is rejected.
/// @return Actual signed numerator after the loop; signed zero or infinity is possible.
/// @throws std::invalid_argument If either operand is nonfinite or denominator is zero, before
/// normalization.
/// @par Example
/// @code
/// std::print("{:.6f}\n", dm::algorithms::goldschmidt_division(22, 7)); // 3.142857
/// @endcode
/// @par References
/// - R. E. Goldschmidt (1964), Applications of Division by Convergence, MIT thesis:
///   https://hdl.handle.net/1721.1/11113
/// @see goldschmidt_run(), goldschmidt_trace()
double goldschmidt_division(double numerator, double denominator);

/// @brief Run the same actual arithmetic with an explicit update budget and work metadata.
/// Same finite-input semantics and limitations as goldschmidt_division(). A zero
/// budget still normalizes and applies the denominator sign, but makes no updates.
/// No history allocation or input mutation. Maximum selected products are 2*max_updates.
/// @param numerator Finite signed dividend.
/// @param denominator Finite nonzero divisor.
/// @param max_updates Nonnegative update budget, default 60; zero is allowed.
/// @return Owned actual result, scaling exponent, initial/final states and loop-product count.
/// @throws std::invalid_argument For invalid operands or a negative update budget.
/// @par Example
/// @code
/// const auto r = dm::algorithms::goldschmidt_run(3, 4, 0);
/// std::print("{} {}\n", r.quotient, r.updates); // 0.375 0
/// @endcode
/// @see goldschmidt_division(), GoldschmidtRun
GoldschmidtRun goldschmidt_run(double numerator, double denominator, int max_updates = 60);

/// @brief Record the shared native loop, admitting each whole update before its arithmetic.
/// Same arithmetic and budget rules as goldschmidt_run(); no replay, truncation,
/// partial result or retained input references. Initial normalization is not a
/// history record and occurs before online update admission. Ordinary FP flags
/// from work already performed are not rolled back if a later admission fails.
/// @param numerator Finite signed dividend.
/// @param denominator Finite nonzero signed divisor.
/// @param max_updates Nonnegative update budget, default 60.
/// @param max_records Whole-update quota, default 1000000; zero disables the policy cap.
/// @return Owned run plus exactly run.updates complete records.
/// @throws std::invalid_argument For invalid inputs/budget, before normalization or quota checks.
/// @throws std::length_error If a next complete record exceeds quota or vector capacity.
/// @throws std::bad_alloc If record allocation fails.
/// @par Example
/// @code
/// const auto t = dm::algorithms::goldschmidt_trace(3, 4, 2, 2);
/// std::print("{} {}\n", t.steps.size(), t.run.loop_multiplications); // 2 4
/// @endcode
/// @see GoldschmidtTrace, goldschmidt_run()
GoldschmidtTrace goldschmidt_trace(
    double      numerator,
    double      denominator,
    int         max_updates = 60,
    std::size_t max_records = 1000000);
} // namespace datamunge::algorithms
