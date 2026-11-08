#pragma once
/// @file borwein_pi.hpp
/// @brief Shared double-precision Borwein iteration, actual diagnostics and an exact-arithmetic bound model.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace datamunge::algorithms {
/// @brief Algebraically equivalent ways to evaluate Borwein's next y in floating point.
/// Both modes retain the historical seed 6-4*sqrt(2). Rationalisation preserves
/// small y updates when the computed fourth root rounds to one; it does not
/// guarantee a more accurate final pi estimate or undo earlier rounding.
/// @par Example
/// @code
/// const auto mode=dm::algorithms::BorweinEvaluation::rationalised;
/// std::print("{}\n",mode==dm::algorithms::BorweinEvaluation::rationalised); // true
/// @endcode
/// @see borwein_pi_trace()
enum class BorweinEvaluation {
    direct, ///< Historical (1-r)/(1+r), where r=(1-y^4)^(1/4).
    rationalised ///< y^4/((1+r)*(1+r+r^2+r^3)), avoiding subtraction of nearly equal ones.
};

/// @brief One actual completed quartic update, before any early-stop decision.
/// Iteration is one-based. All values are owned doubles from the canonical loop;
/// unchanged refers to the stored inverse-pi value, not exact convergence to pi.
/// @par Example
/// @code
/// const auto step=dm::algorithms::borwein_pi_trace(1).steps.front();
/// std::print("{} {:.6f}\n",step.iteration,step.estimate); // 1 3.141593
/// @endcode
/// @see BorweinTrace
struct BorweinStep {
    int iteration{0}; ///< One-based number of this actual update.
    double previous_y{0}; ///< y before this update.
    double y_fourth{0}; ///< Computed previous_y^4, before subtracting from one.
    double fourth_root{0}; ///< Computed (1-y_fourth)^(1/4).
    double y{0}; ///< Updated y, using the selected evaluation mode.
    double inverse_pi{0}; ///< Updated approximation a to 1/pi.
    double estimate{0}; ///< Rounded reciprocal 1/inverse_pi.
    double scale{0}; ///< Actual 2^(2k+3), with k=iteration-1.
    double correction{0}; ///< Actual scale*y*(1+y+y^2) subtracted from the product.
    bool unchanged{false}; ///< Updated a equals its previous stored double; the kernel stops here.
    bool root_rounded_to_one{false}; ///< Computed fourth_root is exactly one despite positive y_fourth.
};

/// @brief Owned actual Borwein states and stopping information, without fabricated plateau records.
/// The seed is separate from the update vector. Negative requested iteration
/// counts are clamped to zero, matching the historical scalar/sequence API.
/// A recorded unchanged update is included, then the kernel stops. Scalar calls
/// use the same kernel without allocating records; sequences pad repeated final
/// estimates to retain their historical requested length.
/// @par Example
/// @code
/// const auto trace=dm::algorithms::borwein_pi_trace(4);
/// std::print("{} {} {}\n",trace.requested_iterations,trace.iterations_performed,trace.stagnated); // 4 3 true
/// @endcode
/// @see borwein_pi_trace(), BorweinStep
struct BorweinTrace {
    BorweinEvaluation evaluation{BorweinEvaluation::direct}; ///< Actual y-expression used.
    int requested_iterations{0}; ///< Effective maximum updates, after clamping negative input to zero.
    int iterations_performed{0}; ///< Actual completed updates, including an unchanged final update.
    double initial_y{0}; ///< Computed sqrt(2)-1.
    double initial_inverse_pi{0}; ///< Historical computed seed 6-4*sqrt(2).
    double initial_estimate{0}; ///< Rounded reciprocal of the initial seed.
    double estimate{0}; ///< Final rounded pi estimate from this invocation.
    bool stagnated{false}; ///< An update left stored a unchanged; this is not a true-error certificate.
    std::vector<BorweinStep> steps; ///< Actual updates only; empty for zero effective iterations.
};

/// @brief Private shared iteration and owned-sequence adapter.
namespace borwein_detail {
/// @brief Execute actual double updates with optional recording/sequence capture.
/// @tparam Record Whether to append owned actual diagnostic records.
/// @param iterations Maximum updates, lower-clamped to zero.
/// @param evaluation Direct or rationalised next-y expression.
/// @param estimates Optional owned sequence to populate and pad, otherwise nullptr.
/// @return Seed, final estimate, actual update count and optional actual records.
/// @throws std::invalid_argument For an unknown evaluation enum value.
/// @throws std::length_error If a requested sequence exceeds vector capacity.
/// @throws std::overflow_error If a scale or positive finite state becomes unrepresentable.
/// Allocation failures propagate; no shared state is changed.
template<bool Record>
inline BorweinTrace iterate(int iterations,BorweinEvaluation evaluation,std::vector<double>* estimates=nullptr) {
    if (evaluation!=BorweinEvaluation::direct && evaluation!=BorweinEvaluation::rationalised)
        throw std::invalid_argument("unknown Borwein evaluation mode");
    BorweinTrace trace;
    trace.evaluation=evaluation;
    trace.requested_iterations=std::max(0,iterations);
    const double square_root=std::sqrt(2.0);
    double y=square_root-1;
    double a=6-4*square_root;
    trace.initial_y=y;
    trace.initial_inverse_pi=a;
    trace.initial_estimate=trace.estimate=1/a;
    if (estimates) {
        const auto count=static_cast<std::uint64_t>(trace.requested_iterations)+1;
        if (count>estimates->max_size()) throw std::length_error("Borwein sequence exceeds vector capacity");
        estimates->reserve(static_cast<std::size_t>(count));
        estimates->push_back(trace.initial_estimate);
    }
    for (int k=0;k<trace.requested_iterations;++k) {
        const auto y_fourth=y*y*y*y;
        const auto root=std::pow(1-y_fourth,.25);
        const auto next_y=evaluation==BorweinEvaluation::direct ? (1-root)/(1+root)
            : y_fourth/((1+root)*(1+root+root*root+root*root*root));
        if (k>(std::numeric_limits<int>::max()-3)/2)
            throw std::overflow_error("Borwein scale exponent exceeds int");
        const auto scale=std::ldexp(1.0,2*k+3);
        if (!std::isfinite(scale)) throw std::overflow_error("Borwein scale exceeds double");
        const auto p1=(1+next_y)*(1+next_y);
        const auto p4=p1*p1;
        const auto correction=scale*next_y*(1+next_y+next_y*next_y);
        const auto next_a=a*p4-correction;
        if (!std::isfinite(next_a) || next_a<=0 || !std::isfinite(next_y))
            throw std::overflow_error("Borwein state is not positive finite");
        trace.iterations_performed=k+1;
        trace.estimate=1/next_a;
        trace.stagnated=next_a==a;
        if constexpr (Record)
            trace.steps.push_back({k+1,y,y_fourth,root,next_y,next_a,trace.estimate,scale,
                                   correction,trace.stagnated,root==1 && y_fourth>0});
        if (estimates) estimates->push_back(trace.estimate);
        y=next_y;
        a=next_a;
        if (trace.stagnated) break;
    }
    if (estimates)
        estimates->resize(static_cast<std::size_t>(trace.requested_iterations)+1,trace.estimate);
    return trace;
}
} // namespace borwein_detail

/// @brief Historical one-argument double pi estimator, using direct evaluation.
/// Retains the original seed, lower clamp, short-run arithmetic order and
/// unambiguous one-argument function signature. It now stops at the first
/// unchanged stored inverse-pi value, retaining the plateau for large iteration
/// requests rather than producing inf*0 and NaN. For K actual updates, O(K)
/// fixed-precision operations and O(1) auxiliary storage, with no trace allocation
/// or input/shared-state mutation. The result is not guaranteed correctly rounded
/// and does not certify digits. Rationalised evaluation is available in the trace.
/// @param iterations Maximum updates; zero/negative values return the reciprocal seed.
/// @return Rounded double reciprocal of the last actual stored inverse-pi value.
/// @throws std::overflow_error For an unrepresentable scale or invalid finite state.
/// @par Example
/// @code
/// std::print("{:.6f} {:.6f}\n",dm::algorithms::borwein_pi(0),dm::algorithms::borwein_pi(4)); // 2.914214 3.141593
/// @endcode
/// @par References
/// - J. M. Borwein and P. B. Borwein, Pi and the AGM, Wiley (1987), Iteration 5.3.
/// - R. P. Brent, The Borwein Brothers, Pi and the AGM, Algorithm BB4,
///   https://arxiv.org/abs/1802.07558 (2018, revised 2021).
/// @see borwein_pi_trace(), borwein_pi_sequence()
inline double borwein_pi(int iterations) {
    return borwein_detail::iterate<false>(iterations,BorweinEvaluation::direct).estimate;
}

/// @brief Historical owned seed and successive double pi estimates, using direct evaluation.
/// Returns max(0,iterations)+1 values. After the first unchanged stored a, later
/// entries copy the plateau without more arithmetic updates. The shared kernel
/// ensures scalar/sequence equality at every requested prefix. Preserves the
/// unambiguous original one-argument signature. For N effective requested updates,
/// O(N+1) time/output storage even after stagnation, and O(1) auxiliary state.
/// No input/shared-state mutation. Allocation failures propagate; no partial
/// vector is returned. The output is a sequence of doubles, not certified digits.
/// @param iterations Requested updates; zero/negative values produce only the seed.
/// @return Owned estimates indexed by requested update number, including the seed at zero.
/// @throws std::length_error If the complete output exceeds vector capacity.
/// @throws std::overflow_error For an unrepresentable scale or invalid finite state.
/// @par Example
/// @code
/// const auto values=dm::algorithms::borwein_pi_sequence(4);
/// std::print("{} {:.6f}\n",values.size(),values.back()); // 5 3.141593
/// @endcode
/// @par References
/// - J. M. Borwein and P. B. Borwein (1987), Pi and the AGM, Iteration 5.3.
/// - R. P. Brent, https://arxiv.org/abs/1802.07558, Algorithm BB4.
/// @see borwein_pi(), borwein_pi_trace()
inline std::vector<double> borwein_pi_sequence(int iterations) {
    std::vector<double> values;
    borwein_detail::iterate<false>(iterations,BorweinEvaluation::direct,&values);
    return values;
}

/// @brief Record actual double Borwein updates, ending at a stored-value plateau.
/// Uses the scalar/sequence kernel, with the seed separate and no fabricated
/// repeated records. For K actual updates, O(K) time and owned record storage.
/// An unchanged stored a is finite-precision stagnation, not proof of exact
/// pi or a theorem-derived error certificate. No input/shared-state mutation.
/// @param iterations Maximum updates, lower-clamped to zero.
/// @param evaluation Direct historical expression by default, or rationalised next-y.
/// @return Owned seed, actual states, final estimate and stopping information.
/// @throws std::invalid_argument For an unknown evaluation enum value.
/// @throws std::overflow_error For an unrepresentable scale or invalid finite state.
/// Allocation errors propagate, without a partial returned trace.
/// @par Example
/// @code
/// const auto trace=dm::algorithms::borwein_pi_trace(512);
/// std::print("{} {}\n",trace.iterations_performed,trace.stagnated); // 3 true
/// @endcode
/// @par References
/// - J. M. Borwein and P. B. Borwein (1987), Pi and the AGM, Iteration 5.3.
/// - R. P. Brent, https://arxiv.org/abs/1802.07558, Algorithm BB4.
/// @see BorweinTrace, borwein_pi()
inline BorweinTrace borwein_pi_trace(int iterations,BorweinEvaluation evaluation=BorweinEvaluation::direct) {
    return borwein_detail::iterate<true>(iterations,evaluation);
}

/// @brief Approximate log10 of Brent's exact-arithmetic Borwein pi-error upper bound.
/// Evaluates log10(pi^2*4^(n+2)*exp(-2*pi*4^n)) directly in the log domain,
/// where n=max(0,iterations). This models exact arithmetic; it is not a bound
/// on rounded scalar/trace results and is not interval/certified evaluation.
/// Avoids exponentiating a tiny bound. Returns negative infinity when the
/// negative logarithm exceeds double's range. O(1) work/storage, no allocation
/// or input/shared-state mutation. Uses the standard-library rounded pi reference.
/// @param iterations Exact-arithmetic update index, lower-clamped to zero.
/// @return Approximate decimal logarithm of the positive theoretical upper bound.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::borwein_pi_log10_error_bound(2)<-40); // true
/// @endcode
/// @par References
/// - R. P. Brent, The Borwein Brothers, Pi and the AGM, equation (29) and
///   Corollary 2, https://arxiv.org/abs/1802.07558 (2018, revised 2021).
/// @see borwein_pi_trace()
inline double borwein_pi_log10_error_bound(int iterations) noexcept {
    const int n=std::max(0,iterations);
    if (n>(std::numeric_limits<double>::max_exponent-2)/2)
        return -std::numeric_limits<double>::infinity();
    const double pi=std::acos(-1.0);
    return 2*std::log10(pi)+(static_cast<double>(n)+2)*std::log10(4.)
        -std::ldexp(2*pi/std::log(10.),2*n);
}
} // namespace datamunge::algorithms
