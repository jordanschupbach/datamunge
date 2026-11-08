#pragma once
/// @file cordic.hpp
/// @brief Inspectable floating circular CORDIC, actual rotations and bounded accuracy sweeps.
#include <datamunge/algorithms/pi_and_cordic.hpp>
#include <cstddef>
#include <vector>
namespace datamunge::algorithms {
/// @brief Number of nonzero scales 2^-i available in IEEE binary64, i=0..1074.
/// Iteration counts are in [0,1075]. Further zero-scale iterations would not alter
/// the recurrence. This is a work/domain limit, not an accuracy promise.
/// @par Example
/// @code
/// std::cout << dm::algorithms::cordic_rotation_limit; // 1075
/// @endcode
inline constexpr int cordic_rotation_limit=1075;

/// @brief Owned actual rounded-period reduction and optional half-turn fold.
/// principal is remainder(theta,period), with period=2*pi_double. reduced lies
/// within [-pi_double/2,pi_double/2]; flipped means a half turn was removed and
/// both final coordinates must be negated. This does not perform high-precision
/// reduction modulo mathematical 2*pi: large input angles can have large phase error.
/// @par Example
/// @code
/// const auto r=dm::algorithms::cordic_reduce(2.5);
/// std::cout << r.flipped << ' ' << (r.reduced<0); // 1 1
/// @endcode
/// @see cordic_reduce()
struct CordicReduction {
    double theta{};     ///< Original finite angle in radians.
    double period{};    ///< Actual double-valued period used by remainder.
    double principal{}; ///< Actual remainder in [-period/2,period/2].
    double reduced{};   ///< Actual folded target for the rotation kernel.
    bool flipped{};     ///< Negate both final sine/cosine values when true.
};

/// @brief One actual cached circular-stage constant, copied on request.
/// shift=2^-index, angle=atan(shift), inverse_gain_step=1/sqrt(1+shift²), all
/// rounded doubles. The shared immutable table is lazily built with libm calls;
/// the subsequent recurrence performs lookups and floating multiplications.
/// @par Example
/// @code
/// const auto c=dm::algorithms::cordic_stage_constants(1);
/// std::cout << c[0].index << ' ' << c[0].shift; // 0 1
/// @endcode
/// @see cordic_stage_constants()
struct CordicConstant {
    int index{};                ///< Zero-based stage in 0..1074.
    double shift{};             ///< Nonzero binary scale for this stage.
    double angle{};             ///< Rounded atan(shift), radians.
    double inverse_gain_step{}; ///< Rounded compensation factor for this stage.
};

/// @brief Bounded actual per-call operations, excluding one-time shared table construction.
/// These are algorithmic evaluations, not machine-instruction/throughput counts.
/// Every requested stage executes even at zero residual. Trace normalization
/// calls are additional diagnostics and do not change the producing recurrence.
/// @par Example
/// @code
/// const auto a=dm::algorithms::cordic_sincos_analyze(1,3);
/// std::cout << a.work.rotations << ' ' << a.work.normalizations; // 3 0
/// @endcode
/// @see CordicAnalysis
struct CordicWork {
    std::size_t range_reductions{};      ///< One rounded-period reduction per successful call.
    std::size_t rotations{};             ///< Exactly the requested iteration count.
    std::size_t constant_lookups{};      ///< One stage-record lookup per rotation.
    std::size_t inverse_gain_products{}; ///< One accumulated compensation product per rotation.
    std::size_t final_products{};        ///< Two final x/y compensation products.
    std::size_t normalizations{};        ///< One extra hypot/division diagnostic per recorded step.
};

/// @brief Owned actual approximation and final folded-frame recurrence state.
/// result includes half-turn restoration; raw_x/raw_y and residual refer to the
/// folded frame, initially (1,0,reduction.reduced). inverse_gain is the product
/// actually accumulated for exactly iterations stages, initially one. No field
/// certifies error to the true original angle. Zero stages is a valid unrotated
/// approximation. The model uses double arithmetic, including multiplications.
/// @par Example
/// @code
/// const auto a=dm::algorithms::cordic_sincos_analyze(0,0);
/// std::cout << a.raw_x << ' ' << a.inverse_gain; // 1 1
/// @endcode
/// @see cordic_sincos_analyze(), CordicReduction
struct CordicAnalysis {
    SinCos result;              ///< Actual compensated/restored output.
    CordicReduction reduction;  ///< Actual producing reduction.
    int iterations{};           ///< Requested and completed stages.
    double raw_x{1};            ///< Actual unscaled x after the last stage.
    double raw_y{};             ///< Actual unscaled y after the last stage.
    double residual{};          ///< Actual reduced target residual, radians.
    double inverse_gain{1};     ///< Actual accumulated K_N compensation.
    CordicWork work;            ///< Actual per-call work.
};

/// @brief Owned actual circular update, including before/after state and normalized direction.
/// Constants are copied from the producing table. direction is +1 at residual
/// >=0 (including either zero), otherwise -1. Before/after coordinates are raw;
/// unit_x/unit_y divide the actual after-state by hypot and remain in the folded
/// frame, without half-turn restoration. They reproduce the former report's
/// plotted normalization; they are not a rerun or a compensated-output snapshot.
/// @par Example
/// @code
/// const auto t=dm::algorithms::cordic_sincos_trace(1,1);
/// std::cout << t.steps[0].x << ' ' << t.steps[0].y; // 1 1
/// @endcode
/// @see CordicTrace
struct CordicStep {
    CordicConstant constant; ///< Actual producing stage constant.
    int direction{};         ///< Actual residual-selected direction, -1 or +1.
    double x_before{};       ///< Raw x before update.
    double y_before{};       ///< Raw y before update.
    double z_before{};       ///< Residual before update.
    double gain_before{};    ///< Accumulated compensation before update.
    double x{};              ///< Raw x after update.
    double y{};              ///< Raw y after update.
    double z{};              ///< Residual after update.
    double inverse_gain{};   ///< Accumulated compensation after update.
    double unit_x{};         ///< After-state x/hypot(x,y), folded frame.
    double unit_y{};         ///< After-state y/hypot(x,y), folded frame.
};

/// @brief Owned actual rotation sequence and the result it produced, without replay.
/// Exactly one constant-size record per completed stage; zero stages is empty.
/// Caller inputs, constants returned elsewhere, steps and result own independent
/// values. Theta(1+N) work and O(N) extra trace storage for N<=1075.
/// @par Example
/// @code
/// std::cout << dm::algorithms::cordic_sincos_trace(1,12).steps.size(); // 12
/// @endcode
/// @see cordic_sincos_trace()
struct CordicTrace {
    CordicAnalysis analysis;       ///< Actual result, final state and work.
    std::vector<CordicStep> steps; ///< Actual before/after records, in stage order.
};

/// @brief Expose the checked rounded-period reduction used by the CORDIC kernel.
/// Uses remainder with the double constant 2*pi_double and a half-turn fold.
/// Accepts every finite angle, but reduction accuracy to mathematical 2*pi can
/// deteriorate severely for large magnitudes. No mutation or shared work state.
/// @param theta Finite original angle, radians.
/// @return Owned actual input/period/principal/folded values and restoration flag.
/// @throws std::invalid_argument Nonfinite angle.
/// @par Example
/// @code
/// std::cout << dm::algorithms::cordic_reduce(.5).reduced; // 0.5
/// @endcode
/// @par References
/// - Volder (1959), The CORDIC Trigonometric Computing Technique,
///   https://doi.org/10.1109/TEC.1959.5222693 . Rounded-period reduction is this model's policy.
/// @see cordic_sincos_analyze()
CordicReduction cordic_reduce(double theta);

/// @brief Copy the actual immutable constants used by the recurrence, in stage order.
/// The shared table is initialized thread-safely with atan/sqrt under the normal
/// floating environment and then read only. Returned storage is independent.
/// Count zero returns empty without constructing the table. O(N) copying.
/// @param iterations Stage count in [0,1075].
/// @return Owned shift/angle/compensation records for stages 0..N-1.
/// @throws std::invalid_argument Count outside [0,1075].
/// @throws std::bad_alloc Returned constant storage cannot be allocated.
/// @par Example
/// @code
/// std::cout << dm::algorithms::cordic_stage_constants(3).size(); // 3
/// @endcode
/// @par References
/// - Volder (1959), https://doi.org/10.1109/TEC.1959.5222693 .
/// @see CordicConstant, cordic_rotation_limit
std::vector<CordicConstant> cordic_stage_constants(int iterations);

/// @brief Execute the canonical floating circular recurrence and expose its actual final state.
/// Same result as cordic_sincos(). Validates finite theta and N in [0,1075]
/// before reduction/table access. Uses actual cached atan/compensation constants,
/// raw initial (1,0), N residual-sign updates, accumulated K_N and final sign fold.
/// No early stop at zero residual, no forced exact identities or error certificate.
/// Theta(1+N) work, O(1) per-call scratch after shared table construction. Large
/// angles retain the documented rounded-period limitation. No input mutation.
/// @param theta Finite original angle, radians.
/// @param iterations Completed stage count in [0,1075], default forty.
/// @return Owned output, reduction, raw final state, compensation and actual work.
/// @throws std::invalid_argument Nonfinite theta or invalid count.
/// @par Example
/// @code
/// const auto a=dm::algorithms::cordic_sincos_analyze(0,0);
/// std::cout << a.result.sin << ' ' << a.result.cos; // 0 1
/// @endcode
/// @par References
/// - Volder (1959), https://doi.org/10.1109/TEC.1959.5222693 .
/// @see cordic_sincos(), cordic_sincos_trace()
CordicAnalysis cordic_sincos_analyze(double theta,int iterations=40);

/// @brief Retain the producing CORDIC kernel's actual before/after steps without replay.
/// Same recurrence/result as analyze; normalization of each actual raw step is
/// an additional trace diagnostic. Full finite/count validation and the complete
/// one-record-per-stage cap precede allocation, reduction and table access.
/// @param theta Finite original angle, radians.
/// @param iterations Stage count in [0,1075], default forty.
/// @param max_steps Positive complete record cap, default 1075; zero disables this optional cap.
/// @return Owned actual steps and their producing analysis/result.
/// @throws std::invalid_argument Nonfinite theta or invalid stage count.
/// @throws std::length_error Positive complete record cap exceeded.
/// @throws std::bad_alloc Actual step storage cannot be allocated.
/// @par Example
/// @code
/// const auto t=dm::algorithms::cordic_sincos_trace(1,3,3);
/// std::cout << t.analysis.work.normalizations; // 3
/// @endcode
/// @par References
/// - Volder (1959), https://doi.org/10.1109/TEC.1959.5222693 .
/// @see cordic_sincos_analyze(), CordicStep
CordicTrace cordic_sincos_trace(double theta,int iterations=40,std::size_t max_steps=1075);

/// @brief Measured worst errors for one requested iteration count against the host libm.
/// Empty angle inputs set has_samples=false and leave zero placeholders for
/// errors/worst_angle. Positive samples select the first input with the largest
/// max(abs(sine error),abs(cosine error)); norms measure abs(hypot(result)-1).
/// Values are diagnostics against std::sin/cos, not certified mathematical bounds.
/// @par Example
/// @code
/// const auto e=dm::algorithms::cordic_error_sweep({0},{0});
/// std::cout << e[0].samples << ' ' << e[0].max_component_error; // 1 0
/// @endcode
/// @see cordic_error_sweep()
struct CordicErrorRow {
    int iterations{};             ///< Requested stage count.
    std::size_t samples{};         ///< Number of original angles measured.
    bool has_samples{};           ///< Whether extrema/witness are present.
    double max_sin_error{};        ///< Maximum absolute sine difference to libm.
    double max_cos_error{};        ///< Maximum absolute cosine difference to libm.
    double max_component_error{};  ///< Largest of the two component errors.
    double max_norm_error{};       ///< Maximum absolute unit-norm defect.
    double worst_angle{};          ///< First input attaining max_component_error; absent when empty.
};

/// @brief Measure native CORDIC on supplied angles/counts with a complete evaluation budget.
/// Full angle/count validation precedes checked angles.size()*counts.size()
/// preflight and allocation. Repeated counts retain order; empty counts return
/// empty and empty angles return absent rows. References use host std::sin/cos
/// of the original angles, exposing both iteration and range-reduction error.
/// O(S+C+S*sum(N+1)) time and O(S+C) storage for S angles/C counts, after the one-time
/// table setup. No global RNG or input mutation.
/// @param angles Finite original angles, radians, in witness tie order.
/// @param counts Stage counts, each in [0,1075].
/// @param max_evaluations Complete call cap, default 100000; zero unlimited.
/// @return Owned error rows in count order with explicit empty-input state.
/// @throws std::invalid_argument Nonfinite angle or invalid stage count, including in otherwise empty sweeps.
/// @throws std::length_error Evaluation product overflow, positive complete cap or container capacity exceeded.
/// @throws std::bad_alloc Reference/result storage cannot be allocated.
/// @par Example
/// @code
/// const auto e=dm::algorithms::cordic_error_sweep({0,.5,1},{12,44});
/// std::cout << e.size() << ' ' << e[0].samples; // 2 3
/// @endcode
/// @see CordicErrorRow, cordic_sincos_analyze()
std::vector<CordicErrorRow> cordic_error_sweep(const std::vector<double>& angles,
    const std::vector<int>& counts,std::size_t max_evaluations=100000);
} // namespace datamunge::algorithms
