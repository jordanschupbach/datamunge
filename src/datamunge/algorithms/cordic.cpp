/// @file cordic.cpp
/// @brief One floating circular CORDIC kernel, actual state recording and bounded diagnostics.
#include <datamunge/algorithms/cordic.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Immutable stage data and checked shared recurrence helpers.
namespace {
constexpr double kCordicPi=3.14159265358979323846; ///< Original rounded double pi constant.
using Constants=std::array<CordicConstant,cordic_rotation_limit>; ///< All nonzero binary64 stages.
static_assert(std::numeric_limits<double>::is_iec559 && std::numeric_limits<double>::digits==53 &&
    std::numeric_limits<double>::min_exponent==-1021,"CORDIC stage limit requires binary64");
/// @brief Validate the finite angle independently of work/record limits.
/// @param theta Angle to inspect.
/// @throws std::invalid_argument Nonfinite input.
void validate_angle(double theta) {
    if (!std::isfinite(theta)) throw std::invalid_argument("CORDIC: angle must be finite");
}
/// @brief Validate the complete nonzero-scale iteration domain.
/// @param iterations Stage count.
/// @throws std::invalid_argument Count outside [0,1075].
void validate_count(int iterations) {
    if (iterations<0 || iterations>cordic_rotation_limit) throw std::invalid_argument("CORDIC: iterations must be in [0,1075]");
}
/// @brief Build constants using the original scale, atan and compensation expressions.
/// @return Owned immutable-table contents for all nonzero binary64 scales.
Constants make_constants() {
    Constants out{};double scale=1;
    for (int i=0;i<cordic_rotation_limit;++i) {
        out[static_cast<std::size_t>(i)]={i,scale,std::atan(scale),1.0/std::sqrt(1.0+scale*scale)};
        scale*=.5;
    }
    return out;
}
/// @brief Return the thread-safe lazy immutable stage table.
/// @return Shared read-only table; initialization performs libm work once.
const Constants& constants() {
    /// @brief Immutable constants, initialized once under the ordinary floating environment.
    static const Constants value=make_constants();
    return value;
}
/// @brief Apply the original rounded-period reduction to an already finite input.
/// @param theta Validated finite angle.
/// @return Actual remainder, folded angle and sign-restoration decision.
CordicReduction reduce_finite(double theta) {
    const double period=2*kCordicPi,principal=std::remainder(theta,period);
    double reduced=principal;bool flip=false;
    if (reduced>kCordicPi/2) {reduced-=kCordicPi;flip=true;}
    else if (reduced<-kCordicPi/2) {reduced+=kCordicPi;flip=true;}
    return {theta,period,principal,reduced,flip};
}
/// @brief Execute the single producing recurrence, optionally retaining actual steps.
/// @tparam Record Keep before/after records and extra unit-direction diagnostics.
/// @param theta Original angle, fully validated here.
/// @param iterations Completed-stage request, fully validated here.
/// @param max_steps Optional positive complete trace cap; ignored when not recording.
/// @return Owned actual result/work and optional producing records.
/// @throws std::invalid_argument Nonfinite angle or invalid stage count.
/// @throws std::length_error Positive complete trace cap exceeded.
template<bool Record>
CordicTrace run(double theta,int iterations,std::size_t max_steps) {
    validate_angle(theta);validate_count(iterations);
    if constexpr (Record) {
        if (max_steps && static_cast<std::size_t>(iterations)>max_steps) throw std::length_error("CORDIC: complete trace cap");
    }
    CordicTrace out;auto& a=out.analysis;
    a.reduction=reduce_finite(theta);a.residual=a.reduction.reduced;a.iterations=iterations;a.work.range_reductions=1;
    if constexpr (Record) out.steps.reserve(static_cast<std::size_t>(iterations));
    const Constants* table=iterations?&constants():nullptr;
    for (int i=0;i<iterations;++i) {
        const auto& c=(*table)[static_cast<std::size_t>(i)];
        const double direction=a.residual>=0?1.0:-1.0;
        CordicStep step;
        if constexpr (Record) {
            step.constant=c;step.direction=static_cast<int>(direction);
            step.x_before=a.raw_x;step.y_before=a.raw_y;step.z_before=a.residual;step.gain_before=a.inverse_gain;
        }
        const double nx=a.raw_x-direction*a.raw_y*c.shift,ny=a.raw_y+direction*a.raw_x*c.shift;
        a.raw_x=nx;a.raw_y=ny;a.residual-=direction*c.angle;a.inverse_gain*=c.inverse_gain_step;
        ++a.work.rotations;++a.work.constant_lookups;++a.work.inverse_gain_products;
        if constexpr (Record) {
            step.x=a.raw_x;step.y=a.raw_y;step.z=a.residual;step.inverse_gain=a.inverse_gain;
            const double norm=std::hypot(a.raw_x,a.raw_y);step.unit_x=a.raw_x/norm;step.unit_y=a.raw_y/norm;
            ++a.work.normalizations;out.steps.push_back(step);
        }
    }
    a.result={a.raw_y*a.inverse_gain,a.raw_x*a.inverse_gain};a.work.final_products=2;
    if (a.reduction.flipped) {a.result.sin=-a.result.sin;a.result.cos=-a.result.cos;}
    return out;
}
} // namespace
CordicReduction cordic_reduce(double theta) {validate_angle(theta);return reduce_finite(theta);}
std::vector<CordicConstant> cordic_stage_constants(int iterations) {
    validate_count(iterations);if (!iterations) return {};
    const auto& table=constants();return {table.begin(),table.begin()+iterations};
}
CordicAnalysis cordic_sincos_analyze(double theta,int iterations) {return run<false>(theta,iterations,0).analysis;}
CordicTrace cordic_sincos_trace(double theta,int iterations,std::size_t max_steps) {return run<true>(theta,iterations,max_steps);}
SinCos cordic_sincos(double theta,int iterations) {return run<false>(theta,iterations,0).analysis.result;}
std::vector<CordicErrorRow> cordic_error_sweep(const std::vector<double>& angles,
    const std::vector<int>& counts,std::size_t max_evaluations) {
    for (double angle:angles) validate_angle(angle);
    for (int count:counts) validate_count(count);
    if (!counts.empty() && angles.size()>std::numeric_limits<std::size_t>::max()/counts.size()) throw std::length_error("CORDIC sweep: evaluation overflow");
    if (max_evaluations && angles.size()*counts.size()>max_evaluations) throw std::length_error("CORDIC sweep: complete evaluation cap");
    if (counts.empty()) return {};
    std::vector<SinCos> reference;reference.reserve(angles.size());
    for (double theta:angles) reference.push_back({std::sin(theta),std::cos(theta)});
    std::vector<CordicErrorRow> out;out.reserve(counts.size());
    for (int count:counts) {
        CordicErrorRow row;row.iterations=count;row.samples=angles.size();row.has_samples=!angles.empty();
        for (std::size_t i=0;i<angles.size();++i) {
            const auto value=cordic_sincos(angles[i],count);
            const double se=std::abs(value.sin-reference[i].sin),ce=std::abs(value.cos-reference[i].cos),score=std::max(se,ce);
            row.max_sin_error=std::max(row.max_sin_error,se);row.max_cos_error=std::max(row.max_cos_error,ce);
            row.max_norm_error=std::max(row.max_norm_error,std::abs(std::hypot(value.sin,value.cos)-1));
            if (!i || score>row.max_component_error) {row.max_component_error=score;row.worst_angle=angles[i];}
        }
        out.push_back(row);
    }
    return out;
}
} // namespace datamunge::algorithms
