/// @file exponential_backoff.cpp
/// @brief Finite constant-operation capped windows and shared explicit/seeded actual idle-wait emission.
#include <datamunge/algorithms/exponential_backoff.hpp>
#include <cmath>
#include <limits>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Validated scalar geometry and shared actual window/schedule kernels.
namespace backoff_detail {
/// @brief Validated finite window parameters and their base-two decomposition.
struct Plan {
    double base=0; ///< Nonnegative finite initial window.
    double cap=0; ///< Nonnegative finite cap, possibly below base.
    double base_fraction=0; ///< frexp mantissa of the initial window.
    double cap_fraction=0; ///< frexp mantissa of the cap.
    long long exponent_gap=0; ///< Widened cap exponent minus base exponent.
};
/// @brief Validate the finite nonnegative base/cap pair and prepare safe comparisons.
/// @param base Requested initial window, with signed zero allowed.
/// @param cap Requested maximum window, with signed zero allowed.
/// @return Scalar decompositions; no dynamic storage or doubling is needed.
/// @throws std::invalid_argument On nonfinite or negative parameters.
Plan plan(double base,double cap){
    if(!std::isfinite(base)||!std::isfinite(cap)||base<0||cap<0)throw std::invalid_argument("backoff: finite nonnegative base and cap required");
    Plan p;p.base=base;p.cap=cap;int base_exp=0,cap_exp=0;p.base_fraction=std::frexp(base,&base_exp);p.cap_fraction=std::frexp(cap,&cap_exp);p.exponent_gap=static_cast<long long>(cap_exp)-base_exp;return p;
}
/// @brief Validate a zero-based attempt or output count without allocation.
/// @param value Requested nonnegative index/count.
/// @throws std::invalid_argument On a negative value.
void nonnegative(int value){if(value<0)throw std::invalid_argument("backoff: nonnegative attempt/count required");}
/// @brief Validate an explicit inclusive-endpoint unit variate.
/// @param u Caller supplied value.
/// @throws std::invalid_argument On nonfinite data or values outside [0,1].
void unit(double u){if(!std::isfinite(u)||u<0||u>1)throw std::invalid_argument("backoff: finite u in [0,1] required");}
/// @brief Compute one actual checked capped window with a fixed number of math calls.
/// @param k Validated nonnegative attempt.
/// @param p Validated scalar plan.
/// @return Actual window and branch flags; u/delay retain their neutral defaults.
BackoffStep window(int k,const Plan& p){
    BackoffStep out;out.attempt=k;
    if(p.base>=p.cap){out.window=p.cap;out.capped=true;}
    else if(p.base==0||k==0)out.window=p.base;
    else if(static_cast<long long>(k)>p.exponent_gap||(static_cast<long long>(k)==p.exponent_gap&&p.base_fraction>=p.cap_fraction)){out.window=p.cap;out.capped=true;}
    else{out.window=std::ldexp(p.base,k);out.scaled=true;}
    return out;
}
/// @brief Apply the actual validated variate to the actual finite window.
/// @param k Validated nonnegative attempt.
/// @param p Validated scalar plan.
/// @param u Validated inclusive [0,1] variate.
/// @return Actual window/variate/product and scalar branch flags.
BackoffStep step(int k,const Plan& p,double u){auto out=window(k,p);out.u=u;out.delay=u*out.window;return out;}
/// @brief Check both intrinsic schedule containers before any reservation or draws.
/// @param n Validated nonnegative step count.
/// @throws std::length_error On record or prefix container geometry limits.
void schedule_geometry(int n){const auto count=static_cast<std::size_t>(n);if(count>std::vector<BackoffStep>{}.max_size()||count>std::vector<double>{}.max_size())throw std::length_error("backoff: schedule container limits");}
/// @brief Advance the original nonzero-state xorshift64 stream by one actual draw.
/// @param state Owned local nonzero state, mutated modulo2^64.
/// @return Actual high-53-bit unit value in [0,1).
double draw(std::uint64_t& state){state^=state<<13;state^=state>>7;state^=state<<17;return (state>>11)*0x1p-53;}
/// @brief Emit actual records and cumulative idle waits, without explanatory replay.
/// @param n Validated step count fitting both containers.
/// @param p Validated scalar window plan.
/// @param uniforms Optional validated nonempty N-element explicit sequence.
/// @param state Optional owned local nonzero RNG state; mutually exclusive with uniforms.
/// @return Owned actual records, prefixes, final total and sampling-mode flag.
/// @throws std::overflow_error If a wait exceeds the remaining finite prefix capacity.
/// @throws std::bad_alloc On output reservation.
BackoffSchedule emit(int n,const Plan& p,const std::vector<double>* uniforms,std::uint64_t* state){
    BackoffSchedule out;out.jittered=uniforms||state;out.steps.reserve(n);out.retry_times.reserve(n);
    for(int k=0;k<n;++k){const auto u=state?draw(*state):uniforms?(*uniforms)[static_cast<std::size_t>(k)]:1.;auto actual=step(k,p,u);
        if(actual.delay>std::numeric_limits<double>::max()-out.total_delay)throw std::overflow_error("backoff: cumulative idle wait exceeds finite capacity");
        out.total_delay+=actual.delay;out.steps.push_back(actual);out.retry_times.push_back(out.total_delay);}
    return out;
}
} // namespace backoff_detail
BackoffStep exponential_backoff_step(int k,double base,double cap,double u){backoff_detail::nonnegative(k);const auto p=backoff_detail::plan(base,cap);backoff_detail::unit(u);return backoff_detail::step(k,p,u);}
std::vector<double> exponential_backoff_windows(int n,double base,double cap){
    backoff_detail::nonnegative(n);const auto p=backoff_detail::plan(base,cap);std::vector<double> out;if(static_cast<std::size_t>(n)>out.max_size())throw std::length_error("backoff: window container limit");out.reserve(n);
    for(int k=0;k<n;++k)out.push_back(backoff_detail::window(k,p).window);return out;
}
BackoffSchedule exponential_backoff_schedule(int n,double base,double cap,const std::vector<double>& uniforms){
    backoff_detail::nonnegative(n);const auto p=backoff_detail::plan(base,cap);
    if(!uniforms.empty()&&uniforms.size()!=static_cast<std::size_t>(n))throw std::invalid_argument("backoff: explicit uniforms must match N");
    for(auto u:uniforms)backoff_detail::unit(u);backoff_detail::schedule_geometry(n);return backoff_detail::emit(n,p,uniforms.empty()?nullptr:&uniforms,nullptr);
}
BackoffSchedule exponential_backoff_seeded(int n,double base,double cap,std::uint64_t seed){
    backoff_detail::nonnegative(n);const auto p=backoff_detail::plan(base,cap);if(!seed)throw std::invalid_argument("backoff: nonzero xorshift seed required");
    backoff_detail::schedule_geometry(n);return backoff_detail::emit(n,p,nullptr,&seed);
}
} // namespace datamunge::algorithms
