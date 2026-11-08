/// @file euler_method.cpp
/// @brief One validated forward-Euler kernel with intrinsic trajectory and optional actual derivative capture.
#include <datamunge/algorithms/euler_method.hpp>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
namespace datamunge::algorithms {
/// @brief Checked history geometry, finite time preparation and actual callback execution.
namespace euler_detail {
/// @brief Add history counts without wrapping.
/// @param a Existing count.
/// @param b Additional count.
/// @return Checked sum.
/// @throws std::length_error On size_t overflow.
std::size_t add(std::size_t a,std::size_t b){if(b>std::numeric_limits<std::size_t>::max()-a)throw std::length_error("Euler count overflow");return a+b;}
/// @brief Multiply history counts without wrapping.
/// @param a First factor.
/// @param b Second factor.
/// @return Checked product.
/// @throws std::length_error On size_t overflow.
std::size_t multiply(std::size_t a,std::size_t b){if(a&&b>std::numeric_limits<std::size_t>::max()/a)throw std::length_error("Euler history geometry overflow");return a*b;}
/// @brief Complete intrinsic history/count geometry, excluding optional derivatives.
struct Geometry {
    std::size_t dimension; ///< Positive state dimension.
    std::size_t steps; ///< Positive committed step count.
    std::size_t samples; ///< steps+1 intrinsic samples.
    std::size_t slot; ///< One logical sample/step slot plus dimension entries.
    std::size_t updates; ///< steps*dimension scalar updates.
};
/// @brief Validate every intrinsic count and required container limit before copying.
/// @param dimension State dimension.
/// @param steps Requested signed step count.
/// @return Complete checked geometry.
/// @throws std::invalid_argument On empty dimension or nonpositive steps.
/// @throws std::length_error On arithmetic/container geometry limits.
Geometry geometry(std::size_t dimension,int steps){
    if(!dimension||steps<1)throw std::invalid_argument("Euler needs nonempty state and positive steps");
    const auto n=static_cast<std::size_t>(steps),samples=add(n,1),slot=add(dimension,1);
    const auto cells=multiply(samples,dimension),updates=multiply(n,dimension);
    if(dimension>std::vector<double>{}.max_size()||samples>std::vector<double>{}.max_size()||samples>std::vector<std::vector<double>>{}.max_size()||cells>std::vector<double>{}.max_size())
        throw std::length_error("Euler intrinsic history exceeds container geometry");
    (void)multiply(samples,slot);return {dimension,n,samples,slot,updates};
}
/// @brief Check and count the complete optional trace before callbacks or copies.
/// @param size Validated intrinsic geometry.
/// @return Full logical sample and derivative capture count.
/// @throws std::length_error On count or derivative container geometry limits.
std::size_t trace_size(const Geometry& size){
    if(size.steps>std::vector<std::vector<double>>{}.max_size())throw std::length_error("Euler derivative container exceeded");
    return multiply(add(size.samples,size.steps),size.slot);
}
/// @brief Validate callbacks, endpoints and initial state without invoking user code.
/// @param f Borrowed callback.
/// @param y0 Borrowed initial state.
/// @param t0 Initial time.
/// @param t1 Final time.
/// @param steps Requested step count.
/// @return Validated intrinsic geometry.
/// @throws std::invalid_argument On empty callback, nonfinite input or invalid dimension/steps.
/// @throws std::length_error On intrinsic count/container limits.
Geometry validate(const OdeSystem& f,const std::vector<double>& y0,double t0,double t1,int steps){
    if(!f||!std::isfinite(t0)||!std::isfinite(t1))throw std::invalid_argument("Euler requires a callback and finite endpoints");
    const auto size=geometry(y0.size(),steps);
    for(double x:y0)if(!std::isfinite(x))throw std::invalid_argument("Euler initial state must be finite");
    return size;
}
/// @brief Convert an extended finite time quantity to finite double.
/// @param x Extended time or nominal step.
/// @return Finite converted double.
/// @throws std::overflow_error On nonfinite/out-of-double-range arithmetic.
double finite_time(long double x){if(!std::isfinite(x)||std::abs(x)>std::numeric_limits<double>::max())throw std::overflow_error("Euler time arithmetic overflow");return static_cast<double>(x);}
/// @brief Prepare the entire intrinsic grid before any actual callback.
/// @param t0 Finite initial time.
/// @param t1 Finite final time, either direction or equal.
/// @param size Validated geometry.
/// @param h Output nominal finite step size.
/// @return Owned complete finite grid with exact requested endpoints.
/// @throws std::overflow_error On nonfinite step/grid arithmetic.
/// @throws std::underflow_error If a nonzero interval's h becomes zero.
/// @throws std::runtime_error If a nonzero interval's grid cannot advance strictly.
/// @throws std::bad_alloc If time storage is unavailable.
std::vector<double> grid(double t0,double t1,const Geometry& size,double& h){
    const long double interval=static_cast<long double>(t1)-t0;h=finite_time(interval/size.steps);
    if(t0!=t1&&h==0)throw std::underflow_error("Euler nonzero time step underflowed");
    std::vector<double> times(size.samples);times[0]=t0;
    for(std::size_t i=1;i<size.samples;++i){const double next=i==size.steps?t1:finite_time(static_cast<long double>(t0)+static_cast<long double>(i)*h);
        if((t1>t0&&next<=times[i-1])||(t1<t0&&next>=times[i-1]))throw std::runtime_error("Euler rounded time grid cannot advance");
        times[i]=next;}
    return times;
}
/// @brief Execute each actual Euler callback once, optionally retaining its derivative.
/// @param f Validated borrowed derivative callback.
/// @param y0 Validated borrowed initial state.
/// @param t0 Initial endpoint.
/// @param t1 Final endpoint.
/// @param size Complete intrinsic geometry.
/// @param trace Optional trace with full capture geometry/quota already validated.
/// @return Owned intrinsic trajectory, nominal h and actual work.
/// @throws std::invalid_argument On derivative shape or nonfinite derivative.
/// @throws std::overflow_error On step/grid/product/state arithmetic overflow.
/// @throws std::underflow_error On a vanished nonzero time step.
/// @throws std::runtime_error On rounded grid stagnation; caller callback exceptions propagate.
/// @throws std::bad_alloc If required history/state storage cannot be allocated.
EulerOutcome core(const OdeSystem& f,const std::vector<double>& y0,double t0,double t1,const Geometry& size,EulerTrace* trace){
    EulerOutcome out;out.solution.t=grid(t0,t1,size,out.step_size);out.solution.y.reserve(size.samples);
    if(trace)trace->derivatives.reserve(size.steps);out.solution.y.push_back(y0);
    for(std::size_t n=0;n<size.steps;++n){
        ++out.work.rhs_calls;auto derivative=f(out.solution.t[n],out.solution.y.back());
        if(derivative.size()!=size.dimension)throw std::invalid_argument("Euler derivative dimension mismatch");
        for(double x:derivative)if(!std::isfinite(x))throw std::invalid_argument("Euler derivative must be finite");
        std::vector<double> next(size.dimension);
        for(std::size_t i=0;i<size.dimension;++i){
            const double increment=out.step_size*derivative[i];
            if(!std::isfinite(increment))throw std::overflow_error("Euler derivative increment overflow");
            next[i]=out.solution.y.back()[i]+increment;
            if(!std::isfinite(next[i]))throw std::overflow_error("Euler state update overflow");
        }
        out.solution.y.push_back(std::move(next));++out.work.steps;out.work.scalar_updates+=size.dimension;
        if(trace)trace->derivatives.push_back(std::move(derivative));
    }
    return out;
}
} // namespace euler_detail
std::size_t euler_trace_size(std::size_t dimension,int steps){return euler_detail::trace_size(euler_detail::geometry(dimension,steps));}
EulerOutcome euler_method_work(const OdeSystem& f,const std::vector<double>& y0,double t0,double t1,int steps){
    const auto size=euler_detail::validate(f,y0,t0,t1,steps);return euler_detail::core(f,y0,t0,t1,size,nullptr);
}
OdeSolution euler_method(const OdeSystem& f,std::vector<double> y0,double t0,double t1,int steps){return euler_method_work(f,y0,t0,t1,steps).solution;}
EulerTrace euler_method_trace(const OdeSystem& f,const std::vector<double>& y0,double t0,double t1,int steps,std::size_t max_entries){
    const auto size=euler_detail::validate(f,y0,t0,t1,steps);const auto entries=euler_detail::trace_size(size);
    if(max_entries&&entries>max_entries)throw std::length_error("Euler capture limit exceeded");
    EulerTrace trace;trace.captured_entries=entries;trace.outcome=euler_detail::core(f,y0,t0,t1,size,&trace);return trace;
}
} // namespace datamunge::algorithms
