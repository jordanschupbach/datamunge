#include <datamunge/algorithms/crank_nicolson.hpp>
/// @file crank_nicolson.cpp
/// @brief Shared validated CN evolution with one Thomas factorization and actual owned states.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

namespace datamunge::algorithms {
/// @brief Private checked dimensions, heat sampling, diagnostics and Thomas evolution.
namespace {
/// @brief Check a size sum before addition.
/// @param a First size.
/// @param b Second size.
/// @return Exact representable sum.
/// @throws std::length_error If the sum exceeds size_t.
std::size_t heat_add(std::size_t a,std::size_t b) {
  if(b>std::numeric_limits<std::size_t>::max()-a) throw std::length_error("heat: size sum overflow");
  return a+b;
}
/// @brief Check a size product before multiplication.
/// @param a First size.
/// @param b Second size.
/// @return Exact representable product.
/// @throws std::length_error If the product exceeds size_t.
std::size_t heat_multiply(std::size_t a,std::size_t b) {
  if(a && b>std::numeric_limits<std::size_t>::max()/a) throw std::length_error("heat: size product overflow");
  return a*b;
}
/// @brief Require finite double storage for an arithmetic result.
/// @param value Widened result, possibly already computed in double.
/// @return Rounded finite double value; signed zero is allowed.
/// @throws std::overflow_error If the value is nonfinite or outside double range.
double heat_finite(long double value) {
  if(!std::isfinite(value) || std::fabs(value)>std::numeric_limits<double>::max())
    throw std::overflow_error("heat: nonfinite or unrepresentable arithmetic");
  return static_cast<double>(value);
}
/// @brief Validate scalar heat parameters and all derived grid/step coefficients.
/// @param L Positive finite length.
/// @param D Nonnegative finite diffusivity.
/// @param nx Positive interior count.
/// @param dt Nonnegative finite step.
/// @param nsteps Nonnegative number of steps.
/// @return Fully checked metadata, including representable stencil diagonal.
/// @throws std::invalid_argument For invalid scalar domains.
/// @throws std::overflow_error For unrepresentable derived coefficients or time.
/// @throws std::length_error For unrepresentable dimension/work sizes.
HeatGrid heat_grid(double L,double D,int nx,double dt,int nsteps) {
  if(!std::isfinite(L) || L<=0 || !std::isfinite(D) || D<0 ||
     !std::isfinite(dt) || dt<0 || nx<1 || nsteps<0)
    throw std::invalid_argument("heat: invalid grid or time parameters");
  if(static_cast<std::uintmax_t>(nx)>std::numeric_limits<std::size_t>::max() ||
     static_cast<std::uintmax_t>(nsteps)>std::numeric_limits<std::size_t>::max())
    throw std::length_error("heat: dimension not representable");
  const auto n=static_cast<std::size_t>(nx);
  (void)heat_add(n,2);(void)heat_multiply(n,static_cast<std::size_t>(nsteps));
  const auto dx=heat_finite(static_cast<long double>(L)/(static_cast<long double>(nx)+1));
  if(dx==0 || !std::isfinite(nx*dx) || nx*dx>=L)
    throw std::overflow_error("heat: rounded grid loses interior coordinates");
  const auto ratio=heat_finite((static_cast<long double>(D)*dt)/(2*static_cast<long double>(dx)*dx));
  if(D>0 && dt>0 && ratio==0) throw std::overflow_error("heat: positive ratio underflow");
  (void)heat_finite(1.+2.*ratio);
  const auto time=heat_finite(static_cast<long double>(dt)*nsteps);
  return {L,D,nx,dx,dt,ratio,nsteps,time};
}
/// @brief Construct owned full-grid coordinates with exact domain endpoints.
/// @param grid Checked metadata.
/// @return nx+2 finite coordinates in ascending order.
/// @throws std::length_error If container capacity is exceeded.
/// @throws std::bad_alloc If coordinate storage cannot be allocated.
std::vector<double> heat_coordinates(const HeatGrid& grid) {
  std::vector<double> x(static_cast<std::size_t>(grid.interior_points)+2);
  for(std::size_t i=1;i+1<x.size();++i) x[i]=i*grid.dx;
  x.back()=grid.length;return x;
}
/// @brief Capture one actual full profile, RHS, scaled norm and solve residual.
/// @param u Actual nx interior values.
/// @param rhs Actual uneliminated RHS, empty for the initial state.
/// @param grid Validated grid metadata.
/// @param index Actual completed step count.
/// @return Owned nx+2 state with scaled RMS and rowwise backward error.
/// @throws std::overflow_error If a diagnostic cannot fit finite double storage.
/// @throws std::length_error If vector capacity is exceeded.
/// @throws std::bad_alloc If state storage cannot be allocated.
HeatStep heat_state(const std::vector<double>& u,const std::vector<double>& rhs,const HeatGrid& grid,int index) {
  HeatStep state;state.index=index;state.time=heat_finite(static_cast<long double>(index)*grid.dt);
  state.profile.resize(u.size()+2,0);std::copy(u.begin(),u.end(),state.profile.begin()+1);state.rhs=rhs;
  for(const auto v:u) state.max_abs=std::max(state.max_abs,std::fabs(v));
  if(state.max_abs) {
    long double sum=0;
    for(const auto v:u) { const auto z=static_cast<long double>(v)/state.max_abs;sum+=z*z; }
    state.rms=heat_finite(static_cast<long double>(state.max_abs)*std::sqrt(sum/u.size()));
  }
  if(!rhs.empty()) {
    const long double r=grid.r,diagonal=1.+2.*grid.r;
    for(std::size_t i=0;i<u.size();++i) {
      const long double center=diagonal*u[i],left=i?r*u[i-1]:0,right=i+1<u.size()?r*u[i+1]:0;
      const auto residual=std::fabs(center-left-right-rhs[i]);
      const auto scale=std::fabs(center)+std::fabs(left)+std::fabs(right)+std::fabs(rhs[i]);
      const auto value=scale?heat_finite(residual/scale):0;
      state.scaled_residual=std::max(state.scaled_residual,value);
    }
  }
  return state;
}
/// @brief Execute the sole sampled CN/Thomas recurrence, optionally retaining actual states.
/// @param u0 Callback sampled exactly once per interior point in ascending order.
/// @param grid Fully checked grid metadata.
/// @param trace Optional owned trace; null skips histories and diagnostics.
/// @return Owned nx+2 final profile with zero endpoints.
/// @throws std::invalid_argument For nonfinite callback results.
/// @throws std::overflow_error For nonfinite stencil, factor or solve arithmetic.
/// @throws std::length_error If container sizes exceed capacity.
/// @throws std::bad_alloc If working/result storage cannot be allocated.
/// Callback exceptions propagate unchanged.
std::vector<double> heat_kernel(const std::function<double(double)>& u0,const HeatGrid& grid,CrankNicolsonHeatTrace* trace) {
  const auto n=static_cast<std::size_t>(grid.interior_points);
  std::vector<double> u(n);
  for(std::size_t i=0;i<n;++i) {
    u[i]=u0((i+1)*grid.dx);
    if(!std::isfinite(u[i])) throw std::invalid_argument("heat: nonfinite initial profile");
  }
  if(trace) {
    trace->grid=grid;trace->coordinates=heat_coordinates(grid);
    trace->states.reserve(heat_add(static_cast<std::size_t>(grid.steps),1));
    trace->states.push_back(heat_state(u,{},grid,0));
  }
  const double r=grid.r,diagonal=1.+2.*r;
  std::vector<double> pivots,multipliers;
  if(grid.steps) {
    pivots.assign(n,diagonal);multipliers.assign(n,0);
    for(std::size_t i=1;i<n;++i) {
      multipliers[i]=heat_finite(-r/pivots[i-1]);
      pivots[i]=heat_finite(pivots[i]-multipliers[i]*(-r));
      if(pivots[i]<=0) throw std::overflow_error("heat: nonpositive Thomas pivot");
    }
    if(trace) { trace->pivots=pivots;trace->multipliers=multipliers;trace->factorizations=1; }
  }
  std::vector<double> rhs(n),reduced(n),next(n);
  for(int step=0;step<grid.steps;++step) {
    for(std::size_t i=0;i<n;++i) {
      const auto left=i?u[i-1]:0,right=i+1<n?u[i+1]:0;
      rhs[i]=heat_finite(r*left+(1.-2.*r)*u[i]+r*right);
    }
    reduced=rhs;
    for(std::size_t i=1;i<n;++i) reduced[i]=heat_finite(reduced[i]-multipliers[i]*reduced[i-1]);
    next[n-1]=heat_finite(reduced[n-1]/pivots[n-1]);
    for(std::size_t i=n-1;i-- >0;) next[i]=heat_finite((reduced[i]-(-r)*next[i+1])/pivots[i]);
    u.swap(next);
    if(trace) {
      trace->states.push_back(heat_state(u,rhs,grid,step+1));
      ++trace->solves;trace->rhs_entries+=n;
    }
  }
  std::vector<double> out(n+2,0);std::copy(u.begin(),u.end(),out.begin()+1);return out;
}
} // namespace

std::vector<double> crank_nicolson_heat(const std::function<double(double)>& u0,double L,double D,int nx,double dt,int nsteps) {
  if(!u0) throw std::invalid_argument("heat: empty initial-profile callback");
  const auto grid=heat_grid(L,D,nx,dt,nsteps);return heat_kernel(u0,grid,nullptr);
}
CrankNicolsonHeatTrace crank_nicolson_heat_trace(const std::function<double(double)>& u0,double L,double D,int nx,double dt,int nsteps,std::size_t max_values) {
  if(!u0) throw std::invalid_argument("heat: empty initial-profile callback");
  const auto grid=heat_grid(L,D,nx,dt,nsteps);
  const auto n=static_cast<std::size_t>(nx),steps=static_cast<std::size_t>(nsteps);
  const auto values=heat_add(heat_multiply(heat_add(n,2),heat_add(steps,1)),heat_multiply(n,steps));
  if(max_values && values>max_values) throw std::length_error("heat: complete history budget exceeded");
  CrankNicolsonHeatTrace result;(void)heat_kernel(u0,grid,&result);return result;
}
HeatSineMode heat_sine_mode(double L,double D,int nx,int mode,double dt,int nsteps) {
  if(mode<1 || mode>nx) throw std::invalid_argument("heat: mode outside [1,nx]");
  HeatSineMode result;result.grid=heat_grid(L,D,nx,dt,nsteps);result.mode=mode;
  result.coordinates=heat_coordinates(result.grid);
  const auto n=static_cast<std::size_t>(nx),size=n+2;
  result.initial.resize(size,0);result.continuous.resize(size,0);result.semidiscrete.resize(size,0);result.discrete.resize(size,0);
  const long double pi=std::acos(-1.L),angle=mode*pi/(2*(static_cast<long double>(nx)+1)),s=std::sin(angle);
  result.eigenvalue=heat_finite(-4*static_cast<long double>(D)*s*s/(static_cast<long double>(result.grid.dx)*result.grid.dx));
  const long double z=-static_cast<long double>(dt)*result.eigenvalue/2;
  result.amplification=heat_finite((1-z)/(1+z));
  const auto discrete=std::pow(static_cast<long double>(result.amplification),nsteps);
  const long double wave=mode*pi/L;
  const auto continuous=std::exp(-static_cast<long double>(D)*wave*wave*result.grid.final_time);
  const auto spatial=std::exp(static_cast<long double>(result.eigenvalue)*result.grid.final_time);
  for(std::size_t i=1;i<=n;++i) {
    result.initial[i]=std::sin(static_cast<double>(mode*pi*i/(static_cast<long double>(nx)+1)));
    result.continuous[i]=heat_finite(continuous*result.initial[i]);
    result.semidiscrete[i]=heat_finite(spatial*result.initial[i]);
    result.discrete[i]=heat_finite(discrete*result.initial[i]);
  }
  return result;
}
double heat_profile_error(const std::vector<double>& actual,const std::vector<double>& reference) {
  if(actual.size()!=reference.size()) throw std::invalid_argument("heat: profile size mismatch");
  long double maximum=0;
  for(std::size_t i=0;i<actual.size();++i) {
    if(!std::isfinite(actual[i]) || !std::isfinite(reference[i])) throw std::invalid_argument("heat: nonfinite comparison value");
    maximum=std::max(maximum,std::fabs(static_cast<long double>(actual[i])-reference[i]));
  }
  return heat_finite(maximum);
}
} // namespace datamunge::algorithms
