#include <datamunge/algorithms/cubic_spline.hpp>
/// @file cubic_spline.cpp
/// @brief Shared validated natural-moment solve, local-power derivatives and complete sampled diagnostics.
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace datamunge::algorithms {
/// @brief Private shape/input checks and the sole natural-fit/evaluation kernels.
namespace {
/// @brief Convert a computed value to finite double storage.
/// @param value Possibly widened arithmetic result.
/// @return Rounded finite double, including zero.
/// @throws std::overflow_error If the value is nonfinite or exceeds double range.
double spline_finite(long double value) {
  if(!std::isfinite(value) || std::fabs(value)>std::numeric_limits<double>::max())
    throw std::overflow_error("cubic spline: nonfinite or unrepresentable arithmetic");
  return static_cast<double>(value);
}
/// @brief Require all entries of a read-only numeric array to be finite.
/// @param values Input values, including allowed empty arrays.
/// @throws std::invalid_argument If any entry is nonfinite.
void spline_finite_array(const std::vector<double>& values) {
  for(const auto value:values) if(!std::isfinite(value)) throw std::invalid_argument("cubic spline: nonfinite input");
}
/// @brief Validate finite strictly increasing knot coordinates.
/// @param xs Knot coordinates; empty/singleton arrays are allowed.
/// @throws std::invalid_argument For a nonfinite, unordered or duplicate knot.
void spline_knots(const std::vector<double>& xs) {
  spline_finite_array(xs);
  for(std::size_t i=1;i<xs.size();++i)if(!(xs[i]>xs[i-1])) throw std::invalid_argument("cubic spline: knots must increase");
}
/// @brief Validate complete natural-fit inputs without allocation.
/// @param xs Finite increasing knots.
/// @param ys Finite equally sized ordinates.
/// @throws std::invalid_argument For invalid knots, ordinate shape or finite values.
void spline_data(const std::vector<double>& xs,const std::vector<double>& ys) {
  if(xs.size()!=ys.size())throw std::invalid_argument("cubic spline: mismatched data lengths");
  spline_knots(xs);spline_finite_array(ys);
}
/// @brief Validate every public-record coefficient and its exact degenerate/nondegenerate shape.
/// @param s Read-only model, which need not satisfy C2/natural equations.
/// @throws std::invalid_argument For invalid knots, coefficient shapes or nonfinite values.
void spline_model(const CubicSpline& s) {
  spline_knots(s.xs);const auto n=s.xs.size(),count=n>=2?n-1:0;
  if(s.a.size()!=(n==1?1:count) || s.b.size()!=count || s.c.size()!=count || s.d.size()!=count)
    throw std::invalid_argument("cubic spline: malformed coefficient arrays");
  spline_finite_array(s.a);spline_finite_array(s.b);spline_finite_array(s.c);spline_finite_array(s.d);
}
/// @brief Require an implemented value/derivative order.
/// @param order Order zero through three.
/// @throws std::invalid_argument If order exceeds three.
void spline_order(unsigned order) {
  if(order>3)throw std::invalid_argument("cubic spline: derivative order outside [0,3]");
}
/// @brief Evaluate one previously validated model/query/order with endpoint-piece extrapolation.
/// @param s Structurally validated owned/read-only coefficients.
/// @param x Finite query coordinate.
/// @param order Valid derivative order in [0,3].
/// @return Rounded value or derivative; zero/constant conventions for small models.
/// @throws std::overflow_error For nonrepresentable offsets or evaluation arithmetic.
double spline_evaluate(const CubicSpline& s,double x,unsigned order) {
  const auto n=s.xs.size();
  if(n==0)return 0;
  if(n==1)return order?0:s.a[0];
  const auto position=static_cast<std::size_t>(std::upper_bound(s.xs.begin(),s.xs.end(),x)-s.xs.begin());
  const auto i=position==0?0:std::min(position-1,n-2);
  if(order==3)return spline_finite(6.*s.d[i]);
  const auto t=spline_finite(x-s.xs[i]);
  if(order==2)return spline_finite(2.*s.c[i]+6.*t*s.d[i]);
  if(order==1)return spline_finite(s.b[i]+t*(2.*s.c[i]+3.*t*s.d[i]));
  return spline_finite(s.a[i]+t*(s.b[i]+t*(s.c[i]+t*s.d[i])));
}
/// @brief Execute the canonical natural moment solve, optionally retaining actual arrays/work.
/// Validates all input before the complete optional cap or allocation. The
/// same arithmetic constructs legacy and traced models; returned arrays own data.
/// @param xs Increasing finite knot coordinates.
/// @param ys Finite equally sized ordinates.
/// @param trace Optional actual-array destination; null skips trace copies.
/// @param max_knots Optional complete knot cap, zero disables it.
/// @return Owned zero/constant/linear/natural-cubic local-power model.
/// @throws std::invalid_argument For invalid data.
/// @throws std::length_error For an exceeded complete cap or container capacity.
/// @throws std::overflow_error For nonfinite widths, secants, solve arithmetic or coefficients.
/// @throws std::bad_alloc If working/result storage cannot be allocated.
CubicSpline spline_fit(const std::vector<double>& xs,const std::vector<double>& ys,NaturalCubicSplineTrace* trace,std::size_t max_knots) {
  spline_data(xs,ys);const auto n=xs.size();
  if(max_knots && n>max_knots)throw std::length_error("cubic spline: complete knot cap exceeded");
  CubicSpline s;s.xs=xs;
  if(n<2) { s.a=ys;if(trace)trace->moments.assign(n,0);return s; }
  std::vector<double> h(n-1),secants(n-1),moments(n,0);
  for(std::size_t i=0;i+1<n;++i) {
    h[i]=spline_finite(xs[i+1]-xs[i]);
    secants[i]=spline_finite((static_cast<long double>(ys[i+1])-ys[i])/h[i]);
  }
  if(trace) { trace->widths=h;trace->secants=secants; }
  if(n>=3) {
    const auto k=n-2;std::vector<double> lower(k,0),diag(k),upper(k,0),rhs(k);
    for(std::size_t i=0;i<k;++i) {
      if(i)lower[i]=h[i];
      diag[i]=spline_finite(2.L*(static_cast<long double>(h[i])+h[i+1]));
      if(i+1<k)upper[i]=h[i+1];
      rhs[i]=spline_finite(6.L*(static_cast<long double>(secants[i+1])-secants[i]));
    }
    if(trace) { trace->lower=lower;trace->diagonal=diag;trace->upper=upper;trace->rhs=rhs;trace->multipliers.assign(k,0); }
    for(std::size_t i=1;i<k;++i) {
      const auto multiplier=spline_finite(lower[i]/diag[i-1]);
      diag[i]=spline_finite(diag[i]-multiplier*upper[i-1]);
      rhs[i]=spline_finite(rhs[i]-multiplier*rhs[i-1]);
      if(diag[i]<=0)throw std::overflow_error("cubic spline: nonpositive Thomas pivot");
      if(trace) { trace->multipliers[i]=multiplier;++trace->eliminations; }
    }
    moments[n-2]=spline_finite(rhs[k-1]/diag[k-1]);
    for(std::size_t i=k-1;i-- >0;)moments[i+1]=spline_finite((rhs[i]-upper[i]*moments[i+2])/diag[i]);
    if(trace) { trace->pivots=diag;trace->reduced_rhs=rhs;trace->back_substitutions=k; }
  }
  s.a.resize(n-1);s.b.resize(n-1);s.c.resize(n-1);s.d.resize(n-1);
  for(std::size_t i=0;i+1<n;++i) {
    s.a[i]=ys[i];
    s.b[i]=spline_finite(static_cast<long double>(secants[i])-static_cast<long double>(h[i])*(2.L*moments[i]+moments[i+1])/6);
    s.c[i]=moments[i]/2;
    s.d[i]=spline_finite((static_cast<long double>(moments[i+1])-moments[i])/(6.L*h[i]));
    if(trace)++trace->coefficient_intervals;
  }
  if(trace)trace->moments=moments;
  return s;
}
} // namespace

CubicSpline natural_cubic_spline(const std::vector<double>& xs,const std::vector<double>& ys) {
  return spline_fit(xs,ys,nullptr,0);
}
NaturalCubicSplineTrace natural_cubic_spline_trace(const std::vector<double>& xs,const std::vector<double>& ys,std::size_t max_knots) {
  NaturalCubicSplineTrace result;result.spline=spline_fit(xs,ys,&result,max_knots);return result;
}
double cubic_spline_eval(const CubicSpline& spline,double x) {
  if(!std::isfinite(x))throw std::invalid_argument("cubic spline: nonfinite query");
  spline_model(spline);return spline_evaluate(spline,x,0);
}
PreparedCubicSpline::PreparedCubicSpline(CubicSpline spline):model_(std::move(spline)) { spline_model(model_); }
double PreparedCubicSpline::evaluate(double x,unsigned derivative) const {
  spline_order(derivative);
  if(!std::isfinite(x))throw std::invalid_argument("cubic spline: nonfinite query");
  return spline_evaluate(model_,x,derivative);
}
std::vector<double> PreparedCubicSpline::evaluate(const std::vector<double>& queries,unsigned derivative) const {
  spline_order(derivative);spline_finite_array(queries);std::vector<double> result;result.reserve(queries.size());
  for(const auto x:queries)result.push_back(spline_evaluate(model_,x,derivative));return result;
}
InterpolationErrorScan interpolation_error_scan(const std::function<double(double)>& model,const std::function<double(double)>& reference,const std::vector<double>& queries,std::size_t max_queries) {
  if(!model || !reference)throw std::invalid_argument("interpolation error: empty callback");
  spline_finite_array(queries);
  if(max_queries && queries.size()>max_queries)throw std::length_error("interpolation error: complete query cap exceeded");
  InterpolationErrorScan result;result.queries=queries;result.predictions.reserve(queries.size());result.references.reserve(queries.size());result.absolute_errors.reserve(queries.size());result.has_samples=!queries.empty();
  for(const auto x:queries) {
    const auto predicted=model(x),expected=reference(x);
    if(!std::isfinite(predicted) || !std::isfinite(expected))throw std::invalid_argument("interpolation error: nonfinite callback value");
    const auto error=spline_finite(std::fabs(static_cast<long double>(predicted)-expected));
    result.predictions.push_back(predicted);result.references.push_back(expected);result.absolute_errors.push_back(error);
    if(error>result.max_abs_error) { result.max_abs_error=error;result.worst_index=result.absolute_errors.size()-1; }
  }
  if(result.max_abs_error>0) {
    long double sum=0;
    for(const auto error:result.absolute_errors) { const auto z=static_cast<long double>(error)/result.max_abs_error;sum+=z*z; }
    result.rms_error=spline_finite(static_cast<long double>(result.max_abs_error)*std::sqrt(sum/queries.size()));
  }
  return result;
}
} // namespace datamunge::algorithms
