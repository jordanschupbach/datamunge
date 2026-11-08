/// @file
/// @brief Checked original weighted-blend kernel with optional actual rows or subdivision diagonals.
#include <datamunge/algorithms/de_casteljau.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace datamunge::algorithms {
/// @brief Private finite validation, complete count checks and common curve evaluator.
namespace {
/// @brief Multiply two size counts before any allocation or arithmetic wraparound.
/// @param a First count.
/// @param b Second count.
/// @return Exact representable product.
/// @throws std::length_error Product exceeds size_t.
std::size_t multiply(std::size_t a,std::size_t b) {
  if(a && b>std::numeric_limits<std::size_t>::max()/a)throw std::length_error("de Casteljau count overflow");
  return a*b;
}
/// @brief Enforce a zero-disabled inclusive resource ceiling.
/// @param count Prospective complete count.
/// @param limit Zero disables the optional cap.
/// @throws std::length_error Nonzero cap exceeded.
void cap(std::size_t count,std::size_t limit) {
  if(limit && count>limit)throw std::length_error("de Casteljau complete resource cap exceeded");
}
/// @brief Validate a parameter and every control coordinate before caps or output.
/// @param control Borrowed points, unchanged, empty allowed.
/// @param t Parameter to check for finiteness, not unit-range membership.
/// @throws std::invalid_argument Any nonfinite coordinate or parameter.
void validate(const std::vector<PlanarPoint>& control,double t) {
  if(!std::isfinite(t))throw std::invalid_argument("de Casteljau parameter must be finite");
  for(const auto& p:control)if(!std::isfinite(p.x)||!std::isfinite(p.y))throw std::invalid_argument("de Casteljau controls must be finite");
}
/// @brief Evaluate one validated curve, optionally recording actual rows or boundary diagonals.
/// @param control Finite borrowed controls, unchanged.
/// @param t Finite parameter, possibly outside [0,1] for value/trace calls.
/// @param trace Optional full-triangle target with checked geometry/cap already established.
/// @param split Optional diagonal target with checked geometry/2m cap and nonempty input.
/// @return Actual final point or legacy empty (0,0) sentinel.
/// @throws std::overflow_error Any computed coordinate blend is nonfinite.
/// @throws std::length_error Container capacity exceeded.
/// @throws std::bad_alloc Allocation fails.
PlanarPoint evaluate(const std::vector<PlanarPoint>& control,double t,DeCasteljauTrace* trace,BezierSubdivision* split) {
  if(control.empty())return {};
  std::vector<PlanarPoint> p(control);std::size_t blends=0;
  if(trace) {trace->levels.reserve(p.size());trace->levels.push_back(p);}
  if(split) {split->left.reserve(p.size());split->right.reserve(p.size());split->left.push_back(p.front());split->right.push_back(p.back());}
  const double one_minus=1.0-t;
  for(std::size_t active=p.size();active>1;) {
    --active;
    for(std::size_t i=0;i<active;++i) {
      p[i].x=one_minus*p[i].x+t*p[i+1].x;p[i].y=one_minus*p[i].y+t*p[i+1].y;
      if(!std::isfinite(p[i].x)||!std::isfinite(p[i].y))throw std::overflow_error("de Casteljau nonfinite weighted blend");
      if(trace||split)++blends;
    }
    if(trace)trace->levels.emplace_back(p.begin(),p.begin()+active);
    if(split) {split->left.push_back(p.front());split->right.push_back(p[active-1]);}
  }
  if(trace) {trace->point=p[0];trace->point_blends=blends;}
  if(split) {std::reverse(split->right.begin(),split->right.end());split->point_blends=blends;}
  return p[0];
}
} // namespace
DeCasteljauShape de_casteljau_shape(std::size_t control_points) {
  if(!control_points)return {};
  if(control_points==std::numeric_limits<std::size_t>::max())throw std::length_error("de Casteljau triangle dimension overflow");
  const auto m=control_points;
  const auto stored=m%2?multiply(m,(m+1)/2):multiply(m/2,m+1);
  const auto blends=m%2?multiply(m,(m-1)/2):multiply(m/2,m-1);
  return {m,stored,blends,multiply(2,blends)};
}
PlanarPoint de_casteljau(const std::vector<PlanarPoint>& control,double t) {
  validate(control,t);return evaluate(control,t,nullptr,nullptr);
}
DeCasteljauTrace de_casteljau_trace(const std::vector<PlanarPoint>& control,double t,std::size_t max_points) {
  validate(control,t);DeCasteljauTrace out;out.parameter=t;out.shape=de_casteljau_shape(control.size());cap(out.shape.stored_points,max_points);
  evaluate(control,t,&out,nullptr);return out;
}
BezierSubdivision de_casteljau_subdivide(const std::vector<PlanarPoint>& control,double t,std::size_t max_points) {
  validate(control,t);
  if(control.empty()||t<0||t>1)throw std::invalid_argument("Bezier subdivision requires controls and a unit parameter");
  (void)de_casteljau_shape(control.size());cap(multiply(2,control.size()),max_points);
  BezierSubdivision out;out.parameter=t;evaluate(control,t,nullptr,&out);return out;
}
std::vector<PlanarPoint> de_casteljau_sample(const std::vector<PlanarPoint>& control,std::size_t segments,double first,double last,std::size_t max_points,std::size_t max_blends) {
  validate(control,first);if(!std::isfinite(last)||!segments)throw std::invalid_argument("Bezier sampling requires finite bounds and positive segments");
  if(segments==std::numeric_limits<std::size_t>::max())throw std::length_error("Bezier sample count overflow");
  const auto count=segments+1;const auto shape=de_casteljau_shape(control.size());
  cap(count,max_points);cap(multiply(count,shape.point_blends),max_blends);
  std::vector<PlanarPoint> out;out.reserve(count);
  for(std::size_t i=0;i<count;++i) {
    const double t=i==0?first:(i==segments?last:std::lerp(first,last,static_cast<double>(i)/static_cast<double>(segments)));
    out.push_back(evaluate(control,t,nullptr,nullptr));
  }
  return out;
}
} // namespace datamunge::algorithms
