/// @file cohen_sutherland.cpp
/// @brief Validated widened Cohen-Sutherland kernel and conditional actual decision recording.
#include <datamunge/algorithms/cohen_sutherland.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace datamunge::algorithms {
/// @brief Private finite validation, working geometry and one executing clipping kernel.
namespace cohen_sutherland_detail {
/// @brief Working scalar, widened from all public double inputs before subtraction.
using Real=long double;

/// @brief Check finite coordinates and inclusive rectangle order before any classification.
/// @tparam N Complete coordinate count, at least four.
/// @param coordinates Complete coordinate list, ending in xmin,ymin,xmax,ymax.
/// @throws std::invalid_argument If a value is nonfinite or either rectangle interval is reversed.
template<std::size_t N>
void validate(const std::array<double,N>& coordinates) {
  static_assert(N>=4);
  for(const auto x:coordinates)if(!std::isfinite(x))throw std::invalid_argument("Cohen-Sutherland requires finite coordinates");
  if(coordinates[N-4]>coordinates[N-2] || coordinates[N-3]>coordinates[N-1])
    throw std::invalid_argument("Cohen-Sutherland requires ordered rectangle bounds");
}

/// @brief Combine outside bits for finite working coordinates and validated bounds.
/// @param x Current x.
/// @param y Current y.
/// @param box Inclusive xmin,ymin,xmax,ymax.
/// @return Left=1,right=2,bottom=4,top=8; boundaries are inside.
std::uint8_t code(Real x,Real y,const std::array<Real,4>& box) {
  std::uint8_t result=0;
  if(x<box[0])result|=1;else if(x>box[2])result|=2;
  if(y<box[1])result|=4;else if(y>box[3])result|=8;
  return result;
}

/// @brief Reject an unrepresentable working result before it can affect region tests.
/// @param value Computed working scalar.
/// @return The same finite value.
/// @throws std::overflow_error If value is not finite.
Real finite(Real value) {
  if(!std::isfinite(value))throw std::overflow_error("Cohen-Sutherland working arithmetic overflow");
  return value;
}

/// @brief Interpolate with a widened line determinant, bounded fallback and endpoint-range projection.
/// @param a Selected outside coordinate on the pinned axis.
/// @param b Other endpoint coordinate on the pinned axis, inside that half-plane.
/// @param edge Chosen boundary on the pinned axis.
/// @param free_a Outside endpoint's other coordinate.
/// @param free_b Other endpoint's other coordinate.
/// @return Finite free coordinate between the current free endpoint values.
/// @throws std::overflow_error If differences/ratio are nonfinite or a valid crossing is lost numerically.
Real intersection(Real a,Real b,Real edge,Real free_a,Real free_b) {
  const auto numerator=finite(edge-a),denominator=finite(b-a);
  if(denominator==0)throw std::overflow_error("Cohen-Sutherland lost crossing denominator");
  const auto t=finite(numerator/denominator);
  if(t<0 || t>1)throw std::overflow_error("Cohen-Sutherland crossing ratio outside segment");
  // Keep the small boundary term separate from large endpoint coordinates.
  // This preserves x=y at huge scales, and exact small-integer corner hits.
  const auto p=free_a*b,q=free_b*a,delta=free_b-free_a,term=edge*delta;
  const auto determinant=p-q,sum=determinant+term;
  const bool underflow=(p==0 && free_a!=0 && b!=0) || (q==0 && free_b!=0 && a!=0) ||
                       (term==0 && edge!=0 && delta!=0);
  const auto value=std::isfinite(p) && std::isfinite(q) && std::isfinite(delta) &&
      std::isfinite(term) && std::isfinite(determinant) && std::isfinite(sum) && !underflow
      ? finite(sum/denominator):finite(std::lerp(free_a,free_b,t));
  return std::clamp(value,std::min(free_a,free_b),std::max(free_a,free_b));
}

/// @brief Convert a finite accepted working coordinate to the historical double result.
/// @param value Accepted coordinate, bounded by the finite original rectangle.
/// @return Finite rounded double coordinate.
/// @throws std::overflow_error If conversion cannot produce a finite double.
double narrow(Real value) {
  if(std::fabs(value)>std::numeric_limits<double>::max())throw std::overflow_error("Cohen-Sutherland result exceeds double");
  const auto result=static_cast<double>(value);
  if(!std::isfinite(result))throw std::overflow_error("Cohen-Sutherland result is nonfinite");
  return result;
}

/// @brief Run up to four actual half-plane cuts, optionally retaining each decision.
/// @tparam Record Whether to own pre-decision records; false allocates no storage.
/// @param input Original x0,y0,x1,y1,xmin,ymin,xmax,ymax, checked before any decision.
/// @return Accepted/rejected legacy result and bounded actual work, plus optional records.
/// @throws std::invalid_argument If validation fails.
/// @throws std::overflow_error If checked arithmetic or the four-edge progress bound fails.
/// @throws std::bad_alloc If a requested record cannot be allocated.
/// @throws std::length_error If vector capacity is exceeded.
template<bool Record>
CohenSutherlandTrace clip(const std::array<double,8>& input) {
  validate(input);
  Real x0=input[0],y0=input[1],x1=input[2],y1=input[3];
  const std::array<Real,4> box{input[4],input[5],input[6],input[7]};
  auto c0=code(x0,y0,box),c1=code(x1,y1,box);
  CohenSutherlandTrace result;result.outcode_evaluations=2;
  unsigned processed=0;
  for(;;) {
    CohenSutherlandStep step;step.x0=x0;step.y0=y0;step.x1=x1;step.y1=y1;step.code0=c0;step.code1=c1;
    ++result.or_tests;
    if((c0|c1)==0) {
      step.action=ClipAction::Accept;
      result.result={true,narrow(x0),narrow(y0),narrow(x1),narrow(y1)};
      if constexpr(Record)result.steps.push_back(step);
      return result;
    }
    ++result.and_tests;
    if((c0&c1)!=0) {
      step.action=ClipAction::Reject;
      if constexpr(Record)result.steps.push_back(step);
      return result;
    }
    const auto outside=c0?c0:c1;
    step.replaced_endpoint=c0?0:1;
    // Reuse the original supporting segment, so earlier rounded intersections
    // cannot tilt the line and turn an exact small-integer corner hit into a miss.
    const Real ax=input[0],ay=input[1],bx=input[2],by=input[3];
    Real x=0,y=0;unsigned edge=0;
    if(outside&8) {edge=8;step.action=ClipAction::Top;y=box[3];x=intersection(ay,by,y,ax,bx);}
    else if(outside&4) {edge=4;step.action=ClipAction::Bottom;y=box[1];x=intersection(ay,by,y,ax,bx);}
    else if(outside&2) {edge=2;step.action=ClipAction::Right;x=box[2];y=intersection(ax,bx,x,ay,by);}
    else {edge=1;step.action=ClipAction::Left;x=box[0];y=intersection(ax,bx,x,ay,by);}
    x=std::clamp(x,std::min(x0,x1),std::max(x0,x1));
    y=std::clamp(y,std::min(y0,y1),std::max(y0,y1));
    // Each pinned half-plane must stay satisfied; there are only four distinct edges.
    if((processed&edge)!=0 || result.intersections==4)
      throw std::overflow_error("Cohen-Sutherland edge progress failed");
    processed|=edge;
    const auto replacement=code(x,y,box);
    if((replacement&processed)!=0)throw std::overflow_error("Cohen-Sutherland lost a processed half-plane");
    step.intersection_x=x;step.intersection_y=y;step.new_code=replacement;
    if(c0){x0=x;y0=y;c0=replacement;}else{x1=x;y1=y;c1=replacement;}
    ++result.intersections;++result.outcode_evaluations;
    if constexpr(Record)result.steps.push_back(step);
  }
}
} // namespace cohen_sutherland_detail

std::uint8_t cohen_sutherland_outcode(double x,double y,double xmin,double ymin,double xmax,double ymax) {
  cohen_sutherland_detail::validate(std::array<double,6>{x,y,xmin,ymin,xmax,ymax});
  return cohen_sutherland_detail::code(x,y,{xmin,ymin,xmax,ymax});
}

ClippedSegment cohen_sutherland_clip(double x0,double y0,double x1,double y1,double xmin,double ymin,double xmax,double ymax) {
  return cohen_sutherland_detail::clip<false>({x0,y0,x1,y1,xmin,ymin,xmax,ymax}).result;
}

CohenSutherlandTrace cohen_sutherland_clip_trace(double x0,double y0,double x1,double y1,double xmin,double ymin,double xmax,double ymax) {
  return cohen_sutherland_detail::clip<true>({x0,y0,x1,y1,xmin,ymin,xmax,ymax});
}
} // namespace datamunge::algorithms
