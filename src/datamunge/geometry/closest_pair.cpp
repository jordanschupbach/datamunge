/// @file closest_pair.cpp
/// @brief Checked all-pairs distance selection, actual incumbent recording and bounded rectangle sampling.
#include <datamunge/geometry/closest_pair.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
namespace datamunge::geometry {
/// @brief Private validation, complete pair forecasts and one optional-recording scan.
namespace closest_pair_detail {
/// @brief Widened coordinate difference and Euclidean-distance working type.
using Real=long double;
/// @brief Increment an actual named operation without wraparound.
/// @param value Mutable actual counter.
/// @throws std::overflow_error If the actual count cannot fit size_t.
void count(std::size_t& value){if(value==std::numeric_limits<std::size_t>::max())throw std::overflow_error("closest_pair: counter overflow");++value;}
/// @brief Check one widened arithmetic result for finite range.
/// @param value Computed candidate.
/// @return Finite value unchanged.
/// @throws std::overflow_error If the candidate is nonfinite.
Real checked(Real value){if(!std::isfinite(value))throw std::overflow_error("closest_pair: nonfinite widened arithmetic");return value;}
/// @brief Compute the complete unordered-pair forecast without an overflowing intermediate product.
/// @param n Validated point count.
/// @return Exact n(n-1)/2.
/// @throws std::length_error If the count exceeds size_t.
std::size_t pairs(std::size_t n){auto a=n,b=n-1;if(a%2==0)a/=2;else b/=2;if(b && a>std::numeric_limits<std::size_t>::max()/b)throw std::length_error("closest_pair: complete pair count overflow");return a*b;}
/// @brief Validate every coordinate before work forecasts/caps or distance evaluation.
/// @param points Original borrowed finite point array.
/// @return Actual complete validation visits.
/// @throws std::invalid_argument If length is below two or any coordinate is nonfinite.
/// @throws std::overflow_error If actual visits exceed size_t.
std::size_t validate(const std::vector<Point2D>& points){if(points.size()<2)throw std::invalid_argument("closest_pair: need at least 2 points");std::size_t visits=0;for(const auto& p:points){if(!std::isfinite(p.x) || !std::isfinite(p.y))throw std::invalid_argument("closest_pair: nonfinite coordinate");count(visits);}return visits;}
/// @brief Execute the common all-pairs scan and optionally copy actual incumbent changes.
/// @tparam Record Whether actual improvement records are retained.
/// @param points Borrowed original points.
/// @param options Complete pair cap and online actual record cap.
/// @return Final owned selection/work and optional actual records.
/// @throws std::invalid_argument If the full point input is invalid.
/// @throws std::length_error If pair count, work cap, online record cap or storage capacity is exceeded.
/// @throws std::overflow_error If widened arithmetic or actual counts cannot fit.
/// @throws std::bad_alloc If optional trace allocation fails.
template<bool Record>
ClosestPairTrace run(const std::vector<Point2D>& points,ClosestPairOptions options) {
  ClosestPairTrace out;out.work.validated_points=validate(points);const auto planned=pairs(points.size());
  if(options.max_pair_visits && planned>options.max_pair_visits)throw std::length_error("closest_pair: complete pair-work cap exceeded");
  bool initialized=false;
  for(std::size_t i=0;i<points.size();++i)for(std::size_t j=i+1;j<points.size();++j) {
    count(out.work.pair_visits);count(out.work.distance_evaluations);
    const auto distance=checked(std::hypot(checked(Real(points[i].x)-points[j].x),checked(Real(points[i].y)-points[j].y)));
    if(distance==0)count(out.work.zero_pairs);
    if(!initialized || distance<out.result.distance) {
      out.result={i,j,points[i],points[j],distance};count(out.work.improvements);initialized=true;
      if constexpr(Record){if(options.max_improvements && out.improvements.size()>=options.max_improvements)throw std::length_error("closest_pair: actual improvement-record cap exceeded");out.improvements.push_back({out.work.pair_visits,i,j,distance});}
    }else if(distance==out.result.distance)count(out.work.ties);
  }
  return out;
}
/// @brief Validate one strict finite interval and its double-range width before sampling.
/// @param low Inclusive lower bound.
/// @param high Exclusive upper bound.
/// @throws std::invalid_argument If bounds are nonfinite/unordered or the width exceeds finite double.
void interval(double low,double high){if(!std::isfinite(low) || !std::isfinite(high) || !(low<high) || Real(high)-low>std::numeric_limits<double>::max())throw std::invalid_argument("rectangle points: invalid finite interval");}
/// @brief Keep one finite generated draw below its exclusive upper bound despite rounding.
/// @param value Generated coordinate from a validated distribution.
/// @param low Validated inclusive lower bound.
/// @param high Validated exclusive upper bound.
/// @return Finite in-interval coordinate; any upper-rounded draw moves one double inward.
/// @throws std::overflow_error If the generated draw is nonfinite.
double interior(double value,double low,double high){if(!std::isfinite(value))throw std::overflow_error("rectangle points: nonfinite draw");return std::clamp(value,low,std::nextafter(high,low));}
}
ClosestPairSelection closest_pair_indexed(const std::vector<Point2D>& points,ClosestPairOptions options){return closest_pair_detail::run<false>(points,options).result;}
ClosestPairTrace closest_pair_trace(const std::vector<Point2D>& points,ClosestPairOptions options){return closest_pair_detail::run<true>(points,options);}
std::vector<Point2D> uniform_rectangle_points(std::size_t count,std::uint32_t seed,double x_min,double x_max,double y_min,double y_max) {
  closest_pair_detail::interval(x_min,x_max);closest_pair_detail::interval(y_min,y_max);
  std::mt19937 engine(seed);std::uniform_real_distribution<double> x(x_min,x_max),y(y_min,y_max);std::vector<Point2D> points;points.reserve(count);
  for(std::size_t i=0;i<count;++i)points.push_back({closest_pair_detail::interior(x(engine),x_min,x_max),closest_pair_detail::interior(y(engine),y_min,y_max)});
  return points;
}
}
