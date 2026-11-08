/// @file cone_algorithm.cpp
/// @brief One checked planar gap kernel, original disk rejection sampling and radial mask diagnostics.
#include <datamunge/geometry/cone_algorithm.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <random>
#include <stdexcept>
#include <utility>

namespace datamunge::geometry {
/// @brief Private widened geometry, checked counts and conditional actual direction recording.
namespace cone_detail {
/// @brief Widened working coordinate/angle type.
using Real=long double;
/// @brief One full angular turn in the working type.
constexpr Real tau=2*std::numbers::pi_v<Real>;

/// @brief Reject nonfinite working arithmetic before distance/angle decisions.
/// @param value Computed working scalar.
/// @return The same finite value.
/// @throws std::overflow_error If value is not finite.
Real finite(Real value) {
  if(!std::isfinite(value))throw std::overflow_error("cone surface: widened arithmetic overflow");
  return value;
}

/// @brief Validate a complete finite planar input before work limits or allocation.
/// @param points Borrowed original points.
/// @return Number of actual successful point-validation visits.
/// @throws std::invalid_argument If any coordinate is nonfinite.
std::size_t validate(const std::vector<Point2D>& points) {
  for(const auto& p:points)if(!std::isfinite(p.x) || !std::isfinite(p.y))
    throw std::invalid_argument("cone surface: coordinates must be finite");
  return points.size();
}

/// @brief Validate positive radius and an open-sector width in (0,2pi].
/// @param radius Candidate finite double distance cutoff.
/// @param threshold Candidate working angle in radians.
/// @throws std::invalid_argument If a control is nonfinite or outside its domain.
void controls(double radius,Real threshold) {
  if(!std::isfinite(radius) || !(radius>0) || !std::isfinite(threshold) || !(threshold>0) || threshold>tau)
    throw std::invalid_argument("cone surface: require positive finite radius and angle in (0,2pi]");
}

/// @brief Count complete ordered nonself pairs without forming an overflowing product.
/// @param n Original point count.
/// @return n(n-1), zero for fewer than two points.
/// @throws std::length_error If the count cannot fit size_t.
std::size_t pairs(std::size_t n) {
  if(n<2)return 0;
  if(n-1>std::numeric_limits<std::size_t>::max()/n)throw std::length_error("cone surface: complete pair count overflow");
  return n*(n-1);
}

/// @brief Increment a bounded actual work counter before unsigned wrap.
/// @param value Counter to increment.
/// @throws std::overflow_error If size_t cannot represent the increment.
void count(std::size_t& value) {
  if(value==std::numeric_limits<std::size_t>::max())throw std::overflow_error("cone surface: counter overflow");
  ++value;
}

/// @brief Sort by actual rounded angle, then original input identity.
/// @param a First direction record.
/// @param b Second direction record.
/// @return True if a precedes b under the deterministic key.
bool less_direction(const ConeDirection& a,const ConeDirection& b) {
  return a.angle<b.angle || (a.angle==b.angle && a.index<b.index);
}

/// @brief Test one actual circular gap, retaining only a strict new largest witness.
/// @param point Current owned query summary.
/// @param from Direction at the starting ray.
/// @param to Direction at the ending ray, modulo a full turn for wrap.
/// @param gap Actual rounded candidate, projected to the finite circular interval.
/// @param work Actual gap-test counter to increment.
/// @throws std::overflow_error If the actual gap counter cannot fit size_t.
void consider(ConePointSummary& point,const ConeDirection& from,const ConeDirection& to,Real gap,ConeSurfaceWork& work) {
  count(work.gap_tests);gap=std::clamp(gap,Real{0},tau);
  if(gap>point.largest_gap) {
    point.largest_gap=gap;point.gap_start=from.angle;point.gap_from=from.index;point.gap_to=to.index;
  }
}

/// @brief Execute all ordered pairs once, optionally owning query summaries and sorted directions.
/// @tparam Inspect Whether to retain each actual query summary.
/// @tparam Record Whether to retain each sorted direction vector; implies Inspect.
/// @param pts Borrowed original finite points, validated completely before preflight.
/// @param radius Positive finite inclusive cutoff.
/// @param threshold Valid working angular threshold.
/// @param limits Complete pair and actual trace-direction caps; zero means unlimited.
/// @return Owned mask/work and requested actual records, without replay.
/// @throws std::invalid_argument If the complete input or controls are invalid.
/// @throws std::length_error If complete pair/count, actual direction cap or capacity is exceeded.
/// @throws std::overflow_error If widened arithmetic/counters cannot fit.
/// @throws std::bad_alloc If allocation fails.
template<bool Inspect,bool Record>
ConeSurfaceTrace run(const std::vector<Point2D>& pts,double radius,Real threshold,ConeSurfaceLimits limits) {
  static_assert(!Record || Inspect);
  ConeSurfaceTrace result;auto& analysis=result.analysis;auto& work=analysis.work;
  work.validated_points=validate(pts);controls(radius,threshold);
  const auto planned=pairs(pts.size());
  if(limits.max_pair_visits && planned>limits.max_pair_visits)throw std::length_error("cone surface: complete pair-work cap");
  analysis.surface.assign(pts.size(),0);
  if constexpr(Inspect)analysis.points.reserve(pts.size());
  if constexpr(Record)result.directions.resize(pts.size());
  std::vector<ConeDirection> directions;
  for(std::size_t i=0;i<pts.size();++i) {
    directions.clear();ConePointSummary point;point.index=i;point.point=pts[i];
    point.gap_from=point.gap_to=pts.size();point.largest_gap=tau;
    for(std::size_t j=0;j<pts.size();++j)if(i!=j) {
      count(work.pair_visits);
      const auto dx=finite(Real(pts[j].x)-pts[i].x),dy=finite(Real(pts[j].y)-pts[i].y);
      if(dx==0 && dy==0){++point.coincident;count(work.coincident_visits);continue;}
      count(work.distance_evaluations);const auto distance=finite(std::hypot(dx,dy));
      if(distance<=radius) {
        if constexpr(Record)
          if(limits.max_directions && work.angle_evaluations>=limits.max_directions)
            throw std::length_error("cone surface: actual direction-record cap");
        count(work.angle_evaluations);
        directions.push_back({j,pts[j],distance,finite(std::atan2(dy,dx))});
      }
    }
    point.neighbors=directions.size();point.insufficient=point.neighbors<2;
    if(point.neighbors==1) {
      point.gap_start=directions[0].angle;point.gap_from=point.gap_to=directions[0].index;
    } else if(point.neighbors>=2) {
      count(work.sort_calls);std::sort(directions.begin(),directions.end(),less_direction);point.largest_gap=0;
      for(std::size_t k=1;k<directions.size();++k)
        consider(point,directions[k-1],directions[k],directions[k].angle-directions[k-1].angle,work);
      consider(point,directions.back(),directions.front(),tau-(directions.back().angle-directions.front().angle),work);
    }
    point.surface=point.largest_gap>=threshold;analysis.surface[i]=point.surface?1:0;
    if constexpr(Inspect)analysis.points.push_back(point);
    if constexpr(Record)result.directions[i]=directions;
  }
  return result;
}
} // namespace cone_detail

std::vector<char> cone_surface_points(const std::vector<Point2D>& pts,double radius,double gap_threshold) {
  return cone_detail::run<false,false>(pts,radius,gap_threshold,{0,0}).analysis.surface;
}

ConeSurfaceAnalysis cone_surface_analyze(const std::vector<Point2D>& pts,double radius,long double gap_threshold,ConeSurfaceLimits limits) {
  return cone_detail::run<true,false>(pts,radius,gap_threshold,limits).analysis;
}

ConeSurfaceTrace cone_surface_trace(const std::vector<Point2D>& pts,double radius,long double gap_threshold,ConeSurfaceLimits limits) {
  return cone_detail::run<true,true>(pts,radius,gap_threshold,limits);
}

std::vector<Point2D> sample_disk_candidates(std::size_t trials,double radius,std::uint32_t seed,std::size_t max_trials) {
  if(!std::isfinite(radius) || !(radius>0) || 2*static_cast<long double>(radius)>std::numeric_limits<double>::max() || !max_trials)
    throw std::invalid_argument("disk candidates: invalid radius, interval width or trial cap");
  if(trials>max_trials)throw std::length_error("disk candidates: complete trial cap");
  std::vector<Point2D> points;points.reserve(trials);
  std::mt19937 engine(seed);std::uniform_real_distribution<double> coordinate(-radius,radius);
  for(std::size_t i=0;i<trials;++i) {
    const double x=coordinate(engine),y=coordinate(engine);
    cone_detail::finite(x);cone_detail::finite(y);
    if(cone_detail::finite(std::hypot(static_cast<long double>(x),static_cast<long double>(y)))<=radius)points.push_back({x,y});
  }
  return points;
}

RadialFlagSummary radial_flag_summary(const std::vector<Point2D>& pts,const std::vector<char>& flags,double inner_radius,double outer_radius,Point2D center) {
  (void)cone_detail::validate(pts);
  if(!std::isfinite(center.x) || !std::isfinite(center.y) || !std::isfinite(inner_radius) ||
      !std::isfinite(outer_radius) || inner_radius<0 || outer_radius<inner_radius || flags.size()!=pts.size())
    throw std::invalid_argument("radial flags: invalid centre, cutoffs or mask length");
  for(const auto flag:flags)if(flag!=0 && flag!=1)throw std::invalid_argument("radial flags: mask must contain only zero/one");
  RadialFlagSummary result;result.point_count=pts.size();
  for(std::size_t i=0;i<pts.size();++i) {
    const auto radius=cone_detail::finite(std::hypot(cone_detail::finite(static_cast<long double>(pts[i].x)-center.x),
                                                   cone_detail::finite(static_cast<long double>(pts[i].y)-center.y)));
    const bool inner=radius<inner_radius,outer=radius>outer_radius;
    if(inner)++result.inner_points;if(outer)++result.outer_points;
    if(flags[i]) {
      ++result.flagged_points;if(inner)++result.inner_flagged;if(outer)++result.outer_flagged;
      if(!result.has_flagged){result.min_flagged_radius=result.max_flagged_radius=radius;result.has_flagged=true;}
      else {result.min_flagged_radius=std::min(result.min_flagged_radius,radius);result.max_flagged_radius=std::max(result.max_flagged_radius,radius);}
    }
  }
  return result;
}
} // namespace datamunge::geometry
