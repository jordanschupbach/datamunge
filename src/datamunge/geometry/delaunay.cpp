/// @file
/// @brief Deterministic exact-sign Bowyer-Watson cavities with symbolic hull faces and local analysis.
#include <datamunge/geometry/delaunay.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <stdexcept>
namespace datamunge::geometry {
/// @brief Private original-identity normalization, symbolic boundary handling and checked counts.
namespace {
/// @brief Directed or canonical undirected pair of original ids, with input-size infinity allowed internally.
using Edge=std::pair<std::size_t,std::size_t>;
/// @brief Unique earliest ids and an original-record-to-earliest-id map.
struct Sites {
  std::vector<std::size_t> unique; ///< Earliest original ids in input order.
  std::vector<std::size_t> representative; ///< Earliest matching id for each original record.
};
/// @brief Validate every coordinate before any allocation, cap or trivial verdict.
/// @param p Complete original sites, unchanged.
/// @throws std::invalid_argument Any coordinate is nonfinite.
void validate(const std::vector<Point2D>& p) {
  for(const auto& x:p)if(!std::isfinite(x.x)||!std::isfinite(x.y))throw std::invalid_argument("Delaunay requires finite coordinates");
}
/// @brief Add counts without wraparound.
/// @param a First count.
/// @param b Second count.
/// @return Exact sum.
/// @throws std::length_error Sum exceeds size_t.
std::size_t add(std::size_t a,std::size_t b) {
  if(b>std::numeric_limits<std::size_t>::max()-a)throw std::length_error("Delaunay count overflow");return a+b;
}
/// @brief Multiply counts without wraparound.
/// @param a First count.
/// @param b Second count.
/// @return Exact product.
/// @throws std::length_error Product exceeds size_t.
std::size_t multiply(std::size_t a,std::size_t b) {
  if(a && b>std::numeric_limits<std::size_t>::max()/a)throw std::length_error("Delaunay count overflow");return a*b;
}
/// @brief Check an inclusive zero-disabled optional resource limit.
/// @param count Complete prospective count.
/// @param limit Zero disables the optional limit.
/// @throws std::length_error Count exceeds a positive limit.
void cap(std::size_t count,std::size_t limit) {if(limit && count>limit)throw std::length_error("Delaunay resource limit exceeded");}
/// @brief Coalesce exact numeric coordinates, retaining earliest original ids and representations.
/// @param p Finite original coordinates, unchanged.
/// @return Owned unique ids in input order and a map for every original record.
/// @throws std::length_error Container capacity exceeded.
/// @throws std::bad_alloc Allocation fails.
Sites normalize(const std::vector<Point2D>& p) {
  Sites s;s.representative.reserve(p.size());std::map<std::pair<double,double>,std::size_t> ids;
  for(std::size_t i=0;i<p.size();++i) {
    const auto [it,inserted]=ids.emplace(std::make_pair(p[i].x,p[i].y),i);s.representative.push_back(it->second);if(inserted)s.unique.push_back(i);
  }
  return s;
}
/// @brief Canonical undirected key for a finite or symbolic edge.
/// @param a First endpoint id.
/// @param b Second endpoint id.
/// @return Ascending pair.
Edge key(std::size_t a,std::size_t b) {return std::minmax(a,b);}
/// @brief Rotate an oriented face so its smallest finite id is first, without reversing it.
/// @param t Finite oriented triangle.
/// @return Same cyclic triangle with canonical first id.
Triangle rotate(Triangle t) {
  if(t.b<t.a && t.b<t.c)return {t.b,t.c,t.a};if(t.c<t.a && t.c<t.b)return {t.c,t.a,t.b};return t;
}
/// @brief Check whether one real/ghost face conflicts with a distinct inserted site.
/// @param t Live finite CCW triangle or ghost with c=p.size().
/// @param i Original id of a distinct, not-yet-inserted site.
/// @param p Finite original coordinates.
/// @param w Actual incircle/hull counters, updated once per call.
/// @return Strict circumcircle interior, or ghost's exterior half-plane/between-edge conflict.
bool conflict(const Triangle& t,std::size_t i,const std::vector<Point2D>& p,DelaunayWork& w) {
  if(t.c<p.size()) {w.incircle_tests=add(w.incircle_tests,1);return incircle_sign(p[t.a],p[t.b],p[t.c],p[i])>0;}
  w.hull_side_tests=add(w.hull_side_tests,1);const int side=orientation_sign(p[t.a],p[t.b],p[i]);if(side)return side>0;
  const auto& a=p[t.a];const auto& b=p[t.b];const auto& q=p[i];
  return q.x>=std::min(a.x,b.x)&&q.x<=std::max(a.x,b.x)&&q.y>=std::min(a.y,b.y)&&q.y<=std::max(a.y,b.y)&&q!=a&&q!=b;
}
/// @brief Execute exact-sign insertion once, optionally retaining actual cavity events.
/// @param p Original finite sites, validated inside before all caps.
/// @param record Retain input and actual events when true.
/// @param point_limit Optional original-record ceiling.
/// @param test_limit Optional cumulative complete-scan conflict ceiling.
/// @param entry_limit Optional cumulative retained-event entry ceiling.
/// @return Complete finite mesh and optional owned trace, no replay or finite super-triangle.
/// @throws std::invalid_argument Nonfinite coordinate.
/// @throws std::length_error Arithmetic, optional cap or capacity exceeded.
/// @throws std::logic_error Cavity orientation/multiplicity/completeness invariant fails.
/// @throws std::bad_alloc Allocation fails.
DelaunayTrace solve(const std::vector<Point2D>& p,bool record,std::size_t point_limit,std::size_t test_limit,std::size_t entry_limit) {
  validate(p);cap(p.size(),point_limit);DelaunayTrace out;if(record)out.points=p;
  auto sites=normalize(p);out.representatives=std::move(sites.unique);out.representative_for_input=std::move(sites.representative);
  const auto& ids=out.representatives;if(ids.size()<3)return out;
  std::size_t third=2;int sign=0;
  for(;third<ids.size();++third) {out.work.seed_orientation_tests=add(out.work.seed_orientation_tests,1);sign=orientation_sign(p[ids[0]],p[ids[1]],p[ids[third]]);if(sign)break;}
  if(!sign)return out;
  out.has_seed=true;out.seed={ids[0],ids[1],ids[third]};if(sign<0)std::swap(out.seed.b,out.seed.c);
  const auto seed=out.seed;const auto infinity=p.size();std::vector<Triangle> faces{seed,{seed.b,seed.a,infinity},{seed.c,seed.b,infinity},{seed.a,seed.c,infinity}};
  std::size_t entries=0,total_tests=0;
  for(const auto i:ids) {
    if(i==seed.a||i==seed.b||i==seed.c)continue;
    total_tests=add(total_tests,faces.size());cap(total_tests,test_limit);
    DelaunayInsertion event;event.input_index=i;event.faces_before=faces.size();std::vector<char> bad(faces.size(),0);
    std::map<Edge,std::pair<Edge,unsigned>> edges;
    for(std::size_t j=0;j<faces.size();++j)if(conflict(faces[j],i,p,out.work)) {
      bad[j]=1;event.removed.push_back(faces[j]);const auto t=faces[j];
      for(const auto e:{Edge{t.a,t.b},Edge{t.b,t.c},Edge{t.c,t.a}}) {
        const auto [it,new_edge]=edges.emplace(key(e.first,e.second),std::make_pair(e,1u));
        if(!new_edge) {if(it->second.second!=1 || it->second.first!=Edge{e.second,e.first})throw std::logic_error("Delaunay nonmanifold cavity");++it->second.second;}
      }
    }
    if(event.removed.empty())throw std::logic_error("Delaunay distinct point has no cavity");
    std::vector<Triangle> next;for(std::size_t j=0;j<faces.size();++j)if(!bad[j])next.push_back(faces[j]);
    for(const auto& [unused,value]:edges)if(value.second==1) {
      const auto e=value.first;event.boundary.push_back(e);Triangle t;
      if(e.first==infinity)t={e.second,i,infinity};else if(e.second==infinity)t={i,e.first,infinity};else t={e.first,e.second,i};
      if(t.c!=infinity && orientation_sign(p[t.a],p[t.b],p[t.c])<=0)throw std::logic_error("Delaunay non-CCW new finite face");
      event.added.push_back(t);next.push_back(t);
    }
    event.faces_after=next.size();out.work.insertions=add(out.work.insertions,1);out.work.removed_faces=add(out.work.removed_faces,event.removed.size());
    out.work.boundary_edges=add(out.work.boundary_edges,event.boundary.size());out.work.created_faces=add(out.work.created_faces,event.added.size());
    if(record) {entries=add(entries,add(1,add(event.removed.size(),add(event.boundary.size(),event.added.size()))));cap(entries,entry_limit);out.steps.push_back(std::move(event));}
    faces=std::move(next);
  }
  for(auto t:faces)if(t.c!=infinity)out.triangles.push_back(rotate(t));
  /// @brief Order canonical oriented faces by their original-id triple.
  /// @param a First canonical face.
  /// @param b Second canonical face.
  /// @return Lexicographic a before b.
  const auto less=[](const Triangle& a,const Triangle& b){return std::array{a.a,a.b,a.c}<std::array{b.a,b.b,b.c};};
  std::sort(out.triangles.begin(),out.triangles.end(),less);return out;
}
/// @brief Approximate a finite triangle's minimum angle using power-of-two scaled coordinates.
/// Scaling avoids overflow in coordinate differences/products. Extreme relative
/// scales may underflow a vector and return zero; the value never controls topology.
/// @param a First finite point.
/// @param b Second finite point.
/// @param c Third finite point, exactly noncollinear by prior validation.
/// @return Approximate smallest angle in degrees, possibly zero after range loss.
double minimum_angle(const Point2D& a,const Point2D& b,const Point2D& c) {
  const double scale=std::max({std::abs(a.x),std::abs(a.y),std::abs(b.x),std::abs(b.y),std::abs(c.x),std::abs(c.y)});const int exponent=scale?std::ilogb(scale):0;
  const std::array<Point2D,3> original{a,b,c};std::array<std::array<long double,2>,3> p;
  for(std::size_t i=0;i<3;++i)p[i]={std::scalbn(static_cast<long double>(original[i].x),-exponent),std::scalbn(static_cast<long double>(original[i].y),-exponent)};
  long double minimum=180;
  for(std::size_t i=0;i<3;++i) {
    const auto j=(i+1)%3,k=(i+2)%3;const auto ux=p[j][0]-p[i][0],uy=p[j][1]-p[i][1],vx=p[k][0]-p[i][0],vy=p[k][1]-p[i][1];
    const auto angle=std::atan2(std::abs(ux*vy-uy*vx),ux*vx+uy*vy)*180/std::acos(-1.L);minimum=std::min(minimum,angle);
  }
  return static_cast<double>(minimum);
}
} // namespace
std::vector<Triangle> delaunay_triangulation(const std::vector<Point2D>& points) {return solve(points,false,0,0,0).triangles;}
DelaunayTrace delaunay_trace(const std::vector<Point2D>& points,std::size_t max_points,std::size_t max_tests,std::size_t max_entries) {return solve(points,true,max_points,max_tests,max_entries);}
DelaunayAnalysis delaunay_analyze(const std::vector<Point2D>& points,const std::vector<Triangle>& triangles,std::size_t max_tests) {
  validate(points);const auto sites=normalize(points);DelaunayAnalysis out;out.distinct_points=sites.unique.size();out.triangles=triangles.size();out.minimum_angle_degrees=std::numeric_limits<double>::quiet_NaN();
  std::set<std::array<std::size_t,3>> seen;std::map<Edge,unsigned> edges;std::set<std::size_t> used;std::vector<Triangle> canonical;canonical.reserve(triangles.size());
  for(const auto t:triangles) {
    if(t.a>=points.size()||t.b>=points.size()||t.c>=points.size())throw std::invalid_argument("Delaunay analysis index out of bounds");
    if(orientation_sign(points[t.a],points[t.b],points[t.c])<=0)throw std::invalid_argument("Delaunay analysis requires nondegenerate CCW faces");
    const auto a=sites.representative[t.a],b=sites.representative[t.b],c=sites.representative[t.c];const auto face=rotate({a,b,c});
    if(!seen.insert({face.a,face.b,face.c}).second)throw std::invalid_argument("Delaunay analysis duplicate face");canonical.push_back(face);
    for(const auto e:{key(a,b),key(b,c),key(c,a)})if(++edges[e]>2)throw std::invalid_argument("Delaunay analysis nonmanifold edge");
    used.insert(a);used.insert(b);used.insert(c);const auto angle=minimum_angle(points[a],points[b],points[c]);if(std::isnan(out.minimum_angle_degrees)||angle<out.minimum_angle_degrees)out.minimum_angle_degrees=angle;
  }
  out.used_points=used.size();out.edges=edges.size();for(const auto& [e,count]:edges)if(count==1)++out.boundary_edges;
  const auto tests=triangles.empty()?0:multiply(triangles.size(),sites.unique.size()-3);cap(tests,max_tests);
  for(const auto t:canonical)for(auto i:sites.unique)if(i!=t.a&&i!=t.b&&i!=t.c) {out.circle_tests=add(out.circle_tests,1);out.interior_violations=add(out.interior_violations,incircle_sign(points[t.a],points[t.b],points[t.c],points[i])>0);}
  return out;
}
std::vector<Point2D> delaunay_uniform_cloud(std::size_t count,std::uint32_t seed,double extent,std::size_t max_points) {
  if(!std::isfinite(extent)||extent<=0)throw std::invalid_argument("Delaunay cloud extent must be finite and positive");cap(count,max_points);
  std::vector<Point2D> points;points.reserve(count);std::mt19937 rng(seed);std::uniform_real_distribution<double> coordinate(0,extent);
  for(std::size_t i=0;i<count;++i)points.push_back({coordinate(rng),coordinate(rng)});return points;
}
} // namespace datamunge::geometry
