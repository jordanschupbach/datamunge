/// @file circle_packing.cpp
/// @brief Checked dense pair relaxation, actual raw snapshots and independent disk-contact measurements.
#include <datamunge/algorithms/circle_packing.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Private validation, dense adjacency, numerical checks and one executing layout kernel.
namespace circle_packing_detail {
/// @brief Canonical unordered graph input edge type.
using Edge=std::pair<std::size_t,std::size_t>;
/// @brief Working distance, displacement and diagnostic accumulation type.
using Real=long double;
/// @brief Legacy initial-angle constant in double precision.
constexpr double pi=3.14159265358979323846;
/// @brief Legacy lower distance used by pair forces, including their target difference.
constexpr Real distance_floor=1e-9L;
/// @brief Legacy normalized-output radius, reset rather than transformed with centres.
constexpr double output_radius=0.45;
/// @brief Check one widened arithmetic result.
/// @param value Candidate finite result.
/// @return value unchanged.
/// @throws std::overflow_error If value is nonfinite.
Real checked(Real value){if(!std::isfinite(value))throw std::overflow_error("circle layout: nonfinite working arithmetic");return value;}
/// @brief Convert a finite working coordinate to supported double storage.
/// @param value Widened coordinate.
/// @return Finite double coordinate.
/// @throws std::overflow_error If value cannot fit finite double.
double narrow(Real value){checked(value);if(std::fabs(value)>std::numeric_limits<double>::max())throw std::overflow_error("circle layout: coordinate exceeds double range");return static_cast<double>(value);}
/// @brief Increment actual work without wrapping.
/// @param value Mutable actual count.
/// @param amount Actual increment, default one.
/// @throws std::overflow_error If size_t cannot represent the incremented count.
void count(std::size_t& value,std::size_t amount=1){if(amount>std::numeric_limits<std::size_t>::max()-value)throw std::overflow_error("circle layout: work counter overflow");value+=amount;}
/// @brief Check a complete logical-work/storage product before allocation or execution.
/// @param a First nonnegative dimension.
/// @param b Second nonnegative dimension.
/// @return Exact size_t product.
/// @throws std::length_error If the complete product exceeds size_t.
std::size_t product_size(std::size_t a,std::size_t b){if(b && a>std::numeric_limits<std::size_t>::max()/b)throw std::length_error("circle layout: complete dimension exceeds size_t");return a*b;}
/// @brief Compute the supported unordered-pair count without overflowing an unreduced product.
/// @param n Vertex count.
/// @return n(n-1)/2, zero for fewer than two vertices.
/// @throws std::length_error If the pair count cannot fit size_t.
std::size_t pairs(std::size_t n){if(n<2)return 0;auto a=n,b=n-1;if(a%2==0)a/=2;else b/=2;return product_size(a,b);}
/// @brief Validate every original endpoint before allocation or any positive cap.
/// @param n Original vertex count.
/// @param edges Borrowed original graph inputs.
/// @return Actual endpoint-validation visits.
/// @throws std::invalid_argument If an endpoint is out of range.
/// @throws std::overflow_error If the visit counter cannot fit size_t.
std::size_t validate_edges(std::size_t n,const std::vector<Edge>& edges){std::size_t visits=0;for(const auto [u,v]:edges){if(u>=n || v>=n)throw std::invalid_argument("circle layout: endpoint out of range");count(visits);}return visits;}
/// @brief Dense unweighted adjacency and actual input-collapsing counts.
struct Graph {
  std::vector<std::vector<bool>> adjacent; ///< Symmetric nonloop adjacency bits.
  std::size_t unique{0}; ///< Actual distinct unordered nonloop edges installed.
  std::size_t loops{0}; ///< Actual loop inputs ignored.
  std::size_t duplicates{0}; ///< Actual repeated/reversed nonloop inputs collapsed.
};
/// @brief Construct dense adjacency only after full endpoint/work validation.
/// @param n Validated vertex count.
/// @param edges Fully validated original input order.
/// @return Owned symmetric adjacency and actual collapsing counts.
/// @throws std::length_error If matrix dimensions/container capacity exceed supported storage.
/// @throws std::overflow_error If actual input-category counts cannot fit.
/// @throws std::bad_alloc If allocation fails.
Graph graph(std::size_t n,const std::vector<Edge>& edges) {
  (void)product_size(n,n);Graph out;out.adjacent.assign(n,std::vector<bool>(n));
  for(const auto [u,v]:edges){if(u==v){count(out.loops);continue;}if(out.adjacent[u][v])count(out.duplicates);else{out.adjacent[u][v]=out.adjacent[v][u]=true;count(out.unique);}}
  return out;
}
/// @brief Measure true centre distance in widened arithmetic, independently of force regularization.
/// @param a First finite centre.
/// @param b Second finite centre.
/// @return Finite nonnegative widened Euclidean distance.
/// @throws std::overflow_error If a working difference/distance is nonfinite.
Real distance(Point2D a,Point2D b){return checked(std::hypot(checked(Real(a.x)-b.x),checked(Real(a.y)-b.y)));}
/// @brief Measure complete disk contacts using an already validated graph and disk data.
/// @param circles Finite nonnegative-radius disk data.
/// @param g Matching validated dense adjacency.
/// @param tolerance Finite nonnegative absolute gap tolerance.
/// @return Actual tolerance classifications and checked aggregates.
/// @throws std::overflow_error If a measurement or actual counter cannot fit.
CirclePackingMetrics inspect(const std::vector<Circle2D>& circles,const Graph& g,Real tolerance) {
  CirclePackingMetrics out;out.circle_count=circles.size();out.unique_edges=g.unique;
  for(const auto& c:circles)if(c.radius==0)count(out.zero_radii);
  for(std::size_t u=0;u<circles.size();++u)for(std::size_t v=u+1;v<circles.size();++v) {
    count(out.pair_visits);const auto gap=checked(distance(circles[u].center,circles[v].center)-checked(Real(circles[u].radius)+circles[v].radius));
    out.min_gap=out.pair_visits==1?gap:std::min(out.min_gap,gap);
    if(gap<-tolerance)count(out.overlapping_pairs);
    if(gap<0)out.overlap_penalty=checked(out.overlap_penalty+checked(gap*gap));
    if(g.adjacent[u][v]) {
      out.max_edge_error=std::max(out.max_edge_error,std::fabs(gap));out.edge_stress=checked(out.edge_stress+checked(gap*gap));
      if(std::fabs(gap)<=tolerance)count(out.edge_contacts);
    }else if(std::fabs(gap)<=tolerance)count(out.nonedge_contacts);
  }
  return out;
}
/// @brief Copy an actual raw state and separately measure its stated pair potential.
/// @param circles Current actual unit-radius working disks.
/// @param g Current fixed adjacency.
/// @param iteration Actual completed sweep count.
/// @param displacement Maximum actual movement in the last sweep, zero initially.
/// @param visits Actual raw diagnostic pair counter.
/// @return Owned state and measured edge/nonedge potential.
/// @throws std::overflow_error If diagnostic arithmetic/counts overflow.
/// @throws std::length_error If disk copy capacity is exceeded.
/// @throws std::bad_alloc If copying fails.
CirclePackingFrame frame(const std::vector<Circle2D>& circles,const Graph& g,std::size_t iteration,Real displacement,std::size_t& visits) {
  CirclePackingFrame out;out.iteration=iteration;out.circles=circles;out.max_displacement=displacement;
  for(std::size_t u=0;u<circles.size();++u)for(std::size_t v=u+1;v<circles.size();++v) {
    count(visits);const auto d=distance(circles[u].center,circles[v].center);
    if(g.adjacent[u][v]){const auto gap=d-2;out.edge_stress=checked(out.edge_stress+checked(gap*gap));}
    else if(d<2.15L){const auto deficit=2.15L-d;out.nonedge_penalty=checked(out.nonedge_penalty+checked(deficit*deficit));}
  }
  out.energy=checked(checked(out.edge_stress+out.nonedge_penalty)/2);return out;
}
/// @brief Widened Jacobi displacement buffer; vertices are committed only after the pair scan.
struct Delta {
  Real x{0}; ///< Accumulated horizontal displacement in current raw units.
  Real y{0}; ///< Accumulated vertical displacement in current raw units.
};
/// @brief Execute the same finite-sweep heuristic with optional actual-state recording.
/// @tparam Record Whether raw snapshots and returned-disk diagnostics are retained.
/// @param n Original vertex count.
/// @param edges Borrowed original undirected input.
/// @param options Finite fixed controls and complete work/snapshot caps.
/// @return Actual normalized output, map/work and optional raw records.
/// @throws std::invalid_argument If any endpoint, step or positive stride is invalid.
/// @throws std::length_error If complete caps/counts/storage capacities are exceeded.
/// @throws std::overflow_error If arithmetic or actual counters cannot fit.
/// @throws std::bad_alloc If allocation fails.
template<bool Record>
CirclePackingTrace run(std::size_t n,const std::vector<Edge>& edges,CirclePackingOptions options) {
  const auto validated=validate_edges(n,edges);
  if(!std::isfinite(options.step_size) || options.step_size<0 || !options.snapshot_stride)throw std::invalid_argument("circle layout: invalid fixed step or zero snapshot stride");
  const auto pair_count=pairs(n),sweeps=n<2?0:options.iterations;const auto planned_visits=product_size(pair_count,sweeps);(void)product_size(n,sweeps);
  if(options.max_pair_visits && planned_visits>options.max_pair_visits)throw std::length_error("circle layout: complete pair-work cap exceeded");
  std::size_t snapshots=0;
  if constexpr(Record) {
    snapshots=sweeps/options.snapshot_stride;
    const auto extra=std::size_t{1}+(sweeps%options.snapshot_stride!=0);
    if(snapshots>std::numeric_limits<std::size_t>::max()-extra)throw std::length_error("circle layout: complete frame count overflow");
    snapshots+=extra;(void)product_size(pair_count,snapshots);
    if(options.max_snapshots && snapshots>options.max_snapshots)throw std::length_error("circle layout: complete snapshot cap exceeded");
  }
  auto g=graph(n,edges);CirclePackingTrace out;out.options=options;out.work.validated_edges=validated;out.work.unique_edges=g.unique;out.work.ignored_self_loops=g.loops;out.work.collapsed_duplicates=g.duplicates;
  std::vector<Circle2D> circles(n);
  for(std::size_t i=0;i<n;++i){const auto angle=2*pi*static_cast<double>(i)/static_cast<double>(n);circles[i].center={3*std::cos(angle),3*std::sin(angle)};}
  if constexpr(Record)out.frames.push_back(frame(circles,g,0,0,out.work.frame_pair_visits));
  std::vector<Delta> delta(n);
  for(std::size_t step=0;step<sweeps;++step) {
    std::fill(delta.begin(),delta.end(),Delta{});
    for(std::size_t u=0;u<n;++u)for(std::size_t v=u+1;v<n;++v) {
      count(out.work.pair_visits);const auto dx=checked(Real(circles[v].center.x)-circles[u].center.x),dy=checked(Real(circles[v].center.y)-circles[u].center.y);
      const auto d=std::max(distance_floor,checked(std::hypot(dx,dy)));const bool adjacent=g.adjacent[u][v];
      const Real target=adjacent?2.0L:2.15L;
      const Real force=checked((d-target)*(adjacent || d<target?Real(options.step_size):0.0L));
      const auto x=checked(dx/d*force),y=checked(dy/d*force);
      delta[u].x=checked(delta[u].x+x);delta[u].y=checked(delta[u].y+y);delta[v].x=checked(delta[v].x-x);delta[v].y=checked(delta[v].y-y);
    }
    Real max_displacement=0;
    for(std::size_t i=0;i<n;++i) {
      const auto before=circles[i].center;const Point2D after{narrow(Real(before.x)+delta[i].x),narrow(Real(before.y)+delta[i].y)};
      max_displacement=std::max(max_displacement,distance(before,after));circles[i].center=after;count(out.work.position_updates);
    }
    count(out.work.completed_iterations);
    if constexpr(Record)if((step+1)%options.snapshot_stride==0 || step+1==sweeps)out.frames.push_back(frame(circles,g,step+1,max_displacement,out.work.frame_pair_visits));
  }
  if(n) {
    Real x=0,y=0;for(const auto& c:circles){x=checked(x+c.center.x);y=checked(y+c.center.y);}
    out.normalization_center={narrow(x/n),narrow(y/n)};Real extent=0;
    for(const auto& c:circles)extent=std::max(extent,distance(c.center,out.normalization_center));out.normalization_scale=extent>0?extent:1;
    for(auto& c:circles){c.center={narrow((Real(c.center.x)-out.normalization_center.x)/out.normalization_scale),narrow((Real(c.center.y)-out.normalization_center.y)/out.normalization_scale)};c.radius=output_radius;}
  }
  out.circles=std::move(circles);
  if constexpr(Record){out.output_metrics=inspect(out.circles,g,1e-9L);out.work.output_pair_visits=out.output_metrics.pair_visits;}
  return out;
}
} // namespace circle_packing_detail
CirclePackingMetrics circle_packing_metrics(const std::vector<Circle2D>& circles,const std::vector<std::pair<std::size_t,std::size_t>>& edges,double tolerance) {
  if(!std::isfinite(tolerance) || tolerance<0)throw std::invalid_argument("circle metrics: invalid tolerance");
  for(const auto& c:circles)if(!std::isfinite(c.center.x) || !std::isfinite(c.center.y) || !std::isfinite(c.radius) || c.radius<0)throw std::invalid_argument("circle metrics: invalid disk");
  (void)circle_packing_detail::validate_edges(circles.size(),edges);(void)circle_packing_detail::pairs(circles.size());
  const auto g=circle_packing_detail::graph(circles.size(),edges);return circle_packing_detail::inspect(circles,g,tolerance);
}
std::vector<Circle2D> circle_packing_layout(std::size_t vertex_count,const std::vector<std::pair<std::size_t,std::size_t>>& edges,std::size_t iterations) {
  CirclePackingOptions options;options.iterations=iterations;return circle_packing_detail::run<false>(vertex_count,edges,options).circles;
}
std::vector<Circle2D> circle_packing_layout_with_options(std::size_t vertex_count,const std::vector<std::pair<std::size_t,std::size_t>>& edges,CirclePackingOptions options) {
  return circle_packing_detail::run<false>(vertex_count,edges,options).circles;
}
CirclePackingTrace circle_packing_trace(std::size_t vertex_count,const std::vector<std::pair<std::size_t,std::size_t>>& edges,CirclePackingOptions options) {
  return circle_packing_detail::run<true>(vertex_count,edges,options);
}
} // namespace datamunge::algorithms
