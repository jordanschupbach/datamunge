/// @file clarke_wright.cpp
/// @brief One checked parallel savings kernel with optional actual-state recording.
#include <datamunge/algorithms/clarke_wright.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Private checked geometry, validation, complete forecasts and executing routing kernel.
namespace clarke_wright_detail {
/// @brief Widened route length, saving and demand accumulation type.
using Real=long double;
/// @brief Borrowed planar input coordinate record.
using Point=std::pair<double,double>;
/// @brief Check one widened arithmetic result.
/// @param value Computed candidate.
/// @return Finite value unchanged.
/// @throws std::overflow_error If value is nonfinite.
Real checked(Real value){if(!std::isfinite(value))throw std::overflow_error("clarke_wright: nonfinite arithmetic");return value;}
/// @brief Increment one actual operation counter without wraparound.
/// @param value Mutable actual counter.
/// @param amount Actual increment, default one.
/// @throws std::overflow_error If size_t cannot hold the incremented value.
void count(std::size_t& value,std::size_t amount=1){if(amount>std::numeric_limits<std::size_t>::max()-value)throw std::overflow_error("clarke_wright: counter overflow");value+=amount;}
/// @brief Check one complete logical work product before allocation/execution.
/// @param a First nonnegative count.
/// @param b Second nonnegative count.
/// @return Exact size_t product.
/// @throws std::length_error If the supported logical count is exceeded.
std::size_t product(std::size_t a,std::size_t b){if(b && a>std::numeric_limits<std::size_t>::max()/b)throw std::length_error("clarke_wright: logical work overflow");return a*b;}
/// @brief Compute supported unordered customer pair count without intermediate overflow.
/// @param m Customer count.
/// @return m(m-1)/2, zero for fewer than two customers.
/// @throws std::length_error If pair count cannot fit size_t.
std::size_t pairs(std::size_t m){if(m<2)return 0;auto a=m,b=m-1;if(a%2==0)a/=2;else b/=2;return product(a,b);}
/// @brief Validate all original coordinates and demands before caps/allocations.
/// @param p Original point array.
/// @param demand Original matching demand array.
/// @param capacity Positive finite common capacity.
/// @param depot In-range depot id; its finite nonnegative demand is ignored.
/// @return Actual complete point-validation visits.
/// @throws std::invalid_argument If shape, domain or customer capacity is invalid.
/// @throws std::overflow_error If actual visits cannot fit size_t.
std::size_t validate(const std::vector<Point>& p,const std::vector<double>& demand,double capacity,std::size_t depot) {
  if(p.size()!=demand.size() || depot>=p.size() || !std::isfinite(capacity) || capacity<=0)throw std::invalid_argument("clarke_wright: invalid shape, depot or capacity");
  std::size_t visits=0;
  for(std::size_t i=0;i<p.size();++i){if(!std::isfinite(p[i].first) || !std::isfinite(p[i].second) || !std::isfinite(demand[i]) || demand[i]<0 || (i!=depot && demand[i]>capacity))throw std::invalid_argument("clarke_wright: invalid coordinate or demand");count(visits);}
  return visits;
}
/// @brief Evaluate actual widened Euclidean geometry and increment its named counter.
/// @param a First validated point.
/// @param b Second validated point.
/// @param work Mutable actual named work.
/// @return Checked finite nonnegative distance.
/// @throws std::overflow_error If widened geometry or the counter overflows.
Real distance(Point a,Point b,ClarkeWrightWork& work){count(work.distance_evaluations);return checked(std::hypot(checked(Real(a.first)-b.first),checked(Real(a.second)-b.second)));}
/// @brief One actual sorted customer-pair saving.
struct Saving {
  std::size_t i; ///< Lower original customer id.
  std::size_t j; ///< Higher original customer id.
  Real value; ///< Actual widened saving, including zero and any tiny rounded negative value.
};
/// @brief Execute endpoint-scanning parallel savings with optional actual decision capture.
/// @tparam Record Whether decisions and accepted customer orders are retained.
/// @param p Original borrowed coordinates.
/// @param demand Original borrowed demands.
/// @param capacity Finite positive common capacity.
/// @param depot Original depot id.
/// @param options Complete work/storage limits.
/// @return Actual result, measured accounting/work and optional decisions.
/// @throws std::invalid_argument If any original input is invalid.
/// @throws std::length_error If complete work caps or supported count/storage capacity is exceeded.
/// @throws std::overflow_error If arithmetic, counters or final double length cannot fit.
/// @throws std::bad_alloc If allocation fails.
template<bool Record>
ClarkeWrightTrace run(const std::vector<Point>& p,const std::vector<double>& demand,double capacity,std::size_t depot,ClarkeWrightOptions options) {
  const auto validated=validate(p,demand,capacity,depot),m=p.size()-1,candidate_count=pairs(m),endpoint_count=product(candidate_count,m);
  if((options.max_savings_pairs && candidate_count>options.max_savings_pairs) || (options.max_endpoint_visits && endpoint_count>options.max_endpoint_visits))throw std::length_error("clarke_wright: complete work cap exceeded");
  if constexpr(Record)if(options.max_decisions && candidate_count>options.max_decisions)throw std::length_error("clarke_wright: complete record cap exceeded");
  ClarkeWrightTrace out;out.work.validated_points=validated;
  std::vector<std::vector<std::size_t>> routes;routes.reserve(m);std::vector<Real> loads;loads.reserve(m);std::vector<Real> radial(p.size());
  for(std::size_t i=0;i<p.size();++i)if(i!=depot){routes.push_back({i});loads.push_back(demand[i]);radial[i]=distance(p[depot],p[i],out.work);out.initial_star_distance=checked(out.initial_star_distance+checked(2*radial[i]));}
  std::vector<Saving> savings;savings.reserve(candidate_count);
  for(std::size_t i=0;i<p.size();++i)if(i!=depot)for(std::size_t j=i+1;j<p.size();++j)if(j!=depot){savings.push_back({i,j,checked(checked(radial[i]+radial[j])-distance(p[i],p[j],out.work))});count(out.work.savings_pairs);}
  /// @brief Actual comparison used by the executing sort; ties are ascending original pair ids.
  /// @param a First constructed finite saving.
  /// @param b Second constructed finite saving.
  /// @return Whether a precedes b in descending-saving/ascending-id order.
  /// @throws std::overflow_error If the actual comparison counter cannot fit.
  auto earlier=[&out](const Saving& a,const Saving& b){count(out.work.sort_comparisons);if(a.value!=b.value)return a.value>b.value;return a.i!=b.i?a.i<b.i:a.j<b.j;};
  std::sort(savings.begin(),savings.end(),earlier);
  if constexpr(Record)out.steps.reserve(candidate_count);
  for(const auto candidate:savings) {
    count(out.work.decisions);ClarkeWrightStep step;step.i=candidate.i;step.j=candidate.j;step.saving=candidate.value;
    for(std::size_t slot=0;slot<m;++slot){count(out.work.endpoint_slot_visits);if(routes[slot].empty())continue;if(routes[slot].front()==candidate.i || routes[slot].back()==candidate.i)step.route_i=slot;if(routes[slot].front()==candidate.j || routes[slot].back()==candidate.j)step.route_j=slot;}
    if(step.route_i)step.load_i=loads[*step.route_i];if(step.route_j)step.load_j=loads[*step.route_j];
    if(!step.route_i || !step.route_j)step.decision=ClarkeWrightDecision::MissingEndpoint;
    else if(step.route_i==step.route_j)step.decision=ClarkeWrightDecision::SameRoute;
    else if(checked(step.load_i+step.load_j)>Real(capacity))step.decision=ClarkeWrightDecision::CapacityExceeded;
    else {
      const auto a=*step.route_i,b=*step.route_j;step.decision=ClarkeWrightDecision::Merged;
      if(routes[a].front()==candidate.i){std::reverse(routes[a].begin(),routes[a].end());step.reversed_i=true;count(out.work.route_reversals);}
      if(routes[b].back()==candidate.j){std::reverse(routes[b].begin(),routes[b].end());step.reversed_j=true;count(out.work.route_reversals);}
      count(out.work.appended_customers,routes[b].size());routes[a].insert(routes[a].end(),routes[b].begin(),routes[b].end());loads[a]=checked(loads[a]+loads[b]);routes[b].clear();count(out.work.merges);out.accepted_savings=checked(out.accepted_savings+candidate.value);
      if constexpr(Record)step.merged_customers=routes[a];
    }
    if constexpr(Record)out.steps.push_back(std::move(step));
  }
  for(std::size_t slot=0;slot<m;++slot)if(!routes[slot].empty()) {
    auto& route=routes[slot];route.insert(route.begin(),depot);route.push_back(depot);
    for(std::size_t i=1;i<route.size();++i){out.measured_distance=checked(out.measured_distance+distance(p[route[i-1]],p[route[i]],out.work));count(out.work.output_edges);}
    if constexpr(Record)out.route_loads.push_back(loads[slot]);out.result.routes.push_back(std::move(route));
  }
  out.accounting_residual=checked(out.measured_distance-checked(out.initial_star_distance-out.accepted_savings));
  if(out.measured_distance>std::numeric_limits<double>::max())throw std::overflow_error("clarke_wright: final distance exceeds double range");
  out.result.total_distance=static_cast<double>(out.measured_distance);return out;
}
}
VehicleRoutes clarke_wright_savings(const std::vector<std::pair<double,double>>& points,const std::vector<double>& demand,double capacity,std::size_t depot){return clarke_wright_detail::run<false>(points,demand,capacity,depot,{}).result;}
VehicleRoutes clarke_wright_savings_with_options(const std::vector<std::pair<double,double>>& points,const std::vector<double>& demand,double capacity,std::size_t depot,ClarkeWrightOptions options){return clarke_wright_detail::run<false>(points,demand,capacity,depot,options).result;}
ClarkeWrightTrace clarke_wright_trace(const std::vector<std::pair<double,double>>& points,const std::vector<double>& demand,double capacity,std::size_t depot,ClarkeWrightOptions options){return clarke_wright_detail::run<true>(points,demand,capacity,depot,options);}
VehicleRouteEvaluation evaluate_vehicle_routes(const std::vector<std::pair<double,double>>& points,const std::vector<double>& demand,double capacity,const std::vector<std::vector<std::size_t>>& routes,std::size_t depot) {
  using namespace clarke_wright_detail;
  (void)validate(points,demand,capacity,depot);
  for(const auto& route:routes)for(auto vertex:route)if(vertex>=points.size())throw std::invalid_argument("vehicle routes: invalid vertex");
  VehicleRouteEvaluation out;std::vector<std::size_t> seen(points.size());ClarkeWrightWork work;
  for(const auto& route:routes) {
    if(route.size()<3 || route.front()!=depot || route.back()!=depot)count(out.malformed_routes);
    Real load=0;
    for(std::size_t i=0;i<route.size();++i) {
      const auto v=route[i];
      if(v==depot){if(i>0 && i+1<route.size())count(out.interior_depot_visits);}
      else{if(seen[v])count(out.repeated_visits);count(seen[v]);load=checked(load+demand[v]);}
      if(i)out.total_distance=checked(out.total_distance+distance(points[route[i-1]],points[v],work));
    }
    out.route_loads.push_back(load);if(load>capacity)count(out.overloaded_routes);
  }
  for(std::size_t v=0;v<points.size();++v)if(v!=depot && !seen[v])count(out.missing_customers);
  out.feasible=!out.malformed_routes && !out.interior_depot_visits && !out.missing_customers && !out.repeated_visits && !out.overloaded_routes;return out;
}
}
