#pragma once
/// @file clarke_wright.hpp
/// @brief Checked Euclidean parallel savings routing with actual merge decisions and work limits.
#include <cstddef>
#include <optional>
#include <utility>
#include <vector>
namespace datamunge::algorithms {
/// @brief Owned depot-closed vehicle routes and their measured total Euclidean length.
/// Each nonempty generated route begins and ends at the chosen depot, with every
/// customer served exactly once. Depot-only input returns no routes and zero
/// distance. This plain record validates nothing and certifies no optimality.
/// The original field order and types are retained. Values own their storage.
/// @par Example
/// @code
/// const auto r=dm::algorithms::clarke_wright_savings({{0,0},{1,0},{2,0}},{0,1,1},2);
/// std::print("{} {}\n",r.routes.size(),r.total_distance); // 1 4
/// @endcode
/// @see clarke_wright_savings(), ClarkeWrightTrace
struct VehicleRoutes {
  std::vector<std::vector<std::size_t>> routes; ///< Original vertex ids, each route closed by its depot at both ends.
  double total_distance{0}; ///< Checked finite double sum along returned routes; not an optimal value certificate.
};
/// @brief Owned independent route-feasibility and widened distance measurements.
/// Every nondepot occurrence contributes demand, including repeated customers or
/// customer endpoints of malformed routes. repeated_visits counts extra visits,
/// not the number of multiply visited customers. Missing customers count once.
/// A malformed route has fewer than three vertices or wrong depot endpoints.
/// Interior depot visits are reported separately. No capacity tolerance applies.
/// This record certifies no optimality and validates no manually assigned fields.
/// @par Example
/// @code
/// const auto m=dm::algorithms::evaluate_vehicle_routes({{0,0},{1,0},{2,0}},{0,1,1},2,{{0,1,2,0}});
/// std::print("{} {}\n",m.feasible,m.total_distance); // true 4
/// @endcode
/// @see evaluate_vehicle_routes(), ClarkeWrightTrace
struct VehicleRouteEvaluation {
  bool feasible{false}; ///< True only when all closure, coverage, uniqueness and capacity checks pass.
  std::size_t malformed_routes{0}; ///< Routes shorter than three vertices or lacking the specified depot endpoints.
  std::size_t interior_depot_visits{0}; ///< Depot occurrences strictly between a route's first and last positions.
  std::size_t missing_customers{0}; ///< Original nondepot vertices with no occurrence anywhere.
  std::size_t repeated_visits{0}; ///< Nondepot occurrences beyond each customer's first occurrence.
  std::size_t overloaded_routes{0}; ///< Routes whose actual widened customer-demand sum exceeds capacity.
  std::vector<long double> route_loads; ///< Actual demand sums aligned with supplied routes, including malformed ones.
  long double total_distance{0}; ///< Checked widened sum over every supplied consecutive route edge.
};
/// @brief Independently measure supplied routes' closure, coverage, capacity and Euclidean distance.
/// Validates the full point/demand/capacity input and every route vertex before
/// measuring. Invalid vertex ids throw; structural/coverage/capacity defects are
/// returned as counts with feasible=false. Empty route collections are feasible
/// only for a depot-only problem. An empty route itself is malformed. Loads and
/// geometry follow clarke_wright_savings(); final length remains long double.
/// O(points+route vertices) work and O(points+routes) auxiliary storage; no input
/// mutation/shared state, and the returned record owns its measurements.
/// @param points Borrowed finite planar coordinates, including the depot.
/// @param demand Matching finite nonnegative demands; depot value validated but ignored.
/// @param capacity Finite positive common capacity, each customer demand at most this value.
/// @param routes Borrowed proposed original-vertex route sequences, including depot boundaries.
/// @param depot Valid original depot id, default zero.
/// @return Owned independent feasibility counts, loads and checked widened edge sum.
/// @throws std::invalid_argument If original input or any supplied route vertex is invalid.
/// @throws std::length_error If supported container capacity is exceeded.
/// @throws std::overflow_error If numeric arithmetic or actual counts overflow.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// const auto m=dm::algorithms::evaluate_vehicle_routes({{0,0},{1,0}},{0,1},1,{});
/// std::print("{} {}\n",m.feasible,m.missing_customers); // false 1
/// @endcode
/// @see VehicleRouteEvaluation, clarke_wright_savings()
VehicleRouteEvaluation evaluate_vehicle_routes(const std::vector<std::pair<double,double>>& points,const std::vector<double>& demand,double capacity,const std::vector<std::vector<std::size_t>>& routes,std::size_t depot=0);
/// @brief Complete-call preflight limits for savings storage, endpoint scans and optional recording.
/// With m customers there are C=m(m-1)/2 candidate pairs and exactly C*m route-slot
/// inspections. All input validation precedes forecast/cap checks, which precede
/// route/candidate allocation. Zero means unlimited. max_decisions applies only
/// to traced calls; no partial routes or trace escape a failure. These logical
/// counts exclude sorting comparisons, geometric arithmetic and allocator work.
/// @par Example
/// @code
/// dm::algorithms::ClarkeWrightOptions o;o.max_savings_pairs=1;o.max_endpoint_visits=2;
/// const auto t=dm::algorithms::clarke_wright_trace({{0,0},{1,0},{2,0}},{0,1,1},2,0,o);
/// std::print("{}\n",t.work.endpoint_slot_visits); // 2
/// @endcode
/// @see clarke_wright_trace(), clarke_wright_savings_with_options()
struct ClarkeWrightOptions {
  std::size_t max_savings_pairs{0}; ///< Positive complete unordered customer-pair cap; zero unlimited.
  std::size_t max_endpoint_visits{0}; ///< Positive complete pair-count times customer-count cap; zero unlimited.
  std::size_t max_decisions{0}; ///< Positive complete trace-record count cap, tracing only; zero unlimited.
};
/// @brief Actual ordered reason for accepting or rejecting a sorted savings pair.
/// Missing endpoint is checked first, then same route, then capacity. A customer
/// in a route interior has no endpoint slot. Merged records include the committed
/// customer order; rejected records never mutate routes. This is not a proof that
/// another global routing could not use the rejected edge.
/// @par Example
/// @code
/// const auto t=dm::algorithms::clarke_wright_trace({{0,0},{1,0},{2,0}},{0,1,1},2);
/// std::print("{}\n",t.steps[0].decision==dm::algorithms::ClarkeWrightDecision::Merged); // true
/// @endcode
/// @see ClarkeWrightStep
enum class ClarkeWrightDecision {
  Merged, ///< Distinct endpoint routes fit capacity and were concatenated.
  MissingEndpoint, ///< At least one customer was interior to its current route.
  SameRoute, ///< Both customer endpoints already belonged to the same route.
  CapacityExceeded ///< The widened sum of distinct route loads exceeded capacity.
};
/// @brief Owned actual sorted-pair decision, endpoint lookup and accepted concatenation.
/// Stable route slots refer to the initial ascending customer order, excluding
/// the depot. A merge retains route_i and clears route_j. Missing optional slots
/// mean an interior customer. Loads are meaningful only for present slots.
/// Reversal flags indicate executed reverse calls, even on singleton routes.
/// merged_customers excludes depot visits and is empty on rejection. No replay.
/// @par Example
/// @code
/// const auto s=dm::algorithms::clarke_wright_trace({{0,0},{1,0},{2,0}},{0,1,1},2).steps[0];
/// std::print("{} {} {}\n",s.i,s.j,s.merged_customers.size()); // 1 2 2
/// @endcode
/// @see ClarkeWrightTrace, ClarkeWrightDecision
struct ClarkeWrightStep {
  std::size_t i{0}; ///< Lower original customer id in this sorted pair.
  std::size_t j{0}; ///< Higher original customer id in this sorted pair.
  long double saving{0}; ///< Actual widened depot-edge saving used for sorting.
  std::optional<std::size_t> route_i; ///< Actual pre-decision endpoint slot of i, absent for an interior customer.
  std::optional<std::size_t> route_j; ///< Actual pre-decision endpoint slot of j, absent for an interior customer.
  long double load_i{0}; ///< Actual route_i load before the decision; zero when absent.
  long double load_j{0}; ///< Actual route_j load before the decision; zero when absent.
  bool reversed_i{false}; ///< Whether route_i was reversed to put i last for an accepted merge.
  bool reversed_j{false}; ///< Whether route_j was reversed to put j first for an accepted merge.
  ClarkeWrightDecision decision{ClarkeWrightDecision::MissingEndpoint}; ///< Actual decision; default record performs no routing.
  std::vector<std::size_t> merged_customers; ///< Actual committed surviving route, accepted merges only.
};
/// @brief Actual named routing operations, not elapsed time or a complete instruction count.
/// Every sorted pair scans all initial route slots, including cleared slots.
/// Geometric evaluations include cached depot distances, customer-pair distances
/// and final output edges. Comparator counts depend on the standard library sort.
/// Appended customers count elements copied into surviving routes, not reversals.
/// @par Example
/// @code
/// const auto w=dm::algorithms::clarke_wright_trace({{0,0},{1,0},{2,0}},{0,1,1},2).work;
/// std::print("{} {} {}\n",w.savings_pairs,w.endpoint_slot_visits,w.merges); // 1 2 1
/// @endcode
/// @see ClarkeWrightTrace
struct ClarkeWrightWork {
  std::size_t validated_points{0}; ///< Actual complete point/demand validation visits, depot included.
  std::size_t savings_pairs{0}; ///< Actual customer-pair savings constructed.
  std::size_t distance_evaluations{0}; ///< Actual widened Euclidean distance evaluations.
  std::size_t sort_comparisons{0}; ///< Actual comparisons made by std::sort.
  std::size_t endpoint_slot_visits{0}; ///< Actual route-slot inspections over all candidate decisions.
  std::size_t decisions{0}; ///< Actual sorted savings pairs processed, including rejected and zero-saving pairs.
  std::size_t merges{0}; ///< Actual successful route concatenations.
  std::size_t route_reversals{0}; ///< Actual std::reverse calls, including singleton ranges.
  std::size_t appended_customers{0}; ///< Actual customers appended from absorbed routes.
  std::size_t output_edges{0}; ///< Actual traversed edges in depot-closed returned routes.
};
/// @brief Owned actual parallel-savings decisions, final routes, loads and distance accounting.
/// Ordinary/traced entry points share the executing kernel without replay.
/// initial_star_distance is twice the sum of cached depot distances. Accepted
/// savings telescope in exact arithmetic; accounting_residual measures the actual
/// widened final distance minus (initial_star_distance-accepted_savings).
/// Rounded sums need not cancel exactly. No optimality, fleet-count or approximation
/// guarantee follows. Calls share no state; no borrowed input references survive.
/// @par Example
/// @code
/// const auto t=dm::algorithms::clarke_wright_trace({{0,0},{1,0},{2,0}},{0,1,1},2);
/// std::print("{} {} {}\n",t.initial_star_distance,t.accepted_savings,t.result.total_distance); // 6 2 4
/// @endcode
/// @see clarke_wright_trace(), VehicleRoutes
struct ClarkeWrightTrace {
  VehicleRoutes result; ///< Actual final closed routes in ascending surviving-slot order.
  std::vector<ClarkeWrightStep> steps; ///< Every actual sorted-pair decision, including rejections.
  std::vector<long double> route_loads; ///< Actual widened loads aligned with result.routes.
  long double initial_star_distance{0}; ///< Actual cached depot-star length before any merges.
  long double accepted_savings{0}; ///< Actual sum of savings on accepted pairs only.
  long double measured_distance{0}; ///< Actual widened edge sum before checked conversion to result.total_distance.
  long double accounting_residual{0}; ///< Measured length minus initial star less accepted savings.
  ClarkeWrightWork work; ///< Actual named counters; no timing inference.
};
/// @brief Construct capacity-feasible Euclidean routes by deterministic parallel Clarke-Wright savings.
/// Starts one route per customer, sorts all customer pairs by descending widened
/// saving d(depot,i)+d(depot,j)-d(i,j), breaking ties by ascending (i,j), then joins
/// distinct endpoint routes whose widened loads fit capacity. Zero-saving pairs
/// are processed too. No improvement phase, fixed fleet size, time windows, split
/// deliveries, optimality or approximation guarantee is supplied. Nonempty points
/// and matching finite nonnegative demands are required; finite capacity is positive,
/// each customer demand is at most capacity, and all coordinates are finite.
/// Depot demand is validated but ignored; depot-only input returns empty routes.
/// No capacity tolerance is applied: literal floating loads can straddle a boundary.
/// Widened checked geometry/loads and deterministic ties can change last-bit costs
/// or earlier unspecified tie choices. The old call/result layout remains.
/// For m customers, O(m^3+m^2 log m) work and O(m^2) auxiliary storage, due to full
/// route-slot lookup for each pair. Input is not mutated; outputs own all storage.
/// @param points Borrowed finite planar coordinates in arbitrary consistent distance units.
/// @param demand Borrowed finite nonnegative demands, same length as points; depot value ignored.
/// @param capacity Finite positive common vehicle capacity, same units as demands.
/// @param depot Original depot vertex id, default zero, in points range.
/// @return Owned feasible heuristic routes and checked finite-double measured total distance.
/// @throws std::invalid_argument If shape, coordinates, demands, capacity or depot is invalid.
/// @throws std::length_error If logical counts/container capacities exceed supported range.
/// @throws std::overflow_error If numeric arithmetic, counters or final double distance cannot fit.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// const auto r=dm::algorithms::clarke_wright_savings({{0,0},{1,0},{2,0}},{0,1,1},2);
/// std::print("{} {}\n",r.routes.size(),r.total_distance); // 1 4
/// @endcode
/// @par References
/// - G. Clarke and J. W. Wright (1964), Scheduling of Vehicles from a Central Depot
///   to a Number of Delivery Points, Operations Research 12(4), 568-581,
///   https://doi.org/10.1287/opre.12.4.568. This implementation is the simple
///   parallel savings variant with endpoint/capacity checks and specified ties.
/// @see clarke_wright_trace(), clarke_wright_savings_with_options()
VehicleRoutes clarke_wright_savings(const std::vector<std::pair<double,double>>& points,const std::vector<double>& demand,double capacity,std::size_t depot=0);
/// @brief Execute the same savings kernel with complete preflight work limits.
/// All input validation precedes caps and allocations. Positive pair and endpoint
/// caps apply; the decision-record cap applies only when tracing. No partial
/// result escapes failure. Same geometry, ties, ownership and complexity as the
/// ordinary call; zero caps are unlimited and inputs remain unchanged.
/// @param points Borrowed finite planar coordinates.
/// @param demand Borrowed finite nonnegative demands, matching point count.
/// @param capacity Finite positive common capacity.
/// @param depot Valid original depot id, default zero.
/// @param options Complete work limits, default unlimited.
/// @return Owned feasible heuristic routes with checked finite-double length.
/// @throws std::invalid_argument If the complete input is invalid.
/// @throws std::length_error If a positive work cap or supported count/storage capacity is exceeded.
/// @throws std::overflow_error If numeric arithmetic, counters or final double length cannot fit.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// dm::algorithms::ClarkeWrightOptions o;o.max_savings_pairs=1;
/// std::print("{}\n",dm::algorithms::clarke_wright_savings_with_options({{0,0},{1,0},{2,0}},{0,1,1},2,0,o).routes.size()); // 1
/// @endcode
/// @see ClarkeWrightOptions, clarke_wright_savings()
VehicleRoutes clarke_wright_savings_with_options(const std::vector<std::pair<double,double>>& points,const std::vector<double>& demand,double capacity,std::size_t depot=0,ClarkeWrightOptions options={});
/// @brief Retain actual savings decisions, accepted route orders, loads and named work.
/// Uses the same kernel without replay. Records all C=m(m-1)/2 candidate decisions
/// and owned accepted concatenations, adding O(m^2) trace storage; no full route
/// partition is copied on rejection. Positive record limits preflight C after
/// input validation and before allocation. Ordinary work remains O(m^3+m^2 log m).
/// @param points Borrowed finite planar coordinates.
/// @param demand Borrowed finite nonnegative demands, matching point count.
/// @param capacity Finite positive common capacity.
/// @param depot Valid original depot id, default zero.
/// @param options Complete work/record limits, default unlimited.
/// @return Owned actual decisions and final result/accounting, without an optimality certificate.
/// @throws std::invalid_argument If the complete input is invalid.
/// @throws std::length_error If a work/record cap or supported count/storage capacity is exceeded.
/// @throws std::overflow_error If numeric arithmetic, counters or final double length cannot fit.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// const auto t=dm::algorithms::clarke_wright_trace({{0,0},{1,0},{2,0}},{0,1,1},2);
/// std::print("{} {}\n",t.steps.size(),t.route_loads[0]); // 1 2
/// @endcode
/// @see ClarkeWrightTrace, ClarkeWrightOptions
ClarkeWrightTrace clarke_wright_trace(const std::vector<std::pair<double,double>>& points,const std::vector<double>& demand,double capacity,std::size_t depot=0,ClarkeWrightOptions options={});
}
