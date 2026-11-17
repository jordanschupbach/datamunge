#pragma once
/// @file nonblocking_switch.hpp
/// @brief Checked two-middle-switch rearrangement through actual bipartite path/cycle coloring.
#include <cstddef>
#include <utility>
#include <vector>
namespace datamunge::algorithms {
/// @brief Complete group/call/capture policies, each zero disabling its limit.
/// @par Example
/// @code
/// dm::algorithms::SwitchRouteOptions o;std::print("{}\n",o.max_groups); // 1000000
/// @endcode
/// @see nonblocking_switch_trace()
struct SwitchRouteOptions {
  std::size_t max_groups{
      1000000}; ///< Sum of input and output group counts before dense incidence allocation.
  std::size_t max_calls{1000000}; ///< Complete original call count before incidence allocation.
  std::size_t max_trace_entries{
      1000000}; ///< Complete logical history units before coloring, ignored without capture.
};
/// @brief Actual middle-switch choices and selected BFS work.
/// @par Example
/// @code
/// auto a=dm::algorithms::nonblocking_switch_analyze(2,2,dm::algorithms::nonblocking_switch_demo_calls());std::print("{} {}\n",a.components,a.incidence_reads); // 1 16
/// @endcode
/// @see nonblocking_switch_trace()
struct SwitchRouteAnalysis {
  std::vector<std::size_t> routes; ///< Owned actual middle choices0 or1 in original call order.
  std::size_t components{0};       ///< Actual path/even-cycle components, including isolated calls.
  std::size_t queue_pops{0};       ///< Actual processed calls, one per original call.
  std::size_t incidence_reads{
      0}; ///< Actual candidate reads, including each call itself at each endpoint; at most4C.
  std::size_t assignments{0}; ///< Actual initial/discovered color assignments, exactly C.
};
/// @brief Actual first assignment of one call to a middle switch.
/// @par Example
/// @code
/// auto t=dm::algorithms::nonblocking_switch_trace(2,2,dm::algorithms::nonblocking_switch_demo_calls());std::print("{}\n",t.assignments[0].middle); // 0
/// @endcode
/// @see nonblocking_switch_trace()
struct SwitchRouteAssignment {
  std::size_t call{0};      ///< Actual original call ID receiving a color.
  std::size_t middle{0};    ///< Actual assigned middle switch0 or1.
  std::size_t component{0}; ///< Actual zero-based component traversal ID.
  std::size_t parent{0};    ///< Actual discovering call ID, or C sentinel for a component start.
  bool        via_input{
      false}; ///< True discovery via shared input; false shared output, ignored for a start.
  bool component_start{false}; ///< Actual outer-scan initial assignment, always middle0.
};
/// @brief Actual shared-endpoint incidence candidate visited by BFS.
/// @par Example
/// @code
/// auto t=dm::algorithms::nonblocking_switch_trace(2,2,dm::algorithms::nonblocking_switch_demo_calls());std::print("{}\n",t.scans[0].from); // 0
/// @endcode
/// @see nonblocking_switch_trace()
struct SwitchRouteScan {
  std::size_t from{0};          ///< Actual dequeued original call ID.
  std::size_t candidate{0};     ///< Actual incidence candidate, including self.
  bool        via_input{false}; ///< True input incidence list, false output incidence list.
  std::size_t color_before{2};  ///< Actual candidate color before this visit;2 means unassigned.
  bool        assigned{false};  ///< Whether this visit actually assigned/enqueued the candidate.
};
/// @brief Owned complete calls and actual assignment/incidence histories.
/// @par Example
/// @code
/// auto t=dm::algorithms::nonblocking_switch_trace(2,2,dm::algorithms::nonblocking_switch_demo_calls());std::print("{}\n",t.captured_entries); // 34
/// @endcode
/// @see nonblocking_switch_trace()
struct SwitchRouteTrace {
  std::size_t input_groups{0};  ///< Original input group count.
  std::size_t output_groups{0}; ///< Original output group count.
  std::vector<std::pair<std::size_t, std::size_t>>
      connections; ///< Owned original input/output group pairs, parallel calls allowed.
  SwitchRouteAnalysis                analysis;    ///< Actual same-producer route choices and work.
  std::vector<SwitchRouteAssignment> assignments; ///< Actual first-assignment order, no replay.
  std::vector<SwitchRouteScan> scans; ///< Actual dequeue/input-list/output-list candidate order.
  std::size_t                  captured_entries{0}; ///< Complete units2+4C+actual incidence reads.
};
/// @brief Assign every degree-at-most-two bipartite call to one of two middle switches.
/// Calls are edges between input/output groups; parallel edges are permitted.
/// Degree<=2 makes components paths or even cycles. Alternating0/1 along them
/// avoids sharing a middle link at either endpoint. This is a restricted
/// rearrangeable model, not a general Clos/strictly-nonblocking sizing theorem.
/// Outer start scan and incident lists use original call order; each component
/// starts0, then BFS scans input incidences before output incidences. Endpoints
/// are completely checked before group/call caps, then degrees before coloring.
/// Original API disables optional policies. O(I+O+C) work/storage; incidence
/// candidate reads<=4C. Does not mutate inputs or retain caller references.
/// @param input_groups Number I of input subswitch groups, zero valid for no calls.
/// @param output_groups Number O of output subswitch groups, zero valid for no calls.
/// @param connections Original complete input/output group-ID pairs.
/// @return Owned actual std::vector<std::size_t> choices/work in original order.
/// @throws std::invalid_argument If an endpoint is out of range or any group degree exceeds2.
/// @throws std::length_error If complete count/container geometry is exceeded.
/// @throws std::bad_alloc If incidence/queue/result allocation fails.
/// @par Example
/// @code
/// auto r=dm::algorithms::nonblocking_switch_routes(2,2,dm::algorithms::nonblocking_switch_demo_calls());
/// std::print("{}\n",r); // [0, 1, 0, 1]
/// @endcode
/// @see SwitchRouteAnalysis, SwitchRouteTrace
std::vector<std::size_t> nonblocking_switch_routes(
    std::size_t                                             input_groups,
    std::size_t                                             output_groups,
    const std::vector<std::pair<std::size_t, std::size_t>>& connections);
/// @brief Assign every degree-at-most-two bipartite call to one of two middle switches.
/// Calls are edges between input/output groups; parallel edges are permitted.
/// Degree<=2 makes components paths or even cycles. Alternating0/1 along them
/// avoids sharing a middle link at either endpoint. This is a restricted
/// rearrangeable model, not a general Clos/strictly-nonblocking sizing theorem.
/// Outer start scan and incident lists use original call order; each component
/// starts0, then BFS scans input incidences before output incidences. Endpoints
/// are completely checked before group/call caps, then degrees before coloring.
/// Original API disables optional policies. O(I+O+C) work/storage; incidence
/// candidate reads<=4C. Does not mutate inputs or retain caller references.
/// @param input_groups Number I of input subswitch groups, zero valid for no calls.
/// @param output_groups Number O of output subswitch groups, zero valid for no calls.
/// @param connections Original complete input/output group-ID pairs.
/// @param options Complete projected group/call/capture policies, defaults1000000, zero disables each.
/// @return Owned actual SwitchRouteAnalysis choices/work in original order.
/// @throws std::invalid_argument If an endpoint is out of range or any group degree exceeds2.
/// @throws std::length_error If complete count/container geometry or selected policy is exceeded.
/// @throws std::bad_alloc If incidence/queue/result allocation fails.
/// @par Example
/// @code
/// auto r=dm::algorithms::nonblocking_switch_analyze(2,2,dm::algorithms::nonblocking_switch_demo_calls());
/// std::print("{}\n",r.routes); // [0, 1, 0, 1]
/// @endcode
/// @see SwitchRouteAnalysis, SwitchRouteTrace
SwitchRouteAnalysis nonblocking_switch_analyze(
    std::size_t                                             input_groups,
    std::size_t                                             output_groups,
    const std::vector<std::pair<std::size_t, std::size_t>>& connections,
    SwitchRouteOptions                                      options = {});
/// @brief Capture actual two-middle-switch path/even-cycle rearrangement decisions.
/// Calls are edges between input/output groups; parallel edges are permitted.
/// Degree<=2 makes components paths or even cycles. Alternating0/1 along them
/// avoids sharing a middle link at either endpoint. This is a restricted
/// rearrangeable model, not a general Clos/strictly-nonblocking sizing theorem.
/// Outer start scan and incident lists use original call order; each component
/// starts0, then BFS scans input incidences before output incidences. Endpoints
/// are completely checked before group/call caps, then degrees before coloring.
/// Original API disables optional policies. O(I+O+C) work/storage; incidence
/// candidate reads<=4C. Does not mutate inputs or retain caller references.
/// @param input_groups Number I of input subswitch groups, zero valid for no calls.
/// @param output_groups Number O of output subswitch groups, zero valid for no calls.
/// @param connections Original complete input/output group-ID pairs.
/// @param options Complete projected group/call/capture policies, defaults1000000, zero disables each.
/// @return Owned actual calls, histories and same-producer result.
/// @throws std::invalid_argument If an endpoint is out of range or any group degree exceeds2.
/// @throws std::length_error If complete count/container geometry or selected policy is exceeded.
/// @throws std::bad_alloc If incidence/queue/result allocation fails.
/// @par Example
/// @code
/// auto r=dm::algorithms::nonblocking_switch_trace(2,2,dm::algorithms::nonblocking_switch_demo_calls());
/// std::print("{}\n",r.analysis.routes); // [0, 1, 0, 1]
/// @endcode
/// @see SwitchRouteAnalysis, SwitchRouteTrace
SwitchRouteTrace nonblocking_switch_trace(
    std::size_t                                             input_groups,
    std::size_t                                             output_groups,
    const std::vector<std::pair<std::size_t, std::size_t>>& connections,
    SwitchRouteOptions                                      options = {});
/// @brief Validate a proposed two-middle-switch routing certificate independently of BFS tie choices.
/// Same complete endpoint/degree/geometry domain as the ordinary router, then
/// checks exactly one0/1 route per call and no repeated input/output middle link.
/// O(I+O+C) work/storage, no mutation, accepts a component-wise flipped coloring.
/// @param input_groups Original input group count.
/// @param output_groups Original output group count.
/// @param connections Original complete validated-domain call pairs.
/// @param routes Proposed original-order middle IDs.
/// @return True iff supplied choices are a feasible certificate; invalid choices/shape return false.
/// @throws std::invalid_argument If original endpoint/degree domain is invalid.
/// @throws std::length_error If group/container geometry is unrepresentable.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::nonblocking_switch_feasible(2,2,dm::algorithms::nonblocking_switch_demo_calls(),{1,0,1,0})); // true
/// @endcode
/// @see nonblocking_switch_routes()
bool nonblocking_switch_feasible(
    std::size_t                                             input_groups,
    std::size_t                                             output_groups,
    const std::vector<std::pair<std::size_t, std::size_t>>& connections,
    const std::vector<std::size_t>&                         routes);
/// @brief Return the original four-cycle call order(0,0),(0,1),(1,1),(1,0).
/// @return Owned original2-by2 call fixture.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::nonblocking_switch_demo_calls().size()); // 4
/// @endcode
/// @see nonblocking_switch_cycle_calls()
std::vector<std::pair<std::size_t, std::size_t>> nonblocking_switch_demo_calls();
/// @brief Generate original-order bipartite cycles, including two parallel calls when groups=1.
/// For each i append(i,i),(i,(i+1)%groups); groups0 gives empty input.
/// @param groups Equal input/output group count, at most1000000.
/// @return Owned2*groups calls, with all nonempty endpoint degrees2.
/// @throws std::length_error If the explicit fixture group cap is exceeded.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::nonblocking_switch_cycle_calls(3).size()); // 6
/// @endcode
/// @see nonblocking_switch_demo_calls()
std::vector<std::pair<std::size_t, std::size_t>> nonblocking_switch_cycle_calls(std::size_t groups);
} // namespace datamunge::algorithms
