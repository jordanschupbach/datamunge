#pragma once
/// @file grasp_analysis.hpp
/// @brief Original MAX-CUT fixture, independent partition diagnostics and bounded exact
/// enumeration.
#include <datamunge/algorithms/grasp.hpp>
namespace datamunge::algorithms {
/// @brief Separately evaluated supplied partition, not a producing GRASP trace.
/// Floating gains use the same sequential incident sums and 1e-12 comparison as
/// local search. tolerance_local certifies only those computed tests. It is not
/// a global or exact-real optimum certificate. All storage is owned.
/// @par Example
/// @code
/// const auto d=dm::algorithms::grasp_cut_check({{{1,2}},{{0,2}}},{0,1});
/// std::print("{} {}\n",d.value,d.tolerance_local); // 2 true
/// @endcode
/// @see grasp_cut_check()
struct GraspCutCheck {
  double              value{0};   ///< Actual independently evaluated u<v crossing sum.
  std::vector<double> flip_gains; ///< Actual separately evaluated per-vertex gains.
  bool tolerance_local{false}; ///< Every computed gain is <=1e-12, true vacuously for empty input.
};
/// @brief Owned exact-enumeration result and actual examined complement classes.
/// Exact refers to exhaustive partitions, not exact arithmetic on arbitrary
/// floating weights. Values use sequential double sums; integer fixtures with
/// exact representable sums admit exact comparisons. Side zero is fixed at
/// vertex zero, removing complement symmetry; strict value ties keep the first.
/// @par Example
/// @code
/// const auto e=dm::algorithms::grasp_exact_max_cut({{{1,2}},{{0,2}}});
/// std::print("{} {}\n",e.result.value,e.partitions); // 2 2
/// @endcode
/// @see grasp_exact_max_cut()
struct GraspExactCut {
  GraspMaxCutResult result;        ///< Actual best enumerated partition/value.
  std::size_t       partitions{0}; ///< Actual examined masks; one for the empty graph.
};
/// @brief Return the original report's seeded fourteen-vertex integer-weight graph.
/// Preserves the original seed12345, xorshift draw sequence, edge-presence modulo
/// two and weight1+word%9 choices. Contains fifty undirected edges on this fixed
/// unsigned generator. No download/global RNG or input references.
/// @return Owned symmetric adjacency in original insertion order.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::grasp_demo_graph().size()); // 14
/// @endcode
/// @see grasp_max_cut_trace()
GraspAdjacency grasp_demo_graph();
/// @brief Independently evaluate a supplied complete cut and every possible single flip.
/// Uses the shared validated graph and objective/delta primitives, but is separate
/// work and does not enter a run's counters. O(A log A+n) validation plus O(A+n)
/// arithmetic/storage. Requires matching side length and labels zero or one.
/// @param adj Borrowed finite symmetric loopless weighted graph, not retained.
/// @param side Borrowed complete 0/1 partition, unmodified.
/// @return Owned actual objective/gains and computed tolerance-local flag.
/// @throws std::invalid_argument On malformed graph or side labels/shape.
/// @throws std::overflow_error On nonfinite objective/gains.
/// @throws std::length_error On dimension/count/container limits.
/// @throws std::bad_alloc On validation/output allocation failure.
/// @par Example
/// @code
/// const auto d=dm::algorithms::grasp_cut_check({{{1,2}},{{0,2}}},{0,0});
/// std::print("{} {}\n",d.value,d.flip_gains[0]); // 0 2
/// @endcode
/// @see GraspCutCheck, grasp_max_cut()
GraspCutCheck grasp_cut_check(const GraspAdjacency& adj, const std::vector<int>& side);
/// @brief Enumerate all small-graph cuts with vertex zero fixed to side zero.
/// Hard limit24 vertices; complete partition count2^(n-1) (one if n=0) is checked
/// against max_partitions before enumeration. Zero disables only the policy cap.
/// Basic/symmetry graph validation occurs before this partition cap. Uses no
/// GRASP search, RNG or local improvement. O(2^n(A+n)) work and O(A+n) storage;
/// actual partition count is returned. This is a bounded verification tool.
/// @param adj Borrowed finite symmetric loopless adjacency, at most24 rows.
/// @param max_partitions Complete mask cap, default1000000; zero disables policy cap.
/// @return Owned best exhaustive partition and actual examined complement classes.
/// @throws std::invalid_argument On malformed graph.
/// @throws std::length_error On hard24 bound, policy cap or counts/container limits.
/// @throws std::overflow_error On nonfinite sequential cut values.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto e=dm::algorithms::grasp_exact_max_cut({{{1,2}},{{0,2}}},2);
/// std::print("{}\n",e.partitions); // 2
/// @endcode
/// @see GraspExactCut, grasp_cut_check()
GraspExactCut grasp_exact_max_cut(const GraspAdjacency& adj, std::size_t max_partitions = 1000000);
} // namespace datamunge::algorithms
