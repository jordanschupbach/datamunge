#pragma once
/// @file nearest_neighbor_tsp.hpp
/// @brief Greedy complete-graph tours with actual candidate decisions and bounded enumeration references.
#include <cstddef>
#include <datamunge/algorithms/tour_result.hpp>
#include <utility>
#include <vector>
namespace datamunge::algorithms {
/// @brief Actual greedy tour and selected scan/sum work counts.
/// Candidate checks include visited IDs; distance comparisons occur only after
/// the first eligible candidate has established a current best. Counts are not
/// matrix-load or CPU-operation totals. Final tour sum includes the closing edge.
/// @par Example
/// @code
/// auto a=dm::algorithms::nearest_neighbor_tsp_analyze({{0,1,2},{1,0,1},{2,1,0}});
/// std::print("{} {}\n",a.result.cost,a.candidate_checks); // 4 6
/// @endcode
/// @see nearest_neighbor_tsp_analyze()
struct NearestTSPAnalysis {
  TourResult  result;              ///< Actual closed greedy tour and finite sequential sum.
  std::size_t candidate_checks{0}; ///< Actual visited-ID tests, n*(n−1).
  std::size_t distance_comparisons{
      0};                      ///< Actual strict distance comparisons among eligible candidates.
  std::size_t edges_summed{0}; ///< Actual returned-tour edges summed, n for nonempty input.
};
/// @brief Actual eligible candidate and current-best update at one greedy visit.
/// Candidates appear by ascending ID; first=true means no prior best, so no
/// distance comparison occurred. Equal-distance later candidates never replace.
/// @par Example
/// @code
/// auto t=dm::algorithms::nearest_neighbor_tsp_trace({{0,1,2},{1,0,1},{2,1,0}});
/// std::print("{}\n",t.trials.front().candidate); // 1
/// @endcode
/// @see NearestTSPTrace
struct NearestTSPTrial {
  std::size_t from{0};         ///< Actual current city.
  std::size_t candidate{0};    ///< Actual unvisited city being inspected.
  double      distance{0};     ///< Actual matrix edge weight observed at this candidate.
  bool        first{false};    ///< True for the first eligible candidate, without a comparison.
  bool        improved{false}; ///< Actual strict improvement or first-candidate initialization.
};
/// @brief Actual chosen next city and its eligible-trial span.
/// @par Example
/// @code
/// auto t=dm::algorithms::nearest_neighbor_tsp_trace({{0,1,2},{1,0,1},{2,1,0}});
/// std::print("{}\n",t.choices.front().to); // 1
/// @endcode
/// @see NearestTSPTrace
struct NearestTSPChoice {
  std::size_t from{0};        ///< Actual departure city.
  std::size_t to{0};          ///< Actual nearest eligible city, smallest ID on ties.
  double      distance{0};    ///< Actual selected edge weight.
  std::size_t first_trial{0}; ///< Actual first eligible-trial index for this choice.
  std::size_t trial_count{0}; ///< Actual eligible candidates examined.
};
/// @brief Actual post-route sequential edge sum, including closure.
/// @par Example
/// @code
/// auto t=dm::algorithms::nearest_neighbor_tsp_trace({{0,1,2},{1,0,1},{2,1,0}});
/// std::print("{}\n",t.edges.back().cumulative); // 4
/// @endcode
/// @see NearestTSPTrace
struct NearestTSPEdge {
  std::size_t from{0};       ///< Actual previous tour city.
  std::size_t to{0};         ///< Actual next tour city, including final start.
  double      distance{0};   ///< Actual edge cost read during summation.
  double      cumulative{0}; ///< Actual rounded finite running total after this edge.
};
/// @brief Owned original matrix/start and actual greedy/summation history.
/// Capture units count start, input rows/weights, result route IDs, eligible
/// trial records, choice records and edge-sum records. Scalar fields within a
/// record are not additional units. Empty input retains start but no route.
/// @par Example
/// @code
/// auto t=dm::algorithms::nearest_neighbor_tsp_trace({{0,1,2},{1,0,1},{2,1,0}});
/// std::print("{}\n",t.captured_entries); // 25
/// @endcode
/// @see nearest_neighbor_tsp_trace()
struct NearestTSPTrace {
  std::vector<std::vector<double>> input;    ///< Owned complete original distance matrix.
  std::size_t                      start{0}; ///< Original start, ignored for an empty matrix.
  NearestTSPAnalysis               analysis; ///< Actual closed tour, cost and selected work.
  std::vector<NearestTSPTrial>     trials;   ///< Actual eligible candidate observations.
  std::vector<NearestTSPChoice>    choices;  ///< Actual selected city at each greedy step.
  std::vector<NearestTSPEdge>      edges;    ///< Actual final sequential cost accumulation.
  std::size_t                      captured_entries{0}; ///< Actual admitted logical history units.
};
/// @brief Greedily visit the closest unvisited city and return to the start.
/// Complete finite nonnegative square matrix required; asymmetry, zero distances,
/// nonzero diagonal and triangle-inequality violations are allowed. Scan IDs in
/// ascending order and replace only on strict distance improvement, preserving
/// smallest-ID ties and the legacy selected route. No optimality/constant-ratio
/// guarantee for arbitrary matrices. O(n^2) scan work, O(n) driver/output storage,
/// excluding caller's input matrix. Singleton returns[start,start] with its
/// diagonal cost; empty returns zero/empty and ignores start.
/// @param distance Complete nonnegative finite directed edge matrix.
/// @param start Valid city ID when nonempty, default0; ignored if empty.
/// @return Owned actual closed tour and finite sequential rounded cost.
/// @throws std::invalid_argument If shape/weights or nonempty start are invalid.
/// @throws std::length_error If complete matrix/route count geometry is unrepresentable.
/// @throws std::overflow_error If the tour's produced cost becomes nonfinite.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::nearest_neighbor_tsp({{0,1,2},{1,0,1},{2,1,0}}).cost); // 4
/// @endcode
/// @par References
/// - D. J. Rosenkrantz, R. E. Stearns and P. M. Lewis II (1977), An analysis of several heuristics for the traveling salesman problem, SIAM Journal on Computing6,563–581, doi:10.1137/0206041.
/// @see nearest_neighbor_tsp_analyze(), enumerate_tsp()
TourResult nearest_neighbor_tsp(
    const std::vector<std::vector<double>>& distance,
    std::size_t                             start = 0);
/// @brief Return actual selected scan/sum counts with the ordinary greedy tour.
/// Same producer, guards, route and complexity as nearest_neighbor_tsp(), no replay.
/// @param distance Complete finite nonnegative square edge matrix.
/// @param start Valid nonempty city ID, default0, ignored for empty.
/// @return Actual result and selected observed work.
/// @throws std::invalid_argument If matrix/start is invalid.
/// @throws std::length_error If complete geometry is unrepresentable.
/// @throws std::overflow_error If produced tour sum is nonfinite.
/// @par Example
/// @code
/// auto a=dm::algorithms::nearest_neighbor_tsp_analyze({{0,1},{1,0}});
/// std::print("{} {}\n",a.result.cost,a.distance_comparisons); // 2 0
/// @endcode
/// @see NearestTSPAnalysis
NearestTSPAnalysis nearest_neighbor_tsp_analyze(
    const std::vector<std::vector<double>>& distance,
    std::size_t                             start = 0);
/// @brief Capture actual eligible candidates, greedy choices and final edge sums.
/// No replay; same ordinary route/cost. Admit complete input rows/weights/start
/// before copying, then actual output/history records online. No partial trace
/// escapes a quota failure. Extra history/input-copy storage follows admitted units.
/// @param distance Original complete finite nonnegative matrix, copied into output.
/// @param start Original start, default0; ignored for empty input.
/// @param max_entries Actual capture cap, default1000000; zero disables.
/// @return Owned actual original inputs, decisions, cost sums and work.
/// @throws std::invalid_argument If matrix/start is invalid, before capture.
/// @throws std::length_error If geometry or quota exceeds its limit.
/// @throws std::overflow_error If the actual tour sum is nonfinite.
/// @throws std::bad_alloc If history allocation fails.
/// @par Example
/// @code
/// auto t=dm::algorithms::nearest_neighbor_tsp_trace({{0,1},{1,0}});
/// std::print("{}\n",t.analysis.result.tour); // [0, 1, 0]
/// @endcode
/// @see NearestTSPTrace
NearestTSPTrace nearest_neighbor_tsp_trace(
    const std::vector<std::vector<double>>& distance,
    std::size_t                             start       = 0,
    std::size_t                             max_entries = 1000000);
/// @brief Enumerate all small anchored tours as a bounded optimal-cost reference.
/// Permute remaining IDs in lexicographic order, retaining the first strict cost
/// minimum. Uses the same sequential cost sum order as the greedy call; overflowed
/// candidates saturate to infinity and cannot defeat a finite minimum.
/// This certifies that rounded objective over enumerated tours, not exact real
/// arithmetic for arbitrary noninteger weights. O(n*(n−1)!) work, O(n) driver
/// storage. No pruning or symmetry reduction; directed matrices allowed.
/// @param distance Complete finite nonnegative square matrix.
/// @param start Valid nonempty start, default0; ignored if empty.
/// @param max_vertices Complete vertex cap, default10; zero disables and may be expensive.
/// @return Owned first lexicographic minimum-cost closed tour; empty for empty input.
/// @throws std::invalid_argument If matrix/start is invalid.
/// @throws std::length_error If complete geometry or nonzero vertex cap is exceeded.
/// @throws std::overflow_error If every enumerated tour has a nonfinite sequential sum.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::enumerate_tsp(dm::algorithms::nearest_neighbor_tsp_adversary()).cost); // 4
/// @endcode
/// @see nearest_neighbor_tsp()
TourResult enumerate_tsp(
    const std::vector<std::vector<double>>& distance,
    std::size_t                             start        = 0,
    std::size_t                             max_vertices = 10);
/// @brief Generate a three-city directed nonmetric greedy-trap family.
/// Rows are(0,1,2),(1,0,M),(1,1,0). From0, greedy costs M+2 while the other
/// anchored cycle costs4, so M>=2 makes4 an optimum. Integers in the report
/// avoid floating-sum ambiguity. No symmetry or metric claim is made.
/// @param expensive Finite M>=2, default100.
/// @return Owned original-ID3-by3 distance matrix.
/// @throws std::invalid_argument If M is nonfinite or below2.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::nearest_neighbor_tsp(dm::algorithms::nearest_neighbor_tsp_adversary()).cost); // 102
/// @endcode
/// @see enumerate_tsp()
std::vector<std::vector<double>> nearest_neighbor_tsp_adversary(double expensive = 100);
/// @brief Return a small native planar city fixture for geometric route comparisons.
/// @return Owned points(0,0),(4,0),(4,3),(0,3),(1,1),(3,2), in original-ID order.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::nearest_neighbor_tsp_demo_points().size()); // 6
/// @endcode
/// @see nearest_neighbor_tsp(), euclidean_tsp_distances()
std::vector<std::pair<double, double>> nearest_neighbor_tsp_demo_points();
} // namespace datamunge::algorithms
