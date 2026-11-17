#pragma once
/// @file optics.hpp
/// @brief Checked serial scan-seed OPTICS with actual neighborhood, reachability and selection histories.
#include <cstddef>
#include <utility>
#include <vector>
namespace datamunge::algorithms {
/// @brief Owned original-layout processing order and per-ID density distances.
/// @par Example
/// @code
/// auto r=dm::algorithms::optics({{0.},{.5},{1.},{4.}},.75,2);std::print("{}\n",r.ordering); // [0, 1, 2, 3]
/// @endcode
/// @see optics_trace()
struct OpticsResult {
  std::vector<std::size_t>
      ordering; ///< Actual original point IDs in processing order, complete permutation.
  std::vector<double>
      reachability; ///< Actual per-original-ID reachability, positive infinity when undefined.
  std::vector<double>
      core_distance; ///< Actual per-original-ID min_pts-th neighborhood distance, infinity when undefined.
};
/// @brief Actual selected scan-seed producer counts, excluding validation/sort-comparator counts.
/// @par Example
/// @code
/// dm::algorithms::OpticsWork w;std::print("{}\n",w.distance_calls); // 0
/// @endcode
/// @see optics_trace()
struct OpticsWork {
  std::size_t neighborhood_queries{0}; ///< Actual one per processed point.
  std::size_t distance_calls{0}; ///< Actual all-points distances, n*n for successful complete run.
  std::size_t coordinate_differences{
      0}; ///< Actual original double component differences in distance attempts.
  std::size_t widened_distances{0}; ///< Actual complete long-double hypot fallback distances.
  std::size_t update_candidates{
      0}; ///< Actual neighbor occurrences inspected by reachability update, including processed/self.
  std::size_t seed_insertions{
      0};                   ///< Actual previously undefined unprocessed reachability discoveries.
  std::size_t decreases{0}; ///< Actual strictly smaller finite reachability replacements.
  std::size_t seed_scan_entries{
      0}; ///< Actual candidate-seed slots inspected in linear minimum scans.
  std::size_t seed_extractions{0}; ///< Actual swap-with-last minimum removals.
};
/// @brief Owned actual OPTICS result, generating controls and selected work.
/// @par Example
/// @code
/// auto r=dm::algorithms::optics_run({{0.},{1.}},2.,2);std::print("{}\n",r.work.distance_calls); // 4
/// @endcode
/// @see optics_trace()
struct OpticsRun {
  OpticsResult result;    ///< Actual original-layout ordering/distances.
  double       eps{0};    ///< Original finite nonnegative inclusive neighborhood radius.
  std::size_t min_pts{0}; ///< Original positive required neighbor count including the point itself.
  OpticsWork  work;       ///< Actual selected serial production counts.
};
/// @brief Actual queried neighbor list in ascending original ID order and selected core distance.
/// @par Example
/// @code
/// dm::algorithms::OpticsNeighborhood n;std::print("{}\n",n.point); // 0
/// @endcode
/// @see optics_trace()
struct OpticsNeighborhood {
  std::size_t point{0}; ///< Actual queried original point ID.
  std::vector<std::pair<std::size_t, double>>
         neighbors; ///< Actual inclusive-radius neighbors, self included.
  double core_distance{
      0}; ///< Actual min_pts-th sorted distance or positive infinity if insufficient neighbors.
};
/// @brief Actual inspected update candidate, including explicitly skipped processed points.
/// @par Example
/// @code
/// dm::algorithms::OpticsReachUpdate u;std::print("{}\n",u.inserted); // false
/// @endcode
/// @see optics_trace()
struct OpticsReachUpdate {
  std::size_t from{0};     ///< Actual processed core source.
  std::size_t to{0};       ///< Actual neighbor point being inspected.
  double      distance{0}; ///< Actual finite neighbor distance.
  double      core{0};     ///< Actual finite source core distance.
  double      before{0};   ///< Actual previous target reachability, possibly infinity.
  double      proposed{0}; ///< Actual max(core,distance),0 placeholder when skipped processed=true.
  double      after{0};    ///< Actual committed or unchanged target reachability.
  bool        processed{false}; ///< True skips candidate before proposing a reachability.
  bool        inserted{false};  ///< Actual first finite reachability discovery and seed insertion.
  bool        decreased{false}; ///< Actual strictly lower finite target reachability.
};
/// @brief Actual seed list and finite priorities observed by one linear minimum extraction.
/// @par Example
/// @code
/// dm::algorithms::OpticsSelection s;std::print("{}\n",s.winning_position); // 0
/// @endcode
/// @see optics_trace()
struct OpticsSelection {
  std::vector<std::size_t> seeds;          ///< Owned actual seed order before scan/removal.
  std::vector<double>      reachabilities; ///< Owned actual corresponding finite priorities.
  std::size_t              winning_position{
      0}; ///< Actual first strict minimum position; equal priorities retain earlier slot.
  std::size_t point{0}; ///< Actual winning original ID, removed by swap with final seed.
};
/// @brief Actual committed processing-order observation.
/// @par Example
/// @code
/// dm::algorithms::OpticsVisit v;std::print("{}\n",v.region_start); // false
/// @endcode
/// @see optics_trace()
struct OpticsVisit {
  std::size_t point{0};         ///< Actual appended original point ID.
  double      core_distance{0}; ///< Actual core distance, infinity if undefined.
  double reachability{0}; ///< Actual original-ID reachability at visit, infinity for outer starts.
  bool   region_start{false}; ///< True only when chosen by the ascending-ID outer loop.
};
/// @brief Owned complete input/controls and actual same-producer search/update/scan histories.
/// @par Example
/// @code
/// auto t=dm::algorithms::optics_trace({{0.},{1.}},2.,2);std::print("{}\n",t.visits.size()); // 2
/// @endcode
/// @see optics_trace()
struct OpticsTrace {
  std::vector<std::vector<double>> data; ///< Owned complete original coordinate rows.
  OpticsRun                        run;  ///< Actual result, original controls and work.
  std::vector<OpticsNeighborhood>
      neighborhoods; ///< Every actual neighbor query and selected core distance.
  std::vector<OpticsReachUpdate> updates; ///< Every actual update candidate, including skips.
  std::vector<OpticsSelection>
      selections;                  ///< Every actual linear extraction with before-seed priorities.
  std::vector<OpticsVisit> visits; ///< Every actual committed processed point.
  std::size_t              captured_entries{
      0}; ///< Actual units2+ND+3N, neighborhoods1+2K, updates1,selections1+2K,visits1.
};
/// @brief Owned actual extraction labels, contiguous cluster sizes and noise count.
/// @par Example
/// @code
/// auto r=dm::algorithms::optics_cluster_summary({0,0,-1});std::print("{}\n",r.noise); // 1
/// @endcode
/// @see optics_trace()
struct OpticsClusterSummary {
  std::vector<int>
      labels; ///< Owned original-ID labels, -1 noise and contiguous nonnegative clusters.
  std::vector<std::size_t> sizes; ///< Actual original-label cluster counts, every entry positive.
  std::size_t              noise{0}; ///< Actual -1 label count.
};
/// @brief Owned presentation-only processing positions and displayed reachabilities.
/// @par Example
/// @code
/// dm::algorithms::OpticsReachabilityProfile p;std::print("{}\n",p.position.size()); // 0
/// @endcode
/// @see optics_trace()
struct OpticsReachabilityProfile {
  std::vector<double> position; ///< Actual zero-based processing positions, not original point IDs.
  std::vector<double>
      displayed; ///< Actual finite reachability or explicit finite display ceiling for undefined values.
  std::vector<char> undefined; ///< 1 iff original reachability was positive infinity.
  double undefined_height{0};  ///< Actual finite presentation ceiling, never an inferred distance.
};
/// @brief Produce the canonical serial OPTICS density ordering and actual distances.
/// Complete rectangular finite coordinates, finite eps>=0 and min_pts>=1 are
/// validated before allocation/search. Zero-dimensional points are accepted;
/// empty data returns empty outputs. Self counts toward min_pts, duplicates are
/// separate observations, and radius membership is inclusive distance<=eps.
/// Ordinary finite sum-of-squares distance order is retained; nonfinite double
/// intermediates use complete long-double hypot before finite-double admission.
/// Outer starts are ascending point IDs. Seed priorities use a LINEAR scan, first
/// strict minimum, then swap-with-last removal; no heap or spatial index exists.
/// This preserves original valid safe-arithmetic ordering/tie behavior.
/// O(N squared*(D+log N)) worst-case work including sorting each neighborhood,
/// O(N+ND) ordinary state/input, no caller mutation or retained references.
/// @param data Complete original coordinate rows, arbitrary common dimension.
/// @param eps Finite nonnegative inclusive generating radius.
/// @param min_pts Positive minimum neighbors, including self, even if greater than N.
/// @return Owned actual OpticsResult.
/// @throws std::invalid_argument If complete shape/finite/radius/count controls fail.
/// @throws std::length_error If complete geometry/count or selected history quota fails.
/// @throws std::overflow_error If a computed distance cannot be finite double.
/// @throws std::bad_alloc If owned storage allocation fails.
/// @par Example
/// @code
/// auto r=dm::algorithms::optics({{0.},{1.}},2.,2);std::print("{}\n",r.ordering); // [0, 1]
/// @endcode
/// @par References
/// - M. Ankerst, M. M. Breunig, H.-P. Kriegel and J. Sander(1999), OPTICS: Ordering Points To Identify the Clustering Structure, https://sigmodrecord.org/1999/06/25/optics-ordering-points-to-identify-the-clustering-structure/. Native candidate selection uses a linear scan, not a priority queue.
/// @see optics_extract_dbscan(), OpticsWork
OpticsResult optics(const std::vector<std::vector<double>>& data, double eps, std::size_t min_pts);
/// @brief Produce the canonical serial OPTICS density ordering and actual distances.
/// Complete rectangular finite coordinates, finite eps>=0 and min_pts>=1 are
/// validated before allocation/search. Zero-dimensional points are accepted;
/// empty data returns empty outputs. Self counts toward min_pts, duplicates are
/// separate observations, and radius membership is inclusive distance<=eps.
/// Ordinary finite sum-of-squares distance order is retained; nonfinite double
/// intermediates use complete long-double hypot before finite-double admission.
/// Outer starts are ascending point IDs. Seed priorities use a LINEAR scan, first
/// strict minimum, then swap-with-last removal; no heap or spatial index exists.
/// This preserves original valid safe-arithmetic ordering/tie behavior.
/// O(N squared*(D+log N)) worst-case work including sorting each neighborhood,
/// O(N+ND) ordinary state/input, no caller mutation or retained references.
/// @param data Complete original coordinate rows, arbitrary common dimension.
/// @param eps Finite nonnegative inclusive generating radius.
/// @param min_pts Positive minimum neighbors, including self, even if greater than N.
/// @return Owned actual OpticsRun.
/// @throws std::invalid_argument If complete shape/finite/radius/count controls fail.
/// @throws std::length_error If complete geometry/count or selected history quota fails.
/// @throws std::overflow_error If a computed distance cannot be finite double.
/// @throws std::bad_alloc If owned storage allocation fails.
/// @par Example
/// @code
/// auto r=dm::algorithms::optics_run({{0.},{1.}},2.,2);std::print("{}\n",r.result.core_distance); // [1, 1]
/// @endcode
/// @par References
/// - M. Ankerst, M. M. Breunig, H.-P. Kriegel and J. Sander(1999), OPTICS: Ordering Points To Identify the Clustering Structure, https://sigmodrecord.org/1999/06/25/optics-ordering-points-to-identify-the-clustering-structure/. Native candidate selection uses a linear scan, not a priority queue.
/// @see optics_extract_dbscan(), OpticsWork
OpticsRun optics_run(const std::vector<std::vector<double>>& data, double eps, std::size_t min_pts);
/// @brief Capture actual neighborhood, reachability and scan-selection decisions.
/// Complete rectangular finite coordinates, finite eps>=0 and min_pts>=1 are
/// validated before allocation/search. Zero-dimensional points are accepted;
/// empty data returns empty outputs. Self counts toward min_pts, duplicates are
/// separate observations, and radius membership is inclusive distance<=eps.
/// Ordinary finite sum-of-squares distance order is retained; nonfinite double
/// intermediates use complete long-double hypot before finite-double admission.
/// Outer starts are ascending point IDs. Seed priorities use a LINEAR scan, first
/// strict minimum, then swap-with-last removal; no heap or spatial index exists.
/// This preserves original valid safe-arithmetic ordering/tie behavior.
/// O(N squared*(D+log N)) worst-case work including sorting each neighborhood,
/// O(N+ND) ordinary state/input, no caller mutation or retained references.
/// Trace logical quota counts controls2,input coordinatesND,result entries3N;
/// each neighborhood record1+neighbor pairs2K, each update1, each selection
///1+seed/priorities2K and each visit1. Fixed fields/vector capacity excluded.
/// Default1000000,0 disables; admission precedes each retained copy/append.
/// Detailed history can be quadratic; no partial history escapes failure.
/// @param data Complete original coordinate rows, arbitrary common dimension.
/// @param eps Finite nonnegative inclusive generating radius.
/// @param min_pts Positive minimum neighbors, including self, even if greater than N.
/// @param max_entries Actual complete/online logical history quota, default1000000,0 disables.
/// @return Owned actual OpticsTrace.
/// @throws std::invalid_argument If complete shape/finite/radius/count controls fail.
/// @throws std::length_error If complete geometry/count or selected history quota fails.
/// @throws std::overflow_error If a computed distance cannot be finite double.
/// @throws std::bad_alloc If owned storage allocation fails.
/// @par Example
/// @code
/// auto t=dm::algorithms::optics_trace({{0.},{1.}},2.,2);std::print("{}\n",t.selections[0].point); // 1
/// @endcode
/// @par References
/// - M. Ankerst, M. M. Breunig, H.-P. Kriegel and J. Sander(1999), OPTICS: Ordering Points To Identify the Clustering Structure, https://sigmodrecord.org/1999/06/25/optics-ordering-points-to-identify-the-clustering-structure/. Native candidate selection uses a linear scan, not a priority queue.
/// @see optics_extract_dbscan(), OpticsWork
OpticsTrace optics_trace(
    const std::vector<std::vector<double>>& data,
    double                                  eps,
    std::size_t                             min_pts,
    std::size_t                             max_entries = 1000000);
/// @brief Apply the canonical processing-order reachability-cut labeling convention.
/// Validate complete result dimensions, permutation and nonnegative finite or+inf
/// distances before indexing. Cut must be finite and nonnegative. A reachability
/// above cut starts a cluster only if core<=cut; otherwise label-1. A reachability
/// within cut inherits current cluster. Border assignments depend on ordering;
/// no unique underlying truth or universal identical DBSCAN label sequence claimed.
/// Original result lacks generating radius; caller must ensure cut<=its eps.
/// O(N) work/output, no result mutation or retained references.
/// @param result Complete actual result/run, unchanged.
/// @param eps_cluster Finite nonnegative desired reachability-cut radius.
/// @return Owned per-original-ID labels,-1 noise and contiguous cluster IDs.
/// @throws std::invalid_argument If complete result or cut validation fails.
/// @throws std::length_error If cluster ID count exceeds int range.
/// @throws std::bad_alloc If owned storage allocation fails.
/// @par Example
/// @code
/// auto r=dm::algorithms::optics({{0.},{.5},{1.},{4.}},.75,2);std::print("{}\n",dm::algorithms::optics_extract_dbscan(r,.75)); // [0, 0, 0, -1]
/// @endcode
/// @par References
/// - M. Ankerst, M. M. Breunig, H.-P. Kriegel and J. Sander(1999), OPTICS: Ordering Points To Identify the Clustering Structure, https://sigmodrecord.org/1999/06/25/optics-ordering-points-to-identify-the-clustering-structure/. Native candidate selection uses a linear scan, not a priority queue.
/// @see optics_cluster_summary()
std::vector<int> optics_extract_dbscan(const OpticsResult& result, double eps_cluster);
/// @brief Apply the canonical processing-order reachability-cut labeling convention.
/// Validate complete result dimensions, permutation and nonnegative finite or+inf
/// distances before indexing. Cut must be finite and nonnegative. A reachability
/// above cut starts a cluster only if core<=cut; otherwise label-1. A reachability
/// within cut inherits current cluster. Border assignments depend on ordering;
/// no unique underlying truth or universal identical DBSCAN label sequence claimed.
/// Run overload additionally checks cut<=original generating eps.
/// O(N) work/output, no result mutation or retained references.
/// @param result Complete actual result/run, unchanged.
/// @param eps_cluster Finite nonnegative desired reachability-cut radius.
/// @return Owned per-original-ID labels,-1 noise and contiguous cluster IDs.
/// @throws std::invalid_argument If complete result or cut validation fails.
/// @throws std::length_error If cluster ID count exceeds int range.
/// @throws std::bad_alloc If owned storage allocation fails.
/// @par Example
/// @code
/// auto r=dm::algorithms::optics_run({{0.},{.5},{1.},{4.}},.75,2);std::print("{}\n",dm::algorithms::optics_extract_dbscan(r,.75)); // [0, 0, 0, -1]
/// @endcode
/// @par References
/// - M. Ankerst, M. M. Breunig, H.-P. Kriegel and J. Sander(1999), OPTICS: Ordering Points To Identify the Clustering Structure, https://sigmodrecord.org/1999/06/25/optics-ordering-points-to-identify-the-clustering-structure/. Native candidate selection uses a linear scan, not a priority queue.
/// @see optics_cluster_summary()
std::vector<int> optics_extract_dbscan(const OpticsRun& result, double eps_cluster);
/// @brief Count complete original-ID noise and contiguous cluster labels without refitting.
/// Accept-1 or contiguous nonnegative IDs, no empty clusters or IDs>=input size.
/// Empty/all-noise vectors have no clusters. O(N) work/storage, no input mutation.
/// @param labels Complete raw canonical extraction labels.
/// @return Owned copied labels, positive cluster sizes and noise count.
/// @throws std::invalid_argument If any label or contiguity check fails.
/// @throws std::bad_alloc If owned storage allocation fails.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::optics_cluster_summary({0,0,1,-1}).sizes); // [2, 1]
/// @endcode
/// @see optics_extract_dbscan()
OpticsClusterSummary optics_cluster_summary(const std::vector<int>& labels);
/// @brief Prepare actual processing-order reachabilities with explicit presentation-only infinity markers.
/// Complete result validation precedes indexing. Actual undefined+inf values are
/// replaced only in displayed output and flagged1; result remains unchanged.
/// A positive supplied height is used verbatim. Default0 chooses1.15 times largest
/// finite reachability, or1 when all finite values are0/absent. O(N) work/output.
/// No fitted distance or actual infinity is silently converted in the result.
/// @param result Complete original-ID result, unchanged.
/// @param undefined_height Finite nonnegative presentation ceiling,0 selects automatic.
/// @return Owned zero-based positions, finite displayed heights and undefined flags.
/// @throws std::invalid_argument If complete result or height validation fails.
/// @throws std::overflow_error If automatic ceiling cannot be finite double.
/// @throws std::bad_alloc If owned output allocation fails.
/// @par Example
/// @code
/// auto p=dm::algorithms::optics_reachability_profile(dm::algorithms::optics({{0.}},0.,2));std::print("{} {}\n",int(p.undefined[0]),p.displayed[0]); // 1 1
/// @endcode
/// @see optics_trace()
OpticsReachabilityProfile optics_reachability_profile(
    const OpticsResult& result,
    double              undefined_height = 0);
/// @brief Produce the original seeded three-density Gaussian/noise report fixture.
/// Local MT19937_64 seed30, fresh normal distribution per blob:40points at(0,0)
/// sd.20,40at(8,0)sd.70,40at(4,7)sd.35; then12uniform[-2,10] pairs.
/// Standard distribution transforms are library dependent; no cross-library bit
/// identity or labeled-cluster recovery guarantee. O(132) work/output, no RNG shared.
/// @return Fresh original-order132two-dimensional points.
/// @throws std::bad_alloc If output storage allocation fails.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::optics_demo_data().size()); // 132
/// @endcode
/// @see optics_run()
std::vector<std::vector<double>> optics_demo_data();
/// @brief Original checked distance-helper compatibility namespace.
namespace detail {
/// @brief Preserve the original detail distance name with checked complete coordinates.
/// Same finite ordinary/fallback arithmetic as the canonical neighbor producer.
/// @param a Complete finite coordinate vector.
/// @param b Matching complete finite coordinate vector.
/// @return Finite double Euclidean distance,0 for two zero-dimensional points.
/// @throws std::invalid_argument If complete shape/finite checks fail.
/// @throws std::overflow_error If distance cannot be finite double.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::detail::optics_euclidean({0,0},{3,4})); // 5
/// @endcode
/// @see optics_trace()
double optics_euclidean(const std::vector<double>& a, const std::vector<double>& b);
} // namespace detail
} // namespace datamunge::algorithms
