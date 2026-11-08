#pragma once
/// @file closest_pair_search.hpp
/// @brief Indexed all-pairs proximity search, actual strict improvements and reproducible rectangle sampling.
#include <datamunge/geometry/point2d.hpp>
#include <cstddef>
#include <cstdint>
#include <vector>
namespace datamunge::geometry {
/// @brief Owned closest input identities, copied coordinates and widened measured distance.
/// first<second refer to original input positions, including distinct records with
/// identical coordinates. Distances are rounded long-double hypot evaluations,
/// not exact predicates. Equal measured distances keep the lexicographically first
/// index pair. Default storage alone is not a valid search result. Calls own copies.
/// @par Example
/// @code
/// const auto r=dm::geometry::closest_pair_indexed({{0,0},{3,4},{0,1}});
/// std::print("{} {} {}\n",r.first,r.second,r.distance); // 0 2 1
/// @endcode
/// @see closest_pair_indexed(), closest_pair()
struct ClosestPairSelection {
  std::size_t first{0}; ///< Lower original input index of the selected distinct records.
  std::size_t second{0}; ///< Higher original input index of the selected distinct records.
  Point2D a; ///< Owned coordinate copy at first.
  Point2D b; ///< Owned coordinate copy at second.
  long double distance{0}; ///< Actual widened Euclidean distance; can exceed finite double range.
};
/// @brief Pair-work and optional actual-improvement recording limits.
/// Zero means unlimited. The complete n(n-1)/2 pair forecast is checked after all
/// coordinates validate, before execution. max_improvements is an online cap on
/// actual trace records, including initialization; it is ignored by untraced calls.
/// Its required count depends on data/order and is not forecast as every pair.
/// Failed work/record/range/allocation checks expose no partial result or trace.
/// @par Example
/// @code
/// dm::geometry::ClosestPairOptions o;o.max_pair_visits=3;o.max_improvements=2;
/// std::print("{}\n",dm::geometry::closest_pair_trace({{0,0},{3,4},{0,1}},o).improvements.size()); // 2
/// @endcode
/// @see closest_pair_trace(), ClosestPairWork
struct ClosestPairOptions {
  std::size_t max_pair_visits{0}; ///< Positive complete unordered-pair cap; zero unlimited.
  std::size_t max_improvements{0}; ///< Positive actual improvement-record cap, tracing only; zero unlimited.
};
/// @brief Owned actual strict incumbent improvement from the executing pair scan.
/// The first comparison initializes the incumbent and is included. Later records
/// have strictly smaller measured distances; ties never replace the incumbent.
/// Indices retain original identities, and comparison is one-based pair-scan work.
/// No candidate evaluation is replayed to build these records.
/// @par Example
/// @code
/// const auto s=dm::geometry::closest_pair_trace({{0,0},{3,4},{0,1}}).improvements.back();
/// std::print("{} {} {}\n",s.comparison,s.first,s.second); // 2 0 2
/// @endcode
/// @see ClosestPairTrace
struct ClosestPairImprovement {
  std::size_t comparison{0}; ///< Actual one-based unordered-pair visit at capture.
  std::size_t first{0}; ///< Lower original input index at this improvement.
  std::size_t second{0}; ///< Higher original input index at this improvement.
  long double distance{0}; ///< Actual new rounded widened incumbent distance.
};
/// @brief Actual validation, distance and incumbent work, excluding allocator costs.
/// All unordered pairs are visited in i-major/j-minor ascending order, even after
/// finding zero distance. Ties count candidates equal to the current incumbent,
/// excluding initialization; zero_pairs counts every measured zero-distance pair.
/// Counts are not elapsed time and do not measure exact-real arithmetic costs.
/// @par Example
/// @code
/// const auto w=dm::geometry::closest_pair_trace({{0,0},{1,0},{0,1}}).work;
/// std::print("{} {} {}\n",w.pair_visits,w.improvements,w.ties); // 3 1 1
/// @endcode
/// @see ClosestPairTrace
struct ClosestPairWork {
  std::size_t validated_points{0}; ///< Actual finite-coordinate validation visits.
  std::size_t pair_visits{0}; ///< Actual distinct unordered candidate pairs visited.
  std::size_t distance_evaluations{0}; ///< Actual widened hypot evaluations, one per pair.
  std::size_t improvements{0}; ///< Actual strict incumbent changes, including first-pair initialization.
  std::size_t ties{0}; ///< Candidates equal to the current rounded incumbent after initialization.
  std::size_t zero_pairs{0}; ///< Actual measured zero-distance pairs; no early stop is taken.
};
/// @brief Owned actual improvements and final indexed result from one all-pairs execution.
/// Shares the executing kernel with ordinary search without replay. Stores only
/// K strict improvements, not a full candidate history: O(K) additional storage,
/// with K at most n(n-1)/2. The final improvement is the returned selection.
/// Input coordinates are copied into result only; records retain ids and distances.
/// No borrowed input references survive and calls share no mutable state.
/// @par Example
/// @code
/// const auto t=dm::geometry::closest_pair_trace({{0,0},{3,4},{0,1}});
/// std::print("{} {}\n",t.result.distance,t.work.pair_visits); // 1 3
/// @endcode
/// @see closest_pair_trace(), ClosestPairSelection
struct ClosestPairTrace {
  ClosestPairSelection result; ///< Actual final original-index selection and coordinate copies.
  std::vector<ClosestPairImprovement> improvements; ///< Actual first/strictly improving incumbent states.
  ClosestPairWork work; ///< Actual named validation/pair/update counters.
};
/// @brief Find the closest distinct input records by a checked widened all-pairs scan.
/// Requires at least two finite points. Validates the whole input before a positive
/// pair cap, then visits every i<j in ascending index order. Uses widened coordinate
/// differences and hypot per pair, avoiding unsafe double squares; equal rounded
/// distances retain the first pair. Duplicate coordinates are allowed and remain
/// distinct input identities. No early stop, sorting or spatial pruning is used.
/// O(n^2) work and O(1) auxiliary storage; no input mutation or shared state.
/// Rounded hypot comparison is not an exact geometric predicate. The widened
/// result can exceed double range; closest_pair() checks legacy conversion.
/// @param points Borrowed original finite planar coordinates, length at least two.
/// @param options Positive complete pair cap; trace-record cap ignored, defaults unlimited.
/// @return Owned selected original ids/coordinates and actual widened distance.
/// @throws std::invalid_argument If fewer than two points or any coordinate is nonfinite.
/// @throws std::length_error If the pair count exceeds size_t or a positive work cap.
/// @throws std::overflow_error If widened arithmetic or actual counts cannot fit.
/// @par Example
/// @code
/// const auto r=dm::geometry::closest_pair_indexed({{2,3},{2,3},{8,9}});
/// std::print("{} {} {}\n",r.first,r.second,r.distance); // 0 1 0
/// @endcode
/// @par References
/// - M. I. Shamos and D. Hoey (1975), Closest-point problems, FOCS, 151-162,
///   https://doi.org/10.1109/SFCS.1975.8. Background for the faster geometric
///   problem; this implementation deliberately enumerates all pairs instead.
/// @see closest_pair(), closest_pair_trace()
ClosestPairSelection closest_pair_indexed(const std::vector<Point2D>& points,ClosestPairOptions options={});
/// @brief Capture actual strict incumbent improvements while executing the same checked scan.
/// Shares the indexed kernel without replay. Includes first-pair initialization;
/// ties are counted but not recorded. Complete work caps preflight after validation;
/// positive actual record caps are enforced online before inserting an improvement.
/// No partial trace escapes failure. O(n^2) work and O(K) additional owned storage,
/// where K is the actual improvement count, at worst n(n-1)/2.
/// @param points Borrowed original finite planar coordinates, length at least two.
/// @param options Complete work cap and online actual-record cap; defaults unlimited.
/// @return Owned actual improvements, final selection and named work.
/// @throws std::invalid_argument If point count or any coordinate is invalid.
/// @throws std::length_error If supported pair/storage count or a positive work/record cap is exceeded.
/// @throws std::overflow_error If widened arithmetic or actual counters cannot fit.
/// @throws std::bad_alloc If trace allocation fails.
/// @par Example
/// @code
/// const auto t=dm::geometry::closest_pair_trace({{0,0},{3,4},{0,1}});
/// std::print("{} {}\n",t.improvements.size(),t.improvements.back().comparison); // 2 2
/// @endcode
/// @see ClosestPairTrace, ClosestPairOptions
ClosestPairTrace closest_pair_trace(const std::vector<Point2D>& points,ClosestPairOptions options={});
/// @brief Generate owned reproducible finite points in an axis-aligned rectangle.
/// A local std::mt19937 seeded by seed draws x then y using separate
/// std::uniform_real_distribution<double> objects. Each coordinate lies in its
/// half-open interval; an upper-rounded draw is clamped to the adjacent interior
/// value. Bounds are finite/strictly ordered with interval widths fitting finite
/// double. Count zero returns empty after validating bounds. The engine is standard,
/// but distribution-to-floating mapping can differ across standard libraries;
/// reproducibility is within the selected implementation, not a portable bitstream.
/// These are rounded pseudorandom samples, not exact continuous/cryptographic draws.
/// O(count) work/storage; local state only, no global RNG mutation.
/// @param count Number of points, including zero.
/// @param seed Local 32-bit MT19937 seed, default zero.
/// @param x_min Finite inclusive horizontal lower bound, default zero.
/// @param x_max Finite exclusive horizontal upper bound, default ten.
/// @param y_min Finite inclusive vertical lower bound, default zero.
/// @param y_max Finite exclusive vertical upper bound, default ten.
/// @return Owned finite samples in input generation order.
/// @throws std::invalid_argument If bounds are nonfinite/unordered or a width exceeds finite double.
/// @throws std::length_error If vector capacity cannot support count.
/// @throws std::overflow_error If a generated coordinate is nonfinite.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// const auto p=dm::geometry::uniform_rectangle_points(60,424242);
/// std::print("{}\n",p.size()); // 60
/// @endcode
/// @par References
/// - ISO C++ random-number library: std::mt19937 and std::uniform_real_distribution.
/// @see closest_pair_indexed(), closest_pair_trace()
std::vector<Point2D> uniform_rectangle_points(std::size_t count,std::uint32_t seed=0,double x_min=0,double x_max=10,double y_min=0,double y_max=10);
}
