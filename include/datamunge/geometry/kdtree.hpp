#pragma once
/// @file kdtree.hpp
/// @brief Deterministic owned planar k-d tree with widened distances and actual nearest-query evidence.
#include <cstddef>
#include <datamunge/geometry/point2d.hpp>
#include <memory>
#include <queue>
#include <vector>
namespace datamunge::geometry {
/// @brief Actual nearest-query phase: visited point or post-near-subtree far-plane decision.
/// @par Example
/// @code
/// const auto t=dm::geometry::KDTree2D({{0,0}}).nearest_trace({1,1});
/// std::print("{}\n",t.probes[0].action==dm::geometry::KDProbeAction::Visit); // true
/// @endcode
/// @see KDNearestProbe
enum class KDProbeAction {
  Visit,   ///< Actual point distance/strict or id-tie best update.
  FarPlane ///< Actual nonnull far-child lower-bound decision after the near traversal.
};
/// @brief Owned actual point or branch decision from the producing nearest traversal.
/// Visit.measure is point distance; FarPlane.measure is absolute split-coordinate
/// difference. Only FarPlane has a meaningful child index/explore_far flag.
/// best indices use size_t max before any candidate; best distance initially infinity.
/// @par Example
/// @code
/// const auto t=dm::geometry::KDTree2D({{0,0}}).nearest_trace({1,0});
/// std::print("{} {}\n",t.probes[0].original_index,t.probes[0].best_index_after); // 0 0
/// @endcode
/// @see KDNearestTrace
struct KDNearestProbe {
  KDProbeAction action{KDProbeAction::Visit}; ///< Actual operation kind.
  std::size_t   original_index{0};            ///< Current node's original input identity.
  std::size_t   child_index{0}; ///< Actual far-child input identity, size_t max on Visit.
  int           axis{0};        ///< Actual split axis0:x,1:y.
  long double   measure{0};     ///< Actual point distance or plane bound by phase.
  std::size_t   best_index_before{
      0}; ///< Actual incumbent id before operation, max sentinel initially.
  std::size_t best_index_after{0};     ///< Actual incumbent id after operation.
  long double best_distance_before{0}; ///< Actual incumbent distance before operation.
  long double best_distance_after{0};  ///< Actual incumbent distance after operation.
  bool        explore_far{false}; ///< Actual nonnull far-child traversal decision, false on Visit.
};
/// @brief Selected actual nearest-query work, not input copying, comparisons or elapsed time.
/// One point-distance computation per visited node; plane tests only for nonnull
/// far children. Pruned subtrees need not contain only one point.
/// @par Example
/// @code
/// std::print("{}\n",dm::geometry::KDTree2D({{0,0}}).nearest_trace({1,0}).work.visited_nodes); // 1
/// @endcode
/// @see KDNearestTrace
struct KDSearchWork {
  std::size_t visited_nodes{0};   ///< Actual entered nodes/point-distance computations.
  std::size_t plane_tests{0};     ///< Actual nonnull far-child lower-bound decisions.
  std::size_t pruned_subtrees{0}; ///< Actual nonnull far children excluded by plane bound.
};
/// @brief Owned original points, actual answer and chronological nearest-query operations.
/// Trace copies original points in input order and captures actual visits/far
/// decisions, without replay. Adds O(n+records) copy/storage beyond query work;
/// no tree/input references survive. Long-double distances are rounded metrics,
/// not exact-real nearest oracles for arbitrarily close inputs.
/// @par Example
/// @code
/// const auto t=dm::geometry::KDTree2D({{0,0},{3,4}}).nearest_trace({1,0});
/// std::print("{} {}\n",t.original_index,t.distance); // 0 1
/// @endcode
/// @see KDTree2D::nearest_trace()
struct KDNearestTrace {
  Point2D     query;             ///< Owned finite query.
  Point2D     point;             ///< Owned actual selected stored point.
  std::size_t original_index{0}; ///< Actual selected original identity, including duplicate points.
  long double distance{0};       ///< Actual widened hypot distance to selected point.
  std::vector<Point2D>        points; ///< Owned complete original points, input order.
  KDSearchWork                work;   ///< Actual selected query work, excluding copies/validation.
  std::vector<KDNearestProbe> probes; ///< Actual chronological point/plane decisions.
};
/// @brief Own a balanced alternating-axis median tree for planar nearest/k/radius queries.
/// Finite points required. Stable deterministic build ties use other coordinate,
/// then original input index; each node retains original identity. Nearest and
/// k-nearest equal computed distances prefer earliest input index, including
/// duplicates. Distances use widened coordinate subtraction and hypot, avoiding
/// the old double square overflow/underflow; they remain rounded machine values.
/// Construction expected O(n log(n+1)) work/O(n) owned storage; tree height O(log n).
/// Queries can visit every node: no universal average O(log n) guarantee without
/// distribution assumptions. No insertion/removal. Const queries use local state
/// and share no mutable state; original caller data are not retained.
/// @par Example
/// @code
/// const dm::geometry::KDTree2D tree({{0,0},{3,4}});
/// std::print("{}\n",tree.nearest({1,0}).x); // 0
/// @endcode
/// @par References
/// - Bentley(1975), Multidimensional binary search trees used for associative searching, https://doi.org/10.1145/361002.361007.
/// @see nearest_trace(), k_nearest(), points_in_radius()
class KDTree2D {
 public:
  /// @brief Validate/admit points and build the owned balanced median tree.
  /// The legacy by-value parameter copy occurs before validation; finite/input
  /// admission precedes internal indexed-record/node allocation. Complete point
  /// cap counts records, excluding call-boundary copy/capacity/byte/RAM overhead.
  /// @param points Owned value argument, possibly empty; duplicates allowed.
  /// @param max_points Complete optional point ceiling, default million, zero off.
  /// @throws std::invalid_argument On a nonfinite coordinate.
  /// @throws std::length_error On signed-length/container/optional point limits.
  /// @throws std::bad_alloc On allocation failure.
  /// @par Example
  /// @code
  /// std::print("{}\n",dm::geometry::KDTree2D({{0,0}},1).size()); // 1
  /// @endcode
  explicit KDTree2D(std::vector<Point2D> points, std::size_t max_points = 1000000);
  /// @brief Report whether the original point set is empty.
  /// @return True exactly for zero stored points.
  bool empty() const { return size_ == 0; }
  /// @brief Count stored points, including distinct identities at equal coordinates.
  /// @return Original owned point count.
  std::size_t size() const { return size_; }
  /// @brief Return a nearest stored point under widened rounded hypot distance.
  /// Equal computed distances prefer the earliest original input. O(n) worst
  /// visited work/O(log n) recursion storage; no allocation/history/input copying.
  /// @param query Borrowed finite query, unchanged.
  /// @return Owned selected original coordinate, not a reference.
  /// @throws std::invalid_argument On nonfinite query, even for an empty tree.
  /// @throws std::logic_error On empty tree.
  /// @throws std::overflow_error If widened distance is nonfinite on this platform.
  /// @par Example
  /// @code
  /// std::print("{}\n",dm::geometry::KDTree2D({{0,0},{2,0}}).nearest({1,0}).x); // 0
  /// @endcode
  /// @see nearest_trace()
  Point2D nearest(const Point2D& query) const;
  /// @brief Return up to k closest coordinates, ordered by widened distance then original id.
  /// Duplicate coordinates retain multiplicity. k=0/empty returns empty after finite
  /// query validation. O(n log(min(k,n)+1)) worst work, O(k+log n) working storage.
  /// @param query Borrowed finite query.
  /// @param k Requested result count, clamped to tree size before heap use.
  /// @return Owned nearest-first coordinates, possibly fewer than k.
  /// @throws std::invalid_argument On nonfinite query.
  /// @throws std::overflow_error On nonfinite widened distance.
  /// @throws std::bad_alloc On heap/result allocation failure.
  /// @par Example
  /// @code
  /// std::print("{}\n",dm::geometry::KDTree2D({{0,0},{2,0}}).k_nearest({1,0},5).size()); // 2
  /// @endcode
  /// @see nearest(), points_in_radius()
  std::vector<Point2D> k_nearest(const Point2D& query, std::size_t k) const;
  /// @brief Return all stored coordinates with widened distance<=radius, in tree traversal order.
  /// Inclusive boundary, duplicate multiplicity. Negative/NaN radius rejects;
  /// positive infinity includes every point without requiring a finite norm.
  /// O(n) worst work, O(log n) recursion plus O(results) owned output.
  /// @param query Borrowed finite query, validated even for empty tree.
  /// @param radius Nonnegative radius in coordinate units, positive infinity allowed.
  /// @return Owned matching coordinates, no guaranteed distance sorting.
  /// @throws std::invalid_argument On nonfinite query or negative/NaN radius.
  /// @throws std::overflow_error On nonfinite widened distance for a finite-radius query.
  /// @throws std::bad_alloc On result allocation failure.
  /// @par Example
  /// @code
  /// std::print("{}\n",dm::geometry::KDTree2D({{0,0},{2,0}}).points_in_radius({1,0},1).size()); // 2
  /// @endcode
  /// @see k_nearest()
  std::vector<Point2D> points_in_radius(const Point2D& query, double radius) const;
  /// @brief Own the original cloud and record actual nearest traversal operations once.
  /// Same answer/arithmetic/ties as nearest(). Complete point copies add O(n);
  /// each point/plane record is admitted before its distance/decision. Default
  /// million-record cap, zero off; excludes vector capacity/byte/RAM overhead.
  /// Failed call exposes no partial trace and never mutates the tree.
  /// @param query Borrowed finite query.
  /// @param max_entries Actual point/plane record ceiling, default million, zero off.
  /// @return Owned original cloud, actual nearest answer/work/chronology.
  /// @throws std::invalid_argument On nonfinite query.
  /// @throws std::logic_error On empty tree.
  /// @throws std::length_error On enabled history ceiling/container/work overflow.
  /// @throws std::overflow_error On nonfinite widened distance.
  /// @throws std::bad_alloc On capture allocation failure.
  /// @par Example
  /// @code
  /// std::print("{}\n",dm::geometry::KDTree2D({{0,0}}).nearest_trace({1,0}).work.visited_nodes); // 1
  /// @endcode
  /// @see KDNearestProbe
  KDNearestTrace nearest_trace(const Point2D& query, std::size_t max_entries = 1000000) const;

 private:
  /// @brief Owned point with original identity during deterministic median partitioning.
  struct Item {
    Point2D     point; ///< Original owned coordinates.
    std::size_t index; ///< Original input identity.
  };
  /// @brief Owned balanced node with original point identity and split axis.
  struct Node {
    Item                  item;    ///< Actual median coordinate and original identity.
    int                   axis{0}; ///< Actual split axis0/1.
    std::unique_ptr<Node> left;    ///< Owned lower ordered half.
    std::unique_ptr<Node> right;   ///< Owned upper ordered half.
  };
  /// @brief Heap entry borrowing one node during one k-nearest query only.
  struct Entry {
    long double distance; ///< Actual widened point distance.
    const Node* node;     ///< Local borrowed node, never returned.
  };
  /// @brief Order heap entries nearest-first for a max-heap of the worst retained candidate.
  struct Compare {
    /// @brief Compare distance then original identity.
    /// @param a First local heap entry.
    /// @param b Second local heap entry.
    /// @return True when a ranks before b.
    bool operator()(const Entry& a, const Entry& b) const;
  };
  using Heap = std::priority_queue<
      Entry,
      std::vector<Entry>,
      Compare>;                   ///< Local worst-retained-candidate max-heap.
  std::unique_ptr<Node> root_;    ///< Owned balanced root; null for empty input.
  std::size_t           size_{0}; ///< Original point count, duplicates included.

  /// @brief Median-partition and own one balanced subtree.
  /// @param items Mutable owned construction records.
  /// @param lo Inclusive subtree beginning.
  /// @param hi Exclusive subtree end.
  /// @param depth Current depth controlling alternating axis.
  /// @return Owned subtree or null if empty.
  /// @throws std::bad_alloc On node allocation failure.
  static std::unique_ptr<Node>
  build(std::vector<Item>& items, std::size_t lo, std::size_t hi, std::size_t depth);
  /// @brief Shared actual nearest traversal, optionally recording operations.
  /// @param node Borrowed current node, possibly null.
  /// @param query Finite query.
  /// @param best Mutable actual incumbent, initially infinite/max sentinel.
  /// @param trace Optional owned producing trace.
  /// @param cap Optional actual-record ceiling, zero off.
  /// @throws std::length_error On recording/work limit.
  /// @throws std::overflow_error On nonfinite widened distance.
  /// @throws std::bad_alloc On history allocation failure.
  static void nearest_recurse(
      const Node*     node,
      const Point2D&  query,
      Entry&          best,
      KDNearestTrace* trace,
      std::size_t     cap);
  /// @brief Shared exact tree schedule for k-nearest widened candidates.
  /// @param node Borrowed current node.
  /// @param query Finite query.
  /// @param k Positive admitted result count.
  /// @param heap Local worst-retained heap.
  /// @throws std::overflow_error On nonfinite widened distance.
  /// @throws std::bad_alloc On heap allocation failure.
  static void k_recurse(const Node* node, const Point2D& query, std::size_t k, Heap& heap);
  /// @brief Traverse a finite/infinite-radius query with inclusive plane tests.
  /// @param node Borrowed current node.
  /// @param query Finite query.
  /// @param radius Nonnegative actual radius, possibly infinity.
  /// @param out Owned output under construction.
  /// @throws std::overflow_error On nonfinite distance for a finite radius.
  /// @throws std::bad_alloc On output allocation failure.
  static void radius_recurse(
      const Node*           node,
      const Point2D&        query,
      long double           radius,
      std::vector<Point2D>& out);
  /// @brief Copy actual node coordinates into original input order without rebuilding the tree.
  /// @param node Borrowed current node.
  /// @param points Preallocated complete original-index output.
  static void original_points(const Node* node, std::vector<Point2D>& points);
};
} // namespace datamunge::geometry
