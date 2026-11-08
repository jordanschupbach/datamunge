#pragma once
/// @file d_star.hpp
/// @brief Persistent exact unit-grid D* Lite, actual repair evidence and checked route rendering.
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>
#include <datamunge/plot/plot.hpp>

namespace datamunge::algorithms {
/// @brief Exact infinity sentinel for D* Lite labels and both components of an empty queue key.
/// All finite labels are at most cell_count-1; priorities include a checked integer
/// moving-start modifier. This value is not a finite path cost.
inline constexpr std::uint64_t d_star_infinity=std::numeric_limits<std::uint64_t>::max();

/// @brief Owned grid route preserving the historical field order and failure convention.
/// Coordinates are(row,column), endpoints included, four-neighbor unit steps.
/// On failure path is empty,cost zero,found false; cost is meaningful only if found.
/// @par Example
/// @code
/// const auto p=dm::algorithms::d_star_replan({{false,false},{false,false}},{0,0},{1,1});
/// std::print("{} {:.0f}\n",p.found,p.cost); // true 2
/// @endcode
/// @see d_star_replan(), DStarLiteGrid
struct GridPathResult {
  std::vector<std::pair<int,int>> path; ///< Owned inclusive row/column route; empty when not found.
  double cost{0}; ///< Number of unit steps on success, exactly representable; zero on failure.
  bool found{false}; ///< Whether a route exists between unblocked endpoints.
};

/// @brief Lexicographic integer priority(min(g,rhs)+Manhattan(start,u)+km,min(g,rhs)).
/// Both components use d_star_infinity for an infinite minimum. Remaining heap
/// ties use flattened cell id, which is not part of the mathematical key.
/// @par Example
/// @code
/// const dm::algorithms::DStarKey key{2,0};
/// std::print("{}\n",key.first); // 2
/// @endcode
using DStarKey=std::pair<std::uint64_t,std::uint64_t>;

/// @brief One validated occupancy change, applied in input order by update_cells().
/// Repeated coordinates are allowed; the last value wins. Default is an unblocked
/// origin update, not a request to initialize a planner.
/// @par Example
/// @code
/// dm::algorithms::DStarLiteGrid p({{false,false},{false,false}},{0,0},{1,1});
/// p.update_cells({{{0,1},true}});
/// std::print("{:.0f}\n",p.repair().route.cost); // 2
/// @endcode
/// @see DStarLiteGrid::update_cells()
struct GridCellUpdate {
  std::pair<int,int> cell{0,0}; ///< In-bounds(row,column).
  bool blocked{false}; ///< True closes every incident edge; false restores unit-cost edges.
};

/// @brief Logical work since construction or the previous successful repair.
/// Includes pending map/start updates and current repair. Counts exclude input
/// validation, heap comparisons/swaps, snapshots and final route extraction.
/// Neighbor checks count actual Bellman backup neighbors, including blocked ones.
/// Work is not elapsed time; a resumed capped repair may include prior completed pops.
/// @par Example
/// @code
/// dm::algorithms::DStarLiteGrid p({{false}},{0,0},{0,0});
/// const auto r=p.repair();
/// std::print("{} {}\n",r.work.pops,r.work.lowerings); // 1 1
/// @endcode
/// @see DStarRepair, DStarTrace
struct DStarWork {
  std::size_t changed_cells=0; ///< Effective occupancy-bit changes, including repeated edits.
  std::size_t start_moves=0; ///< Calls that actually change the start coordinate.
  std::size_t vertex_updates=0; ///< UpdateVertex calls, including goal/blocked backups.
  std::size_t neighbor_checks=0; ///< In-bounds neighbors visited by nonblocked nongoal Bellman backups.
  std::size_t queue_inserts=0; ///< Absent-to-present queue insertions, including initialization.
  std::size_t queue_removals=0; ///< Present-to-absent removals, including popped entries.
  std::size_t pops=0; ///< Actual minimum-key removals during repairs.
  std::size_t lowerings=0; ///< Overconsistent pops assigning g=rhs.
  std::size_t raisings=0; ///< Underconsistent pops invalidating g to infinity.
  std::size_t key_refreshes=0; ///< Old moving-start key reinsertions, without a label expansion.
  std::size_t queue_peak=0; ///< Largest live indexed-queue size, including interval's initial queue.
};

/// @brief Owned route and actual interval work from one successful repair call.
/// Default fields are an uncomputed placeholder. All caller data remain owned by
/// the planner or this return value; no borrowed references are retained here.
/// @par Example
/// @code
/// dm::algorithms::DStarLiteGrid p({{false,false}},{0,0},{0,1});
/// std::print("{:.0f}\n",p.repair().route.cost); // 1
/// @endcode
/// @see DStarLiteGrid::repair()
struct DStarRepair {
  GridPathResult route; ///< Optimal route for current occupancy/start/fixed goal.
  DStarWork work; ///< Actual interval counters, reset in planner only after successful return preparation.
};

/// @brief Actual D* Lite pop classification.
/// A key refresh requeues an old moving-start key; lower sets g=rhs; raise sets
/// g=infinity and recomputes this vertex and its predecessors.
/// @par Example
/// @code
/// dm::algorithms::DStarLiteGrid p({{false}},{0,0},{0,0});
/// const auto t=p.repair_trace();
/// std::print("{}\n",t.steps[0].action==dm::algorithms::DStarAction::lower); // true
/// @endcode
/// @see DStarStep
enum class DStarAction {rekey, ///< Old key smaller than current key; refresh without expanding labels.
                        lower, ///< g>rhs; commit improved g and update predecessors.
                        raise ///< g<rhs; invalidate g and update self/predecessors.
};

/// @brief Owned scalar operands/results of one actual popped vertex, without replay.
/// rhs_after includes any self update; predecessor backups may affect other vertices.
/// @par Example
/// @code
/// dm::algorithms::DStarLiteGrid p({{false}},{0,0},{0,0});
/// const auto t=p.repair_trace();
/// std::print("{} {}\n",t.steps[0].vertex,t.steps[0].g_after); // 0 0
/// @endcode
/// @see DStarTrace
struct DStarStep {
  std::size_t vertex=0; ///< Flattened row-major id, row*columns+column.
  DStarKey old_key{d_star_infinity,d_star_infinity}; ///< Stored key actually removed.
  DStarKey current_key{d_star_infinity,d_star_infinity}; ///< Calculated key before modifying this pop.
  std::uint64_t g_before=d_star_infinity; ///< Actual pre-pop committed label.
  std::uint64_t rhs_before=d_star_infinity; ///< Actual pre-pop one-step label.
  std::uint64_t g_after=d_star_infinity; ///< Actual committed label after this operation.
  std::uint64_t rhs_after=d_star_infinity; ///< Actual one-step label after this operation.
  DStarAction action=DStarAction::rekey; ///< Actual branch taken.
};

/// @brief Owned initial/final labels, original live queue and actual repair pops.
/// Entry accounting is5*N+initial_open.size()+steps.size(): grid cells,four label
/// arrays,one per initial queue record and one per pop record. Scalar metadata,
/// planner working storage, paths and allocator bytes are excluded. The complete
/// base cost is checked before copies; each new pop entry before queue/label mutation.
/// On a trace cap exception completed earlier pops remain in the planner; a later
/// repair resumes correctly. work_before distinguishes already pending work from
/// pops captured in this call. Initial/final labels outside the needed frontier
/// can remain stale/infinite; only current-start optimality is guaranteed at termination.
/// @par Example
/// @code
/// dm::algorithms::DStarLiteGrid p({{false}},{0,0},{0,0});
/// const auto t=p.repair_trace();
/// std::print("{} {}\n",t.steps.size(),t.final_g[0]); // 1 0
/// @endcode
/// @see DStarLiteGrid::repair_trace()
struct DStarTrace {
  std::vector<std::vector<bool>> blocked; ///< Owned occupancy at this repair, unchanged during capture.
  std::pair<int,int> start{0,0}; ///< Query start at capture.
  std::pair<int,int> goal{0,0}; ///< Fixed goal.
  std::uint64_t key_modifier=0; ///< Checked accumulated Manhattan movement.
  std::vector<std::uint64_t> initial_g; ///< Row-major actual committed labels before this call.
  std::vector<std::uint64_t> initial_rhs; ///< Row-major actual one-step labels before this call.
  std::vector<std::uint64_t> final_g; ///< Row-major committed labels at successful termination.
  std::vector<std::uint64_t> final_rhs; ///< Row-major one-step labels at successful termination.
  std::vector<std::pair<std::size_t,DStarKey>> initial_open; ///< Actual indexed-heap entries and stored keys.
  std::vector<DStarStep> steps; ///< Actual pops of this call only.
  DStarKey final_top{d_star_infinity,d_star_infinity}; ///< Minimum stored key at termination, or infinity pair.
  DStarWork work_before; ///< Pending interval work before this capture's pops.
  DStarRepair repair; ///< Optimal route and work since previous successful repair.
  std::size_t captured_entries=0; ///< Logical count described above.
};

/// @brief Persistent D* Lite for a rectangular four-neighbor unit-cost occupancy grid.
/// Owns g/rhs labels and an indexed binary heap with at most one entry per vertex.
/// Planning runs backward from a fixed goal; Manhattan priorities and a checked
/// movement modifier let changed edges/new starts reuse prior state. Cell bounds
/// must fit int and total cells<=INT_MAX. A shortest simple route has at most N-1
/// edges; backups at N or more are saturated to infinity, retaining every possible
/// optimum and bounding integer labels. Priorities use exact uint64_t arithmetic;
/// modifier headroom of4*N is reserved before moving the start. Blocked goal has
/// rhs=infinity; unblocking restores its zero boundary. Blocked endpoints yield no
/// route. No diagonals, weighted terrain, moving goal, sensor inference or automatic
/// movement are included. Copying owns an independent full planner. Concurrent
/// mutation of one instance needs synchronization. Each repair can touch the whole
/// grid; there is no guaranteed speedup over fresh search. Heap operations are
/// O(log N), backup degree<=4, persistent storage O(N), plus caller-requested trace.
/// @par Example
/// @code
/// dm::algorithms::DStarLiteGrid p({{false,false},{false,false}},{0,0},{1,1});
/// (void)p.repair();
/// std::print("{}\n",p.repair().work.pops); // 0
/// @endcode
/// @par References
/// - S. Koenig and M. Likhachev (2002), D* Lite, AAAI,476-483.
/// https://www.cs.cmu.edu/~motionplanning/papers/sbp_papers/integrated3/koenig_dstarlite_aaai02b.pdf
/// @see d_star_replan(), DStarTrace
class DStarLiteGrid {
public:
  /// @brief Initialize owned occupancy, infinite labels and the free goal's zero rhs.
  /// Does not run repair. All shapes/endpoints/cell limits are validated first.
  /// Buffers and indexed-heap capacity are allocated once; queue updates allocate
  /// no additional heap storage. Constructor work includes the initial goal insert.
  /// @param blocked Nonempty rectangular grid; true cells close all incident edges.
  /// @param start In-bounds initial(row,column), including blocked cells.
  /// @param goal In-bounds fixed goal, including blocked cells.
  /// @param max_cells Optional complete-cell cap, default1000000; zero disables only cap.
  /// @throws std::invalid_argument For empty/ragged shape or out-of-bounds endpoints.
  /// @throws std::length_error For shape/count/implementation/cell-budget limits.
  /// @throws std::bad_alloc On storage allocation failure.
  /// @par Example
  /// See DStarLiteGrid's worked initialization/repair example.
  /// @see repair()
  DStarLiteGrid(const std::vector<std::vector<bool>>& blocked,std::pair<int,int> start,
                std::pair<int,int> goal,std::size_t max_cells=1000000);

  /// @brief Deep-copy a planner while retaining preallocated full-cell heap capacity.
  /// Rebuilds owned buffers, then copies labels, stored keys, queue positions and
  /// pending work. O(N) time/storage; later queue updates still allocate no storage.
  /// @param other Borrowed valid planner, unchanged.
  /// @throws std::bad_alloc If independent buffers cannot be allocated.
  /// @par Example
  /// @code
  /// dm::algorithms::DStarLiteGrid a({{false,false}},{0,0},{0,1});
  /// auto b=a;b.update_cells({{{0,0},true}});
  /// std::print("{} {}\n",a.repair().route.found,b.repair().route.found); // true false
  /// @endcode
  /// @see operator=()
  DStarLiteGrid(const DStarLiteGrid& other)
    :DStarLiteGrid(other.blocked_,other.start_,other.goal_,0) {
    modifier_=other.modifier_;g_=other.g_;rhs_=other.rhs_;heap_=other.heap_;heap_.reserve(count_);
    position_=other.position_;keys_=other.keys_;pending_=other.pending_;
  }

  /// @brief Replace a planner with an independent copy, preserving old state on allocation failure.
  /// Uses a fully constructed replacement before nonallocating swaps. Self assignment
  /// is a no-op. O(N) time/new storage; borrowed grid references are invalidated.
  /// @param other Valid source planner, unchanged.
  /// @return Reference to this planner, valid until its destruction.
  /// @throws std::bad_alloc If replacement buffers cannot be allocated; this is unchanged.
  /// @par Example
  /// @code
  /// dm::algorithms::DStarLiteGrid a({{false,false}},{0,0},{0,1});
  /// dm::algorithms::DStarLiteGrid b({{false}},{0,0},{0,0});
  /// b=a;std::print("{:.0f}\n",b.repair().route.cost); // 1
  /// @endcode
  /// @see DStarLiteGrid(const DStarLiteGrid&)
  DStarLiteGrid& operator=(const DStarLiteGrid& other) {
    if(this!=&other) {
      DStarLiteGrid replacement(other);using std::swap;
      swap(rows_,replacement.rows_);swap(columns_,replacement.columns_);swap(count_,replacement.count_);
      swap(blocked_,replacement.blocked_);swap(start_,replacement.start_);swap(goal_,replacement.goal_);
      swap(modifier_,replacement.modifier_);swap(g_,replacement.g_);swap(rhs_,replacement.rhs_);
      swap(heap_,replacement.heap_);swap(position_,replacement.position_);swap(keys_,replacement.keys_);
      swap(pending_,replacement.pending_);
    }
    return *this;
  }

  /// @brief Apply a validated batch of occupancy edits without discarding any g labels.
  /// Validates all coordinates before mutation. Edits are applied in order; repeats
  /// are allowed and only effective bit changes count. Recomputes each changed cell
  /// and its in-bounds neighbors, updating queue membership. O(k log N), no allocation.
  /// @param changes Borrowed edits; empty batch and unchanged bits are no-ops.
  /// @throws std::invalid_argument If any coordinate is outside this grid; no edits applied.
  /// @throws std::length_error If conservative work-counter headroom is unavailable, before edits.
  /// @par Example
  /// See GridCellUpdate's executed blocked-cell example.
  /// @see repair(), move_start()
  void update_cells(const std::vector<GridCellUpdate>& changes);

  /// @brief Change the query start and accumulate its Manhattan key modifier without clearing the heap.
  /// Any in-bounds new start is accepted, including a blocked cell or nonadjacent
  /// relocation. No physical movement is inferred. Same coordinate is a no-op.
  /// O(1), with exact checked modifier headroom; no allocation.
  /// @param start New(row,column).
  /// @throws std::invalid_argument For out-of-bounds start, before mutation.
  /// @throws std::overflow_error If modifier would exhaust reserved integer key headroom.
  /// @throws std::length_error If work-counter headroom is unavailable, before mutation.
  /// @par Example
  /// @code
  /// dm::algorithms::DStarLiteGrid p({{false,false}},{0,0},{0,1});
  /// (void)p.repair();p.move_start({0,1});
  /// std::print("{:.0f}\n",p.repair().route.cost); // 0
  /// @endcode
  /// @see repair()
  void move_start(std::pair<int,int> start);

  /// @brief Repair inconsistent labels and return an optimal current route plus interval work.
  /// Stops when the start is locally consistent and no smaller queue key remains.
  /// Retains unused frontier entries for later calls. Equal route choices use the
  /// smallest row-major successor id; successful cost equals path.size()-1.
  /// No capture history is kept. Pending work resets only after successful result
  /// preparation; allocation failure during route output leaves valid repaired state.
  /// @return Owned inclusive route, or the historical false/empty/zero failure, and actual work.
  /// @throws std::length_error For conservative work-counter headroom limits.
  /// @throws std::bad_alloc If result storage allocation fails.
  /// @throws std::logic_error If internal queue/path invariants are violated.
  /// @par Example
  /// See DStarLiteGrid's executed cold and unchanged-repair example.
  /// @see repair_trace()
  DStarRepair repair();

  /// @brief Run the same repair loop with owned initial/final fields and actual pops, without replay.
  /// See DStarTrace for logical entry accounting. Cap failures occur before the next
  /// pop; completed prior pops remain usable, and repair()/repair_trace() can resume.
  /// Working memory is O(N); capture adds O(N+P), where P is this call's pop count.
  /// @param max_trace_entries Optional complete base/per-pop cap, default1000000; zero disables only cap.
  /// @return Owned input/labels/open entries/pop operands/final route and work.
  /// @throws std::length_error For cap/count/work headroom before affected capture/step.
  /// @throws std::bad_alloc For captured/result storage; already completed progress is retained.
  /// @throws std::logic_error If internal path/queue invariants are violated.
  /// @par Example
  /// See DStarTrace's executed single-cell trace example.
  /// @see repair()
  DStarTrace repair_trace(std::size_t max_trace_entries=1000000);

  /// @brief Borrow current owned occupancy, valid until this planner is mutated/destroyed.
  /// @return Const grid reference; caller cannot modify it through this reference.
  const std::vector<std::vector<bool>>& grid() const noexcept {return blocked_;}
  /// @brief Current start coordinate.
  /// @return Owned(row,column).
  std::pair<int,int> start() const noexcept {return start_;}
  /// @brief Fixed goal coordinate.
  /// @return Owned(row,column).
  std::pair<int,int> goal() const noexcept {return goal_;}
  /// @brief Number of inconsistent vertices retained in the indexed queue.
  /// @return Live queue size, at most cell count.
  std::size_t queued() const noexcept {return heap_.size();}

private:
  /// @brief Convert a validated coordinate to a row-major id.
  /// @param cell In-bounds coordinate.
  /// @return Flattened id.
  std::size_t index(std::pair<int,int> cell) const;
  /// @brief List in-bounds four-neighbors in down/up/right/left order.
  /// @param vertex Valid id.
  /// @param count Receives neighbor count0..4.
  /// @return Fixed array, with only first count entries valid.
  std::array<std::size_t,4> neighbors(std::size_t vertex,std::size_t& count) const;
  /// @brief Compute current exact lexicographic key for one vertex.
  /// @param vertex Valid id.
  /// @return Finite checked-headroom key, or the infinity pair.
  DStarKey key(std::size_t vertex) const;
  /// @brief Compare stored heap keys, breaking mathematical ties by vertex id.
  /// @param a First id.
  /// @param b Second id.
  /// @return Whether a precedes b.
  bool before(std::size_t a,std::size_t b) const;
  /// @brief Swap two heap positions and maintain their inverse indices.
  /// @param a Valid heap position.
  /// @param b Valid heap position.
  void swap_heap(std::size_t a,std::size_t b);
  /// @brief Restore upward heap order from one valid position.
  /// @param position Valid heap index.
  void sift_up(std::size_t position);
  /// @brief Restore downward heap order from one valid position.
  /// @param position Valid heap index.
  void sift_down(std::size_t position);
  /// @brief Remove a vertex if queued, updating inverse positions and logical removal work.
  /// @param vertex Valid id.
  void erase(std::size_t vertex);
  /// @brief Insert an absent vertex with a stored key, using preallocated capacity.
  /// @param vertex Absent id.
  /// @param priority Stored key.
  void insert(std::size_t vertex,DStarKey priority);
  /// @brief Recompute one Bellman rhs and its exact queue membership.
  /// @param vertex Valid id, including goal/blocked cells.
  void update_vertex(std::size_t vertex);
  /// @brief Reserve conservative logical counter headroom before a mutating operation.
  /// @param units Cell count for repair, edit count for updates or1 for movement.
  /// @throws std::length_error Before mutation if multiplication/counter headroom cannot fit.
  void work_room(std::size_t units) const;
  /// @brief Shared repair loop; trace optionally receives actual pre/post operands and fields.
  /// @param trace Null for ordinary repair, otherwise output accumulator.
  /// @param limit Optional logical cap.
  /// @return Route and accumulated interval work; resets pending work after output succeeds.
  /// @throws std::length_error On budget/work bounds; allocation/output invariants can also fail.
  DStarRepair run(DStarTrace* trace,std::size_t limit);
  /// @brief Extract a strict g-descent route with smallest-id ties.
  /// @return Owned valid route, or empty failure for blocked/unreachable endpoints.
  /// @throws std::logic_error If a finite start cannot yield a valid descending route.
  GridPathResult route() const;

  int rows_=0; ///< Validated positive row count.
  int columns_=0; ///< Validated positive column count.
  std::size_t count_=0; ///< Validated total cells, at most INT_MAX.
  std::vector<std::vector<bool>> blocked_; ///< Owned current occupancy.
  std::pair<int,int> start_; ///< Current query coordinate.
  std::pair<int,int> goal_; ///< Fixed terminal coordinate.
  std::uint64_t modifier_=0; ///< Accumulated movement, with4*N headroom below infinity.
  std::vector<std::uint64_t> g_; ///< Retained exact committed row-major labels.
  std::vector<std::uint64_t> rhs_; ///< Retained exact one-step row-major labels.
  std::vector<std::size_t> heap_; ///< Indexed binary heap of vertex ids.
  std::vector<std::size_t> position_; ///< Inverse heap positions; absent uses SIZE_MAX.
  std::vector<DStarKey> keys_; ///< Stored priorities, intentionally lazy after moving start.
  DStarWork pending_; ///< Work accumulated since construction/previous successful repair.
};

/// @brief One-shot optimal grid route using a newly constructed D* Lite planner.
/// Preserves the original function signature, result shape and blocked/outside-
/// terminal false/empty/zero convenience. Each call starts fresh; persistent reuse
/// requires DStarLiteGrid. Empty/ragged grids throw. No optional cell cap is imposed,
/// but the implementation count bound INT_MAX remains. Unit four-neighbor movement.
/// @param blocked Borrowed nonempty rectangular occupancy, unchanged.
/// @param start Query coordinate; outside/blocked gives no route.
/// @param goal Goal coordinate; outside/blocked gives no route.
/// @return Owned inclusive optimal route with row-major ties, or historical failure result.
/// @throws std::invalid_argument For empty/ragged grid.
/// @throws std::length_error For shape/count/implementation limits.
/// @throws std::bad_alloc On allocation failure.
/// @throws std::logic_error If internal path/queue invariants are violated.
/// @par Example
/// @code
/// const auto r=dm::algorithms::d_star_replan({{false,false},{false,false}},{0,0},{1,1});
/// std::print("{} {:.0f}\n",r.path.size(),r.cost); // 3 2
/// @endcode
/// @par References
/// - S. Koenig and M. Likhachev (2002), D* Lite, AAAI,476-483.
/// https://www.cs.cmu.edu/~motionplanning/papers/sbp_papers/integrated3/koenig_dstarlite_aaai02b.pdf
/// @see DStarLiteGrid
GridPathResult d_star_replan(const std::vector<std::vector<bool>>& blocked,std::pair<int,int> start,std::pair<int,int> goal);

/// @brief Build a validated owned occupancy grid from obstacle coordinates.
/// All dimensions/counts/coordinates are validated before allocation. Repeated
/// obstacles are harmless; coordinates use(row,column), zero-based.
/// @param rows Positive row count.
/// @param columns Positive column count.
/// @param obstacles Borrowed in-bounds blocked coordinates.
/// @param max_cells Optional complete-cell cap, default1000000; zero disables only cap.
/// @return Owned rectangular bool grid, initially free except listed cells.
/// @throws std::invalid_argument For nonpositive dimensions or invalid obstacle coordinate.
/// @throws std::length_error For count/implementation/cap limits.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto g=dm::algorithms::grid_obstacle_mask(2,3,{{0,1}});
/// std::print("{}\n",g[0][1]); // true
/// @endcode
/// @see DStarLiteGrid
std::vector<std::vector<bool>> grid_obstacle_mask(int rows,int columns,
             const std::vector<std::pair<int,int>>& obstacles={},std::size_t max_cells=1000000);

/// @brief Render a checked grid route with native polygons and actual cell coordinates.
/// Uses columns to the right and rows downward. Black cells are blocked, blue is
/// the route, green/orange terminals are S/G(or S=G). Validates every supplied path
/// step, occupancy, endpoint and cost before creating a plot; supports single cells.
/// This is a renderer/feasibility check, not an independent optimality certificate.
/// @param blocked Borrowed nonempty rectangular grid.
/// @param route Borrowed owned-result record; may be a no-route empty/zero result.
/// @param start In-bounds starting cell.
/// @param goal In-bounds goal.
/// @param max_cells Optional complete-cell drawing cap, default4096; zero disables only cap.
/// @return Owned RPlot; caller can configure title/layout and save to a local file.
/// @throws std::invalid_argument For malformed grid/endpoints/path/cost/result.
/// @throws std::length_error For shape/count/cell-budget limits.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto g=dm::algorithms::grid_obstacle_mask(2,2);
/// const auto r=dm::algorithms::d_star_replan(g,{0,0},{1,1});
/// const auto p=dm::algorithms::plot_grid_route(g,r,{0,0},{1,1});
/// std::print("{}\n",p.series().empty()); // false
/// @endcode
/// @see d_star_replan()
graphics::RPlot plot_grid_route(const std::vector<std::vector<bool>>& blocked,
             const GridPathResult& route,std::pair<int,int> start,std::pair<int,int> goal,
             std::size_t max_cells=4096);
} // namespace datamunge::algorithms
