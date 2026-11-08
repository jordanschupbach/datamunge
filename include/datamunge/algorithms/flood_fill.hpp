#pragma once
/// @file flood_fill.hpp
/// @brief Exact-value iterative grid flood fill with actual traversal evidence.
#include <array>
#include <cstddef>
#include <vector>
namespace datamunge::algorithms {
/// @brief Reason a validated flood-fill operation did or did not traverse.
/// Empty includes zero-width rectangular grids. SameValue changes no cells.
/// @par Example
/// @code
/// std::cout << (dm::algorithms::FloodFillStop::Filled == dm::algorithms::FloodFillStop::Filled); // 1
/// @endcode
/// @see FloodFillWork
 enum class FloodFillStop {
    Filled, ///< A valid seed with a distinct replacement was traversed.
    EmptyGrid, ///< There are no cells.
    SeedOutside, ///< Seed row or column is outside the nonempty grid.
    SameValue ///< Replacement already equals the seed value.
};
/// @brief One ordered neighbor slot; invalid slots have zero row and column.
/// Four directions are up/down/left/right; eight are NW/N/NE/W/E/SW/S/SE.
/// @par Example
/// @code
/// dm::algorithms::FloodFillNeighbor p{};
/// std::cout << p.in_bounds; // 0
/// @endcode
/// @see flood_fill_neighborhood()
struct FloodFillNeighbor {
    std::size_t direction{}; ///< Zero-based slot in the selected direction order.
    bool in_bounds{}; ///< Whether row and col identify a real neighbor.
    std::size_t row{}; ///< Neighbor row, or zero when outside.
    std::size_t col{}; ///< Neighbor column, or zero when outside.
};
/// @brief Fixed-capacity ordered neighborhood without allocation or grid storage.
/// Unused slots are zero initialized; count is zero for an invalid center.
/// @par Example
/// @code
/// dm::algorithms::FloodFillNeighborhood n{};
/// std::cout << n.count; // 0
/// @endcode
/// @see flood_fill_neighborhood()
struct FloodFillNeighborhood {
    std::array<FloodFillNeighbor,8> candidates{}; ///< Ordered slots; inspect only count entries.
    std::size_t count{}; ///< Four or eight slots for a valid center, otherwise zero.
};
/// @brief Enumerate the actual fill direction order using guarded size_t geometry.
/// No product, narrowing, allocation or grid access; virtual SIZE_MAX dimensions
/// are supported. Boundary guards precede coordinate subtraction/addition.
/// @param rows Number of rows, including zero.
/// @param cols Number of columns, including zero.
/// @param row Center row; outside centers return no slots.
/// @param col Center column; outside centers return no slots.
/// @param eight_connected Eight directions if true; four by default.
/// @return Owned fixed-capacity slots; outside coordinates are zero placeholders.
/// @par Example
/// @code
/// const auto n=dm::algorithms::flood_fill_neighborhood(2,2,0,0);
/// std::cout << n.count << ' ' << n.candidates[1].in_bounds; // 4 1
/// @endcode
/// @see flood_fill()
FloodFillNeighborhood flood_fill_neighborhood(std::size_t rows,std::size_t cols,
    std::size_t row,std::size_t col,bool eight_connected=false) noexcept;
/// @brief Owned scalar measurements of one actual iterative fill.
/// Each discovered cell is painted when pushed and popped once. Neighbor slots
/// include outside probes; in_bounds_neighbors counts actual value reads.
/// A preexisting replacement-valued cell is a barrier, not a traversal bridge.
/// @par Example
/// @code
/// dm::algorithms::FloodFillWork w{};
/// std::cout << w.filled; // 0
/// @endcode
/// @see flood_fill_work(), FloodFillTrace
struct FloodFillWork {
    std::size_t rows{}; ///< Validated outer row count.
    std::size_t cols{}; ///< Validated common width.
    std::size_t cells{}; ///< Checked product rows times cols.
    std::size_t seed_row{}; ///< Requested seed row.
    std::size_t seed_col{}; ///< Requested seed column.
    int target{}; ///< Original seed value; zero placeholder without a valid seed.
    bool has_target{}; ///< Whether target was read from a valid seed.
    int new_value{}; ///< Requested replacement value.
    FloodFillStop stop{FloodFillStop::EmptyGrid}; ///< Validated operation outcome.
    std::size_t filled{}; ///< Number of popped and changed cells.
    std::size_t discovered{}; ///< Seed plus accepted neighbor pushes.
    std::size_t neighbor_slots{}; ///< All attempted direction slots.
    std::size_t in_bounds_neighbors{}; ///< Actual in-bounds target comparisons.
    std::size_t accepted_neighbors{}; ///< Successful neighbor discoveries, excluding the seed.
    std::size_t peak_stack{}; ///< Maximum number of pending coordinates.
};
/// @brief One actual LIFO pop and its contiguous range of neighbor probes.
/// Stack sizes exclude/include the current scan's discoveries respectively.
/// @par Example
/// @code
/// dm::algorithms::FloodFillVisit v{};
/// std::cout << v.probe_end; // 0
/// @endcode
/// @see FloodFillTrace
struct FloodFillVisit {
    std::size_t row{}; ///< Popped row.
    std::size_t col{}; ///< Popped column.
    std::size_t stack_after_pop{}; ///< Pending cells immediately after removal.
    std::size_t stack_after_scan{}; ///< Pending cells after all direction slots.
    std::size_t probe_begin{}; ///< Inclusive index in the trace probe vector.
    std::size_t probe_end{}; ///< Exclusive index in the trace probe vector.
};
/// @brief One actual direction attempt and its value before any discovery paint.
/// Outside slots have value zero; accepted means an in-bounds target match.
/// @par Example
/// @code
/// dm::algorithms::FloodFillProbe p{};
/// std::cout << p.accepted; // 0
/// @endcode
/// @see FloodFillTrace
struct FloodFillProbe {
    FloodFillNeighbor location; ///< Ordered slot and guarded coordinates.
    int value_before{}; ///< Observed cell value, or zero outside the grid.
    bool accepted{}; ///< Whether this attempt painted and pushed the neighbor.
};
/// @brief Owned input, output and evidence recorded during the actual fill.
/// No retained views or replay; modifying caller storage cannot change this result.
/// @par Example
/// @code
/// dm::algorithms::FloodFillTrace t{};
/// std::cout << t.visits.size(); // 0
/// @endcode
/// @see flood_fill_trace()
struct FloodFillTrace {
    std::vector<std::vector<int>> input; ///< Independent original grid.
    std::vector<std::vector<int>> output; ///< Independent painted grid.
    FloodFillWork work; ///< Actual scalar measurements.
    std::vector<FloodFillVisit> visits; ///< Actual pop order.
    std::vector<FloodFillProbe> probes; ///< Actual direction attempts in visit order.
};
/// @brief Paint the exact-value seed component and measure the actual traversal.
/// Full rectangular validation and checked cell product precede all no-ops and
/// mutation. Signed int values are unrestricted. Empty/outside/same-value fills
/// change nothing. Marks on push, scans four directions U/D/L/R or eight
/// NW/N/NE/W/E/SW/S/SE, then pops LIFO. O(rows+K) time and O(K) stack for K
/// changed cells; no history quota. Independent calls have no shared mutable state.
/// @param grid Mutable rectangular row-major integer grid.
/// @param sr Seed row.
/// @param sc Seed column.
/// @param new_value Replacement value.
/// @param eight_connected Eight directions if true; four by default.
/// @return Scalar measurements, including zero filled for no-ops.
/// @throws std::invalid_argument Ragged grid, before mutation even for no-ops.
/// @throws std::length_error Cell product or stack capacity is unrepresentable.
/// @throws std::overflow_error An actual work counter would overflow.
/// @throws std::bad_alloc Stack allocation fails; earlier paint may remain.
/// @par Example
/// @code
/// std::vector<std::vector<int>> g{{1,1},{0,1}};
/// std::cout << dm::algorithms::flood_fill_work(g,0,0,9).neighbor_slots; // 12
/// @endcode
/// @par References
/// - https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_floodfill.html
///   Exact-value interpretation only; this API defaults to four connectivity.
/// @see flood_fill(), flood_fill_trace()
FloodFillWork flood_fill_work(std::vector<std::vector<int>>& grid,std::size_t sr,
    std::size_t sc,int new_value,bool eight_connected=false);
/// @brief Paint the seed component, preserving the original count-returning API.
/// Uses flood_fill_work() and its validation, direction order, complexity and
/// partial-mutation behavior on resource failure. Full signed int values are valid.
/// @param grid Mutable rectangular row-major integer grid.
/// @param sr Seed row.
/// @param sc Seed column.
/// @param new_value Replacement value.
/// @param eight_connected Eight directions if true; four by default.
/// @return Changed cell count; zero for empty, outside or same-value requests.
/// @throws std::invalid_argument Ragged grid before mutation.
/// @throws std::length_error Cell product or stack capacity is unrepresentable.
/// @throws std::overflow_error An actual counter would overflow.
/// @throws std::bad_alloc Stack allocation fails; earlier paint may remain.
/// @par Example
/// @code
/// std::vector<std::vector<int>> g{{1,9,1}};
/// std::cout << dm::algorithms::flood_fill(g,0,0,9); // 1
/// @endcode
/// @par References
/// - https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_floodfill.html
/// @see flood_fill_work(), flood_fill_trace()
std::size_t flood_fill(std::vector<std::vector<int>>& grid,std::size_t sr,
    std::size_t sc,int new_value,bool eight_connected=false);
/// @brief Copy a grid and record actual pop/probe evidence while painting its copy.
/// Validates shape first. Before copying, admits the whole upper entry bound
/// 2*rows+(D+3)*cells for an active fill (D=4 or 8), or 2*(rows+cells)
/// for a no-op. Counts outer-row entries, cell entries, visits and probes, not
/// bytes or peak RAM. Actual histories contain only K visits and D*K probes.
/// Default cap is one million; zero disables only the optional cap. Checked
/// arithmetic/container bounds remain. O(rows+cells+D*K) time/storage.
/// Caller input is unchanged on success or failure; no partial result escapes.
/// @param grid Read-only rectangular row-major grid.
/// @param sr Seed row.
/// @param sc Seed column.
/// @param new_value Replacement value.
/// @param eight_connected Eight directions if true; four by default.
/// @param max_entries Whole upper entry cap; default one million, zero unlimited.
/// @return Independent owned input/output, actual work, visits and probes.
/// @throws std::invalid_argument Ragged grid, before quota checking/copying.
/// @throws std::length_error Bound overflows, exceeds a positive cap or container limit.
/// @throws std::overflow_error An actual work counter would overflow.
/// @throws std::bad_alloc Result or stack allocation fails.
/// @par Example
/// @code
/// const auto t=dm::algorithms::flood_fill_trace({{1,1},{1,1}},0,0,9);
/// std::cout << t.work.filled << ' ' << t.work.peak_stack; // 4 2
/// @endcode
/// @par References
/// - https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_floodfill.html
/// @see flood_fill(), FloodFillTrace
FloodFillTrace flood_fill_trace(const std::vector<std::vector<int>>& grid,
    std::size_t sr,std::size_t sc,int new_value,bool eight_connected=false,
    std::size_t max_entries=1000000);
/// @brief Construct the original six-by-eight binary flood-fill report fixture.
/// Upper-left and lower-right foreground components have eight and five cells.
/// Returns fresh owned storage; no external dataset or random state.
/// @return Six rows of eight integer cells.
/// @throws std::bad_alloc Grid allocation fails.
/// @par Example
/// @code
/// auto g=dm::algorithms::flood_fill_example();
/// std::cout << dm::algorithms::flood_fill(g,0,0,9); // 8
/// @endcode
/// @see flood_fill()
std::vector<std::vector<int>> flood_fill_example();
} // namespace datamunge::algorithms
