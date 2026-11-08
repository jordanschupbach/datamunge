#pragma once
/// @file connected_components.hpp
/// @brief Checked two-pass foreground labelling, actual union histories and blob measurements.
#include <array>
#include <cstddef>
#include <vector>

namespace datamunge::algorithms {
/// @brief Owned dense component IDs, ordered by first row-major component occurrence.
/// Zero is background; IDs are independent of internal union-find roots. Original
/// fields and layout are preserved. All nonzero input ints mean foreground.
/// @par Example
/// @code
/// const auto c = dm::algorithms::connected_components({{1,0,1}});
/// std::cout << c.count << ' ' << c.labels[0][2]; // 2 2
/// @endcode
/// @see connected_components(), connected_component_stats()
struct ConnectedComponents {
    std::vector<std::vector<int>> labels; ///< Owned rectangular label image, zero background.
    int count = 0;                       ///< Number of components, at most INT_MAX.
};

/// @brief Complete-image and actual-history budgets; zero means unlimited.
/// Full shape validation precedes the complete max_cells cap and allocation.
/// max_events counts foreground-cell plus union-attempt records online; analysis
/// and output-only calls ignore the event cap.
/// @par Example
/// @code
/// dm::algorithms::ComponentLabelOptions o;
/// std::cout << o.max_cells << ' ' << o.max_events; // 10000000 1000000
/// @endcode
/// @see connected_components_trace()
struct ComponentLabelOptions {
    std::size_t max_cells{10000000}; ///< Complete cell cap.
    std::size_t max_events{1000000}; ///< Actual cell/union record cap.
};

/// @brief Actual two-scan and balanced disjoint-set operation counts.
/// Neighbor probes count in-bounds earlier positions for foreground cells only;
/// admitted labels retain duplicates. find_steps counts path-halving iterations,
/// not original edges. parent_writes counts union/halving assignments, including
/// idempotent halving writes, excluding initial set creation. Snapshots and later
/// measurements perform no counted operations. All counters are checked.
/// @par Example
/// @code
/// const auto a = dm::algorithms::connected_components_analyze({{1,0,1}});
/// std::cout << a.work.labels_created << ' ' << a.work.successful_unions; // 2 0
/// @endcode
/// @see ComponentLabelAnalysis
struct ComponentLabelWork {
    std::size_t validated_cells{};   ///< Checked rows times columns.
    std::size_t first_pass_cells{};  ///< All first-scan cells, including background.
    std::size_t second_pass_cells{}; ///< All second-scan cells, including background.
    std::size_t foreground{};        ///< Nonzero input cells.
    std::size_t neighbor_probes{};   ///< In-bounds earlier positions inspected.
    std::size_t neighbor_labels{};   ///< Admitted nonzero provisional labels.
    std::size_t labels_created{};    ///< Positive provisional sets made.
    std::size_t union_calls{};       ///< All attempts, including self/redundant unions.
    std::size_t successful_unions{}; ///< Attachments between distinct roots.
    std::size_t find_calls{};        ///< Actual representative queries.
    std::size_t find_steps{};        ///< Path-halving loop iterations.
    std::size_t parent_writes{};     ///< Union and halving assignments.
    std::size_t remapped_cells{};    ///< Foreground cells assigned dense output IDs.
};

/// @brief Owned actual first-pass foreground decision with its union-record range.
/// earlier_labels[0..earlier_count) follows up, left, upper-left, upper-right,
/// omitting background/out-of-bounds positions and retaining duplicate labels.
/// assigned is their minimum or a newly created ID. The union range is half-open.
/// @par Example
/// @code
/// const auto t = dm::algorithms::connected_components_trace({{1,0,1},{1,1,1}});
/// std::cout << t.cells.back().assigned << ' ' << t.cells.back().earlier_count; // 1 2
/// @endcode
/// @see ComponentLabelUnion, ComponentLabelTrace
struct ComponentLabelCell {
    std::size_t row{};                 ///< Zero-based foreground row.
    std::size_t col{};                 ///< Zero-based foreground column.
    std::array<int,4> earlier_labels{}; ///< Actual admitted labels; unused entries zero.
    std::size_t earlier_count{};       ///< Admitted count, at most four.
    int assigned{};                   ///< Provisional value actually written.
    bool created{};                   ///< Whether a new set was created.
    std::size_t union_begin{};         ///< First actual union-record index.
    std::size_t union_end{};           ///< One-past-last union-record index.
};

/// @brief Actual union operands, pre-attachment roots and resulting attachment.
/// Rank determines attachment; equal ranks choose the smaller root and increment
/// its rank. Redundant attempts have merged=false, winner=root_a and loser=0.
/// Internal root IDs are distinct from the final dense component labels.
/// @par Example
/// @code
/// const auto t = dm::algorithms::connected_components_trace({{1,0,1},{1,1,1}});
/// std::cout << t.unions.back().merged << ' ' << t.unions.back().winner; // 0 1
/// @endcode
/// @see ComponentLabelCell
struct ComponentLabelUnion {
    int a{};      ///< First provisional operand.
    int b{};      ///< Second provisional operand.
    int root_a{}; ///< Actual root of a before attachment.
    int root_b{}; ///< Actual root of b before attachment.
    int winner{}; ///< Resulting representative.
    int loser{};  ///< Attached former root, or zero when redundant.
    bool merged{};///< Whether distinct roots were attached.
};

/// @brief Owned final labelling and work, without history allocation.
/// count equals labels_created minus successful_unions. Only foreground cells
/// query representatives during renumbering; both scans visit background too.
/// @par Example
/// @code
/// const auto a = dm::algorithms::connected_components_analyze({{1,0,1},{1,1,1}});
/// std::cout << a.result.count << ' ' << a.work.successful_unions; // 1 1
/// @endcode
/// @see connected_components_analyze()
struct ComponentLabelAnalysis {
    ConnectedComponents result; ///< Owned dense output map.
    ComponentLabelWork work;     ///< Actual producing work.
};

/// @brief Owned actual two-pass history, without prefix reruns or synthetic decisions.
/// provisional is the map after scan one. parents is the actual forest after
/// scan two, including dummy zero; halving need not flatten it fully.
/// root_to_dense is the actual remap, zero at non-root/unused entries. Input,
/// snapshots and results own independent storage.
/// @par Example
/// @code
/// const auto t = dm::algorithms::connected_components_trace({{1,0,1},{1,1,1}});
/// std::cout << t.provisional[0][2] << ' ' << t.analysis.result.labels[0][2]; // 2 1
/// @endcode
/// @see connected_components_trace()
struct ComponentLabelTrace {
    ComponentLabelAnalysis analysis;            ///< Actual result and counters.
    std::vector<std::vector<int>> provisional;  ///< Completed first scan.
    std::vector<ComponentLabelCell> cells;       ///< One record per foreground cell.
    std::vector<ComponentLabelUnion> unions;     ///< All actual union attempts.
    std::vector<int> parents;                    ///< Actual final parent forest.
    std::vector<int> root_to_dense;              ///< Actual final remap table.
};

/// @brief Label rectangular nonzero foreground with orthogonal or diagonal adjacency.
/// Full shape validation precedes allocation, including zero-width inputs. Empty
/// and rectangular zero-width images retain shape and have zero components.
/// Two row-major scans use union by rank and path halving. IDs follow first
/// component occurrence, independently of root numbers. No input mutation/global
/// state. O(R+N alpha(N)) time, O(R+N) storage for R rows and N cells. No optional cap.
/// @param image Rectangular integer image; every nonzero int is foreground.
/// @param eight_connected False (default) uses four neighbours; true uses eight.
/// @return Owned dense IDs and count with the original member layout.
/// @throws std::invalid_argument Ragged rows.
/// @throws std::length_error Shape/counter overflow or more than INT_MAX provisional labels.
/// @par Example
/// @code
/// const auto four = dm::algorithms::connected_components({{1,0},{0,1}});
/// const auto eight = dm::algorithms::connected_components({{1,0},{0,1}}, true);
/// std::cout << four.count << ' ' << eight.count; // 2 1
/// @endcode
/// @par References
/// - Rosenfeld and Pfaltz (1966), Sequential Operations in Digital Picture Processing,
///   https://doi.org/10.1145/321356.321357 , two-pass equivalence labelling.
/// - Tarjan (1975), Efficiency of a Good But Not Linear Set Union Algorithm,
///   https://doi.org/10.1145/321879.321884 , balanced union and path compression.
/// - Tarjan and van Leeuwen (1984), Worst-Case Analysis of Set Union Algorithms,
///   https://doi.org/10.1145/62.2160 , rank plus path halving.
/// @see connected_components_analyze(), connected_components_trace()
ConnectedComponents connected_components(const std::vector<std::vector<int>>& image,
    bool eight_connected = false);

/// @brief Run the same kernel with actual work and no history allocation.
/// Same foreground/order/complexity contract as connected_components(). Full
/// rectangular validation precedes complete cell-cap preflight; event cap ignored.
/// @param image Rectangular integer image, zero background.
/// @param eight_connected Include diagonal adjacency when true; default false.
/// @param options Complete cell cap, default ten million; zero unlimited.
/// @return Owned result and actual counters.
/// @throws std::invalid_argument Ragged rows.
/// @throws std::length_error Cell cap, shape/counter overflow or int-label exhaustion.
/// @par Example
/// @code
/// const auto a = dm::algorithms::connected_components_analyze({{1,0,1}});
/// std::cout << a.work.first_pass_cells << ' ' << a.work.second_pass_cells; // 3 3
/// @endcode
/// @par References
/// - Rosenfeld and Pfaltz (1966), https://doi.org/10.1145/321356.321357 .
/// - Tarjan and van Leeuwen (1984), https://doi.org/10.1145/62.2160 .
/// @see connected_components(), ComponentLabelOptions
ComponentLabelAnalysis connected_components_analyze(const std::vector<std::vector<int>>& image,
    bool eight_connected = false, ComponentLabelOptions options = {});

/// @brief Retain actual decisions and the forest from the executing kernel without replay.
/// Same work/validation as analyze plus O(R+N) owned history. Positive max_events
/// counts actual foreground-cell plus union-attempt records online, never
/// background; zero is unlimited. Failure returns no partial trace.
/// @param image Rectangular integer image, zero background.
/// @param eight_connected Include diagonals when true, default false.
/// @param options Complete cell and online record caps.
/// @return Owned actual first-pass map, decisions, final forest and remap.
/// @throws std::invalid_argument Ragged rows.
/// @throws std::length_error Cell/event cap, shape/counter overflow or int-label exhaustion.
/// @par Example
/// @code
/// const auto t = dm::algorithms::connected_components_trace({{1,0,1},{1,1,1}});
/// std::cout << t.cells.size() << ' ' << t.analysis.result.count; // 5 1
/// @endcode
/// @par References
/// - Rosenfeld and Pfaltz (1966), https://doi.org/10.1145/321356.321357 .
/// - Tarjan and van Leeuwen (1984), https://doi.org/10.1145/62.2160 .
/// @see connected_components_analyze(), ComponentLabelTrace
ComponentLabelTrace connected_components_trace(const std::vector<std::vector<int>>& image,
    bool eight_connected = false, ComponentLabelOptions options = {});

/// @brief Pixel-index area, bounds, centroid and exposed-edge measurements for one label.
/// Bounds are inclusive; centroids average zero-based pixel-centre indices.
/// perimeter4 counts exposed orthogonal edges including holes/image edges,
/// regardless of label connectivity; it is not a smooth Euclidean perimeter.
/// @par Example
/// @code
/// const auto s = dm::algorithms::connected_component_stats(
///     dm::algorithms::connected_components({{1,1}}));
/// std::cout << s[0].area << ' ' << s[0].centroid_col << ' ' << s[0].perimeter4; // 2 0.5 6
/// @endcode
/// @see connected_component_stats()
struct ConnectedComponentStats {
    int label{};                ///< Supplied positive label.
    std::size_t area{};          ///< Number of labelled cells, positive.
    std::size_t min_row{};       ///< Inclusive minimum row.
    std::size_t min_col{};       ///< Inclusive minimum column.
    std::size_t max_row{};       ///< Inclusive maximum row.
    std::size_t max_col{};       ///< Inclusive maximum column.
    double centroid_row{};      ///< Mean zero-based row index.
    double centroid_col{};      ///< Mean zero-based column index.
    std::size_t perimeter4{};    ///< Exposed orthogonal edge count.
};

/// @brief Measure dense positive labels with checked integer counts and widened centroid sums.
/// Validates rectangular shape, nonnegative count, IDs in 0..count and that every
/// positive ID occurs. Does not certify connectivity: a supplied disconnected ID
/// is measured as one group. Full label-range validation and count<=foreground preflight
/// precede count-sized allocations. Empty/all-background count=0 returns empty.
/// O(R+N+count) time and O(count) additional storage; no input mutation/global state.
/// @param components Produced or externally constructed dense label map.
/// @return Owned measurements in ascending label order.
/// @throws std::invalid_argument Invalid shape/count/labels or missing positive IDs.
/// @throws std::length_error Shape/measurement counter overflow.
/// @par Example
/// @code
/// const auto s = dm::algorithms::connected_component_stats(
///     dm::algorithms::connected_components({{1,1},{1,0}}));
/// std::cout << s[0].area << ' ' << s[0].perimeter4; // 3 8
/// @endcode
/// @see ConnectedComponentStats, connected_components()
std::vector<ConnectedComponentStats> connected_component_stats(const ConnectedComponents& components);

/// @brief Original deterministic 12-by-12 fixture with four disjoint inclusive rectangles.
/// Rectangles (row,col): (1,1)-(3,3), (1,7)-(2,10), (6,2)-(9,3),
/// (7,7)-(10,10). Areas 9,8,8,16; both connectivity choices retain four blobs.
/// No RNG, download or global state; returned storage is owned.
/// @return Owned binary image.
/// @par Example
/// @code
/// std::cout << dm::algorithms::connected_components(
///     dm::algorithms::component_blobs_example()).count; // 4
/// @endcode
/// @see component_checkerboard(), connected_components()
std::vector<std::vector<int>> component_blobs_example();

/// @brief Generate a bounded checkerboard with foreground at (0,0), no RNG/global state.
/// Dimensions must be positive. Complete checked rows*cols cap precedes allocation.
/// O(rows*cols) time/storage; parity avoids adding row and column indices.
/// @param rows Positive row count.
/// @param cols Positive column count.
/// @param max_cells Complete cell cap, default one million; zero unlimited.
/// @return Owned zero/one image.
/// @throws std::invalid_argument Zero row or column count.
/// @throws std::length_error Shape overflow or cell cap exceeded.
/// @par Example
/// @code
/// const auto c = dm::algorithms::component_checkerboard(3,3);
/// std::cout << dm::algorithms::connected_components(c).count; // 5
/// @endcode
/// @see component_blobs_example(), connected_components()
std::vector<std::vector<int>> component_checkerboard(std::size_t rows, std::size_t cols,
    std::size_t max_cells = 1000000);
} // namespace datamunge::algorithms
