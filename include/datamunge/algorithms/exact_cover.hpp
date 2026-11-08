#pragma once
/// @file exact_cover.hpp
/// @brief Checked iterative dancing links, actual search/mesh witnesses and reusable domino encodings.
#include <datamunge/plot/diagram.hpp>
#include <cstddef>
#include <vector>
namespace datamunge::algorithms {

/// @brief First exact-cover verdict and selected original row indices.
/// Preserves the original bool/vector field layout. A solved result covers every
/// required column exactly once; an unsolved result has an empty row vector. Empty
/// universes are solved by the empty selection. Row order is actual search order,
/// not sorted order. Copies own rows; default construction is unsolved.
///
///
/// @par Example
/// @code
/// const auto r=dm::algorithms::exact_cover(2,{{0},{1}});
/// std::print("{} {}\n",r.solved,r.rows.size()); // true 2
/// @endcode
///
/// @see exact_cover(), ExactCoverOutcome
struct ExactCoverResult {
    bool solved=false; ///< Whether a first cover was found.
    std::vector<int> rows; ///< Owned selected original row IDs, in actual selection order.
};

/// @brief Owned snapshot of one actual index-linked dancing-links node.
/// Node IDs are indices into a trace mesh. Root is zero; required-column headers
/// are 1..C and data nodes follow input row/entry order. column is a HEADER NODE ID,
/// not a zero-based problem column. Root column/row are -1; header column points to
/// itself and row is -1. Data row is the original input row ID and size is zero.
/// Header size is its current vertical list length. Mutable records do not validate
/// their own links; they own integers rather than pointers into another object.
///
///
/// @par Example
/// @code
/// const auto t=dm::algorithms::exact_cover_trace(1,{{0}});
/// std::print("{} {}\n",t.initial_nodes[0].right,t.final_nodes[0].right); // 1 1
/// @endcode
///
/// @see ExactCoverTrace
struct ExactCoverNode {
    int left=0; ///< Actual left neighbour node ID.
    int right=0; ///< Actual right neighbour node ID.
    int up=0; ///< Actual upward neighbour node ID.
    int down=0; ///< Actual downward neighbour node ID.
    int column=-1; ///< Header node ID, or -1 for root.
    int row=-1; ///< Original row ID, or -1 for headers/root.
    int size=0; ///< Actual column size for headers; zero otherwise.

    /// @brief Compare every stored link, identity and size field exactly.
    /// No mesh traversal or validation is performed.
    ///
    /// @param other Borrowed snapshot record.
    /// @return Whether all seven integer fields agree.
    ///
    /// @par Example
    /// @code
    /// const dm::algorithms::ExactCoverNode a,b;
    /// std::print("{}\n",a==b); // true
    /// @endcode
    ///
    /// @see ExactCoverTrace
    bool operator==(const ExactCoverNode& other) const = default;
};

/// @brief Kinds of actual search events, including restoration after a found solution.
/// choose_column and dead_end are recorded before covering the chosen column.
/// select_row follows covering all its columns. unselect_row follows restoring its
/// other columns and popping its row; the pivot column remains covered until its
/// frame exits. Thus unselect_row can be cleanup after success as well as a failed
/// branch. solution precedes cleanup; finished follows complete mesh restoration.
///
///
/// @par Example
/// @code
/// const auto event=dm::algorithms::ExactCoverEvent::unselect_row;
/// std::print("{}\n",event==dm::algorithms::ExactCoverEvent::dead_end); // false
/// @endcode
///
/// @see ExactCoverStep
enum class ExactCoverEvent {
    choose_column, ///< Selected the smallest active minimum-size column.
    select_row, ///< Actually selected a candidate and covered its other columns.
    unselect_row, ///< Actually restored the candidate and removed its row.
    dead_end, ///< Selected column has zero remaining compatible rows.
    solution, ///< Every required header is covered; output rows retained.
    finished ///< All runtime links/sizes restored; search has terminated.
};

/// @brief Checked logical operations from the actual iterative dancing-links execution.
/// Initialization, validation, allocations, snapshots, copying and CPU time are
/// excluded. Column scans count every active header visited by the minimum-size
/// heuristic. Link writes count assignments even if their value is unchanged.
/// Every completed successful or failed search restores its mesh: covers=uncovers
/// and row_attempts=row_restorations. Search remains exponential in the worst case.
///
///
/// @par Example
/// @code
/// const auto r=dm::algorithms::exact_cover_work(2,{{0},{1}});
/// std::print("{} {} {}\n",r.work.row_attempts,r.work.row_restorations,r.work.maximum_depth); // 2 2 2
/// @endcode
///
/// @see ExactCoverOutcome
struct ExactCoverWork {
    std::size_t search_entries=0; ///< Entered search levels, including zero-column success/dead ends.
    std::size_t column_scans=0; ///< Actual active headers visited in minimum-size selection.
    std::size_t row_attempts=0; ///< Actual candidate selections.
    std::size_t row_restorations=0; ///< Actual candidate removals during failed-branch or success cleanup.
    std::size_t dead_ends=0; ///< Entries with a required column of size zero.
    std::size_t covers=0; ///< Actual cover-column calls.
    std::size_t uncovers=0; ///< Actual inverse uncover-column calls.
    std::size_t header_link_writes=0; ///< Two assignments per cover/uncover.
    std::size_t data_link_writes=0; ///< Two assignments per data unlink/relink.
    std::size_t column_size_updates=0; ///< One decrement/increment per data unlink/relink.
    std::size_t maximum_depth=0; ///< Largest actual selected-row count.
};

/// @brief Owned first-cover result and actual search/restoration work.
/// Uses the same deterministic search as exact_cover(). The runtime mesh is fully
/// restored even after success; the first selected-row result is retained separately.
/// Ordinary/work calls retain no optional input, event or mesh-snapshot history.
///
///
/// @par Example
/// @code
/// const auto r=dm::algorithms::exact_cover_work(1,{});
/// std::print("{} {}\n",r.result.solved,r.work.dead_ends); // false 1
/// @endcode
///
/// @see exact_cover_work(), ExactCoverTrace
struct ExactCoverOutcome {
    ExactCoverResult result; ///< Owned first solution or unsolved/empty rows.
    ExactCoverWork work; ///< Actual logical counts, including complete restoration.
};

/// @brief Owned actual residual-column and selected-row state at a search event.
/// column_sizes has C entries: -1 for a covered column and actual nonnegative
/// vertical-list size for an active column. partial_rows is the actual selected-row
/// stack at this event; depth equals its length. column/row use zero-based problem
/// IDs, or -1 when inapplicable. At unselect_row, the pivot column is still covered
/// even though that candidate has been popped. These are live captures, not replay.
///
///
/// @par Example
/// @code
/// const auto t=dm::algorithms::exact_cover_trace(1,{{0}});
/// std::print("{} {}\n",t.steps.front().column,t.steps.front().column_sizes[0]); // 0 1
/// @endcode
///
/// @see ExactCoverEvent, ExactCoverTrace
struct ExactCoverStep {
    ExactCoverEvent kind=ExactCoverEvent::finished; ///< Actual event kind.
    std::size_t depth=0; ///< Actual current selected-row count.
    int column=-1; ///< Chosen/pivot problem column, or -1.
    int row=-1; ///< Selected/unselected original row ID, or -1.
    std::vector<int> column_sizes; ///< Owned full residual size vector, with -1 covered markers.
    std::vector<int> partial_rows; ///< Owned actual selected rows, in stack order.
};

/// @brief Owned actual search states and initial/final dancing-links meshes.
/// All inputs, nodes, events and output rows are owned. Initial/final nodes are
/// copied from the actual runtime mesh; every successful/failed search unwinds fully,
/// so those arrays agree exactly. No explanation is reconstructed by another solver.
/// Let R be input row count, I total incidences and K=1+C+I nodes. Known base capture
/// B=R+I+2K counts row records/incidences plus two NODE RECORD snapshots (each entire
/// seven-field node counts as one record). Each event costs 1+C+depth: event record,
/// column sizes and selected-row IDs. B is reserved before graph/copy allocation;
/// events are reserved before snapshot copies. Default one million; zero disables
/// only quota. Ordinary output/work, runtime mesh/stack/validation, vector headers,
/// capacities and allocator bytes are excluded: this is not peak RAM. Any failure
/// returns no partial trace; caller input is never mutated.
///
///
/// @par Example
/// @code
/// const auto t=dm::algorithms::exact_cover_trace(1,{{0}});
/// std::print("{} {}\n",t.outcome.result.solved,t.initial_nodes==t.final_nodes); // true true
/// @endcode
///
/// @see exact_cover_trace(), ExactCoverStep
struct ExactCoverTrace {
    int num_columns=0; ///< Number of required problem columns.
    std::vector<std::vector<int>> input_rows; ///< Owned validated rows, retaining original entry order.
    std::vector<ExactCoverNode> initial_nodes; ///< Actual mesh immediately after construction.
    std::vector<ExactCoverNode> final_nodes; ///< Actual fully restored mesh after search/cleanup.
    std::vector<ExactCoverStep> steps; ///< Owned actual events in execution order.
    ExactCoverOutcome outcome; ///< Actual result and search/restoration counts.
    std::size_t captured_entries=0; ///< Defined logical mixed record/incidence/event-entry count.
};

/// @brief Owned exact-cover encoding of adjacent domino placements on a rectangular board.
/// Cells use row-major IDs r*board_columns+c with row zero at the top. Each
/// placement has exactly two orthogonally adjacent cells. The canonical builder
/// visits cells row-major, adding horizontal then vertical placements. Identical
/// dominoes need no piece-identity column. Public mutable fields are validated by
/// domino_tiling_diagram(); exact_cover() separately validates general incidence.
///
///
/// @par Example
/// @code
/// const auto p=dm::algorithms::domino_exact_cover(4,6);
/// std::print("{} {}\n",p.num_columns,p.placements.size()); // 24 38
/// @endcode
///
/// @see domino_exact_cover(), domino_tiling_diagram()
struct DominoCoverProblem {
    int board_rows=0; ///< Nonnegative board height.
    int board_columns=0; ///< Nonnegative board width.
    int num_columns=0; ///< Number of required cells, equal to height*width.
    std::vector<std::vector<int>> placements; ///< Owned adjacent cell pairs, in canonical builder order.
};

/// @brief Check exact-cover index/container geometry without allocating a mesh.
/// Requires root+column headers+nonzeros to fit positive int node IDs and all node
/// containers; row IDs must fit int. This is a count check, not available-memory
/// validation, and does not examine individual incidence lists.
///
/// @param num_columns Nonnegative number of required columns.
/// @param row_count Number of input row records.
/// @param nonzeros Total number of row-column incidences.
/// @return Checked root+header+incidence node count.
/// @throws std::invalid_argument If num_columns is negative.
/// @throws std::length_error On checked count, int-index or container limits.
///
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::exact_cover_node_count(3,4,9)); // 13
/// @endcode
///
/// @see exact_cover(), ExactCoverNode
std::size_t exact_cover_node_count(int num_columns,std::size_t row_count,std::size_t nonzeros);

/// @brief Find the first required-column exact cover using checked iterative dancing links.
/// Every column is required exactly once (no secondary columns). Each row must
/// contain distinct in-range column IDs; unsorted order is allowed and preserved.
/// Duplicate whole rows are separate alternatives; empty rows are ignored. The empty
/// universe is solved by no rows. All inputs are checked before early success/access.
/// Minimum-size active column ties use the first column in ascending header order;
/// its row candidates keep input order. Iterative depth-first search follows Algorithm X
/// without recursive call-stack growth. It stops at the first cover and restores the
/// complete mesh afterwards on both success/failure. Work remains exponential in
/// general; runtime mesh/stack storage is O(columns+rows+incidences), excluding output.
/// Independent calls share no mutable state; concurrent caller mutation is unsafe.
/// The original compiled signature/result field layout are retained. Ordinary/work
/// calls have no optional history or capture quota and never mutate input.
///
/// @param num_columns Nonnegative required-column count C.
/// @param rows Borrowed incidence rows; each contains distinct indices in [0,C).
/// @return Owned first solution in actual selection order, or unsolved with empty rows.
/// @throws std::invalid_argument For negative columns, invalid column IDs or repeated IDs within a row.
/// @throws std::length_error On node/row/int/container/counter geometry overflow.
/// @throws std::bad_alloc If required storage cannot be allocated.
///
/// @par Example
/// @code
/// const auto r=dm::algorithms::exact_cover(3,{{0,1},{0,2},{1,2},{0,1,2}});
/// std::print("{} {}\n",r.solved,r.rows[0]); // true 3
/// @endcode
///
/// @par References
/// - D. E. Knuth (2000). Dancing Links. arXiv:cs/0011047. https://arxiv.org/abs/cs/0011047
///
/// @see exact_cover_work(), exact_cover_trace()
ExactCoverResult exact_cover(int num_columns,const std::vector<std::vector<int>>& rows);

/// @brief Return the first exact cover and actual search/restoration operations.
/// Every column is required exactly once (no secondary columns). Each row must
/// contain distinct in-range column IDs; unsorted order is allowed and preserved.
/// Duplicate whole rows are separate alternatives; empty rows are ignored. The empty
/// universe is solved by no rows. All inputs are checked before early success/access.
/// Minimum-size active column ties use the first column in ascending header order;
/// its row candidates keep input order. Iterative depth-first search follows Algorithm X
/// without recursive call-stack growth. It stops at the first cover and restores the
/// complete mesh afterwards on both success/failure. Work remains exponential in
/// general; runtime mesh/stack storage is O(columns+rows+incidences), excluding output.
/// Independent calls share no mutable state; concurrent caller mutation is unsafe.
/// Retains no optional input/events/mesh snapshots or capture quota. Counters include
/// restoring all selected rows and columns after the first cover has been retained.
///
/// @param num_columns Nonnegative required-column count C.
/// @param rows Borrowed incidence rows; each contains distinct indices in [0,C).
/// @return Owned first-cover result and actual logical work.
/// @throws std::invalid_argument For negative columns, invalid column IDs or repeated IDs within a row.
/// @throws std::length_error On node/row/int/container/counter geometry overflow.
/// @throws std::bad_alloc If required storage cannot be allocated.
///
/// @par Example
/// @code
/// const auto r=dm::algorithms::exact_cover_work(2,{{0},{1}});
/// std::print("{} {}\n",r.result.rows.size(),r.work.covers==r.work.uncovers); // 2 true
/// @endcode
///
/// @par References
/// - D. E. Knuth (2000). Dancing Links. arXiv:cs/0011047. https://arxiv.org/abs/cs/0011047
///
/// @see ExactCoverWork, exact_cover()
ExactCoverOutcome exact_cover_work(int num_columns,const std::vector<std::vector<int>>& rows);

/// @brief Capture the same actual exact-cover search and its restored mesh with a quota.
/// Every column is required exactly once (no secondary columns). Each row must
/// contain distinct in-range column IDs; unsorted order is allowed and preserved.
/// Duplicate whole rows are separate alternatives; empty rows are ignored. The empty
/// universe is solved by no rows. All inputs are checked before early success/access.
/// Minimum-size active column ties use the first column in ascending header order;
/// its row candidates keep input order. Iterative depth-first search follows Algorithm X
/// without recursive call-stack growth. It stops at the first cover and restores the
/// complete mesh afterwards on both success/failure. Work remains exponential in
/// general; runtime mesh/stack storage is O(columns+rows+incidences), excluding output.
/// Independent calls share no mutable state; concurrent caller mutation is unsafe.
/// Capture B=R+I+2K plus sum(1+C+depth) follows ExactCoverTrace. The known base
/// is checked before optional input/mesh copies or runtime graph allocation; each
/// event is checked before its state snapshot. Quota failure exposes no partial
/// trace. Optional history can be much larger than the sparse runtime mesh.
///
/// @param num_columns Nonnegative required-column count C.
/// @param rows Borrowed incidence rows; each contains distinct indices in [0,C).
/// @param max_entries Maximum defined logical capture entries (default 1000000); zero disables only quota.
/// @return Owned actual input, initial/final meshes, events and outcome.
/// @throws std::invalid_argument For negative columns, invalid column IDs or repeated IDs within a row.
/// @throws std::length_error On node/row/int/container/counter geometry overflow.
/// @throws std::bad_alloc If required storage cannot be allocated.
/// @throws std::length_error If known base or any next event exceeds enabled max_entries.
///
/// @par Example
/// @code
/// const auto t=dm::algorithms::exact_cover_trace(1,{{0}});
/// std::print("{} {}\n",t.steps.size(),t.captured_entries); // 5 20
/// @endcode
///
/// @par References
/// - D. E. Knuth (2000). Dancing Links. arXiv:cs/0011047. https://arxiv.org/abs/cs/0011047
///
/// @see ExactCoverTrace, ExactCoverEvent
ExactCoverTrace exact_cover_trace(int num_columns,const std::vector<std::vector<int>>& rows,std::size_t max_entries=1000000);

/// @brief Build every horizontal/vertical adjacent domino placement for a rectangular board.
/// Rows/columns are nonnegative; a zero-area board has zero cells/placements.
/// For positive dimensions, placement count is h(w-1)+(h-1)w. Cell/pair/node/count
/// geometry is checked before allocation. Construction and owned storage are linear
/// in cells+placements. Canonical row-major, horizontal-before-vertical order retains
/// the original report encoding and therefore its stable first tiling choice.
///
/// @param board_rows Nonnegative height h.
/// @param board_columns Nonnegative width w.
/// @return Owned required-cell/adjacent-placement encoding.
/// @throws std::invalid_argument If a board dimension is negative.
/// @throws std::length_error On cell/placement/node/container arithmetic limits.
/// @throws std::bad_alloc If required storage cannot be allocated.
///
/// @par Example
/// @code
/// const auto p=dm::algorithms::domino_exact_cover(4,6);
/// const auto r=dm::algorithms::exact_cover(p.num_columns,p.placements);
/// std::print("{} {}\n",r.solved,r.rows.size()); // true 12
/// @endcode
///
/// @par References
/// - D. E. Knuth (2000). Dancing Links. arXiv:cs/0011047. https://arxiv.org/abs/cs/0011047
///
/// @see DominoCoverProblem, exact_cover()
DominoCoverProblem domino_exact_cover(int board_rows,int board_columns);

/// @brief Draw a validated selected domino cover as an owned native C++ diagram.
/// Uses actual placement cell IDs; row zero is drawn at the top. Selected rows get
/// coloured rectangles and original row-ID labels. A solved result must cover every
/// cell exactly once. An unsolved result must have no selected rows and is labelled
/// reported unsolved, without independently proving nonexistence. Mutable problem
/// shape, every adjacent pair and selected ID are checked before access. Zero area
/// returns an informative empty-board diagram. Time/storage O(cells+placements).
/// The diagram owns primitives/text and borrows no input. Primitive mutation follows
/// Diagram thread-safety; concurrent calls are independent for unmodified inputs.
///
/// @param problem Borrowed rectangular-board encoding with valid adjacent cell pairs.
/// @param result Borrowed solved exact cover, or unsolved/empty selection.
/// @param unit_pixels Positive finite native coordinate scale (default 64).
/// @param max_width Positive finite maximum diagram width in pixels (default 1100).
/// @param max_cells Maximum board cells to render (default 100000); zero disables only this quota.
/// @return Owned diagram ready to save as a local image.
/// @throws std::invalid_argument On invalid scale, dimensions, problem incidence or selected cover.
/// @throws std::length_error On board geometry or enabled cell-render quota.
/// @throws std::bad_alloc If required diagram/validation storage cannot be allocated.
///
/// @par Example
/// @code
/// const auto p=dm::algorithms::domino_exact_cover(2,2);
/// const auto r=dm::algorithms::exact_cover(p.num_columns,p.placements);
/// dm::algorithms::domino_tiling_diagram(p,r).save_svg("domino.svg");
/// std::print("{}\n",r.rows.size()); // 2
/// @endcode
///
/// @see domino_exact_cover(), graphics::Diagram
graphics::Diagram domino_tiling_diagram(const DominoCoverProblem& problem,const ExactCoverResult& result,double unit_pixels=64,double max_width=1100,std::size_t max_cells=100000);

} // namespace datamunge::algorithms
