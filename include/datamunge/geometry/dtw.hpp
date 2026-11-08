#pragma once
/// @file dtw.hpp
/// @brief Checked scalar absolute-loss DTW, actual dynamic-programming evidence and native alignment plots.
#include <cstddef>
#include <cstdint>
#include <vector>
#include <datamunge/plot/plot.hpp>
namespace datamunge::geometry {
/// @brief Owned unrestricted scalar DTW result, preserving the original three-field layout.
/// Distance is the unnormalized sum of absolute differences along the returned
/// inclusive path. Default construction is an uncomputed placeholder. Copies own
/// independent buffers. Positive lengths fit int; steps advance a, b or both by one.
/// @par Example
/// @code
/// const auto r=dm::geometry::dynamic_time_warping({1,2,3},{1,1,2,3});
/// std::print("{} {}\n",r.distance,r.path_a.size()); // 0 4
/// @endcode
/// @see dynamic_time_warping(), dtw_path_cost()
struct DTWResult {
 double distance{0}; ///< Finite nonnegative summed absolute loss for a computed result.
 std::vector<int> path_a; ///< Owned zero-based a indices, from zero to a.size()-1.
 std::vector<int> path_b; ///< Owned parallel b indices, from zero to b.size()-1.
};
/// @brief Logical work of the shared DTW recurrence and actual traceback.
/// These are counts, not time. Validation, allocation, copies, comparisons and
/// rendering are excluded. Each positive cell reads three predecessor slots,
/// including infinite boundary slots. Every returned pair counts one traceback step.
/// @par Example
/// @code
/// const auto r=dm::geometry::dynamic_time_warping_work({2},{5});
/// std::print("{} {} {}\n",r.work.cells,r.work.predecessor_reads,r.work.traceback_steps); // 1 3 1
/// @endcode
/// @see DTWComputation
struct DTWWork {
 std::uint64_t cells=0; ///< Exactly a.size()*b.size() recurrence evaluations.
 std::uint64_t predecessor_reads=0; ///< Three slots per recurrence cell, including borders.
 std::uint64_t traceback_steps=0; ///< Number of actual output index pairs, including origin.
};
/// @brief Owned ordinary result and logical work, without retained cost-table history.
/// Scratch consists of two cost rows and one predecessor byte per positive cell.
/// No input references survive return. Copies own independent result paths.
/// @par Example
/// @code
/// const auto c=dm::geometry::dynamic_time_warping_work({0,1},{0,1});
/// std::print("{} {}\n",c.result.distance,c.work.cells); // 0 4
/// @endcode
/// @see dynamic_time_warping_work(), DTWTrace
struct DTWComputation {
 DTWResult result; ///< Optimal path and its unnormalized absolute loss.
 DTWWork work; ///< Actual recurrence and traceback counts.
};
/// @brief Selected predecessor of an actual positive DTW cell.
/// advance_a changes the first coordinate, advance_b the second. Finite ties
/// prefer diagonal, then advance_a, then advance_b. Unrepresentable accumulated
/// costs have no finite predecessor witness; successful traceback never uses them.
/// @par Example
/// @code
/// const auto t=dm::geometry::dynamic_time_warping_trace({1},{1});
/// std::print("{}\n",t.predecessors[0]==dm::geometry::DTWPredecessor::origin); // true
/// @endcode
/// @see DTWTrace
enum class DTWPredecessor:std::uint8_t {
 origin, ///< Positive cell (0,0), reached from the implicit zero border origin.
 diagonal, ///< Previous positive cell (i-1,j-1).
 advance_a, ///< Previous positive cell (i-1,j).
 advance_b, ///< Previous positive cell (i,j-1).
 unrepresentable ///< Positive accumulated cost exceeds finite double range.
};
/// @brief Owned input, actual accumulated costs and chosen predecessors from one DTW run.
/// Positive cells are row-major at i*b.size()+j; sentinel border is implicit.
/// Costs and predecessor codes are captured during the original recurrence,
/// not reconstructed afterward. Infinite cells mark unrepresentable costs.
/// Logical capture entries are a.size()+b.size()+2*a.size()*b.size(): one
/// per input scalar, cost and direction. Result paths, rolling scratch rows,
/// scalar metadata and allocator bytes are excluded. Copies are independent.
/// @par Example
/// @code
/// const auto t=dm::geometry::dynamic_time_warping_trace({1},{4});
/// std::print("{} {}\n",t.cumulative[0],t.captured_entries); // 3 4
/// @endcode
/// @see dynamic_time_warping_trace(), dtw_trace_entries()
struct DTWTrace {
 std::vector<double> a; ///< Owned first input, unchanged by alignment.
 std::vector<double> b; ///< Owned second input.
 std::vector<double> cumulative; ///< Actual positive-cell prefix costs, row-major.
 std::vector<DTWPredecessor> predecessors; ///< Actual selected directions, parallel to cumulative.
 DTWResult result; ///< Finite optimal distance and selected actual traceback.
 DTWWork work; ///< Counts from the same recurrence/traceback.
 std::size_t captured_entries=0; ///< Complete logical capture cost described above.
};
/// @brief Compute the complete logical capture size without allocating a DTW matrix.
/// Checks positive lengths, int path-index representability and size_t geometry.
/// A representable count does not promise that the corresponding buffers fit memory.
/// @param n Positive first length, at most INT_MAX.
/// @param m Positive second length, at most INT_MAX.
/// @return n+m+2*n*m logical entries, excluding result paths and scratch rows.
/// @throws std::invalid_argument For either zero length.
/// @throws std::length_error For index, product or capture-count overflow.
/// @par Example
/// @code
/// std::print("{}\n",dm::geometry::dtw_trace_entries(3,4)); // 31
/// @endcode
/// @see dynamic_time_warping_trace()
[[nodiscard]] std::size_t dtw_trace_entries(std::size_t n,std::size_t m);
/// @brief Compute unrestricted absolute-loss DTW and actual logical work without cost history.
/// Owns the result; borrows finite inputs without mutation. No band, slope bound,
/// step weighting, length normalization or fitted parameters. Finite ties choose
/// diagonal, then advance_a, then advance_b. O(n*m) recurrence time, O(n*m+n+m)
/// scratch/output, with only two double cost rows and n*m predecessor bytes.
/// Complete input/geometry validation precedes allocation. Positive overflowing
/// off-path costs remain infinity; only a nonfinite final optimum is rejected.
/// Under ordinary double arithmetic the returned witness sums to the returned
/// distance in forward order. Concurrent independent calls share no mutable state.
/// @param a Nonempty finite scalar sequence, length at most INT_MAX.
/// @param b Nonempty finite scalar sequence, length at most INT_MAX.
/// @return Independent optimal result and counts, with no retained input/cost history.
/// @throws std::invalid_argument For empty or nonfinite input.
/// @throws std::length_error For geometry/index/container bounds.
/// @throws std::overflow_error If the final minimum is not finite double.
/// @throws std::bad_alloc If scratch or result storage cannot be allocated.
/// @par Example
/// See DTWWork and DTWComputation for executed work examples.
/// @par References
/// - H. Sakoe and S. Chiba (1978), Dynamic programming algorithm optimization for
///   spoken word recognition, IEEE TASSP26(1):43-49. doi:10.1109/TASSP.1978.1163055.
///   This is the unrestricted scalar absolute-loss variant, without the paper's window/slope rules.
/// @see dynamic_time_warping(), dynamic_time_warping_trace()
[[nodiscard]] DTWComputation dynamic_time_warping_work(const std::vector<double>& a,const std::vector<double>& b);
/// @brief Preserve the original two-argument unrestricted absolute-loss DTW entry point.
/// Same validation, ownership, deterministic ties, numerical limits and complexity
/// as dynamic_time_warping_work(); discards only its work counters. This distance
/// is not a metric: unequal stretched sequences can have zero cost and the triangle
/// inequality can fail. Equal-length diagonal alignment bounds it by rigid L1 loss.
/// @param a Nonempty finite first sequence, length at most INT_MAX.
/// @param b Nonempty finite second sequence, length at most INT_MAX.
/// @return Owned optimal distance and inclusive zero-based parallel int paths.
/// @throws std::invalid_argument For empty or nonfinite input.
/// @throws std::length_error For geometry/index/container bounds.
/// @throws std::overflow_error If the final optimum is not finite double.
/// @throws std::bad_alloc If scratch or result storage cannot be allocated.
/// @par Example
/// See DTWResult for the executed original stretched-sequence example.
/// @par References
/// - H. Sakoe and S. Chiba (1978), IEEE TASSP26(1):43-49, doi:10.1109/TASSP.1978.1163055.
/// @see dynamic_time_warping_work(), dtw_path_cost()
[[nodiscard]] inline DTWResult dynamic_time_warping(const std::vector<double>& a,const std::vector<double>& b) {
 return dynamic_time_warping_work(a,b).result;
}
/// @brief Capture actual positive-cell costs and predecessor choices from the shared DTW kernel.
/// Validates complete finite inputs/geometry before checking the optional complete
/// capture budget, then copies inputs and runs once. Owns every snapshot. No partial
/// trace/result escapes on failure. Same variant/ties/numerical rules as ordinary DTW.
/// @param a Nonempty finite first input, length at most INT_MAX.
/// @param b Nonempty finite second input, length at most INT_MAX.
/// @param max_entries Complete logical entry cap, default 1000000; zero disables only cap.
/// @return Independent input, costs, directions, optimal result, counts and capture cost.
/// @throws std::invalid_argument For empty or nonfinite input, before budget checks.
/// @throws std::length_error For geometry/index/container/capture limits.
/// @throws std::overflow_error If the final optimum is not finite double.
/// @throws std::bad_alloc If owned or working storage cannot be allocated.
/// @par Example
/// See DTWTrace and DTWPredecessor for executed capture examples.
/// @par References
/// - H. Sakoe and S. Chiba (1978), IEEE TASSP26(1):43-49, doi:10.1109/TASSP.1978.1163055.
/// @see dtw_trace_entries(), dynamic_time_warping_work()
[[nodiscard]] DTWTrace dynamic_time_warping_trace(const std::vector<double>& a,const std::vector<double>& b,std::size_t max_entries=1000000);
/// @brief Validate a complete monotone witness and sum its unshifted absolute losses.
/// Checks all finite input and path geometry before numerical accumulation: equal
/// nonempty path lengths, in-bounds indices, exact endpoints and unit advances
/// (1,0), (0,1), (1,1). This certifies feasibility/cost, not optimality. O(n+m+L)
/// validation/sum, no allocations, input unchanged; no shared mutable state.
/// @param a Nonempty finite first sequence.
/// @param b Nonempty finite second sequence.
/// @param path_a Borrowed signed first indices, zero through a.size()-1.
/// @param path_b Borrowed parallel signed second indices, zero through b.size()-1.
/// @return Finite forward-order sum of absolute differences on the supplied witness.
/// @throws std::invalid_argument For invalid finite inputs or path shape/indices/steps/endpoints.
/// @throws std::length_error For input index/geometry bounds.
/// @throws std::overflow_error If the supplied path's accumulated cost is not finite double.
/// @par Example
/// @code
/// const auto r=dm::geometry::dynamic_time_warping({1,2},{1,2});
/// std::print("{}\n",dm::geometry::dtw_path_cost({1,2},{1,2},r.path_a,r.path_b)); // 0
/// @endcode
/// @see DTWResult, plot_dtw_alignment()
[[nodiscard]] double dtw_path_cost(const std::vector<double>& a,const std::vector<double>& b,const std::vector<int>& path_a,const std::vector<int>& path_b);
/// @brief Draw two scalar series and their validated DTW correspondence segments natively.
/// Costs use unshifted inputs; only drawing adds offset to a and subtracts it from b.
/// Requires result.distance to equal the validated forward witness sum exactly;
/// it does not independently prove optimality. Validates complete inputs/witness,
/// finite plotted coordinates and optional path-pair cap before drawing allocation.
/// Owns all series; no file I/O or model mutation. Concurrent calls share no state.
/// @param a Nonempty finite first sequence.
/// @param b Nonempty finite second sequence.
/// @param result Borrowed complete feasible result with exact finite witness cost.
/// @param offset Finite nonnegative vertical display separation per series, default 1.6.
/// @param max_pairs Optional complete correspondence cap, default 4096; zero disables only cap.
/// @return Editable owned RPlot with two lines and gray correspondence segments.
/// @throws std::invalid_argument For invalid inputs/witness/distance/offset.
/// @throws std::length_error For input geometry/index limits or exceeded nonzero pair cap.
/// @throws std::overflow_error For witness cost or shifted drawing-coordinate overflow.
/// @throws std::bad_alloc If owned plot storage cannot be allocated.
/// @par Example
/// @code
/// const auto r=dm::geometry::dynamic_time_warping({1,2},{1,2});
/// const auto p=dm::geometry::plot_dtw_alignment({1,2},{1,2},r);
/// std::print("{}\n",p.series().size()); // 3
/// @endcode
/// @see dtw_path_cost(), graphics::RPlot
[[nodiscard]] graphics::RPlot plot_dtw_alignment(const std::vector<double>& a,const std::vector<double>& b,const DTWResult& result,double offset=1.6,std::size_t max_pairs=4096);
} // namespace datamunge::geometry
