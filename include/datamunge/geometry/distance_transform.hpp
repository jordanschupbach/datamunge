#pragma once
/// @file distance_transform.hpp
/// @brief Separable Euclidean distance fields and actual one-dimensional parabola envelopes.
#include <datamunge/plot/plot.hpp>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>
namespace datamunge::geometry {
/// @brief Logical work of actual one-dimensional envelope sweeps, aggregated for a grid.
/// Counts exclude validation, output initialization, copying and square roots. Empty-site
/// lines scan inputs but evaluate no finite parabola. All counters are checked against wrap.
/// @par Example
/// See squared_distance_transform_1d_trace().
/// @see DistanceTransformTrace
struct DistanceTransformWork {
    std::size_t lines{0}; ///< One per invoked 1-D transform; zero for an empty grid shape.
    std::size_t input_values{0}; ///< Samples inspected during envelope construction, including infinity.
    std::size_t finite_candidates{0}; ///< Finite-cost parabolas considered.
    std::size_t intersections{0}; ///< Actual candidate-versus-last-envelope intersection tests.
    std::size_t envelope_pushes{0}; ///< Candidate insertions, one per finite candidate.
    std::size_t envelope_pops{0}; ///< Previously inserted candidates removed from the envelope.
    std::size_t selection_advances{0}; ///< Forward winner-pointer moves while evaluating integer samples.
    std::size_t evaluations{0}; ///< Finite-parabola cost evaluations, one per output on nonempty envelopes.
    std::size_t max_envelope_size{0}; ///< Largest active envelope during any line, not a sum over lines.
};
/// @brief Operands and decision of one actual parabola-envelope comparison.
/// Candidate q is to the right of previous p. A crossing at or before the previous
/// left boundary removes p; otherwise q is appended with this new boundary.
/// Wide boundaries are retained rather than truncated to double.
/// @par Example
/// See squared_distance_transform_1d_trace().
/// @see SquaredDistanceTrace
struct ParabolaComparison {
    std::size_t candidate; ///< New finite-cost sample index q.
    std::size_t previous; ///< Last active sample index p before comparison.
    long double intersection; ///< Crossing of the two equal-curvature parabolas.
    long double previous_begin; ///< Current left boundary of p, possibly negative infinity.
    bool removed; ///< intersection <= previous_begin; p was removed before the next comparison.
};
/// @brief Owned actual one-dimensional generalized squared-distance transform and envelope.
/// values[q]=min_p(costs[p]+(q-p)^2). Exact ties select the smallest finite p.
/// All-infinity input has infinity values, nearest sentinel costs.size(), and empty envelope.
/// Empty input yields empty vectors. The finite-cost objective may be negative; this is a
/// min-plus cost transform, not a square-root distance for arbitrary input costs.
/// @par Example
/// See squared_distance_transform_1d_trace().
/// @see ParabolaComparison
struct SquaredDistanceTrace {
    std::vector<double> costs; ///< Owned original input for a captured trace; no borrowed views.
    std::vector<double> values; ///< Transformed double costs.
    std::vector<std::size_t> nearest_index; ///< Selected finite p, or input length when no candidate exists.
    std::vector<std::size_t> envelope_sites; ///< Active candidates in increasing index order.
    std::vector<long double> breakpoints; ///< One more than active candidates, with -/+infinity ends; empty if none.
    std::vector<ParabolaComparison> comparisons; ///< Actual construction decisions, without replay or full snapshots.
    DistanceTransformWork work; ///< Actual envelope and evaluation counters.
};
/// @brief Owned distances, squared distances and nearest-site coordinates on a rectangular unit grid.
/// Nonzero input cells are sites. Ties choose smaller column, then smaller row, following
/// column-first separability. Without sites, both distance fields are positive infinity,
/// nearest_row uses row count and nearest_column uses column count. Rows are numbered downward.
/// Distances are between cell centres; no padding, outside sites, obstacles or path-cost model.
/// @par Example
/// See euclidean_distance_transform_features().
/// @see DistanceTransformTrace
struct EuclideanDistanceField {
    std::vector<std::vector<double>> distance; ///< Euclidean centre distances in grid units.
    std::vector<std::vector<double>> squared_distance; ///< Squared grid-unit distances before square roots.
    std::vector<std::vector<std::size_t>> nearest_row; ///< Chosen site row, or number of rows.
    std::vector<std::vector<std::size_t>> nearest_column; ///< Chosen site column, or number of columns.
};
/// @brief Owned column-pass intermediates and final feature field from one execution.
/// No full per-line histories are retained. Storage and work are O(rows*columns), plus
/// O(rows+columns) line scratch. The optional cell cap excludes caller input storage.
/// @par Example
/// See euclidean_distance_transform_trace().
/// @see EuclideanDistanceField
struct DistanceTransformTrace {
    EuclideanDistanceField result; ///< Final distances and selected original sites.
    std::vector<std::vector<double>> column_squared; ///< min_site_row (row-site_row)^2 in each original column.
    std::vector<std::vector<std::size_t>> column_nearest_row; ///< Chosen vertical site row, or row-count sentinel.
    DistanceTransformWork work; ///< Combined actual column and row sweep counters.
};
/// @brief Computes min_p(costs[p]+(q-p)^2) at integer sample locations in linear time.
/// Uses the Felzenszwalb-Huttenlocher lower envelope. Finite costs of either sign and
/// positive infinity are accepted; infinity candidates are skipped, so no infinity-minus-
/// infinity arithmetic or finite missing-site sentinel is used. Empty input returns empty;
/// all-infinity input returns infinity. Indices use size_t and are converted before products.
/// Intersections/evaluations use long double and output costs use double; the exact metric
/// algorithm still has floating-point rounding, not arbitrary-precision guarantees.
/// O(n) work/storage, no caller mutation, shared state or captured comparison histories.
/// @param costs Borrowed finite costs or positive infinity; NaN/negative infinity are invalid.
/// @return Owned transformed double costs, same length as input.
/// @throws std::invalid_argument A NaN or negative-infinity cost is present.
/// @throws std::length_error Container or counter capacity exceeded.
/// @throws std::overflow_error A transformed cost or crossing is nonfinite/unrepresentable.
/// @par Example
/// @code
/// const double inf=std::numeric_limits<double>::infinity();
/// std::print("{}\n",dm::geometry::squared_distance_transform_1d({0,inf,9,inf,0})); // [0, 1, 4, 1, 0]
/// @endcode
/// @par References
/// - P. F. Felzenszwalb and D. P. Huttenlocher (2012), Distance Transforms of Sampled
///   Functions. Theory of Computing 8, 415-428. https://doi.org/10.4086/toc.2012.v008a019
/// @see squared_distance_transform_1d_trace()
[[nodiscard]] std::vector<double> squared_distance_transform_1d(const std::vector<double>& costs);
/// @brief Captures the actual lower-envelope comparisons, final boundaries and chosen samples.
/// Shares the ordinary 1-D kernel, with the smallest-index tie convention. Storage remains
/// O(n), with at most two construction comparisons per finite candidate. No prefix replay.
/// Complete input validation and the optional sample cap precede envelope/output allocation.
/// @param costs Finite or positive-infinity input costs.
/// @param max_samples Complete input cap, default 1000000; zero disables only this cap.
/// @return Owned input, output, nearest indices, envelope, comparisons and work.
/// @throws std::invalid_argument NaN or negative infinity in costs.
/// @throws std::length_error Optional cap, container or counter capacity exceeded.
/// @throws std::overflow_error Unrepresentable output or crossing.
/// @par Example
/// @code
/// const double inf=std::numeric_limits<double>::infinity();
/// const auto t=dm::geometry::squared_distance_transform_1d_trace({0,inf,9,inf,0});
/// std::print("{} {} {}\n",t.envelope_sites,t.work.intersections,t.work.envelope_pops); // [0, 4] 3 1
/// @endcode
/// @par References
/// - Felzenszwalb and Huttenlocher (2012), doi:10.4086/toc.2012.v008a019; records the executing envelope variant.
/// @see SquaredDistanceTrace
[[nodiscard]] SquaredDistanceTrace squared_distance_transform_1d_trace(const std::vector<double>& costs,std::size_t max_samples=1000000);
/// @brief Returns Euclidean distances to nonzero site cells, preserving the original convenience API.
/// Uses column then row squared transforms and a final square root. No artificial outside
/// background or wrapping. Rejects ragged grids. Empty input returns empty; R zero-width
/// rows return R empty output rows. No sites gives positive infinity. O(R*C+R) work and
/// O(R*C+R+C) storage including line scratch, without captured intermediate matrices.
/// This is the exact Euclidean metric algorithm rather than a chamfer approximation;
/// returned floating distances and square roots remain subject to rounding.
/// @param grid Borrowed rectangular char grid; every nonzero value marks a site.
/// @return Owned row-major distance matrix of the same shape.
/// @throws std::invalid_argument Ragged input.
/// @throws std::length_error Shape/counter/container capacity exceeded.
/// @throws std::overflow_error Unrepresentable transformed cost/crossing.
/// @par Example
/// @code
/// const auto d=dm::geometry::euclidean_distance_transform({{1,0},{0,0}});
/// std::print("{:.6f}\n",d[1][1]); // 1.414214
/// @endcode
/// @par References
/// - Felzenszwalb and Huttenlocher (2012), doi:10.4086/toc.2012.v008a019.
/// @see euclidean_distance_transform_features(), euclidean_distance_transform_trace()
[[nodiscard]] std::vector<std::vector<double>> euclidean_distance_transform(const std::vector<std::vector<char>>& grid);
/// @brief Computes squared/Euclidean fields and original nearest-site coordinates in one fit.
/// Same validation, shape, metric and column-then-row tie behavior as the convenience API.
/// Input is unchanged; all arrays are owned. O(R*C+R) work and O(R*C+R+C) storage.
/// @param grid Rectangular char mask, nonzero means site.
/// @return Owned distance, squared-distance and nearest-row/column matrices.
/// @throws std::invalid_argument Ragged input.
/// @throws std::length_error Shape/counter/container capacity exceeded.
/// @throws std::overflow_error Unrepresentable transformed cost/crossing.
/// @par Example
/// @code
/// const auto f=dm::geometry::euclidean_distance_transform_features({{1,0},{0,0}});
/// std::print("{:.0f} {} {}\n",f.squared_distance[1][1],f.nearest_row[1][1],f.nearest_column[1][1]); // 2 0 0
/// @endcode
/// @par References
/// - Felzenszwalb and Huttenlocher (2012), doi:10.4086/toc.2012.v008a019 (separability).
/// @see EuclideanDistanceField
[[nodiscard]] EuclideanDistanceField euclidean_distance_transform_features(const std::vector<std::vector<char>>& grid);
/// @brief Captures the actual column pass, selected sites, final fields and combined sweep work.
/// No per-line comparison histories or caller views are retained. Same tie/shape/metric
/// contracts as the feature API; optional complete-cell cap is checked before output allocation.
/// @param grid Rectangular input mask, including empty/zero-width shapes.
/// @param max_cells Optional cell cap, default 1000000; zero disables only this cap.
/// @return Owned intermediate column matrices, final features and logical counters.
/// @throws std::invalid_argument Ragged input.
/// @throws std::length_error Shape, optional cap, counter or container capacity exceeded.
/// @throws std::overflow_error Unrepresentable transformed cost/crossing.
/// @par Example
/// @code
/// const auto t=dm::geometry::euclidean_distance_transform_trace({{1,0},{0,0}});
/// std::print("{:.0f} {}\n",t.column_squared[1][0],t.work.input_values); // 1 8
/// @endcode
/// @par References
/// - Felzenszwalb and Huttenlocher (2012), doi:10.4086/toc.2012.v008a019.
/// @see DistanceTransformTrace
[[nodiscard]] DistanceTransformTrace euclidean_distance_transform_trace(const std::vector<std::vector<char>>& grid,std::size_t max_cells=1000000);
/// @brief Builds a bounded rectangular zero mask with explicitly listed site cells set to one.
/// Duplicates are harmless. Validates every coordinate and count before allocation, even
/// for empty shapes. No random/shared state; O(R*C+R+sites) work and O(R*C+R) owned storage.
/// @param rows Row count, including zero.
/// @param columns Column count, including zero.
/// @param sites Borrowed (row,column) pairs; each coordinate must lie inside the shape.
/// @param max_cells Complete cell cap, default 1000000; zero disables only this cap.
/// @return Owned rectangular char grid.
/// @throws std::invalid_argument Any site is out of range.
/// @throws std::length_error Count/container capacity or optional cap exceeded.
/// @par Example
/// @code
/// const auto g=dm::geometry::distance_site_grid(4,5,{{0,0}});
/// std::print("{:.0f}\n",dm::geometry::euclidean_distance_transform(g)[3][4]); // 5
/// @endcode
/// @see euclidean_distance_transform()
[[nodiscard]] std::vector<std::vector<char>> distance_site_grid(std::size_t rows,std::size_t columns,
    const std::vector<std::pair<std::size_t,std::size_t>>& sites,std::size_t max_cells=1000000);
/// @brief Plots a distance field with downward row labels, pale-to-dark blue cells and black sites.
/// Uses only field.distance, so manually constructed fields need not provide other arrays.
/// Values must be nonnegative finite or positive infinity; infinity cells are grey. Optional
/// numeric labels omit zero sites. Returns a plot without file I/O; work/storage O(cells).
/// Cells are plot rectangles; numeric distances use the unit-grid metric regardless of canvas size.
/// @param field Borrowed nonempty rectangular distance matrix.
/// @param annotate Draw finite nonzero numeric cell labels, default false.
/// @param max_cells Complete polygon cap, default 4096; zero disables only this cap.
/// @return Owned RPlot with colour-range/site legend and row/column axis labels.
/// @throws std::invalid_argument Empty/ragged field or negative/NaN distance.
/// @throws std::length_error Count or optional polygon cap exceeded.
/// @par Example
/// @code
/// const auto f=dm::geometry::euclidean_distance_transform_features({{1,0},{0,0}});
/// const auto p=dm::geometry::plot_distance_field(f);
/// std::print("plot ready\n"); // plot ready
/// @endcode
/// @see EuclideanDistanceField, graphics::RPlot
[[nodiscard]] graphics::RPlot plot_distance_field(const EuclideanDistanceField& field,bool annotate=false,std::size_t max_cells=4096);
/// @brief Compatibility helpers retained for existing internal callers.
namespace detail {
/// @brief Forwards the former internal 1-D spelling to the canonical compiled kernel.
/// @param f Finite costs or positive infinity; empty is accepted.
/// @return Owned generalized squared-distance values.
/// @throws std::invalid_argument NaN or negative infinity.
/// @throws std::length_error Capacity/counter overflow.
/// @throws std::overflow_error Unrepresentable transformed cost/crossing.
/// @see squared_distance_transform_1d()
inline std::vector<double> dt_1d(const std::vector<double>& f){return squared_distance_transform_1d(f);}
} // namespace detail
} // namespace datamunge::geometry
