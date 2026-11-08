#pragma once
/// @file floyd_steinberg.hpp
/// @brief Checked raster error diffusion with actual floating-point update evidence.
#include <array>
#include <cstddef>
#include <vector>
namespace datamunge::algorithms {
/// @brief Historical cutoff: below128 becomes zero;128 and above becomes255.
/// For arbitrary doubles this differs from a nearest-level cutoff at127.5.
/// @par Example
/// @code
/// std::cout << dm::algorithms::floyd_steinberg_threshold; // 128
/// @endcode
/// @see floyd_steinberg_dither()
inline constexpr double floyd_steinberg_threshold=128.0;
/// @brief One guarded future-neighbor slot and its fixed numerator over16.
/// Outside coordinates are zero placeholders. Direction order is right, down-left,
/// down, down-right with numerators7,3,5,1. No grid storage is retained.
/// @par Example
/// @code
/// dm::algorithms::FloydSteinbergNeighbor n{};
/// std::cout << n.in_bounds; // 0
/// @endcode
/// @see floyd_steinberg_stencil()
struct FloydSteinbergNeighbor {
    std::size_t direction{}; ///< Direction0..3.
    std::size_t numerator{}; ///< Integer numerator of the weight over16.
    bool in_bounds{}; ///< Whether the destination exists.
    std::size_t row{}; ///< Destination row, zero outside.
    std::size_t col{}; ///< Destination column, zero outside.
};
/// @brief Owned fixed-capacity future-neighbor geometry.
/// A valid center has four slots, including outside ones; an invalid center has none.
/// @par Example
/// @code
/// dm::algorithms::FloydSteinbergStencil s{};
/// std::cout << s.count; // 0
/// @endcode
/// @see floyd_steinberg_stencil()
struct FloydSteinbergStencil {
    std::array<FloydSteinbergNeighbor,4> candidates{}; ///< Direction slots; inspect only count entries.
    std::size_t count{}; ///< Four for a valid center, otherwise zero.
};
/// @brief Enumerate guarded raster diffusion destinations without accessing a grid.
/// No allocation, narrowing or dimension multiplication; SIZE_MAX dimensions work.
/// All four weights are retained even at edges; outside coordinates are zero.
/// @param rows Number of rows, possibly zero.
/// @param cols Number of columns, possibly zero.
/// @param row Center row; an outside center returns no slots.
/// @param col Center column; an outside center returns no slots.
/// @return Owned right/down-left/down/down-right slots with numerators7/3/5/1.
/// @par Example
/// @code
/// const auto s=dm::algorithms::floyd_steinberg_stencil(2,2,0,0);
/// std::cout << s.count << ' ' << s.candidates[1].in_bounds; // 4 0
/// @endcode
/// @see floyd_steinberg_dither()
FloydSteinbergStencil floyd_steinberg_stencil(std::size_t rows,std::size_t cols,
    std::size_t row,std::size_t col) noexcept;
/// @brief Actual raster counts and signed tone-balance diagnostics.
/// Nominal grayscale units are0..255; finite values outside that range are accepted
/// without clamping. Boundary loss is the sum of error times discarded weight in
/// long double, not an absolute error. Balance residual includes floating rounding.
/// Empty images have zero counts/sums/extrema. Diagnostics are not perceptual scores.
/// @par Example
/// @code
/// dm::algorithms::FloydSteinbergWork w{};
/// std::cout << w.pixels; // 0
/// @endcode
/// @see floyd_steinberg_work()
struct FloydSteinbergWork {
    std::size_t rows{}; ///< Validated row count.
    std::size_t cols{}; ///< Common row width.
    std::size_t pixels{}; ///< Processed cell count.
    std::size_t white{}; ///< Output255 count.
    std::size_t black{}; ///< Output0 count.
    std::size_t neighbor_slots{}; ///< Four attempted directions per pixel.
    std::size_t updates{}; ///< Actual in-bounds destination updates.
    std::size_t discarded_slots{}; ///< Outside directions, not renormalized.
    long double input_sum{}; ///< Original intensity sum.
    long double output_sum{}; ///< Binary output intensity sum.
    long double boundary_loss{}; ///< Signed discarded weighted errors.
    long double balance_residual{}; ///< Input sum minus output sum minus boundary loss.
    double working_min{}; ///< Minimum actual pre-quantization value, zero empty.
    double working_max{}; ///< Maximum actual pre-quantization value, zero empty.
};
/// @brief Owned binary integer output and actual native work.
/// Values are0or255; storage is an int grid, not a packed one-bit bitmap.
/// @par Example
/// @code
/// dm::algorithms::FloydSteinbergResult r{};
/// std::cout << r.output.size(); // 0
/// @endcode
/// @see floyd_steinberg_work()
struct FloydSteinbergResult {
    std::vector<std::vector<int>> output{}; ///< Owned row-major binary values.
    FloydSteinbergWork work{}; ///< Actual counts and signed balance diagnostics.
};
/// @brief One actual floating-point update to a future cell.
/// Only in-bounds updates are stored, in the original direction order. The transfer
/// is computed as error*numerator/16.0 in double, preserving the historical order.
/// @par Example
/// @code
/// dm::algorithms::FloydSteinbergUpdate u{};
/// std::cout << u.direction; // 0
/// @endcode
/// @see FloydSteinbergTrace
struct FloydSteinbergUpdate {
    std::size_t direction{}; ///< Right0/down-left1/down2/down-right3.
    std::size_t row{}; ///< Destination row.
    std::size_t col{}; ///< Destination column.
    double value_before{}; ///< Destination before addition.
    double transfer{}; ///< Actual double weighted transfer.
    double value_after{}; ///< Destination after addition.
};
/// @brief One actual raster quantization and its update range.
/// Boundary loss uses the missing numerator over16 in long double. Values already
/// include all prior diffusion; they are not clamped to the nominal grayscale range.
/// @par Example
/// @code
/// dm::algorithms::FloydSteinbergStep s{};
/// std::cout << s.update_begin; // 0
/// @endcode
/// @see floyd_steinberg_trace()
struct FloydSteinbergStep {
    std::size_t row{}; ///< Processed row.
    std::size_t col{}; ///< Processed column.
    double value_before{}; ///< Actual pre-quantization value.
    int quantized{}; ///< Actual0or255 decision at cutoff128.
    double error{}; ///< Actual double value_before minus quantized.
    std::size_t discarded_numerator{}; ///< Sum of outside direction numerators,0..16.
    long double boundary_loss{}; ///< Signed discarded share for this step.
    std::size_t update_begin{}; ///< Inclusive index in the updates vector.
    std::size_t update_end{}; ///< Exclusive index in the updates vector.
};
/// @brief Owned original/output grids and actual raster/update evidence.
/// Records are captured during the ordinary kernel, without replay. Caller storage
/// is never retained or modified; failures return no partial result.
/// @par Example
/// @code
/// const auto t=dm::algorithms::floyd_steinberg_trace({{128.0}});
/// std::cout << t.output[0][0]; // 255
/// @endcode
/// @see floyd_steinberg_trace()
struct FloydSteinbergTrace {
    std::vector<std::vector<double>> input{}; ///< Owned original finite intensities.
    std::vector<std::vector<int>> output{}; ///< Owned binary output.
    FloydSteinbergWork work{}; ///< Actual scalar measurements.
    std::vector<FloydSteinbergStep> steps{}; ///< One actual row-major step per pixel.
    std::vector<FloydSteinbergUpdate> updates{}; ///< Only actual in-bounds updates, at most four per pixel.
};
/// @brief Dither an owned grayscale copy, preserving the original by-value API.
/// Row-major left-to-right processing uses cutoff128, no clamping and ordered
/// error*numerator/16.0 updates with7/3/5/1 to right/down-left/down/down-right.
/// Outside shares are discarded without renormalization; local/global means need
/// not match exactly. All rows are validated before any finite-value checks.
/// Every source value must be finite; values outside nominal0..255 are accepted.
/// Actual double arithmetic must remain finite. No shared mutable state.
/// O(rows+pixels) work and O(rows+pixels) owned scratch plus returned int storage.
/// Lvalue input is copied by the language before body validation; no second input
/// copy is made. Caller input is unchanged; an rvalue may be consumed.
/// @param image Owned rectangular finite intensities; empty/zero-width rows valid.
/// @return Owned0/255 integer grid with the input shape; not packed one-bit data.
/// @throws std::invalid_argument Ragged rows or any nonfinite source value.
/// @throws std::length_error Checked cell/work product or container capacity exhausted.
/// @throws std::overflow_error Actual transfer/update or diagnostic becomes nonfinite.
/// @throws std::bad_alloc Scratch/output allocation fails.
/// @par Example
/// @code
/// const auto d=dm::algorithms::floyd_steinberg_dither({{127.75,128.0}});
/// std::cout << d[0][0]; // 0
/// @endcode
/// @par References
/// - R.W.Floyd and L.Steinberg (1976), An Adaptive Algorithm for Spatial Greyscale,
///   Proceedings SID17(2),75-77; bibliographic reference.
/// - J.M.Speigle and X.-f.Feng (2000), Enhanced Error Diffusion with Thresholds
///   Modulated by Spatio-Chromatically Correlated Noise, section3 describes
///   the traditional128 cutoff and weights; its later modifications are not used.
///   https://corporate.jp.sharp/rd/journal-76/pdf/76-05.pdf
/// @see floyd_steinberg_work(), floyd_steinberg_trace()
std::vector<std::vector<int>> floyd_steinberg_dither(std::vector<std::vector<double>> image);
/// @brief Return owned output and actual work without retaining histories.
/// Row-major left-to-right processing uses cutoff128, no clamping and ordered
/// error*numerator/16.0 updates with7/3/5/1 to right/down-left/down/down-right.
/// Outside shares are discarded without renormalization; local/global means need
/// not match exactly. All rows are validated before any finite-value checks.
/// Every source value must be finite; values outside nominal0..255 are accepted.
/// Actual double arithmetic must remain finite. No shared mutable state.
/// Validates before copying caller input. O(rows+pixels) time and scratch/output
/// storage. Separate input/output values do not alias the caller.
/// @param image Read-only rectangular finite intensities.
/// @return Owned binary result and actual counts/boundary-loss diagnostics.
/// @throws std::invalid_argument Ragged rows or nonfinite source.
/// @throws std::length_error Checked work product or container capacity exhausted.
/// @throws std::overflow_error Actual double arithmetic or diagnostics become nonfinite.
/// @throws std::bad_alloc Scratch/output allocation fails.
/// @par Example
/// @code
/// const auto r=dm::algorithms::floyd_steinberg_work({{128.,128.},{128.,128.}});
/// std::cout << r.work.white << ' ' << r.work.boundary_loss; // 2 2
/// @endcode
/// @par References
/// - R.W.Floyd and L.Steinberg (1976), An Adaptive Algorithm for Spatial Greyscale,
///   Proceedings SID17(2),75-77; bibliographic reference.
/// - J.M.Speigle and X.-f.Feng (2000), Enhanced Error Diffusion with Thresholds
///   Modulated by Spatio-Chromatically Correlated Noise, section3 describes
///   the traditional128 cutoff and weights; its later modifications are not used.
///   https://corporate.jp.sharp/rd/journal-76/pdf/76-05.pdf
/// @see floyd_steinberg_dither(), FloydSteinbergWork
FloydSteinbergResult floyd_steinberg_work(const std::vector<std::vector<double>>& image);
/// @brief Copy validated input and record the actual quantizations and updates.
/// Row-major left-to-right processing uses cutoff128, no clamping and ordered
/// error*numerator/16.0 updates with7/3/5/1 to right/down-left/down/down-right.
/// Outside shares are discarded without renormalization; local/global means need
/// not match exactly. All rows are validated before any finite-value checks.
/// Every source value must be finite; values outside nominal0..255 are accepted.
/// Actual double arithmetic must remain finite. No shared mutable state.
/// Before copying, preflights complete upper entry bound2*rows+7*pixels:
/// two outer-row arrays, two cell arrays, one step and at most four updates per
/// cell. Temporary scratch, bytes and peak RAM are not the quota units. Default
/// cap one million; zero disables only the optional cap. Checked arithmetic and
/// container limits remain. Actual histories retain only valid updates.
/// O(rows+pixels) work/storage; no replay or caller mutation, no partial result.
/// @param image Read-only rectangular finite intensities.
/// @param max_entries Whole upper entry cap, default one million; zero unlimited.
/// @return Independent input/output, work and actual row-major histories.
/// @throws std::invalid_argument Ragged rows or nonfinite source, before quota checks.
/// @throws std::length_error Whole bound overflows/exceeds cap or a container limit.
/// @throws std::overflow_error Actual arithmetic or diagnostics become nonfinite.
/// @throws std::bad_alloc Copies or history allocation fail.
/// @par Example
/// @code
/// const auto t=dm::algorithms::floyd_steinberg_trace({{128.}});
/// std::cout << t.steps.size() << ' ' << t.updates.size(); // 1 0
/// @endcode
/// @par References
/// - R.W.Floyd and L.Steinberg (1976), An Adaptive Algorithm for Spatial Greyscale,
///   Proceedings SID17(2),75-77; bibliographic reference.
/// - J.M.Speigle and X.-f.Feng (2000), Enhanced Error Diffusion with Thresholds
///   Modulated by Spatio-Chromatically Correlated Noise, section3 describes
///   the traditional128 cutoff and weights; its later modifications are not used.
///   https://corporate.jp.sharp/rd/journal-76/pdf/76-05.pdf
/// @see floyd_steinberg_work(), FloydSteinbergTrace
FloydSteinbergTrace floyd_steinberg_trace(const std::vector<std::vector<double>>& image,
    std::size_t max_entries=1000000);
/// @brief Apply the same128 cutoff independently, without diffusion.
/// Full rectangular/finite validation precedes output allocation. Finite values
/// outside0..255 are accepted without clamping. O(rows+pixels) time/output storage.
/// @param image Read-only rectangular finite intensities; empty shapes valid.
/// @return Owned0/255 grid with matching shape.
/// @throws std::invalid_argument Ragged rows or nonfinite values.
/// @throws std::length_error Shape product or container capacity exhausted.
/// @throws std::bad_alloc Output allocation fails.
/// @par Example
/// @code
/// const auto d=dm::algorithms::threshold_dither({{127.75,128.}});
/// std::cout << d[0][0] << ' ' << d[0][1]; // 0 255
/// @endcode
/// @see floyd_steinberg_dither(), floyd_steinberg_threshold
std::vector<std::vector<int>> threshold_dither(const std::vector<std::vector<double>>& image);
/// @brief Construct the original deterministic left-to-right0..255 grayscale ramp.
/// Every row uses255.0*column/(cols-1) in double; a single column is defined as0.
/// Complete product/cap preflight precedes allocation. O(rows*cols) work/storage.
/// No randomness, dataset, mutation or shared state.
/// @param rows Positive row count.
/// @param cols Positive column count.
/// @param max_pixels Whole cell cap, default one million; zero unlimited.
/// @return Fresh owned grayscale rows.
/// @throws std::invalid_argument Either dimension is zero.
/// @throws std::length_error Product/cap/container bound exhausted.
/// @throws std::bad_alloc Grid allocation fails.
/// @par Example
/// @code
/// const auto g=dm::algorithms::dither_gradient(1,3);
/// std::cout << g[0][1]; // 127.5
/// @endcode
/// @see floyd_steinberg_dither()
std::vector<std::vector<double>> dither_gradient(std::size_t rows,std::size_t cols,
    std::size_t max_pixels=1000000);
/// @brief Column-band input/output means and actual binary density.
/// Empty bands have zero means/fraction and pixels0. These are arithmetic summaries,
/// not perceptual fidelity scores or proof of an algorithm having produced the output.
/// @par Example
/// @code
/// dm::algorithms::DitherRegionStats s{};
/// std::cout << s.pixels; // 0
/// @endcode
/// @see dither_region_stats()
struct DitherRegionStats {
    std::size_t pixels{}; ///< Number of queried cells.
    std::size_t white{}; ///< Queried255 outputs.
    std::size_t black{}; ///< Queried0 outputs.
    long double input_mean{}; ///< Source mean, zero empty.
    long double output_mean{}; ///< Output mean, zero empty.
    long double white_fraction{}; ///< White count divided by pixels, zero empty.
    long double mean_error{}; ///< Output mean minus source mean, zero empty.
};
/// @brief Validate complete images, then summarize one half-open column band.
/// Both grids must have identical rectangular shapes. Every source value must be
/// finite and every output value0or255, including outside the queried band.
/// Full validation precedes range checks. Empty shapes/bands return zero summaries.
/// O(rows+pixels) validation/work; O(1) extra storage, no mutation/shared state.
/// @param input Read-only finite grayscale source; outside0..255 permitted.
/// @param output Read-only0/255 integer grid with matching shape.
/// @param col_begin Inclusive column index, at most the width.
/// @param col_end Exclusive index, between col_begin and the width.
/// @return Owned counts, means and signed output-minus-input mean error.
/// @throws std::invalid_argument Invalid shapes, source/output values or column range.
/// @throws std::length_error Shape product cannot be represented.
/// @throws std::overflow_error A sum becomes nonfinite.
/// @par Example
/// @code
/// const auto s=dm::algorithms::dither_region_stats({{0.,255.}},{{0,255}},0,2);
/// std::cout << s.white_fraction; // 0.5
/// @endcode
/// @see DitherRegionStats, floyd_steinberg_work()
DitherRegionStats dither_region_stats(const std::vector<std::vector<double>>& input,
    const std::vector<std::vector<int>>& output,std::size_t col_begin,std::size_t col_end);
} // namespace datamunge::algorithms
