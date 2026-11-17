#pragma once
/// @file sobel.hpp
/// @brief Checked unnormalized Sobel correlations with actual pixel contributions.
#include <array>
#include <cstddef>
#include <vector>
/// @brief Canonical checked native producers and owned values.
namespace datamunge::algorithms {
/// @brief SobelResult owns actual producer values or explicit invocation policies; fields state units and defaults.
/// @par Example
/// @code
/// const dm::algorithms::SobelResult value{};
/// std::print("{}\n",value.gx.size()); // 0
/// @endcode
struct SobelResult {
  std::vector<std::vector<double>>
      gx; ///< Actual horizontal response of right-positive correlation kernel.
  std::vector<std::vector<double>>
      gy; ///< Actual vertical response, positive toward increasing row index.
  std::vector<std::vector<double>> magnitude; ///< Actual hypot(gx,gy); forced zero at border.
};
/// @brief SobelOptions owns actual producer values or explicit invocation policies; fields state units and defaults.
/// @par Example
/// @code
/// const dm::algorithms::SobelOptions value{};
/// std::print("{}\n",value.max_pixels); // 1000000
/// @endcode
struct SobelOptions {
  std::size_t max_pixels{1000000}; ///< Complete rectangular input cell count; zero disables.
  std::size_t max_history_values{
      1000000}; ///< Captured pixel records plus18 contribution values each; zero disables.
};
/// @brief SobelPixel owns actual producer values or explicit invocation policies; fields state units and defaults.
/// @par Example
/// @code
/// const dm::algorithms::SobelPixel value{};
/// std::print("{}\n",value.gx); // 0
/// @endcode
struct SobelPixel {
  std::size_t row{};    ///< Actual interior input row.
  std::size_t column{}; ///< Actual interior input column.
  std::array<double, 9>
      horizontal_terms{}; ///< Actual row-major neighborhood samples multiplied by kx coefficients.
  std::array<double, 9>
         vertical_terms{}; ///< Actual row-major neighborhood samples multiplied by ky coefficients.
  double gx{};             ///< Actual summed horizontal terms.
  double gy{};             ///< Actual summed vertical terms.
  double magnitude{};      ///< Actual hypot response.
};
/// @brief SobelRun owns actual producer values or explicit invocation policies; fields state units and defaults.
/// @par Example
/// @code
/// const dm::algorithms::SobelRun value{};
/// std::print("{}\n",value.sample_visits); // 0
/// @endcode
struct SobelRun {
  SobelResult result;              ///< Actual full-size responses, borders forced zero.
  std::size_t interior_pixels{};   ///< Actual evaluated interior neighborhoods.
  std::size_t sample_visits{};     ///< Actual nine sample reads per evaluated interior pixel.
  double      maximum_magnitude{}; ///< Actual maximum over evaluated pixels; zero if no interior.
  std::vector<SobelPixel> history; ///< Optional actual producing pixel contributions.
};
/// @brief SobelEdges owns actual producer values or explicit invocation policies; fields state units and defaults.
/// @par Example
/// @code
/// const dm::algorithms::SobelEdges value{};
/// std::print("{}\n",value.threshold); // 0
/// @endcode
struct SobelEdges {
  double threshold{}; ///< Fraction times actual maximum magnitude.
  double maximum{};   ///< Actual scanned maximum magnitude.
  std::vector<std::array<std::size_t, 2>>
      pixels; ///< Actual row,column indices with magnitude strictly greater than threshold.
};
/// @brief Correlate a finite rectangular image with the Sobel masks.
/// Unnormalized kx=[[-1,0,1],[-2,0,2],[-1,0,1]], ky=[[-1,-2,-1],[0,0,0],[1,2,1]]. These are correlations, not flipped-kernel mathematical convolutions. gx increases to the right; gy increases down image rows. All outer border responses are forced zero, so no padded boundary gradient is estimated. Empty/zero-width inputs and images below3 in either dimension have no interior work. Finite rectangular input required; nonrepresentable sums/magnitudes throw. O(rows*columns) time/storage, optional actual history O(interior pixels). Complete cell/history geometry checked before computation. Borrowed input and owned output; no shared state.
/// @param image Complete borrowed finite row-major grayscale values; arbitrary finite intensity scale.
/// @param options Complete pixel and optional actual-contribution capture policies; zero disables policies.
/// @return Owned actual response matrices, counters, maximum and optional pixel histories.
/// @throws std::invalid_argument Ragged rows or nonfinite intensity.
/// @throws std::length_error Pixel/history/counter/container capacity exceeded.
/// @throws std::overflow_error Correlation or magnitude not representable.
/// @par Example
/// @code
/// auto r=dm::algorithms::sobel_run({{0,1,2},{0,1,2},{0,1,2}});
/// std::print("{} {}\n",r.result.gx[1][1],r.sample_visits); // 8 9
/// @endcode
/// @par References
/// - I. Sobel and G. Feldman, A3x3 Isotropic Gradient Operator for Image Processing, Stanford Artificial Intelligence Project,1968.
/// @see sobel()
SobelRun sobel_run(const std::vector<std::vector<double>>& image, const SobelOptions& options = {});
/// @brief Correlate a finite rectangular image with the Sobel masks and capture actual contributions.
/// Unnormalized kx=[[-1,0,1],[-2,0,2],[-1,0,1]], ky=[[-1,-2,-1],[0,0,0],[1,2,1]]. These are correlations, not flipped-kernel mathematical convolutions. gx increases to the right; gy increases down image rows. All outer border responses are forced zero, so no padded boundary gradient is estimated. Empty/zero-width inputs and images below3 in either dimension have no interior work. Finite rectangular input required; nonrepresentable sums/magnitudes throw. O(rows*columns) time/storage, optional actual history O(interior pixels). Complete cell/history geometry checked before computation. Borrowed input and owned output; no shared state.
/// @param image Complete borrowed finite row-major grayscale values; arbitrary finite intensity scale.
/// @param options Complete pixel and optional actual-contribution capture policies; zero disables policies.
/// @return Owned actual response matrices, counters, maximum and optional pixel histories.
/// @throws std::invalid_argument Ragged rows or nonfinite intensity.
/// @throws std::length_error Pixel/history/counter/container capacity exceeded.
/// @throws std::overflow_error Correlation or magnitude not representable.
/// @par Example
/// @code
/// auto r=dm::algorithms::sobel_trace({{0,1,2},{0,1,2},{0,1,2}});
/// std::print("{} {}\n",r.result.gx[1][1],r.sample_visits); // 8 9
/// @endcode
/// @par References
/// - I. Sobel and G. Feldman, A3x3 Isotropic Gradient Operator for Image Processing, Stanford Artificial Intelligence Project,1968.
/// @see sobel()
SobelRun sobel_trace(
    const std::vector<std::vector<double>>& image,
    const SobelOptions&                     options = {});
/// @brief Return the preserved checked Sobel responses.
/// Same actual producer with default policies and forced zero outer border.
/// @param img Complete finite rectangular grayscale matrix.
/// @return Owned gx,gy,magnitude matrices of the same shape.
/// @throws std::invalid_argument Malformed image.
/// @throws std::length_error Default geometry/capacity exceeded.
/// @throws std::overflow_error Nonrepresentable response.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::sobel({{0,1,2},{0,1,2},{0,1,2}}).gx[1][1]); // 8
/// @endcode
/// @see sobel_run()
SobelResult sobel(const std::vector<std::vector<double>>& img);
/// @brief Generate the original32x32 bright-square fixture.
/// Rows/columns8..23 contain255, all others0. Owned deterministic synthetic image, not a photograph or annotated edge dataset.
/// @return Original complete32x32 matrix.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::sobel_square_example()[8][8]); // 255
/// @endcode
/// @see sobel_run()
std::vector<std::vector<double>> sobel_square_example();
/// @brief Select actual magnitude pixels above a relative threshold.
/// Scans the supplied complete rectangular finite nonnegative magnitude matrix, computes its maximum, then selects magnitude>fraction*maximum. Fraction in[0,1]; zero magnitude images produce no edges. gx/gy are not inspected. O(pixels) time/output, at most one million pixels, owned indices, no shared state. This is simple thresholding, not Canny suppression/hysteresis or a semantic-edge classifier.
/// @param result Borrowed magnitude matrix in a Sobel result.
/// @param fraction Finite relative threshold in[0,1], default0.5.
/// @return Actual maximum, threshold and selected row/column indices.
/// @throws std::invalid_argument Malformed/nonfinite/negative magnitude or invalid fraction.
/// @throws std::length_error Complete pixel geometry exceeds one million.
/// @par Example
/// @code
/// auto r=dm::algorithms::sobel_edges(dm::algorithms::sobel({{0,1,2},{0,1,2},{0,1,2}}));
/// std::print("{}\n",r.pixels.size()); // 1
/// @endcode
/// @see sobel_run()
SobelEdges sobel_edges(const SobelResult& result, double fraction = .5);
} // namespace datamunge::algorithms
