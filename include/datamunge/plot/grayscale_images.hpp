#pragma once
/// @file grayscale_images.hpp
/// @brief Native equal-scale grayscale raster panels, with black zero and white255.
#include <datamunge/plot/diagram.hpp>
#include <cstddef>
#include <string>
#include <vector>
namespace datamunge::graphics {
/// @brief Owned rectangular grayscale values and a panel title.
/// Rendering requires finite values in0..255; the source values are never changed.
/// @par Example
/// @code
/// dm::graphics::GrayscaleImagePanel p{{{0.,128.,255.}}, "Gray"};
/// std::cout << p.values[0][1]; // 128
/// @endcode
/// @see diagram_grayscale_images()
struct GrayscaleImagePanel {
    std::vector<std::vector<double>> values; ///< Owned row-major grayscale values.
    std::string title; ///< UTF-8 heading, copied into the diagram.
};
/// @brief Draw one or two grayscale grids as equal-scale native pixel cells.
/// Zero is black and255 white. Display colors round to the nearest8-bit gray;
/// numeric source values and algorithms are untouched. Rows increase downwards.
/// Default color-only cells have invisible borders and ten-pixel native units;
/// optional four-significant-digit labels widen cells to fit. The figure scales
/// uniformly to at most1100px width. Color-only views suit halftone patterns.
/// Full shape/finite/range validation precedes the complete total-cell cap and
/// rendering. Empty grids are valid. One or two panels only; diagrams own their
/// primitives and retain no input views. O(total rows+cells) work/storage.
/// No input mutation or shared mutable state.
/// @param panels One or two rectangular finite0..255 grayscale descriptions.
/// @param show_values Print compact numeric values when true; default false.
/// @param max_cells Complete cell cap, default10000; zero unlimited.
/// @return Owned editable native diagram with equal horizontal/vertical units.
/// @throws std::invalid_argument Panel count, ragged rows or invalid grayscale values.
/// @throws std::length_error Cell sum overflows or exceeds a positive cap.
/// @throws std::bad_alloc Diagram/text allocation fails.
/// @par Example
/// @code
/// const auto d=dm::graphics::diagram_grayscale_images({{{{0.,255.}},"Binary"}});
/// std::cout << d.to_svg().empty(); // 0
/// @endcode
/// @see GrayscaleImagePanel, Diagram
Diagram diagram_grayscale_images(const std::vector<GrayscaleImagePanel>& panels,
    bool show_values=false,std::size_t max_cells=10000);
} // namespace datamunge::graphics
