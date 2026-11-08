#pragma once
/// @file label_images.hpp
/// @brief Native discrete-label grid figures with aligned panels and explicit pixel cells.
#include <datamunge/plot/diagram.hpp>
#include <cstddef>
#include <string>
#include <vector>
namespace datamunge::graphics {
/// @brief Owned rectangular nonnegative label grid and its panel heading.
/// Zero is background; positive IDs may have gaps, allowing provisional maps.
/// Storage is independent of the labeller. Colours repeat after eight positive IDs.
/// @par Example
/// @code
/// dm::graphics::LabelImagePanel p{{{1,0},{0,2}}, "Labels"};
/// std::cout << p.labels[1][1]; // 2
/// @endcode
/// @see diagram_label_images()
struct LabelImagePanel {
    std::vector<std::vector<int>> labels; ///< Owned rectangular nonnegative labels.
    std::string title;                    ///< UTF-8 panel heading.
};
/// @brief Draw one or two label grids as native equal-scale cell diagrams.
/// Rows increase downwards. Zero cells are white; positive IDs use eight cycling
/// pastel colours, so colour alone is not a unique component identifier. Optional
/// numeric labels include background zeros. Cells widen for the longest integer
/// label; the complete figure fits within 1100 pixels. Full shape/value validation
/// precedes the complete total-cell cap and rendering. Empty grids are valid;
/// zero or more than two panels is invalid. No mutation/global state.
/// O(total rows + total cells) construction work and returned diagram storage.
/// @param panels One or two owned panel descriptions; labels must be nonnegative.
/// @param show_labels Print integer IDs when true (default); false uses colour only.
/// @param max_cells Complete total-cell cap, default ten thousand; zero unlimited.
/// @return Owned editable native diagram; caller saves it to a local image file.
/// @throws std::invalid_argument Invalid panel count, ragged rows or negative labels.
/// @throws std::length_error Total cell count overflows or exceeds its positive cap.
/// @throws std::bad_alloc Diagram allocation fails.
/// @par Example
/// @code
/// const auto d = dm::graphics::diagram_label_images({{{{1,0},{0,2}}, "Labels"}});
/// std::cout << d.to_svg().empty(); // 0
/// @endcode
/// @see Diagram, LabelImagePanel
Diagram diagram_label_images(const std::vector<LabelImagePanel>& panels,
    bool show_labels = true, std::size_t max_cells = 10000);
/// @brief Owned rectangular signed-value grid and its panel heading.
/// Values are categories, not magnitudes; every int, including INT_MIN, is valid.
/// @par Example
/// @code
/// dm::graphics::ValueImagePanel p{{{-1,0,9}}, "Values"};
/// std::cout << p.values[0][0]; // -1
/// @endcode
/// @see diagram_value_images()
struct ValueImagePanel {
    std::vector<std::vector<int>> values; ///< Owned rectangular signed values.
    std::string title; ///< UTF-8 panel heading.
};
/// @brief Draw one or two signed categorical grids with consistent shared colors.
/// Zero is white. The sorted union of nonzero values across the supplied panels
/// selects an eight-color cycle; a value has the same color within this figure,
/// but its color can differ across separate calls. More than eight categories
/// reuse colors, so retain labels when identity matters. Rows increase downwards;
/// equal-scale cells widen to fit signed labels, with maximum figure width1100px.
/// Complete shape validation precedes the total-cell cap and palette allocation.
/// Empty grids are valid; the returned editable diagram owns its primitives.
/// O(total rows + N log(V+1)) construction work for N cells and V nonzero values;
/// O(N+V) returned/temporary storage. No mutation or shared mutable state.
/// @param panels One or two owned rectangular signed-value descriptions.
/// @param show_values Print integer values when true (default).
/// @param max_cells Total-cell cap, default10000; zero disables the optional cap.
/// @return Owned native diagram, independent of the input panels.
/// @throws std::invalid_argument Invalid panel count or ragged rows.
/// @throws std::length_error Cell-count overflow or positive cap exceeded.
/// @throws std::bad_alloc Palette or diagram allocation fails.
/// @par Example
/// @code
/// const auto d=dm::graphics::diagram_value_images({{{{-1,0,9}}, "Values"}});
/// std::cout << d.to_svg().empty(); // 0
/// @endcode
/// @see Diagram, ValueImagePanel, diagram_label_images()
Diagram diagram_value_images(const std::vector<ValueImagePanel>& panels,
    bool show_values=true,std::size_t max_cells=10000);
} // namespace datamunge::graphics
