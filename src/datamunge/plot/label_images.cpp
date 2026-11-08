/// @file label_images.cpp
/// @brief Validated native one/two-panel discrete label-grid rendering.
#include <algorithm>
#include <array>
#include <datamunge/plot/label_images.hpp>
#include <limits>
#include <map>
#include <stdexcept>
namespace datamunge::graphics {
/// @brief Private borrowed views and common cell renderer.
namespace {
/// @brief Read-only grid/title references used only during rendering.
struct GridView {
  const std::vector<std::vector<int>>& labels; ///< Borrowed grid, never retained.
  const std::string&                   title;  ///< Borrowed heading, never retained.
};
/// @brief Render validated views with legacy ID colors or sorted signed categories.
/// @param panels Borrowed views, at most two.
/// @param count Number of active views, already validated.
/// @param show_labels Whether numeric labels are printed.
/// @param max_cells Complete optional cell cap.
/// @param categorical Use sorted signed categories rather than positive ID indexing.
/// @return Owned diagram with the original cell geometry and primitive order.
/// @throws std::invalid_argument Ragged rows or negative legacy IDs.
/// @throws std::length_error Cell sum overflow or positive cap exceeded.
/// @throws std::bad_alloc Palette or diagram allocation fails.
Diagram render(
    const std::array<GridView, 2>& panels,
    std::size_t                    count,
    bool                           show_labels,
    std::size_t                    max_cells,
    bool                           categorical) {
  std::size_t total = 0, digits = 1;
  for (std::size_t panel_index = 0; panel_index < count; ++panel_index) {
    const auto& panel = panels[panel_index];
    const auto  cols  = panel.labels.empty() ? 0 : panel.labels[0].size();
    for (const auto& row : panel.labels) {
      if (row.size() != cols)
        throw std::invalid_argument("label grids: ragged rows");
      for (int label : row) {
        if (!categorical && label < 0)
          throw std::invalid_argument("label grids: negative label");
        digits = std::max(digits, std::to_string(label).size());
      }
    }
    if (cols && panel.labels.size() > (std::numeric_limits<std::size_t>::max() - total) / cols)
      throw std::length_error("label grids: shape sum overflow");
    total += panel.labels.size() * cols;
  }
  if (max_cells && total > max_cells)
    throw std::length_error("label grids: complete cell cap");
  std::map<int, std::size_t> categories;
  if (categorical) {
    for (std::size_t i = 0; i < count; ++i)
      for (const auto& row : panels[i].labels)
        for (int value : row)
          if (value)
            categories.try_emplace(value, 0);
    std::size_t ordinal = 0;
    for (auto& entry : categories)
      entry.second = ordinal++;
  }
  /// @brief Fixed eight-colour positive-ID cycle; background is handled separately.
  const std::array<RGB, 8> palette{
      {{254, 202, 202},
       {191, 219, 254},
       {187, 247, 208},
       {254, 240, 138},
       {233, 213, 255},
       {165, 243, 252},
       {254, 215, 170},
       {251, 207, 232}}};
  Diagram       d(show_labels ? std::max(52., 14. * digits * .7 + 16) : 52., 1100);
  DiagramStroke stroke;
  stroke.color  = {148, 163, 184};
  stroke.width  = .7;
  double offset = 0;
  for (std::size_t panel_index = 0; panel_index < count; ++panel_index) {
    const auto& panel = panels[panel_index];
    const auto  cols  = panel.labels.empty() ? 0 : panel.labels[0].size();
    d.text({offset + (cols ? double(cols - 1) / 2 : 0), 1.25}, panel.title, 20);
    for (std::size_t r = 0; r < panel.labels.size(); ++r)
      for (std::size_t c = 0; c < cols; ++c) {
        const int    label = panel.labels[r][c];
        const double x = offset + c, y = -double(r);
        const RGB    fill =
            label
                   ? palette
                      [(categorical ? categories.at(label) : static_cast<std::size_t>(label) - 1) %
                       palette.size()]
                   : RGB{255, 255, 255};
        d.polygon(
            {{x - .5, y - .5}, {x + .5, y - .5}, {x + .5, y + .5}, {x - .5, y + .5}}, fill, stroke);
        if (show_labels)
          d.text({x, y}, std::to_string(label), 14);
      }
    offset += std::max(1., double(cols)) + 3;
  }
  return d;
}
} // namespace
Diagram diagram_label_images(
    const std::vector<LabelImagePanel>& panels,
    bool                                show_labels,
    std::size_t                         max_cells) {
  if (panels.empty() || panels.size() > 2)
    throw std::invalid_argument("label grids: require one or two panels");
  const auto& second = panels.size() == 2 ? panels[1] : panels[0];
  return render(
      {GridView{panels[0].labels, panels[0].title}, GridView{second.labels, second.title}},
      panels.size(),
      show_labels,
      max_cells,
      false);
}
Diagram diagram_value_images(
    const std::vector<ValueImagePanel>& panels,
    bool                                show_values,
    std::size_t                         max_cells) {
  if (panels.empty() || panels.size() > 2)
    throw std::invalid_argument("value grids: require one or two panels");
  const auto& second = panels.size() == 2 ? panels[1] : panels[0];
  return render(
      {GridView{panels[0].values, panels[0].title}, GridView{second.values, second.title}},
      panels.size(),
      show_values,
      max_cells,
      true);
}
} // namespace datamunge::graphics
