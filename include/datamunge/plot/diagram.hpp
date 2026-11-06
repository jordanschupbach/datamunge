#pragma once

/// @file diagram.hpp
/// @brief Native vector diagrams: labelled nodes, boundary-anchored arrows and geometric paths.

#include <datamunge/plot/plot.hpp>
#include <string>
#include <vector>

namespace datamunge::graphics {

/// @brief A diagram coordinate; x grows rightwards and y upwards, in diagram units.
/// @par Example
/// @code
/// const dm::graphics::DiagramPoint origin{0, 0};
/// std::print("{} {}\n", origin.x, origin.y);  // 0 0
/// @endcode
struct DiagramPoint {
  double x{0}; ///< Horizontal coordinate, in diagram units.
  double y{0}; ///< Vertical coordinate, in diagram units.
};

/// @brief Appearance of a native diagram path, polygon or ellipse.
/// Values are owned; widths are pixels and RGB opacity is respected. All numeric values
/// must be finite. Width must be positive; RGB channels and opacity are clamped on export.
/// @par Example
/// @code
/// dm::graphics::DiagramStroke arrow;
/// arrow.arrow = true;
/// arrow.color = {37, 99, 235};
/// dm::graphics::Diagram().path({{0, 0}, {2, 1}}, arrow).save_svg("arrow.svg");
/// @endcode
/// @see Diagram::path(), Diagram::edge()
struct DiagramStroke {
  RGB    color{51, 65, 85}; ///< Outline or line colour, including opacity.
  double width{2};          ///< Positive stroke width in pixels.
  bool   arrow{
      false}; ///< Add a filled arrowhead at the final path endpoint; ignored for closed shapes.
  bool dashed{false}; ///< Use a dashed outline.
};

/// @brief Node appearance and minimum size; text expands the box when necessary.
/// Explicit newlines are retained; long lines wrap at word boundaries within width.
/// An unbreakable word may expand the box. Padding is applied on all sides. Nodes are
/// rounded rectangles, or circles when circle is true. Numeric values must be finite;
/// dimensions and font size are positive, padding and corner radius nonnegative.
/// @par Example
/// @code
/// dm::graphics::DiagramNodeStyle style;
/// style.fill = {255, 237, 213};
/// dm::graphics::Diagram().node("step", {0, 0}, "Evaluate\nresidual", style).save_svg("step.svg");
/// @endcode
/// @see Diagram::node()
struct DiagramNodeStyle {
  double width{3.8};          ///< Minimum width and preferred wrapping width, in diagram units.
  double height{1.2};         ///< Minimum height in diagram units.
  double font_size{18};       ///< Text size in pixels before optional whole-diagram scaling.
  double padding{12};         ///< Interior padding in pixels.
  double corner_radius{9};    ///< Rectangle corner radius in pixels.
  RGB    fill{239, 246, 255}; ///< Interior colour; set alpha to zero for transparent fill.
  RGB    border{51, 65, 85};  ///< Border colour.
  RGB    ink{15, 23, 42};     ///< Label colour.
  bool   circle{false};       ///< Enclose the complete label in a circle instead of a rectangle.
};

/// @brief Editable native method diagrams, exported as standalone SVG without external renderers.
/// Coordinates have equal scales on both axes and y points upwards. Nodes own their labels,
/// wrap words and size themselves from the plotting library's embedded Computer Modern font
/// metrics. Text supports UTF-8 and explicit newlines (not TeX markup). Visible labels
/// use native glyph outlines so viewers cannot substitute a differently sized font;
/// selectable source text and accessible labels are retained. Unicode digit scripts,
/// superscript plus/minus/T and script ell use native base glyphs. Other characters absent
/// from the embedded font render as question marks; accessible labels retain the original
/// text. Arrows join node boundaries. Paths, polygons, ellipses and free labels share one
/// coordinate system.
/// Export fits all primitives, labels, stroke widths and arrowheads, leaving a pixel margin.
/// A diagram wider than max_width is scaled uniformly, retaining its aspect ratio.
/// Mutation is not thread-safe; const export may be called concurrently on an unmodified object.
/// Rendering and storage are linear in the number of primitives and their text/path lengths.
/// @par Example
/// @code
/// dm::graphics::Diagram diagram;
/// diagram.node("input", {0, 0}, "Observations\ny")
///        .node("fit", {5, 0}, "Least squares\nminimise residual")
///        .edge("input", "fit")
///        .text({2.5, -1.5}, "The residual is perpendicular to the design columns.");
/// diagram.save_svg("method.svg");
/// @endcode
/// @see RPlot, DiagramNodeStyle, DiagramStroke
class Diagram {
 public:
  /// @brief Creates an empty diagram with equal horizontal and vertical unit scales.
  /// @param unit_pixels Pixels per diagram unit (default 64); positive and finite.
  /// @param max_width Maximum exported width in pixels (default 1100); positive and finite.
  /// @throws std::invalid_argument If a scale or width is not positive and finite.
  /// @par Example
  /// @code
  /// dm::graphics::Diagram(48, 900).text({0, 0}, "Native coordinates").save_svg("scaled.svg");
  /// @endcode
  explicit Diagram(double unit_pixels = 64, double max_width = 1100);

  /// @brief Adds a uniquely named, automatically sized label box or circle.
  /// @param id Nonempty identifier, unique within this diagram.
  /// @param centre Node centre in diagram units.
  /// @param label Owned UTF-8 text; newlines separate lines. Empty labels are allowed.
  /// @param style Node sizing and colours (default DiagramNodeStyle).
  /// @return This diagram for chaining.
  /// @throws std::invalid_argument For a duplicate/empty id, nonfinite coordinates or invalid
  /// style.
  /// @par Example
  /// @code
  /// dm::graphics::Diagram().node("input", {0, 0}, "Observed data\ny").save_svg("node.svg");
  /// @endcode
  Diagram&
  node(std::string id, DiagramPoint centre, std::string label, DiagramNodeStyle style = {});

  /// @brief Joins two existing node boundaries with an arrow and optional midpoint label.
  /// The label is offset to the left of the directed edge and backed by a white rectangle.
  /// Circle boundaries use radial intersections; rectangles use their enclosing box.
  /// @param from Source node identifier.
  /// @param to Destination node identifier, different from from and at a different centre.
  /// @param label UTF-8 edge label; empty omits it.
  /// @param stroke Line appearance; an arrow is always added regardless of stroke.arrow.
  /// @return This diagram for chaining.
  /// @throws std::invalid_argument For missing/coincident or overlapping nodes, self edges or
  /// invalid stroke.
  /// @par Example
  /// @code
  /// dm::graphics::Diagram().node("a", {0, 0}, "Input").node("b", {5, 0}, "Output")
  ///     .edge("a", "b", "transform").save_svg("edge.svg");
  /// @endcode
  Diagram& edge(
      const std::string& from,
      const std::string& to,
      std::string        label  = {},
      DiagramStroke      stroke = {});

  /// @brief Adds a centred multiline UTF-8 label without a box.
  /// @param centre Label centre in diagram units.
  /// @param label Owned text; newlines separate lines, empty text is allowed.
  /// @param font_size Positive finite size in pixels (default 18).
  /// @param ink Text colour (default dark slate); alpha must be finite.
  /// @return This diagram for chaining.
  /// @throws std::invalid_argument For nonfinite coordinates or invalid font size/opacity.
  /// @par Example
  /// @code
  /// dm::graphics::Diagram().text({0, 0}, "Residual ≠ coefficient error",
  /// 18).save_svg("label.svg");
  /// @endcode
  Diagram&
  text(DiagramPoint centre, std::string label, double font_size = 18, RGB ink = {15, 23, 42});

  /// @brief Adds a polyline, optionally dashed and ending in an arrowhead.
  /// @param points At least two finite coordinates in diagram units, in drawing order.
  ///               Adjacent repeated points are removed; at least two distinct points must remain.
  /// @param stroke Line appearance (default DiagramStroke).
  /// @return This diagram for chaining.
  /// @throws std::invalid_argument For insufficient points, nonfinite values or invalid stroke.
  /// @par Example
  /// @code
  /// dm::graphics::DiagramStroke arrow;
  /// arrow.arrow = true;
  /// dm::graphics::Diagram().path({{0, 0}, {2, 0}, {2, 1}}, arrow).save_svg("path.svg");
  /// @endcode
  Diagram& path(std::vector<DiagramPoint> points, DiagramStroke stroke = {});

  /// @brief Adds a closed filled polygon; it does not draw an arrowhead.
  /// @param points At least three finite vertices in diagram units.
  /// @param fill Interior colour; alpha must be finite.
  /// @param stroke Outline appearance; arrow is ignored.
  /// @return This diagram for chaining.
  /// @throws std::invalid_argument For insufficient vertices, nonfinite values or invalid stroke.
  /// @par Example
  /// @code
  /// dm::graphics::Diagram().polygon({{0, 0}, {2, 1}, {3, 3}, {1, 2}}, {219, 234, 254})
  ///     .save_svg("parallelogram.svg");
  /// @endcode
  /// @see path(), ellipse()
  Diagram& polygon(std::vector<DiagramPoint> points, RGB fill, DiagramStroke stroke = {});

  /// @brief Adds a filled ellipse or a circle when the radii are equal.
  /// @param centre Centre in diagram units.
  /// @param radius_x Positive finite horizontal radius in diagram units.
  /// @param radius_y Positive finite vertical radius in diagram units.
  /// @param fill Interior colour; alpha must be finite.
  /// @param stroke Outline appearance; arrow is ignored.
  /// @return This diagram for chaining.
  /// @throws std::invalid_argument For nonfinite coordinates or invalid radii/style.
  /// @par Example
  /// @code
  /// dm::graphics::Diagram().ellipse({0, 0}, 1, 1, {219, 234, 254}).save_svg("circle.svg");
  /// @endcode
  /// @see polygon()
  Diagram& ellipse(
      DiagramPoint  centre,
      double        radius_x,
      double        radius_y,
      RGB           fill,
      DiagramStroke stroke = {});

  /// @brief Renders a complete SVG with fitted bounds and embedded plotting font.
  /// The viewBox retains native units; width/height apply any max_width scaling uniformly.
  /// No files, displays, subprocesses or external fonts are required.
  /// @return Owned standalone SVG text with embedded glyph outlines, accessible labels and a title.
  /// @throws std::runtime_error If the diagram has no primitives.
  /// @par Example
  /// @code
  /// const auto svg = dm::graphics::Diagram().text({0, 0}, "Input → output").to_svg();
  /// std::print("{}\n", svg.find("<svg") != std::string::npos);  // true
  /// @endcode
  /// @see save_svg()
  std::string to_svg() const;

  /// @brief Writes to_svg() to a local file; existing files are replaced.
  /// @param path Output filename; its parent directory must already exist.
  /// @throws std::runtime_error For an empty diagram or a file write failure.
  /// @par Example
  /// @code
  /// dm::graphics::Diagram().text({0, 0}, "Local vector output").save_svg("saved.svg");
  /// @endcode
  /// @see to_svg()
  void save_svg(const std::string& path) const;

 private:
  /// @brief One owned drawing operation; all coordinates and extents are in pixels.
  struct Item {
    std::string               kind;     ///< node, text, path, polygon, or ellipse.
    std::string               id;       ///< Node identifier, or empty.
    DiagramPoint              centre;   ///< Centre of a box, ellipse or label in pixel coordinates.
    std::vector<DiagramPoint> points;   ///< Polyline/polygon vertices in pixel coordinates.
    std::vector<std::string>  lines;    ///< Wrapped UTF-8 label lines.
    double                    width{0}; ///< Pixel width of box, ellipse or label.
    double                    height{0}; ///< Pixel height of box, ellipse or label.
    DiagramNodeStyle          style;     ///< Font, padding, colours and circle setting.
    DiagramStroke             stroke;    ///< Path/outline style.
    RGB                       fill{255, 255, 255, 0}; ///< Shape fill; transparent by default.
  };
  double            unit_;      ///< Pixels per diagram coordinate unit.
  double            max_width_; ///< Maximum output width in pixels.
  std::vector<Item> items_;     ///< Drawing operations in insertion order; copied by value.
};

} // namespace datamunge::graphics
