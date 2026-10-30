#pragma once

/// @file geometry.hpp
/// @brief std::format / std::print support for datamunge::geometry.
/// Included by <datamunge/format.hpp>; requires C++20 <format>.
///
/// Points print without field names (`Point2D(1, 2)`), result structs as
/// `TypeName(field=value, ...)`, and point lists via a capped preview.
///
/// @par Example
/// @code
/// const dm::geometry::Circle c{{0, 0}, 2.5};
/// std::print("{}\n", c);   // Circle(center=Point2D(0, 0), radius=2.5)
/// @endcode

#include <datamunge/format/text_formatter.hpp>
#include <datamunge/geometry/bentley_ottmann.hpp>
#include <datamunge/geometry/geometry.hpp>

#include <format>
#include <string>

/// @brief Prints a Point2D as "Point2D(x, y)".
template <>
struct std::formatter<datamunge::geometry::Point2D>
    : datamunge::format_detail::text_formatter<datamunge::geometry::Point2D> {
  /// @brief The point's coordinates.
  /// @param p The point to print.
  /// @return "Point2D(x, y)".
  static std::string text(const datamunge::geometry::Point2D& p) { return std::format("Point2D({}, {})", p.x, p.y); }
};

/// @brief Prints a Segment as "Segment(Point2D(..), Point2D(..))".
template <>
struct std::formatter<datamunge::geometry::Segment>
    : datamunge::format_detail::text_formatter<datamunge::geometry::Segment> {
  /// @brief The segment's two endpoints.
  /// @param s The segment to print.
  /// @return "Segment(a, b)".
  static std::string text(const datamunge::geometry::Segment& s) { return std::format("Segment({}, {})", s.a, s.b); }
};

/// @brief Prints a BoundingBox as "BoundingBox(min=Point2D(..), max=Point2D(..))".
template <>
struct std::formatter<datamunge::geometry::BoundingBox>
    : datamunge::format_detail::text_formatter<datamunge::geometry::BoundingBox> {
  /// @brief The box's corners.
  /// @param b The box to print.
  /// @return "BoundingBox(min=..., max=...)".
  static std::string text(const datamunge::geometry::BoundingBox& b) {
    return std::format("BoundingBox(min={}, max={})", b.min, b.max);
  }
};

/// @brief Prints a ClosestPairResult as "ClosestPairResult(a=.., b=.., distance=..)".
template <>
struct std::formatter<datamunge::geometry::ClosestPairResult>
    : datamunge::format_detail::text_formatter<datamunge::geometry::ClosestPairResult> {
  /// @brief The pair and its distance.
  /// @param r The result to print.
  /// @return "ClosestPairResult(a=..., b=..., distance=...)".
  static std::string text(const datamunge::geometry::ClosestPairResult& r) {
    return std::format("ClosestPairResult(a={}, b={}, distance={})", r.a, r.b, r.distance);
  }
};

/// @brief Prints a DiameterResult as "DiameterResult(a=.., b=.., distance=..)".
template <>
struct std::formatter<datamunge::geometry::DiameterResult>
    : datamunge::format_detail::text_formatter<datamunge::geometry::DiameterResult> {
  /// @brief The diameter endpoints and length.
  /// @param r The result to print.
  /// @return "DiameterResult(a=..., b=..., distance=...)".
  static std::string text(const datamunge::geometry::DiameterResult& r) {
    return std::format("DiameterResult(a={}, b={}, distance={})", r.a, r.b, r.distance);
  }
};

/// @brief Prints a Delaunay Triangle (vertex indices) as "Triangle(a, b, c)".
template <>
struct std::formatter<datamunge::geometry::Triangle>
    : datamunge::format_detail::text_formatter<datamunge::geometry::Triangle> {
  /// @brief The three vertex indices.
  /// @param t The triangle to print.
  /// @return "Triangle(a, b, c)".
  static std::string text(const datamunge::geometry::Triangle& t) {
    return std::format("Triangle({}, {}, {})", t.a, t.b, t.c);
  }
};

/// @brief Prints a Circle as "Circle(center=Point2D(..), radius=r)".
template <>
struct std::formatter<datamunge::geometry::Circle>
    : datamunge::format_detail::text_formatter<datamunge::geometry::Circle> {
  /// @brief The centre and radius.
  /// @param c The circle to print.
  /// @return "Circle(center=..., radius=...)".
  static std::string text(const datamunge::geometry::Circle& c) {
    return std::format("Circle(center={}, radius={})", c.center, c.radius);
  }
};

/// @brief Prints a DTWResult as "DTWResult(distance=.., path_length=.., path_a=[..], path_b=[..])".
template <>
struct std::formatter<datamunge::geometry::DTWResult>
    : datamunge::format_detail::text_formatter<datamunge::geometry::DTWResult> {
  /// @brief The warping cost and a preview of the alignment path.
  /// @param r The result to print.
  /// @return "DTWResult(distance=..., path_length=..., path_a=..., path_b=...)".
  static std::string text(const datamunge::geometry::DTWResult& r) {
    using datamunge::format_detail::preview;
    return std::format("DTWResult(distance={}, path_length={}, path_a={}, path_b={})", r.distance, r.path_a.size(),
                       preview(r.path_a), preview(r.path_b));
  }
};

/// @brief Prints a GeoHashTable as "GeoHashTable(quant=.., bins=..)".
template <>
struct std::formatter<datamunge::geometry::GeoHashTable>
    : datamunge::format_detail::text_formatter<datamunge::geometry::GeoHashTable> {
  /// @brief The bin size and number of occupied bins.
  /// @param t The table to print.
  /// @return "GeoHashTable(quant=..., bins=...)".
  static std::string text(const datamunge::geometry::GeoHashTable& t) {
    return std::format("GeoHashTable(quant={}, bins={})", t.quant, t.bins.size());
  }
};

/// @brief Prints a GeoMatch with its vote count and recovered similarity transform.
template <>
struct std::formatter<datamunge::geometry::GeoMatch>
    : datamunge::format_detail::text_formatter<datamunge::geometry::GeoMatch> {
  /// @brief "GeoMatch(not found)" or the matched bases and transform.
  /// @param m The match to print.
  /// @return The one-line description.
  static std::string text(const datamunge::geometry::GeoMatch& m) {
    if (!m.found) return std::format("GeoMatch(found=false, votes={})", m.votes);
    return std::format(
        "GeoMatch(found=true, votes={}, model=({}, {}), scene=({}, {}), scale={}, angle={}, translation={}, inliers={})",
        m.votes, m.model_i, m.model_j, m.scene_a, m.scene_b, m.scale, m.angle, m.translation, m.inliers);
  }
};

/// @brief Prints an IcpResult as "IcpResult(angle=.., translation=.., iterations=.., rmse=..)".
template <>
struct std::formatter<datamunge::geometry::IcpResult>
    : datamunge::format_detail::text_formatter<datamunge::geometry::IcpResult> {
  /// @brief The recovered rigid transform and fit quality.
  /// @param r The result to print.
  /// @return "IcpResult(angle=..., translation=..., iterations=..., rmse=...)".
  static std::string text(const datamunge::geometry::IcpResult& r) {
    return std::format("IcpResult(angle={}, translation={}, iterations={}, rmse={})", r.angle, r.translation,
                       r.iterations, r.rmse);
  }
};

/// @brief Prints a MinimumBoundingRectangle as its size, area, and corners.
template <>
struct std::formatter<datamunge::geometry::MinimumBoundingRectangle>
    : datamunge::format_detail::text_formatter<datamunge::geometry::MinimumBoundingRectangle> {
  /// @brief Width, height, area and the corner list.
  /// @param r The rectangle to print.
  /// @return "MinimumBoundingRectangle(width=..., height=..., area=..., corners=[...])".
  static std::string text(const datamunge::geometry::MinimumBoundingRectangle& r) {
    return std::format("MinimumBoundingRectangle(width={}, height={}, area={}, corners={})", r.width, r.height, r.area,
                       datamunge::format_detail::preview(r.corners));
  }
};

/// @brief Prints a VoronoiDiagram as "VoronoiDiagram(vertices=.., cells=..)".
template <>
struct std::formatter<datamunge::geometry::VoronoiDiagram>
    : datamunge::format_detail::text_formatter<datamunge::geometry::VoronoiDiagram> {
  /// @brief The number of Voronoi vertices and cells.
  /// @param v The diagram to print.
  /// @return "VoronoiDiagram(vertices=..., cells=...)".
  static std::string text(const datamunge::geometry::VoronoiDiagram& v) {
    return std::format("VoronoiDiagram(vertices={}, cells={})", v.vertices.size(), v.cells.size());
  }
};

/// @brief Prints a KDTree2D as "KDTree2D(size=n)".
template <>
struct std::formatter<datamunge::geometry::KDTree2D>
    : datamunge::format_detail::text_formatter<datamunge::geometry::KDTree2D> {
  /// @brief The number of stored points.
  /// @param t The tree to print.
  /// @return "KDTree2D(size=n)".
  static std::string text(const datamunge::geometry::KDTree2D& t) { return std::format("KDTree2D(size={})", t.size()); }
};

/// @brief Prints a BestBinFirst index as "BestBinFirst(size=n, points=[..])" with an elided point preview.
template <>
struct std::formatter<datamunge::geometry::BestBinFirst>
    : datamunge::format_detail::text_formatter<datamunge::geometry::BestBinFirst> {
  /// @brief The point count and the first few points.
  /// @param t The index to print.
  /// @return "BestBinFirst(size=n, points=[...])".
  static std::string text(const datamunge::geometry::BestBinFirst& t) {
    return std::format("BestBinFirst(size={}, points={})", t.size(), datamunge::format_detail::preview(t.points(), 4));
  }
};

/// @brief Prints a BestBinFirst::Result as "BestBinFirst::Result(index=.., examined=..)".
template <>
struct std::formatter<datamunge::geometry::BestBinFirst::Result>
    : datamunge::format_detail::text_formatter<datamunge::geometry::BestBinFirst::Result> {
  /// @brief The found index and inspection count.
  /// @param r The result to print.
  /// @return "BestBinFirst::Result(index=..., examined=...)".
  static std::string text(const datamunge::geometry::BestBinFirst::Result& r) {
    return std::format("BestBinFirst::Result(index={}, examined={})", r.index, r.examined);
  }
};

/// @brief Prints a TriangulationLocator as "TriangulationLocator(points=.., triangles=.., edges=..)".
template <>
struct std::formatter<datamunge::geometry::TriangulationLocator>
    : datamunge::format_detail::text_formatter<datamunge::geometry::TriangulationLocator> {
  /// @brief The sizes of the referenced triangulation and its edge map.
  /// @param l The locator to print.
  /// @return "TriangulationLocator(points=..., triangles=..., edges=...)".
  static std::string text(const datamunge::geometry::TriangulationLocator& l) {
    return std::format("TriangulationLocator(points={}, triangles={}, edges={})", l.points ? l.points->size() : 0,
                       l.tris ? l.tris->size() : 0, l.across.size());
  }
};

/// @brief Prints an Orientation as its enumerator name (Collinear, Clockwise, CounterClockwise).
template <>
struct std::formatter<datamunge::geometry::Orientation>
    : datamunge::format_detail::text_formatter<datamunge::geometry::Orientation> {
  /// @brief The enumerator name, or "Orientation(<int>)" for an unknown value.
  /// @param o The orientation to print.
  /// @return The enumerator name.
  static std::string text(datamunge::geometry::Orientation o) {
    using datamunge::geometry::Orientation;
    switch (o) {
      case Orientation::Collinear: return "Collinear";
      case Orientation::Clockwise: return "Clockwise";
      case Orientation::CounterClockwise: return "CounterClockwise";
    }
    return std::format("Orientation({})", static_cast<int>(o));
  }
};
