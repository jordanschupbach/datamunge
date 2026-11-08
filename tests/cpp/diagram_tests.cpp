/// @file diagram_tests.cpp
/// @brief Native diagram contracts: geometry, labels, anchoring, ownership and error handling.
#include <gtest/gtest.h>

#include <cmath>
#include <datamunge/plot/diagram.hpp>
#include <filesystem>
#include <fstream>
#include <limits>
#include <regex>
#include <set>

namespace {
using namespace datamunge::graphics;

TEST(Diagram, EscapesLabelsAndOwnsACopy) {
  Diagram     original;
  std::string label = "x < y & z\nλ → μ";
  original.node("a", {0, 0}, label);
  Diagram copy = original;
  label        = "changed";
  original.text({10, 0}, label);
  const auto svg = copy.to_svg();
  EXPECT_NE(svg.find("x &lt; y &amp; z"), std::string::npos);
  EXPECT_NE(svg.find("λ → μ"), std::string::npos);
  EXPECT_EQ(svg.find("changed"), std::string::npos);
  EXPECT_NE(svg.find("<path id=\"dm-g-"), std::string::npos);
  EXPECT_NE(svg.find("aria-label=\"λ → μ\""), std::string::npos);
  EXPECT_EQ(svg.find("@font-face"), std::string::npos);
}

TEST(Diagram, ArrowsEndAtRectangleBoundaries) {
  Diagram d;
  d.node("a", {0, 0}, "a").node("b", {5, 0}, "b").edge("a", "b");
  const auto  svg = d.to_svg();
  std::smatch match;
  ASSERT_TRUE(
      std::regex_search(
          svg, match, std::regex("<polyline points=\"([0-9.]+),([0-9.]+) ([0-9.]+),([0-9.]+)")));
  EXPECT_NEAR(std::stod(match[3]) - std::stod(match[1]), (5 - 3.8) * 64, .002);
  EXPECT_EQ(match[2], match[4]);
  EXPECT_NE(svg.find("<polygon points="), std::string::npos);
}

TEST(Diagram, UndirectedLinksPreserveBoundaryAndLabelWithoutArrowhead) {
  DiagramStroke style;
  style.arrow = true;
  style.color = {220, 38, 38};
  style.width = 3;
  Diagram linked;
  linked.node("a", {0, 0}, "A").node("b", {5, 0}, "B").link("a", "b", "u & v", style);
  const auto svg = linked.to_svg();
  EXPECT_EQ(svg.find("<polygon points="), std::string::npos);
  EXPECT_NE(svg.find("u &amp; v"), std::string::npos);
  std::smatch match;
  ASSERT_TRUE(
      std::regex_search(
          svg, match, std::regex("<polyline points=\"([0-9.]+),([0-9.]+) ([0-9.]+),([0-9.]+)")));
  EXPECT_NEAR(std::stod(match[3]) - std::stod(match[1]), (5 - 3.8) * 64, .002);
  Diagram directed;
  style.arrow = false;
  directed.node("a", {0, 0}, "A").node("b", {5, 0}, "B").edge("a", "b", "", style);
  EXPECT_NE(directed.to_svg().find("<polygon points="), std::string::npos);
}

TEST(Diagram, UndirectedLinksShareCircleClippingAndRejectAmbiguousGeometry) {
  DiagramNodeStyle c;
  c.circle = true;
  c.width = c.height = 1;
  c.padding          = 2;
  Diagram d;
  d.node("a", {0, 0}, "A", c).node("b", {3, 4}, "B", c);
  EXPECT_NO_THROW(d.link("a", "b"));
  EXPECT_EQ(d.to_svg().find("<polygon points="), std::string::npos);
  EXPECT_THROW(d.link("a", "missing"), std::invalid_argument);
  EXPECT_THROW(d.link("a", "a"), std::invalid_argument);
  Diagram overlap;
  overlap.node("a", {0, 0}, "A").node("b", {.1, 0}, "B");
  EXPECT_THROW(overlap.link("a", "b"), std::invalid_argument);
}

TEST(Diagram, MathematicalScriptsHaveNativeOutlinesAndKeepTheirAccessibleText) {
  const auto svg = Diagram().text({0, 0}, "v₁ + x² + H⁻¹ + Vᵀ").to_svg();
  EXPECT_NE(svg.find("aria-label=\"v₁ + x² + H⁻¹ + Vᵀ\""), std::string::npos);
  const std::regex scale_pattern("scale\\(([0-9.]+) -[0-9.]+\\)");
  std::set<double> scales;
  for (std::sregex_iterator it(svg.begin(), svg.end(), scale_pattern), end; it != end; ++it)
    scales.insert(std::stod((*it)[1]));
  ASSERT_GE(scales.size(), 2);
  EXPECT_GT(*scales.begin(), 0);
  EXPECT_LT(*scales.begin(), *scales.rbegin());
  EXPECT_NE(svg.find("<use href=\"#dm-g-"), std::string::npos);
  EXPECT_EQ(svg.find("font-family"), std::string::npos);
}

TEST(Diagram, LogicalConnectivesHaveDistinctNativeOutlines) {
  const auto            logical  = Diagram().text({0, 0}, "∧∨").to_svg();
  const auto            fallback = Diagram().text({0, 0}, "?").to_svg();
  const std::regex      pattern("<path id=\"dm-g-([0-9]+)\"");
  std::set<std::string> ids;
  for (std::sregex_iterator it(logical.begin(), logical.end(), pattern), end; it != end; ++it)
    ids.insert((*it)[1]);
  ASSERT_EQ(ids.size(), 2);
  std::smatch question;
  ASSERT_TRUE(std::regex_search(fallback, question, pattern));
  EXPECT_EQ(ids.count(question[1]), 0);
  EXPECT_NE(logical.find("aria-label=\"∧∨\""), std::string::npos);
}

TEST(Diagram, CircleAnchorsAreRadialAndNodeTextExpands) {
  DiagramNodeStyle small;
  small.circle = true;
  small.width = small.height = .5;
  Diagram d;
  d.node("a", {0, 0}, "A", small).node("b", {3, 4}, "B", small).edge("a", "b");
  const auto svg = d.to_svg();
  EXPECT_NE(svg.find("<ellipse"), std::string::npos);
  Diagram expanded;
  expanded.node("large", {0, 0}, "An_unbreakable_label_that_must_expand_the_node", small);
  std::smatch match;
  const auto  wide = expanded.to_svg();
  ASSERT_TRUE(std::regex_search(wide, match, std::regex("rx=\"([0-9.]+)\" ry=\"([0-9.]+)")));
  EXPECT_EQ(match[1], match[2]);
  EXPECT_GT(std::stod(match[1]), 100);
}

TEST(Diagram, FittedBoundsIncludeFreeLabelsAndRespectMaximumWidth) {
  Diagram d(64, 400);
  d.text({-20, 10}, "Left edge\nsecond line").text({20, -10}, "Right edge");
  const auto  svg = d.to_svg();
  std::smatch match;
  ASSERT_TRUE(
      std::regex_search(
          svg,
          match,
          std::regex(
              "width=\"([0-9.]+)\" height=\"([0-9.]+)\" viewBox=\"0 0 ([0-9.]+) ([0-9.]+)")));
  EXPECT_NEAR(std::stod(match[1]), 400, .001);
  EXPECT_NEAR(
      std::stod(match[1]) / std::stod(match[2]), std::stod(match[3]) / std::stod(match[4]), .001);
  EXPECT_GT(std::stod(match[3]), 40 * 64);
  EXPECT_GT(std::stod(match[4]), 20 * 64);
}

TEST(Diagram, PreservesGeometryStylesAndRemovesRepeatedPathPoints) {
  DiagramStroke style;
  style.arrow  = true;
  style.dashed = true;
  style.color  = {37, 99, 235, .4};
  Diagram d;
  d.path({{0, 0}, {0, 0}, {1, 0}, {1, 1}}, style)
      .polygon({{2, 0}, {3, 0}, {3, 1}}, {255, 0, 0, .2})
      .ellipse({4, 0}, 1, .5, {0, 0, 255, 0});
  const auto svg = d.to_svg();
  EXPECT_NE(svg.find("stroke-dasharray=\"6 4\""), std::string::npos);
  EXPECT_NE(svg.find("stroke-opacity=\"0.400\""), std::string::npos);
  EXPECT_NE(svg.find("fill-opacity=\"0.200\""), std::string::npos);
  EXPECT_NE(svg.find("rx=\"64.000\" ry=\"32.000\""), std::string::npos);
  EXPECT_EQ(svg.find("nan"), std::string::npos);
}

TEST(Diagram, RejectsInvalidInputsAndAmbiguousEdges) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(Diagram(0), std::invalid_argument);
  EXPECT_THROW(Diagram(64, nan), std::invalid_argument);
  EXPECT_THROW(Diagram().to_svg(), std::runtime_error);
  EXPECT_THROW(Diagram().text({nan, 0}, "x"), std::invalid_argument);
  EXPECT_THROW(Diagram().path({{0, 0}, {0, 0}}), std::invalid_argument);
  EXPECT_THROW(Diagram().polygon({{0, 0}, {1, 1}}, {}), std::invalid_argument);
  EXPECT_THROW(Diagram().ellipse({0, 0}, 0, 1, {}), std::invalid_argument);
  Diagram d;
  d.node("a", {0, 0}, "A");
  EXPECT_THROW(d.node("a", {5, 0}, "duplicate"), std::invalid_argument);
  EXPECT_THROW(d.edge("a", "missing"), std::invalid_argument);
  EXPECT_THROW(d.edge("a", "a"), std::invalid_argument);
  d.node("b", {.1, 0}, "overlap");
  EXPECT_THROW(d.edge("a", "b"), std::invalid_argument);
  DiagramNodeStyle bad;
  bad.padding = -1;
  EXPECT_THROW(d.node("bad", {5, 0}, "bad", bad), std::invalid_argument);
}

TEST(Diagram, SavesStandaloneSvgAndReportsWriteFailures) {
  const auto path = std::filesystem::temp_directory_path() / "datamunge-native-diagram-test.svg";
  Diagram    d;
  d.text({0, 0}, "Native vector output");
  d.save_svg(path.string());
  std::ifstream     input(path, std::ios::binary);
  const std::string actual((std::istreambuf_iterator<char>(input)), {});
  EXPECT_EQ(actual, d.to_svg());
  std::filesystem::remove(path);
  EXPECT_THROW(d.save_svg((path / "missing" / "image.svg").string()), std::runtime_error);
}
} // namespace
