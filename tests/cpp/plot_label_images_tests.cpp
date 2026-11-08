#include <gtest/gtest.h>

#include <datamunge/plot/label_images.hpp>
#include <limits>
#include <stdexcept>
using namespace datamunge::graphics;
TEST(LabelImages, ValidatesAllInputsBeforeRenderingCap) {
  EXPECT_THROW(diagram_label_images({}), std::invalid_argument);
  EXPECT_THROW(diagram_label_images(std::vector<LabelImagePanel>(3)), std::invalid_argument);
  EXPECT_THROW(diagram_label_images({{{{1}, {}}, "ragged"}}, true, 1), std::invalid_argument);
  EXPECT_THROW(
      diagram_label_images({{{{1, 1}}, "first"}, {{{-1}}, "second"}}, true, 1),
      std::invalid_argument);
  EXPECT_THROW(
      diagram_label_images({{{{1, 1}}, "first"}, {{{2}}, "second"}}, true, 2), std::length_error);
  EXPECT_FALSE(
      diagram_label_images({{{{1, 1}}, "first"}, {{{2}}, "second"}}, true, 3).to_svg().empty());
}
TEST(LabelImages, EmptyProvisionalAndFullIntLabelsRenderAsOwnedDiagrams) {
  EXPECT_FALSE(diagram_label_images({{{}, "Empty"}}).to_svg().empty());
  std::vector<LabelImagePanel> panels{{{{1, 0, 7}, {0, 3, 0}}, "Provisional"}};
  const auto                   d      = diagram_label_images(panels);
  const auto                   before = d.to_svg();
  panels[0].labels[0][0]              = 99;
  EXPECT_EQ(d.to_svg(), before);
  EXPECT_FALSE(
      diagram_label_images({{{{std::numeric_limits<int>::max()}}, "Large ID"}}).to_svg().empty());
  EXPECT_FALSE(diagram_label_images(panels, false).to_svg().empty());
}

TEST(ValueImages, ValidatesShapesAndCompleteCellCap) {
  EXPECT_THROW(diagram_value_images({}), std::invalid_argument);
  EXPECT_THROW(diagram_value_images(std::vector<ValueImagePanel>(3)), std::invalid_argument);
  EXPECT_THROW(
      diagram_value_images({{{{1, 1}}, "First"}, {{{1}, {}}, "Ragged"}}, true, 1),
      std::invalid_argument);
  EXPECT_THROW(diagram_value_images({{{{-1, 0, 9}}, "Signed"}}, true, 2), std::length_error);
  EXPECT_NO_THROW(diagram_value_images({{{{-1, 0, 9}}, "Signed"}}, true, 3));
  EXPECT_NO_THROW(diagram_value_images({{{{-1, 0, 9}}, "Signed"}}, false, 0));
  EXPECT_FALSE(diagram_value_images({{{}, "Empty"}}).to_svg().empty());
}
TEST(ValueImages, SharedSignedCategoriesHaveConsistentOwnedColors) {
  std::vector<ValueImagePanel> panels{{{{1, 9, 0}}, "Before"}, {{{9, 1, 0}}, "After"}};
  const auto                   d   = diagram_value_images(panels);
  const auto                   svg = d.to_svg();
  panels[0].values[0][0]           = -77;
  EXPECT_EQ(d.to_svg(), svg);
  // The sorted shared union assigns distinct first/second palette entries.
  EXPECT_NE(svg.find("rgb(254,202,202)"), std::string::npos);
  EXPECT_NE(svg.find("rgb(191,219,254)"), std::string::npos);
  EXPECT_FALSE(
      diagram_value_images(
          {{{{std::numeric_limits<int>::min(), 0, std::numeric_limits<int>::max()}}, "Full int"}})
          .to_svg()
          .empty());
  // Legacy IDs 1 and 9 retain the original cyclic color alias.
  const auto old = diagram_label_images({{{{1, 9}}, "Legacy"}}).to_svg();
  EXPECT_EQ(old.find("rgb(191,219,254)"), std::string::npos);
}
