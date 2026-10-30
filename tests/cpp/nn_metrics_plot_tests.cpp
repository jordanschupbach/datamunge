#include <gtest/gtest.h>

#include <datamunge/nn/nn.hpp>
#include <filesystem>
#include <fstream>
#include <limits>
#include <regex>
namespace n = datamunge::nn;
TEST(NnConfusionPlot, CountsAndOrientationAreExactAndRenderingDoesNotMutate) {
  const auto counts = n::tensor({3, 1, 2, 4}, {2, 2}).to(n::DType::Int64);
  const auto cm     = n::ConfusionMatrix::from_counts(counts);
  EXPECT_DOUBLE_EQ(cm.accuracy(), .7);
  const auto svg = cm.to_svg({.labels = {"cat & dog", "<other>"}});
  EXPECT_NE(
      svg.find("data-true=\"0\" data-predicted=\"1\" data-count=\"1\" data-value=\"1\""),
      std::string::npos);
  EXPECT_NE(svg.find("cat &amp; dog"), std::string::npos);
  EXPECT_NE(svg.find("&lt;other&gt;"), std::string::npos);
  EXPECT_EQ(cm.to_tensor().to_vector<std::int64_t>(), counts.to_vector<std::int64_t>());
  EXPECT_EQ(svg, cm.to_svg({.labels = {"cat & dog", "<other>"}}));
}
TEST(NnConfusionPlot, NormalizationsUseTheirActualSupportsAndZeroDenominatorsAreZero) {
  const auto cm =
      n::ConfusionMatrix::from_counts(n::tensor({3, 1, 2, 4}, {2, 2}).to(n::DType::Int64));
  for (const auto [mode, expected] :
       std::vector<std::pair<n::ConfusionMatrixNormalization, std::string>>{
           {n::ConfusionMatrixNormalization::True, "0.25"},
           {n::ConfusionMatrixNormalization::Predicted, "0.2"},
           {n::ConfusionMatrixNormalization::All, "0.1"}}) {
    const auto svg = cm.to_svg({.normalization = mode});
    EXPECT_NE(
        svg.find(
            "data-true=\"0\" data-predicted=\"1\" data-count=\"1\" data-value=\"" + expected +
            "\""),
        std::string::npos);
  }
  for (auto mode :
       {n::ConfusionMatrixNormalization::True,
        n::ConfusionMatrixNormalization::Predicted,
        n::ConfusionMatrixNormalization::All}) {
    auto svg = n::ConfusionMatrix(2).to_svg({.normalization = mode});
    EXPECT_EQ(svg.find("nan"), std::string::npos);
    EXPECT_EQ(svg.find("inf"), std::string::npos);
    EXPECT_NE(svg.find("0.0%"), std::string::npos);
  }
  EXPECT_NE(n::ConfusionMatrix(1).to_svg().find("data-count=\"0\""), std::string::npos);
}
TEST(NnConfusionPlot, ImportedIntegerCountsAreOwnedAndInvalidInputIsRejected) {
  auto counts = n::tensor({1, 2, 3, 4}, {2, 2}).to(n::DType::Int32);
  auto cm     = n::ConfusionMatrix::from_counts(n::transpose(counts, {1, 0}));
  EXPECT_EQ(cm.cells()[0][1], 3u);
  EXPECT_THROW(n::ConfusionMatrix::from_counts(n::Tensor{}), std::invalid_argument);
  EXPECT_THROW(n::ConfusionMatrix::from_counts(n::ones({2, 2})), std::invalid_argument);
  EXPECT_THROW(
      n::ConfusionMatrix::from_counts(n::zeros({2, 3}, n::DType::Int64)), std::invalid_argument);
  EXPECT_THROW(
      n::ConfusionMatrix::from_counts(n::tensor({0, -1, 2, 3}, {2, 2}).to(n::DType::Int64)),
      std::invalid_argument);
  EXPECT_EQ(cm.cells()[1][0], 2u);
}
TEST(NnConfusionPlot, OptionsAreValidatedAndSvgFilesRoundTrip) {
  n::ConfusionMatrix cm(2);
  EXPECT_THROW(cm.to_svg({.labels = {"only one"}}), std::invalid_argument);
  EXPECT_THROW(
      cm.to_svg({.normalization = static_cast<n::ConfusionMatrixNormalization>(999)}),
      std::invalid_argument);
  EXPECT_THROW(
      cm.to_svg({.cell_size = std::numeric_limits<double>::quiet_NaN()}), std::invalid_argument);
  EXPECT_THROW(cm.to_svg({.font_size = 0}), std::invalid_argument);
  EXPECT_THROW(cm.to_svg({.high_color = "red\"<script>"}), std::invalid_argument);
  const auto p = std::filesystem::temp_directory_path() / "nn-confusion-svg-test.svg";
  cm.save_svg(p.string());
  std::ifstream file(p);
  std::string   svg((std::istreambuf_iterator<char>(file)), {});
  file.close();
  EXPECT_EQ(svg, cm.to_svg());
  std::filesystem::remove(p);
  EXPECT_THROW(cm.save_svg((p / "missing.svg").string()), std::runtime_error);
}
