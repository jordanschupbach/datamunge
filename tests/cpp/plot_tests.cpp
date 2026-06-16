#include <gtest/gtest.h>

#include <datamunge/plot/plot.hpp>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

using namespace datamunge::plot;

TEST(Plot, SavesSvgWithExpectedElements) {
  const auto out_dir = std::filesystem::temp_directory_path();
  const auto path = out_dir / "datamunge_plot_test.svg";

  auto plot = ScatterPlot::create();
  plot.points({0.0, 1.0, 2.0}, {1.0, 3.0, 2.0}, "pts", {37, 99, 235}, 5.0)
      .line({0.0, 2.0}, {1.0, 2.0}, "fit", {220, 38, 38}, 2.5)
      .title("Test Scatter")
      .x_label("x axis")
      .y_label("y axis");

  plot.save_svg(path.string());

  std::ifstream in(path);
  ASSERT_TRUE(in.good());
  const std::string svg((std::istreambuf_iterator<char>(in)),
                        std::istreambuf_iterator<char>());

  EXPECT_NE(svg.find("<svg"), std::string::npos);
  EXPECT_NE(svg.find("Test Scatter"), std::string::npos);
  EXPECT_NE(svg.find("<circle"), std::string::npos);
  EXPECT_NE(svg.find("<polyline"), std::string::npos);
  EXPECT_NE(svg.find("pts"), std::string::npos);
  EXPECT_NE(svg.find("fit"), std::string::npos);

  std::filesystem::remove(path);
}

TEST(Plot, RejectsUnsupportedExtension) {
  auto plot = LinePlot::create();
  plot.line({0.0, 1.0}, {0.0, 1.0});
  EXPECT_THROW(plot.save("bad-format.xyz"), std::invalid_argument);
}

TEST(Plot, SavesPngWithExpectedSignature) {
  const auto out_dir = std::filesystem::temp_directory_path();
  const auto path = out_dir / "datamunge_plot_test.png";

  auto plot = LinePlot::create();
  plot.line({0.0, 1.0, 2.0}, {1.0, 3.0, 2.0}, "line", {220, 38, 38}, 3.0)
      .title("PNG Test");
  plot.save(path.string());

  std::ifstream in(path, std::ios::binary);
  ASSERT_TRUE(in.good());
  char signature[8] = {};
  in.read(signature, sizeof(signature));
  ASSERT_EQ(in.gcount(), static_cast<std::streamsize>(sizeof(signature)));
  EXPECT_EQ(static_cast<unsigned char>(signature[0]), 0x89U);
  EXPECT_EQ(signature[1], 'P');
  EXPECT_EQ(signature[2], 'N');
  EXPECT_EQ(signature[3], 'G');

  std::filesystem::remove(path);
}

TEST(Plot, SavesPdfWithExpectedHeader) {
  const auto out_dir = std::filesystem::temp_directory_path();
  const auto path = out_dir / "datamunge_plot_test.pdf";

  auto plot = ScatterPlot::create();
  plot.points({0.0, 1.0}, {1.0, 0.0}, "pts", {37, 99, 235}, 5.0)
      .title("PDF Test");
  plot.save(path.string());

  std::ifstream in(path, std::ios::binary);
  ASSERT_TRUE(in.good());
  std::string header(8, '\0');
  in.read(header.data(), static_cast<std::streamsize>(header.size()));
  ASSERT_EQ(in.gcount(), static_cast<std::streamsize>(header.size()));
  EXPECT_EQ(header.substr(0, 5), "%PDF-");

  std::filesystem::remove(path);
}

TEST(Plot, BarChartSaveUsesBars) {
  const auto out_dir = std::filesystem::temp_directory_path();
  const auto path = out_dir / "datamunge_plot_bar_test.svg";

  auto plot = BarChart::create();
  plot.bars({1.0, 2.0, 3.0}, {2.0, 4.0, 3.0}, "bars", {22, 163, 74}, 0.6);
  plot.save_svg(path.string());

  std::ifstream in(path);
  ASSERT_TRUE(in.good());
  const std::string svg((std::istreambuf_iterator<char>(in)),
                        std::istreambuf_iterator<char>());
  EXPECT_NE(svg.find("<rect"), std::string::npos);

  std::filesystem::remove(path);
}
