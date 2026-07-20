// Demonstrates datamunge::plot::RPlot / RLayout -- the R-base-graphics-style plotting library.
#include <datamunge/datamunge.hpp>
#include <datamunge/plot/plot.hpp>

#include <cmath>
#include <filesystem>
#include <iostream>
#include <vector>

using datamunge::Callback;
using datamunge::plot::RGB;
using datamunge::plot::RLayout;
using datamunge::plot::RPlot;

namespace {
// Mimics R's curve(expr, from, to) -- the callback is evaluated at each sample point.
class SineWave final : public Callback {
 public:
  double call(double x) override { return std::sin(x); }
};
} // namespace

int main() {
  const auto out_dir = std::filesystem::path("build/debug/examples");
  std::filesystem::create_directories(out_dir);

  // plot(x, y, type = "p"/"l"/"b"), then points()/lines()/abline() layered on afterward --
  // mirrors R's incremental "plot() then add to it" style.
  auto scatter = RPlot::plot({1.0, 2.0, 3.0, 4.0, 5.0}, {2.1, 3.9, 6.2, 7.8, 10.1}, "p", "observed");
  scatter.abline(0.0, 2.0, RGB{220, 38, 38}, 1.5); // y = 2x reference line
  scatter.title("plot() + abline()").x_label("x").y_label("y");
  scatter.save_svg((out_dir / "rplot_scatter.svg").string());

  // hist(): equal-width binning over the data range.
  std::vector<double> samples = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6, 7};
  auto                hist = RPlot::hist(samples, 6, "counts", RGB{96, 165, 250});
  hist.save_svg((out_dir / "rplot_hist.svg").string());

  // barplot(): categorical positions with x_tick_labels.
  auto bars = RPlot::barplot({23.0, 41.0, 12.0}, {"A", "B", "C"});
  bars.title("barplot()");
  bars.save_svg((out_dir / "rplot_barplot.svg").string());

  // boxplot(): Tukey five-number summary per group, drawn via the low-level box() verb too.
  auto box = RPlot::boxplot({{2, 4, 4, 4, 5, 5, 7, 9}, {1, 2, 2, 2, 3, 3, 3, 3, 4, 20}}, {"low variance", "has outlier"});
  box.title("boxplot()");
  box.save_svg((out_dir / "rplot_boxplot.svg").string());

  // pie(): wedge areas proportional to value, axes hidden automatically.
  auto pie = RPlot::pie({35.0, 25.0, 20.0, 20.0}, {"Q1", "Q2", "Q3", "Q4"});
  pie.title("pie()");
  pie.save_svg((out_dir / "rplot_pie.svg").string());

  // curve(): samples a Callback (director-enabled in every SWIG binding) over a range.
  SineWave sine;
  auto     curve = RPlot::curve(sine, 0.0, 2.0 * 3.14159265358979323846, 200, "sin(x)", RGB{124, 58, 237});
  curve.title("curve()");
  curve.save_svg((out_dir / "rplot_curve.svg").string());

  // qqnorm() + qqline(): standard-normal Q-Q plot with a fitted reference line through Q1/Q3.
  std::vector<double> residuals = {-2.1, -1.3, -0.8, -0.4, -0.1, 0.2, 0.5, 0.9, 1.4, 2.3};
  auto                qq = RPlot::qqnorm(residuals);
  qq.qqline(residuals);
  qq.save_svg((out_dir / "rplot_qqnorm.svg").string());

  // par(mfrow = c(1, 2))-style multi-panel composition via RLayout.
  auto layout = RLayout::create(1, 2);
  layout.add(scatter).add(hist);
  layout.save_svg((out_dir / "rplot_layout.svg").string());

  std::cout << "Wrote 7 SVGs to " << out_dir.string() << "\n";
  return 0;
}
