#include <datamunge/plot/plot.hpp>

#include <filesystem>
#include <iostream>
#include <vector>

using namespace datamunge::plot;

int main() {
  const auto out_dir = std::filesystem::path("build/debug/examples/plot_output");
  std::filesystem::create_directories(out_dir);

  auto scatter = ScatterPlot::create();
  scatter.points({0.1, 0.6, 1.0, 1.4, 1.9, 2.4},
                 {1.2, 1.7, 1.1, 2.4, 2.8, 2.2},
                 "samples",
                 {37, 99, 235},
                 6.0)
      .title("Random-ish Scatter")
      .x_label("feature x")
      .y_label("feature y")
      .size(960, 640)
      .background({250, 250, 252});

  auto line = LinePlot::create();
  line.line({0, 1, 2, 3, 4, 5},
            {0.2, 0.9, 1.4, 1.1, 1.9, 2.4},
            "trend",
            {220, 38, 38},
            3.0)
      .title("Line Plot")
      .x_label("step")
      .y_label("value");

  auto bars = BarChart::create();
  bars.bars({1, 2, 3, 4, 5},
            {4, 7, 3, 9, 6},
            "counts",
            {22, 163, 74},
            0.7)
      .title("Bar Chart")
      .x_label("bucket")
      .y_label("count");

  scatter.save((out_dir / "scatter.svg").string());
  scatter.save((out_dir / "scatter.pdf").string());
  scatter.save((out_dir / "scatter.png").string());
  line.save((out_dir / "line.svg").string());
  line.save((out_dir / "line.png").string());
  bars.save((out_dir / "bars.svg").string());
  bars.save((out_dir / "bars.pdf").string());

  scatter.show("datamunge-scatter");
  std::cout << "Wrote plots to " << out_dir << "\n";
  std::cout << "Closed scatter plot window\n";

  return 0;
}
