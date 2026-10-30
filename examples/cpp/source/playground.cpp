// CMD: just TARGET=playground run-cpp
// REPL: just cpp-repl

#include <dm/dm.hpp>
#include <print>

/// @brief Shows dataset plots, grouped summaries, and ToothGrowth moment statistics.
int main(void) {

  std::print("Plot of the Iris dataset\n");
  
  // Import and print data
  const auto iris_data = dm::datasets::iris();
  std::print("{}\n", iris_data.head(10));
  
  // Set up palletes
  const auto species = dm::factor(dm::c(iris_data.get_column<std::string>("Species").values()));
  const auto palette = dm::c(dm::RGB{255, 255, 80}, dm::RGB{100, 255, 255}, dm::RGB{220, 100, 255});
  
  // Plot Iris data
  dm::Plot plt = dm::Plot()
    .set_title("Iris Dataset")
    .set_subtitle("Sepal Width vs Sepal Length")
    .set_xlab("Sepal Width")
    .set_ylab("Sepal Length") |
  dm::points(iris_data.get_column("Sepal.Width").values(), iris_data.get_column("Sepal.Length").values())
    .set_col(palette.select(species)) |
  dm::legend()
    .set_labels(dm::c("setosa", "versicolor", "virginica"))
    .set_col(palette);
  std::print("Showing figure...\n");
  plt.show();

  // Show histogram of sepal length
  dm::Histogram hist = dm::Histogram(iris_data.get_column("Sepal.Length").values())
    .set_title("Iris Sepal Length Distribution")
    .set_xlab("Sepal Length");
  hist.show();
    
  // Simulate doppler data
  auto n = 1000;
  auto x = dm::runif(0, 1, n);
  std::print("{}\n", x);
  auto ytrue = dm::apply(x, dm::function([](auto x) {return dm::sin(20 / (x + 0.25));}));
  auto y = ytrue + dm::rnorm(0, 0.2, n);
  std::print("{}\n", y);
  
  // Plot fig
  std::print("Showing figure...\n");
  auto plt2 = dm::Plot(x, y).set_col("blue");
  plt2.show();
  std::print("Done.\n");
  
  // Scatterplot matrix, points colored by species with the same palette as above:
  auto plt3 = dm::Splom(iris_data)
    .set_title("Iris Pairs")
    .set_col(species, palette);
  plt3.show();
  
  const auto tooth_data = dm::datasets::tooth_growth();
  std::print("{}\n", tooth_data);
  
  
  const auto supp = dm::factor(dm::c(tooth_data.get_column<std::string>("supp").values()));
  const auto supp_palette = dm::c(dm::RGB{100, 255, 255}, dm::RGB{220, 100, 255});
  
  // Scatterplot matrix, points colored by species with the same palette as above:
  auto plt4 = dm::Splom(tooth_data)
    .set_title("Iris Pairs")
    .set_col(supp, supp_palette);
  plt4.show();

  // dplyr-style pipe: mean of every numeric column per (supp, dose) group
  const auto tooth_means = tooth_data | dm::group_by("supp", "dose") | dm::summarise_each(dm::mean);
  std::print("{}\n", tooth_means);
  std::print("{}\n", dm::info(tooth_means));

  // Filter rows, add a column, summarise per group, then sort
  const auto tooth_summary = tooth_data
    | dm::filter_rows([](const auto& row) { return row.get_double("dose") >= 1; })
    | dm::mutate("len_per_dose", [](const auto& row) { return row.get_double("len") / row.get_double("dose"); })
    | dm::group_by("supp", "dose")
    | dm::summarise_each(dm::mean)
    | dm::arrange("supp", dm::desc("dose"));
  std::print("{}\n", tooth_summary);

  // Skewness and excess kurtosis using sample standard deviation (n - 1).
  const auto other_stats = dm::function([](const auto& x) {
    const auto m = dm::mean(x);
    const auto n = dm::length(x);
    const auto s = dm::sd(x);
    const auto skew = dm::round(dm::sum(dm::pow(x - m, 3) / dm::pow(s, 3)) / n, 3);
    const auto kurtosis = dm::round(dm::sum(dm::pow(x - m, 4) / dm::pow(s, 4)) / n - 3, 3);
    return dm::c(skew, kurtosis);
  });
  // Reduce the whole column once; element-wise application would receive single values.
  const auto tooth_stats = other_stats(dm::c(tooth_data.get_column("len").values()));
  std::print("ToothGrowth [skewness, excess kurtosis]: {}\n", tooth_stats);
  
  // const auto lm1 = dm::stats::LM(tooth_data, "len ~ supp * dose");
  const auto lm1 = dm::stats::LM(dm::formula("len ~ supp * dose"), tooth_data);
  // and also (through auto type coercion):
  // const auto lm1 = dm::stats::LM("len ~ supp * dose", tooth_data);
  lm1.print_summary();
  std::print("{}\n", lm1);
  std::print("{}\n", dm::dir(lm1));
  std::print( "{}\n", lm1.t_values());
  std::print( "{}\n", lm1.rank());

  return 0;
}
