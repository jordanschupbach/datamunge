#include <datamunge/dstruct/dstruct.hpp>
#include <datamunge/stats/stats.hpp>

#include <iostream>
#include <vector>

using datamunge::dstruct::DataFrame;
using datamunge::stats::LM;
using datamunge::stats::PredictionInterval;

int main() {
  const std::vector<double> hp = {110, 110, 93, 110, 175, 105, 245, 62, 95, 123};
  const std::vector<double> wt = {2.62, 2.875, 2.32, 3.215, 3.44, 3.46, 3.57, 3.19, 3.15, 3.44};
  const std::vector<std::string> transmission = {"manual", "manual", "manual", "automatic", "automatic",
                                                  "automatic", "automatic", "automatic", "automatic", "automatic"};
  const std::vector<double> mpg = {21.0, 21.0, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2};

  DataFrame cars;
  cars.add_column("hp", hp);
  cars.add_column("wt", wt);
  cars.add_column("transmission", transmission);
  cars.add_column("mpg", mpg);

  std::cout << "Fitting: mpg ~ hp + wt + transmission\n\n";
  LM model(cars, "mpg ~ hp + wt + transmission");
  model.print_summary(std::cout);

  std::cout << "\nSequential ANOVA:\n";
  for (const auto& row : model.anova())
    std::cout << "  " << row.term << ": df=" << row.degrees_of_freedom << " SS=" << row.sum_sq
               << " F=" << row.f_value << " p=" << row.p_value << "\n";

  DataFrame newcars;
  newcars.add_column("hp", std::vector<double>{150.0, 90.0});
  newcars.add_column("wt", std::vector<double>{3.0, 2.5});
  newcars.add_column("transmission", std::vector<std::string>{"manual", "automatic"});

  const auto prediction = model.predict(newcars, PredictionInterval::Confidence);
  std::cout << "\nPredictions with 95% confidence intervals:\n";
  for (std::size_t i = 0; i < prediction.fit.size(); ++i)
    std::cout << "  fit=" << prediction.fit[i] << "  [" << prediction.lower[i] << ", " << prediction.upper[i]
               << "]\n";

  model.save_diagnostic_plots("lm_ex_diagnostics");
  std::cout << "\nSaved diagnostic plots as lm_ex_diagnostics_*.svg\n";

  return 0;
}
