/// @file structured_svm_analysis.cpp
/// @brief Original local Gaussian cohort and actual two-feature region predictions.
#include <cmath>
#include <datamunge/algorithms/structured_svm_analysis.hpp>
#include <random>
#include <stdexcept>
namespace datamunge::algorithms {
StructuredSVMCohort structured_svm_example_cohort(std::size_t per_class, std::uint64_t seed) {
  if (per_class > 100000)
    throw std::length_error("SVM cohort count");
  StructuredSVMCohort              r;
  std::mt19937_64                  rng(seed);
  std::normal_distribution<double> jitter(0, .5);
  const double                     centers[3][2] = {{0, 0}, {6, 0}, {3, 6}};
  for (int c = 0; c < 3; ++c)
    for (std::size_t i = 0; i < per_class; ++i) {
      r.features.push_back({centers[c][0] + jitter(rng), centers[c][1] + jitter(rng)});
      r.labels.push_back(c);
    }
  return r;
}
StructuredSVMRegions structured_svm_regions(
    const StructuredSVMModel& model,
    double                    low,
    double                    high,
    std::size_t               points) {
  if (model.dim != 2 || !std::isfinite(low) || !std::isfinite(high) || !(low < high) ||
      points < 2 || points > 1000)
    throw std::invalid_argument("SVM two-dimensional grid");
  structured_svm_predict(model, {low, low});
  StructuredSVMRegions r;
  r.x.resize(model.num_classes);
  r.y.resize(model.num_classes);
  for (std::size_t i = 0; i < points; ++i)
    for (std::size_t j = 0; j < points; ++j) {
      double x = std::lerp(low, high, double(i) / double(points - 1)),
             y = std::lerp(low, high, double(j) / double(points - 1));
      auto c   = structured_svm_predict(model, {x, y});
      r.x[c].push_back(x);
      r.y[c].push_back(y);
    }
  return r;
}
} // namespace datamunge::algorithms
