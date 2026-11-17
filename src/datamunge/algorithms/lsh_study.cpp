/// @file lsh_study.cpp
/// @brief Native original seeded collision/search workload, preserving generator consumption and exact baselines.
#include <cmath>
#include <datamunge/algorithms/lsh_study.hpp>
#include <numbers>
#include <random>
namespace datamunge::algorithms {
LSHDemoStudy lsh_original_study() {
  LSHDemoStudy                     out;
  std::mt19937_64                  rng(3);
  std::normal_distribution<double> g(0, 1);
  for (int degrees = 0; degrees <= 180; degrees += 20) {
    const double          theta = degrees * std::numbers::pi / 180;
    const double          x = std::cos(theta), y = std::sin(theta);
    std::size_t           agreements = 0;
    constexpr std::size_t trials     = 20000;
    for (std::size_t t = 0; t < trials; ++t) {
      double r0 = g(rng), r1 = g(rng);
      double da = r0 * 1. + r1 * 0., db = r0 * x + r1 * y;
      agreements += (da >= 0) == (db >= 0);
    }
    out.angles.push_back(
        {degrees,
         agreements,
         trials,
         double(agreements) / trials,
         lsh_collision_probability(theta)});
  }
  out.data.assign(2000, std::vector<double>(30));
  for (auto& v : out.data)
    for (double& x : v)
      x = g(rng);
  CosineLSH                                  index(out.data, 15, 14, 99);
  std::uniform_int_distribution<std::size_t> pick(0, 1999);
  std::normal_distribution<double>           noise(0, .3);
  for (int t = 0; t < 200; ++t) {
    const auto source = pick(rng);
    auto       q      = out.data[source];
    for (double& x : q)
      x += noise(rng);
    std::size_t exact = 0;
    double      best  = -2;
    for (std::size_t i = 0; i < out.data.size(); ++i) {
      const double s = CosineLSH::cosine(q, out.data[i]);
      if (s > best) {
        best  = s;
        exact = i;
      }
    }
    const auto actual = index.query_analysis(q);
    out.queries.push_back(
        {std::move(q), source, exact, actual.best, actual.examined, best, actual.similarity});
  }
  return out;
}
} // namespace datamunge::algorithms
