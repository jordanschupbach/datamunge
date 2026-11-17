/// @file miser.cpp
/// @brief Original Gaussian fixture and native replicated comparison.
#include <datamunge/algorithms/miser.hpp>
#include <numbers>
namespace datamunge::algorithms {
double miser_gaussian(const std::vector<double>& p) {
  if (p.size() != 2 || !std::isfinite(p[0]) || !std::isfinite(p[1]))
    throw std::invalid_argument("miser_gaussian: two finite coordinates required");
  auto dx = p[0] - .5, dy = p[1] - .5;
  return std::exp(-200 * (dx * dx + dy * dy));
}
double miser_gaussian_integral() {
  const auto e = std::erf(std::sqrt(50.));
  return std::numbers::pi / 200 * e * e;
}
std::vector<MiserComparison> miser_gaussian_comparison(const std::vector<long>& budgets, int reps) {
  if (reps <= 0)
    throw std::invalid_argument("miser_gaussian_comparison: positive replicates required");
  if (reps > 10000)
    throw std::length_error("miser_gaussian_comparison: replicate cap exceeded");
  for (auto n : budgets)
    if (n < 2)
      throw std::invalid_argument("miser_gaussian_comparison: at least two calls required");
  std::vector<MiserComparison> out;
  const auto                   truth = miser_gaussian_integral();
  for (auto n : budgets) {
    double me = 0, pe = 0;
    for (int seed = 1; seed <= reps; ++seed) {
      auto mv = miser_integrate(miser_gaussian, {0, 0}, {1, 1}, n, seed).value;
      me += (mv - truth) * (mv - truth);
      std::mt19937_64                        rng(1000 + seed);
      std::uniform_real_distribution<double> u(0, 1);
      double                                 sum = 0;
      for (long k = 0; k < n; ++k) {
        auto x = u(rng), y = u(rng);
        sum += miser_gaussian({x, y});
      }
      auto pv = sum / n;
      pe += (pv - truth) * (pv - truth);
    }
    out.push_back({n, reps, std::sqrt(me / reps), std::sqrt(pe / reps)});
  }
  return out;
}
} // namespace datamunge::algorithms
