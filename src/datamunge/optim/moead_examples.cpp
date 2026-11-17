/// @file moead_examples.cpp
/// @brief Canonical ZDT1 objective, analytic reference and original decomposition illustration.
#include <algorithm>
#include <cmath>
#include <datamunge/optim/moead.hpp>
#include <limits>
#include <stdexcept>
namespace datamunge::optim {
ZDT1Objective::ZDT1Objective(std::size_t n) : variables_(n) {
  if (n < 2)
    throw std::invalid_argument("ZDT1Objective: at least two variables required");
}
std::vector<double> ZDT1Objective::evaluate(const std::vector<double>& x) {
  if (x.size() != variables_)
    throw std::invalid_argument("ZDT1Objective: invalid dimension");
  for (auto v : x)
    if (!std::isfinite(v) || v < 0 || v > 1)
      throw std::invalid_argument("ZDT1Objective: finite unit bounds required");
  double sum = 0;
  for (std::size_t i = 1; i < x.size(); ++i)
    sum += x[i];
  const auto g = 1 + 9 * sum / double(x.size() - 1);
  return {x[0], g * (1 - std::sqrt(x[0] / g))};
}
ZDT1FrontSummary zdt1_front_summary(const std::vector<ParetoPoint>& front) {
  if (front.empty())
    throw std::invalid_argument("zdt1_front_summary: nonempty front required");
  ZDT1FrontSummary s;
  s.points          = front.size();
  s.min_f1          = std::numeric_limits<double>::infinity();
  s.max_f1          = -s.min_f1;
  s.max_gap         = -s.min_f1;
  long double total = 0;
  for (auto& p : front) {
    auto& f = p.objectives;
    if (f.size() != 2 || !std::isfinite(f[0]) || !std::isfinite(f[1]) || f[0] < 0 || f[0] > 1)
      throw std::invalid_argument("zdt1_front_summary: invalid objectives");
    s.min_f1 = std::min(s.min_f1, f[0]);
    s.max_f1 = std::max(s.max_f1, f[0]);
    auto gap = f[1] - (1 - std::sqrt(f[0]));
    total += gap;
    s.max_gap = std::max(s.max_gap, gap);
  }
  s.mean_gap = double(total / front.size());
  return s;
}
std::vector<ParetoPoint> zdt1_front(std::size_t n) {
  if (n < 2)
    throw std::invalid_argument("zdt1_front: at least two points required");
  if (n > 1000000)
    throw std::length_error("zdt1_front: point cap exceeded");
  std::vector<ParetoPoint> out;
  out.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    double x = double(i) / (n - 1);
    out.push_back({{}, {x, 1 - std::sqrt(x)}});
  }
  return out;
}
std::vector<MOEADDiscreteChoice> moead_original_choices() {
  const std::vector<std::vector<double>> candidates{
      {0, 1}, {.25, .5}, {.5, .293}, {.75, .134}, {1, 0}},
      weights{{1, 0}, {.7, .3}, {.5, .5}, {.3, .7}, {0, 1}};
  std::vector<MOEADDiscreteChoice> out;
  for (auto& w : weights) {
    /// @brief Score a supplied original candidate with the illustrated fixed ideal.
    auto score = [&](const std::vector<double>& f) {
      double value = 0;
      for (std::size_t k = 0; k < f.size(); ++k)
        value = std::max(value, std::max(w[k], 1e-6) * std::abs(f[k]));
      return value;
    };
    std::size_t best = 0;
    for (std::size_t i = 1; i < candidates.size(); ++i)
      if (score(candidates[i]) < score(candidates[best]))
        best = i;
    out.push_back({w, candidates[best], score(candidates[best])});
  }
  return out;
}
} // namespace datamunge::optim
