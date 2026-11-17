/// @file stochastic_universal_sampling.cpp
/// @brief One checked actual widened-pointer selection producer.
#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/selection_rng.hpp>
#include <datamunge/algorithms/stochastic_universal_sampling.hpp>
#include <limits>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Translation-unit-private checked producing kernels; no shared mutable state.
namespace {
/// @brief Produce all pointers from exactly one actual local draw.
/// @param fitness Complete weights.
/// @param requested Pick count.
/// @param seed Local seed.
/// @param o Policies.
/// @param capture Retain pointers.
/// @return Owned actual outcomes.
SUSRun run(
    const std::vector<double>& fitness,
    int                        requested,
    std::uint64_t              seed,
    const SUSOptions&          o,
    bool                       capture) {
  if (requested < 0)
    throw std::invalid_argument("SUS: negative picks");
  const auto n = static_cast<std::size_t>(requested), m = fitness.size();
  if ((o.max_population && m > o.max_population) || (o.max_picks && n > o.max_picks) ||
      (capture && o.max_records && n > o.max_records))
    throw std::length_error("SUS: complete geometry");
  double maximum = 0;
  for (double w : fitness) {
    if (!std::isfinite(w) || w < 0)
      throw std::invalid_argument("SUS: finite nonnegative weights");
    maximum = std::max(maximum, w);
  }
  if (n && maximum == 0)
    throw std::invalid_argument("SUS: positive total");
  SUSRun r;
  r.counts.assign(m, 0);
  r.expected.assign(m, 0);
  r.cumulative.assign(m, 0);
  if (!n)
    return r;
  std::vector<long double> scaled(m);
  long double              total = 0;
  for (std::size_t i = 0; i < m; ++i) {
    scaled[i] = static_cast<long double>(fitness[i]) / maximum;
    if (fitness[i] > 0 && scaled[i] == 0)
      throw std::overflow_error("SUS: lost positive weight");
    const auto next = total + scaled[i];
    if (!std::isfinite(next) || (scaled[i] > 0 && next == total))
      throw std::overflow_error("SUS: cumulative precision");
    total           = next;
    r.cumulative[i] = total;
  }
  r.spacing = total / n;
  if (!std::isfinite(r.spacing) || r.spacing <= 0)
    throw std::overflow_error("SUS: spacing");
  for (std::size_t i = 0; i < m; ++i)
    r.expected[i] = n * scaled[i] / total;
  detail::SelRng rng(seed);
  r.unit_offset  = rng.uniform();
  r.random_words = 1;
  r.indices.reserve(n);
  std::size_t index = 0;
  for (std::size_t k = 0; k < n; ++k) {
    long double pointer = (static_cast<long double>(r.unit_offset) + k) * r.spacing;
    if (pointer >= total)
      pointer = std::nextafter(total, 0.L);
    while (index + 1 < m && r.cumulative[index] <= pointer) {
      ++index;
      ++r.boundary_advances;
    }
    if (fitness[index] <= 0)
      throw std::overflow_error("SUS: zero interval selected");
    r.indices.push_back(index);
    ++r.counts[index];
    if (capture)
      r.pointers.push_back({k, pointer, index});
  }
  return r;
}
} // namespace
SUSRun sus_run(const std::vector<double>& w, int n, std::uint64_t seed, const SUSOptions& o) {
  return run(w, n, seed, o, false);
}
SUSRun sus_trace(const std::vector<double>& w, int n, std::uint64_t seed, const SUSOptions& o) {
  return run(w, n, seed, o, true);
}
std::vector<std::size_t>
stochastic_universal_sampling(const std::vector<double>& w, int n, std::uint64_t seed) {
  return sus_run(w, n, seed).indices;
}
} // namespace datamunge::algorithms
