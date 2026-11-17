/// @file wang_landau.cpp
/// @brief Bounded original Wang-Landau toy-spin kernel with actual proposal and flatness records.
#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/wang_landau.hpp>
#include <numeric>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Execute the original spin-flip and flatness producer once with explicit resource stopping.
WangLandauWork wang_landau_work(int N, WangLandauOptions o) {
  if (N < 1 || N > 1000 || !std::isfinite(o.flatness) || o.flatness <= 0 || o.flatness > 1 ||
      !std::isfinite(o.log_f_min) || o.log_f_min <= 0 || o.log_f_min >= 1 ||
      o.check_interval > 10000000)
    throw std::invalid_argument("Wang-Landau inputs");
  std::size_t bins = N + 1;
  if (o.max_moves > 100000000 || o.max_move_records > 100000 ||
      o.max_check_records > 1000000 / bins)
    throw std::length_error("Wang-Landau budget");
  std::size_t    interval = o.check_interval ? o.check_interval : bins * 1000;
  WangLandauWork r;
  r.log_density.assign(bins, 0);
  r.histogram.assign(bins, 0);
  std::vector<char> spin(N);
  int               E = 0;
  detail::WlRng     rng(o.seed);
  while (r.log_f_remaining > o.log_f_min && r.moves < o.max_moves) {
    std::fill(r.histogram.begin(), r.histogram.end(), 0);
    bool flat = false;
    while (!flat && r.moves < o.max_moves) {
      std::size_t todo = std::min(interval, o.max_moves - r.moves);
      for (std::size_t step = 0; step < todo; ++step) {
        auto   i      = rng.below(N);
        int    before = E, proposed = E + (spin[i] ? -1 : 1);
        double ratio    = r.log_density[E] - r.log_density[proposed];
        bool   random   = ratio < 0;
        double u        = random ? rng.uniform() : 0;
        bool   accepted = !random || u < std::exp(ratio);
        if (accepted) {
          spin[i] ^= 1;
          E = proposed;
          ++r.accepted;
        }
        if (r.move_records.size() < o.max_move_records)
          r.move_records.push_back(
              {before, proposed, E, i, ratio, u, r.log_f_remaining, random, accepted});
        r.log_density[E] += r.log_f_remaining;
        ++r.histogram[E];
        ++r.moves;
      }
      if (todo < interval)
        break;
      std::size_t minimum = *std::min_element(r.histogram.begin(), r.histogram.end()),
                  total   = std::accumulate(r.histogram.begin(), r.histogram.end(), std::size_t{});
      double mean         = double(total) / bins;
      flat                = minimum > 0 && double(minimum) >= o.flatness * mean;
      ++r.checks;
      if (r.check_records.size() < o.max_check_records)
        r.check_records.push_back({r.moves, r.log_f_remaining, r.histogram, minimum, mean, flat});
    }
    if (flat) {
      r.log_f_remaining *= .5;
      ++r.completed_stages;
    } else
      break;
  }
  r.complete  = r.log_f_remaining <= o.log_f_min;
  double base = r.log_density[0];
  for (auto& x : r.log_density)
    x -= base;
  return r;
}
/// @brief Require the same bounded producer to finish before returning the legacy vector result.
std::vector<double> wang_landau_dos(int N, double flatness, double log_f_min, std::uint64_t seed) {
  WangLandauOptions o;
  o.flatness  = flatness;
  o.log_f_min = log_f_min;
  o.seed      = seed;
  auto r      = wang_landau_work(N, o);
  if (!r.complete)
    throw std::runtime_error("Wang-Landau move limit");
  return r.log_density;
}
} // namespace datamunge::algorithms
