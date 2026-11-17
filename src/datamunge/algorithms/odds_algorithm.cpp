/// @file odds_algorithm.cpp
/// @brief Canonical checked odds-threshold producer and separate fixed-threshold probability study.
#include <cmath>
#include <datamunge/algorithms/odds_algorithm.hpp>
#include <limits>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Private complete validation and one shared actual stopping loop.
namespace {
/// @brief Validate every caller probability before any adaptive suffix work.
/// @param p Complete original probability vector.
/// @throws std::invalid_argument If any probability is nonfinite or outside[0,1].
void validate(const std::vector<double>& p) {
  for (double x : p)
    if (!std::isfinite(x) || x < 0 || x > 1)
      throw std::invalid_argument("odds: finite probabilities in[0,1] required");
}
/// @brief Execute actual backwards odds decisions with optional owned observations.
/// @param p Original probability vector, completely validated here.
/// @param out Optional owned trace sink.
/// @param cap Actual logical input/result/record quota,0 disables.
/// @return Actual same-producer result, selected suffix state and work.
/// @throws std::invalid_argument If complete probabilities fail.
/// @throws std::length_error If complete capture geometry or online quota fails.
OddsRun run(const std::vector<double>& p, OddsTrace* out, std::size_t cap) {
  validate(p);
  OddsRun r;
  if (out) {
    if (p.size() > std::numeric_limits<std::size_t>::max() - 2)
      throw std::length_error("odds: input geometry");
    auto base = p.size() + 2;
    if (cap && base > cap)
      throw std::length_error("odds: capture base");
    out->captured_entries = base;
    out->probabilities    = p;
  }
  for (std::size_t i = p.size(); i-- > 0;) {
    if (out) {
      if (out->captured_entries == std::numeric_limits<std::size_t>::max() ||
          (cap && out->captured_entries == cap))
        throw std::length_error("odds: capture quota");
      ++out->captured_entries;
    }
    OddsStep s;
    s.event       = i + 1;
    s.probability = p[i];
    s.failure     = 1 - static_cast<long double>(p[i]);
    s.certain     = p[i] == 1;
    s.odds        = s.certain ? 0 : static_cast<long double>(p[i]) / s.failure;
    s.sum_before  = r.odds_sum;
    s.none_before = r.no_success;
    s.one_before  = r.exactly_one;
    r.odds_sum += s.odds;
    r.exactly_one = static_cast<long double>(p[i]) * r.no_success + s.failure * r.exactly_one;
    r.no_success *= s.failure;
    r.certain = s.certain;
    ++r.visits;
    r.result.threshold = i + 1;
    s.sum_after        = r.odds_sum;
    s.none_after       = r.no_success;
    s.one_after        = r.exactly_one;
    s.reached          = s.certain || r.odds_sum >= 1;
    if (out)
      out->steps.push_back(s);
    if (s.reached)
      break;
  }
  r.result.win_probability = double(r.exactly_one);
  return r;
}
} // namespace
OddsResult odds_algorithm(const std::vector<double>& p) {
  return run(p, nullptr, 0).result;
}
OddsRun odds_algorithm_run(const std::vector<double>& p) {
  return run(p, nullptr, 0);
}
OddsTrace odds_algorithm_trace(const std::vector<double>& p, std::size_t max_entries) {
  OddsTrace t;
  t.run = run(p, &t, max_entries);
  return t;
}
std::vector<OddsThresholdPoint> odds_threshold_curve(
    const std::vector<double>& p,
    std::size_t                max_points) {
  validate(p);
  if (p.size() == std::numeric_limits<std::size_t>::max() ||
      p.size() + 1 > std::vector<OddsThresholdPoint>().max_size() ||
      (max_points && p.size() + 1 > max_points))
    throw std::length_error("odds: complete curve geometry");
  std::vector<OddsThresholdPoint> rows(p.size() + 1);
  rows.back()      = {p.size() + 1, 0, 1};
  long double none = 1, one = 0;
  for (std::size_t i = p.size(); i-- > 0;) {
    auto q = 1 - static_cast<long double>(p[i]);
    one    = static_cast<long double>(p[i]) * none + q * one;
    none *= q;
    rows[i] = {i + 1, double(one), double(none)};
  }
  return rows;
}
std::vector<double> odds_algorithm_demo_probabilities() {
  return {.10, .15, .05, .30, .20, .40, .25, .35, .12, .28};
}
std::vector<double> odds_secretary_probabilities(std::size_t n, std::size_t max_values) {
  if (n > std::vector<double>().max_size() || (max_values && n > max_values))
    throw std::length_error("odds: complete secretary output geometry");
  std::vector<double> p(n);
  for (std::size_t i = 0; i < n; ++i)
    p[i] = 1. / static_cast<double>(i + 1);
  return p;
}
} // namespace datamunge::algorithms
