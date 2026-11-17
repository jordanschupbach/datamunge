/// @file jaro_winkler.cpp
/// @brief Shared original greedy byte matching with actual phase events and checked bounds.
#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/jaro_winkler.hpp>
#include <limits>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Private input/event admission and one producing matching kernel.
namespace {
/// @brief Increment selected work without wrapping.
/// @param value Mutable actual count.
/// @throws std::length_error On exhausted size_t count.
void increment(std::size_t& value) {
  if (value == std::numeric_limits<std::size_t>::max())
    throw std::length_error("Jaro count overflow");
  ++value;
}
/// @brief Admit one optional event before its actual inspection.
/// @param out Borrowed current output.
/// @param options Actual-record cap policy.
/// @param capture Whether chronology is retained.
/// @throws std::length_error On enabled cap or exhausted vector capacity.
void admit(const JaroWinklerTrace& out, JaroWinklerOptions options, bool capture) {
  if (capture && ((options.max_events && out.events.size() >= options.max_events) ||
                  out.events.size() == out.events.max_size()))
    throw std::length_error("Jaro actual event cap");
}
/// @brief Execute one raw-byte greedy match, order comparison and optional prefix pass.
/// @param a Borrowed complete first bytes.
/// @param b Borrowed complete second bytes.
/// @param scale Prefix weight, checked when prefix pass is enabled.
/// @param options Complete input and optional actual-event policies.
/// @param capture Whether to copy input/retain chronological scalar events.
/// @param prefix Whether to perform the actual Winkler prefix pass.
/// @return Actual same-run scores/work and optional owned trace.
/// @throws std::invalid_argument On invalid active prefix scale.
/// @throws std::length_error On enabled cap or size/count/container overflow.
/// @throws std::bad_alloc On input/flags/event allocation failure.
JaroWinklerTrace execute(
    const std::string& a,
    const std::string& b,
    double             scale,
    JaroWinklerOptions options,
    bool               capture,
    bool               prefix) {
  if (prefix && (!std::isfinite(scale) || scale < 0 || scale > .25))
    throw std::invalid_argument("Winkler finite scale in [0,.25] required");
  if (a.size() > std::numeric_limits<std::size_t>::max() - b.size() ||
      (options.max_total_bytes && a.size() + b.size() > options.max_total_bytes))
    throw std::length_error("Jaro complete input bytes");
  JaroWinklerTrace out;
  auto&            r = out.run;
  if (capture) {
    out.first  = a;
    out.second = b;
  }
  if (a.empty() || b.empty()) {
    r.jaro = r.winkler = a.empty() && b.empty() ? 1 : 0;
    return out;
  }
  const auto la = a.size(), lb = b.size();
  r.window = std::max(la, lb) / 2;
  if (r.window)
    --r.window;
  std::vector<bool> am(la, false), bm(lb, false);
  for (std::size_t i = 0; i < la; ++i) {
    const auto lo = i > r.window ? i - r.window : 0;
    const auto hi = i >= lb || r.window >= lb - i - 1 ? lb : i + r.window + 1;
    for (std::size_t j = lo; j < hi; ++j) {
      admit(out, options, capture);
      increment(r.work.candidate_slots);
      JaroWinklerEvent event;
      event.first_index  = i;
      event.second_index = j;
      event.occupied     = bm[j];
      if (!event.occupied) {
        increment(r.work.candidate_byte_tests);
        event.compared    = true;
        event.first_byte  = static_cast<std::uint8_t>(a[i]);
        event.second_byte = static_cast<std::uint8_t>(b[j]);
        event.equal       = event.first_byte == event.second_byte;
        if (event.equal) {
          am[i] = true;
          bm[j] = true;
          ++r.matches;
        }
      }
      if (capture)
        out.events.push_back(event);
      if (event.compared && event.equal)
        break;
    }
  }
  if (r.matches) {
    std::size_t k = 0;
    for (std::size_t i = 0; i < la; ++i) {
      if (!am[i])
        continue;
      while (k < lb && !bm[k])
        ++k;
      if (k == lb)
        throw std::logic_error("Jaro matched-count invariant");
      admit(out, options, capture);
      increment(r.work.order_byte_tests);
      const auto ac = static_cast<std::uint8_t>(a[i]), bc = static_cast<std::uint8_t>(b[k]);
      const bool equal = ac == bc;
      if (!equal)
        ++r.order_mismatches;
      if (capture)
        out.events.push_back({JaroWinklerPhase::Order, i, k, false, true, ac, bc, equal});
      ++k;
    }
    const double m = static_cast<double>(r.matches);
    r.jaro         = (m / static_cast<double>(la) + m / static_cast<double>(lb) +
              (m - static_cast<double>(r.order_mismatches) / 2.0) / m) /
             3.0;
  }
  if (prefix) {
    const auto limit = std::min<std::size_t>({la, lb, 4});
    while (r.prefix < limit) {
      admit(out, options, capture);
      increment(r.work.prefix_byte_tests);
      const auto i  = r.prefix;
      const auto ac = static_cast<std::uint8_t>(a[i]), bc = static_cast<std::uint8_t>(b[i]);
      const bool equal = ac == bc;
      if (capture)
        out.events.push_back({JaroWinklerPhase::Prefix, i, i, false, true, ac, bc, equal});
      if (!equal)
        break;
      ++r.prefix;
    }
    r.winkler = r.jaro + static_cast<double>(r.prefix) * scale * (1.0 - r.jaro);
  } else
    r.winkler = r.jaro;
  return out;
}
} // namespace
double jaro_similarity(const std::string& a, const std::string& b) {
  return execute(a, b, 0, {0, 0}, false, false).run.jaro;
}
double jaro_winkler_similarity(const std::string& a, const std::string& b, double prefix_scale) {
  return execute(a, b, prefix_scale, {0, 0}, false, true).run.winkler;
}
JaroWinklerRun jaro_winkler_run(
    const std::string& a,
    const std::string& b,
    double             prefix_scale,
    JaroWinklerOptions options) {
  return execute(a, b, prefix_scale, options, false, true).run;
}
JaroWinklerTrace jaro_winkler_trace(
    const std::string& a,
    const std::string& b,
    double             prefix_scale,
    JaroWinklerOptions options) {
  return execute(a, b, prefix_scale, options, true, true);
}
} // namespace datamunge::algorithms
