/// @file manacher.cpp
/// @brief Shared safe tagged-byte mirror-radius producer and actual frontier/expansion observations.
#include <algorithm>
#include <datamunge/algorithms/manacher.hpp>
#include <format>
#include <limits>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Private checked complete geometry and optional actual capture helpers.
namespace {
/// @brief Add actual logical counts without wrapping.
/// @param a Current nonnegative count.
/// @param b Additional nonnegative count.
/// @return Exact representable sum.
std::size_t add(std::size_t a, std::size_t b) {
  if (b > std::numeric_limits<std::size_t>::max() - a)
    throw std::length_error("Manacher: count overflow");
  return a + b;
}
/// @brief Admit known base slots or one actual history record before copying it.
/// @param trace Optional actual owned capture output.
/// @param amount Actual next logical slots/records.
/// @param quota Optional selected capture cap.
void admit(ManacherTrace* trace, std::size_t amount, std::size_t quota) {
  if (!trace)
    return;
  trace->captured_entries = add(trace->captured_entries, amount);
  if (quota && trace->captured_entries > quota)
    throw std::length_error("Manacher: capture quota");
}
/// @brief Transform original bytes into distinct control-tagged integers and run the mirror scan once.
/// @param s Complete original bytes.
/// @param trace Optional actual owned input/symbol/history output.
/// @param quota Optional selected logical capture limit.
/// @return Actual radii, leftmost longest result and selected producing operations.
ManacherAnalysis produce(const std::string& s, ManacherTrace* trace, std::size_t quota) {
  if (s.size() > (std::numeric_limits<std::size_t>::max() - 3) / 2)
    throw std::length_error("Manacher: transform overflow");
  const std::size_t m = 2 * s.size() + 3;
  if (m > std::vector<int>().max_size() || m > std::vector<std::size_t>().max_size())
    throw std::length_error("Manacher: transform/radius container geometry");
  if (trace) {
    admit(trace, add(s.size(), add(m, m)), quota);
    trace->input = s;
  }
  std::vector<int> symbols;
  symbols.reserve(m);
  symbols.push_back(257);
  for (unsigned char c : s) {
    symbols.push_back(256);
    symbols.push_back(c);
  }
  symbols.push_back(256);
  symbols.push_back(258);
  if (trace)
    trace->symbols = symbols;
  ManacherAnalysis out;
  out.radii.assign(m, 0);
  std::size_t center = 0, right = 0;
  for (std::size_t i = 1; i + 1 < m; ++i) {
    ++out.work.centers;
    const auto  center_before = center, right_before = right;
    const bool  reused = i < right;
    std::size_t mirror = std::numeric_limits<std::size_t>::max(), mirror_radius = 0, bound = 0;
    if (reused) {
      ++out.work.mirror_reuses;
      const auto distance = i - center;
      if (distance > center)
        throw std::logic_error("Manacher: invalid mirror invariant");
      mirror        = center - distance;
      mirror_radius = out.radii[mirror];
      bound         = right - i;
      out.radii[i]  = std::min(mirror_radius, bound);
    }
    const auto initial = out.radii[i];
    while (out.radii[i] < i && out.radii[i] + 1 < m - i) {
      const auto left = i - out.radii[i] - 1, r = i + out.radii[i] + 1;
      const bool equal = symbols[left] == symbols[r];
      ++out.work.comparisons;
      if (trace) {
        admit(trace, 1, quota);
        trace->comparisons.push_back({i, out.radii[i], left, r, symbols[left], symbols[r], equal});
      }
      if (!equal)
        break;
      ++out.radii[i];
      ++out.work.expansions;
    }
    const bool update = i + out.radii[i] > right;
    if (update) {
      center = i;
      right  = i + out.radii[i];
      ++out.work.frontier_updates;
    }
    const bool best = out.radii[i] > out.longest.length;
    if (best) {
      out.best_center = i;
      out.longest     = {(i - out.radii[i]) / 2, out.radii[i]};
      ++out.work.best_updates;
    }
    if (trace) {
      admit(trace, 1, quota);
      trace->centers.push_back(
          {i,
           center_before,
           right_before,
           reused,
           mirror,
           mirror_radius,
           bound,
           initial,
           out.radii[i],
           update,
           center,
           right,
           best});
    }
  }
  return out;
}
} // namespace
PalindromeResult longest_palindrome(const std::string& s) {
  return produce(s, nullptr, 0).longest;
}
std::vector<std::size_t> palindrome_radii(const std::string& s) {
  return produce(s, nullptr, 0).radii;
}
ManacherAnalysis manacher_analyze(const std::string& s) {
  return produce(s, nullptr, 0);
}
ManacherTrace manacher_trace(const std::string& s, std::size_t max_entries) {
  ManacherTrace t;
  t.analysis = produce(s, &t, max_entries);
  return t;
}
std::string manacher_symbol_label(int symbol) {
  if (symbol < 0 || symbol > 258)
    throw std::invalid_argument("Manacher: symbol outside byte/tag domain");
  if (symbol == 256)
    return "separator";
  if (symbol == 257)
    return "left guard";
  if (symbol == 258)
    return "right guard";
  return std::format("byte0x{:02X}", symbol);
}
} // namespace datamunge::algorithms
