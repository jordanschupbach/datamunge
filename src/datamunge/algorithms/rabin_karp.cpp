/// @file rabin_karp.cpp
/// @brief One checked bytewise rolling-hash producer for ordinary, counted and traced matching.
#include <datamunge/algorithms/rabin_karp.hpp>
#include <limits>
#include <stdexcept>
/// @brief Canonical byte matching implementation.
namespace datamunge::algorithms {
/// @brief Private rolling-hash producer.
namespace {
/// @brief Execute matching with optional actual window capture.
/// @param text Text bytes.
/// @param pattern Pattern bytes.
/// @param o Hash and execution policies.
/// @param capture Whether to retain actual windows.
/// @return Exact matches and actual selected work.
RabinKarpResult solve(
    const std::string&      text,
    const std::string&      pattern,
    const RabinKarpOptions& o,
    bool                    capture) {
  if (o.modulus < 2 || o.base == 0 || o.base >= o.modulus)
    throw std::invalid_argument("rabin_karp: hash parameters");
  RabinKarpResult r;
  const auto      n = text.size(), m = pattern.size();
  /// @brief Admit and count one actual selected event.
  const auto tick = [&](std::size_t& counter) {
    if (r.operations == std::numeric_limits<std::size_t>::max() ||
        (o.max_operations && r.operations >= o.max_operations))
      throw std::length_error("rabin_karp: work policy");
    ++r.operations;
    ++counter;
  };
  /// @brief Admit one actual visited-window record before executing the visit.
  const auto admit = [&]() {
    if (capture && o.max_trace_entries && r.windows.size() >= o.max_trace_entries)
      throw std::length_error("rabin_karp: window history policy");
    tick(r.visited_windows);
  };
  if (m == 0) {
    if (n == std::numeric_limits<std::size_t>::max())
      throw std::length_error("rabin_karp: position capacity");
    for (std::size_t i = 0;; ++i) {
      admit();
      r.matches.push_back(i);
      if (capture)
        r.windows.push_back({i, 0, true, true, 0});
      if (i == n)
        break;
    }
    return r;
  }
  if (m > n)
    return r;
  r.high = 1;
  for (std::size_t i = 1; i < m; ++i) {
    tick(r.hash_updates);
    r.high = r.high * o.base % o.modulus;
  }
  std::uint64_t hash = 0;
  for (std::size_t i = 0; i < m; ++i) {
    tick(r.hash_updates);
    r.pattern_hash = (r.pattern_hash * o.base + static_cast<unsigned char>(pattern[i])) % o.modulus;
    tick(r.hash_updates);
    hash = (hash * o.base + static_cast<unsigned char>(text[i])) % o.modulus;
  }
  for (std::size_t start = 0;; ++start) {
    admit();
    const bool  candidate = hash == r.pattern_hash;
    bool        match     = candidate;
    std::size_t compared  = 0;
    if (candidate)
      for (std::size_t i = 0; i < m; ++i) {
        tick(r.compared_bytes);
        ++compared;
        if (static_cast<unsigned char>(text[start + i]) != static_cast<unsigned char>(pattern[i])) {
          match = false;
          break;
        }
      }
    if (match)
      r.matches.push_back(start);
    if (capture)
      r.windows.push_back({start, hash, candidate, match, compared});
    if (start == n - m)
      break;
    tick(r.hash_updates);
    const std::uint64_t leading  = static_cast<unsigned char>(text[start]),
                        incoming = static_cast<unsigned char>(text[start + m]);
    hash                         = (hash + o.modulus - leading * r.high % o.modulus) % o.modulus;
    hash                         = (hash * o.base + incoming) % o.modulus;
  }
  return r;
}
} // namespace
RabinKarpResult
rabin_karp_run(const std::string& t, const std::string& p, const RabinKarpOptions& o) {
  return solve(t, p, o, false);
}
RabinKarpResult
rabin_karp_trace(const std::string& t, const std::string& p, const RabinKarpOptions& o) {
  return solve(t, p, o, true);
}
std::vector<std::size_t> rabin_karp_search(const std::string& t, const std::string& p) {
  RabinKarpOptions o;
  o.max_operations    = 0;
  o.max_trace_entries = 0;
  return rabin_karp_run(t, p, o).matches;
}
} // namespace datamunge::algorithms
