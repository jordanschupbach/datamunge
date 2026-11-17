/// @file pollard_kangaroo.cpp
/// @brief Original seeded kangaroos with exact widened trail exponents and actual-site histories.
#include <algorithm>
#include <datamunge/algorithms/integer_arithmetic.hpp>
#include <datamunge/algorithms/modular_exponentiation.hpp>
#include <datamunge/algorithms/pollard_kangaroo.hpp>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <vector>
namespace datamunge::algorithms {
/// @brief Local deterministic table generation and shared checked producing walks.
namespace {
/// @brief Native unsigned fixed-width aliases; distance arithmetic is exact in the wider type.
using U64 = std::uint64_t;
/// @brief Distance/trail exponent capacity, enough for all uint64 interval-dependent native phase lengths.
using U128 = unsigned __int128;
/// @brief Original xor-shift table generator, including its zero-state substitute.
struct XorShift {
  U64 state; ///< Actual current generator word.

  /// @brief Initialise the original word or nonzero substitute.
  explicit XorShift(U64 seed) : state(seed ? seed : 0x9E3779B97F4A7C15ULL) {}
  /// @brief Return the actual next13/7/17 xor-shift word modulo2^64.
  U64 next() {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
  }
};
/// @brief Increment a selected actual counter without wraparound.
void increment(std::size_t& n) {
  if (n == std::numeric_limits<std::size_t>::max())
    throw std::overflow_error("kangaroo counter range");
  ++n;
}
/// @brief Shared exact-distance producer with optional actual-site records.
KangarooTrace
run(U64 g, U64 h, U64 P, U64 a, U64 b, U64 seed, KangarooOptions options, bool capture) {
  if (P < 2)
    throw std::invalid_argument("kangaroo modulus must be >=2");
  g %= P;
  h %= P;
  if (std::gcd(g, P) != 1 || std::gcd(h, P) != 1)
    throw std::invalid_argument("kangaroo requires unit base and target");
  KangarooTrace t;
  auto&         w       = t.run;
  auto&         r       = w.result;
  std::size_t   records = 0;
  /// @brief Admit one actual retained table/jump/collision record.
  auto record = [&]() {
    if (records == std::numeric_limits<std::size_t>::max() ||
        (options.max_trace_records && records >= options.max_trace_records))
      throw std::length_error("kangaroo history policy");
    ++records;
  };
  /// @brief Count and call exact canonical modular powering.
  auto power = [&](U64 base, U64 exponent) {
    increment(w.power_calls);
    return mod_pow(base, exponent, P);
  };
  /// @brief Admit one producing group jump before arithmetic or mutation.
  auto jump = [&]() {
    if (r.jumps == std::numeric_limits<U64>::max())
      throw std::overflow_error("kangaroo jump count range");
    if (options.max_jumps && r.jumps >= options.max_jumps)
      throw std::length_error("kangaroo jump policy");
    ++r.jumps;
  };
  if (b < a)
    return t;
  if (power(g, a) == h) {
    r.x     = a;
    r.found = true;
    return t;
  }
  if (power(g, b) == h) {
    r.x     = b;
    r.found = true;
    return t;
  }
  const U64 m = std::max<U64>(2, integer_sqrt_u64(b - a)), N = 8 * m + 64, wbound = 20 * m + 256;
  for (std::size_t attempt = 0; attempt < options.max_attempts; ++attempt) {
    if (options.max_jumps && (r.jumps > options.max_jumps || N > options.max_jumps - r.jumps))
      throw std::length_error("kangaroo planned tame phase exceeds remaining jump policy");
    std::unordered_map<U64, U128> trail;
    if (N > std::numeric_limits<std::size_t>::max() / 2 || 2 * N > trail.max_size())
      throw std::length_error("kangaroo trail capacity");
    trail.reserve(static_cast<std::size_t>(2 * N));
    increment(w.attempts);
    XorShift         rng(seed + 0x9E3779B97F4A7C15ULL * static_cast<U64>(attempt + 1));
    std::vector<U64> steps(32), powers(32);
    for (unsigned j = 0; j < 32; ++j) {
      steps[j]  = 1 + rng.next() % (2 * m);
      powers[j] = power(g, steps[j]);
      if (capture) {
        record();
        t.table.push_back({attempt, j, steps[j], powers[j]});
      }
    }
    U64  y        = power(g, b);
    U128 distance = 0;
    trail[y]      = b;
    increment(w.tame_assignments);
    for (U64 i = 0; i < N; ++i) {
      const auto before = y;
      const auto slot   = static_cast<unsigned>(y % 32);
      if (capture)
        record();
      jump();
      y = mul_mod(y, powers[slot], P);
      distance += steps[slot];
      if (trail.contains(y))
        increment(w.trail_overwrites);
      trail[y] = static_cast<U128>(b) + distance;
      increment(w.tame_assignments);
      if (capture)
        t.jumps.push_back(
            {attempt,
             false,
             before,
             y,
             slot,
             steps[slot],
             distance,
             static_cast<U128>(b) + distance});
    }
    U64  wild = h;
    U128 dw   = 0;
    for (U64 i = 0; i < wbound; ++i) {
      increment(w.wild_lookups);
      const auto it = trail.find(wild);
      if (it != trail.end()) {
        bool in = false, verified = false;
        U64  candidate = 0;
        if (it->second >= dw) {
          const auto difference = it->second - dw;
          if (difference >= a && difference <= b) {
            in        = true;
            candidate = static_cast<U64>(difference);
            verified  = power(g, candidate) == h;
          }
        }
        if (capture) {
          record();
          t.collisions.push_back({attempt, wild, it->second, dw, in, candidate, verified});
        }
        if (verified) {
          r.x     = candidate;
          r.found = true;
          return t;
        }
      }
      const auto before = wild;
      const auto slot   = static_cast<unsigned>(wild % 32);
      if (capture)
        record();
      jump();
      wild = mul_mod(wild, powers[slot], P);
      dw += steps[slot];
      if (capture)
        t.jumps.push_back({attempt, true, before, wild, slot, steps[slot], dw, 0});
    }
  }
  return t;
}
} // namespace
KangarooResult pollard_kangaroo(std::uint64_t g, std::uint64_t h, std::uint64_t P, std::uint64_t a, std::uint64_t b, std::uint64_t seed) {
  return run(g, h, P, a, b, seed, {}, false).run.result;
}
KangarooRun
pollard_kangaroo_run(std::uint64_t g, std::uint64_t h, std::uint64_t P, std::uint64_t a, std::uint64_t b, std::uint64_t seed, KangarooOptions options) {
  return run(g, h, P, a, b, seed, options, false).run;
}
KangarooTrace
pollard_kangaroo_trace(std::uint64_t g, std::uint64_t h, std::uint64_t P, std::uint64_t a, std::uint64_t b, std::uint64_t seed, KangarooOptions options) {
  return run(g, h, P, a, b, seed, options, true);
}
} // namespace datamunge::algorithms
