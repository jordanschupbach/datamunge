/// @file pbkdf2.cpp
/// @brief Canonical producing implementation and actual-site diagnostics; public contracts are in the paired header.
#include <datamunge/algorithms/hex_encoding.hpp>
#include <datamunge/algorithms/hmac.hpp>
#include <datamunge/algorithms/pbkdf2.hpp>
#include <limits>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief File-local validated kernels shared by ordinary and observed entry points.
namespace {
/// @brief Shared derivation, optionally recording the actual chain after each HMAC.
Pbkdf2Run derive(
    const std::string& p,
    const std::string& s,
    std::uint32_t      c,
    std::size_t        len,
    Pbkdf2Options      o,
    bool               trace) {
  if (c == 0)
    throw std::invalid_argument("PBKDF2 iterations must be positive");
  const auto blocks = len / 32 + (len % 32 != 0);
  if (blocks > std::numeric_limits<std::uint32_t>::max() ||
      blocks > std::numeric_limits<std::size_t>::max() / c)
    throw std::length_error("PBKDF2 standard length/work range");
  const auto calls = blocks * c;
  if ((o.max_hmac_calls && calls > o.max_hmac_calls) ||
      (trace && o.max_trace_entries && calls > o.max_trace_entries))
    throw std::length_error("PBKDF2 work/history policy");
  if (s.size() > std::string().max_size() - 4)
    throw std::length_error("PBKDF2 salt suffix range");
  Pbkdf2Run r;
  r.key.reserve(len);
  if (trace)
    r.steps.reserve(calls);
  for (std::size_t bi = 0; bi < blocks; ++bi) {
    const auto  i   = static_cast<std::uint32_t>(bi + 1);
    std::string msg = s;
    for (int shift : {24, 16, 8, 0})
      msg.push_back(static_cast<char>((i >> shift) & 255));
    auto u = hmac_sha256(p, msg);
    ++r.hmac_calls;
    auto t = u;
    if (trace)
      r.steps.push_back({i, 1, u, t});
    for (std::uint32_t j = 1; j < c; ++j) {
      std::string prev(reinterpret_cast<const char*>(u.data()), u.size());
      u = hmac_sha256(p, prev);
      ++r.hmac_calls;
      for (std::size_t k = 0; k < 32; ++k)
        t[k] ^= u[k];
      if (trace)
        r.steps.push_back({i, j + 1, u, t});
    }
    for (std::size_t k = 0; k < 32 && r.key.size() < len; ++k)
      r.key.push_back(t[k]);
  }
  return r;
}
} // namespace
Pbkdf2Run pbkdf2_hmac_sha256_run(
    const std::string& p,
    const std::string& s,
    std::uint32_t      c,
    std::size_t        n,
    Pbkdf2Options      o) {
  return derive(p, s, c, n, o, false);
}
Pbkdf2Run pbkdf2_hmac_sha256_trace(
    const std::string& p,
    const std::string& s,
    std::uint32_t      c,
    std::size_t        n,
    Pbkdf2Options      o) {
  return derive(p, s, c, n, o, true);
}
std::vector<std::uint8_t>
pbkdf2_hmac_sha256(const std::string& p, const std::string& s, std::uint32_t c, std::size_t n) {
  return derive(p, s, c, n, {}, false).key;
}
std::string
pbkdf2_hmac_sha256_hex(const std::string& p, const std::string& s, std::uint32_t c, std::size_t n) {
  if (n > std::string().max_size() / 2)
    throw std::length_error("PBKDF2 hex capacity");
  auto key = pbkdf2_hmac_sha256(p, s, c, n);
  return to_hex(std::string_view(reinterpret_cast<const char*>(key.data()), key.size()));
}
} // namespace datamunge::algorithms
