/// @file lfsr.cpp
/// @brief LFSR generation, period search, linear complexity profiles and the Geffe combiner.

#include <datamunge/algorithms/lfsr.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace datamunge::algorithms {

std::vector<int> lfsr_sequence(const std::vector<int>& connection, const std::vector<int>& seed, std::size_t n) {
  if (connection.empty() || connection[0] != 1)
    throw std::invalid_argument("lfsr_sequence: the connection polynomial must start with 1");
  const std::size_t L = connection.size() - 1;
  if (seed.size() != L) throw std::invalid_argument("lfsr_sequence: the seed must hold one bit per register stage");
  std::vector<int> s(std::max(n, L));
  for (std::size_t i = 0; i < L; ++i) s[i] = seed[i] & 1;
  for (std::size_t i = L; i < n; ++i) {
    int bit = 0;
    for (std::size_t j = 1; j <= L; ++j) bit ^= connection[j] & s[i - j];
    s[i] = bit;
  }
  s.resize(n);
  return s;
}

long long lfsr_period(const std::vector<int>& connection, const std::vector<int>& seed, long long max_steps) {
  if (connection.size() < 2 || connection[0] != 1 || (connection.back() & 1) == 0)
    throw std::invalid_argument("lfsr_period: the connection polynomial must be {1, ..., 1}");
  const std::size_t L = connection.size() - 1;
  if (L > 62) throw std::invalid_argument("lfsr_period: at most 62 stages");
  if (seed.size() != L) throw std::invalid_argument("lfsr_period: the seed must hold one bit per register stage");
  // State bit j holds s_{i-1-j}; the taps mask has bit j-1 set when c_j = 1.
  std::uint64_t taps = 0, state = 0;
  for (std::size_t j = 1; j <= L; ++j)
    if (connection[j] & 1) taps |= std::uint64_t{1} << (j - 1);
  for (std::size_t j = 0; j < L; ++j)
    if (seed[L - 1 - j] & 1) state |= std::uint64_t{1} << j;
  if (state == 0) throw std::invalid_argument("lfsr_period: the all-zero seed never moves");
  const std::uint64_t mask = (std::uint64_t{1} << L) - 1;
  const std::uint64_t start = state;
  for (long long step = 1; step <= max_steps; ++step) {
    const auto bit = static_cast<std::uint64_t>(std::popcount(state & taps) & 1);
    state = ((state << 1) | bit) & mask;
    if (state == start) return step;
  }
  return 0;
}

std::vector<int> lfsr_seed(long long code, int length) {
  std::vector<int> seed(static_cast<std::size_t>(std::max(length, 0)));
  for (int i = 0; i < length; ++i) seed[static_cast<std::size_t>(i)] = static_cast<int>((code >> i) & 1);
  return seed;
}

std::vector<int> lfsr_correlation_scan(const std::vector<int>& connection, const std::vector<int>& observed) {
  if (connection.empty() || connection[0] != 1)
    throw std::invalid_argument("lfsr_correlation_scan: the connection polynomial must start with 1");
  const std::size_t L = connection.size() - 1;
  if (L == 0 || L > 30) throw std::invalid_argument("lfsr_correlation_scan: need 1 to 30 stages");
  std::uint64_t taps = 0;
  for (std::size_t j = 1; j <= L; ++j)
    if (connection[j] & 1) taps |= std::uint64_t{1} << (j - 1);
  const std::uint64_t mask = (std::uint64_t{1} << L) - 1;
  const std::size_t head = std::min(L, observed.size());
  std::vector<int> score(std::size_t{1} << L, 0);
  for (std::uint64_t code = 0; code <= mask; ++code) {
    int agree = 0;
    std::uint64_t state = 0;  // bit j holds s_{i-1-j}
    for (std::size_t i = 0; i < L; ++i) {
      const auto bit = (code >> i) & 1;
      if (i < head && (observed[i] == 0 || observed[i] == 1)) agree += static_cast<int>(bit) == observed[i];
      state = (state << 1) | bit;
    }
    for (std::size_t i = L; i < observed.size(); ++i) {
      const auto bit = static_cast<std::uint64_t>(std::popcount(state & taps) & 1);
      if (observed[i] == 0 || observed[i] == 1) agree += static_cast<int>(bit) == observed[i];
      state = ((state << 1) | bit) & mask;
    }
    score[code] = agree;
  }
  return score;
}

std::vector<int> linear_complexity_profile(const std::vector<int>& s) {
  const std::size_t n = s.size();
  std::vector<int> c(n + 1, 0), b(n + 1, 0), t, profile(n);
  c[0] = b[0] = 1;
  std::size_t L = 0, m = 1;
  for (std::size_t i = 0; i < n; ++i) {
    int d = s[i] & 1;
    for (std::size_t j = 1; j <= L; ++j) d ^= c[j] & s[i - j];
    if (d == 0) {
      ++m;
    } else if (2 * L <= i) {
      t = c;
      for (std::size_t j = 0; j + m <= n; ++j) c[j + m] ^= b[j];
      L = i + 1 - L;
      b = t;
      m = 1;
    } else {
      for (std::size_t j = 0; j + m <= n; ++j) c[j + m] ^= b[j];
      ++m;
    }
    profile[i] = static_cast<int>(L);
  }
  return profile;
}

/// Rueppel's mean linear complexity of n fair bits; see the declaration in lfsr.hpp.
double expected_linear_complexity(std::size_t n) {
  if (n == 0) return 0.0;
  const double nd = static_cast<double>(n);
  return nd / 2.0 + (4.0 + static_cast<double>(n % 2)) / 18.0 - std::ldexp(nd / 3.0 + 2.0 / 9.0, -static_cast<int>(std::min<std::size_t>(n, 2000)));
}

std::vector<int> geffe_sequence(const std::vector<int>& x1, const std::vector<int>& x2, const std::vector<int>& x3) {
  const std::size_t n = std::min({x1.size(), x2.size(), x3.size()});
  std::vector<int> z(n);
  for (std::size_t i = 0; i < n; ++i) z[i] = (x2[i] & 1) ? (x1[i] & 1) : (x3[i] & 1);
  return z;
}

std::vector<int> bytes_to_bits(std::string_view text) {
  std::vector<int> bits;
  bits.reserve(8 * text.size());
  for (const char ch : text) {
    const auto byte = static_cast<unsigned char>(ch);
    for (int k = 7; k >= 0; --k) bits.push_back((byte >> k) & 1);
  }
  return bits;
}

std::string bits_to_bytes(const std::vector<int>& bits) {
  std::string out((bits.size() + 7) / 8, '\0');
  for (std::size_t i = 0; i < bits.size(); ++i)
    if (bits[i] & 1) out[i / 8] = static_cast<char>(static_cast<unsigned char>(out[i / 8]) | (0x80u >> (i % 8)));
  return out;
}

std::vector<int> xor_bits(const std::vector<int>& a, const std::vector<int>& b) {
  const std::size_t n = std::min(a.size(), b.size());
  std::vector<int> c(n);
  for (std::size_t i = 0; i < n; ++i) c[i] = (a[i] ^ b[i]) & 1;
  return c;
}

}  // namespace datamunge::algorithms
