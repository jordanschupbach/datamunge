/// @file sha256.cpp
/// @brief Single checked FIPS digest producer with native actual-state capture.
#include <algorithm>
#include <bit>
#include <datamunge/algorithms/hex_encoding.hpp>
#include <datamunge/algorithms/sha256.hpp>
#include <limits>
#include <stdexcept>
#include <utility>
/// @brief Canonical checked native producers and owned algorithm values.
namespace datamunge::algorithms {
/// @brief Private invocation-local checked helpers.
namespace {
/// @brief Private fixed FIPS round constants and rotation helpers.
namespace detail {

/// @brief FIPS180-4 SHA-256 cube-root round constants.
constexpr std::uint32_t sha256_k[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u,
    0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu,
    0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu,
    0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
    0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu,
    0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u,
    0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u,
    0xc67178f2u};

/// @brief Rotate one uint32 word right by an internal nonzero count.
/// @param x Actual word.
/// @param c Internal count in[1,31].
/// @return Actual modulo32 rotation.
std::uint32_t sha256_rotr(std::uint32_t x, int c) {
  return (x >> c) | (x << (32 - c));
}

} // namespace detail
/// @brief Execute one complete checked actual digest, optionally retaining real rounds.
/// @param message Borrowed raw bytes.
/// @param options Complete bounds.
/// @param capture Retain actual schedules and rounds.
/// @return Owned actual digest/work/history.
SHA256Run produce(const std::string& message, const SHAOptions& options, bool capture) {
  const auto n = message.size(), max = std::numeric_limits<std::size_t>::max();
  if ((options.max_bytes && n > options.max_bytes) ||
      n > std::numeric_limits<std::uint64_t>::max() / 8 || n > max - 72)
    throw std::length_error("sha256: complete input capacity/policy");
  const auto blocks = n / 64 + 1 + (n % 64 >= 56);
  if (blocks > max / 64 || blocks > max / 64 || blocks > max / 1233)
    throw std::length_error("sha256: work/capture capacity");
  const auto padded = 64 * blocks;
  if (padded > std::string().max_size() ||
      (capture && options.max_trace_entries && blocks * 1233 > options.max_trace_entries))
    throw std::length_error("sha256: complete padding/capture policy");

  std::uint32_t h[8] = {
      0x6a09e667u,
      0xbb67ae85u,
      0x3c6ef372u,
      0xa54ff53au,
      0x510e527fu,
      0x9b05688cu,
      0x1f83d9abu,
      0x5be0cd19u};

  std::string msg;
  msg.reserve(padded);
  msg                         = message;
  const std::uint64_t bit_len = static_cast<std::uint64_t>(message.size()) * 8u;
  msg.push_back(static_cast<char>(0x80));
  while (msg.size() % 64 != 56)
    msg.push_back(0);
  for (int i = 7; i >= 0; --i)
    msg.push_back(static_cast<char>((bit_len >> (8 * i)) & 0xFF));

  SHA256Run out;
  out.message_bytes = message.size();
  out.padded_bytes  = msg.size();
  for (std::size_t off = 0; off < msg.size(); off += 64) {
    ++out.compression_blocks;
    SHA256Block block;
    block.offset = off;
    std::uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
      w[i] = (static_cast<std::uint32_t>(static_cast<std::uint8_t>(msg[off + 4 * i])) << 24) |
             (static_cast<std::uint32_t>(static_cast<std::uint8_t>(msg[off + 4 * i + 1])) << 16) |
             (static_cast<std::uint32_t>(static_cast<std::uint8_t>(msg[off + 4 * i + 2])) << 8) |
             (static_cast<std::uint32_t>(static_cast<std::uint8_t>(msg[off + 4 * i + 3])));
    }
    for (int i = 16; i < 64; ++i) {
      std::uint32_t s0 =
          detail::sha256_rotr(w[i - 15], 7) ^ detail::sha256_rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      std::uint32_t s1 =
          detail::sha256_rotr(w[i - 2], 17) ^ detail::sha256_rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
      ++out.schedule_expansions;
    }

    if (capture) {
      std::copy(w, w + 64, block.schedule.begin());
      block.before = {h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]};
      block.rounds.reserve(64);
    }
    std::uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
    for (int i = 0; i < 64; ++i) {
      ++out.rounds;
      SHA256Round round;
      if (capture) {
        round.index  = i;
        round.word   = w[i];
        round.before = {a, b, c, d, e, f, g, hh};
      }
      std::uint32_t S1 =
          detail::sha256_rotr(e, 6) ^ detail::sha256_rotr(e, 11) ^ detail::sha256_rotr(e, 25);
      std::uint32_t ch = (e & f) ^ (~e & g);
      std::uint32_t t1 = hh + S1 + ch + detail::sha256_k[i] + w[i];
      std::uint32_t S0 =
          detail::sha256_rotr(a, 2) ^ detail::sha256_rotr(a, 13) ^ detail::sha256_rotr(a, 22);
      std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
      std::uint32_t t2  = S0 + maj;
      hh                = g;
      g                 = f;
      f                 = e;
      e                 = d + t1;
      d                 = c;
      c                 = b;
      b                 = a;
      a                 = t1 + t2;
      if (capture) {
        round.after = {a, b, c, d, e, f, g, hh};
        block.rounds.push_back(std::move(round));
      }
    }
    h[0] += a;
    h[1] += b;
    h[2] += c;
    h[3] += d;
    h[4] += e;
    h[5] += f;
    h[6] += g;
    h[7] += hh;
    if (capture) {
      block.after = {h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]};
      out.blocks.push_back(std::move(block));
    }
  }

  std::array<std::uint8_t, 32> digest{};
  for (int w = 0; w < 8; ++w)
    for (int i = 0; i < 4; ++i)
      digest[4 * w + i] = static_cast<std::uint8_t>((h[w] >> (24 - 8 * i)) & 0xFF);
  out.digest = digest;
  return out;
}
} // namespace
SHA256Run sha256_run(const std::string& m, const SHAOptions& o) {
  return produce(m, o, false);
}
SHA256Run sha256_trace(const std::string& m, const SHAOptions& o) {
  return produce(m, o, true);
}
std::array<std::uint8_t, 32> sha256(const std::string& m) {
  SHAOptions options;options.max_bytes=0;return sha256_run(m,options).digest;
}
std::string sha256_hex(const std::string& m) {
  return to_hex(sha256(m));
}
HashAvalanche sha256_avalanche(const std::string& m, std::size_t cap, const SHAOptions& o) {
  const auto max = std::numeric_limits<std::size_t>::max();
  if (m.size() > max / 8)
    throw std::length_error("avalanche: bit count capacity");
  const auto bits = m.size() * 8;
  if ((cap && bits > cap) || bits == max)
    throw std::length_error("avalanche: complete flip policy");
  const auto blocks = m.size() / 64 + 1 + (m.size() % 64 >= 56);
  if (blocks && bits + 1 > max / blocks)
    throw std::length_error("avalanche: cumulative block count capacity");
  HashAvalanche out;
  out.histogram.assign(257, 0);
  const auto baseline = sha256_run(m, o);
  ++out.hash_calls;
  out.compression_blocks += baseline.compression_blocks;
  out.baseline_hex = to_hex(baseline.digest);
  out.changed_bits.reserve(bits);
  std::string variant = m;
  long double sum     = 0;
  for (std::size_t bit = 0; bit < bits; ++bit) {
    const auto mask   = static_cast<unsigned char>(1U << (bit % 8));
    variant[bit / 8]  = static_cast<char>(static_cast<unsigned char>(variant[bit / 8]) ^ mask);
    const auto result = sha256_run(variant, o);
    ++out.hash_calls;
    out.compression_blocks += result.compression_blocks;
    std::size_t changes = 0;
    for (std::size_t j = 0; j < 32; ++j)
      changes += std::popcount(static_cast<unsigned>(baseline.digest[j] ^ result.digest[j]));
    out.changed_bits.push_back(changes);
    ++out.histogram[changes];
    sum += changes;
    variant[bit / 8] = m[bit / 8];
  }
  out.mean = bits ? static_cast<double>(sum / bits) : 0;
  return out;
}
} // namespace datamunge::algorithms
