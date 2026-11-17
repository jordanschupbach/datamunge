/// @file sha1.cpp
/// @brief Single checked FIPS digest producer with native actual-state capture.
#include <algorithm>
#include <bit>
#include <datamunge/algorithms/hex_encoding.hpp>
#include <datamunge/algorithms/sha1.hpp>
#include <limits>
#include <stdexcept>
#include <utility>
/// @brief Canonical checked native producers and owned algorithm values.
namespace datamunge::algorithms {
/// @brief Private invocation-local checked helpers.
namespace {
/// @brief Private fixed FIPS round constants and rotation helpers.
namespace detail {

/// @brief Rotate one uint32 word by an internal nonzero count.
/// @param x Actual word.
/// @param c Internal count in[1,31].
/// @return Actual modulo32 rotation.
std::uint32_t sha1_rotl(std::uint32_t x, int c) {
  return (x << c) | (x >> (32 - c));
}

} // namespace detail
/// @brief Execute one complete checked actual digest, optionally retaining real rounds.
/// @param message Borrowed raw bytes.
/// @param options Complete bounds.
/// @param capture Retain actual schedules and rounds.
/// @return Owned actual digest/work/history.
SHA1Run produce(const std::string& message, const SHAOptions& options, bool capture) {
  const auto n = message.size(), max = std::numeric_limits<std::size_t>::max();
  if ((options.max_bytes && n > options.max_bytes) ||
      n > std::numeric_limits<std::uint64_t>::max() / 8 || n > max - 72)
    throw std::length_error("sha1: complete input capacity/policy");
  const auto blocks = n / 64 + 1 + (n % 64 >= 56);
  if (blocks > max / 64 || blocks > max / 80 || blocks > max / 1051)
    throw std::length_error("sha1: work/capture capacity");
  const auto padded = 64 * blocks;
  if (padded > std::string().max_size() ||
      (capture && options.max_trace_entries && blocks * 1051 > options.max_trace_entries))
    throw std::length_error("sha1: complete padding/capture policy");

  std::uint32_t h0 = 0x67452301u, h1 = 0xEFCDAB89u, h2 = 0x98BADCFEu, h3 = 0x10325476u,
                h4 = 0xC3D2E1F0u;

  // Padding: append 0x80, zeros to length r 56 (mod 64), then 64-bit big-endian length.
  std::string msg;
  msg.reserve(padded);
  msg                         = message;
  const std::uint64_t bit_len = static_cast<std::uint64_t>(message.size()) * 8u;
  msg.push_back(static_cast<char>(0x80));
  while (msg.size() % 64 != 56)
    msg.push_back(0);
  for (int i = 7; i >= 0; --i)
    msg.push_back(static_cast<char>((bit_len >> (8 * i)) & 0xFF));

  SHA1Run out;
  out.message_bytes = message.size();
  out.padded_bytes  = msg.size();
  for (std::size_t off = 0; off < msg.size(); off += 64) {
    ++out.compression_blocks;
    SHA1Block block;
    block.offset = off;
    std::uint32_t w[80];
    for (int i = 0; i < 16; ++i) {
      w[i] = (static_cast<std::uint32_t>(static_cast<std::uint8_t>(msg[off + 4 * i])) << 24) |
             (static_cast<std::uint32_t>(static_cast<std::uint8_t>(msg[off + 4 * i + 1])) << 16) |
             (static_cast<std::uint32_t>(static_cast<std::uint8_t>(msg[off + 4 * i + 2])) << 8) |
             (static_cast<std::uint32_t>(static_cast<std::uint8_t>(msg[off + 4 * i + 3])));
    }
    for (int i = 16; i < 80; ++i) {
      w[i] = detail::sha1_rotl(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
      ++out.schedule_expansions;
    }

    if (capture) {
      std::copy(w, w + 80, block.schedule.begin());
      block.before = {h0, h1, h2, h3, h4};
      block.rounds.reserve(80);
    }
    std::uint32_t a = h0, b = h1, c = h2, d = h3, e = h4;
    for (int i = 0; i < 80; ++i) {
      ++out.rounds;
      SHA1Round round;
      if (capture) {
        round.index  = i;
        round.word   = w[i];
        round.before = {a, b, c, d, e};
      }
      std::uint32_t f, k;
      if (i < 20) {
        f = (b & c) | (~b & d);
        k = 0x5A827999u;
      } else if (i < 40) {
        f = b ^ c ^ d;
        k = 0x6ED9EBA1u;
      } else if (i < 60) {
        f = (b & c) | (b & d) | (c & d);
        k = 0x8F1BBCDCu;
      } else {
        f = b ^ c ^ d;
        k = 0xCA62C1D6u;
      }
      std::uint32_t temp = detail::sha1_rotl(a, 5) + f + e + k + w[i];
      e                  = d;
      d                  = c;
      c                  = detail::sha1_rotl(b, 30);
      b                  = a;
      a                  = temp;
      if (capture) {
        round.after = {a, b, c, d, e};
        block.rounds.push_back(std::move(round));
      }
    }
    h0 += a;
    h1 += b;
    h2 += c;
    h3 += d;
    h4 += e;
    if (capture) {
      block.after = {h0, h1, h2, h3, h4};
      out.blocks.push_back(std::move(block));
    }
  }

  std::array<std::uint8_t, 20> digest{};
  const std::uint32_t          hs[5] = {h0, h1, h2, h3, h4};
  for (int w = 0; w < 5; ++w)
    for (int i = 0; i < 4; ++i)
      digest[4 * w + i] = static_cast<std::uint8_t>((hs[w] >> (24 - 8 * i)) & 0xFF);
  out.digest = digest;
  return out;
}
} // namespace
SHA1Run sha1_run(const std::string& m, const SHAOptions& o) {
  return produce(m, o, false);
}
SHA1Run sha1_trace(const std::string& m, const SHAOptions& o) {
  return produce(m, o, true);
}
std::array<std::uint8_t, 20> sha1(const std::string& m) {
  SHAOptions options;options.max_bytes=0;return sha1_run(m,options).digest;
}
std::string sha1_hex(const std::string& m) {
  return to_hex(sha1(m));
}
HashAvalanche sha1_avalanche(const std::string& m, std::size_t cap, const SHAOptions& o) {
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
  out.histogram.assign(161, 0);
  const auto baseline = sha1_run(m, o);
  ++out.hash_calls;
  out.compression_blocks += baseline.compression_blocks;
  out.baseline_hex = to_hex(baseline.digest);
  out.changed_bits.reserve(bits);
  std::string variant = m;
  long double sum     = 0;
  for (std::size_t bit = 0; bit < bits; ++bit) {
    const auto mask   = static_cast<unsigned char>(1U << (bit % 8));
    variant[bit / 8]  = static_cast<char>(static_cast<unsigned char>(variant[bit / 8]) ^ mask);
    const auto result = sha1_run(variant, o);
    ++out.hash_calls;
    out.compression_blocks += result.compression_blocks;
    std::size_t changes = 0;
    for (std::size_t j = 0; j < 20; ++j)
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
