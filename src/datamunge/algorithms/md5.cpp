/// @file md5.cpp
/// @brief Canonical md5 API, producing histories and checked native contracts.

#include <algorithm>
#include <bit>
#include <datamunge/algorithms/hex_encoding.hpp>
#include <datamunge/algorithms/md5.hpp>
#include <limits>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Implementation-only helpers for canonical md5 production and capture.
namespace {
/// Per-step left-rotation amounts s[0..63].
inline constexpr int md5_s[64] = {7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
                                  5, 9,  14, 20, 5, 9,  14, 20, 5, 9,  14, 20, 5, 9,  14, 20,
                                  4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
                                  6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21};

/// Per-step additive constants K[i] = floor(2^32 * |sin(i+1)|).
inline constexpr std::uint32_t md5_k[64] = {
    0xd76aa478u, 0xe8c7b756u, 0x242070dbu, 0xc1bdceeeu, 0xf57c0fafu, 0x4787c62au, 0xa8304613u,
    0xfd469501u, 0x698098d8u, 0x8b44f7afu, 0xffff5bb1u, 0x895cd7beu, 0x6b901122u, 0xfd987193u,
    0xa679438eu, 0x49b40821u, 0xf61e2562u, 0xc040b340u, 0x265e5a51u, 0xe9b6c7aau, 0xd62f105du,
    0x02441453u, 0xd8a1e681u, 0xe7d3fbc8u, 0x21e1cde6u, 0xc33707d6u, 0xf4d50d87u, 0x455a14edu,
    0xa9e3e905u, 0xfcefa3f8u, 0x676f02d9u, 0x8d2a4c8au, 0xfffa3942u, 0x8771f681u, 0x6d9d6122u,
    0xfde5380cu, 0xa4beea44u, 0x4bdecfa9u, 0xf6bb4b60u, 0xbebfbc70u, 0x289b7ec6u, 0xeaa127fau,
    0xd4ef3085u, 0x04881d05u, 0xd9d4d039u, 0xe6db99e5u, 0x1fa27cf8u, 0xc4ac5665u, 0xf4292244u,
    0x432aff97u, 0xab9423a7u, 0xfc93a039u, 0x655b59c3u, 0x8f0ccc92u, 0xffeff47du, 0x85845dd1u,
    0x6fa87e4fu, 0xfe2ce6e0u, 0xa3014314u, 0x4e0811a1u, 0xf7537e82u, 0xbd3af235u, 0x2ad7d2bbu,
    0xeb86d391u};

/// @brief Checked logical byte/record count addition.
/// @param a Existing count.
/// @param b Additional count.
/// @return Exact sum.
/// @throws std::length_error On size_t overflow.
std::size_t add(std::size_t a, std::size_t b) {
  if (b > SIZE_MAX - a)
    throw std::length_error("MD5 geometry");
  return a + b;
}
/// @brief Canonical RFC1321 padding/compression producer and optional actual capture.
/// @param message Original arbitrary bytes.
/// @param trace Optional complete owned history.
/// @param limit Optional complete logical cap, zero off.
/// @return Actual digest and selected work.
/// @throws std::length_error Before copying if padded or complete capture geometry exceeds limits.
MD5Analysis produce(const std::string& message, MD5Trace* trace, std::size_t limit) {
  auto padding = 9 + (119 - message.size() % 64) % 64;
  auto total   = add(message.size(), padding);
  if (total > message.max_size())
    throw std::length_error("MD5 padded string geometry");
  auto blocks = total / 64;
  if (trace) {
    auto count = add(add(add(message.size(), total), total), add(blocks, 16));
    if (limit && count > limit)
      throw std::length_error("MD5 complete capture quota");
    trace->captured_entries = count;
    trace->input            = message;
    trace->steps.reserve(total);
    trace->blocks.reserve(blocks);
  }
  MD5Analysis out;
  out.input_bytes        = message.size();
  out.padded_bytes       = total;
  out.encoded_bit_length = static_cast<std::uint64_t>(message.size()) * 8u;
  std::string msg        = message;
  msg.reserve(total);
  msg.push_back(static_cast<char>(0x80));
  while (msg.size() % 64 != 56)
    msg.push_back(0);
  for (unsigned i = 0; i < 8; ++i)
    msg.push_back(static_cast<char>((out.encoded_bit_length >> (8 * i)) & 255));
  if (trace)
    trace->padded = msg;
  std::array<std::uint32_t, 4> chain{0x67452301u, 0xefcdab89u, 0x98badcfeu, 0x10325476u};
  for (std::size_t off = 0; off < total; off += 64) {
    std::array<std::uint32_t, 16> words{};
    for (std::size_t i = 0; i < 16; ++i)
      for (unsigned byte = 0; byte < 4; ++byte)
        words[i] |= std::uint32_t(static_cast<unsigned char>(msg[off + 4 * i + byte]))
                    << (8 * byte);
    auto before = chain;
    auto state  = chain;
    for (std::size_t i = 0; i < 64; ++i) {
      auto          old = state;
      std::uint32_t function;
      std::size_t   g;
      if (i < 16) {
        function = (state[1] & state[2]) | (~state[1] & state[3]);
        g        = i;
      } else if (i < 32) {
        function = (state[3] & state[1]) | (~state[3] & state[2]);
        g        = (5 * i + 1) % 16;
      } else if (i < 48) {
        function = state[1] ^ state[2] ^ state[3];
        g        = (3 * i + 5) % 16;
      } else {
        function = state[2] ^ (state[1] | ~state[3]);
        g        = (7 * i) % 16;
      }
      std::uint32_t sum     = function + state[0] + md5_k[i] + words[g];
      unsigned      shift   = md5_s[i];
      auto          rotated = std::rotl(sum, static_cast<int>(shift));
      state                 = {state[3], state[1] + rotated, state[1], state[2]};
      if (trace)
        trace->steps.push_back(
            {off / 64, i, old, function, g, words[g], md5_k[i], shift, sum, rotated, state});
      ++out.steps;
    }
    for (std::size_t w = 0; w < 4; ++w)
      chain[w] += state[w];
    if (trace)
      trace->blocks.push_back({off, words, before, state, chain});
    ++out.blocks;
  }
  for (std::size_t w = 0; w < 4; ++w)
    for (unsigned byte = 0; byte < 4; ++byte)
      out.digest[4 * w + byte] = static_cast<std::uint8_t>((chain[w] >> (8 * byte)) & 255);
  return out;
}
} // namespace
MD5Digest md5(const std::string& message) {
  return produce(message, nullptr, 0).digest;
}
std::string md5_hex(const std::string& message) {
  return to_hex(md5(message));
}
MD5Analysis md5_analyze(const std::string& message) {
  return produce(message, nullptr, 0);
}
MD5Trace md5_trace(const std::string& message, std::size_t limit) {
  MD5Trace out;
  out.analysis = produce(message, &out, limit);
  return out;
}
unsigned md5_hamming(const MD5Digest& a, const MD5Digest& b) {
  unsigned count = 0;
  for (std::size_t i = 0; i < 16; ++i)
    count += std::popcount(static_cast<unsigned>(a[i] ^ b[i]));
  return count;
}
unsigned md5_state_hamming(
    const std::array<std::uint32_t, 4>& a,
    const std::array<std::uint32_t, 4>& b) {
  unsigned count = 0;
  for (std::size_t i = 0; i < 4; ++i)
    count += std::popcount(a[i] ^ b[i]);
  return count;
}
MD5Avalanche md5_avalanche(const std::string& message, std::size_t limit) {
  if (message.size() > SIZE_MAX / 8)
    throw std::length_error("MD5 flip geometry");
  auto n = message.size() * 8;
  if (limit && n > limit)
    throw std::length_error("MD5 complete flip cap");
  MD5Avalanche out;
  out.baseline = md5(message);
  out.flips.reserve(n);
  long double sum     = 0;
  out.minimum_changed = n ? 128 : 0;
  for (std::size_t bit = 0; bit < n; ++bit) {
    auto changed = message;
    changed[bit / 8] =
        static_cast<char>(static_cast<unsigned char>(changed[bit / 8]) ^ (1u << (bit % 8)));
    unsigned distance = md5_hamming(out.baseline, md5(changed));
    out.flips.push_back({bit, distance});
    sum += distance;
    out.minimum_changed = std::min(out.minimum_changed, distance);
    out.maximum_changed = std::max(out.maximum_changed, distance);
  }
  out.mean_changed = n ? double(sum / n) : 0;
  return out;
}
std::string md5_demo_message() {
  return "avalanche";
}
std::vector<std::pair<std::string, std::string>> md5_reference_vectors() {
  return {
      {"", "d41d8cd98f00b204e9800998ecf8427e"},
      {"a", "0cc175b9c0f1b6a831c399e269772661"},
      {"abc", "900150983cd24fb0d6963f7d28e17f72"},
      {"message digest", "f96b697d7cb7938d525a2f31aaf161d0"},
      {"abcdefghijklmnopqrstuvwxyz", "c3fcd3d76192e4007dfb496cca67e13b"},
      {"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
       "d174ab98d277d9f5a5611c2c9f419d9f"},
      {"12345678901234567890123456789012345678901234567890123456789012345678901234567890",
       "57edf4a22be3c955ac49da2e2107b67a"}};
}
} // namespace datamunge::algorithms
