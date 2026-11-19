// Copyright (c) 2014 Google, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//
// FarmHash, by Geoff Pike

#pragma once

/// @file farmhash_fingerprint.hpp
/// @brief Portable farmhashna fingerprint path from Google FarmHash (0d859a8).
#include <bit>
#include <cstddef>
#include <cstdint>
#include <utility>

/// @brief Internal stable fingerprint implementation from Google FarmHash.
namespace datamunge::nn::farmhash_detail {

/// @brief Size of a byte sequence.
using std::size_t;
/// @brief Unsigned 8-bit byte.
using std::uint8_t;
/// @brief Unsigned 32-bit mixing word.
using std::uint32_t;
/// @brief Unsigned 64-bit mixing word.
using std::uint64_t;

/// @brief First portable FarmHash mixing constant.
inline constexpr uint64_t k0 = 0xc3a5c85c97cb3127ULL;
/// @brief Second portable FarmHash mixing constant.
inline constexpr uint64_t k1 = 0xb492b66fbe98f273ULL;
/// @brief Third portable FarmHash mixing constant.
inline constexpr uint64_t k2 = 0x9ae16a3b2f90404fULL;

/// @brief Load an unaligned word in little-endian byte order.
template<class Word>
inline Word Fetch(const char* p) {
  Word word = 0;
  for (size_t i = 0; i < sizeof(Word); ++i)
    word |= Word(static_cast<unsigned char>(p[i])) << (8*i);
  return word;
}

/// @brief Mix high bits into the low bits.
inline uint64_t ShiftMix(uint64_t val) {
  return val ^ (val >> 47);
}

/// @brief Mix two words using the supplied multiplier.
inline uint64_t HashLen16(uint64_t u, uint64_t v, uint64_t mul) {
  // Murmur-inspired hashing.
  uint64_t a = (u ^ v) * mul;
  a ^= (a >> 47);
  uint64_t b = (v ^ a) * mul;
  b ^= (b >> 47);
  b *= mul;
  return b;
}

/// @brief Fingerprint an input of at most 16 bytes.
inline uint64_t HashLen0to16(const char *s, size_t len) {
  if (len >= 8) {
    uint64_t mul = k2 + len * 2;
    uint64_t a = Fetch<uint64_t>(s) + k2;
    uint64_t b = Fetch<uint64_t>(s + len - 8);
    uint64_t c = std::rotr(b, 37) * mul + a;
    uint64_t d = (std::rotr(a, 25) + b) * mul;
    return HashLen16(c, d, mul);
  }
  if (len >= 4) {
    uint64_t mul = k2 + len * 2;
    uint64_t a = Fetch<uint32_t>(s);
    return HashLen16(len + (a << 3), Fetch<uint32_t>(s + len - 4), mul);
  }
  if (len > 0) {
    uint8_t a = s[0];
    uint8_t b = s[len >> 1];
    uint8_t c = s[len - 1];
    uint32_t y = static_cast<uint32_t>(a) + (static_cast<uint32_t>(b) << 8);
    uint32_t z = len + (static_cast<uint32_t>(c) << 2);
    return ShiftMix(y * k2 ^ z * k0) * k2;
  }
  return k2;
}

// This probably works well for 16-byte strings as well, but it may be overkill
// in that case.
/// @brief Fingerprint an input of 17 to 32 bytes.
inline uint64_t HashLen17to32(const char *s, size_t len) {
  uint64_t mul = k2 + len * 2;
  uint64_t a = Fetch<uint64_t>(s) * k1;
  uint64_t b = Fetch<uint64_t>(s + 8);
  uint64_t c = Fetch<uint64_t>(s + len - 8) * mul;
  uint64_t d = Fetch<uint64_t>(s + len - 16) * k2;
  return HashLen16(std::rotr(a + b, 43) + std::rotr(c, 30) + d,
                   a + std::rotr(b + k2, 18) + c, mul);
}

// Return a 16-byte hash for 48 bytes.  Quick and dirty.
// Callers do best to use "random-looking" values for a and b.
/// @brief Mix a 32-byte block and two seed words into two state words.
inline std::pair<uint64_t, uint64_t> WeakHashLen32WithSeeds(
    uint64_t w, uint64_t x, uint64_t y, uint64_t z, uint64_t a, uint64_t b) {
  a += w;
  b = std::rotr(b + a + z, 21);
  uint64_t c = a;
  a += x;
  a += y;
  b += std::rotr(a, 44);
  return std::make_pair(a + z, b + c);
}

// Return a 16-byte hash for s[0] ... s[31], a, and b.  Quick and dirty.
/// @brief Mix a 32-byte block and two seed words into two state words.
inline std::pair<uint64_t, uint64_t> WeakHashLen32WithSeeds(
    const char* s, uint64_t a, uint64_t b) {
  return WeakHashLen32WithSeeds(Fetch<uint64_t>(s),
                                Fetch<uint64_t>(s + 8),
                                Fetch<uint64_t>(s + 16),
                                Fetch<uint64_t>(s + 24),
                                a,
                                b);
}

// Return an 8-byte hash for 33 to 64 bytes.
/// @brief Fingerprint an input of 33 to 64 bytes.
inline uint64_t HashLen33to64(const char *s, size_t len) {
  uint64_t mul = k2 + len * 2;
  uint64_t a = Fetch<uint64_t>(s) * k2;
  uint64_t b = Fetch<uint64_t>(s + 8);
  uint64_t c = Fetch<uint64_t>(s + len - 8) * mul;
  uint64_t d = Fetch<uint64_t>(s + len - 16) * k2;
  uint64_t y = std::rotr(a + b, 43) + std::rotr(c, 30) + d;
  uint64_t z = HashLen16(y, a + std::rotr(b + k2, 18) + c, mul);
  uint64_t e = Fetch<uint64_t>(s + 16) * mul;
  uint64_t f = Fetch<uint64_t>(s + 24);
  uint64_t g = (y + Fetch<uint64_t>(s + len - 32)) * mul;
  uint64_t h = (z + Fetch<uint64_t>(s + len - 24)) * mul;
  return HashLen16(std::rotr(e + f, 43) + std::rotr(g, 30) + h,
                   e + std::rotr(f + a, 18) + g, mul);
}

/// @brief Compute the portable FarmHash fingerprint for any input length.
inline uint64_t Hash64(const char *s, size_t len) {
  const uint64_t seed = 81;
  if (len <= 32) {
    if (len <= 16) {
      return HashLen0to16(s, len);
    } else {
      return HashLen17to32(s, len);
    }
  } else if (len <= 64) {
    return HashLen33to64(s, len);
  }

  // For strings over 64 bytes we loop.  Internal state consists of
  // 56 bytes: v, w, x, y, and z.
  uint64_t x = seed;
  uint64_t y = seed * k1 + 113;
  uint64_t z = ShiftMix(y * k2 + 113) * k2;
  std::pair<uint64_t, uint64_t> v = std::make_pair(0, 0);
  std::pair<uint64_t, uint64_t> w = std::make_pair(0, 0);
  x = x * k2 + Fetch<uint64_t>(s);

  // Set end so that after the loop we have 1 to 64 bytes left to process.
  const char* end = s + ((len - 1) / 64) * 64;
  const char* last64 = end + ((len - 1) & 63) - 63;
  do {
    x = std::rotr(x + y + v.first + Fetch<uint64_t>(s + 8), 37) * k1;
    y = std::rotr(y + v.second + Fetch<uint64_t>(s + 48), 42) * k1;
    x ^= w.second;
    y += v.first + Fetch<uint64_t>(s + 40);
    z = std::rotr(z + w.first, 33) * k1;
    v = WeakHashLen32WithSeeds(s, v.second * k1, x + w.first);
    w = WeakHashLen32WithSeeds(s + 32, z + w.second, y + Fetch<uint64_t>(s + 16));
    std::swap(z, x);
    s += 64;
  } while (s != end);
  uint64_t mul = k1 + ((z & 0xff) << 1);
  // Make s point to the last 64 bytes of input.
  s = last64;
  w.first += ((len - 1) & 63);
  v.first += w.first;
  w.first += v.first;
  x = std::rotr(x + y + v.first + Fetch<uint64_t>(s + 8), 37) * mul;
  y = std::rotr(y + v.second + Fetch<uint64_t>(s + 48), 42) * mul;
  x ^= w.second * 9;
  y += v.first * 9 + Fetch<uint64_t>(s + 40);
  z = std::rotr(z + w.first, 33) * mul;
  v = WeakHashLen32WithSeeds(s, v.second * mul, x + w.first);
  w = WeakHashLen32WithSeeds(s + 32, z + w.second, y + Fetch<uint64_t>(s + 16));
  std::swap(z, x);
  return HashLen16(HashLen16(v.first, w.first, mul) + ShiftMix(y) * k0 + z,
                   HashLen16(v.second, w.second, mul) + x,
                   mul);
}

}  // namespace datamunge::nn::farmhash_detail
