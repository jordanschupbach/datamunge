#pragma once
/// @file hmac_analysis.hpp
/// @brief Complete bounded HMAC key-bit sensitivity with exact full-range counts.
#include <algorithm>
#include <bit>
#include <datamunge/algorithms/hmac.hpp>
#include <vector>
namespace datamunge::algorithms {
/// @brief Owned public RFC4231 HMAC-SHA-256 input and expected full/prefix hex.
/// expected_hex contains 64 characters except case 5's 32-character 128-bit prefix.
/// The full-tag API remains 32 bytes for every case. No references are retained.
/// @par Example
/// @code
/// const dm::algorithms::HmacReferenceCase c;std::print("{}\n",c.case_number); // 0
/// @endcode
/// @see hmac_rfc4231_cases()
struct HmacReferenceCase {
  std::size_t case_number{0}; ///< RFC case identifier1..7, zero for default construction.
  std::string key;            ///< Owned exact public key bytes.
  std::string message;        ///< Owned exact public message bytes.
  std::string expected_hex;   ///< Published lowercase full tag, or case5's leading128 bits.
};
/// @brief Return all seven public RFC4231 HMAC-SHA-256 fixtures in case order.
/// Binary bytes and exact ASCII messages follow sections 4.2..4.8. Case 5 publishes
/// only 128 bits; compare that prefix explicitly rather than claiming a full-tag
/// reference. Fixed bounded fixture storage; returned strings own their memory.
/// @return Owned seven-element array of inputs and published expected hex.
/// @throws std::bad_alloc On string allocation failure.
/// @par Example
/// @code
/// const auto cases=dm::algorithms::hmac_rfc4231_cases();
/// std::print("{} {}\n",cases.size(),cases[4].expected_hex.size()); // 7 32
/// @endcode
/// @par References
/// - M. Nystrom (2005), RFC4231, https://www.rfc-editor.org/rfc/rfc4231.html .
/// @see hmac_sha256_hex(), HmacReferenceCase
inline std::array<HmacReferenceCase, 7> hmac_rfc4231_cases() {
  std::string ascending;
  for (unsigned i = 1; i <= 25; ++i)
    ascending.push_back(static_cast<char>(i));
  return {
      {{1,
        std::string(20, '\x0b'),
        "Hi There",
        "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7"},
       {2,
        "Jefe",
        "what do ya want for nothing?",
        "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"},
       {3,
        std::string(20, '\xaa'),
        std::string(50, '\xdd'),
        "773ea91e36800e46854db8ebd09181a72959098b3ef8c122d9635514ced565fe"},
       {4,
        ascending,
        std::string(50, '\xcd'),
        "82558a389a443c0ea4cc819899f2083a85f0faa3e578f8077a2e3ff46729665b"},
       {5, std::string(20, '\x0c'), "Test With Truncation", "a3b6167473100ee06e0c796c2955552b"},
       {6,
        std::string(131, '\xaa'),
        "Test Using Larger Than Block-Size Key - Hash Key First",
        "60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54"},
       {7,
        std::string(131, '\xaa'),
        "This is a test using a larger than block-size key and a larger than block-size data. The "
        "key needs to be hashed before being used by the HMAC algorithm.",
        "9b09ffa71b942fcb27635fbcd5b0e944bfdc63644f0713938a7f51535c3a35e2"}}};
}
/// @brief Count differing bits across two complete 32-byte tags, from 0 through 256.
/// Fixed work over all bytes; no constant-time machine-code guarantee is claimed.
/// @param a First borrowed tag.
/// @param b Second borrowed tag.
/// @return Exact Hamming distance.
/// @par Example
/// @code
/// dm::algorithms::HmacTag a{},b{};b[0]=255;
/// std::print("{}\n",dm::algorithms::hmac_tag_distance(a,b)); // 8
/// @endcode
inline std::size_t hmac_tag_distance(const HmacTag& a, const HmacTag& b) {
  std::size_t result = 0;
  for (std::size_t i = 0; i < a.size(); ++i)
    result += std::popcount(static_cast<unsigned>(a[i] ^ b[i]));
  return result;
}
/// @brief Owned complete experiment, not a randomness or unforgeability certificate.
/// All supplied key bits are flipped once, least-significant bit first within
/// each byte. Histograms cover every possible distance without clipping.
/// @par Example
/// @code
/// const dm::algorithms::HmacKeyAvalanche r;std::print("{}\n",r.histogram.size()); // 257
/// @endcode
struct HmacKeyAvalanche {
  HmacTag                  baseline{}; ///< Actual original complete-key tag.
  std::vector<std::size_t> distances;  ///< Actual tag distance for every key bit in byte/LSB order.
  std::array<std::size_t, 257> histogram{}; ///< Exact frequency for each distance0..256.
  bool        has_flips{false}; ///< False for an empty key; summary zeros then are sentinels.
  std::size_t minimum{0};       ///< Actual smallest distance, or zero sentinel without flips.
  std::size_t maximum{0};       ///< Actual largest distance, or zero sentinel without flips.
  double      mean{0};          ///< Mean actual distance as double, or zero sentinel without flips.
};
/// @brief Flip each supplied key bit and summarize actual HMAC changes, with complete admission.
/// Input policy and full 8k flip count are checked before baseline hashing/copies.
/// A failed limit throws; there is no sampling, truncation or partial result.
/// Calls the shared HMAC kernel for one baseline and 8k changed keys. For k key
/// bytes and m message bytes, time O((8k+1)(k+m)) and additional working/output
/// storage O(k+m+8k), excluding borrowed input. Empty key still computes a baseline
/// but returns no flips and has_flips=false. Values describe only this fixture;
/// a mean near 128 does not prove PRF behavior or authentication security.
/// @param key Complete borrowed key bytes, including zeros/high bytes.
/// @param message Complete borrowed message bytes.
/// @param max_flips Complete key-bit cap, default 4096; zero disables policy.
/// @param options Complete combined input-byte policy, default 1m; zero disables.
/// @return Owned baseline, every actual distance and exact full-range summary.
/// @throws std::length_error On complete input/flip/container/count representation limits.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto r=dm::algorithms::hmac_key_avalanche("k","m");
/// std::print("{} {}\n",r.has_flips,r.distances.size()); // true 8
/// @endcode
/// @see hmac_sha256_trace(), hmac_tag_distance()
inline HmacKeyAvalanche hmac_key_avalanche(
    const std::string& key,
    const std::string& message,
    std::size_t        max_flips = 4096,
    HmacOptions        options   = {}) {
  hmac_detail::validate(key.size(), message.size(), options);
  if (key.size() > std::numeric_limits<std::size_t>::max() / 8)
    throw std::length_error("HMAC flip count");
  const auto count = key.size() * 8;
  if ((max_flips && count > max_flips) || count > std::vector<std::size_t>{}.max_size())
    throw std::length_error("HMAC complete flip cap");
  // Bound the exact distance sum before any expensive work.
  if (count > std::numeric_limits<std::size_t>::max() / 256)
    throw std::length_error("HMAC distance sum");
  HmacKeyAvalanche r;
  r.baseline = hmac_sha256(key, message);
  r.distances.reserve(count);
  std::size_t sum = 0;
  for (std::size_t bit = 0; bit < count; ++bit) {
    auto changed = key;
    changed[bit / 8] =
        static_cast<char>(static_cast<unsigned char>(changed[bit / 8]) ^ (1u << (bit % 8)));
    const auto d = hmac_tag_distance(r.baseline, hmac_sha256(changed, message));
    r.distances.push_back(d);
    ++r.histogram[d];
    sum += d;
  }
  if (count) {
    r.has_flips = true;
    r.minimum   = *std::min_element(r.distances.begin(), r.distances.end());
    r.maximum   = *std::max_element(r.distances.begin(), r.distances.end());
    r.mean      = double(sum) / double(count);
  }
  return r;
}
} // namespace datamunge::algorithms
