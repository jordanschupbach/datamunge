/// @file siphash.cpp
/// @brief One original-reference ARX producer with actual round capture.
#include <bit>
#include <datamunge/algorithms/siphash.hpp>
#include <limits>
#include <stdexcept>
/// @brief Canonical checked native producers and owned algorithm values.
namespace datamunge::algorithms {
/// @brief Private invocation-local checked helpers.
namespace {
/// @brief Load a little-endian word.
/// @param p Eight readable bytes.
/// @return Actual decoded word.
std::uint64_t load(const std::uint8_t* p) {
  std::uint64_t v = 0;
  for (int i = 0; i < 8; ++i)
    v |= std::uint64_t(p[i]) << (8 * i);
  return v;
}
/// @brief Execute the canonical SipRound in place.
/// @param v Four mutable words.
void round(std::array<std::uint64_t, 4>& v) {
  auto& v0 = v[0];
  auto& v1 = v[1];
  auto& v2 = v[2];
  auto& v3 = v[3];
  v0 += v1;
  v1 = std::rotl(v1, 13);
  v1 ^= v0;
  v0 = std::rotl(v0, 32);
  v2 += v3;
  v3 = std::rotl(v3, 16);
  v3 ^= v2;
  v0 += v3;
  v3 = std::rotl(v3, 21);
  v3 ^= v0;
  v2 += v1;
  v1 = std::rotl(v1, 17);
  v1 ^= v2;
  v2 = std::rotl(v2, 32);
}
/// @brief Hash once with actual optional rounds.
/// @param key Key.
/// @param message Raw bytes.
/// @param o Policies.
/// @param capture Retain round states.
/// @return Actual hash and work.
SipHashRun run(
    const std::array<std::uint8_t, 16>& key,
    const std::string&                  message,
    const SipHashOptions&               o,
    bool                                capture) {
  const auto len = message.size();
  if (o.max_bytes && len > o.max_bytes)
    throw std::length_error("siphash: bytes");
  const auto blocks = len / 8 + 1;
  if (blocks > (std::numeric_limits<std::size_t>::max() - 4) / 2)
    throw std::length_error("siphash: rounds capacity");
  const auto rounds = 2 * blocks + 4;
  SipHashRun r;
  if (capture &&
      ((o.max_round_records && rounds > o.max_round_records) || rounds > r.history.max_size()))
    throw std::length_error("siphash: history");
  const auto                   k0 = load(key.data()), k1 = load(key.data() + 8);
  std::array<std::uint64_t, 4> v{
      0x736f6d6570736575ULL ^ k0,
      0x646f72616e646f6dULL ^ k1,
      0x6c7967656e657261ULL ^ k0,
      0x7465646279746573ULL ^ k1};
  auto stir = [&](std::size_t block, bool final, std::size_t index) {
    const auto before = v;
    round(v);
    ++r.rounds;
    if (capture)
      r.history.push_back({block, final, index, before, v});
  };
  const auto* data = reinterpret_cast<const std::uint8_t*>(message.data());
  for (std::size_t block = 0; block < blocks; ++block) {
    std::uint64_t m = 0;
    if (block + 1 < blocks)
      m = load(data + block * 8);
    else {
      m = static_cast<std::uint64_t>(len) << 56;
      for (std::size_t i = 0; i < len % 8; ++i)
        m |= std::uint64_t(data[block * 8 + i]) << (8 * i);
    }
    v[3] ^= m;
    stir(block, false, 0);
    stir(block, false, 1);
    v[0] ^= m;
    ++r.blocks;
  }
  v[2] ^= 0xff;
  for (std::size_t i = 0; i < 4; ++i)
    stir(blocks - 1, true, i);
  r.hash = v[0] ^ v[1] ^ v[2] ^ v[3];
  return r;
}
} // namespace
SipHashRun siphash24_run(
    const std::array<std::uint8_t, 16>& k,
    const std::string&                  m,
    const SipHashOptions&               o) {
  return run(k, m, o, false);
}
SipHashRun siphash24_trace(
    const std::array<std::uint8_t, 16>& k,
    const std::string&                  m,
    const SipHashOptions&               o) {
  return run(k, m, o, true);
}
std::uint64_t siphash24(const std::array<std::uint8_t, 16>& k, const std::string& m) {
  SipHashOptions o;
  o.max_bytes = 0;
  return siphash24_run(k, m, o).hash;
}
std::array<std::uint8_t, 16> siphash_example_key() {
  std::array<std::uint8_t, 16> k{};
  for (std::size_t i = 0; i < 16; ++i)
    k[i] = static_cast<std::uint8_t>(i);
  return k;
}
std::string siphash_example_message(std::size_t n) {
  if (n > 1000000)
    throw std::length_error("siphash: fixture bytes");
  std::string s(n, '\0');
  for (std::size_t i = 0; i < n; ++i)
    s[i] = static_cast<char>(i % 256);
  return s;
}
std::vector<std::string> siphash_example_words() {
  return {"the",  "quick",   "brown",  "fox",     "jumps",  "over",  "lazy",      "dog",
          "hash", "table",   "flood",  "attack",  "bucket", "chain", "collision", "secret",
          "key",  "siphash", "denial", "service", "alpha",  "beta",  "gamma",     "delta",
          "one",  "two",     "three",  "four",    "five",   "six",   "seven",     "eight"};
}
std::vector<std::size_t> siphash_bucket_counts(
    const std::array<std::uint8_t, 16>& key,
    const std::vector<std::string>&     messages,
    std::size_t                         buckets) {
  if (!buckets)
    throw std::invalid_argument("siphash: buckets");
  if (buckets > 1000000 || messages.size() > 1000000)
    throw std::length_error("siphash: bucket fixture count");
  std::vector<std::size_t> counts(buckets);
  for (const auto& m : messages)
    ++counts[siphash24(key, m) % buckets];
  return counts;
}
} // namespace datamunge::algorithms
