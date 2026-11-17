#include <gtest/gtest.h>

#include <datamunge/algorithms/hmac_analysis.hpp>
#include <numeric>
using namespace datamunge::algorithms;
TEST(HmacTrace, PadsDigestsBoundaries) {
  for (std::size_t k : {0, 1, 20, 63, 64, 65, 131})
    for (std::size_t m : {0, 1, 55, 56, 63, 64, 119, 120}) {
      const std::string key(k, '\xaa'), msg(m, '\xdd');
      const auto        t = hmac_sha256_trace(key, msg, {0});
      EXPECT_EQ(t.run.tag, hmac_sha256(key, msg));
      EXPECT_EQ(t.run.key_hashed, k > 64);
      std::string normalized = key;
      if (k > 64) {
        const auto d = sha256(key);
        EXPECT_EQ(t.key_digest, d);
        normalized.assign(reinterpret_cast<const char*>(d.data()), d.size());
      }
      normalized.resize(64, 0);
      std::string inner(64, 0), outer(64, 0);
      for (std::size_t i = 0; i < 64; ++i) {
        EXPECT_EQ(t.key_block[i], static_cast<unsigned char>(normalized[i]));
        EXPECT_EQ(t.inner_pad[i], t.key_block[i] ^ 0x36);
        EXPECT_EQ(t.outer_pad[i], t.key_block[i] ^ 0x5c);
        inner[i] = char(t.inner_pad[i]);
        outer[i] = char(t.outer_pad[i]);
      }
      const auto d = sha256(inner + msg);
      EXPECT_EQ(d, t.inner_digest);
      EXPECT_EQ(
          sha256(outer + std::string(reinterpret_cast<const char*>(d.data()), d.size())),
          t.run.tag);
      EXPECT_EQ(t.run.work.sha_calls, k > 64 ? 3 : 2);
      EXPECT_EQ(t.run.work.bytes_hashed, 160 + m + (k > 64 ? k : 0));
      const auto padded = [](std::size_t bytes) { return (bytes + 9 + 63) / 64; };
      EXPECT_EQ(t.run.work.padded_blocks, padded(64 + m) + padded(96) + (k > 64 ? padded(k) : 0));
    }
}
TEST(HmacTrace, ReferenceAndAdmission) {
  EXPECT_EQ(
      hmac_sha256_hex(std::string(20, '\x0b'), "Hi There"),
      "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7");
  EXPECT_EQ(
      hmac_sha256_hex("Jefe", "what do ya want for nothing?"),
      "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843");
  EXPECT_EQ(hmac_sha256_run("k", "m", {2}).tag, hmac_sha256("k", "m"));
  EXPECT_THROW(hmac_sha256_trace("k", "m", {1}), std::length_error);
  EXPECT_THROW(hmac_detail::blocks(std::numeric_limits<std::size_t>::max()), std::length_error);
  auto t         = hmac_sha256_trace("k", "m");
  auto c         = t;
  c.inner_pad[0] = 0;
  EXPECT_NE(t.inner_pad[0], c.inner_pad[0]);
}
TEST(HmacTrace, CompleteSensitivityAndBinaryInputs) {
  const std::string key("\0\xff", 2), msg("a\0b", 3);
  const auto        r = hmac_key_avalanche(key, msg, 16, {5});
  EXPECT_EQ(r.distances.size(), 16);
  EXPECT_TRUE(r.has_flips);
  EXPECT_EQ(std::accumulate(r.histogram.begin(), r.histogram.end(), std::size_t{0}), 16);
  std::array<std::size_t, 257> hist{};
  std::size_t                  sum = 0;
  for (std::size_t bit = 0; bit < 16; ++bit) {
    auto changed      = key;
    changed[bit / 8]  = char(static_cast<unsigned char>(changed[bit / 8]) ^ (1u << (bit % 8)));
    const auto  a     = hmac_sha256(changed, msg);
    std::size_t count = 0;
    for (std::size_t i = 0; i < 32; ++i)
      for (unsigned b = 0; b < 8; ++b)
        count += ((a[i] ^ r.baseline[i]) >> b) & 1;
    EXPECT_EQ(r.distances[bit], count);
    ++hist[count];
    sum += count;
  }
  EXPECT_EQ(r.histogram, hist);
  EXPECT_DOUBLE_EQ(r.mean, double(sum) / 16);
  EXPECT_THROW(hmac_key_avalanche(key, msg, 15), std::length_error);
  EXPECT_THROW(hmac_key_avalanche(key, msg, 16, {4}), std::length_error);
  auto empty = hmac_key_avalanche("", msg);
  EXPECT_FALSE(empty.has_flips);
  EXPECT_TRUE(empty.distances.empty());
  HmacTag a{}, b{};
  b.fill(255);
  EXPECT_EQ(hmac_tag_distance(a, b), 256);
  EXPECT_EQ(hmac_tag_distance(a, a), 0);
}

TEST(HmacTrace, AllPublishedFixtures) {
  const auto cases = hmac_rfc4231_cases();
  ASSERT_EQ(cases.size(), 7);
  for (const auto& c : cases) {
    const auto full = hmac_sha256_hex(c.key, c.message);
    EXPECT_EQ(full.size(), 64);
    EXPECT_EQ(full.substr(0, c.expected_hex.size()), c.expected_hex);
    EXPECT_EQ(c.expected_hex.size(), c.case_number == 5 ? 32 : 64);
    EXPECT_EQ(hmac_sha256_trace(c.key, c.message).run.key_hashed, c.case_number >= 6);
  }
}
