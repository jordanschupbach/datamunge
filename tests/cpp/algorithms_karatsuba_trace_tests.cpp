#include <gtest/gtest.h>

#include <datamunge/algorithms/karatsuba.hpp>
#include <limits>
using namespace datamunge::algorithms;
TEST(KaratsubaTrace, SmallExactProductsAndDecimalConventions) {
  for (unsigned a = 0; a < 100; ++a)
    for (unsigned b = 0; b < 100; ++b)
      EXPECT_EQ(karatsuba_multiply(std::to_string(a), std::to_string(b)), std::to_string(a * b));
  EXPECT_EQ(karatsuba_multiply("", "123"), "0");
  EXPECT_EQ(karatsuba_multiply("00012", "0034"), "408");
  for (const auto& s : {"-1", "+1", "1.2", "1 2", "A"})
    EXPECT_THROW((void)karatsuba_multiply(s, "2"), std::invalid_argument);
}
TEST(KaratsubaTrace, ClosedFormCarryAndPowerProductsAcrossCrossover) {
  for (std::size_t n : {1, 16, 31, 32, 33, 64, 65, 128, 256}) {
    const std::string nine(n, '9');
    const auto        t = karatsuba_trace(nine, nine);
    EXPECT_EQ(t.product, std::string(n - 1, '9') + "8" + std::string(n - 1, '0') + "1");
    EXPECT_EQ(t.product, karatsuba_multiply(nine, nine));
    EXPECT_EQ(
        karatsuba_multiply("1" + std::string(n, '0'), "1" + std::string(n + 1, '0')),
        "1" + std::string(2 * n + 1, '0'));
    std::size_t leaves = 0, products = 0, maxdepth = 0;
    ASSERT_EQ(t.work.recursive_calls, t.calls.size());
    for (std::size_t i = 0; i < t.calls.size(); ++i) {
      const auto& c = t.calls[i];
      maxdepth      = std::max(maxdepth, c.depth);
      EXPECT_EQ(c.leaf, c.left_digits <= 32 || c.right_digits <= 32);
      if (c.leaf) {
        ++leaves;
        products += c.left_digits * c.right_digits;
        EXPECT_EQ(c.split, 0U);
      } else {
        EXPECT_EQ(c.split, std::min(c.left_digits, c.right_digits) / 2);
        std::size_t children = 0;
        bool        roles[4] = {};
        for (std::size_t j = i + 1; j < t.calls.size(); ++j)
          if (t.calls[j].parent == i) {
            ++children;
            roles[int(t.calls[j].role)] = true;
            EXPECT_EQ(t.calls[j].depth, c.depth + 1);
          }
        EXPECT_EQ(children, 3U);
        EXPECT_TRUE(roles[1] && roles[2] && roles[3]);
      }
      if (i)
        EXPECT_LT(c.parent, i);
    }
    EXPECT_EQ(t.work.schoolbook_leaves, leaves);
    EXPECT_EQ(t.work.digit_products, products);
    EXPECT_EQ(t.work.max_depth, maxdepth);
  }
}
TEST(KaratsubaTrace, CompleteInputActualCallsDepthAndOwnedBytes) {
  EXPECT_NO_THROW((void)karatsuba_trace("12", "34", {4, 1, 1}));
  EXPECT_THROW((void)karatsuba_trace("12", "34", {3, 1, 1}), std::length_error);
  EXPECT_THROW(
      (void)karatsuba_trace(std::string(33, '9'), std::string(33, '9'), {0, 1, 0}),
      std::length_error);
  EXPECT_THROW(
      (void)karatsuba_trace(std::string(128, '9'), std::string(128, '9'), {0, 0, 1}),
      std::length_error);
  std::string a = "0012";
  const auto  t = karatsuba_trace(a, "34");
  a[0]          = '9';
  EXPECT_EQ(t.left, "0012");
  EXPECT_EQ(t.calls[0].left_digits, 2U);
}
