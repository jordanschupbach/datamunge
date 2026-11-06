/// @file tex_format_tests.cpp
/// @brief Raw TeX std::format support and standard string format specifications.
#include <gtest/gtest.h>

#include <datamunge/format/tex.hpp>

TEST(TexFormat, UsesRawSourceAndSupportsStringSpecifications) {
  const datamunge::Tex x("x+y");
  EXPECT_EQ(std::format("{}", x), "x+y");
  EXPECT_EQ(std::format("{:>6}", x), "   x+y");
  EXPECT_EQ(std::format("{:.1}", x), "x");
  EXPECT_EQ(std::format("{}", x.underbrace("s")), R"(\underbrace{x+y}_{s})");
  EXPECT_EQ(std::format("{}", x.underset("s")), R"(\underset{s}{x+y})");
  EXPECT_EQ(std::format("{}", datamunge::Tex()), "");
}
