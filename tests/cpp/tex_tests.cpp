/// @file tex_tests.cpp
/// @brief TeX value ownership, composition, escaping and byte interoperability.
#include <gtest/gtest.h>

#include <algorithm>
#include <datamunge/tex.hpp>
#include <sstream>
#include <string>

using datamunge::Tex;

TEST(Tex, OwnsRawBytesAndCopiesIndependently) {
  std::string raw = "x+y";
  const Tex   x(raw);
  raw.assign("changed");
  auto copy = x;
  copy      = Tex("z");
  EXPECT_EQ(x, "x+y");
  EXPECT_EQ(copy, "z");
  const Tex binary(std::string_view("x\0y", 3));
  EXPECT_EQ(binary.size(), 3);
  EXPECT_EQ(std::string(binary), std::string("x\0y", 3));
  EXPECT_EQ(binary.find(std::string_view("\0y", 2)), 1);
  EXPECT_TRUE(Tex().empty());
  EXPECT_EQ(Tex().display(), "$$\n\n$$");
  EXPECT_EQ(Tex().inline_math(), "$$");
}

TEST(Tex, DistinguishesBracesFromDirectAnnotations) {
  const Tex x("x+y");
  EXPECT_EQ(x.underset("i"), R"(\underset{i}{x+y})");
  EXPECT_EQ(x.overset("n"), R"(\overset{n}{x+y})");
  EXPECT_EQ(x.underbrace("i"), R"(\underbrace{x+y}_{i})");
  EXPECT_EQ(x.overbrace("n"), R"(\overbrace{x+y}^{n})");
  EXPECT_EQ(x.underbrace(""), R"(\underbrace{x+y}_{})");
  EXPECT_EQ(x.underset(R"(X^{T})"), R"(\underset{X^{T}}{x+y})");
  EXPECT_EQ(x.underbrace(R"(X^{T})"), R"(\underbrace{x+y}_{X^{T}})");
  EXPECT_EQ(x, "x+y");
}

TEST(Tex, GroupsWholeFragmentsAndComposesWithoutMutation) {
  const Tex numerator("a+b");
  EXPECT_EQ(numerator.frac("c+d"), R"(\frac{a+b}{c+d})");
  EXPECT_EQ(numerator.frac(Tex("c+d").sqrt()), R"(\frac{a+b}{\sqrt{c+d}})");
  EXPECT_EQ(numerator.sqrt(), R"(\sqrt{a+b})");
  EXPECT_EQ(numerator.subscript("i").superscript("2"), R"({{a+b}_{i}}^{2})");
  EXPECT_EQ(numerator.delimit().superscript("-1"), R"({\left( a+b \right)}^{-1})");
  EXPECT_EQ(numerator.delimit(R"(\langle)", R"(\rangle)"), R"(\left\langle a+b \right\rangle)");
  EXPECT_EQ(numerator.delimit(".", "|"), R"(\left. a+b \right|)");
  const auto equation = "z = " + numerator.underbrace("s") + " + " + Tex("c");
  EXPECT_EQ(equation, R"(z = \underbrace{a+b}_{s} + c)");
  EXPECT_EQ(equation.display(), "$$\nz = \\underbrace{a+b}_{s} + c\n$$");
  EXPECT_EQ(equation.inline_math(), "$z = \\underbrace{a+b}_{s} + c$");
  EXPECT_EQ(numerator, "a+b");
}

TEST(Tex, EscapesLiteralTextWhilePreservingMathSource) {
  EXPECT_EQ(
      Tex::text(R"(\{}$&#%_^~)"),
      R"(\text{\textbackslash{}\{\}\$\&\#\%\_\textasciicircum{}\textasciitilde{}})");
  EXPECT_EQ(Tex::text("a\nb\tc\rd"), R"(\text{a b c d})");
  EXPECT_EQ(Tex::text("café"), "\\text{café}");
  EXPECT_EQ(Tex::text(""), R"(\text{})");
  EXPECT_EQ(
      Tex("x").underbrace(Tex::text("fit_1 (95%)")), R"(\underbrace{x}_{\text{fit\_1 (95\%)}})");
  EXPECT_EQ(Tex(R"(\beta_{i})"), R"(\beta_{i})");
}

TEST(Tex, InteroperatesWithStringsStreamsAndRawByteIteration) {
  const Tex         x("x+y");
  const std::string raw = x;
  EXPECT_EQ(raw, x);
  EXPECT_EQ(x, raw);
  EXPECT_EQ("x+y", x);
  EXPECT_EQ(x, Tex("x+y"));
  EXPECT_FALSE(x == Tex("y+x"));
  EXPECT_EQ(x.find("z"), std::string::npos);
  EXPECT_EQ(x.find("", x.size()), x.size());
  EXPECT_EQ(std::count(x.begin(), x.end(), '+'), 1);
  EXPECT_STREQ(x.c_str(), "x+y");
  std::ostringstream stream;
  stream << x;
  EXPECT_EQ(stream.str(), raw);
}
