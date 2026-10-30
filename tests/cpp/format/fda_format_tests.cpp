// std::format / std::print support for datamunge::fda types.
#include <dm/dm.hpp>

#include <gtest/gtest.h>

#include <format>
#include <string>
#include <vector>

static_assert(std::formattable<dm::fda::LocalBSplineEvaluation, char>);
static_assert(std::formattable<dm::fda::BSplineBasis, char>);
static_assert(std::formattable<dm::fda::BSpline, char>);
static_assert(std::formattable<dm::fda::PSplineOptions, char>);
static_assert(std::formattable<dm::fda::PSplineFitResult, char>);

TEST(FdaFormat, BasisAndLocalEvaluation) {
    const auto basis = dm::fda::BSplineBasis::open_uniform(1, 3);
    EXPECT_EQ(std::format("{}", basis), "BSplineBasis(degree=1, basis_size=3, domain=[0, 1], knots=[0, 0, 0.5, 1, 1])");
    const dm::fda::LocalBSplineEvaluation e{{0, 1}, {0.5, 0.5}};
    EXPECT_EQ(std::format("{}", e), "LocalBSplineEvaluation(indices=[0, 1], values=[0.5, 0.5])");
    std::printf("%s\n", std::format("{}", dm::fda::BSplineBasis::open_uniform(3, 6)).c_str());
}

TEST(FdaFormat, TensorBSpline) {
    const auto s = dm::fda::BSpline::open_uniform({1, 2}, {3, 4}, {0.0, 0.0}, {1.0, 1.0});
    EXPECT_EQ(std::format("{}", s), "BSpline(dimensions=2, basis_size=12, degrees=[1, 2], sizes=[3, 4])");
}

TEST(FdaFormat, OptionsAndFit) {
    EXPECT_EQ(std::format("{}", dm::fda::PSplineOptions{}), "PSplineOptions(lambda=1, penalty_order=2)");
    const std::vector<double> x{0.0, 0.25, 0.5, 0.75, 1.0};
    const std::vector<double> y{0.0, 0.3, 0.4, 0.8, 1.0};
    const auto fit = dm::fda::fit_pspline(x, y, 1, 3, {});
    const std::string text = std::format("{}", fit);
    std::printf("%s\n", text.c_str());
    EXPECT_EQ(text.find("PSplineFitResult(n=5, lambda=1, penalty_order=2)\n  basis: BSplineBasis(degree=1"), 0u);
    EXPECT_NE(text.find("edf="), std::string::npos);
}
