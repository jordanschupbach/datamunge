// std::format / std::print support for datamunge::autodiff types.
#include <dm/dm.hpp>

#include <gtest/gtest.h>

#include <format>
#include <string>

static_assert(std::formattable<dm::autodiff::Dual, char>);
static_assert(std::formattable<dm::autodiff::HyperDual, char>);
static_assert(std::formattable<dm::autodiff::Tape, char>);
static_assert(std::formattable<dm::autodiff::Var, char>);

TEST(AutodiffFormat, DualPrintsValueAndDerivative) {
    EXPECT_EQ(std::format("{}", dm::autodiff::Dual(2.0, 1.0)), "Dual(value=2, derivative=1)");
    EXPECT_EQ(std::format("{:>30}", dm::autodiff::Dual(1.5)), " Dual(value=1.5, derivative=0)");
}

TEST(AutodiffFormat, HyperDualPrintsAllParts) {
    EXPECT_EQ(std::format("{}", dm::autodiff::HyperDual(1.0, 0.5, 0.25, 2.0)),
              "HyperDual(value=1, eps1=0.5, eps2=0.25, eps1eps2=2)");
}

TEST(AutodiffFormat, TapeAndVar) {
    dm::autodiff::Tape tape;
    const dm::autodiff::Var x(tape, 3.0);
    const dm::autodiff::Var y = x * 2.0;
    EXPECT_EQ(std::format("{}", tape), "Tape(nodes=2)");
    EXPECT_EQ(std::format("{}", y), "Var(value=6, index=1)");
}
