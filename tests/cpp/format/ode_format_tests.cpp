// std::format / std::print support for datamunge::ode types.
#include <dm/dm.hpp>

#include <gtest/gtest.h>

#include <format>
#include <string>

static_assert(std::formattable<dm::ode::StepMethod, char>);
static_assert(std::formattable<dm::ode::ODEOptions, char>);
static_assert(std::formattable<dm::ode::ODESolution, char>);
static_assert(std::formattable<dm::ode::ODESolver, char>);

TEST(OdeFormat, StepMethodPrintsEnumeratorName) {
    EXPECT_EQ(std::format("{}", dm::ode::StepMethod::RK45), "RK45");
    EXPECT_EQ(std::format("{}", dm::ode::StepMethod::AdamsMoulton), "AdamsMoulton");
    EXPECT_EQ(std::format("{}", static_cast<dm::ode::StepMethod>(42)), "StepMethod(42)");
}

TEST(OdeFormat, OptionsPrintAllFields) {
    EXPECT_EQ(std::format("{}", dm::ode::ODEOptions{}),
              "ODEOptions(method=RK4, multistep_order=4, step_size=0.01, abs_tol=1e-06, rel_tol=1e-06, max_step=0)");
}

TEST(OdeFormat, SolutionAndSolver) {
    const dm::ode::ODESolver solver(dm::ode::ODEOptions{.step_size = 0.25});
    const auto sol = solver.solve_builtin("exponential_decay", {1.0}, {1.0}, 0.0, 1.0);
    const std::string text = std::format("{}", sol);
    std::printf("%s\n", text.c_str());
    EXPECT_EQ(text.find("ODESolution(points=5, t=[0, 1], steps_taken=4"), 0u);
    EXPECT_NE(text.find("y_end=[0.36"), std::string::npos);
    EXPECT_EQ(std::format("{}", dm::ode::ODESolution{}), "ODESolution(points=0)");
    EXPECT_EQ(std::format("{}", solver),
              "ODESolver(method=RK4, multistep_order=4, step_size=0.25, abs_tol=1e-06, rel_tol=1e-06, max_step=0)");
}
