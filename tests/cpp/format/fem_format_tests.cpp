// std::format / std::print support for datamunge::fem types.
#include <dm/dm.hpp>

#include <gtest/gtest.h>

#include <format>
#include <string>

namespace fe = dm::fem;

static_assert(std::formattable<fe::BCType, char>);
static_assert(std::formattable<fe::BoundaryCondition1D, char>);
static_assert(std::formattable<fe::Mesh1D, char>);
static_assert(std::formattable<fe::Mesh2D, char>);
static_assert(std::formattable<fe::FEM1DResult, char>);
static_assert(std::formattable<fe::FEM1DTimeSeries, char>);
static_assert(std::formattable<fe::FEM2DResult, char>);

TEST(FemFormat, EnumAndBoundaryCondition) {
    EXPECT_EQ(std::format("{}", fe::BCType::Neumann), "Neumann");
    EXPECT_EQ(std::format("{}", static_cast<fe::BCType>(9)), "BCType(9)");
    EXPECT_EQ(std::format("{}", fe::BoundaryCondition1D{fe::BCType::Dirichlet, 1.5, 0.0}),
              "BoundaryCondition1D(type=Dirichlet, value=1.5)");
    EXPECT_EQ(std::format("{}", fe::BoundaryCondition1D{fe::BCType::Robin, 2, 0.5}),
              "BoundaryCondition1D(type=Robin, value=2, robin_coefficient=0.5)");
}

TEST(FemFormat, Meshes) {
    EXPECT_EQ(std::format("{}", fe::make_uniform_mesh1d(0.0, 1.0, 4)),
              "Mesh1D(nodes=5, elements=4, domain=[0, 1], x=[0, 0.25, 0.5, 0.75, 1])");
    EXPECT_EQ(std::format("{}", fe::Mesh1D{}), "Mesh1D(nodes=0)");
    EXPECT_EQ(std::format("{}", fe::make_rectangular_mesh2d(0, 0, 2, 1, 2, 1)),
              "Mesh2D(nodes=6, triangles=4, bbox=[0, 2] x [0, 1])");
}

TEST(FemFormat, Results) {
    const auto mesh = fe::make_uniform_mesh1d(0.0, 1.0, 2);
    const fe::BoundaryCondition1D zero{};
    const auto r = fe::FEM1D::solve_builtin(mesh, {1.0}, {0.0}, "zero", {}, zero, zero);
    EXPECT_EQ(std::format("{}", r), "FEM1DResult(nodes=3, domain=[0, 1], u=[0, 0, 0])");
    const fe::FEM1DTimeSeries ts{{0, 1}, {0, 0.5}, {{1, 1}, {0.5, 0.25}}};
    EXPECT_EQ(std::format("{}", ts), "FEM1DTimeSeries(nodes=2, steps=2, t=[0, 0.5], u_end=[0.5, 0.25])");
    fe::FEM2DResult r2;
    r2.nodes.resize(3);
    r2.u = {0.5, -1, 2};
    EXPECT_EQ(std::format("{}", r2), "FEM2DResult(nodes=3, u_min=-1, u_max=2)");
}
