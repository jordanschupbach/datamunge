#include <gtest/gtest.h>

#include <datamunge/fem/fem.hpp>

#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <vector>

using namespace datamunge::fem;
using datamunge::geometry::Point2D;

namespace {

constexpr double kPi = 3.14159265358979323846;

double max_abs_error(const FEM1DResult& result, const std::function<double(double)>& exact) {
    double worst = 0.0;
    for (std::size_t i = 0; i < result.size(); ++i) {
        worst = std::max(worst, std::abs(result.value_at(i) - exact(result.node_at(i))));
    }
    return worst;
}

class CustomConstantSource : public ScalarField1D {
public:
    explicit CustomConstantSource(double c) : c_(c) {}
    double evaluate(double) override { return c_; }

private:
    double c_;
};

} // namespace

// ---------------------------------------------------------------------------
// Mesh1D
// ---------------------------------------------------------------------------

TEST(Mesh1D, UniformMeshHasCorrectNodesAndSpacing) {
    Mesh1D mesh = make_uniform_mesh1d(0.0, 2.0, 4);
    EXPECT_EQ(mesh.num_nodes(), 5U);
    EXPECT_EQ(mesh.num_elements(), 4U);
    EXPECT_DOUBLE_EQ(mesh.node_at(0), 0.0);
    EXPECT_DOUBLE_EQ(mesh.node_at(4), 2.0);
    for (std::size_t e = 0; e < mesh.num_elements(); ++e) {
        EXPECT_NEAR(mesh.element_length(e), 0.5, 1e-12);
    }
}

TEST(Mesh1D, RejectsInvalidInputs) {
    EXPECT_THROW(make_uniform_mesh1d(0.0, 1.0, 0), std::invalid_argument);
    EXPECT_THROW(make_uniform_mesh1d(1.0, 1.0, 4), std::invalid_argument);
    EXPECT_THROW(make_uniform_mesh1d(2.0, 1.0, 4), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// FEM1D steady state
// ---------------------------------------------------------------------------

// -u'' = 1 on [0,1], u(0) = u(1) = 0 has exact solution u(x) = x(1-x)/2, a quadratic -- P1 FEM
// is nodally exact for this class of problem (constant source, no reaction term), even on a
// coarse mesh.
TEST(FEM1D, NodallyExactForConstantSourcePureDiffusion) {
    Mesh1D mesh = make_uniform_mesh1d(0.0, 1.0, 5);
    BoundaryCondition1D left{BCType::Dirichlet, 0.0, 0.0};
    BoundaryCondition1D right{BCType::Dirichlet, 0.0, 0.0};

    FEM1DResult result = FEM1D::solve_builtin(mesh, {1.0}, {0.0}, "constant", {1.0}, left, right);

    for (std::size_t i = 0; i < result.size(); ++i) {
        const double x = result.node_at(i);
        EXPECT_NEAR(result.value_at(i), x * (1.0 - x) / 2.0, 1e-10);
    }
}

// -u'' + u = (pi^2+1) sin(pi x) on [0,1], u(0)=u(1)=0 has exact solution u(x) = sin(pi x).
// Refining the mesh should reduce the nodal error at roughly the O(h^2) rate expected of P1
// Galerkin FEM.
TEST(FEM1D, ConvergesForDiffusionReactionManufacturedSolution) {
    auto exact = [](double x) { return std::sin(kPi * x); };
    BoundaryCondition1D left{BCType::Dirichlet, 0.0, 0.0};
    BoundaryCondition1D right{BCType::Dirichlet, 0.0, 0.0};
    const std::vector<double> params = {kPi * kPi + 1.0, kPi, 0.0};

    Mesh1D coarse = make_uniform_mesh1d(0.0, 1.0, 8);
    Mesh1D fine = make_uniform_mesh1d(0.0, 1.0, 64);

    const double err_coarse = max_abs_error(FEM1D::solve_builtin(coarse, {1.0}, {1.0}, "sine", params, left, right), exact);
    const double err_fine = max_abs_error(FEM1D::solve_builtin(fine, {1.0}, {1.0}, "sine", params, left, right), exact);

    EXPECT_LT(err_fine, err_coarse);
    EXPECT_LT(err_fine, 1e-4);
    // 8x refinement (2^3) should shrink error by roughly 2^(3*2)=64x for O(h^2) convergence;
    // allow generous slack either side of the ideal rate.
    EXPECT_GT(err_coarse / err_fine, 20.0);
}

// -u'' = 0 on [0,1], u(0)=0, prescribed outward flux p*u'(1)=2 has exact linear solution
// u(x) = 2x.
TEST(FEM1D, NeumannBoundaryConditionMatchesExactLinearSolution) {
    Mesh1D mesh = make_uniform_mesh1d(0.0, 1.0, 10);
    BoundaryCondition1D left{BCType::Dirichlet, 0.0, 0.0};
    BoundaryCondition1D right{BCType::Neumann, 2.0, 0.0};

    FEM1DResult result = FEM1D::solve_builtin(mesh, {1.0}, {0.0}, "zero", {}, left, right);

    for (std::size_t i = 0; i < result.size(); ++i) {
        EXPECT_NEAR(result.value_at(i), 2.0 * result.node_at(i), 1e-10);
    }
}

// -u'' = 0 on [0,1], u(0)=0, Robin condition u'(1) + 3*(u(1) - 4) = 0 at the right end has
// exact linear solution u(x) = m*x with m = alpha*u_inf/(1+alpha) = 3*4/4 = 3.
TEST(FEM1D, RobinBoundaryConditionMatchesExactLinearSolution) {
    Mesh1D mesh = make_uniform_mesh1d(0.0, 1.0, 10);
    BoundaryCondition1D left{BCType::Dirichlet, 0.0, 0.0};
    BoundaryCondition1D right{BCType::Robin, 4.0, 3.0};

    FEM1DResult result = FEM1D::solve_builtin(mesh, {1.0}, {0.0}, "zero", {}, left, right);

    for (std::size_t i = 0; i < result.size(); ++i) {
        EXPECT_NEAR(result.value_at(i), 3.0 * result.node_at(i), 1e-10);
    }
}

TEST(FEM1D, DirectorSourceMatchesEquivalentBuiltinSource) {
    Mesh1D mesh = make_uniform_mesh1d(0.0, 1.0, 6);
    BoundaryCondition1D left{BCType::Dirichlet, 0.0, 0.0};
    BoundaryCondition1D right{BCType::Dirichlet, 0.0, 0.0};

    CustomConstantSource f(3.5);
    FEM1DResult via_director = FEM1D::solve(mesh, {1.0}, {0.5}, f, left, right);
    FEM1DResult via_builtin = FEM1D::solve_builtin(mesh, {1.0}, {0.5}, "constant", {3.5}, left, right);

    for (std::size_t i = 0; i < via_director.size(); ++i) {
        EXPECT_NEAR(via_director.value_at(i), via_builtin.value_at(i), 1e-12);
    }
}

TEST(FEM1D, RejectsMismatchedCoefficientSizes) {
    Mesh1D mesh = make_uniform_mesh1d(0.0, 1.0, 4);
    BoundaryCondition1D bc{BCType::Dirichlet, 0.0, 0.0};
    EXPECT_THROW(FEM1D::solve_builtin(mesh, {1.0, 2.0, 3.0}, {0.0}, "zero", {}, bc, bc), std::invalid_argument);
    EXPECT_THROW(FEM1D::solve_builtin(mesh, {}, {0.0}, "zero", {}, bc, bc), std::invalid_argument);
}

TEST(FEM1D, RejectsUnknownBuiltinSource) {
    Mesh1D mesh = make_uniform_mesh1d(0.0, 1.0, 4);
    BoundaryCondition1D bc{BCType::Dirichlet, 0.0, 0.0};
    EXPECT_THROW(FEM1D::solve_builtin(mesh, {1.0}, {0.0}, "not_a_real_source", {}, bc, bc), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// FEM1D transient
// ---------------------------------------------------------------------------

// Backward Euler is unconditionally stable and the discrete solution should relax to the same
// steady state that FEM1D::solve() computes directly, for fixed BCs/source held over a long
// enough time horizon.
TEST(FEM1D, TransientSolutionRelaxesToSteadyState) {
    Mesh1D mesh = make_uniform_mesh1d(0.0, 1.0, 10);
    BoundaryCondition1D left{BCType::Dirichlet, 0.0, 0.0};
    BoundaryCondition1D right{BCType::Dirichlet, 0.0, 0.0};
    std::vector<double> u0(mesh.num_nodes(), 0.0);

    FEM1DResult steady = FEM1D::solve_builtin(mesh, {1.0}, {0.5}, "constant", {1.0}, left, right);

    CustomConstantSource f(1.0);
    FEM1DTimeSeries transient = FEM1D::solve_transient(mesh, {1.0}, {0.5}, f, left, right, u0, 20.0, 0.05);

    EXPECT_EQ(transient.num_nodes(), mesh.num_nodes());
    for (std::size_t i = 0; i < mesh.num_nodes(); ++i) {
        EXPECT_NEAR(transient.value_at(transient.num_steps() - 1, i), steady.value_at(i), 1e-3);
    }
    // The initial condition should be preserved verbatim as the first time slice.
    EXPECT_DOUBLE_EQ(transient.time_at(0), 0.0);
    for (std::size_t i = 0; i < mesh.num_nodes(); ++i) {
        EXPECT_DOUBLE_EQ(transient.value_at(0, i), 0.0);
    }
}

TEST(FEM1D, TransientRejectsInvalidInputs) {
    Mesh1D mesh = make_uniform_mesh1d(0.0, 1.0, 4);
    BoundaryCondition1D bc{BCType::Dirichlet, 0.0, 0.0};
    CustomConstantSource f(1.0);
    std::vector<double> wrong_size_u0(2, 0.0);
    EXPECT_THROW(FEM1D::solve_transient(mesh, {1.0}, {0.0}, f, bc, bc, wrong_size_u0, 1.0, 0.1), std::invalid_argument);

    std::vector<double> u0(mesh.num_nodes(), 0.0);
    EXPECT_THROW(FEM1D::solve_transient(mesh, {1.0}, {0.0}, f, bc, bc, u0, 1.0, 0.0), std::invalid_argument);
    EXPECT_THROW(FEM1D::solve_transient(mesh, {1.0}, {0.0}, f, bc, bc, u0, 0.0, 0.1), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Mesh2D
// ---------------------------------------------------------------------------

TEST(Mesh2D, RectangularMeshHasCorrectCountsAndBoundary) {
    Mesh2D mesh = make_rectangular_mesh2d(0.0, 0.0, 1.0, 1.0, 4, 4);
    EXPECT_EQ(mesh.num_nodes(), 25U);
    EXPECT_EQ(mesh.num_triangles(), 32U);

    std::vector<int> boundary = boundary_nodes(mesh);
    // 4x4 grid: 9 interior nodes ((4-1)*(4-1)), so 25 - 9 = 16 boundary nodes.
    EXPECT_EQ(boundary.size(), 16U);
}

TEST(Mesh2D, RejectsInvalidInputs) {
    EXPECT_THROW(make_rectangular_mesh2d(0.0, 0.0, 1.0, 1.0, 0, 4), std::invalid_argument);
    EXPECT_THROW(make_rectangular_mesh2d(0.0, 0.0, 1.0, 1.0, 4, 0), std::invalid_argument);
    EXPECT_THROW(make_rectangular_mesh2d(1.0, 0.0, 1.0, 1.0, 4, 4), std::invalid_argument);
}

TEST(Mesh2D, FromPointsUsesDelaunayTriangulation) {
    std::vector<Point2D> pts = {{0, 0}, {1, 0}, {1, 1}, {0, 1}, {0.5, 0.5}};
    Mesh2D mesh = make_mesh2d_from_points(pts);
    EXPECT_EQ(mesh.num_nodes(), 5U);
    EXPECT_GT(mesh.num_triangles(), 0U);
}

// ---------------------------------------------------------------------------
// FEM2D
// ---------------------------------------------------------------------------

// Laplace's equation (f=0) with boundary data taken from a harmonic (linear) function has that
// same linear function as its exact solution everywhere -- P1 FEM should reproduce it almost
// exactly, interior nodes included, regardless of mesh resolution.
TEST(FEM2D, ExactForHarmonicLinearBoundaryData) {
    Mesh2D mesh = make_rectangular_mesh2d(0.0, 0.0, 1.0, 1.0, 6, 6);
    auto harmonic = [](const Point2D& p) { return p.x + 2.0 * p.y; };

    std::vector<int> boundary = boundary_nodes(mesh);
    std::vector<double> boundary_values;
    boundary_values.reserve(boundary.size());
    for (int node : boundary) {
        boundary_values.push_back(harmonic(mesh.node_at(static_cast<std::size_t>(node))));
    }

    FEM2DResult result = FEM2D::solve_builtin(mesh, 1.0, "zero", {}, boundary, boundary_values);

    for (std::size_t i = 0; i < result.size(); ++i) {
        EXPECT_NEAR(result.value_at(i), harmonic(result.node_at(i)), 1e-8);
    }
}

// -Laplacian(u) = 2*pi^2*sin(pi x)*sin(pi y) on the unit square with u=0 on the boundary has
// exact solution u(x,y) = sin(pi x)*sin(pi y). Refining the mesh should reduce the error.
TEST(FEM2D, ConvergesForPoissonManufacturedSolution) {
    auto exact = [](const Point2D& p) { return std::sin(kPi * p.x) * std::sin(kPi * p.y); };
    const std::vector<double> params = {2.0 * kPi * kPi, kPi, kPi};

    auto max_error = [&](std::size_t n) {
        Mesh2D mesh = make_rectangular_mesh2d(0.0, 0.0, 1.0, 1.0, n, n);
        std::vector<int> boundary = boundary_nodes(mesh);
        std::vector<double> boundary_values(boundary.size(), 0.0);
        FEM2DResult result = FEM2D::solve_builtin(mesh, 1.0, "sine", params, boundary, boundary_values);
        double worst = 0.0;
        for (std::size_t i = 0; i < result.size(); ++i) {
            worst = std::max(worst, std::abs(result.value_at(i) - exact(result.node_at(i))));
        }
        return worst;
    };

    const double err_coarse = max_error(8);
    const double err_fine = max_error(16);

    EXPECT_LT(err_fine, err_coarse);
    EXPECT_LT(err_fine, 0.05);
}

TEST(FEM2D, RejectsInvalidBoundaryData) {
    Mesh2D mesh = make_rectangular_mesh2d(0.0, 0.0, 1.0, 1.0, 4, 4);
    std::vector<int> boundary = boundary_nodes(mesh);
    std::vector<double> too_few(boundary.size() - 1, 0.0);
    EXPECT_THROW(FEM2D::solve_builtin(mesh, 1.0, "zero", {}, boundary, too_few), std::invalid_argument);

    std::vector<double> right_size(boundary.size(), 0.0);
    std::vector<int> bad_index = boundary;
    bad_index[0] = static_cast<int>(mesh.num_nodes()) + 5;
    EXPECT_THROW(FEM2D::solve_builtin(mesh, 1.0, "zero", {}, bad_index, right_size), std::invalid_argument);

    std::vector<int> all_nodes(mesh.num_nodes());
    for (std::size_t i = 0; i < all_nodes.size(); ++i) all_nodes[i] = static_cast<int>(i);
    std::vector<double> all_values(mesh.num_nodes(), 0.0);
    EXPECT_THROW(FEM2D::solve_builtin(mesh, 1.0, "zero", {}, all_nodes, all_values), std::invalid_argument);
}

TEST(FEM2D, RejectsUnknownBuiltinSource) {
    Mesh2D mesh = make_rectangular_mesh2d(0.0, 0.0, 1.0, 1.0, 4, 4);
    std::vector<int> boundary = boundary_nodes(mesh);
    std::vector<double> values(boundary.size(), 0.0);
    EXPECT_THROW(FEM2D::solve_builtin(mesh, 1.0, "not_a_real_source", {}, boundary, values), std::invalid_argument);
}
