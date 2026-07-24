#include <datamunge/fem/fem2d.hpp>

#include <datamunge/linalg/solvers.hpp>

#include <array>
#include <cmath>
#include <stdexcept>

namespace datamunge::fem {

namespace {

using datamunge::geometry::Point2D;

// Per-triangle P1 shape-function gradients (constant over the triangle) and its area, via the
// standard formulas b_i = (y_j - y_k) / (2A), c_i = (x_k - x_j) / (2A) for CCW vertices.
struct TriangleGeometry {
    std::array<double, 3> b{};
    std::array<double, 3> c{};
    double area{};
};

TriangleGeometry triangle_geometry(const Point2D& p0, const Point2D& p1, const Point2D& p2) {
    const double two_area = (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y);
    TriangleGeometry g;
    g.area = 0.5 * two_area;
    g.b = {(p1.y - p2.y) / two_area, (p2.y - p0.y) / two_area, (p0.y - p1.y) / two_area};
    g.c = {(p2.x - p1.x) / two_area, (p0.x - p2.x) / two_area, (p1.x - p0.x) / two_area};
    return g;
}

Point2D midpoint(const Point2D& a, const Point2D& b) { return {(a.x + b.x) / 2.0, (a.y + b.y) / 2.0}; }

class ZeroSource2D : public ScalarField2D {
public:
    double evaluate(double, double) override { return 0.0; }
};

class ConstantSource2D : public ScalarField2D {
public:
    explicit ConstantSource2D(double c) : c_(c) {}
    double evaluate(double, double) override { return c_; }

private:
    double c_;
};

class SineSource2D : public ScalarField2D {
public:
    SineSource2D(double amplitude, double fx, double fy) : amplitude_(amplitude), fx_(fx), fy_(fy) {}
    double evaluate(double x, double y) override { return amplitude_ * std::sin(fx_ * x) * std::sin(fy_ * y); }

private:
    double amplitude_, fx_, fy_;
};

double param_or_default(const std::vector<double>& params, std::size_t index, double fallback) {
    return index < params.size() ? params[index] : fallback;
}

} // namespace

FEM2DResult FEM2D::solve(const Mesh2D& mesh, double k, ScalarField2D& f, const std::vector<int>& dirichlet_nodes,
                          const std::vector<double>& dirichlet_values) {
    const std::size_t n = mesh.num_nodes();
    if (dirichlet_nodes.size() != dirichlet_values.size()) {
        throw std::invalid_argument("FEM2D::solve: dirichlet_nodes and dirichlet_values must have the same size");
    }

    std::vector<bool> is_dirichlet(n, false);
    std::vector<double> dirichlet_value(n, 0.0);
    for (std::size_t i = 0; i < dirichlet_nodes.size(); ++i) {
        const int node = dirichlet_nodes[i];
        if (node < 0 || static_cast<std::size_t>(node) >= n) {
            throw std::invalid_argument("FEM2D::solve: dirichlet_nodes contains an out-of-range index");
        }
        is_dirichlet[static_cast<std::size_t>(node)] = true;
        dirichlet_value[static_cast<std::size_t>(node)] = dirichlet_values[i];
    }
    if (dirichlet_nodes.size() >= n) {
        throw std::invalid_argument("FEM2D::solve: at least one node must be free (unconstrained)");
    }

    // Assemble the full stiffness system as (row, col, value) triplets and the dense load
    // vector, before boundary elimination.
    std::vector<std::size_t> rows, cols;
    std::vector<double> vals;
    std::vector<double> load(n, 0.0);

    for (const auto& tri : mesh.triangles) {
        const std::array<std::size_t, 3> v{tri.a, tri.b, tri.c};
        const Point2D& p0 = mesh.nodes[v[0]];
        const Point2D& p1 = mesh.nodes[v[1]];
        const Point2D& p2 = mesh.nodes[v[2]];
        const TriangleGeometry geom = triangle_geometry(p0, p1, p2);

        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                const double kij = k * geom.area * (geom.b[i] * geom.b[j] + geom.c[i] * geom.c[j]);
                rows.push_back(v[i]);
                cols.push_back(v[j]);
                vals.push_back(kij);
            }
        }

        // Load vector via edge-midpoint quadrature (exact for quadratics, weight area/3 per
        // midpoint): F_i += (area/6) * sum of f at the two edge midpoints touching vertex i.
        const double f01 = f.evaluate(midpoint(p0, p1).x, midpoint(p0, p1).y);
        const double f02 = f.evaluate(midpoint(p0, p2).x, midpoint(p0, p2).y);
        const double f12 = f.evaluate(midpoint(p1, p2).x, midpoint(p1, p2).y);
        load[v[0]] += (geom.area / 6.0) * (f01 + f02);
        load[v[1]] += (geom.area / 6.0) * (f01 + f12);
        load[v[2]] += (geom.area / 6.0) * (f02 + f12);
    }

    // Dirichlet elimination: drop rows/columns touching a constrained node (folding their
    // contribution into the free rows' load vector), then reinsert an identity row/rhs entry
    // for each constrained node -- keeps the reduced system symmetric positive definite.
    datamunge::linalg::SparseCOO<double> A(n, n);
    for (std::size_t idx = 0; idx < vals.size(); ++idx) {
        const std::size_t r = rows[idx];
        const std::size_t c = cols[idx];
        if (is_dirichlet[r]) continue;
        if (is_dirichlet[c]) {
            load[r] -= vals[idx] * dirichlet_value[c];
            continue;
        }
        A.set(r, c, vals[idx]);
    }
    for (std::size_t i = 0; i < n; ++i) {
        if (is_dirichlet[i]) {
            A.set(i, i, 1.0);
            load[i] = dirichlet_value[i];
        }
    }
    A.compress();

    const auto csr = datamunge::linalg::to_csr(A);
    const auto cg_result = datamunge::linalg::cg(csr, load);

    FEM2DResult result;
    result.nodes = mesh.nodes;
    result.u = cg_result.x;
    return result;
}

FEM2DResult FEM2D::solve_builtin(const Mesh2D& mesh, double k, const std::string& source,
                                  const std::vector<double>& source_params, const std::vector<int>& dirichlet_nodes,
                                  const std::vector<double>& dirichlet_values) {
    if (source == "zero") {
        ZeroSource2D f;
        return solve(mesh, k, f, dirichlet_nodes, dirichlet_values);
    }
    if (source == "constant") {
        ConstantSource2D f(param_or_default(source_params, 0, 1.0));
        return solve(mesh, k, f, dirichlet_nodes, dirichlet_values);
    }
    if (source == "sine") {
        SineSource2D f(param_or_default(source_params, 0, 1.0), param_or_default(source_params, 1, 1.0),
                       param_or_default(source_params, 2, 1.0));
        return solve(mesh, k, f, dirichlet_nodes, dirichlet_values);
    }
    throw std::invalid_argument("FEM2D::solve_builtin: unknown source '" + source + "'");
}

} // namespace datamunge::fem
