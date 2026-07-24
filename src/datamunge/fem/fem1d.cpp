#include <datamunge/fem/fem1d.hpp>

#include <cmath>
#include <stdexcept>

namespace datamunge::fem {

namespace {

// Caller must have already validated the size via check_coefficient_size().
double resolve_coefficient(const std::vector<double>& coeff, std::size_t element_index) {
    return coeff.size() == 1 ? coeff[0] : coeff[element_index];
}

void check_coefficient_size(const std::vector<double>& coeff, std::size_t num_elements, const char* name) {
    if (coeff.empty() || (coeff.size() != 1 && coeff.size() != num_elements)) {
        throw std::invalid_argument(std::string("FEM1D: ") + name +
                                     " must have size 1 (constant) or mesh.num_elements()");
    }
}

// 3-point Gauss-Legendre quadrature on [-1, 1]; exact through degree 5, plenty for the smooth
// source terms this "basic" solver targets.
constexpr double kGaussPoints[3] = {-0.7745966692414834, 0.0, 0.7745966692414834};
constexpr double kGaussWeights[3] = {5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0};

// Assembles the tridiagonal system for -(p u')' + q u = f (no boundary terms yet).
struct TridiagonalSystem {
    std::vector<double> lower, diag, upper, rhs;
};

TridiagonalSystem assemble_interior(const Mesh1D& mesh, const std::vector<double>& p, const std::vector<double>& q,
                                     ScalarField1D& f) {
    const std::size_t n = mesh.num_nodes();
    const std::size_t ne = mesh.num_elements();
    TridiagonalSystem sys;
    sys.lower.assign(n, 0.0);
    sys.diag.assign(n, 0.0);
    sys.upper.assign(n, 0.0);
    sys.rhs.assign(n, 0.0);

    for (std::size_t e = 0; e < ne; ++e) {
        const double x_l = mesh.nodes[e];
        const double x_r = mesh.nodes[e + 1];
        const double h = x_r - x_l;
        const double p_e = resolve_coefficient(p, e);
        const double q_e = resolve_coefficient(q, e);

        // Local stiffness (p/h * [[1,-1],[-1,1]]) plus local mass (q*h/6 * [[2,1],[1,2]]).
        const double k_diag = p_e / h;
        const double m_diag = q_e * h / 3.0;
        const double m_off = q_e * h / 6.0;

        sys.diag[e] += k_diag + m_diag;
        sys.diag[e + 1] += k_diag + m_diag;
        sys.upper[e] += -k_diag + m_off; // coupling node e -> e+1
        sys.lower[e + 1] += -k_diag + m_off; // coupling node e+1 -> e

        // Local load vector via 3-point Gauss-Legendre quadrature.
        const double mid = 0.5 * (x_l + x_r);
        const double half = 0.5 * h;
        double f_left = 0.0, f_right = 0.0;
        for (int g = 0; g < 3; ++g) {
            const double x = mid + half * kGaussPoints[g];
            const double w = kGaussWeights[g] * half;
            const double fx = f.evaluate(x);
            f_left += w * fx * (x_r - x) / h;
            f_right += w * fx * (x - x_l) / h;
        }
        sys.rhs[e] += f_left;
        sys.rhs[e + 1] += f_right;
    }

    return sys;
}

// Applies the left/right boundary conditions in place, keeping the system tridiagonal and
// symmetric (Dirichlet elimination folds the removed row/column into the neighbor's rhs).
void apply_boundary_conditions(TridiagonalSystem& sys, const BoundaryCondition1D& left,
                                const BoundaryCondition1D& right) {
    const std::size_t n = sys.diag.size();
    const std::size_t last = n - 1;

    // Natural (Neumann/Robin) terms first -- both are additive and safe to apply before or
    // after Dirichlet elimination, but must be applied before we potentially overwrite a
    // Dirichlet row below.
    if (left.type == BCType::Neumann) {
        sys.rhs[0] += left.value;
    } else if (left.type == BCType::Robin) {
        sys.diag[0] += left.robin_coefficient;
        sys.rhs[0] += left.robin_coefficient * left.value;
    }
    if (right.type == BCType::Neumann) {
        sys.rhs[last] += right.value;
    } else if (right.type == BCType::Robin) {
        sys.diag[last] += right.robin_coefficient;
        sys.rhs[last] += right.robin_coefficient * right.value;
    }

    if (left.type == BCType::Dirichlet) {
        if (n > 1) {
            sys.rhs[1] -= sys.lower[1] * left.value;
            sys.lower[1] = 0.0;
        }
        sys.diag[0] = 1.0;
        sys.upper[0] = 0.0;
        sys.rhs[0] = left.value;
    }
    if (right.type == BCType::Dirichlet) {
        if (n > 1) {
            sys.rhs[last - 1] -= sys.upper[last - 1] * right.value;
            sys.upper[last - 1] = 0.0;
        }
        sys.diag[last] = 1.0;
        sys.lower[last] = 0.0;
        sys.rhs[last] = right.value;
    }
}

std::vector<double> thomas_solve(std::vector<double> lower, std::vector<double> diag, std::vector<double> upper,
                                  std::vector<double> rhs) {
    const std::size_t n = diag.size();
    for (std::size_t i = 1; i < n; ++i) {
        const double m = lower[i] / diag[i - 1];
        diag[i] -= m * upper[i - 1];
        rhs[i] -= m * rhs[i - 1];
    }
    std::vector<double> u(n);
    u[n - 1] = rhs[n - 1] / diag[n - 1];
    for (std::size_t i = n - 1; i-- > 0;) {
        u[i] = (rhs[i] - upper[i] * u[i + 1]) / diag[i];
    }
    return u;
}

class ZeroSource : public ScalarField1D {
public:
    double evaluate(double) override { return 0.0; }
};

class ConstantSource : public ScalarField1D {
public:
    explicit ConstantSource(double c) : c_(c) {}
    double evaluate(double) override { return c_; }

private:
    double c_;
};

class LinearSource : public ScalarField1D {
public:
    LinearSource(double m, double c) : m_(m), c_(c) {}
    double evaluate(double x) override { return m_ * x + c_; }

private:
    double m_, c_;
};

class PolynomialSource : public ScalarField1D {
public:
    explicit PolynomialSource(std::vector<double> coeffs) : coeffs_(std::move(coeffs)) {}
    double evaluate(double x) override {
        double result = 0.0;
        double power = 1.0;
        for (double c : coeffs_) {
            result += c * power;
            power *= x;
        }
        return result;
    }

private:
    std::vector<double> coeffs_;
};

class SineSource : public ScalarField1D {
public:
    SineSource(double amplitude, double freq, double phase) : amplitude_(amplitude), freq_(freq), phase_(phase) {}
    double evaluate(double x) override { return amplitude_ * std::sin(freq_ * x + phase_); }

private:
    double amplitude_, freq_, phase_;
};

double param_or_default(const std::vector<double>& params, std::size_t index, double fallback) {
    return index < params.size() ? params[index] : fallback;
}

} // namespace

FEM1DResult FEM1D::solve(const Mesh1D& mesh, const std::vector<double>& p, const std::vector<double>& q,
                          ScalarField1D& f, const BoundaryCondition1D& left, const BoundaryCondition1D& right) {
    const std::size_t ne = mesh.num_elements();
    check_coefficient_size(p, ne, "p");
    check_coefficient_size(q, ne, "q");

    TridiagonalSystem sys = assemble_interior(mesh, p, q, f);
    apply_boundary_conditions(sys, left, right);

    FEM1DResult result;
    result.nodes = mesh.nodes;
    result.u = thomas_solve(sys.lower, sys.diag, sys.upper, sys.rhs);
    return result;
}

FEM1DResult FEM1D::solve_builtin(const Mesh1D& mesh, const std::vector<double>& p, const std::vector<double>& q,
                                  const std::string& source, const std::vector<double>& source_params,
                                  const BoundaryCondition1D& left, const BoundaryCondition1D& right) {
    if (source == "zero") {
        ZeroSource f;
        return solve(mesh, p, q, f, left, right);
    }
    if (source == "constant") {
        ConstantSource f(param_or_default(source_params, 0, 1.0));
        return solve(mesh, p, q, f, left, right);
    }
    if (source == "linear") {
        LinearSource f(param_or_default(source_params, 0, 1.0), param_or_default(source_params, 1, 0.0));
        return solve(mesh, p, q, f, left, right);
    }
    if (source == "polynomial") {
        PolynomialSource f(source_params);
        return solve(mesh, p, q, f, left, right);
    }
    if (source == "sine") {
        SineSource f(param_or_default(source_params, 0, 1.0), param_or_default(source_params, 1, 1.0),
                     param_or_default(source_params, 2, 0.0));
        return solve(mesh, p, q, f, left, right);
    }
    throw std::invalid_argument("FEM1D::solve_builtin: unknown source '" + source + "'");
}

FEM1DTimeSeries FEM1D::solve_transient(const Mesh1D& mesh, const std::vector<double>& p,
                                        const std::vector<double>& q, ScalarField1D& f,
                                        const BoundaryCondition1D& left, const BoundaryCondition1D& right,
                                        const std::vector<double>& u0, double t_end, double dt) {
    const std::size_t n = mesh.num_nodes();
    const std::size_t ne = mesh.num_elements();
    check_coefficient_size(p, ne, "p");
    check_coefficient_size(q, ne, "q");
    if (u0.size() != n) {
        throw std::invalid_argument("FEM1D::solve_transient: u0 must have one value per mesh node");
    }
    if (dt <= 0.0) {
        throw std::invalid_argument("FEM1D::solve_transient: dt must be positive");
    }
    if (t_end <= 0.0) {
        throw std::invalid_argument("FEM1D::solve_transient: t_end must be positive");
    }

    // Pure mass matrix (coefficient 1, not q) for the u_t term, assembled once since the mesh
    // never changes across time steps.
    std::vector<double> mass_lower(n, 0.0), mass_diag(n, 0.0), mass_upper(n, 0.0);
    for (std::size_t e = 0; e < ne; ++e) {
        const double h = mesh.element_length(e);
        mass_diag[e] += h / 3.0;
        mass_diag[e + 1] += h / 3.0;
        mass_upper[e] += h / 6.0;
        mass_lower[e + 1] += h / 6.0;
    }

    const std::size_t num_steps = static_cast<std::size_t>(std::ceil(t_end / dt));

    FEM1DTimeSeries series;
    series.nodes = mesh.nodes;
    series.times.reserve(num_steps + 1);
    series.u.reserve(num_steps + 1);
    series.times.push_back(0.0);
    series.u.push_back(u0);

    std::vector<double> u_prev = u0;
    for (std::size_t step = 1; step <= num_steps; ++step) {
        const double t = std::min(static_cast<double>(step) * dt, t_end);
        const double actual_dt = t - series.times.back();

        // (M/dt + K) u^{n+1} = M/dt u^n + F, where K/F come from the same steady-state
        // assembly (stiffness + reaction + boundary terms) used by solve().
        TridiagonalSystem sys = assemble_interior(mesh, p, q, f);
        for (std::size_t i = 0; i < n; ++i) {
            sys.diag[i] += mass_diag[i] / actual_dt;
            if (i > 0) sys.lower[i] += mass_lower[i] / actual_dt;
            if (i + 1 < n) sys.upper[i] += mass_upper[i] / actual_dt;
        }

        std::vector<double> mass_dot_u(n, 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            mass_dot_u[i] += mass_diag[i] * u_prev[i];
            if (i > 0) mass_dot_u[i] += mass_lower[i] * u_prev[i - 1];
            if (i + 1 < n) mass_dot_u[i] += mass_upper[i] * u_prev[i + 1];
        }
        for (std::size_t i = 0; i < n; ++i) {
            sys.rhs[i] += mass_dot_u[i] / actual_dt;
        }

        apply_boundary_conditions(sys, left, right);

        std::vector<double> u_next = thomas_solve(sys.lower, sys.diag, sys.upper, sys.rhs);
        series.times.push_back(t);
        series.u.push_back(u_next);
        u_prev = std::move(u_next);
    }

    return series;
}

} // namespace datamunge::fem
