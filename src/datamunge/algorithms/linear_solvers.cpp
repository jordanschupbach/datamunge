#include <datamunge/algorithms/linear_solvers.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

namespace datamunge::algorithms {

namespace {

using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;

Vector matvec(const Matrix& A, const Vector& x) {
    Vector y(A.size(), 0.0);
    for (std::size_t i = 0; i < A.size(); ++i)
        for (std::size_t j = 0; j < x.size(); ++j) y[i] += A[i][j] * x[j];
    return y;
}
double dot(const Vector& a, const Vector& b) {
    double s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}
double residual_norm(const Matrix& A, const Vector& x, const Vector& b) {
    const Vector Ax = matvec(A, x);
    double       s  = 0.0;
    for (std::size_t i = 0; i < b.size(); ++i) s += (Ax[i] - b[i]) * (Ax[i] - b[i]);
    return std::sqrt(s);
}

} // namespace

LinearSolution gaussian_elimination(std::vector<std::vector<double>> a, std::vector<double> b) {
    const std::size_t n = a.size();
    LinearSolution    out;
    for (std::size_t col = 0; col < n; ++col) {
        // Partial pivot: pick the row with the largest magnitude in this column.
        std::size_t pivot = col;
        for (std::size_t r = col + 1; r < n; ++r)
            if (std::fabs(a[r][col]) > std::fabs(a[pivot][col])) pivot = r;
        if (std::fabs(a[pivot][col]) < 1e-12) return out; // singular
        std::swap(a[col], a[pivot]);
        std::swap(b[col], b[pivot]);
        // Eliminate below the pivot.
        for (std::size_t r = col + 1; r < n; ++r) {
            const double factor = a[r][col] / a[col][col];
            for (std::size_t c = col; c < n; ++c) a[r][c] -= factor * a[col][c];
            b[r] -= factor * b[col];
        }
    }
    // Back substitution.
    Vector x(n, 0.0);
    for (std::size_t i = n; i-- > 0;) {
        double s = b[i];
        for (std::size_t j = i + 1; j < n; ++j) s -= a[i][j] * x[j];
        x[i] = s / a[i][i];
    }
    out.solved = true;
    out.x      = std::move(x);
    return out;
}

IterativeSolution gauss_seidel(const std::vector<std::vector<double>>& a, const std::vector<double>& b,
                               int max_iter, double tol) {
    const std::size_t n = a.size();
    IterativeSolution out;
    Vector            x(n, 0.0);
    for (int it = 1; it <= max_iter; ++it) {
        for (std::size_t i = 0; i < n; ++i) {
            double s = b[i];
            for (std::size_t j = 0; j < n; ++j)
                if (j != i) s -= a[i][j] * x[j]; // uses already-updated x[j] for j < i
            x[i] = s / a[i][i];
        }
        out.iterations = it;
        out.residual   = residual_norm(a, x, b);
        if (out.residual < tol) {
            out.converged = true;
            break;
        }
    }
    out.x = std::move(x);
    return out;
}

IterativeSolution conjugate_gradient(const std::vector<std::vector<double>>& a, const std::vector<double>& b,
                                     int max_iter, double tol) {
    const std::size_t n = a.size();
    IterativeSolution out;
    Vector            x(n, 0.0);
    Vector            r = b;                 // residual r0 = b - A x0 (x0 = 0)
    Vector            p = r;                 // initial search direction
    double            rs_old = dot(r, r);

    for (int it = 1; it <= max_iter; ++it) {
        const Vector Ap    = matvec(a, p);
        const double alpha = rs_old / dot(p, Ap);
        for (std::size_t i = 0; i < n; ++i) {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
        }
        out.iterations   = it;
        const double rs_new = dot(r, r);
        out.residual     = std::sqrt(rs_new);
        if (out.residual < tol) {
            out.converged = true;
            break;
        }
        const double beta = rs_new / rs_old;
        for (std::size_t i = 0; i < n; ++i) p[i] = r[i] + beta * p[i];
        rs_old = rs_new;
    }
    out.x = std::move(x);
    return out;
}

std::vector<double> thomas_solve(std::vector<double> sub, std::vector<double> diag,
                                 std::vector<double> super, std::vector<double> rhs) {
    const std::size_t n = diag.size();
    if (n == 0) return {};
    // Forward sweep: eliminate the sub-diagonal.
    for (std::size_t i = 1; i < n; ++i) {
        const double w = sub[i] / diag[i - 1];
        diag[i] -= w * super[i - 1];
        rhs[i] -= w * rhs[i - 1];
    }
    // Back substitution.
    std::vector<double> x(n, 0.0);
    x[n - 1] = rhs[n - 1] / diag[n - 1];
    for (std::size_t i = n - 1; i-- > 0;) x[i] = (rhs[i] - super[i] * x[i + 1]) / diag[i];
    return x;
}

} // namespace datamunge::algorithms
