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
Vector matvec_transpose(const Matrix& A, const Vector& x) {
    const std::size_t n = A.size();
    Vector            y(n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) y[j] += A[i][j] * x[i]; // (A^T x)_j = sum_i A[i][j] x[i]
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

GaussJordanResult gauss_jordan(std::vector<std::vector<double>> a, std::vector<double> b) {
    const std::size_t n = a.size();
    GaussJordanResult out;
    Matrix            inv(n, Vector(n, 0.0));
    for (std::size_t i = 0; i < n; ++i) inv[i][i] = 1.0; // start the inverse as the identity

    for (std::size_t col = 0; col < n; ++col) {
        std::size_t pivot = col;
        for (std::size_t r = col + 1; r < n; ++r)
            if (std::fabs(a[r][col]) > std::fabs(a[pivot][col])) pivot = r;
        if (std::fabs(a[pivot][col]) < 1e-12) return out; // singular
        std::swap(a[col], a[pivot]);
        std::swap(inv[col], inv[pivot]);
        std::swap(b[col], b[pivot]);

        const double d = a[col][col]; // normalize the pivot row to a leading 1
        for (std::size_t k = 0; k < n; ++k) { a[col][k] /= d; inv[col][k] /= d; }
        b[col] /= d;

        for (std::size_t r = 0; r < n; ++r) { // clear the column both above and below the pivot
            if (r == col) continue;
            const double f = a[r][col];
            for (std::size_t k = 0; k < n; ++k) { a[r][k] -= f * a[col][k]; inv[r][k] -= f * inv[col][k]; }
            b[r] -= f * b[col];
        }
    }
    out.solved  = true;
    out.x       = std::move(b);
    out.inverse = std::move(inv);
    return out;
}

IterativeSolution sor(const std::vector<std::vector<double>>& a, const std::vector<double>& b, double omega,
                      int max_iter, double tol) {
    const std::size_t n = a.size();
    IterativeSolution out;
    Vector            x(n, 0.0);
    for (int it = 1; it <= max_iter; ++it) {
        for (std::size_t i = 0; i < n; ++i) {
            double s = b[i];
            for (std::size_t j = 0; j < n; ++j)
                if (j != i) s -= a[i][j] * x[j];
            const double gs = s / a[i][i];                 // the Gauss-Seidel value
            x[i]            = (1.0 - omega) * x[i] + omega * gs; // over-relaxed blend
        }
        out.iterations = it;
        out.residual   = residual_norm(a, x, b);
        if (out.residual < tol) { out.converged = true; break; }
    }
    out.x = std::move(x);
    return out;
}

IterativeSolution biconjugate_gradient(const std::vector<std::vector<double>>& a, const std::vector<double>& b,
                                       int max_iter, double tol) {
    const std::size_t n = a.size();
    IterativeSolution out;
    Vector            x(n, 0.0);
    Vector            r = b;      // residual (x0 = 0)
    Vector            rt = r;     // shadow residual
    Vector            p = r, pt = rt;
    double            rho_old = dot(rt, r);

    for (int it = 1; it <= max_iter; ++it) {
        const Vector Ap  = matvec(a, p);
        const Vector Atp = matvec_transpose(a, pt);
        const double pAp = dot(pt, Ap);
        if (std::fabs(pAp) < 1e-30) break; // breakdown
        const double alpha = rho_old / pAp;
        for (std::size_t i = 0; i < n; ++i) {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
            rt[i] -= alpha * Atp[i];
        }
        out.iterations = it;
        out.residual   = std::sqrt(dot(r, r));
        if (out.residual < tol) { out.converged = true; break; }
        const double rho_new = dot(rt, r);
        if (std::fabs(rho_new) < 1e-30) break; // breakdown
        const double beta = rho_new / rho_old;
        for (std::size_t i = 0; i < n; ++i) {
            p[i]  = r[i] + beta * p[i];
            pt[i] = rt[i] + beta * pt[i];
        }
        rho_old = rho_new;
    }
    out.x = std::move(x);
    return out;
}

std::vector<double> levinson_solve(const std::vector<double>& first_row, const std::vector<double>& rhs) {
    const std::size_t N = first_row.size();
    if (N == 0) return {};
    auto t = [&](long k) { return first_row[static_cast<std::size_t>(k < 0 ? -k : k)]; }; // symmetric Toeplitz

    Vector f = {1.0 / t(0)};  // forward vector
    Vector bvec = {1.0 / t(0)}; // backward vector
    Vector x = {rhs[0] / t(0)};
    for (std::size_t s = 1; s < N; ++s) {
        double ef = 0.0, eb = 0.0, ex = 0.0;
        for (std::size_t i = 0; i < s; ++i) {
            ef += t(static_cast<long>(s) - static_cast<long>(i)) * f[i];       // forward error
            eb += t(-(static_cast<long>(i) + 1)) * bvec[i];                    // backward error
            ex += t(static_cast<long>(s) - static_cast<long>(i)) * x[i];       // solution error
        }
        const double denom = 1.0 - eb * ef;
        Vector       fn(s + 1, 0.0), bn(s + 1, 0.0), xn(s + 1, 0.0);
        for (std::size_t j = 0; j <= s; ++j) {
            const double fpad = (j < s) ? f[j] : 0.0;
            const double bpad = (j > 0) ? bvec[j - 1] : 0.0;
            fn[j] = (fpad - ef * bpad) / denom;
            bn[j] = (bpad - eb * fpad) / denom;
        }
        for (std::size_t j = 0; j <= s; ++j) {
            const double xpad = (j < s) ? x[j] : 0.0;
            xn[j] = xpad + (rhs[s] - ex) * bn[j];
        }
        f = std::move(fn);
        bvec = std::move(bn);
        x = std::move(xn);
    }
    return x;
}

} // namespace datamunge::algorithms
