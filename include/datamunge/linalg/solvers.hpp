#pragma once

#include <datamunge/linalg/preconditioners.hpp>

#include <cmath>
#include <stdexcept>
#include <vector>

namespace datamunge::linalg {

// ============================================================
// Result type returned by all iterative solvers
// ============================================================

template <typename T = double>
struct SolverResult {
    std::vector<T> x;                 // solution vector
    std::size_t    iterations{0};
    T              residual_norm{0};  // ||r_k|| / ||b||  (relative)
    bool           converged{false};
};

// ============================================================
// Options shared by Krylov solvers
// ============================================================

struct SolverOptions {
    double      tol{1e-8};   // relative residual tolerance
    std::size_t max_iter{0}; // 0 → use the system dimension
};

// ============================================================
// Internal BLAS-1 helpers
// ============================================================

namespace detail {

template <typename T>
T dot(const std::vector<T>& a, const std::vector<T>& b) {
    T s = T{};
    for (std::size_t i = 0; i < a.size(); ++i)
        s += a[i] * b[i];
    return s;
}

// y += alpha * x
template <typename T>
void axpy(T alpha, const std::vector<T>& x, std::vector<T>& y) {
    for (std::size_t i = 0; i < x.size(); ++i)
        y[i] += alpha * x[i];
}

// y = x + beta * y
template <typename T>
void xpby(const std::vector<T>& x, T beta, std::vector<T>& y) {
    for (std::size_t i = 0; i < x.size(); ++i)
        y[i] = x[i] + beta * y[i];
}

template <typename T>
T norm2(const std::vector<T>& v) {
    return std::sqrt(dot(v, v));
}

} // namespace detail

// ============================================================
// Conjugate Gradient (CG)
//
// Solves  A * x = b  where A is symmetric positive definite.
//
// Template parameter Matrix must expose:
//   void spmv(const std::vector<T>& x, std::vector<T>& y)  — y = A*x
//
// This requirement is satisfied by SparseCOO<T>, SparseCSR<T>,
// SparseCSC<T>, SparseELL<T>, SparseDIA<T>, and SparseMatrix<T>.
//
// x0:   initial guess; treated as all-zero when empty.
// opts: tolerance and iteration limit.
// ============================================================

template <typename Matrix, typename T = double>
SolverResult<T> cg(const Matrix&         A,
                   const std::vector<T>& b,
                   std::vector<T>        x0   = {},
                   SolverOptions         opts = {}) {
    const std::size_t n = b.size();

    if (x0.empty())
        x0.assign(n, T{});
    else if (x0.size() != n)
        throw std::invalid_argument("cg: x0 size must match b");

    const std::size_t max_iter = opts.max_iter ? opts.max_iter : n;
    const T           tol      = static_cast<T>(opts.tol);

    SolverResult<T> res;
    res.x = std::move(x0);

    // Trivial right-hand side
    const T b_norm = detail::norm2(b);
    if (b_norm == T{}) {
        res.x.assign(n, T{});
        res.converged = true;
        return res;
    }

    std::vector<T> Ap(n), r(n), p(n);

    // r₀ = b − A·x₀
    A.spmv(res.x, Ap);
    for (std::size_t i = 0; i < n; ++i)
        r[i] = b[i] - Ap[i];
    p = r;

    T rsold = detail::dot(r, r);

    // Already converged (e.g. warm start with the exact solution)
    res.residual_norm = std::sqrt(rsold) / b_norm;
    if (res.residual_norm <= tol) {
        res.converged = true;
        return res;
    }

    for (std::size_t k = 0; k < max_iter; ++k) {
        A.spmv(p, Ap);

        const T pAp = detail::dot(p, Ap);
        if (pAp == T{}) break; // p is in the null space of A — shouldn't happen for SPD
        const T alpha = rsold / pAp;

        detail::axpy( alpha, p,  res.x); // x  ←  x + α p
        detail::axpy(-alpha, Ap, r);     // r  ←  r − α Ap

        const T rsnew = detail::dot(r, r);
        res.residual_norm = std::sqrt(rsnew) / b_norm;
        ++res.iterations;

        if (res.residual_norm <= tol) {
            res.converged = true;
            break;
        }

        const T beta = rsnew / rsold;
        detail::xpby(r, beta, p); // p  ←  r + β p
        rsold = rsnew;
    }

    return res;
}

// ============================================================
// Preconditioned Conjugate Gradient (PCG)
//
// Solves  A * x = b  where A is SPD and M ≈ A is SPD.
// Preconditioner must expose:
//   void apply(const std::vector<T>& r, std::vector<T>& z)  — z = M⁻¹ r
//
// Use make_jacobi(A) to build a Jacobi (diagonal) preconditioner.
// When M = I (identity), PCG reduces to standard CG.
// ============================================================

template <typename Matrix, typename Preconditioner, typename T = double>
SolverResult<T> pcg(const Matrix&           A,
                    const std::vector<T>&   b,
                    const Preconditioner&   M,
                    std::vector<T>          x0   = {},
                    SolverOptions           opts = {}) {
    const std::size_t n = b.size();

    if (x0.empty())
        x0.assign(n, T{});
    else if (x0.size() != n)
        throw std::invalid_argument("pcg: x0 size must match b");

    const std::size_t max_iter = opts.max_iter ? opts.max_iter : n;
    const T           tol      = static_cast<T>(opts.tol);

    SolverResult<T> res;
    res.x = std::move(x0);

    const T b_norm = detail::norm2(b);
    if (b_norm == T{}) {
        res.x.assign(n, T{});
        res.converged = true;
        return res;
    }

    std::vector<T> Ap(n), r(n), z(n), p(n);

    // r₀ = b − A·x₀
    A.spmv(res.x, Ap);
    for (std::size_t i = 0; i < n; ++i)
        r[i] = b[i] - Ap[i];

    // Pre-check convergence (warm start)
    res.residual_norm = detail::norm2(r) / b_norm;
    if (res.residual_norm <= tol) {
        res.converged = true;
        return res;
    }

    M.apply(r, z);          // z₀ = M⁻¹ r₀
    p = z;
    T rz_old = detail::dot(r, z);

    for (std::size_t k = 0; k < max_iter; ++k) {
        A.spmv(p, Ap);
        const T pAp = detail::dot(p, Ap);
        if (pAp == T{}) break;
        const T alpha = rz_old / pAp;

        detail::axpy( alpha, p,  res.x); // x  ←  x + α p
        detail::axpy(-alpha, Ap, r);     // r  ←  r − α Ap

        res.residual_norm = detail::norm2(r) / b_norm;
        ++res.iterations;

        if (res.residual_norm <= tol) {
            res.converged = true;
            break;
        }

        M.apply(r, z);                    // z  ←  M⁻¹ r
        const T rz_new = detail::dot(r, z);
        const T beta   = rz_new / rz_old;
        detail::xpby(z, beta, p);         // p  ←  z + β p
        rz_old = rz_new;
    }

    return res;
}

// ============================================================
// Minimum Residual (MINRES)
//
// Solves  A * x = b  where A is symmetric (possibly indefinite).
//
// Uses the Lanczos process with Givens QR rotations to minimise
// ‖b − Ax‖ over the Krylov subspace K_k(A, r₀) at each step.
// Unlike CG, A need not be positive definite.
//
// Template parameter Matrix must expose:
//   void spmv(const std::vector<T>& x, std::vector<T>& y)  — y = A*x
//
// x0:   initial guess; treated as all-zero when empty.
// opts: tolerance and iteration limit.
// ============================================================

template <typename Matrix, typename T = double>
SolverResult<T> minres(const Matrix&         A,
                       const std::vector<T>& b,
                       std::vector<T>        x0   = {},
                       SolverOptions         opts = {}) {
    const std::size_t n = b.size();

    if (x0.empty())
        x0.assign(n, T{});
    else if (x0.size() != n)
        throw std::invalid_argument("minres: x0 size must match b");

    const std::size_t max_iter = opts.max_iter ? opts.max_iter : n;
    const T           tol      = static_cast<T>(opts.tol);

    SolverResult<T> res;
    res.x = std::move(x0);

    const T b_norm = detail::norm2(b);
    if (b_norm == T{}) {
        res.x.assign(n, T{});
        res.converged = true;
        return res;
    }

    // r = b − A·x₀
    std::vector<T> Ax(n);
    A.spmv(res.x, Ax);
    std::vector<T> r(n);
    for (std::size_t i = 0; i < n; ++i)
        r[i] = b[i] - Ax[i];

    T beta1 = detail::norm2(r);
    res.residual_norm = beta1 / b_norm;
    if (res.residual_norm <= tol) {
        res.converged = true;
        return res;
    }

    // Lanczos vectors: v_old = v_{k-1}, v = v_k
    std::vector<T> v(n), v_old(n, T{});
    for (std::size_t i = 0; i < n; ++i)
        v[i] = r[i] / beta1;

    // beta:   β_k used in the Lanczos recurrence (normalises v_k)
    // beta_T: β_k as it appears in column k of the Lanczos tridiagonal T
    //         (0 for k=1 — column 1 of T has no super-diagonal entry above it)
    T beta   = beta1;
    T beta_T = T{};

    // Two previous Givens rotations: G_{k-2} = (c_old2, s_old2),
    //                                G_{k-1} = (c_old, s_old)
    T c_old2 = T{1}, s_old2 = T{};
    T c_old  = T{1}, s_old  = T{};

    // Solution direction vectors: d_cur = d_{k-1}, d_old = d_{k-2}
    std::vector<T> d_cur(n, T{}), d_old(n, T{});

    // φ̄_k (signed): satisfies ‖r_{k-1}‖ = |φ̄_k|; drives the residual estimate.
    // Must carry the sign because phi_k = c_k φ̄_k and c_k can be negative.
    T phi_bar = beta1;

    std::vector<T> Av(n), v_new(n), d_new(n);

    for (std::size_t k = 0; k < max_iter; ++k) {
        // ---- Lanczos recurrence ----
        A.spmv(v, Av);
        const T alpha = detail::dot(v, Av);

        for (std::size_t i = 0; i < n; ++i)
            v_new[i] = Av[i] - alpha * v[i] - beta * v_old[i];

        T beta_new = detail::norm2(v_new);
        if (beta_new > T{}) {
            for (std::size_t i = 0; i < n; ++i)
                v_new[i] /= beta_new;
        }

        // ---- Apply G_{k-2}: fill-in ε_k and modified sub-diag δ̃_k ----
        // Column k of T has β_k (= beta_T) at row k-1 and α_k at row k.
        // G_{k-2} operates on rows k-2 and k-1, introducing ε_k at row k-2.
        const T eps         = s_old2 * beta_T;
        const T delta_tilde = c_old2 * beta_T;

        // ---- Apply G_{k-1}: compute diagonal δ_k and pre-rotation φ̃_k ----
        const T delta     = c_old * delta_tilde + s_old * alpha;
        const T phi_tilde = c_old * alpha        - s_old * delta_tilde;

        // ---- Compute G_k to zero β_{k+1} from φ̃_k ----
        const T gamma = std::sqrt(phi_tilde * phi_tilde + beta_new * beta_new);
        if (gamma == T{}) break;   // null-space: residual is already zero

        const T c = phi_tilde / gamma;   // cosine (may be negative for indefinite A)
        const T s = beta_new  / gamma;   // sine (≥ 0)

        // ---- Update direction: d_k = (v_k − ε_k d_{k-2} − δ_k d_{k-1}) / γ_k ----
        for (std::size_t i = 0; i < n; ++i)
            d_new[i] = (v[i] - eps * d_old[i] - delta * d_cur[i]) / gamma;

        // ---- Update solution: x += φ_k · d_k,  φ_k = c_k |φ̄_k| ----
        const T phi = c * phi_bar;
        detail::axpy(phi, d_new, res.x);

        // ---- Advance residual estimate: φ̄_{k+1} = −s_k φ̄_k ----
        phi_bar = -s * phi_bar;
        res.residual_norm = std::abs(phi_bar) / b_norm;
        ++res.iterations;

        if (res.residual_norm <= tol) {
            res.converged = true;
            break;
        }

        // ---- Advance Givens history and rolling vectors ----
        c_old2 = c_old; s_old2 = s_old;
        c_old  = c;     s_old  = s;

        std::swap(d_old, d_cur); std::swap(d_cur, d_new);
        std::swap(v_old, v);     std::swap(v, v_new);

        beta   = beta_new;
        beta_T = beta_new;
    }

    return res;
}

// ============================================================
// Generalized Minimum Residual (GMRES / GMRES(m))
//
// Solves  A * x = b  for general (possibly non-symmetric) A.
//
// Uses the Arnoldi process to build an orthonormal Krylov basis
// and minimises ‖b − Ax‖ over x₀ + Kₖ(A, r₀) at each step.
// Unlike CG and MINRES, A need not be symmetric.
//
// Memory per restart cycle is O(restart · n): all Arnoldi basis
// vectors must be kept for the back-solve.  For large problems,
// set restart to a modest value (e.g. 30) to bound memory.
//
// restart = 0   — full GMRES (one cycle of up to max_iter steps)
// restart = m>0 — GMRES(m): restart every m steps
//
// Template parameter Matrix must expose:
//   void spmv(const std::vector<T>& x, std::vector<T>& y)  — y = A*x
// ============================================================

template <typename Matrix, typename T = double>
SolverResult<T> gmres(const Matrix&         A,
                      const std::vector<T>& b,
                      std::vector<T>        x0      = {},
                      SolverOptions         opts    = {},
                      std::size_t           restart = 0) {
    const std::size_t n = b.size();

    if (x0.empty())
        x0.assign(n, T{});
    else if (x0.size() != n)
        throw std::invalid_argument("gmres: x0 size must match b");

    const std::size_t max_iter = opts.max_iter ? opts.max_iter : n;
    const T           tol      = static_cast<T>(opts.tol);
    // m: Arnoldi steps per restart cycle (= max_iter when no restart)
    const std::size_t m = (restart > 0) ? std::min(restart, max_iter) : max_iter;

    SolverResult<T> res;
    res.x = std::move(x0);

    const T b_norm = detail::norm2(b);
    if (b_norm == T{}) {
        res.x.assign(n, T{});
        res.converged = true;
        return res;
    }

    // Arnoldi basis V[0..m], each a vector of size n.
    std::vector<std::vector<T>> V(m + 1, std::vector<T>(n, T{}));

    // Upper Hessenberg stored column-major: H[j][i] = H_{i,j}.
    // Column j is filled during Arnoldi step j; G_j then zeroes H[j][j+1].
    // After all rotations: R_{i,j} = H[j][i] (used in the back-solve).
    std::vector<std::vector<T>> H(m, std::vector<T>(m + 1, T{}));

    std::vector<T> cs(m, T{}), sn(m, T{}); // Givens rotation parameters
    std::vector<T> g(m + 1, T{});           // Q^T [β; 0; …; 0] — transformed RHS
    std::vector<T> w(n);                    // scratch for Arnoldi
    std::vector<T> y;                       // back-solve result (sized per cycle)

    do {
        // ---- r = b − A·x,  β = ‖r‖ ----
        A.spmv(res.x, w);
        for (std::size_t i = 0; i < n; ++i) w[i] = b[i] - w[i];
        const T beta = detail::norm2(w);

        res.residual_norm = beta / b_norm;
        if (res.residual_norm <= tol) {
            res.converged = true;
            break;
        }

        // v₀ = r/β;  g = [β, 0, …, 0]
        const T inv_beta = T{1} / beta;
        for (std::size_t i = 0; i < n; ++i) V[0][i] = w[i] * inv_beta;
        std::fill(g.begin(), g.end(), T{});
        g[0] = beta;
        for (auto& col : H) std::fill(col.begin(), col.end(), T{});

        std::size_t k = 0; // Arnoldi steps completed this cycle
        for (; k < m && res.iterations < max_iter; ++k) {
            // ---- Arnoldi: w = A vₖ, modified Gram-Schmidt ----
            A.spmv(V[k], w);
            for (std::size_t j = 0; j <= k; ++j) {
                H[k][j] = detail::dot(w, V[j]);         // h_{j,k}
                for (std::size_t i = 0; i < n; ++i)
                    w[i] -= H[k][j] * V[j][i];
            }
            H[k][k + 1] = detail::norm2(w);             // h_{k+1,k}
            if (H[k][k + 1] > T{}) {
                const T inv = T{1} / H[k][k + 1];
                for (std::size_t i = 0; i < n; ++i)
                    V[k + 1][i] = w[i] * inv;
            }

            // ---- Apply previous rotations G₀,…,G_{k−1} to column k ----
            for (std::size_t j = 0; j < k; ++j) {
                const T h0  = H[k][j];
                const T h1  = H[k][j + 1];
                H[k][j]     =  cs[j] * h0 + sn[j] * h1;
                H[k][j + 1] = -sn[j] * h0 + cs[j] * h1;
            }

            // ---- Compute G_k to zero H[k][k+1] ----
            const T gamma = std::sqrt(H[k][k] * H[k][k] + H[k][k + 1] * H[k][k + 1]);
            if (gamma == T{}) break; // singular direction — stop here

            cs[k] = H[k][k]     / gamma;
            sn[k] = H[k][k + 1] / gamma;
            H[k][k]     = gamma;
            H[k][k + 1] = T{};

            // ---- Update transformed RHS (save g[k] before overwriting) ----
            const T gk = g[k];
            g[k]     =  cs[k] * gk;
            g[k + 1] = -sn[k] * gk;

            res.residual_norm = std::abs(g[k + 1]) / b_norm;
            ++res.iterations;

            if (res.residual_norm <= tol) {
                res.converged = true;
                ++k; // step k is complete; k now = number of completed steps
                break;
            }
        }

        // ---- Back-solve upper triangular R y = g[0..k−1] ----
        // R_{i,j} = H[j][i] for j ≥ i  (column-major H storage)
        y.assign(k, T{});
        for (std::size_t i = k; i-- > 0;) {
            y[i] = g[i];
            for (std::size_t j = i + 1; j < k; ++j)
                y[i] -= H[j][i] * y[j]; // R_{i,j} = H[j][i]
            y[i] /= H[i][i];            // R_{i,i} = H[i][i]
        }

        // ---- Update solution: x += V[:,0..k−1] · y ----
        for (std::size_t j = 0; j < k; ++j)
            detail::axpy(y[j], V[j], res.x);

    } while (restart > 0 && !res.converged && res.iterations < max_iter);

    return res;
}

// ============================================================
// Biconjugate Gradient Stabilized (BiCGSTAB)
//
// Solves  A * x = b  for general (possibly non-symmetric) A.
//
// Unlike GMRES, BiCGSTAB uses a short recurrence and requires
// only O(n) memory regardless of iteration count.  Each iteration
// performs exactly two SpMV operations (A·p and A·s).
//
// The algorithm can break down if the shadow residual r̃₀ becomes
// orthogonal to the current residual (ρ = r̃₀·r = 0) or if ω = 0
// (stagnation).  Both are detected and the solver exits early as
// unconverged.
//
// Template parameter Matrix must expose:
//   void spmv(const std::vector<T>& x, std::vector<T>& y)  — y = A*x
//
// x0:   initial guess; treated as all-zero when empty.
// opts: tolerance and iteration limit.
// ============================================================

template <typename Matrix, typename T = double>
SolverResult<T> bicgstab(const Matrix&         A,
                         const std::vector<T>& b,
                         std::vector<T>        x0   = {},
                         SolverOptions         opts = {}) {
    const std::size_t n = b.size();

    if (x0.empty())
        x0.assign(n, T{});
    else if (x0.size() != n)
        throw std::invalid_argument("bicgstab: x0 size must match b");

    const std::size_t max_iter = opts.max_iter ? opts.max_iter : n;
    const T           tol      = static_cast<T>(opts.tol);

    SolverResult<T> res;
    res.x = std::move(x0);

    const T b_norm = detail::norm2(b);
    if (b_norm == T{}) {
        res.x.assign(n, T{});
        res.converged = true;
        return res;
    }

    // r = b − A·x₀
    std::vector<T> Ax(n);
    A.spmv(res.x, Ax);
    std::vector<T> r(n);
    for (std::size_t i = 0; i < n; ++i)
        r[i] = b[i] - Ax[i];

    res.residual_norm = detail::norm2(r) / b_norm;
    if (res.residual_norm <= tol) {
        res.converged = true;
        return res;
    }

    // Shadow residual r̃₀ = r₀ (fixed; must satisfy r̃₀·r₀ ≠ 0)
    const std::vector<T> r_tilde(r);

    // Scalar state
    T rho_old = T{1};
    T alpha   = T{1};
    T omega   = T{1};

    // Vector state
    std::vector<T> v(n, T{});   // A·p from the previous iteration
    std::vector<T> p(n, T{});   // search direction
    std::vector<T> s(n), t(n);  // half-step residual and A·s

    for (std::size_t k = 0; k < max_iter; ++k) {
        // ---- ρ = r̃₀ · r ----
        const T rho = detail::dot(r_tilde, r);
        if (rho == T{}) break;                          // breakdown

        // ---- Update search direction: p = r + β(p − ω v) ----
        const T beta = (rho / rho_old) * (alpha / omega);
        for (std::size_t i = 0; i < n; ++i)
            p[i] = r[i] + beta * (p[i] - omega * v[i]);

        // ---- v = A·p,  α = ρ / (r̃₀·v) ----
        A.spmv(p, v);
        const T rtv = detail::dot(r_tilde, v);
        if (rtv == T{}) break;                          // breakdown
        alpha = rho / rtv;

        // ---- s = r − α·v ---- (half-step residual)
        for (std::size_t i = 0; i < n; ++i)
            s[i] = r[i] - alpha * v[i];

        // Half-step convergence: x += α·p and exit if ‖s‖ is small
        res.residual_norm = detail::norm2(s) / b_norm;
        if (res.residual_norm <= tol) {
            detail::axpy(alpha, p, res.x);
            res.converged = true;
            ++res.iterations;
            break;
        }

        // ---- t = A·s,  ω = (t·s) / (t·t) ----
        A.spmv(s, t);
        const T tt = detail::dot(t, t);
        if (tt == T{}) break;                           // s in null space of A
        omega = detail::dot(t, s) / tt;

        // ---- Full update: x += α·p + ω·s,  r = s − ω·t ----
        detail::axpy(alpha, p, res.x);
        detail::axpy(omega, s, res.x);
        for (std::size_t i = 0; i < n; ++i)
            r[i] = s[i] - omega * t[i];

        res.residual_norm = detail::norm2(r) / b_norm;
        ++res.iterations;

        if (res.residual_norm <= tol) {
            res.converged = true;
            break;
        }

        if (omega == T{}) break;                        // stagnation breakdown

        rho_old = rho;
    }

    return res;
}

} // namespace datamunge::linalg
