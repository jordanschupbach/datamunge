#include <gtest/gtest.h>

#include <datamunge/linalg/linalg.hpp>

#include <cmath>
#include <vector>

using namespace datamunge::linalg;

// ============================================================
// Helpers
// ============================================================

static SparseCOO<double> make_tridiag(std::size_t n) {
    SparseCOO<double> A(n, n, 3 * n);
    for (std::size_t i = 0; i < n; ++i) {
        A.set(i, i, 2.0);
        if (i > 0)     A.set(i, i - 1, -1.0);
        if (i + 1 < n) A.set(i, i + 1, -1.0);
    }
    A.compress();
    return A;
}

static SparseCOO<double> make_identity(std::size_t n) {
    SparseCOO<double> I(n, n, n);
    for (std::size_t i = 0; i < n; ++i)
        I.set(i, i, 1.0);
    I.compress();
    return I;
}

static double residual_norm(const SparseCOO<double>& A,
                            const std::vector<double>& x,
                            const std::vector<double>& b) {
    std::vector<double> Ax(b.size());
    A.spmv(x, Ax);
    double r = 0.0, bn = 0.0;
    for (std::size_t i = 0; i < b.size(); ++i) {
        r  += (Ax[i] - b[i]) * (Ax[i] - b[i]);
        bn += b[i] * b[i];
    }
    return std::sqrt(r) / std::sqrt(bn);
}

// ============================================================
// Basic convergence
// ============================================================

TEST(CG, IdentityConvergesInOneIteration) {
    auto I = make_identity(5);
    std::vector<double> b = {1.0, 2.0, 3.0, 4.0, 5.0};
    auto res = cg(I, b);
    EXPECT_TRUE(res.converged);
    EXPECT_LE(res.iterations, 1u);
    for (std::size_t i = 0; i < b.size(); ++i)
        EXPECT_NEAR(res.x[i], b[i], 1e-10);
}

TEST(CG, Tridiag5Converges) {
    auto A = make_tridiag(5);
    std::vector<double> b = {1.0, 0.0, 0.0, 0.0, 1.0};
    auto res = cg(A, b);
    EXPECT_TRUE(res.converged);
    EXPECT_LE(res.residual_norm, 1e-8);
    EXPECT_LE(residual_norm(A, res.x, b), 1e-8);
}

TEST(CG, Tridiag100Converges) {
    auto A = make_tridiag(100);
    std::vector<double> b(100, 1.0);
    auto res = cg(A, b);
    EXPECT_TRUE(res.converged);
    EXPECT_LE(res.residual_norm, 1e-8);
    EXPECT_LT(res.iterations, 100u);
}

TEST(CG, ZeroRHSReturnsZeroImmediately) {
    auto A = make_tridiag(10);
    std::vector<double> b(10, 0.0);
    auto res = cg(A, b);
    EXPECT_TRUE(res.converged);
    EXPECT_EQ(res.iterations, 0u);
    for (double v : res.x)
        EXPECT_EQ(v, 0.0);
}

// ============================================================
// Initial guess
// ============================================================

TEST(CG, CustomInitialGuessSameAnswer) {
    auto A = make_tridiag(20);
    std::vector<double> b(20, 1.0);

    // cold start
    auto r0 = cg(A, b);
    EXPECT_TRUE(r0.converged);

    // warm start from the converged solution: residual is already below tol,
    // so the solver returns immediately (0 extra iterations)
    auto r1 = cg(A, b, r0.x);
    EXPECT_TRUE(r1.converged);
    EXPECT_EQ(r1.iterations, 0u);
    for (std::size_t i = 0; i < b.size(); ++i)
        EXPECT_NEAR(r1.x[i], r0.x[i], 1e-8);
}

TEST(CG, InitialGuessWrongSizeThrows) {
    auto A = make_identity(4);
    std::vector<double> b(4, 1.0);
    std::vector<double> bad_x0(3, 0.0);
    EXPECT_THROW(cg(A, b, bad_x0), std::invalid_argument);
}

// ============================================================
// Options
// ============================================================

TEST(CG, MaxIterLimitStopsEarly) {
    auto A = make_tridiag(50);
    std::vector<double> b(50, 1.0);
    SolverOptions opts;
    opts.max_iter = 3;
    auto res = cg(A, b, {}, opts);
    EXPECT_FALSE(res.converged);
    EXPECT_EQ(res.iterations, 3u);
}

TEST(CG, TighterToleranceRequiresMoreIterations) {
    auto A = make_tridiag(30);
    std::vector<double> b(30, 1.0);

    SolverOptions loose, tight;
    loose.tol = 1e-4;
    tight.tol = 1e-10;

    auto r_loose = cg(A, b, {}, loose);
    auto r_tight = cg(A, b, {}, tight);

    EXPECT_TRUE(r_loose.converged);
    EXPECT_TRUE(r_tight.converged);
    EXPECT_LE(r_loose.iterations, r_tight.iterations);
    EXPECT_LE(r_tight.residual_norm, r_loose.residual_norm);
}

// ============================================================
// Format interoperability
// ============================================================

TEST(CG, WorksWithCSR) {
    auto coo = make_tridiag(30);
    auto csr = to_csr(coo);
    std::vector<double> b(30, 1.0);
    auto res = cg(csr, b);
    EXPECT_TRUE(res.converged);
    EXPECT_LE(res.residual_norm, 1e-8);
}

TEST(CG, WorksWithCSC) {
    auto coo = make_tridiag(30);
    auto csc = to_csc(coo);
    std::vector<double> b(30, 1.0);
    auto res = cg(csc, b);
    EXPECT_TRUE(res.converged);
    EXPECT_LE(res.residual_norm, 1e-8);
}

TEST(CG, WorksWithELL) {
    auto coo = make_tridiag(30);
    auto ell = to_ell(coo);
    std::vector<double> b(30, 1.0);
    auto res = cg(ell, b);
    EXPECT_TRUE(res.converged);
    EXPECT_LE(res.residual_norm, 1e-8);
}

TEST(CG, WorksWithDIA) {
    auto coo = make_tridiag(30);
    auto dia = to_dia(coo);
    std::vector<double> b(30, 1.0);
    auto res = cg(dia, b);
    EXPECT_TRUE(res.converged);
    EXPECT_LE(res.residual_norm, 1e-8);
}

TEST(CG, WorksWithSparseMatrix) {
    auto coo = make_tridiag(30);
    SparseMatrix<double> mat(coo);
    std::vector<double> b(30, 1.0);
    auto res = cg(mat, b);
    EXPECT_TRUE(res.converged);
    EXPECT_LE(res.residual_norm, 1e-8);

    // Same result after conversion to CSR
    mat.convert_to(SparseMatrix<double>::Format::CSR);
    auto res_csr = cg(mat, b);
    EXPECT_TRUE(res_csr.converged);
    for (std::size_t i = 0; i < b.size(); ++i)
        EXPECT_NEAR(res_csr.x[i], res.x[i], 1e-8);
}

// ============================================================
// Correctness: verify A*x ≈ b for a known small system
//
//  A = [ 4  1 ]   b = [ 1 ]   x* = [ 3/23, -1/23 ]
//      [ 1  3 ]       [ 0 ]
// ============================================================
TEST(CG, SmallKnownSolution) {
    SparseCOO<double> A(2, 2, 4);
    A.set(0, 0, 4.0);
    A.set(0, 1, 1.0);
    A.set(1, 0, 1.0);
    A.set(1, 1, 3.0);
    A.compress();

    std::vector<double> b = {1.0, 0.0};
    auto res = cg(A, b);
    EXPECT_TRUE(res.converged);
    EXPECT_NEAR(res.x[0],  3.0 / 11.0, 1e-10);
    EXPECT_NEAR(res.x[1], -1.0 / 11.0, 1e-10);
}

// ============================================================
// Scaled identity — all eigenvalues equal → CG converges in 1 iteration.
// A diagonal matrix with distinct eigenvalues needs up to n iterations;
// here we use 3*I so there is exactly one distinct eigenvalue.
// ============================================================
TEST(CG, ScaledIdentityConvergesInOneIteration) {
    constexpr std::size_t n = 10;
    SparseCOO<double> A(n, n, n);
    for (std::size_t i = 0; i < n; ++i)
        A.set(i, i, 3.0);
    A.compress();

    std::vector<double> b(n);
    for (std::size_t i = 0; i < n; ++i)
        b[i] = static_cast<double>(i + 1);

    auto res = cg(A, b);
    EXPECT_TRUE(res.converged);
    EXPECT_EQ(res.iterations, 1u);
    for (std::size_t i = 0; i < n; ++i)
        EXPECT_NEAR(res.x[i], b[i] / 3.0, 1e-10);
}

// ============================================================
// diagonal() — extraction from each format
// ============================================================

static SparseCOO<double> make_4x4() {
    //  A = [ 1  0  2  0 ]
    //      [ 0  3  0  0 ]
    //      [ 4  0  5  0 ]   diagonal = [1, 3, 5, 6]
    //      [ 0  0  0  6 ]
    SparseCOO<double> A(4, 4, 7);
    A.set(0, 0, 1.0); A.set(0, 2, 2.0);
    A.set(1, 1, 3.0);
    A.set(2, 0, 4.0); A.set(2, 2, 5.0);
    A.set(3, 3, 6.0);
    A.compress();
    return A;
}

TEST(Diagonal, COO) {
    auto d = diagonal(make_4x4());
    ASSERT_EQ(d.size(), 4u);
    EXPECT_DOUBLE_EQ(d[0], 1.0);
    EXPECT_DOUBLE_EQ(d[1], 3.0);
    EXPECT_DOUBLE_EQ(d[2], 5.0);
    EXPECT_DOUBLE_EQ(d[3], 6.0);
}

TEST(Diagonal, CSR) {
    auto d = diagonal(to_csr(make_4x4()));
    ASSERT_EQ(d.size(), 4u);
    EXPECT_DOUBLE_EQ(d[0], 1.0);
    EXPECT_DOUBLE_EQ(d[1], 3.0);
    EXPECT_DOUBLE_EQ(d[2], 5.0);
    EXPECT_DOUBLE_EQ(d[3], 6.0);
}

TEST(Diagonal, CSC) {
    auto d = diagonal(to_csc(make_4x4()));
    ASSERT_EQ(d.size(), 4u);
    EXPECT_DOUBLE_EQ(d[0], 1.0);
    EXPECT_DOUBLE_EQ(d[1], 3.0);
    EXPECT_DOUBLE_EQ(d[2], 5.0);
    EXPECT_DOUBLE_EQ(d[3], 6.0);
}

TEST(Diagonal, ELL) {
    auto d = diagonal(to_ell(make_4x4()));
    ASSERT_EQ(d.size(), 4u);
    EXPECT_DOUBLE_EQ(d[0], 1.0);
    EXPECT_DOUBLE_EQ(d[1], 3.0);
    EXPECT_DOUBLE_EQ(d[2], 5.0);
    EXPECT_DOUBLE_EQ(d[3], 6.0);
}

TEST(Diagonal, DIA) {
    // Tridiagonal is a clean DIA matrix with a well-defined main diagonal
    auto coo = make_tridiag(5);
    auto d = diagonal(to_dia(coo));
    ASSERT_EQ(d.size(), 5u);
    for (double v : d)
        EXPECT_DOUBLE_EQ(v, 2.0);
}

TEST(Diagonal, SparseMatrixCOO) {
    SparseMatrix<double> mat(make_4x4());
    auto d = diagonal(mat);
    ASSERT_EQ(d.size(), 4u);
    EXPECT_DOUBLE_EQ(d[2], 5.0);
}

TEST(Diagonal, SparseMatrixCSR) {
    SparseMatrix<double> mat(make_4x4());
    mat.convert_to(SparseMatrix<double>::Format::CSR);
    auto d = diagonal(mat);
    ASSERT_EQ(d.size(), 4u);
    EXPECT_DOUBLE_EQ(d[2], 5.0);
}

TEST(Diagonal, MissingDiagonalEntryIsZero) {
    SparseCOO<double> A(3, 3, 2);
    A.set(0, 1, 7.0);  // no diagonal entries
    A.set(1, 0, 7.0);
    A.compress();
    auto d = diagonal(A);
    for (double v : d)
        EXPECT_DOUBLE_EQ(v, 0.0);
}

// ============================================================
// JacobiPreconditioner — make_jacobi and apply
// ============================================================

TEST(Jacobi, InvDiagIsReciprocalOfDiagonal) {
    auto A = make_4x4();
    auto P = make_jacobi(A);
    ASSERT_EQ(P.inv_diag.size(), 4u);
    EXPECT_DOUBLE_EQ(P.inv_diag[0], 1.0 / 1.0);
    EXPECT_DOUBLE_EQ(P.inv_diag[1], 1.0 / 3.0);
    EXPECT_DOUBLE_EQ(P.inv_diag[2], 1.0 / 5.0);
    EXPECT_DOUBLE_EQ(P.inv_diag[3], 1.0 / 6.0);
}

TEST(Jacobi, ZeroDiagonalFallsBackToOne) {
    SparseCOO<double> A(2, 2, 1);
    A.set(0, 1, 5.0);  // diagonal entries absent → treated as zero
    A.compress();
    auto P = make_jacobi(A);
    EXPECT_DOUBLE_EQ(P.inv_diag[0], 1.0);
    EXPECT_DOUBLE_EQ(P.inv_diag[1], 1.0);
}

TEST(Jacobi, ApplyScalesCorrectly) {
    auto P = make_jacobi(make_tridiag(5)); // diagonal is all 2s
    std::vector<double> r = {2.0, 4.0, 6.0, 8.0, 10.0};
    std::vector<double> z(5);
    P.apply(r, z);
    for (std::size_t i = 0; i < 5; ++i)
        EXPECT_DOUBLE_EQ(z[i], r[i] / 2.0);
}

// ============================================================
// PCG — correctness and iteration count
// ============================================================

TEST(PCG, Tridiag5Converges) {
    auto A = make_tridiag(5);
    auto P = make_jacobi(A);
    std::vector<double> b = {1.0, 0.0, 0.0, 0.0, 1.0};
    auto res = pcg(A, b, P);
    EXPECT_TRUE(res.converged);
    EXPECT_LE(res.residual_norm, 1e-8);
}

TEST(PCG, MatchesCGSolution) {
    auto A = make_tridiag(50);
    auto P = make_jacobi(A);
    std::vector<double> b(50, 1.0);
    auto cg_res  = cg(A, b);
    auto pcg_res = pcg(A, b, P);
    EXPECT_TRUE(pcg_res.converged);
    for (std::size_t i = 0; i < b.size(); ++i)
        EXPECT_NEAR(pcg_res.x[i], cg_res.x[i], 1e-8);
}

TEST(PCG, DiagonalSystemConvergesInOneIteration) {
    // For a pure diagonal matrix, Jacobi preconditioner is exact:
    // M^{-1} A = I  →  PCG converges in exactly 1 iteration.
    constexpr std::size_t n = 20;
    SparseCOO<double> D(n, n, n);
    std::vector<double> b(n);
    for (std::size_t i = 0; i < n; ++i) {
        D.set(i, i, static_cast<double>(i + 1));
        b[i] = static_cast<double>(i + 1);
    }
    D.compress();

    auto P   = make_jacobi(D);
    auto res = pcg(D, b, P);
    EXPECT_TRUE(res.converged);
    EXPECT_EQ(res.iterations, 1u);
    for (std::size_t i = 0; i < n; ++i)
        EXPECT_NEAR(res.x[i], 1.0, 1e-10);
}

TEST(PCG, ZeroRHSReturnsZero) {
    auto A = make_tridiag(10);
    auto P = make_jacobi(A);
    std::vector<double> b(10, 0.0);
    auto res = pcg(A, b, P);
    EXPECT_TRUE(res.converged);
    EXPECT_EQ(res.iterations, 0u);
}

TEST(PCG, WarmStartFromSolutionReturnsImmediately) {
    auto A = make_tridiag(20);
    auto P = make_jacobi(A);
    std::vector<double> b(20, 1.0);
    auto r0 = pcg(A, b, P);
    EXPECT_TRUE(r0.converged);
    auto r1 = pcg(A, b, P, r0.x);
    EXPECT_TRUE(r1.converged);
    EXPECT_EQ(r1.iterations, 0u);
}

TEST(PCG, WorksWithCSR) {
    auto coo = make_tridiag(30);
    auto csr = to_csr(coo);
    auto P   = make_jacobi(csr);
    std::vector<double> b(30, 1.0);
    auto res = pcg(csr, b, P);
    EXPECT_TRUE(res.converged);
    EXPECT_LE(res.residual_norm, 1e-8);
}

TEST(PCG, WorksWithSparseMatrix) {
    auto coo = make_tridiag(30);
    SparseMatrix<double> mat(coo);
    auto P = make_jacobi(mat);
    std::vector<double> b(30, 1.0);
    auto res = pcg(mat, b, P);
    EXPECT_TRUE(res.converged);
    EXPECT_LE(res.residual_norm, 1e-8);
}

TEST(PCG, MaxIterLimitStopsEarly) {
    auto A = make_tridiag(50);
    auto P = make_jacobi(A);
    std::vector<double> b(50, 1.0);
    SolverOptions opts;
    opts.max_iter = 3;
    auto res = pcg(A, b, P, {}, opts);
    EXPECT_FALSE(res.converged);
    EXPECT_EQ(res.iterations, 3u);
}

// ============================================================
// MINRES — minimum residual for symmetric (possibly indefinite) A
// ============================================================

// Build a small symmetric indefinite matrix:
//
//   A = [ 4  1  0 ]
//       [ 1 -2  1 ]   eigenvalues ≈ 4.317, 4, -2.317  (indefinite)
//       [ 0  1  4 ]
//
//   b = [ 5  0  5 ]  →  exact solution x* = [1, 1, 1]
//
static SparseCOO<double> make_sym_indefinite() {
    SparseCOO<double> A(3, 3, 7);
    A.set(0, 0,  4.0); A.set(0, 1,  1.0);
    A.set(1, 0,  1.0); A.set(1, 1, -2.0); A.set(1, 2, 1.0);
    A.set(2, 1,  1.0); A.set(2, 2,  4.0);
    A.compress();
    return A;
}

TEST(MINRES, IdentityConvergesQuickly) {
    auto I = make_identity(5);
    std::vector<double> b = {1.0, 2.0, 3.0, 4.0, 5.0};
    auto res = minres(I, b);
    EXPECT_TRUE(res.converged);
    for (std::size_t i = 0; i < b.size(); ++i)
        EXPECT_NEAR(res.x[i], b[i], 1e-10);
}

TEST(MINRES, Tridiag5Converges) {
    auto A = make_tridiag(5);
    std::vector<double> b = {1.0, 0.0, 0.0, 0.0, 1.0};
    auto res = minres(A, b);
    EXPECT_TRUE(res.converged);
    EXPECT_LE(res.residual_norm, 1e-8);
    EXPECT_LE(residual_norm(A, res.x, b), 1e-8);
}

TEST(MINRES, MatchesCGSolution) {
    // For SPD systems MINRES and CG compute the same solution.
    auto A = make_tridiag(50);
    std::vector<double> b(50, 1.0);
    auto cg_res     = cg(A, b);
    auto minres_res = minres(A, b);
    EXPECT_TRUE(minres_res.converged);
    for (std::size_t i = 0; i < b.size(); ++i)
        EXPECT_NEAR(minres_res.x[i], cg_res.x[i], 1e-7);
}

TEST(MINRES, SymmetricIndefinite) {
    // MINRES handles indefinite symmetric systems; CG does not.
    auto A = make_sym_indefinite();
    std::vector<double> b = {5.0, 0.0, 5.0};
    auto res = minres(A, b);
    EXPECT_TRUE(res.converged);
    EXPECT_NEAR(res.x[0], 1.0, 1e-8);
    EXPECT_NEAR(res.x[1], 1.0, 1e-8);
    EXPECT_NEAR(res.x[2], 1.0, 1e-8);
}

TEST(MINRES, ZeroRHSReturnsZeroImmediately) {
    auto A = make_tridiag(10);
    std::vector<double> b(10, 0.0);
    auto res = minres(A, b);
    EXPECT_TRUE(res.converged);
    EXPECT_EQ(res.iterations, 0u);
    for (double v : res.x)
        EXPECT_EQ(v, 0.0);
}

TEST(MINRES, WarmStartReturnsImmediately) {
    auto A = make_tridiag(20);
    std::vector<double> b(20, 1.0);
    auto r0 = minres(A, b);
    EXPECT_TRUE(r0.converged);
    auto r1 = minres(A, b, r0.x);
    EXPECT_TRUE(r1.converged);
    EXPECT_EQ(r1.iterations, 0u);
}

TEST(MINRES, InitialGuessWrongSizeThrows) {
    auto A = make_identity(4);
    std::vector<double> b(4, 1.0);
    std::vector<double> bad_x0(3, 0.0);
    EXPECT_THROW(minres(A, b, bad_x0), std::invalid_argument);
}

TEST(MINRES, MaxIterLimitStopsEarly) {
    auto A = make_tridiag(50);
    std::vector<double> b(50, 1.0);
    SolverOptions opts;
    opts.max_iter = 3;
    auto res = minres(A, b, {}, opts);
    EXPECT_FALSE(res.converged);
    EXPECT_EQ(res.iterations, 3u);
}

TEST(MINRES, WorksWithCSR) {
    auto coo = make_tridiag(30);
    auto csr = to_csr(coo);
    std::vector<double> b(30, 1.0);
    auto res = minres(csr, b);
    EXPECT_TRUE(res.converged);
    EXPECT_LE(res.residual_norm, 1e-8);
}

TEST(MINRES, WorksWithSparseMatrix) {
    auto coo = make_tridiag(30);
    SparseMatrix<double> mat(coo);
    std::vector<double> b(30, 1.0);
    auto res = minres(mat, b);
    EXPECT_TRUE(res.converged);
    EXPECT_LE(res.residual_norm, 1e-8);
}

// ============================================================
// GMRES — general non-symmetric systems
// ============================================================

// Non-symmetric 3×3 matrix:
//
//   A = [ 4  1  0 ]
//       [ 2  3  1 ]   (A[1][0]=2 ≠ A[0][1]=1 — not symmetric)
//       [ 0  1  3 ]
//
//   b = [ 5  6  4 ]  →  exact solution x* = [1, 1, 1]
static SparseCOO<double> make_nonsym3() {
    SparseCOO<double> A(3, 3, 7);
    A.set(0, 0, 4.0); A.set(0, 1, 1.0);
    A.set(1, 0, 2.0); A.set(1, 1, 3.0); A.set(1, 2, 1.0);
    A.set(2, 1, 1.0); A.set(2, 2, 3.0);
    A.compress();
    return A;
}

TEST(GMRES, IdentityConvergesInOneIteration) {
    auto I = make_identity(5);
    std::vector<double> b = {1.0, 2.0, 3.0, 4.0, 5.0};
    auto res = gmres(I, b);
    EXPECT_TRUE(res.converged);
    EXPECT_LE(res.iterations, 1u);
    for (std::size_t i = 0; i < b.size(); ++i)
        EXPECT_NEAR(res.x[i], b[i], 1e-10);
}

TEST(GMRES, Tridiag5Converges) {
    auto A = make_tridiag(5);
    std::vector<double> b = {1.0, 0.0, 0.0, 0.0, 1.0};
    auto res = gmres(A, b);
    EXPECT_TRUE(res.converged);
    EXPECT_LE(res.residual_norm, 1e-8);
    EXPECT_LE(residual_norm(A, res.x, b), 1e-8);
}

TEST(GMRES, MatchesCGSolution) {
    // For SPD systems GMRES and CG should find the same solution.
    auto A = make_tridiag(30);
    std::vector<double> b(30, 1.0);
    auto cg_res   = cg(A, b);
    auto gmres_res = gmres(A, b);
    EXPECT_TRUE(gmres_res.converged);
    for (std::size_t i = 0; i < b.size(); ++i)
        EXPECT_NEAR(gmres_res.x[i], cg_res.x[i], 1e-7);
}

TEST(GMRES, NonSymmetricSystem) {
    // CG and MINRES require symmetry; GMRES handles non-symmetric A.
    auto A = make_nonsym3();
    std::vector<double> b = {5.0, 6.0, 4.0};
    auto res = gmres(A, b);
    EXPECT_TRUE(res.converged);
    EXPECT_NEAR(res.x[0], 1.0, 1e-8);
    EXPECT_NEAR(res.x[1], 1.0, 1e-8);
    EXPECT_NEAR(res.x[2], 1.0, 1e-8);
}

TEST(GMRES, ZeroRHSReturnsZeroImmediately) {
    auto A = make_tridiag(10);
    std::vector<double> b(10, 0.0);
    auto res = gmres(A, b);
    EXPECT_TRUE(res.converged);
    EXPECT_EQ(res.iterations, 0u);
    for (double v : res.x)
        EXPECT_EQ(v, 0.0);
}

TEST(GMRES, WarmStartReturnsImmediately) {
    auto A = make_tridiag(20);
    std::vector<double> b(20, 1.0);
    auto r0 = gmres(A, b);
    EXPECT_TRUE(r0.converged);
    auto r1 = gmres(A, b, r0.x);
    EXPECT_TRUE(r1.converged);
    EXPECT_EQ(r1.iterations, 0u);
}

TEST(GMRES, InitialGuessWrongSizeThrows) {
    auto A = make_identity(4);
    std::vector<double> b(4, 1.0);
    std::vector<double> bad_x0(3, 0.0);
    EXPECT_THROW(gmres(A, b, bad_x0), std::invalid_argument);
}

TEST(GMRES, MaxIterLimitStopsEarly) {
    auto A = make_tridiag(50);
    std::vector<double> b(50, 1.0);
    SolverOptions opts;
    opts.max_iter = 3;
    auto res = gmres(A, b, {}, opts);
    EXPECT_FALSE(res.converged);
    EXPECT_EQ(res.iterations, 3u);
}

TEST(GMRES, WorksWithCSR) {
    auto coo = make_tridiag(30);
    auto csr = to_csr(coo);
    std::vector<double> b(30, 1.0);
    auto res = gmres(csr, b);
    EXPECT_TRUE(res.converged);
    EXPECT_LE(res.residual_norm, 1e-8);
}

TEST(GMRES, WorksWithSparseMatrix) {
    auto coo = make_tridiag(30);
    SparseMatrix<double> mat(coo);
    std::vector<double> b(30, 1.0);
    auto res = gmres(mat, b);
    EXPECT_TRUE(res.converged);
    EXPECT_LE(res.residual_norm, 1e-8);
}

TEST(GMRES, RestartedConverges) {
    // GMRES(5) on a 20×20 non-symmetric upper-bidiagonal system.
    // A_{i,i} = 3,  A_{i,i+1} = 1  →  b = [4,4,...,4,3],  x* = [1,...,1]
    constexpr std::size_t n = 20;
    SparseCOO<double> A(n, n, 2 * n - 1);
    std::vector<double> b(n);
    for (std::size_t i = 0; i < n; ++i) {
        A.set(i, i, 3.0);
        if (i + 1 < n) A.set(i, i + 1, 1.0);
        b[i] = (i + 1 < n) ? 4.0 : 3.0;
    }
    A.compress();

    auto res = gmres(A, b, {}, {}, /*restart=*/5);
    EXPECT_TRUE(res.converged);
    for (std::size_t i = 0; i < n; ++i)
        EXPECT_NEAR(res.x[i], 1.0, 1e-7);
}

// ============================================================
// BiCGSTAB — short-recurrence solver for non-symmetric A
// ============================================================

TEST(BiCGSTAB, IdentityConvergesInOneIteration) {
    auto I = make_identity(5);
    std::vector<double> b = {1.0, 2.0, 3.0, 4.0, 5.0};
    auto res = bicgstab(I, b);
    EXPECT_TRUE(res.converged);
    EXPECT_LE(res.iterations, 1u);
    for (std::size_t i = 0; i < b.size(); ++i)
        EXPECT_NEAR(res.x[i], b[i], 1e-10);
}

TEST(BiCGSTAB, Tridiag5Converges) {
    auto A = make_tridiag(5);
    std::vector<double> b = {1.0, 0.0, 0.0, 0.0, 1.0};
    auto res = bicgstab(A, b);
    EXPECT_TRUE(res.converged);
    EXPECT_LE(res.residual_norm, 1e-8);
    EXPECT_LE(residual_norm(A, res.x, b), 1e-8);
}

TEST(BiCGSTAB, Tridiag100Converges) {
    auto A = make_tridiag(100);
    std::vector<double> b(100, 1.0);
    auto res = bicgstab(A, b);
    EXPECT_TRUE(res.converged);
    EXPECT_LE(res.residual_norm, 1e-8);
}

TEST(BiCGSTAB, MatchesCGSolution) {
    auto A = make_tridiag(30);
    std::vector<double> b(30, 1.0);
    auto cg_res      = cg(A, b);
    auto bicg_res    = bicgstab(A, b);
    EXPECT_TRUE(bicg_res.converged);
    for (std::size_t i = 0; i < b.size(); ++i)
        EXPECT_NEAR(bicg_res.x[i], cg_res.x[i], 1e-7);
}

TEST(BiCGSTAB, NonSymmetricSystem) {
    // Same 3×3 non-symmetric matrix used in GMRES tests.
    auto A = make_nonsym3();
    std::vector<double> b = {5.0, 6.0, 4.0};
    auto res = bicgstab(A, b);
    EXPECT_TRUE(res.converged);
    EXPECT_NEAR(res.x[0], 1.0, 1e-8);
    EXPECT_NEAR(res.x[1], 1.0, 1e-8);
    EXPECT_NEAR(res.x[2], 1.0, 1e-8);
}

TEST(BiCGSTAB, ZeroRHSReturnsZeroImmediately) {
    auto A = make_tridiag(10);
    std::vector<double> b(10, 0.0);
    auto res = bicgstab(A, b);
    EXPECT_TRUE(res.converged);
    EXPECT_EQ(res.iterations, 0u);
    for (double v : res.x)
        EXPECT_EQ(v, 0.0);
}

TEST(BiCGSTAB, WarmStartReturnsImmediately) {
    auto A = make_tridiag(20);
    std::vector<double> b(20, 1.0);
    auto r0 = bicgstab(A, b);
    EXPECT_TRUE(r0.converged);
    auto r1 = bicgstab(A, b, r0.x);
    EXPECT_TRUE(r1.converged);
    EXPECT_EQ(r1.iterations, 0u);
}

TEST(BiCGSTAB, InitialGuessWrongSizeThrows) {
    auto A = make_identity(4);
    std::vector<double> b(4, 1.0);
    std::vector<double> bad_x0(3, 0.0);
    EXPECT_THROW(bicgstab(A, b, bad_x0), std::invalid_argument);
}

TEST(BiCGSTAB, MaxIterLimitStopsEarly) {
    auto A = make_tridiag(50);
    std::vector<double> b(50, 1.0);
    SolverOptions opts;
    opts.max_iter = 3;
    auto res = bicgstab(A, b, {}, opts);
    EXPECT_FALSE(res.converged);
    EXPECT_LE(res.iterations, 3u);
}

TEST(BiCGSTAB, WorksWithCSR) {
    auto coo = make_tridiag(30);
    auto csr = to_csr(coo);
    std::vector<double> b(30, 1.0);
    auto res = bicgstab(csr, b);
    EXPECT_TRUE(res.converged);
    EXPECT_LE(res.residual_norm, 1e-8);
}

TEST(BiCGSTAB, WorksWithSparseMatrix) {
    auto coo = make_tridiag(30);
    SparseMatrix<double> mat(coo);
    std::vector<double> b(30, 1.0);
    auto res = bicgstab(mat, b);
    EXPECT_TRUE(res.converged);
    EXPECT_LE(res.residual_norm, 1e-8);
}

TEST(GMRES, RestartedNonSymmetric) {
    // GMRES(2) on the 3×3 non-symmetric system.
    // restart < n means restarting discards Krylov history, so more total
    // iterations are needed than full GMRES — raise max_iter accordingly.
    auto A = make_nonsym3();
    std::vector<double> b = {5.0, 6.0, 4.0};
    SolverOptions opts;
    opts.max_iter = 100;
    auto res = gmres(A, b, {}, opts, /*restart=*/2);
    EXPECT_TRUE(res.converged);
    EXPECT_NEAR(res.x[0], 1.0, 1e-7);
    EXPECT_NEAR(res.x[1], 1.0, 1e-7);
    EXPECT_NEAR(res.x[2], 1.0, 1e-7);
}
