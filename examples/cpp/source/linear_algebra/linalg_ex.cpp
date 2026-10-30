#include <datamunge/linalg/linalg.hpp>

#include <iostream>
#include <vector>

using namespace datamunge::linalg;

static void print_vec(const std::vector<double>& v) {
    std::cout << "[";
    for (std::size_t i = 0; i < v.size(); ++i)
        std::cout << (i ? ", " : "") << v[i];
    std::cout << "]";
}

static void section(const char* title) {
    std::cout << "\n---- " << title << " ----\n";
}

int main() {
    // ============================================================
    // Build a 4×4 test matrix in COO format
    //
    //   A = [ 1  0  2  0 ]
    //       [ 0  3  0  0 ]
    //       [ 4  0  0  5 ]
    //       [ 0  0  0  6 ]
    // ============================================================
    SparseCOO<double> A(4, 4, 6);
    A.set(0, 0, 1.0);
    A.set(0, 2, 2.0);
    A.set(1, 1, 3.0);
    A.set(2, 0, 4.0);
    A.set(2, 3, 5.0);
    A.set(3, 3, 6.0);
    A.compress();

    const std::vector<double> x = {1.0, 2.0, 3.0, 4.0};
    // Expected: A*x = [1+6, 6, 4+20, 24] = [7, 6, 24, 24]

    // ============================================================
    // COO — allocating and buffer-reuse SpMV, tiled variant
    // ============================================================
    section("COO");
    std::cout << "A: " << A.rows() << "x" << A.cols() << "  nnz=" << A.nnz() << "\n";
    std::cout << "x = "; print_vec(x); std::cout << "\n";

    auto y_coo = A.spmv(x);
    std::cout << "A*x (allocating)   = "; print_vec(y_coo); std::cout << "\n";

    std::vector<double> y_buf(A.rows());
    A.spmv(x, y_buf);
    std::cout << "A*x (buffer reuse) = "; print_vec(y_buf); std::cout << "\n";

    A.spmv_tiled(x, y_buf, 2);
    std::cout << "A*x (tiled r=2)    = "; print_vec(y_buf); std::cout << "\n";

    // ============================================================
    // CSR — built from COO; inner-loop reads are sequential
    // ============================================================
    section("CSR");
    auto csr = to_csr(A);
    std::cout << "CSR: nrows=" << csr.nrows << " ncols=" << csr.ncols
              << " nnz=" << csr.nnz() << "\n";
    std::cout << "row_ptr: [";
    for (std::size_t i = 0; i <= csr.nrows; ++i)
        std::cout << (i ? ", " : "") << csr.row_ptr[i];
    std::cout << "]\n";

    auto y_csr = csr.spmv(x);
    std::cout << "A*x (CSR)          = "; print_vec(y_csr); std::cout << "\n";

    csr.spmv(x, y_buf);
    std::cout << "A*x (CSR buf)      = "; print_vec(y_buf); std::cout << "\n";

    // ============================================================
    // CSC — built from COO; col_ptr mirrors CSR's row_ptr
    // ============================================================
    section("CSC");
    auto csc = to_csc(A);
    std::cout << "CSC: nnz=" << csc.nnz() << "\n";
    std::cout << "col_ptr: [";
    for (std::size_t j = 0; j <= csc.ncols; ++j)
        std::cout << (j ? ", " : "") << csc.col_ptr[j];
    std::cout << "]\n";

    auto y_csc = csc.spmv(x);
    std::cout << "A*x (CSC)          = "; print_vec(y_csc); std::cout << "\n";

    // ============================================================
    // ELL — padded to max_nnz_per_row; shows padding overhead
    // ============================================================
    section("ELL");
    auto ell = to_ell(A);
    std::cout << "ELL: max_nnz_per_row=" << ell.max_nnz_per_row
              << " nnz=" << ell.nnz()
              << " nnz_padded=" << ell.nnz_padded() << "\n";

    auto y_ell = ell.spmv(x);
    std::cout << "A*x (ELL)          = "; print_vec(y_ell); std::cout << "\n";

    // ============================================================
    // DIA — best for banded matrices; build a 5×5 tridiagonal
    //
    //   B = [ 2 -1  0  0  0 ]
    //       [-1  2 -1  0  0 ]
    //       [ 0 -1  2 -1  0 ]
    //       [ 0  0 -1  2 -1 ]
    //       [ 0  0  0 -1  2 ]
    // ============================================================
    section("DIA (tridiagonal)");
    SparseCOO<double> B(5, 5, 13);
    for (std::size_t i = 0; i < 5; ++i) {
        B.set(i, i, 2.0);
        if (i > 0)     B.set(i, i - 1, -1.0);
        if (i < 4)     B.set(i, i + 1, -1.0);
    }
    B.compress();

    auto dia = to_dia(B);
    std::cout << "DIA: num_diags=" << dia.num_diags()
              << " offsets=[";
    for (std::size_t d = 0; d < dia.offsets.size(); ++d)
        std::cout << (d ? ", " : "") << dia.offsets[d];
    std::cout << "]\n";

    const std::vector<double> xb = {1.0, 1.0, 1.0, 1.0, 1.0};
    auto y_dia = dia.spmv(xb);
    // B*[1,1,1,1,1] = [1, 0, 0, 0, 1]
    std::cout << "B*[1,1,1,1,1] (DIA) = "; print_vec(y_dia); std::cout << "\n";

    // ============================================================
    // Round-trips: verify each format → COO → matches original
    // ============================================================
    section("Round-trips (format → COO)");
    auto coo_from_csr = to_coo(csr);
    auto coo_from_csc = to_coo(csc);
    auto coo_from_ell = to_coo(ell);
    auto coo_from_dia = to_coo(dia);

    std::cout << "COO←CSR nnz=" << coo_from_csr.nnz()
              << "  A*x = "; print_vec(coo_from_csr.spmv(x)); std::cout << "\n";
    std::cout << "COO←CSC nnz=" << coo_from_csc.nnz()
              << "  A*x = "; print_vec(coo_from_csc.spmv(x)); std::cout << "\n";
    std::cout << "COO←ELL nnz=" << coo_from_ell.nnz()
              << "  A*x = "; print_vec(coo_from_ell.spmv(x)); std::cout << "\n";
    std::cout << "COO←DIA nnz=" << coo_from_dia.nnz()
              << "  B*x = "; print_vec(coo_from_dia.spmv(xb)); std::cout << "\n";

    // ============================================================
    // SparseMatrix<T> — format-polymorphic wrapper
    // ============================================================
    section("SparseMatrix (format-polymorphic wrapper)");

    SparseMatrix<double> mat(std::move(A));   // starts as COO
    std::cout << "format=COO  A*x = "; print_vec(mat.spmv(x)); std::cout << "\n";

    mat.convert_to(SparseMatrix<double>::Format::CSR);
    std::cout << "format=CSR  A*x = "; print_vec(mat.spmv(x)); std::cout << "\n";
    std::cout << "  as_csr()->nnz() = " << mat.as_csr()->nnz() << "\n";

    mat.convert_to(SparseMatrix<double>::Format::CSC);
    std::cout << "format=CSC  A*x = "; print_vec(mat.spmv(x)); std::cout << "\n";

    mat.convert_to(SparseMatrix<double>::Format::ELL);
    std::cout << "format=ELL  A*x = "; print_vec(mat.spmv(x)); std::cout << "\n";
    std::cout << "  as_ell()=" << (mat.as_ell() ? "ok" : "null")
              << "  as_csr()=" << (mat.as_csr() ? "ok" : "null") << "\n";

    mat.convert_to(SparseMatrix<double>::Format::DIA);
    std::cout << "format=DIA  A*x = "; print_vec(mat.spmv(x)); std::cout << "\n";

    mat.convert_to(SparseMatrix<double>::Format::CSR);  // back to CSR (via COO intermediate)
    std::cout << "format=CSR (re-converted)  A*x = "; print_vec(mat.spmv(x)); std::cout << "\n";

    // ============================================================
    // Expression templates still work — result feeds directly into SparseMatrix
    // ============================================================
    section("Expression templates → SparseMatrix");
    SparseCOO<double> P(3, 3), Q(3, 3);
    P.set_batch({0, 1, 2}, {0, 1, 2}, {1.0, 2.0, 3.0});
    Q.set_batch({0, 1, 2}, {2, 0, 1}, {4.0, 5.0, 6.0});

    SparseMatrix<double> mat2(SparseCOO<double>(2.0 * P + Q));
    mat2.convert_to(SparseMatrix<double>::Format::CSR);
    const std::vector<double> x3 = {1.0, 1.0, 1.0};
    std::cout << "(2P+Q)*[1,1,1] via CSR = "; print_vec(mat2.spmv(x3)); std::cout << "\n";

    // ============================================================
    // Conjugate Gradient solver — Ax = b for SPD matrices
    //
    // Solve the 5×5 tridiagonal system:
    //   [ 2 -1  0  0  0 ] [x0]   [1]
    //   [-1  2 -1  0  0 ] [x1]   [0]
    //   [ 0 -1  2 -1  0 ] [x2] = [0]
    //   [ 0  0 -1  2 -1 ] [x3]   [0]
    //   [ 0  0  0 -1  2 ] [x4]   [1]
    // ============================================================
    section("Conjugate Gradient (CG) solver");

    // Build the matrix in COO and use it directly (no conversion needed)
    SparseCOO<double> T5(5, 5, 13);
    for (std::size_t i = 0; i < 5; ++i) {
        T5.set(i, i, 2.0);
        if (i > 0) T5.set(i, i - 1, -1.0);
        if (i < 4) T5.set(i, i + 1, -1.0);
    }
    T5.compress();

    const std::vector<double> rhs = {1.0, 0.0, 0.0, 0.0, 1.0};
    auto sol = cg(T5, rhs);
    std::cout << "converged=" << std::boolalpha << sol.converged
              << "  iterations=" << sol.iterations
              << "  rel_residual=" << sol.residual_norm << "\n";
    std::cout << "x = "; print_vec(sol.x); std::cout << "\n";

    // Verify: the exact solution to this symmetric system is x = [1,1,1,1,1]
    // (T5 * [1,1,1,1,1] = [2-1, -1+2-1, -1+2-1, -1+2-1, -1+2] = [1,0,0,0,1] ✓)
    std::cout << "T5 * x = "; print_vec(T5.spmv(sol.x)); std::cout << "  (should equal rhs)\n";

    // CG works with any format — here with CSR after converting:
    auto T5_csr = to_csr(T5);
    auto sol_csr = cg(T5_csr, rhs);
    std::cout << "CSR solution: "; print_vec(sol_csr.x); std::cout << "\n";

    // SolverOptions: tighter tolerance, limited iterations
    SolverOptions opts;
    opts.tol      = 1e-12;
    opts.max_iter = 2;
    auto sol_limited = cg(T5, rhs, {}, opts);
    std::cout << "max_iter=2 → converged=" << sol_limited.converged
              << "  iterations=" << sol_limited.iterations
              << "  rel_residual=" << sol_limited.residual_norm << "\n";

    // ============================================================
    // Jacobi preconditioner + PCG
    //
    // For a diagonal matrix the Jacobi preconditioner is exact
    // (M⁻¹A = I), so PCG converges in 1 iteration regardless of
    // how many distinct eigenvalues the original matrix has.
    // ============================================================
    section("Preconditioned CG (PCG) with Jacobi preconditioner");

    // Build a diagonal matrix with 5 distinct eigenvalues:
    // D = diag(1, 2, 3, 4, 5),  b = [1, 2, 3, 4, 5]  → x* = [1, 1, 1, 1, 1]
    // Plain CG needs up to 5 iterations (one per distinct eigenvalue).
    // PCG with Jacobi (M = D) collapses to 1.
    SparseCOO<double> D(5, 5, 5);
    std::vector<double> bd = {1.0, 2.0, 3.0, 4.0, 5.0};
    for (std::size_t i = 0; i < 5; ++i)
        D.set(i, i, bd[i]);
    D.compress();

    auto cg_sol  = cg(D, bd);
    auto jac     = make_jacobi(D);
    auto pcg_sol = pcg(D, bd, jac);

    std::cout << "CG:  iterations=" << cg_sol.iterations
              << "  residual=" << cg_sol.residual_norm << "\n";
    std::cout << "PCG: iterations=" << pcg_sol.iterations
              << "  residual=" << pcg_sol.residual_norm << "\n";
    std::cout << "PCG x = "; print_vec(pcg_sol.x); std::cout << "  (expect all 1s)\n";

    // Jacobi also works with every matrix format via make_jacobi()
    auto T5_csr2  = to_csr(T5);
    auto jac_csr  = make_jacobi(T5_csr2);
    auto pcg_csr  = pcg(T5_csr2, rhs, jac_csr);
    std::cout << "PCG (CSR, tridiag): iterations=" << pcg_csr.iterations
              << "  x = "; print_vec(pcg_csr.x); std::cout << "\n";

    // ============================================================
    // MINRES — minimum residual for symmetric (possibly indefinite) A
    //
    // CG minimises the A-norm of the error and requires A to be SPD.
    // MINRES minimises ‖r‖ and only requires symmetry, so it handles
    // indefinite matrices where CG breaks down.
    //
    // SPD case: MINRES and CG recover the same solution.
    //
    // Indefinite case:
    //   A = [ 4  1  0 ]    eigenvalues ≈ 4.317, 4, -2.317
    //       [ 1 -2  1 ]    b = [5, 0, 5],  x* = [1, 1, 1]
    //       [ 0  1  4 ]
    // ============================================================
    section("MINRES solver");

    // SPD system — same 5×5 tridiagonal as in the CG section
    auto minres_sol = minres(T5, rhs);
    std::cout << "MINRES (SPD tridiag): converged=" << minres_sol.converged
              << "  iterations=" << minres_sol.iterations
              << "  rel_residual=" << minres_sol.residual_norm << "\n";
    std::cout << "x = "; print_vec(minres_sol.x); std::cout << "\n";

    // Symmetric indefinite system
    SparseCOO<double> Aindef(3, 3, 7);
    Aindef.set(0, 0,  4.0); Aindef.set(0, 1,  1.0);
    Aindef.set(1, 0,  1.0); Aindef.set(1, 1, -2.0); Aindef.set(1, 2, 1.0);
    Aindef.set(2, 1,  1.0); Aindef.set(2, 2,  4.0);
    Aindef.compress();

    const std::vector<double> b_indef = {5.0, 0.0, 5.0};
    auto sol_indef = minres(Aindef, b_indef);
    std::cout << "MINRES (indefinite):  converged=" << sol_indef.converged
              << "  iterations=" << sol_indef.iterations
              << "  rel_residual=" << sol_indef.residual_norm << "\n";
    std::cout << "x = "; print_vec(sol_indef.x);
    std::cout << "  (expect [1, 1, 1])\n";

    // MINRES also accepts any matrix format
    auto T5_csr3 = to_csr(T5);
    auto sol_csr3 = minres(T5_csr3, rhs);
    std::cout << "MINRES (CSR): "; print_vec(sol_csr3.x); std::cout << "\n";

    // ============================================================
    // BiCGSTAB — short-recurrence solver for non-symmetric A
    //
    // Like GMRES, handles non-symmetric A.  Unlike GMRES, uses
    // O(n) memory (no basis storage) and exactly 2 SpMVs per
    // iteration.  Trade-off: convergence can be less smooth than
    // GMRES for highly non-normal matrices.
    //
    //   A = [ 4  1  0 ]  (non-symmetric),  b=[5,6,4],  x*=[1,1,1]
    //       [ 2  3  1 ]
    //       [ 0  1  3 ]
    // ============================================================
    section("BiCGSTAB solver");

    SparseCOO<double> Ansym(3, 3, 7);
    Ansym.set(0, 0, 4.0); Ansym.set(0, 1, 1.0);
    Ansym.set(1, 0, 2.0); Ansym.set(1, 1, 3.0); Ansym.set(1, 2, 1.0);
    Ansym.set(2, 1, 1.0); Ansym.set(2, 2, 3.0);
    Ansym.compress();
    const std::vector<double> b_nsym = {5.0, 6.0, 4.0};

    auto sol_bicg = bicgstab(Ansym, b_nsym);
    std::cout << "BiCGSTAB (non-symmetric 3×3): converged=" << sol_bicg.converged
              << "  iterations=" << sol_bicg.iterations
              << "  rel_residual=" << sol_bicg.residual_norm << "\n";
    std::cout << "x = "; print_vec(sol_bicg.x);
    std::cout << "  (expect [1, 1, 1])\n";

    // BiCGSTAB also solves SPD systems; uses 2 SpMVs vs CG's 1
    auto sol_bicg_spd = bicgstab(T5, rhs);
    std::cout << "BiCGSTAB (SPD tridiag):       x = "; print_vec(sol_bicg_spd.x);
    std::cout << "  (same as CG)\n";

    // ============================================================
    // GMRES — generalized minimum residual for non-symmetric A
    //
    // CG and MINRES require symmetry.  GMRES works for any
    // non-singular A by building a full Arnoldi basis, but stores
    // all basis vectors: O(m·n) memory per restart cycle.
    //
    // Non-symmetric 3×3 system:
    //   A = [ 4  1  0 ]
    //       [ 2  3  1 ]   (A₁₀ ≠ A₀₁ — not symmetric)
    //       [ 0  1  3 ]
    //   b = [5, 6, 4],  x* = [1, 1, 1]
    //
    // Restarted GMRES(5) on a 20×20 upper-bidiagonal system.
    // ============================================================
    section("GMRES solver");

    auto sol_nsym = gmres(Ansym, b_nsym);
    std::cout << "GMRES (non-symmetric 3×3): converged=" << sol_nsym.converged
              << "  iterations=" << sol_nsym.iterations
              << "  rel_residual=" << sol_nsym.residual_norm << "\n";
    std::cout << "x = "; print_vec(sol_nsym.x);
    std::cout << "  (expect [1, 1, 1])\n";

    // Restarted GMRES(5) — bounds memory at O(5n) per cycle
    constexpr std::size_t Nbidi = 20;
    SparseCOO<double> Abidi(Nbidi, Nbidi, 2 * Nbidi - 1);
    std::vector<double> b_bidi(Nbidi);
    for (std::size_t i = 0; i < Nbidi; ++i) {
        Abidi.set(i, i, 3.0);
        if (i + 1 < Nbidi) Abidi.set(i, i + 1, 1.0);
        b_bidi[i] = (i + 1 < Nbidi) ? 4.0 : 3.0;
    }
    Abidi.compress();

    auto sol_restart = gmres(Abidi, b_bidi, {}, {}, /*restart=*/5);
    std::cout << "GMRES(5) (20×20 bidiag):   converged=" << sol_restart.converged
              << "  iterations=" << sol_restart.iterations
              << "  rel_residual=" << sol_restart.residual_norm << "\n";
    std::cout << "x = "; print_vec(sol_restart.x); std::cout << "\n";

    // Full GMRES also solves SPD systems — same answer as CG
    auto sol_spd = gmres(T5, rhs);
    std::cout << "GMRES (SPD tridiag):       x = "; print_vec(sol_spd.x);
    std::cout << "  (matches CG)\n";

    return 0;
}
