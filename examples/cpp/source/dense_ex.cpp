#include <datamunge/linalg/linalg.hpp>

#include <iomanip>
#include <iostream>

using namespace datamunge::linalg;

static void section(const char* title) {
    std::cout << "\n---- " << title << " ----\n";
}

static void print_vec(const std::vector<double>& v) {
    std::cout << "[";
    for (std::size_t i = 0; i < v.size(); ++i)
        std::cout << (i ? ", " : "") << std::setw(8) << std::setprecision(4) << v[i];
    std::cout << "]";
}

static void print_mat(const DenseMatrix<double>& M) {
    for (std::size_t i = 0; i < M.rows(); ++i) {
        std::cout << "  [";
        for (std::size_t j = 0; j < M.cols(); ++j)
            std::cout << std::setw(8) << std::setprecision(4)
                      << (j ? ", " : "") << M(i, j);
        std::cout << "]\n";
    }
}

int main() {
    // ============================================================
    section("DenseMatrix — construction and arithmetic");
    // ============================================================

    auto I3 = DenseMatrix<double>::identity(3);
    std::cout << "Identity(3):\n";
    print_mat(I3);

    auto Z = DenseMatrix<double>::zeros(2, 3);
    std::cout << "zeros(2,3):\n";
    print_mat(Z);

    DenseMatrix<double> A(2, 3, {1, 2, 3, 4, 5, 6});
    std::cout << "A (2x3) from initializer list:\n";
    print_mat(A);

    DenseMatrix<double> B(2, 3, {7, 8, 9, 10, 11, 12});
    auto C = A + B;
    std::cout << "A + B:\n";
    print_mat(C);

    DenseMatrix<double> S(2, 2, {1, 0, 0, 2});
    auto S2 = S * 2.0;
    std::cout << "S * 2.0:\n";
    print_mat(S2);

    DenseMatrix<double> P(2, 3, {1, 0, 0, 0, 1, 0});
    DenseMatrix<double> Q_ex(3, 2, {1, 2, 3, 4, 5, 6});
    auto PQ = P * Q_ex;
    std::cout << "P(2x3) * Q(3x2):\n";
    print_mat(PQ);

    std::vector<double> xv = {1, 2, 3};
    DenseMatrix<double> M3(2, 3, {1, 2, 3, 4, 5, 6});
    auto mv = M3 * xv;
    std::cout << "M3 * [1,2,3] = ";
    print_vec(mv);
    std::cout << "\n";

    auto At = A.transpose();
    std::cout << "A.transpose():\n";
    print_mat(At);

    std::cout << "norm_frobenius(A) = " << A.norm_frobenius() << "\n";
    std::cout << "trace(I3) = " << I3.trace() << "\n";

    // ============================================================
    section("LU decomposition");
    // ============================================================

    // 3×3 system: x = [2, 3, -1]
    DenseMatrix<double> Alu(3, 3, {2, 1, -1, -3, -1, 2, -2, 1, 2});
    std::vector<double> b_lu = {8, -11, -3};

    auto lu_dec = lu(Alu);
    std::cout << "det(A) = " << lu_dec.det() << "\n";

    auto x_lu = lu_dec.solve(b_lu);
    std::cout << "Solution x: ";
    print_vec(x_lu);
    std::cout << " (expected [2, 3, -1])\n";

    auto Ainv = lu_dec.inverse();
    std::cout << "A * inv(A):\n";
    print_mat(Alu * Ainv);

    // ============================================================
    section("Cholesky decomposition");
    // ============================================================

    DenseMatrix<double> Aspd(3, 3, {4, 2, 0, 2, 2, 1, 0, 1, 2});
    std::vector<double> b_ch = {4, 2, 2};

    auto ch_dec = cholesky(Aspd);
    std::cout << "Cholesky ok = " << std::boolalpha << ch_dec.ok << "\n";
    std::cout << "det(A) = " << ch_dec.det() << "\n";

    auto x_ch = ch_dec.solve(b_ch);
    std::cout << "Solution x: ";
    print_vec(x_ch);
    std::cout << " (expected [2, -2, 2])\n";

    // Verify L * L^T ≈ A
    auto LLt = ch_dec.L * ch_dec.L.transpose();
    std::cout << "L * L^T:\n";
    print_mat(LLt);

    // ============================================================
    section("QR decomposition — square system");
    // ============================================================

    auto qr_dec = qr(Alu);

    // Verify Q orthogonal
    auto QtQ = qr_dec.Q.transpose() * qr_dec.Q;
    auto I3b = DenseMatrix<double>::identity(3);
    std::cout << "||Q^T Q - I|| = " << (QtQ - I3b).norm_max() << "\n";

    // Verify Q*R = A
    auto QR_prod = qr_dec.Q * qr_dec.R;
    std::cout << "||Q*R - A|| = " << (QR_prod - Alu).norm_max() << "\n";

    auto x_qr = qr_dec.solve(b_lu);
    std::cout << "Solution x: ";
    print_vec(x_qr);
    std::cout << " (expected [2, 3, -1])\n";

    // ============================================================
    section("QR decomposition — least-squares");
    // ============================================================

    DenseMatrix<double> Als(3, 2, {1, 0, 0, 1, 1, 1});
    std::vector<double> b_ls = {1, 1, 1};
    auto qr_ls = qr(Als);
    auto x_ls  = qr_ls.solve(b_ls);
    std::cout << "Solution x: ";
    print_vec(x_ls);
    std::cout << " (expected [2/3, 2/3])\n";

    // Residual norm
    auto Ax_ls = Als * x_ls;
    double res2 = 0.0;
    for (std::size_t i = 0; i < b_ls.size(); ++i) {
        double r = Ax_ls[i] - b_ls[i];
        res2 += r * r;
    }
    std::cout << "Residual norm = " << std::sqrt(res2) << "\n";

    // ============================================================
    section("Dense matrix with iterative solvers");
    // ============================================================

    DenseMatrix<double> Aiter(4, 4, {
         4, -1,  0,  0,
        -1,  4, -1,  0,
         0, -1,  4, -1,
         0,  0, -1,  4
    });
    std::vector<double> b_iter = {1, 2, 3, 4};

    auto res_bicg = bicgstab(Aiter, b_iter);
    std::cout << "BiCGSTAB: converged=" << std::boolalpha << res_bicg.converged
              << "  iters=" << res_bicg.iterations
              << "  residual=" << res_bicg.residual_norm << "\n";
    std::cout << "Solution x: ";
    print_vec(res_bicg.x);
    std::cout << "\n";

    auto res_gmres = gmres(Aiter, b_iter);
    std::cout << "GMRES:     converged=" << std::boolalpha << res_gmres.converged
              << "  iters=" << res_gmres.iterations
              << "  residual=" << res_gmres.residual_norm << "\n";
    std::cout << "Solution x: ";
    print_vec(res_gmres.x);
    std::cout << "\n";

    return 0;
}
