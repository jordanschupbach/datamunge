#include <gtest/gtest.h>

#include <datamunge/linalg/dense.hpp>
#include <datamunge/linalg/solvers.hpp>

#include <cmath>
#include <vector>

using namespace datamunge::linalg;

// ============================================================
// DenseMatrix suite
// ============================================================

TEST(DenseMatrix, Zeros) {
    auto A = DenseMatrix<double>::zeros(2, 3);
    EXPECT_EQ(A.rows(), 2u);
    EXPECT_EQ(A.cols(), 3u);
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            EXPECT_EQ(A(i, j), 0.0);
}

TEST(DenseMatrix, Identity) {
    auto I = DenseMatrix<double>::identity(3);
    EXPECT_EQ(I.rows(), 3u);
    EXPECT_EQ(I.cols(), 3u);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            EXPECT_EQ(I(i, j), (i == j) ? 1.0 : 0.0);
    EXPECT_NEAR(I.trace(), 3.0, 1e-15);
}

TEST(DenseMatrix, FromVector) {
    std::vector<double> v = {1, 2, 3, 4};
    DenseMatrix<double> A(2, 2, v);
    EXPECT_EQ(A(0, 0), 1.0);
    EXPECT_EQ(A(0, 1), 2.0);
    EXPECT_EQ(A(1, 0), 3.0);
    EXPECT_EQ(A(1, 1), 4.0);
}

TEST(DenseMatrix, FromInitializerList) {
    DenseMatrix<double> A(2, 2, {1, 2, 3, 4});
    EXPECT_EQ(A(0, 0), 1.0);
    EXPECT_EQ(A(0, 1), 2.0);
    EXPECT_EQ(A(1, 0), 3.0);
    EXPECT_EQ(A(1, 1), 4.0);
}

TEST(DenseMatrix, DataSizeMismatch) {
    std::vector<double> bad = {1, 2, 3};
    EXPECT_THROW((DenseMatrix<double>(2, 2, bad)), std::invalid_argument);
}

TEST(DenseMatrix, AddSubtract) {
    DenseMatrix<double> A(2, 2, {1, 2, 3, 4});
    DenseMatrix<double> B(2, 2, {5, 6, 7, 8});
    auto C = A + B;
    EXPECT_EQ(C(0, 0), 6.0);
    EXPECT_EQ(C(0, 1), 8.0);
    EXPECT_EQ(C(1, 0), 10.0);
    EXPECT_EQ(C(1, 1), 12.0);
    auto D = B - A;
    EXPECT_EQ(D(0, 0), 4.0);
    EXPECT_EQ(D(0, 1), 4.0);
    EXPECT_EQ(D(1, 0), 4.0);
    EXPECT_EQ(D(1, 1), 4.0);
}

TEST(DenseMatrix, ScalarMultiply) {
    DenseMatrix<double> A(2, 2, {1, 2, 3, 4});
    auto B = A * 3.0;
    EXPECT_EQ(B(0, 0), 3.0);
    EXPECT_EQ(B(0, 1), 6.0);
    EXPECT_EQ(B(1, 0), 9.0);
    EXPECT_EQ(B(1, 1), 12.0);
    auto C = 3.0 * A;
    EXPECT_EQ(C(0, 0), 3.0);
    EXPECT_EQ(C(1, 1), 12.0);
}

TEST(DenseMatrix, ShapeMismatch) {
    DenseMatrix<double> A(2, 2, {1, 2, 3, 4});
    DenseMatrix<double> B(2, 3, {1, 2, 3, 4, 5, 6});
    EXPECT_THROW(A + B, std::invalid_argument);
}

TEST(DenseMatrix, Gemv) {
    DenseMatrix<double> A(2, 2, {1, 2, 3, 4});
    std::vector<double> x = {1, 1};
    auto y = A * x;
    EXPECT_NEAR(y[0], 3.0, 1e-15);
    EXPECT_NEAR(y[1], 7.0, 1e-15);
    // spmv gives same result
    std::vector<double> ys(2);
    A.spmv(x, ys);
    EXPECT_NEAR(ys[0], 3.0, 1e-15);
    EXPECT_NEAR(ys[1], 7.0, 1e-15);
}

TEST(DenseMatrix, Gemm) {
    auto I = DenseMatrix<double>::identity(2);
    DenseMatrix<double> B(2, 2, {2, 3, 4, 5});
    auto C = I * B;
    EXPECT_NEAR(C(0, 0), 2.0, 1e-15);
    EXPECT_NEAR(C(0, 1), 3.0, 1e-15);
    EXPECT_NEAR(C(1, 0), 4.0, 1e-15);
    EXPECT_NEAR(C(1, 1), 5.0, 1e-15);
}

TEST(DenseMatrix, GemmShapeMismatch) {
    DenseMatrix<double> A(2, 3, {1, 2, 3, 4, 5, 6});
    DenseMatrix<double> B(2, 3, {1, 2, 3, 4, 5, 6});
    EXPECT_THROW(A * B, std::invalid_argument);
}

TEST(DenseMatrix, Transpose) {
    DenseMatrix<double> A(2, 3, {1, 2, 3, 4, 5, 6});
    auto At = A.transpose();
    EXPECT_EQ(At.rows(), 3u);
    EXPECT_EQ(At.cols(), 2u);
    EXPECT_EQ(At(0, 0), 1.0);
    EXPECT_EQ(At(0, 1), 4.0);
    EXPECT_EQ(At(1, 0), 2.0);
    EXPECT_EQ(At(1, 1), 5.0);
    EXPECT_EQ(At(2, 0), 3.0);
    EXPECT_EQ(At(2, 1), 6.0);
}

TEST(DenseMatrix, NormFrobenius) {
    auto I = DenseMatrix<double>::identity(3);
    EXPECT_NEAR(I.norm_frobenius(), std::sqrt(3.0), 1e-15);
}

TEST(DenseMatrix, Norm1) {
    DenseMatrix<double> A(2, 2, {1, -2, 3, 4});
    // col 0: |1|+|3|=4, col 1: |-2|+|4|=6
    EXPECT_NEAR(A.norm_1(), 6.0, 1e-15);
}

TEST(DenseMatrix, NormInf) {
    DenseMatrix<double> A(2, 2, {1, -2, 3, 4});
    // row 0: |1|+|-2|=3, row 1: |3|+|4|=7
    EXPECT_NEAR(A.norm_inf(), 7.0, 1e-15);
}

TEST(DenseMatrix, NormMax) {
    DenseMatrix<double> A(2, 2, {1, -2, 3, 4});
    EXPECT_NEAR(A.norm_max(), 4.0, 1e-15);
}

TEST(DenseMatrix, TraceSquare) {
    auto I = DenseMatrix<double>::identity(4);
    EXPECT_NEAR(I.trace(), 4.0, 1e-15);
}

TEST(DenseMatrix, TraceNonSquare) {
    DenseMatrix<double> A(3, 2, {1, 0, 0, 1, 0, 0});
    EXPECT_THROW(A.trace(), std::invalid_argument);
}

TEST(DenseMatrix, SolvableWithGMRES) {
    // 3x3 non-symmetric dense system
    DenseMatrix<double> A(3, 3, {4, 1, 0, 1, 3, 1, 0, 1, 2});
    std::vector<double> b = {5, 5, 3};
    auto result = gmres(A, b);
    EXPECT_TRUE(result.converged);
    // Verify A*x ≈ b
    std::vector<double> Ax(3);
    A.spmv(result.x, Ax);
    for (std::size_t i = 0; i < 3; ++i)
        EXPECT_NEAR(Ax[i], b[i], 1e-7);
}

// ============================================================
// LU suite
// ============================================================

TEST(LU, KnownSolution) {
    // A = [[2,1,-1],[-3,-1,2],[-2,1,2]], b = [8,-11,-3] => x = [2,3,-1]
    DenseMatrix<double> A(3, 3, {2, 1, -1, -3, -1, 2, -2, 1, 2});
    std::vector<double> b = {8, -11, -3};
    auto dec = lu(A);
    auto x   = dec.solve(b);
    EXPECT_NEAR(x[0],  2.0, 1e-10);
    EXPECT_NEAR(x[1],  3.0, 1e-10);
    EXPECT_NEAR(x[2], -1.0, 1e-10);
}

TEST(LU, Determinant) {
    DenseMatrix<double> A(3, 3, {2, 1, -1, -3, -1, 2, -2, 1, 2});
    auto dec = lu(A);
    EXPECT_NEAR(dec.det(), -1.0, 1e-10);
}

TEST(LU, Inverse) {
    DenseMatrix<double> A(3, 3, {2, 1, -1, -3, -1, 2, -2, 1, 2});
    auto dec = lu(A);
    auto Ainv = dec.inverse();
    auto prod = A * Ainv;
    auto I3   = DenseMatrix<double>::identity(3);
    EXPECT_LT((prod - I3).norm_max(), 1e-10);
}

TEST(LU, SolveMultipleRHS) {
    DenseMatrix<double> A(3, 3, {2, 1, -1, -3, -1, 2, -2, 1, 2});
    std::vector<double> b = {8, -11, -3};
    // B has two identical columns
    DenseMatrix<double> B(3, 2);
    for (std::size_t i = 0; i < 3; ++i) { B(i, 0) = b[i]; B(i, 1) = b[i]; }
    auto dec = lu(A);
    auto X   = dec.solve(B);
    EXPECT_NEAR(X(0, 0),  2.0, 1e-10);
    EXPECT_NEAR(X(1, 0),  3.0, 1e-10);
    EXPECT_NEAR(X(2, 0), -1.0, 1e-10);
    EXPECT_NEAR(X(0, 1),  2.0, 1e-10);
    EXPECT_NEAR(X(1, 1),  3.0, 1e-10);
    EXPECT_NEAR(X(2, 1), -1.0, 1e-10);
}

TEST(LU, SingularMatrix) {
    DenseMatrix<double> A(2, 2, {1, 2, 2, 4});
    auto dec = lu(A);
    EXPECT_TRUE(dec.singular || std::abs(dec.det()) < 1e-10);
}

TEST(LU, NonSquareThrows) {
    DenseMatrix<double> A(3, 2, {1, 2, 3, 4, 5, 6});
    EXPECT_THROW(lu(A), std::invalid_argument);
}

// ============================================================
// Cholesky suite
// ============================================================

TEST(Cholesky, KnownSPD) {
    // A = [[4,2,0],[2,2,1],[0,1,2]], b = [4,2,2] => x = [2,-2,2]
    DenseMatrix<double> A(3, 3, {4, 2, 0, 2, 2, 1, 0, 1, 2});
    std::vector<double> b = {4, 2, 2};
    auto dec = cholesky(A);
    EXPECT_TRUE(dec.ok);
    auto x = dec.solve(b);
    EXPECT_NEAR(x[0],  2.0, 1e-10);
    EXPECT_NEAR(x[1], -2.0, 1e-10);
    EXPECT_NEAR(x[2],  2.0, 1e-10);
}

TEST(Cholesky, Determinant) {
    // det(A) = (2*1*1)^2 = 4
    DenseMatrix<double> A(3, 3, {4, 2, 0, 2, 2, 1, 0, 1, 2});
    auto dec = cholesky(A);
    EXPECT_NEAR(dec.det(), 4.0, 1e-10);
}

TEST(Cholesky, SolveMultipleRHS) {
    DenseMatrix<double> A(3, 3, {4, 2, 0, 2, 2, 1, 0, 1, 2});
    std::vector<double> b = {4, 2, 2};
    DenseMatrix<double> B(3, 2);
    for (std::size_t i = 0; i < 3; ++i) { B(i, 0) = b[i]; B(i, 1) = b[i]; }
    auto dec = cholesky(A);
    auto X   = dec.solve(B);
    EXPECT_NEAR(X(0, 0),  2.0, 1e-10);
    EXPECT_NEAR(X(1, 0), -2.0, 1e-10);
    EXPECT_NEAR(X(2, 0),  2.0, 1e-10);
    EXPECT_NEAR(X(0, 1),  2.0, 1e-10);
    EXPECT_NEAR(X(1, 1), -2.0, 1e-10);
    EXPECT_NEAR(X(2, 1),  2.0, 1e-10);
}

TEST(Cholesky, NotSPD) {
    DenseMatrix<double> A(2, 2, {1, 0, 0, -1});
    auto dec = cholesky(A);
    EXPECT_FALSE(dec.ok);
}

TEST(Cholesky, NonSquareThrows) {
    DenseMatrix<double> A(3, 2, {1, 0, 0, 1, 0, 0});
    EXPECT_THROW(cholesky(A), std::invalid_argument);
}

// ============================================================
// QR suite
// ============================================================

TEST(QR, SquareExact) {
    // A = [[2,1,-1],[-3,-1,2],[-2,1,2]], b = [8,-11,-3] => x = [2,3,-1]
    DenseMatrix<double> A(3, 3, {2, 1, -1, -3, -1, 2, -2, 1, 2});
    std::vector<double> b = {8, -11, -3};
    auto dec = qr(A);

    // Check Q*R ≈ A
    auto QR_prod = dec.Q * dec.R;
    EXPECT_LT((QR_prod - A).norm_max(), 1e-10);

    // Check Q orthogonal: Q^T * Q ≈ I
    auto QtQ = dec.Q.transpose() * dec.Q;
    auto I3  = DenseMatrix<double>::identity(3);
    EXPECT_LT((QtQ - I3).norm_max(), 1e-10);

    // Solve
    auto x = dec.solve(b);
    EXPECT_NEAR(x[0],  2.0, 1e-7);
    EXPECT_NEAR(x[1],  3.0, 1e-7);
    EXPECT_NEAR(x[2], -1.0, 1e-7);
}

TEST(QR, LeastSquares) {
    // A = [[1,0],[0,1],[1,1]], b = [1,1,1] => x = [2/3, 2/3]
    DenseMatrix<double> A(3, 2, {1, 0, 0, 1, 1, 1});
    std::vector<double> b = {1, 1, 1};
    auto dec = qr(A);
    auto x   = dec.solve(b);
    EXPECT_NEAR(x[0], 2.0 / 3.0, 1e-10);
    EXPECT_NEAR(x[1], 2.0 / 3.0, 1e-10);
}

TEST(QR, IsFullRank) {
    DenseMatrix<double> A(3, 2, {1, 0, 0, 1, 1, 1});
    auto dec = qr(A);
    EXPECT_TRUE(dec.is_full_rank());
}

TEST(QR, SolveMultipleRHS) {
    DenseMatrix<double> A(3, 2, {1, 0, 0, 1, 1, 1});
    std::vector<double> b = {1, 1, 1};
    DenseMatrix<double> B(3, 2);
    for (std::size_t i = 0; i < 3; ++i) { B(i, 0) = b[i]; B(i, 1) = b[i]; }
    auto dec = qr(A);
    auto X   = dec.solve(B);
    EXPECT_NEAR(X(0, 0), 2.0 / 3.0, 1e-10);
    EXPECT_NEAR(X(1, 0), 2.0 / 3.0, 1e-10);
    EXPECT_NEAR(X(0, 1), 2.0 / 3.0, 1e-10);
    EXPECT_NEAR(X(1, 1), 2.0 / 3.0, 1e-10);
}

TEST(QR, NDLessThanMThrows) {
    DenseMatrix<double> A(2, 3, {1, 2, 3, 4, 5, 6});
    EXPECT_THROW(qr(A), std::invalid_argument);
}
