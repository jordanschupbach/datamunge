#include <gtest/gtest.h>

#include <datamunge/linalg/linalg.hpp>

#include <cmath>

using namespace datamunge::linalg;

TEST(JacobiEigen, TwoByTwoKnownEigenpairs) {
    DenseMatrix<double> A(2, 2, {2, 1, 1, 2});
    const auto result = jacobi_eigen(A);

    EXPECT_NEAR(result.eigenvalues[0], 3.0, 1e-10);
    EXPECT_NEAR(result.eigenvalues[1], 1.0, 1e-10);

    for (std::size_t k = 0; k < 2; ++k) {
        const double vx = result.eigenvectors(0, k);
        const double vy = result.eigenvectors(1, k);
        const double ax = A(0, 0) * vx + A(0, 1) * vy;
        const double ay = A(1, 0) * vx + A(1, 1) * vy;
        EXPECT_NEAR(ax, result.eigenvalues[k] * vx, 1e-10);
        EXPECT_NEAR(ay, result.eigenvalues[k] * vy, 1e-10);
        EXPECT_NEAR(vx * vx + vy * vy, 1.0, 1e-10);
    }
}

TEST(JacobiEigen, DiagonalMatrixSortedDescending) {
    DenseMatrix<double> D(4, 4, 0.0);
    D(0, 0) = 3;
    D(1, 1) = 1;
    D(2, 2) = 4;
    D(3, 3) = 2;

    const auto result = jacobi_eigen(D);
    const std::vector<double> expected = {4, 3, 2, 1};
    ASSERT_EQ(result.eigenvalues.size(), 4u);
    for (std::size_t i = 0; i < 4; ++i)
        EXPECT_NEAR(result.eigenvalues[i], expected[i], 1e-10);
}

TEST(JacobiEigen, FiveByFiveRandomSymmetricMatrixSatisfiesEigenrelation) {
    DenseMatrix<double> A(5, 5, {
        4, 1, 2, 0, 1,
        1, 3, 0, 1, 2,
        2, 0, 5, 1, 0,
        0, 1, 1, 6, 1,
        1, 2, 0, 1, 4,
    });
    const auto result = jacobi_eigen(A);

    for (std::size_t k = 0; k < 5; ++k) {
        std::vector<double> Av(5, 0.0);
        for (std::size_t i = 0; i < 5; ++i)
            for (std::size_t j = 0; j < 5; ++j)
                Av[i] += A(i, j) * result.eigenvectors(j, k);
        for (std::size_t i = 0; i < 5; ++i)
            EXPECT_NEAR(Av[i], result.eigenvalues[k] * result.eigenvectors(i, k), 1e-8);
    }

    // Eigenvalues should be sorted descending.
    for (std::size_t k = 1; k < 5; ++k)
        EXPECT_GE(result.eigenvalues[k - 1], result.eigenvalues[k]);

    // Trace should equal the sum of eigenvalues.
    double trace = 0.0;
    for (std::size_t i = 0; i < 5; ++i) trace += A(i, i);
    double eigen_sum = 0.0;
    for (double v : result.eigenvalues) eigen_sum += v;
    EXPECT_NEAR(trace, eigen_sum, 1e-8);
}
