#include <gtest/gtest.h>

#include <datamunge/fda/bspline.hpp>
#include <datamunge/linalg/csr.hpp>

#include <cstddef>
#include <vector>

using datamunge::fda::BSplineBasis;

namespace {

double dense_from_sparse(const datamunge::linalg::SparseCOO<double>& matrix,
                         std::size_t                                 row,
                         std::size_t                                 col) {
    for (std::size_t k = 0; k < matrix.nnz(); ++k) {
        if (matrix.row_indices()[k] == row && matrix.col_indices()[k] == col)
            return matrix.values()[k];
    }
    return 0.0;
}

} // namespace

TEST(BSplineBasis, OpenUniformBuildsClampedKnotVector) {
    const auto basis = BSplineBasis::open_uniform(2, 5, -1.0, 2.0);

    EXPECT_EQ(basis.degree(), 2U);
    EXPECT_EQ(basis.basis_size(), 5U);
    EXPECT_EQ(basis.knots().size(), 8U);
    EXPECT_DOUBLE_EQ(basis.knots().front(), -1.0);
    EXPECT_DOUBLE_EQ(basis.knots()[1], -1.0);
    EXPECT_DOUBLE_EQ(basis.knots()[2], -1.0);
    EXPECT_DOUBLE_EQ(basis.knots()[5], 2.0);
    EXPECT_DOUBLE_EQ(basis.knots()[6], 2.0);
    EXPECT_DOUBLE_EQ(basis.knots()[7], 2.0);
}

TEST(BSplineBasis, EvaluatePointMatchesKnownLinearBasisValues) {
    const BSplineBasis basis(1, {0.0, 0.0, 1.0, 2.0, 2.0});

    {
        const auto local = basis.evaluate_point(0.25);
        ASSERT_EQ(local.indices.size(), 2U);
        EXPECT_EQ(local.indices[0], 0U);
        EXPECT_EQ(local.indices[1], 1U);
        EXPECT_DOUBLE_EQ(local.values[0], 0.75);
        EXPECT_DOUBLE_EQ(local.values[1], 0.25);
    }

    {
        const auto local = basis.evaluate_point(1.5);
        ASSERT_EQ(local.indices.size(), 2U);
        EXPECT_EQ(local.indices[0], 1U);
        EXPECT_EQ(local.indices[1], 2U);
        EXPECT_DOUBLE_EQ(local.values[0], 0.5);
        EXPECT_DOUBLE_EQ(local.values[1], 0.5);
    }

    {
        const auto local = basis.evaluate_point(2.0);
        ASSERT_EQ(local.indices.size(), 1U);
        EXPECT_EQ(local.indices[0], 2U);
        EXPECT_DOUBLE_EQ(local.values[0], 1.0);
    }
}

TEST(BSplineBasis, EvaluateMatrixReturnsSparseByDefault) {
    const BSplineBasis basis(2, {0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0});
    const std::vector<double> points = {0.0, 0.5, 1.0, 1.5, 2.0, 2.5};

    const auto Phi = basis.evaluate(points);

    EXPECT_EQ(Phi.rows(), points.size());
    EXPECT_EQ(Phi.cols(), basis.basis_size());
    EXPECT_TRUE(Phi.is_compressed());
    EXPECT_LE(Phi.nnz(), points.size() * (basis.degree() + 1));

    for (std::size_t row = 0; row < points.size() - 1; ++row) {
        double row_sum = 0.0;
        for (std::size_t col = 0; col < basis.basis_size(); ++col)
            row_sum += dense_from_sparse(Phi, row, col);
        EXPECT_NEAR(row_sum, 1.0, 1e-12);
    }

    double outside_sum = 0.0;
    for (std::size_t col = 0; col < basis.basis_size(); ++col)
        outside_sum += dense_from_sparse(Phi, points.size() - 1, col);
    EXPECT_DOUBLE_EQ(outside_sum, 0.0);
}

TEST(BSplineBasis, DenseEvaluationMatchesSparseEvaluation) {
    const auto basis = BSplineBasis::open_uniform(3, 6, 0.0, 1.0);
    const std::vector<double> points = {0.0, 0.2, 0.4, 0.8, 1.0};

    const auto sparse = basis.evaluate(points);
    const auto dense = basis.evaluate_dense(points);
    const auto csr = datamunge::linalg::to_csr(sparse);

    EXPECT_EQ(dense.rows(), points.size());
    EXPECT_EQ(dense.cols(), basis.basis_size());

    for (std::size_t row = 0; row < dense.rows(); ++row) {
        double row_sum = 0.0;
        for (std::size_t col = 0; col < dense.cols(); ++col) {
            const double sparse_value = dense_from_sparse(sparse, row, col);
            EXPECT_NEAR(dense(row, col), sparse_value, 1e-12);
            row_sum += dense(row, col);
        }
        EXPECT_NEAR(row_sum, 1.0, 1e-12);
    }

    const std::vector<double> coeffs = {1.0, -0.5, 0.25, 1.5, -1.0, 0.75};
    const auto y_sparse = csr.spmv(coeffs);
    const auto y_dense = dense * coeffs;
    for (std::size_t i = 0; i < y_dense.size(); ++i)
        EXPECT_NEAR(y_dense[i], y_sparse[i], 1e-12);
}
