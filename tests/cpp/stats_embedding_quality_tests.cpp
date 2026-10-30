#include <gtest/gtest.h>

#include <datamunge/stats/embedding_quality.hpp>

#include <stdexcept>
#include <string>
#include <vector>

namespace {

using datamunge::linalg::DenseMatrix;

DenseMatrix<double> ten_points() {
    const double        xy[10][2] = {{0, 0}, {1, 3}, {2, 1}, {3, 4}, {4, 0.5}, {5, 2.5}, {6, 5}, {7, 1.5}, {8, 3.5}, {9, 0.2}};
    DenseMatrix<double> X(10, 2, 0.0);
    for (std::size_t i = 0; i < 10; ++i) {
        X(i, 0) = xy[i][0];
        X(i, 1) = xy[i][1];
    }
    return X;
}

// Twelve points in 3-D and the projection 0.7 x + 0.4 y, chosen with no tied distances
// (tie order differs between implementations).
DenseMatrix<double> twelve_points() {
    const double xyz[12][3] = {{9.431, 5.113, 9.762}, {0.808, 6.074, 3.765}, {8.019, 1.745, 8.716},
                               {5.439, 9.022, 4.772}, {4.305, 7.889, 9.842}, {3.697, 9.689, 9.29},
                               {1.777, 6.089, 7.049}, {9.428, 6.657, 1.334}, {4.979, 4.936, 5.002},
                               {9.586, 3.499, 2.238}, {5.221, 6.412, 9.391}, {5.82, 2.678, 9.298}};
    DenseMatrix<double> X(12, 3, 0.0);
    for (std::size_t i = 0; i < 12; ++i)
        for (std::size_t c = 0; c < 3; ++c) X(i, c) = xyz[i][c];
    return X;
}

DenseMatrix<double> projection(const DenseMatrix<double>& X) {
    DenseMatrix<double> Y(X.rows(), 1, 0.0);
    for (std::size_t i = 0; i < X.rows(); ++i) Y(i, 0) = X(i, 0) * 0.7 + X(i, 1) * 0.4;
    return Y;
}

} // namespace

TEST(EmbeddingQuality, IdenticalEmbeddingScoresOne) {
    const auto X = ten_points();
    EXPECT_DOUBLE_EQ(datamunge::stats::trustworthiness(X, X, 3), 1.0);
    EXPECT_DOUBLE_EQ(datamunge::stats::continuity(X, X, 3), 1.0);
}

TEST(EmbeddingQuality, MatchesScikitLearnOnAProjection) {
    // sklearn.manifold.trustworthiness(X, Y, n_neighbors=k) and with the roles swapped.
    const auto X = twelve_points();
    const auto Y = projection(X);
    EXPECT_NEAR(datamunge::stats::trustworthiness(X, Y, 2), 0.6323529411764706, 1e-12);
    EXPECT_NEAR(datamunge::stats::trustworthiness(X, Y, 3), 0.6904761904761905, 1e-12);
    EXPECT_NEAR(datamunge::stats::trustworthiness(X, Y, 5), 0.6791666666666667, 1e-12);
    EXPECT_NEAR(datamunge::stats::continuity(X, Y, 2), 0.75, 1e-12);
    EXPECT_NEAR(datamunge::stats::continuity(X, Y, 3), 0.6706349206349207, 1e-12);
    EXPECT_NEAR(datamunge::stats::continuity(X, Y, 5), 0.7041666666666666, 1e-12);
}

TEST(EmbeddingQuality, RejectsBadArguments) {
    const auto X = ten_points();
    EXPECT_THROW((void)datamunge::stats::trustworthiness(X, X, 0), std::invalid_argument);
    EXPECT_THROW((void)datamunge::stats::trustworthiness(X, X, 5), std::invalid_argument);
    EXPECT_THROW((void)datamunge::stats::continuity(X, DenseMatrix<double>(9, 1, 0.0), 2), std::invalid_argument);
}

TEST(EmbeddingQuality, DataFrameOverloadsMatchTheMatrixVersions) {
    const auto X = twelve_points();
    const auto Y = projection(X);
    datamunge::dstruct::DataFrame frame;
    for (std::size_t c = 0; c < 3; ++c) {
        std::vector<double> col(12);
        for (std::size_t i = 0; i < 12; ++i) col[i] = X(i, c);
        frame.add_column("v" + std::to_string(c), col);
    }
    const std::vector<std::string> cols{"v0", "v1", "v2"};
    EXPECT_DOUBLE_EQ(datamunge::stats::trustworthiness(frame, cols, Y, 3), datamunge::stats::trustworthiness(X, Y, 3));
    EXPECT_DOUBLE_EQ(datamunge::stats::continuity(frame, cols, Y, 3), datamunge::stats::continuity(X, Y, 3));
    EXPECT_THROW((void)datamunge::stats::trustworthiness(frame, {"nope"}, Y, 3), std::invalid_argument);
}
