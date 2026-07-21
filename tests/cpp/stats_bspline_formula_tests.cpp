#include <gtest/gtest.h>

#include <datamunge/dstruct/dstruct.hpp>
#include <datamunge/fda/bspline.hpp>
#include <datamunge/stats/lm.hpp>

#include <cmath>
#include <vector>

using datamunge::dstruct::DataFrame;
using datamunge::fda::BSpline;
using datamunge::stats::LM;

TEST(BSpline, TensorProductDesignIsSparseAndPartitionsUnity) {
    const auto spline = BSpline::open_uniform({1, 1}, {3, 3}, {0.0, 0.0}, {2.0, 2.0});
    const auto design = spline.evaluate({{0.25, 0.5}, {1.5, 1.5}});

    EXPECT_EQ(spline.dimensions(), 2U);
    EXPECT_EQ(design.rows(), 2U);
    EXPECT_EQ(design.cols(), 9U);
    EXPECT_TRUE(design.is_compressed());
    EXPECT_LE(design.nnz(), 8U);
    for (std::size_t row = 0; row < design.rows(); ++row) {
        double sum = 0.0;
        for (std::size_t i = 0; i < design.nnz(); ++i)
            if (design.row_indices()[i] == row) sum += design.values()[i];
        EXPECT_NEAR(sum, 1.0, 1e-12);
    }
}

TEST(LM, BSplineFormulaSupportsOneDimensionalAndSurfaceTerms) {
    constexpr std::size_t n = 64;
    std::vector<double> x(n), y(n), z(n), curve_response(n);
    for (std::size_t i = 0; i < n; ++i) {
        x[i] = static_cast<double>(i % 8) / 7.0;
        y[i] = static_cast<double>(i / 8) / 7.0;
        curve_response[i] = std::sin(x[i]);
        z[i] = std::sin(x[i]) + y[i] * y[i];
    }
    DataFrame frame;
    frame.add_column("x", x);
    frame.add_column("y", y);
    frame.add_column("z", z);
    frame.add_column("curve_response", curve_response);

    LM curve(frame, "curve_response ~ bs(x)");
    EXPECT_EQ(curve.coefficient_names().size(), 6U); // intercept + five non-reference cubic basis functions
    EXPECT_GT(curve.r_squared(), 0.9);

    LM surface(frame, "z ~ bs(x, y)");
    EXPECT_EQ(surface.coefficient_names().size(), 36U); // intercept + 35 non-reference tensor-product columns
    EXPECT_GT(surface.r_squared(), 0.99);
}
