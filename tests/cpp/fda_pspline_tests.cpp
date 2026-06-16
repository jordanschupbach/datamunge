#include <gtest/gtest.h>

#include <datamunge/fda/pspline.hpp>

#include <cmath>
#include <vector>

using datamunge::fda::BSplineBasis;
using datamunge::fda::PSplineOptions;
using datamunge::fda::fit_pspline;
using datamunge::fda::pspline_penalty_matrix;

TEST(PSpline, SecondOrderPenaltyMatchesKnownStructure) {
    const auto penalty = pspline_penalty_matrix(5, 2);

    const double expected[5][5] = {
        {1, -2, 1, 0, 0},
        {-2, 5, -4, 1, 0},
        {1, -4, 6, -4, 1},
        {0, 1, -4, 5, -2},
        {0, 0, 1, -2, 1},
    };

    for (std::size_t i = 0; i < 5; ++i)
        for (std::size_t j = 0; j < 5; ++j)
            EXPECT_DOUBLE_EQ(penalty(i, j), expected[i][j]);
}

TEST(PSpline, ZeroPenaltyInterpolatesNoiselessDataWell) {
    std::vector<double> x(20);
    std::vector<double> y(20);
    for (std::size_t i = 0; i < x.size(); ++i) {
        x[i] = static_cast<double>(i) / static_cast<double>(x.size() - 1);
        y[i] = std::sin(2.0 * 3.14159265358979323846 * x[i]);
    }

    PSplineOptions opts;
    opts.lambda = 0.0;
    opts.penalty_order = 2;
    const auto fit = fit_pspline(x, y, 3, 8, opts);

    EXPECT_LT(fit.residual_sum_of_squares, 0.02);
    EXPECT_GT(fit.r_squared, 0.99);
    EXPECT_EQ(fit.fitted.size(), x.size());
}

TEST(PSpline, LargerPenaltyShrinksRoughness) {
    std::vector<double> x(40);
    std::vector<double> y(40);
    for (std::size_t i = 0; i < x.size(); ++i) {
        x[i] = 2.0 * 3.14159265358979323846 * static_cast<double>(i)
             / static_cast<double>(x.size() - 1);
        y[i] = std::sin(x[i]) + 0.2 * std::cos(4.0 * x[i]);
    }

    const auto basis = BSplineBasis::open_uniform(3, 14, x.front(), x.back());

    PSplineOptions light;
    light.lambda = 1e-4;
    light.penalty_order = 2;
    const auto fit_light = fit_pspline(x, y, basis, light);

    PSplineOptions heavy;
    heavy.lambda = 10.0;
    heavy.penalty_order = 2;
    const auto fit_heavy = fit_pspline(x, y, basis, heavy);

    EXPECT_LT(fit_heavy.penalty, fit_light.penalty);
    EXPECT_LT(fit_heavy.effective_degrees_of_freedom, fit_light.effective_degrees_of_freedom);
    EXPECT_EQ(fit_heavy.predict({x.front(), x.back()}).size(), 2U);
}

TEST(PSpline, RejectsInvalidPenaltyOrder) {
    std::vector<double> x = {0.0, 0.5, 1.0};
    std::vector<double> y = {0.0, 1.0, 0.0};

    PSplineOptions opts;
    opts.penalty_order = 0;
    EXPECT_THROW(fit_pspline(x, y, 1, 3, opts), std::invalid_argument);

    EXPECT_THROW(pspline_penalty_matrix(3, 3), std::invalid_argument);
}
