/// @file bspline_design_tests.cpp
/// @brief B-spline design, named matrix conversion, formula prediction and plot regressions.
#include <gtest/gtest.h>

#include <datamunge/bspline_regression.hpp>
#include <datamunge/datamunge.hpp>
#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/stats/lm.hpp>
#include <datamunge/vector.hpp>
#include <limits>

namespace dm = datamunge;

TEST(BsplineDesign, FreeFactoryOwnsKnotsAndSupportsDegreeAndInterceptOptions) {
  auto       knots  = dm::c(0, 0.5, 1);
  const auto basis  = dm::bs(0, 1, knots, 1, true);
  knots[1]          = 0.9;
  const auto design = basis.eval({0, 0.25, 0.5, 0.75, 1});
  ASSERT_EQ(design.cols(), 3u);
  EXPECT_DOUBLE_EQ(design(1, 0), 0.5);
  EXPECT_DOUBLE_EQ(design(1, 1), 0.5);
  EXPECT_DOUBLE_EQ(design(2, 1), 1);
  EXPECT_DOUBLE_EQ(design(4, 2), 1);
  const auto defaults = dm::bs(0, 1, dm::seql(0, 1, 3));
  EXPECT_EQ(defaults.eval({}).cols(), 4u);
  EXPECT_DOUBLE_EQ(defaults.eval({1})(0, 3), 1);
  EXPECT_THROW(dm::bs(1, 0, {}), std::invalid_argument);
}

TEST(BsplineDesign, LinearBasisHasHandCheckableValuesAndInterceptPolicy) {
  const dm::Bspline basis(0, 1, {0, 0.5, 1}, 1);
  const auto        x = basis.eval({0, 0.25, 0.5, 0.75, 1});
  ASSERT_EQ(x.cols(), 2u);
  EXPECT_EQ(x.colnames(), (std::vector<std::string>{"V1", "V2"}));
  const dm::Mat<> expected = {{0, 0}, {0.5, 0}, {1, 0}, {0.5, 0.5}, {0, 1}};
  for (std::size_t i = 0; i < x.rows(); ++i)
    for (std::size_t j = 0; j < x.cols(); ++j)
      EXPECT_DOUBLE_EQ(x(i, j), expected(i, j));
  const auto full = dm::Bspline(0, 1, {0.5}, 1, true).eval({0, 0.25, 0.5, 0.75, 1});
  for (std::size_t i = 0; i < full.rows(); ++i)
    EXPECT_DOUBLE_EQ(full(i, 0) + full(i, 1) + full(i, 2), 1);
}

TEST(BsplineDesign, CubicMatchesCanonicalEvaluatorAndReusesTrainingKnots) {
  const dm::Bspline           basis(0, 1, {0, 0.3, 0.7, 1});
  const std::vector<double>   points{0, 0.1, 0.3, 0.6, 0.7, 0.9, 1};
  const auto                  actual = basis.eval(points);
  const dm::fda::BSplineBasis native(3, {0, 0, 0, 0, 0.3, 0.7, 1, 1, 1, 1});
  const auto                  expected = native.evaluate_dense(points);
  ASSERT_EQ(actual.cols() + 1, expected.cols());
  for (std::size_t i = 0; i < actual.rows(); ++i)
    for (std::size_t j = 0; j < actual.cols(); ++j)
      EXPECT_DOUBLE_EQ(actual(i, j), expected(i, j + 1));
  const auto subset = basis.eval({0.3, 0.7});
  for (std::size_t j = 0; j < subset.cols(); ++j) {
    EXPECT_DOUBLE_EQ(subset(0, j), actual(2, j));
    EXPECT_DOUBLE_EQ(subset(1, j), actual(4, j));
  }
  const auto empty = basis.eval({});
  EXPECT_EQ(empty.rows(), 0u);
  EXPECT_EQ(empty.cols(), actual.cols());
}

TEST(BsplineDesign, RejectsInvalidKnotsBoundsAndCoordinates) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(dm::Bspline(1, 0, {}), std::invalid_argument);
  EXPECT_THROW(dm::Bspline(0, nan, {}), std::invalid_argument);
  EXPECT_THROW(dm::Bspline(0, 1, {0.5, 0.5}), std::invalid_argument);
  EXPECT_THROW(dm::Bspline(0, 1, {0.8, 0.2}), std::invalid_argument);
  EXPECT_THROW(dm::Bspline(0, 1, {-0.1}), std::invalid_argument);
  EXPECT_THROW(dm::Bspline(0, 1, {nan}), std::invalid_argument);
  const dm::Bspline basis(0, 1, {});
  EXPECT_THROW(basis.eval({-0.1}), std::invalid_argument);
  EXPECT_THROW(basis.eval({1.1}), std::invalid_argument);
  EXPECT_THROW(basis.eval({nan}), std::invalid_argument);
}

TEST(BsplineDesign, NumericFrameConversionsPreserveOrderValuesAndOwnership) {
  dm::SpMat<>                    matrix = {{0, 2}, {3, 0}};
  const auto                   joined = dm::cbind(dm::c(5, 6), matrix);
  const dm::dstruct::DataFrame frame(joined, {"y", "a", "b"});
  EXPECT_EQ(frame.columns(), (std::vector<std::string>{"y", "a", "b"}));
  EXPECT_DOUBLE_EQ(frame.double_at("y", 1), 6);
  EXPECT_DOUBLE_EQ(frame.double_at("a", 0), 0);
  EXPECT_DOUBLE_EQ(frame.double_at("b", 0), 2);
  const dm::dstruct::DataFrame automatic(matrix);
  matrix.set(0, 1, 9);
  EXPECT_DOUBLE_EQ(automatic.double_at("V2", 0), 2);
  const dm::dstruct::DataFrame vector(dm::seql(0, 1, 3), {"x"});
  EXPECT_EQ(vector.ncols(), 1u);
  EXPECT_DOUBLE_EQ(vector.double_at("x", 1), 0.5);
  const dm::BasicMat<false, float> floats = {{1, 2}};
  EXPECT_DOUBLE_EQ(dm::dstruct::DataFrame(floats).double_at("V2", 0), 2);
}

TEST(BsplineDesign, RejectsIncompatibleBindingsAndColumnNames) {
  const dm::Mat<> matrix = {{1, 2}, {3, 4}};
  EXPECT_THROW(dm::cbind(dm::c(1), matrix), std::invalid_argument);
  EXPECT_THROW((dm::DataFrame(matrix, {"x"})), std::invalid_argument);
  EXPECT_THROW((dm::DataFrame(matrix, {"x", "x"})), std::invalid_argument);
  EXPECT_THROW((dm::DataFrame(matrix, {"", "y"})), std::invalid_argument);
  EXPECT_THROW((dm::DataFrame(dm::Mat<>(2, 0))), std::invalid_argument);
  EXPECT_THROW((dm::DataFrame(std::vector<double>{1}, {"x", "y"})), std::invalid_argument);
}

TEST(BsplineDesign, FormulaFitPredictsOnTheSameBasisWithImplicitIntercept) {
  const auto      x      = dm::seql(0, 1, 31);
  const auto      y      = 1 + 2 * x + 3 * x * x;
  const auto      basis  = dm::bs(0, 1, {0, 0.5, 1});
  const dm::Mat<> design = basis.eval(x);
  auto            names  = design.colnames();
  names.insert(names.begin(), "y");
  const dm::LM model(dm::DataFrame(dm::cbind(y, design), names), "y ~ .");
  EXPECT_EQ(model.rank(), design.cols() + 1);
  const auto grid = dm::seql(0, 1, 19);
  const auto fit  = model.predict(dm::DataFrame(basis.eval(grid), design.colnames()));
  for (std::size_t i = 0; i < grid.size(); ++i)
    EXPECT_NEAR(fit[i], 1 + 2 * grid[i] + 3 * grid[i] * grid[i], 1e-10);
  const auto plot = dm::plot(x, y).lines(grid, fit, "Fitted", dm::RGB{20, 100, 20});
  EXPECT_EQ(plot.series().size(), 2u);
  EXPECT_NE(plot.to_svg().find("Fitted"), std::string::npos);
}

TEST(BsplineDesign, SparseEvaluationMatchesDenseBoundariesSupportAndEmptyInputs) {
  for (bool intercept : {false, true}) {
    const auto basis = dm::bs(0, 1, {0, 0.25, 0.5, 0.75, 1}, 3, intercept);
    const auto x = dm::seql(0, 1, 101);
    const auto dense = basis.eval(x);
    const dm::SpMat<> sparse = basis.speval(x);
    EXPECT_LE(sparse.nnz(), 4 * x.size());
    EXPECT_EQ(sparse.cols(), dense.cols());
    for (std::size_t i = 0; i < x.size(); ++i)
      for (std::size_t j = 0; j < dense.cols(); ++j)
        EXPECT_DOUBLE_EQ(sparse(i, j), dense(i, j));
    const auto empty = basis.speval({});
    EXPECT_EQ(empty.rows(), 0u);
    EXPECT_EQ(empty.cols(), dense.cols());
    EXPECT_EQ(empty.nnz(), 0u);
    EXPECT_THROW(basis.speval({-0.1}), std::invalid_argument);
    EXPECT_THROW(basis.speval({std::numeric_limits<double>::quiet_NaN()}), std::invalid_argument);
  }
  static_assert(std::is_same_v<dm::SpMat<float>, dm::Mat<dm::Sparse, float>>);
}

TEST(MatrixLM, DenseSparseAndFormulaFitsAgreeIncludingDiagnosticsAndAnova) {
  const auto x = dm::seql(0, 1, 101);
  const auto y = dm::apply(x, [](double z) { return 1 + 2*z + std::sin(17*z)*0.1; });
  const auto basis = dm::bs(0, 1, {0, 0.5, 1});
  const auto dense = basis.eval(x);
  const auto sparse = basis.speval(x);
  const auto dfit = dm::lm(y, dense), sfit = dm::lm(y, sparse);
  auto names = dense.colnames(); names.insert(names.begin(), "y");
  const auto formula = dm::lm(dm::DataFrame(dm::cbind(y, dense), names), "y ~ .");
  ASSERT_EQ(sfit.rank(), formula.rank());
  EXPECT_TRUE(sfit.has_intercept());
  EXPECT_EQ(sfit.degrees_of_freedom(), x.size() - sfit.rank());
  for (std::size_t j = 0; j < sfit.rank(); ++j) {
    EXPECT_NEAR(sfit.coefficients()[j], formula.coefficients()[j], 1e-8);
    EXPECT_NEAR(sfit.standard_errors()[j], formula.standard_errors()[j], 1e-9);
    EXPECT_NEAR(dfit.coefficients()[j], formula.coefficients()[j], 1e-10);
  }
  const auto grid = dm::seql(0, 1, 29);
  const auto expected = formula.predict(dm::DataFrame(basis.eval(grid)));
  const auto actual = sfit.predict(basis.speval(grid));
  for (std::size_t i = 0; i < grid.size(); ++i) {
    EXPECT_NEAR(actual[i], expected[i], 1e-9);
    EXPECT_NEAR(sfit.predict(basis.eval(grid))[i], expected[i], 1e-9);
    EXPECT_NEAR(dfit.predict(basis.speval(grid))[i], expected[i], 1e-9);
  }
  EXPECT_NEAR(sfit.r_squared(), formula.r_squared(), 1e-10);
  EXPECT_NEAR(sfit.f_statistic(), formula.f_statistic(), 1e-6);
  for (std::size_t i = 0; i < x.size(); ++i) {
    EXPECT_NEAR(sfit.leverage()[i], formula.leverage()[i], 1e-8);
    EXPECT_NEAR(sfit.cooks_distance()[i], formula.cooks_distance()[i], 1e-6);
  }
  const auto st = std::unique_ptr<dm::DataFrame>(sfit.anova());
  const auto ft = std::unique_ptr<dm::DataFrame>(formula.anova());
  ASSERT_EQ(st->nrows(), ft->nrows());
  for (std::size_t i = 0; i < st->nrows(); ++i)
    EXPECT_NEAR(st->numeric_at("sum_sq", i), ft->numeric_at("sum_sq", i), 1e-8);
}

TEST(MatrixLM, NoInterceptExactFitZeroResponseAndOwnedTypedInputs) {
  const dm::SpMat<float> x = {{1}, {2}, {3}, {4}};
  const auto no_intercept = dm::lm(dm::c(2, 4, 6, 8), x, false);
  EXPECT_FALSE(no_intercept.has_intercept());
  EXPECT_EQ(no_intercept.rank(), 1u);
  EXPECT_NEAR(no_intercept.predict(dm::SpMat<float>{{5}})[0], 10, 1e-10);
  const dm::LM zeros(dm::c(0, 0, 0, 0), x);
  EXPECT_NEAR(zeros.predict(dm::SpMat<>{{5}})[0], 0, 1e-10);
  const auto model = [] {
    auto x = dm::bs(0, 1, {}, 1).speval({0, 0.25, 0.5, 0.75, 1});
    dm::LM fit(dm::c(1, 1.5, 2, 2.5, 3), x);
    x.set(0, 0, 10);
    return fit;
  }();
  EXPECT_NEAR(model.predict(dm::Mat<>{{0.5}})[0], 2, 1e-10);
  EXPECT_TRUE(model.predict(dm::SpMat<>(0, 1)).empty());
  const dm::LM intercept_only(dm::c(1, 2, 3), dm::SpMat<>(3, 0));
  EXPECT_NEAR(intercept_only.predict(dm::SpMat<>(2, 0))[0], 2, 1e-10);
}

TEST(MatrixLM, RejectsBadShapesNonfiniteRankDeficiencyAndSolverFailure) {
  const auto nan = std::numeric_limits<double>::quiet_NaN();
  const dm::SpMat<> x = {{0}, {1}, {2}, {3}};
  EXPECT_THROW((dm::LM(dm::c(1, 2), x)), std::invalid_argument);
  EXPECT_THROW((dm::LM(dm::c(1, 2), dm::SpMat<>{{0}, {1}})), std::invalid_argument);
  EXPECT_THROW((dm::LM(dm::c(1, 2, 3, nan), x)), std::invalid_argument);
  EXPECT_THROW((dm::LM(dm::c(1, 2, 3, 4), dm::SpMat<>{{0}, {1}, {nan}, {3}})), std::invalid_argument);
  EXPECT_THROW((dm::LM(dm::c(1, 2, 3, 4), dm::SpMat<>{{1}, {1}, {1}, {1}})), std::runtime_error);
  EXPECT_THROW((dm::LM(dm::c(1, 2, 3, 4), x, true, {{1e-30, 1}, true})), std::runtime_error);
  const dm::LM fit(dm::c(1, 3, 5, 7), x);
  EXPECT_THROW(fit.predict(dm::SpMat<>(1, 2)), std::invalid_argument);
  EXPECT_THROW(fit.predict(dm::SpMat<>{{nan}}), std::invalid_argument);
  EXPECT_THROW(fit.predict(dm::Mat<>{{nan}}), std::invalid_argument);
  const dm::DataFrame data(dm::Mat<>{{0, 1}, {1, 3}, {2, 5}, {3, 7}}, {"x", "y"});
  const dm::LM formula(data, "y ~ x");
  EXPECT_THROW(formula.predict(dm::SpMat<>{{1}}), std::invalid_argument);
}
