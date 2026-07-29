#include <gtest/gtest.h>

#include <datamunge/datasets/datasets.hpp>
#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/stats/kernel_ridge_regression.hpp>

#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using datamunge::dstruct::DataFrame;
using datamunge::stats::KernelRidgeRegression;
using datamunge::stats::KernelRidgeRegressionOptions;

namespace {

constexpr double kTwoPi = 6.283185307179586;

// Noisy samples of y = sin(2*pi*x) on [0,1]; the true (noiseless) function is available for scoring.
DataFrame make_sin(std::vector<double>& x_out, std::size_t n = 120, unsigned seed = 20240815) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> ux(0.0, 1.0);
  std::normal_distribution<double> noise(0.0, 0.15);
  std::vector<double> x(n), y(n);
  for (std::size_t i = 0; i < n; ++i) {
    x[i] = ux(rng);
    y[i] = std::sin(kTwoPi * x[i]) + noise(rng);
  }
  x_out = x;
  DataFrame d;
  d.add_column("x", x);
  d.add_column("y", y);
  return d;
}

double truth_rmse(const KernelRidgeRegression& m, std::size_t grid = 400) {
  std::vector<double> gx(grid);
  for (std::size_t i = 0; i < grid; ++i) gx[i] = static_cast<double>(i) / static_cast<double>(grid - 1);
  DataFrame g;
  g.add_column("x", gx);
  const auto pred = m.predict(g);
  double ss = 0.0;
  for (std::size_t i = 0; i < grid; ++i) {
    const double d = pred[i] - std::sin(kTwoPi * gx[i]);
    ss += d * d;
  }
  return std::sqrt(ss / static_cast<double>(grid));
}

} // namespace

TEST(KernelRidgeRegression, RecoversNonlinearFunction) {
  std::vector<double> x;
  const auto data = make_sin(x);
  KernelRidgeRegression m(data, "y ~ x", KernelRidgeRegressionOptions{"rbf"});
  EXPECT_EQ(m.observations(), 120u);
  EXPECT_TRUE(m.lambda_was_selected());
  EXPECT_TRUE(m.gamma_was_selected());
  EXPECT_GT(m.r_squared(), 0.90);       // leave-one-out
  EXPECT_LT(truth_rmse(m), 0.05);        // close to the true sine
}

TEST(KernelRidgeRegression, LinearKernelIsRidgeRegression) {
  std::vector<double> x;
  const auto data = make_sin(x);
  const std::size_t n = x.size();
  const double lam = 1.0;
  KernelRidgeRegression m(data, "y ~ x", KernelRidgeRegressionOptions{"linear", lam});
  const auto krr_pred = m.predict(data);

  // Closed-form primal ridge on the standardized single predictor (population sd, matching the class).
  std::vector<double> y(n);
  for (std::size_t i = 0; i < n; ++i) y[i] = data.double_at("y", i);
  double xm = 0.0, ym = 0.0;
  for (std::size_t i = 0; i < n; ++i) { xm += x[i]; ym += y[i]; }
  xm /= n; ym /= n;
  double var = 0.0;
  for (std::size_t i = 0; i < n; ++i) var += (x[i] - xm) * (x[i] - xm);
  const double sd = std::sqrt(var / n);
  double sxx = 0.0, sxy = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double xs = (x[i] - xm) / sd;
    sxx += xs * xs;
    sxy += xs * (y[i] - ym);
  }
  const double beta = sxy / (sxx + lam);
  for (std::size_t i = 0; i < n; ++i) {
    const double ridge = ym + ((x[i] - xm) / sd) * beta;
    EXPECT_NEAR(krr_pred[i], ridge, 1e-8);
  }
}

TEST(KernelRidgeRegression, RbfBeatsLinearOnNonlinearData) {
  std::vector<double> x;
  const auto data = make_sin(x);
  KernelRidgeRegression rbf(data, "y ~ x", KernelRidgeRegressionOptions{"rbf"});
  KernelRidgeRegression lin(data, "y ~ x", KernelRidgeRegressionOptions{"linear"});
  EXPECT_GT(rbf.r_squared(), lin.r_squared() + 0.2);
  EXPECT_LT(truth_rmse(rbf), truth_rmse(lin));
}

TEST(KernelRidgeRegression, EffectiveDofFallsWithLambda) {
  std::vector<double> x;
  const auto data = make_sin(x);
  KernelRidgeRegression a(data, "y ~ x", KernelRidgeRegressionOptions{"rbf", 0.01, 5.0});
  KernelRidgeRegression b(data, "y ~ x", KernelRidgeRegressionOptions{"rbf", 1.0, 5.0});
  KernelRidgeRegression c(data, "y ~ x", KernelRidgeRegressionOptions{"rbf", 100.0, 5.0});
  EXPECT_GT(a.effective_degrees_of_freedom(), b.effective_degrees_of_freedom());
  EXPECT_GT(b.effective_degrees_of_freedom(), c.effective_degrees_of_freedom());
  EXPECT_GT(c.effective_degrees_of_freedom(), 0.0);
  EXPECT_LT(a.effective_degrees_of_freedom(), static_cast<double>(a.observations()));
}

TEST(KernelRidgeRegression, AutoSelectionPopulatesCvCurve) {
  std::vector<double> x;
  const auto data = make_sin(x);
  KernelRidgeRegression sel(data, "y ~ x", KernelRidgeRegressionOptions{"rbf", -1.0, 5.0});
  EXPECT_EQ(sel.lambda_grid().size(), sel.cv_mean_squared_error().size());
  EXPECT_GT(sel.lambda_grid().size(), 1u);

  KernelRidgeRegression fixed(data, "y ~ x", KernelRidgeRegressionOptions{"rbf", 0.1, 5.0});
  EXPECT_FALSE(fixed.lambda_was_selected());
  EXPECT_TRUE(fixed.lambda_grid().empty());
  EXPECT_THROW(fixed.plot_cv_curve(), std::invalid_argument);
}

TEST(KernelRidgeRegression, PredictReproducesDualFitOnTrainingData) {
  std::vector<double> x;
  const auto data = make_sin(x);
  KernelRidgeRegression m(data, "y ~ x", KernelRidgeRegressionOptions{"rbf", 0.1, 1.0});
  const auto pred = m.predict(data);
  ASSERT_EQ(pred.size(), data.nrows());
  for (double v : pred) EXPECT_TRUE(std::isfinite(v));
  // Resubstitution fit is at least as good as the leave-one-out fit.
  std::vector<double> y(data.nrows());
  double ym = 0.0;
  for (std::size_t i = 0; i < data.nrows(); ++i) { y[i] = data.double_at("y", i); ym += y[i]; }
  ym /= data.nrows();
  double sr = 0.0, st = 0.0;
  for (std::size_t i = 0; i < data.nrows(); ++i) {
    sr += (y[i] - pred[i]) * (y[i] - pred[i]);
    st += (y[i] - ym) * (y[i] - ym);
  }
  EXPECT_GE(1.0 - sr / st, m.r_squared() - 1e-9);
}

TEST(KernelRidgeRegression, DropsRowsWithNullResponse) {
  DataFrame frame;
  frame.add_column("x", std::vector<std::optional<double>>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
  frame.add_column("y", std::vector<std::optional<double>>{2.0, 4.0, std::nullopt, 8.0, 9.0, 3.0});
  KernelRidgeRegression m(frame, "y ~ x", KernelRidgeRegressionOptions{"rbf", 0.1, 1.0});
  EXPECT_EQ(m.observations(), 5u);
}

TEST(KernelRidgeRegression, RejectsInvalidOptions) {
  std::vector<double> x;
  const auto data = make_sin(x, 40);
  EXPECT_THROW(KernelRidgeRegression(data, "y ~ x", KernelRidgeRegressionOptions{"bogus"}), std::invalid_argument);
  KernelRidgeRegressionOptions zero_grid{"rbf"};
  zero_grid.n_lambda_grid = 0;
  EXPECT_THROW(KernelRidgeRegression(data, "y ~ x", zero_grid), std::invalid_argument);
}

TEST(KernelRidgeRegression, PlotFitGuardsMultiPredictor) {
  const auto iris = datamunge::datasets::iris();
  KernelRidgeRegression single(iris, "Petal.Length ~ Petal.Width", KernelRidgeRegressionOptions{"rbf"});
  EXPECT_FALSE(single.plot_fit(iris).series().empty());

  KernelRidgeRegression multi(iris, "Petal.Length ~ Petal.Width + Sepal.Length",
                              KernelRidgeRegressionOptions{"rbf"});
  EXPECT_THROW(multi.plot_fit(iris), std::invalid_argument);
}
