#include <gtest/gtest.h>

#include <datamunge/dstruct/dstruct.hpp>
#include <datamunge/stats/stats.hpp>

#include <cmath>
#include <vector>

using datamunge::dstruct::DataFrame;
using datamunge::stats::LM;
using datamunge::stats::LmOptions;
using datamunge::stats::PredictionInterval;

namespace {

// Matches the synthetic dataset used to generate reference values from real
// R (`lm()`, `anova()`, `confint()`, `hatvalues()`, `cooks.distance()`,
// `predict()`) — see the R transcript referenced in the PR description.
// color reference level is "blue" (alphabetically first), matching R's
// default factor level ordering.
DataFrame make_reference_frame() {
  const std::vector<double> x1 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  const std::vector<double> x2 = {2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11};
  const std::vector<std::string> color = {"red",  "blue", "red",  "green", "blue", "red",
                                          "green", "blue", "red",  "green", "blue", "green"};
  const std::vector<double> noise = {0.3, -0.2, 0.5, -0.4, 0.1, -0.6, 0.25, -0.15, 0.35, -0.3, 0.2, -0.1};

  std::vector<double> y(x1.size());
  for (std::size_t i = 0; i < x1.size(); ++i) {
    const double effect = (color[i] == "green") ? 1.5 : (color[i] == "red" ? 0.5 : 0.0);
    y[i] = 3.0 + 2.0 * x1[i] - 1.5 * x2[i] + effect + noise[i];
  }

  DataFrame df;
  df.add_column("x1", x1);
  df.add_column("x2", x2);
  df.add_column("color", color);
  df.add_column("y", y);
  df.add_column("w", std::vector<double>{1, 2, 1, 1, 3, 1, 2, 1, 1, 2, 1, 3});
  return df;
}

} // namespace

TEST(LM, MainEffectsMatchRSummary) {
  const auto df = make_reference_frame();
  LM model(df, "y ~ x1 + x2 + color");

  ASSERT_EQ(model.coefficient_names().size(), 5u);
  EXPECT_EQ(model.coefficient_names()[0], "(Intercept)");
  EXPECT_EQ(model.coefficient_names()[1], "x1");
  EXPECT_EQ(model.coefficient_names()[2], "x2");
  EXPECT_EQ(model.coefficient_names()[3], "colorgreen");
  EXPECT_EQ(model.coefficient_names()[4], "colorred");

  const std::vector<double> expected_coef = {2.96774, 1.71334, -1.21030, 1.51453, 0.51047};
  const std::vector<double> expected_se   = {0.15115, 0.06442, 0.06138, 0.14420, 0.14420};
  const std::vector<double> expected_t    = {19.64, 26.60, -19.72, 10.50, 3.54};
  for (std::size_t j = 0; j < 5; ++j) {
    EXPECT_NEAR(model.coefficients()[j], expected_coef[j], 1e-4);
    EXPECT_NEAR(model.standard_errors()[j], expected_se[j], 1e-4);
    EXPECT_NEAR(model.t_values()[j], expected_t[j], 1e-2);
  }

  EXPECT_NEAR(model.sigma(), 0.194, 1e-3);
  EXPECT_EQ(model.degrees_of_freedom(), 7u);
  EXPECT_NEAR(model.r_squared(), 0.9966, 1e-3);
  EXPECT_NEAR(model.adjusted_r_squared(), 0.9947, 1e-3);
  EXPECT_NEAR(model.f_statistic(), 516.3, 0.5);
  EXPECT_NEAR(model.f_p_value(), 1.005e-08, 1e-9);
}

TEST(LM, ConfidenceIntervalsMatchR) {
  const auto df = make_reference_frame();
  LM model(df, "y ~ x1 + x2 + color");

  const auto intervals = model.confidence_intervals(0.95);
  ASSERT_EQ(intervals.size(), 5u);

  const std::vector<std::pair<double, double>> expected = {
      {2.6103270, 3.3251459}, {1.5610243, 1.8656649}, {-1.3554523, -1.0651558},
      {1.1735371, 1.8555169}, {0.1694831, 0.8514629},
  };
  for (std::size_t j = 0; j < 5; ++j) {
    EXPECT_NEAR(intervals[j].lower, expected[j].first, 1e-3);
    EXPECT_NEAR(intervals[j].upper, expected[j].second, 1e-3);
  }
}

TEST(LM, LeverageAndCooksDistanceMatchR) {
  const auto df = make_reference_frame();
  LM model(df, "y ~ x1 + x2 + color");

  const std::vector<double> expected_leverage = {0.3969595, 0.5135135, 0.3023649, 0.4239865, 0.3716216, 0.4915541,
                                                  0.4915541, 0.3716216, 0.4239865, 0.3023649, 0.5135135, 0.3969595};
  const std::vector<double> expected_cooks    = {0.004895127, 0.154779143, 0.164087837, 0.074460156, 0.149009580,
                                                  0.950533188, 0.018889285, 0.108977105, 0.020334363, 0.001741832,
                                                  0.095198205, 0.169460332};

  ASSERT_EQ(model.leverage().size(), 12u);
  for (std::size_t i = 0; i < 12; ++i) {
    EXPECT_NEAR(model.leverage()[i], expected_leverage[i], 1e-5);
    EXPECT_NEAR(model.cooks_distance()[i], expected_cooks[i], 1e-4);
  }
}

TEST(LM, AnovaSequentialSumSquaresMatchR) {
  const auto df = make_reference_frame();
  LM model(df, "y ~ x1 + x2 + color");

  const auto rows = model.anova();
  ASSERT_EQ(rows.size(), 4u); // x1, x2, color, Residuals

  EXPECT_EQ(rows[0].term, "x1");
  EXPECT_NEAR(rows[0].sum_sq, 51.991, 1e-2);
  EXPECT_NEAR(rows[0].f_value, 1381.01, 1.0);

  EXPECT_EQ(rows[1].term, "x2");
  EXPECT_NEAR(rows[1].sum_sq, 21.606, 1e-2);
  EXPECT_NEAR(rows[1].f_value, 573.91, 1.0);

  EXPECT_EQ(rows[2].term, "color");
  EXPECT_EQ(rows[2].degrees_of_freedom, 2u);
  EXPECT_NEAR(rows[2].sum_sq, 4.155, 1e-2);
  EXPECT_NEAR(rows[2].f_value, 55.18, 0.5);

  EXPECT_EQ(rows[3].term, "Residuals");
  EXPECT_NEAR(rows[3].sum_sq, 0.264, 1e-2);
}

TEST(LM, PredictWithConfidenceIntervalMatchesR) {
  const auto df = make_reference_frame();
  LM model(df, "y ~ x1 + x2 + color");

  DataFrame newdata;
  newdata.add_column("x1", std::vector<double>{13, 14});
  newdata.add_column("x2", std::vector<double>{14, 13});
  newdata.add_column("color", std::vector<std::string>{"blue", "red"});

  const auto pred = model.predict(newdata, PredictionInterval::Confidence);
  ASSERT_EQ(pred.fit.size(), 2u);

  EXPECT_NEAR(pred.fit[0], 8.296959, 1e-3);
  EXPECT_NEAR(pred.fit[1], 11.731081, 1e-3);
  EXPECT_NEAR(pred.se_fit[0], 0.1618654, 1e-3);
  EXPECT_NEAR(pred.se_fit[1], 0.2144246, 1e-3);
  EXPECT_NEAR(pred.lower[0], 7.914209, 1e-2);
  EXPECT_NEAR(pred.upper[0], 8.67971, 1e-2);
  EXPECT_NEAR(pred.lower[1], 11.224047, 1e-2);
  EXPECT_NEAR(pred.upper[1], 12.23811, 1e-2);

  // The plain predict() overload should return the same fitted values.
  const auto simple = model.predict(newdata);
  EXPECT_NEAR(simple[0], pred.fit[0], 1e-9);
  EXPECT_NEAR(simple[1], pred.fit[1], 1e-9);
}

TEST(LM, WeightedLeastSquaresMatchesR) {
  const auto df = make_reference_frame();
  LmOptions  options;
  options.weights_column = "w";
  LM model(df, "y ~ x1 + x2", options);

  const std::vector<double> expected_coef = {3.0484, 1.9874, -1.3890};
  const std::vector<double> expected_se   = {0.4417, 0.1971, 0.2046};
  for (std::size_t j = 0; j < 3; ++j) {
    EXPECT_NEAR(model.coefficients()[j], expected_coef[j], 1e-3);
    EXPECT_NEAR(model.standard_errors()[j], expected_se[j], 1e-3);
  }
  EXPECT_NEAR(model.sigma(), 0.8577, 1e-3);
  EXPECT_NEAR(model.r_squared(), 0.9582, 1e-3);
  EXPECT_NEAR(model.f_statistic(), 103.2, 0.2);
}

TEST(LM, InteractionTermMatchesR) {
  const auto df = make_reference_frame();
  LM model(df, "y ~ x1 * color");

  const std::vector<std::string> expected_names = {"(Intercept)", "x1",         "colorgreen",
                                                    "colorred",   "x1:colorgreen", "x1:colorred"};
  ASSERT_EQ(model.coefficient_names().size(), expected_names.size());
  for (std::size_t j = 0; j < expected_names.size(); ++j)
    EXPECT_EQ(model.coefficient_names()[j], expected_names[j]);

  const std::vector<double> expected_coef = {4.08166667, 0.33166667, 0.07955782,
                                             -1.51568027, 0.28363946, 0.23602041};
  for (std::size_t j = 0; j < expected_coef.size(); ++j)
    EXPECT_NEAR(model.coefficients()[j], expected_coef[j], 1e-4);
}

TEST(LM, PolyRawMatchesR) {
  const auto df = make_reference_frame();
  LM model(df, "y ~ poly(x1, 2) + x2");

  ASSERT_EQ(model.coefficient_names().size(), 4u);
  EXPECT_EQ(model.coefficient_names()[1], "poly(x1,2)1");
  EXPECT_EQ(model.coefficient_names()[2], "poly(x1,2)2");

  const std::vector<double> expected_coef = {3.618717532, 1.792624459, 0.008429071, -1.356130952};
  for (std::size_t j = 0; j < expected_coef.size(); ++j)
    EXPECT_NEAR(model.coefficients()[j], expected_coef[j], 1e-4);
  EXPECT_NEAR(model.r_squared(), 0.9445825, 1e-4);
}

TEST(LM, NoInterceptMatchesR) {
  const auto df = make_reference_frame();
  LM model(df, "y ~ x1 + x2 - 1");

  ASSERT_FALSE(model.has_intercept());
  ASSERT_EQ(model.coefficient_names().size(), 2u);
  EXPECT_NEAR(model.coefficients()[0], 2.104920, 1e-4);
  EXPECT_NEAR(model.coefficients()[1], -1.153413, 1e-4);
}

TEST(LM, DotFormulaIncludesAllOtherColumns) {
  const auto df = make_reference_frame();
  LM full(df, "y ~ x1 + x2 + color + w");
  LM dot(df, "y ~ .");

  ASSERT_EQ(full.coefficient_names().size(), dot.coefficient_names().size());
  for (std::size_t j = 0; j < full.coefficients().size(); ++j)
    EXPECT_NEAR(full.coefficients()[j], dot.coefficients()[j], 1e-9);
}

TEST(LM, DropsRowsWithMissingValues) {
  auto df = make_reference_frame();
  df.set_null("x1", 0);
  df.set_null("y", 5);

  LM model(df, "y ~ x1 + x2 + color");
  EXPECT_EQ(model.observations(), 10u);
}

TEST(LM, PredictThrowsOnUnseenCategoricalLevel) {
  const auto df = make_reference_frame();
  LM model(df, "y ~ x1 + x2 + color");

  DataFrame newdata;
  newdata.add_column("x1", std::vector<double>{1.0});
  newdata.add_column("x2", std::vector<double>{1.0});
  newdata.add_column("color", std::vector<std::string>{"purple"});

  EXPECT_THROW(model.predict(newdata), std::runtime_error);
}

TEST(LM, IExpressionAndLogTransformsFit) {
  DataFrame df;
  std::vector<double> x(20);
  std::vector<double> y(20);
  for (int i = 0; i < 20; ++i) {
    x[i] = static_cast<double>(i + 1);
    y[i] = 2.0 + 0.5 * x[i] * x[i] + 3.0 * std::log(x[i]);
  }
  df.add_column("x", x);
  df.add_column("y", y);

  LM model(df, "y ~ I(x^2) + log(x)");
  EXPECT_NEAR(model.coefficients()[0], 2.0, 1e-6);
  EXPECT_NEAR(model.coefficients()[1], 0.5, 1e-6);
  EXPECT_NEAR(model.coefficients()[2], 3.0, 1e-6);
  EXPECT_NEAR(model.r_squared(), 1.0, 1e-9);
}

TEST(LM, SummaryContainsKeyStatistics) {
  const auto df = make_reference_frame();
  LM model(df, "y ~ x1 + x2 + color");
  const auto text = model.summary();
  EXPECT_NE(text.find("Coefficients:"), std::string::npos);
  EXPECT_NE(text.find("colorgreen"), std::string::npos);
  EXPECT_NE(text.find("F-statistic"), std::string::npos);
  EXPECT_NE(text.find("R-squared"), std::string::npos);
}
