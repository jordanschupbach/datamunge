#include <gtest/gtest.h>

#include <datamunge/algorithms/logitboost.hpp>
#include <limits>
using namespace datamunge::algorithms;
TEST(LogitBoostTrace, ActualNewtonAndStump) {
  std::vector<std::vector<double>> x{{-1}, {1}};
  std::vector<int>                 y{0, 1};
  auto                             t = logitboost_trace(x, y, 3);
  ASSERT_EQ(t.rounds.size(), 3);
  EXPECT_EQ(t.rounds[0].responses, (std::vector<double>{-2, 2}));
  EXPECT_EQ(t.rounds[0].weights, (std::vector<double>{.25, .25}));
  EXPECT_EQ(t.rounds[0].stump.left_value, -2);
  EXPECT_EQ(t.rounds[0].stump.right_value, 2);
  auto ordinary = logitboost_train(x, y, 3);
  for (std::size_t k = 0; k < 3; ++k) {
    EXPECT_EQ(t.model.log_likelihood[k], ordinary.log_likelihood[k]);
    auto&           s  = t.rounds[k];
    double          ll = 0;
    LogitBoostModel prefix;
    prefix.stumps.assign(t.model.stumps.begin(), t.model.stumps.begin() + k + 1);
    for (std::size_t i = 0; i < 2; ++i) {
      double p = logitboost_probability(prefix, x[i]);
      EXPECT_EQ(p, s.probabilities[i]);
      ll += y[i] ? std::log(p) : std::log(1 - p);
    }
    EXPECT_DOUBLE_EQ(ll, s.log_likelihood);
  }
  EXPECT_EQ(logitboost_predict(t.model, {-1}), 0);
  EXPECT_EQ(logitboost_predict(t.model, {1}), 1);
  auto clipped = logitboost_trace(x, y, 1, .5);
  EXPECT_EQ(clipped.rounds[0].responses, (std::vector<double>{-.5, .5}));
}
TEST(LogitBoostTrace, CompleteGuardsAndFiniteThresholds) {
  EXPECT_THROW(logitboost_train({{1}, {2, 3}}, {0, 1}), std::invalid_argument);
  EXPECT_THROW(logitboost_train({{1}}, {2}), std::invalid_argument);
  EXPECT_THROW(logitboost_train({{}}, {0}), std::invalid_argument);
  EXPECT_THROW(logitboost_train({{NAN}}, {0}), std::invalid_argument);
  EXPECT_THROW(logitboost_train({{1}}, {0}, 0, 0), std::invalid_argument);
  EXPECT_THROW(logitboost_trace({{-1}, {1}}, {0, 1}, 3, 4, 20), std::length_error);
  EXPECT_EQ(logitboost_trace({{-1}, {1}}, {0, 1}, 3, 4, 21).rounds.size(), 3);
  const auto m = std::numeric_limits<double>::max();
  auto       t = logitboost_trace({{-m}, {m}}, {0, 1}, 1);
  EXPECT_TRUE(std::isfinite(t.rounds[0].stump.threshold));
  EXPECT_EQ(logitboost_predict(t.model, {-m}), 0);
  EXPECT_EQ(logitboost_predict(t.model, {m}), 1);
  EXPECT_THROW(logitboost_probability(t.model, {}), std::invalid_argument);
}
#include <datamunge/algorithms/logitboost_study.hpp>
TEST(LogitBoostStudy, OriginalFixtureAndIndependentHoldout) {
 const auto d=logitboost_diagonal_data();ASSERT_EQ(d.labels.size(),271);for(std::size_t i=0;i<d.labels.size();++i){EXPECT_GE(std::abs(d.features[i][0]+d.features[i][1]),.3);EXPECT_EQ(d.labels[i],d.features[i][0]+d.features[i][1]>0);}
 const auto s=logitboost_iris_study();ASSERT_EQ(s.training.labels.size(),80);ASSERT_EQ(s.testing.labels.size(),20);ASSERT_EQ(s.model.stumps.size(),40);for(std::size_t i=0;i<20;++i){EXPECT_EQ(s.probabilities[i],logitboost_probability(s.model,s.testing.features[i]));EXPECT_EQ(s.predictions[i],s.probabilities[i]>=.5);}
}
