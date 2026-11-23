#include <datamunge/stats/anova_measures.hpp>
#include <gtest/gtest.h>

#include <stdexcept>

namespace {

using namespace datamunge::stats;
using Matrix = datamunge::linalg::DenseMatrix<double>;

TEST(AnovaMeasures, OverallAndPartialEffectSizesUseTheirOwnDenominators) {
  const auto fit = anova_effect_sizes(4, 6, 20, 1, 6);
  EXPECT_DOUBLE_EQ(fit.eta_squared, .2);
  EXPECT_DOUBLE_EQ(fit.partial_eta_squared, .4);
  EXPECT_DOUBLE_EQ(fit.omega_squared, 3.0 / 21);
  EXPECT_DOUBLE_EQ(fit.partial_omega_squared, 3.0 / 11);
  const auto scaled = anova_effect_sizes(4e300, 6e300, 20e300, 1, 6);
  EXPECT_NEAR(scaled.partial_omega_squared, fit.partial_omega_squared, 1e-15);
  const auto negative = anova_effect_sizes(0, 6, 6, 1, 6);
  EXPECT_DOUBLE_EQ(negative.partial_omega_squared, -1.0 / 7);
  EXPECT_THROW(anova_effect_sizes(0, 0, 1, 1, 6), std::invalid_argument);
  EXPECT_THROW(anova_effect_sizes(4, 6, 20, 1, 0), std::invalid_argument);
}

TEST(AnovaMeasures, IntraclassCorrelationsMatchANOVAMeanSquaresFromR) {
  const Matrix ratings(5, 3, {4, 6, 7, 6, 8, 9, 7, 9, 9, 12, 14, 15, 13, 15, 18});
  constexpr auto one = IccModel::OneWayRandom;
  constexpr auto two = IccModel::TwoWayRandom;
  constexpr auto mixed = IccModel::TwoWayMixed;
  constexpr auto agreement = IccDefinition::AbsoluteAgreement;
  constexpr auto consistency = IccDefinition::Consistency;
  constexpr auto single = IccUnit::Single;
  constexpr auto average = IccUnit::Average;
  EXPECT_NEAR(intraclass_correlation(ratings, one, agreement, single).estimate, .84686774941995357, 1e-14);
  EXPECT_NEAR(intraclass_correlation(ratings, one, agreement, average).estimate, .94315245478036169, 1e-14);
  EXPECT_NEAR(intraclass_correlation(ratings, two, agreement, single).estimate, .85333333333333339, 1e-14);
  EXPECT_NEAR(intraclass_correlation(ratings, two, agreement, average).estimate, .94581280788177347, 1e-14);
  EXPECT_NEAR(intraclass_correlation(ratings, two, consistency, single).estimate, .97709923664122145, 1e-14);
  const auto fit = intraclass_correlation(ratings, mixed, consistency, average);
  EXPECT_NEAR(fit.estimate, .99224806201550386, 1e-14);
  EXPECT_EQ(fit.model, mixed);
  EXPECT_EQ(fit.definition, consistency);
  EXPECT_EQ(fit.unit, average);
  EXPECT_EQ(fit.targets, 5u);
  EXPECT_EQ(fit.raters, 3u);
  EXPECT_THROW(intraclass_correlation(ratings, one, consistency, single), std::invalid_argument);
}

TEST(AnovaMeasures, ConsistencyRemovesRaterOffsetsAndDetectsUndefinedDesigns) {
  constexpr auto two = IccModel::TwoWayMixed;
  constexpr auto agreement = IccDefinition::AbsoluteAgreement;
  constexpr auto consistency = IccDefinition::Consistency;
  constexpr auto single = IccUnit::Single;
  const Matrix shifted(3, 3, {0, 1e200, 2, 1, 1e200, 3, 2, 1e200, 4});
  const Matrix centered(3, 3, {0, 0, 2, 1, 0, 3, 2, 0, 4});
  EXPECT_NEAR(intraclass_correlation(shifted, two, consistency, single).estimate,
              intraclass_correlation(centered, two, consistency, single).estimate, 1e-15);
  const Matrix raters_only(3, 3, {1, 2, 4, 1, 2, 4, 1, 2, 4});
  EXPECT_DOUBLE_EQ(intraclass_correlation(raters_only, two, agreement, single).estimate, 0);
  EXPECT_THROW(intraclass_correlation(raters_only, two, consistency, single), std::invalid_argument);
  EXPECT_LT(intraclass_correlation(Matrix(3, 2, {0, 2, 2, 0, 1, 1}),
      IccModel::OneWayRandom, agreement, single).estimate, 0);
}

} // namespace
