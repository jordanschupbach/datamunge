#include <gtest/gtest.h>

#include <datamunge/nn/reconstruction_anomaly.hpp>
namespace n = datamunge::nn;
namespace m = n::models;
TEST(NnReconstructionAnomaly, ScoresFlattenDifferentShapesWidenBeforeSubtractingAndStopPaths) {
  auto x = n::tensor({0., 1., 2., 3.}, {2, 2}).requires_grad_();
  auto y = n::zeros({2, 1, 2}, n::DType::Float64).requires_grad_();
  auto s = m::ReconstructionScores(x, y);
  EXPECT_EQ(s.shape(), (n::Shape{2}));
  EXPECT_EQ(s.dtype(), n::DType::Float64);
  EXPECT_EQ(s.to_vector<double>(), (std::vector<double>{.5, 6.5}));
  EXPECT_FALSE(s.requires_grad());
  EXPECT_TRUE(x.requires_grad());
  EXPECT_TRUE(y.requires_grad());
  auto           copy = s;
  n::NoGradGuard guard;
  x.fill_(9);
  y.fill_(2);
  EXPECT_EQ(copy.to_vector<double>(), (std::vector<double>{.5, 6.5}));
  const auto strided = n::transpose(n::tensor({0., 2., 1., 3.}, {2, 2}), {1, 0});
  EXPECT_TRUE(n::equal_bits(s, m::ReconstructionScores(strided, n::zeros({2, 2}))));
  auto vector = m::ReconstructionScores(n::tensor({-2., 3.}, {2}), n::tensor({1., 1.}, {2}));
  EXPECT_EQ(vector.to_vector<double>(), (std::vector<double>{9, 4}));
  const auto close = n::from_vector(std::vector<float>{1.0000001192092896f}, {1, 1});
  EXPECT_EQ(
      m::ReconstructionScores(close, n::ones({1, 1})).item(),
      std::pow(double(close.item()) - 1, 2));
}
TEST(NnReconstructionAnomaly, InkUsesExplicitReferenceAndFeatureOrderWithoutClipping) {
  auto x = n::tensor({0., 1., 1., 1.}, {2, 2}).requires_grad_();
  auto s = m::InkScores(x, .5);
  EXPECT_EQ(s.to_vector<double>(), (std::vector<double>{0, .5}));
  EXPECT_FALSE(s.requires_grad());
  EXPECT_TRUE(x.requires_grad());
  EXPECT_EQ(m::InkScores(n::tensor({1e16, 1., -1e16}, {1, 3}, n::DType::Float64), .25).item(), .25);
  EXPECT_EQ(m::InkScores(n::tensor({-2., -4.}, {1, 2}), 1).item(), 4);
  auto t = n::transpose(n::tensor({0., 1., 1., 1.}, {2, 2}), {1, 0});
  EXPECT_TRUE(n::equal_bits(s, m::InkScores(t, .5)));
  {
    n::NoGradGuard guard;
    s.fill_(99);
  }
  EXPECT_EQ(x.at({0, 0}), 0);
}
TEST(NnReconstructionAnomaly, NearestRankHasNoInterpolationFuzzOrImplicitQueryRefit) {
  auto scores = n::select(n::arange(0, 40).reshape({20, 2}), 1, 0);
  scores.requires_grad_();
  EXPECT_EQ(m::NominalThreshold(scores), 36);
  EXPECT_EQ(m::NominalThreshold(scores, 1), 38);
  EXPECT_EQ(m::NominalThreshold(scores, std::numeric_limits<double>::denorm_min()), 0);
  EXPECT_EQ(m::NominalThreshold(scores, .5), 18);
  EXPECT_EQ(m::NominalThreshold(scores, std::nextafter(.5, 1.)), 20);
  EXPECT_TRUE(scores.requires_grad());
  EXPECT_EQ(scores.at({1}), 2);
  EXPECT_EQ(m::NominalThreshold(n::tensor({7.}, {1})), 7);
  EXPECT_EQ(m::NominalThreshold(n::tensor({-1., -3., -2.}, {3}), .5), -2);
  // Test/anomaly scores are never arguments to calibration.
  const double threshold = m::NominalThreshold(n::arange(0, 20));
  EXPECT_EQ(threshold, 18);
  const auto r =
      m::RankAnomalies(n::tensor({18., 19.}, {2}), n::tensor({18., 1000.}, {2}), threshold);
  EXPECT_EQ(r.normal_fpr, .5);
  EXPECT_EQ(r.anomaly_tpr, .5);
}
TEST(NnReconstructionAnomaly, RankingTiesStrictThresholdsAndSuppliedPrevalenceAreExplicit) {
  auto       normal  = n::tensor({1., 2.}, {2}, n::DType::Float64).requires_grad_();
  const auto anomaly = n::tensor({2., 3.}, {2});
  auto       r       = m::RankAnomalies(normal, anomaly, 2);
  EXPECT_EQ(r.auroc, .875);
  EXPECT_NEAR(r.average_precision, 5. / 6, 1e-15);
  EXPECT_EQ(r.normal_fpr, 0);
  EXPECT_EQ(r.anomaly_tpr, .5);
  const auto below = m::RankAnomalies(normal, anomaly, std::nextafter(2., -INFINITY));
  EXPECT_EQ(below.normal_fpr, .5);
  EXPECT_EQ(below.anomaly_tpr, 1);
  const auto above = m::RankAnomalies(normal, anomaly, std::nextafter(2., INFINITY));
  EXPECT_EQ(above.normal_fpr, 0);
  EXPECT_EQ(above.anomaly_tpr, .5);
  const auto tied = m::RankAnomalies(n::ones({3}), n::ones({2}, n::DType::Float64), 1);
  EXPECT_EQ(tied.auroc, .5);
  EXPECT_EQ(tied.average_precision, .4);
  EXPECT_EQ(tied.normal_fpr, 0);
  EXPECT_EQ(tied.anomaly_tpr, 0);
  const auto perm = m::RankAnomalies(n::tensor({2., 1.}, {2}), n::tensor({3., 2.}, {2}), 2);
  EXPECT_EQ(perm.auroc, r.auroc);
  EXPECT_EQ(perm.average_precision, r.average_precision);
  const auto perfect = m::RankAnomalies(n::tensor({-3., -2.}, {2}), n::tensor({-1.}, {1}), -2);
  EXPECT_EQ(perfect.auroc, 1);
  EXPECT_EQ(perfect.average_precision, 1);
  const auto inverted = m::RankAnomalies(n::tensor({2., 3.}, {2}), n::tensor({1.}, {1}), 0);
  EXPECT_EQ(inverted.auroc, 0);
  EXPECT_NEAR(inverted.average_precision, 1. / 3, 1e-15);
  EXPECT_TRUE(normal.requires_grad());
}
TEST(NnReconstructionAnomaly, StressTransformsAreOwnedDetachedStridedAndPixelExact) {
  std::vector<float> data(2 * 784);
  for (std::size_t i = 0; i < data.size(); ++i)
    data[i] = float(int(i % 61) - 30) / 30;
  auto original = n::transpose(n::from_vector(data, {2, 28, 28, 1}), {0, 2, 1, 3}).requires_grad_();
  const auto             before = original.contiguous().to_vector<float>();
  std::vector<n::Tensor> outputs;
  for (int kind = 0; kind < 3; ++kind) {
    auto out = m::AnomalyCorruption(original, kind);
    EXPECT_TRUE(out.is_contiguous());
    EXPECT_FALSE(out.requires_grad());
    EXPECT_EQ(out.shape(), original.shape());
    const auto values = out.to_vector<float>();
    for (int b = 0; b < 2; ++b)
      for (int y = 0; y < 28; ++y)
        for (int c = 0; c < 28; ++c) {
          const auto i        = b * 784 + y * 28 + c;
          float      expected = before[i];
          if (kind == 0 && y < 8 && c < 8)
            expected = 1;
          else if (kind == 1)
            expected *= .35f;
          else if (kind == 2)
            expected = c >= 6 ? before[b * 784 + y * 28 + c - 6] : 0;
          EXPECT_EQ(values[i], expected);
        }
    outputs.push_back(out);
  }
  EXPECT_TRUE(original.requires_grad());
  EXPECT_EQ(original.contiguous().to_vector<float>(), before);
  {
    n::NoGradGuard guard;
    original.fill_(9);
  }
  EXPECT_NE(outputs[2].at({0, 0, 10, 0}), 9);
}
TEST(NnReconstructionAnomaly, UnmodifiedAndShiftedSignedZeroBitsRemainExact) {
  auto       x     = n::full({1, 28, 28, 1}, -0.);
  const auto patch = m::AnomalyCorruption(x, 0), shift = m::AnomalyCorruption(x, 2),
             dark = m::AnomalyCorruption(x, 1);
  EXPECT_EQ(patch.at({0, 0, 0, 0}), 1);
  EXPECT_TRUE(std::signbit(patch.at({0, 20, 20, 0})));
  EXPECT_FALSE(std::signbit(shift.at({0, 0, 0, 0})));
  EXPECT_TRUE(std::signbit(shift.at({0, 0, 6, 0})));
  EXPECT_TRUE(std::signbit(dark.at({0, 0, 0, 0})));
}
TEST(NnReconstructionAnomaly, InvalidVectorsSamplesQuantilesAndComputedOverflowAreRejected) {
  for (const auto& bad :
       {n::Tensor(),
        n::zeros({0}),
        n::ones({2, 2}),
        n::ones({2}, n::DType::Int32),
        n::full({2}, NAN)}) {
    EXPECT_THROW(m::NominalThreshold(bad), std::invalid_argument);
    EXPECT_THROW(m::RankAnomalies(bad, n::ones({2}), 0), std::invalid_argument);
  }
  for (double q : {0., -1., 1.00001, double(INFINITY), double(NAN)})
    EXPECT_THROW(m::NominalThreshold(n::ones({2}), q), std::invalid_argument);
  for (double t : {INFINITY, -INFINITY, NAN})
    EXPECT_THROW(m::RankAnomalies(n::ones({2}), n::ones({2}), t), std::invalid_argument);
  EXPECT_THROW(m::ReconstructionScores(n::ones({2, 3}), n::ones({3, 2})), std::invalid_argument);
  EXPECT_THROW(m::ReconstructionScores(n::ones({2, 3}), n::ones({2, 2})), std::invalid_argument);
  EXPECT_THROW(m::ReconstructionScores(n::zeros({0, 2}), n::zeros({0, 2})), std::invalid_argument);
  EXPECT_THROW(m::InkScores(n::ones({2, 2}, n::DType::Int32), 0), std::invalid_argument);
  EXPECT_THROW(m::InkScores(n::ones({2, 2}), INFINITY), std::invalid_argument);
  const auto huge = n::full({1, 2}, std::numeric_limits<double>::max(), n::DType::Float64);
  EXPECT_THROW(
      m::ReconstructionScores(huge, n::zeros({1, 2}, n::DType::Float64)), std::invalid_argument);
  EXPECT_THROW(m::InkScores(huge, 0), std::invalid_argument);
  EXPECT_THROW(
      m::InkScores(
          n::full({1}, std::numeric_limits<double>::max(), n::DType::Float64),
          -std::numeric_limits<double>::max()),
      std::invalid_argument);
}
TEST(NnReconstructionAnomaly, StressRejectsMalformedGeometryKindsAndNonfiniteInput) {
  for (const auto& x :
       {n::Tensor(),
        n::ones({28, 28, 1}),
        n::ones({2, 784}),
        n::zeros({0, 28, 28, 1}),
        n::ones({1, 28, 28, 3}),
        n::ones({1, 28, 28, 1}, n::DType::Float64),
        n::full({1, 28, 28, 1}, INFINITY)})
    EXPECT_THROW(m::AnomalyCorruption(x, 0), std::invalid_argument);
  EXPECT_THROW(m::AnomalyCorruption(n::ones({1, 28, 28, 1}), -1), std::invalid_argument);
  EXPECT_THROW(m::AnomalyCorruption(n::ones({1, 28, 28, 1}), 3), std::invalid_argument);
}
