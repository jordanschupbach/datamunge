#include <gtest/gtest.h>

#include <chrono>
#include <datamunge/nn/cifar10_study.hpp>
#include <datamunge/nn/classification_analysis.hpp>
#include <datamunge/nn/image_overlap.hpp>

namespace n = datamunge::nn;

TEST(NnClassificationAnalysis, UniformThreeClassScoresHaveKnownProperScores) {
  const auto y   = n::tensor({0, 1, 2}).to(n::DType::Int64);
  const auto out = n::study::classification_diagnostics(n::zeros({3, 3}), y);
  EXPECT_NEAR(out.nll, std::log(3.), 1e-12);
  EXPECT_NEAR(out.brier, 2. / 3, 1e-12);
  EXPECT_NEAR(out.accuracy, 1. / 3, 1e-12);
  EXPECT_NEAR(out.ece, 0., 1e-12);
  EXPECT_EQ(std::accumulate(out.bin_count.begin(), out.bin_count.end(), 0), 3);
}

TEST(NnClassificationAnalysis, TemperatureIsValidationFittedAndPreservesDecisions) {
  const auto   z = n::tensor({4., 0., 4., 0.}, {2, 2}), y = n::tensor({0, 1}).to(n::DType::Int64);
  const double t = n::study::fit_temperature(z, y);
  EXPECT_GT(t, 9.9);
  const auto raw    = n::study::classification_diagnostics(z, y),
             scaled = n::study::classification_diagnostics(z, y, t);
  EXPECT_LT(scaled.nll, raw.nll);
  EXPECT_LT(scaled.brier, raw.brier);
  EXPECT_DOUBLE_EQ(scaled.accuracy, raw.accuracy);
  EXPECT_NEAR(n::study::fit_temperature(z + 50, y), t, 1e-6);
  EXPECT_DOUBLE_EQ(n::study::fit_temperature(n::zeros({2, 2}), y), 1.);
}

TEST(NnClassificationAnalysis, ConfidenceOneIsIncludedInLastBinAndEmptyBinsAreFinite) {
  const auto out = n::study::classification_diagnostics(
      n::tensor({1000., -1000.}, {1, 2}), n::zeros({1}, n::DType::Int64));
  EXPECT_EQ(out.bin_count.back(), 1);
  EXPECT_DOUBLE_EQ(out.nll, 0);
  EXPECT_DOUBLE_EQ(out.brier, 0);
  EXPECT_DOUBLE_EQ(out.ece, 0);
  EXPECT_EQ(out.bin_count.front(), 0);
}

TEST(NnClassificationAnalysis, RejectsInvalidScoresLabelsAndOptions) {
  const auto z = n::zeros({2, 3}), y = n::zeros({2}, n::DType::Int64);
  EXPECT_THROW(n::study::classification_diagnostics(z, y, 0), std::invalid_argument);
  EXPECT_THROW(
      n::study::classification_diagnostics(n::full({2, 3}, 1e308, n::DType::Float64), y, 1e-100),
      std::invalid_argument);
  EXPECT_THROW(n::study::classification_diagnostics(z, y, 1, 0), std::invalid_argument);
  EXPECT_THROW(n::study::classification_diagnostics(z, n::ones({2})), std::invalid_argument);
  EXPECT_THROW(
      n::study::classification_diagnostics(z, n::full({2}, 3, n::DType::Int64)),
      std::invalid_argument);
  EXPECT_THROW(
      n::study::classification_diagnostics(n::full({2, 3}, NAN), y), std::invalid_argument);
  EXPECT_THROW(
      n::study::classification_diagnostics(n::zeros({0, 3}), n::zeros({0}, n::DType::Int64)),
      std::invalid_argument);
}

TEST(NnCifar10Study, SeededAugmentationKeepsGeometryAndRejectsWrongResolution) {
  n::Generator first(13), second(13);
  const auto   images = n::linspace(0, 1, 2 * 32 * 32 * 3).reshape({2, 32, 32, 3});
  const auto   a      = n::study::augment_cifar10(images, first),
             b        = n::study::augment_cifar10(images, second);
  EXPECT_EQ(a.shape(), images.shape());
  EXPECT_DOUBLE_EQ(n::max(n::abs(a - b)).item(), 0);
  EXPECT_GE(n::min(a).item(), 0);
  EXPECT_LE(n::max(a).item(), 1);
  EXPECT_THROW(n::study::augment_cifar10(n::zeros({1, 28, 28, 3}), first), std::invalid_argument);
}

TEST(NnCifar10Study, PublicConfigurationRetainsAllEightResidualBlocks) {
  n::set_num_threads(2);
  n::ParameterStore p(17), state(17);
  const auto        z =
      n::study::cifar10_logits(p, state, n::zeros({1, 32, 32, 3}), n::zeros({3}), n::ones({3}));
  EXPECT_EQ(z.shape(), (n::Shape{1, 10}));
  EXPECT_EQ(p.count(), 701466);
  EXPECT_TRUE(p.contains("stage4/block2/conv2/W"));
}

TEST(NnCifar10Study, RejectsPersistedRowOverlapBeforeDatasetAcquisition) {
  auto ids = n::arange(0, 45000, 1, n::DType::Int64).to_vector<int64_t>();
  ids[1]   = ids[0];
  const auto path =
      std::filesystem::path(::testing::TempDir()) /
      ("datamunge-cifar-overlap-" +
       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".npz");
  n::save_npz(
      path.string(),
      {{"mean", n::zeros({3})},
       {"sd", n::ones({3})},
       {"fit_ids", n::from_vector(ids, {45000})},
       {"validation_ids", n::arange(45000, 50000, 1, n::DType::Int64)}});
  EXPECT_THROW(n::study::restore_cifar10_study(path.string()), std::invalid_argument);
  std::filesystem::remove(path);
}

TEST(NnImageOverlap, CountsUnorderedPairsConflictsAndUniqueLaterRows) {
  const n::ImageDataset fit{
      n::tensor({0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7}, {3, 2, 2, 1}),
      n::tensor({0, 0, 1}).to(n::DType::Int64)};
  const n::ImageDataset val{
      n::tensor({4, 5, 6, 7, 9, 9, 9, 9}, {2, 2, 2, 1}), n::tensor({2, 3}).to(n::DType::Int64)};
  const n::ImageDataset test{
      n::tensor({0, 1, 2, 3, 4, 5, 6, 7, 9, 9, 9, 9}, {3, 2, 2, 1}),
      n::tensor({0, 1, 3}).to(n::DType::Int64)};
  const auto a = n::study::audit_image_overlap({fit, val, test});
  EXPECT_EQ(a.pairs[0][0], 1);
  EXPECT_EQ(a.pairs[0][1], 1);
  EXPECT_EQ(a.pairs[0][2], 3);
  EXPECT_EQ(a.pairs[1][2], 2);
  EXPECT_EQ(a.pairs[2][2], 0);
  EXPECT_EQ(a.conflicting_label_pairs, 2);
  EXPECT_EQ(a.rows_matching_first[1], (std::vector<int64_t>{0}));
  EXPECT_EQ(a.rows_matching_first[2], (std::vector<int64_t>{0, 1}));
}

TEST(NnImageOverlap, CopiesNoncontiguousImagesAndUsesExactBytes) {
  const auto x = n::tensor({0, 1, 2, 3}, {1, 2, 2, 1});
  const auto y = n::zeros({1}, n::DType::Int64);
  const auto z = n::transpose(x, {0, 2, 1, 3});
  EXPECT_FALSE(z.is_contiguous());
  const auto a = n::study::audit_image_overlap({{z, y}, {z.contiguous(), y}, {z + 1e-5, y}});
  EXPECT_EQ(a.pairs[0][1], 1);
  EXPECT_EQ(a.pairs[0][2], 0);
  EXPECT_THROW(
      n::study::audit_image_overlap({{n::full({1, 2, 2, 1}, NAN), y}}), std::invalid_argument);
  EXPECT_THROW(
      n::study::audit_image_overlap({{x, y}, {n::zeros({1, 3, 2, 1}), y}}), std::invalid_argument);
}
