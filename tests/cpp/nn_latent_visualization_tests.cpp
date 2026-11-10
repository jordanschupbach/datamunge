#include <gtest/gtest.h>

#include <datamunge/nn/gradcheck.hpp>
#include <datamunge/nn/latent_visualization.hpp>
namespace n = datamunge::nn;
namespace m = n::models;

TEST(NnLatentVisualization, DoubleRowsFlattenLogicalAxesOwnStorageAndRetainInputPaths) {
  auto x = n::transpose(
               n::tensor({0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.}, {2, 3, 2}), {1, 0, 2})
               .requires_grad_();
  const auto before = x.contiguous().to_vector<double>();
  auto       rows   = m::DoubleSampleRows(x);
  EXPECT_EQ(rows.dtype(), n::DType::Float64);
  EXPECT_EQ(rows.shape(), (n::Shape{3, 4}));
  EXPECT_TRUE(rows.is_contiguous());
  EXPECT_EQ(rows.to_vector<double>(), before);
  const auto g = n::grad(n::sum(rows), {x})[0];
  EXPECT_EQ(g.to_vector<double>(), std::vector<double>(12, 1));
  {
    n::NoGradGuard guard;
    rows.fill_(44);
  }
  EXPECT_EQ(x.contiguous().to_vector<double>(), before);
  EXPECT_EQ(m::DoubleSampleRows(n::ones({3})).shape(), (n::Shape{3, 1}));
}
TEST(NnLatentVisualization, CenteredSampleVarianceAndFrozenSingleQueryTransform) {
  const auto t   = n::DType::Float64;
  auto       fit = n::tensor({-1., 0., 0., 0., 1., 0.}, {3, 2}, t).requires_grad_();
  m::SvdPca  p(fit, 1);
  EXPECT_EQ(p.mean().to_vector<double>(), (std::vector<double>{0, 0}));
  EXPECT_NEAR(p.variances().item(), 1, 1e-14);
  EXPECT_FALSE(p.mean().requires_grad());
  EXPECT_FALSE(p.components().requires_grad());
  EXPECT_FALSE(p.scores().requires_grad());
  const auto query         = n::tensor({2., 3.}, {1, 2}, t),
             reconstructed = p.Reconstruct(p.Transform(query));
  EXPECT_NEAR(reconstructed.at({0, 0}), 2, 1e-14);
  EXPECT_NEAR(reconstructed.at({0, 1}), 0, 1e-14);
  const auto score = p.Transform(query).detach();
  p.mean().fill_(9);
  p.components().fill_(9);
  p.variances().fill_(9);
  p.scores().fill_(9);
  {
    n::NoGradGuard guard;
    fit.fill_(9);
  }
  EXPECT_TRUE(n::equal_bits(score, p.Transform(query)));
  EXPECT_EQ(p.Reconstruct(n::zeros({2, 1}, t)).to_vector<double>(), std::vector<double>(4, 0));
}
TEST(NnLatentVisualization, PrefixReconstructionConstantWideMatricesAndNoVarianceScaling) {
  const auto t   = n::DType::Float64;
  const auto fit = n::tensor({-3., -1., -3., 1., 3., -1., 3., 1.}, {4, 2}, t);
  m::SvdPca  p(fit, 2);
  EXPECT_NEAR(p.variances().at({0}), 12, 1e-12);
  EXPECT_NEAR(p.variances().at({1}), 4. / 3, 1e-12);
  const auto raw = n::tensor({2., .5}, {1, 2}, t), all = p.Transform(raw);
  EXPECT_NEAR(p.Reconstruct(all).at({0, 0}), 2, 1e-13);
  EXPECT_NEAR(p.Reconstruct(all).at({0, 1}), .5, 1e-13);
  const auto one = p.Reconstruct(n::slice(all, {0, 0}, {1, 1}));
  EXPECT_NEAR(one.at({0, 0}), 2, 1e-13);
  EXPECT_NEAR(one.at({0, 1}), 0, 1e-13);
  m::SvdPca constant(n::full({3, 4}, 7, t), 3);
  EXPECT_EQ(constant.variances().to_vector<double>(), std::vector<double>(3, 0));
  EXPECT_EQ(
      constant.Reconstruct(constant.scores()).to_vector<double>(), std::vector<double>(12, 7));
  m::SvdPca wide(n::tensor({0., 1., 2., 3., 4., 5.}, {2, 3}, t), 2);
  EXPECT_EQ(wide.components().shape(), (n::Shape{3, 2}));
  EXPECT_EQ(wide.Transform(n::zeros({1, 3}, t)).shape(), (n::Shape{1, 2}));
}
TEST(NnLatentVisualization, QueryAndInverseFirstSecondDerivativesWithStoppedFittingState) {
  const auto t       = n::DType::Float64;
  auto       fitting = n::tensor({-3., -1., -3., 1., 3., -1., 3., 1.}, {4, 2}, t).requires_grad_();
  m::SvdPca  p(fitting, 2);
  auto       query          = n::tensor({.2, .4, -.3, .6}, {2, 2}, t).requires_grad_();
  n::TensorFunction project = [&](const auto& v) { return n::mean(n::square(p.Transform(v[0]))); };
  EXPECT_TRUE(n::gradcheck(project, {query}).ok);
  EXPECT_TRUE(n::gradgradcheck(project, {query}).ok);
  n::TensorFunction reconstruct = [&](const auto& v) {
    return n::mean(n::square(p.Reconstruct(v[0])));
  };
  EXPECT_TRUE(n::gradcheck(reconstruct, {query}).ok);
  EXPECT_TRUE(n::gradgradcheck(reconstruct, {query}).ok);
  const auto g = n::grad(n::sum(p.Transform(query)) + 0 * n::sum(fitting), {fitting})[0];
  EXPECT_EQ(g.to_vector<double>(), std::vector<double>(8, 0));
  const auto strided = n::transpose(n::transpose(query).contiguous());
  EXPECT_TRUE(n::equal_bits(p.Transform(strided), p.Transform(query)));
}
TEST(NnLatentVisualization, EuclideanFeatureOrderDistanceTiesAndDetachedInt32Neighbors) {
  const auto fit   = n::tensor({-1., 0., 1., 0., 0., 2., 4., 0.}, {4, 2}, n::DType::Float64);
  const auto query = n::zeros({1, 2}).requires_grad_();
  auto       rows  = m::NearestFitRows(fit, query, 4);
  EXPECT_EQ(rows.dtype(), n::DType::Int32);
  EXPECT_EQ(rows.shape(), (n::Shape{1, 4}));
  EXPECT_EQ(rows.to_vector<int>(), (std::vector<int>{0, 1, 2, 3}));
  EXPECT_FALSE(rows.requires_grad());
  EXPECT_TRUE(
      n::equal_bits(
          m::NearestFitRows(n::transpose(n::transpose(fit).contiguous()), query, 4), rows));
  const auto images = fit.reshape({4, 1, 2, 1});
  EXPECT_TRUE(n::equal_bits(m::NearestFitRows(images, query.reshape({1, 2, 1}), 4), rows));
  rows.fill_(0);
  EXPECT_EQ(fit.at({0, 0}), -1);
}
TEST(NnLatentVisualization, OverlapMembershipDuplicatesAndPosthocFirstNeighborLabels) {
  const auto ref  = n::from_vector(std::vector<int>{0, 1, 1, 1}, {2, 2}),
             cand = n::from_vector(std::vector<std::int64_t>{1, 2, 1, 1}, {2, 2});
  EXPECT_EQ(m::NeighborOverlap(ref, cand), .75);
  EXPECT_EQ(
      m::NeighborOverlap(
          n::from_vector(std::vector<int>{1, 1}, {1, 2}),
          n::from_vector(std::vector<int>{1, 7}, {1, 2})),
      1);
  const auto neighbors = n::from_vector(std::vector<int>{0, 1, 1, 0}, {2, 2});
  const auto labels    = n::from_vector(std::vector<std::int64_t>{4, 8});
  EXPECT_EQ(m::NeighborLabelAccuracy(neighbors, labels, n::from_vector(std::vector<int>{4, 8})), 1);
  EXPECT_EQ(m::NeighborLabelAccuracy(neighbors, labels, n::from_vector(std::vector<int>{8, 4})), 0);
  EXPECT_EQ(
      m::NeighborLabelAccuracy(neighbors, labels, n::from_vector(std::vector<int>{4, 9})), .5);
}
TEST(NnLatentVisualization, ProjectionRejectsBadRowsWidthsCountsAndNonfiniteComputation) {
  for (const auto& x : std::vector<n::Tensor>{
           n::Tensor{},
           n::from_scalar(1.f),
           n::zeros({0, 2}),
           n::ones({3, 2}, n::DType::Int32),
           n::full({3, 2}, NAN)})
    EXPECT_THROW(m::DoubleSampleRows(x), std::invalid_argument);
  EXPECT_THROW(m::SvdPca(n::ones({1, 3}), 1), std::invalid_argument);
  EXPECT_THROW(m::SvdPca(n::ones({3, 2}), 0), std::invalid_argument);
  EXPECT_THROW(m::SvdPca(n::ones({2, 3}), 3), std::invalid_argument);
  EXPECT_THROW(m::SvdPca(n::full({2, 2}, 1e308, n::DType::Float64), 1), std::invalid_argument);
  m::SvdPca p(n::ones({3, 2}), 1);
  EXPECT_THROW(p.Transform(n::ones({2, 3})), std::invalid_argument);
  EXPECT_THROW(p.Reconstruct(n::ones({2, 2})), std::invalid_argument);
  EXPECT_THROW(p.Reconstruct(n::ones({1, 2, 1})), std::invalid_argument);
}
TEST(NnLatentVisualization, RetrievalRejectsInvalidLabelsTablesGeometryAndDistanceOverflow) {
  EXPECT_THROW(m::NearestFitRows(n::ones({2, 3}), n::ones({1, 4}), 1), std::invalid_argument);
  EXPECT_THROW(m::NearestFitRows(n::ones({2, 3}), n::ones({1, 3}), 0), std::invalid_argument);
  EXPECT_THROW(m::NearestFitRows(n::ones({2, 3}), n::ones({1, 3}), 3), std::invalid_argument);
  EXPECT_THROW(
      m::NearestFitRows(
          n::full({2, 1}, 1e308, n::DType::Float64), n::full({1, 1}, -1e308, n::DType::Float64), 1),
      std::invalid_argument);
  const auto good = n::zeros({2, 2}, n::DType::Int32), labels = n::zeros({2}, n::DType::Int64);
  for (const auto& bad : std::vector<n::Tensor>{
           n::Tensor{},
           n::zeros({2, 2}),
           n::zeros({0, 2}, n::DType::Int32),
           n::full({2, 2}, -1, n::DType::Int32)})
    EXPECT_THROW(m::NeighborOverlap(good, bad), std::invalid_argument);
  EXPECT_THROW(m::NeighborOverlap(good, n::zeros({1, 2}, n::DType::Int32)), std::invalid_argument);
  EXPECT_THROW(
      m::NeighborLabelAccuracy(good, labels, n::zeros({1}, n::DType::Int32)),
      std::invalid_argument);
  EXPECT_THROW(
      m::NeighborLabelAccuracy(n::full({2, 2}, 2, n::DType::Int32), labels, labels),
      std::invalid_argument);
  EXPECT_THROW(
      m::NeighborLabelAccuracy(good, n::full({2}, -1, n::DType::Int64), labels),
      std::invalid_argument);
}
