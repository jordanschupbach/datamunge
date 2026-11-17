
#include <gtest/gtest.h>

#include <dm/dm.hpp>

#include "../fixtures/nn_grad_cam_checks.hpp"
TEST(GradCam, EveryHostWeightAlphaMapAnd24TrueExpAffineHigherDerivatives) {
  FormulaChecks();
}
TEST(GradCam, AllThirteenSharedModelOutputsMatchOriginalWorkflowAndImmutableWeights) {
  namespace n = dm::nn;
  namespace m = n::models;
  n::set_num_threads(2);
  n::ParameterStore source(1103);
  {
    n::NoGradGuard guard;
    m::SaliencyCnn(source, n::zeros({1, 28, 28, 1}));
  }
  const auto pools = m::LoadSaliency(std::filesystem::path(dm::datasets::data_home()) / "saliency");
  std::vector<std::int64_t> rows;
  std::array<bool, 10>      seen{};
  const auto                labels = pools[2].labels.to_vector<int>();
  for (std::size_t row = 0; row < labels.size(); ++row)
    if (!seen[labels[row]]) {
      seen[labels[row]] = true;
      rows.push_back(row);
    }
  const auto d = m::SaliencyRows(pools[2], rows);
  n::Tensor  targets;
  {
    n::NoGradGuard guard;
    targets = n::argmax(m::SaliencyCnn(source, d.images).logits, 1).to(n::DType::Int32);
  }
  const auto before = source.flatten();
  const auto a      = m::SaliencyCam(source, d.images, targets);
  const auto net    = m::SaliencyCnn(source, d.images);
  const auto gradient =
      n::grad(n::sum(m::SaliencyScore(net.logits, targets)), {net.conv2})[0].detach();
  n::NoGradGuard guard;
  const auto     activation = net.conv2.detach();
  const auto     cams       = m::ClassActivationMaps(activation, gradient);
  const auto     normalized = m::NormalizeCam(cams.grad_map), plus = m::NormalizeCam(cams.plus_map);
  auto           resize = [](const n::Tensor& map) {
    return n::resize_bilinear(n::expand_dims(map, 3), {28, 28}, false, true)
        .reshape({map.size(0), 28, 28});
  };
  const std::vector<n::Tensor> b{
      activation,
      gradient,
      cams.grad_weights,
      cams.grad_signed,
      cams.grad_map,
      cams.positive_alpha,
      cams.plus_weights,
      cams.plus_map,
      normalized,
      plus,
      resize(normalized),
      resize(plus),
      net.logits.detach()};
  std::vector<n::Tensor> values = {
      a.activation,
      a.gradient,
      a.maps.grad_weights,
      a.maps.grad_signed,
      a.maps.grad_map,
      a.maps.positive_alpha,
      a.maps.plus_weights,
      a.maps.plus_map,
      a.normalized,
      a.plus_normalized,
      a.resized,
      a.plus_resized,
      a.logits};
  ASSERT_EQ(values.size(), b.size());
  for (int i = 0; i < 13; ++i)
    EXPECT_EQ(n::max(n::abs(values[i] - b[i])).item(), 0);
  EXPECT_EQ(before, source.flatten());
  EXPECT_LE(n::max(n::abs(m::SaliencyCamTail(source, a.activation) - a.logits)).item(), 2e-6);
}
TEST(GradCam, InvalidNegativeFeatureNonfiniteAndNormalizationContracts) {
  namespace n = dm::nn;
  namespace m = n::models;
  EXPECT_THROW(
      m::ClassActivationMaps(n::full({1, 2, 2, 1}, -1), n::ones({1, 2, 2, 1})),
      std::invalid_argument);
  EXPECT_THROW(
      m::ClassActivationMaps(n::ones({1, 2, 2, 1}), n::full({1, 2, 2, 1}, NAN)),
      std::invalid_argument);
  EXPECT_EQ(n::sum(m::NormalizeCam(n::zeros({2, 3, 3}))).item(), 0);
  EXPECT_THROW(m::NormalizeCam(n::full({1, 2, 2}, -1)), std::invalid_argument);
}
