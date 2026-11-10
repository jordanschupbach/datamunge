#include <gtest/gtest.h>

#include <datamunge/nn/encoder_pretraining.hpp>
#include <datamunge/nn/gradcheck.hpp>
#include <datamunge/nn/optim.hpp>
namespace n = datamunge::nn;
namespace m = n::models;

TEST(NnEncoderPretraining, PopulationScalingPreservesOriginalTwoPassRoundingAndStrides) {
  for (auto type : {n::DType::Float32, n::DType::Float64}) {
    auto code = n::tensor({1., 5., 2., 5., 4., 5.}, {3, 2}, type).requires_grad_();
    auto fit  = m::FitTransferScaling(code);
    EXPECT_EQ(fit.center.dtype(), type);
    EXPECT_EQ(fit.center.at({0}), type == n::DType::Float32 ? double(float(7. / 3)) : 7. / 3);
    EXPECT_EQ(
        fit.scale.at({0}),
        type == n::DType::Float32 ? double(float(std::sqrt(14. / 9))) : std::sqrt(14. / 9));
    EXPECT_EQ(fit.scale.at({1}), type == n::DType::Float32 ? double(float(1e-6)) : 1e-6);
    EXPECT_FALSE(fit.center.requires_grad());
    EXPECT_FALSE(fit.scale.requires_grad());
    const auto strided = n::transpose(n::transpose(code).contiguous());
    EXPECT_FALSE(strided.is_contiguous());
    EXPECT_TRUE(n::equal_bits(m::FitTransferScaling(strided).center, fit.center));
    EXPECT_TRUE(n::equal_bits(m::FitTransferScaling(strided).scale, fit.scale));
    const auto single = m::FitTransferScaling(n::full({1, 2}, -4, type), .25);
    EXPECT_EQ(single.center.to_vector<double>(), (std::vector<double>{-4, -4}));
    EXPECT_EQ(single.scale.to_vector<double>(), (std::vector<double>{.25, .25}));
    fit.center.fill_(99);
    EXPECT_EQ(code.at({0, 0}), 1);
  }
}
TEST(NnEncoderPretraining, FeaturesKeepInputPathAndStopBothEncoderPaths) {
  const auto t      = n::DType::Float64;
  auto       image  = n::full({2, 3}, .4, t).requires_grad_();
  auto       weight = n::full({3, 2}, .5, t).requires_grad_();
  auto       bias   = n::full({2}, .2, t).requires_grad_();
  auto       code   = m::FrozenEncoderFeatures(image, weight, bias);
  auto       gradients =
      n::grad(n::sum(code) + 0 * n::sum(weight) + 0 * n::sum(bias), {image, weight, bias});
  EXPECT_EQ(gradients[0].to_vector<double>(), (std::vector<double>(6, 1)));
  EXPECT_EQ(gradients[1].to_vector<double>(), (std::vector<double>(6, 0)));
  EXPECT_EQ(gradients[2].to_vector<double>(), (std::vector<double>(2, 0)));
  n::TensorFunction f = [&](const auto& a) {
    return n::mean(n::square(m::FrozenEncoderFeatures(a[0], weight, bias)));
  };
  EXPECT_TRUE(n::gradcheck(f, {image}).ok);
  EXPECT_TRUE(n::gradgradcheck(f, {image}).ok);
  const auto strided = n::transpose(n::full({3, 2}, .4, t));
  EXPECT_TRUE(n::equal_bits(m::FrozenEncoderFeatures(strided, weight, bias), code));
}
TEST(NnEncoderPretraining, ClassifierCopiesAllFrozenStateAndExportsIndependentArrays) {
  auto w = n::ones({3, 2}).requires_grad_(), b = n::zeros({2}).requires_grad_();
  auto center = n::zeros({2}).requires_grad_(), scale = n::ones({2}).requires_grad_();
  m::FrozenEncoderClassifier model(w, b, center, scale, 2);
  n::ParameterStore          head(4703);
  const auto                 first = model.Forward(head, n::ones({2, 3}));
  {
    n::NoGradGuard guard;
    w.fill_(9);
    b.fill_(9);
    center.fill_(9);
    scale.fill_(9);
  }
  model.encoder_weight().fill_(42);
  model.encoder_bias().fill_(42);
  model.center().fill_(42);
  model.scale().fill_(42);
  const auto again = model.Forward(head, n::ones({2, 3}));
  EXPECT_TRUE(n::equal_bits(first.logits, again.logits));
  EXPECT_FALSE(model.encoder_weight().requires_grad());
  EXPECT_FALSE(first.loss.defined());
  EXPECT_EQ(head.size(), 2u);
  EXPECT_EQ(head.count(), 6);
}
TEST(NnEncoderPretraining, EveryHeadDerivativeAndInputSecondDerivativesMatchFiniteDifferences) {
  const auto                 t = n::DType::Float64;
  m::FrozenEncoderClassifier model(
      n::full({3, 2}, .5, t), n::full({2}, .2, t), n::full({2}, .1, t), n::full({2}, .7, t), 2);
  n::ParameterStore   head(4703, t);
  auto                image   = n::tensor({.1, .2, .3, .4, .5, .6}, {2, 3}, t).requires_grad_();
  auto                labels  = n::from_vector(std::vector<int>{0, 1});
  auto                out     = model.Forward(head, image, labels);
  const auto          initial = head.flatten();
  std::vector<double> analytic;
  for (const auto& g : n::grad(out.loss, head.parameters())) {
    const auto values = g.to_vector<double>();
    analytic.insert(analytic.end(), values.begin(), values.end());
  }
  for (std::size_t j = 0; j < initial.size(); ++j) {
    auto value = initial;
    value[j] += 1e-6;
    head.unflatten(value);
    const auto plus = model.Forward(head, image, labels).loss.item();
    value[j] -= 2e-6;
    head.unflatten(value);
    const auto minus = model.Forward(head, image, labels).loss.item();
    EXPECT_NEAR((plus - minus) / 2e-6, analytic[j], 1e-9);
  }
  head.unflatten(initial);
  n::TensorFunction f = [&](const auto& a) { return model.Forward(head, a[0], labels).loss; };
  EXPECT_TRUE(n::gradcheck(f, {image}).ok);
  EXPECT_TRUE(n::gradgradcheck(f, {image}).ok);
}
TEST(NnEncoderPretraining, OriginalHeadCensusLabelControlAndCallerOwnedOptimizer) {
  m::FrozenEncoderClassifier model(
      n::full({784, 32}, .01), n::zeros({32}), n::zeros({32}), n::ones({32}));
  n::ParameterStore head(4703);
  auto              image = n::full({2, 784}, .3), labels = n::from_vector(std::vector<int>{1, 2});
  auto              out = model.Forward(head, image, labels);
  EXPECT_EQ(head.size(), 2u);
  EXPECT_EQ(head.count(), 330);
  const auto before  = head.flatten();
  const auto frozen  = model.encoder_weight();
  const auto correct = n::grad(out.loss, head.parameters());
  const auto wrong   = n::grad(
      model.Forward(head, image, n::from_vector(std::vector<int>{2, 3})).loss, head.parameters());
  EXPECT_GT(n::sum(n::abs(correct[0] - wrong[0])).item(), 1);
  n::OptimizerConfig config;
  config.kind = n::OptimizerKind::Adam;
  n::Optimizer optimizer(head, .003, config);
  optimizer.step(correct);
  EXPECT_EQ(optimizer.iterations(), 1);
  EXPECT_NE(before, head.flatten());
  EXPECT_TRUE(n::equal_bits(frozen, model.encoder_weight()));
  EXPECT_EQ(model.encoder_weight().numel() + model.encoder_bias().numel(), 25120);
}
TEST(NnEncoderPretraining, InvalidTargetsAndImagesFailBeforeHeadCreation) {
  m::FrozenEncoderClassifier model(n::ones({3, 2}), n::zeros({2}), n::zeros({2}), n::ones({2}), 2);
  n::ParameterStore          head(4703);
  for (const auto& label : std::vector<n::Tensor>{
           n::zeros({2}),
           n::zeros({1}, n::DType::Int32),
           n::full({2}, -1, n::DType::Int32),
           n::full({2}, 2, n::DType::Int64)})
    EXPECT_THROW(model.Forward(head, n::ones({2, 3}), label), std::invalid_argument);
  for (const auto& image : std::vector<n::Tensor>{
           n::Tensor{},
           n::ones({0, 3}),
           n::ones({2, 4}),
           n::ones({2, 3}, n::DType::Float64),
           n::full({2, 3}, NAN)})
    EXPECT_THROW(model.Forward(head, image), std::invalid_argument);
  EXPECT_EQ(head.size(), 0u);
  EXPECT_NO_THROW(model.Forward(head, n::ones({2, 3}), n::zeros({2}, n::DType::Int64)));
}
TEST(NnEncoderPretraining, RejectsInvalidStateScalingAndFloors) {
  const auto w = n::ones({3, 2}), b = n::zeros({2}), scale = n::ones({2});
  EXPECT_THROW(m::FrozenEncoderClassifier(w, b, b, scale, 0), std::invalid_argument);
  EXPECT_THROW(m::FrozenEncoderClassifier(w, b, b, b), std::invalid_argument);
  EXPECT_THROW(m::FrozenEncoderClassifier(w, b, b, n::full({2}, INFINITY)), std::invalid_argument);
  EXPECT_THROW(
      m::FrozenEncoderFeatures(n::ones({1, 3}), w, n::zeros({2}, n::DType::Int32)),
      std::invalid_argument);
  for (const auto& code : std::vector<n::Tensor>{
           n::Tensor{},
           n::ones({2}),
           n::ones({0, 2}),
           n::ones({2, 2}, n::DType::Int32),
           n::full({2, 2}, INFINITY)})
    EXPECT_THROW(m::FitTransferScaling(code), std::invalid_argument);
  for (double floor : {0., -1., double(NAN), double(INFINITY), 1e100, 1e-100})
    EXPECT_THROW(m::FitTransferScaling(n::ones({2, 2}), floor), std::invalid_argument);
  EXPECT_THROW(
      m::FitTransferScaling(n::full({2, 2}, 1e308, n::DType::Float64)), std::invalid_argument);
}
