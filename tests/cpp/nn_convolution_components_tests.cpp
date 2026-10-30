#include <datamunge/nn/nn.hpp>
#include <gtest/gtest.h>

namespace n = datamunge::nn;
TEST(NnConvolutionComponents, PatchRowsAndOverlapHaveKnownOrder) {
  const auto type = n::DType::Float64;
  auto x = n::linspace(1, 9, 9, type).reshape({1, 3, 3, 1}).detach().requires_grad_();
  const auto patches = n::im2col2d(x, 2, 2, n::conv_options(1, n::Padding::Valid));
  EXPECT_EQ(patches.to_vector<double>(),
            (std::vector<double>{1, 2, 4, 5, 2, 3, 5, 6, 4, 5, 7, 8, 5, 6, 8, 9}));
  const auto gradient = n::grad(n::sum(patches), {x})[0];
  EXPECT_EQ(gradient.to_vector<double>(), (std::vector<double>{1, 2, 1, 2, 4, 2, 1, 2, 1}));
}
TEST(NnConvolutionComponents, PatchGemmAndGradientsAgreeForAsymmetricGeometry) {
  const auto type = n::DType::Float64;
  n::Generator rng(149);
  for (auto padding : {n::Padding::Same, n::Padding::Valid, n::Padding::Explicit}) {
    auto x = n::normal({2, 7, 8, 2}, 0, .3, rng, type).requires_grad_();
    auto k = n::normal({2, 3, 2, 3}, 0, .3, rng, type).requires_grad_();
    n::Conv2DOptions options;
    options.padding = padding;
    options.strides = {2, 1};
    options.dilations = {1, 2};
    options.explicit_paddings = {1, 2, 2, 0};
    const auto native = n::conv2d(x, k, options);
    const auto patches = n::im2col2d(x, 2, 3, options);
    const auto manual = n::matmul(patches, k.reshape({12, 3})).reshape(native.shape());
    EXPECT_LT(n::max(n::abs(manual - native)).item(), 1e-12);
    const auto a = n::grad(n::sum(manual), {x, k});
    const auto b = n::grad(n::sum(native), {x, k});
    EXPECT_LT(n::max(n::abs(a[0] - b[0])).item(), 1e-12);
    EXPECT_LT(n::max(n::abs(a[1] - b[1])).item(), 1e-12);
  }
}
TEST(NnConvolutionComponents, InvalidPatchGeometryIsRejected) {
  const auto x = n::ones({1, 3, 3, 1});
  EXPECT_THROW(n::im2col2d(x, 0, 2), std::invalid_argument);
  EXPECT_THROW(n::im2col2d(n::ones({3, 3}), 2, 2), std::invalid_argument);
  auto options = n::conv_options(1);
  options.groups = 2;
  EXPECT_THROW(n::im2col2d(x, 2, 2, options), std::invalid_argument);
  options.groups = 1;
  options.strides[0] = 0;
  EXPECT_THROW(n::im2col2d(x, 2, 2, options), std::invalid_argument);
  options = n::conv_options(1, n::Padding::Explicit);
  options.explicit_paddings[0] = -1;
  EXPECT_THROW(n::im2col2d(x, 2, 2, options), std::invalid_argument);
}
TEST(NnConvolutionComponents, ExpandedKernelPreservesTapGradients) {
  auto k = n::ones({2, 2, 1, 1}, n::DType::Float64).requires_grad_();
  const auto expanded = n::expand_dilated_kernel(k, 2);
  EXPECT_EQ(expanded.to_vector<double>(), (std::vector<double>{1, 0, 1, 0, 0, 0, 1, 0, 1}));
  EXPECT_EQ(n::grad(n::sum(expanded), {k})[0].to_vector<double>(),
            (std::vector<double>{1, 1, 1, 1}));
  EXPECT_THROW(n::expand_dilated_kernel(k, 0), std::invalid_argument);
}
TEST(NnConvolutionComponents, RectangularSeparableCompositionPreservesBothFactorGradients) {
  const auto type = n::DType::Float64;
  n::Generator rng(19);
  auto d = n::normal({2, 3, 2, 2}, 0, .3, rng, type).requires_grad_();
  auto p = n::normal({1, 1, 4, 3}, 0, .3, rng, type).requires_grad_();
  const auto x = n::normal({1, 5, 6, 2}, 0, .3, rng, type);
  const auto a = n::conv2d(n::depthwise_conv2d(x, d), p);
  const auto b = n::conv2d(x, n::compose_separable_kernel(d, p));
  EXPECT_LT(n::max(n::abs(a - b)).item(), 1e-12);
  const auto ga = n::grad(n::sum(a), {d, p}), gb = n::grad(n::sum(b), {d, p});
  EXPECT_LT(n::max(n::abs(ga[0] - gb[0])).item(), 1e-12);
  EXPECT_LT(n::max(n::abs(ga[1] - gb[1])).item(), 1e-12);
  EXPECT_THROW(n::compose_separable_kernel(d, n::ones({1, 1, 3, 3})), std::invalid_argument);
}
TEST(NnConvolutionComponents, SharedMnistArchitectureHasExpectedShapesAndCount) {
  n::ParameterStore p(17);
  const auto out = n::models::MnistCnn(p, n::zeros({2, 28, 28, 1}));
  EXPECT_EQ(p.count(), 105866);
  EXPECT_EQ(out.pool1.shape(), (n::Shape{2, 14, 14, 16}));
  EXPECT_EQ(out.pool2.shape(), (n::Shape{2, 7, 7, 32}));
  EXPECT_EQ(out.logits.shape(), (n::Shape{2, 10}));
}
TEST(NnConvolutionComponents, SparseLeNetCountMissingChannelsAndFloat64Prototypes) {
  n::ParameterStore p(17, n::DType::Float64);
  const auto out = n::models::LeNet5(p, n::zeros({1, 784}, n::DType::Float64));
  EXPECT_EQ(p.count(), 60000);
  EXPECT_EQ(out.logits.dtype(), n::DType::Float64);
  auto input = n::full({1, 14, 14, 6}, .2, n::DType::Float64).requires_grad_();
  const auto c3 = n::models::LeNetC3(p, "c3", input);
  const auto gradient = n::grad(n::sum(n::slice_axis(c3, 3, 0, 1)), {input})[0];
  EXPECT_EQ(n::max(n::abs(n::slice_axis(gradient, 3, 3, 6))).item(), 0);
  const auto codes = n::cast(n::models::LeNetDigitCodes(), n::DType::Float64);
  EXPECT_EQ(n::argmax(n::models::LeNetRbfLogits(codes), 1).to_vector<int64_t>(),
            (std::vector<int64_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}));
}
TEST(NnConvolutionComponents, NinPatchMlpMatchesEveryParameterGradient) {
  const auto type = n::DType::Float64;
  n::ParameterStore p(31, type);
  n::Generator rng(31);
  const auto x = n::normal({1, 4, 5, 2}, 0, .3, rng, type);
  const auto native = n::models::MlpConv(p, "mlp", x, 4, 3, 2, 3);
  auto patches = n::im2col2d(x, 3, 3);
  const std::array<std::string, 3> names{"spatial", "mix1", "mix2"};
  const std::array<int, 3> in{18, 4, 3}, out{4, 3, 2};
  for (int layer = 0; layer < 3; ++layer)
    patches = n::relu(
        n::matmul(patches,
                  p.at("mlp/" + names[layer] + "/W").reshape({in[layer], out[layer]})) +
        p.at("mlp/" + names[layer] + "/b"));
  EXPECT_LT(n::max(n::abs(native.reshape(patches.shape()) - patches)).item(), 1e-12);
  const auto a = n::grad(n::sum(native), p.parameters()),
             b = n::grad(n::sum(patches), p.parameters());
  for (size_t i = 0; i < a.size(); ++i)
    EXPECT_LT(n::max(n::abs(a[i] - b[i])).item(), 1e-12);
}
TEST(NnConvolutionComponents, VggDepthSchedulesAndSlidingHeadAgree) {
  EXPECT_EQ(n::models::VggBlockDepths(11), (std::array<int, 5>{1, 1, 2, 2, 2}));
  EXPECT_EQ(n::models::VggBlockDepths(16), (std::array<int, 5>{2, 2, 3, 3, 3}));
  EXPECT_EQ(n::models::VggBlockDepths(19), (std::array<int, 5>{2, 2, 4, 4, 4}));
  EXPECT_THROW(n::models::VggBlockDepths(12), std::invalid_argument);
  const auto type = n::DType::Float64;
  n::models::VggConfig cfg;
  cfg.depth = 11;
  cfg.image_size = 64;
  cfg.classes = 3;
  cfg.width_divisor = 8;
  cfg.dense_width = 16;
  n::ParameterStore p(17, type);
  n::NoGradGuard guard;
  const auto model = n::models::Vgg(p, n::zeros({1, 64, 64, 3}, type), cfg);
  EXPECT_EQ(model.pooling[4].shape(), (n::Shape{1, 2, 2, 64}));
  n::Generator rng(17);
  const auto feature = n::normal({1, 3, 4, 64}, 0, .2, rng, type);
  const auto scores = n::models::VggConvolutionalHead(p, feature, cfg);
  for (int row = 0; row < 2; ++row)
    for (int column = 0; column < 3; ++column) {
      auto h = n::slice(feature, {0, row, column, 0}, {1, 2, 2, 64}).reshape({1, 256});
      h = n::dense(p, "fc6", h, 16, n::Activation::Relu);
      h = n::dense(p, "fc7", h, 16, n::Activation::Relu);
      const auto logits = n::dense(p, "fc8", h, 3);
      EXPECT_LT(
          n::max(n::abs(n::slice(scores, {0, row, column, 0}, {1, 1, 1, 3}).reshape({1, 3}) -
                        logits))
              .item(),
          1e-12);
    }
}
TEST(NnConvolutionComponents, ClassicConfigsRejectInvalidDepthAndDropout) {
  n::models::NinConfig nin;
  nin.width_divisor = 3;
  EXPECT_THROW(nin.Validate(), std::invalid_argument);
  nin.width_divisor = 8;
  nin.keep_probability = 0;
  EXPECT_THROW(nin.Validate(), std::invalid_argument);
  n::models::VggConfig vgg;
  vgg.depth = 12;
  EXPECT_THROW(vgg.Validate(), std::invalid_argument);
  vgg.depth = 11;
  vgg.image_size = 31;
  EXPECT_THROW(vgg.Validate(), std::invalid_argument);
}
