#include <dm/dm.hpp>
#include <gtest/gtest.h>
namespace nn = dm::nn;
using nn::Tensor;
namespace m = nn::models;
namespace check {
template <typename A, typename B>
void Near(A a, B b, double tolerance, const std::string &message) {
  EXPECT_NEAR(double(a), double(b), tolerance) << message;
}
inline void Require(bool value, const std::string &message) {
  if (!value)
    throw std::runtime_error(message);
}
} // namespace check
namespace fcn_checks {

std::vector<float> Wave(std::size_t n, double scale, double offset = 0, bool cosine = false) {
  std::vector<float> v(n);
  for (std::size_t i = 0; i < n; ++i)
    v[i] =
        float(offset + (cosine ? std::cos(scale * double(i)) : std::sin(scale * double(i))));
  return v;
}

/// Max/min gradient norm (and count) over every parameter; requires finite (and nonzero when
/// `nonzero`).
std::pair<double, double> GradientNorms(const nn::ParameterStore &p, const Tensor &objective,
                                        bool nonzero, const std::string &what) {
  const auto params = p.parameters();
  const auto grads = nn::grad(objective, params);
  check::Require(grads.size() == params.size(), what + ": every parameter has a derivative");
  double low = INFINITY, high = 0;
  for (std::size_t k = 0; k < grads.size(); ++k) {
    const double norm =
        std::sqrt(nn::sum(nn::square(grads[k].to(nn::DType::Float64))).item<double>());
    check::Require(std::isfinite(norm) && (!nonzero || norm > 0),
                   "finite nonzero " + what + " derivative: " + p.names()[k]);
    low = std::min(low, norm);
    high = std::max(high, norm);
  }
  return {low, high};
}

void FusionChecks() {
  Tensor a = nn::from_vector(Wave(24, .7), {1, 3, 4, 2}),
         b = nn::from_vector(Wave(160, .31, -.2, true), {1, 8, 10, 2});
  const auto weights = Wave(96, .13);
  a.requires_grad_(true);
  b.requires_grad_(true);
  const Tensor fused = m::FcnFuseSkip(a, b, 2, false, 1);
  const auto gradients =
      nn::grad(nn::sum(fused * nn::from_vector(weights, {1, 6, 8, 2})), {a, b});
  const auto av = a.to_vector<float>(), bv = b.to_vector<float>(),
             out = fused.to_vector<float>();
  const auto ga_out = gradients[0].to_vector<float>(),
             gb_out = gradients[1].to_vector<float>();
  std::vector<double> expected(6 * 8 * 2), ga(3 * 4 * 2), gb(8 * 10 * 2);
  for (int r = 0; r < 6; ++r)
    for (int c = 0; c < 8; ++c)
      for (int k = 0; k < 2; ++k) {
        const int index = (r * 8 + c) * 2 + k;
        expected[std::size_t(index)] = bv[std::size_t(((r + 1) * 10 + c + 1) * 2 + k)];
        gb[std::size_t(((r + 1) * 10 + c + 1) * 2 + k)] = weights[std::size_t(index)];
      }
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 4; ++c)
      for (int dy = 0; dy < 4; ++dy)
        for (int dx = 0; dx < 4; ++dx) {
          const int y = 2 * r + dy - 1, x = 2 * c + dx - 1;
          if (y < 0 || x < 0 || y >= 6 || x >= 8)
            continue;
          const double factor = (1 - std::abs(dy - 1.5) / 2) * (1 - std::abs(dx - 1.5) / 2);
          for (int k = 0; k < 2; ++k) {
            const int destination = (y * 8 + x) * 2 + k, source = (r * 4 + c) * 2 + k;
            expected[std::size_t(destination)] += factor * av[std::size_t(source)];
            ga[std::size_t(source)] += factor * weights[std::size_t(destination)];
          }
        }
  check::Require(fused.shape() == nn::Shape({1, 6, 8, 2}), "SAME bilinear fusion shape");
  double error = 0;
  for (std::size_t i = 0; i < out.size(); ++i)
    error = std::max(error, std::abs(out[i] - expected[i]));
  for (std::size_t i = 0; i < ga_out.size(); ++i)
    error = std::max(error, std::abs(ga_out[i] - ga[i]));
  for (std::size_t i = 0; i < gb_out.size(); ++i)
    error = std::max(error, std::abs(gb_out[i] - gb[i]));
  check::Near(error, 0, 1e-6,
              "host bilinear scatter, signed skip crop and every input gradient");
  const auto kernel = m::FcnBilinearKernel(2, 2).to_vector<float>(); // [4, 4, 2, 2]
  check::Near(kernel[((1 * 4 + 1) * 2 + 0) * 2 + 0], .5625, 0, "bilinear kernel centre");
  check::Near(kernel[((1 * 4 + 1) * 2 + 0) * 2 + 1], 0, 0, "no cross-class mixing");
  std::cout << "  signed bilinear/skip/crop values and every input derivative match host "
               "scatter, error="
            << error << std::endl;
}

m::FcnConfig CompactConfig(int stride) {
  m::FcnConfig c;
  c.stride = stride;
  c.classes = 4;
  c.width_divisor = 8;
  c.fc_width = 64;
  c.compact = true;
  c.keep_probability = 1;
  return c;
}

void ModelChecks() {
  const int strides[] = {32, 16, 8};
  const std::int64_t full_counts[] = {134346581, 134357354, 134362751};
  const std::int64_t compact_counts[] = {271916, 272176, 272308};
  for (int v = 0; v < 3; ++v) {
    nn::ParameterStore p(17);
    m::FcnConfig c;
    c.stride = strides[v];
    const auto net = m::Fcn(p, nn::from_vector(Wave(8 * 8 * 3, .13), {1, 8, 8, 3}), c, true);
    check::Require(p.count() == full_counts[v],
                   "full FCN trainable count excludes frozen bilinear filters");
    check::Require(net.logits.shape() == nn::Shape({1, 8, 8, 21}),
                   "full FCN crops logits to the input grid");
    GradientNorms(p, nn::mean(nn::square(net.logits)), false, "full FCN");
    check::Require(p.size() == std::size_t(32 + 2 * v), "full FCN parameter connections");
    std::cout << "  full FCN-" << strides[v] << "s: " << p.count() << " parameters, "
              << p.size() << " derivatives (executed on an 8x8 image)" << std::endl;
  }
  for (int v = 0; v < 3; ++v) {
    nn::ParameterStore p(17);
    double low = INFINITY, high = 0;
    for (int side : {95, 96, 97}) {
      const auto net = m::Fcn(
          p, nn::from_vector(Wave(std::size_t(2 * side * side * 3), .13), {2, side, side, 3}),
          CompactConfig(strides[v]), true);
      check::Require(p.count() == compact_counts[v], "compact FCN parameter count");
      check::Require(net.logits.shape() == nn::Shape({2, side, side, 4}),
                     "FCN logits align to original image dimensions");
      if (side != 96)
        continue;
      const auto norms = GradientNorms(p, nn::mean(nn::square(net.logits)), true, "FCN");
      low = std::min(low, norms.first);
      high = std::max(high, norms.second);
    }
    std::cout << "  compact FCN-" << strides[v] << "s: " << p.count()
              << " parameters; exact95/96/97 output grids; gradient norms [" << low << ','
              << high << "]" << std::endl;
  }
}

} // namespace fcn_checks
TEST(NnFcn, IndependentGeometryValueAndDerivativeFixtures) {
  fcn_checks::FusionChecks();
}
TEST(NnFcn, FullAndCompactGraphsEveryGradientAndDynamicGrids) {
  nn::set_num_threads(2);
  fcn_checks::ModelChecks();
}
namespace unet_checks {

Tensor Wave(const nn::Shape &shape, double scale) {
  std::int64_t n = 1;
  for (auto d : shape)
    n *= d;
  std::vector<float> v(static_cast<std::size_t>(n));
  for (std::size_t i = 0; i < v.size(); ++i)
    v[i] = float(std::sin(scale * double(i)));
  return nn::from_vector(v, shape);
}

/// [min, max] gradient norm over every parameter; requires finite (and nonzero when `nonzero`).
std::pair<double, double> GradientNorms(const nn::ParameterStore &p, const Tensor &objective,
                                        bool nonzero, const std::string &what) {
  const auto params = p.parameters();
  const auto grads = nn::grad(objective, params);
  check::Require(grads.size() == params.size(), what + ": every parameter has a derivative");
  double low = INFINITY, high = 0;
  for (std::size_t k = 0; k < grads.size(); ++k) {
    const double norm =
        std::sqrt(nn::sum(nn::square(grads[k].to(nn::DType::Float64))).item<double>());
    check::Require(std::isfinite(norm) && (!nonzero || norm > 0),
                   "finite nonzero " + what + " derivative: " + p.names()[k]);
    low = std::min(low, norm);
    high = std::max(high, norm);
  }
  return {low, high};
}

void CropChecks() {
  check::Require(m::UnetOutputSide(572) == 388 && m::UnetOutputSide(96, 4, false) == 96,
                 "canonical and SAME U-Net geometry");
  bool rejected = false;
  try {
    m::UnetOutputSide(64);
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  check::Require(rejected, "too-small valid input rejected");
  std::vector<float> values(9 * 12 * 2);
  for (std::size_t i = 0; i < values.size(); ++i)
    values[i] = float(i) - 100;
  Tensor x = nn::from_vector(values, {1, 9, 12, 2});
  x.requires_grad_(true);
  const Tensor cropped = m::UnetCenterCropTo(x, nn::zeros({1, 4, 7, 3}));
  const auto gradient = nn::grad(nn::sum(cropped), {x})[0].to_vector<float>();
  check::Require(cropped.shape() == nn::Shape({1, 4, 7, 2}),
                 "centre crop preserves source channels");
  const auto out = cropped.to_vector<float>();
  for (int r = 0; r < 4; ++r)
    for (int c = 0; c < 7; ++c)
      for (int k = 0; k < 2; ++k)
        check::Near(out[std::size_t((r * 7 + c) * 2 + k)],
                    values[std::size_t(((r + 2) * 12 + c + 2) * 2 + k)], 0,
                    "odd difference uses floor-centred crop");
  for (int r = 0; r < 9; ++r)
    for (int c = 0; c < 12; ++c)
      for (int k = 0; k < 2; ++k)
        check::Near(gradient[std::size_t((r * 12 + c) * 2 + k)],
                    r >= 2 && r < 6 && c >= 2 && c < 9 ? 1 : 0, 0,
                    "crop Jacobian masks exactly the retained region");
  std::cout << "  572->388 geometry, too-small rejection and signed rectangular "
               "centre-crop/Jacobian checks passed"
            << std::endl;
}

m::UnetConfig CompactConfig() {
  m::UnetConfig c;
  c.classes = 4;
  c.input_channels = 3;
  c.base_width = 8;
  c.valid = false;
  c.dropout = 0;
  return c;
}

void ModelChecks() {
  {
    nn::ParameterStore p(17);
    const int side = 188;
    check::Require(m::UnetOutputSide(side) == 4, "smallest valid full U-Net input");
    const auto net = m::Unet(p, Wave({1, side, side, 1}, .13), {}, false);
    check::Require(p.count() == 31030658,
                   "original grayscale/two-class U-Net parameter count");
    check::Require(net.logits.shape() == nn::Shape({1, 4, 4, 2}),
                   "valid U-Net output geometry");
    check::Require(net.skips[0].shape()[1] == 184 && net.decoder.back().shape()[1] == 4,
                   "valid skip/decoder sizes");
    GradientNorms(p, nn::mean(nn::square(net.logits)), false, "full U-Net");
    check::Require(p.size() == 46, "full U-Net parameter connections");
    std::cout
        << "  original U-Net: 31030658 parameters, 46 derivatives (executed on 188x188 -> "
           "4x4), four cropped skips"
        << std::endl;
  }
  nn::ParameterStore p(17);
  double low = INFINITY, high = 0;
  for (int side : {96, 112, 128}) {
    const auto net = m::Unet(p, Wave({2, side, side, 3}, .13), CompactConfig(), true);
    check::Require(net.logits.shape() == nn::Shape({2, side, side, 4}),
                   "SAME U-Net output alignment");
    if (side != 96)
      continue;
    const auto norms = GradientNorms(p, nn::mean(nn::square(net.logits)), true, "U-Net");
    low = norms.first;
    high = norms.second;
  }
  std::cout << "  compact U-Net: " << p.count()
            << " parameters; 96/112/128 alignment; all gradient norms [" << low << ',' << high
            << "]" << std::endl;
}

} // namespace unet_checks
TEST(NnUnet, IndependentGeometryValueAndDerivativeFixtures) {
  unet_checks::CropChecks();
}
TEST(NnUnet, FullAndCompactGraphsEveryGradientAndDynamicGrids) {
  nn::set_num_threads(2);
  unet_checks::ModelChecks();
}
TEST(NnSegmentationData, DeterministicSeedsAlignedClassesAndIndependentBatchStorage) {
  const auto a = m::TensorSegmentationShapes(3, 32, 11),
             b = m::TensorSegmentationShapes(3, 32, 11),
             c = m::TensorSegmentationShapes(3, 32, 12);
  EXPECT_EQ(a.images.to_vector<float>(), b.images.to_vector<float>());
  EXPECT_EQ(a.labels.to_vector<int>(), b.labels.to_vector<int>());
  EXPECT_NE(a.labels.to_vector<int>(), c.labels.to_vector<int>());
  EXPECT_GE(nn::min(a.images).item(), 0);
  EXPECT_LE(nn::max(a.images).item(), 1);
  EXPECT_GE(nn::min(a.labels).item(), 0);
  EXPECT_LE(nn::max(a.labels).item(), 3);
  const auto labels = a.labels.to_vector<int>();
  EXPECT_TRUE(std::any_of(labels.begin(), labels.end(), [](int v) {
    return v > 0;
  }));
  auto batch =
      m::TensorSegmentationBatch(a, nn::from_vector(std::vector<std::int64_t>{2, 0, 2}));
  EXPECT_EQ(nn::slice_axis(batch.first, 0, 0, 1).to_vector<float>(),
            nn::slice_axis(a.images, 0, 2, 3).to_vector<float>());
  batch.first.fill_(0);
  EXPECT_GT(nn::sum(a.images).item(), 0);
}
TEST(NnSegmentationData, FitClassPseudocountWeightsAndWeightedCrossEntropyDerivative) {
  const auto labels = nn::from_vector(std::vector<std::int64_t>{0, 0, 1}, {1, 1, 3}),
             weights = m::SegmentationClassWeights(labels, 3);
  EXPECT_NEAR(weights.at({0}), 1, 1e-7);
  EXPECT_NEAR(weights.at({1}), std::sqrt(1.5), 1e-7);
  EXPECT_NEAR(weights.at({2}), std::sqrt(3.), 1e-7);
  auto logits = nn::zeros({1, 1, 3, 3}, nn::DType::Float64).requires_grad_();
  const auto w = nn::from_vector(std::vector<double>{1, 2, 3}),
             loss = m::SegmentationCrossEntropy(logits, labels, 3, w);
  EXPECT_NEAR(loss.item(), std::log(3.), 1e-12);
  const auto g = nn::grad(loss, {logits})[0].reshape({3, 3}).to_vector<double>();
  const int truth[] = {0, 0, 1};
  const double importance[] = {1, 1, 2};
  for (int i = 0; i < 3; ++i)
    for (int c = 0; c < 3; ++c)
      EXPECT_NEAR(g[i * 3 + c], importance[i] / 4 * (1. / 3 - (c == truth[i] ? 1 : 0)), 1e-12);
}
TEST(NnSegmentationData, ConfusionMeansIncludePredictionOnlyAndExcludeAbsentClasses) {
  const auto y = nn::from_vector(std::vector<std::int64_t>{0, 0, 1, 1, 1}),
             p = nn::from_vector(std::vector<std::int64_t>{0, 1, 1, 1, 2});
  const auto result = m::SegmentationPredictionMetrics(p, y, 4);
  EXPECT_EQ(result.confusion.to_vector<int>(),
            std::vector<int>({1, 1, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0}));
  EXPECT_DOUBLE_EQ(result.pixel_accuracy.item(), .6);
  EXPECT_DOUBLE_EQ(result.mean_iou.item(), (1. / 2 + 2. / 4 + 0) / 3);
  EXPECT_EQ(result.iou.at({3}), -1);
  const auto empty = m::SegmentationConfusionMetrics(nn::zeros({4, 4}, nn::DType::Int64));
  EXPECT_EQ(empty.mean_iou.item(), 0);
  EXPECT_EQ(empty.pixel_accuracy.item(), 0);
}
TEST(NnSegmentationData, PalettePreservesFourClassOrdering) {
  const auto rgb =
      m::ColorSegmentationLabels(nn::from_vector(std::vector<std::int64_t>{0, 1, 2, 3}));
  EXPECT_EQ(rgb.shape(), nn::Shape({4, 3}));
  EXPECT_NEAR(rgb.at({1, 0}), 230. / 255, 1e-7);
  EXPECT_NEAR(rgb.at({2, 1}), 210. / 255, 1e-7);
  EXPECT_NEAR(rgb.at({3, 2}), 240. / 255, 1e-7);
}
TEST(NnSegmentationData, InvalidRowsClassWeightsAndGeometryReject) {
  EXPECT_THROW(m::TensorSegmentationShapes(0, 32), std::invalid_argument);
  EXPECT_THROW(m::SegmentationClassWeights(nn::full({1}, 4., nn::DType::Int64), 4),
               std::invalid_argument);
  EXPECT_THROW(m::SegmentationCrossEntropy(nn::zeros({1, 3}), nn::zeros({1}, nn::DType::Int64),
                                           3, nn::zeros({3})),
               std::invalid_argument);
}
