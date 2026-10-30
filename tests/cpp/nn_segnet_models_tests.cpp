#include <dm/dm.hpp>
#include <gtest/gtest.h>
namespace nn = dm::nn;
namespace m = nn::models;
using nn::Tensor;
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
namespace {

Tensor Wave(const nn::Shape &shape, double scale) {
  std::int64_t n = 1;
  for (auto d : shape)
    n *= d;
  std::vector<float> v(static_cast<std::size_t>(n));
  for (std::size_t i = 0; i < v.size(); ++i)
    v[i] = float(std::sin(scale * double(i)));
  return nn::from_vector(v, shape);
}

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

void PoolChecks() {
  std::vector<float> value(2 * 5 * 7 * 2), values(2 * 2 * 3 * 2), weight(value.size());
  for (std::size_t i = 0; i < value.size(); ++i) {
    value[i] = std::sin(.37f * float(i)) + .001f * float(i) - .4f;
    weight[i] = std::cos(.17f * float(i));
  }
  for (std::size_t i = 0; i < values.size(); ++i)
    values[i] = .2f * float(i) - 3;
  Tensor input = nn::from_vector(value, {2, 5, 7, 2}),
         pooled = nn::from_vector(values, {2, 2, 3, 2});
  input.requires_grad_(true);
  pooled.requires_grad_(true);
  const auto record = m::SegNetPoolWithIndices(input);
  const Tensor w = nn::from_vector(weight, {2, 5, 7, 2});
  const Tensor restored = m::SegNetIndexUnpool(pooled, record),
               pipeline = m::SegNetIndexUnpool(record.values, record);
  const auto pooled_gradient = nn::grad(nn::sum(restored * w), {pooled})[0].to_vector<float>();
  const auto input_gradient = nn::grad(nn::sum(pipeline * w), {input})[0].to_vector<float>();
  const auto restored_values = restored.to_vector<float>();
  std::vector<double> expected(value.size()), dx(value.size());
  double error = 0;
  for (int n = 0; n < 2; ++n)
    for (int r = 0; r < 2; ++r)
      for (int c = 0; c < 3; ++c)
        for (int k = 0; k < 2; ++k) {
          float maximum = -INFINITY;
          int best = -1;
          for (int dy = 0; dy < 2; ++dy)
            for (int dxx = 0; dxx < 2; ++dxx) {
              const int index = ((n * 5 + 2 * r + dy) * 7 + 2 * c + dxx) * 2 + k;
              if (value[std::size_t(index)] > maximum) {
                maximum = value[std::size_t(index)];
                best = index;
              }
            }
          const int slot = ((n * 2 + r) * 3 + c) * 2 + k;
          check::Require(record.indices.reshape({-1}).at({slot}) == best,
                         "argmax includes batch and channel offsets");
          expected[std::size_t(best)] = values[std::size_t(slot)];
          dx[std::size_t(best)] = weight[std::size_t(best)];
          error = std::max(error, std::abs(double(pooled_gradient[std::size_t(slot)]) -
                                           weight[std::size_t(best)]));
        }
  for (std::size_t i = 0; i < expected.size(); ++i) {
    error = std::max(error, std::abs(restored_values[i] - expected[i]));
    error = std::max(error, std::abs(input_gradient[i] - dx[i]));
  }
  check::Require(restored.shape() == nn::Shape({2, 5, 7, 2}),
                 "unpool restores the odd pre-pool shape");
  check::Near(error, 0, 1e-6, "host sparse unpool and every pooled/original derivative");
  std::cout
      << "  negative/multi-batch/channel/odd-grid index unpool and every derivative match "
         "host winners; error="
      << error << std::endl;
}

m::SegNetConfig CompactConfig() {
  m::SegNetConfig c;
  c.classes = 4;
  c.width_divisor = 8;
  return c;
}

void ModelChecks() {
  {
    nn::ParameterStore p(17), state(17);
    const auto net = m::SegNet(p, state, Wave({1, 64, 64, 3}, .13), {}, true);
    check::Require(p.count() == 29449355 && state.count() == 15872,
                   "full SegNet trainable/state counts");
    check::Require(net.logits.shape() == nn::Shape({1, 64, 64, 11}),
                   "full SegNet logits on the input grid");
    GradientNorms(p, nn::mean(nn::square(net.logits)), false, "full SegNet");
    check::Require(p.size() == 102, "full SegNet all parameter connections");
    std::cout << "  full SegNet: 29449355 parameters, 15872 state values, 102 derivatives "
                 "(executed on 64x64)"
              << std::endl;
  }
  nn::ParameterStore p(17), state(17);
  double low = INFINITY, high = 0;
  std::int64_t index_bytes = 0, feature_bytes = 0;
  for (int side : {96, 97}) {
    const auto net = m::SegNet(p, state, Wave({2, side, side, 3}, .13), CompactConfig(), true);
    check::Require(net.logits.shape() == nn::Shape({2, side, side, 4}),
                   "saved shapes restore odd input dimensions");
    if (side != 96)
      continue;
    for (const auto &record : net.pooling) {
      index_bytes += 8 * std::int64_t(record.indices.numel());
      feature_bytes += 4 * nn::shape_numel(record.shape);
    }
    const auto norms = GradientNorms(p, nn::mean(nn::square(net.logits)), true, "SegNet");
    low = norms.first;
    high = norms.second;
  }
  check::Require(2 * index_bytes == feature_bytes,
                 "int64 nonoverlapping 2x2 index cache is half full float-feature cache");
  std::cout << "  compact SegNet: " << p.count()
            << " parameters; 96/97 alignment; derivative norms [" << low << ',' << high
            << "]; index cache=" << index_bytes << " vs full-feature cache=" << feature_bytes
            << " bytes" << std::endl;
}

} // namespace
TEST(NnSegNet, IndependentOddNegativeMultiBatchChannelWinnersAndEveryDerivative) {
  PoolChecks();
}
TEST(NnSegNet, Full102GradientCountsCompactNonzeroOddGridsAndCacheBytes) {
  nn::set_num_threads(2);
  ModelChecks();
}
TEST(NnSegNet, TiedMaximaSelectFirstPixelAndZeroOddEdges) {
  const auto record = m::SegNetPoolWithIndices(nn::ones({2, 3, 3, 1}));
  EXPECT_EQ(record.indices.to_vector<std::int64_t>(), std::vector<std::int64_t>({0, 9}));
  const auto restored = m::SegNetIndexUnpool(nn::full({2, 1, 1, 1}, -3.), record);
  EXPECT_EQ(restored.to_vector<float>(),
            std::vector<float>({-3, 0, 0, 0, 0, 0, 0, 0, 0, -3, 0, 0, 0, 0, 0, 0, 0, 0}));
}
TEST(NnSegNet, InvalidWinnerRangesDuplicatesAndGeometryReject) {
  auto record = m::SegNetPoolWithIndices(nn::ones({1, 4, 4, 1}));
  record.indices = nn::zeros(record.indices.shape(), nn::DType::Int64);
  EXPECT_THROW(m::SegNetIndexUnpool(record.values, record), std::invalid_argument);
  record.indices = nn::full(record.indices.shape(), 100., nn::DType::Int64);
  EXPECT_THROW(m::SegNetIndexUnpool(record.values, record), std::invalid_argument);
  EXPECT_THROW(m::SegNetPoolWithIndices(nn::zeros({1, 1, 4, 1})), std::invalid_argument);
  EXPECT_THROW(m::SegNetPoolWithIndices(nn::full({1, 2, 2, 1}, NAN)), std::invalid_argument);
}
