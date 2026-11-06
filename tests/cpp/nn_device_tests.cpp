#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <datamunge/compute/context.hpp>
#include <datamunge/nn/autograd.hpp>
#include <datamunge/nn/conv.hpp>
#include <datamunge/nn/io.hpp>
#include <datamunge/nn/memory.hpp>
#include <datamunge/nn/ops.hpp>
#include <datamunge/nn/params.hpp>
#include <datamunge/nn/resnet_models.hpp>
#include <datamunge/nn/squeezenet_models.hpp>
#include <datamunge/nn/tensor.hpp>
#include <limits>
namespace nn = datamunge::nn;
namespace c  = datamunge::compute;

TEST(NnDevice, CpuContextCopiesViewsAndGradientTransfers) {
  const auto baseline = nn::live_memory_bytes();
  c::Context cpu({}, 2);
  {
    auto x = nn::from_vector(std::vector<double>{1, 2, 3, 4, 5, 6}, {2, 3});
    EXPECT_EQ(x.device().backend, c::Backend::CPU);
    EXPECT_TRUE(x.context().shares_with(nn::empty({1}).context()));
    x.requires_grad_();
    auto transferred = x.transpose().to(cpu);
    EXPECT_TRUE(transferred.context().shares_with(cpu));
    EXPECT_FALSE(transferred.impl()->storage == x.impl()->storage);
    EXPECT_EQ(transferred.to_vector<double>(), (std::vector<double>{1, 4, 2, 5, 3, 6}));
    transferred.sum().backward();
    EXPECT_EQ(x.grad().to_vector<double>(), (std::vector<double>{1, 1, 1, 1, 1, 1}));
    auto view = transferred.transpose();
    EXPECT_TRUE(view.context().shares_with(cpu));
    EXPECT_TRUE(view.detach().impl()->storage == transferred.impl()->storage);
  }
  EXPECT_EQ(cpu.live_bytes(), 0);
  EXPECT_EQ(nn::live_memory_bytes(), baseline);
  nn::Tensor retained;
  {
    c::Context temporary;
    retained = nn::full({1}, 3, nn::DType::Int64, temporary);
  }
  EXPECT_EQ(retained.item<std::int64_t>(), 3);
}
TEST(NnDevice, AllocationShapesDtypesAndIntegralFillValidation) {
  const c::Context cpu;
  EXPECT_THROW(nn::empty({-1}, nn::DType::Float32, cpu), std::invalid_argument);
  EXPECT_THROW(nn::empty({1}, static_cast<nn::DType>(999), cpu), std::invalid_argument);
  const auto largest = std::numeric_limits<std::int64_t>::max();
  EXPECT_THROW(nn::empty({largest, 2}, nn::DType::Float32, cpu), std::overflow_error);
  EXPECT_THROW(nn::empty({largest}, nn::DType::Float64, cpu), std::overflow_error);
  EXPECT_THROW(nn::empty({0, largest, largest}, nn::DType::Bool, cpu), std::overflow_error);
  EXPECT_EQ(nn::zeros({0, 3}, nn::DType::Float32, cpu).numel(), 0);
  EXPECT_THROW(
      nn::full({1}, std::numeric_limits<double>::infinity(), nn::DType::Int32, cpu),
      std::invalid_argument);
  EXPECT_THROW(nn::full({1}, std::ldexp(1., 63), nn::DType::Int64, cpu), std::invalid_argument);
  EXPECT_EQ(
      nn::full({1}, -std::ldexp(1., 63), nn::DType::Int64, cpu).item<std::int64_t>(),
      std::numeric_limits<std::int64_t>::min());
  EXPECT_THROW((void)nn::Tensor{}.device(), std::logic_error);
  EXPECT_THROW((void)nn::Tensor{}.to(cpu), std::logic_error);
}
namespace {
void exercise(c::Backend backend) {
  c::Context gpu({backend, 0}), cpu;
  const auto before = nn::live_memory_bytes();
  {
    const auto integer = nn::from_vector(
        std::vector<std::int64_t>{
            9007199254740993LL,
            2,
            3,
            -9007199254740993LL,
            std::numeric_limits<std::int64_t>::min(),
            std::numeric_limits<std::int64_t>::max()},
        {2, 3});
    auto device = integer.to(gpu);
    EXPECT_EQ(device.device().backend, backend);
    EXPECT_TRUE(device.context().shares_with(gpu));
    EXPECT_THROW(device.data_ptr<std::int64_t>(), std::logic_error);
    EXPECT_THROW(device.to_vector<std::int64_t>(), std::logic_error);
    EXPECT_THROW(nn::add(device, device), std::logic_error);
    EXPECT_THROW(nn::encode_npy(device), std::logic_error);
    EXPECT_EQ(device.to(cpu).to_vector<std::int64_t>(), integer.to_vector<std::int64_t>());
    const auto transpose = device.transpose();
    EXPECT_TRUE(transpose.impl()->storage == device.impl()->storage);
    auto packed = transpose.contiguous();
    EXPECT_TRUE(packed.context().shares_with(gpu));
    EXPECT_TRUE(packed.is_contiguous());
    EXPECT_EQ(
        packed.to(cpu).to_vector<std::int64_t>(), integer.transpose().to_vector<std::int64_t>());
    EXPECT_EQ(
        nn::slice_axis(device, 1, 2, -4, -1).clone().to(cpu).to_vector<std::int64_t>(),
        (std::vector<std::int64_t>{
            3,
            2,
            9007199254740993LL,
            std::numeric_limits<std::int64_t>::max(),
            std::numeric_limits<std::int64_t>::min(),
            -9007199254740993LL}));
    const auto broadcast = nn::broadcast_to(nn::slice(device, {0, 0}, {1, 3}), {4, 3});
    EXPECT_EQ(
        broadcast.clone().to(cpu).to_vector<std::int64_t>(),
        (std::vector<std::int64_t>{
            9007199254740993LL,
            2,
            3,
            9007199254740993LL,
            2,
            3,
            9007199254740993LL,
            2,
            3,
            9007199254740993LL,
            2,
            3}));
    c::Context other({backend, 0});
    EXPECT_EQ(
        transpose.to(other).to(cpu).to_vector<std::int64_t>(),
        integer.transpose().to_vector<std::int64_t>());
    const auto flags = nn::from_vector(std::vector<bool>{true, false, true, false, true}).to(gpu);
    EXPECT_EQ(flags.impl()->storage->nbytes, 5);
    EXPECT_EQ(flags.impl()->storage->bytes.size(), 8);
    EXPECT_EQ(
        nn::slice_axis(flags, 0, 1, 4).clone().to(cpu).to_vector<bool>(),
        (std::vector<bool>{false, true, false}));
    EXPECT_EQ(
        nn::slice_axis(flags, 0, 1, 4).to(cpu).to_vector<bool>(),
        (std::vector<bool>{false, true, false}));
    for (const auto dtype :
         {nn::DType::Bool,
          nn::DType::Int32,
          nn::DType::Int64,
          nn::DType::Float32,
          nn::DType::Float64}) {
      const auto expected = nn::full({67}, 1, dtype);
      const auto fill     = nn::full({67}, 1, dtype, gpu);
      EXPECT_TRUE(nn::equal_bits(fill.to(cpu), expected));
      EXPECT_TRUE(nn::equal_bits(nn::zeros_like(fill).to(cpu), nn::zeros({67}, dtype)));
      EXPECT_TRUE(nn::equal_bits(nn::ones_like(fill).to(cpu), expected));
      EXPECT_TRUE(nn::equal_bits(nn::full_like(fill, 0).to(cpu), nn::zeros({67}, dtype)));
    }
    // Cross the portable 65535-workgroup X limit and verify the second grid row.
    const std::int64_t large_count = 4194241;
    const auto         large       = nn::full({large_count}, 7, nn::DType::Int32, gpu).to(cpu);
    EXPECT_EQ(large.at({0}), 7);
    EXPECT_EQ(large.at({large_count - 1}), 7);
    const auto real = nn::full({65}, -3.125, nn::DType::Float64, gpu);
    EXPECT_EQ(real.to(cpu).to_vector<double>(), std::vector<double>(65, -3.125));
    const auto empty = nn::zeros({0, 2}, nn::DType::Float32, gpu).transpose().clone();
    EXPECT_TRUE(empty.context().shares_with(gpu));
    EXPECT_TRUE(empty.to(cpu).to_vector<float>().empty());
    EXPECT_THROW(empty.to_vector<float>(), std::logic_error);
    EXPECT_THROW(
        nn::matmul(nn::empty({0, 2}, nn::DType::Float32, gpu), nn::zeros({2, 3})),
        std::logic_error);
    EXPECT_THROW(nn::is_finite(device), std::logic_error);
    EXPECT_THROW(nn::is_nan(device), std::logic_error);
    EXPECT_THROW(nn::empty({std::int64_t(1) << 32}, nn::DType::Bool, gpu), std::length_error);
  }
  EXPECT_EQ(gpu.live_bytes(), 0);
  EXPECT_EQ(nn::live_memory_bytes(), before);
  {
    auto x = nn::from_vector(std::vector<float>{2, 3, 4}, {3});
    x.requires_grad_();
    const auto y = x.to(gpu).to(cpu);
    nn::mul(y, y).sum().backward();
    EXPECT_EQ(x.grad().to_vector<float>(), (std::vector<float>{4, 6, 8}));
    auto leaf = nn::full({}, 2, nn::DType::Float64, gpu);
    leaf.requires_grad_();
    leaf.backward();
    EXPECT_TRUE(leaf.grad().context().shares_with(gpu));
    EXPECT_EQ(leaf.grad().to(cpu).item<double>(), 1);
    EXPECT_THROW(leaf.set_grad(nn::scalar(1, nn::DType::Float64)), std::invalid_argument);
    c::Context foreign({backend, 0});
    EXPECT_THROW(
        nn::grad({leaf}, {leaf}, {nn::full({}, 1, nn::DType::Float64, foreign)}),
        std::invalid_argument);
    EXPECT_THROW(
        nn::grad({leaf}, {leaf}, {nn::scalar(1, nn::DType::Float64)}), std::invalid_argument);
    auto input = nn::full({2}, 1, nn::DType::Float32, gpu);
    input.requires_grad_();
    auto a = nn::zeros_like(input), b = nn::zeros_like(input);
    bool checked = false;
    nn::record(
        "TwoDeviceOutputs",
        {input},
        {a, b},
        [&](const std::vector<nn::Tensor>& grads, const std::vector<nn::Tensor>&) {
          checked = true;
          EXPECT_EQ(grads[1].device().backend, backend);
          EXPECT_EQ(grads[1].to(cpu).to_vector<float>(), (std::vector<float>{0, 0}));
          return std::vector<nn::Tensor>{grads[0]};
        });
    a.backward(nn::ones_like(a));
    EXPECT_TRUE(checked);
    EXPECT_EQ(input.grad().to(cpu).to_vector<float>(), (std::vector<float>{1, 1}));
  }
  EXPECT_EQ(gpu.live_bytes(), 0);
  EXPECT_EQ(nn::live_memory_bytes(), before);
}
} // namespace
TEST(NnDevice, VulkanStorageViewsFillsAndTransferAutograd) {
  const auto info = c::probe({c::Backend::Vulkan, 0});
  if (!info.available)
    GTEST_SKIP() << info.diagnostic;
  exercise(c::Backend::Vulkan);
}
TEST(NnDevice, CudaStorageViewsFillsAndTransferAutograd) {
  const auto info = c::probe({c::Backend::CUDA, 0});
  if (!info.available) {
    ASSERT_FALSE(std::getenv("DATAMUNGE_REQUIRE_CUDA")) << info.diagnostic;
    GTEST_SKIP() << info.diagnostic;
  }
  exercise(c::Backend::CUDA);
}

TEST(NnDevice, CudaMatmulTransposesBroadcastViewsAndGradients) {
  const auto info = c::probe({c::Backend::CUDA, 0});
  if (!info.available) {
    ASSERT_FALSE(std::getenv("DATAMUNGE_REQUIRE_CUDA")) << info.diagnostic;
    GTEST_SKIP() << info.diagnostic;
  }
  c::Context gpu({c::Backend::CUDA, 0}), cpu;
  ASSERT_TRUE(gpu.supports_matmul()) << "CUDA contexts always run the in-house GEMM";
  const auto baseline = nn::live_memory_bytes();
  {
    for (auto dtype : {nn::DType::Float32, nn::DType::Float64}) {
      const auto tolerance = dtype == nn::DType::Float32 ? 1e-5 : 1e-12;
      auto a = nn::cast(nn::from_vector(std::vector<double>{1, -2, 3, 4, .5, -6}, {2, 3}), dtype);
      auto b = nn::cast(
          nn::from_vector(std::vector<double>{.5, -1, 2, 3, 4, -2, 1, -3, 7, .25, 5, 6}, {3, 4}),
          dtype);
      for (bool ta : {false, true})
        for (bool tb : {false, true}) {
          auto ah = (ta ? a.transpose() : a).contiguous();
          auto bh = (tb ? b.transpose() : b).contiguous();
          auto ad = ah.to(gpu), bd = bh.to(gpu);
          ad.requires_grad_();
          bd.requires_grad_();
          ah.requires_grad_();
          bh.requires_grad_();
          auto actual = nn::matmul(ad, bd, ta, tb), expected = nn::matmul(ah, bh, ta, tb);
          EXPECT_TRUE(actual.context().shares_with(gpu));
          auto values    = nn::cast(actual.to(cpu), nn::DType::Float64).to_vector<double>();
          auto reference = nn::cast(expected, nn::DType::Float64).to_vector<double>();
          for (std::size_t i = 0; i < values.size(); ++i)
            EXPECT_NEAR(values[i], reference[i], tolerance);
          auto gd = nn::grad({actual}, {ad, bd}, {nn::ones_like(actual)});
          auto gh = nn::grad({expected}, {ah, bh}, {nn::ones_like(expected)});
          for (int j = 0; j < 2; ++j) {
            EXPECT_TRUE(gd[j].context().shares_with(gpu));
            const auto g = nn::cast(gd[j].to(cpu), nn::DType::Float64).to_vector<double>();
            const auto r = nn::cast(gh[j], nn::DType::Float64).to_vector<double>();
            for (std::size_t i = 0; i < g.size(); ++i)
              EXPECT_NEAR(g[i], r[i], tolerance);
          }
        }
      // Partially broadcast batches and noncontiguous negative-stride matrix views.
      auto ah  = nn::broadcast_to(a, {2, 1, 2, 3});
      auto bh  = nn::broadcast_to(b, {1, 3, 3, 4});
      auto av  = nn::slice_axis(ah.to(gpu), -1, 2, -4, -1);
      auto bv  = nn::swapaxes(bh.to(gpu), -1, -2);
      auto got = nn::matmul(av, bv, false, true).to(cpu);
      auto ref = nn::matmul(nn::slice_axis(ah, -1, 2, -4, -1), bh);
      EXPECT_EQ(got.shape(), (nn::Shape{2, 3, 2, 4}));
      auto g = nn::cast(got, nn::DType::Float64).to_vector<double>();
      auto r = nn::cast(ref, nn::DType::Float64).to_vector<double>();
      for (std::size_t i = 0; i < g.size(); ++i)
        EXPECT_NEAR(g[i], r[i], tolerance);
      auto ba = nn::broadcast_to(a, {2, 2, 3}).contiguous().detach();
      auto bb = nn::broadcast_to(b, {2, 3, 4}).contiguous().detach();
      auto da = ba.to(gpu), db = bb.to(gpu);
      ba.requires_grad_();
      bb.requires_grad_();
      da.requires_grad_();
      db.requires_grad_();
      auto gh = nn::grad({nn::matmul(ba, bb)}, {ba, bb}, {nn::ones({2, 2, 4}, dtype)});
      auto gd = nn::grad({nn::matmul(da, db)}, {da, db}, {nn::full({2, 2, 4}, 1, dtype, gpu)});
      for (int j = 0; j < 2; ++j) {
        const auto g = nn::cast(gd[j].to(cpu), nn::DType::Float64).to_vector<double>();
        const auto r = nn::cast(gh[j], nn::DType::Float64).to_vector<double>();
        for (std::size_t i = 0; i < g.size(); ++i)
          EXPECT_NEAR(g[i], r[i], tolerance);
      }
      auto z = nn::matmul(nn::empty({2, 0}, dtype, gpu), nn::empty({0, 4}, dtype, gpu));
      EXPECT_EQ(
          nn::cast(z.to(cpu), nn::DType::Float64).to_vector<double>(), std::vector<double>(8, 0));
      EXPECT_EQ(nn::matmul(nn::empty({0, 3}, dtype, gpu), b.to(gpu)).numel(), 0);
    }
    auto x = nn::ones({2, 2}).to(gpu);
    EXPECT_THROW(nn::matmul(x, nn::ones({2, 2})), std::invalid_argument);
    EXPECT_THROW(
        nn::matmul(x, nn::ones({2, 2}, nn::DType::Float64).to(gpu)), std::invalid_argument);
    auto i = nn::ones({2, 2}, nn::DType::Int64).to(gpu);
    EXPECT_THROW(nn::matmul(i, i), std::invalid_argument);
    c::Context foreign({c::Backend::CUDA, 0});
    EXPECT_THROW(nn::matmul(x, x.to(foreign)), std::invalid_argument);
  }
  EXPECT_EQ(gpu.live_bytes(), 0);
  EXPECT_EQ(nn::live_memory_bytes(), baseline);
}

namespace {
/// Maximum absolute difference between a device result and a host reference.
double max_error(const nn::Tensor& device, const nn::Tensor& host) {
  EXPECT_EQ(device.shape(), host.shape());
  const auto a = nn::cast(device.to(c::Context{}), nn::DType::Float64).to_vector<double>();
  const auto b = nn::cast(host, nn::DType::Float64).to_vector<double>();
  double     e = 0;
  for (std::size_t i = 0; i < a.size(); ++i)
    e = std::max(e, std::abs(a[i] - b[i]));
  return e;
}
} // namespace

TEST(NnDevice, CudaInferenceOpsMatchCpu) {
  const auto info = c::probe({c::Backend::CUDA, 0});
  if (!info.available) {
    ASSERT_FALSE(std::getenv("DATAMUNGE_REQUIRE_CUDA")) << info.diagnostic;
    GTEST_SKIP() << info.diagnostic;
  }
  c::Context gpu({c::Backend::CUDA, 0});
  ASSERT_TRUE(gpu.supports_matmul()) << "CUDA contexts always run the in-house GEMM";
  const auto baseline = nn::live_memory_bytes();
  {
    nn::NoGradGuard guard;
    nn::Generator   rng(5);
    for (auto dtype : {nn::DType::Float32, nn::DType::Float64}) {
      const double tolerance = dtype == nn::DType::Float32 ? 2e-5 : 1e-12;
      const auto   x         = nn::normal({3, 9, 7, 5}, 0, 1, rng, dtype);
      const auto   xd        = x.to(gpu);
      // Same/valid/explicit padding, strides, dilation and the pointwise GEMM shortcut.
      std::vector<std::pair<std::int64_t, nn::Conv2DOptions>> cases;
      cases.push_back({3, nn::conv_options(1)});
      cases.push_back({3, nn::conv_options(2, nn::Padding::Valid)});
      cases.push_back({1, nn::conv_options(1)});
      cases.push_back({1, nn::conv_options(2)});
      cases.push_back({3, nn::conv_options(1, nn::Padding::Same, 2)});
      nn::Conv2DOptions explicit_pad = nn::conv_options(2, nn::Padding::Explicit);
      explicit_pad.explicit_paddings = {1, 2, 0, 1};
      cases.push_back({5, explicit_pad});
      for (const auto& [k, options] : cases) {
        const auto f = nn::normal({k, k, 5, 4}, 0, 1, rng, dtype);
        EXPECT_LT(max_error(nn::conv2d(xd, f.to(gpu), options), nn::conv2d(x, f, options)),
                  tolerance * 10);
      }
      // A transposed (noncontiguous) device view is packed before the kernel runs.
      const auto xt = nn::swapaxes(x, 1, 2);
      const auto f  = nn::normal({3, 3, 5, 2}, 0, 1, rng, dtype);
      EXPECT_LT(max_error(nn::conv2d(nn::swapaxes(xd, 1, 2), f.to(gpu)), nn::conv2d(xt, f)),
                tolerance * 10);

      const auto with_nan = nn::concat(
          {nn::reshape(x, {-1}), nn::full({1}, std::numeric_limits<double>::quiet_NaN(), dtype)}, 0);
      EXPECT_EQ(max_error(nn::relu(with_nan.to(gpu)), nn::relu(with_nan)), 0);

      const auto bias = nn::normal({5}, 0, 1, rng, dtype), pixel = nn::reshape(bias, {1, 1, 1, 5});
      const auto y    = nn::normal({3, 9, 7, 5}, 0, 1, rng, dtype);
      EXPECT_EQ(max_error(nn::add(xd, y.to(gpu)), nn::add(x, y)), 0);
      EXPECT_EQ(max_error(nn::add(xd, bias.to(gpu)), nn::add(x, bias)), 0);
      EXPECT_EQ(max_error(nn::add(bias.to(gpu), xd), nn::add(bias, x)), 0);
      EXPECT_EQ(max_error(nn::bias_add(xd, bias.to(gpu)), nn::bias_add(x, bias)), 0);
      EXPECT_EQ(max_error(xd + pixel.to(gpu), x + pixel), 0);
      EXPECT_THROW(nn::add(xd, nn::ones({3, 1, 1, 1}, dtype).to(gpu)), std::logic_error);

      const auto gamma = nn::normal({5}, 1, .2, rng, dtype), beta = nn::normal({5}, 0, 1, rng, dtype);
      const auto mean = nn::normal({5}, 0, 1, rng, dtype),
                 variance = nn::abs(nn::normal({5}, 1, .5, rng, dtype));
      EXPECT_LT(
          max_error(
              nn::batch_norm_infer(xd, gamma.to(gpu), beta.to(gpu), mean.to(gpu), variance.to(gpu), 1e-5),
              nn::batch_norm_infer(x, gamma, beta, mean, variance, 1e-5)),
          tolerance);
      for (bool keep : {false, true})
        EXPECT_LT(max_error(nn::global_avg_pool2d(xd, keep), nn::global_avg_pool2d(x, keep)),
                  tolerance);
      EXPECT_THROW(nn::add(xd, x), std::invalid_argument);

      // Concat on every axis, including a transposed (noncontiguous) operand.
      const auto z = nn::normal({3, 9, 7, 2}, 0, 1, rng, dtype);
      EXPECT_EQ(max_error(nn::concat({xd, z.to(gpu), xd}, 3), nn::concat({x, z, x}, 3)), 0);
      EXPECT_EQ(max_error(nn::concat({xd, y.to(gpu)}, 0), nn::concat({x, y}, 0)), 0);
      EXPECT_EQ(max_error(nn::concat({xd, y.to(gpu)}, 2), nn::concat({x, y}, 2)), 0);
      EXPECT_EQ(max_error(nn::concat({nn::swapaxes(xd, 1, 2), nn::swapaxes(y.to(gpu), 1, 2)}, 1),
                          nn::concat({nn::swapaxes(x, 1, 2), nn::swapaxes(y, 1, 2)}, 1)),
                0);
      // Constant padding (rank 4 and rank 2), including the -inf used by ceil pooling.
      // Bit-exact comparison, since infinite values cannot be subtracted.
      const c::Context host;
      const auto inf = -std::numeric_limits<double>::infinity();
      const std::vector<std::pair<std::int64_t, std::int64_t>> p4{{0, 1}, {2, 0}, {0, 1}, {1, 1}};
      EXPECT_TRUE(nn::equal_bits(nn::pad(xd, p4, nn::PadMode::Constant, 1.5).to(host),
                                 nn::pad(x, p4, nn::PadMode::Constant, 1.5)));
      const auto m = nn::normal({4, 6}, 0, 1, rng, dtype);
      EXPECT_TRUE(nn::equal_bits(nn::pad(m.to(gpu), {{1, 0}, {0, 2}}, nn::PadMode::Constant, inf).to(host),
                                 nn::pad(m, {{1, 0}, {0, 2}}, nn::PadMode::Constant, inf)));
      // Max pooling with SAME padding (skipped taps), VALID windows and the ceil-pool pattern,
      // where -inf padding never beats real data and fully padded channels stay -inf.
      for (auto padding : {nn::Padding::Same, nn::Padding::Valid})
        EXPECT_TRUE(nn::equal_bits(nn::max_pool2d(xd, 3, 2, padding).to(host),
                                   nn::max_pool2d(x, 3, 2, padding)));
      const auto negative = -nn::abs(x);
      EXPECT_TRUE(nn::equal_bits(
          nn::max_pool2d(nn::pad(negative.to(gpu), p4, nn::PadMode::Constant, inf), 3, 2,
                         nn::Padding::Valid)
              .to(host),
          nn::max_pool2d(nn::pad(negative, p4, nn::PadMode::Constant, inf), 3, 2,
                         nn::Padding::Valid)));
    }
    // A complete compact ResNet copied to the GPU with ParameterStore::to.
    nn::models::ResNetConfig cfg;
    cfg.depth = 18;
    cfg.classes = 10;
    cfg.width_divisor = 8;
    cfg.small_stem = true;
    nn::ParameterStore p(3), state(3);
    const auto images = nn::normal({4, 16, 16, 3}, 0, 1, rng);
    {
      nn::EnableGradGuard train;
      nn::models::ResNet(p, state, images, cfg, true); // non-default moving statistics
    }
    auto       dp = p.to(gpu), ds = state.to(gpu);
    EXPECT_EQ(dp.names(), p.names());
    EXPECT_EQ(dp.count(), p.count());
    EXPECT_TRUE(dp.parameters().front().context().shares_with(gpu));
    const auto expected = nn::models::ResNet(p, state, images, cfg, false).logits;
    const auto actual   = nn::models::ResNet(dp, ds, images.to(gpu), cfg, false).logits;
    EXPECT_LT(max_error(actual, expected), 1e-4);

    // A full-width CIFAR SqueezeNet (concat, ceil pooling, biased ReLU convs, class-map head).
    for (int version : {10, 11}) {
      nn::models::SqueezeNetConfig sq;
      sq.version    = version;
      sq.cifar_stem = true;
      sq.image_size = 32;
      sq.classes    = 10;
      nn::ParameterStore sp(7);
      const auto         cifar = nn::normal({3, 32, 32, 3}, 0, 1, rng);
      const auto host_logits = nn::models::SqueezeNet(sp, cifar, sq, false).logits;
      auto       sp_gpu      = sp.to(gpu);
      const auto gpu_logits  = nn::models::SqueezeNet(sp_gpu, cifar.to(gpu), sq, false).logits;
      EXPECT_LT(max_error(gpu_logits, host_logits), 1e-5);
    }
  }
  {
    // Forward-only: a GPU input that needs gradients is rejected while tracking.
    auto x = nn::ones({2, 3}).to(gpu);
    x.requires_grad_();
    EXPECT_THROW(nn::relu(x), std::logic_error);
    nn::Conv2DOptions grouped = nn::conv_options(1);
    grouped.groups = 2;
    nn::NoGradGuard guard;
    EXPECT_THROW(nn::conv2d(nn::ones({1, 4, 4, 2}).to(gpu), nn::ones({1, 1, 1, 2}).to(gpu), grouped),
                 std::logic_error);
  }
  EXPECT_EQ(gpu.live_bytes(), 0);
  EXPECT_EQ(nn::live_memory_bytes(), baseline);
}
