#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <datamunge/compute/context.hpp>
#include <datamunge/nn/autograd.hpp>
#include <datamunge/nn/io.hpp>
#include <datamunge/nn/memory.hpp>
#include <datamunge/nn/ops.hpp>
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
  if (!gpu.supports_matmul()) {
    ASSERT_FALSE(std::getenv("DATAMUNGE_REQUIRE_CUDA")) << "cuBLAS is disabled";
    GTEST_SKIP() << "cuBLAS is disabled";
  }
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
