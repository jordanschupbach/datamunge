#include <gtest/gtest.h>

#include <array>
#include <cstdlib>
#include <cstring>
#include <datamunge/compute/context.hpp>
#include <iostream>

#include "../fixtures/compute/axpy_spirv.hpp"
namespace c = datamunge::compute;
namespace {
// The third ABI parameter is an eight-byte by-value {float,uint32} block.
const char* axpy_ptx = R"ptx(.version 6.0
.target sm_50
.address_size 64
.visible .entry main(.param .u64 x,.param .u64 y,.param .align 4 .b8 args[8]){
.reg .pred p;
.reg .b32 r<6>;
.reg .b64 rd<6>;
.reg .f32 f<4>;
ld.param.u64 rd0,[x];ld.param.u64 rd1,[y];
ld.param.f32 f0,[args];ld.param.u32 r0,[args+4];
mov.u32 r1,%ctaid.x;mov.u32 r2,%ntid.x;mov.u32 r3,%tid.x;
mad.lo.u32 r4,r1,r2,r3;setp.ge.u32 p,r4,r0;@p bra done;
mul.wide.u32 rd2,r4,4;add.u64 rd3,rd0,rd2;add.u64 rd4,rd1,rd2;
ld.global.f32 f1,[rd3];ld.global.f32 f2,[rd4];fma.rn.f32 f3,f0,f1,f2;st.global.f32 [rd4],f3;
done: ret;
})ptx";
void        exercise(c::Backend backend) {
  c::Context         context({backend, 0});
  std::vector<float> x(257), y(257);
  for (std::size_t i = 0; i < x.size(); ++i) {
    x[i] = float(i) + .25f;
    y[i] = -float(i);
  }
  auto xb = context.allocate(x.size() * sizeof(float)),
       yb = context.allocate(y.size() * sizeof(float));
  context.upload(xb, x.data(), xb.size());
  context.upload(yb, y.data(), yb.size());
  EXPECT_THROW(xb.host_data(), std::logic_error);
  EXPECT_THROW(context.parallel_for(1, [](std::size_t) {}), std::logic_error);
  c::KernelSource source;
  source.buffers         = 2;
  source.parameter_bytes = 8;
  source.ptx             = axpy_ptx;
  source.spirv.assign(
      std::begin(compute_fixture::axpy_spirv), std::end(compute_fixture::axpy_spirv));
  const auto program = context.program(source);
  struct Args {
    float         a;
    std::uint32_t n;
  } args{2, 257};
  context.dispatch(program, {xb, yb}, &args, sizeof(args), {5, 1, 1});
  context.download(y.data(), yb, yb.size());
  for (std::size_t i = 0; i < y.size(); ++i)
    EXPECT_FLOAT_EQ(y[i], float(i) + .5f);
  EXPECT_THROW(context.dispatch(program, {xb}, &args, sizeof(args)), std::invalid_argument);
  EXPECT_THROW(context.dispatch(program, {xb, yb}, &args, 4), std::invalid_argument);
  EXPECT_THROW(
      context.dispatch(program, {xb, yb}, &args, sizeof(args), {0, 1, 1}), std::invalid_argument);
  auto bytes = context.allocate(7);
  context.fill(bytes, 0);
  auto                        slice = bytes.slice(1, 3);
  std::array<std::uint8_t, 3> input{1, 2, 3};
  context.upload(slice, input.data(), 3);
  std::array<std::uint8_t, 7> out{};
  context.download(out.data(), bytes, out.size());
  EXPECT_EQ(out, (std::array<std::uint8_t, 7>{0, 1, 2, 3, 0, 0, 0}));
  context.fill(slice, 8);
  context.download(out.data(), bytes, out.size());
  EXPECT_EQ(out, (std::array<std::uint8_t, 7>{0, 8, 8, 8, 0, 0, 0}));
  auto copy = context.allocate(xb.size());
  context.copy(copy, xb, xb.size());
  std::vector<float> copied(x.size());
  context.download(copied.data(), copy, copy.size());
  EXPECT_EQ(copied, x);
  context.synchronize();
}
} // namespace
TEST(ComputeGpu, VulkanNativeBufferAndShaderExecution) {
  const auto info = c::probe({c::Backend::Vulkan, 0});
  if (!info.available)
    GTEST_SKIP() << info.diagnostic;
  std::cout << "Vulkan compute device: " << info.name << '\n';
  exercise(c::Backend::Vulkan);
}
TEST(ComputeGpu, CudaNativeBufferAndPtxExecution) {
  const auto info = c::probe({c::Backend::CUDA, 0});
  if (!info.available) {
    ASSERT_FALSE(std::getenv("DATAMUNGE_REQUIRE_CUDA")) << info.diagnostic;
    GTEST_SKIP() << info.diagnostic;
  }
  std::cout << "CUDA compute device: " << info.name << '\n';
  exercise(c::Backend::CUDA);
}
TEST(ComputeGpu, CudaMatrixAdapterValidatesViewsAndEmptyProducts) {
  const auto info = c::probe({c::Backend::CUDA, 0});
  if (!info.available) {
    ASSERT_FALSE(std::getenv("DATAMUNGE_REQUIRE_CUDA")) << info.diagnostic;
    GTEST_SKIP() << info.diagnostic;
  }
  c::Context gpu({c::Backend::CUDA, 0});
  if (!gpu.supports_matmul()) {
    ASSERT_FALSE(std::getenv("DATAMUNGE_REQUIRE_CUDA")) << "cuBLAS is disabled";
    auto x = gpu.allocate(4);
    EXPECT_THROW(gpu.matmul(x, x, x, {}), std::logic_error);
    GTEST_SKIP() << "cuBLAS is disabled";
  }
  auto                       a = gpu.allocate(24), b = gpu.allocate(24), output = gpu.allocate(32);
  const std::array<float, 6> av{1, 2, 3, 4, 5, 6}, bv{7, 8, 9, 10, 11, 12};
  gpu.upload(a, av.data(), 24);
  gpu.upload(b, bv.data(), 24);
  gpu.fill(output, 0x7f);
  auto             dst = output.slice(4, 16);
  c::MatrixProduct product{2, 2, 3};
  gpu.matmul(dst, a, b, product);
  std::array<float, 4> result{};
  gpu.download(result.data(), dst, 16);
  EXPECT_EQ(result, (std::array<float, 4>{58, 64, 139, 154}));
  std::array<unsigned char, 32> bytes{};
  gpu.download(bytes.data(), output, 32);
  EXPECT_EQ(bytes[0], 0x7f);
  EXPECT_EQ(bytes[20], 0x7f);
  auto small = output.slice(4, 15), unaligned = output.slice(1, 16);
  EXPECT_THROW(gpu.matmul(small, a, b, product), std::out_of_range);
  EXPECT_THROW(gpu.matmul(unaligned, a, b, product), std::invalid_argument);
  EXPECT_THROW(gpu.matmul(a, a, b, product), std::invalid_argument);
  c::Context other({c::Backend::CUDA, 0});
  auto       foreign = other.allocate(24);
  EXPECT_THROW(gpu.matmul(dst, foreign, b, product), std::invalid_argument);
  product.type = static_cast<c::MatrixType>(999);
  EXPECT_THROW(gpu.matmul(dst, a, b, product), std::invalid_argument);
  product.type = c::MatrixType::Float32;
  product.rows = std::size_t(1) << 31;
  EXPECT_THROW(gpu.matmul(dst, a, b, product), std::invalid_argument);
  product = {std::size_t(0x7fffffff), std::size_t(0x7fffffff), 1, c::MatrixType::Float64};
  EXPECT_THROW(gpu.matmul(dst, a, b, product), std::overflow_error);
  product    = {2, 2, 0};
  auto empty = gpu.allocate(0);
  gpu.matmul(dst, empty, empty, product);
  gpu.download(result.data(), dst, 16);
  EXPECT_EQ(result, (std::array<float, 4>{0, 0, 0, 0}));
  product = {0, 2, 0};
  EXPECT_NO_THROW(gpu.matmul(empty, empty, empty, product));
  // Invalid PTX reports the driver JIT's compiler log and leaves the context usable.
  c::KernelSource bad;
  bad.ptx = ".version 6.0\n.target sm_50\n.address_size 64\ninvalid instruction;";
  try {
    gpu.program(bad);
    FAIL() << "invalid PTX accepted";
  } catch (const std::runtime_error& error) {
    EXPECT_NE(std::string(error.what()).find("cuModuleLoadDataEx"), std::string::npos);
    EXPECT_NE(std::string(error.what()).find("error"), std::string::npos);
  }
  gpu.synchronize();
}
TEST(ComputeGpu, CpuTransfersAndUndefinedProgramsAreValidated) {
  c::Context    context;
  auto          buffer = context.allocate(8);
  std::uint64_t input = 9007199254740993ull, output = 0;
  context.upload(buffer, &input, 8);
  context.download(&output, buffer, 8);
  EXPECT_EQ(input, output);
  EXPECT_THROW(context.upload(buffer, nullptr, 1), std::invalid_argument);
  EXPECT_THROW(context.download(nullptr, buffer, 1), std::invalid_argument);
  EXPECT_THROW(context.upload(buffer, &input, 9), std::out_of_range);
  EXPECT_THROW(context.program({}), std::logic_error);
  EXPECT_THROW(context.dispatch({}, {}, nullptr, 0), std::invalid_argument);
  EXPECT_FALSE(context.supports_matmul());
  EXPECT_THROW(context.matmul(buffer, buffer, buffer, {}), std::logic_error);
}
