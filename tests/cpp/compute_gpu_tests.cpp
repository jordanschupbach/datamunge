#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <random>

#include <array>
#include <cstdint>
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
TEST(ComputeGpu, CudaInHouseGemmMatchesHostOnAwkwardShapes) {
  const auto info = c::probe({c::Backend::CUDA, 0});
  if (!info.available) {
    ASSERT_FALSE(std::getenv("DATAMUNGE_REQUIRE_CUDA")) << info.diagnostic;
    GTEST_SKIP() << info.diagnostic;
  }
  c::Context gpu({c::Backend::CUDA, 0});
  ASSERT_TRUE(gpu.supports_matmul());
  std::mt19937                           rng(17);
  std::uniform_real_distribution<double> u(-1, 1);
  // From {129, 65, 600} on the outputs have few tiles and k is long, so the planner splits every
  // tile into k chunks (short chunks for {129, 65, 600} and {40, 30, 300}, long ones for the last
  // three) and reduces them in a second pass; CudaInHouseGemmLargeShapesMatchHost covers the big
  // tiles and the mixed whole/split plans.
  const std::size_t shapes[][3] = {{37, 53, 71},   {130, 257, 9},  {1, 300, 40},   {300, 1, 40},
                                   {1, 1, 1},      {129, 65, 600}, {40, 30, 300},  {300, 130, 520},
                                   {37, 20, 3001}, {144, 64, 4100}, {5, 7, 70000}};
  for (const auto& shape : shapes)
    for (bool ta : {false, true})
      for (bool tb : {false, true})
        for (int precision = 0; precision < 2; ++precision) {
          const std::size_t   m = shape[0], n = shape[1], k = shape[2];
          std::vector<double> a(m * k), b(k * n);
          for (auto& v : a) v = u(rng);
          for (auto& v : b) v = u(rng);
          std::vector<double> reference(m * n);
          for (std::size_t i = 0; i < m; ++i)
            for (std::size_t j = 0; j < n; ++j) {
              double sum = 0;
              for (std::size_t l = 0; l < k; ++l)
                sum += (ta ? a[l * m + i] : a[i * k + l]) * (tb ? b[j * k + l] : b[l * n + j]);
              reference[i * n + j] = sum;
            }
          const c::MatrixProduct product{m, n, k, precision == 0 ? c::MatrixType::Float32 : c::MatrixType::Float64, ta, tb};
          std::vector<double>    result(m * n);
          if (precision == 0) {
            std::vector<float> af(a.begin(), a.end()), bf(b.begin(), b.end()), cf(m * n);
            // Recompute the float reference from the rounded inputs.
            for (std::size_t i = 0; i < m; ++i)
              for (std::size_t j = 0; j < n; ++j) {
                double sum = 0;
                for (std::size_t l = 0; l < k; ++l)
                  sum += double(ta ? af[l * m + i] : af[i * k + l]) * double(tb ? bf[j * k + l] : bf[l * n + j]);
                reference[i * n + j] = sum;
              }
            auto da = gpu.allocate(af.size() * 4), db = gpu.allocate(bf.size() * 4), dc = gpu.allocate(cf.size() * 4);
            gpu.upload(da, af.data(), af.size() * 4);
            gpu.upload(db, bf.data(), bf.size() * 4);
            gpu.matmul(dc, da, db, product);
            gpu.download(cf.data(), dc, cf.size() * 4);
            result.assign(cf.begin(), cf.end());
          } else {
            auto da = gpu.allocate(a.size() * 8), db = gpu.allocate(b.size() * 8), dc = gpu.allocate(m * n * 8);
            gpu.upload(da, a.data(), a.size() * 8);
            gpu.upload(db, b.data(), b.size() * 8);
            gpu.matmul(dc, da, db, product);
            gpu.download(result.data(), dc, m * n * 8);
          }
          const double tolerance = precision == 0 ? 1e-4 : 1e-10;
          for (std::size_t i = 0; i < m * n; ++i)
            ASSERT_NEAR(result[i], reference[i], tolerance * (1 + std::abs(reference[i])) * std::sqrt(double(k)))
                << "m=" << m << " n=" << n << " k=" << k << " ta=" << ta << " tb=" << tb << " f" << (precision ? 64 : 32)
                << " element " << i;
        }
}

namespace {
/// One in-house GEMM of random [-1, 1) operands on `gpu`; returns the product and the operands as
/// the device saw them (rounded to float for Float32).
struct GemmRun {
  std::vector<double> a, b, c;
};
GemmRun run_gemm(c::Context& gpu, std::size_t m, std::size_t n, std::size_t k, bool ta, bool tb, bool f32,
                 std::uint32_t seed) {
  std::mt19937                           rng(seed);
  std::uniform_real_distribution<double> u(-1, 1);
  GemmRun                                r{std::vector<double>(m * k), std::vector<double>(k * n), {}};
  for (auto& v : r.a) v = f32 ? double(float(u(rng))) : u(rng);
  for (auto& v : r.b) v = f32 ? double(float(u(rng))) : u(rng);
  const c::MatrixProduct product{m, n, k, f32 ? c::MatrixType::Float32 : c::MatrixType::Float64, ta, tb};
  r.c.resize(m * n);
  if (f32) {
    std::vector<float> af(r.a.begin(), r.a.end()), bf(r.b.begin(), r.b.end()), cf(m * n);
    auto da = gpu.allocate(af.size() * 4), db = gpu.allocate(bf.size() * 4), dc = gpu.allocate(cf.size() * 4);
    gpu.upload(da, af.data(), af.size() * 4);
    gpu.upload(db, bf.data(), bf.size() * 4);
    gpu.matmul(dc, da, db, product);
    gpu.download(cf.data(), dc, cf.size() * 4);
    r.c.assign(cf.begin(), cf.end());
  } else {
    auto da = gpu.allocate(r.a.size() * 8), db = gpu.allocate(r.b.size() * 8), dc = gpu.allocate(m * n * 8);
    gpu.upload(da, r.a.data(), r.a.size() * 8);
    gpu.upload(db, r.b.data(), r.b.size() * 8);
    gpu.matmul(dc, da, db, product);
    gpu.download(r.c.data(), dc, m * n * 8);
  }
  return r;
}
} // namespace

TEST(ComputeGpu, CudaInHouseGemmLargeShapesMatchHost) {
  const auto info = c::probe({c::Backend::CUDA, 0});
  if (!info.available) {
    ASSERT_FALSE(std::getenv("DATAMUNGE_REQUIRE_CUDA")) << info.diagnostic;
    GTEST_SKIP() << info.diagnostic;
  }
  c::Context gpu({c::Backend::CUDA, 0});
  // Outputs big enough for the 128 x 256 and 128 x 128 float tiles, with ragged edges (rows,
  // columns and k not multiples of the tile or of the 4-wide vectors, which also turns off the
  // 128-bit loads of the row-contiguous operands), and tile counts that are not whole waves, so
  // the planner computes full waves of tiles whole and splits the leftover tiles in k.
  const std::size_t shapes[][3] = {{2048, 2048, 512}, {1999, 2001, 999}, {1500, 1700, 1300},
                                   {3000, 777, 2000}, {1023, 1025, 257}, {640, 2304, 1153}};
  std::mt19937      pick(5);
  std::uint32_t     seed = 1;
  for (const auto& shape : shapes)
    for (bool ta : {false, true})
      for (bool tb : {false, true})
        for (bool f32 : {true, false}) {
          const std::size_t m = shape[0], n = shape[1], k = shape[2];
          const auto        r = run_gemm(gpu, m, n, k, ta, tb, f32, seed++);
          // Every element of the last row and column (tile edges) and 2000 random ones.
          std::vector<std::size_t> rows, cols;
          for (std::size_t j = 0; j < n; ++j) rows.push_back(m - 1), cols.push_back(j);
          for (std::size_t i = 0; i < m; ++i) rows.push_back(i), cols.push_back(n - 1);
          for (int s = 0; s < 2000; ++s) rows.push_back(pick() % m), cols.push_back(pick() % n);
          for (std::size_t e = 0; e < rows.size(); ++e) {
            const std::size_t i = rows[e], j = cols[e];
            double            sum = 0, mag = 0;
            for (std::size_t l = 0; l < k; ++l) {
              const double x = (ta ? r.a[l * m + i] : r.a[i * k + l]) * (tb ? r.b[j * k + l] : r.b[l * n + j]);
              sum += x;
              mag += std::abs(x);
            }
            // Accumulation error bound of a length-k sum of products in the device precision.
            const double eps = f32 ? 5.96e-8 : 1.11e-16;
            ASSERT_LE(std::abs(r.c[i * n + j] - sum), (double(k) + 4) * eps * mag + 1e-300)
                << "m=" << m << " n=" << n << " k=" << k << " ta=" << ta << " tb=" << tb << " f" << (f32 ? 32 : 64)
                << " element (" << i << ", " << j << ")";
          }
        }
}

TEST(ComputeGpu, CudaInHouseGemmIsDeterministic) {
  const auto info = c::probe({c::Backend::CUDA, 0});
  if (!info.available) {
    ASSERT_FALSE(std::getenv("DATAMUNGE_REQUIRE_CUDA")) << info.diagnostic;
    GTEST_SKIP() << info.diagnostic;
  }
  c::Context gpu({c::Backend::CUDA, 0});
  // Whole tiles only, whole waves plus split leftovers, and every tile split into long k chunks:
  // the same device and shape must give bit-identical results (no atomics, the chunks' partial
  // sums added in a fixed order: groups of eight chunks, then the groups).
  const std::size_t shapes[][3] = {{512, 512, 512}, {2048, 2048, 2048}, {1999, 2001, 999}, {144, 64, 8192},
                                   {5, 7, 70000}};
  for (const auto& shape : shapes)
    for (bool f32 : {true, false})
      for (bool ta : {false, true}) {
        const bool tb = !ta;
        const auto first = run_gemm(gpu, shape[0], shape[1], shape[2], ta, tb, f32, 99);
        for (int repeat = 0; repeat < 2; ++repeat) {
          const auto again = run_gemm(gpu, shape[0], shape[1], shape[2], ta, tb, f32, 99);
          ASSERT_EQ(std::memcmp(first.c.data(), again.c.data(), first.c.size() * sizeof(double)), 0)
              << "m=" << shape[0] << " n=" << shape[1] << " k=" << shape[2] << " ta=" << ta << " f"
              << (f32 ? 32 : 64);
        }
      }
}

TEST(ComputeGpu, CudaMatrixAdapterValidatesViewsAndEmptyProducts) {
  const auto info = c::probe({c::Backend::CUDA, 0});
  if (!info.available) {
    ASSERT_FALSE(std::getenv("DATAMUNGE_REQUIRE_CUDA")) << info.diagnostic;
    GTEST_SKIP() << info.diagnostic;
  }
  c::Context gpu({c::Backend::CUDA, 0});
  if (!gpu.supports_matmul()) {
    // Every CUDA device runs the in-house GEMM kernels.
    FAIL() << "CUDA contexts always support matmul";
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
