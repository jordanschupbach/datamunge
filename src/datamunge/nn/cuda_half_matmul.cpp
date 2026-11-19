/// @file cuda_half_matmul.cpp
/// @brief Native binary16 matrix products using CUDA Driver API kernels.

#include <datamunge/nn/cuda_half_matmul.hpp>

#include <array>
#include <limits>
#include <stdexcept>

namespace datamunge::nn {
/// @brief CUDA conversion and product programs with their launch geometry.
namespace {

/// @brief CUDA block size for both flat-index kernels.
constexpr std::uint32_t block_size = 128;

/// @brief Convert one Float32 element per thread to packed IEEE binary16.
constexpr char pack_ptx[] = R"ptx(
.version 6.4
.target sm_53
.address_size 64
.visible .entry pack_half(
  .param .u64 source,
  .param .u64 destination,
  .param .align 4 .b8 count[4]
) {
  .reg .pred %outside;
  .reg .u32 %i, %block, %lane, %width, %n;
  .reg .u64 %in, %out, %offset, %address;
  .reg .f32 %value;
  .reg .b16 %half;
  mov.u32 %block, %ctaid.x;
  mov.u32 %lane, %tid.x;
  mov.u32 %width, %ntid.x;
  mad.lo.u32 %i, %block, %width, %lane;
  ld.param.u32 %n, [count];
  setp.ge.u32 %outside, %i, %n;
  @%outside bra DONE;
  ld.param.u64 %in, [source];
  ld.param.u64 %out, [destination];
  mul.wide.u32 %offset, %i, 4;
  add.u64 %address, %in, %offset;
  ld.global.f32 %value, [%address];
  cvt.rn.f16.f32 %half, %value;
  mul.wide.u32 %offset, %i, 2;
  add.u64 %address, %out, %offset;
  st.global.b16 [%address], %half;
DONE:
  ret;
}
)ptx";

/// @brief One dot product per thread with Float32 FMA and binary16 rounding.
constexpr char product_ptx[] = R"ptx(
.version 6.4
.target sm_53
.address_size 64
.visible .entry half_product(
  .param .u64 a,
  .param .u64 b,
  .param .u64 c,
  .param .align 4 .b8 dims[16]
) {
  .reg .pred %outside, %transposed, %done;
  .reg .u32 %i, %block, %lane, %width, %m, %n, %k, %transpose;
  .reg .u32 %total, %row, %col, %step, %ai, %bi, %alt;
  .reg .u64 %ap, %bp, %cp, %offset, %address;
  .reg .b16 %av, %bv, %half;
  .reg .f32 %x, %y, %acc, %rounded;
  mov.u32 %block, %ctaid.x;
  mov.u32 %lane, %tid.x;
  mov.u32 %width, %ntid.x;
  mad.lo.u32 %i, %block, %width, %lane;
  ld.param.u32 %m, [dims];
  ld.param.u32 %n, [dims+4];
  ld.param.u32 %k, [dims+8];
  ld.param.u32 %transpose, [dims+12];
  mul.lo.u32 %total, %m, %n;
  setp.ge.u32 %outside, %i, %total;
  @%outside bra DONE;
  ld.param.u64 %ap, [a];
  ld.param.u64 %bp, [b];
  ld.param.u64 %cp, [c];
  div.u32 %row, %i, %n;
  rem.u32 %col, %i, %n;
  setp.ne.u32 %transposed, %transpose, 0;
  mov.u32 %step, 0;
  mov.f32 %acc, 0f00000000;
LOOP:
  mad.lo.u32 %ai, %row, %k, %step;
  mad.lo.u32 %alt, %step, %m, %row;
  selp.u32 %ai, %alt, %ai, %transposed;
  mad.lo.u32 %bi, %step, %n, %col;
  mul.wide.u32 %offset, %ai, 2;
  add.u64 %address, %ap, %offset;
  ld.global.b16 %av, [%address];
  mul.wide.u32 %offset, %bi, 2;
  add.u64 %address, %bp, %offset;
  ld.global.b16 %bv, [%address];
  cvt.f32.f16 %x, %av;
  cvt.f32.f16 %y, %bv;
  fma.rn.f32 %acc, %x, %y, %acc;
  add.u32 %step, %step, 1;
  setp.ge.u32 %done, %step, %k;
  @!%done bra LOOP;
  cvt.rn.f16.f32 %half, %acc;
  cvt.f32.f16 %rounded, %half;
  mul.wide.u32 %offset, %i, 4;
  add.u64 %address, %cp, %offset;
  st.global.f32 [%address], %rounded;
DONE:
  ret;
}
)ptx";

/// @brief Compute a nonoverflowing ceiling for a flat element count.
std::array<std::uint32_t, 3> launch_grid(std::uint32_t count) {
  return {1 + (count - 1) / block_size, 1, 1};
}

} // namespace

CudaHalfMatmul::CudaHalfMatmul(const compute::Context& context) : context_(context) {
  if (context_.device().backend != compute::Backend::CUDA)
    throw std::invalid_argument("CudaHalfMatmul: a CUDA context is required");
  pack_ = context_.program({"pack_half", {}, pack_ptx, 2, 4, {block_size, 1, 1}});
  product_ = context_.program({"half_product", {}, product_ptx, 3, 16, {block_size, 1, 1}});
}

Tensor CudaHalfMatmul::operator()(const Tensor& a, const Tensor& b, bool transpose_a) const {
  constexpr auto limit = std::numeric_limits<std::uint32_t>::max();
  /// @brief Check host matrix geometry and the flat CUDA indexing range.
  const auto valid = [=](const Tensor& t) {
    return t.defined() && t.dim() == 2 && t.numel() > 0 && t.numel() <= limit &&
           t.dtype() == DType::Float32 && t.device().backend == compute::Backend::CPU;
  };
  if (!valid(a) || !valid(b) || a.size(transpose_a ? 0 : 1) != b.size(0))
    throw std::invalid_argument("CudaHalfMatmul: expected compatible CPU Float32 matrices");
  const auto rows = a.size(transpose_a ? 1 : 0);
  const auto columns = b.size(1);
  if (rows > limit / columns)
    throw std::invalid_argument("CudaHalfMatmul: output exceeds CUDA flat indexing range");
  const auto count = static_cast<std::uint32_t>(rows * columns);

  /// @brief Upload host Float32 and convert it into a packed device operand.
  const auto pack = [&](const Tensor& value) {
    const auto size = static_cast<std::uint32_t>(value.numel());
    auto source = context_.allocate(std::size_t(size) * sizeof(float));
    auto dest = context_.allocate(std::size_t(size) * sizeof(std::uint16_t));
    const auto contiguous = value.contiguous();
    context_.upload(source, contiguous.data_ptr<float>(), source.size());
    context_.dispatch(pack_, {source, dest}, &size, sizeof(size), launch_grid(size));
    return dest;
  };
  const auto left = pack(a);
  const auto right = pack(b);
  auto device_output = context_.allocate(std::size_t(count) * sizeof(float));
  const std::array args{static_cast<std::uint32_t>(rows), static_cast<std::uint32_t>(columns),
                        static_cast<std::uint32_t>(b.size(0)), std::uint32_t(transpose_a)};
  context_.dispatch(product_, {left, right, device_output}, args.data(), sizeof(args), launch_grid(count));
  auto result = empty({rows, columns}, DType::Float32);
  context_.download(result.data_ptr<float>(), device_output, device_output.size());
  return result;
}

} // namespace datamunge::nn
