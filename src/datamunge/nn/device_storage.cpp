/// @file device_storage.cpp
/// @brief Native CUDA/Vulkan bit-preserving tensor packing and scalar-pattern fills.
#include "device_storage.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>

#include "storage_shader.hpp"
namespace datamunge::nn::detail {
/// @brief One retained native pipeline; allocations on its context can share it.
struct StorageProgram {
  compute::Context context; ///< Retains the creating execution owner.
  compute::Program program; ///< Native storage pipeline.
};
/// @brief Private portable storage launch layout and pipeline acquisition.
namespace {
/// @brief Packed uint32 push constants/by-value CUDA arguments, identical to storage.comp.
struct Parameters {
  std::uint32_t bytes;       ///< Logical output byte count, excluding word padding.
  std::uint32_t width;       ///< Scalar byte width, one/four/eight.
  std::uint32_t rank;        ///< Number of shape/stride pairs in the metadata binding.
  std::uint32_t base;        ///< Source element offset, modulo 2^32 for valid bounded views.
  std::uint32_t mode;        ///< Zero packs a view; one repeats a scalar pattern.
  std::uint32_t low;         ///< Pattern's first four little-endian bytes.
  std::uint32_t high;        ///< Pattern's optional remaining four bytes.
  std::uint32_t reserved{0}; ///< Zero ABI padding.
};
/// @brief CUDA Driver PTX equivalent of storage.comp, using exact byte loads and word writes.
/// No floating arithmetic or device int64 feature is needed; 64-bit registers address memory.
constexpr const char* storage_ptx = R"ptx(.version 6.0
.target sm_50
.address_size 64
.visible .entry main(.param .u64 input,.param .u64 output,.param .u64 layout,.param .align 4 .b8 args[32]) {
.reg .pred p<4>;
.reg .b32 r<36>;
.reg .b64 d<8>;
ld.param.u64 d0,[input];ld.param.u64 d1,[output];ld.param.u64 d2,[layout];
ld.param.u32 r0,[args];ld.param.u32 r1,[args+4];ld.param.u32 r2,[args+8];
ld.param.u32 r3,[args+12];ld.param.u32 r4,[args+16];ld.param.u32 r5,[args+20];ld.param.u32 r6,[args+24];
mov.u32 r24,%ctaid.y;mov.u32 r25,%nctaid.x;mov.u32 r26,%ctaid.x;
mad.lo.u32 r24,r24,r25,r26;mov.u32 r25,%ntid.x;mov.u32 r26,%tid.x;
mad.lo.u32 r7,r24,r25,r26;add.u32 r20,r0,3;shr.u32 r20,r20,2;
setp.ge.u32 p0,r7,r20;@p0 bra done;
shl.b32 r8,r7,2;mov.u32 r9,0;mov.u32 r11,0;
byte_loop:
add.u32 r10,r8,r9;setp.ge.u32 p0,r10,r0;@p0 bra store;
rem.u32 r13,r10,r1;setp.eq.u32 p1,r4,1;@p1 bra fill;
div.u32 r14,r10,r1;mov.u32 r15,r2;mov.u32 r19,r3;
dimension_loop:
setp.eq.u32 p0,r15,0;@p0 bra load;
sub.u32 r15,r15,1;mul.wide.u32 d3,r15,8;add.u64 d3,d2,d3;
ld.global.u32 r16,[d3];ld.global.u32 r17,[d3+4];
rem.u32 r18,r14,r16;div.u32 r14,r14,r16;mad.lo.u32 r19,r18,r17,r19;bra dimension_loop;
load:
mad.lo.u32 r21,r19,r1,r13;cvt.u64.u32 d3,r21;add.u64 d3,d0,d3;
ld.global.u8 r12,[d3];bra combine;
fill:
setp.lt.u32 p2,r13,4;selp.u32 r12,r5,r6,p2;and.b32 r22,r13,3;shl.b32 r22,r22,3;
shr.u32 r12,r12,r22;and.b32 r12,r12,255;
combine:
shl.b32 r23,r9,3;shl.b32 r12,r12,r23;or.b32 r11,r11,r12;
add.u32 r9,r9,1;setp.lt.u32 p0,r9,4;@p0 bra byte_loop;
store:
mul.wide.u32 d4,r7,4;add.u64 d4,d1,d4;st.global.u32 [d4],r11;
done: ret;
})ptx";
/// @brief Acquire a lazily compiled program, retaining no global strong context ownership.
/// A one-entry thread-local weak cache shares the pipeline across simultaneously live allocations.
/// @param storage Destination storage on the selected native context.
/// @return Pipeline retained by that storage, with no allocation ownership cycle.
StorageProgram& program(Storage& storage) {
  if (!storage.program) {
    /// @brief Weak last-context pipeline cache; never keeps a driver/context alive on its own.
    static thread_local std::weak_ptr<StorageProgram> cached;
    auto                                              existing = cached.lock();
    if (existing && existing->context.shares_with(storage.context))
      storage.program = std::move(existing);
    else {
      compute::KernelSource source;
      source.buffers         = 3;
      source.parameter_bytes = sizeof(Parameters);
      source.local_size      = {64, 1, 1};
      source.ptx             = storage_ptx;
      source.spirv.assign(std::begin(storage_spirv), std::end(storage_spirv));
      storage.program = std::make_shared<StorageProgram>(
          StorageProgram{storage.context, storage.context.program(source)});
      cached = storage.program;
    }
  }
  return *storage.program;
}
/// @brief Submit a portable two-dimensional grid, with at most 65535 groups per axis.
/// @param out Newly allocated packed output.
/// @param input Input or dummy buffer from the same context.
/// @param layout Shape/stride metadata or an unused buffer for fills.
/// @param args Exact 32-byte native parameter layout.
void launch(
    Tensor&                out,
    const compute::Buffer& input,
    const compute::Buffer& layout,
    const Parameters&      args) {
  if (!args.bytes)
    return;
  auto&      storage  = *out.impl()->storage;
  auto&      pipeline = program(storage);
  const auto count    = ((args.bytes + 3) / 4 + 63) / 64;
  const auto x        = std::min(count, std::uint32_t(65535));
  storage.context.dispatch(
      pipeline.program,
      {input, storage.bytes, layout},
      &args,
      sizeof(args),
      {x, (count + x - 1) / x, 1});
}
} // namespace
const compute::Context& default_tensor_context() {
  /// @brief Shared single-lane CPU execution owner; buffers retain it after static teardown.
  static const compute::Context context;
  return context;
}
Tensor pack_device(const Tensor& source) {
  Tensor out                   = empty(source.shape(), source.dtype(), source.context());
  out.impl()->storage->program = std::atomic_load(&source.impl()->storage->program);
  if (!out.numel())
    return out;
  std::vector<std::uint32_t> dimensions;
  dimensions.reserve(std::size_t(source.dim()) * 2);
  for (std::size_t i = 0; i < source.shape().size(); ++i) {
    dimensions.push_back(static_cast<std::uint32_t>(source.shape()[i]));
    dimensions.push_back(static_cast<std::uint32_t>(source.strides()[i]));
  }
  auto layout = source.context().allocate(std::max(std::size_t(4), dimensions.size() * 4));
  source.context().upload(layout, dimensions.data(), dimensions.size() * 4);
  Parameters args{
      static_cast<std::uint32_t>(out.impl()->storage->nbytes),
      static_cast<std::uint32_t>(dtype_size(source.dtype())),
      static_cast<std::uint32_t>(source.dim()),
      static_cast<std::uint32_t>(source.offset()),
      0,
      0,
      0};
  launch(out, source.impl()->storage->bytes, layout, args);
  std::atomic_store(&source.impl()->storage->program, out.impl()->storage->program);
  return out;
}
void fill_device(Tensor& destination, const void* pattern, std::size_t width) {
  const std::uint16_t endian = 1;
  if (*reinterpret_cast<const unsigned char*>(&endian) != 1)
    throw std::runtime_error("native tensor storage requires a little-endian GPU host");
  std::array<unsigned char, 8> bytes{};
  std::memcpy(bytes.data(), pattern, width);
  Parameters args{
      static_cast<std::uint32_t>(destination.impl()->storage->nbytes),
      static_cast<std::uint32_t>(width),
      0,
      0,
      1,
      0,
      0};
  std::memcpy(&args.low, bytes.data(), 4);
  std::memcpy(&args.high, bytes.data() + 4, 4);
  const auto& buffer = destination.impl()->storage->bytes;
  launch(destination, buffer, buffer, args);
}
} // namespace datamunge::nn::detail
