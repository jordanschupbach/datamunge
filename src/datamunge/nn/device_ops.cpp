/// @file device_ops.cpp
/// @brief Native CUDA forward kernels for nn inference (see shaders/nn/inference.cu).
#include "device_ops.hpp"

#include <datamunge/nn/autograd.hpp>
#include <datamunge/nn/dtype.hpp>

#include <algorithm>
#include <atomic>
#include <stdexcept>
#include <string>
#include <vector>

#include "device_kernels_ptx.hpp"
#include "device_storage.hpp"
namespace datamunge::nn::detail {
namespace {
/// @brief Byte layouts identical to the by-value blocks in shaders/nn/inference.cu.
struct Elementwise {
  std::uint64_t count, period;
};
struct Patches {
  std::uint64_t count;
  std::int64_t  h, w, c, kh, kw, oh, ow, sh, sw, dh, dw, pt, pl;
};
struct Normalize {
  std::uint64_t count, channels;
  double        eps;
};
struct Pool {
  std::uint64_t count, pixels, channels;
};
struct Block {
  std::uint64_t count, row, out_row, offset;
};
struct Padding {
  std::uint64_t count;
  std::int64_t  out[4], in[4], before[4];
  double        value;
};
struct MaxPool {
  std::uint64_t count;
  std::int64_t  h, w, c, kh, kw, oh, ow, sh, sw, pt, pl;
};
static_assert(sizeof(Elementwise) == 16 && sizeof(Patches) == 112 && sizeof(Normalize) == 24 &&
              sizeof(Pool) == 24 && sizeof(Block) == 32 && sizeof(Padding) == 112 &&
              sizeof(MaxPool) == 96);

constexpr std::uint32_t threads = 256; ///< Block width shared by every inference entry.
/// Patch matrices are built a few images at a time so their buffer stays below this size.
constexpr std::size_t patch_budget = std::size_t(256) << 20;

/// @brief Validate that every operand is a CUDA float tensor on one context, without gradients.
/// @return The shared dtype.
DType check_operands(const char* op, const std::vector<const Tensor*>& operands) {
  const Tensor& first = *operands.front();
  if (first.device().backend != compute::Backend::CUDA)
    throw std::logic_error(
        std::string(op) + ": native numerical kernels are implemented for CUDA only");
  const DType dt = first.dtype();
  if (dt != DType::Float32 && dt != DType::Float64)
    throw std::logic_error(std::string(op) + ": CUDA kernels need Float32 or Float64 input");
  const bool tracking = is_grad_enabled();
  for (const Tensor* t : operands) {
    if (!t->defined() || !t->context().shares_with(first.context()))
      throw std::invalid_argument(
          std::string(op) + ": all operands must live on the same CUDA context");
    if (t->dtype() != dt)
      throw std::invalid_argument(std::string(op) + ": CUDA operands must share one dtype");
    if (tracking && t->requires_grad())
      throw std::logic_error(
          std::string(op) +
          ": GPU backward is not implemented; run GPU inference under NoGradGuard");
  }
  return dt;
}

/// @brief Byte view of a packed tensor's logical elements.
compute::Buffer bytes(const Tensor& packed) {
  const auto width = dtype_size(packed.dtype());
  return packed.impl()->storage->bytes.slice(
      std::size_t(packed.offset()) * width, std::size_t(packed.numel()) * width);
}

/// @brief Compiled entry `<name>_f32|f64`, cached on the output's retained pipeline.
/// Inputs adopt that pipeline too, so long-lived tensors (parameters) keep it compiled.
compute::Program& kernel(
    Tensor&                            out,
    const std::vector<const Tensor*>&  inputs,
    const std::string&                 name,
    std::uint32_t                      buffers,
    std::uint32_t                      parameter_bytes) {
  auto&      storage = *out.impl()->storage;
  auto&      owner   = storage_program(storage);
  const auto shared  = storage.program;
  for (const Tensor* t : inputs)
    if (!std::atomic_load(&t->impl()->storage->program))
      std::atomic_store(&t->impl()->storage->program, shared);
  const std::string entry = name + (out.dtype() == DType::Float32 ? "_f32" : "_f64");
  std::lock_guard   guard(owner.lock);
  auto              found = owner.kernels.find(entry);
  if (found == owner.kernels.end()) {
    compute::KernelSource source;
    source.entry           = entry;
    source.ptx             = inference_ptx;
    source.buffers         = buffers;
    source.parameter_bytes = parameter_bytes;
    source.local_size      = {threads, 1, 1};
    found = owner.kernels.emplace(entry, owner.context.program(source)).first;
  }
  return found->second;
}

/// @brief Launch one thread per element over a 2-D grid of at most 65535 blocks per axis.
template <class Parameters>
void launch(
    Tensor&                            out,
    const std::vector<const Tensor*>&  inputs,
    const std::string&                 name,
    std::vector<compute::Buffer>       buffers,
    const Parameters&                  parameters,
    std::uint64_t                      count) {
  if (!count)
    return;
  auto&               program = kernel(
      out, inputs, name, std::uint32_t(buffers.size()), std::uint32_t(sizeof(Parameters)));
  const std::uint64_t blocks = (count + threads - 1) / threads;
  const std::uint64_t x      = std::min<std::uint64_t>(blocks, 65535);
  const std::uint64_t y      = (blocks + x - 1) / x;
  if (y > 65535)
    throw std::length_error(name + ": tensor too large for one native launch");
  out.context().dispatch(
      program,
      buffers,
      &parameters,
      sizeof(Parameters),
      {std::uint32_t(x), std::uint32_t(y), 1});
}
} // namespace

bool is_native(const Tensor& tensor) {
  return tensor.defined() && tensor.device().backend != compute::Backend::CPU;
}

Tensor device_add(const Tensor& a, const Tensor& b) {
  const DType dt = check_operands("add", {&a, &b});
  // Floating addition commutes exactly, so put the broadcast operand second.
  const bool    swap  = a.numel() < b.numel();
  const Tensor& big   = swap ? b : a;
  const Tensor& small = swap ? a : b;
  Shape         tail  = small.shape();
  while (!tail.empty() && tail.front() == 1)
    tail.erase(tail.begin());
  const Shape& full = big.shape();
  if (tail.size() > full.size() || !std::equal(tail.begin(), tail.end(), full.end() - tail.size()) ||
      (small.dim() > big.dim()))
    throw std::logic_error(
        "add: CUDA supports equal shapes or broadcasting over trailing axes, got " +
        shape_str(a.shape()) + " and " + shape_str(b.shape()));
  const Tensor x = big.contiguous(), y = small.contiguous();
  Tensor       out = empty(full, dt, big.context());
  const auto   n   = std::uint64_t(out.numel());
  launch(
      out, {&x, &y}, "add", {bytes(x), bytes(y), out.impl()->storage->bytes},
      Elementwise{n, std::max<std::uint64_t>(1, std::uint64_t(y.numel()))}, n);
  return out;
}

Tensor device_relu(const Tensor& input) {
  const DType  dt  = check_operands("relu", {&input});
  const Tensor x   = input.contiguous();
  Tensor       out = empty(x.shape(), dt, x.context());
  const auto   n   = std::uint64_t(out.numel());
  launch(out, {&x}, "relu", {bytes(x), out.impl()->storage->bytes}, Elementwise{n, n}, n);
  return out;
}

Tensor device_conv2d(const Tensor& input, const Tensor& weights, const DeviceConvGeometry& g) {
  const DType dt = check_operands("conv2d", {&input, &weights});
  if (g.groups != 1)
    throw std::logic_error("conv2d: CUDA path supports ungrouped convolution only");
  if (!input.context().supports_matmul())
    throw std::logic_error("conv2d: this device has no GEMM (Context::matmul)");
  const Tensor x = input.contiguous(), f = weights.contiguous();
  Tensor       out = empty({g.n, g.oh, g.ow, g.o}, dt, x.context());
  if (!out.numel())
    return out;
  const std::int64_t k = g.kh * g.kw * g.c, rows = g.oh * g.ow;
  const auto         width = dtype_size(dt);
  const bool         pointwise =
      g.kh == 1 && g.kw == 1 && g.sh == 1 && g.sw == 1 && g.pt == 0 && g.pl == 0 &&
      g.oh == g.h && g.ow == g.w;
  const std::int64_t per_image = rows * k * std::int64_t(width);
  const std::int64_t chunk =
      pointwise ? g.n : std::clamp<std::int64_t>(std::int64_t(patch_budget) / per_image, 1, g.n);
  Tensor patches = pointwise ? Tensor() : empty({chunk * rows, k}, dt, x.context());
  const auto filter = bytes(f);
  auto       target = out.impl()->storage->bytes;
  for (std::int64_t first = 0; first < g.n; first += chunk) {
    const std::int64_t images = std::min(chunk, g.n - first);
    const auto image_bytes = std::size_t(g.h * g.w * g.c) * width;
    auto       source      = bytes(x).slice(std::size_t(first) * image_bytes,
                                 std::size_t(images) * image_bytes);
    if (!pointwise) {
      const auto count = std::uint64_t(images * rows * k);
      launch(
          patches, {&x, &f}, "im2col", {source, patches.impl()->storage->bytes},
          Patches{count, g.h, g.w, g.c, g.kh, g.kw, g.oh, g.ow, g.sh, g.sw, g.dh, g.dw, g.pt, g.pl},
          count);
      source = patches.impl()->storage->bytes.slice(0, std::size_t(images * rows * k) * width);
    }
    compute::MatrixProduct product{
        std::size_t(images * rows),
        std::size_t(g.o),
        std::size_t(k),
        dt == DType::Float32 ? compute::MatrixType::Float32 : compute::MatrixType::Float64};
    auto destination = target.slice(
        std::size_t(first * rows * g.o) * width, std::size_t(images * rows * g.o) * width);
    x.context().matmul(destination, source, filter, product);
  }
  return out;
}

Tensor device_batch_norm_infer(
    const Tensor& input,
    const Tensor& gamma_in,
    const Tensor& beta_in,
    const Tensor& mean_in,
    const Tensor& variance_in,
    double        eps) {
  const DType dt =
      check_operands("batch_norm", {&input, &gamma_in, &beta_in, &mean_in, &variance_in});
  const std::int64_t c = input.dim() ? input.shape().back() : 1;
  for (const Tensor* t : {&gamma_in, &beta_in, &mean_in, &variance_in})
    if (t->numel() != c)
      throw std::invalid_argument("batch_norm: CUDA statistics must have one value per channel");
  const Tensor x = input.contiguous(), gamma = gamma_in.contiguous(), beta = beta_in.contiguous(),
               mean = mean_in.contiguous(), variance = variance_in.contiguous();
  Tensor     out = empty(x.shape(), dt, x.context());
  const auto n   = std::uint64_t(out.numel());
  launch(
      out, {&x, &gamma, &beta, &mean, &variance}, "batch_norm",
      {bytes(x), bytes(gamma), bytes(beta), bytes(mean), bytes(variance),
       out.impl()->storage->bytes},
      Normalize{n, std::uint64_t(std::max<std::int64_t>(c, 1)), eps}, n);
  return out;
}

Tensor device_global_avg_pool2d(const Tensor& input, bool keepdims) {
  const DType  dt = check_operands("global_avg_pool2d", {&input});
  const Tensor x  = input.contiguous();
  const auto&  s  = x.shape();
  Tensor out = empty(keepdims ? Shape{s[0], 1, 1, s[3]} : Shape{s[0], s[3]}, dt, x.context());
  const auto n = std::uint64_t(out.numel());
  launch(
      out, {&x}, "global_average", {bytes(x), out.impl()->storage->bytes},
      Pool{n, std::uint64_t(s[1] * s[2]), std::uint64_t(s[3])}, n);
  return out;
}

Tensor device_concat(const std::vector<Tensor>& tensors, std::int64_t axis, const Shape& shape) {
  std::vector<const Tensor*> operands;
  for (const auto& t : tensors)
    operands.push_back(&t);
  const DType dt  = check_operands("concat", operands);
  Tensor      out = empty(shape, dt, tensors.front().context());
  // View every tensor as [outer, axis * inner]; each input fills a column band of each row.
  std::int64_t inner = 1, outer = 1;
  for (std::size_t d = 0; d < shape.size(); ++d) {
    if (std::int64_t(d) < axis)
      outer *= shape[d];
    else if (std::int64_t(d) > axis)
      inner *= shape[d];
  }
  const auto out_row = std::uint64_t(shape[std::size_t(axis)] * inner);
  if (!outer || !out_row)
    return out;
  std::uint64_t offset = 0;
  for (const auto& t : tensors) {
    const Tensor x   = t.contiguous();
    const auto   row = std::uint64_t(x.shape()[std::size_t(axis)] * inner);
    if (row)
      launch(
          out, {&x}, "copy_block", {bytes(x), out.impl()->storage->bytes},
          Block{row * std::uint64_t(outer), row, out_row, offset}, row * std::uint64_t(outer));
    offset += row;
  }
  return out;
}

Tensor device_pad_constant(
    const Tensor& input, const std::vector<std::pair<std::int64_t, std::int64_t>>& paddings,
    double value) {
  const DType dt = check_operands("pad", {&input});
  if (input.dim() > 4)
    throw std::logic_error("pad: CUDA constant padding supports rank <= 4");
  const Tensor x = input.contiguous();
  Padding      p{};
  Shape        shape = x.shape();
  const auto   lead  = 4 - x.dim();
  for (int d = 0; d < 4; ++d) {
    const bool real = d >= lead;
    p.in[d]         = real ? x.shape()[std::size_t(d - lead)] : 1;
    p.before[d]     = real ? paddings[std::size_t(d - lead)].first : 0;
    p.out[d]        = p.in[d] + p.before[d] + (real ? paddings[std::size_t(d - lead)].second : 0);
    if (real)
      shape[std::size_t(d - lead)] = p.out[d];
  }
  Tensor out = empty(shape, dt, x.context());
  p.count    = std::uint64_t(out.numel());
  p.value    = value;
  launch(out, {&x}, "pad_constant", {bytes(x), out.impl()->storage->bytes}, p, p.count);
  return out;
}

Tensor device_max_pool2d(const Tensor& input, const DevicePoolGeometry& g) {
  const DType  dt  = check_operands("max_pool2d", {&input});
  const Tensor x   = input.contiguous();
  Tensor       out = empty({g.n, g.oh, g.ow, g.c}, dt, x.context());
  const auto   n   = std::uint64_t(out.numel());
  launch(
      out, {&x}, "max_pool", {bytes(x), out.impl()->storage->bytes},
      MaxPool{n, g.h, g.w, g.c, g.kh, g.kw, g.oh, g.ow, g.sh, g.sw, g.pt, g.pl}, n);
  return out;
}
} // namespace datamunge::nn::detail
