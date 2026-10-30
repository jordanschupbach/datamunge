#pragma once

/// @file context.hpp
/// @brief Explicit compute devices, owning byte buffers and persistent CPU workers.

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

/// @brief Explicit execution contexts and byte buffers for numerical libraries.
namespace datamunge::compute {

/// @brief Physical execution backend; a GPU selection never falls back to CPU.
/// @par Example
/// @code
/// const auto cpu = dm::compute::Backend::CPU;
/// std::print("{}\n", dm::compute::backend_name(cpu)); // cpu
/// @endcode
enum class Backend {
  CPU,   ///< Host memory and native CPU execution.
  CUDA,  ///< NVIDIA device memory and CUDA execution.
  Vulkan ///< Vulkan storage buffers and compute execution.
};

/// @brief Return a stable lowercase backend identifier.
/// @param backend Backend enum value.
/// @return cpu, cuda or vulkan.
/// @throws std::invalid_argument Unknown enum value.
/// @par Example
/// @code
/// std::print("{}\n", dm::compute::backend_name(dm::compute::Backend::CPU)); // cpu
/// @endcode
std::string backend_name(Backend backend);

/// @brief An explicit backend and ordinal, with no hidden current-device state.
/// CPU accepts index zero only; GPU indices refer to backend enumeration order.
/// @par Example
/// @code
/// const dm::compute::Device device;
/// std::print("{}\n", device.index); // 0
/// @endcode
struct Device {
  Backend       backend{Backend::CPU}; ///< Selected backend; CPU by default.
  std::uint32_t index{0};              ///< Zero-based device index.
};

/// @brief Scalar precision for packed native matrix products.
/// @par Example
/// @code
/// const auto precision = dm::compute::MatrixType::Float32;
/// @endcode
enum class MatrixType {
  Float32, ///< IEEE binary32 inputs, accumulation and output; TF32 is disabled.
  Float64  ///< IEEE binary64 inputs, accumulation and output.
};

/// @brief Packed row-major C[m,n] = op(A)[m,k] op(B)[k,n], with no host evaluation.
/// Stored A is [m,k] or [k,m]; stored B is [k,n] or [n,k], according to the
/// transpose flags. Each dimension must fit a signed 32-bit cuBLAS dimension.
/// Zero output dimensions do no work; zero inner dimension writes exact zeros.
/// @par Example
/// @code
/// dm::compute::MatrixProduct product;
/// product.rows=2; product.columns=4; product.inner=3;
/// @endcode
/// @par References
/// - NVIDIA cuBLAS: https://docs.nvidia.com/cuda/archive/12.8.0/cublas/index.html
/// @see Context::matmul()
struct MatrixProduct {
  std::size_t rows{0};                   ///< Logical output rows m, default zero.
  std::size_t columns{0};                ///< Logical output columns n, default zero.
  std::size_t inner{0};                  ///< Reduction dimension k, default zero.
  MatrixType  type{MatrixType::Float32}; ///< Input/output precision, default Float32.
  bool        transpose_a{false};        ///< Stored A is [k,m] when true; default false.
  bool        transpose_b{false};        ///< Stored B is [n,k] when true; default false.
};

/// @brief Describe one backend without treating missing support as an available GPU.
/// @par Example
/// @code
/// const auto info = dm::compute::probe(dm::compute::Device{});
/// std::print("{}\n", info.available); // true
/// @endcode
struct DeviceInfo {
  Device      device;           ///< Requested device identifier.
  bool        available{false}; ///< Whether a context can be constructed.
  std::string name;             ///< Device name when available.
  std::string diagnostic;       ///< Nonempty explanation when unavailable.
};

/// @brief Probe device availability; never creates a fallback context.
/// @param device Requested backend/index, CPU zero by default.
/// @return Device description with explicit unavailable diagnostics.
/// @throws std::invalid_argument Unknown backend value.
/// @par Example
/// @code
/// std::print("{}\n", dm::compute::probe().available); // true
/// @endcode
DeviceInfo probe(Device device = {});

/// @brief Private shared compute ownership and backend resources.
namespace detail {
/// @brief Private shared context state retained by buffers and copied contexts.
struct ContextState;
/// @brief Private allocation owner; views retain the same instance.
struct Allocation;
/// @brief Private backend-owned native program and its creating context.
struct ProgramState;
} // namespace detail

/// @brief Shared allocation or bounded byte view retaining its creating context.
/// Copies alias bytes. A view survives destruction of the original handle.
/// Host pointers are exposed only for CPU allocations. Concurrent writes to
/// overlapping bytes need caller synchronization. A default handle is undefined.
/// @par Example
/// @code
/// const dm::compute::Buffer buffer;
/// std::print("{}\n", buffer.defined()); // false
/// @endcode
class Buffer {
 public:
  /// @brief Construct an undefined buffer handle without allocating.
  /// @par Example
  /// @code
  /// const dm::compute::Buffer buffer;
  /// std::print("{}\n", buffer.size()); // 0
  /// @endcode
  Buffer() = default;
  /// @brief Test whether this handle owns or views an allocation.
  /// @return True even for a defined zero-byte allocation.
  bool defined() const noexcept { return allocation_ != nullptr; }
  /// @brief Logical byte length of this view.
  /// @return Zero for an undefined or empty buffer.
  std::size_t size() const noexcept { return size_; }
  /// @brief Return the allocation's explicit execution device.
  /// @return Backend and ordinal.
  /// @throws std::logic_error Undefined buffer.
  Device device() const;
  /// @brief Create an alias bounded within this view.
  /// @param offset Byte offset within the current view.
  /// @param size Number of bytes to retain, including zero.
  /// @return Owning view sharing allocation and context lifetime.
  /// @throws std::out_of_range Requested range exceeds this view.
  /// @throws std::logic_error Undefined buffer.
  /// @par Example
  /// @code
  /// const dm::compute::Context context;
  /// const auto view = context.allocate(16).slice(4, 8);
  /// std::print("{}\n", view.size()); // 8
  /// @endcode
  Buffer slice(std::size_t offset, std::size_t size) const;
  /// @brief Borrow this CPU view's first byte, aligned only when its offset is aligned.
  /// @return Host pointer valid while an allocation handle survives.
  /// @throws std::logic_error Undefined or non-CPU buffer.
  /// @par Example
  /// @code
  /// const dm::compute::Context context;
  /// auto bytes = context.allocate(1);
  /// bytes.host_data()[0] = 7;
  /// std::print("{}\n", int(bytes.host_data()[0])); // 7
  /// @endcode
  std::uint8_t* host_data();
  /// @brief Borrow this CPU view's bytes read-only.
  /// @return Pointer with the same lifetime/alignment contract as host_data().
  /// @throws std::logic_error Undefined or non-CPU buffer.
  /// @see host_data()
  const std::uint8_t* host_data() const;

 private:
  /// @brief Creating context validates and accesses allocation/program ownership.
  friend class Context;
  std::shared_ptr<detail::Allocation> allocation_; ///< Shared resource owner.
  std::size_t                         offset_{0};  ///< First byte in the underlying allocation.
  std::size_t                         size_{0};    ///< Bytes accessible through this view.
};

/// @brief Native program source and its explicit, backend-neutral argument layout.
/// Vulkan takes SPIR-V words; CUDA takes null-terminated PTX text. Programs have
/// buffer parameters first, followed by one by-value parameter block when nonempty.
/// Vulkan uses descriptor set zero with sequential storage-buffer bindings and a
/// push-constant block. CUDA uses the equivalent pointer parameters and packed block.
/// @par Example
/// @code
/// dm::compute::KernelSource source;
/// std::print("{}\n",source.entry); // main
/// @endcode
struct KernelSource {
  std::string                entry{"main"};      ///< Native entry-point name.
  std::vector<std::uint32_t> spirv;              ///< Vulkan binary words; empty for CUDA.
  std::string                ptx;                ///< CUDA PTX text; empty for Vulkan.
  std::uint32_t              buffers{0};         ///< Number of ordered buffer bindings/parameters.
  std::uint32_t              parameter_bytes{0}; ///< Exact push-constant/by-value-block byte count.
  std::array<std::uint32_t, 3> local_size{
      64,
      1,
      1}; ///< CUDA block dimensions; must match Vulkan shader local size.
};
/// @brief Shared native GPU kernel retaining its context and compiled resources.
/// Default construction is undefined; context.program compiles a backend program.
/// @par Example
/// @code
/// const dm::compute::Program program;
/// std::print("{}\n",program.defined()); // false
/// @endcode
class Program {
 public:
  /// @brief Construct an undefined native program handle.
  /// @par Example
  /// @code
  /// const dm::compute::Program program;
  /// @endcode
  Program() = default;
  /// @brief Test whether native compilation succeeded for this handle.
  /// @return Whether a backend program is retained.
  bool defined() const noexcept { return state_ != nullptr; }

 private:
  /// @brief Creating context validates and accesses allocation/program ownership.
  friend class Context;
  std::shared_ptr<detail::ProgramState> state_; ///< Shared native program/context ownership.
};
/// @brief Execution context with persistent workers and shared, bounded allocations.
/// Context copies share workers and allocations. CPU and current GPU commands are synchronous;
/// parallel_for serializes submissions on one context and propagates callback
/// exceptions after all workers finish. Separate contexts can run concurrently.
/// No global thread count or current-device setting is changed.
/// @par Example
/// @code
/// const dm::compute::Context context({}, 2);
/// std::vector<int> values(4);
/// context.parallel_for(4, [&](std::size_t i) { values[i] = int(i*i); });
/// std::print("{}\n", values[3]); // 9
/// @endcode
class Context {
 public:
  /// @brief Create an explicit context; unavailable backends fail before allocation.
  /// @param device Backend/index; default CPU zero.
  /// @param workers CPU execution lanes including the caller; default one, must be positive.
  /// @throws std::invalid_argument Invalid device or zero workers.
  /// @throws std::runtime_error Requested backend unavailable.
  /// @par Example
  /// @code
  /// const dm::compute::Context context({}, 2);
  /// std::print("{}\n", context.workers()); // 2
  /// @endcode
  explicit Context(Device device = {}, std::size_t workers = 1);
  /// @brief Return the context's backend/index.
  /// @return Selected device, unchanged for this context's lifetime.
  Device device() const noexcept;
  /// @brief Test whether another handle shares this context's execution and allocation ownership.
  /// @param other Context to compare without changing either owner.
  /// @return True for copies of one context, including CPU contexts.
  bool shares_with(const Context& other) const noexcept;
  /// @brief Number of CPU execution lanes including the submitting thread.
  /// @return Positive lane count.
  std::size_t workers() const noexcept;
  /// @brief Allocate owned bytes; size zero still yields a defined allocation.
  /// @param bytes Logical allocation length.
  /// @return Shared buffer retaining its context; initial bytes are uninitialized.
  /// @throws std::bad_alloc Allocation failure.
  /// @par Example
  /// @code
  /// const dm::compute::Context context;
  /// const auto bytes = context.allocate(16);
  /// std::print("{}\n", bytes.size()); // 16
  /// @endcode
  Buffer allocate(std::size_t bytes) const;
  /// @brief Copy bytes between buffers created by this context; overlap is supported on CPU.
  /// GPU overlap requires a temporary buffer; identical views are a no-op. Vulkan
  /// copy offsets and byte counts must be multiples of four.
  /// @param destination Destination view with at least bytes accessible bytes.
  /// @param source Source view with at least bytes accessible bytes.
  /// @param bytes Number of bytes to copy; zero is valid.
  /// @throws std::invalid_argument Undefined/foreign buffers, overlapping GPU ranges or unaligned
  /// Vulkan copy.
  /// @throws std::out_of_range Byte count exceeds either view.
  /// @par Example
  /// @code
  /// const dm::compute::Context context;
  /// auto a=context.allocate(1), b=context.allocate(1);
  /// a.host_data()[0]=3; context.copy(b,a,1);
  /// std::print("{}\n",int(b.host_data()[0])); // 3
  /// @endcode
  void copy(Buffer& destination, const Buffer& source, std::size_t bytes) const;
  /// @brief Fill one owned buffer view with an exact byte value.
  /// @param buffer Destination created by this context.
  /// @param value Byte value, zero by default.
  /// @throws std::invalid_argument Undefined buffer or another context's allocation.
  /// @par Example
  /// @code
  /// const dm::compute::Context context;
  /// auto bytes=context.allocate(1); context.fill(bytes,42);
  /// std::print("{}\n",int(bytes.host_data()[0])); // 42
  /// @endcode
  void fill(Buffer& buffer, std::uint8_t value = 0) const;
  /// @brief Explicitly copy host bytes into one owned buffer view.
  /// @param destination Destination allocated by this context.
  /// @param source Host memory valid for bytes bytes; may be null only when bytes is zero.
  /// @param bytes Number of bytes, within the destination view.
  /// @throws std::invalid_argument Foreign/undefined destination or null source.
  /// @throws std::out_of_range Transfer exceeds the view.
  /// @throws std::runtime_error Native GPU error.
  /// @par Example
  /// @code
  /// const dm::compute::Context context;auto buffer=context.allocate(4);
  /// const int value=42;context.upload(buffer,&value,sizeof(value));
  /// @endcode
  void upload(Buffer& destination, const void* source, std::size_t bytes) const;
  /// @brief Explicitly copy device/host buffer bytes into caller-owned host memory.
  /// @param destination Host destination, nonnull when bytes is positive.
  /// @param source Buffer allocated by this context.
  /// @param bytes Number of bytes, within the source view.
  /// @throws std::invalid_argument Foreign/undefined source or null destination.
  /// @throws std::out_of_range Transfer exceeds the view.
  /// @throws std::runtime_error Native GPU error.
  /// @par Example
  /// @code
  /// const dm::compute::Context context;auto buffer=context.allocate(1);context.fill(buffer,7);
  /// unsigned char value=0;context.download(&value,buffer,1);
  /// std::print("{}\n",int(value)); // 7
  /// @endcode
  void download(void* destination, const Buffer& source, std::size_t bytes) const;
  /// @brief Test whether this build/context provides the native matrix adapter.
  /// @return True for CUDA with DATAMUNGE_ENABLE_CUBLAS; false for CPU/Vulkan.
  /// No device allocation or numerical computation is performed.
  bool supports_matmul() const noexcept;
  /// @brief Synchronously overwrite a packed matrix using native GPU arithmetic.
  /// Inputs and output must be scalar-aligned views from this context, with at
  /// least the requested packed bytes. Extra trailing bytes are untouched. Output
  /// must not overlap either input's used range; inputs may alias each other.
  /// Calls on one context are serialized. GPU data never passes through host math.
  /// @param destination Output C allocation/view, overwritten; beta is zero.
  /// @param a Packed stored A, unmodified.
  /// @param b Packed stored B, unmodified.
  /// @param product Logical shape, precision and optional transposes.
  /// @throws std::logic_error Native matrix adapter not enabled for this context.
  /// @throws std::invalid_argument Foreign/undefined/unaligned buffers, invalid precision,
  /// dimension above INT_MAX, or overlapping output/input ranges.
  /// @throws std::overflow_error Requested byte counts overflow size_t.
  /// @throws std::out_of_range A buffer is too small.
  /// @throws std::runtime_error CUDA/cuBLAS failure.
  /// @par Example
  /// @code
  /// const dm::compute::Device device{dm::compute::Backend::CUDA,0};
  /// if (dm::compute::probe(device).available) {
  ///   const dm::compute::Context gpu(device);
  ///   if (gpu.supports_matmul()) {
  ///     const float left[]{1,2}, right[]{3,4};
  ///     auto a=gpu.allocate(sizeof(left)), b=gpu.allocate(sizeof(right));
  ///     auto c=gpu.allocate(sizeof(float));
  ///     gpu.upload(a,left,sizeof(left)); gpu.upload(b,right,sizeof(right));
  ///     gpu.matmul(c,a,b,{1,1,2});
  ///     float result=0; gpu.download(&result,c,sizeof(result));
  ///     if (result!=11) throw std::runtime_error("Unexpected matrix product");
  ///   }
  /// }
  /// @endcode
  /// @par References
  /// - NVIDIA cuBLAS GEMM:
  /// https://docs.nvidia.com/cuda/archive/12.8.0/cublas/index.html#cublas-t-gemm
  /// @see MatrixProduct, supports_matmul()
  void matmul(Buffer& destination, const Buffer& a, const Buffer& b, const MatrixProduct& product)
      const;
  /// @brief Compile a native CUDA PTX or Vulkan SPIR-V program on this context.
  /// @param source Exact native code plus buffer/parameter-block layout.
  /// @return Shared native program retaining this context.
  /// @throws std::logic_error CPU context has no native GPU compiler.
  /// @throws std::invalid_argument Invalid layout or missing backend code.
  /// @throws std::runtime_error Driver compilation/validation failure.
  /// @par Example
  /// @code
  /// const dm::compute::Context cpu;
  /// try { cpu.program({}); } catch(const std::logic_error&) { /* Native programs require a GPU
  /// context. */ }
  /// @endcode
  Program program(const KernelSource& source) const;
  /// @brief Execute a native GPU kernel over explicit workgroup counts.
  /// This initial command API is synchronous. Native execution never invokes a
  /// host numerical callback. Programs and buffers must belong to this context.
  /// @param program Native kernel from this context.
  /// @param buffers Ordered allocation views, matching its declared binding count.
  /// @param parameters Packed host bytes for its by-value/push-constant block.
  /// @param parameter_bytes Exact declared block length; null parameters allowed only for zero.
  /// @param groups Positive workgroup counts in X,Y,Z, default one each.
  /// @throws std::invalid_argument Wrong context, shape, parameter length or null arguments.
  /// Vulkan descriptor offsets must meet the selected device's storage-buffer
  /// alignment, and bound ranges must fit its storage-buffer range limit.
  /// @throws std::runtime_error Native dispatch failure.
  /// @par Example
  /// @code
  /// const dm::compute::Context cpu;const dm::compute::Program empty;
  /// try { cpu.dispatch(empty,{},nullptr,0); } catch(const std::invalid_argument&) { /* Undefined
  /// handles are rejected. */ }
  /// @endcode
  void dispatch(
      const Program&               program,
      const std::vector<Buffer>&   buffers,
      const void*                  parameters,
      std::size_t                  parameter_bytes,
      std::array<std::uint32_t, 3> groups = {1, 1, 1}) const;
  /// @brief Synchronously evaluate one independent callback per index on CPU.
  /// The callback must not submit work to the same context (reentry is rejected).
  /// A worker error cancels unclaimed indices, waits for running callbacks and
  /// rethrows the first exception. The context remains usable afterwards.
  /// @param count Number of indices [0,count); zero invokes nothing.
  /// @param function Borrowed callable, alive until all work completes.
  /// @throws std::invalid_argument Empty callable.
  /// @throws std::logic_error GPU context or reentry from a callback on this context.
  /// @throws std::exception Callback exception, after outstanding work is joined.
  /// @par Example
  /// @code
  /// const dm::compute::Context context({},2);
  /// std::vector<int> out(3);
  /// context.parallel_for(3,[&](std::size_t i){out[i]=int(i);});
  /// std::print("{}\n",out[2]); // 2
  /// @endcode
  void parallel_for(std::size_t count, const std::function<void(std::size_t)>& function) const;
  /// @brief Wait for work submitted to this context; CPU submissions are already synchronous.
  /// @throws std::logic_error Called from a callback on this context.
  /// @par Example
  /// @code
  /// const dm::compute::Context context; context.synchronize();
  /// @endcode
  void synchronize() const;
  /// @brief Current logical bytes held by this context's live allocations.
  /// @return Shared views count once; zero-byte allocations contribute zero.
  std::size_t live_bytes() const noexcept;

 private:
  std::shared_ptr<detail::ContextState> state_; ///< Shared backend/workers, retained by buffers.
};
} // namespace datamunge::compute
