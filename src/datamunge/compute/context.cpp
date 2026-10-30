/// @file context.cpp
/// @brief Context-owned CPU allocations and an exception-safe persistent worker pool.
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <datamunge/compute/context.hpp>
#include <exception>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>

#include "gpu.hpp"

namespace datamunge::compute {
namespace detail {
/// @brief One link in the current thread's nested callback ownership chain.
struct CallbackScope;
/// @brief Innermost callback owner on this thread; no process-global execution context.
thread_local const CallbackScope* active_scope = nullptr;
/// @brief Restore a thread's previous callback owner on exceptional or normal exit.
struct CallbackScope {
  const ContextState*  context;  ///< Context whose callback is active.
  const CallbackScope* previous; ///< Enclosing callback scope, possibly another context.

  /// @brief Publish a nested callback owner.
  explicit CallbackScope(const ContextState* state) : context(state), previous(active_scope) {
    active_scope = this;
  }
  /// @brief Restore the enclosing callback chain.
  ~CallbackScope() { active_scope = previous; }
};
/// @brief Test all enclosing callbacks, preventing indirect same-context reentry.
bool executing(const ContextState* context) {
  for (auto* scope = active_scope; scope; scope = scope->previous)
    if (scope->context == context)
      return true;
  return false;
}

/// @brief Shared synchronization, native backend ownership and reusable CPU work dispatch.
struct ContextState {
  Device                     device;   ///< Explicit backend/index.
  std::shared_ptr<GpuDevice> gpu;      ///< Native GPU executor, null for CPU.
  std::size_t                lanes;    ///< Caller plus persistent worker threads.
  std::atomic<std::size_t>   bytes{0}; ///< Logical live allocation bytes.
  std::vector<std::thread>   threads;  ///< Joined when the last context/allocation releases state.
  mutable std::mutex         submission; ///< Serializes complete jobs from independent callers.
  std::mutex                 mutex; ///< Protects generation, callback and completion/error state.
  std::condition_variable    ready; ///< Announces a fresh job or shutdown.
  std::condition_variable    finished;      ///< Announces all worker threads left the current job.
  bool                       stop{false};   ///< Shutdown flag under mutex.
  std::size_t                generation{0}; ///< Monotonic job publication counter.
  std::size_t                remaining{0};  ///< Workers still participating in the current job.
  std::size_t                count{0}; ///< Published work size, immutable while a job executes.
  std::atomic<std::size_t>   next{0};  ///< Next unclaimed chunk start.
  std::atomic<bool>          cancel{false}; ///< Stops assigning new callbacks after an exception.
  const std::function<void(std::size_t)>* function{nullptr}; ///< Borrowed synchronous job.
  std::exception_ptr error; ///< First callback failure; rethrown by the submitter.

  /// @brief Start workers; roll back a partially created pool if thread creation fails.
  /// @param selected Validated CPU device.
  /// @param workers Positive execution-lane count.
  /// @param backend Native GPU owner, empty for CPU.
  ContextState(Device selected, std::size_t workers, std::shared_ptr<GpuDevice> backend = {})
      : device(selected), gpu(std::move(backend)), lanes(gpu ? 1 : workers) {
    try {
      for (std::size_t i = 1; i < lanes; ++i)
        threads.emplace_back([this] { worker(); });
    } catch (...) {
      shutdown();
      throw;
    }
  }
  /// @brief Join workers before destroying synchronization state.
  ~ContextState() { shutdown(); }
  /// @brief Signal and join every successfully started worker.
  void shutdown() noexcept {
    {
      std::lock_guard<std::mutex> lock(mutex);
      stop = true;
    }
    ready.notify_all();
    for (auto& thread : threads)
      if (thread.joinable())
        thread.join();
  }
  /// @brief Claim bounded chunks and propagate the first exception through shared state.
  void execute() noexcept {
    CallbackScope scope(this);
    try {
      constexpr std::size_t chunk = 16;
      while (!cancel.load(std::memory_order_relaxed)) {
        auto        begin = next.load(std::memory_order_relaxed);
        std::size_t end;
        do {
          if (begin >= count)
            return;
          end = begin + std::min(chunk, count - begin);
        } while (!next.compare_exchange_weak(begin, end, std::memory_order_relaxed));
        for (auto i = begin; i < end && !cancel.load(std::memory_order_relaxed); ++i)
          (*function)(i);
      }
    } catch (...) {
      std::lock_guard<std::mutex> lock(mutex);
      if (!error)
        error = std::current_exception();
      cancel.store(true, std::memory_order_relaxed);
    }
  }
  /// @brief Wait on generations and account for completion even when a callback fails.
  void worker() noexcept {
    std::size_t                  seen = 0;
    std::unique_lock<std::mutex> lock(mutex);
    for (;;) {
      ready.wait(lock, [&] { return stop || generation != seen; });
      if (stop)
        return;
      seen = generation;
      lock.unlock();
      execute();
      lock.lock();
      if (--remaining == 0)
        finished.notify_one();
    }
  }
};

/// @brief Shared logical allocation with backend storage and context lifetime retention.
struct Allocation {
  std::shared_ptr<ContextState>   context; ///< Retains the allocator and workers.
  std::unique_ptr<std::uint8_t[]> data;    ///< CPU-owned storage, including one byte for size zero.
  std::shared_ptr<GpuAllocation>  native;  ///< GPU allocation, null for CPU.
  std::size_t                     size;    ///< Logical bytes charged once to the context.

  /// @brief Allocate bytes and charge them only after allocation succeeds.
  /// @param owner Creating context, retained.
  /// @param bytes Logical allocation size.
  Allocation(std::shared_ptr<ContextState> owner, std::size_t bytes)
      : context(std::move(owner)),
        data(context->gpu ? nullptr : new std::uint8_t[std::max(std::size_t(1), bytes)]),
        native(context->gpu ? context->gpu->allocate(bytes) : nullptr),
        size(bytes) {
    context->bytes.fetch_add(size, std::memory_order_relaxed);
  }
  /// @brief Release this allocation's charge when its last view is destroyed.
  ~Allocation() { context->bytes.fetch_sub(size, std::memory_order_relaxed); }
};
/// @brief Native program retaining its exact creating context and argument contract.
struct ProgramState {
  std::shared_ptr<ContextState> context; ///< Retains backend until after program destruction.
  std::shared_ptr<GpuProgram>   native;  ///< Native backend kernel owner.
  KernelSource                  source;  ///< Explicit launch layout and original code.
};
/// @brief Construct the selected native backend, rejecting unsupported identifiers.
std::shared_ptr<GpuDevice> make_gpu(Device device) {
  switch (device.backend) {
    case Backend::CUDA:
      return make_cuda(device.index);
    case Backend::Vulkan:
      return make_vulkan(device.index);
    case Backend::CPU:
      break;
  }
  throw std::invalid_argument("CPU has no native GPU backend");
}
} // namespace detail

std::string backend_name(Backend backend) {
  switch (backend) {
    case Backend::CPU:
      return "cpu";
    case Backend::CUDA:
      return "cuda";
    case Backend::Vulkan:
      return "vulkan";
  }
  throw std::invalid_argument("unknown compute backend");
}
DeviceInfo probe(Device device) {
  const auto name = backend_name(device.backend);
  if (device.backend == Backend::CPU && device.index == 0)
    return {device, true, "CPU", {}};
  if (device.backend == Backend::CPU)
    return {device, false, {}, "CPU device index must be zero"};
  try {
    auto gpu = detail::make_gpu(device);
    return {device, true, gpu->name(), {}};
  } catch (const std::exception& e) {
    return {device, false, {}, e.what()};
  }
}
Context::Context(Device device, std::size_t workers) {
  if (!workers)
    throw std::invalid_argument("compute workers must be positive");
  if (device.backend == Backend::CPU && device.index != 0)
    throw std::invalid_argument("CPU device index must be zero");
  backend_name(device.backend);
  auto gpu = device.backend == Backend::CPU ? nullptr : detail::make_gpu(device);
  state_   = std::make_shared<detail::ContextState>(device, workers, std::move(gpu));
}
Device Context::device() const noexcept {
  return state_->device;
}
bool Context::shares_with(const Context& other) const noexcept {
  return state_ == other.state_;
}
std::size_t Context::workers() const noexcept {
  return state_->lanes;
}
Buffer Context::allocate(std::size_t bytes) const {
  Buffer buffer;
  buffer.allocation_ = std::make_shared<detail::Allocation>(state_, bytes);
  buffer.size_       = bytes;
  return buffer;
}
Device Buffer::device() const {
  if (!allocation_)
    throw std::logic_error("undefined compute buffer");
  return allocation_->context->device;
}
Buffer Buffer::slice(std::size_t offset, std::size_t size) const {
  if (!allocation_)
    throw std::logic_error("undefined compute buffer");
  if (offset > size_ || size > size_ - offset)
    throw std::out_of_range("compute buffer view bounds");
  Buffer out = *this;
  out.offset_ += offset;
  out.size_ = size;
  return out;
}
std::uint8_t* Buffer::host_data() {
  if (device().backend != Backend::CPU)
    throw std::logic_error("GPU buffer has no host pointer");
  return allocation_->data.get() + offset_;
}
const std::uint8_t* Buffer::host_data() const {
  if (device().backend != Backend::CPU)
    throw std::logic_error("GPU buffer has no host pointer");
  return allocation_->data.get() + offset_;
}
void Context::copy(Buffer& destination, const Buffer& source, std::size_t bytes) const {
  if (!destination.defined() || !source.defined() || destination.allocation_->context != state_ ||
      source.allocation_->context != state_)
    throw std::invalid_argument("compute copy requires buffers from this context");
  if (bytes > destination.size_ || bytes > source.size_)
    throw std::out_of_range("compute copy bounds");
  if (!state_->gpu)
    std::memmove(destination.host_data(), source.host_data(), bytes);
  else {
    std::lock_guard<std::mutex> lock(state_->submission);
    state_->gpu->copy(
        {destination.allocation_->native, destination.offset_, destination.size_},
        {source.allocation_->native, source.offset_, source.size_},
        bytes);
  }
}
void Context::fill(Buffer& buffer, std::uint8_t value) const {
  if (!buffer.defined() || buffer.allocation_->context != state_)
    throw std::invalid_argument("compute fill requires a buffer from this context");
  if (!state_->gpu)
    std::memset(buffer.host_data(), value, buffer.size_);
  else {
    std::lock_guard<std::mutex> lock(state_->submission);
    state_->gpu->fill({buffer.allocation_->native, buffer.offset_, buffer.size_}, value);
  }
}
void Context::parallel_for(std::size_t count, const std::function<void(std::size_t)>& function)
    const {
  if (state_->gpu)
    throw std::logic_error("GPU execution requires a native program, not a host callback");
  if (!function)
    throw std::invalid_argument("empty compute callback");
  if (detail::executing(state_.get()))
    throw std::logic_error("compute context callback reentry");
  if (!count)
    return;
  std::lock_guard<std::mutex> submit(state_->submission);
  {
    std::lock_guard<std::mutex> lock(state_->mutex);
    state_->function  = &function;
    state_->count     = count;
    state_->next      = 0;
    state_->cancel    = false;
    state_->error     = nullptr;
    state_->remaining = state_->threads.size();
    ++state_->generation;
  }
  state_->ready.notify_all();
  state_->execute();
  std::exception_ptr error;
  {
    std::unique_lock<std::mutex> lock(state_->mutex);
    state_->finished.wait(lock, [&] { return state_->remaining == 0; });
    state_->function = nullptr;
    error            = state_->error;
  }
  if (error)
    std::rethrow_exception(error);
}
void Context::synchronize() const {
  if (detail::executing(state_.get()))
    throw std::logic_error("compute context callback reentry");
  std::lock_guard<std::mutex> submit(state_->submission);
  if (state_->gpu)
    state_->gpu->synchronize();
}
std::size_t Context::live_bytes() const noexcept {
  return state_->bytes.load(std::memory_order_relaxed);
}
void Context::upload(Buffer& destination, const void* source, std::size_t bytes) const {
  if (!destination.defined() || destination.allocation_->context != state_ || (!source && bytes))
    throw std::invalid_argument("invalid upload ownership/source");
  if (bytes > destination.size_)
    throw std::out_of_range("upload bounds");
  if (!bytes)
    return;
  if (!state_->gpu)
    std::memmove(destination.host_data(), source, bytes);
  else {
    std::lock_guard<std::mutex> lock(state_->submission);
    state_->gpu->upload(
        {destination.allocation_->native, destination.offset_, destination.size_}, source, bytes);
  }
}
void Context::download(void* destination, const Buffer& source, std::size_t bytes) const {
  if (!source.defined() || source.allocation_->context != state_ || (!destination && bytes))
    throw std::invalid_argument("invalid download ownership/destination");
  if (bytes > source.size_)
    throw std::out_of_range("download bounds");
  if (!bytes)
    return;
  if (!state_->gpu)
    std::memmove(destination, source.host_data(), bytes);
  else {
    std::lock_guard<std::mutex> lock(state_->submission);
    state_->gpu->download(
        destination, {source.allocation_->native, source.offset_, source.size_}, bytes);
  }
}
bool Context::supports_matmul() const noexcept {
  return state_->gpu && state_->gpu->supports_matmul();
}
void Context::matmul(
    Buffer&              destination,
    const Buffer&        a,
    const Buffer&        b,
    const MatrixProduct& product) const {
  if (!supports_matmul())
    throw std::logic_error("native matrix adapter unavailable; enable CUDA and cuBLAS");
  if (product.type != MatrixType::Float32 && product.type != MatrixType::Float64)
    throw std::invalid_argument("invalid native matrix precision");
  for (auto n : {product.rows, product.columns, product.inner})
    if (n > std::size_t(std::numeric_limits<int>::max()))
      throw std::invalid_argument("native matrix dimensions exceed INT_MAX");
  const auto scalar = product.type == MatrixType::Float32 ? sizeof(float) : sizeof(double);
  /// Compute checked packed byte lengths, including empty matrices.
  const auto bytes = [scalar](std::size_t x, std::size_t y) {
    if (y && x > std::numeric_limits<std::size_t>::max() / scalar / y)
      throw std::overflow_error("native matrix byte count overflow");
    return x * y * scalar;
  };
  const auto na = bytes(product.rows, product.inner);
  const auto nb = bytes(product.inner, product.columns);
  const auto nc = bytes(product.rows, product.columns);
  /// Validate ownership, scalar alignment and capacity before touching native resources.
  const auto validate = [&](const Buffer& buffer, std::size_t required) {
    if (!buffer.defined() || buffer.allocation_->context != state_ || buffer.offset_ % scalar)
      throw std::invalid_argument(
          "native matrix buffers must be aligned and owned by this context");
    if (buffer.size_ < required)
      throw std::out_of_range("native matrix buffer too small");
  };
  validate(a, na);
  validate(b, nb);
  validate(destination, nc);
  /// Compare bounded used ranges without overflowing offset-plus-size arithmetic.
  const auto overlaps = [&](const Buffer& input, std::size_t count) {
    if (!nc || !count || input.allocation_ != destination.allocation_)
      return false;
    const auto d = destination.offset_, s = input.offset_;
    return d <= s ? s - d < nc : d - s < count;
  };
  if (overlaps(a, na) || overlaps(b, nb))
    throw std::invalid_argument("native matrix output overlaps an input");
  std::lock_guard<std::mutex> lock(state_->submission);
  state_->gpu->matmul(
      {destination.allocation_->native, destination.offset_, destination.size_},
      {a.allocation_->native, a.offset_, a.size_},
      {b.allocation_->native, b.offset_, b.size_},
      product);
}
Program Context::program(const KernelSource& source) const {
  if (!state_->gpu)
    throw std::logic_error("CPU context has no native GPU program compiler");
  std::uint64_t lanes = 1;
  for (auto n : source.local_size) {
    if (!n || n > 1024)
      throw std::invalid_argument("invalid kernel local size");
    lanes *= n;
  }
  if (lanes > 1024 || source.buffers > 32 || source.parameter_bytes > 4096 ||
      source.entry.empty() || source.entry.find('\0') != std::string::npos)
    throw std::invalid_argument("invalid native program layout");
  std::lock_guard<std::mutex> lock(state_->submission);
  auto                        native = state_->gpu->program(source);
  Program                     result;
  result.state_          = std::make_shared<detail::ProgramState>();
  result.state_->context = state_;
  result.state_->source  = source;
  result.state_->native  = std::move(native);
  return result;
}
void Context::dispatch(
    const Program&               program,
    const std::vector<Buffer>&   buffers,
    const void*                  parameters,
    std::size_t                  parameter_bytes,
    std::array<std::uint32_t, 3> groups) const {
  if (!program.defined() || program.state_->context != state_ ||
      buffers.size() != program.state_->source.buffers ||
      parameter_bytes != program.state_->source.parameter_bytes || (!parameters && parameter_bytes))
    throw std::invalid_argument("invalid native program/arguments");
  for (auto n : groups)
    if (!n)
      throw std::invalid_argument("kernel group dimensions must be positive");
  std::vector<detail::GpuArgument> arguments;
  arguments.reserve(buffers.size());
  for (const auto& b : buffers) {
    if (!b.defined() || b.allocation_->context != state_ || !b.size_)
      throw std::invalid_argument("kernel buffers must be nonempty and owned by this context");
    arguments.push_back({b.allocation_->native, b.offset_, b.size_});
  }
  std::lock_guard<std::mutex> lock(state_->submission);
  state_->gpu->dispatch(*program.state_->native, arguments, parameters, parameter_bytes, groups);
}
} // namespace datamunge::compute
