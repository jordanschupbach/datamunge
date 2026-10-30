/// @file cuda.cpp
/// @brief Optional CUDA Driver API memory and native PTX execution, without host numerical
/// fallback.
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#ifdef DATAMUNGE_ENABLE_CUBLAS
#include <cublas_v2.h>
#endif

#include "gpu.hpp"
#ifdef DATAMUNGE_ENABLE_CUDA
#ifndef _WIN32
#include <dlfcn.h>
namespace datamunge::compute::detail {
/// @brief Private native driver resource and execution helpers.
namespace {
/// @brief CUDA stable driver ABI declarations; SDK types remain private to the backend.
using Result = int;
/// @brief NVIDIA device pointer, represented by the stable 64-bit driver ABI.
using DevicePointer = std::uint64_t;
/// @brief Dynamically loaded driver functions, retained until all native resources are destroyed.
struct Driver {
  void* library{nullptr}; ///< Native driver shared library.

  /// @brief Native cuInit entrypoint initializes the driver before device discovery.
  Result (*init)(unsigned){};

  /// @brief Native cuDeviceGetCount.
  Result (*device_count)(int*){};

  /// @brief Native cuDeviceGet.
  Result (*device_get)(int*, int){};

  /// @brief Native cuDeviceGetName.
  Result (*device_name)(char*, int, int){};

  /// @brief Native Stable cuCtxCreate_v2 ABI.
  Result (*context_create)(void**, unsigned, int){};

  /// @brief Native cuCtxDestroy_v2.
  Result (*context_destroy)(void*){};

  /// @brief Native cuCtxPushCurrent_v2.
  Result (*push)(void*){};

  /// @brief Native cuCtxPopCurrent_v2.
  Result (*pop)(void**){};

  /// @brief Native cuMemAlloc_v2.
  Result (*allocate)(DevicePointer*, std::size_t){};

  /// @brief Native cuMemFree_v2.
  Result (*free)(DevicePointer){};

  /// @brief Native cuMemcpyHtoD_v2.
  Result (*upload)(DevicePointer, const void*, std::size_t){};

  /// @brief Native cuMemcpyDtoH_v2.
  Result (*download)(void*, DevicePointer, std::size_t){};

  /// @brief Native cuMemcpyDtoD_v2.
  Result (*copy)(DevicePointer, DevicePointer, std::size_t){};

  /// @brief Native cuMemsetD8_v2.
  Result (*fill)(DevicePointer, unsigned char, std::size_t){};

  /// @brief Native cuModuleLoadDataEx.
  Result (*module_load)(void**, const void*, unsigned, int*, void**){};

  /// @brief Native cuModuleUnload.
  Result (*module_unload)(void*){};

  /// @brief Native cuModuleGetFunction.
  Result (*function)(void**, void*, const char*){};

  /// @brief Native cuLaunchKernel dispatches a module function with explicit grid/block dimensions.
  Result (*launch)(
      void*,
      unsigned,
      unsigned,
      unsigned,
      unsigned,
      unsigned,
      unsigned,
      unsigned,
      void*,
      void**,
      void**){}; ///< cuLaunchKernel.

  /// @brief Native cuCtxSynchronize.
  Result (*synchronize)(){};

  /// @brief Native cuGetErrorString.
  Result (*error_string)(Result, const char**){};

  /// @brief Load one required driver symbol or fail before creating a device.
  template<class T>
  void load(T& target, const char* name) {
    target = reinterpret_cast<T>(dlsym(library, name));
    if (!target)
      throw std::runtime_error(std::string("missing CUDA driver symbol: ") + name);
  }
  /// @brief Open the installed NVIDIA driver and resolve the exact versioned interfaces used here.
  Driver() {
    const char* override = std::getenv("DATAMUNGE_CUDA_DRIVER_LIBRARY");
    if (override && *override) {
      library = dlopen(override, RTLD_NOW | RTLD_LOCAL);
    } else {
      library = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
      // NixOS exposes the running host driver outside the normal loader search path.
      if (!library)
        library = dlopen("/run/opengl-driver/lib/libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    }
    if (!library)
      throw std::runtime_error(std::string("CUDA driver unavailable: ") + dlerror());
    try {
      load(init, "cuInit");
      load(device_count, "cuDeviceGetCount");
      load(device_get, "cuDeviceGet");
      load(device_name, "cuDeviceGetName");
      load(context_create, "cuCtxCreate_v2");
      load(context_destroy, "cuCtxDestroy_v2");
      load(push, "cuCtxPushCurrent_v2");
      load(pop, "cuCtxPopCurrent_v2");
      load(allocate, "cuMemAlloc_v2");
      load(free, "cuMemFree_v2");
      load(upload, "cuMemcpyHtoD_v2");
      load(download, "cuMemcpyDtoH_v2");
      load(copy, "cuMemcpyDtoD_v2");
      load(fill, "cuMemsetD8_v2");
      load(module_load, "cuModuleLoadDataEx");
      load(module_unload, "cuModuleUnload");
      load(function, "cuModuleGetFunction");
      load(launch, "cuLaunchKernel");
      load(synchronize, "cuCtxSynchronize");
      load(error_string, "cuGetErrorString");
    } catch (...) {
      dlclose(library);
      library = nullptr;
      throw;
    }
  }
  /// @brief Unload the driver only after its native resource owners are gone.
  ~Driver() {
    if (library)
      dlclose(library);
  }
  /// @brief Convert a failed driver result into an actionable exception.
  void check(Result result, const char* operation) {
    if (result == 0)
      return;
    const char* text = nullptr;
    error_string(result, &text);
    throw std::runtime_error(
        std::string(operation) + ": " + (text ? text : "unknown CUDA error") + " (CUDA error " +
        std::to_string(result) + ")");
  }
};
/// @brief Push a context on this thread and restore the caller's previous context on exit.
struct Scope {
  Driver& driver; ///< Borrowed live driver.

  /// @brief Push the selected native context on this thread.
  explicit Scope(Driver& d, void* context) : driver(d) {
    driver.check(driver.push(context), "cuCtxPushCurrent");
  }
  /// @brief Restore the previous current context without throwing.
  ~Scope() {
    void* popped = nullptr;
    (void)driver.pop(&popped);
  }
};
/// @brief CUDA device allocation retaining the context through its outer buffer owner.
struct CudaMemory : GpuAllocation {
  Driver*       driver;     ///< Borrowed backend driver, retained by public allocation context.
  void*         context;    ///< Allocating CUDA context.
  DevicePointer pointer{0}; ///< Native device address.

  /// @brief Retain the allocating driver/context for device-memory cleanup.
  CudaMemory(Driver& d, void* c) : driver(&d), context(c) {}
  /// @brief Release owned device memory through the allocating context.
  ~CudaMemory() {
    try {
      Scope scope(*driver, context);
      if (pointer)
        (void)driver->free(pointer);
    } catch (...) {
    }
  }
};
/// @brief Module and function owned by one CUDA context.
struct CudaProgram : GpuProgram {
  Driver*                      driver;            ///< Retained by outer program context.
  void*                        context;           ///< Native CUDA context.
  void*                        module{nullptr};   ///< Driver-compiled module.
  void*                        function{nullptr}; ///< Entry point in module.
  std::array<std::uint32_t, 3> local;             ///< Block dimensions.

  /// @brief Retain the native module owner and launch block dimensions.
  CudaProgram(Driver& d, void* c, std::array<std::uint32_t, 3> size)
      : driver(&d), context(c), local(size) {}
  /// @brief Unload the compiled module through its creating context.
  ~CudaProgram() {
    try {
      Scope scope(*driver, context);
      if (module)
        (void)driver->module_unload(module);
    } catch (...) {
    }
  }
};
/// @brief Native CUDA executor; all public submissions are serialized by its owning context.
class CudaDevice final : public GpuDevice {
  Driver      driver_;           ///< Shared-library and driver function ownership.
  void*       context_{nullptr}; ///< Exclusive native CUDA context.
  std::string name_;             ///< Physical GPU name.
#ifdef DATAMUNGE_ENABLE_CUBLAS
  cublasHandle_t blas_{nullptr}; ///< Lazily created matrix adapter, bound to this CUDA context.

  /// @brief Surface a cuBLAS failure with its native operation and status code.
  static void check_blas(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error(
          std::string(operation) + ": cuBLAS status " + std::to_string(status));
  }
  /// @brief Create a retained handle with prescribed Float32/Float64 arithmetic.
  void prepare_blas() {
    if (blas_)
      return;
    cublasHandle_t handle = nullptr;
    check_blas(cublasCreate(&handle), "cublasCreate");
    try {
      check_blas(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH), "cublasSetMathMode");
    } catch (...) {
      cublasDestroy(handle);
      throw;
    }
    blas_ = handle;
  }
#endif
 public:
  /// @brief Initialize the selected physical GPU and restore the caller's prior thread context.
  explicit CudaDevice(std::uint32_t index) {
    driver_.check(driver_.init(0), "cuInit");
    int count = 0;
    driver_.check(driver_.device_count(&count), "cuDeviceGetCount");
    if (index >= static_cast<std::uint32_t>(count))
      throw std::runtime_error("CUDA device index unavailable");
    int device = 0;
    driver_.check(driver_.device_get(&device, int(index)), "cuDeviceGet");
    char text[256]{};
    driver_.check(driver_.device_name(text, sizeof(text), device), "cuDeviceGetName");
    name_ = text;
    driver_.check(driver_.context_create(&context_, 0, device), "cuCtxCreate_v2");
    void* popped = nullptr;
    try {
      driver_.check(driver_.pop(&popped), "cuCtxPopCurrent");
    } catch (...) {
      driver_.context_destroy(context_);
      context_ = nullptr;
      throw;
    }
  }
  /// @brief Release the native context before unloading the retained driver.
  ~CudaDevice() override {
#ifdef DATAMUNGE_ENABLE_CUBLAS
    try {
      Scope scope(driver_, context_);
      if (blas_)
        (void)cublasDestroy(blas_);
    } catch (...) {
    }
#endif
    if (context_)
      (void)driver_.context_destroy(context_);
  }
  std::string                    name() const override { return name_; }
  std::shared_ptr<GpuAllocation> allocate(std::size_t bytes) override {
    Scope scope(driver_, context_);
    auto  result = std::make_shared<CudaMemory>(driver_, context_);
    driver_.check(
        driver_.allocate(&result->pointer, std::max(std::size_t(1), bytes)), "cuMemAlloc");
    return result;
  }
  /// @brief Resolve a validated public allocation view into its native CUDA address.
  DevicePointer address(const GpuArgument& argument) {
    auto* memory = dynamic_cast<CudaMemory*>(argument.allocation.get());
    if (!memory || memory->context != context_)
      throw std::invalid_argument("foreign CUDA allocation");
    return memory->pointer + argument.offset;
  }
  void upload(const GpuArgument& dst, const void* src, std::size_t bytes) override {
    if (!bytes)
      return;
    Scope scope(driver_, context_);
    driver_.check(driver_.upload(address(dst), src, bytes), "cuMemcpyHtoD");
  }
  void download(void* dst, const GpuArgument& src, std::size_t bytes) override {
    if (!bytes)
      return;
    Scope scope(driver_, context_);
    driver_.check(driver_.download(dst, address(src), bytes), "cuMemcpyDtoH");
  }
  void copy(const GpuArgument& dst, const GpuArgument& src, std::size_t bytes) override {
    if (!bytes)
      return;
    Scope scope(driver_, context_);
    auto  d = address(dst), s = address(src);
    if (d == s)
      return;
    if (dst.allocation == src.allocation && ((d < s && bytes > s - d) || (s < d && bytes > d - s)))
      throw std::invalid_argument("overlapping GPU copies require a temporary buffer");
    driver_.check(driver_.copy(d, s, bytes), "cuMemcpyDtoD");
  }
  void fill(const GpuArgument& dst, std::uint8_t byte) override {
    if (!dst.size)
      return;
    Scope scope(driver_, context_);
    driver_.check(driver_.fill(address(dst), byte, dst.size), "cuMemsetD8");
  }
  bool supports_matmul() const noexcept override {
#ifdef DATAMUNGE_ENABLE_CUBLAS
    return true;
#else
    return false;
#endif
  }
  void matmul(
      const GpuArgument&   dst,
      const GpuArgument&   a,
      const GpuArgument&   b,
      const MatrixProduct& product) override {
#ifdef DATAMUNGE_ENABLE_CUBLAS
    Scope scope(driver_, context_);
    if (!product.rows || !product.columns)
      return;
    if (!product.inner) {
      const auto bytes = product.rows * product.columns *
                         (product.type == MatrixType::Float32 ? sizeof(float) : sizeof(double));
      driver_.check(driver_.fill(address(dst), 0, bytes), "cuMemsetD8");
      driver_.check(driver_.synchronize(), "cuCtxSynchronize");
      return;
    }
    prepare_blas();
    const auto m = int(product.rows), n = int(product.columns), k = int(product.inner);
    const auto ta  = product.transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    const auto tb  = product.transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N;
    const auto lda = product.transpose_a ? m : k, ldb = product.transpose_b ? k : n;
    // Packed row-major C = op(A) op(B) is column-major C^T = op(B)^T op(A)^T.
    if (product.type == MatrixType::Float32) {
      const float one = 1, zero = 0;
      check_blas(
          cublasSgemm(
              blas_,
              tb,
              ta,
              n,
              m,
              k,
              &one,
              reinterpret_cast<const float*>(address(b)),
              ldb,
              reinterpret_cast<const float*>(address(a)),
              lda,
              &zero,
              reinterpret_cast<float*>(address(dst)),
              n),
          "cublasSgemm");
    } else {
      const double one = 1, zero = 0;
      check_blas(
          cublasDgemm(
              blas_,
              tb,
              ta,
              n,
              m,
              k,
              &one,
              reinterpret_cast<const double*>(address(b)),
              ldb,
              reinterpret_cast<const double*>(address(a)),
              lda,
              &zero,
              reinterpret_cast<double*>(address(dst)),
              n),
          "cublasDgemm");
    }
    driver_.check(driver_.synchronize(), "cuCtxSynchronize");
#else
    GpuDevice::matmul(dst, a, b, product);
#endif
  }
  std::shared_ptr<GpuProgram> program(const KernelSource& source) override {
    if (source.ptx.empty() || source.ptx.find('\0') != std::string::npos)
      throw std::invalid_argument("CUDA program requires nonempty PTX text");
    Scope scope(driver_, context_);
    auto  program = std::make_shared<CudaProgram>(driver_, context_, source.local_size);
    char  log[4096]{};
    // CU_JIT_ERROR_LOG_BUFFER / CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES, stable Driver ABI.
    int        options[]{5, 6};
    void*      values[]{log, reinterpret_cast<void*>(std::uintptr_t(sizeof(log)))};
    const auto status =
        driver_.module_load(&program->module, source.ptx.c_str(), 2, options, values);
    if (status != 0) {
      try {
        driver_.check(status, "cuModuleLoadDataEx");
      } catch (const std::runtime_error& error) {
        throw std::runtime_error(std::string(error.what()) + ": " + log);
      }
    }
    driver_.check(
        driver_.function(&program->function, program->module, source.entry.c_str()),
        "cuModuleGetFunction");
    return program;
  }
  void dispatch(
      GpuProgram&                     source,
      const std::vector<GpuArgument>& buffers,
      const void*                     parameters,
      std::size_t                     bytes,
      std::array<std::uint32_t, 3>    groups) override {
    auto* program = dynamic_cast<CudaProgram*>(&source);
    if (!program || program->context != context_)
      throw std::invalid_argument("foreign CUDA program");
    Scope                      scope(driver_, context_);
    std::vector<DevicePointer> pointers;
    for (const auto& b : buffers)
      pointers.push_back(address(b));
    std::vector<void*> args;
    for (auto& pointer : pointers)
      args.push_back(&pointer);
    if (bytes)
      args.push_back(const_cast<void*>(parameters));
    driver_.check(
        driver_.launch(
            program->function,
            groups[0],
            groups[1],
            groups[2],
            program->local[0],
            program->local[1],
            program->local[2],
            0,
            nullptr,
            args.data(),
            nullptr),
        "cuLaunchKernel");
    driver_.check(driver_.synchronize(), "cuCtxSynchronize");
  }
  void synchronize() override {
    Scope scope(driver_, context_);
    driver_.check(driver_.synchronize(), "cuCtxSynchronize");
  }
};
} // namespace
std::shared_ptr<GpuDevice> make_cuda(std::uint32_t index) {
  return std::make_shared<CudaDevice>(index);
}
} // namespace datamunge::compute::detail
#else
namespace datamunge::compute::detail {
std::shared_ptr<GpuDevice> make_cuda(std::uint32_t) {
  throw std::runtime_error("CUDA backend currently requires a POSIX driver loader");
}
} // namespace datamunge::compute::detail
#endif
#else
namespace datamunge::compute::detail {
std::shared_ptr<GpuDevice> make_cuda(std::uint32_t) {
  throw std::runtime_error("CUDA backend was not enabled at build time");
}
} // namespace datamunge::compute::detail
#endif
