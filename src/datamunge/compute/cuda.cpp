/// @file cuda.cpp
/// @brief Optional CUDA Driver API memory and native PTX execution, without host numerical
/// fallback.
///
/// Context::matmul runs the library's own GEMM kernels (shaders/compute/blas.cu, embedded as
/// PTX in cuda_blas_ptx.hpp and JIT-compiled by the driver on first use), so the CUDA backend
/// needs nothing beyond the NVIDIA driver. Builds with DATAMUNGE_ENABLE_CUBLAS keep a cuBLAS
/// path for comparison only; it runs when the environment variable DATAMUNGE_CUDA_GEMM is
/// "cublas" when the device is created.
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <tuple>

#ifdef DATAMUNGE_ENABLE_CUBLAS
#include <cublas_v2.h>
#endif

#include "cuda_blas_ptx.hpp"
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

  /// @brief Native cuDeviceGetAttribute.
  Result (*device_attribute)(int*, int, int){};

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

  /// @brief Native cuOccupancyMaxActiveBlocksPerMultiprocessor (GEMM planning).
  Result (*occupancy)(int*, void*, int, std::size_t){};

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
      load(device_attribute, "cuDeviceGetAttribute");
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
      load(occupancy, "cuOccupancyMaxActiveBlocksPerMultiprocessor");
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
/// @brief By-value parameter block of the gemm_* entries in shaders/compute/blas.cu.
struct GemmParameters {
  std::uint64_t m, n, k;
  std::uint64_t lda, ldb, ldc;
  std::uint32_t transpose_a, transpose_b;
  double        alpha, beta;
  std::uint64_t whole;  ///< Tiles computed whole, one block each.
  std::uint64_t splits; ///< k chunks of every later tile.
  std::uint64_t kchunk; ///< k per chunk, a multiple of 16.
};
static_assert(sizeof(GemmParameters) == 96, "GemmParameters must match shaders/compute/blas.cu");
/// @brief By-value parameter block of the reduce_* entries (the split tiles of one product).
struct ReduceParameters {
  std::uint64_t m, n, ldc, bm, bn, whole, splits, count;
  double        alpha, beta;
};
static_assert(sizeof(ReduceParameters) == 80, "ReduceParameters must match shaders/compute/blas.cu");

/// @brief One GEMM entry of blas.cu: output tile, block size and cost-model constants.
struct GemmKernel {
  const char*   entry;
  std::uint64_t bm, bn;
  unsigned      threads;
  bool          f32;
  /// @brief One wave of every multiprocessor's resident blocks with chunk length kc takes
  /// a + b * kc microseconds, measured on an RTX 5080 Laptop GPU (only the ratios between
  /// entries matter for the choice).
  double a, b;
  void*  function{nullptr};
  int    occupancy{1}; ///< Resident blocks per multiprocessor, from the driver.
};

/// @brief How one product runs: tiles [0, whole) whole, every later tile in `splits` k chunks.
struct GemmPlan {
  const GemmKernel* kernel{nullptr};
  std::uint64_t     whole{0}, splits{1}, kchunk{0};
};

/// @brief Native CUDA executor; all public submissions are serialized by its owning context.
class CudaDevice final : public GpuDevice {
  Driver      driver_;           ///< Shared-library and driver function ownership.
  void*       context_{nullptr}; ///< Exclusive native CUDA context.
  std::string name_;             ///< Physical GPU name.
  void*       gemm_module_{nullptr}; ///< In-house GEMM module, JIT-compiled on first matmul.
  /// @brief GEMM entries (see shaders/compute/blas.cu for the tilings).
  GemmKernel gemm_kernels_[6]{{"gemm_f32_128x256", 128, 256, 256, true, 11.7, 0.1356},
                              {"gemm_f32_128x128", 128, 128, 128, true, 11.3, 0.1474},
                              {"gemm_f32_128x64", 128, 64, 128, true, 9.5, 0.111},
                              {"gemm_f32_64x64", 64, 64, 256, true, 6.5, 0.0795},
                              {"gemm_f32_128x32", 128, 32, 256, true, 5.5, 0.102},
                              {"gemm_f64_64x32", 64, 32, 256, false, 5.8, 0.734}};
  void*         reduce_f32_{nullptr};    ///< Split-K reduction entries.
  void*         reduce_f64_{nullptr};
  int           multiprocessors_{1};     ///< Streaming multiprocessors, for GEMM planning.
  DevicePointer workspace_{0};           ///< Split-K partial slabs, grown on demand.
  std::size_t   workspace_bytes_{0};
  /// @brief Plans of recent shapes (m, n, k, float), so repeated products skip the search.
  std::map<std::tuple<std::uint64_t, std::uint64_t, std::uint64_t, bool>, GemmPlan> plans_;
  bool          use_cublas_{false};      ///< DATAMUNGE_CUDA_GEMM=cublas in a cuBLAS build.

  /// @brief Load the embedded GEMM PTX once; the caller holds the context scope.
  void prepare_gemm() {
    if (gemm_module_)
      return;
    char  log[4096]{};
    int   options[]{5, 6}; // CU_JIT_ERROR_LOG_BUFFER / CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES
    void* values[]{log, reinterpret_cast<void*>(std::uintptr_t(sizeof(log)))};
    void* module = nullptr;
    if (const auto status = driver_.module_load(&module, blas_ptx, 2, options, values); status != 0) {
      try {
        driver_.check(status, "cuModuleLoadDataEx(gemm)");
      } catch (const std::runtime_error& error) {
        throw std::runtime_error(std::string(error.what()) + ": " + log);
      }
    }
    try {
      for (auto& kernel : gemm_kernels_) {
        driver_.check(driver_.function(&kernel.function, module, kernel.entry), "cuModuleGetFunction(gemm)");
        driver_.check(driver_.occupancy(&kernel.occupancy, kernel.function, int(kernel.threads), 0),
                      "cuOccupancyMaxActiveBlocksPerMultiprocessor(gemm)");
        kernel.occupancy = std::max(kernel.occupancy, 1);
      }
      driver_.check(driver_.function(&reduce_f32_, module, "reduce_f32"), "cuModuleGetFunction(reduce_f32)");
      driver_.check(driver_.function(&reduce_f64_, module, "reduce_f64"), "cuModuleGetFunction(reduce_f64)");
    } catch (...) {
      (void)driver_.module_unload(module);
      throw;
    }
    gemm_module_ = module;
  }
  /// @brief Workspace of at least `bytes`, reallocated (never shrunk) as needed.
  DevicePointer workspace(std::size_t bytes) {
    if (bytes > workspace_bytes_) {
      if (workspace_)
        (void)driver_.free(workspace_);
      workspace_       = 0;
      workspace_bytes_ = 0;
      driver_.check(driver_.allocate(&workspace_, bytes), "cuMemAlloc(gemm workspace)");
      workspace_bytes_ = bytes;
    }
    return workspace_;
  }
  /// @brief Pick the entry and k split with the lowest modelled time for an m x n x k product.
  ///
  /// The busiest multiprocessor runs ceil(blocks / multiprocessors) blocks in rounds of the
  /// entry's occupancy; a full round costs b * kc (GemmKernel), a partly filled one sqrt(fill)
  /// of that, and every launch a. Split tiles add a reduction pass that reads their slabs.
  /// Candidates per entry: every tile whole; with fewer tiles than resident blocks, every tile
  /// split into 2..64 chunks of at least 32; with more, full waves of whole tiles and only the
  /// leftover tiles split (instead of a last, mostly idle wave). Deterministic for a given device
  /// and shape; the split workspace is capped at 64 MiB.
  GemmPlan plan(std::uint64_t m, std::uint64_t n, std::uint64_t k, bool f32) {
    const auto key = std::make_tuple(m, n, k, f32);
    if (const auto found = plans_.find(key); found != plans_.end())
      return found->second;
    const std::uint64_t sms   = std::uint64_t(multiprocessors_);
    const double        width = f32 ? sizeof(float) : sizeof(double);
    GemmPlan            best;
    double              best_time = 0;
    for (const auto& g : gemm_kernels_) {
      if (g.f32 != f32)
        continue;
      const std::uint64_t occ = std::uint64_t(g.occupancy);
      const std::uint64_t tiles = ((m + g.bm - 1) / g.bm) * ((n + g.bn - 1) / g.bn), slots = sms * occ;
      const auto          rounds = [&](std::uint64_t blocks, double kc) {
        const std::uint64_t per = (blocks + sms - 1) / sms;
        return g.b * kc * (double(per / occ) + std::sqrt(double(per % occ) / double(occ)));
      };
      const auto consider = [&](std::uint64_t whole, std::uint64_t splits) {
        std::uint64_t kchunk = k;
        if (splits > 1) {
          kchunk = ((k + splits - 1) / splits + 15) / 16 * 16;
          splits = (k + kchunk - 1) / kchunk;
        }
        if (splits <= 1) {
          whole  = tiles;
          splits = 1;
          kchunk = k;
        }
        const double slabs = double(tiles - whole) * double(g.bm * g.bn) * width;
        if (slabs * double(splits) > double(64u << 20))
          return;
        double time = g.a + rounds(whole, double(k));
        if (whole < tiles)
          time += rounds((tiles - whole) * splits, double(kchunk)) + 2 + slabs * double(splits + 1) / 2.5e6;
        if (!best.kernel || time < best_time) {
          best      = {&g, whole, splits, kchunk};
          best_time = time;
        }
      };
      consider(tiles, 1);
      for (std::uint64_t splits = 2; splits <= 64 && k / splits >= 32; ++splits) {
        if (tiles < slots)
          consider(0, splits);
        else if (tiles % slots)
          consider(tiles / slots * slots, splits);
      }
    }
    if (plans_.size() >= 256)
      plans_.clear();
    plans_.emplace(key, best);
    return best;
  }
  /// @brief C = op(A) op(B) with the in-house kernels: one block per whole output tile or k chunk
  /// of a split tile (see plan()), then, when tiles were split, a reduction with one thread per
  /// vector of their outputs and group of eight chunks (blocks of up to 256 threads).
  void gemm(DevicePointer c, DevicePointer a, DevicePointer b, const MatrixProduct& product) {
    prepare_gemm();
    const bool          f32 = product.type == MatrixType::Float32;
    const std::uint64_t m = product.rows, n = product.columns, k = product.inner;
    const std::uint64_t width = f32 ? sizeof(float) : sizeof(double);
    const GemmPlan      p     = plan(m, n, k, f32);
    const GemmKernel&   g     = *p.kernel;
    const std::uint64_t tiles = ((m + g.bm - 1) / g.bm) * ((n + g.bn - 1) / g.bn);
    const std::uint64_t blocks = p.whole + (tiles - p.whole) * p.splits;
    if (blocks > 0x7fffffffull)
      throw std::length_error("matmul: product too large for one native launch");
    GemmParameters parameters{m, n, k, product.transpose_a ? m : k, product.transpose_b ? k : n, n,
                              product.transpose_a ? 1u : 0u, product.transpose_b ? 1u : 0u, 1.0, 0.0,
                              p.whole, p.splits, p.kchunk};
    const std::uint64_t split_tiles = tiles - p.whole;
    DevicePointer       w = split_tiles ? workspace(std::size_t(split_tiles * p.splits * g.bm * g.bn * width))
                                        : DevicePointer(0);
    void*               args[]{&a, &b, &c, &w, &parameters};
    driver_.check(
        driver_.launch(g.function, unsigned(blocks), 1, 1, g.threads, 1, 1, 0, nullptr, args, nullptr),
        "cuLaunchKernel(gemm)");
    if (split_tiles) {
      const std::uint64_t vectors = split_tiles * g.bm * g.bn / (f32 ? 4 : 2);
      ReduceParameters    r{m, n, n, g.bm, g.bn, p.whole, p.splits, vectors, 1.0, 0.0};
      void*               reduce_args[]{&w, &c, &r};
      // Blocks of X vectors x G groups of eight chunks: splits <= 64 (plan()), so G <= 8 and
      // X * G <= 256. X is narrowed while fewer than eight blocks per multiprocessor would run.
      const std::uint64_t groups = (p.splits + 7) / 8;
      std::uint64_t       x      = 32 * std::max<std::uint64_t>(1, 8 / groups);
      while (x > 32 && (vectors + x - 1) / x < 8 * std::uint64_t(multiprocessors_))
        x /= 2;
      driver_.check(driver_.launch(f32 ? reduce_f32_ : reduce_f64_, unsigned((vectors + x - 1) / x), 1, 1,
                                   unsigned(x), unsigned(groups), 1, 0, nullptr, reduce_args, nullptr),
                    "cuLaunchKernel(gemm reduce)");
    }
  }
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
    driver_.check(driver_.device_attribute(&multiprocessors_, 16, device), "cuDeviceGetAttribute");
    multiprocessors_ = std::max(multiprocessors_, 1);
#ifdef DATAMUNGE_ENABLE_CUBLAS
    if (const char* gemm = std::getenv("DATAMUNGE_CUDA_GEMM"))
      use_cublas_ = std::strcmp(gemm, "cublas") == 0;
#endif
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
    try {
      Scope scope(driver_, context_);
      if (workspace_)
        (void)driver_.free(workspace_);
      if (gemm_module_)
        (void)driver_.module_unload(gemm_module_);
#ifdef DATAMUNGE_ENABLE_CUBLAS
      if (blas_)
        (void)cublasDestroy(blas_);
#endif
    } catch (...) {
    }
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
  bool supports_matmul() const noexcept override { return true; }
  void matmul(
      const GpuArgument&   dst,
      const GpuArgument&   a,
      const GpuArgument&   b,
      const MatrixProduct& product) override {
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
    if (!use_cublas_) {
      gemm(address(dst), address(a), address(b), product);
      driver_.check(driver_.synchronize(), "cuCtxSynchronize");
      return;
    }
#ifdef DATAMUNGE_ENABLE_CUBLAS
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
