// CMD: just run-nn hello_tf
// Prints information about the datamunge::nn build and about the GPU(s) in
// this machine.
//
// Port of tflow's hello_tf. Two sources of truth:
//   1. datamunge::nn itself: version, the dtypes and kernels it was built
//      with (GEMM and LAPACK backends, intra-op threads), CPU capabilities.
//      TF lists its physical and session devices here; nn is CPU only (the
//      plan's D4: GPU, float16 and bfloat16 are out of scope), so the one
//      "device" is the host CPU with its thread pool.
//   2. NVIDIA's management library (NVML), loaded at runtime with dlopen so the
//      program builds without CUDA headers and still works if no driver exists
//      (kept unchanged from TF: it reports the hardware, not the framework).
// A small matmul timing replaces TF's DeviceAttributes listing as evidence that
// the kernels run.

#include <dlfcn.h>

#include <chrono>
#include <cstdio>
#include <datamunge/nn/nn.hpp>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#if __has_include(<datamunge/version.h>)
#include <datamunge/version.h>
#endif
#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

#include "common/checks.hpp"

namespace nn = datamunge::nn;

namespace {

std::string HumanBytes(unsigned long long bytes) {
  const char* units[] = {"B", "KiB", "MiB", "GiB", "TiB"};
  double      v       = static_cast<double>(bytes);
  int         u       = 0;
  while (v >= 1024.0 && u < 4) {
    v /= 1024.0;
    ++u;
  }
  char buf[32];
  std::snprintf(buf, sizeof buf, "%.1f %s", v, units[u]);
  return buf;
}

// ---------------------------------------------------------------------------
// datamunge::nn
// ---------------------------------------------------------------------------

#if defined(__APPLE__)
std::string SysctlString(const char* name) {
  char        buf[256];
  std::size_t size = sizeof buf;
  if (sysctlbyname(name, buf, &size, nullptr, 0) != 0) return "";
  return std::string(buf, size > 0 ? size - 1 : 0);
}
long long SysctlInt(const char* name) {
  long long   v    = 0;
  std::size_t size = sizeof v;
  if (sysctlbyname(name, &v, &size, nullptr, 0) != 0) return -1;
  return size == 4 ? static_cast<long long>(*reinterpret_cast<int*>(&v)) : v;
}
#endif

int ReportNn() {
  Header("datamunge::nn");
#ifdef DATAMUNGE_VERSION
  std::cout << "Version:              " << DATAMUNGE_VERSION << "\n";
#else
  std::cout << "Version:              (version.h not generated; built outside CMake)\n";
#endif
  std::cout << "Execution model:      eager, define-by-run autograd (no graph, no session)\n";
  std::cout << "Dtypes:              ";
  for (nn::DType d : {nn::DType::Bool, nn::DType::Int32, nn::DType::Int64, nn::DType::Float32, nn::DType::Float64})
    std::cout << " " << nn::dtype_name(d) << "(" << nn::dtype_size(d) << "B)";
  std::cout << "\n";
  std::cout << "GEMM backend:         " << nn::gemm_backend() << "\n";
  std::cout << "LAPACK:               " << (nn::linalg_has_lapack() ? "yes" : "no (built-in kernels)") << "\n";
  std::cout << "Intra-op threads:     " << nn::get_num_threads() << " (DATAMUNGE_NUM_THREADS overrides)\n";
  // TF's ListAllPhysicalDevices: nn has exactly one device, the host CPU.
  std::cout << "Devices (1):\n  /device:CPU:0\n";
  std::cout << "Built with GPU support: no (CPU-only by design; GPU is out of scope)\n";

  // Evidence that the kernels run: one float32 matmul through the GEMM backend.
  nn::Generator gen(1);
  const nn::Tensor a  = nn::normal({512, 512}, 0.0, 1.0, gen);
  const auto       t0 = std::chrono::steady_clock::now();
  const nn::Tensor c  = nn::matmul(a, a);
  const double     s  = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("512x512 float32 matmul: %.2f ms (%.1f GFLOP/s), checksum finite: %s\n", s * 1e3,
              2.0 * 512 * 512 * 512 / s / 1e9, std::isfinite(nn::sum(c).item<float>()) ? "yes" : "no");
  check::Require(c.shape() == nn::Shape({512, 512}) && std::isfinite(nn::sum(c).item<float>()), "matmul runs");

  Header("CPU (as seen by datamunge)");
  std::cout << "Hardware threads:     " << std::thread::hardware_concurrency() << "\n";
#if defined(__APPLE__)
  std::cout << "Model:                " << SysctlString("machdep.cpu.brand_string") << "\n";
  std::cout << "Physical cores:       " << SysctlInt("hw.physicalcpu") << "\n";
  std::cout << "Logical cores:        " << SysctlInt("hw.logicalcpu") << "\n";
  std::cout << "Memory:               " << HumanBytes(static_cast<unsigned long long>(SysctlInt("hw.memsize")))
            << "\n";
  struct Feature {
    const char* name;
    const char* key;
  };
  const Feature features[] = {
      {"NEON", "hw.optional.neon"},          {"FP16", "hw.optional.arm.FEAT_FP16"},
      {"DotProd", "hw.optional.arm.FEAT_DotProd"}, {"BF16", "hw.optional.arm.FEAT_BF16"},
      {"I8MM", "hw.optional.arm.FEAT_I8MM"}, {"SME", "hw.optional.arm.FEAT_SME"},
      {"AVX", "hw.optional.avx1_0"},         {"AVX2", "hw.optional.avx2_0"},
      {"FMA", "hw.optional.fma"},            {"AVX512F", "hw.optional.avx512f"},
  };
  std::cout << "CPU features:        ";
  for (const auto& f : features)
    if (SysctlInt(f.key) == 1) std::cout << " " << f.name;
  std::cout << "\n";
#elif defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
  struct Feature {
    const char* name;
    bool        on;
  };
  const Feature features[] = {
      {"SSE4.2", __builtin_cpu_supports("sse4.2")}, {"AVX", __builtin_cpu_supports("avx")},
      {"AVX2", __builtin_cpu_supports("avx2")},     {"FMA", __builtin_cpu_supports("fma")},
      {"AVX512F", __builtin_cpu_supports("avx512f")},
  };
  std::cout << "CPU features:        ";
  for (const auto& f : features)
    if (f.on) std::cout << " " << f.name;
  std::cout << "\n";
#else
  std::cout << "CPU features:         (not probed on this platform)\n";
#endif
  return 0;
}

// ---------------------------------------------------------------------------
// NVIDIA GPU via NVML (dlopen'd; no CUDA SDK needed at build time)
// ---------------------------------------------------------------------------

// Minimal subset of nvml.h. The ABI of these is stable across driver releases.
using nvmlDevice_t = struct nvmlDevice_st*;
using nvmlReturn_t = int;  // 0 == NVML_SUCCESS
struct nvmlMemory_t {
  unsigned long long total, free, used;
};
struct nvmlUtilization_t {
  unsigned int gpu, memory;
};
constexpr int kNvmlTemperatureGpu = 0;
constexpr int kNvmlClockGraphics  = 0;
constexpr int kNvmlClockMem       = 2;

struct Nvml {
  void* handle = nullptr;
  nvmlReturn_t (*Init)()                                                         = nullptr;
  nvmlReturn_t (*Shutdown)()                                                     = nullptr;
  const char* (*ErrorString)(nvmlReturn_t)                                       = nullptr;
  nvmlReturn_t (*SystemGetDriverVersion)(char*, unsigned)                        = nullptr;
  nvmlReturn_t (*SystemGetNVMLVersion)(char*, unsigned)                          = nullptr;
  nvmlReturn_t (*SystemGetCudaDriverVersion)(int*)                               = nullptr;
  nvmlReturn_t (*DeviceGetCount)(unsigned*)                                      = nullptr;
  nvmlReturn_t (*DeviceGetHandleByIndex)(unsigned, nvmlDevice_t*)                = nullptr;
  nvmlReturn_t (*DeviceGetName)(nvmlDevice_t, char*, unsigned)                   = nullptr;
  nvmlReturn_t (*DeviceGetMemoryInfo)(nvmlDevice_t, nvmlMemory_t*)               = nullptr;
  nvmlReturn_t (*DeviceGetCudaComputeCapability)(nvmlDevice_t, int*, int*)       = nullptr;
  nvmlReturn_t (*DeviceGetTemperature)(nvmlDevice_t, int, unsigned*)             = nullptr;
  nvmlReturn_t (*DeviceGetUtilizationRates)(nvmlDevice_t, nvmlUtilization_t*)   = nullptr;
  nvmlReturn_t (*DeviceGetPowerUsage)(nvmlDevice_t, unsigned*)                   = nullptr;
  nvmlReturn_t (*DeviceGetClockInfo)(nvmlDevice_t, int, unsigned*)               = nullptr;
  nvmlReturn_t (*DeviceGetNumGpuCores)(nvmlDevice_t, unsigned*)                  = nullptr;

  template <typename F>
  bool Load(F& fn, const char* name) {
    fn = reinterpret_cast<F>(dlsym(handle, name));
    return fn != nullptr;
  }

  bool Open() {
    // NixOS keeps driver libraries under /run/opengl-driver; other distros put
    // them on the default search path.
    const char* candidates[] = {
        "libnvidia-ml.so.1",
        "/run/opengl-driver/lib/libnvidia-ml.so.1",
        "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1",
        "/usr/lib64/libnvidia-ml.so.1",
    };
    for (const char* c : candidates) {
      handle = dlopen(c, RTLD_NOW | RTLD_LOCAL);
      if (handle) break;
    }
    if (!handle) return false;
    return Load(Init, "nvmlInit_v2") && Load(Shutdown, "nvmlShutdown") && Load(ErrorString, "nvmlErrorString") &&
           Load(SystemGetDriverVersion, "nvmlSystemGetDriverVersion") &&
           Load(SystemGetNVMLVersion, "nvmlSystemGetNVMLVersion") &&
           Load(SystemGetCudaDriverVersion, "nvmlSystemGetCudaDriverVersion") &&
           Load(DeviceGetCount, "nvmlDeviceGetCount_v2") &&
           Load(DeviceGetHandleByIndex, "nvmlDeviceGetHandleByIndex_v2") && Load(DeviceGetName, "nvmlDeviceGetName") &&
           Load(DeviceGetMemoryInfo, "nvmlDeviceGetMemoryInfo") &&
           Load(DeviceGetCudaComputeCapability, "nvmlDeviceGetCudaComputeCapability") &&
           Load(DeviceGetTemperature, "nvmlDeviceGetTemperature") &&
           Load(DeviceGetUtilizationRates, "nvmlDeviceGetUtilizationRates") &&
           Load(DeviceGetPowerUsage, "nvmlDeviceGetPowerUsage") && Load(DeviceGetClockInfo, "nvmlDeviceGetClockInfo") &&
           Load(DeviceGetNumGpuCores, "nvmlDeviceGetNumGpuCores");
  }

  ~Nvml() {
    if (handle) dlclose(handle);
  }
};

int ReportGpu() {
  Header("GPU (via NVIDIA NVML)");
  Nvml nvml;
  if (!nvml.Open()) {
    std::cout << "libnvidia-ml.so.1 not found or incomplete; no NVIDIA driver available.\n";
    return 0;
  }
  nvmlReturn_t rc = nvml.Init();
  if (rc != 0) {
    std::cout << "nvmlInit failed: " << nvml.ErrorString(rc) << "\n";
    return 0;
  }

  char buf[128];
  if (nvml.SystemGetDriverVersion(buf, sizeof buf) == 0) std::cout << "Driver version:       " << buf << "\n";
  if (nvml.SystemGetNVMLVersion(buf, sizeof buf) == 0) std::cout << "NVML version:         " << buf << "\n";
  int cuda = 0;
  if (nvml.SystemGetCudaDriverVersion(&cuda) == 0)
    std::cout << "CUDA driver supports: CUDA " << cuda / 1000 << "." << (cuda % 1000) / 10 << "\n";

  unsigned count = 0;
  rc             = nvml.DeviceGetCount(&count);
  if (rc != 0) {
    std::cout << "nvmlDeviceGetCount failed: " << nvml.ErrorString(rc) << "\n";
    nvml.Shutdown();
    return 0;
  }
  std::cout << "GPU count:            " << count << " (visible to the driver; nn does not use them)\n";

  for (unsigned i = 0; i < count; ++i) {
    nvmlDevice_t dev = nullptr;
    if (nvml.DeviceGetHandleByIndex(i, &dev) != 0) continue;
    std::cout << "GPU " << i << ":\n";
    if (nvml.DeviceGetName(dev, buf, sizeof buf) == 0) std::cout << "    name:               " << buf << "\n";
    int major = 0, minor = 0;
    if (nvml.DeviceGetCudaComputeCapability(dev, &major, &minor) == 0)
      std::cout << "    compute capability: " << major << "." << minor << "\n";
    unsigned cores = 0;
    if (nvml.DeviceGetNumGpuCores(dev, &cores) == 0) std::cout << "    CUDA cores:         " << cores << "\n";
    nvmlMemory_t mem{};
    if (nvml.DeviceGetMemoryInfo(dev, &mem) == 0)
      std::cout << "    memory:             " << HumanBytes(mem.used) << " used / " << HumanBytes(mem.total)
                << " total\n";
    unsigned clock = 0;
    if (nvml.DeviceGetClockInfo(dev, kNvmlClockGraphics, &clock) == 0)
      std::cout << "    graphics clock:     " << clock << " MHz\n";
    if (nvml.DeviceGetClockInfo(dev, kNvmlClockMem, &clock) == 0)
      std::cout << "    memory clock:       " << clock << " MHz\n";
    unsigned temp = 0;
    if (nvml.DeviceGetTemperature(dev, kNvmlTemperatureGpu, &temp) == 0)
      std::cout << "    temperature:        " << temp << " C\n";
    unsigned mw = 0;
    if (nvml.DeviceGetPowerUsage(dev, &mw) == 0)
      std::cout << "    power draw:         " << std::fixed << std::setprecision(1) << mw / 1000.0 << " W\n";
    nvmlUtilization_t util{};
    if (nvml.DeviceGetUtilizationRates(dev, &util) == 0)
      std::cout << "    utilization:        " << util.gpu << "% gpu, " << util.memory << "% memory\n";
  }
  nvml.Shutdown();
  return 0;
}

}  // namespace

int main() {
  try {
    int rc = ReportNn();
    if (rc != 0) return rc;
    return ReportGpu();
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
