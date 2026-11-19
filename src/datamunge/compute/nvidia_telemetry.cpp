/// @file nvidia_telemetry.cpp
/// @brief Runtime NVML loading without requiring the CUDA SDK.

#include <datamunge/compute/nvidia_telemetry.hpp>

#if defined(__linux__)
#include <dlfcn.h>
#include <array>
#include <memory>
#endif

namespace datamunge::compute {

#if defined(__linux__)
namespace {

/// @brief Opaque NVML device handle.
using DeviceHandle = struct nvmlDevice_st*;
/// @brief NVML status code, zero on success.
using Status = int;

/// @brief ABI layout returned by nvmlDeviceGetMemoryInfo.
struct Memory {
  unsigned long long total; ///< Total framebuffer bytes.
  unsigned long long free; ///< Unused framebuffer bytes.
  unsigned long long used; ///< Used framebuffer bytes.
};

/// @brief ABI layout returned by nvmlDeviceGetUtilizationRates.
struct Utilization {
  unsigned gpu; ///< Percent of sample interval running kernels.
  unsigned memory; ///< Percent of sample interval accessing memory.
};

/// @brief Load a typed NVML function pointer or null for a missing symbol.
template<class Function>
Function symbol(void* library, const char* name) {
  return reinterpret_cast<Function>(dlsym(library, name));
}

/// @brief Run a single-output query with optional leading arguments.
template<class Value, class... Args>
std::optional<Value> read_metric(void* library, const char* name, Args... args) {
  const auto query = symbol<Status (*)(Args..., Value*)>(library, name);
  Value value{};
  if (query && query(args..., &value) == 0) return value;
  return std::nullopt;
}

/// @brief Query a version or device name into a fixed NVML-sized buffer.
template<class... Args>
std::string read_string(void* library, const char* name, Args... args) {
  const auto query = symbol<Status (*)(Args..., char*, unsigned)>(library, name);
  std::array<char, 256> value{};
  if (query && query(args..., value.data(), value.size()) == 0) return value.data();
  return {};
}

/// @brief Release NVML initialization before unloading the shared library.
struct Session {
  Status (*shutdown)(); ///< Matching shutdown function for one successful initialization.
  ~Session() { (void)shutdown(); } ///< Drop one NVML initialization reference.
};

/// @brief Query one enumerated device, leaving unsupported metrics empty.
NvidiaDeviceTelemetry read_device(void* library, unsigned index, DeviceHandle handle) {
  NvidiaDeviceTelemetry out;
  out.index = index;
  out.name = read_string(library, "nvmlDeviceGetName", handle);
  const auto capability =
      symbol<Status (*)(DeviceHandle, int*, int*)>(library, "nvmlDeviceGetCudaComputeCapability");
  int major = 0;
  int minor = 0;
  if (capability && capability(handle, &major, &minor) == 0)
    out.compute_capability = {major, minor};
  out.cores = read_metric<unsigned>(library, "nvmlDeviceGetNumGpuCores", handle);
  if (const auto memory = read_metric<Memory>(library, "nvmlDeviceGetMemoryInfo", handle)) {
    out.memory_total_bytes = memory->total;
    out.memory_used_bytes = memory->used;
  }
  out.graphics_clock_mhz = read_metric<unsigned>(library, "nvmlDeviceGetClockInfo", handle, 0);
  out.memory_clock_mhz = read_metric<unsigned>(library, "nvmlDeviceGetClockInfo", handle, 2);
  out.temperature_celsius = read_metric<unsigned>(library, "nvmlDeviceGetTemperature", handle, 0);
  if (const auto power = read_metric<unsigned>(library, "nvmlDeviceGetPowerUsage", handle))
    out.power_watts = *power / 1000.;
  if (const auto utilization = read_metric<Utilization>(library, "nvmlDeviceGetUtilizationRates", handle)) {
    out.gpu_utilization_percent = utilization->gpu;
    out.memory_utilization_percent = utilization->memory;
  }
  return out;
}

} // namespace
#endif

NvidiaTelemetry nvidia_telemetry() {
  NvidiaTelemetry out;
#if defined(__linux__)
  const auto close = [](void* handle) { (void)dlclose(handle); };
  std::unique_ptr<void, decltype(close)> library(dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL), close);
  if (!library)
    library.reset(dlopen("/run/opengl-driver/lib/libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL));
  if (!library) {
    out.diagnostic = "NVML library unavailable";
    return out;
  }
  const auto init = symbol<Status (*)()>(library.get(), "nvmlInit_v2");
  const auto shutdown = symbol<Status (*)()>(library.get(), "nvmlShutdown");
  const auto get_device = symbol<Status (*)(unsigned, DeviceHandle*)>(library.get(), "nvmlDeviceGetHandleByIndex_v2");
  if (!init || !shutdown || !get_device) {
    out.diagnostic = "NVML library lacks required functions";
    return out;
  }
  if (const auto status = init(); status != 0) {
    out.diagnostic = "NVML initialization failed (status " + std::to_string(status) + ")";
    return out;
  }
  const Session session{shutdown};
  const auto count = read_metric<unsigned>(library.get(), "nvmlDeviceGetCount_v2");
  if (!count) {
    out.diagnostic = "NVML could not enumerate devices";
    return out;
  }
  out.driver_version = read_string(library.get(), "nvmlSystemGetDriverVersion");
  out.nvml_version = read_string(library.get(), "nvmlSystemGetNVMLVersion");
  if (const auto version = read_metric<int>(library.get(), "nvmlSystemGetCudaDriverVersion"))
    out.cuda_version = {*version / 1000, (*version % 1000) / 10};
  for (unsigned i = 0; i < *count; ++i) {
    DeviceHandle handle = nullptr;
    if (get_device(i, &handle) == 0)
      out.devices.push_back(read_device(library.get(), i, handle));
    else {
      NvidiaDeviceTelemetry unavailable;
      unavailable.index = i;
      out.devices.push_back(std::move(unavailable));
    }
  }
#else
  out.diagnostic = "NVML telemetry currently requires a Linux driver loader";
#endif
  return out;
}

} // namespace datamunge::compute
