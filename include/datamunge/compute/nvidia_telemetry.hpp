#pragma once

/// @file nvidia_telemetry.hpp
/// @brief Optional NVIDIA Management Library hardware discovery and measurements.

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace datamunge::compute {

/// @brief Measurements for one NVIDIA device at the time of an NVML query.
///
/// Empty optionals indicate metrics unavailable on the device or driver.
/// @par Example
/// @code
/// const dm::compute::NvidiaDeviceTelemetry device;
/// std::println("{}", device.power_watts.has_value()); // false
/// @endcode
/// @see nvidia_telemetry()
struct NvidiaDeviceTelemetry {
  unsigned index{}; ///< NVML ordinal, independent of CUDA ordinal.
  std::string name; ///< Device name when available.
  std::optional<std::pair<int, int>> compute_capability; ///< CUDA capability major and minor.
  std::optional<unsigned> cores; ///< Number of CUDA cores.
  std::optional<std::uint64_t> memory_total_bytes; ///< Total framebuffer bytes.
  std::optional<std::uint64_t> memory_used_bytes; ///< Used framebuffer bytes.
  std::optional<unsigned> graphics_clock_mhz; ///< Graphics clock in MHz.
  std::optional<unsigned> memory_clock_mhz; ///< Memory clock in MHz.
  std::optional<unsigned> temperature_celsius; ///< GPU temperature in degrees Celsius.
  std::optional<double> power_watts; ///< GPU and associated circuitry power in watts.
  std::optional<unsigned> gpu_utilization_percent; ///< Recent time spent executing GPU kernels, in percent.
  std::optional<unsigned> memory_utilization_percent; ///< Recent time spent accessing device memory, in percent.
};

/// @brief Owned NVIDIA driver information and per-device telemetry.
/// @par Example
/// @code
/// const dm::compute::NvidiaTelemetry telemetry;
/// std::println("{}", telemetry.devices.size()); // 0
/// @endcode
/// @see nvidia_telemetry()
struct NvidiaTelemetry {
  std::string driver_version; ///< NVIDIA driver version when available.
  std::string nvml_version; ///< NVML version when available.
  std::optional<std::pair<int, int>> cuda_version; ///< Driver-supported CUDA major and minor.
  std::vector<NvidiaDeviceTelemetry> devices; ///< Devices detected by NVML.
  std::string diagnostic; ///< Explanation if the library, initialization or enumeration failed.
};

/// @brief Query NVIDIA devices and their current readings through the installed NVML.
///
/// The loader is optional, and this query works independently of CUDA compute
/// support and CUDA_VISIBLE_DEVICES. Each metric is sampled separately. NVML
/// may refuse individual measurements or access to specific devices.
/// @return An owned snapshot with optional metrics and any enumeration diagnostic.
///
/// @par Example
/// @code
/// const auto result = dm::compute::nvidia_telemetry();
/// if (!result.diagnostic.empty()) std::println("{}", result.diagnostic);
/// for (const auto& device : result.devices) std::println("{}", device.name);
/// @endcode
/// @par References
/// - NVIDIA NVML API, https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html
/// @see device_count(), probe()
NvidiaTelemetry nvidia_telemetry();

} // namespace datamunge::compute
