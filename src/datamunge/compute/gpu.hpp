#pragma once
/// @file gpu.hpp
/// @brief Private backend-neutral GPU allocation and kernel execution boundary.
#include <array>
#include <datamunge/compute/context.hpp>
#include <memory>
#include <stdexcept>
namespace datamunge::compute::detail {
/// @brief Opaque backend-owned device allocation.
struct GpuAllocation {
  /// @brief Release the backend allocation through its concrete owner.
  virtual ~GpuAllocation() = default;
};
/// @brief Opaque compiled native GPU program.
struct GpuProgram {
  /// @brief Release the concrete module, pipeline and binding resources.
  virtual ~GpuProgram() = default;
};
/// @brief Bound allocation view passed to a GPU program.
struct GpuArgument {
  std::shared_ptr<GpuAllocation> allocation; ///< Backend resource.
  std::size_t                    offset;     ///< Offset in bytes.
  std::size_t                    size;       ///< Accessible logical bytes.
};
/// @brief Backend-owned resources; all calls are serialized by the public context.
struct GpuDevice {
  /// @brief Release the native device context after every resource owner has finished.
  virtual ~GpuDevice() = default;
  /// @brief Return the driver-reported device description.
  /// @return Owned device name.
  virtual std::string name() const = 0;
  /// @brief Allocate native storage, retaining a physical sentinel for logical zero bytes.
  /// @param bytes Logical allocation size.
  /// @return Backend resource retained by the public allocation owner.
  virtual std::shared_ptr<GpuAllocation> allocate(std::size_t bytes) = 0;
  /// @brief Copy validated host bytes into a native allocation view.
  virtual void upload(const GpuArgument&, const void*, std::size_t) = 0;
  /// @brief Copy a validated native allocation view into caller host memory.
  virtual void download(void*, const GpuArgument&, std::size_t) = 0;
  /// @brief Copy disjoint validated native ranges without host numerical evaluation.
  virtual void copy(const GpuArgument&, const GpuArgument&, std::size_t) = 0;
  /// @brief Initialize native bytes to one exact byte value.
  virtual void fill(const GpuArgument&, std::uint8_t) = 0;
  /// @brief Report whether this backend was compiled with a numerical matrix adapter.
  /// @return False unless a concrete backend overrides the capability.
  virtual bool supports_matmul() const noexcept { return false; }
  /// @brief Overwrite a validated packed output using backend-native matrix arithmetic.
  /// @throws std::logic_error Backend lacks the matrix adapter.
  virtual void
  matmul(const GpuArgument&, const GpuArgument&, const GpuArgument&, const MatrixProduct&) {
    throw std::logic_error("native matrix adapter unavailable; enable CUDA and cuBLAS");
  }
  /// @brief Compile the selected native source into a retained backend program.
  virtual std::shared_ptr<GpuProgram> program(const KernelSource&) = 0;
  /// @brief Bind explicit resources/parameters and synchronously dispatch native workgroups.
  virtual void dispatch(
      GpuProgram&,
      const std::vector<GpuArgument>&,
      const void*,
      std::size_t,
      std::array<std::uint32_t, 3>) = 0;
  /// @brief Wait for native execution and surface pending driver errors.
  virtual void synchronize() = 0;
};
/// @brief Construct an explicit CUDA backend or throw an availability diagnostic.
std::shared_ptr<GpuDevice> make_cuda(std::uint32_t index);
/// @brief Construct an explicit Vulkan backend or throw an availability diagnostic.
std::shared_ptr<GpuDevice> make_vulkan(std::uint32_t index);
} // namespace datamunge::compute::detail
