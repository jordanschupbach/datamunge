#pragma once
/// @file device_storage.hpp
/// @brief Private native tensor packing/fill kernels and shared default CPU ownership.
#include <datamunge/nn/tensor.hpp>

#include <map>
#include <mutex>
#include <string>
namespace datamunge::nn::detail {
/// @brief One retained native pipeline; allocations on its context can share it.
/// Numerical kernels compiled on the same context are cached here too, so they
/// live exactly as long as some tensor on that context retains this object.
struct StorageProgram {
  compute::Context                        context; ///< Retains the creating execution owner.
  compute::Program                        program; ///< Native storage pipeline.
  std::mutex                              lock;    ///< Guards kernels.
  std::map<std::string, compute::Program> kernels; ///< Numerical entries by name.
};
/// @brief Acquire the storage's retained pipeline, compiling or sharing it on first use.
/// @param storage GPU storage on the context that will run native kernels.
/// @return Pipeline owner retained by that storage.
StorageProgram& storage_program(Storage& storage);
/// @brief Return the process default CPU context without creating workers per allocation.
/// @return Shared single-lane context; allocations retain its ownership independently.
const compute::Context& default_tensor_context();
/// @brief Pack a valid GPU view into independent native storage without host evaluation.
/// @param source Defined GPU tensor with valid view metadata.
/// @return Packed tensor on the source's retained context, without recorded history.
/// @throws std::runtime_error Native compilation, dispatch or binding-limit failure.
Tensor pack_device(const Tensor& source);
/// @brief Fill a newly allocated packed GPU tensor with an exact scalar representation.
/// @param destination Packed offset-zero GPU tensor; no aliases may write concurrently.
/// @param pattern Host scalar bytes, little endian on supported GPU hosts.
/// @param width Dtype size, one, four or eight bytes.
/// @throws std::runtime_error Native compilation/dispatch failure or unsupported host endian.
void fill_device(Tensor& destination, const void* pattern, std::size_t width);
} // namespace datamunge::nn::detail
