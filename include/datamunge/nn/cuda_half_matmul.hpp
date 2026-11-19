#pragma once

/// @file cuda_half_matmul.hpp
/// @brief CUDA matrix products with IEEE binary16 operands and outputs.

#include <datamunge/nn/tensor.hpp>

namespace datamunge::nn {

/// @brief Run dense products on CUDA with packed binary16 inputs and Float32 accumulation.
///
/// Host Float32 inputs are converted to binary16 on the GPU. Each result
/// is rounded to binary16 before widening to Float32 for host access.
/// The context retains the compiled kernels across calls. No host fallback
/// or automatic differentiation is performed.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const dm::compute::Context gpu({dm::compute::Backend::CUDA, 0});
/// const n::CudaHalfMatmul product(gpu);
/// std::println("{}", product(n::ones({1, 2}), n::ones({2, 1})).item()); // 2
/// @endcode
/// @see round_float16(), DynamicLossScale
class CudaHalfMatmul {
 public:
  /// @brief Compile the binary16 conversion and product kernels.
  /// @param context CUDA context retaining the device and native programs.
  /// @throws std::invalid_argument A non-CUDA context.
  /// @throws std::runtime_error Driver failure or unsupported native instructions.
  /// @par Example
  /// @code
  /// const dm::nn::CudaHalfMatmul product(dm::compute::Context({dm::compute::Backend::CUDA, 0}));
  /// @endcode
  explicit CudaHalfMatmul(const compute::Context& context);

  /// @brief Compute A times B, optionally transposing the first matrix.
  /// @param a Nonempty CPU Float32 matrix.
  /// @param b Nonempty CPU Float32 matrix with compatible inner dimension.
  /// @param transpose_a Use the transpose of A, default false.
  /// @return Owning CPU Float32 tensor containing the rounded binary16 result.
  /// @throws std::invalid_argument Invalid shape, dtype, device, or kernel index range.
  /// @throws std::runtime_error Native allocation, transfer, or execution failure.
  /// @par Example
  /// @code
  /// namespace n = dm::nn;
  /// const n::CudaHalfMatmul product(dm::compute::Context({dm::compute::Backend::CUDA, 0}));
  /// std::println("{}", product(n::ones({2, 1}),n::ones({2, 1}),true).item()); // 2
  /// @endcode
  [[nodiscard]] Tensor operator()(const Tensor& a, const Tensor& b, bool transpose_a = false) const;

 private:
  compute::Context context_; ///< CUDA allocation and launch context.
  compute::Program pack_; ///< GPU Float32-to-binary16 conversion program.
  compute::Program product_; ///< GPU product with binary16 output rounding.
};

} // namespace datamunge::nn
