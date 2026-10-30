#pragma once
/// @file mixed_precision.hpp
/// @brief CPU emulation of low-precision storage, rounded VJPs and host-driven loss scaling.
#include <datamunge/nn/params.hpp>
#include <string>
#include <vector>
namespace datamunge::nn {
/// @brief Value-storage precision emulated on ordinary CPU floating tensors, not hardware dtypes.
/// Float16/BFloat16 round values to their representable sets but keep Float32 tensor
/// storage and arithmetic. No GPU, memory-size reduction or throughput gain is promised.
/// @par Example
/// @code
/// std::print("{}\n",dm::nn::precision_name(dm::nn::PrecisionEmulation::Float16)); // float16
/// @endcode
/// @par References
/// - https://www.tensorflow.org/guide/mixed_precision
/// @see quantize_precision(), DynamicLossScale
enum class PrecisionEmulation {
  Float32, ///< Identity mode, without reduced-precision rounding.
  Float16, ///< Binary16: five exponent bits, ten stored fraction bits, gradual underflow.
  BFloat16 ///< Eight exponent bits and seven stored fraction bits.
};
/// @brief Names the emulated storage format.
/// @param precision Valid precision enumerator.
/// @return Owned float32, float16 or bfloat16 label.
/// @throws std::invalid_argument Unknown enumerator.
/// @par Example
/// See PrecisionEmulation.
std::string precision_name(PrecisionEmulation precision);
/// @brief Rounds a float to binary16's value set, returning the result in a float.
/// Round-to-nearest, ties-to-even is independent of the ambient rounding mode.
/// Subnormals are retained, signed zero is preserved, overflow becomes signed infinity,
/// and input infinities/NaNs are passed through. No mutation/shared state; O(1).
/// @param value Float32 value, including nonfinite values.
/// @return Rounded value, with no native Float16 tensor allocation.
/// @par Example
/// @code
/// std::print("{}\n",dm::nn::round_float16(1e-8f)==0); // true
/// @endcode
/// @par References
/// - https://www.tensorflow.org/guide/mixed_precision
float round_float16(float value);
/// @brief Rounds a float to bfloat16's value set, ties to even, returned as a float.
/// Preserves signed zero and nonfinite values; finite overflow can become infinity.
/// No mutation/shared state; O(1).
/// @param value Float32 value.
/// @return Float32 carrier of the rounded value.
/// @par Example
/// @code
/// std::print("{:.0f}\n",dm::nn::round_bfloat16(1.001f)); // 1
/// @endcode
/// @par References
/// - https://www.tensorflow.org/guide/mixed_precision
float round_bfloat16(float value);
/// @brief Rounds tensor values for a storage-emulation experiment, without a differentiable cast.
/// @param x Defined Float32/Float64 tensor; shape is preserved.
/// @param precision Valid emulation format.
/// @return Float32 mode aliases x; low-precision modes own a new contiguous Float32 carrier.
/// @throws std::invalid_argument Invalid enum, undefined or nonfloating tensor.
/// @par Example
/// @code
/// const auto x=dm::nn::from_vector(std::vector<float>{1e-8f});
/// std::print("{}\n",dm::nn::round_tensor_precision(x,dm::nn::PrecisionEmulation::Float16).item()==0);
/// // true
/// @endcode
/// @par References
/// - https://www.tensorflow.org/guide/mixed_precision
/// @see quantize_precision()
Tensor round_tensor_precision(const Tensor& x, PrecisionEmulation precision);
/// @brief Custom autograd operation rounding both values and incoming gradients to a format.
/// The VJP rounds the incoming gradient; it is a storage-emulation surrogate, not the
/// mathematical derivative of a piecewise-constant rounding function. Work/storage O(n).
/// @param x Defined Float32 input, borrowed and unchanged.
/// @param precision Valid format; Float32 is identity.
/// @return New rounded Float32 tensor with the stated custom VJP, or x in identity mode.
/// @throws std::invalid_argument Invalid format or non-Float32 input.
/// @par Example
/// @code
/// const auto x=dm::nn::from_vector(std::vector<float>{1}).requires_grad_();
/// const auto q=dm::nn::quantize_precision(x,dm::nn::PrecisionEmulation::Float16);
/// std::print("{}\n",dm::nn::grad(dm::nn::sum(q)*1e-8,{x})[0].item()==0); // true
/// @endcode
/// @par References
/// - https://www.tensorflow.org/guide/mixed_precision
/// @see unscale_gradients()
Tensor quantize_precision(const Tensor& x, PrecisionEmulation precision);
/// @brief Dense layer with Float32 master parameters and rounded compute-edge storage.
/// Rounds input, weights, matmul output, bias and bias-add output at the selected format;
/// the underlying arithmetic still runs in Float32. Creates named Glorot-uniform W and
/// zero b on first call; capture optimizer parameters after that creation. The store's
/// initializer RNG advances only for newly created parameters. No shared global state.
/// @param store Mutable Float32 parameter store, borrowing existing names on later calls.
/// @param name Layer prefix, producing name/W and name/b.
/// @param x Float32 matrix [N,in].
/// @param in Positive input width.
/// @param out Positive output width.
/// @param precision Emulated storage format.
/// @return Differentiable [N,out] Float32 output.
/// @throws std::invalid_argument Invalid schema, shape, dtype or precision.
/// @par Example
/// @code
/// dm::nn::ParameterStore p;
/// const auto
/// y=dm::nn::mixed_dense(p,"d",dm::nn::zeros({2,3}),3,1,dm::nn::PrecisionEmulation::Float16);
/// std::print("{}\n",y.shape()[0]); // 2
/// @endcode
/// @par References
/// - https://www.tensorflow.org/guide/mixed_precision
/// @see quantize_precision()
Tensor mixed_dense(
    ParameterStore&    store,
    const std::string& name,
    const Tensor&      x,
    int                in,
    int                out,
    PrecisionEmulation precision);
/// @brief Host loss-scale policy, advanced only after explicitly applied/skipped updates.
/// A good-step interval doubles the scale up to maximum. overflow() halves it down
/// to minimum and clears the good-step count. Neither call changes a model/optimizer.
/// Value-owned state; serialize it separately for exact dynamic-scale resume. Do not
/// mutate one instance concurrently. No GPU or automatic optimizer integration.
/// @par Example
/// @code
/// dm::nn::DynamicLossScale s(128,2,1,256);s.applied();s.applied();s.overflow();
/// std::print("{:.0f}\n",s.value()); // 128
/// @endcode
/// @par References
/// - https://www.tensorflow.org/guide/mixed_precision
/// @see unscale_gradients()
class DynamicLossScale {
 public:
  /// @brief Creates a bounded scale with a successful-update growth interval.
  /// @param initial Finite starting scale, default 128, within the stated bounds.
  /// @param growth_interval Positive successful-update count, default 2000.
  /// @param minimum Finite positive lower bound, default one.
  /// @param maximum Finite upper bound no smaller than initial, default 65536.
  /// @throws std::invalid_argument Invalid bounds, scale or interval.
  /// @par Example
  /// See the class example.
  explicit DynamicLossScale(
      float initial         = 128,
      int   growth_interval = 2000,
      float minimum         = 1,
      float maximum         = 65536);
  /// @brief Current loss multiplier.
  /// @return Finite positive current scale.
  float value() const { return scale_; }
  /// @brief Backs off after a skipped nonfinite-gradient attempt, clearing good-step history.
  /// @par Example
  /// See the class example.
  void overflow();
  /// @brief Records one applied update and grows scale when the interval is reached.
  /// @par Example
  /// See the class example.
  void applied();

 private:
  float scale_;    ///< Current bounded positive multiplier.
  float minimum_;  ///< Positive backoff floor.
  float maximum_;  ///< Finite growth ceiling.
  int   interval_; ///< Positive count required to grow.
  int   good_{0};  ///< Consecutive successful calls since backoff/growth.
};
/// @brief Unscales gradients into independent Float32 buffers, checking all values are finite.
/// @param scaled Defined Float32 gradient tensors; an empty list succeeds with empty output.
/// @param scale Finite positive loss multiplier.
/// @param result Non-null output pointer. Success replaces it; a nonfinite value clears it.
/// @return True if all unscaled entries are finite; false otherwise. No optimizer call occurs.
/// @throws std::invalid_argument Invalid scale, null result, undefined or non-Float32 gradient;
/// argument errors leave result unchanged.
/// @par Example
/// @code
/// std::vector<dm::nn::Tensor> g;
/// const bool ok=dm::nn::unscale_gradients({dm::nn::from_vector(std::vector<float>{2})},2,&g);
/// std::print("{} {:.0f}\n",ok,g[0].item()); // true 1
/// @endcode
/// @par References
/// - https://www.tensorflow.org/guide/mixed_precision
/// @see quantize_precision(), DynamicLossScale
bool unscale_gradients(const std::vector<Tensor>& scaled, float scale, std::vector<Tensor>* result);
} // namespace datamunge::nn
