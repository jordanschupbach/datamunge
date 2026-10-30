#pragma once
/// @file mixed_precision.hpp
/// @brief Compatibility forwards for existing examples; algorithms live in the native core.
#include <datamunge/nn/mixed_precision.hpp>
namespace mixed {
/// @brief Native neural-network API used by these compatibility forwards.
namespace nn = datamunge::nn;
/// @brief Existing example spelling for the native storage-emulation enum.
using Precision = nn::PrecisionEmulation;
/// @brief Forwards the old example spelling to the native precision label.
/// @param p Valid native precision enumerator; see nn::precision_name().
/// @return Owned native format label, with the native validation contract.
inline std::string PrecisionName(Precision p) {
  return nn::precision_name(p);
}
/// @brief Forwards binary16 rounding to the core implementation.
/// @param v Float32 value, including nonfinite values; see nn::round_float16().
/// @return Float32 carrier of the binary16-rounded value.
inline float RoundHalf(float v) {
  return nn::round_float16(v);
}
/// @brief Forwards bfloat16 rounding to the core implementation.
/// @param v Float32 value, including nonfinite values; see nn::round_bfloat16().
/// @return Float32 carrier of the bfloat16-rounded value.
inline float RoundBFloat16(float v) {
  return nn::round_bfloat16(v);
}
/// @brief Forwards tensor value rounding to the core implementation.
/// @param x Defined floating tensor, borrowed unchanged.
/// @param p Native format; see nn::round_tensor_precision() for validation/aliasing.
/// @return Rounded Float32 carrier, or the original tensor in identity mode.
inline nn::Tensor RoundTensor(const nn::Tensor& x, Precision p) {
  return nn::round_tensor_precision(x, p);
}
/// @brief Forwards the value/gradient storage-emulation operation.
/// @param x Float32 tensor, borrowed unchanged.
/// @param p Native format; see nn::quantize_precision() for the surrogate VJP.
/// @return Differentiable rounded tensor, or x in identity mode.
inline nn::Tensor Quantize(const nn::Tensor& x, Precision p) {
  return nn::quantize_precision(x, p);
}
/// @brief Forwards the Float32-master dense-layer operation.
/// @param store Mutable Float32 store; parameters are created on first call.
/// @param name Layer prefix, producing name/W and name/b.
/// @param x Float32 matrix [N,in], borrowed unchanged.
/// @param in Positive input width.
/// @param out Positive output width.
/// @param p Emulated storage format; see nn::mixed_dense() for validation.
/// @return Differentiable [N,out] Float32 output.
inline nn::Tensor MixedDense(
    nn::ParameterStore& store,
    const std::string&  name,
    const nn::Tensor&   x,
    int                 in,
    int                 out,
    Precision           p) {
  return nn::mixed_dense(store, name, x, in, out, p);
}
/// @brief Preserves the example's callback spellings while owning the core scale policy.
/// Constructors and policy behavior are inherited from nn::DynamicLossScale.
class DynamicLossScale : public nn::DynamicLossScale {
 public:
  using nn::DynamicLossScale::DynamicLossScale;
  /// @brief Forwards a skipped overflowing attempt.
  void Overflow() { overflow(); }
  /// @brief Forwards an applied optimizer update.
  void Applied() { applied(); }
};
/// @brief Forwards finite-checked gradient unscaling to the core implementation.
/// @param g Defined Float32 gradients, borrowed unchanged.
/// @param scale Positive finite loss multiplier.
/// @param out Non-null output pointer; see nn::unscale_gradients() for mutation/errors.
/// @return True on finite gradients, false on nonfinite entries.
inline bool
UnscaleGradients(const std::vector<nn::Tensor>& g, float scale, std::vector<nn::Tensor>* out) {
  return nn::unscale_gradients(g, scale, out);
}
} // namespace mixed
