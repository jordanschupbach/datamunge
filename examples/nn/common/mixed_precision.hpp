// Emulated float16/bfloat16 compute, mixed-precision dense layers and
// host-driven dynamic loss scaling (port of tflow's common/nn/mixed_precision.h).
//
// nn has no float16 or bfloat16 dtype (the plan keeps them out of scope), so
// low precision is emulated on float32 tensors: `Quantize` rounds every value
// to the nearest representable float16 / bfloat16 (round to nearest even,
// float16 overflow to +-inf and gradual underflow through subnormals) in the
// forward pass AND rounds the incoming gradient in the backward pass. Placing
// it at every edge TF would type as half (casts, matmul and bias outputs)
// reproduces what the TF graph stores in half precision: tiny gradients flush
// to zero, scaled ones overflow. Arithmetic inside one op still accumulates in
// float32, which TF's CPU half kernels mostly do too.
//
// Dense weights and bias stay float32 master copies; read values and inputs
// are quantized for the matmul/bias add, predictions are float32 for the loss.
// Differentiate scale*loss, unscale its gradients, and step the optimizer only
// when all are finite, so a skipped update leaves weights, moments and the
// optimizer counter untouched.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <datamunge/nn/nn.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace mixed {

namespace nn = datamunge::nn;

enum class Precision { Float32, Float16, BFloat16 };

inline std::string PrecisionName(Precision p) {
  return p == Precision::Float32 ? "float32" : p == Precision::Float16 ? "float16" : "bfloat16";
}

/// float -> IEEE binary16 -> float, round to nearest even (as Eigen::half).
inline float RoundHalf(float value) {
  if (!std::isfinite(value)) return value;
  std::uint32_t bits;
  std::memcpy(&bits, &value, 4);
  const std::uint32_t sign = bits & 0x80000000u;
  const float         a    = std::fabs(value);
  float               out;
  if (a >= 65520.f) {  // rounds past the largest half (65504)
    out = INFINITY;
  } else if (a < 6.103515625e-05f) {  // subnormal half: multiples of 2^-24
    out = std::nearbyint(a * 16777216.f) / 16777216.f;  // default rounding mode is to nearest even
  } else {
    // Normal half: keep 10 mantissa bits of the float's 23.
    std::uint32_t ab;
    std::memcpy(&ab, &a, 4);
    const std::uint32_t lsb = (ab >> 13) & 1u;
    ab += 0x0FFFu + lsb;
    ab &= ~0x1FFFu;
    std::memcpy(&out, &ab, 4);
    if (out > 65504.f) out = INFINITY;
  }
  std::uint32_t ob;
  std::memcpy(&ob, &out, 4);
  ob |= sign;
  std::memcpy(&out, &ob, 4);
  return out;
}

/// float -> bfloat16 -> float, round to nearest even (as Eigen::bfloat16).
inline float RoundBFloat16(float value) {
  if (std::isnan(value)) return value;
  std::uint32_t bits;
  std::memcpy(&bits, &value, 4);
  bits += 0x7FFFu + ((bits >> 16) & 1u);
  bits &= 0xFFFF0000u;
  float out;
  std::memcpy(&out, &bits, 4);
  return out;
}

inline nn::Tensor RoundTensor(const nn::Tensor& x, Precision p) {
  if (p == Precision::Float32) return x;
  auto v = x.contiguous().to(nn::DType::Float32).to_vector<float>();
  for (auto& e : v) e = p == Precision::Float16 ? RoundHalf(e) : RoundBFloat16(e);
  return nn::from_vector(v, {x.numel()}).reshape(x.shape());
}

/// Rounds values (forward) and gradients (backward) to `p`: one tensor that
/// TF would hold in half precision.
inline nn::Tensor Quantize(const nn::Tensor& x, Precision p) {
  if (p == Precision::Float32) return x;
  const nn::Function f(
      p == Precision::Float16 ? "QuantizeFloat16" : "QuantizeBFloat16",
      [p](const std::vector<nn::Tensor>& in) { return std::vector<nn::Tensor>{RoundTensor(in[0], p)}; },
      [p](const std::vector<nn::Tensor>&, const std::vector<nn::Tensor>&, const std::vector<nn::Tensor>& g) {
        return std::vector<nn::Tensor>{RoundTensor(g[0], p)};
      });
  return f(x);
}

/// x [N, in] @ W [in, out] + b with W, b float32 masters in `store`
/// (W Glorot-uniform, b zero) and the compute in `compute` precision.
inline nn::Tensor MixedDense(nn::ParameterStore& store, const std::string& name, const nn::Tensor& x, int in,
                             int out, Precision compute) {
  const auto w = store.get(name + "/W", {in, out}, nn::init::uniform(std::sqrt(6.0 / (in + out))));
  const auto b = store.get(name + "/b", {out}, nn::init::zeros());
  const auto q = [compute](const nn::Tensor& t) { return Quantize(t, compute); };
  return q(nn::bias_add(q(nn::matmul(q(x), q(w))), q(b)));
}

class DynamicLossScale {
 public:
  explicit DynamicLossScale(float initial = 128, int growth_interval = 2000, float minimum = 1, float maximum = 65536)
      : scale_(initial), minimum_(minimum), maximum_(maximum), interval_(growth_interval) {
    if (!std::isfinite(initial) || !std::isfinite(minimum) || !std::isfinite(maximum) || minimum <= 0 ||
        initial < minimum || initial > maximum || growth_interval <= 0)
      throw std::invalid_argument("invalid dynamic loss-scale range or growth interval");
  }
  float value() const { return scale_; }
  void  Overflow() {
    scale_ = std::max(minimum_, scale_ / 2);
    good_  = 0;
  }
  void Applied() {
    if (++good_ == interval_) {
      scale_ = std::min(maximum_, scale_ * 2);
      good_  = 0;
    }
  }

 private:
  float scale_, minimum_, maximum_;
  int   interval_, good_ = 0;
};

/// Divides every float32 gradient by `scale` into fresh tensors; false (and
/// an empty result) when any unscaled value is non-finite.
inline bool UnscaleGradients(const std::vector<nn::Tensor>& scaled, float scale, std::vector<nn::Tensor>* result) {
  if (!std::isfinite(scale) || scale <= 0) throw std::invalid_argument("loss scale must be positive");
  result->clear();
  for (const auto& tensor : scaled) {
    if (tensor.dtype() != nn::DType::Float32) throw std::invalid_argument("master gradients must be float32");
    auto v = tensor.contiguous().to_vector<float>();
    for (auto& e : v) {
      e /= scale;
      if (!std::isfinite(e)) {
        result->clear();
        return false;
      }
    }
    result->push_back(nn::from_vector(v, {tensor.numel()}).reshape(tensor.shape()));
  }
  return true;
}

}  // namespace mixed
