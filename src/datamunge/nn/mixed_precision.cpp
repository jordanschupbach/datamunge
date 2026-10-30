/// @file mixed_precision.cpp
/// @brief Deterministic binary16/bfloat16 rounding, emulated compute edges and loss-scale policy.
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <datamunge/nn/autograd.hpp>
#include <datamunge/nn/mixed_precision.hpp>
#include <datamunge/nn/ops.hpp>
#include <limits>
#include <stdexcept>
namespace datamunge::nn {
std::string precision_name(PrecisionEmulation p) {
  switch (p) {
    case PrecisionEmulation::Float32:
      return "float32";
    case PrecisionEmulation::Float16:
      return "float16";
    case PrecisionEmulation::BFloat16:
      return "bfloat16";
  }
  throw std::invalid_argument("unknown precision emulation");
}
float round_float16(float value) {
  if (!std::isfinite(value))
    return value;
  const auto  bits = std::bit_cast<std::uint32_t>(value), sign = bits & 0x80000000u;
  const float a = std::fabs(value);
  float       out;
  if (a >= 65520.f)
    out = std::numeric_limits<float>::infinity();
  else if (a < 6.103515625e-05f) {
    const double scaled = static_cast<double>(a) * 16777216, low = std::floor(scaled),
                 fraction = scaled - low;
    const double rounded  = low + (fraction > .5 || (fraction == .5 && std::fmod(low, 2) != 0));
    out                   = static_cast<float>(rounded / 16777216);
  } else {
    auto ab = std::bit_cast<std::uint32_t>(a);
    ab += 0x0fffu + ((ab >> 13) & 1u);
    ab &= ~0x1fffu;
    out = std::bit_cast<float>(ab);
    if (out > 65504.f)
      out = std::numeric_limits<float>::infinity();
  }
  return std::bit_cast<float>(std::bit_cast<std::uint32_t>(out) | sign);
}
float round_bfloat16(float value) {
  if (std::isnan(value))
    return value;
  auto bits = std::bit_cast<std::uint32_t>(value);
  bits += 0x7fffu + ((bits >> 16) & 1u);
  bits &= 0xffff0000u;
  return std::bit_cast<float>(bits);
}
Tensor round_tensor_precision(const Tensor& x, PrecisionEmulation p) {
  (void)precision_name(p);
  if (!x.defined() || (x.dtype() != DType::Float32 && x.dtype() != DType::Float64))
    throw std::invalid_argument("round_tensor_precision: floating tensor required");
  if (p == PrecisionEmulation::Float32)
    return x;
  auto v = x.contiguous().to(DType::Float32).to_vector<float>();
  for (auto& value : v)
    value = p == PrecisionEmulation::Float16 ? round_float16(value) : round_bfloat16(value);
  return from_vector(v, {x.numel()}).reshape(x.shape());
}
Tensor quantize_precision(const Tensor& x, PrecisionEmulation p) {
  (void)precision_name(p);
  if (!x.defined() || x.dtype() != DType::Float32)
    throw std::invalid_argument("quantize_precision: Float32 input required");
  if (p == PrecisionEmulation::Float32)
    return x;
  const Function f(
      p == PrecisionEmulation::Float16 ? "QuantizeFloat16" : "QuantizeBFloat16",
      [p](const std::vector<Tensor>& in) {
        return std::vector<Tensor>{round_tensor_precision(in[0], p)};
      },
      [p](const std::vector<Tensor>&, const std::vector<Tensor>&, const std::vector<Tensor>& g) {
        return std::vector<Tensor>{round_tensor_precision(g[0], p)};
      });
  return f(x);
}
Tensor mixed_dense(
    ParameterStore&    store,
    const std::string& name,
    const Tensor&      x,
    int                in,
    int                out,
    PrecisionEmulation p) {
  (void)precision_name(p);
  if (store.dtype() != DType::Float32 || !x.defined() || x.dtype() != DType::Float32 || in <= 0 ||
      out <= 0 || x.shape().size() != 2 || x.size(1) != in)
    throw std::invalid_argument(
        "mixed_dense: Float32 [N,in] input/store and positive widths required");
  const auto w = store.get(
      name + "/W", {in, out}, init::uniform(std::sqrt(6. / (static_cast<double>(in) + out))));
  const auto b = store.get(name + "/b", {out}, init::zeros());
  return quantize_precision(
      bias_add(
          quantize_precision(matmul(quantize_precision(x, p), quantize_precision(w, p)), p),
          quantize_precision(b, p)),
      p);
}
DynamicLossScale::DynamicLossScale(float initial, int interval, float minimum, float maximum)
    : scale_(initial), minimum_(minimum), maximum_(maximum), interval_(interval) {
  if (!std::isfinite(initial) || !std::isfinite(minimum) || !std::isfinite(maximum) ||
      minimum <= 0 || initial < minimum || initial > maximum || interval <= 0)
    throw std::invalid_argument("DynamicLossScale: invalid bounds/interval");
}
void DynamicLossScale::overflow() {
  scale_ = std::max(minimum_, scale_ / 2);
  good_  = 0;
}
void DynamicLossScale::applied() {
  if (++good_ == interval_) {
    scale_ = std::min(maximum_, scale_ * 2);
    good_  = 0;
  }
}
bool unscale_gradients(
    const std::vector<Tensor>& scaled,
    float                      scale,
    std::vector<Tensor>*       result) {
  if (!result || !std::isfinite(scale) || scale <= 0)
    throw std::invalid_argument(
        "unscale_gradients: positive finite scale and output pointer required");
  for (const auto& t : scaled)
    if (!t.defined() || t.dtype() != DType::Float32)
      throw std::invalid_argument("unscale_gradients: Float32 gradients required");
  std::vector<Tensor> out;
  for (const auto& t : scaled) {
    auto v = t.contiguous().to_vector<float>();
    for (auto& value : v) {
      value /= scale;
      if (!std::isfinite(value)) {
        result->clear();
        return false;
      }
    }
    out.push_back(from_vector(v, {t.numel()}).reshape(t.shape()));
  }
  *result = std::move(out);
  return true;
}
} // namespace datamunge::nn
