#pragma once
/// @file patch_restoration.hpp
/// @brief Target-free patch visibility and exact observed/predicted composition.
#include <cmath>
#include <datamunge/nn/ops.hpp>
#include <stdexcept>

namespace datamunge::nn::models {
/// @brief Shape/value checks shared by pure patch-selection utilities.
namespace restoration_detail {
/// @brief Validate finite nonempty floating patch arrays.
/// @param patches Float32/64 values[B,P,D], all dimensions positive.
/// @throws std::invalid_argument For invalid shape, dtype or any nonfinite value.
inline void Patches(const Tensor& patches) {
  if (!patches.defined() || patches.dim() != 3 || patches.size(0) <= 0 || patches.size(1) <= 0 ||
      patches.size(2) <= 0 ||
      (patches.dtype() != DType::Float32 && patches.dtype() != DType::Float64))
    throw std::invalid_argument("restoration needs nonempty floating patch rows[B,P,D]");
  for (double v : patches.contiguous().to_vector<double>())
    if (!std::isfinite(v))
      throw std::invalid_argument("restoration patch values must be finite");
}
/// @brief Validate a detached binary floating hidden-patch mask.
/// @param hidden Matching-dtype[B,P] values exactly zero or one; no autograd path.
/// @param patches Already-validated patch values[B,P,D].
/// @throws std::invalid_argument For invalid mask shape/dtype/values or a live gradient flag.
inline void Hidden(const Tensor& hidden, const Tensor& patches) {
  if (!hidden.defined() || hidden.shape() != Shape{patches.size(0), patches.size(1)} ||
      hidden.dtype() != patches.dtype() || hidden.requires_grad())
    throw std::invalid_argument("restoration mask must be detached matching floating[B,P]");
  for (double v : hidden.contiguous().to_vector<double>())
    if (v != 0 && v != 1)
      throw std::invalid_argument("restoration hidden mask must be binary");
}
} // namespace restoration_detail
/// @brief Keep visible patch values and replace hidden patches with one explicit constant.
///
/// For hidden[b,p]=1, every coordinate is fill; for zero, every original value
/// is copied bit-for-bit, including signed zero. Finite signed Float32/64
/// patches[B,P,D] and a detached matching-dtype binary mask[B,P] are required.
/// Every dimension is positive; all-visible and all-hidden masks are allowed.
/// All input values must be finite, including unselected hidden coordinates.
/// No pixel-range restriction, clipping, encoder, target or fitting is added.
/// Float32 narrows fill once; finite underflow is accepted, overflow rejected.
/// Strides are accepted. Visible input derivatives remain live; hidden input
/// derivatives are zero. Output owns lifetime; inputs unchanged, no shared
/// state or random draws. Work/storage O(B*P*D), plus O(B*P) mask conversion.
/// @param patches Finite Float32/64 patch values[B,P,D], B,P,D positive.
/// @param hidden Detached same-dtype binary mask[B,P], one meaning hidden.
/// @param fill Finite replacement representable without overflow, default0.
/// @return Owning patch array of the same shape/dtype, with visible values preserved exactly.
/// @throws std::invalid_argument For invalid tensors/mask or nonfinite/unrepresentable fill.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// const auto visible=n::models::VisiblePatchValues(n::ones({1,2,3}),n::tensor({0.,1.},{1,2}));
/// // First patch is[1,1,1]; second patch is[0,0,0]. No hidden target is used.
/// @endcode
/// @see CompositePatchValues(), MaePatches(), MaeImage()
inline Tensor VisiblePatchValues(const Tensor& patches, const Tensor& hidden, double fill = 0) {
  restoration_detail::Patches(patches);
  restoration_detail::Hidden(hidden, patches);
  if (!std::isfinite(fill) || (patches.dtype() == DType::Float32 && !std::isfinite(float(fill))))
    throw std::invalid_argument("restoration fill must be finite in the computation dtype");
  return where(expand_dims(cast(hidden, DType::Bool), 2), full({}, fill, patches.dtype()), patches);
}
/// @brief Fill hidden patches from predictions while copying observed patches exactly.
///
/// Selects prediction at hidden[b,p]=1 and observed at zero. Matching finite
/// signed Float32/64[B,P,D] arrays and a detached same-dtype binary mask[B,P]
/// are required, with B,P,D positive. All-visible/all-hidden masks and strides
/// are accepted; every input value must be finite, including unselected ones.
/// Branch selection preserves selected value bits, including signed zero;
/// no averaging, blending, clamping or target values enter. Inputs unchanged,
/// owning same-shape/dtype output, no shared state, fitting or RNG. Continuous
/// first/second paths remain live in selected branches; unselected branch
/// derivatives are zero. Work/storage O(B*P*D), plus O(B*P) mask conversion.
/// @param observed Finite Float32/64 observed patch values[B,P,D].
/// @param prediction Finite exactly matching-shape/dtype predicted patch values.
/// @param hidden Detached same-dtype binary mask[B,P], one selecting prediction.
/// @return Owning composite patches, with all visible observed coordinates bit-exact.
/// @throws std::invalid_argument For invalid shapes/dtypes/values or a live/nonbinary mask.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// const auto out=n::models::CompositePatchValues(n::zeros({1,2,3}),n::ones({1,2,3}),
///     n::tensor({0.,1.},{1,2}));
/// // First patch is observed zeros; second patch is predicted ones.
/// @endcode
/// @see VisiblePatchValues(), MaePatches(), MaeImage(), MaeLoss()
inline Tensor
CompositePatchValues(const Tensor& observed, const Tensor& prediction, const Tensor& hidden) {
  restoration_detail::Patches(observed);
  restoration_detail::Patches(prediction);
  if (observed.shape() != prediction.shape() || observed.dtype() != prediction.dtype())
    throw std::invalid_argument("restoration prediction must match observed patches");
  restoration_detail::Hidden(hidden, observed);
  return where(expand_dims(cast(hidden, DType::Bool), 2), prediction, observed);
}
} // namespace datamunge::nn::models
