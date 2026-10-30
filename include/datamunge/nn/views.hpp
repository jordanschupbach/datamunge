#pragma once

/// @file views.hpp
/// @brief Strided views: when a reshape can share storage, zero-copy sliding windows, and
///        whether two tensors may overlap in memory.
///
/// A tensor reads element \f$(i_0,\dots,i_{r-1})\f$ at storage position
/// \f$\text{offset}+\sum_k i_k s_k\f$, where \f$s_k\f$ are its strides. Transposes, slices and broadcasts
/// change only the strides and offset, so they never copy. A reshape can keep the same storage exactly when
/// every group of old axes that merges into new axes is itself laid out contiguously; view_strides()
/// decides that and gives the new strides, and nn::reshape() uses it.

#include "datamunge/nn/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>

namespace datamunge::nn {

/// @brief Strides that let a tensor of @p shape and @p strides be read as @p new_shape without copying
///        (NumPy's no-copy reshape rule), or nothing if a copy is needed.
///
/// Size-1 axes are ignored. The old and new shapes are split into the shortest groups of axes with equal
/// products; a view exists if and only if, inside each old group, every stride equals the next axis's
/// length times the next stride. Each new group then gets row-major strides ending in the group's last old stride.
///
/// @param shape  Current shape.
/// @param strides  Current strides, in elements.
/// @param new_shape  Requested shape, with the same number of elements (no -1).
/// @return The new strides, or std::nullopt.
/// @throws std::invalid_argument  If the element counts differ or the shapes and strides have different ranks.
[[nodiscard]] std::optional<Shape> view_strides(const Shape& shape, const Shape& strides, const Shape& new_shape);

/// @brief Whether nn::reshape(x, new_shape) returns a view of @p x rather than a copy.
/// @param x  A defined tensor.
/// @param new_shape  Requested shape (no -1).
/// @return True when view_strides() succeeds.
[[nodiscard]] bool can_view(const Tensor& x, const Shape& new_shape);

/// @brief Overlapping windows along one axis, as a view (NumPy `sliding_window_view`, PyTorch `unfold`).
///
/// The result has @p x's shape with axis @p axis replaced by the number of windows
/// \f$W=\lfloor(n-\text{size})/\text{step}\rfloor+1\f$, and a new last axis of length @p size:
/// `out[..., w, ..., k] = x[..., w * step + k, ...]`. It shares @p x's storage, so it costs no memory
/// whatever the overlap. It is differentiable; the gradient adds each window's gradient back to
/// the positions it covers.
///
/// @param x  Input tensor.
/// @param axis  Axis to window; negative counts from the end.
/// @param size  Window length, 1 to the axis length.
/// @param step  Distance between window starts (default 1).
/// @return The view, of rank x.dim() + 1.
/// @throws std::invalid_argument  For a bad axis, a size outside [1, n], or a step below 1.
[[nodiscard]] Tensor sliding_window(const Tensor& x, std::int64_t axis, std::int64_t size, std::int64_t step = 1);

/// @brief Whether two tensors may share memory: the same storage buffer with overlapping element ranges.
///
/// The test compares the first and last storage positions each tensor can touch, so it is exact for
/// contiguous tensors and conservative for interleaved strided views (two views of alternate
/// elements may be reported as overlapping although they are disjoint).
///
/// @param a  A tensor.
/// @param b  Another tensor.
/// @return False when either is undefined or empty, or the buffers or ranges are disjoint.
[[nodiscard]] bool may_share_memory(const Tensor& a, const Tensor& b);

/// @brief Bytes of the storage buffer behind @p x, which can exceed `numel() * dtype_size()` for a
///        view (a slice keeps its whole parent alive) or fall short of it (a broadcast or sliding window).
/// @param x  A defined tensor.
/// @return The buffer size in bytes.
[[nodiscard]] std::size_t storage_bytes(const Tensor& x);

}  // namespace datamunge::nn
