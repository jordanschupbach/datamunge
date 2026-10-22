#pragma once

// Internal helpers shared by the nn kernels. Not installed.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/ops.hpp"
#include "datamunge/nn/tensor.hpp"

namespace datamunge::nn::detail {

template <typename T>
struct Tag {
  using type = T;
};

/// Calls f(Tag<T>{}) with T the C++ type of `dtype`.
template <typename F>
decltype(auto) dispatch(DType dtype, F&& f) {
  switch (dtype) {
    case DType::Bool: return f(Tag<bool>{});
    case DType::Int32: return f(Tag<std::int32_t>{});
    case DType::Int64: return f(Tag<std::int64_t>{});
    case DType::Float32: return f(Tag<float>{});
    case DType::Float64: return f(Tag<double>{});
  }
  throw std::logic_error("nn: unknown dtype");
}

/// Like dispatch, but only for float dtypes.
template <typename F>
decltype(auto) dispatch_float(DType dtype, const char* what, F&& f) {
  switch (dtype) {
    case DType::Float32: return f(Tag<float>{});
    case DType::Float64: return f(Tag<double>{});
    default: throw std::invalid_argument(std::string(what) + ": expected a float tensor, got " + dtype_name(dtype));
  }
}

inline std::int64_t normalize_axis(std::int64_t axis, std::int64_t rank, const char* what) {
  const std::int64_t a = axis < 0 ? axis + rank : axis;
  if (a < 0 || a >= rank)
    throw std::invalid_argument(std::string(what) + ": axis " + std::to_string(axis) + " out of range for rank " +
                                std::to_string(rank));
  return a;
}

/// Sorted unique non-negative axes; an empty list means all axes.
inline std::vector<std::int64_t> normalize_axes(const std::vector<std::int64_t>& axes, std::int64_t rank,
                                                const char* what) {
  std::vector<std::int64_t> out;
  if (axes.empty()) {
    for (std::int64_t i = 0; i < rank; ++i) out.push_back(i);
    return out;
  }
  for (auto a : axes) out.push_back(normalize_axis(a, rank, what));
  std::sort(out.begin(), out.end());
  if (std::adjacent_find(out.begin(), out.end()) != out.end())
    throw std::invalid_argument(std::string(what) + ": repeated axis");
  return out;
}

/// A tensor viewing `base`'s storage with a new shape/strides/offset.
inline Tensor make_view(const Tensor& base, Shape shape, Shape strides, std::int64_t offset) {
  auto impl     = std::make_shared<TensorImpl>();
  impl->storage = base.impl()->storage;
  impl->shape   = std::move(shape);
  impl->strides = std::move(strides);
  impl->offset  = offset;
  impl->dtype   = base.dtype();
  return Tensor(std::move(impl));
}

/// Strides that read `t` as if broadcast to `out_shape` (0 on stretched axes).
inline Shape broadcast_strides(const Tensor& t, const Shape& out_shape) {
  const auto& shape   = t.shape();
  const auto& strides = t.strides();
  Shape       out(out_shape.size(), 0);
  const auto  lead = out_shape.size() - shape.size();
  for (std::size_t i = 0; i < shape.size(); ++i) out[lead + i] = shape[i] == 1 ? 0 : strides[i];
  return out;
}

/// Walks every element of `shape` for K operands with the given strides,
/// calling f(offsets, n, steps) once per innermost row: element j of the row
/// is at offsets[k] + j * steps[k] for operand k.
template <std::size_t K, typename F>
void strided_rows(const Shape& shape, const std::array<const Shape*, K>& strides, std::array<std::int64_t, K> base,
                  F&& f) {
  const std::size_t rank = shape.size();
  for (auto s : shape)
    if (s == 0) return;
  if (rank == 0) {
    std::array<std::int64_t, K> steps{};
    f(base, std::int64_t{1}, steps);
    return;
  }
  std::array<std::int64_t, K> steps{};
  for (std::size_t k = 0; k < K; ++k) steps[k] = (*strides[k])[rank - 1];
  const std::int64_t        inner = shape[rank - 1];
  std::vector<std::int64_t> counter(rank, 0);
  std::array<std::int64_t, K> off = base;
  while (true) {
    f(off, inner, steps);
    // advance the outer counters (all but the last axis)
    std::int64_t axis = static_cast<std::int64_t>(rank) - 2;
    while (axis >= 0) {
      const auto a = static_cast<std::size_t>(axis);
      ++counter[a];
      for (std::size_t k = 0; k < K; ++k) off[k] += (*strides[k])[a];
      if (counter[a] < shape[a]) break;
      for (std::size_t k = 0; k < K; ++k) off[k] -= (*strides[k])[a] * shape[a];
      counter[a] = 0;
      --axis;
    }
    if (axis < 0) return;
  }
}

template <typename T>
T* raw_data(const Tensor& t) {
  return reinterpret_cast<T*>(t.impl()->storage->bytes.get());
}

/// Elementwise map into a new packed tensor of `out_dtype`. `x` must already
/// have dtype In.
template <typename In, typename Out, typename F>
Tensor map_unary(const Tensor& x, DType out_dtype, F&& f) {
  Tensor     out = empty(x.shape(), out_dtype);
  Out*       dst = out.data_ptr<Out>();
  const In*  src = raw_data<In>(x);
  if (x.is_contiguous()) {
    const In*          s = src + x.offset();
    const std::int64_t n = x.numel();
    for (std::int64_t i = 0; i < n; ++i) dst[i] = f(s[i]);
    return out;
  }
  const Shape                         out_strides = contiguous_strides(x.shape());
  const std::array<const Shape*, 2> st{&out_strides, &x.strides()};
  strided_rows<2>(x.shape(), st, {0, x.offset()}, [&](const auto& off, std::int64_t n, const auto& step) {
    for (std::int64_t j = 0; j < n; ++j) dst[off[0] + j] = f(src[off[1] + j * step[1]]);
  });
  return out;
}

/// Elementwise binary map with broadcasting. `a` and `b` must already have
/// dtype In.
template <typename In, typename Out, typename F>
Tensor map_binary(const Tensor& a, const Tensor& b, DType out_dtype, F&& f) {
  const Shape shape = broadcast_shapes(a.shape(), b.shape());
  Tensor      out   = empty(shape, out_dtype);
  Out*        dst   = out.data_ptr<Out>();
  const In*   pa    = raw_data<In>(a);
  const In*   pb    = raw_data<In>(b);
  if (a.shape() == shape && b.shape() == shape && a.is_contiguous() && b.is_contiguous()) {
    const In*          sa = pa + a.offset();
    const In*          sb = pb + b.offset();
    const std::int64_t n  = out.numel();
    for (std::int64_t i = 0; i < n; ++i) dst[i] = f(sa[i], sb[i]);
    return out;
  }
  if (b.numel() == 1 && a.shape() == shape && a.is_contiguous()) {
    const In           vb = pb[b.offset()];
    const In*          sa = pa + a.offset();
    const std::int64_t n  = out.numel();
    for (std::int64_t i = 0; i < n; ++i) dst[i] = f(sa[i], vb);
    return out;
  }
  const Shape                         out_strides = contiguous_strides(shape);
  const Shape                         sa          = broadcast_strides(a, shape);
  const Shape                         sb          = broadcast_strides(b, shape);
  const std::array<const Shape*, 3> st{&out_strides, &sa, &sb};
  strided_rows<3>(shape, st, {0, a.offset(), b.offset()}, [&](const auto& off, std::int64_t n, const auto& step) {
    for (std::int64_t j = 0; j < n; ++j) dst[off[0] + j] = f(pa[off[1] + j * step[1]], pb[off[2] + j * step[2]]);
  });
  return out;
}

/// The same tensor without history, for use inside kernels.
inline Tensor nograd(const Tensor& t) { return t.defined() ? t.detach() : t; }

/// Converts without recording (kernel use).
Tensor cast_raw(const Tensor& x, DType dtype);

/// Converts with recording if needed and returns `x` itself if already `dtype`.
inline Tensor as_dtype(const Tensor& x, DType dtype) { return x.dtype() == dtype ? x : cast(x, dtype); }

/// Shape with 1 at the reduced axes (for broadcasting a reduced gradient back).
inline Shape keepdims_shape(const Shape& shape, const std::vector<std::int64_t>& axes) {
  Shape out = shape;
  for (auto a : axes) out[static_cast<std::size_t>(a)] = 1;
  return out;
}

}  // namespace datamunge::nn::detail
