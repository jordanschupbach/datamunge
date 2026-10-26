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
#include "datamunge/nn/parallel.hpp"
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

/// Elementwise kernels run in parallel above this many elements.
constexpr std::int64_t kParallelGrain = 32768;

/// strided_rows split over the leading axis across threads (f must be thread-safe).
template <std::size_t K, typename F>
void parallel_strided_rows(const Shape& shape, const std::array<const Shape*, K>& strides,
                           std::array<std::int64_t, K> base, F&& f) {
  const std::int64_t n = shape_numel(shape);
  if (shape.empty() || n < 2 * kParallelGrain || shape[0] < 2) {
    strided_rows<K>(shape, strides, base, f);
    return;
  }
  const std::int64_t per_row = n / shape[0];
  parallel_for(0, shape[0], std::max<std::int64_t>(1, kParallelGrain / std::max<std::int64_t>(per_row, 1)),
               [&](std::int64_t lo, std::int64_t hi) {
                 Shape sub = shape;
                 sub[0]    = hi - lo;
                 std::array<std::int64_t, K> off = base;
                 for (std::size_t k = 0; k < K; ++k) off[k] += lo * (*strides[k])[0];
                 strided_rows<K>(sub, strides, off, f);
               });
}

/// out[i] = f(i) for i in [0, n), in parallel for large n.
template <typename F>
void parallel_elements(std::int64_t n, F&& f) {
  parallel_for(0, n, kParallelGrain, [&](std::int64_t lo, std::int64_t hi) {
    for (std::int64_t i = lo; i < hi; ++i) f(i);
  });
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
    const In* s = src + x.offset();
    parallel_elements(x.numel(), [&](std::int64_t i) { dst[i] = f(s[i]); });
    return out;
  }
  const Shape                         out_strides = contiguous_strides(x.shape());
  const std::array<const Shape*, 2> st{&out_strides, &x.strides()};
  parallel_strided_rows<2>(x.shape(), st, {0, x.offset()}, [&](const auto& off, std::int64_t n, const auto& step) {
    for (std::int64_t j = 0; j < n; ++j) dst[off[0] + j] = f(src[off[1] + j * step[1]]);
  });
  return out;
}

/// Elementwise binary map with broadcasting. `a` and `b` must already have
/// dtype In.
template<typename In, typename Out, typename F>
Tensor map_binary(const Tensor& a, const Tensor& b, DType out_dtype, F&& f) {
  const Shape shape = broadcast_shapes(a.shape(), b.shape());
  Tensor      out   = empty(shape, out_dtype);
  Out*        dst   = out.data_ptr<Out>();
  const In*   pa    = raw_data<In>(a);
  const In*   pb    = raw_data<In>(b);
  if (a.shape() == shape && b.shape() == shape && a.is_contiguous() && b.is_contiguous()) {
    const In* sa = pa + a.offset();
    const In* sb = pb + b.offset();
    parallel_elements(out.numel(), [&](std::int64_t i) { dst[i] = f(sa[i], sb[i]); });
    return out;
  }
  // A broadcast scalar keeps its logical shape but every stride is zero.
  // Treat it like its scalar storage, avoiding the strided loop in VJPs (and
  // other uniform broadcasts) without expanding or copying the seed.
  const auto scalar_storage = [](const Tensor& t) {
    return t.numel() == 1 ||
           (t.numel() > 0 &&
            std::all_of(t.strides().begin(), t.strides().end(), [](std::int64_t stride) {
              return stride == 0;
            }));
  };
  if (scalar_storage(b) && a.shape() == shape && a.is_contiguous()) {
    const In  vb = pb[b.offset()];
    const In* sa = pa + a.offset();
    parallel_elements(out.numel(), [&](std::int64_t i) { dst[i] = f(sa[i], vb); });
    return out;
  }
  if (scalar_storage(a) && b.shape() == shape && b.is_contiguous()) {
    const In  va = pa[a.offset()];
    const In* sb = pb + b.offset();
    parallel_elements(out.numel(), [&](std::int64_t i) { dst[i] = f(va, sb[i]); });
    return out;
  }
  const Shape out_strides = contiguous_strides(shape);
  const Shape sa          = broadcast_strides(a, shape);
  const Shape sb          = broadcast_strides(b, shape);
  // Two transposed matrices otherwise stream one cache line per scalar.
  // Tile both dimensions, keeping the input columns and output row spans hot.
  // Positive, non-broadcast strides also support padded/offset matrix views.
  if (shape.size() == 2 && shape[0] >= 16 && shape[1] >= 32 && out.numel() >= 32768 && sa[0] == 1 &&
      sb[0] == 1 && sa[1] >= shape[0] && sb[1] >= shape[0]) {
    constexpr std::int64_t tile_rows = 16, tile_cols = 32;
    const auto             row_tiles = (shape[0] + tile_rows - 1) / tile_rows;
    const auto             col_tiles = (shape[1] + tile_cols - 1) / tile_cols;
    // Parallelize over both dimensions, so a thin transposed matrix still
    // exposes enough independent work without changing the global grain policy.
    parallel_for(
        0,
        row_tiles * col_tiles,
        std::max<std::int64_t>(1, kParallelGrain / (tile_rows * tile_cols)),
        [&](std::int64_t lo, std::int64_t hi) {
          for (auto tile = lo; tile < hi; ++tile) {
            const auto row_begin = (tile / col_tiles) * tile_rows;
            const auto col_begin = (tile % col_tiles) * tile_cols;
            const auto row_end   = std::min(shape[0], row_begin + tile_rows);
            const auto col_end   = std::min(shape[1], col_begin + tile_cols);
            for (auto row = row_begin; row < row_end; ++row) {
              const In* left   = pa + a.offset() + row;
              const In* right  = pb + b.offset() + row;
              Out*      target = dst + row * shape[1];
              for (auto j = col_begin; j < col_end; ++j)
                target[j] = f(left[j * sa[1]], right[j * sb[1]]);
            }
          }
        });
    return out;
  }
  const std::array<const Shape*, 3> st{&out_strides, &sa, &sb};
  parallel_strided_rows<3>(
      shape,
      st,
      {0, a.offset(), b.offset()},
      [&](const auto& off, std::int64_t n, const auto& step) {
        for (std::int64_t j = 0; j < n; ++j)
          dst[off[0] + j] = f(pa[off[1] + j * step[1]], pb[off[2] + j * step[2]]);
      });
  return out;
}

/// The same tensor without history, for use inside kernels.
inline Tensor nograd(const Tensor& t) {
  return t.defined() ? t.detach() : t;
}

/// Converts without recording (kernel use).
Tensor cast_raw(const Tensor& x, DType dtype);

/// Converts with recording if needed and returns `x` itself if already `dtype`.
inline Tensor as_dtype(const Tensor& x, DType dtype) {
  return x.dtype() == dtype ? x : cast(x, dtype);
}

/// Shape with 1 at the reduced axes (for broadcasting a reduced gradient back).
inline Shape keepdims_shape(const Shape& shape, const std::vector<std::int64_t>& axes) {
  Shape out = shape;
  for (auto a : axes)
    out[static_cast<std::size_t>(a)] = 1;
  return out;
}

}  // namespace datamunge::nn::detail
