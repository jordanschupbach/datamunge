// Shape ops. Most return views that share storage with their input.

#include <numeric>

#include "detail.hpp"

namespace datamunge::nn {

using detail::as_dtype;
using detail::dispatch;
using detail::make_view;
using detail::nograd;

namespace {

using Grads = std::vector<Tensor>;

/// One axis of a strided view: `count` elements from `start` every `step`.
struct AxisSlice {
  std::int64_t start{0};
  std::int64_t step{1};
  std::int64_t count{0};
};

Tensor strided_view_raw(const Tensor& x, const std::vector<AxisSlice>& slices) {
  Shape        shape   = x.shape();
  Shape        strides = x.strides();
  std::int64_t offset  = x.offset();
  for (std::size_t i = 0; i < slices.size(); ++i) {
    const auto& s = slices[i];
    if (s.count > 0) offset += s.start * strides[i];
    shape[i]   = s.count;
    strides[i] = strides[i] * s.step;
  }
  return make_view(x, shape, strides, offset);
}

Tensor strided_embed(const Tensor& g, const Shape& shape, const std::vector<AxisSlice>& slices);

/// Differentiable strided view.
Tensor strided_view(const Tensor& x, const std::vector<AxisSlice>& slices) {
  Tensor      out      = strided_view_raw(nograd(x), slices);
  const Shape in_shape = x.shape();
  record("StridedSlice", {x}, {out}, [in_shape, slices](const Grads& g, const Grads&) {
    return Grads{strided_embed(g[0], in_shape, slices)};
  });
  return out;
}

/// zeros(shape) with `g` written at the strided positions (gradient of strided_view).
Tensor strided_embed(const Tensor& g, const Shape& shape, const std::vector<AxisSlice>& slices) {
  Tensor out = zeros(shape, g.dtype());
  {
    NoGradGuard guard;
    Tensor      target = strided_view_raw(out, slices);
    target.copy_(nograd(g));
  }
  record("StridedSliceGrad", {g}, {out}, [slices](const Grads& gg, const Grads&) {
    return Grads{strided_view(gg[0], slices)};
  });
  return out;
}

std::vector<AxisSlice> full_slices(const Shape& shape) {
  std::vector<AxisSlice> s;
  for (auto d : shape) s.push_back({0, 1, d});
  return s;
}

/// Python slice semantics for one axis of length n.
AxisSlice python_slice(std::int64_t n, std::int64_t start, std::int64_t stop, std::int64_t step) {
  if (step == 0) throw std::invalid_argument("slice: step must be nonzero");
  auto clamp = [&](std::int64_t v, bool is_stop) {
    if (v < 0) v += n;
    if (step > 0) return std::min(std::max<std::int64_t>(v, 0), n);
    (void)is_stop;
    return std::min(std::max<std::int64_t>(v, -1), n - 1);
  };
  const std::int64_t b = clamp(start, false);
  const std::int64_t e = clamp(stop, true);
  std::int64_t       count = 0;
  if (step > 0 && e > b) count = (e - b + step - 1) / step;
  if (step < 0 && b > e) count = (b - e + (-step) - 1) / (-step);
  return {b, step, count};
}

}  // namespace

// ---- broadcasting ------------------------------------------------------------------------

Shape broadcast_shapes(const Shape& a, const Shape& b) {
  const std::size_t rank = std::max(a.size(), b.size());
  Shape             out(rank, 1);
  for (std::size_t i = 0; i < rank; ++i) {
    const std::int64_t da = i < rank - a.size() ? 1 : a[i - (rank - a.size())];
    const std::int64_t db = i < rank - b.size() ? 1 : b[i - (rank - b.size())];
    if (da != db && da != 1 && db != 1)
      throw std::invalid_argument("cannot broadcast shapes " + shape_str(a) + " and " + shape_str(b));
    out[i] = da == 1 ? db : da;
  }
  return out;
}

Tensor broadcast_to(const Tensor& x, const Shape& shape) {
  if (x.shape() == shape) return x;
  if (broadcast_shapes(x.shape(), shape) != shape)
    throw std::invalid_argument("broadcast_to: cannot broadcast " + shape_str(x.shape()) + " to " + shape_str(shape));
  Tensor      out      = make_view(nograd(x), shape, detail::broadcast_strides(x, shape), x.offset());
  const Shape in_shape = x.shape();
  record("BroadcastTo", {x}, {out}, [in_shape](const Grads& g, const Grads&) { return Grads{sum_to(g[0], in_shape)}; });
  return out;
}

Tensor sum_to(const Tensor& x, const Shape& shape) {
  if (x.shape() == shape) return x;
  const std::size_t lead = x.shape().size() - shape.size();
  if (x.shape().size() < shape.size()) throw std::invalid_argument("sum_to: target has higher rank");
  Axes axes;
  for (std::size_t i = 0; i < x.shape().size(); ++i) {
    if (i < lead) {
      axes.push_back(static_cast<std::int64_t>(i));
    } else if (shape[i - lead] == 1 && x.shape()[i] != 1) {
      axes.push_back(static_cast<std::int64_t>(i));
    } else if (shape[i - lead] != x.shape()[i]) {
      throw std::invalid_argument("sum_to: cannot reduce " + shape_str(x.shape()) + " to " + shape_str(shape));
    }
  }
  if (axes.empty()) return reshape(x, shape);
  return reshape(sum(x, axes, true), shape);
}

// ---- reshape family -------------------------------------------------------------------------

Tensor reshape(const Tensor& x, const Shape& shape_in) {
  Shape        shape = shape_in;
  std::int64_t known = 1;
  int          infer = -1;
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] == -1) {
      if (infer >= 0) throw std::invalid_argument("reshape: more than one -1 in " + shape_str(shape_in));
      infer = static_cast<int>(i);
    } else {
      if (shape[i] < 0) throw std::invalid_argument("reshape: bad dimension in " + shape_str(shape_in));
      known *= shape[i];
    }
  }
  if (infer >= 0) {
    if (known == 0 || x.numel() % known != 0)
      throw std::invalid_argument("reshape: cannot infer -1 reshaping " + shape_str(x.shape()) + " to " +
                                  shape_str(shape_in));
    shape[static_cast<std::size_t>(infer)] = x.numel() / known;
  }
  if (shape_numel(shape) != x.numel())
    throw std::invalid_argument("reshape: cannot reshape " + shape_str(x.shape()) + " (" + std::to_string(x.numel()) +
                                " elements) to " + shape_str(shape_in));
  Tensor out;
  {
    NoGradGuard  guard;
    const Tensor packed = nograd(x).contiguous();
    out                 = make_view(packed, shape, contiguous_strides(shape), packed.offset());
  }
  const Shape in_shape = x.shape();
  record("Reshape", {x}, {out}, [in_shape](const Grads& g, const Grads&) { return Grads{reshape(g[0], in_shape)}; });
  return out;
}

Tensor flatten(const Tensor& x) { return reshape(x, {-1}); }

Tensor expand_dims(const Tensor& x, std::int64_t axis_in) {
  const std::int64_t axis  = detail::normalize_axis(axis_in, x.dim() + 1, "expand_dims");
  Shape              shape = x.shape();
  shape.insert(shape.begin() + axis, 1);
  return reshape(x, shape);
}

Tensor squeeze(const Tensor& x, const Axes& axes_in) {
  Shape shape;
  if (axes_in.empty()) {
    for (auto d : x.shape())
      if (d != 1) shape.push_back(d);
  } else {
    const auto        axes = detail::normalize_axes(axes_in, x.dim(), "squeeze");
    std::vector<bool> drop(x.shape().size(), false);
    for (auto a : axes) {
      if (x.shape()[static_cast<std::size_t>(a)] != 1)
        throw std::invalid_argument("squeeze: axis " + std::to_string(a) + " of " + shape_str(x.shape()) +
                                    " is not size 1");
      drop[static_cast<std::size_t>(a)] = true;
    }
    for (std::size_t i = 0; i < x.shape().size(); ++i)
      if (!drop[i]) shape.push_back(x.shape()[i]);
  }
  return reshape(x, shape);
}

Tensor transpose(const Tensor& x, const Axes& perm_in) {
  const std::int64_t rank = x.dim();
  Axes               perm = perm_in;
  if (perm.empty())
    for (std::int64_t i = rank - 1; i >= 0; --i) perm.push_back(i);
  if (static_cast<std::int64_t>(perm.size()) != rank)
    throw std::invalid_argument("transpose: permutation has " + std::to_string(perm.size()) + " axes for rank " +
                                std::to_string(rank));
  std::vector<bool> seen(static_cast<std::size_t>(rank), false);
  Shape             shape, strides;
  for (auto& p : perm) {
    p = detail::normalize_axis(p, rank, "transpose");
    if (seen[static_cast<std::size_t>(p)]) throw std::invalid_argument("transpose: repeated axis");
    seen[static_cast<std::size_t>(p)] = true;
    shape.push_back(x.shape()[static_cast<std::size_t>(p)]);
    strides.push_back(x.strides()[static_cast<std::size_t>(p)]);
  }
  Tensor out = make_view(nograd(x), shape, strides, x.offset());
  Axes   inverse(perm.size());
  for (std::size_t i = 0; i < perm.size(); ++i) inverse[static_cast<std::size_t>(perm[i])] = static_cast<std::int64_t>(i);
  record("Transpose", {x}, {out}, [inverse](const Grads& g, const Grads&) { return Grads{transpose(g[0], inverse)}; });
  return out;
}

Tensor swapaxes(const Tensor& x, std::int64_t axis0, std::int64_t axis1) {
  Axes perm(static_cast<std::size_t>(x.dim()));
  std::iota(perm.begin(), perm.end(), 0);
  std::swap(perm[static_cast<std::size_t>(detail::normalize_axis(axis0, x.dim(), "swapaxes"))],
            perm[static_cast<std::size_t>(detail::normalize_axis(axis1, x.dim(), "swapaxes"))]);
  return transpose(x, perm);
}

// ---- slicing ------------------------------------------------------------------------------------

Tensor slice(const Tensor& x, const Shape& begin, const Shape& size) {
  if (static_cast<std::int64_t>(begin.size()) != x.dim() || begin.size() != size.size())
    throw std::invalid_argument("slice: begin and size need one entry per axis of " + shape_str(x.shape()));
  auto slices = full_slices(x.shape());
  for (std::size_t i = 0; i < begin.size(); ++i) {
    const std::int64_t n = x.shape()[i];
    const std::int64_t b = begin[i];
    const std::int64_t s = size[i] == -1 ? n - b : size[i];
    if (b < 0 || s < 0 || b + s > n)
      throw std::invalid_argument("slice: begin " + shape_str(begin) + " size " + shape_str(size) +
                                  " out of range for " + shape_str(x.shape()));
    slices[i] = {b, 1, s};
  }
  return strided_view(x, slices);
}

Tensor slice_axis(const Tensor& x, std::int64_t axis_in, std::int64_t start, std::int64_t stop, std::int64_t step) {
  const std::int64_t axis   = detail::normalize_axis(axis_in, x.dim(), "slice_axis");
  auto               slices = full_slices(x.shape());
  slices[static_cast<std::size_t>(axis)] = python_slice(x.shape()[static_cast<std::size_t>(axis)], start, stop, step);
  return strided_view(x, slices);
}

Tensor strided_slice(const Tensor& x, const Shape& begin, const Shape& end, const Shape& strides) {
  if (begin.size() != end.size() || (!strides.empty() && strides.size() != begin.size()) ||
      static_cast<std::int64_t>(begin.size()) > x.dim())
    throw std::invalid_argument("strided_slice: begin/end/strides lengths do not match");
  auto slices = full_slices(x.shape());
  for (std::size_t i = 0; i < begin.size(); ++i)
    slices[i] = python_slice(x.shape()[i], begin[i], end[i], strides.empty() ? 1 : strides[i]);
  return strided_view(x, slices);
}

Tensor select(const Tensor& x, std::int64_t axis_in, std::int64_t index) {
  const std::int64_t axis = detail::normalize_axis(axis_in, x.dim(), "select");
  const std::int64_t n    = x.shape()[static_cast<std::size_t>(axis)];
  const std::int64_t i    = index < 0 ? index + n : index;
  if (i < 0 || i >= n) throw std::out_of_range("select: index " + std::to_string(index) + " out of range for axis of size " + std::to_string(n));
  auto slices                            = full_slices(x.shape());
  slices[static_cast<std::size_t>(axis)] = {i, 1, 1};
  return squeeze(strided_view(x, slices), {axis});
}

// ---- concat / split / stack ---------------------------------------------------------------------

Tensor concat(const std::vector<Tensor>& tensors, std::int64_t axis_in) {
  if (tensors.empty()) throw std::invalid_argument("concat: no tensors");
  const std::int64_t rank = tensors[0].dim();
  const std::int64_t axis = detail::normalize_axis(axis_in, rank, "concat");
  DType              dt   = tensors[0].dtype();
  Shape              shape = tensors[0].shape();
  shape[static_cast<std::size_t>(axis)] = 0;
  for (const auto& t : tensors) {
    if (t.dim() != rank) throw std::invalid_argument("concat: rank mismatch");
    for (std::int64_t i = 0; i < rank; ++i)
      if (i != axis && t.shape()[static_cast<std::size_t>(i)] != shape[static_cast<std::size_t>(i)])
        throw std::invalid_argument("concat: shapes " + shape_str(tensors[0].shape()) + " and " +
                                    shape_str(t.shape()) + " differ off the concat axis");
    shape[static_cast<std::size_t>(axis)] += t.shape()[static_cast<std::size_t>(axis)];
    dt = promote_types(dt, t.dtype());
  }
  std::vector<Tensor> inputs;
  for (const auto& t : tensors) inputs.push_back(as_dtype(t, dt));
  Tensor out = empty(shape, dt);
  Shape  sizes;
  {
    NoGradGuard  guard;
    std::int64_t at = 0;
    for (const auto& t : inputs) {
      const std::int64_t n      = t.shape()[static_cast<std::size_t>(axis)];
      auto               slices = full_slices(shape);
      slices[static_cast<std::size_t>(axis)] = {at, 1, n};
      Tensor target = strided_view_raw(out, slices);
      target.copy_(nograd(t));
      at += n;
      sizes.push_back(n);
    }
  }
  record("ConcatV2", inputs, {out}, [sizes, axis](const Grads& g, const Grads&) { return split(g[0], sizes, axis); });
  return out;
}

std::vector<Tensor> split(const Tensor& x, const Shape& sizes_in, std::int64_t axis_in) {
  const std::int64_t axis  = detail::normalize_axis(axis_in, x.dim(), "split");
  const std::int64_t n     = x.shape()[static_cast<std::size_t>(axis)];
  Shape              sizes = sizes_in;
  std::int64_t       known = 0;
  int                infer = -1;
  for (std::size_t i = 0; i < sizes.size(); ++i) {
    if (sizes[i] == -1) {
      if (infer >= 0) throw std::invalid_argument("split: more than one -1 size");
      infer = static_cast<int>(i);
    } else {
      known += sizes[i];
    }
  }
  if (infer >= 0) sizes[static_cast<std::size_t>(infer)] = n - known;
  if (std::accumulate(sizes.begin(), sizes.end(), std::int64_t{0}) != n)
    throw std::invalid_argument("split: sizes " + shape_str(sizes_in) + " do not add up to " + std::to_string(n));
  std::vector<Tensor> outs;
  {
    std::int64_t at = 0;
    for (auto s : sizes) {
      auto slices                            = full_slices(x.shape());
      slices[static_cast<std::size_t>(axis)] = {at, 1, s};
      outs.push_back(strided_view_raw(nograd(x), slices));
      at += s;
    }
  }
  record("Split", {x}, outs, [axis](const Grads& g, const Grads&) { return Grads{concat(g, axis)}; });
  return outs;
}

std::vector<Tensor> split(const Tensor& x, std::int64_t count, std::int64_t axis) {
  const std::int64_t n = x.shape()[static_cast<std::size_t>(detail::normalize_axis(axis, x.dim(), "split"))];
  if (count <= 0 || n % count != 0)
    throw std::invalid_argument("split: axis of size " + std::to_string(n) + " does not split into " +
                                std::to_string(count) + " equal parts");
  return split(x, Shape(static_cast<std::size_t>(count), n / count), axis);
}

Tensor stack(const std::vector<Tensor>& tensors, std::int64_t axis_in) {
  if (tensors.empty()) throw std::invalid_argument("stack: no tensors");
  const std::int64_t  axis = detail::normalize_axis(axis_in, tensors[0].dim() + 1, "stack");
  std::vector<Tensor> expanded;
  for (const auto& t : tensors) {
    if (t.shape() != tensors[0].shape())
      throw std::invalid_argument("stack: shapes " + shape_str(tensors[0].shape()) + " and " + shape_str(t.shape()) +
                                  " differ");
    expanded.push_back(expand_dims(t, axis));
  }
  return concat(expanded, axis);
}

std::vector<Tensor> unstack(const Tensor& x, std::int64_t axis_in) {
  const std::int64_t  axis = detail::normalize_axis(axis_in, x.dim(), "unstack");
  std::vector<Tensor> parts = split(x, Shape(static_cast<std::size_t>(x.shape()[static_cast<std::size_t>(axis)]), 1), axis);
  for (auto& p : parts) p = squeeze(p, {axis});
  return parts;
}

Tensor tile(const Tensor& x, const Shape& multiples) {
  if (static_cast<std::int64_t>(multiples.size()) != x.dim())
    throw std::invalid_argument("tile: need one multiple per axis of " + shape_str(x.shape()));
  Shape interleaved, broadcast, final_shape;
  for (std::size_t i = 0; i < multiples.size(); ++i) {
    interleaved.push_back(1);
    interleaved.push_back(x.shape()[i]);
    broadcast.push_back(multiples[i]);
    broadcast.push_back(x.shape()[i]);
    final_shape.push_back(multiples[i] * x.shape()[i]);
  }
  return reshape(broadcast_to(reshape(x, interleaved), broadcast), final_shape);
}

Tensor repeat(const Tensor& x, std::int64_t repeats, std::int64_t axis_in) {
  const std::int64_t axis = detail::normalize_axis(axis_in, x.dim(), "repeat");
  Shape              with_one = x.shape();
  with_one.insert(with_one.begin() + axis + 1, 1);
  Shape broadcast = with_one;
  broadcast[static_cast<std::size_t>(axis + 1)] = repeats;
  Shape final_shape = x.shape();
  final_shape[static_cast<std::size_t>(axis)] *= repeats;
  return reshape(broadcast_to(reshape(x, with_one), broadcast), final_shape);
}

// ---- pad / reverse --------------------------------------------------------------------------------

Tensor pad(const Tensor& x, const std::vector<std::pair<std::int64_t, std::int64_t>>& paddings, PadMode mode,
           double value) {
  if (static_cast<std::int64_t>(paddings.size()) != x.dim())
    throw std::invalid_argument("pad: need one (before, after) pair per axis of " + shape_str(x.shape()));
  for (const auto& p : paddings)
    if (p.first < 0 || p.second < 0) throw std::invalid_argument("pad: negative padding");

  if (mode == PadMode::Constant) {
    Shape shape = x.shape();
    auto  slices = full_slices(x.shape());
    for (std::size_t i = 0; i < paddings.size(); ++i) {
      shape[i] += paddings[i].first + paddings[i].second;
      slices[i] = {paddings[i].first, 1, x.shape()[i]};
    }
    Tensor out = full(shape, value, x.dtype());
    {
      NoGradGuard guard;
      Tensor      interior = strided_view_raw(out, slices);
      interior.copy_(nograd(x));
    }
    record("Pad", {x}, {out}, [slices](const Grads& g, const Grads&) { return Grads{strided_view(g[0], slices)}; });
    return out;
  }

  // Reflect (edge not repeated) and symmetric (edge repeated) as a gather per axis.
  Tensor out = x;
  for (std::size_t i = 0; i < paddings.size(); ++i) {
    const auto [before, after] = paddings[i];
    if (before == 0 && after == 0) continue;
    const std::int64_t n = x.shape()[i];
    const std::int64_t limit = mode == PadMode::Reflect ? n - 1 : n;
    if (before > limit || after > limit)
      throw std::invalid_argument("pad: mirror padding larger than the axis allows");
    std::vector<std::int64_t> index;
    for (std::int64_t k = -before; k < n + after; ++k) {
      std::int64_t j = k;
      if (j < 0) j = mode == PadMode::Reflect ? -j : -j - 1;
      if (j >= n) j = mode == PadMode::Reflect ? 2 * (n - 1) - j : 2 * n - 1 - j;
      index.push_back(j);
    }
    out = gather(out, from_vector(index), static_cast<std::int64_t>(i));
  }
  return out;
}

Tensor reverse(const Tensor& x, const Axes& axes_in) {
  if (axes_in.empty()) return x;
  const auto axes   = detail::normalize_axes(axes_in, x.dim(), "reverse");
  auto       slices = full_slices(x.shape());
  for (auto a : axes) {
    const std::int64_t n                 = x.shape()[static_cast<std::size_t>(a)];
    slices[static_cast<std::size_t>(a)] = {n - 1, -1, n};
  }
  return strided_view(x, slices);
}

}  // namespace datamunge::nn
