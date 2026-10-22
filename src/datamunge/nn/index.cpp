// Index ops: gather/scatter families, one_hot, top_k, argsort.

#include <numeric>

#include "detail.hpp"

namespace datamunge::nn {

using detail::dispatch;
using detail::nograd;

namespace {

using Grads = std::vector<Tensor>;

/// How out-of-range indices are treated. Strict: negatives wrap, others
/// throw (TF Gather on CPU). Drop: anything outside [0, n) is ignored, and
/// gathers read 0 there (TF UnsortedSegmentSum).
enum class IndexMode { Strict, Drop };

std::vector<std::int64_t> index_values(const Tensor& indices, const char* what) {
  if (!is_integral(indices.dtype()))
    throw std::invalid_argument(std::string(what) + ": indices must be int32 or int64, got " +
                                dtype_name(indices.dtype()));
  return indices.to_vector<std::int64_t>();
}

/// Resolves an index against an axis of size n; -1 means "skip".
std::int64_t resolve(std::int64_t i, std::int64_t n, IndexMode mode, const char* what) {
  if (mode == IndexMode::Drop) return (i < 0 || i >= n) ? -1 : i;
  const std::int64_t r = i < 0 ? i + n : i;
  if (r < 0 || r >= n)
    throw std::out_of_range(std::string(what) + ": index " + std::to_string(i) + " out of range for axis of size " +
                            std::to_string(n));
  return r;
}

struct AxisSplit {
  std::int64_t outer{1}, n{1}, inner{1};
};

AxisSplit split_at(const Shape& shape, std::int64_t axis) {
  AxisSplit s;
  for (std::int64_t i = 0; i < axis; ++i) s.outer *= shape[static_cast<std::size_t>(i)];
  s.n = shape[static_cast<std::size_t>(axis)];
  for (std::size_t i = static_cast<std::size_t>(axis) + 1; i < shape.size(); ++i) s.inner *= shape[i];
  return s;
}

Tensor scatter_add_axis(const Tensor& updates, const Tensor& indices, std::int64_t axis, std::int64_t n,
                        IndexMode mode);

/// out = x.shape[:axis] + indices.shape + x.shape[axis+1:]
Tensor gather_axis(const Tensor& x, const Tensor& indices, std::int64_t axis, IndexMode mode) {
  const auto      idx = index_values(indices, "gather");
  const AxisSplit s   = split_at(x.shape(), axis);
  Shape           shape(x.shape().begin(), x.shape().begin() + axis);
  shape.insert(shape.end(), indices.shape().begin(), indices.shape().end());
  shape.insert(shape.end(), x.shape().begin() + axis + 1, x.shape().end());
  std::vector<std::int64_t> resolved(idx.size());
  for (std::size_t k = 0; k < idx.size(); ++k) resolved[k] = resolve(idx[k], s.n, mode, "gather");
  const auto K   = static_cast<std::int64_t>(idx.size());
  Tensor     out = zeros(shape, x.dtype());
  {
    NoGradGuard  guard;
    const Tensor xc = nograd(x).contiguous();
    dispatch(x.dtype(), [&](auto tag) {
      using T    = typename decltype(tag)::type;
      const T* src = xc.data_ptr<T>();
      T*       dst = out.data_ptr<T>();
      for (std::int64_t o = 0; o < s.outer; ++o)
        for (std::int64_t k = 0; k < K; ++k) {
          const std::int64_t j = resolved[static_cast<std::size_t>(k)];
          if (j < 0) continue;
          std::copy(src + (o * s.n + j) * s.inner, src + (o * s.n + j + 1) * s.inner, dst + (o * K + k) * s.inner);
        }
    });
  }
  const std::int64_t n = s.n;
  record("GatherV2", {x}, {out}, [indices, axis, n, mode](const Grads& g, const Grads&) {
    return Grads{scatter_add_axis(g[0], indices, axis, n, mode)};
  });
  return out;
}

/// Inverse of gather_axis: sums `updates` (shape pre + indices.shape + post)
/// into zeros of shape pre + [n] + post.
Tensor scatter_add_axis(const Tensor& updates, const Tensor& indices, std::int64_t axis, std::int64_t n,
                        IndexMode mode) {
  const auto   idx  = index_values(indices, "scatter");
  const auto   K    = static_cast<std::int64_t>(idx.size());
  const auto   irank = static_cast<std::int64_t>(indices.dim());
  const Shape& us   = updates.shape();
  if (static_cast<std::int64_t>(us.size()) < axis + irank)
    throw std::invalid_argument("scatter: updates shape " + shape_str(us) + " too small for indices " +
                                shape_str(indices.shape()));
  for (std::int64_t i = 0; i < irank; ++i)
    if (us[static_cast<std::size_t>(axis + i)] != indices.shape()[static_cast<std::size_t>(i)])
      throw std::invalid_argument("scatter: updates shape " + shape_str(us) + " does not match indices shape " +
                                  shape_str(indices.shape()) + " at axis " + std::to_string(axis));
  Shape shape(us.begin(), us.begin() + axis);
  shape.push_back(n);
  shape.insert(shape.end(), us.begin() + axis + irank, us.end());
  std::int64_t outer = 1, inner = 1;
  for (std::int64_t i = 0; i < axis; ++i) outer *= us[static_cast<std::size_t>(i)];
  for (std::size_t i = static_cast<std::size_t>(axis + irank); i < us.size(); ++i) inner *= us[i];
  std::vector<std::int64_t> resolved(idx.size());
  for (std::size_t k = 0; k < idx.size(); ++k) resolved[k] = resolve(idx[k], n, mode, "scatter");
  Tensor out = zeros(shape, updates.dtype());
  {
    NoGradGuard  guard;
    const Tensor uc = nograd(updates).contiguous();
    dispatch(updates.dtype(), [&](auto tag) {
      using T    = typename decltype(tag)::type;
      const T* src = uc.data_ptr<T>();
      T*       dst = out.data_ptr<T>();
      for (std::int64_t o = 0; o < outer; ++o)
        for (std::int64_t k = 0; k < K; ++k) {
          const std::int64_t j = resolved[static_cast<std::size_t>(k)];
          if (j < 0) continue;
          const T* from = src + (o * K + k) * inner;
          T*       to   = dst + (o * n + j) * inner;
          for (std::int64_t i = 0; i < inner; ++i) {
            if constexpr (std::is_same<T, bool>::value)
              to[i] = to[i] || from[i];
            else
              to[i] += from[i];
          }
        }
    });
  }
  record("ScatterAdd", {updates}, {out}, [indices, axis, mode](const Grads& g, const Grads&) {
    return Grads{gather_axis(g[0], indices, axis, mode)};
  });
  return out;
}

/// Row-major flat positions (over the first k axes of `shape`) addressed by
/// the last axis of `indices`. Validates the range.
Tensor flatten_nd_indices(const Tensor& indices, const Shape& shape, const char* what) {
  if (indices.dim() < 1) throw std::invalid_argument(std::string(what) + ": indices need rank >= 1");
  const std::int64_t k = indices.shape().back();
  if (k > static_cast<std::int64_t>(shape.size()))
    throw std::invalid_argument(std::string(what) + ": index depth " + std::to_string(k) + " exceeds rank of " +
                                shape_str(shape));
  const auto  idx = index_values(indices, what);
  Shape       out_shape(indices.shape().begin(), indices.shape().end() - 1);
  const auto  count = static_cast<std::int64_t>(idx.size()) / std::max<std::int64_t>(k, 1);
  std::vector<std::int64_t> flat(static_cast<std::size_t>(k == 0 ? shape_numel(out_shape) : count), 0);
  for (std::int64_t r = 0; r < count && k > 0; ++r) {
    std::int64_t f = 0;
    for (std::int64_t j = 0; j < k; ++j) {
      const std::int64_t n = shape[static_cast<std::size_t>(j)];
      const std::int64_t v = idx[static_cast<std::size_t>(r * k + j)];
      if (v < 0 || v >= n)
        throw std::out_of_range(std::string(what) + ": index " + std::to_string(v) + " out of range for axis " +
                                std::to_string(j) + " of size " + std::to_string(n));
      f = f * n + v;
    }
    flat[static_cast<std::size_t>(r)] = f;
  }
  return from_vector(flat, out_shape.empty() ? Shape{} : out_shape).reshape(out_shape);
}

}  // namespace

Tensor gather(const Tensor& x, const Tensor& indices, std::int64_t axis) {
  return gather_axis(x, indices, detail::normalize_axis(axis, x.dim(), "gather"), IndexMode::Strict);
}

Tensor gather_nd(const Tensor& x, const Tensor& indices) {
  const std::int64_t k = indices.shape().empty() ? 0 : indices.shape().back();
  Shape              lead(x.shape().begin(), x.shape().begin() + k);
  Shape              flat_shape{shape_numel(lead)};
  flat_shape.insert(flat_shape.end(), x.shape().begin() + k, x.shape().end());
  const Tensor flat = flatten_nd_indices(indices, x.shape(), "gather_nd");
  return gather_axis(reshape(x, flat_shape), flat, 0, IndexMode::Strict);
}

Tensor scatter_nd(const Tensor& indices, const Tensor& updates, const Shape& shape) {
  const std::int64_t k = indices.shape().empty() ? 0 : indices.shape().back();
  const Tensor       flat = flatten_nd_indices(indices, shape, "scatter_nd");
  Shape              lead(shape.begin(), shape.begin() + k);
  Tensor             summed = scatter_add_axis(updates, flat, 0, shape_numel(lead), IndexMode::Strict);
  return reshape(summed, shape);
}

Tensor scatter_nd_add(const Tensor& x, const Tensor& indices, const Tensor& updates) {
  return add(x, scatter_nd(indices, cast(updates, x.dtype()), x.shape()));
}

Tensor index_add(const Tensor& x, std::int64_t axis_in, const Tensor& index, const Tensor& src) {
  const std::int64_t axis = detail::normalize_axis(axis_in, x.dim(), "index_add");
  if (index.dim() != 1) throw std::invalid_argument("index_add: index must be 1-D");
  return add(x, scatter_add_axis(cast(src, x.dtype()), index, axis, x.shape()[static_cast<std::size_t>(axis)],
                                 IndexMode::Strict));
}

Tensor unsorted_segment_sum(const Tensor& data, const Tensor& segment_ids, std::int64_t num_segments) {
  const Shape& ds = data.shape();
  const Shape& is = segment_ids.shape();
  if (is.size() > ds.size() || !std::equal(is.begin(), is.end(), ds.begin()))
    throw std::invalid_argument("unsorted_segment_sum: segment_ids shape " + shape_str(is) +
                                " must be a prefix of data shape " + shape_str(ds));
  return scatter_add_axis(data, segment_ids, 0, num_segments, IndexMode::Drop);
}

Tensor unsorted_segment_mean(const Tensor& data, const Tensor& segment_ids, std::int64_t num_segments) {
  Tensor dataf  = cast(data, to_floating(data.dtype()));
  Tensor sums   = unsorted_segment_sum(dataf, segment_ids, num_segments);
  Tensor counts = unsorted_segment_sum(ones(segment_ids.shape(), dataf.dtype()), segment_ids, num_segments);
  Shape  cshape{num_segments};
  for (std::size_t i = 1; i < sums.shape().size(); ++i) cshape.push_back(1);
  return div(sums, reshape(maximum(counts, ones({}, dataf.dtype())), cshape));
}

Tensor one_hot(const Tensor& indices, std::int64_t depth, double on, double off, DType dtype) {
  const auto idx   = index_values(indices, "one_hot");
  Shape      shape = indices.shape();
  shape.push_back(depth);
  Tensor out = full(shape, off, dtype);
  dispatch(dtype, [&](auto tag) {
    using T = typename decltype(tag)::type;
    T* p    = out.data_ptr<T>();
    for (std::size_t i = 0; i < idx.size(); ++i)
      if (idx[i] >= 0 && idx[i] < depth) p[static_cast<std::int64_t>(i) * depth + idx[i]] = static_cast<T>(on);
  });
  return out;
}

namespace {

/// Stable ordering of each last-axis row.
Tensor sort_indices(const Tensor& x, bool descending, std::int64_t keep) {
  if (x.dim() < 1) throw std::invalid_argument("sort: need rank >= 1");
  const std::int64_t n    = x.shape().back();
  const std::int64_t rows = n ? x.numel() / n : 0;
  Shape              shape = x.shape();
  shape.back()             = keep;
  Tensor out = empty(shape, DType::Int64);
  NoGradGuard  guard;
  const Tensor xc = nograd(x).contiguous();
  dispatch(x.dtype(), [&](auto tag) {
    using T                 = typename decltype(tag)::type;
    const T*      src       = xc.data_ptr<T>();
    std::int64_t* dst       = out.data_ptr<std::int64_t>();
    std::vector<std::int64_t> order(static_cast<std::size_t>(n));
    for (std::int64_t r = 0; r < rows; ++r) {
      const T* row = src + r * n;
      std::iota(order.begin(), order.end(), 0);
      auto cmp = [&](std::int64_t a, std::int64_t b) { return descending ? row[a] > row[b] : row[a] < row[b]; };
      if (keep < n)
        std::partial_sort(order.begin(), order.begin() + keep, order.end(), [&](std::int64_t a, std::int64_t b) {
          return cmp(a, b) || (!cmp(b, a) && a < b);
        });
      else
        std::stable_sort(order.begin(), order.end(), cmp);
      std::copy(order.begin(), order.begin() + keep, dst + r * keep);
    }
  });
  return out;
}

}  // namespace

std::pair<Tensor, Tensor> top_k(const Tensor& x, std::int64_t k) {
  if (x.dim() < 1) throw std::invalid_argument("top_k: need rank >= 1");
  const std::int64_t n = x.shape().back();
  if (k < 0 || k > n) throw std::invalid_argument("top_k: k = " + std::to_string(k) + " for a last axis of " + std::to_string(n));
  Tensor indices = sort_indices(x, true, k);
  // Values via a differentiable gather on the flattened tensor.
  const std::int64_t        rows = n ? x.numel() / n : 0;
  std::vector<std::int64_t> flat(static_cast<std::size_t>(rows * k));
  const std::int64_t*       ip = indices.data_ptr<std::int64_t>();
  for (std::int64_t r = 0; r < rows; ++r)
    for (std::int64_t j = 0; j < k; ++j) flat[static_cast<std::size_t>(r * k + j)] = r * n + ip[r * k + j];
  Tensor values = reshape(gather(reshape(x, {-1}), from_vector(flat), 0), indices.shape());
  return {values, indices};
}

Tensor argsort(const Tensor& x, bool descending) { return sort_indices(x, descending, x.dim() ? x.shape().back() : 0); }

}  // namespace datamunge::nn
