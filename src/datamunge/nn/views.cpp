/// @file views.cpp
/// @brief view_strides(), can_view(), sliding_window(), may_share_memory() and storage_bytes().

#include "datamunge/nn/views.hpp"

#include "detail.hpp"

namespace datamunge::nn {

using detail::make_view;
using detail::nograd;

std::optional<Shape> view_strides(const Shape& shape, const Shape& strides, const Shape& new_shape) {
    if (shape.size() != strides.size()) throw std::invalid_argument("view_strides: shape and strides differ in rank");
    if (shape_numel(shape) != shape_numel(new_shape))
        throw std::invalid_argument("view_strides: cannot view " + shape_str(shape) + " as " + shape_str(new_shape));
    if (shape_numel(shape) == 0) return contiguous_strides(new_shape);
    // Drop size-1 axes: their strides never matter.
    Shape dims, steps;
    for (std::size_t i = 0; i < shape.size(); ++i)
        if (shape[i] != 1) dims.push_back(shape[i]), steps.push_back(strides[i]);
    const std::size_t old_rank = dims.size(), new_rank = new_shape.size();
    Shape             out(new_rank, 1);
    std::size_t       oi = 0, oj = 1, ni = 0, nj = 1;
    while (ni < new_rank && oi < old_rank) {
        std::int64_t np = new_shape[ni], op = dims[oi];
        while (np != op) {
            if (np < op) np *= new_shape[nj++];
            else op *= dims[oj++];
        }
        // The old axes oi..oj-1 merge into the new axes ni..nj-1: they must be contiguous among themselves.
        for (std::size_t k = oi; k + 1 < oj; ++k)
            if (steps[k] != dims[k + 1] * steps[k + 1]) return std::nullopt;
        out[nj - 1] = steps[oj - 1];
        for (std::size_t k = nj - 1; k > ni; --k) out[k - 1] = out[k] * new_shape[k];
        ni = nj++;
        oi = oj++;
    }
    // Trailing new axes have length 1; give them the last stride (any value would do).
    const std::int64_t last = ni > 0 ? out[ni - 1] : 1;
    for (std::size_t k = ni; k < new_rank; ++k) out[k] = last;
    return out;
}

bool can_view(const Tensor& x, const Shape& new_shape) {
    return view_strides(x.shape(), x.strides(), new_shape).has_value();
}

/// @brief Helpers private to this file.
namespace {

/// @brief Shape, strides and offset of the window view, shared by the forward and gradient passes.
struct WindowGeometry {
    /// Output shape.
    Shape shape;
    /// Output strides.
    Shape strides;
};

/// @brief The view geometry of sliding_window() for an input of @p shape and @p strides.
/// @param shape  Input shape.
/// @param strides  Input strides.
/// @param axis  Normalised axis.
/// @param size  Window length.
/// @param step  Window step.
/// @return The geometry.
WindowGeometry window_geometry(const Shape& shape, const Shape& strides, std::int64_t axis, std::int64_t size,
                               std::int64_t step) {
    const auto a = static_cast<std::size_t>(axis);
    WindowGeometry g{shape, strides};
    g.shape[a]   = (shape[a] - size) / step + 1;
    g.strides[a] = strides[a] * step;
    g.shape.push_back(size);
    g.strides.push_back(strides[a]);
    return g;
}

/// @brief zeros(shape) with each window of @p g added back where it came from (the gradient of sliding_window()).
/// @param g  Gradient of the window view.
/// @param shape  Input shape.
/// @param axis  Normalised axis.
/// @param size  Window length.
/// @param step  Window step.
/// @return The input gradient; differentiable, with sliding_window() as its own gradient.
Tensor window_sum(const Tensor& g, const Shape& shape, std::int64_t axis, std::int64_t size, std::int64_t step) {
    Tensor out = zeros(shape, g.dtype());
    {
        NoGradGuard  guard;
        const Tensor gg = nograd(g);
        // Window offset k covers positions k, k + step, ...: one strided add per offset.
        const auto   geo = window_geometry(shape, out.strides(), axis, size, step);
        Shape        part_shape(geo.shape.begin(), geo.shape.end() - 1);
        Shape        part_strides(geo.strides.begin(), geo.strides.end() - 1);
        for (std::int64_t k = 0; k < size; ++k) {
            Tensor target = make_view(out, part_shape, part_strides, out.offset() + k * out.strides()[static_cast<std::size_t>(axis)]);
            target.add_(select(gg, -1, k));
        }
    }
    record("SlidingWindowGrad", {g}, {out}, [axis, size, step](const std::vector<Tensor>& gg, const std::vector<Tensor>&) {
        return std::vector<Tensor>{sliding_window(gg[0], axis, size, step)};
    });
    return out;
}

}  // namespace

Tensor sliding_window(const Tensor& x, std::int64_t axis_in, std::int64_t size, std::int64_t step) {
    const std::int64_t axis = detail::normalize_axis(axis_in, x.dim(), "sliding_window");
    const std::int64_t n    = x.shape()[static_cast<std::size_t>(axis)];
    if (size < 1 || size > n)
        throw std::invalid_argument("sliding_window: size " + std::to_string(size) + " outside [1, " + std::to_string(n) + "]");
    if (step < 1) throw std::invalid_argument("sliding_window: step must be at least 1");
    const auto   geo   = window_geometry(x.shape(), x.strides(), axis, size, step);
    Tensor       out   = make_view(nograd(x), geo.shape, geo.strides, x.offset());
    const Shape  shape = x.shape();
    record("SlidingWindow", {x}, {out}, [shape, axis, size, step](const std::vector<Tensor>& g, const std::vector<Tensor>&) {
        return std::vector<Tensor>{window_sum(g[0], shape, axis, size, step)};
    });
    return out;
}

bool may_share_memory(const Tensor& a, const Tensor& b) {
    if (!a.defined() || !b.defined() || a.numel() == 0 || b.numel() == 0) return false;
    if (a.impl()->storage != b.impl()->storage) return false;
    // Byte range [lo, hi) each tensor can touch.
    auto range = [](const Tensor& t) {
        std::int64_t lo = t.offset(), hi = t.offset();
        for (std::size_t i = 0; i < t.shape().size(); ++i) {
            const std::int64_t reach = (t.shape()[i] - 1) * t.strides()[i];
            (reach < 0 ? lo : hi) += reach;
        }
        const auto item = static_cast<std::int64_t>(dtype_size(t.dtype()));
        return std::pair{lo * item, (hi + 1) * item};
    };
    const auto [alo, ahi] = range(a);
    const auto [blo, bhi] = range(b);
    return alo < bhi && blo < ahi;
}

std::size_t storage_bytes(const Tensor& x) {
    if (!x.defined()) throw std::logic_error("storage_bytes: undefined tensor");
    return x.impl()->storage->nbytes;
}

}  // namespace datamunge::nn
