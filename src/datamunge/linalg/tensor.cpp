#include <datamunge/linalg/tensor.hpp>

#include "../nn/gemm.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace datamunge::linalg {

namespace {

std::size_t total_size(const std::vector<std::size_t>& shape) {
    std::size_t n = 1;
    for (const auto d : shape) n *= d;
    return n;
}

void require_numeric(const Tensor& t, const char* op) {
    if (t.dtype() == TensorDType::String)
        throw std::invalid_argument(std::string("Tensor::") + op +
                                     ": operation requires a numeric (float64/bool) dtype, got 'string'");
}

std::vector<std::size_t> broadcast_shape(const std::vector<std::size_t>& a, const std::vector<std::size_t>& b) {
    const std::size_t na = a.size(), nb = b.size(), n = std::max(na, nb);
    std::vector<std::size_t> result(n);
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t da = (i < na) ? a[na - 1 - i] : 1;
        const std::size_t db = (i < nb) ? b[nb - 1 - i] : 1;
        if (da != db && da != 1 && db != 1)
            throw std::invalid_argument("Tensor: shapes are not broadcastable");
        result[n - 1 - i] = da == 1 ? db : da;
    }
    return result;
}

std::vector<std::size_t> shape_without_axis(const std::vector<std::size_t>& shape, std::size_t axis, bool keepdims) {
    std::vector<std::size_t> result;
    result.reserve(shape.size());
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i == axis) {
            if (keepdims) result.push_back(1);
        } else {
            result.push_back(shape[i]);
        }
    }
    return result;
}

// Row-major buffers need no coordinate allocation per element. Broadcast offsets
// advance once per row, with zero strides on expanded dimensions.
template <typename Out, typename A, typename B, typename Fn>
std::vector<Out> broadcast_values(const std::vector<std::size_t>& shape,
                                 const std::vector<std::size_t>& a_shape,
                                 const std::vector<std::size_t>& b_shape,
                                 const A& a, const B& b, Fn fn) {
    std::vector<Out> values(total_size(shape));
    if (values.empty()) return values;
    if (a_shape == shape && b_shape == shape) {
        for (std::size_t i = 0; i < values.size(); ++i) values[i] = fn(a[i], b[i]);
        return values;
    }
    if (a.size() == 1 && b_shape == shape) {
        for (std::size_t i = 0; i < values.size(); ++i) values[i] = fn(a[0], b[i]);
        return values;
    }
    if (b.size() == 1 && a_shape == shape) {
        for (std::size_t i = 0; i < values.size(); ++i) values[i] = fn(a[i], b[0]);
        return values;
    }
    const auto rank = shape.size();
    std::vector<std::size_t> sa(rank, 0), sb(rank, 0), index(rank, 0);
    std::size_t as = 1, bs = 1;
    for (std::size_t i = 0; i < rank; ++i) {
        if (i < a_shape.size()) {
            auto d = a_shape[a_shape.size() - 1 - i];
            sa[rank - 1 - i] = d == 1 ? 0 : as;
            as *= d;
        }
        if (i < b_shape.size()) {
            auto d = b_shape[b_shape.size() - 1 - i];
            sb[rank - 1 - i] = d == 1 ? 0 : bs;
            bs *= d;
        }
    }
    const auto inner = shape.back();
    std::size_t ao = 0, bo = 0;
    for (std::size_t base = 0; base < values.size(); base += inner) {
        for (std::size_t j = 0; j < inner; ++j)
            values[base + j] = fn(a[ao + j * sa.back()], b[bo + j * sb.back()]);
        for (std::size_t axis = rank - 1; axis-- > 0;) {
            if (++index[axis] < shape[axis]) { ao += sa[axis]; bo += sb[axis]; break; }
            index[axis] = 0;
            ao -= (shape[axis] - 1) * sa[axis];
            bo -= (shape[axis] - 1) * sb[axis];
        }
    }
    return values;
}

template <typename Data, typename BinOp>
Tensor elementwise_binary(const Tensor& a, const Tensor& b, const Data& av, const Data& bv,
                          const char* op, BinOp fn) {
    require_numeric(a, op);
    require_numeric(b, op);
    const auto shape = broadcast_shape(a.shape(), b.shape());
    return std::visit([&](const auto& x, const auto& y) -> Tensor {
        using X = typename std::decay_t<decltype(x)>::value_type;
        using Y = typename std::decay_t<decltype(y)>::value_type;
        if constexpr (std::is_same_v<X, std::string> || std::is_same_v<Y, std::string>) {
            throw std::logic_error("Tensor: unexpected string arithmetic");
        } else {
            return Tensor::from_values(shape, broadcast_values<double>(shape, a.shape(), b.shape(), x, y, fn));
        }
    }, av, bv);
}

template <typename Data, typename Cmp>
Tensor elementwise_compare(const Tensor& a, const Tensor& b, const Data& av, const Data& bv,
                           const char* op, Cmp fn) {
    if (a.dtype() != b.dtype())
        throw std::invalid_argument(std::string("Tensor::") + op + ": dtype mismatch between operands");
    const auto shape = broadcast_shape(a.shape(), b.shape());
    return std::visit([&](const auto& x, const auto& y) -> Tensor {
        if constexpr (std::is_same_v<std::decay_t<decltype(x)>, std::decay_t<decltype(y)>>) {
            return Tensor::from_bool_values(shape, broadcast_values<int>(shape, a.shape(), b.shape(), x, y, fn));
        } else {
            throw std::logic_error("Tensor: unexpected comparison dtype mismatch");
        }
    }, av, bv);
}

enum class AxisReduceOp { Sum, Mean, Max, Min, Prod, ArgMax, ArgMin };

// Float storage is already packed. Only bool operands need numeric conversion.
template <typename Data>
const double* numeric_buffer(const Data& data, std::vector<double>& converted) {
    if (const auto* values = std::get_if<std::vector<double>>(&data)) return values->data();
    const auto& bits = std::get<std::vector<std::uint8_t>>(data);
    converted.assign(bits.begin(), bits.end());
    return converted.data();
}

template <typename Data, typename Fn>
auto numeric_values(const Data& data, Fn fn) -> decltype(fn(std::declval<const std::vector<double>&>())) {
    using Result = decltype(fn(std::declval<const std::vector<double>&>()));
    return std::visit([&](const auto& values) -> Result {
        using Value = typename std::decay_t<decltype(values)>::value_type;
        if constexpr (std::is_same_v<Value, std::string>)
            throw std::logic_error("Tensor: unexpected string arithmetic");
        else return fn(values);
    }, data);
}

template <typename Data>
Tensor reduce_along_axis(const Tensor& t, const Data& data, std::size_t axis, bool keepdims,
                         AxisReduceOp op, const char* name) {
    require_numeric(t, name);
    if (axis >= t.ndim()) throw std::out_of_range(std::string("Tensor::") + name + ": axis out of range");
    const std::size_t axis_len = t.shape()[axis];
    if (axis_len == 0) throw std::invalid_argument(std::string("Tensor::") + name + ": cannot reduce over empty axis");
    const auto out_shape = shape_without_axis(t.shape(), axis, keepdims);
    std::vector<double> result(total_size(out_shape));
    if (result.empty()) return Tensor::from_values(out_shape, std::move(result));
    std::size_t inner = 1;
    for (std::size_t d = axis + 1; d < t.ndim(); ++d) inner *= t.shape()[d];
    numeric_values(data, [&](const auto& values) {
        for (std::size_t k = 0; k < result.size(); ++k) {
            double acc = 0.0;
            std::size_t best_j = 0;
            switch (op) {
                case AxisReduceOp::Sum:
                case AxisReduceOp::Mean: break;
                case AxisReduceOp::Prod: acc = 1.0; break;
                case AxisReduceOp::Max:
                case AxisReduceOp::ArgMax: acc = -std::numeric_limits<double>::infinity(); break;
                case AxisReduceOp::Min:
                case AxisReduceOp::ArgMin: acc = std::numeric_limits<double>::infinity(); break;
            }
            const std::size_t base = (k / inner) * axis_len * inner + k % inner;
            for (std::size_t j = 0; j < axis_len; ++j) {
                const double v = values[base + j * inner];
                switch (op) {
                    case AxisReduceOp::Sum:
                    case AxisReduceOp::Mean: acc += v; break;
                    case AxisReduceOp::Prod: acc *= v; break;
                    case AxisReduceOp::Max: if (v > acc) acc = v; break;
                    case AxisReduceOp::Min: if (v < acc) acc = v; break;
                    case AxisReduceOp::ArgMax: if (v > acc) { acc = v; best_j = j; } break;
                    case AxisReduceOp::ArgMin: if (v < acc) { acc = v; best_j = j; } break;
                }
            }
            if (op == AxisReduceOp::Mean) acc /= static_cast<double>(axis_len);
            if (op == AxisReduceOp::ArgMax || op == AxisReduceOp::ArgMin) acc = static_cast<double>(best_j);
            result[k] = acc;
        }
        return 0;
    });
    return Tensor::from_values(out_shape, std::move(result));
}

} // namespace

Tensor::Tensor() : Tensor(std::vector<std::size_t>{}, TensorDType::Float64) {}

Tensor::Tensor(std::vector<std::size_t> shape, TensorDType dtype) : shape_(std::move(shape)) {
    compute_strides();
    const std::size_t n = total_size(shape_);
    switch (dtype) {
        case TensorDType::Float64: data_ = std::vector<double>(n, 0.0); break;
        case TensorDType::Bool: data_ = std::vector<std::uint8_t>(n, 0); break;
        case TensorDType::String: data_ = std::vector<std::string>(n, std::string{}); break;
    }
}

void Tensor::compute_strides() {
    strides_.assign(shape_.size(), 0);
    std::size_t stride = 1;
    for (std::size_t i = shape_.size(); i-- > 0;) {
        strides_[i] = stride;
        stride *= shape_[i];
    }
}

std::size_t Tensor::flat_index(const std::vector<std::size_t>& index) const {
    if (index.size() != shape_.size())
        throw std::invalid_argument("Tensor: index rank does not match tensor rank");
    std::size_t flat = 0;
    for (std::size_t i = 0; i < index.size(); ++i) {
        if (index[i] >= shape_[i]) throw std::out_of_range("Tensor: index out of range");
        flat += index[i] * strides_[i];
    }
    return flat;
}

Tensor Tensor::zeros(std::vector<std::size_t> shape) { return Tensor(std::move(shape), TensorDType::Float64); }

Tensor Tensor::ones(std::vector<std::size_t> shape) { return full(std::move(shape), 1.0); }

Tensor Tensor::full(std::vector<std::size_t> shape, double value) {
    Tensor t(std::move(shape), TensorDType::Float64);
    std::get<std::vector<double>>(t.data_).assign(t.size(), value);
    return t;
}

Tensor Tensor::from_values(std::vector<std::size_t> shape, std::vector<double> values) {
    if (values.size() != total_size(shape))
        throw std::invalid_argument("Tensor::from_values: value count does not match shape");
    Tensor t;
    t.shape_ = std::move(shape);
    t.compute_strides();
    std::get<std::vector<double>>(t.data_) = std::move(values);
    return t;
}

Tensor Tensor::from_bool_values(std::vector<std::size_t> shape, std::vector<int> values) {
    if (values.size() != total_size(shape))
        throw std::invalid_argument("Tensor::from_bool_values: value count does not match shape");
    Tensor t(std::move(shape), TensorDType::Bool);
    auto& vec = std::get<std::vector<std::uint8_t>>(t.data_);
    for (std::size_t i = 0; i < values.size(); ++i) vec[i] = values[i] != 0 ? 1 : 0;
    return t;
}

Tensor Tensor::from_string_values(std::vector<std::size_t> shape, std::vector<std::string> values) {
    if (values.size() != total_size(shape))
        throw std::invalid_argument("Tensor::from_string_values: value count does not match shape");
    Tensor t(std::move(shape), TensorDType::String);
    std::get<std::vector<std::string>>(t.data_) = std::move(values);
    return t;
}

Tensor Tensor::arange(double start, double stop, double step) {
    if (step == 0.0) throw std::invalid_argument("Tensor::arange: step must be nonzero");
    std::vector<double> values;
    if (step > 0.0) {
        for (double v = start; v < stop; v += step) values.push_back(v);
    } else {
        for (double v = start; v > stop; v += step) values.push_back(v);
    }
    const std::size_t n = values.size();
    return from_values({n}, std::move(values));
}

Tensor Tensor::eye(std::size_t n) {
    Tensor t({n, n}, TensorDType::Float64);
    auto& vec = std::get<std::vector<double>>(t.data_);
    for (std::size_t i = 0; i < n; ++i) vec[i * n + i] = 1.0;
    return t;
}

std::size_t Tensor::ndim() const noexcept { return shape_.size(); }

const std::vector<std::size_t>& Tensor::shape() const noexcept { return shape_; }

std::size_t Tensor::size() const noexcept {
    return std::visit([](const auto& v) { return v.size(); }, data_);
}

TensorDType Tensor::dtype() const noexcept {
    switch (data_.index()) {
        case 0: return TensorDType::Float64;
        case 1: return TensorDType::Bool;
        default: return TensorDType::String;
    }
}

std::string Tensor::dtype_name() const {
    switch (dtype()) {
        case TensorDType::Float64: return "float64";
        case TensorDType::Bool: return "bool";
        case TensorDType::String: return "string";
    }
    return "unknown";
}

double Tensor::at(const std::vector<std::size_t>& index) const { return at_flat(flat_index(index)); }

void Tensor::set(const std::vector<std::size_t>& index, double value) { set_flat(flat_index(index), value); }

std::string Tensor::string_at(const std::vector<std::size_t>& index) const {
    return string_at_flat(flat_index(index));
}

void Tensor::set_string(const std::vector<std::size_t>& index, const std::string& value) {
    set_string_flat(flat_index(index), value);
}

double Tensor::at_flat(std::size_t i) const {
    if (i >= size()) throw std::out_of_range("Tensor::at_flat: index out of range");
    if (const auto* v = std::get_if<std::vector<double>>(&data_)) return (*v)[i];
    if (const auto* v = std::get_if<std::vector<std::uint8_t>>(&data_)) return (*v)[i] != 0 ? 1.0 : 0.0;
    throw std::logic_error("Tensor::at: tensor dtype is 'string' -- use string_at instead");
}

void Tensor::set_flat(std::size_t i, double value) {
    if (i >= size()) throw std::out_of_range("Tensor::set_flat: index out of range");
    if (auto* v = std::get_if<std::vector<double>>(&data_)) {
        (*v)[i] = value;
        return;
    }
    if (auto* v = std::get_if<std::vector<std::uint8_t>>(&data_)) {
        (*v)[i] = value != 0.0 ? 1 : 0;
        return;
    }
    throw std::logic_error("Tensor::set: tensor dtype is 'string' -- use set_string instead");
}

std::string Tensor::string_at_flat(std::size_t i) const {
    if (i >= size()) throw std::out_of_range("Tensor::string_at_flat: index out of range");
    const auto* v = std::get_if<std::vector<std::string>>(&data_);
    if (!v) throw std::logic_error("Tensor::string_at: tensor dtype is not 'string'");
    return (*v)[i];
}

void Tensor::set_string_flat(std::size_t i, const std::string& value) {
    if (i >= size()) throw std::out_of_range("Tensor::set_string_flat: index out of range");
    auto* v = std::get_if<std::vector<std::string>>(&data_);
    if (!v) throw std::logic_error("Tensor::set_string: tensor dtype is not 'string'");
    (*v)[i] = value;
}

Tensor Tensor::reshape(std::vector<std::size_t> new_shape) const {
    if (total_size(new_shape) != size()) throw std::invalid_argument("Tensor::reshape: total size mismatch");
    Tensor result(*this);
    result.shape_ = std::move(new_shape);
    result.compute_strides();
    return result;
}

Tensor Tensor::flatten() const { return reshape({size()}); }

Tensor Tensor::transpose(std::vector<std::size_t> permutation) const {
    if (permutation.empty()) {
        permutation.resize(ndim());
        for (std::size_t i = 0; i < ndim(); ++i) permutation[i] = ndim() - 1 - i;
    }
    if (permutation.size() != ndim())
        throw std::invalid_argument("Tensor::transpose: permutation size does not match tensor rank");
    // Identity permutations are valid by construction and still make an
    // independent vector copy, without a zero-fill or stride reconstruction.
    bool identity = true;
    for (std::size_t axis = 0; axis < ndim(); ++axis)
        if (permutation[axis] != axis) { identity = false; break; }
    if (identity && total_size(shape_) == size()) return *this;
    std::vector<bool> seen(ndim(), false);
    for (const auto p : permutation) {
        if (p >= ndim() || seen[p]) throw std::invalid_argument("Tensor::transpose: invalid permutation");
        seen[p] = true;
    }
    std::vector<std::size_t> new_shape(ndim());
    for (std::size_t i = 0; i < ndim(); ++i) new_shape[i] = shape_[permutation[i]];

    Tensor result(new_shape, dtype());
    if (size() == 0) return result;
    if (ndim() == 2 && permutation[0] == 1 && shape_[0] >= 64 && shape_[1] >= 64 && size() >= 32768) {
        if (const auto* source = std::get_if<std::vector<double>>(&data_)) {
            auto& dest = std::get<std::vector<double>>(result.data_);
            // Bounded tiles reuse source cache lines, with contiguous output
            // writes. Keep this path outside the generic dtype/rank visitor.
            constexpr std::size_t tile_rows = 8, tile_cols = 32;
            const auto rows = shape_[0], cols = shape_[1];
            for (std::size_t c0 = 0; c0 < cols; c0 += tile_cols)
                for (std::size_t r0 = 0; r0 < rows; r0 += tile_rows)
                    for (std::size_t c = c0; c < std::min(cols, c0 + tile_cols); ++c)
                        for (std::size_t r = r0; r < std::min(rows, r0 + tile_rows); ++r)
                            dest[c * rows + r] = (*source)[r * cols + c];
            return result;
        }
    }
    std::vector<std::size_t> source_strides(ndim());
    for (std::size_t i = 0; i < ndim(); ++i) source_strides[i] = strides_[permutation[i]];
    std::visit([&](const auto& source) {
        auto& dest = std::get<std::decay_t<decltype(source)>>(result.data_);
        if (ndim() == 0) { dest[0] = source[0]; return; }
        const auto inner = new_shape.back();
        std::vector<std::size_t> index(ndim(), 0);
        std::size_t offset = 0;
        for (std::size_t base = 0; base < size(); base += inner) {
            for (std::size_t j = 0; j < inner; ++j) dest[base + j] = source[offset + j * source_strides.back()];
            for (std::size_t axis = ndim() - 1; axis-- > 0;) {
                if (++index[axis] < new_shape[axis]) { offset += source_strides[axis]; break; }
                index[axis] = 0;
                offset -= (new_shape[axis] - 1) * source_strides[axis];
            }
        }
    }, data_);
    return result;
}

Tensor Tensor::squeeze() const {
    std::vector<std::size_t> new_shape;
    for (const auto d : shape_)
        if (d != 1) new_shape.push_back(d);
    return reshape(new_shape);
}

Tensor Tensor::squeeze_axis(std::size_t axis) const {
    if (axis >= ndim()) throw std::out_of_range("Tensor::squeeze_axis: axis out of range");
    if (shape_[axis] != 1) throw std::invalid_argument("Tensor::squeeze_axis: dimension is not size 1");
    std::vector<std::size_t> new_shape;
    for (std::size_t i = 0; i < ndim(); ++i)
        if (i != axis) new_shape.push_back(shape_[i]);
    return reshape(new_shape);
}

Tensor Tensor::expand_dims(std::size_t axis) const {
    if (axis > ndim()) throw std::out_of_range("Tensor::expand_dims: axis out of range");
    std::vector<std::size_t> new_shape = shape_;
    new_shape.insert(new_shape.begin() + static_cast<std::ptrdiff_t>(axis), 1);
    return reshape(new_shape);
}

Tensor Tensor::slice(std::size_t axis, std::size_t start, std::size_t stop, std::size_t step) const {
    if (axis >= ndim()) throw std::out_of_range("Tensor::slice: axis out of range");
    if (step == 0) throw std::invalid_argument("Tensor::slice: step must be nonzero");
    if (start > stop || stop > shape_[axis]) throw std::out_of_range("Tensor::slice: start/stop out of range");
    const std::size_t out_len = (stop - start + step - 1) / step;
    std::vector<std::size_t> new_shape = shape_;
    new_shape[axis] = out_len;
    Tensor result(new_shape, dtype());
    if (result.size() == 0) return result;
    const auto inner = strides_[axis];
    const auto outer = size() / (shape_[axis] * inner);
    std::visit([&](const auto& source) {
        auto& dest = std::get<std::decay_t<decltype(source)>>(result.data_);
        for (std::size_t i = 0; i < outer; ++i)
            for (std::size_t j = 0; j < out_len; ++j)
                std::copy_n(source.begin() + (i * shape_[axis] + start + j * step) * inner,
                            inner, dest.begin() + (i * out_len + j) * inner);
    }, data_);
    return result;
}

Tensor Tensor::index_select(std::size_t axis, const std::vector<std::size_t>& indices) const {
    if (axis >= ndim()) throw std::out_of_range("Tensor::index_select: axis out of range");
    for (const auto i : indices)
        if (i >= shape_[axis]) throw std::out_of_range("Tensor::index_select: index out of range");
    std::vector<std::size_t> new_shape = shape_;
    new_shape[axis] = indices.size();
    Tensor result(new_shape, dtype());
    if (result.size() == 0) return result;
    const auto inner = strides_[axis];
    const auto outer = size() / (shape_[axis] * inner);
    std::visit([&](const auto& source) {
        auto& dest = std::get<std::decay_t<decltype(source)>>(result.data_);
        for (std::size_t i = 0; i < outer; ++i)
            for (std::size_t j = 0; j < indices.size(); ++j)
                std::copy_n(source.begin() + (i * shape_[axis] + indices[j]) * inner,
                            inner, dest.begin() + (i * indices.size() + j) * inner);
    }, data_);
    return result;
}

Tensor Tensor::concatenate(const std::vector<Tensor>& tensors, std::size_t axis) {
    if (tensors.empty()) throw std::invalid_argument("Tensor::concatenate: no tensors given");
    const auto& first = tensors.front();
    if (axis >= first.ndim()) throw std::out_of_range("Tensor::concatenate: axis out of range");
    std::vector<std::size_t> new_shape = first.shape_;
    std::size_t axis_total = 0;
    for (const auto& t : tensors) {
        if (t.ndim() != first.ndim()) throw std::invalid_argument("Tensor::concatenate: rank mismatch");
        if (t.dtype() != first.dtype()) throw std::invalid_argument("Tensor::concatenate: dtype mismatch");
        for (std::size_t d = 0; d < first.ndim(); ++d) {
            if (d == axis) continue;
            if (t.shape_[d] != first.shape_[d])
                throw std::invalid_argument("Tensor::concatenate: shape mismatch on non-concatenation axis");
        }
        axis_total += t.shape_[axis];
    }
    new_shape[axis] = axis_total;
    Tensor result(new_shape, first.dtype());
    if (result.size() == 0) return result;
    const auto inner = first.strides_[axis];
    const auto outer = result.size() / (axis_total * inner);
    std::visit([&](auto& dest) {
        using Values = std::decay_t<decltype(dest)>;
        for (std::size_t i = 0; i < outer; ++i) {
            std::size_t offset = 0;
            for (const auto& tensor : tensors) {
                const auto& source = std::get<Values>(tensor.data_);
                const auto count = tensor.shape_[axis] * inner;
                std::copy_n(source.begin() + i * count, count,
                            dest.begin() + i * axis_total * inner + offset);
                offset += count;
            }
        }
    }, result.data_);
    return result;
}

Tensor Tensor::stack(const std::vector<Tensor>& tensors, std::size_t axis) {
    if (tensors.empty()) throw std::invalid_argument("Tensor::stack: no tensors given");
    std::vector<Tensor> expanded;
    expanded.reserve(tensors.size());
    for (const auto& t : tensors) expanded.push_back(t.expand_dims(axis));
    return concatenate(expanded, axis);
}

Tensor Tensor::add(const Tensor& other) const {
    return elementwise_binary(*this, other, data_, other.data_, "add", [](double x, double y) { return x + y; });
}
Tensor Tensor::subtract(const Tensor& other) const {
    return elementwise_binary(*this, other, data_, other.data_, "subtract", [](double x, double y) { return x - y; });
}
Tensor Tensor::multiply(const Tensor& other) const {
    return elementwise_binary(*this, other, data_, other.data_, "multiply", [](double x, double y) { return x * y; });
}
Tensor Tensor::divide(const Tensor& other) const {
    return elementwise_binary(*this, other, data_, other.data_, "divide", [](double x, double y) { return x / y; });
}
Tensor Tensor::power(const Tensor& other) const {
    return elementwise_binary(*this, other, data_, other.data_, "power", [](double x, double y) { return std::pow(x, y); });
}

Tensor Tensor::add_scalar(double scalar) const { return apply([scalar](double x) { return x + scalar; }); }
Tensor Tensor::subtract_scalar(double scalar) const { return apply([scalar](double x) { return x - scalar; }); }
Tensor Tensor::multiply_scalar(double scalar) const { return apply([scalar](double x) { return x * scalar; }); }
Tensor Tensor::divide_scalar(double scalar) const { return apply([scalar](double x) { return x / scalar; }); }
Tensor Tensor::power_scalar(double exponent) const {
    return apply([exponent](double x) { return std::pow(x, exponent); });
}

Tensor Tensor::negate() const { return apply([](double x) { return -x; }); }
Tensor Tensor::abs() const { return apply([](double x) { return std::fabs(x); }); }
Tensor Tensor::sqrt() const { return apply([](double x) { return std::sqrt(x); }); }
Tensor Tensor::exp() const { return apply([](double x) { return std::exp(x); }); }
Tensor Tensor::log() const { return apply([](double x) { return std::log(x); }); }

Tensor Tensor::apply(const std::function<double(double)>& fn) const {
    require_numeric(*this, "apply");
    Tensor result(shape_, TensorDType::Float64);
    auto& dest = std::get<std::vector<double>>(result.data_);
    std::visit([&](const auto& source) {
        using Value = typename std::decay_t<decltype(source)>::value_type;
        if constexpr (!std::is_same_v<Value, std::string>)
            for (std::size_t i = 0; i < size(); ++i) dest[i] = fn(static_cast<double>(source[i]));
    }, data_);
    return result;
}

Tensor Tensor::equal(const Tensor& other) const {
    return elementwise_compare(*this, other, data_, other.data_, "equal", std::equal_to<>{});
}
Tensor Tensor::not_equal(const Tensor& other) const {
    return elementwise_compare(*this, other, data_, other.data_, "not_equal", std::not_equal_to<>{});
}
Tensor Tensor::less(const Tensor& other) const {
    return elementwise_compare(*this, other, data_, other.data_, "less", std::less<>{});
}
Tensor Tensor::less_equal(const Tensor& other) const {
    return elementwise_compare(*this, other, data_, other.data_, "less_equal", std::less_equal<>{});
}
Tensor Tensor::greater(const Tensor& other) const {
    return elementwise_compare(*this, other, data_, other.data_, "greater", std::greater<>{});
}
Tensor Tensor::greater_equal(const Tensor& other) const {
    return elementwise_compare(*this, other, data_, other.data_, "greater_equal", std::greater_equal<>{});
}

double Tensor::sum() const {
    require_numeric(*this, "sum");
    return numeric_values(data_, [](const auto& values) {
        double sum = 0.0;
        for (const auto value : values) sum += static_cast<double>(value);
        return sum;
    });
}

double Tensor::mean() const {
    require_numeric(*this, "mean");
    if (size() == 0) throw std::invalid_argument("Tensor::mean: tensor is empty");
    return sum() / static_cast<double>(size());
}

double Tensor::max() const {
    require_numeric(*this, "max");
    if (size() == 0) throw std::invalid_argument("Tensor::max: tensor is empty");
    return numeric_values(data_, [](const auto& values) {
        double best = values[0];
        for (std::size_t i = 1; i < values.size(); ++i) best = std::max(best, static_cast<double>(values[i]));
        return best;
    });
}

double Tensor::min() const {
    require_numeric(*this, "min");
    if (size() == 0) throw std::invalid_argument("Tensor::min: tensor is empty");
    return numeric_values(data_, [](const auto& values) {
        double best = values[0];
        for (std::size_t i = 1; i < values.size(); ++i) best = std::min(best, static_cast<double>(values[i]));
        return best;
    });
}

double Tensor::prod() const {
    require_numeric(*this, "prod");
    return numeric_values(data_, [](const auto& values) {
        double product = 1.0;
        for (const auto value : values) product *= static_cast<double>(value);
        return product;
    });
}

std::size_t Tensor::argmax() const {
    require_numeric(*this, "argmax");
    if (size() == 0) throw std::invalid_argument("Tensor::argmax: tensor is empty");
    return numeric_values(data_, [](const auto& values) {
        std::size_t best = 0;
        for (std::size_t i = 1; i < values.size(); ++i)
            if (values[i] > values[best]) best = i;
        return best;
    });
}

std::size_t Tensor::argmin() const {
    require_numeric(*this, "argmin");
    if (size() == 0) throw std::invalid_argument("Tensor::argmin: tensor is empty");
    return numeric_values(data_, [](const auto& values) {
        std::size_t best = 0;
        for (std::size_t i = 1; i < values.size(); ++i)
            if (values[i] < values[best]) best = i;
        return best;
    });
}

bool Tensor::all() const {
    require_numeric(*this, "all");
    return numeric_values(data_, [](const auto& values) {
        for (const auto value : values) if (value == 0) return false;
        return true;
    });
}

bool Tensor::any() const {
    require_numeric(*this, "any");
    return numeric_values(data_, [](const auto& values) {
        for (const auto value : values) if (value != 0) return true;
        return false;
    });
}

Tensor Tensor::sum_axis(std::size_t axis, bool keepdims) const {
    return reduce_along_axis(*this, data_, axis, keepdims, AxisReduceOp::Sum, "sum_axis");
}
Tensor Tensor::mean_axis(std::size_t axis, bool keepdims) const {
    return reduce_along_axis(*this, data_, axis, keepdims, AxisReduceOp::Mean, "mean_axis");
}
Tensor Tensor::max_axis(std::size_t axis, bool keepdims) const {
    return reduce_along_axis(*this, data_, axis, keepdims, AxisReduceOp::Max, "max_axis");
}
Tensor Tensor::min_axis(std::size_t axis, bool keepdims) const {
    return reduce_along_axis(*this, data_, axis, keepdims, AxisReduceOp::Min, "min_axis");
}
Tensor Tensor::prod_axis(std::size_t axis, bool keepdims) const {
    return reduce_along_axis(*this, data_, axis, keepdims, AxisReduceOp::Prod, "prod_axis");
}
Tensor Tensor::argmax_axis(std::size_t axis, bool keepdims) const {
    return reduce_along_axis(*this, data_, axis, keepdims, AxisReduceOp::ArgMax, "argmax_axis");
}
Tensor Tensor::argmin_axis(std::size_t axis, bool keepdims) const {
    return reduce_along_axis(*this, data_, axis, keepdims, AxisReduceOp::ArgMin, "argmin_axis");
}

Tensor Tensor::matmul(const Tensor& other) const {
    require_numeric(*this, "matmul");
    require_numeric(other, "matmul");
    if (ndim() != 2 || other.ndim() != 2)
        throw std::invalid_argument("Tensor::matmul: both tensors must be 2-D (use dot/outer for 1-D)");
    if (shape_[1] != other.shape_[0]) throw std::invalid_argument("Tensor::matmul: inner dimensions do not match");
    const std::size_t m = shape_[0], k = shape_[1], n = other.shape_[1];
    Tensor result({m, n}, TensorDType::Float64);
    if (m == 0 || n == 0 || k == 0) return result;
    std::vector<double> left_conversion, right_conversion;
    const double* left = numeric_buffer(data_, left_conversion);
    const double* right = numeric_buffer(other.data_, right_conversion);
    double* output = std::get<std::vector<double>>(result.data_).data();
    // The shared GEMM's CBLAS interface uses int dimensions. Keep a raw-buffer
    // fallback for larger shapes instead of silently narrowing their dimensions.
    if (std::max({m, n, k}) <= static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        datamunge::nn::detail::gemm(false, false, static_cast<std::int64_t>(m),
            static_cast<std::int64_t>(n), static_cast<std::int64_t>(k), 1.0,
            left, static_cast<std::int64_t>(k), right, static_cast<std::int64_t>(n),
            0.0, output, static_cast<std::int64_t>(n));
    } else {
        for (std::size_t i = 0; i < m; ++i)
            for (std::size_t p = 0; p < k; ++p)
                for (std::size_t j = 0; j < n; ++j)
                    output[i * n + j] += left[i * k + p] * right[p * n + j];
    }
    return result;
}

double Tensor::dot(const Tensor& other) const {
    require_numeric(*this, "dot");
    require_numeric(other, "dot");
    if (ndim() != 1 || other.ndim() != 1)
        throw std::invalid_argument("Tensor::dot: both tensors must be 1-D (use matmul for 2-D)");
    if (shape_[0] != other.shape_[0]) throw std::invalid_argument("Tensor::dot: length mismatch");
    std::vector<double> left_conversion, right_conversion;
    const double* left = numeric_buffer(data_, left_conversion);
    const double* right = numeric_buffer(other.data_, right_conversion);
    double s = 0.0;
    for (std::size_t i = 0; i < shape_[0]; ++i) s += left[i] * right[i];
    return s;
}

Tensor Tensor::outer(const Tensor& other) const {
    require_numeric(*this, "outer");
    require_numeric(other, "outer");
    if (ndim() != 1 || other.ndim() != 1) throw std::invalid_argument("Tensor::outer: both tensors must be 1-D");
    const std::size_t m = shape_[0], n = other.shape_[0];
    Tensor result({m, n}, TensorDType::Float64);
    std::vector<double> left_conversion, right_conversion;
    const double* left = numeric_buffer(data_, left_conversion);
    const double* right = numeric_buffer(other.data_, right_conversion);
    auto& values = std::get<std::vector<double>>(result.data_);
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j) values[i * n + j] = left[i] * right[j];
    return result;
}

std::string Tensor::to_string(std::size_t max_elements) const {
    std::ostringstream oss;
    oss << "Tensor(shape=[";
    for (std::size_t i = 0; i < shape_.size(); ++i) {
        if (i) oss << ", ";
        oss << shape_[i];
    }
    oss << "], dtype=" << dtype_name() << ")\n";
    const std::size_t n = std::min(size(), max_elements);
    for (std::size_t i = 0; i < n; ++i) {
        if (i) oss << ", ";
        if (dtype() == TensorDType::String)
            oss << '"' << string_at_flat(i) << '"';
        else
            oss << at_flat(i);
    }
    if (size() > max_elements) oss << ", ... (" << (size() - max_elements) << " more)";
    return oss.str();
}

} // namespace datamunge::linalg
