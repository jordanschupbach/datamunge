#include <datamunge/linalg/tensor.hpp>

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace datamunge::linalg {

namespace {

std::size_t total_size(const std::vector<std::size_t>& shape) {
    std::size_t n = 1;
    for (const auto d : shape) n *= d;
    return n;
}

/// @brief Advances a row-major multi-index odometer (last axis fastest).
/// @return false once the index has wrapped back around to all zeros.
bool increment_index(std::vector<std::size_t>& index, const std::vector<std::size_t>& shape) {
    for (std::size_t i = index.size(); i-- > 0;) {
        if (++index[i] < shape[i]) return true;
        index[i] = 0;
    }
    return false;
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
        result[n - 1 - i] = std::max(da, db);
    }
    return result;
}

std::vector<std::size_t> broadcast_operand_index(const std::vector<std::size_t>& out_idx,
                                                   const std::vector<std::size_t>& operand_shape) {
    const std::size_t n_out = out_idx.size();
    const std::size_t n_op = operand_shape.size();
    std::vector<std::size_t> op_idx(n_op);
    for (std::size_t i = 0; i < n_op; ++i) {
        const std::size_t out_pos = n_out - n_op + i;
        op_idx[i] = (operand_shape[i] == 1) ? 0 : out_idx[out_pos];
    }
    return op_idx;
}

void copy_element(const Tensor& src, const std::vector<std::size_t>& src_idx, Tensor& dst,
                   const std::vector<std::size_t>& dst_idx) {
    if (src.dtype() == TensorDType::String)
        dst.set_string(dst_idx, src.string_at(src_idx));
    else
        dst.set(dst_idx, src.at(src_idx));
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

template <typename BinOp>
Tensor elementwise_binary(const Tensor& a, const Tensor& b, const char* op, BinOp fn) {
    require_numeric(a, op);
    require_numeric(b, op);
    const auto out_shape = broadcast_shape(a.shape(), b.shape());
    Tensor result(out_shape, TensorDType::Float64);
    if (result.size() == 0) return result;
    std::vector<std::size_t> idx(out_shape.size(), 0);
    std::size_t k = 0;
    do {
        const auto a_idx = broadcast_operand_index(idx, a.shape());
        const auto b_idx = broadcast_operand_index(idx, b.shape());
        result.set_flat(k++, fn(a.at(a_idx), b.at(b_idx)));
    } while (increment_index(idx, out_shape));
    return result;
}

template <typename Cmp>
Tensor elementwise_compare(const Tensor& a, const Tensor& b, const char* op, Cmp fn) {
    if (a.dtype() != b.dtype())
        throw std::invalid_argument(std::string("Tensor::") + op + ": dtype mismatch between operands");
    const auto out_shape = broadcast_shape(a.shape(), b.shape());
    Tensor result(out_shape, TensorDType::Bool);
    if (result.size() == 0) return result;
    const bool is_string = a.dtype() == TensorDType::String;
    std::vector<std::size_t> idx(out_shape.size(), 0);
    std::size_t k = 0;
    do {
        const auto a_idx = broadcast_operand_index(idx, a.shape());
        const auto b_idx = broadcast_operand_index(idx, b.shape());
        const bool value = is_string ? fn(a.string_at(a_idx), b.string_at(b_idx)) : fn(a.at(a_idx), b.at(b_idx));
        result.set_flat(k++, value ? 1.0 : 0.0);
    } while (increment_index(idx, out_shape));
    return result;
}

enum class AxisReduceOp { Sum, Mean, Max, Min, Prod, ArgMax, ArgMin };

Tensor reduce_along_axis(const Tensor& t, std::size_t axis, bool keepdims, AxisReduceOp op, const char* name) {
    require_numeric(t, name);
    if (axis >= t.ndim()) throw std::out_of_range(std::string("Tensor::") + name + ": axis out of range");
    const std::size_t axis_len = t.shape()[axis];
    if (axis_len == 0) throw std::invalid_argument(std::string("Tensor::") + name + ": cannot reduce over empty axis");

    const auto out_shape = shape_without_axis(t.shape(), axis, keepdims);
    Tensor result(out_shape, TensorDType::Float64);

    const auto iter_shape = shape_without_axis(t.shape(), axis, false);
    std::vector<std::size_t> iter_idx(iter_shape.size(), 0);
    std::vector<std::size_t> src_idx(t.ndim(), 0);
    std::size_t k = 0;
    do {
        std::size_t oi = 0;
        for (std::size_t i = 0; i < t.ndim(); ++i) {
            if (i != axis) src_idx[i] = iter_idx[oi++];
        }
        double acc = 0.0;
        std::size_t best_j = 0;
        switch (op) {
            case AxisReduceOp::Sum:
            case AxisReduceOp::Mean: acc = 0.0; break;
            case AxisReduceOp::Prod: acc = 1.0; break;
            case AxisReduceOp::Max:
            case AxisReduceOp::ArgMax: acc = -std::numeric_limits<double>::infinity(); break;
            case AxisReduceOp::Min:
            case AxisReduceOp::ArgMin: acc = std::numeric_limits<double>::infinity(); break;
        }
        for (std::size_t j = 0; j < axis_len; ++j) {
            src_idx[axis] = j;
            const double v = t.at(src_idx);
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
        result.set_flat(k++, acc);
    } while (increment_index(iter_idx, iter_shape));
    return result;
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
    Tensor t(std::move(shape), TensorDType::Float64);
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
    Tensor result(std::move(new_shape), dtype());
    result.data_ = data_;
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
    std::vector<bool> seen(ndim(), false);
    for (const auto p : permutation) {
        if (p >= ndim() || seen[p]) throw std::invalid_argument("Tensor::transpose: invalid permutation");
        seen[p] = true;
    }
    std::vector<std::size_t> new_shape(ndim());
    for (std::size_t i = 0; i < ndim(); ++i) new_shape[i] = shape_[permutation[i]];

    Tensor result(new_shape, dtype());
    if (size() == 0) return result;
    std::vector<std::size_t> idx(ndim(), 0);
    do {
        std::vector<std::size_t> new_idx(ndim());
        for (std::size_t i = 0; i < ndim(); ++i) new_idx[i] = idx[permutation[i]];
        copy_element(*this, idx, result, new_idx);
    } while (increment_index(idx, shape_));
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
    if (out_len == 0) return result;
    std::vector<std::size_t> idx(ndim(), 0);
    do {
        std::vector<std::size_t> src_idx = idx;
        src_idx[axis] = start + idx[axis] * step;
        copy_element(*this, src_idx, result, idx);
    } while (increment_index(idx, new_shape));
    return result;
}

Tensor Tensor::index_select(std::size_t axis, const std::vector<std::size_t>& indices) const {
    if (axis >= ndim()) throw std::out_of_range("Tensor::index_select: axis out of range");
    for (const auto i : indices)
        if (i >= shape_[axis]) throw std::out_of_range("Tensor::index_select: index out of range");
    std::vector<std::size_t> new_shape = shape_;
    new_shape[axis] = indices.size();
    Tensor result(new_shape, dtype());
    if (indices.empty()) return result;
    std::vector<std::size_t> idx(ndim(), 0);
    do {
        std::vector<std::size_t> src_idx = idx;
        src_idx[axis] = indices[idx[axis]];
        copy_element(*this, src_idx, result, idx);
    } while (increment_index(idx, new_shape));
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
    std::size_t offset = 0;
    for (const auto& t : tensors) {
        if (t.size() > 0) {
            std::vector<std::size_t> idx(t.ndim(), 0);
            do {
                std::vector<std::size_t> dst_idx = idx;
                dst_idx[axis] += offset;
                copy_element(t, idx, result, dst_idx);
            } while (increment_index(idx, t.shape_));
        }
        offset += t.shape_[axis];
    }
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
    return elementwise_binary(*this, other, "add", [](double x, double y) { return x + y; });
}
Tensor Tensor::subtract(const Tensor& other) const {
    return elementwise_binary(*this, other, "subtract", [](double x, double y) { return x - y; });
}
Tensor Tensor::multiply(const Tensor& other) const {
    return elementwise_binary(*this, other, "multiply", [](double x, double y) { return x * y; });
}
Tensor Tensor::divide(const Tensor& other) const {
    return elementwise_binary(*this, other, "divide", [](double x, double y) { return x / y; });
}
Tensor Tensor::power(const Tensor& other) const {
    return elementwise_binary(*this, other, "power", [](double x, double y) { return std::pow(x, y); });
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
    for (std::size_t i = 0; i < size(); ++i) result.set_flat(i, fn(at_flat(i)));
    return result;
}

Tensor Tensor::equal(const Tensor& other) const {
    return elementwise_compare(*this, other, "equal", std::equal_to<>{});
}
Tensor Tensor::not_equal(const Tensor& other) const {
    return elementwise_compare(*this, other, "not_equal", std::not_equal_to<>{});
}
Tensor Tensor::less(const Tensor& other) const {
    return elementwise_compare(*this, other, "less", std::less<>{});
}
Tensor Tensor::less_equal(const Tensor& other) const {
    return elementwise_compare(*this, other, "less_equal", std::less_equal<>{});
}
Tensor Tensor::greater(const Tensor& other) const {
    return elementwise_compare(*this, other, "greater", std::greater<>{});
}
Tensor Tensor::greater_equal(const Tensor& other) const {
    return elementwise_compare(*this, other, "greater_equal", std::greater_equal<>{});
}

double Tensor::sum() const {
    require_numeric(*this, "sum");
    double s = 0.0;
    for (std::size_t i = 0; i < size(); ++i) s += at_flat(i);
    return s;
}

double Tensor::mean() const {
    require_numeric(*this, "mean");
    if (size() == 0) throw std::invalid_argument("Tensor::mean: tensor is empty");
    return sum() / static_cast<double>(size());
}

double Tensor::max() const {
    require_numeric(*this, "max");
    if (size() == 0) throw std::invalid_argument("Tensor::max: tensor is empty");
    double best = at_flat(0);
    for (std::size_t i = 1; i < size(); ++i) best = std::max(best, at_flat(i));
    return best;
}

double Tensor::min() const {
    require_numeric(*this, "min");
    if (size() == 0) throw std::invalid_argument("Tensor::min: tensor is empty");
    double best = at_flat(0);
    for (std::size_t i = 1; i < size(); ++i) best = std::min(best, at_flat(i));
    return best;
}

double Tensor::prod() const {
    require_numeric(*this, "prod");
    double p = 1.0;
    for (std::size_t i = 0; i < size(); ++i) p *= at_flat(i);
    return p;
}

std::size_t Tensor::argmax() const {
    require_numeric(*this, "argmax");
    if (size() == 0) throw std::invalid_argument("Tensor::argmax: tensor is empty");
    std::size_t best = 0;
    for (std::size_t i = 1; i < size(); ++i)
        if (at_flat(i) > at_flat(best)) best = i;
    return best;
}

std::size_t Tensor::argmin() const {
    require_numeric(*this, "argmin");
    if (size() == 0) throw std::invalid_argument("Tensor::argmin: tensor is empty");
    std::size_t best = 0;
    for (std::size_t i = 1; i < size(); ++i)
        if (at_flat(i) < at_flat(best)) best = i;
    return best;
}

bool Tensor::all() const {
    require_numeric(*this, "all");
    for (std::size_t i = 0; i < size(); ++i)
        if (at_flat(i) == 0.0) return false;
    return true;
}

bool Tensor::any() const {
    require_numeric(*this, "any");
    for (std::size_t i = 0; i < size(); ++i)
        if (at_flat(i) != 0.0) return true;
    return false;
}

Tensor Tensor::sum_axis(std::size_t axis, bool keepdims) const {
    return reduce_along_axis(*this, axis, keepdims, AxisReduceOp::Sum, "sum_axis");
}
Tensor Tensor::mean_axis(std::size_t axis, bool keepdims) const {
    return reduce_along_axis(*this, axis, keepdims, AxisReduceOp::Mean, "mean_axis");
}
Tensor Tensor::max_axis(std::size_t axis, bool keepdims) const {
    return reduce_along_axis(*this, axis, keepdims, AxisReduceOp::Max, "max_axis");
}
Tensor Tensor::min_axis(std::size_t axis, bool keepdims) const {
    return reduce_along_axis(*this, axis, keepdims, AxisReduceOp::Min, "min_axis");
}
Tensor Tensor::prod_axis(std::size_t axis, bool keepdims) const {
    return reduce_along_axis(*this, axis, keepdims, AxisReduceOp::Prod, "prod_axis");
}
Tensor Tensor::argmax_axis(std::size_t axis, bool keepdims) const {
    return reduce_along_axis(*this, axis, keepdims, AxisReduceOp::ArgMax, "argmax_axis");
}
Tensor Tensor::argmin_axis(std::size_t axis, bool keepdims) const {
    return reduce_along_axis(*this, axis, keepdims, AxisReduceOp::ArgMin, "argmin_axis");
}

Tensor Tensor::matmul(const Tensor& other) const {
    require_numeric(*this, "matmul");
    require_numeric(other, "matmul");
    if (ndim() != 2 || other.ndim() != 2)
        throw std::invalid_argument("Tensor::matmul: both tensors must be 2-D (use dot/outer for 1-D)");
    if (shape_[1] != other.shape_[0]) throw std::invalid_argument("Tensor::matmul: inner dimensions do not match");
    const std::size_t m = shape_[0], k = shape_[1], n = other.shape_[1];
    Tensor result({m, n}, TensorDType::Float64);
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            double s = 0.0;
            for (std::size_t p = 0; p < k; ++p) s += at({i, p}) * other.at({p, j});
            result.set({i, j}, s);
        }
    }
    return result;
}

double Tensor::dot(const Tensor& other) const {
    require_numeric(*this, "dot");
    require_numeric(other, "dot");
    if (ndim() != 1 || other.ndim() != 1)
        throw std::invalid_argument("Tensor::dot: both tensors must be 1-D (use matmul for 2-D)");
    if (shape_[0] != other.shape_[0]) throw std::invalid_argument("Tensor::dot: length mismatch");
    double s = 0.0;
    for (std::size_t i = 0; i < shape_[0]; ++i) s += at_flat(i) * other.at_flat(i);
    return s;
}

Tensor Tensor::outer(const Tensor& other) const {
    require_numeric(*this, "outer");
    require_numeric(other, "outer");
    if (ndim() != 1 || other.ndim() != 1) throw std::invalid_argument("Tensor::outer: both tensors must be 1-D");
    const std::size_t m = shape_[0], n = other.shape_[0];
    Tensor result({m, n}, TensorDType::Float64);
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j) result.set({i, j}, at_flat(i) * other.at_flat(j));
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
