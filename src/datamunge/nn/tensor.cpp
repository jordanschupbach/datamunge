#include "datamunge/nn/tensor.hpp"

#include "datamunge/nn/memory.hpp"

#include <cmath>
#include <cstring>
#include <iomanip>
#include <sstream>

#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/ops.hpp"
#include "detail.hpp"

static_assert(sizeof(bool) == 1, "nn stores bool tensors as one byte per element");

namespace datamunge::nn {

using detail::dispatch;
using detail::Tag;

std::string dtype_name(DType dtype) {
  switch (dtype) {
    case DType::Bool: return "bool";
    case DType::Int32: return "int32";
    case DType::Int64: return "int64";
    case DType::Float32: return "float32";
    case DType::Float64: return "float64";
  }
  return "unknown";
}

std::size_t dtype_size(DType dtype) {
  switch (dtype) {
    case DType::Bool: return 1;
    case DType::Int32: return 4;
    case DType::Int64: return 8;
    case DType::Float32: return 4;
    case DType::Float64: return 8;
  }
  return 0;
}

std::int64_t shape_numel(const Shape& shape) {
  std::int64_t n = 1;
  for (auto d : shape) {
    if (d < 0) throw std::invalid_argument("negative dimension in shape " + shape_str(shape));
    n *= d;
  }
  return n;
}

std::string shape_str(const Shape& shape) {
  std::string out = "[";
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (i) out += ", ";
    out += std::to_string(shape[i]);
  }
  return out + "]";
}

Shape contiguous_strides(const Shape& shape) {
  Shape        strides(shape.size(), 1);
  std::int64_t acc = 1;
  for (std::size_t i = shape.size(); i-- > 0;) {
    strides[i] = acc;
    acc *= std::max<std::int64_t>(shape[i], 1);
  }
  return strides;
}

Storage::Storage(std::size_t n) : bytes(new unsigned char[n ? n : 1]), nbytes(n) {}

namespace detail {

void throw_dtype_mismatch(const char* what, DType have, DType want) {
  throw std::invalid_argument(std::string(what) + ": tensor is " + dtype_name(have) + ", requested " +
                              dtype_name(want));
}

void throw_undefined(const char* what) {
  throw std::logic_error(std::string(what) + ": use of an undefined tensor");
}

double read_as_double(const TensorImpl& impl, std::int64_t element_offset) {
  const unsigned char* base = impl.storage->bytes.get();
  return dispatch(impl.dtype, [&](auto tag) -> double {
    using T = typename decltype(tag)::type;
    return static_cast<double>(reinterpret_cast<const T*>(base)[element_offset]);
  });
}

Tensor cast_raw(const Tensor& x, DType dtype) {
  if (x.dtype() == dtype) return x;
  return dispatch(x.dtype(), [&](auto in_tag) {
    using In = typename decltype(in_tag)::type;
    return dispatch(dtype, [&](auto out_tag) {
      using Out = typename decltype(out_tag)::type;
      if constexpr (std::is_same<Out, bool>::value)
        return map_unary<In, Out>(x, dtype, [](In v) { return v != In(0); });
      else
        return map_unary<In, Out>(x, dtype, [](In v) { return static_cast<Out>(v); });
    });
  });
}

}  // namespace detail

// ---- Tensor members ----------------------------------------------------------------

std::int64_t Tensor::size(std::int64_t axis) const {
  return shape()[static_cast<std::size_t>(detail::normalize_axis(axis, dim(), "Tensor::size"))];
}

bool Tensor::is_contiguous() const {
  const auto&  s = shape();
  const auto&  st = strides();
  std::int64_t expected = 1;
  for (std::size_t i = s.size(); i-- > 0;) {
    if (s[i] == 1) continue;
    if (st[i] != expected) return false;
    expected *= s[i];
  }
  return true;
}

double Tensor::at(const std::vector<std::int64_t>& index) const {
  if (static_cast<std::int64_t>(index.size()) != dim())
    throw std::invalid_argument("Tensor::at: index has " + std::to_string(index.size()) + " entries for a rank " +
                                std::to_string(dim()) + " tensor");
  std::int64_t off = offset();
  for (std::size_t i = 0; i < index.size(); ++i) {
    std::int64_t k = index[i] < 0 ? index[i] + shape()[i] : index[i];
    if (k < 0 || k >= shape()[i]) throw std::out_of_range("Tensor::at: index out of range");
    off += k * strides()[i];
  }
  return detail::read_as_double(*impl_, off);
}

Tensor Tensor::contiguous() const {
  if (is_contiguous()) return *this;
  Tensor out = dispatch(dtype(), [&](auto tag) {
    using T = typename decltype(tag)::type;
    return detail::map_unary<T, T>(*this, dtype(), [](T v) { return v; });
  });
  record("Contiguous", {*this}, {out}, [](const std::vector<Tensor>& g, const std::vector<Tensor>&) {
    return std::vector<Tensor>{g[0]};
  });
  return out;
}

Tensor Tensor::clone() const {
  Tensor out = dispatch(dtype(), [&](auto tag) {
    using T = typename decltype(tag)::type;
    return detail::map_unary<T, T>(*this, dtype(), [](T v) { return v; });
  });
  record("Clone", {*this}, {out}, [](const std::vector<Tensor>& g, const std::vector<Tensor>&) {
    return std::vector<Tensor>{g[0]};
  });
  return out;
}

Tensor Tensor::to(DType dtype) const { return cast(*this, dtype); }

Tensor Tensor::detach() const {
  auto impl     = std::make_shared<TensorImpl>();
  impl->storage = checked().storage;
  impl->shape   = checked().shape;
  impl->strides = checked().strides;
  impl->offset  = checked().offset;
  impl->dtype   = checked().dtype;
  return Tensor(std::move(impl));
}

Tensor& Tensor::requires_grad_(bool value) {
  auto& impl = checked();
  if (value && !is_floating(impl.dtype))
    throw std::invalid_argument("requires_grad_: only float tensors can require gradients, this one is " +
                                dtype_name(impl.dtype));
  if (!value && impl.grad_fn)
    throw std::invalid_argument("requires_grad_(false) on a non-leaf tensor; use detach() instead");
  impl.requires_grad = value;
  return *this;
}

Tensor Tensor::grad() const { return checked().grad ? Tensor(checked().grad) : Tensor(); }

void Tensor::set_grad(const Tensor& grad) {
  if (grad.defined() && grad.shape() != shape())
    throw std::invalid_argument("set_grad: gradient shape " + shape_str(grad.shape()) + " does not match tensor shape " +
                                shape_str(shape()));
  checked().grad = grad.impl_ptr();
}

void Tensor::zero_grad() { checked().grad.reset(); }

std::string Tensor::grad_fn_name() const { return checked().grad_fn ? checked().grad_fn->name() : std::string(); }

void Tensor::backward(const Tensor& grad_output, bool retain_graph, bool create_graph) const {
  nn::backward({*this}, grad_output.defined() ? std::vector<Tensor>{grad_output} : std::vector<Tensor>{},
               retain_graph, create_graph);
}

namespace {

void check_inplace(const Tensor& t, const char* what) {
  if (t.requires_grad() && is_grad_enabled())
    throw std::logic_error(std::string(what) +
                           ": in-place update of a tensor that requires grad while grad mode is on; wrap it in "
                           "NoGradGuard");
}

}  // namespace

Tensor& Tensor::copy_(const Tensor& src) {
  check_inplace(*this, "copy_");
  Tensor s;
  {
    NoGradGuard guard;
    s = detail::cast_raw(src.detach(), dtype());
    // Read everything first when the source shares our storage.
    if (s.impl()->storage == checked().storage) s = s.clone();
  }
  if (broadcast_shapes(s.shape(), shape()) != shape())
    throw std::invalid_argument("copy_: source shape " + shape_str(src.shape()) + " does not broadcast to " +
                                shape_str(shape()));
  const Shape ss = detail::broadcast_strides(s, shape());
  dispatch(dtype(), [&](auto tag) {
    using T                             = typename decltype(tag)::type;
    T*                                dst  = detail::raw_data<T>(*this);
    const T*                          from = detail::raw_data<T>(s);
    const std::array<const Shape*, 2> st{&strides(), &ss};
    detail::strided_rows<2>(shape(), st, {offset(), s.offset()}, [&](const auto& off, std::int64_t n,
                                                                     const auto& step) {
      for (std::int64_t j = 0; j < n; ++j) dst[off[0] + j * step[0]] = from[off[1] + j * step[1]];
    });
  });
  ++checked().storage->version;
  return *this;
}

Tensor& Tensor::fill_(double value) {
  check_inplace(*this, "fill_");
  dispatch(dtype(), [&](auto tag) {
    using T                             = typename decltype(tag)::type;
    T*                                dst = detail::raw_data<T>(*this);
    const T                           v   = static_cast<T>(value);
    const std::array<const Shape*, 1> st{&strides()};
    detail::strided_rows<1>(shape(), st, {offset()}, [&](const auto& off, std::int64_t n, const auto& step) {
      for (std::int64_t j = 0; j < n; ++j) dst[off[0] + j * step[0]] = v;
    });
  });
  ++checked().storage->version;
  return *this;
}

Tensor& Tensor::add_(const Tensor& other, double alpha) {
  check_inplace(*this, "add_");
  Tensor updated;
  {
    NoGradGuard guard;
    updated = alpha == 1.0 ? add(detach(), other.detach()) : add(detach(), mul(other.detach(), scalar_like(alpha, other)));
  }
  return copy_(updated);
}

Tensor& Tensor::mul_(const Tensor& other) {
  check_inplace(*this, "mul_");
  Tensor updated;
  {
    NoGradGuard guard;
    updated = mul(detach(), other.detach());
  }
  return copy_(updated);
}

Tensor& Tensor::mul_(double value) { return mul_(scalar_like(value, *this)); }

Tensor Tensor::reshape(const Shape& shape) const { return nn::reshape(*this, shape); }
Tensor Tensor::transpose(const std::vector<std::int64_t>& perm) const { return nn::transpose(*this, perm); }
Tensor Tensor::sum(const std::vector<std::int64_t>& axes, bool keepdims) const { return nn::sum(*this, axes, keepdims); }
Tensor Tensor::mean(const std::vector<std::int64_t>& axes, bool keepdims) const {
  return nn::mean(*this, axes, keepdims);
}

// ---- printing ------------------------------------------------------------------------

namespace {

std::string format_value(DType dtype, double v) {
  if (dtype == DType::Bool) return v != 0 ? "true" : "false";
  if (is_integral(dtype)) return std::to_string(static_cast<long long>(v));
  std::ostringstream os;
  os << std::setprecision(6) << v;
  return os.str();
}

void render(const Tensor& t, std::size_t axis, std::int64_t offset, bool elide, std::string& out) {
  const auto& shape = t.shape();
  if (axis == shape.size()) {
    out += format_value(t.dtype(), detail::read_as_double(*t.impl(), offset));
    return;
  }
  out += "[";
  const std::int64_t n    = shape[axis];
  const std::int64_t edge = 3;
  for (std::int64_t i = 0; i < n; ++i) {
    if (elide && n > 2 * edge && i == edge) {
      out += "..., ";
      i = n - edge - 1;
      continue;
    }
    render(t, axis + 1, offset + i * t.strides()[axis], elide, out);
    if (i + 1 < n) out += ", ";
  }
  out += "]";
}

}  // namespace

std::string Tensor::values_str(std::int64_t max_elements) const {
  std::string out;
  render(*this, 0, offset(), numel() > max_elements, out);
  return out;
}

std::string Tensor::str(std::int64_t max_elements) const {
  if (!defined()) return "Tensor(undefined)";
  std::string out = dtype_name(dtype()) + shape_str(shape()) + " " + values_str(max_elements);
  if (requires_grad()) out += grad_fn_name().empty() ? " (requires_grad)" : " (grad_fn=" + grad_fn_name() + ")";
  return out;
}

std::ostream& operator<<(std::ostream& out, const Tensor& tensor) { return out << tensor.str(); }

// ---- factories ------------------------------------------------------------------------

Tensor empty(const Shape& shape, DType dtype) {
  auto impl     = std::make_shared<TensorImpl>();
  // A deleter (rather than make_shared) lets nn/memory.hpp count live bytes.
  const std::size_t nbytes = static_cast<std::size_t>(shape_numel(shape)) * dtype_size(dtype);
  impl->storage = std::shared_ptr<Storage>(new Storage(nbytes), [](Storage* s) {
    detail::note_storage_free(static_cast<std::int64_t>(s->nbytes));
    delete s;
  });
  detail::note_storage_alloc(static_cast<std::int64_t>(nbytes));
  impl->shape   = shape;
  impl->strides = contiguous_strides(shape);
  impl->dtype   = dtype;
  return Tensor(std::move(impl));
}

Tensor zeros(const Shape& shape, DType dtype) {
  Tensor out = empty(shape, dtype);
  std::memset(out.impl()->storage->bytes.get(), 0, out.impl()->storage->nbytes);
  return out;
}

Tensor full(const Shape& shape, double value, DType dtype) {
  Tensor out = empty(shape, dtype);
  dispatch(dtype, [&](auto tag) {
    using T             = typename decltype(tag)::type;
    T*                p = out.data_ptr<T>();
    const std::int64_t n = out.numel();
    for (std::int64_t i = 0; i < n; ++i) p[i] = static_cast<T>(value);
  });
  return out;
}

Tensor ones(const Shape& shape, DType dtype) { return full(shape, 1.0, dtype); }
Tensor zeros_like(const Tensor& like) { return zeros(like.shape(), like.dtype()); }
Tensor ones_like(const Tensor& like) { return ones(like.shape(), like.dtype()); }
Tensor full_like(const Tensor& like, double value) { return full(like.shape(), value, like.dtype()); }
Tensor scalar(double value, DType dtype) { return full({}, value, dtype); }

Tensor arange(double start, double stop, double step, DType dtype) {
  if (step == 0) throw std::invalid_argument("arange: step must be nonzero");
  const double       span = (stop - start) / step;
  const std::int64_t n    = span > 0 ? static_cast<std::int64_t>(std::ceil(span)) : 0;
  Tensor             out  = empty({n}, dtype);
  dispatch(dtype, [&](auto tag) {
    using T = typename decltype(tag)::type;
    T* p    = out.data_ptr<T>();
    for (std::int64_t i = 0; i < n; ++i) p[i] = static_cast<T>(start + static_cast<double>(i) * step);
  });
  return out;
}

Tensor linspace(double start, double stop, std::int64_t count, DType dtype) {
  if (count < 0) throw std::invalid_argument("linspace: negative count");
  Tensor out = empty({count}, dtype);
  dispatch(dtype, [&](auto tag) {
    using T = typename decltype(tag)::type;
    T* p    = out.data_ptr<T>();
    for (std::int64_t i = 0; i < count; ++i)
      p[i] = static_cast<T>(count == 1 ? start : start + (stop - start) * static_cast<double>(i) / static_cast<double>(count - 1));
  });
  return out;
}

Tensor eye(std::int64_t n, DType dtype) {
  Tensor out = zeros({n, n}, dtype);
  dispatch(dtype, [&](auto tag) {
    using T = typename decltype(tag)::type;
    T* p    = out.data_ptr<T>();
    for (std::int64_t i = 0; i < n; ++i) p[i * n + i] = static_cast<T>(1);
  });
  return out;
}

Tensor tensor(std::initializer_list<double> values, Shape shape, DType dtype) {
  return cast(from_vector(std::vector<double>(values), std::move(shape)), dtype);
}

Tensor tensor(std::initializer_list<std::initializer_list<double>> rows, DType dtype) {
  const std::int64_t  r = static_cast<std::int64_t>(rows.size());
  const std::int64_t  c = r ? static_cast<std::int64_t>(rows.begin()->size()) : 0;
  std::vector<double> values;
  for (const auto& row : rows) {
    if (static_cast<std::int64_t>(row.size()) != c) throw std::invalid_argument("tensor: ragged rows");
    values.insert(values.end(), row.begin(), row.end());
  }
  return cast(from_vector(values, {r, c}), dtype);
}

// ---- cast (the op) ---------------------------------------------------------------------

Tensor cast(const Tensor& x, DType dtype) {
  if (x.dtype() == dtype) return x;
  Tensor out = detail::cast_raw(x, dtype);
  if (is_floating(dtype) && is_floating(x.dtype())) {
    const DType from = x.dtype();
    record("Cast", {x}, {out}, [from](const std::vector<Tensor>& g, const std::vector<Tensor>&) {
      return std::vector<Tensor>{cast(g[0], from)};
    });
  }
  return out;
}

bool equal_bits(const Tensor& a, const Tensor& b) {
  if (!a.defined() || !b.defined()) return a.defined() == b.defined();
  if (a.dtype() != b.dtype() || a.shape() != b.shape()) return false;
  if (a.numel() == 0) return true;
  const Tensor      pa = a.contiguous(), pb = b.contiguous();
  const std::size_t width = dtype_size(a.dtype());
  const auto*       da = pa.impl()->storage->bytes.get() + static_cast<std::size_t>(pa.offset()) * width;
  const auto*       db = pb.impl()->storage->bytes.get() + static_cast<std::size_t>(pb.offset()) * width;
  return std::memcmp(da, db, static_cast<std::size_t>(a.numel()) * width) == 0;
}

}  // namespace datamunge::nn
