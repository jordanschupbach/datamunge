#pragma once

/// @file tensor.hpp
/// @brief `nn::Tensor`: a typed, strided N-d array that can record autograd history.
///
/// A Tensor is a cheap handle (a shared pointer) to a `TensorImpl`. Copying a
/// Tensor aliases it, like a Python reference; use `clone()` for an
/// independent copy. Several tensors can view one `Storage` buffer with
/// different shapes, strides and offsets, which keeps reshape, transpose,
/// slicing and broadcasting free of copies.
///
/// Storage is row-major. Kernels that need packed memory call `contiguous()`,
/// which returns the tensor itself when it already is.

#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "datamunge/nn/dtype.hpp"

namespace datamunge::nn {

/// Dimensions of a tensor. A scalar has an empty shape.
using Shape = std::vector<std::int64_t>;

/// Number of elements of a shape (1 for a scalar).
std::int64_t shape_numel(const Shape& shape);

/// "[2, 3]" style rendering of a shape.
std::string shape_str(const Shape& shape);

/// Row-major strides for a packed tensor of this shape.
Shape contiguous_strides(const Shape& shape);

/// A reference-counted, untyped byte buffer shared by tensor views.
///
/// `version` counts in-place writes. Autograd records it when it saves a
/// tensor for the backward pass and refuses to run a backward that would read
/// data modified after it was saved.
struct Storage {
  explicit Storage(std::size_t nbytes);

  std::unique_ptr<unsigned char[]> bytes;
  std::size_t                      nbytes;
  std::uint64_t                    version{0};
};

class Node;

/// Internal state behind a Tensor handle.
struct TensorImpl {
  std::shared_ptr<Storage> storage;
  Shape                    shape;
  Shape                    strides;  // in elements
  std::int64_t             offset{0};  // in elements
  DType                    dtype{DType::Float32};

  // Autograd. A tensor with `requires_grad` and no `grad_fn` is a leaf: its
  // gradient is accumulated into `grad` by `backward()`.
  bool                        requires_grad{false};
  std::shared_ptr<Node>       grad_fn;
  std::uint32_t               output_nr{0};
  std::weak_ptr<Node>         grad_accumulator;
  std::shared_ptr<TensorImpl> grad;
};

class Tensor;

/// Template helpers used by the inline accessors below.
namespace detail {
[[noreturn]] void throw_dtype_mismatch(const char* what, DType have, DType want);
[[noreturn]] void throw_undefined(const char* what);
double            read_as_double(const TensorImpl& impl, std::int64_t element_offset);
}  // namespace detail

/// A typed N-d array with optional reverse-mode autograd history.
class Tensor {
 public:
  /// An undefined tensor (no storage). Used as "no gradient" in autograd.
  Tensor() = default;
  explicit Tensor(std::shared_ptr<TensorImpl> impl) : impl_(std::move(impl)) {}

  // ---- metadata -------------------------------------------------------------

  [[nodiscard]] bool         defined() const noexcept { return impl_ != nullptr; }
  [[nodiscard]] const Shape& shape() const { return checked().shape; }
  [[nodiscard]] const Shape& strides() const { return checked().strides; }
  [[nodiscard]] std::int64_t offset() const { return checked().offset; }
  [[nodiscard]] DType        dtype() const { return checked().dtype; }
  /// Number of dimensions (0 for a scalar).
  [[nodiscard]] std::int64_t dim() const { return static_cast<std::int64_t>(checked().shape.size()); }
  /// Size of one dimension; negative axes count from the end.
  [[nodiscard]] std::int64_t size(std::int64_t axis) const;
  [[nodiscard]] std::int64_t numel() const { return shape_numel(checked().shape); }
  /// True when the elements are packed in row-major order from `offset`.
  [[nodiscard]] bool         is_contiguous() const;
  /// True when both handles point at the same TensorImpl.
  [[nodiscard]] bool         is_same(const Tensor& other) const noexcept { return impl_ == other.impl_; }

  // ---- element access -------------------------------------------------------

  /// Pointer to the first element (at `offset`). The dtype must match T.
  template <typename T>
  T* data_ptr() {
    check_type<T>("data_ptr");
    return reinterpret_cast<T*>(checked().storage->bytes.get()) + checked().offset;
  }
  template <typename T>
  const T* data_ptr() const {
    check_type<T>("data_ptr");
    return reinterpret_cast<const T*>(checked().storage->bytes.get()) + checked().offset;
  }

  /// The value of a one-element tensor, converted to T.
  template <typename T = double>
  T item() const {
    if (numel() != 1) throw std::invalid_argument("Tensor::item: tensor has " + std::to_string(numel()) + " elements, expected 1");
    return static_cast<T>(detail::read_as_double(checked(), checked().offset));
  }

  /// One element, converted to double. Negative indices count from the end.
  [[nodiscard]] double at(const std::vector<std::int64_t>& index) const;

  /// All elements in row-major order, converted to T.
  template <typename T>
  std::vector<T> to_vector() const;

  // ---- copies and conversions ----------------------------------------------

  /// This tensor if already packed, else a packed copy (differentiable).
  [[nodiscard]] Tensor contiguous() const;
  /// An independent packed copy (differentiable).
  [[nodiscard]] Tensor clone() const;
  /// Converts to another dtype (differentiable between float dtypes).
  [[nodiscard]] Tensor to(DType dtype) const;
  /// A view of the same data with no autograd history.
  [[nodiscard]] Tensor detach() const;

  // ---- autograd -------------------------------------------------------------

  [[nodiscard]] bool requires_grad() const { return checked().requires_grad; }
  /// Marks a leaf tensor as needing gradients. Only float tensors can.
  Tensor& requires_grad_(bool value = true);
  /// True for tensors not produced by a recorded op.
  [[nodiscard]] bool is_leaf() const { return checked().grad_fn == nullptr; }
  /// Gradient accumulated by `backward()`; undefined until then.
  [[nodiscard]] Tensor grad() const;
  void                 set_grad(const Tensor& grad);
  /// Drops the accumulated gradient.
  void                 zero_grad();
  /// Name of the op that produced this tensor ("" for leaves).
  [[nodiscard]] std::string grad_fn_name() const;
  /// Runs reverse mode from this tensor and accumulates into leaf `grad()`s.
  /// A non-scalar tensor needs an explicit `grad_output` of its shape.
  void backward(const Tensor& grad_output = Tensor(), bool retain_graph = false, bool create_graph = false) const;

  // ---- in-place updates (not recorded; for optimizers and initializers) -----
  //
  // These write into the storage and bump its version. They refuse to touch a
  // tensor that requires grad while grad mode is on, since that would
  // silently corrupt the graph; wrap updates in `NoGradGuard`.

  /// Copies `src` (broadcast to this shape, converted to this dtype) in.
  Tensor& copy_(const Tensor& src);
  Tensor& fill_(double value);
  Tensor& zero_() { return fill_(0.0); }
  /// this += alpha * other
  Tensor& add_(const Tensor& other, double alpha = 1.0);
  /// this -= alpha * other
  Tensor& sub_(const Tensor& other, double alpha = 1.0) { return add_(other, -alpha); }
  /// this *= other (other broadcast to this shape)
  Tensor& mul_(const Tensor& other);
  Tensor& mul_(double value);

  // ---- shorthands for common ops (see ops.hpp) ------------------------------

  [[nodiscard]] Tensor reshape(const Shape& shape) const;
  [[nodiscard]] Tensor transpose(const std::vector<std::int64_t>& perm = {}) const;
  [[nodiscard]] Tensor sum(const std::vector<std::int64_t>& axes = {}, bool keepdims = false) const;
  [[nodiscard]] Tensor mean(const std::vector<std::int64_t>& axes = {}, bool keepdims = false) const;

  // ---- printing ---------------------------------------------------------------

  /// "float32[2, 3] [[1, 2, 3], [4, 5, 6]]", eliding the middle past `max_elements`.
  [[nodiscard]] std::string str(std::int64_t max_elements = 64) const;
  /// Values only, NumPy style.
  [[nodiscard]] std::string values_str(std::int64_t max_elements = 64) const;

  // ---- internals --------------------------------------------------------------

  [[nodiscard]] TensorImpl*                        impl() const noexcept { return impl_.get(); }
  [[nodiscard]] const std::shared_ptr<TensorImpl>& impl_ptr() const noexcept { return impl_; }

 private:
  const TensorImpl& checked() const {
    if (!impl_) detail::throw_undefined("Tensor");
    return *impl_;
  }
  TensorImpl& checked() {
    if (!impl_) detail::throw_undefined("Tensor");
    return *impl_;
  }
  template <typename T>
  void check_type(const char* what) const {
    if (dtype_of<T>::value != checked().dtype) detail::throw_dtype_mismatch(what, checked().dtype, dtype_of<T>::value);
  }

  std::shared_ptr<TensorImpl> impl_;
};

std::ostream& operator<<(std::ostream& out, const Tensor& tensor);

// ---- factories ----------------------------------------------------------------

/// Uninitialised packed tensor.
Tensor empty(const Shape& shape, DType dtype = DType::Float32);
Tensor zeros(const Shape& shape, DType dtype = DType::Float32);
Tensor ones(const Shape& shape, DType dtype = DType::Float32);
Tensor full(const Shape& shape, double value, DType dtype = DType::Float32);
Tensor zeros_like(const Tensor& like);
Tensor ones_like(const Tensor& like);
Tensor full_like(const Tensor& like, double value);
/// A 0-d tensor.
Tensor scalar(double value, DType dtype = DType::Float32);
/// [start, start + step, ...) stopping before `stop` (TF Range).
Tensor arange(double start, double stop, double step = 1.0, DType dtype = DType::Float32);
/// `count` evenly spaced values from start to stop inclusive.
Tensor linspace(double start, double stop, std::int64_t count, DType dtype = DType::Float32);
/// n x n identity.
Tensor eye(std::int64_t n, DType dtype = DType::Float32);

/// Copies `values` into a tensor of the matching dtype. An empty `shape`
/// means a 1-D tensor of `values.size()` elements, so a one-element vector
/// gives shape [1], not a 0-d scalar: use `from_scalar` / `scalar`, or
/// `.reshape({})`, for a 0-d tensor.
template <typename T>
Tensor from_vector(const std::vector<T>& values, Shape shape = {});

/// A 0-d tensor holding `value` exactly, with the dtype matching T
/// (float -> float32, double -> float64, int32/int64, bool).
template <typename T>
Tensor from_scalar(T value);

/// Bit-for-bit equality: same dtype, same shape and identical element bytes
/// (so NaN == NaN with the same payload, while 0.0 != -0.0). Two undefined
/// tensors are equal; an undefined and a defined one are not. Strided views
/// compare by their logical (row-major) elements.
bool equal_bits(const Tensor& a, const Tensor& b);

/// Literal tensors: `tensor({1, 2, 3})`, `tensor({1, 2, 3, 4}, {2, 2})`.
Tensor tensor(std::initializer_list<double> values, Shape shape = {}, DType dtype = DType::Float32);
/// Literal matrix: `tensor({{1, 2}, {3, 4}})`.
Tensor tensor(std::initializer_list<std::initializer_list<double>> rows, DType dtype = DType::Float32);

// ---- template definitions -------------------------------------------------------

template <typename T>
std::vector<T> Tensor::to_vector() const {
  static_assert(std::is_arithmetic<T>::value, "to_vector needs an arithmetic type");
  const Tensor   packed = contiguous();
  std::vector<T> out(static_cast<std::size_t>(packed.numel()));
  for (std::size_t i = 0; i < out.size(); ++i)
    out[i] = static_cast<T>(detail::read_as_double(*packed.impl(), packed.offset() + static_cast<std::int64_t>(i)));
  return out;
}

template <typename T>
Tensor from_scalar(T value) {
  Tensor out = empty(Shape{}, dtype_of<T>::value);
  *out.data_ptr<T>() = value;
  return out;
}

template <typename T>
Tensor from_vector(const std::vector<T>& values, Shape shape) {
  if (shape.empty()) shape = {static_cast<std::int64_t>(values.size())};
  if (shape_numel(shape) != static_cast<std::int64_t>(values.size()))
    throw std::invalid_argument("from_vector: " + std::to_string(values.size()) + " values do not fill shape " + shape_str(shape));
  Tensor out = empty(shape, dtype_of<T>::value);
  T*     dst = out.data_ptr<T>();
  for (std::size_t i = 0; i < values.size(); ++i) dst[i] = values[i];
  return out;
}

}  // namespace datamunge::nn
