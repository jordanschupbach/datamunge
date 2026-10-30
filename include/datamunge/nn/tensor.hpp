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

#include "datamunge/compute/context.hpp"
#include "datamunge/nn/dtype.hpp"

namespace datamunge::nn {

/// Dimensions of a tensor. A scalar has an empty shape.
using Shape = std::vector<std::int64_t>;

/// @brief Count logical elements, one for a scalar and zero if any dimension is zero.
/// @param shape Nonnegative axis lengths, empty for a scalar.
/// @return Checked product; all dimensions are validated even for empty tensors.
/// @throws std::invalid_argument If any dimension is negative.
/// @throws std::overflow_error The element count exceeds INT64_MAX.
/// @par Example
/// @code
/// std::print("{}\n",dm::nn::shape_numel({2,3})); // 6
/// @endcode
/// @see contiguous_strides(), Tensor::numel()
std::int64_t shape_numel(const Shape& shape);

/// "[2, 3]" style rendering of a shape.
std::string shape_str(const Shape& shape);

/// @brief Compute row-major strides in elements, treating zero dimensions as one for strides.
/// @param shape Nonnegative axis lengths, empty for a scalar.
/// @return Independent strides with the same rank as shape.
/// @throws std::invalid_argument A negative dimension.
/// @throws std::overflow_error Any stride or the running element product exceeds INT64_MAX.
/// @par Example
/// @code
/// const auto strides=dm::nn::contiguous_strides({2,3});
/// std::print("{}\n",strides[0]); // 3
/// @endcode
/// @see shape_numel(), Tensor::strides()
Shape contiguous_strides(const Shape& shape);

/// @brief Private GPU storage program shared by live tensor allocations.
namespace detail {
/// @brief Retained native packing/fill pipeline for one explicit context.
struct StorageProgram;
} // namespace detail

/// @brief Reference-counted, device-owned bytes shared by tensor views.
/// Storage retains its compute context. CPU bytes have a host pointer; GPU bytes
/// never do. GPU buffers include up to three padding bytes for word access;
/// nbytes and NN memory counters exclude padding. version tracks in-place writes.
/// Direct buffer writes bypass version tracking, as does data_ptr().
/// @par Example
/// @code
/// const dm::nn::Storage storage(16);
/// std::print("{}\n", storage.nbytes); // 16
/// @endcode
/// @see Tensor, compute::Buffer
struct Storage {
  /// @brief Allocate uninitialized CPU bytes in the shared default CPU context.
  /// @param nbytes Logical byte count; zero still owns a valid allocation.
  /// @throws std::bad_alloc Allocation failed.
  /// @see Storage(std::size_t, const compute::Context&) for explicit devices.
  explicit Storage(std::size_t nbytes);
  /// @brief Allocate uninitialized bytes in an explicit retained context.
  /// @param nbytes Logical byte count; GPU storage is limited to UINT32_MAX-3 bytes.
  /// @param context Existing execution owner, retained by copy.
  /// @throws std::length_error GPU storage exceeds the portable kernel address range.
  /// @throws std::bad_alloc Host allocation failed.
  /// @throws std::runtime_error Native device allocation failed.
  /// @par Example
  /// @code
  /// const dm::compute::Context context;
  /// const dm::nn::Storage storage(8,context);
  /// @endcode
  Storage(std::size_t nbytes, const compute::Context& context);
  /// @brief Disallow copying allocation ownership with an independent version counter.
  Storage(const Storage&) = delete;
  /// @brief Disallow replacing a live allocation/version owner by assignment.
  Storage& operator=(const Storage&) = delete;

  compute::Context context;    ///< Retained execution/allocation owner.
  compute::Buffer  bytes;      ///< Owning buffer; GPU size includes word padding.
  std::size_t      nbytes;     ///< Logical tensor bytes, excluding word padding.
  std::uint64_t    version{0}; ///< In-place counter checked by saved autograd values.
  std::shared_ptr<detail::StorageProgram>
      program; ///< Lazily retained native packing/fill pipeline.
};

class Node;

/// Internal state behind a Tensor handle.
struct TensorImpl {
  /// @brief Shared backing allocation; different views may refer to the same buffer.
  std::shared_ptr<Storage> storage;
  /// @brief Logical axis lengths; an empty shape denotes a scalar.
  Shape shape;
  /// @brief Per-axis memory increments in elements, including zero for broadcast axes.
  Shape strides;
  /// @brief Position of the first logical element relative to the storage base, in elements.
  std::int64_t offset{0};
  /// @brief Element representation used to interpret bytes in storage.
  DType dtype{DType::Float32};

  // Autograd. A tensor with `requires_grad` and no `grad_fn` is a leaf: its
  // gradient is accumulated into `grad` by `backward()`.
  bool                  requires_grad{false}; ///< Whether autograd should track this tensor.
  std::shared_ptr<Node> grad_fn;              ///< Producing node, or null for a leaf tensor.
  std::uint32_t         output_nr{0};         ///< Index of this output in its producing node.
  std::weak_ptr<Node>   grad_accumulator; ///< Cached leaf accumulator without an ownership cycle.
  std::shared_ptr<TensorImpl> grad; ///< Accumulated leaf gradient; null until assigned or computed.
};

class Tensor;

/// Template helpers used by the inline accessors below.
namespace detail {
/// @brief Throws std::invalid_argument identifying the operation and actual/requested dtypes.
[[noreturn]] void throw_dtype_mismatch(const char* what, DType have, DType want);
/// @brief Throws std::logic_error identifying an operation on an undefined tensor.
[[noreturn]] void throw_undefined(const char* what);
/// @brief Reads one element as double at an absolute storage offset, without bounds checks.
/// The caller supplies valid CPU storage and an offset in elements; large int64 values may round.
/// @throws std::logic_error GPU storage requires an explicit host transfer.
double read_as_double(const TensorImpl& impl, std::int64_t element_offset);
} // namespace detail

/// @brief A typed N-d array with explicit device storage and reverse-mode autograd history.
/// Native storage, views, packing, fill and explicit transfers support CPU/CUDA/Vulkan.
/// Host element access and numerical operations without GPU kernels throw for GPU
/// tensors. Transfer to an explicit CPU context before reading host values.
/// @par Example
/// @code
/// const auto x=dm::nn::zeros({2,3});
/// std::print("{}\n",x.numel()); // 6
/// @endcode
/// @see to(), device(), compute::Context
/// Copies alias the same implementation. Metadata and data access on an undefined
/// tensor throw std::logic_error unless a method explicitly accepts that state.
class Tensor {
 public:
  /// An undefined tensor (no storage). Used as "no gradient" in autograd.
  Tensor() = default;
  /// @brief Shares an implementation without copying its data; null creates an undefined tensor.
  explicit Tensor(std::shared_ptr<TensorImpl> impl) : impl_(std::move(impl)) {}

  // ---- metadata -------------------------------------------------------------

  /// @brief Returns whether the handle has an implementation, including zero-element tensors.
  [[nodiscard]] bool defined() const noexcept { return impl_ != nullptr; }
  /// @brief Borrows logical axis lengths from the shared implementation.
  [[nodiscard]] const Shape& shape() const { return checked().shape; }
  /// @brief Borrows per-axis increments in elements, which may be zero or negative for views.
  [[nodiscard]] const Shape& strides() const { return checked().strides; }
  /// @brief Returns the first element's offset from the storage base, in elements.
  [[nodiscard]] std::int64_t offset() const { return checked().offset; }
  /// @brief Returns the element representation of the backing storage.
  [[nodiscard]] DType dtype() const { return checked().dtype; }
  /// Number of dimensions (0 for a scalar).
  [[nodiscard]] std::int64_t dim() const {
    return static_cast<std::int64_t>(checked().shape.size());
  }
  /// Size of one dimension; negative axes count from the end.
  [[nodiscard]] std::int64_t size(std::int64_t axis) const;
  /// @brief Returns the product of the axis lengths (one for a scalar).
  [[nodiscard]] std::int64_t numel() const { return shape_numel(checked().shape); }
  /// True when the elements are packed in row-major order from `offset`.
  [[nodiscard]] bool is_contiguous() const;
  /// True when both handles point at the same TensorImpl.
  [[nodiscard]] bool is_same(const Tensor& other) const noexcept { return impl_ == other.impl_; }

  // ---- element access -------------------------------------------------------

  /// @brief Borrow a typed CPU pointer at offset(); dtype must match T.
  /// @tparam T Element type recognized by dtype_of.
  /// @return Borrowed pointer; strided views are not necessarily contiguous.
  /// @throws std::invalid_argument If T does not match dtype().
  /// @throws std::logic_error Undefined tensor or GPU storage; use an explicit to(cpu) transfer.
  /// Direct writes bypass storage version tracking; use in-place methods for tracked updates.
  template<typename T>
  T* data_ptr() {
    check_type<T>("data_ptr");
    return reinterpret_cast<T*>(checked().storage->bytes.host_data()) + checked().offset;
  }
  /// @brief Borrows a read-only CPU pointer at offset(); strided views need not be contiguous.
  /// @tparam T Element type recognized by dtype_of; it must match dtype().
  /// @return Borrowed read-only pointer valid while an allocation alias survives.
  /// @throws std::invalid_argument If T does not match dtype().
  /// @throws std::logic_error Undefined tensor or GPU storage.
  template<typename T>
  const T* data_ptr() const {
    check_type<T>("data_ptr");
    return reinterpret_cast<const T*>(checked().storage->bytes.host_data()) + checked().offset;
  }

  /// @brief Copies one tensor value to a host value using the destination numeric type.
  ///
  /// Any rank is allowed when there is exactly one element. Int64 storage is
  /// converted directly, preserving every bit when T is Int64; it does not pass
  /// through double. Other conversions use ordinary static_cast semantics, so
  /// callers must choose a destination that can represent the stored value.
  /// This reads storage without changing the tensor or recording a gradient.
  ///
  /// @tparam T Destination numeric type, default double.
  /// @return An owning host value converted to T.
  /// @throws std::invalid_argument If the tensor does not contain exactly one value.
  /// @throws std::logic_error If this tensor is undefined or has GPU storage; transfer to CPU
  /// first.
  /// @par Example
  /// @code
  /// const auto value = dm::nn::from_vector(std::vector<std::int64_t>{9007199254740993LL});
  /// const auto id = value.item<std::int64_t>(); // 9007199254740993, exactly.
  /// @endcode
  /// @see to_vector(), data_ptr()
  template<typename T = double>
  T item() const {
    if (numel() != 1)
      throw std::invalid_argument(
          "Tensor::item: tensor has " + std::to_string(numel()) + " elements, expected 1");
    if (dtype() == DType::Int64)
      return static_cast<T>(*data_ptr<std::int64_t>());
    return static_cast<T>(detail::read_as_double(checked(), checked().offset));
  }

  /// One element, converted to double. Negative indices count from the end.
  [[nodiscard]] double at(const std::vector<std::int64_t>& index) const;

  /// @brief Copies all logical tensor elements to an owning row-major host vector.
  ///
  /// Strided views are packed in logical order before conversion; empty tensors
  /// return an empty vector. Int64 elements convert directly without an
  /// intermediate double, preserving exact Int64 source IDs across the full
  /// signed range. Other numeric conversions use static_cast; the destination
  /// must represent the requested values. This does not mutate source storage.
  ///
  /// @tparam T Arithmetic destination element type.
  /// @return An independent vector of converted values in logical row-major order.
  /// @throws std::logic_error If this tensor is undefined or has GPU storage; transfer to CPU
  /// first.
  /// @par Example
  /// @code
  /// const auto values = dm::nn::from_vector(
  ///     std::vector<std::int64_t>{9007199254740993LL, 2, 3, -9007199254740993LL}, {2, 2});
  /// const auto copy = values.transpose().to_vector<std::int64_t>();
  /// // {9007199254740993, 3, 2, -9007199254740993}, all exact.
  /// @endcode
  /// @see item(), contiguous(), data_ptr()
  template<typename T>
  std::vector<T> to_vector() const;

  // ---- copies and conversions ----------------------------------------------

  /// @brief Return this handle if packed, otherwise make a native packed copy.
  /// Logical order, dtype, device context and differentiability are preserved.
  /// GPU packing uses native byte kernels, including negative/broadcast strides.
  /// @return Alias for contiguous input, independent storage otherwise.
  /// @throws std::logic_error Undefined tensor.
  /// @throws std::runtime_error Native GPU packing or device binding limits fail.
  /// @par Example
  /// @code
  /// const auto x=dm::nn::zeros({2,3}).transpose().contiguous();
  /// std::print("{}\n",x.is_contiguous()); // true
  /// @endcode
  /// @see clone(), to()
  [[nodiscard]] Tensor contiguous() const;
  /// @brief Make an independent native packed copy, preserving dtype, device and gradient history.
  /// GPU storage stays on the device; no host transfer or numerical conversion occurs.
  /// @return Owning contiguous copy in logical row-major order.
  /// @throws std::logic_error Undefined tensor.
  /// @throws std::runtime_error Native GPU packing or device binding limits fail.
  /// @par Example
  /// @code
  /// const auto x=dm::nn::zeros({2});const auto copy=x.clone();
  /// std::print("{}\n",copy.is_same(x)); // false
  /// @endcode
  /// @see contiguous(), to()
  [[nodiscard]] Tensor clone() const;
  /// @brief Return the backing allocation's explicit backend and ordinal.
  /// @return CPU zero by default; views retain the original device.
  /// @throws std::logic_error Undefined tensor.
  [[nodiscard]] compute::Device device() const { return checked().storage->context.device(); }
  /// @brief Borrow the execution owner retained by this tensor's storage.
  /// @return Context reference valid while the tensor or a storage alias survives.
  /// @throws std::logic_error Undefined tensor.
  [[nodiscard]] const compute::Context& context() const { return checked().storage->context; }
  /// @brief Copy logical values to packed storage in an explicit context, preserving dtype.
  /// Always allocates an independent copy. This is a synchronous transfer boundary;
  /// cross-context GPU transfers stage bytes through host memory. GPU views are
  /// packed on their device. Floating transfers record an identity backward that
  /// returns gradients to the source context. Unsupported numerical GPU operations
  /// still throw; transfer support does not imply model/backend parity.
  /// @param context Destination owner, retained by the result.
  /// @return Independent contiguous tensor with the same shape and dtype.
  /// @throws std::logic_error Undefined tensor.
  /// @throws std::length_error GPU byte range exceeded.
  /// @throws std::runtime_error Native allocation, packing or transfer failed.
  /// @par Example
  /// @code
  /// const dm::compute::Context cpu;
  /// const auto original=dm::nn::from_vector(std::vector<float>{1,2});
  /// const auto copy=original.to(cpu);
  /// std::print("{}\n",copy.numel()); // 2
  /// @endcode
  /// @see device(), context(), clone()
  [[nodiscard]] Tensor to(const compute::Context& context) const;
  /// Converts to another dtype (differentiable between float dtypes).
  [[nodiscard]] Tensor to(DType dtype) const;
  /// A view of the same data with no autograd history.
  [[nodiscard]] Tensor detach() const;

  // ---- autograd -------------------------------------------------------------

  /// @brief Returns whether autograd tracks operations involving this tensor.
  [[nodiscard]] bool requires_grad() const { return checked().requires_grad; }
  /// Sets whether gradients are required and returns this handle.
  /// @throws std::invalid_argument If enabling gradients for a non-floating dtype,
  /// or disabling them on a non-leaf; detach() produces an untracked view instead.
  Tensor& requires_grad_(bool value = true);
  /// True for tensors not produced by a recorded op.
  [[nodiscard]] bool is_leaf() const { return checked().grad_fn == nullptr; }
  /// Gradient accumulated by `backward()`; undefined until then.
  [[nodiscard]] Tensor grad() const;
  /// @brief Shares @p grad as the accumulated gradient; an undefined tensor clears it.
  /// @throws std::invalid_argument If a defined gradient has a different shape or incompatible native context.
  /// The gradient dtype is not validated or converted by this assignment.
  void set_grad(const Tensor& grad);
  /// Drops the accumulated gradient.
  void zero_grad();
  /// Name of the op that produced this tensor ("" for leaves).
  [[nodiscard]] std::string grad_fn_name() const;
  /// Runs reverse mode from this tensor and accumulates into leaf `grad()`s.
  /// A non-scalar tensor needs an explicit `grad_output` of its shape.
  /// @param grad_output Initial gradient, or undefined for implicit scalar seeding.
  /// @param retain_graph Keep saved state for a subsequent backward pass.
  /// @param create_graph Record differentiable backward operations for higher derivatives.
  void backward(
      const Tensor& grad_output  = Tensor(),
      bool          retain_graph = false,
      bool          create_graph = false) const;

  // ---- in-place updates (not recorded; for optimizers and initializers) -----
  //
  // These write into the storage and bump its version. They refuse to touch a
  // tensor that requires grad while grad mode is on, since that would
  // silently corrupt the graph; wrap updates in `NoGradGuard`.

  /// Copies `src` (broadcast to this shape, converted to this dtype) in.
  Tensor& copy_(const Tensor& src);
  /// @brief Fills all elements after converting @p value to dtype(), and returns this handle.
  /// Increments the storage version; tracked tensors require a NoGradGuard.
  Tensor& fill_(double value);
  /// @brief Fills all elements with zero, using the same version and grad checks as fill_().
  Tensor& zero_() { return fill_(0.0); }
  /// this += alpha * other
  Tensor& add_(const Tensor& other, double alpha = 1.0);
  /// this -= alpha * other
  Tensor& sub_(const Tensor& other, double alpha = 1.0) { return add_(other, -alpha); }
  /// this *= other (other broadcast to this shape)
  Tensor& mul_(const Tensor& other);
  /// @brief Multiplies elements by @p value converted to dtype(), then returns this handle.
  /// Increments the storage version; tracked tensors require a NoGradGuard.
  Tensor& mul_(double value);

  // ---- shorthands for common ops (see ops.hpp) ------------------------------

  /// @brief Reshapes without changing element order; copies if packing is needed.
  /// One dimension may be -1 to infer its length; the total element count must match.
  [[nodiscard]] Tensor reshape(const Shape& shape) const;
  /// @brief Returns a storage-sharing view with permuted axes; an empty permutation reverses them.
  /// Negative axes count from the end; repeated or out-of-range axes are invalid.
  [[nodiscard]] Tensor transpose(const std::vector<std::int64_t>& perm = {}) const;
  /// @brief Sums over @p axes (all axes when empty), retaining length-one axes if @p keepdims.
  [[nodiscard]] Tensor sum(const std::vector<std::int64_t>& axes = {}, bool keepdims = false) const;
  /// @brief Averages over @p axes (all when empty), retaining reduced axes if @p keepdims.
  /// Non-floating inputs are converted to Float32; floating inputs retain their dtype.
  [[nodiscard]] Tensor mean(const std::vector<std::int64_t>& axes = {}, bool keepdims = false)
      const;

  // ---- printing ---------------------------------------------------------------

  /// "float32[2, 3] [[1, 2, 3], [4, 5, 6]]", eliding the middle past `max_elements`.
  [[nodiscard]] std::string str(std::int64_t max_elements = 64) const;
  /// Values only, NumPy style.
  [[nodiscard]] std::string values_str(std::int64_t max_elements = 64) const;

  // ---- internals --------------------------------------------------------------

  /// @brief Borrows mutable internal state, or null when undefined; does not extend its lifetime.
  [[nodiscard]] TensorImpl* impl() const noexcept { return impl_.get(); }
  /// @brief Borrows the shared pointer; copy it to extend the implementation's lifetime.
  [[nodiscard]] const std::shared_ptr<TensorImpl>& impl_ptr() const noexcept { return impl_; }

 private:
  /// @brief Borrows internal state for reading; throws std::logic_error when undefined.
  const TensorImpl& checked() const {
    if (!impl_)
      detail::throw_undefined("Tensor");
    return *impl_;
  }
  /// @brief Borrows internal state for mutation; throws std::logic_error when undefined.
  TensorImpl& checked() {
    if (!impl_)
      detail::throw_undefined("Tensor");
    return *impl_;
  }
  /// @brief Validates typed element access, naming the calling operation in mismatch errors.
  /// @tparam T Required element representation, mapped through dtype_of.
  template<typename T>
  void check_type(const char* what) const {
    if (dtype_of<T>::value != checked().dtype)
      detail::throw_dtype_mismatch(what, checked().dtype, dtype_of<T>::value);
  }

  /// @brief Shared tensor metadata and autograd state; null represents an undefined handle.
  std::shared_ptr<TensorImpl> impl_;
};

/// @brief Writes tensor.str() to @p out and returns the stream for chaining.
std::ostream& operator<<(std::ostream& out, const Tensor& tensor);

// ---- factories ----------------------------------------------------------------

/// @brief Allocate uninitialized packed storage in the shared default CPU context.
/// @param shape Nonnegative dimensions, empty for a scalar.
/// @param dtype Valid element representation; Float32 by default.
/// @return Independent CPU tensor with no gradient history.
/// @throws std::invalid_argument Invalid shape or dtype.
/// @throws std::overflow_error Unrepresentable shape, strides or byte count.
/// @throws std::bad_alloc Host allocation failure.
/// @see empty(const Shape&, DType, const compute::Context&) for a runnable example.
Tensor empty(const Shape& shape, DType dtype = DType::Float32);
/// @brief Allocate uninitialized contiguous storage in an explicit retained context.
/// @param shape Nonnegative dimensions; an empty shape creates one scalar.
/// @param dtype Valid Bool/Int32/Int64/Float32/Float64 representation.
/// @param context Destination execution/allocation owner.
/// @return Independent packed tensor with no gradient history.
/// @throws std::invalid_argument Invalid shape or dtype.
/// @throws std::overflow_error Shape or byte count cannot be represented.
/// @throws std::length_error GPU byte range exceeds UINT32_MAX-3.
/// @throws std::runtime_error Native device allocation failed.
/// @par Example
/// @code
/// const dm::compute::Context cpu;
/// const auto x=dm::nn::empty({2,3},dm::nn::DType::Float32,cpu);
/// std::print("{}\n",x.numel()); // 6
/// @endcode
/// @see zeros(), full(), Tensor::to()
Tensor empty(const Shape& shape, DType dtype, const compute::Context& context);
/// @brief Allocate a zero-filled packed tensor in the shared default CPU context.
/// @param shape Nonnegative dimensions, empty for a scalar.
/// @param dtype Valid element representation; Float32 by default.
/// @return Independent zero-filled CPU tensor.
/// @throws std::invalid_argument Invalid shape or dtype.
/// @throws std::overflow_error Unrepresentable shape, strides or byte count.
/// @see zeros(const Shape&, DType, const compute::Context&) for a runnable example.
Tensor zeros(const Shape& shape, DType dtype = DType::Float32);
/// @brief Allocates a packed tensor filled with ones converted to @p dtype.
Tensor ones(const Shape& shape, DType dtype = DType::Float32);
/// @brief Allocate a scalar fill in the shared default CPU context.
/// @param shape Nonnegative dimensions, empty for a scalar.
/// @param value Scalar converted to dtype; Int32/Int64 require a finite in-range value.
/// @param dtype Valid representation; Float32 by default.
/// @return Independent packed CPU tensor.
/// @throws std::invalid_argument Invalid shape, dtype or integral conversion.
/// @throws std::overflow_error Unrepresentable shape, strides or byte count.
/// @see full(const Shape&, double, DType, const compute::Context&) for a runnable example.
Tensor full(const Shape& shape, double value, DType dtype = DType::Float32);
/// @brief Allocate zero-filled contiguous storage on an explicit device.
/// @param shape Nonnegative dimensions; an empty shape creates a scalar.
/// @param dtype Valid element representation.
/// @param context Destination execution owner.
/// @return Independent tensor containing exact zero bytes.
/// @throws std::invalid_argument Invalid shape or dtype.
/// @throws std::overflow_error Unrepresentable allocation size.
/// @throws std::length_error GPU byte range exceeded.
/// @throws std::runtime_error Native allocation or fill failed.
/// @par Example
/// @code
/// const dm::compute::Context cpu;
/// const auto x=dm::nn::zeros({1},dm::nn::DType::Int64,cpu);
/// std::print("{}\n",x.item<std::int64_t>()); // 0
/// @endcode
/// @see empty(), full()
Tensor zeros(const Shape& shape, DType dtype, const compute::Context& context);
/// @brief Allocate a native device tensor repeating one converted scalar bit pattern.
/// CPU and GPU fills use the same scalar conversion; GPU repetition runs natively.
/// Nonfinite/out-of-range values for integral dtypes are rejected.
/// @param shape Nonnegative dimensions; an empty shape creates a scalar.
/// @param value Scalar converted to dtype, then repeated.
/// @param dtype Valid element representation.
/// @param context Destination execution owner.
/// @return Independent packed tensor with no gradient history.
/// @throws std::invalid_argument Invalid shape, dtype or integral conversion.
/// @throws std::overflow_error Unrepresentable allocation size.
/// @throws std::length_error GPU byte range exceeded.
/// @throws std::runtime_error Native allocation, compilation or fill failed.
/// @par Example
/// @code
/// const dm::compute::Context cpu;
/// const auto x=dm::nn::full({1},2.5,dm::nn::DType::Float64,cpu);
/// std::print("{}\n",x.item<double>()); // 2.5
/// @endcode
/// @see zeros(), full_like()
Tensor full(const Shape& shape, double value, DType dtype, const compute::Context& context);
/// @brief Allocate zeros matching like's shape, dtype and retained device context.
/// @param like Defined prototype, read-only; gradient tracking is not inherited.
/// @return Independent packed zero-filled tensor.
/// @throws std::logic_error Undefined prototype.
/// @throws std::runtime_error Native allocation/fill failed.
/// @par Example
/// @code
/// const auto x=dm::nn::zeros_like(dm::nn::ones({2}));
/// std::print("{}\n",x.numel()); // 2
/// @endcode
/// @see zeros(), ones_like()
Tensor zeros_like(const Tensor& like);
/// @brief Allocate ones matching like's shape, dtype and retained device context.
/// @param like Defined prototype, read-only; gradient tracking is not inherited.
/// @return Independent packed one-filled tensor.
/// @throws std::logic_error Undefined prototype.
/// @throws std::runtime_error Native allocation/compilation/fill failed.
/// @par Example
/// @code
/// const auto x=dm::nn::ones_like(dm::nn::zeros({1}));
/// std::print("{}\n",x.item<double>()); // 1
/// @endcode
/// @see full_like(), zeros_like()
Tensor ones_like(const Tensor& like);
/// @brief Allocate a scalar fill matching like's shape, dtype and retained device context.
/// @param like Defined prototype, read-only; gradient tracking is not inherited.
/// @param value Scalar converted with the same checked integral rules as full().
/// @return Independent packed tensor on the prototype's context.
/// @throws std::logic_error Undefined prototype.
/// @throws std::invalid_argument Invalid integral conversion.
/// @throws std::runtime_error Native allocation/compilation/fill failed.
/// @par Example
/// @code
/// const auto x=dm::nn::full_like(dm::nn::zeros({1}),3);
/// std::print("{}\n",x.item<double>()); // 3
/// @endcode
/// @see full(), ones_like()
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
template<typename T>
Tensor from_vector(const std::vector<T>& values, Shape shape = {});

/// A 0-d tensor holding `value` exactly, with the dtype matching T
/// (float -> float32, double -> float64, int32/int64, bool).
template<typename T>
Tensor from_scalar(T value);

/// Bit-for-bit equality: same dtype, same shape and identical element bytes
/// (so NaN == NaN with the same payload, while 0.0 != -0.0). Two undefined
/// tensors are equal; an undefined and a defined one are not. Strided views
/// compare by their logical (row-major) elements.
bool equal_bits(const Tensor& a, const Tensor& b);

/// Literal tensors: `tensor({1, 2, 3})`, `tensor({1, 2, 3, 4}, {2, 2})`.
Tensor tensor(std::initializer_list<double> values, Shape shape = {}, DType dtype = DType::Float32);
/// Literal matrix: `tensor({{1, 2}, {3, 4}})`.
Tensor tensor(
    std::initializer_list<std::initializer_list<double>> rows,
    DType                                                dtype = DType::Float32);

// ---- template definitions -------------------------------------------------------

template<typename T>
std::vector<T> Tensor::to_vector() const {
  static_assert(std::is_arithmetic<T>::value, "to_vector needs an arithmetic type");
  if(device().backend!=compute::Backend::CPU)
    throw std::logic_error("to_vector: transfer GPU tensors to an explicit CPU context first");
  const Tensor   packed = contiguous();
  std::vector<T> out(static_cast<std::size_t>(packed.numel()));
  if (packed.dtype() == DType::Int64) {
    const auto* values = packed.data_ptr<std::int64_t>();
    for (std::size_t i = 0; i < out.size(); ++i)
      out[i] = static_cast<T>(values[i]);
  } else {
    for (std::size_t i = 0; i < out.size(); ++i)
      out[i] = static_cast<T>(
          detail::read_as_double(*packed.impl(), packed.offset() + static_cast<std::int64_t>(i)));
  }
  return out;
}

template<typename T>
Tensor from_scalar(T value) {
  Tensor out         = empty(Shape{}, dtype_of<T>::value);
  *out.data_ptr<T>() = value;
  return out;
}

template<typename T>
Tensor from_vector(const std::vector<T>& values, Shape shape) {
  if (shape.empty())
    shape = {static_cast<std::int64_t>(values.size())};
  if (shape_numel(shape) != static_cast<std::int64_t>(values.size()))
    throw std::invalid_argument(
        "from_vector: " + std::to_string(values.size()) + " values do not fill shape " +
        shape_str(shape));
  Tensor out = empty(shape, dtype_of<T>::value);
  T*     dst = out.data_ptr<T>();
  for (std::size_t i = 0; i < values.size(); ++i)
    dst[i] = values[i];
  return out;
}

} // namespace datamunge::nn
