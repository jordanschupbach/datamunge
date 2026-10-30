#pragma once

/// @file ops.hpp
/// @brief Tensor ops. All float-valued ops are differentiable.
///
/// Conventions (NumPy / TensorFlow):
/// - Binary ops broadcast: shapes are aligned at the trailing dimension and
///   size-1 dimensions stretch. Mixed dtypes promote (see `promote_types`).
/// - A C++ `double` operand takes the tensor's dtype when the tensor is
///   floating, so `x * 2.0` stays float32 for a float32 `x`.
/// - Axes may be negative (counted from the end). An empty axis list in a
///   reduction means "all axes".
/// - Ops that only make sense on floats (exp, sqrt, mean, softmax, ...)
///   convert integer and bool inputs to float32.

#include <cstdint>
#include <utility>
#include <vector>

#include "datamunge/nn/tensor.hpp"

namespace datamunge::nn {

using Axes = std::vector<std::int64_t>;

// ---- conversions --------------------------------------------------------------------

/// Converts to another dtype. Differentiable between float dtypes.
Tensor cast(const Tensor& x, DType dtype);

// ---- binary arithmetic ------------------------------------------------------------------

Tensor add(const Tensor& a, const Tensor& b);
Tensor sub(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);
/// True division; integer inputs give a float32 result.
Tensor div(const Tensor& a, const Tensor& b);
/// Elementwise a^b. The gradient for b is taken where a > 0.
Tensor pow(const Tensor& a, const Tensor& b);
Tensor pow(const Tensor& a, double exponent);
/// Elementwise max/min. On ties the whole gradient goes to `a` (TF Maximum/Minimum).
Tensor maximum(const Tensor& a, const Tensor& b);
Tensor minimum(const Tensor& a, const Tensor& b);
/// (a - b)^2 (TF SquaredDifference).
Tensor squared_difference(const Tensor& a, const Tensor& b);
/// atan(a / b) with the correct quadrant.
Tensor atan2(const Tensor& a, const Tensor& b);

/// Scalar forms; the scalar takes the tensor's dtype (see `scalar_like`).
Tensor add(const Tensor& a, double b);
Tensor sub(const Tensor& a, double b);
Tensor sub(double a, const Tensor& b);
Tensor mul(const Tensor& a, double b);
Tensor div(const Tensor& a, double b);
Tensor div(double a, const Tensor& b);

Tensor operator+(const Tensor& a, const Tensor& b);
Tensor operator-(const Tensor& a, const Tensor& b);
Tensor operator*(const Tensor& a, const Tensor& b);
Tensor operator/(const Tensor& a, const Tensor& b);
Tensor operator+(const Tensor& a, double b);
Tensor operator-(const Tensor& a, double b);
Tensor operator*(const Tensor& a, double b);
Tensor operator/(const Tensor& a, double b);
Tensor operator+(double a, const Tensor& b);
Tensor operator-(double a, const Tensor& b);
Tensor operator*(double a, const Tensor& b);
Tensor operator/(double a, const Tensor& b);
Tensor operator-(const Tensor& x);

/// A 0-d tensor holding `value`, in `like`'s dtype when it is floating and
/// float32 otherwise (or `like`'s dtype if `value` is integral).
Tensor scalar_like(double value, const Tensor& like);

// ---- unary math ---------------------------------------------------------------------------

Tensor neg(const Tensor& x);
Tensor square(const Tensor& x);
Tensor sqrt(const Tensor& x);
Tensor rsqrt(const Tensor& x);
Tensor reciprocal(const Tensor& x);
Tensor exp(const Tensor& x);
Tensor log(const Tensor& x);
Tensor log1p(const Tensor& x);
Tensor expm1(const Tensor& x);
Tensor abs(const Tensor& x);
/// -1, 0 or 1 (zero gradient).
Tensor sign(const Tensor& x);
/// floor/ceil/round have zero gradient (use a custom function for a straight-through estimator).
Tensor floor(const Tensor& x);
Tensor ceil(const Tensor& x);
/// Rounds half to even, like TF Round.
Tensor round(const Tensor& x);
Tensor sin(const Tensor& x);
Tensor cos(const Tensor& x);
Tensor tan(const Tensor& x);
Tensor asin(const Tensor& x);
Tensor acos(const Tensor& x);
Tensor atan(const Tensor& x);
Tensor sinh(const Tensor& x);
Tensor cosh(const Tensor& x);
Tensor erf(const Tensor& x);
Tensor erfc(const Tensor& x);
Tensor lgamma(const Tensor& x);
Tensor digamma(const Tensor& x);
/// Trigamma, psi'(x). Not differentiable.
Tensor trigamma(const Tensor& x);
/// Limits values to [low, high]; the gradient passes where low <= x <= high.
Tensor clip(const Tensor& x, double low, double high);
Tensor clip(const Tensor& x, const Tensor& low, const Tensor& high);

// ---- activations ------------------------------------------------------------------------

Tensor relu(const Tensor& x);
Tensor relu6(const Tensor& x);
Tensor leaky_relu(const Tensor& x, double alpha = 0.2);
Tensor elu(const Tensor& x, double alpha = 1.0);
Tensor selu(const Tensor& x);
Tensor sigmoid(const Tensor& x);
Tensor tanh(const Tensor& x);
/// log(1 + exp(x)), computed stably.
Tensor softplus(const Tensor& x);
/// x / (1 + |x|)
Tensor softsign(const Tensor& x);
/// x * sigmoid(x) (Swish with beta = 1).
Tensor silu(const Tensor& x);
/// Gaussian error linear unit; `approximate` uses the tanh form.
Tensor gelu(const Tensor& x, bool approximate = false);
/// x * tanh(softplus(x))
Tensor mish(const Tensor& x);
/// log(sigmoid(x)), computed stably.
Tensor log_sigmoid(const Tensor& x);

// ---- comparisons and selection (results are bool, not differentiable) -----------------

Tensor equal(const Tensor& a, const Tensor& b);
Tensor not_equal(const Tensor& a, const Tensor& b);
Tensor less(const Tensor& a, const Tensor& b);
Tensor less_equal(const Tensor& a, const Tensor& b);
Tensor greater(const Tensor& a, const Tensor& b);
Tensor greater_equal(const Tensor& a, const Tensor& b);
Tensor equal(const Tensor& a, double b);
Tensor not_equal(const Tensor& a, double b);
Tensor less(const Tensor& a, double b);
Tensor less_equal(const Tensor& a, double b);
Tensor greater(const Tensor& a, double b);
Tensor greater_equal(const Tensor& a, double b);
Tensor logical_and(const Tensor& a, const Tensor& b);
Tensor logical_or(const Tensor& a, const Tensor& b);
Tensor logical_xor(const Tensor& a, const Tensor& b);
Tensor logical_not(const Tensor& x);
Tensor is_finite(const Tensor& x);
Tensor is_nan(const Tensor& x);
/// Elementwise `condition ? a : b` with broadcasting (TF SelectV2). Differentiable in a and b.
Tensor where(const Tensor& condition, const Tensor& a, const Tensor& b);
Tensor where(const Tensor& condition, const Tensor& a, double b);
Tensor where(const Tensor& condition, double a, const Tensor& b);

// ---- reductions ----------------------------------------------------------------------------

Tensor sum(const Tensor& x, const Axes& axes = {}, bool keepdims = false);
Tensor mean(const Tensor& x, const Axes& axes = {}, bool keepdims = false);
/// Max/min. The gradient is shared equally between tied maxima (TF Max).
Tensor max(const Tensor& x, const Axes& axes = {}, bool keepdims = false);
Tensor min(const Tensor& x, const Axes& axes = {}, bool keepdims = false);
Tensor prod(const Tensor& x, const Axes& axes = {}, bool keepdims = false);
/// Population variance (divides by N) and its square root.
Tensor variance(const Tensor& x, const Axes& axes = {}, bool keepdims = false);
Tensor std_dev(const Tensor& x, const Axes& axes = {}, bool keepdims = false);
/// log(sum(exp(x))), computed stably.
Tensor logsumexp(const Tensor& x, const Axes& axes = {}, bool keepdims = false);
/// Index of the first max/min along one axis (int64).
Tensor argmax(const Tensor& x, std::int64_t axis = -1, bool keepdims = false);
Tensor argmin(const Tensor& x, std::int64_t axis = -1, bool keepdims = false);
/// Bool reductions.
Tensor any(const Tensor& x, const Axes& axes = {}, bool keepdims = false);
Tensor all(const Tensor& x, const Axes& axes = {}, bool keepdims = false);
/// Number of nonzero elements (int64).
Tensor count_nonzero(const Tensor& x, const Axes& axes = {}, bool keepdims = false);
Tensor softmax(const Tensor& x, std::int64_t axis = -1);
Tensor log_softmax(const Tensor& x, std::int64_t axis = -1);
/// Inclusive (or exclusive) running sum along an axis, optionally from the end.
Tensor cumsum(const Tensor& x, std::int64_t axis, bool exclusive = false, bool reverse = false);
/// Euclidean norm over `axes`.
Tensor norm(const Tensor& x, const Axes& axes = {}, bool keepdims = false);

// ---- shape ops --------------------------------------------------------------------------------

/// One dimension may be -1 and is inferred. Returns a view whenever view_strides() allows one
/// (always for a contiguous input), and otherwise a packed copy.
Tensor reshape(const Tensor& x, const Shape& shape);
/// Collapses to 1-D.
Tensor flatten(const Tensor& x);
Tensor expand_dims(const Tensor& x, std::int64_t axis);
/// Removes the given size-1 axes, or all of them when `axes` is empty.
Tensor squeeze(const Tensor& x, const Axes& axes = {});
/// Permutes axes; an empty `perm` reverses them (TF Transpose).
Tensor transpose(const Tensor& x, const Axes& perm = {});
/// Swaps two axes.
Tensor swapaxes(const Tensor& x, std::int64_t axis0, std::int64_t axis1);
/// TF Slice: `size[i] == -1` takes the rest of axis i.
Tensor slice(const Tensor& x, const Shape& begin, const Shape& size);
/// Python-style x[start:stop:step] along one axis. Negative start/stop count
/// from the end and are clamped; step may be negative.
Tensor slice_axis(const Tensor& x, std::int64_t axis, std::int64_t start, std::int64_t stop, std::int64_t step = 1);
/// Python-style slicing of the leading axes: begin/end/strides per axis.
Tensor strided_slice(const Tensor& x, const Shape& begin, const Shape& end, const Shape& strides = {});
/// x[..., index, ...] along `axis`, dropping that axis.
Tensor select(const Tensor& x, std::int64_t axis, std::int64_t index);
Tensor concat(const std::vector<Tensor>& tensors, std::int64_t axis);
/// Splits into pieces of the given sizes along an axis (one size may be -1).
std::vector<Tensor> split(const Tensor& x, const Shape& sizes, std::int64_t axis);
/// Splits into `count` equal pieces.
std::vector<Tensor> split(const Tensor& x, std::int64_t count, std::int64_t axis);
Tensor stack(const std::vector<Tensor>& tensors, std::int64_t axis = 0);
std::vector<Tensor> unstack(const Tensor& x, std::int64_t axis = 0);
/// Repeats the whole tensor `multiples[i]` times along axis i.
Tensor tile(const Tensor& x, const Shape& multiples);
/// Repeats each element `repeats` times along an axis.
Tensor repeat(const Tensor& x, std::int64_t repeats, std::int64_t axis);
/// Broadcasts to a shape. Returns a zero-copy view.
Tensor broadcast_to(const Tensor& x, const Shape& shape);
/// The shape two shapes broadcast to (throws if incompatible).
Shape broadcast_shapes(const Shape& a, const Shape& b);
/// Sums `x` down to `shape` (the inverse of broadcasting; used by gradients).
Tensor sum_to(const Tensor& x, const Shape& shape);

enum class PadMode { Constant, Reflect, Symmetric };
/// Pads each axis with (before, after) elements (TF Pad / MirrorPad).
Tensor pad(const Tensor& x, const std::vector<std::pair<std::int64_t, std::int64_t>>& paddings,
           PadMode mode = PadMode::Constant, double value = 0.0);
/// Reverses the order of elements along the given axes (none if `axes` is empty, like TF ReverseV2).
Tensor reverse(const Tensor& x, const Axes& axes);

// ---- index ops ---------------------------------------------------------------------------------

/// TF GatherV2: output shape is x.shape[:axis] + indices.shape + x.shape[axis+1:].
/// Negative indices count from the end. Differentiable in x.
Tensor gather(const Tensor& x, const Tensor& indices, std::int64_t axis = 0);
/// TF GatherNd: the last axis of `indices` addresses the leading axes of x.
Tensor gather_nd(const Tensor& x, const Tensor& indices);
/// TF OneHot: appends an axis of size `depth`. Out-of-range indices give all `off`.
Tensor one_hot(const Tensor& indices, std::int64_t depth, double on = 1.0, double off = 0.0,
               DType dtype = DType::Float32);
/// x with src added at `index` along `axis` (repeated indices accumulate).
/// `src` has x's shape except `index.numel()` along axis.
Tensor index_add(const Tensor& x, std::int64_t axis, const Tensor& index, const Tensor& src);
/// Sums rows of `data` into `num_segments` buckets: out[s] = sum of data[i] with ids[i] == s.
/// Ids outside [0, num_segments) are dropped (TF UnsortedSegmentSum).
Tensor unsorted_segment_sum(const Tensor& data, const Tensor& segment_ids, std::int64_t num_segments);
/// Mean of each segment (empty segments give 0).
Tensor unsorted_segment_mean(const Tensor& data, const Tensor& segment_ids, std::int64_t num_segments);
/// x with updates added at the positions `indices` address (TF TensorScatterAdd).
Tensor scatter_nd_add(const Tensor& x, const Tensor& indices, const Tensor& updates);
/// zeros(shape) with updates added at `indices` (TF ScatterNd).
Tensor scatter_nd(const Tensor& indices, const Tensor& updates, const Shape& shape);
/// Largest k values along the last axis, sorted descending, and their int64 indices.
std::pair<Tensor, Tensor> top_k(const Tensor& x, std::int64_t k);
/// Indices that sort the last axis (stable).
Tensor argsort(const Tensor& x, bool descending = false);

// ---- matrix ops -----------------------------------------------------------------------------------

/// @brief Batched matrix product over the last two axes, like TF BatchMatMulV2.
/// Leading axes broadcast. CPU operands use dtype promotion; CUDA operands must
/// share one context and the same Float32/Float64 dtype. CUDA uses cuBLAS with
/// native packing for strided views and no host numerical fallback. Both inputs
/// need rank >= 2. Zero inner dimension produces zeros. The result retains the
/// CUDA context, or uses the default CPU context for CPU inputs.
/// Gradients with matching batch shapes run on CUDA; gradients requiring batch
/// broadcast reduction currently fail until native reductions are implemented.
/// Floating results can vary in their last bits with batch shape/backend.
/// @param a Left operand, with stored final axes [m,k] (or [k,m] when transposed).
/// @param b Right operand, with stored final axes [k,n] (or [n,k] when transposed).
/// @param transpose_a Transpose the final two axes of a; default false.
/// @param transpose_b Transpose the final two axes of b; default false.
/// @return Packed tensor with broadcast batch shape followed by [m,n].
/// @throws std::invalid_argument Invalid rank, incompatible dimensions/batches,
/// mixed devices/contexts, or unsupported CUDA dtype.
/// @throws std::logic_error CUDA matrix adapter is disabled or backend unsupported.
/// @throws std::runtime_error Device allocation, packing or cuBLAS failure.
/// @par Example
/// @code
/// const auto a=dm::nn::from_vector(std::vector<float>{1,2,3,4},{2,2});
/// const auto b=dm::nn::from_vector(std::vector<float>{5,6},{2,1});
/// std::print("{}\n",dm::nn::matmul(a,b).to_vector<float>()[0]); // 17
/// @endcode
/// @par References
/// - NVIDIA cuBLAS GEMM: https://docs.nvidia.com/cuda/archive/12.8.0/cublas/index.html#cublas-t-gemm
/// @see compute::Context::matmul()
Tensor matmul(const Tensor& a, const Tensor& b, bool transpose_a = false, bool transpose_b = false);
/// x + bias along the last axis (NHWC channels).
Tensor bias_add(const Tensor& x, const Tensor& bias);
/// Inner product of two 1-D tensors.
Tensor dot(const Tensor& a, const Tensor& b);
/// Outer product of two 1-D tensors.
Tensor outer(const Tensor& a, const Tensor& b);

}  // namespace datamunge::nn
