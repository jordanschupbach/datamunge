// Matrix products.

#include "detail.hpp"

namespace datamunge::nn {

using detail::as_dtype;
using detail::dispatch;
using detail::nograd;

namespace {

using Grads = std::vector<Tensor>;

/// C[b] = A[b] @ B[b] for packed [batch, m, k] and [batch, k, n] blocks.
template <typename T>
void gemm_batched(const T* a, const T* b, T* c, std::int64_t batch, std::int64_t m, std::int64_t k, std::int64_t n) {
  using Acc = typename std::conditional<std::is_integral<T>::value, std::int64_t, T>::type;
  std::vector<Acc> row(static_cast<std::size_t>(n));
  for (std::int64_t p = 0; p < batch; ++p) {
    const T* ap = a + p * m * k;
    const T* bp = b + p * k * n;
    T*       cp = c + p * m * n;
    for (std::int64_t i = 0; i < m; ++i) {
      std::fill(row.begin(), row.end(), Acc(0));
      for (std::int64_t l = 0; l < k; ++l) {
        const Acc av = static_cast<Acc>(ap[i * k + l]);
        const T*  br = bp + l * n;
        for (std::int64_t j = 0; j < n; ++j) row[static_cast<std::size_t>(j)] += av * static_cast<Acc>(br[j]);
      }
      for (std::int64_t j = 0; j < n; ++j) cp[i * n + j] = static_cast<T>(row[static_cast<std::size_t>(j)]);
    }
  }
}

}  // namespace

Tensor matmul(const Tensor& a, const Tensor& b, bool transpose_a, bool transpose_b) {
  if (a.dim() < 2 || b.dim() < 2)
    throw std::invalid_argument("matmul: inputs need rank >= 2, got " + shape_str(a.shape()) + " and " +
                                shape_str(b.shape()));
  DType dt = promote_types(a.dtype(), b.dtype());
  if (dt == DType::Bool) dt = DType::Int64;
  Tensor ac = as_dtype(a, dt), bc = as_dtype(b, dt);

  Tensor out;
  {
    NoGradGuard  guard;
    const Tensor av = transpose_a ? swapaxes(nograd(ac), -1, -2) : nograd(ac);
    const Tensor bv = transpose_b ? swapaxes(nograd(bc), -1, -2) : nograd(bc);
    const std::int64_t m = av.shape()[av.shape().size() - 2], k = av.shape().back();
    const std::int64_t k2 = bv.shape()[bv.shape().size() - 2], n = bv.shape().back();
    if (k != k2)
      throw std::invalid_argument("matmul: inner dimensions differ: " + shape_str(av.shape()) + " @ " +
                                  shape_str(bv.shape()) + (transpose_a || transpose_b ? " (after transposes)" : ""));
    const Shape batch_a(av.shape().begin(), av.shape().end() - 2);
    const Shape batch_b(bv.shape().begin(), bv.shape().end() - 2);
    const Shape batch = broadcast_shapes(batch_a, batch_b);
    Shape       sa = batch, sb = batch, so = batch;
    sa.insert(sa.end(), {m, k});
    sb.insert(sb.end(), {k, n});
    so.insert(so.end(), {m, n});
    const Tensor pa = broadcast_to(av, sa).contiguous();
    const Tensor pb = broadcast_to(bv, sb).contiguous();
    out             = empty(so, dt);
    const std::int64_t nb = shape_numel(batch);
    dispatch(dt, [&](auto tag) {
      using T = typename decltype(tag)::type;
      if constexpr (!std::is_same<T, bool>::value)
        gemm_batched<T>(pa.data_ptr<T>(), pb.data_ptr<T>(), out.data_ptr<T>(), nb, m, k, n);
    });
  }
  SavedTensor sa(ac), sb(bc);
  const bool  need_a = ac.requires_grad(), need_b = bc.requires_grad();
  record("BatchMatMulV2", {ac, bc}, {out}, [sa, sb, transpose_a, transpose_b, need_a, need_b](const Grads& g, const Grads&) {
    const Tensor& A = sa.get();
    const Tensor& B = sb.get();
    Tensor        ga, gb;
    if (!transpose_a && !transpose_b) {
      if (need_a) ga = matmul(g[0], B, false, true);
      if (need_b) gb = matmul(A, g[0], true, false);
    } else if (transpose_a && !transpose_b) {
      if (need_a) ga = matmul(B, g[0], false, true);
      if (need_b) gb = matmul(A, g[0], false, false);
    } else if (!transpose_a && transpose_b) {
      if (need_a) ga = matmul(g[0], B, false, false);
      if (need_b) gb = matmul(g[0], A, true, false);
    } else {
      if (need_a) ga = matmul(B, g[0], true, true);
      if (need_b) gb = matmul(g[0], A, true, true);
    }
    return Grads{ga.defined() ? sum_to(ga, A.shape()) : ga, gb.defined() ? sum_to(gb, B.shape()) : gb};
  });
  return out;
}

Tensor bias_add(const Tensor& x, const Tensor& bias) {
  if (bias.dim() != 1 || x.dim() < 1 || bias.shape()[0] != x.shape().back())
    throw std::invalid_argument("bias_add: bias " + shape_str(bias.shape()) + " must be 1-D and match the last axis of " +
                                shape_str(x.shape()));
  return add(x, bias);
}

Tensor dot(const Tensor& a, const Tensor& b) {
  if (a.dim() != 1 || b.dim() != 1 || a.shape()[0] != b.shape()[0])
    throw std::invalid_argument("dot: need two 1-D tensors of equal length, got " + shape_str(a.shape()) + " and " +
                                shape_str(b.shape()));
  return sum(mul(a, b));
}

Tensor outer(const Tensor& a, const Tensor& b) {
  if (a.dim() != 1 || b.dim() != 1) throw std::invalid_argument("outer: need two 1-D tensors");
  return mul(reshape(a, {-1, 1}), reshape(b, {1, -1}));
}

}  // namespace datamunge::nn
