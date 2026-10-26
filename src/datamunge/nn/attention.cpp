// Attention building blocks: masks, scaled dot-product attention (plain and
// fused), multi-head attention with a KV cache, additive attention and
// positional encodings.

#include "datamunge/nn/attention.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <tuple>

#include "datamunge/nn/layers.hpp"
#include "detail.hpp"
#include "gemm.hpp"

namespace datamunge::nn {

using detail::as_dtype;
using detail::dispatch_float;
using detail::nograd;

namespace {

using I64   = std::int64_t;
using Grads = std::vector<Tensor>;

// Score given to excluded keys on the plain path: finite, so a row with no
// allowed key gives a uniform softmax (zeroed afterwards) rather than NaN.
constexpr double kExcluded = -1e30;

// Fused kernel tile sizes (queries x keys per block).
constexpr I64 kBlockQ = 64;
constexpr I64 kBlockK = 128;

// Auto picks the fused kernel from this many scores per head on (measured on
// darwin/Accelerate: fused forward+backward is ~2x faster from 32 x 32 up and
// ~5x at 512 x 512, and its memory does not grow with Tq * Tk).
constexpr I64 kFuseMinScores = 32 * 32;

Shape lead_shape(const Tensor& t) { return Shape(t.shape().begin(), t.shape().end() - 2); }

/// Bool keep mask of `mask` (bool, or integer read as nonzero), undefined for float masks.
Tensor keep_part(const Tensor& mask) {
  if (!mask.defined() || is_floating(mask.dtype())) return Tensor();
  return mask.dtype() == DType::Bool ? mask : not_equal(mask, 0.0);
}

/// Combined keep mask from a user keep mask and the causal flag (undefined when neither).
Tensor combined_keep(const Tensor& keep, bool causal, I64 Tq, I64 Tk) {
  if (!causal) return keep;
  const Tensor c = causal_keep_mask(Tq, Tk);
  return keep.defined() ? logical_and(keep, c) : c;
}

/// softmax(scores + bias) over the last axis with excluded (keep == false)
/// entries; rows with nothing kept give zero weights.
Tensor masked_softmax(Tensor scores, const Tensor& bias, const Tensor& keep) {
  if (bias.defined()) scores = add(scores, bias);
  if (!keep.defined()) return softmax(scores, -1);
  Tensor     w       = softmax(where(keep, scores, kExcluded), -1);
  const auto row_any = any(keep, {-1}, true);
  if (!all(row_any).item<bool>()) w = mul(w, cast(row_any, w.dtype()));
  return w;
}

void check_mask_shape(const Tensor& mask, const Shape& full, const char* what) {
  if (!mask.defined()) return;
  bool ok = mask.dim() <= static_cast<I64>(full.size());
  if (ok) {
    try {
      ok = broadcast_shapes(mask.shape(), full) == full;
    } catch (const std::exception&) {
      ok = false;
    }
  }
  if (!ok)
    throw std::invalid_argument(std::string(what) + ": mask " + shape_str(mask.shape()) + " does not broadcast to " +
                                shape_str(full));
}

// ---- fused kernel ------------------------------------------------------------------------------

/// Strided reader of a mask broadcast to [lead..., Tq, Tk]: element (b, i, j)
/// lives at base[boff[b] + i * sr + j * sc].
struct MaskView {
  std::vector<I64> boff;
  I64              sr{0}, sc{0};
  bool             defined{false};
};

MaskView mask_view(const Tensor& m, const Shape& lead, I64 Tq, I64 Tk) {
  MaskView view;
  if (!m.defined()) return view;
  view.defined = true;
  Shape full   = lead;
  full.push_back(Tq);
  full.push_back(Tk);
  const Shape st = detail::broadcast_strides(m, full);
  const I64   Bt = shape_numel(lead);
  view.boff.resize(static_cast<std::size_t>(Bt));
  for (I64 b = 0; b < Bt; ++b) {
    I64 rem = b, off = m.offset();
    for (I64 a = static_cast<I64>(lead.size()) - 1; a >= 0; --a) {
      const auto ax = static_cast<std::size_t>(a);
      off += (rem % lead[ax]) * st[ax];
      rem /= lead[ax];
    }
    view.boff[static_cast<std::size_t>(b)] = off;
  }
  view.sr = st[full.size() - 2];
  view.sc = st[full.size() - 1];
  return view;
}

/// Everything the fused kernels need to know about one call.
template <typename T>
struct FusedProblem {
  I64      Bt{0}, Tq{0}, Tk{0}, D{0}, Dv{0};
  T        scale{1};
  bool     causal{false};
  I64      off{0};  // causal: key j allowed for query i when j <= i + off
  const T* Q{nullptr};
  const T* K{nullptr};
  const T* V{nullptr};
  const T* bias{nullptr};
  MaskView bview;
  const bool* keep{nullptr};
  MaskView    kview;

  /// Adds the bias to a [nq, nk] block of scaled scores and sets excluded entries to -inf.
  void mask_block(T* S, I64 b, I64 i0, I64 nq, I64 j0, I64 nk) const {
    constexpr T ninf = -std::numeric_limits<T>::infinity();
    for (I64 r = 0; r < nq; ++r) {
      T*        row = S + r * nk;
      const I64 i   = i0 + r;
      if (bias) {
        const T* bp = bias + bview.boff[static_cast<std::size_t>(b)] + i * bview.sr + j0 * bview.sc;
        for (I64 c = 0; c < nk; ++c) row[c] += bp[c * bview.sc];
      }
      if (keep) {
        const bool* kp = keep + kview.boff[static_cast<std::size_t>(b)] + i * kview.sr + j0 * kview.sc;
        for (I64 c = 0; c < nk; ++c)
          if (!kp[c * kview.sc]) row[c] = ninf;
      }
      if (causal)
        for (I64 c = std::max<I64>(0, i + off + 1 - j0); c < nk; ++c) row[c] = ninf;
    }
  }
  /// Keys [0, key_limit(i0, nq)) can be visible to queries i0..i0+nq-1.
  I64 key_limit(I64 i0, I64 nq) const { return causal ? std::clamp<I64>(i0 + nq + off, 0, Tk) : Tk; }
};

/// Forward: O [Bt, Tq, Dv] and the row log-sum-exp LSE [Bt, Tq] (+inf for rows with nothing visible).
template <typename T>
void fused_forward(const FusedProblem<T>& p, T* O, T* LSE) {
  const I64 nqb = (p.Tq + kBlockQ - 1) / kBlockQ;
  const I64 scratch_q = std::min(kBlockQ, p.Tq), scratch_k = std::min(kBlockK, p.Tk);
  parallel_for(0, p.Bt * nqb, 1, [&](I64 lo, I64 hi) {
    std::vector<T> S(static_cast<std::size_t>(scratch_q * scratch_k)), acc(static_cast<std::size_t>(scratch_q * p.Dv));
    std::vector<T> m(static_cast<std::size_t>(scratch_q)), l(static_cast<std::size_t>(scratch_q));
    for (I64 task = lo; task < hi; ++task) {
      const I64 b = task / nqb, i0 = (task % nqb) * kBlockQ, nq = std::min(kBlockQ, p.Tq - i0);
      const T*  q = p.Q + (b * p.Tq + i0) * p.D;
      std::fill(m.begin(), m.end(), -std::numeric_limits<T>::infinity());
      std::fill(l.begin(), l.end(), T(0));
      std::fill(acc.begin(), acc.begin() + nq * p.Dv, T(0));
      const I64 jmax = p.key_limit(i0, nq);
      for (I64 j0 = 0; j0 < jmax; j0 += kBlockK) {
        const I64 nk = std::min(kBlockK, jmax - j0);
        detail::gemm(false, true, nq, nk, p.D, p.scale, q, p.D, p.K + (b * p.Tk + j0) * p.D, p.D, T(0), S.data(), nk);
        p.mask_block(S.data(), b, i0, nq, j0, nk);
        for (I64 r = 0; r < nq; ++r) {
          T* row  = S.data() + r * nk;
          T  rmax = -std::numeric_limits<T>::infinity();
          for (I64 c = 0; c < nk; ++c) rmax = std::max(rmax, row[c]);
          const T mnew = std::max(m[r], rmax);
          if (mnew == -std::numeric_limits<T>::infinity()) {  // nothing visible yet
            std::fill(row, row + nk, T(0));
            continue;
          }
          const T alpha = std::exp(m[r] - mnew);  // 0 when m was -inf
          T       sum   = 0;
          for (I64 c = 0; c < nk; ++c) {
            row[c] = std::exp(row[c] - mnew);
            sum += row[c];
          }
          l[r] = l[r] * alpha + sum;
          m[r] = mnew;
          if (alpha != T(1)) {
            T* a = acc.data() + r * p.Dv;
            for (I64 c = 0; c < p.Dv; ++c) a[c] *= alpha;
          }
        }
        detail::gemm(false, false, nq, p.Dv, nk, T(1), S.data(), nk, p.V + (b * p.Tk + j0) * p.Dv, p.Dv, T(1),
                     acc.data(), p.Dv);
      }
      for (I64 r = 0; r < nq; ++r) {
        T*       o = O + (b * p.Tq + i0 + r) * p.Dv;
        const T* a = acc.data() + r * p.Dv;
        if (l[r] > 0) {
          const T inv = T(1) / l[r];
          for (I64 c = 0; c < p.Dv; ++c) o[c] = a[c] * inv;
          LSE[b * p.Tq + i0 + r] = m[r] + std::log(l[r]);
        } else {
          std::fill(o, o + p.Dv, T(0));
          LSE[b * p.Tq + i0 + r] = std::numeric_limits<T>::infinity();
        }
      }
    }
  });
}

/// First-order backward with the scores recomputed block by block.
template <typename T>
void fused_backward(const FusedProblem<T>& p, const T* O, const T* LSE, const T* dO, T* dQ, T* dK, T* dV) {
  const I64 Bt = p.Bt, Tq = p.Tq, Tk = p.Tk, D = p.D, Dv = p.Dv;
  // Delta_i = sum_c dO_ic O_ic (= sum_j P_ij dP_ij).
  std::vector<T> delta(static_cast<std::size_t>(Bt * Tq));
  detail::parallel_elements(Bt * Tq, [&](I64 r) {
    T s = 0;
    for (I64 c = 0; c < Dv; ++c) s += dO[r * Dv + c] * O[r * Dv + c];
    delta[static_cast<std::size_t>(r)] = s;
  });
  std::fill(dQ, dQ + Bt * Tq * D, T(0));
  std::fill(dK, dK + Bt * Tk * D, T(0));
  std::fill(dV, dV + Bt * Tk * Dv, T(0));

  struct Scratch {
    std::vector<T> P, dP;
    explicit Scratch(I64 count) : P(static_cast<std::size_t>(count)), dP(static_cast<std::size_t>(count)) {}
  };
  // One (query block, key block) tile; accumulates into dK/dV and/or dQ.
  auto tile = [&](Scratch& s, I64 b, I64 i0, I64 nq, I64 j0, I64 nk, bool do_kv, bool do_q) {
    const T* q  = p.Q + (b * Tq + i0) * D;
    const T* k  = p.K + (b * Tk + j0) * D;
    const T* v  = p.V + (b * Tk + j0) * Dv;
    const T* go = dO + (b * Tq + i0) * Dv;
    T*       P  = s.P.data();
    T*       dP = s.dP.data();
    detail::gemm(false, true, nq, nk, D, p.scale, q, D, k, D, T(0), P, nk);
    p.mask_block(P, b, i0, nq, j0, nk);
    for (I64 r = 0; r < nq; ++r) {
      const T lse = LSE[b * Tq + i0 + r];
      for (I64 c = 0; c < nk; ++c) P[r * nk + c] = std::exp(P[r * nk + c] - lse);  // 0 when masked or lse = +inf
    }
    detail::gemm(false, true, nq, nk, Dv, T(1), go, Dv, v, Dv, T(0), dP, nk);
    for (I64 r = 0; r < nq; ++r) {
      const T dl = delta[static_cast<std::size_t>(b * Tq + i0 + r)];
      for (I64 c = 0; c < nk; ++c) dP[r * nk + c] = P[r * nk + c] * (dP[r * nk + c] - dl);  // dS
    }
    if (do_kv) {
      detail::gemm(true, false, nk, Dv, nq, T(1), P, nk, go, Dv, T(1), dV + (b * Tk + j0) * Dv, Dv);
      detail::gemm(true, false, nk, D, nq, p.scale, dP, nk, q, D, T(1), dK + (b * Tk + j0) * D, D);
    }
    if (do_q) detail::gemm(false, false, nq, D, nk, p.scale, dP, nk, k, D, T(1), dQ + (b * Tq + i0) * D, D);
  };
  const I64 nqb = (Tq + kBlockQ - 1) / kBlockQ, nkb = (Tk + kBlockK - 1) / kBlockK;
  if (Bt >= get_num_threads()) {
    // Enough independent heads: one pass per head computes all three gradients.
    parallel_for(0, Bt, 1, [&](I64 lo, I64 hi) {
      Scratch s(std::min(kBlockQ, Tq) * std::min(kBlockK, Tk));
      for (I64 b = lo; b < hi; ++b)
        for (I64 qb = 0; qb < nqb; ++qb) {
          const I64 i0 = qb * kBlockQ, nq = std::min(kBlockQ, Tq - i0), jmax = p.key_limit(i0, nq);
          for (I64 j0 = 0; j0 < jmax; j0 += kBlockK) tile(s, b, i0, nq, j0, std::min(kBlockK, jmax - j0), true, true);
        }
    });
    return;
  }
  // Few heads: parallelise over key blocks for dK/dV, then over query blocks for dQ.
  parallel_for(0, Bt * nkb, 1, [&](I64 lo, I64 hi) {
    Scratch s(std::min(kBlockQ, Tq) * std::min(kBlockK, Tk));
    for (I64 task = lo; task < hi; ++task) {
      const I64 b = task / nkb, j0 = (task % nkb) * kBlockK, nk = std::min(kBlockK, Tk - j0);
      for (I64 i0 = 0; i0 < Tq; i0 += kBlockQ) {
        const I64 nq = std::min(kBlockQ, Tq - i0);
        if (p.key_limit(i0, nq) <= j0) continue;
        tile(s, b, i0, nq, j0, std::min(nk, p.key_limit(i0, nq) - j0), true, false);
      }
    }
  });
  parallel_for(0, Bt * nqb, 1, [&](I64 lo, I64 hi) {
    Scratch s(std::min(kBlockQ, Tq) * std::min(kBlockK, Tk));
    for (I64 task = lo; task < hi; ++task) {
      const I64 b = task / nqb, i0 = (task % nqb) * kBlockQ, nq = std::min(kBlockQ, Tq - i0), jmax = p.key_limit(i0, nq);
      for (I64 j0 = 0; j0 < jmax; j0 += kBlockK) tile(s, b, i0, nq, j0, std::min(kBlockK, jmax - j0), false, true);
    }
  });
}

template <typename T>
FusedProblem<T> make_problem(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& bias, const Tensor& keep,
                             bool causal, double scale) {
  FusedProblem<T> p;
  const Shape     lead = lead_shape(q);
  p.Bt                 = shape_numel(lead);
  p.Tq                 = q.size(-2);
  p.Tk                 = k.size(-2);
  p.D                  = q.size(-1);
  p.Dv                 = v.size(-1);
  p.scale              = static_cast<T>(scale);
  p.causal             = causal;
  p.off                = p.Tk - p.Tq;
  p.Q                  = q.data_ptr<T>();
  p.K                  = k.data_ptr<T>();
  p.V                  = v.data_ptr<T>();
  if (bias.defined()) {
    p.bias  = detail::raw_data<T>(bias);
    p.bview = mask_view(bias, lead, p.Tq, p.Tk);
  }
  if (keep.defined()) {
    p.keep  = detail::raw_data<bool>(keep);
    p.kview = mask_view(keep, lead, p.Tq, p.Tk);
  }
  return p;
}

/// Gradients of the attention output with recorded ops (create_graph, or a bias that needs a gradient).
Grads composite_backward(const Tensor& Q, const Tensor& K, const Tensor& V, const Tensor& bias, const Tensor& keep,
                         bool causal, double scale, const Tensor& dO, bool need_q, bool need_k, bool need_v,
                         bool need_b) {
  const Tensor kp = combined_keep(keep, causal, Q.size(-2), K.size(-2));
  const Tensor P  = masked_softmax(mul(matmul(Q, K, false, true), scale), bias, kp);
  const Tensor dP = matmul(dO, V, false, true);
  const Tensor dS = mul(P, sub(dP, sum(mul(P, dP), {-1}, true)));
  Grads        g(4);
  if (need_q) g[0] = mul(matmul(dS, K), scale);
  if (need_k) g[1] = mul(matmul(dS, Q, true, false), scale);
  if (need_v) g[2] = matmul(P, dO, true, false);
  if (need_b) g[3] = sum_to(dS, bias.shape());
  return g;
}

/// The fused primitive on q, k, v with equal leading axes; bias float (or undefined), keep bool (or undefined).
Tensor fused_attention(const Tensor& q_in, const Tensor& k_in, const Tensor& v_in, const Tensor& bias,
                       const Tensor& keep, bool causal, double scale) {
  const Tensor q = q_in.contiguous(), k = k_in.contiguous(), v = v_in.contiguous();
  Shape        out_shape = q.shape();
  out_shape.back()       = v.size(-1);
  Tensor out             = empty(out_shape, q.dtype());
  Shape  lse_shape       = lead_shape(q);
  lse_shape.push_back(q.size(-2));
  Tensor lse = empty(lse_shape, q.dtype());
  {
    NoGradGuard  guard;
    const Tensor qk = nograd(q), kk = nograd(k), vk = nograd(v), bk = nograd(bias);
    dispatch_float(q.dtype(), "scaled_dot_product_attention", [&](auto tag) {
      using T = typename decltype(tag)::type;
      fused_forward<T>(make_problem<T>(qk, kk, vk, bk, keep, causal, scale), out.data_ptr<T>(), lse.data_ptr<T>());
    });
  }
  const SavedTensor sq(q), sk(k), sv(v), sb(bias);
  const bool need_q = q.requires_grad(), need_k = k.requires_grad(), need_v = v.requires_grad();
  const bool need_b = bias.defined() && bias.requires_grad();
  record(
      "FusedAttention", {q, k, v, bias}, {out},
      [sq, sk, sv, sb, keep, lse, causal, scale, need_q, need_k, need_v, need_b](const Grads& g, const Grads& outs) {
        const Tensor &Q = sq.get(), &K = sk.get(), &V = sv.get(), &B = sb.get();
        if (is_grad_enabled() || need_b)
          return composite_backward(Q, K, V, B, keep, causal, scale, g[0], need_q, need_k, need_v, need_b);
        Tensor       dq = empty(Q.shape(), Q.dtype()), dk = empty(K.shape(), K.dtype()), dv = empty(V.shape(), V.dtype());
        const Tensor go = g[0].contiguous(), o = outs[0].contiguous();
        dispatch_float(Q.dtype(), "scaled_dot_product_attention backward", [&](auto tag) {
          using T = typename decltype(tag)::type;
          fused_backward<T>(make_problem<T>(Q, K, V, B, keep, causal, scale), o.data_ptr<T>(), lse.data_ptr<T>(),
                            go.data_ptr<T>(), dq.data_ptr<T>(), dk.data_ptr<T>(), dv.data_ptr<T>());
        });
        return Grads{need_q ? dq : Tensor(), need_k ? dk : Tensor(), need_v ? dv : Tensor(), Tensor()};
      },
      true);
  return out;
}

}  // namespace

// ---- masks ------------------------------------------------------------------------------------------

Tensor causal_keep_mask(I64 Tq, I64 Tk) {
  const Tensor rows = reshape(arange(0, static_cast<double>(Tq), 1, DType::Int64), {Tq, 1});
  const Tensor cols = reshape(arange(0, static_cast<double>(Tk), 1, DType::Int64), {1, Tk});
  return less_equal(cols, add(rows, scalar(static_cast<double>(Tk - Tq), DType::Int64)));
}
Tensor causal_keep_mask(I64 T) { return causal_keep_mask(T, T); }

Tensor causal_mask(I64 Tq, I64 Tk, DType dtype, double value) {
  return additive_mask(causal_keep_mask(Tq, Tk), dtype, value);
}
Tensor causal_mask(I64 T, DType dtype, double value) { return causal_mask(T, T, dtype, value); }

Tensor padding_keep_mask(const Tensor& lengths, I64 T) {
  if (lengths.dim() != 1) throw std::invalid_argument("padding_keep_mask: lengths must be [N]");
  const Tensor pos = reshape(arange(0, static_cast<double>(T), 1, DType::Int64), {1, T});
  const Tensor len = reshape(cast(lengths, DType::Int64), {-1, 1});
  return reshape(less(pos, len), {lengths.size(0), 1, 1, T});
}

Tensor padding_keep_mask_from_ids(const Tensor& ids, I64 pad_id) {
  if (ids.dim() != 2) throw std::invalid_argument("padding_keep_mask_from_ids: ids must be [N, T]");
  return reshape(not_equal(ids, static_cast<double>(pad_id)), {ids.size(0), 1, 1, ids.size(1)});
}

Tensor additive_mask(const Tensor& keep, DType dtype, double value) {
  return where(keep_part(keep), zeros({}, dtype), full({}, value, dtype));
}

// ---- scaled dot-product attention -----------------------------------------------------------------

namespace {

struct Prepared {
  Tensor q, k, v, bias, keep;
  DType  dt;
  double scale;
};

Prepared prepare(const Tensor& q, const Tensor& k, const Tensor& v, const AttentionOptions& o) {
  if (q.dim() < 2 || k.dim() < 2 || v.dim() < 2)
    throw std::invalid_argument("scaled_dot_product_attention: q, k and v need rank >= 2");
  if (q.size(-1) < 1)
    throw std::invalid_argument("scaled_dot_product_attention: key width must be positive");
  if (q.size(-1) != k.size(-1) || k.size(-2) != v.size(-2))
    throw std::invalid_argument("scaled_dot_product_attention: shapes " + shape_str(q.shape()) + ", " +
                                shape_str(k.shape()) + ", " + shape_str(v.shape()) +
                                " do not fit q [..., Tq, d], k [..., Tk, d], v [..., Tk, dv]");
  Prepared p;
  p.dt = promote_types(promote_types(q.dtype(), k.dtype()), v.dtype());
  if (!is_floating(p.dt)) p.dt = DType::Float32;
  p.q     = as_dtype(q, p.dt);
  p.k     = as_dtype(k, p.dt);
  p.v     = as_dtype(v, p.dt);
  p.scale = o.scale > 0 ? o.scale : 1.0 / std::sqrt(static_cast<double>(q.size(-1)));
  Shape full = broadcast_shapes(broadcast_shapes(lead_shape(q), lead_shape(k)), lead_shape(v));
  full.push_back(q.size(-2));
  full.push_back(k.size(-2));
  check_mask_shape(o.mask, full, "scaled_dot_product_attention");
  if (o.mask.defined()) {
    if (is_floating(o.mask.dtype()))
      p.bias = as_dtype(o.mask, p.dt);
    else
      p.keep = keep_part(o.mask);
  }
  return p;
}

bool fuse(const Prepared& p, const AttentionOptions& o) {
  switch (o.impl) {
    case AttentionImpl::Plain: return false;
    case AttentionImpl::Fused:
      if (o.need_weights) throw std::invalid_argument("scaled_dot_product_attention: the fused path returns no weights");
      if (o.dropout > 0) throw std::invalid_argument("scaled_dot_product_attention: the fused path has no dropout");
      return true;
    case AttentionImpl::Auto: break;
  }
  if (o.need_weights || o.dropout > 0) return false;
  if (p.bias.defined() && p.bias.requires_grad() && is_grad_enabled()) return false;
  return p.q.size(-2) * p.k.size(-2) >= kFuseMinScores;
}

}  // namespace

bool attention_would_fuse(const Shape& q_shape, const Shape& k_shape, const AttentionOptions& o) {
  if (o.impl != AttentionImpl::Auto) return o.impl == AttentionImpl::Fused;
  if (o.need_weights || o.dropout > 0 || q_shape.size() < 2 || k_shape.size() < 2) return false;
  if (o.mask.defined() && is_floating(o.mask.dtype()) && o.mask.requires_grad() && is_grad_enabled()) return false;
  return q_shape[q_shape.size() - 2] * k_shape[k_shape.size() - 2] >= kFuseMinScores;
}

Attended scaled_dot_product_attention(const Tensor& q, const Tensor& k, const Tensor& v, const AttentionOptions& o) {
  if (!(o.dropout >= 0 && o.dropout < 1)) throw std::invalid_argument("scaled_dot_product_attention: dropout must be in [0, 1)");
  const Prepared p = prepare(q, k, v, o);
  Attended       a;
  if (fuse(p, o)) {
    // Broadcast the leading axes (copies only when they differ) so the kernel sees equal batches.
    Shape lead = broadcast_shapes(broadcast_shapes(lead_shape(p.q), lead_shape(p.k)), lead_shape(p.v));
    auto  full = [&](const Tensor& t) {
      Shape s = lead;
      s.push_back(t.size(-2));
      s.push_back(t.size(-1));
      return t.shape() == s ? t : broadcast_to(t, s);
    };
    a.output = fused_attention(full(p.q), full(p.k), full(p.v), p.bias, p.keep, o.causal, p.scale);
    return a;
  }
  const Tensor keep    = combined_keep(p.keep, o.causal, p.q.size(-2), p.k.size(-2));
  const Tensor weights = masked_softmax(mul(matmul(p.q, p.k, false, true), p.scale), p.bias, keep);
  Tensor       mixed   = weights;
  if (o.dropout > 0) mixed = dropout(weights, o.dropout, true, o.generator ? *o.generator : default_generator());
  a.output = matmul(mixed, p.v);
  if (o.need_weights) a.weights = weights;
  return a;
}

Attended scaled_dot_product_attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& mask) {
  AttentionOptions o;
  o.mask         = mask;
  o.need_weights = true;
  o.impl         = AttentionImpl::Plain;
  return scaled_dot_product_attention(q, k, v, o);
}

// ---- KV cache ----------------------------------------------------------------------------------------

std::pair<Tensor, Tensor> KvCache::append(const Tensor& k_new, const Tensor& v_new) {
  if (k_new.dim() != 4 || v_new.dim() != 4 || k_new.size(2) != v_new.size(2))
    throw std::invalid_argument("KvCache::append: keys and values must be [N, heads, t, dh] with the same t");
  const I64 t = k_new.size(2);
  if (length_ > 0) {
    const Tensor k0 = keys(), v0 = values();
    if (k0.size(0) != k_new.size(0) || k0.size(1) != k_new.size(1) || k0.size(3) != k_new.size(3) ||
        v0.size(3) != v_new.size(3) || k0.dtype() != k_new.dtype() || v0.dtype() != v_new.dtype())
      throw std::invalid_argument("KvCache::append: new keys/values " + shape_str(k_new.shape()) +
                                  " do not match the cache " + shape_str(k0.shape()));
  }
  if (is_grad_enabled()) {
    // Recorded concatenation: gradients flow into every cached step.
    if (length_ == 0) {
      k_buf_ = k_new;
      v_buf_ = v_new;
    } else {
      k_buf_ = concat({keys(), k_new}, 2);
      v_buf_ = concat({values(), v_new}, 2);
    }
    buffered_ = false;
    length_ += t;
    return {k_buf_, v_buf_};
  }
  // Inference: write into a buffer with spare capacity and return views.
  const I64 need = length_ + t;
  if (!buffered_ || k_buf_.size(2) < need) {
    const I64 cap = std::max<I64>({need, 16, buffered_ ? 2 * k_buf_.size(2) : 0});
    Tensor    kb  = zeros({k_new.size(0), k_new.size(1), cap, k_new.size(3)}, k_new.dtype());
    Tensor    vb  = zeros({v_new.size(0), v_new.size(1), cap, v_new.size(3)}, v_new.dtype());
    if (length_ > 0) {
      slice_axis(kb, 2, 0, length_).copy_(keys());
      slice_axis(vb, 2, 0, length_).copy_(values());
    }
    k_buf_    = kb;
    v_buf_    = vb;
    buffered_ = true;
  }
  slice_axis(k_buf_, 2, length_, need).copy_(k_new);
  slice_axis(v_buf_, 2, length_, need).copy_(v_new);
  length_ = need;
  return {keys(), values()};
}

Tensor KvCache::keys() const {
  if (length_ == 0) return Tensor();
  return buffered_ ? slice_axis(k_buf_, 2, 0, length_) : k_buf_;
}
Tensor KvCache::values() const {
  if (length_ == 0) return Tensor();
  return buffered_ ? slice_axis(v_buf_, 2, 0, length_) : v_buf_;
}

void KvCache::reset() {
  k_buf_    = Tensor();
  v_buf_    = Tensor();
  length_   = 0;
  buffered_ = false;
}

// ---- multi-head attention ----------------------------------------------------------------------------

Tensor split_heads(const Tensor& x, I64 heads) {
  if (x.dim() != 3 || heads < 1 || x.size(2) % heads != 0)
    throw std::invalid_argument("split_heads: expected [N, T, heads * dh], got " + shape_str(x.shape()));
  return transpose(reshape(x, {x.size(0), x.size(1), heads, x.size(2) / heads}), {0, 2, 1, 3});
}

Tensor merge_heads(const Tensor& x) {
  if (x.dim() != 4) throw std::invalid_argument("merge_heads: expected [N, heads, T, dh]");
  return reshape(transpose(x, {0, 2, 1, 3}), {x.size(0), x.size(2), x.size(1) * x.size(3)});
}

Attended multi_head_attention(ParameterStore& store, const std::string& name, const Tensor& x_q, const Tensor& x_kv,
                              I64 heads, const MultiHeadOptions& options) {
  return multi_head_attention(store, name, x_q, x_kv, x_kv, heads, options);
}

Attended multi_head_attention(ParameterStore& store, const std::string& name, const Tensor& x_q, const Tensor& x_k,
                              const Tensor& x_v, I64 heads, const MultiHeadOptions& o) {
  if (x_q.dim() != 3 || x_k.dim() != 3 || x_v.dim() != 3)
    throw std::invalid_argument("multi_head_attention: inputs must be [N, T, features]");
  const I64 d_model = x_q.size(2);
  if (d_model < 1) throw std::invalid_argument("multi_head_attention: d_model (the query feature size) must be positive");
  if (heads < 1 || d_model % heads != 0)
    throw std::invalid_argument("multi_head_attention: d_model " + std::to_string(d_model) +
                                " is not divisible by heads " + std::to_string(heads));
  const I64 dh  = d_model / heads;
  const I64 kvh = o.kv_heads > 0 ? o.kv_heads : heads;
  if (heads % kvh != 0) throw std::invalid_argument("multi_head_attention: heads must be a multiple of kv_heads");
  auto proj = [&](const std::string& w, const std::string& b, const Tensor& x, I64 out) {
    const I64    in = x.size(2);
    const Tensor W  = store.get(name + "/" + w, {in, out}, o.init);
    Tensor       y  = reshape(matmul(reshape(x, {-1, in}), W), {x.size(0), x.size(1), out});
    if (o.bias) y = bias_add(y, store.get(name + "/" + b, {out}, init::zeros()));
    return y;
  };
  Tensor    q      = split_heads(proj("Wq", "bq", x_q, d_model), heads);
  Tensor    k      = split_heads(proj("Wk", "bk", x_k, kvh * dh), kvh);
  Tensor    v      = split_heads(proj("Wv", "bv", x_v, kvh * dh), kvh);
  const I64 offset = o.cache ? o.cache->length() : 0;
  if (o.rotary) {
    q = apply_rotary(q, offset, o.rotary_base, o.rotary_layout);
    k = apply_rotary(k, offset, o.rotary_base, o.rotary_layout);
  }
  if (o.cache) std::tie(k, v) = o.cache->append(k, v);
  if (kvh != heads) {
    k = repeat(k, heads / kvh, 1);
    v = repeat(v, heads / kvh, 1);
  }
  AttentionOptions ao;
  ao.mask         = o.mask;
  ao.causal       = o.causal;
  ao.dropout      = o.training ? o.dropout : 0.0;
  ao.generator    = o.generator;
  ao.need_weights = o.need_weights;
  ao.impl         = o.impl;
  Attended a      = scaled_dot_product_attention(q, k, v, ao);
  a.output        = proj("Wo", "bo", merge_heads(a.output), d_model);
  return a;
}

Attended additive_attention(ParameterStore& store, const std::string& name, const Tensor& q, const Tensor& k,
                            const Tensor& v, I64 units, const Tensor& mask) {
  if (q.dim() != 2 || k.dim() != 3 || v.dim() != 3 || k.size(0) != q.size(0) || v.size(1) != k.size(1))
    throw std::invalid_argument("additive_attention: expected q [N, dq], k [N, T, dk], v [N, T, dv]");
  const I64    N = k.size(0), T = k.size(1), dq = q.size(1), dk = k.size(2);
  const Tensor Wq = store.get(name + "/Wq", {dq, units}, init::glorot_uniform());
  const Tensor Wk = store.get(name + "/Wk", {dk, units}, init::glorot_uniform());
  const Tensor vv = store.get(name + "/v", {units, 1}, init::glorot_uniform());
  const Tensor qp = expand_dims(matmul(q, Wq), 1);                                 // [N, 1, units]
  const Tensor kp = reshape(matmul(reshape(k, {-1, dk}), Wk), {N, T, units});    // [N, T, units]
  const Tensor hidden = tanh(add(qp, kp));
  const Tensor scores = reshape(matmul(reshape(hidden, {-1, units}), vv), {N, T});  // [N, T]
  const bool   additive = mask.defined() && is_floating(mask.dtype());
  const Tensor bias     = additive ? as_dtype(mask, scores.dtype()) : Tensor();
  const Tensor keep     = additive ? Tensor() : keep_part(mask);
  Attended a;
  a.weights = masked_softmax(scores, bias, keep);
  a.output  = reshape(matmul(expand_dims(a.weights, 1), v), {N, v.size(2)});
  return a;
}

// ---- positional encodings -------------------------------------------------------------------------------

Tensor sinusoidal_encoding(I64 T, I64 d, double base, DType dtype, I64 offset) {
  if (T < 0 || d < 2 || d % 2 != 0)
    throw std::invalid_argument("sinusoidal_encoding: length must be >= 0 and width even and >= 2 (got T=" +
                                std::to_string(T) + ", d=" + std::to_string(d) + ")");
  if (!std::isfinite(base) || base <= 0) throw std::invalid_argument("sinusoidal_encoding: base must be positive");
  if (offset < 0) throw std::invalid_argument("sinusoidal_encoding: offset must be nonnegative");
  std::vector<double> table(static_cast<std::size_t>(T * d));
  for (I64 t = 0; t < T; ++t)
    for (I64 i = 0; 2 * i < d; ++i) {
      const double theta = static_cast<double>(t + offset) / std::pow(base, 2.0 * static_cast<double>(i) / d);
      table[static_cast<std::size_t>(t * d + 2 * i)] = std::sin(theta);
      if (2 * i + 1 < d) table[static_cast<std::size_t>(t * d + 2 * i + 1)] = std::cos(theta);
    }
  return cast(from_vector(table, {T, d}), dtype);
}

Tensor learned_encoding(ParameterStore& store, const std::string& name, I64 max_len, I64 d, double stddev) {
  return store.get(name + "/pos", {max_len, d}, init::normal(stddev));
}

namespace {

/// Inverse frequency of column c: base^(-2j/d) for the pair j that c belongs to.
std::vector<double> rotary_inv_freq(I64 d, double base, RotaryLayout layout) {
  if (d < 2 || d % 2 != 0)
    throw std::invalid_argument("rotary embedding: the feature size must be even and >= 2 (got " + std::to_string(d) + ")");
  if (!std::isfinite(base) || base <= 0) throw std::invalid_argument("rotary embedding: base must be positive");
  std::vector<double> f(static_cast<std::size_t>(d));
  for (I64 c = 0; c < d; ++c) {
    const I64 j = layout == RotaryLayout::Interleaved ? c / 2 : c % (d / 2);
    f[static_cast<std::size_t>(c)] = std::pow(base, -2.0 * static_cast<double>(j) / static_cast<double>(d));
  }
  return f;
}

/// (x_a, x_b) -> (-x_b, x_a) for every pair, so x cos + rotate(x) sin rotates each pair.
Tensor rotate_pairs(const Tensor& x, RotaryLayout layout) {
  const I64 d = x.size(-1);
  if (layout == RotaryLayout::HalfSplit) {
    const auto h = split(x, 2, -1);
    return concat({neg(h[1]), h[0]}, -1);
  }
  Shape pairs = x.shape();
  pairs.back() = d / 2;
  pairs.push_back(2);
  const auto parts = split(reshape(x, pairs), 2, -1);
  return reshape(concat({neg(parts[1]), parts[0]}, -1), x.shape());
}

}  // namespace

std::pair<Tensor, Tensor> rotary_tables(I64 T, I64 d, double base, DType dtype, I64 offset, RotaryLayout layout) {
  if (T < 0) throw std::invalid_argument("rotary_tables: length must be nonnegative");
  if (offset < 0)
    throw std::invalid_argument("rotary embedding: position offset must be nonnegative (got " + std::to_string(offset) + ")");
  const auto          f = rotary_inv_freq(d, base, layout);
  std::vector<double> c(static_cast<std::size_t>(T * d)), s(c.size());
  for (I64 t = 0; t < T; ++t)
    for (I64 j = 0; j < d; ++j) {
      const double theta                   = static_cast<double>(t + offset) * f[static_cast<std::size_t>(j)];
      c[static_cast<std::size_t>(t * d + j)] = std::cos(theta);
      s[static_cast<std::size_t>(t * d + j)] = std::sin(theta);
    }
  return {cast(from_vector(c, {T, d}), dtype), cast(from_vector(s, {T, d}), dtype)};
}

Tensor apply_rotary(const Tensor& x, I64 offset, double base, RotaryLayout layout) {
  if (x.dim() < 2) throw std::invalid_argument("apply_rotary: x must be [..., T, d]");
  const DType dt     = is_floating(x.dtype()) ? x.dtype() : DType::Float32;
  const auto  tables = rotary_tables(x.size(-2), x.size(-1), base, dt, offset, layout);
  const Tensor xc    = as_dtype(x, dt);
  return add(mul(xc, tables.first), mul(rotate_pairs(xc, layout), tables.second));
}

Tensor apply_rotary(const Tensor& x, const Tensor& positions, double base, RotaryLayout layout) {
  if (x.dim() < 2) throw std::invalid_argument("apply_rotary: x must be [..., T, d]");
  const DType dt = is_floating(x.dtype()) ? x.dtype() : DType::Float32;
  const I64   d  = x.size(-1);
  const auto  f  = rotary_inv_freq(d, base, layout);
  Tensor      angles = mul(expand_dims(cast(positions, DType::Float64), -1), from_vector(f, {d}));  // [..., T, d]
  if (positions.dim() == 2 && x.dim() == 4) angles = expand_dims(angles, 1);  // [N, 1, T, d] over heads
  const Tensor c = cast(cos(angles), dt), s = cast(sin(angles), dt);
  const Tensor xc = as_dtype(x, dt);
  return add(mul(xc, c), mul(rotate_pairs(xc, layout), s));
}

std::vector<double> alibi_slopes(I64 heads) {
  if (heads < 1) throw std::invalid_argument("alibi_slopes: heads must be positive (got " + std::to_string(heads) + ")");
  // Press et al.'s get_slopes: the geometric sequence for the largest power of
  // two P <= heads, then every other slope of the 2P sequence for the rest.
  I64 power = 1;
  while (power * 2 <= heads) power *= 2;
  std::vector<double> slopes;
  for (I64 h = 1; h <= power; ++h) slopes.push_back(std::pow(2.0, -8.0 * static_cast<double>(h) / static_cast<double>(power)));
  for (I64 h = 1; static_cast<I64>(slopes.size()) < heads; h += 2)
    slopes.push_back(std::pow(2.0, -4.0 * static_cast<double>(h) / static_cast<double>(power)));
  return slopes;
}

Tensor alibi_bias(I64 heads, I64 T, bool causal, DType dtype) { return alibi_bias(heads, T, T, causal, dtype); }

Tensor alibi_bias(I64 heads, I64 Tq, I64 Tk, bool causal, DType dtype) {
  if (Tq < 0 || Tk < 0)
    throw std::invalid_argument("alibi_bias: lengths must be nonnegative (got Tq=" + std::to_string(Tq) + ", Tk=" +
                                std::to_string(Tk) + ")");
  const auto          slopes = alibi_slopes(heads);
  std::vector<double> b(static_cast<std::size_t>(heads * Tq * Tk));
  for (I64 h = 0; h < heads; ++h)
    for (I64 i = 0; i < Tq; ++i) {
      const I64 m = i + Tk - Tq;
      for (I64 n = 0; n < Tk; ++n)
        b[static_cast<std::size_t>((h * Tq + i) * Tk + n)] =
            causal && n > m ? -1e9 : -slopes[static_cast<std::size_t>(h)] * static_cast<double>(std::abs(m - n)) + 0.0;
    }
  return cast(from_vector(b, {heads, Tq, Tk}), dtype);
}

Tensor relative_position_buckets(I64 Tq, I64 Tk, bool bidirectional, I64 num_buckets, I64 max_distance) {
  if (Tq < 0 || Tk < 0)
    throw std::invalid_argument("relative_position_buckets: lengths must be nonnegative (got Tq=" + std::to_string(Tq) +
                                ", Tk=" + std::to_string(Tk) + ")");
  if (num_buckets < 4 || (bidirectional && num_buckets % 2 != 0))
    throw std::invalid_argument("relative_position_buckets: num_buckets must be >= 4 (and even when bidirectional), got " +
                                std::to_string(num_buckets));
  if (max_distance <= (bidirectional ? num_buckets / 2 : num_buckets) / 2)
    throw std::invalid_argument("relative_position_buckets: max_distance " + std::to_string(max_distance) +
                                " must exceed the exact range " +
                                std::to_string((bidirectional ? num_buckets / 2 : num_buckets) / 2));
  std::vector<std::int64_t> out(static_cast<std::size_t>(Tq * Tk));
  for (I64 i = 0; i < Tq; ++i)
    for (I64 j = 0; j < Tk; ++j) {
      I64 n = -(j - (i + Tk - Tq));  // query minus key
      I64 ret = 0, nb = num_buckets;
      if (bidirectional) {
        nb /= 2;
        if (n < 0) ret += nb;
        n = std::abs(n);
      } else {
        n = std::max<I64>(n, 0);
      }
      const I64 max_exact = nb / 2;
      if (n < max_exact) {
        ret += n;
      } else {
        const double ratio = std::log(static_cast<double>(n) / max_exact) /
                             std::log(static_cast<double>(max_distance) / max_exact);
        ret += std::min<I64>(max_exact + static_cast<I64>(ratio * static_cast<double>(nb - max_exact)), nb - 1);
      }
      out[static_cast<std::size_t>(i * Tk + j)] = ret;
    }
  return from_vector(out, {Tq, Tk});
}

Tensor relative_position_bias(ParameterStore& store, const std::string& name, I64 heads, I64 Tq, I64 Tk,
                              bool bidirectional, I64 num_buckets, I64 max_distance) {
  const Tensor table   = store.get(name + "/rel_bias", {num_buckets, heads}, init::zeros());
  const Tensor buckets = relative_position_buckets(Tq, Tk, bidirectional, num_buckets, max_distance);
  return transpose(gather(table, buckets, 0), {2, 0, 1});  // [Tq, Tk, heads] -> [heads, Tq, Tk]
}

}  // namespace datamunge::nn
