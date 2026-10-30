// CTC loss / greedy decoding and linear-chain CRF (log-likelihood, Viterbi).
//
// ctc_loss is one recorded primitive whose forward pass also produces the
// gradient (TF CTCLoss). The CRF log-normaliser is a composition of nn ops
// (a log-sum-exp recursion), so it is differentiable to any order.

#include "datamunge/nn/sequence.hpp"

#include <limits>

#include "datamunge/nn/parallel.hpp"
#include "detail.hpp"

namespace datamunge::nn {

using detail::as_dtype;
using detail::dispatch_float;
using detail::nograd;

namespace {

using I64   = std::int64_t;
using Grads = std::vector<Tensor>;

constexpr double kNegInf = -std::numeric_limits<double>::infinity();

double log_add(double a, double b) {
  if (a == kNegInf) return b;
  if (b == kNegInf) return a;
  return a > b ? a + std::log1p(std::exp(b - a)) : b + std::log1p(std::exp(a - b));
}

std::vector<I64> int_values(const Tensor& t, I64 expected, const char* what, const char* name) {
  if (!t.defined()) throw std::invalid_argument(std::string(what) + ": " + name + " is undefined");
  if (is_floating(t.dtype())) throw std::invalid_argument(std::string(what) + ": " + name + " must be an int tensor");
  if (t.dim() != 1 || t.shape()[0] != expected)
    throw std::invalid_argument(std::string(what) + ": " + name + " must have shape [" + std::to_string(expected) +
                                "], got " + shape_str(t.shape()));
  return t.to_vector<I64>();
}

/// Layout of a CTC logits tensor.
struct CtcGeometry {
  I64  T{0}, B{0}, C{0};
  bool time_major{true};
  [[nodiscard]] I64 at(I64 t, I64 b) const { return (time_major ? t * B + b : b * T + t) * C; }
};

CtcGeometry ctc_geometry(const Tensor& logits, bool time_major, const char* what) {
  if (logits.dim() != 3)
    throw std::invalid_argument(std::string(what) + ": logits must be rank 3, got " + shape_str(logits.shape()));
  CtcGeometry g;
  g.time_major = time_major;
  g.T          = time_major ? logits.shape()[0] : logits.shape()[1];
  g.B          = time_major ? logits.shape()[1] : logits.shape()[0];
  g.C          = logits.shape()[2];
  if (g.C < 1) throw std::invalid_argument(std::string(what) + ": no classes");
  return g;
}

I64 resolve_blank(I64 blank, I64 C, const char* what) {
  const I64 b = blank < 0 ? blank + C : blank;
  if (b < 0 || b >= C) throw std::invalid_argument(std::string(what) + ": blank index out of range");
  return b;
}

/// Forward-backward for one sequence. Writes d loss / d logits for frames
/// [0, Tb) into `grad` (layout g) and returns the loss. `lp` is scratch.
template <typename T>
double ctc_one(const CtcGeometry& g, I64 b, const T* x, T* grad, I64 Tb, const std::vector<I64>& label, I64 blank,
               bool ignore_infeasible) {
  const I64 C = g.C, L = static_cast<I64>(label.size()), S = 2 * L + 1;
  I64       required = L;
  for (I64 i = 1; i < L; ++i) required += label[i] == label[i - 1];
  if (required > Tb) {
    if (ignore_infeasible) return 0.0;
    throw std::invalid_argument("ctc_loss: not enough time for target transition sequence (required: " +
                                std::to_string(required) + ", available: " + std::to_string(Tb) + ")");
  }
  if (Tb == 0) return 0.0;  // empty label, no frames: probability 1
  // log-softmax per valid frame
  std::vector<double> lp(static_cast<std::size_t>(Tb * C));
  for (I64 t = 0; t < Tb; ++t) {
    const T* row = x + g.at(t, b);
    double   m   = kNegInf;
    for (I64 c = 0; c < C; ++c) m = std::max(m, static_cast<double>(row[c]));
    double z = 0;
    for (I64 c = 0; c < C; ++c) z += std::exp(static_cast<double>(row[c]) - m);
    const double lz = m + std::log(z);
    for (I64 c = 0; c < C; ++c) lp[t * C + c] = static_cast<double>(row[c]) - lz;
  }
  auto ext = [&](I64 s) { return (s & 1) ? label[s >> 1] : blank; };
  // skip transition s-2 -> s allowed for a label differing from the previous label
  auto can_skip = [&](I64 s) { return s >= 2 && (s & 1) && ext(s) != ext(s - 2); };
  std::vector<double> alpha(static_cast<std::size_t>(Tb * S), kNegInf), beta(alpha.size(), kNegInf);
  alpha[0] = lp[blank];
  if (S > 1) alpha[1] = lp[ext(1)];
  for (I64 t = 1; t < Tb; ++t) {
    const double* prev = &alpha[(t - 1) * S];
    double*       cur  = &alpha[t * S];
    for (I64 s = 0; s < S; ++s) {
      double v = prev[s];
      if (s >= 1) v = log_add(v, prev[s - 1]);
      if (can_skip(s)) v = log_add(v, prev[s - 2]);
      cur[s] = v == kNegInf ? kNegInf : v + lp[t * C + ext(s)];
    }
  }
  // beta[t][s]: log-probability of frames t..Tb-1 given state s at frame t (incl. frame t's emission)
  beta[(Tb - 1) * S + S - 1] = lp[(Tb - 1) * C + blank];
  if (S > 1) beta[(Tb - 1) * S + S - 2] = lp[(Tb - 1) * C + ext(S - 2)];
  for (I64 t = Tb - 2; t >= 0; --t) {
    const double* next = &beta[(t + 1) * S];
    double*       cur  = &beta[t * S];
    for (I64 s = 0; s < S; ++s) {
      double v = next[s];
      if (s + 1 < S) v = log_add(v, next[s + 1]);
      if (s + 2 < S && can_skip(s + 2)) v = log_add(v, next[s + 2]);
      cur[s] = v == kNegInf ? kNegInf : v + lp[t * C + ext(s)];
    }
  }
  const double* last  = &alpha[(Tb - 1) * S];
  const double  log_p = S > 1 ? log_add(last[S - 1], last[S - 2]) : last[S - 1];
  // gradient: softmax - posterior occupancy of each class
  std::vector<double> occ(static_cast<std::size_t>(C));
  for (I64 t = 0; t < Tb; ++t) {
    std::fill(occ.begin(), occ.end(), kNegInf);
    for (I64 s = 0; s < S; ++s) {
      const double a = alpha[t * S + s], be = beta[t * S + s];
      if (a == kNegInf || be == kNegInf) continue;
      const I64 c = ext(s);
      occ[c]      = log_add(occ[c], a + be - lp[t * C + c]);
    }
    T* gr = grad + g.at(t, b);
    for (I64 c = 0; c < C; ++c) {
      const double post = occ[c] == kNegInf ? 0.0 : std::exp(occ[c] - log_p);
      gr[c]             = static_cast<T>(std::exp(lp[t * C + c]) - post);
    }
  }
  return -log_p;
}

}  // namespace

// ---- CTC --------------------------------------------------------------------------------------

CtcLossResult ctc_loss_with_gradient(const Tensor& logits, const Tensor& labels, const Tensor& label_lengths,
                                     const Tensor& logit_lengths, const CtcOptions& options) {
  const char*       what = "ctc_loss";
  const CtcGeometry g    = ctc_geometry(logits, options.time_major, what);
  const I64         blank = resolve_blank(options.blank, g.C, what);
  if (labels.dim() != 2 || labels.shape()[0] != g.B || is_floating(labels.dtype()))
    throw std::invalid_argument("ctc_loss: labels must be an int tensor [B, L_max]");
  const auto lab_len = int_values(label_lengths, g.B, what, "label_lengths");
  const auto log_len = int_values(logit_lengths, g.B, what, "logit_lengths");
  const I64  Lmax    = labels.shape()[1];
  const auto lab     = labels.to_vector<I64>();
  std::vector<std::vector<I64>> seqs(static_cast<std::size_t>(g.B));
  for (I64 b = 0; b < g.B; ++b) {
    if (lab_len[b] < 0 || lab_len[b] > Lmax) throw std::invalid_argument("ctc_loss: label length out of range");
    if (log_len[b] < 0 || log_len[b] > g.T) throw std::invalid_argument("ctc_loss: logit length out of range");
    for (I64 i = 0; i < lab_len[b]; ++i) {
      const I64 c = lab[b * Lmax + i];
      if (c < 0 || c >= g.C || c == blank)
        throw std::invalid_argument("ctc_loss: label " + std::to_string(c) + " is out of range or the blank");
      seqs[b].push_back(c);
    }
  }
  Tensor loss = empty({g.B}, logits.dtype());
  Tensor grad = zeros(logits.shape(), logits.dtype());
  {
    NoGradGuard  guard;
    const Tensor xk = nograd(logits).contiguous();
    dispatch_float(logits.dtype(), what, [&](auto tag) {
      using T    = typename decltype(tag)::type;
      const T* x = xk.data_ptr<T>();
      T*       gr = grad.data_ptr<T>();
      T*       lo = loss.data_ptr<T>();
      parallel_for(0, g.B, 1, [&](I64 b0, I64 b1) {
        for (I64 b = b0; b < b1; ++b)
          lo[b] = static_cast<T>(
              ctc_one<T>(g, b, x, gr, log_len[b], seqs[b], blank, options.ignore_longer_outputs_than_inputs));
      });
    });
  }
  const SavedTensor saved(grad);
  const bool        time_major = options.time_major;
  record("CTCLoss", {logits}, {loss}, [saved, time_major](const Grads& gr, const Grads&) {
    // First order only: the saved gradient is a constant (TF does the same).
    const Tensor& G = saved.get();
    const Tensor  w = time_major ? reshape(gr[0], {1, -1, 1}) : reshape(gr[0], {-1, 1, 1});
    return Grads{mul(G, w)};
  });
  return {loss, grad};
}

Tensor ctc_loss(const Tensor& logits, const Tensor& labels, const Tensor& label_lengths, const Tensor& logit_lengths,
                const CtcOptions& options) {
  return ctc_loss_with_gradient(logits, labels, label_lengths, logit_lengths, options).loss;
}

std::vector<std::int64_t> ctc_collapse(const std::vector<std::int64_t>& path, std::int64_t blank,
                                       bool merge_repeated) {
  std::vector<std::int64_t> out;
  std::int64_t              previous = -1;
  bool                      first    = true;
  for (auto c : path) {
    if (c != blank && !(merge_repeated && !first && c == previous)) out.push_back(c);
    previous = c;
    first    = false;
  }
  return out;
}

CtcDecoded ctc_greedy_decode(const Tensor& logits, const Tensor& logit_lengths, const CtcOptions& options,
                             bool merge_repeated) {
  const char*       what  = "ctc_greedy_decode";
  const CtcGeometry g     = ctc_geometry(logits, options.time_major, what);
  const I64         blank = resolve_blank(options.blank, g.C, what);
  std::vector<I64>  lengths(static_cast<std::size_t>(g.B), g.T);
  if (logit_lengths.defined()) lengths = int_values(logit_lengths, g.B, what, "logit_lengths");
  CtcDecoded out;
  out.sequences.resize(static_cast<std::size_t>(g.B));
  std::vector<double> neg(static_cast<std::size_t>(g.B), 0.0), lp(static_cast<std::size_t>(g.B), 0.0);
  {
    NoGradGuard  guard;
    const Tensor xk = nograd(logits).contiguous();
    dispatch_float(logits.dtype(), what, [&](auto tag) {
      using T    = typename decltype(tag)::type;
      const T* x = xk.data_ptr<T>();
      for (I64 b = 0; b < g.B; ++b) {
        if (lengths[b] < 0 || lengths[b] > g.T) throw std::invalid_argument("ctc_greedy_decode: length out of range");
        std::vector<I64> path;
        for (I64 t = 0; t < lengths[b]; ++t) {
          const T* row = x + g.at(t, b);
          I64      best = 0;
          for (I64 c = 1; c < g.C; ++c)
            if (row[c] > row[best]) best = c;
          double z = 0;
          for (I64 c = 0; c < g.C; ++c) z += std::exp(static_cast<double>(row[c]) - static_cast<double>(row[best]));
          neg[b] -= static_cast<double>(row[best]);
          lp[b] -= std::log(z);
          path.push_back(best);
        }
        out.sequences[b] = ctc_collapse(path, blank, merge_repeated);
      }
    });
  }
  std::size_t width = 0;
  for (const auto& s : out.sequences) width = std::max(width, s.size());
  out.dense = full({g.B, static_cast<I64>(width)}, -1.0, DType::Int64);
  auto* d   = out.dense.data_ptr<std::int64_t>();
  for (I64 b = 0; b < g.B; ++b)
    for (std::size_t i = 0; i < out.sequences[b].size(); ++i) d[b * static_cast<I64>(width) + static_cast<I64>(i)] = out.sequences[b][i];
  out.neg_sum_logits = cast(reshape(from_vector(neg), {g.B, 1}), logits.dtype());
  out.path_log_prob  = cast(from_vector(lp), logits.dtype());
  return out;
}

// ---- linear-chain CRF ---------------------------------------------------------------------------

namespace {

struct CrfShape {
  I64 B{0}, T{0}, K{0};
  std::vector<I64> lengths;
};

CrfShape crf_shape(const Tensor& potentials, const Tensor& lengths, const Tensor& transitions, const Tensor& start,
                   const Tensor& end, const char* what) {
  for(const Tensor* value:{&potentials,&lengths,&transitions,&start,&end})
    if(value->defined())detail::require_cpu(*value,what);
  if (potentials.dim() != 3)
    throw std::invalid_argument(std::string(what) + ": potentials must be [B, T, K], got " +
                                shape_str(potentials.shape()));
  if (!is_floating(potentials.dtype())) throw std::invalid_argument(std::string(what) + ": float potentials expected");
  CrfShape s;
  s.B = potentials.shape()[0];
  s.T = potentials.shape()[1];
  s.K = potentials.shape()[2];
  if (transitions.shape() != Shape{s.K, s.K})
    throw std::invalid_argument(std::string(what) + ": transitions must be [K, K]");
  for (const Tensor* v : {&start, &end})
    if (v->defined() && v->shape() != Shape{s.K})
      throw std::invalid_argument(std::string(what) + ": start/end must be [K]");
  s.lengths = int_values(lengths, s.B, what, "lengths");
  for (auto n : s.lengths)
    if (n < 0 || n > s.T) throw std::invalid_argument(std::string(what) + ": length out of range");
  return s;
}

/// [B, T] float mask of valid positions.
Tensor position_mask(const CrfShape& s, DType dtype) {
  std::vector<double> m(static_cast<std::size_t>(s.B * s.T), 0.0);
  for (I64 b = 0; b < s.B; ++b)
    for (I64 t = 0; t < s.lengths[b]; ++t) m[b * s.T + t] = 1.0;
  return cast(from_vector(m, {s.B, s.T}), dtype);
}

/// [B, 1] bool: length > t.
Tensor valid_at(const CrfShape& s, I64 t) {
  Tensor out = empty({s.B, 1}, DType::Bool);
  bool*  p   = out.data_ptr<bool>();
  for (I64 b = 0; b < s.B; ++b) p[b] = s.lengths[b] > t;
  return out;
}

}  // namespace

Tensor crf_sequence_score(const Tensor& potentials, const Tensor& tags, const Tensor& lengths,
                          const Tensor& transitions, const Tensor& start, const Tensor& end) {
  const char*    what = "crf_sequence_score";
  const CrfShape s    = crf_shape(potentials, lengths, transitions, start, end, what);
  if (tags.shape() != Shape{s.B, s.T} || is_floating(tags.dtype()))
    throw std::invalid_argument("crf_sequence_score: tags must be an int tensor [B, T]");
  const DType dt   = potentials.dtype();
  const Tensor A   = as_dtype(transitions, dt);
  // Tags outside the valid positions become -1, which one_hot maps to zeros.
  auto tv = tags.to_vector<I64>();
  std::vector<I64> last(static_cast<std::size_t>(s.B), -1);
  for (I64 b = 0; b < s.B; ++b) {
    for (I64 t = 0; t < s.T; ++t) {
      I64& v = tv[b * s.T + t];
      if (t >= s.lengths[b]) v = -1;
      else if (v < 0 || v >= s.K) throw std::invalid_argument("crf_sequence_score: tag out of range");
    }
    if (s.lengths[b] > 0) last[b] = tv[b * s.T + s.lengths[b] - 1];
  }
  const Tensor oh    = one_hot(from_vector(tv, {s.B, s.T}), s.K, 1.0, 0.0, dt);  // [B, T, K]
  Tensor       score = sum(mul(oh, potentials), {1, 2});
  if (s.T > 1) {
    const Tensor prev = reshape(slice_axis(oh, 1, 0, s.T - 1), {-1, s.K});
    const Tensor cur  = reshape(slice_axis(oh, 1, 1, s.T), {-1, s.K});
    const Tensor pair = sum(mul(matmul(prev, A), cur), {1});  // zero unless both positions are valid
    score             = add(score, sum(reshape(pair, {s.B, s.T - 1}), {1}));
  }
  if (start.defined() && s.T > 0) score = add(score, sum(mul(select(oh, 1, 0), as_dtype(start, dt)), {1}));
  if (end.defined()) score = add(score, sum(mul(one_hot(from_vector(last), s.K, 1.0, 0.0, dt), as_dtype(end, dt)), {1}));
  return score;
}

Tensor crf_forward(const Tensor& potentials, const Tensor& lengths, const Tensor& transitions, const Tensor& start) {
  const CrfShape s = crf_shape(potentials, lengths, transitions, start, Tensor(), "crf_forward");
  if (s.T == 0) return zeros({s.B, 0, s.K}, potentials.dtype());
  const Tensor A = expand_dims(as_dtype(transitions, potentials.dtype()), 0);  // [1, K, K]
  const auto   u = unstack(potentials, 1);
  // Empty sequences keep the initial state (start scores, or zeros), as in tflow.
  Tensor init = start.defined() ? add(zeros_like(u[0]), as_dtype(start, potentials.dtype())) : zeros_like(u[0]);
  Tensor alpha = where(valid_at(s, 0), start.defined() ? add(u[0], as_dtype(start, potentials.dtype())) : u[0], init);
  std::vector<Tensor> trace{alpha};
  for (I64 t = 1; t < s.T; ++t) {
    const Tensor cand = add(u[static_cast<std::size_t>(t)], logsumexp(add(expand_dims(alpha, 2), A), {1}));
    alpha             = where(valid_at(s, t), cand, alpha);
    trace.push_back(alpha);
  }
  return stack(trace, 1);
}

Tensor crf_log_norm(const Tensor& potentials, const Tensor& lengths, const Tensor& transitions, const Tensor& start,
                    const Tensor& end) {
  const CrfShape s = crf_shape(potentials, lengths, transitions, start, end, "crf_log_norm");
  if (s.T == 0) return zeros({s.B}, potentials.dtype());
  const Tensor alphas = crf_forward(potentials, lengths, transitions, start);
  Tensor       last   = select(alphas, 1, s.T - 1);
  if (end.defined()) last = add(last, as_dtype(end, potentials.dtype()));
  const Tensor z = logsumexp(last, {1});
  return where(reshape(valid_at(s, 0), {s.B}), z, scalar_like(0.0, z));
}

Tensor crf_log_likelihood(const Tensor& potentials, const Tensor& tags, const Tensor& lengths,
                          const Tensor& transitions, const Tensor& start, const Tensor& end) {
  return sub(crf_sequence_score(potentials, tags, lengths, transitions, start, end),
             crf_log_norm(potentials, lengths, transitions, start, end));
}

CrfDecoded crf_decode(const Tensor& potentials, const Tensor& lengths, const Tensor& transitions, const Tensor& start,
                      const Tensor& end, std::int32_t pad_tag) {
  const CrfShape s  = crf_shape(potentials, lengths, transitions, start, end, "crf_decode");
  const auto     U  = potentials.to_vector<double>();
  const auto     A  = transitions.to_vector<double>();
  const auto     st = start.defined() ? start.to_vector<double>() : std::vector<double>(s.K, 0.0);
  const auto     en = end.defined() ? end.to_vector<double>() : std::vector<double>(s.K, 0.0);
  CrfDecoded out;
  out.tags   = full({s.B, s.T}, pad_tag, DType::Int32);
  out.scores = zeros({s.B}, DType::Float64);
  auto* tags = out.tags.data_ptr<std::int32_t>();
  auto* sc   = out.scores.data_ptr<double>();
  const I64 K = s.K;
  parallel_for(0, s.B, 1, [&](I64 b0, I64 b1) {
    std::vector<double> prev(K), next(K);
    std::vector<I64>    back;
    for (I64 b = b0; b < b1; ++b) {
      const I64 n = s.lengths[b];
      if (n == 0) continue;
      back.assign(static_cast<std::size_t>(n * K), 0);
      const double* u = U.data() + b * s.T * K;
      for (I64 k = 0; k < K; ++k) prev[k] = st[k] + u[k];
      for (I64 t = 1; t < n; ++t) {
        for (I64 to = 0; to < K; ++to) {
          double best = kNegInf;
          I64    arg  = 0;
          for (I64 from = 0; from < K; ++from) {
            const double v = prev[from] + A[from * K + to];
            if (v > best) best = v, arg = from;
          }
          next[to]           = best + u[t * K + to];
          back[t * K + to]   = arg;
        }
        prev.swap(next);
      }
      double best = kNegInf;
      I64    tag  = 0;
      for (I64 k = 0; k < K; ++k)
        if (prev[k] + en[k] > best) best = prev[k] + en[k], tag = k;
      sc[b] = best;
      for (I64 t = n - 1; t >= 0; --t) {
        tags[b * s.T + t] = static_cast<std::int32_t>(tag);
        if (t > 0) tag = back[t * K + tag];
      }
    }
  });
  return out;
}

}  // namespace datamunge::nn
