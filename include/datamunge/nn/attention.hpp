#pragma once

/// @file attention.hpp
/// @brief Attention building blocks: scaled dot-product attention (plain and
///        fused, memory-efficient), masks, multi-head attention with a KV
///        cache, additive (Bahdanau) attention and positional encodings
///        (sinusoidal, learned, rotary, ALiBi, relative position buckets).
///
/// Layout follows tflow's `common/nn/attention.h` and `positional.h`:
/// queries [..., Tq, d], keys [..., Tk, d], values [..., Tk, dv], attention
/// weights [..., Tq, Tk] with the softmax over the last (key) axis, and
/// multi-head tensors [N, heads, T, dh].
///
/// Masks come in two kinds, both broadcastable to [..., Tq, Tk]:
/// - a *keep* mask (bool, or an integer tensor read as nonzero = keep): false
///   entries are excluded from the softmax;
/// - an *additive* float bias added to the scaled scores (tflow's convention:
///   0 where allowed, -1e9 where not; also ALiBi and relative-position biases).
/// A query row with no allowed key gets zero weights and a zero output
/// (instead of the NaN a softmax over only -inf would give).
///
/// Causal masking aligns the last query with the last key ("bottom right"):
/// query i may see key j when j <= i + (Tk - Tq). With Tq == Tk this is the
/// usual lower triangle; with a KV cache (Tq new queries after Tk - Tq cached
/// keys) it is what incremental decoding needs.

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "datamunge/nn/params.hpp"
#include "datamunge/nn/random.hpp"
#include "datamunge/nn/tensor.hpp"

namespace datamunge::nn {

// ---- masks ---------------------------------------------------------------------------------

/// Bool [Tq, Tk] keep mask: true where key j <= query i + (Tk - Tq).
Tensor causal_keep_mask(std::int64_t Tq, std::int64_t Tk);
/// Square form of causal_keep_mask.
Tensor causal_keep_mask(std::int64_t T);
/// tflow's CausalMask: additive [Tq, Tk] mask, 0 where allowed and `value`
/// (default -1e9) elsewhere.
Tensor causal_mask(std::int64_t Tq, std::int64_t Tk, DType dtype = DType::Float32, double value = -1e9);
/// Square form of causal_mask.
Tensor causal_mask(std::int64_t T, DType dtype = DType::Float32, double value = -1e9);
/// Bool keep mask [N, 1, 1, T] from per-example lengths [N]: position t is
/// kept when t < lengths[n]. Broadcasts against [N, heads, Tq, T].
Tensor padding_keep_mask(const Tensor& lengths, std::int64_t T);
/// Bool keep mask [N, 1, 1, T] from token ids [N, T]: ids equal to `pad_id` are masked out.
Tensor padding_keep_mask_from_ids(const Tensor& ids, std::int64_t pad_id = 0);
/// Converts a keep mask to an additive one: 0 where kept, `value` elsewhere.
Tensor additive_mask(const Tensor& keep, DType dtype = DType::Float32, double value = -1e9);

// ---- scaled dot-product attention -------------------------------------------------------------

/// Which kernel scaled_dot_product_attention runs.
enum class AttentionImpl {
  /// Fused when the weights are not requested, there is no attention dropout,
  /// the additive mask needs no gradient and Tq * Tk >= 1024; plain otherwise.
  Auto,
  /// softmax(q k^T * scale + mask) v with recorded ops; materialises the weights.
  Plain,
  /// Blocked online-softmax kernel that never holds the [Tq, Tk] matrix of a
  /// head; its backward recomputes the scores block by block.
  Fused,
};

/// Options of scaled_dot_product_attention.
struct AttentionOptions {
  /// Keep mask (bool / integer) or additive float bias, broadcastable to [..., Tq, Tk]; may be undefined.
  Tensor mask;
  /// Also mask keys after each query (see the file comment for the alignment).
  bool causal{false};
  /// Scores are q k^T * scale; 0 means 1 / sqrt(d).
  double scale{0.0};
  /// Dropout rate on the attention weights (only when > 0; forces the plain path).
  double dropout{0.0};
  /// Random stream for dropout (the default generator when null).
  Generator* generator{nullptr};
  /// Return the attention weights (forces the plain path).
  bool need_weights{false};
  AttentionImpl impl{AttentionImpl::Auto};
};

/// Attention output [..., Tq, dv] and, when requested, weights [..., Tq, Tk]
/// (before dropout). Same shape as tflow's `nn::Attended`.
struct Attended {
  Tensor output, weights;
};

/// softmax(q k^T * scale + mask) v. q [..., Tq, d], k [..., Tk, d], v [..., Tk, dv];
/// leading axes broadcast. Differentiable in q, k, v and a float mask, to
/// second order on the plain path (and on the fused path, whose create_graph
/// backward is built from recorded ops).
Attended scaled_dot_product_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                                      const AttentionOptions& options = {});
/// tflow's ScaledDotProductAttention(q, k, v, d, mask): plain path, additive
/// (or keep) mask, weights returned. The scale is 1 / sqrt(q.size(-1)).
Attended scaled_dot_product_attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& mask);

/// True when AttentionImpl::Auto would take the fused kernel for these shapes and options.
bool attention_would_fuse(const Shape& q_shape, const Shape& k_shape, const AttentionOptions& options);

// ---- KV cache ----------------------------------------------------------------------------------

/// Keys and values of the positions decoded so far, [N, heads, T, dh] each,
/// grown by `append` one step (or chunk) at a time. Without autograd history
/// the cache writes into a buffer with doubling capacity and hands out views
/// (no per-step copy of the whole cache); tensors that need gradients are
/// concatenated instead, so gradients flow through the cache.
class KvCache {
 public:
  KvCache() = default;
  /// Appends k_new, v_new [N, heads, t, dh] along the time axis and returns
  /// the full keys and values.
  std::pair<Tensor, Tensor> append(const Tensor& k_new, const Tensor& v_new);
  /// Keys / values so far (undefined before the first append).
  [[nodiscard]] Tensor keys() const;
  [[nodiscard]] Tensor values() const;
  /// Number of cached positions.
  [[nodiscard]] std::int64_t length() const noexcept { return length_; }
  /// Forgets everything.
  void reset();

 private:
  Tensor       k_buf_, v_buf_;  // [N, H, capacity, dh] (no-grad mode) or exact size (grad mode)
  std::int64_t length_{0};
  bool         buffered_{false};
};

// ---- multi-head attention ----------------------------------------------------------------------

/// Pair layout of rotary embeddings.
enum class RotaryLayout {
  /// Pairs (x_{2j}, x_{2j+1}) (tflow's ApplyRotary, the RoFormer paper, GPT-J).
  Interleaved,
  /// Pairs (x_j, x_{j + d/2}) (GPT-NeoX / LLaMA "rotate_half").
  HalfSplit,
};

/// Options of multi_head_attention.
struct MultiHeadOptions {
  /// Keep mask or additive mask broadcastable to [N, heads, Tq, Tk].
  Tensor mask;
  bool   causal{false};
  /// Key/value heads for grouped-query (1 = multi-query) attention; 0 means `heads`.
  std::int64_t kv_heads{0};
  /// Learn biases "<name>/bq", "/bk", "/bv", "/bo" (tflow's layer has none).
  bool bias{false};
  /// Initializer of the projections (default: U(+-sqrt(3 / fan_in)) as tflow's AddUniform).
  Initializer init{init::lecun_uniform()};
  /// Attention-weight dropout while `training`.
  double     dropout{0.0};
  bool       training{false};
  Generator* generator{nullptr};
  bool       need_weights{false};
  AttentionImpl impl{AttentionImpl::Auto};
  /// Apply rotary embeddings to queries and keys (positions continue from the cache length).
  bool         rotary{false};
  double       rotary_base{10000.0};
  RotaryLayout rotary_layout{RotaryLayout::Interleaved};
  /// Incremental decoding: the new keys/values are appended and the queries
  /// attend to everything cached (combine with `causal` for chunked prefill).
  KvCache* cache{nullptr};
};

/// tflow's MultiHeadAttention. x_q [N, Tq, d_model] gives the queries, x_kv
/// [N, Tk, d_kv] the keys and values (pass x_q again for self-attention).
/// Parameters "<name>/Wq" [d_model, d_model], "/Wk" and "/Wv" [d_kv, kv_heads dh],
/// "/Wo" [d_model, d_model]. Returns output [N, Tq, d_model] and, with
/// need_weights, weights [N, heads, Tq, Tk]. Throws std::invalid_argument when
/// d_model is 0 or `heads` does not divide it.
Attended multi_head_attention(ParameterStore& store, const std::string& name, const Tensor& x_q, const Tensor& x_kv,
                              std::int64_t heads, const MultiHeadOptions& options = {});
/// Separate key and value inputs (DETR adds positions to queries and keys only).
Attended multi_head_attention(ParameterStore& store, const std::string& name, const Tensor& x_q, const Tensor& x_k,
                              const Tensor& x_v, std::int64_t heads, const MultiHeadOptions& options = {});

/// [N, T, heads * dh] -> [N, heads, T, dh].
Tensor split_heads(const Tensor& x, std::int64_t heads);
/// [N, heads, T, dh] -> [N, T, heads * dh].
Tensor merge_heads(const Tensor& x);

/// tflow's AdditiveAttention (Bahdanau): one query per example. q [N, dq],
/// k [N, T, dk], v [N, T, dv] -> output [N, dv], weights [N, T];
/// score_t = v . tanh(q Wq + k_t Wk) + mask. Parameters "<name>/Wq" [dq, units],
/// "/Wk" [dk, units], "/v" [units, 1] (Glorot uniform). `mask` is additive or keep, [N, T].
Attended additive_attention(ParameterStore& store, const std::string& name, const Tensor& q, const Tensor& k,
                            const Tensor& v, std::int64_t units, const Tensor& mask = Tensor());

// ---- positional encodings -------------------------------------------------------------------------

/// tflow's SinusoidalTable [T, d]: PE[t, 2i] = sin(t / base^(2i/d)),
/// PE[t, 2i+1] = cos(t / base^(2i/d)), positions offset..offset+T-1.
/// Throws std::invalid_argument unless d is even and >= 2, T >= 0, offset >= 0 and base > 0.
Tensor sinusoidal_encoding(std::int64_t T, std::int64_t d, double base = 10000.0, DType dtype = DType::Float32,
                           std::int64_t offset = 0);
/// tflow's LearnedEncoding: trainable table "<name>/pos" [max_len, d] drawn N(0, stddev).
Tensor learned_encoding(ParameterStore& store, const std::string& name, std::int64_t max_len, std::int64_t d,
                        double stddev = 0.02);

/// cos and sin tables [T, d] for rotary embeddings (each angle repeated for
/// both members of its pair, in the given layout), positions offset..offset+T-1.
std::pair<Tensor, Tensor> rotary_tables(std::int64_t T, std::int64_t d, double base = 10000.0,
                                        DType dtype = DType::Float32, std::int64_t offset = 0,
                                        RotaryLayout layout = RotaryLayout::Interleaved);
/// tflow's ApplyRotary on x [..., T, d] (d even): pair j at position
/// offset + t is rotated by (offset + t) * base^(-2j/d). Differentiable.
/// Throws std::invalid_argument for odd or zero d, a negative offset or base <= 0.
Tensor apply_rotary(const Tensor& x, std::int64_t offset = 0, double base = 10000.0,
                    RotaryLayout layout = RotaryLayout::Interleaved);
/// Rotary embedding at explicit integer positions [T] (or [N, T] broadcast over heads when x is [N, H, T, d]).
Tensor apply_rotary(const Tensor& x, const Tensor& positions, double base = 10000.0,
                    RotaryLayout layout = RotaryLayout::Interleaved);

/// ALiBi slopes (Press et al. 2022, tflow's AlibiSlopes). For a power of two
/// H, head h (1-based) gets 2^(-8 h / H). Otherwise, with P the largest power
/// of two below H, the first P heads get 2^(-8 h / P) and the remaining H - P
/// heads 2^(-4 h / P) for h = 1, 3, 5, ... (the paper's get_slopes; e.g.
/// 3 heads -> [2^-4, 2^-8, 2^-2]). Throws std::invalid_argument for heads < 1.
std::vector<double> alibi_slopes(std::int64_t heads);
/// tflow's AlibiBias [heads, T, T]: -slope_h |m - n|, and -1e9 for n > m when `causal`.
Tensor alibi_bias(std::int64_t heads, std::int64_t T, bool causal = true, DType dtype = DType::Float32);
/// Rectangular form [heads, Tq, Tk] for decoding: query i sits at position i + (Tk - Tq).
Tensor alibi_bias(std::int64_t heads, std::int64_t Tq, std::int64_t Tk, bool causal, DType dtype = DType::Float32);

/// T5's relative position buckets [Tq, Tk] (int64) of n - m (key minus query,
/// queries at positions i + (Tk - Tq)): exact for small distances, then
/// logarithmic up to `max_distance`; separate buckets for each sign when bidirectional.
/// Throws std::invalid_argument unless num_buckets >= 4 (even when bidirectional)
/// and max_distance exceeds the exact range (num_buckets / 4 bidirectional, / 2 otherwise).
Tensor relative_position_buckets(std::int64_t Tq, std::int64_t Tk, bool bidirectional = true,
                                 std::int64_t num_buckets = 32, std::int64_t max_distance = 128);
/// Learned relative position bias [heads, Tq, Tk] from the table
/// "<name>/rel_bias" [num_buckets, heads] (zero initialised), to add to the scores.
Tensor relative_position_bias(ParameterStore& store, const std::string& name, std::int64_t heads, std::int64_t Tq,
                              std::int64_t Tk, bool bidirectional = true, std::int64_t num_buckets = 32,
                              std::int64_t max_distance = 128);

}  // namespace datamunge::nn
