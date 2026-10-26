// Compact original post-norm Transformer encoder-decoder and its masked NMT
// likelihood, partials and own-token greedy decoding: port of tflow's
// common/nn/transformer_nmt.h and common/ml/transformer_nmt.h.
//
// T 6 tokens per side, D 8, 2 heads, FF 16 (ReLU), byte vocabulary 256; a
// shared embedding "embedding/W" (N(0, .1)) scaled by sqrt(D) plus the
// sinusoidal table; encoder self-attention, causal decoder self-attention and
// cross-attention are tflow's bias-free MultiHeadAttention (attn::, i.e.
// nn::multi_head_attention with Wq/Wk/Wv/Wo U(+-sqrt(3/D))) with
// attn::AttentionAllowed masks from the source/target lengths; five post-norm
// LNs whose outputs are multiplied by the length mask; head "head/W|b".
// 5760 parameters in 33 tensors with tflow's names.
#pragma once

#include <datamunge/nn/nn.hpp>

#include <cmath>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "common/attention.hpp"
#include "common/checks.hpp"

namespace nmt {

namespace nn = datamunge::nn;
using nn::Tensor;
using Outputs = std::map<std::string, Tensor>;

/// tflow's ml::SeqBatch (int32 [N, 6] tokens, [N] lengths and ids).
struct SeqBatch {
  Tensor input, target, teacher, input_lengths, output_lengths, source_ids;
};

inline SeqBatch SeqRows(const SeqBatch& b, const std::vector<std::int64_t>& r) {
  return {nn::take_rows(b.input, r),         nn::take_rows(b.target, r),         nn::take_rows(b.teacher, r),
          nn::take_rows(b.input_lengths, r), nn::take_rows(b.output_lengths, r), nn::take_rows(b.source_ids, r)};
}

inline nn::NamedTensors Named(const SeqBatch& b) {
  return {{"input", b.input},
          {"teacher", b.teacher},
          {"target", b.target},
          {"input_lengths", b.input_lengths},
          {"output_lengths", b.output_lengths},
          {"source_ids", b.source_ids}};
}

struct TransformerNmtPath {
  Tensor logits, memory, hidden, encoder_attention, decoder_attention, cross_attention, source_embedded, target_embedded;
};

/// Float mask [N, T, 1]: 1 where t < lengths[n].
inline Tensor LengthMask(const Tensor& lengths, std::int64_t T) {
  const auto         l = nn::cast(lengths, nn::DType::Int64).to_vector<std::int64_t>();
  std::vector<float> m;
  for (auto n : l)
    for (std::int64_t t = 0; t < T; ++t) m.push_back(t < n ? 1.f : 0.f);
  return nn::from_vector(m, {static_cast<std::int64_t>(l.size()), T, 1});
}

/// `source` / `teacher` are float one-hot [N, 6, 256].
inline TransformerNmtPath TransformerNmt(nn::ParameterStore& store, const Tensor& source, const Tensor& teacher,
                                         const Tensor& source_lengths, const Tensor& target_lengths) {
  constexpr std::int64_t T = 6, D = 8, H = 2, V = 256, F = 16;
  const Tensor sm = LengthMask(source_lengths, T), tm = LengthMask(target_lengths, T);
  const Tensor embedding  = store.get("embedding/W", {V, D}, nn::init::normal(.1));
  const Tensor sinusoidal = nn::sinusoidal_encoding(T, D);
  auto embed = [&](const Tensor& input, const Tensor& mask) {
    return nn::mul(nn::add(nn::mul(nn::matmul(input, embedding), std::sqrt(float(D))), sinusoidal), mask);
  };
  const Tensor sx = embed(source, sm), tx = embed(teacher, tm);
  auto norm = [&](const Tensor& x, const std::string& name, const Tensor& mask) {
    return nn::mul(nn::layer_norm(store, name, x, 1e-5), mask);
  };
  auto ff = [&](const Tensor& x, const std::string& name) {
    const double limit = std::sqrt(6.f / (D + F));
    const Tensor w1 = store.get(name + "/W1", {D, F}, nn::init::uniform(limit)), b1 = store.get(name + "/b1", {F}, nn::init::zeros());
    const Tensor w2 = store.get(name + "/W2", {F, D}, nn::init::uniform(limit)), b2 = store.get(name + "/b2", {D}, nn::init::zeros());
    return nn::bias_add(nn::matmul(nn::relu(nn::bias_add(nn::matmul(x, w1), b1)), w2), b2);
  };
  auto mask = [&](const Tensor& kl, const Tensor& ql, bool causal) {
    return nn::expand_dims(attn::AttentionAllowed(kl, ql, T, T, causal), 1);
  };
  const auto enc = attn::MultiHeadAttention(store, "encoder/self", sx, sx, D, H, mask(source_lengths, source_lengths, false));
  Tensor     memory = norm(nn::add(sx, enc.output), "encoder/ln1", sm);
  memory            = norm(nn::add(memory, ff(memory, "encoder/ff")), "encoder/ln2", sm);
  const auto dec = attn::MultiHeadAttention(store, "decoder/self", tx, tx, D, H, mask(target_lengths, target_lengths, true));
  Tensor     hidden = norm(nn::add(tx, dec.output), "decoder/ln1", tm);
  const auto cross =
      attn::MultiHeadAttention(store, "decoder/cross", hidden, memory, D, H, mask(source_lengths, target_lengths, false));
  hidden = norm(nn::add(hidden, cross.output), "decoder/ln2", tm);
  hidden = norm(nn::add(hidden, ff(hidden, "decoder/ff")), "decoder/ln3", tm);
  const Tensor hw     = store.get("head/W", {D, V}, nn::init::uniform(std::sqrt(6.f / (D + V))));
  const Tensor hb     = store.get("head/b", {V}, nn::init::zeros());
  const Tensor logits = nn::mul(nn::bias_add(nn::matmul(hidden, hw), hb), tm);
  return {logits, memory, hidden, enc.weights, dec.weights, cross.weights, sx, tx};
}

/// tflow's ml::TransformerNmtModel: token CE averaged over valid target
/// positions, Adam .003, partials of the loss w.r.t. the source / teacher
/// one-hots and the encoder memory, and the causal prefix gradient
/// d sum(logits[:, :2, :7]) / d teacher one-hot.
class TransformerNmtModel {
 public:
  nn::ParameterStore             store{46003};
  std::unique_ptr<nn::Optimizer> optimizer;

  explicit TransformerNmtModel(bool training = true) {
    {
      nn::NoGradGuard guard;
      const Tensor    ids = nn::zeros({1, 6}, nn::DType::Int32), len = nn::full({1}, 6, nn::DType::Int32);
      TransformerNmt(store, nn::one_hot(ids, 256), nn::one_hot(ids, 256), len, len);
    }
    if (training) Train();
  }
  void Train() {
    nn::OptimizerConfig c;
    c.kind    = nn::OptimizerKind::Adam;
    optimizer = std::make_unique<nn::Optimizer>(store, .003, c);
  }

  static Tensor Loss(const TransformerNmtPath& p, const SeqBatch& b) {
    const Tensor mask = nn::squeeze(LengthMask(b.output_lengths, 6), {2});
    const Tensor ce   = nn::neg(nn::sum(nn::mul(nn::log_softmax(p.logits, -1), nn::one_hot(b.target, 256)), {2}));
    return nn::div(nn::sum(nn::mul(ce, mask)), nn::maximum(nn::sum(mask), nn::scalar_like(1.0, mask)));
  }

  TransformerNmtPath Forward(const SeqBatch& b) {
    return TransformerNmt(store, nn::one_hot(b.input, 256), nn::one_hot(b.teacher, 256), b.input_lengths, b.output_lengths);
  }

  Outputs Evaluate(const SeqBatch& b) {
    Tensor source = nn::one_hot(b.input, 256), teacher = nn::one_hot(b.teacher, 256);
    source.requires_grad_(true);
    teacher.requires_grad_(true);
    const auto      p      = TransformerNmt(store, source, teacher, b.input_lengths, b.output_lengths);
    const Tensor    loss   = Loss(p, b);
    const Tensor    prefix = nn::sum(nn::slice(p.logits, {0, 0, 0}, {-1, 2, 7}));
    nn::GradOptions go;
    go.retain_graph = true;
    go.allow_unused = true;
    // nn::grad differentiates with respect to intermediate tensors (the
    // encoder memory) as TF's AddSymbolicGradients does.
    const auto g  = nn::grad(std::vector<Tensor>{loss}, {source, teacher, p.memory}, {}, go);
    const auto pg = nn::grad(std::vector<Tensor>{prefix}, {teacher}, {}, go);
    auto def = [](const Tensor& x, const Tensor& like) { return x.defined() ? x.detach() : nn::zeros_like(like.detach()); };
    return {{"loss", loss.detach()},
            {"logits", p.logits.detach()},
            {"memory", p.memory.detach()},
            {"hidden", p.hidden.detach()},
            {"encoder_attention", p.encoder_attention.detach()},
            {"decoder_attention", p.decoder_attention.detach()},
            {"cross_attention", p.cross_attention.detach()},
            {"source_embedded", p.source_embedded.detach()},
            {"target_embedded", p.target_embedded.detach()},
            {"prefix_gradient", def(pg[0], teacher)},
            {"partial0", def(g[0], source)},
            {"partial1", def(g[1], teacher)},
            {"partial2", def(g[2], p.memory)}};
  }

  float LossValue(const SeqBatch& b) {
    nn::NoGradGuard guard;
    return Loss(Forward(b), b).item<float>();
  }

  std::vector<Tensor> RawGradients(const SeqBatch& b) {
    nn::GradOptions go;
    go.allow_unused = true;
    auto       g    = nn::grad(Loss(Forward(b), b), store.parameters(), go);
    const auto ps   = store.parameters();
    for (std::size_t i = 0; i < g.size(); ++i) g[i] = g[i].defined() ? g[i].detach() : nn::zeros_like(ps[i]);
    return g;
  }

  nn::NamedTensors Arrays(const SeqBatch* b = nullptr) {
    nn::NamedTensors out;
    if (b) {
      const auto g = RawGradients(*b);
      for (std::size_t i = 0; i < g.size(); ++i) out.emplace_back(store.names()[i], g[i]);
    } else {
      for (const auto& name : store.names()) out.emplace_back(name, store.at(name).detach().clone());
    }
    return out;
  }

  void Update(const SeqBatch& b) { optimizer->step(RawGradients(b)); }

  /// Own-previous-token greedy decoding from BOS (1): step t feeds the tokens
  /// so far with output length t + 1 and takes the argmax at position t; a row
  /// stops after EOS (2). `trace` receives the cross-attention row of each step.
  Tensor Greedy(const SeqBatch& b, Tensor* trace = nullptr) {
    nn::NoGradGuard           guard;
    const std::int64_t        n = b.input.size(0);
    std::vector<std::int32_t> tokens(static_cast<std::size_t>(n * 6), 0), generated(static_cast<std::size_t>(n * 6), 0);
    for (std::int64_t j = 0; j < n; ++j) tokens[static_cast<std::size_t>(j * 6)] = 1;
    std::vector<bool>  done(static_cast<std::size_t>(n), false);
    std::vector<float> tr(static_cast<std::size_t>(n * 2 * 6 * 6), 0.f);
    for (int t = 0; t < 6; ++t) {
      const Tensor teacher = nn::from_vector(tokens, {n, 6});
      const auto   p       = TransformerNmt(store, nn::one_hot(b.input, 256), nn::one_hot(teacher, 256), b.input_lengths,
                                            nn::full({n}, t + 1, nn::DType::Int32));
      const auto   z       = p.logits.to_vector<float>();
      const auto   ca      = p.cross_attention.to_vector<float>();
      for (std::int64_t j = 0; j < n; ++j) {
        if (done[static_cast<std::size_t>(j)]) continue;
        const float* row  = z.data() + (j * 6 + t) * 256;
        int          best = 0;
        for (int k = 1; k < 256; ++k)
          if (row[k] > row[best]) best = k;
        generated[static_cast<std::size_t>(j * 6 + t)] = best;
        if (t < 5) tokens[static_cast<std::size_t>(j * 6 + t + 1)] = best;
        for (int h = 0; h < 2; ++h)
          for (int k = 0; k < 6; ++k) {
            const auto i = static_cast<std::size_t>(((j * 2 + h) * 6 + t) * 6 + k);
            tr[i]        = ca[i];
          }
        done[static_cast<std::size_t>(j)] = best == 2;
      }
    }
    if (trace) *trace = nn::from_vector(tr, {n, 2, 6, 6});
    return nn::from_vector(generated, {n, 6});
  }
};

}  // namespace nmt
