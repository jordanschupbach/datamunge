// T5-style pre-RMS encoder-decoder with relative bucket biases and a shared
// token/output table; port of tflow common/{nn,ml}/t5_text.h.
#pragma once
#include "common/bert_mlm.hpp"
#include "common/positional.hpp"
#include "common/tf_replay.hpp"
namespace t5 {
namespace nn = datamunge::nn;
using nn::Tensor;
using A = bert::Batch;
struct Path {
  Tensor logits, memory, hidden, encoder_attention, decoder_attention, cross_attention;
};
inline Path Forward(
    nn::ParameterStore& p,
    const Tensor&       source,
    const Tensor&       teacher,
    const Tensor&       sl,
    const Tensor&       tl) {
  constexpr int T = 48, D = 8;
  auto table = p.get("token/W", {72, D}, nn::init::normal(.2)), sm = bert::LengthMask(sl, T),
       tm     = bert::LengthMask(tl, T);
  auto linear = [&](const Tensor& x, const std::string& name, int out) {
    return nn::matmul(x, p.get(name, {x.size(2), out}, nn::init::normal(.1)));
  };
  auto norm = [&](const Tensor& x, const std::string& name, const Tensor& mask) {
    return nn::rms_norm(p, name, x, 1e-6) * mask;
  };
  auto relative = [&](const std::string& name, bool bidir) {
    return positional::RelativePositionBias(
        p.get(name, {32, 2}, nn::init::normal(.02)), T, T, bidir, 32, 128);
  };
  auto attention = [&](const Tensor&      x,
                       const Tensor&      kv,
                       const std::string& name,
                       const Tensor&      kl,
                       const Tensor&      ql,
                       bool               causal,
                       const Tensor&      bias) {
    auto q      = bert::Heads(linear(x, name + "/Wq", D), 2),
         k      = bert::Heads(linear(kv, name + "/Wk", D), 2),
         v      = bert::Heads(linear(kv, name + "/Wv", D), 2);
    auto logits = nn::matmul(q, k, false, true);
    if (bias.defined())
      logits = logits + bias;
    auto allowed     = nn::expand_dims(attn::AttentionAllowed(kl, ql, T, T, causal), 1);
    auto probability = nn::softmax(nn::where(allowed, logits, nn::full(logits.shape(), -1e9)), -1) *
                       nn::cast(allowed, nn::DType::Float32);
    auto merged = nn::reshape(
        nn::transpose(nn::matmul(probability, v), {0, 2, 1, 3}), {source.size(0), T, D});
    return std::pair<Tensor, Tensor>{linear(merged, name + "/Wo", D), probability};
  };
  auto ff = [&](const Tensor& x, const std::string& name) {
    return linear(nn::relu(linear(x, name + "/Wi", 16)), name + "/Wo", D);
  };
  auto memory     = nn::matmul(source, table) * sm;
  auto normalized = norm(memory, "encoder/self_norm", sm);
  auto enc        = attention(
      normalized, normalized, "encoder/self", sl, sl, false, relative("encoder/relative", true));
  memory      = (memory + enc.first) * sm;
  memory      = (memory + ff(norm(memory, "encoder/ff_norm", sm), "encoder/ff")) * sm;
  memory      = norm(memory, "encoder/final_norm", sm);
  auto hidden = nn::matmul(teacher, table) * tm;
  auto dn     = norm(hidden, "decoder/self_norm", tm);
  auto dec = attention(dn, dn, "decoder/self", tl, tl, true, relative("decoder/relative", false));
  hidden   = (hidden + dec.first) * tm;
  auto cross =
      attention(norm(hidden, "decoder/cross_norm", tm), memory, "decoder/cross", sl, tl, false, {});
  hidden      = (hidden + cross.first) * tm;
  hidden      = (hidden + ff(norm(hidden, "decoder/ff_norm", tm), "decoder/ff")) * tm;
  hidden      = norm(hidden, "decoder/final_norm", tm);
  auto logits = nn::matmul(hidden * float(1 / std::sqrt(float(D))), table, false, true) * tm;
  return {logits, memory, hidden, enc.second, dec.second, cross.second};
}
class Model {
 public:
  nn::ParameterStore             parameters{46803};
  std::unique_ptr<nn::Optimizer> optimizer;
  explicit Model(bool training = true) {
    {
      nn::NoGradGuard guard;
      Forward(
          parameters,
          nn::zeros({1, 48, 72}),
          nn::zeros({1, 48, 72}),
          nn::full({1}, 48, nn::DType::Int32),
          nn::full({1}, 48, nn::DType::Int32));
    }
    if (training) {
      nn::OptimizerConfig opt;
      opt.kind  = nn::OptimizerKind::Adam;
      optimizer = std::make_unique<nn::Optimizer>(parameters, .003, opt);
    }
  }
  static Tensor Loss(const Tensor& logits, const A& b) {
    auto mask = nn::squeeze(bert::LengthMask(b.at("output_lengths"), 48), {2});
    auto ce   = -nn::sum(nn::log_softmax(logits, -1) * nn::one_hot(b.at("target"), 72), {2});
    return nn::sum(ce * mask) / nn::maximum(nn::sum(mask), nn::scalar(1));
  }
  A Run(const A& b, bool gradients = false, const A& overrides = {}) {
    A saved;
    for (const auto& [k, v] : overrides) {
      saved[k] = parameters.at(k).detach().clone();
      parameters.set(k, v);
    }
    struct Restore {
      nn::ParameterStore& p;
      A&                  saved;
      ~Restore() {
        for (const auto& [k, v] : saved)
          p.set(k, v);
      }
    } restore{parameters, saved};
    auto source  = nn::one_hot(b.at("input"), 72).requires_grad_(),
         teacher = nn::one_hot(b.at("teacher"), 72).requires_grad_();
    auto path = Forward(parameters, source, teacher, b.at("input_lengths"), b.at("output_lengths"));
    auto loss = Loss(path.logits, b);
    std::vector<Tensor> inputs{source, teacher, path.memory};
    if (gradients) {
      auto p = parameters.parameters();
      inputs.insert(inputs.end(), p.begin(), p.end());
    }
    nn::GradOptions retained;
    retained.retain_graph = true;
    auto g                = nn::grad(loss, inputs, retained);
    auto prefix =
        nn::grad(nn::sum(nn::slice(path.logits, {0, 0, 0}, {source.size(0), 2, 7})), {teacher})[0];
    A out{
        {"loss", loss},
        {"logits", path.logits},
        {"memory", path.memory},
        {"hidden", path.hidden},
        {"encoder_attention", path.encoder_attention},
        {"decoder_attention", path.decoder_attention},
        {"cross_attention", path.cross_attention},
        {"prefix_gradient", prefix},
        {"partial0", g[0]},
        {"partial1", g[1]},
        {"partial2", g[2]}};
    if (gradients)
      for (size_t i = 0; i < parameters.size(); ++i)
        out[parameters.names()[i]] = g[i + 3];
    for (auto& [k, v] : out)
      v = v.detach().clone();
    return out;
  }
  void Update(const A& b) {
    auto path = Forward(
        parameters,
        nn::one_hot(b.at("input"), 72),
        nn::one_hot(b.at("teacher"), 72),
        b.at("input_lengths"),
        b.at("output_lengths"));
    optimizer->minimize(Loss(path.logits, b));
  }
  A      Arrays() const { return replay::Arrays(parameters); }
  Tensor Greedy(const A& b, Tensor* trace = nullptr) {
    nn::NoGradGuard    guard;
    int                n = b.at("input").size(0);
    std::vector<int>   tokens(n * 48), generated(n * 48);
    std::vector<bool>  done(n);
    std::vector<float> attention(n * 2 * 48 * 48);
    auto               source = nn::one_hot(b.at("input"), 72);
    for (int t = 0; t < 48; ++t) {
      auto path = Forward(
          parameters,
          source,
          nn::one_hot(nn::from_vector(tokens, {n, 48}), 72),
          b.at("input_lengths"),
          nn::full({n}, t + 1, nn::DType::Int32));
      auto z = path.logits.to_vector<float>(), a = path.cross_attention.to_vector<float>();
      for (int row = 0; row < n; ++row) {
        if (done[row])
          continue;
        int best = 0;
        for (int c = 1; c < 72; ++c)
          if (z[(row * 48 + t) * 72 + c] > z[(row * 48 + t) * 72 + best])
            best = c;
        generated[row * 48 + t] = best;
        if (t < 47)
          tokens[row * 48 + t + 1] = best;
        if (trace)
          for (int h = 0; h < 2; ++h)
            for (int k = 0; k < 48; ++k)
              attention[((row * 2 + h) * 48 + t) * 48 + k] = a[((row * 2 + h) * 48 + t) * 48 + k];
        done[row] = best == 1;
      }
    }
    if (trace)
      *trace = nn::from_vector(attention, {n, 2, 48, 48});
    return nn::from_vector(generated, {n, 48});
  }
};
} // namespace t5
