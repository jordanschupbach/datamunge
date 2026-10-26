// Completion-only causal decoder likelihood, pretrained GPT transfer and own-
// token prompt generation; port of tflow common/ml/instruction_tuning.h.
#pragma once
#include "common/bert_mlm.hpp"
#include "common/decoder_transformer.hpp"
#include "common/tf_replay.hpp"
namespace instruction {
namespace nn = datamunge::nn;
using nn::Tensor;
using A = bert::Batch;
class Model {
 public:
  lm::DecoderConfig              config;
  nn::ParameterStore             parameters{48103};
  std::unique_ptr<nn::Optimizer> optimizer;
  explicit Model(bool training = true) {
    config.input_classes = config.output_classes = 67;
    {
      nn::NoGradGuard guard;
      lm::DecoderTransformer(parameters, nn::zeros({1, 64, 67}), config);
    }
    if (training) {
      nn::OptimizerConfig opt;
      opt.kind  = nn::OptimizerKind::Adam;
      optimizer = std::make_unique<nn::Optimizer>(parameters, .003, opt);
    }
  }
  lm::DecoderPath Forward(const Tensor& encoded, const Tensor& lengths) {
    auto path      = lm::DecoderTransformer(parameters, encoded, config);
    auto mask      = bert::LengthMask(lengths, 64);
    path.logits    = path.logits * mask;
    path.hidden    = path.hidden * mask;
    path.embedded  = path.embedded * mask;
    path.attention = path.attention * nn::expand_dims(mask, 1);
    return path;
  }
  static Tensor Loss(const Tensor& logits, const A& b) {
    auto ce = -nn::sum(nn::log_softmax(logits, -1) * nn::one_hot(b.at("target"), 67), {2});
    return nn::sum(ce * b.at("selected")) / nn::maximum(nn::sum(b.at("selected")), nn::scalar(1));
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
    auto                encoded = nn::one_hot(b.at("input"), 67).requires_grad_();
    auto                path    = Forward(encoded, b.at("lengths"));
    auto                loss    = Loss(path.logits, b);
    std::vector<Tensor> inputs{encoded, path.logits};
    if (gradients) {
      auto p = parameters.parameters();
      inputs.insert(inputs.end(), p.begin(), p.end());
    }
    nn::GradOptions retained;
    retained.retain_graph = true;
    auto g                = nn::grad(loss, inputs, retained);
    auto prefix           = nn::grad(
        nn::sum(nn::slice(path.logits, {0, 32, 0}, {encoded.size(0), 1, 1})), {encoded})[0];
    A out{
        {"logits", path.logits},
        {"hidden", path.hidden},
        {"attention", path.attention},
        {"embedded", path.embedded},
        {"loss", loss},
        {"input_gradient", g[0]},
        {"logit_gradient", g[1]},
        {"prefix_gradient", prefix}};
    if (gradients)
      for (size_t i = 0; i < parameters.size(); ++i)
        out[parameters.names()[i]] = g[i + 2];
    for (auto& [k, v] : out)
      v = v.detach().clone();
    return out;
  }
  Tensor Predict(const A& b) {
    nn::NoGradGuard guard;
    return Forward(nn::one_hot(b.at("input"), 67), b.at("lengths")).logits;
  }
  void Update(const A& b) {
    auto path = Forward(nn::one_hot(b.at("input"), 67), b.at("lengths"));
    optimizer->minimize(Loss(path.logits, b));
  }
  A    Arrays() const { return replay::Arrays(parameters); }
  void Import(const A& source) {
    for (auto name : parameters.names()) {
      auto value = parameters.at(name), old = source.at(name);
      if (name == "token/W" || name == "head/b") {
        auto x = value.to_vector<float>(), y = old.to_vector<float>();
        check::Require(
            old.shape() == (name == "token/W" ? nn::Shape{65, 16} : nn::Shape{65}),
            "pretrained65tokenrows/headbias");
        std::copy(y.begin(), y.end(), x.begin());
        parameters.set(name, nn::from_vector(x, value.shape()));
      } else if (name == "head/W") {
        check::Require(old.shape() == nn::Shape({16, 65}), "pretrained65headcolumns");
        auto x = value.to_vector<float>(), y = old.to_vector<float>();
        for (int row = 0; row < 16; ++row)
          for (int col = 0; col < 65; ++col)
            x[row * 67 + col] = y[row * 65 + col];
        parameters.set(name, nn::from_vector(x, value.shape()));
      } else {
        check::Require(value.shape() == old.shape(), "same GPT layer shape");
        parameters.set(name, old);
      }
    }
  }
  A Generate(const Tensor& prompts, const Tensor& prompt_lengths, int maximum = 12) {
    check::Require(
        prompts.dtype() == nn::DType::Int32 && prompts.dim() == 2 && prompts.size(1) == 64 &&
            maximum > 0 && maximum <= 16,
        "instruction prompt buffers");
    int              n = prompts.size(0);
    std::vector<int> codes(n * maximum, 65), lengthout(n), contexts(n * maximum * 64, 65),
        context_lengths(n * maximum);
    std::vector<float> logits(n * maximum * 67);
    auto               pids = prompts.to_vector<int>(), plens = prompt_lengths.to_vector<int>();
    for (int row = 0; row < n; ++row) {
      int p = plens[row];
      check::Require(p > 0 && p + maximum <= 64, "bounded nonempty prompt plus generation");
      std::vector<int> context(64, 65);
      std::copy_n(pids.begin() + row * 64, p, context.begin());
      for (int step = 0; step < maximum; ++step) {
        std::copy(context.begin(), context.end(), contexts.begin() + (row * maximum + step) * 64);
        context_lengths[row * maximum + step] = p + step;
        auto scores = Predict({{"input", nn::from_vector(context, {1, 64})},
                               {"lengths", nn::full({1}, p + step, nn::DType::Int32)}})
                          .to_vector<float>();
        int best = 0;
        for (int c = 0; c < 67; ++c) {
          logits[(row * maximum + step) * 67 + c] = scores[(p + step - 1) * 67 + c];
          if (scores[(p + step - 1) * 67 + c] > scores[(p + step - 1) * 67 + best])
            best = c;
        }
        codes[row * maximum + step] = best;
        lengthout[row]              = step + 1;
        if (best == 66)
          break;
        context[p + step] = best;
      }
    }
    return {
        {"codes", nn::from_vector(codes, {n, maximum})},
        {"lengths", nn::from_vector(lengthout, {n})},
        {"logits", nn::from_vector(logits, {n, maximum, 67})},
        {"contexts", nn::from_vector(contexts, {n, maximum, 64})},
        {"context_lengths", nn::from_vector(context_lengths, {n, maximum})}};
  }
};
} // namespace instruction
