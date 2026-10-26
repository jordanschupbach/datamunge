// Task-specific classification, first-piece NER and context-only QA heads over
// the trained BERT encoder. Port of tflow common/ml/bert_finetune.h.
#pragma once
#include "common/bert_mlm.hpp"
#include "common/tf_replay.hpp"
namespace finebert {
namespace nn = datamunge::nn;
using nn::Tensor;
using A = bert::Batch;
struct Path {
  Tensor            logits;
  bert::EncoderPath core;
};
class Model {
 public:
  std::string                    task;
  int                            time, classes;
  nn::ParameterStore             parameters{46503};
  std::unique_ptr<nn::Optimizer> optimizer;
  explicit Model(std::string name, bool training = true)
      : task(std::move(name)), time(task == "qa" ? 256 : 128), classes(task == "ner" ? 13 : 2) {
    check::Require(task == "classification" || task == "ner" || task == "qa", "three BERT tasks");
    {
      nn::NoGradGuard guard;
      Forward(
          nn::zeros({1, time, 70}),
          {{"segments", nn::zeros({1, time}, nn::DType::Int32)},
           {"lengths", nn::full({1}, time, nn::DType::Int32)},
           {"weights", nn::ones({1, time})}});
    }
    if (training) {
      nn::OptimizerConfig opt;
      opt.kind  = nn::OptimizerKind::Adam;
      optimizer = std::make_unique<nn::Optimizer>(parameters, .001, opt);
    }
  }
  Path Forward(const Tensor& encoded, const A& b) {
    Path out;
    out.core = bert::BertEncoder(parameters, encoded, b.at("segments"), b.at("lengths"), time);
    if (task == "classification") {
      auto cls = nn::reshape(
          nn::slice(out.core.hidden, {0, 0, 0}, {encoded.size(0), 1, 8}), {encoded.size(0), 8});
      out.logits =
          bert::Linear(parameters, "head", nn::tanh(bert::Linear(parameters, "pooler", cls, 8)), 2);
    } else {
      out.logits = bert::Linear(parameters, "head", out.core.hidden, classes);
      if (task == "qa")
        out.logits = nn::where(
            nn::greater(nn::expand_dims(b.at("weights"), 2), nn::scalar(0)),
            out.logits,
            nn::full(out.logits.shape(), -1e9));
    }
    return out;
  }
  Tensor Loss(const Tensor& logits, const A& b) const {
    if (task == "classification")
      return -nn::mean(nn::sum(nn::log_softmax(logits, -1) * nn::one_hot(b.at("labels"), 2), {1}));
    if (task == "ner") {
      auto ce = -nn::sum(nn::log_softmax(logits, -1) * nn::one_hot(b.at("labels"), classes), {2});
      return nn::sum(ce * b.at("weights")) / nn::maximum(nn::sum(b.at("weights")), nn::scalar(1));
    }
    auto lp = nn::log_softmax(nn::reshape(nn::transpose(logits, {0, 2, 1}), {-1, time}), -1);
    return -nn::mean(nn::sum(lp * nn::one_hot(nn::reshape(b.at("labels"), {-1}), time), {1}));
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
    auto                encoded = nn::one_hot(b.at("input"), 70).requires_grad_();
    auto                path    = Forward(encoded, b);
    auto                loss    = Loss(path.logits, b);
    std::vector<Tensor> inputs{encoded};
    if (gradients) {
      auto p = parameters.parameters();
      inputs.insert(inputs.end(), p.begin(), p.end());
    }
    auto g = nn::grad(loss, inputs);
    A    out{
           {"logits", path.logits},
           {"hidden", path.core.hidden},
           {"attention", path.core.attention},
           {"loss", loss},
           {"input_gradient", g[0]}};
    if (gradients)
      for (size_t i = 0; i < parameters.size(); ++i)
        out[parameters.names()[i]] = g[i + 1];
    for (auto& [k, v] : out)
      v = v.detach().clone();
    return out;
  }
  A    Arrays() const { return replay::Arrays(parameters); }
  void ImportEncoder(const A& source) {
    int count = 0;
    for (auto k : parameters.names())
      if (source.count(k)) {
        auto value = parameters.at(k), old = source.at(k);
        if (k == "position/W") {
          check::Require(old.shape() == nn::Shape({32, 8}), "trained32positiontable");
          auto x = value.to_vector<float>(), y = old.to_vector<float>();
          std::copy(y.begin(), y.end(), x.begin());
          parameters.set(k, nn::from_vector(x, value.shape()));
        } else {
          check::Require(value.shape() == old.shape(), "matching pretrained tensor");
          parameters.set(k, old);
        }
        ++count;
      }
    check::Require(count == 21, "all21encoder tensors transferred,no MLM head");
  }
  Tensor Predict(const A& b) {
    nn::NoGradGuard guard;
    return Forward(nn::one_hot(b.at("input"), 70), b).logits;
  }
  void Update(const A& b) {
    auto path = Forward(nn::one_hot(b.at("input"), 70), b);
    optimizer->minimize(Loss(path.logits, b));
  }
};
} // namespace finebert
