// Shared BERT masked-mean/L2 sentence towers, cosine regression and in-batch
// negative retrieval; port of tflow common/{nn,ml}/sentence_embedding(s).h.
#pragma once
#include "common/bert_mlm.hpp"
#include "common/tf_replay.hpp"
namespace sentence_embeddings {
namespace nn = datamunge::nn;
using nn::Tensor;
using A = bert::Batch;
struct Path {
  bert::EncoderPath encoder;
  Tensor            mean, embedding;
};
class Model {
 public:
  bool                           dual;
  nn::ParameterStore             parameters{47503};
  std::unique_ptr<nn::Optimizer> optimizer;
  explicit Model(bool retrieval, bool training = true) : dual(retrieval) {
    {
      nn::NoGradGuard guard;
      Forward(nn::zeros({1, 128, 70}), nn::full({1}, 128, nn::DType::Int32));
    }
    if (training) {
      nn::OptimizerConfig opt;
      opt.kind  = nn::OptimizerKind::Adam;
      optimizer = std::make_unique<nn::Optimizer>(parameters, .003, opt);
    }
  }
  Path Forward(const Tensor& encoded, const Tensor& lengths) {
    auto core = bert::BertEncoder(
        parameters, encoded, nn::zeros({encoded.size(0), 128}, nn::DType::Int32), lengths, 128);
    auto mean =
        nn::sum(core.hidden, {1}) /
        nn::expand_dims(nn::maximum(nn::cast(lengths, nn::DType::Float32), nn::scalar(1)), 1);
    auto norm = nn::sqrt(nn::maximum(nn::sum(nn::square(mean), {1}, true), nn::scalar(1e-12)));
    return {core, mean, mean / norm};
  }
  A Fields(const Path& a, const Path& b, const A& batch) const {
    auto cosine = nn::sum(a.embedding * b.embedding, {1}),
         scores = nn::matmul(a.embedding, b.embedding, false, true);
    auto loss =
        dual ? -nn::mean(nn::sum(nn::eye(scores.size(0)) * nn::log_softmax(scores * 10., -1), {1}))
             : nn::mean(nn::square(cosine - batch.at("score")));
    return {
        {"embedding_a", a.embedding},
        {"embedding_b", b.embedding},
        {"mean_a", a.mean},
        {"mean_b", b.mean},
        {"hidden_a", a.encoder.hidden},
        {"hidden_b", b.encoder.hidden},
        {"attention_a", a.encoder.attention},
        {"attention_b", b.encoder.attention},
        {"cosine", cosine},
        {"scores", scores},
        {"loss", loss}};
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
    auto ea = nn::one_hot(b.at("input_a"), 70).requires_grad_(),
         eb = nn::one_hot(b.at("input_b"), 70).requires_grad_();
    auto a = Forward(ea, b.at("lengths_left")), right = Forward(eb, b.at("lengths_right"));
    auto out = Fields(a, right, b);
    std::vector<Tensor> inputs{ea, eb};
    if (gradients) {
      auto p = parameters.parameters();
      inputs.insert(inputs.end(), p.begin(), p.end());
    }
    auto g                  = nn::grad(out.at("loss"), inputs);
    out["input_gradient_a"] = g[0];
    out["input_gradient_b"] = g[1];
    if (gradients)
      for (size_t i = 0; i < parameters.size(); ++i)
        out[parameters.names()[i]] = g[i + 2];
    for (auto& [k, v] : out)
      v = v.detach().clone();
    return out;
  }
  Tensor Encode(const Tensor& input, const Tensor& lengths) {
    check::Require(
        input.dim() == 2 && input.size(1) == 128 && input.dtype() == nn::DType::Int32 &&
            lengths.dtype() == nn::DType::Int32 && lengths.numel() == input.size(0),
        "sentence ID/length matrices");
    for (auto id : input.to_vector<int>())
      check::Require(id >= 0 && id < 70, "sentence ID range");
    for (auto length : lengths.to_vector<int>())
      check::Require(length >= 0 && length <= 128, "sentence length range");
    nn::NoGradGuard guard;
    return Forward(nn::one_hot(input, 70), lengths).embedding;
  }
  void Update(const A& b) {
    auto a     = Forward(nn::one_hot(b.at("input_a"), 70), b.at("lengths_left")),
         right = Forward(nn::one_hot(b.at("input_b"), 70), b.at("lengths_right"));
    optimizer->minimize(Fields(a, right, b).at("loss"));
  }
  A    Arrays() const { return replay::Arrays(parameters); }
  void Import(const A& source) {
    int count = 0;
    for (auto name : parameters.names()) {
      auto value = parameters.at(name), old = source.at(name);
      if (name == "position/W") {
        auto x = value.to_vector<float>(), y = old.to_vector<float>();
        check::Require(old.shape() == nn::Shape({32, 8}), "trained32positiontable");
        std::copy(y.begin(), y.end(), x.begin());
        parameters.set(name, nn::from_vector(x, value.shape()));
      } else {
        check::Require(value.shape() == old.shape(), "sentence import shape");
        parameters.set(name, old);
      }
      ++count;
    }
    check::Require(count == 21, "all21 encoder tensors imported");
  }
};
} // namespace sentence_embeddings
