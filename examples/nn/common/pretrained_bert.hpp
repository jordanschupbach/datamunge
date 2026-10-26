// Published Small BERT inference from the actual 44 exported variable arrays.
// NPZ is the cross-framework weight boundary: no SavedModel loader is needed.
// Equations match the inspected source graph, including key-only masks and
// nonzero padded query states. The unused MLM head remains in the inventory.
#pragma once
#include "common/tf_replay.hpp"
#include "common/wordpiece.hpp"
namespace pretrained_bert {
namespace nn = datamunge::nn;
using nn::Tensor;
using A = replay::Map;
class Model {
  A weights;

 public:
  explicit Model(const std::filesystem::path& file)
      : weights(replay::ToMap(nn::load_npz(file.string()))) {
    check::Require(weights.size() == 44, "44 published stored tensors");
    std::int64_t count = 0;
    for (const auto& [k, v] : weights) {
      check::Require(v.dtype() == nn::DType::Float32, "float32 published weights");
      count += v.numel();
    }
    check::Require(count == 4433210, "actual pretrained variable count");
  }
  A Arrays() const {
    A out;
    for (const auto& [k, v] : weights)
      out[k] = v.clone();
    return out;
  }
  A Run(const A& b) const {
    for (auto key : {"input_word_ids", "input_mask", "input_type_ids"})
      check::Require(
          b.count(key) && b.at(key).dtype() == nn::DType::Int32 && b.at(key).dim() == 2,
          "BERT input keys/int32/rank2");
    auto ids = b.at("input_word_ids"), mask = b.at("input_mask"), types = b.at("input_type_ids");
    int  n = ids.size(0), t = ids.size(1);
    check::Require(
        ids.shape() == mask.shape() && ids.shape() == types.shape() && n > 0 && t >= 2 && t <= 512,
        "matching BERT input shapes/max512");
    auto iv = ids.to_vector<int>(), mv = mask.to_vector<int>(), tv = types.to_vector<int>();
    for (int row = 0; row < n; ++row) {
      int active = 0;
      for (int j = 0; j < t; ++j) {
        int i = row * t + j;
        check::Require(iv[i] >= 0 && iv[i] < 30522, "BERT vocabulary range");
        check::Require(mv[i] == 0 || mv[i] == 1, "binary attention mask");
        check::Require(tv[i] == 0 || tv[i] == 1, "two segment IDs");
        active += mv[i];
      }
      check::Require(active > 0, "at least one visible token");
    }
    nn::NoGradGuard guard;
    auto            w    = [&](const std::string& key) { return weights.at(key); };
    auto            norm = [&](const Tensor& x, const std::string& prefix) {
      auto c = x - nn::mean(x, {-1}, true);
      return c / nn::sqrt(nn::mean(nn::square(c), {-1}, true) + 1e-12) * w(prefix + "/gamma") +
             w(prefix + "/beta");
    };
    auto h = norm(
        nn::gather(w("word_embeddings/embeddings"), ids) +
            nn::slice(w("position_embedding/embeddings"), {0, 0}, {t, 128}) +
            nn::gather(w("type_embeddings/embeddings"), types),
        "embeddings/layer_norm");
    std::vector<Tensor> states;
    auto bias = nn::reshape((1. - nn::cast(mask, nn::DType::Float32)) * -1e9, {n, 1, 1, t});
    for (int l = 0; l < 2; ++l) {
      auto prefix  = "transformer/layer_" + std::to_string(l);
      auto project = [&](const std::string& name) {
        auto p     = prefix + "/self_attention/" + name;
        auto value = nn::matmul(h, nn::reshape(w(p + "/kernel"), {128, 128})) +
                     nn::reshape(w(p + "/bias"), {128});
        return nn::transpose(nn::reshape(value, {n, t, 2, 64}), {0, 2, 1, 3});
      };
      auto q = project("query"), k = project("key"), v = project("value");
      auto probability = nn::softmax(nn::matmul(q * .125, k, false, true) + bias, -1);
      auto context =
          nn::reshape(nn::transpose(nn::matmul(probability, v), {0, 2, 1, 3}), {n, t, 128});
      auto out =
          nn::matmul(
              context,
              nn::reshape(w(prefix + "/self_attention/attention_output/kernel"), {128, 128})) +
          w(prefix + "/self_attention/attention_output/bias");
      h      = norm(h + out, prefix + "/self_attention_layer_norm");
      auto f = nn::gelu(
          nn::matmul(h, w(prefix + "/intermediate/kernel")) + w(prefix + "/intermediate/bias"),
          true);
      out = nn::matmul(f, w(prefix + "/output/kernel")) + w(prefix + "/output/bias");
      h   = norm(h + out, prefix + "/output_layer_norm");
      states.push_back(h);
    }
    auto cls = nn::squeeze(nn::slice(h, {0, 0, 0}, {n, 1, 128}), {1});
    auto pooled =
        nn::tanh(nn::matmul(cls, w("pooler_transform/kernel")) + w("pooler_transform/bias"));
    return {
        {"bert_encoder", pooled},
        {"bert_encoder_1", states[0]},
        {"bert_encoder_2", states[1]},
        {"bert_encoder_3", pooled},
        {"bert_encoder_4", states[1]}};
  }
};
} // namespace pretrained_bert
