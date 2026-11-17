// Two-hop end-to-end memory network with adjacent embedding and answer tying
// (port of tflow's common/nn/memnet.h, 312 parameters) and the key-value
// memory network with separate key/value feature maps and learned hop
// transforms (port of tflow's common/nn/key_value.h, 264 parameters), plus the
// shared question-answering model of tflow's memnet.cc / key_value.cc.
// Facts [B, 6, 2] and queries [B] are token ids < 13; lengths [B] count the
// valid facts; targets [B] are one of 4 answers.
#pragma once

#include <memory>
#include <sstream>
#include <utility>
#include <vector>

#include "common/memory_task.hpp"

namespace memory_nets {

/// Masked softmax attention over the valid facts, renormalised by
/// max(sum, 1) so that an empty memory reads zero (tflow's hop weights).
inline Tensor HopWeights(const Tensor& scores, const Tensor& valid) {
  return nn::models::HopWeights(scores,valid);
}

/// Bool [B, 6]: fact slot < length.
inline Tensor FactValid(const Tensor& lengths) {
  return nn::models::FactValid(lengths);
}

/// facts one-hot [B, 6, 2, 13], query one-hot [B, 13]. Fields: logits [B, 4],
/// states [B, 3, 8], attention [B, 2, 6], reads [B, 2, 8], m0, m1, m2 [B, 6, 8].
inline Outputs MemNet(nn::ParameterStore& p, const Tensor& facts, const Tensor& query, const Tensor& lengths) {
  auto y=nn::models::AdjacentMemoryNetwork(p,facts,query,lengths);return {{"logits",y.logits},{"states",y.states},{"attention",y.attention},{"reads",y.reads},{"m0",y.m0},{"m1",y.m1},{"m2",y.m2}};
}

/// Same inputs; keys from the first token of each fact, values from the
/// second (shared embedding A), hop transforms R0, R1 (identity init),
/// answers B. Fields: logits, states, attention, reads, keys, values.
inline Outputs KeyValueMemory(nn::ParameterStore& p, const Tensor& facts, const Tensor& query, const Tensor& lengths) {
  auto y=nn::models::KeyValueMemoryNetwork(p,facts,query,lengths);return {{"logits",y.logits},{"states",y.states},{"attention",y.attention},{"reads",y.reads},{"keys",y.keys},{"values",y.values}};
}

using QaPath = std::function<Outputs(nn::ParameterStore&, const Tensor&, const Tensor&, const Tensor&)>;

/// tflow's memnet/key_value Model: one-hot leaves "encoded"/"qencoded" (their
/// loss gradients are input_gradient / query_gradient), mean softmax CE,
/// Adam .01.
inline std::unique_ptr<MemoryModel> QaModel(std::uint64_t seed, QaPath path, bool training, const Batch& example) {
  auto m = std::make_unique<MemoryModel>(seed, [path](nn::ParameterStore& p, const Batch& leaves, const Batch& b) {
    return path(p, leaves.at("encoded"), leaves.at("qencoded"), b.at("lengths"));
  });
  m->leaves = [](const Batch& b) {
    return Batch{{"encoded", nn::one_hot(b.at("facts"), 13)}, {"qencoded", nn::one_hot(b.at("query"), 13)}};
  };
  m->loss = [](const Outputs& out, const Batch& b) {
    return nn::neg(nn::mean(nn::sum(nn::mul(nn::log_softmax(out.at("logits"), -1), nn::one_hot(b.at("target"), 4)), {1}), {0}));
  };
  m->leaf_gradients = {{"encoded", "input_gradient"}, {"qencoded", "query_gradient"}};
  m->Build(example, training, .01);
  return m;
}

inline std::unique_ptr<MemoryModel> MemNetModel(bool training, const Batch& example) {
  return QaModel(45303, MemNet, training, example);
}
inline std::unique_ptr<MemoryModel> KeyValueModel(bool training, const Batch& example) {
  return QaModel(45403, KeyValueMemory, training, example);
}

}  // namespace memory_nets
