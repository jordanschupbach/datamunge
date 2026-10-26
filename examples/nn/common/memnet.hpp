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
  const Tensor masked = nn::where(valid, scores, nn::add(nn::mul(scores, 0.0), -1e9));
  const Tensor raw    = nn::mul(nn::softmax(masked, -1), nn::cast(valid, nn::DType::Float32));
  return nn::div(raw, nn::maximum(nn::sum(raw, {1}, true), nn::full({1}, 1.0)));
}

/// Bool [B, 6]: fact slot < length.
inline Tensor FactValid(const Tensor& lengths) {
  return nn::less(nn::expand_dims(nn::arange(0, 6, 1, nn::DType::Float32), 0),
                  nn::expand_dims(nn::cast(lengths, nn::DType::Float32), 1));
}

/// facts one-hot [B, 6, 2, 13], query one-hot [B, 13]. Fields: logits [B, 4],
/// states [B, 3, 8], attention [B, 2, 6], reads [B, 2, 8], m0, m1, m2 [B, 6, 8].
inline Outputs MemNet(nn::ParameterStore& p, const Tensor& facts, const Tensor& query, const Tensor& lengths) {
  std::vector<float> positions(16);
  for (int j = 0; j < 2; ++j)
    for (int k = 0; k < 8; ++k) positions[j * 8 + k] = j == 0 ? .5f : (k + 1) / 8.f;
  const Tensor pos   = nn::from_vector(positions, {1, 1, 2, 8});
  const Tensor valid = FactValid(lengths);
  std::vector<Tensor> tables, memory;
  for (int i = 0; i < 3; ++i) {
    const Tensor table = p.get("embedding" + std::to_string(i) + "/W", {13, 8}, nn::init::normal(.1));
    tables.push_back(table);
    const Tensor embedded = nn::reshape(nn::matmul(nn::reshape(facts, {-1, 13}), table), {-1, 6, 2, 8});
    const Tensor sentence = nn::sum(nn::mul(embedded, pos), {2});
    memory.push_back(nn::where(nn::expand_dims(valid, 2), sentence, nn::zeros_like(sentence)));
  }
  Tensor              u = nn::matmul(query, tables[0]);
  std::vector<Tensor> states{u}, attention, reads;
  for (int hop = 0; hop < 2; ++hop) {
    const Tensor weights = HopWeights(nn::sum(nn::mul(memory[hop], nn::expand_dims(u, 1)), {2}), valid);
    const Tensor read    = nn::sum(nn::mul(memory[hop + 1], nn::expand_dims(weights, 2)), {1});
    u                    = nn::add(u, read);
    states.push_back(u);
    attention.push_back(weights);
    reads.push_back(read);
  }
  const Tensor answers = nn::slice(tables[2], {8, 0}, {4, 8});
  return {{"logits", nn::matmul(u, answers, false, true)},
          {"states", nn::stack_time(states)},
          {"attention", nn::stack_time(attention)},
          {"reads", nn::stack_time(reads)},
          {"m0", memory[0]},
          {"m1", memory[1]},
          {"m2", memory[2]}};
}

/// Same inputs; keys from the first token of each fact, values from the
/// second (shared embedding A), hop transforms R0, R1 (identity init),
/// answers B. Fields: logits, states, attention, reads, keys, values.
inline Outputs KeyValueMemory(nn::ParameterStore& p, const Tensor& facts, const Tensor& query, const Tensor& lengths) {
  const Tensor embedding = p.get("embedding/A", {13, 8}, nn::init::normal(.1));
  const Tensor answers   = p.get("answers/B", {4, 8}, nn::init::normal(.1));
  const Tensor valid     = FactValid(lengths);
  auto encode = [&](std::int64_t role) {
    const Tensor onehot  = nn::reshape(nn::slice(facts, {0, 0, role, 0}, {-1, 6, 1, 13}), {-1, 13});
    const Tensor vectors = nn::reshape(nn::matmul(onehot, embedding), {-1, 6, 8});
    return nn::where(nn::expand_dims(valid, 2), vectors, nn::zeros_like(vectors));
  };
  const Tensor        keys = encode(0), values = encode(1);
  Tensor              u = nn::matmul(query, embedding);
  std::vector<Tensor> states{u}, attention, reads;
  for (int hop = 0; hop < 2; ++hop) {
    const Tensor weights  = HopWeights(nn::sum(nn::mul(keys, nn::expand_dims(u, 1)), {2}), valid);
    const Tensor read     = nn::sum(nn::mul(values, nn::expand_dims(weights, 2)), {1});
    const Tensor rotation = p.get("hop" + std::to_string(hop) + "/R", {8, 8},
                                  [](const nn::Shape&, nn::DType d, nn::Generator&) { return nn::eye(8, d); });
    u = nn::matmul(nn::add(u, read), rotation);
    states.push_back(u);
    attention.push_back(weights);
    reads.push_back(read);
  }
  return {{"logits", nn::matmul(u, answers, false, true)},
          {"states", nn::stack_time(states)},
          {"attention", nn::stack_time(attention)},
          {"reads", nn::stack_time(reads)},
          {"keys", keys},
          {"values", values}};
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
