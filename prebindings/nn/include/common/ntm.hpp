// Neural Turing Machine content/location addressing and differentiable
// erase/add memory. Port of tflow's common/nn/ntm.h (4 slots x 4 wide memory,
// LSTM(8) controller, one read and one write head; 851 parameters).
#pragma once

#include <memory>

#include "common/memory_task.hpp"

namespace memory_nets {

struct NtmAddress {
  Tensor content, interpolated, shifted, weights;
};

/// Content addressing (cosine + softplus strength), gated interpolation with
/// the previous weights, a 3-tap circular shift and sharpening. memory
/// [B, 4, 4], parameters [B, 10] = key(4) beta(1) gate(1) shift(3) gamma(1),
/// previous [B, 4].
inline NtmAddress NtmAddressing(const Tensor& memory, const Tensor& parameters, const Tensor& previous) {
  auto y=nn::models::NtmAddressing(memory,parameters,previous);return {y.content,y.interpolated,y.shifted,y.weights};
}

struct NtmState {
  Tensor h, c, memory, read, read_weights, write_weights;
};
struct NtmStep {
  NtmState state;
  Tensor   logits, interface;
};

/// One NTM step: controller LSTM on [input, previous read], a 28-wide
/// interface (write address 10, read address 10, erase 4, add 4), erase/add
/// write, then the read from the written memory.
inline NtmStep NtmCell(nn::ParameterStore& p, const Tensor& input, const NtmState& previous) {
  auto y=nn::models::NtmCell(p,input,{previous.h,previous.c,previous.memory,previous.read,previous.read_weights,previous.write_weights});return {{y.h,y.c,y.memory,y.read,y.read_weights,y.write_weights},y.logits,y.interface};
}

/// The nine-step unroll with valid steps t <= 2 * length (tflow's nn::Ntm).
/// Fields: logits [B,9,3], interface [B,9,28] (zero on invalid steps) and the
/// state traces h, c [B,10,8], memory [B,10,4,4], read, read_weights,
/// write_weights [B,10,4] (index 0 = initial state).
inline Outputs Ntm(nn::ParameterStore& p, const Tensor& x, const Tensor& lengths) {
  auto y=nn::models::NtmSequence(p,x,lengths);return {{"logits",y.logits},{"h",y.h},{"c",y.c},{"memory",y.memory},{"read",y.read},{"read_weights",y.read_weights},{"write_weights",y.write_weights},{"interface",y.interface}};
}

/// Bit cross-entropy over the three target bits of the recall steps
/// length < t <= 2 * length (tflow's ntm/dnc loss).
inline Tensor CopyLoss(const Outputs& out, const Batch& b) {
  const Tensor& logits    = out.at("logits");
  const Tensor  positions = nn::expand_dims(nn::arange(0, 9, 1, nn::DType::Float32), 0);
  const Tensor  source    = nn::expand_dims(nn::cast(b.at("lengths"), nn::DType::Float32), 1);
  const Tensor  mask = nn::cast(nn::logical_and(nn::greater(positions, source), nn::less_equal(positions, nn::mul(source, 2.0))),
                                nn::DType::Float32);
  const Tensor  ce   = nn::sub(nn::softplus(logits), nn::mul(b.at("target"), logits));
  return nn::div(nn::sum(nn::mul(ce, nn::expand_dims(mask, 2)), {0, 1, 2}), nn::mul(nn::sum(mask, {0, 1}), 3.0));
}

/// The binary-copy model of tflow's ntm.cc / dnc.cc around a forward pass.
inline std::unique_ptr<MemoryModel> CopyModel(std::uint64_t seed,
                                              std::function<Outputs(nn::ParameterStore&, const Tensor&, const Tensor&)> path,
                                              bool training, const Batch& example) {
  auto m = std::make_unique<MemoryModel>(seed, [path](nn::ParameterStore& p, const Batch& leaves, const Batch& b) {
    return path(p, leaves.at("input"), b.at("lengths"));
  });
  m->leaves         = [](const Batch& b) { return Batch{{"input", b.at("input")}}; };
  m->loss           = CopyLoss;
  m->leaf_gradients = {{"input", "input_gradient"}};
  m->Build(example, training, .005);
  return m;
}

/// tflow's ntm Model: seed 45103, Adam .005.
inline std::unique_ptr<MemoryModel> NtmModel(bool training, const Batch& example) { return CopyModel(45103, Ntm, training, example); }

/// Bit accuracy and exact-copy count of the recall steps (tflow's ntm/dnc metrics).
struct CopyScore {
  int correct = 0, total = 0, exact = 0;
};
inline CopyScore ScoreCopy(const Tensor& prediction, const Batch& test) {
  const auto p = prediction.to_vector<float>(), y = test.at("target").to_vector<float>();
  const auto lengths = test.at("lengths").to_vector<std::int32_t>();
  CopyScore  s;
  for (std::size_t b = 0; b < lengths.size(); ++b) {
    bool      match = true;
    const int n     = lengths[b];
    for (int t = n + 1; t <= 2 * n; ++t)
      for (int j = 0; j < 3; ++j) {
        const std::size_t i    = (b * 9 + t) * 3 + j;
        const bool        good = (p[i] > 0) == (y[i] > 0);
        s.correct += good;
        match &= good;
        ++s.total;
      }
    s.exact += match;
  }
  return s;
}

/// Rows {0, 1} of the test split with every input step after 2 * length set to `value`.
inline Batch PadCopyInputs(const Batch& test, float value) {
  Batch      pad     = Rows(test, {0, 1});
  auto       input   = pad.at("input").to_vector<float>();
  const auto lengths = pad.at("lengths").to_vector<std::int32_t>();
  for (int b = 0; b < 2; ++b)
    for (int t = 2 * lengths[b] + 1; t < 9; ++t)
      for (int j = 0; j < 4; ++j) input[(b * 9 + t) * 4 + j] = value;
  pad["input"] = nn::from_vector(input, {2, 9, 4});
  return pad;
}

}  // namespace memory_nets
