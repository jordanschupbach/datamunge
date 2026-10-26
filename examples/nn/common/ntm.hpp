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
  auto part         = [&](std::int64_t b, std::int64_t w) { return Cols(parameters, b, w); };
  const Tensor key  = part(0, 4);
  const Tensor dot  = nn::sum(nn::mul(memory, nn::expand_dims(key, 1)), {2});
  const Tensor mn   = nn::sqrt(nn::add(nn::sum(nn::square(memory), {2}), 1e-6));
  const Tensor kn   = nn::sqrt(nn::add(nn::sum(nn::square(key), {1}, true), 1e-6));
  const Tensor cosine  = nn::div(dot, nn::mul(mn, kn));
  const Tensor beta    = nn::softplus(part(4, 1));
  const Tensor content = nn::softmax(nn::mul(beta, cosine), -1);
  const Tensor gate    = nn::sigmoid(part(5, 1));
  const Tensor interpolated = nn::add(nn::mul(gate, content), nn::mul(nn::sub(1.0, gate), previous));
  const Tensor shift = nn::softmax(part(6, 3), -1);
  const Tensor left  = nn::concat({Cols(interpolated, 1, 3), Cols(interpolated, 0, 1)}, 1);
  const Tensor right = nn::concat({Cols(interpolated, 3, 1), Cols(interpolated, 0, 3)}, 1);
  const Tensor shifted = nn::add(nn::add(nn::mul(left, Cols(shift, 0, 1)), nn::mul(interpolated, Cols(shift, 1, 1))),
                                 nn::mul(right, Cols(shift, 2, 1)));
  const Tensor gamma     = nn::add(nn::softplus(part(9, 1)), 1.0);
  const Tensor sharpened = nn::exp(nn::mul(gamma, nn::log(nn::add(shifted, 1e-8))));
  const Tensor weights   = nn::div(sharpened, nn::sum(sharpened, {1}, true));
  return {content, interpolated, shifted, weights};
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
  const auto controller = nn::lstm_step(p, "controller", nn::concat({input, previous.read}, 1), {previous.h, previous.c});
  const Tensor interface = nn::dense(p, "interface", controller.h, 28);
  const auto   write     = NtmAddressing(previous.memory, Cols(interface, 0, 10), previous.write_weights);
  const auto   read      = NtmAddressing(previous.memory, Cols(interface, 10, 10), previous.read_weights);
  const Tensor erase     = nn::sigmoid(Cols(interface, 20, 4));
  const Tensor add       = nn::tanh(Cols(interface, 24, 4));
  const Tensor weights   = nn::expand_dims(write.weights, 2);
  const Tensor memory    = nn::add(nn::mul(previous.memory, nn::sub(1.0, nn::mul(weights, nn::expand_dims(erase, 1)))),
                                   nn::mul(weights, nn::expand_dims(add, 1)));
  const Tensor value     = nn::sum(nn::mul(memory, nn::expand_dims(read.weights, 2)), {1});
  const Tensor logits    = nn::dense(p, "output", nn::concat({controller.h, value}, 1), 3);
  return {{controller.h, controller.c, memory, value, read.weights, write.weights}, logits, interface};
}

/// The nine-step unroll with valid steps t <= 2 * length (tflow's nn::Ntm).
/// Fields: logits [B,9,3], interface [B,9,28] (zero on invalid steps) and the
/// state traces h, c [B,10,8], memory [B,10,4,4], read, read_weights,
/// write_weights [B,10,4] (index 0 = initial state).
inline Outputs Ntm(nn::ParameterStore& p, const Tensor& x, const Tensor& lengths) {
  const std::int64_t batch   = x.shape()[0];
  const Tensor       initial = p.get("memory/initial", {4, 4}, nn::init::normal(.01));
  const Tensor       memory  = nn::add(nn::zeros({batch, 4, 4}), initial);
  const Tensor       weights = nn::add(nn::zeros({batch, 4}), nn::from_vector(std::vector<float>{1, 0, 0, 0}, {1, 4}));
  const Tensor       read    = nn::sum(nn::mul(memory, nn::expand_dims(weights, 2)), {1});
  NtmState state{nn::zeros({batch, 8}), nn::zeros({batch, 8}), memory, read, weights, weights};
  std::vector<Tensor> hs{state.h}, cs{state.c}, ms{memory}, rs{read}, rw{weights}, ww{weights}, logits, interfaces;
  const auto steps = nn::time_steps(x);
  for (int t = 0; t < 9; ++t) {
    const auto   next  = NtmCell(p, steps[t], state);
    const Tensor valid = nn::expand_dims(nn::greater(nn::add(nn::mul(nn::cast(lengths, nn::DType::Float32), 2.0), 1.0), double(t)), 1);
    auto choose        = [&](const Tensor& a, const Tensor& b) { return nn::where(valid, a, b); };
    state = {choose(next.state.h, state.h),
             choose(next.state.c, state.c),
             nn::where(nn::expand_dims(valid, 2), next.state.memory, state.memory),
             choose(next.state.read, state.read),
             choose(next.state.read_weights, state.read_weights),
             choose(next.state.write_weights, state.write_weights)};
    logits.push_back(choose(next.logits, nn::zeros_like(next.logits)));
    interfaces.push_back(choose(next.interface, nn::zeros_like(next.interface)));
    hs.push_back(state.h);
    cs.push_back(state.c);
    ms.push_back(state.memory);
    rs.push_back(state.read);
    rw.push_back(state.read_weights);
    ww.push_back(state.write_weights);
  }
  return {{"logits", nn::stack_time(logits)}, {"h", nn::stack_time(hs)},          {"c", nn::stack_time(cs)},
          {"memory", nn::stack(ms, 1)},       {"read", nn::stack_time(rs)},       {"read_weights", nn::stack_time(rw)},
          {"write_weights", nn::stack_time(ww)}, {"interface", nn::stack_time(interfaces)}};
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
