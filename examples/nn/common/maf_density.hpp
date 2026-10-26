// MAF: strict degree-masked Gaussian autoregression on two coordinates, a parallel
// density pass and a sequential exact inverse. Port of tflow's
// common/nn/maf_density.h and common/ml/maf_density.h (harness in
// flow_density.hpp). The masks follow MADE (common/made_density.hpp has the
// 64-pixel version): input degrees (1, 2) (reversed for odd blocks), 16 hidden
// units of degree floor(j / 8), heads (mu, raw s) per coordinate with the
// coordinate's degree; input <= hidden and STRICT hidden < output, so a head never
// sees its own or a later coordinate. W ~ N(0, .3) / N(0, .01), b = .05 / 0,
// s = .8 tanh(raw). Store seed 9603 (IAF: 9703).
#pragma once

#include "common/real_nvp_density.hpp"

namespace flowd {

/// tflow's MafInventory(block): degrees (int32) and the two masks (float).
inline Map MafInventory(int block) {
  std::vector<std::int32_t> input(2), hidden(16), output(4);
  std::vector<float>        im(2 * 16), om(16 * 4);
  for (int c = 0; c < 2; ++c) input[c] = block % 2 ? 2 - c : c + 1;
  for (int j = 0; j < 16; ++j) hidden[j] = j / 8;
  for (int c = 0; c < 4; ++c) output[c] = input[c % 2];
  for (int c = 0; c < 2; ++c)
    for (int j = 0; j < 16; ++j) im[c * 16 + j] = input[c] <= hidden[j];
  for (int j = 0; j < 16; ++j)
    for (int c = 0; c < 4; ++c) om[j * 4 + c] = hidden[j] < output[c];
  return {{"input_degrees", nn::from_vector(input, {2})},   {"hidden_degrees", nn::from_vector(hidden, {16})},
          {"output_degrees", nn::from_vector(output, {4})}, {"input_mask", nn::from_vector(im, {2, 16})},
          {"output_mask", nn::from_vector(om, {16, 4})}};
}

struct MafConditionerPath {
  Tensor mean, scale;
};
inline MafConditionerPath MafConditioner(nn::ParameterStore& p, const Tensor& x, int block) {
  const std::string name  = "autoregressive" + std::to_string(block);
  const auto        masks = MafInventory(block);
  const Tensor      iw = p.get(name + "/hidden/W", {2, 16}, nn::init::normal(.3)), ib = p.get(name + "/hidden/b", {16}, nn::init::constant(.05)),
               ow = p.get(name + "/heads/W", {16, 4}, nn::init::normal(.01)), ob = p.get(name + "/heads/b", {4}, nn::init::zeros());
  const Tensor hidden = nn::tanh(nn::bias_add(nn::matmul(x, iw * masks.at("input_mask")), ib));
  const Tensor heads  = nn::bias_add(nn::matmul(hidden, ow * masks.at("output_mask")), ob);
  return {Col(heads, 0, 2), 0.8 * nn::tanh(Col(heads, 2, 2))};
}

/// Fields shared by MAF and IAF, per block in tflow's names.
struct MafStages {
  std::vector<Tensor> stages, logdet_stages, scale_stages, shift_stages, conditioner_inputs;
  explicit MafStages(int n = 4) : stages(n), logdet_stages(n), scale_stages(n), shift_stages(n), conditioner_inputs(n) {}
  void To(Named& f) const {
    for (int i = 0; i < 4; ++i) {
      f.emplace_back("conditioner_input" + std::to_string(i), conditioner_inputs[i]);
      CouplingFields(f, i, stages[i], logdet_stages[i], scale_stages[i], shift_stages[i]);
    }
  }
};

/// One sequential autoregressive pass over the two coordinates of block i, in its
/// dependency order: value_c = step(c, x_c, conditioner(recovered so far)). Adds
/// each used scale column to `ld` when given.
template <typename Step>
Tensor Sequential(nn::ParameterStore& p, const Tensor& x, int i, Step step, Tensor* ld = nullptr) {
  Tensor recovered = x * 0.0;
  for (int ordinal = 0; ordinal < 2; ++ordinal) {
    const int    coordinate = i % 2 ? 1 - ordinal : ordinal;
    const auto   c          = MafConditioner(p, recovered, i);
    const Tensor scale      = Col(c.scale, coordinate);
    const Tensor value      = step(Col(x, coordinate), Col(c.mean, coordinate), scale);
    const Tensor other      = Col(recovered, 1 - coordinate);
    recovered               = coordinate ? nn::concat({other, value}, 1) : nn::concat({value, other}, 1);
    if (ld) *ld = *ld + nn::reshape(scale, {-1});
  }
  return recovered;
}

/// tflow's conditioner_jacobian<i>_<k>: d sum(head k) / d conditioner input, k = mu0, mu1, s0, s1
/// (the conditioner re-run on a differentiable copy of its recorded input).
inline void ConditionerJacobians(nn::ParameterStore& p, const MafStages& st, Map& out) {
  for (int i = 0; i < 4; ++i) {
    Tensor input = st.conditioner_inputs[i].detach().clone();
    input.requires_grad_();
    const auto c = MafConditioner(p, input, i);
    for (int k = 0; k < 4; ++k)
      out["conditioner_jacobian" + std::to_string(i) + "_" + std::to_string(k)] =
          nn::grad(nn::sum(Col(k < 2 ? c.mean : c.scale, k % 2)), {input}, nn::GradOptions{true, false, false})[0];
  }
}

class MafDensity : public FlowModel {
 public:
  explicit MafDensity(bool training = true) : FlowModel(9603) { Build(training); }
  FlowPath Forward(const Tensor& x) override {
    FlowPath  out;
    MafStages st;
    Tensor    h = x, ld = ZeroLogdet(x);
    for (int i = 0; i < 4; ++i) {
      const Tensor observed     = h;
      const auto   c            = MafConditioner(store, observed, i);
      h                         = (observed - c.mean) * nn::exp(-c.scale);
      const Tensor contribution = -nn::sum(c.scale, {1});
      ld                        = ld + contribution;
      st.conditioner_inputs[i] = observed, st.stages[i] = h, st.scale_stages[i] = c.scale, st.shift_stages[i] = c.mean,
      st.logdet_stages[i] = contribution;
    }
    GlobalScale(store, out, h, ld);
    st.To(out.fields);
    return out;
  }
  FlowInverse Inverse(const Tensor& z) override {
    auto [h, ld] = UnscaleGlobal(store, z);
    for (int i = 3; i >= 0; --i)
      h = Sequential(
          store, h, i, [](const Tensor& base, const Tensor& mean, const Tensor& scale) { return base * nn::exp(scale) + mean; }, &ld);
    return {h, ld};
  }
  void Extra(const Tensor&, const FlowPath& path, Map& out) override { ConditionerJacobians(store, StagesOf(path), out); }

  static MafStages StagesOf(const FlowPath& path) {
    MafStages st;
    for (const auto& [k, v] : path.fields)
      if (k.rfind("conditioner_input", 0) == 0) st.conditioner_inputs[k.back() - '0'] = v;
    return st;
  }

 protected:
  struct Unbuilt {};
  /// For subclasses that build themselves (IAF).
  MafDensity(Unbuilt, std::uint64_t seed) : FlowModel(seed) {}
};

}  // namespace flowd
