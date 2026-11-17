// Generic eager harness for tflow's small explicit-unroll recurrent models
// (the ml::<Name>Model classes of tflow's common/ml/{bidirectional_rnn,
// clockwork_rnn,conv_lstm,gated_sequence,gradient_clipping,indrnn,mgu}.h).
//
// Every tflow model there has the same shape: placeholders for a sequence
// batch, its valid lengths and zero initial states; a cell unrolled with an
// explicit length mask; a head; a loss and a "chosen logit" whose gradients
// with respect to the inputs and initial states are fetched next to the
// forward values; Adam (or SGD with a gradient policy) over every parameter;
// a SaveV2 checkpoint. Here the graph is the `Forward` function, the session
// fetches are eager calls, gradients come from nn::grad and checkpoints are
// .npz files of the parameter store.
//
// A Forward returns named tensors and must contain "logits". Evaluate()
// returns the forward fields, "probability", "loss", "input_gradient", one
// gradient per initial state (named by `initial_gradient_names`) and any
// probe gradients, exactly like tflow's Evaluate fetches.
#pragma once

#include <cstring>
#include <datamunge/nn/nn.hpp>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "common/checks.hpp"

namespace rnn_cells {

namespace nn = datamunge::nn;
using nn::Tensor;
using Outputs = std::map<std::string, Tensor>;

/// (store, input, lengths, initial states) -> named outputs incl. "logits".
using Forward = std::function<Outputs(nn::ParameterStore&, const Tensor&, const Tensor&, const std::vector<Tensor>&)>;
/// (outputs, labels) -> {loss, chosen}: the training loss and the scalar whose
/// input/initial-state gradients Evaluate reports.
using LossFn = std::function<std::pair<Tensor, Tensor>(const Outputs&, const Tensor&)>;

/// A gradient probe: d target(outputs) / d (input when wrt == 0, else initial state wrt - 1).
struct Probe {
  std::string                           name;
  std::function<Tensor(const Outputs&)> target;
  int                                   wrt{0};
};

/// One batch fed to a model: inputs, int32 lengths and labels.
struct Feed {
  Tensor inputs, lengths, labels;
};

/// Softmax cross-entropy over two classes with int32 labels, and the summed
/// target logit (tflow's `loss` and `chosen`).
inline std::pair<Tensor, Tensor> SoftmaxLoss(const Outputs& out, const Tensor& labels) {
  const Tensor& logits  = out.at("logits");
  const Tensor  one_hot = nn::one_hot(labels, logits.shape().back());
  const Tensor  loss    = nn::neg(nn::mean(nn::sum(nn::mul(one_hot, nn::log_softmax(logits, -1)), {1}), {0}));
  return {loss, nn::sum(nn::mul(logits, one_hot))};
}

inline bool SameBits(const Tensor& a, const Tensor& b) {
  if (a.dtype() != b.dtype() || a.shape() != b.shape()) return false;
  auto same = [](const auto& x, const auto& y) {
    return x.size() == y.size() && std::memcmp(x.data(), y.data(), x.size() * sizeof(x[0])) == 0;
  };
  switch (a.dtype()) {
    case nn::DType::Float32: return same(a.to_vector<float>(), b.to_vector<float>());
    case nn::DType::Float64: return same(a.to_vector<double>(), b.to_vector<double>());
    case nn::DType::Int32: return same(a.to_vector<std::int32_t>(), b.to_vector<std::int32_t>());
    default: return same(a.to_vector<std::int64_t>(), b.to_vector<std::int64_t>());
  }
}

inline bool SameBits(const nn::NamedTensors& a, const nn::NamedTensors& b) {
  if (a.size() != b.size()) return false;
  for (std::size_t i = 0; i < a.size(); ++i)
    if (a[i].first != b[i].first || !SameBits(a[i].second, b[i].second)) return false;
  return true;
}

inline nn::NamedTensors Named(const Outputs& out) { return {out.begin(), out.end()}; }

inline Tensor Find(const nn::NamedTensors& arrays, const std::string& name) {
  for (const auto& [key, value] : arrays)
    if (key == name) return value;
  throw std::runtime_error("missing array " + name);
}

/// A gradient from nn::grad (detached), or zeros shaped like `like` when the
/// target did not depend on it (allow_unused).
inline Tensor DefinedGradient(const Tensor& g, const Tensor& like) {
  return g.defined() ? g.detach() : nn::zeros_like(like.detach());
}

inline double MaxAbsDiff(const Tensor& a, const Tensor& b) {
  const auto x = a.to_vector<double>(), y = b.to_vector<double>();
  if (x.size() != y.size()) return INFINITY;
  double m = 0;
  for (std::size_t i = 0; i < x.size(); ++i) m = std::max(m, std::abs(x[i] - y[i]));
  return m;
}

class RecurrentModel {
 public:
  nn::ParameterStore                 store;
  Forward                            forward;
  LossFn                             loss_fn = SoftmaxLoss;
  std::function<Tensor(const Tensor&)> probability = [](const Tensor& logits) { return nn::softmax(logits, -1); };
  /// Per-example shapes of the input and of each zero initial state.
  nn::Shape                          input_shape;
  std::vector<nn::Shape>             initial_shapes;
  /// Names of the initial-state gradients in Evaluate (empty: not reported).
  std::vector<std::string>           initial_gradient_names;
  /// Forward fields returned by Predict (Evaluate returns these plus losses and gradients).
  std::vector<std::string>           fields;
  std::vector<Probe>                 evaluate_probes;
  /// Optional transform of the raw parameter gradients before the optimizer (gradient policies).
  std::function<std::vector<Tensor>(const std::vector<Tensor>&)> gradient_policy;
  /// Optional projection after each update (IndRNN coefficient bound).
  std::function<void(nn::ParameterStore&)> after_update;
  /// Extra named inputs a Forward may read through a captured model pointer
  /// (tflow's additional placeholders: dropout masks, timestamps, leak). Set by
  /// the caller around Evaluate/Gradient/Update; empty by default.
  std::map<std::string, Tensor>      side_inputs;
  /// Optional extra Evaluate fields (e.g. gradients w.r.t. a side input), added to the result.
  std::function<void(Outputs& result, const Feed&)> evaluate_extra;
  std::unique_ptr<nn::Optimizer>     optimizer;

  RecurrentModel(std::uint64_t seed, Forward f, nn::Shape input, std::vector<nn::Shape> initials,
                 std::vector<std::string> initial_names, std::vector<std::string> fetch)
      : store(seed), forward(std::move(f)), input_shape(std::move(input)), initial_shapes(std::move(initials)),
        initial_gradient_names(std::move(initial_names)), fields(std::move(fetch)) {}

  /// Creates every parameter (tflow builds them with the graph) by running one
  /// example through the forward pass; with `training`, adds the optimizer.
  void Build(bool training, double learning_rate = .01, nn::OptimizerKind kind = nn::OptimizerKind::Adam) {
    {
      nn::NoGradGuard guard;
      nn::Shape       shape{1};
      shape.insert(shape.end(), input_shape.begin(), input_shape.end());
      forward(store, nn::zeros(shape), nn::zeros({1}, nn::DType::Int32), Initials(1));
    }
    if (training) {
      nn::OptimizerConfig config;
      config.kind = kind;
      optimizer   = std::make_unique<nn::Optimizer>(store, learning_rate, config);
    }
  }

  std::vector<Tensor> Initials(std::int64_t n) const {
    std::vector<Tensor> out;
    for (const auto& s : initial_shapes) {
      nn::Shape shape{n};
      shape.insert(shape.end(), s.begin(), s.end());
      out.push_back(nn::zeros(shape));
    }
    return out;
  }

  /// Sequence and lengths only, no labels (tflow's Predict).
  Outputs Predict(const Feed& x) {
    nn::NoGradGuard guard;
    Outputs         out = forward(store, x.inputs, x.lengths, Initials(x.inputs.shape()[0]));
    Outputs         result;
    for (const auto& f : fields) result[f] = f == "probability" ? probability(out.at("logits")) : out.at(f);
    return result;
  }

  // Transitional bundles (forward + loss + gradients [+ optimizer step]).
  // The 4.1 examples bidirectional_rnn, clockwork_rnn, conv_lstm,
  // gradient_clipping, gru, indrnn, layer_norm_lstm, lstm and mgu write these
  // steps out themselves; Evaluate/ProbeGradients/Gradient/RawGradients/Update
  // remain only for the examples that have not been unbundled yet (the other
  // 4.1 cells and 4.3 memory networks). Delete them once nothing uses them.

  /// Forward fields, loss and the full-BPTT input/initial-state gradients of the chosen logit.
  Outputs Evaluate(const Feed& x) {
    Tensor              input = x.inputs.detach().clone();
    std::vector<Tensor> initials = Initials(x.inputs.shape()[0]);
    input.requires_grad_(true);
    for (auto& s : initials) s.requires_grad_(true);
    Outputs out = forward(store, input, x.lengths, initials);
    out["probability"] = probability(out.at("logits"));
    const auto [loss, chosen] = loss_fn(out, x.labels);
    std::vector<Tensor> wrt{input};
    wrt.insert(wrt.end(), initials.begin(), initials.end());
    nn::GradOptions options;
    options.retain_graph = true;
    options.allow_unused = true;
    const auto g = nn::grad(std::vector<Tensor>{chosen}, wrt, {}, options);
    Outputs    result;
    for (const auto& f : fields) result[f] = out.at(f).detach();
    result["loss"]           = loss.detach();
    result["input_gradient"] = Defined(g[0], input);
    for (std::size_t k = 0; k < initial_gradient_names.size(); ++k)
      if (!initial_gradient_names[k].empty()) result[initial_gradient_names[k]] = Defined(g[k + 1], initials[k]);
    for (const auto& probe : evaluate_probes) {
      const Tensor& target = wrt.at(static_cast<std::size_t>(probe.wrt));
      result[probe.name]   = Defined(nn::grad(std::vector<Tensor>{probe.target(out)}, {target}, {}, options)[0], target);
    }
    if (evaluate_extra) evaluate_extra(result, x);
    return result;
  }

  /// Gradients of each probe target (tflow's ModuleGradients / ChannelGradients).
  Outputs ProbeGradients(const Feed& x, const std::vector<Probe>& probes) {
    Tensor              input = x.inputs.detach().clone();
    std::vector<Tensor> initials = Initials(x.inputs.shape()[0]);
    input.requires_grad_(true);
    for (auto& s : initials) s.requires_grad_(true);
    const Outputs   out = forward(store, input, x.lengths, initials);
    std::vector<Tensor> wrt{input};
    wrt.insert(wrt.end(), initials.begin(), initials.end());
    nn::GradOptions options;
    options.retain_graph = true;
    options.allow_unused = true;
    Outputs result;
    for (const auto& probe : probes) {
      const Tensor& target = wrt.at(static_cast<std::size_t>(probe.wrt));
      result[probe.name]   = Defined(nn::grad(std::vector<Tensor>{probe.target(out)}, {target}, {}, options)[0], target);
    }
    return result;
  }

  /// Raw loss gradient of every parameter, by name.
  nn::NamedTensors Gradient(const Feed& x) {
    const auto           g = RawGradients(x);
    nn::NamedTensors     result;
    for (std::size_t i = 0; i < g.size(); ++i) result.emplace_back(store.names()[i], g[i]);
    return result;
  }

  std::vector<Tensor> RawGradients(const Feed& x) {
    Outputs out = forward(store, x.inputs, x.lengths, Initials(x.inputs.shape()[0]));
    const auto params = store.parameters();
    nn::GradOptions options;
    options.allow_unused = true;
    auto g = nn::grad(std::vector<Tensor>{loss_fn(out, x.labels).first}, params, {}, options);
    for (std::size_t i = 0; i < g.size(); ++i) g[i] = Defined(g[i], params[i]);
    return g;
  }

  void Update(const Feed& x) {
    auto g = RawGradients(x);
    if (gradient_policy) g = gradient_policy(g);
    optimizer->step(g);
    if (after_update) after_update(store);
  }

  /// Copies of every parameter, by name.
  nn::NamedTensors Arrays() const {
    nn::NamedTensors out;
    for (const auto& name : store.names()) out.emplace_back(name, store.at(name).detach().clone());
    return out;
  }

  void Save(const std::filesystem::path& file) const { store.save(file.string()); }
  void Restore(const std::filesystem::path& file) { store.load(file.string(), true); }

  /// Overwrites every parameter from an .npz written by tflow (initial_arrays.npz).
  void LoadArrays(const std::filesystem::path& file) {
    const auto arrays = nn::load_npz(file.string());
    check::Require(arrays.size() == store.size(), "init arrays cover exactly the model's parameters");
    for (const auto& [name, value] : arrays) {
      check::Require(store.contains(name), "init array " + name + " is a model parameter");
      store.set(name, nn::cast(value, store.at(name).dtype()));
    }
  }

 private:
  static Tensor Defined(const Tensor& g, const Tensor& like) { return DefinedGradient(g, like); }
};

}  // namespace rnn_cells
