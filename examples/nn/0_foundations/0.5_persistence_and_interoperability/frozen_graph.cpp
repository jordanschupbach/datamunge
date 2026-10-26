// CMD: just run-nn frozen_graph
// Freeze a trained model into a portable, inference-only weights artifact.
//
// Port of tflow's frozen_graph. Train a scalar affine regressor with Adam,
// retain only the parameters the prediction depends on, and export their
// evaluated values. Save the artifact, mutate the live model, then load the
// file into a completely fresh model whose parameters were never initialised
// (NaN). Predictions must match the trained snapshot exactly, including a new
// batch size; optimizer slots and the step counter must be absent.
//
// Differences from tflow: an eager library has no GraphDef, so "freezing"
// cannot embed constants into a serialized graph (loading a frozen GraphDef
// is out of scope for nn). The artifact is the inference function's
// parameters as .npz (output/nn/models/frozen_affine/weights.npz) plus
// signature.txt naming the input/output contract; the code that computes
// `predictions` ships with the consumer. Pruning is done by asking autograd
// which parameters the requested output reaches (nn::grad with allow_unused)
// instead of walking GraphDef ancestors, and "prune training subgraph" is
// checked against the full training state (parameters + nn::Optimizer
// slots). Device placement has no counterpart (nn is CPU only).

#include <cmath>
#include <datamunge/nn/nn.hpp>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include "common/checks.hpp"
#include "common/paths.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;

namespace {

struct Affine {
  nn::ParameterStore store;
  Tensor             w, b;
  explicit Affine(double initial)
      : w(store.get("model/w", {}, nn::init::constant(initial))),
        b(store.get("model/b", {}, nn::init::constant(initial))) {}
  Tensor Predict(const Tensor& x) const { return x * w + b; }
};

/// The model's named outputs, each a function of the features (and labels).
using Endpoint = std::function<Tensor(const Tensor& x, const Tensor& y)>;

/// The values of every store parameter the named outputs depend on; unknown
/// output names throw std::invalid_argument (TF's FreezeGraph contract).
nn::NamedTensors Freeze(const nn::ParameterStore& store, const std::map<std::string, Endpoint>& endpoints,
                        const std::vector<std::string>& outputs) {
  const Tensor probe = nn::tensor({1});
  std::vector<bool> used(store.size(), false);
  for (const auto& name : outputs) {
    auto it = endpoints.find(name);
    if (it == endpoints.end()) throw std::invalid_argument("Freeze: unknown output '" + name + "'");
    nn::GradOptions options;
    options.allow_unused = true;
    const auto grads = nn::grad(nn::sum(it->second(probe, probe)), store.parameters(), options);
    for (std::size_t i = 0; i < grads.size(); ++i) used[i] = used[i] || grads[i].defined();
  }
  nn::NamedTensors frozen;
  for (std::size_t i = 0; i < store.size(); ++i)
    if (used[i]) frozen.emplace_back(store.names()[i], store.parameters()[i].detach().clone());
  return frozen;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int epochs = 500;
    for (int i = 1; i < argc; ++i) {
      const std::string argument(argv[i]);
      if (argument.rfind("epochs=", 0) != 0) throw std::invalid_argument("expected epochs=N");
      epochs = std::stoi(argument.substr(7));
    }
    check::Require(epochs > 0, "positive training budget required");
    Affine        model(0);
    nn::Optimizer optimizer(model.store, .05, nn::OptimizerConfig{nn::OptimizerKind::Adam});
    const std::map<std::string, Endpoint> endpoints{
        {"predictions", [&](const Tensor& x, const Tensor&) { return model.Predict(x); }},
        {"loss", [&](const Tensor& x, const Tensor& y) { return nn::mean(nn::square(model.Predict(x) - y)); }},
    };
    const Tensor train_x = nn::tensor({-2, -1, 0, 1, 2}), train_y = nn::tensor({-4.5, -2.5, -.5, 1.5, 3.5});
    for (int i = 0; i < epochs; ++i) optimizer.minimize(endpoints.at("loss")(train_x, train_y));
    const Tensor query = nn::tensor({-3, .5, 4});
    std::vector<float> expected;
    {
      nn::NoGradGuard guard;
      expected = model.Predict(query).to_vector<float>();
    }
    const auto q = query.to_vector<double>();
    for (int i = 0; i < 3; ++i) {
      check::Require(std::isfinite(expected[i]), "finite prediction");
      if (epochs >= 500) check::Near(expected[i], 2 * q[i] - .5, 1e-5, "trained affine regressor");
    }
    const double new_batch = model.Predict(nn::tensor({5})).item();

    // Full training state, the analogue of TF's source GraphDef.
    nn::NamedTensors training;
    for (std::size_t i = 0; i < model.store.size(); ++i)
      training.emplace_back("params/" + model.store.names()[i], model.store.parameters()[i].detach());
    for (const auto& [name, value] : optimizer.state()) training.emplace_back("optimizer/" + name, value);
    const auto frozen = Freeze(model.store, endpoints, {"predictions"});
    check::Require(frozen.size() < training.size(), "prune training subgraph");
    for (const auto& [name, value] : frozen)
      check::Require(name.rfind("model/", 0) == 0 && name.find("optimizer") == std::string::npos &&
                         nn::is_floating(value.dtype()),
                     "no mutable/optimizer state in artifact");

    const auto dir  = paths::OutputDir("models/frozen_affine");
    const auto path = dir / "weights.npz";
    nn::save_npz(path.string(), frozen);
    {
      std::ofstream contract(dir / "signature.txt");
      contract << "input=features float32 [batch]\noutput=predictions float32 [batch]\n"
                  "predictions = features * model/w + model/b\n";
    }
    {
      nn::NoGradGuard guard;
      model.w.copy_(nn::scalar(-100));
      model.b.copy_(nn::scalar(100));
    }

    Affine serving(std::numeric_limits<double>::quiet_NaN());  // never initialised
    serving.store.load(path.string(), /*strict=*/true);
    nn::NoGradGuard guard;
    const auto restored = serving.Predict(query).to_vector<float>();
    for (int i = 0; i < 3; ++i) check::Near(restored[i], expected[i], 0, "exact frozen/live prediction agreement");
    check::Near(serving.Predict(nn::tensor({5})).item(), new_batch, 0, "new batch shape");
    if (epochs >= 500) check::Near(new_batch, 9.5, 1e-5, "converged new batch");
    check::Throws([&] { Freeze(model.store, endpoints, {"does_not_exist"}); }, "unknown export output must fail");
    std::cout << "== Frozen weights export\n"
              << "  training state arrays=" << training.size() << ", frozen arrays=" << frozen.size() << '\n'
              << "  predictions=(" << expected[0] << ',' << expected[1] << ',' << expected[2]
              << "), exact after reload\n"
              << "  fresh model needs no initialization; live-model mutations do not affect artifact\n"
              << "  wrote " << path.string() << " (" << std::filesystem::file_size(path)
              << " bytes) and signature.txt\n"
              << "All frozen-graph checks passed.\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
  return 0;
}
