// CMD: just run-nn saved_model_inference
// Load a Python-trained model's weights and serve its affine function in C++.
//
// Port of tflow's saved_model_inference. Preparation (in tflow_examples):
// python3 scripts/train_export_affine.py
// That helper trains y=.75*x0-1.5*x1+.25 with GradientTape, exports a tf.Module
// with a dynamic batch dimension, and writes reference predictions and weights
// (weights.npz next to the SavedModel). The default export lives under
// ~/projects/tflow_examples/output/models/python_affine. Optional model=PATH
// selects another export: its saved_model directory or the export directory.
// When the default export is missing, the helper is run once automatically
// (inside tflow's Nix shell, writing to that default location).
//
// Validate the weights' dtypes/shapes against the serving contract
// (features [None,2] -> predictions [None,1]), compare four known
// predictions, run a different batch size and verify incompatible feature
// widths fail. No Python runtime is needed in this C++ process.
//
// Differences from tflow: loading a SavedModel (graph functions, variables
// and SignatureDefs) is out of scope for nn, so the C++ side reads the
// Python-trained variables from the export's weights.npz (nn::load_npz into a
// ParameterStore) and the serving function features @ weights + bias is
// written in nn ops. The signature check therefore validates the stored
// variables' shapes instead of the SignatureDef's tensor specs.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <datamunge/nn/nn.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "common/checks.hpp"
#include "common/paths.hpp"

namespace nn = datamunge::nn;
namespace fs = std::filesystem;
using nn::Tensor;

namespace {

/// tflow's preparation step, run once when the default export is missing:
/// python3 scripts/train_export_affine.py --tflite inside tflow's Nix shell.
inline void PrepareDefaultExport(const fs::path& dir) {
  const fs::path tflow = paths::DataRoot().parent_path().parent_path();
  if (fs::exists(dir / "weights.npz") || !fs::exists(tflow / "scripts" / "train_export_affine.py")) return;
  const std::string command = "cd '" + tflow.string() +
                              "' && nix develop --accept-flake-config --command python3 "
                              "scripts/train_export_affine.py --tflite --output '" + dir.string() + "'";
  std::cout << "Preparing " << dir.string() << " with tflow's train_export_affine.py\n";
  std::cout.flush();
  if (std::system(command.c_str()) != 0) std::cerr << "train_export_affine.py failed\n";
}
}  // namespace

int main(int argc, char** argv) {
  try {
    fs::path path = paths::DataRoot().parent_path() / "models" / "python_affine" / "saved_model";
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg.rfind("model=", 0) != 0 || arg.size() == 6) throw std::invalid_argument("expected model=PATH");
      path = arg.substr(6);
    }
    if (argc == 1) PrepareDefaultExport(path.parent_path());
    const fs::path export_dir = fs::exists(path / "weights.npz") ? path : path.parent_path();
    const fs::path weights    = export_dir / "weights.npz";
    check::Require(fs::exists(weights), weights.string() +
                                            " missing; run python3 scripts/train_export_affine.py in tflow_examples "
                                            "first, or pass model=PATH");
    nn::ParameterStore store;
    store.load(weights.string(), /*strict=*/false);
    check::Require(store.size() == 2 && store.contains("weights") && store.contains("bias"),
                   "named weights/bias contract required");
    const Tensor w = store.at("weights"), b = store.at("bias");
    check::Require(w.dtype() == nn::DType::Float32 && b.dtype() == nn::DType::Float32, "float32 serving contract");
    check::Require(w.shape() == nn::Shape({2, 1}), "features must have shape [None,2]");
    check::Require(b.shape() == nn::Shape({1}), "predictions must have shape [None,1]");
    nn::NoGradGuard guard;
    const auto serve = [&](const Tensor& features) { return nn::matmul(features, w) + b; };

    const Tensor query  = nn::tensor({{0, 0}, {1, 0}, {0, 1}, {2, -3}});
    const Tensor output = serve(query);
    check::Require(output.shape() == nn::Shape({4, 1}), "dynamic inference batch");
    std::cout << "== Python-trained affine model inference\n"
              << "  features -> " << weights.string() << " [weights " << nn::shape_str(w.shape()) << ", bias "
              << nn::shape_str(b.shape()) << "]\n"
              << "  predictions:";
    const std::vector<double> expected{.25, 1, -1.25, 6.25};
    const auto                values = output.to_vector<float>();
    for (int i = 0; i < 4; ++i) {
      check::Near(values[i], expected[i], 1e-5, "Python-trained affine prediction");
      std::cout << ' ' << values[i];
    }
    if (std::ifstream reference(export_dir / "reference.tsv"); reference) {
      double max_error = 0, x0, x1, prediction;
      while (reference >> x0 >> x1 >> prediction)
        max_error = std::max(max_error, std::abs(serve(nn::tensor({{x0, x1}})).item() - prediction));
      check::Near(max_error, 0, 1e-6, "match Python reference predictions");
      std::cout << "\n  Python reference max error=" << max_error;
    }
    check::Near(serve(nn::tensor({{-2, 4}})).item(), -7.25, 1e-5, "single-row inference");
    check::Throws([&] { serve(nn::tensor({{1, 2, 3}})); }, "wrong feature width must fail");
    std::cout << "\n  batch sizes 4 and 1 passed; incompatible width rejected\n"
                 "All SavedModel inference checks passed.\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
  return 0;
}
