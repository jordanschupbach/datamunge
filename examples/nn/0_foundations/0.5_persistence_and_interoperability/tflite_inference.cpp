// CMD: just run-nn tflite_inference
// Run a Python-trained, exported affine model from C++ with preallocated buffers.
//
// Port of tflow's tflite_inference. Preparation (in tflow_examples):
// python3 scripts/train_export_affine.py --tflite
// The export helper trains an affine model, converts its serving signature,
// and writes independent Python reference predictions (reference.tsv) and the
// trained variables (weights.npz). This C++ executable uses no Python
// interpreter. Optional dir=PATH selects the export directory (default
// ~/projects/tflow_examples/output/models/python_affine; when it is missing
// the helper is run once automatically inside tflow's Nix shell).
//
// Inspect the signature tensor types, allocate input/output buffers once,
// copy input in, run synchronously, and read back output. Reuse one "compiled"
// model and its buffers for four rows and a repeat. Like tflow this uses a
// batch size of one; it does not resize dynamic inputs.
//
// Differences from tflow: loading a .tflite flatbuffer through LiteRT is out
// of scope for nn (the converted model is float32, not quantised, so there is
// no quantised arithmetic to reproduce). The compiled model is the export's
// weights.npz bound to an nn affine function with fixed [1,2] / [1,1] float32
// buffers; "invalid flatbuffer must fail" becomes "invalid weights archive
// must fail" (16 zero bytes written to output/nn/models/tflite_inference/).

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

void Shape(const Tensor& t, std::int64_t width) {
  check::Require(t.dtype() == nn::DType::Float32 && t.shape() == nn::Shape({1, width}) && t.is_contiguous(),
                 "expected dense float32 tensor [1," + std::to_string(width) + "]");
}

/// Weights plus the buffers they run on, allocated once and reused.
struct CompiledModel {
  Tensor weights, bias, input = nn::zeros({1, 2}), output = nn::zeros({1, 1});
  explicit CompiledModel(const fs::path& file) {
    const auto arrays = nn::load_npz(file.string());
    for (const auto& [name, value] : arrays) {
      if (name == "weights") weights = value;
      if (name == "bias") bias = value;
    }
    check::Require(arrays.size() == 2 && weights.defined() && bias.defined(), "one input/output required");
    check::Require(weights.shape() == nn::Shape({2, 1}) && bias.shape() == nn::Shape({1}),
                   "features/predictions signature");
  }
  void Run() {
    nn::NoGradGuard guard;
    output.copy_(nn::matmul(input, weights) + bias);
  }
};


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
    fs::path dir = paths::DataRoot().parent_path() / "models" / "python_affine";
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg.rfind("dir=", 0) != 0 || arg.size() == 4) throw std::invalid_argument("expected dir=PATH");
      dir = arg.substr(4);
    }
    if (argc == 1) PrepareDefaultExport(dir);
    check::Require(fs::exists(dir / "weights.npz"),
                   (dir / "weights.npz").string() +
                       " missing; run python3 scripts/train_export_affine.py --tflite in tflow_examples first");
    std::ifstream reference(dir / "reference.tsv");
    check::Require(bool(reference), "Python reference.tsv required");
    struct Row {
      float x0, x1, prediction;
    };
    std::vector<Row> rows;
    Row              row{};
    while (reference >> row.x0 >> row.x1 >> row.prediction) rows.push_back(row);
    check::Require(reference.eof() && rows.size() == 4, "expected four Python reference rows");

    CompiledModel model(dir / "weights.npz");
    Shape(model.input, 2);
    Shape(model.output, 1);
    const std::size_t in_bytes  = model.input.numel() * sizeof(float);
    const std::size_t out_bytes = model.output.numel() * sizeof(float);
    check::Require(in_bytes >= 2 * sizeof(float) && out_bytes >= sizeof(float), "buffers must cover tensor data");
    std::cout << "== Exported affine model inference with reusable buffers\n"
              << "  model bytes=" << fs::file_size(dir / "weights.npz") << "; input/output buffer bytes=" << in_bytes
              << '/' << out_bytes << '\n';
    double max_error = 0;
    rows.push_back(rows.front());  // Reuse buffers after a completely different input.
    for (const auto& r : rows) {
      const float features[]{r.x0, r.x1};
      model.input.copy_(nn::tensor({{features[0], features[1]}}));
      model.Run();
      const float prediction = model.output.to_vector<float>()[0];
      check::Near(prediction, r.prediction, 1e-6, "match Python prediction");
      check::Near(prediction, .75 * features[0] - 1.5 * features[1] + .25, 1e-5, "trained affine target");
      max_error = std::max(max_error, std::abs(double(prediction) - r.prediction));
      std::cout << "  [" << features[0] << ',' << features[1] << "] -> " << prediction << '\n';
    }
    const auto junk = paths::OutputDir("models/tflite_inference") / "invalid.npz";
    {
      std::ofstream out(junk, std::ios::binary | std::ios::trunc);
      const char    zeros[16]{};
      out.write(zeros, sizeof(zeros));
    }
    check::Throws([&] { CompiledModel invalid(junk); }, "invalid weights archive must fail");
    std::cout << "  Python max error=" << max_error << "; invalid archive rejected\n"
                 "All TFLite inference checks passed.\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
  return 0;
}
