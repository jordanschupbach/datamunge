// CMD: just run-nn checkpoint_io
// Checkpoints with float, int64 and (byte-encoded) string variables.
//
// Port of tflow's checkpoint_io. Save a weight matrix, a counter larger than
// float32 can represent exactly, and vocabulary strings (including UTF-8 and
// an embedded NUL). Mutate every variable, restore, then load into a fresh
// set of variables that were never initialised (NaN / sentinel values).
// Compare serialized tensor snapshots and predictions exactly. Inspect the
// checkpoint's metadata and verify missing tensor names fail before any
// assignments run.
//
// Keys are variable names, so reconstructed models must preserve names,
// dtypes and intended shapes. This is a model-state checkpoint, not a complete
// process snapshot; save samplers/RNG state separately. For optimizer state and
// training continuation, see the early_stopping_checkpoints example in 0.3.
//
// Differences from tflow: TF's SaveV2/RestoreV2 tensor bundle becomes one .npz
// archive (nn::save_npz/load_npz) and the TensorProto snapshot becomes the
// NPY encoding of each tensor (nn::encode_npy). nn has no string dtype, so the
// vocabulary is stored as two numeric variables, "vocabulary/bytes" (int32
// [2,16], zero padded UTF-8 bytes) and "vocabulary/lengths" (int64 [2]); the
// byte-exact UTF-8 / embedded-NUL checks run on the decoded strings. The
// checkpoint is written to output/nn/checkpoints/checkpoint_io/state.npz.

#include <cstdint>
#include <datamunge/nn/nn.hpp>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "common/checks.hpp"
#include "common/paths.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;

namespace {

constexpr std::int64_t kCounter = (std::int64_t{1} << 40) + 7;
constexpr std::int64_t kWidth   = 16;  // bytes per vocabulary entry

/// TF's nn::Checkpoint for eager tensors: named variables saved to / restored
/// from one .npz archive. Restore validates every name, dtype and shape before
/// it assigns anything, so a failed restore leaves the variables untouched.
class Checkpoint {
 public:
  explicit Checkpoint(std::vector<std::pair<std::string, Tensor>> variables) : variables_(std::move(variables)) {}

  void Save(const std::string& path) const {
    nn::NamedTensors arrays;
    for (const auto& [name, tensor] : variables_) arrays.emplace_back(name, tensor);
    nn::save_npz(path, arrays);
  }

  void Restore(const std::string& path) {
    const auto           arrays = nn::load_npz(path);
    std::vector<Tensor> values;
    for (const auto& [name, variable] : variables_) {
      const Tensor* found = nullptr;
      for (const auto& [key, value] : arrays)
        if (key == name) found = &value;
      if (!found) throw std::invalid_argument("checkpoint " + path + " has no tensor '" + name + "'");
      if (found->dtype() != variable.dtype() || found->shape() != variable.shape())
        throw std::invalid_argument("checkpoint tensor '" + name + "' is " + nn::dtype_name(found->dtype()) + ' ' +
                                    nn::shape_str(found->shape()) + ", variable is " +
                                    nn::dtype_name(variable.dtype()) + ' ' + nn::shape_str(variable.shape()));
      values.push_back(*found);
    }
    for (std::size_t i = 0; i < values.size(); ++i) variables_[i].second.copy_(values[i]);
  }

 private:
  std::vector<std::pair<std::string, Tensor>> variables_;
};

/// Fixed-width byte encoding of a string vector (nn has no string dtype).
void AssignStrings(Tensor& bytes, Tensor& lengths, const std::vector<std::string>& words) {
  std::vector<std::int32_t> data(words.size() * kWidth, 0);
  std::vector<std::int64_t> sizes;
  for (std::size_t i = 0; i < words.size(); ++i) {
    check::Require(static_cast<std::int64_t>(words[i].size()) <= kWidth, "vocabulary entry too long");
    for (std::size_t j = 0; j < words[i].size(); ++j)
      data[i * kWidth + j] = static_cast<unsigned char>(words[i][j]);
    sizes.push_back(static_cast<std::int64_t>(words[i].size()));
  }
  bytes.copy_(nn::from_vector(data, {static_cast<std::int64_t>(words.size()), kWidth}));
  lengths.copy_(nn::from_vector(sizes, {static_cast<std::int64_t>(words.size())}));
}

std::vector<std::string> Strings(const Tensor& bytes, const Tensor& lengths) {
  const auto data  = bytes.to_vector<std::int32_t>();
  const auto sizes = lengths.to_vector<std::int64_t>();
  std::vector<std::string> words;
  for (std::size_t i = 0; i < sizes.size(); ++i) {
    std::string word;
    for (std::int64_t j = 0; j < sizes[i]; ++j) word.push_back(static_cast<char>(data[i * kWidth + j]));
    words.push_back(word);
  }
  return words;
}

struct Model {
  Tensor     weights = nn::full({2, 2}, std::numeric_limits<double>::quiet_NaN());
  Tensor     step    = nn::scalar(0, nn::DType::Int64);
  Tensor     bytes   = nn::full({2, kWidth}, -1, nn::DType::Int32);
  Tensor     lengths = nn::full({2}, -1, nn::DType::Int64);
  Checkpoint checkpoint;

  explicit Model(bool wrong_name = false)
      : checkpoint({{"weights", weights},
                    {wrong_name ? "renamed_step" : "step", step},
                    {"vocabulary/bytes", bytes},
                    {"vocabulary/lengths", lengths}}) {}

  void Init() {
    weights.copy_(nn::tensor({{1, 2}, {3, 4}}));
    step.copy_(nn::scalar(static_cast<double>(kCounter), nn::DType::Int64));  // exact in double
    AssignStrings(bytes, lengths, {"café", std::string("a\0b", 3)});
  }
  void Mutate() {
    weights.copy_(nn::full({2, 2}, -9));
    step.copy_(nn::scalar(-1, nn::DType::Int64));
    AssignStrings(bytes, lengths, {"changed", "changed"});
  }
  Tensor Predict(const Tensor& features) const { return nn::matmul(features, weights); }
  std::vector<std::string> Vocabulary() const { return Strings(bytes, lengths); }
};

std::vector<std::vector<unsigned char>> Snapshot(const Model& model) {
  return {nn::encode_npy(model.weights), nn::encode_npy(model.step), nn::encode_npy(model.bytes),
          nn::encode_npy(model.lengths)};
}

void RoundTrip() {
  const std::string prefix = (paths::OutputDir("checkpoints/checkpoint_io") / "state.npz").string();
  Model             model;
  model.Init();
  const auto original = Snapshot(model);
  check::Require(model.step.item<std::int64_t>() == kCounter, "exact int64 counter");
  check::Require(model.Vocabulary() == std::vector<std::string>{"café", std::string("a\0b", 3)},
                 "UTF-8 and embedded-NUL strings encoded");
  model.checkpoint.Save(prefix);
  model.Mutate();
  check::Require(Snapshot(model)[0] != original[0] && Snapshot(model)[1] != original[1] &&
                     Snapshot(model)[2] != original[2] && Snapshot(model)[3] != original[3],
                 "mutation changes all checkpoint state");
  model.checkpoint.Restore(prefix);
  check::Require(Snapshot(model) == original, "exact mixed-dtype restore");
  for (const auto& [name, value] : nn::load_npz(prefix))
    std::cout << "  " << name << ": " << nn::dtype_name(value.dtype()) << ' ' << nn::shape_str(value.shape()) << '\n';

  Model restored;  // never initialised: NaN weights, sentinel counter/strings
  restored.checkpoint.Restore(prefix);
  check::Require(Snapshot(restored) == original, "fresh model exact restore");
  const auto prediction = restored.Predict(nn::tensor({{2, 3}})).to_vector<double>();
  check::Near(prediction[0], 11, 0, "restored matrix column zero");
  check::Near(prediction[1], 16, 0, "restored matrix column one");
  check::Require(restored.Vocabulary() == model.Vocabulary(), "fresh model vocabulary");

  Model wrong(true);
  wrong.Mutate();
  const auto sentinel = Snapshot(wrong);
  check::Throws([&] { wrong.checkpoint.Restore(prefix); }, "missing checkpoint tensor must fail");
  check::Require(Snapshot(wrong) == sentinel, "missing-name restore cannot partially assign");
  std::cout << "  same-model and fresh-model tensors match exactly; predictions=(11,16)\n"
               "  int64 counter="
            << restored.step.item<std::int64_t>()
            << "; UTF-8 and embedded-NUL strings preserved\n"
               "  missing tensor key rejected without modifying sentinel state\n";
}

}  // namespace

int main() {
  try {
    std::cout << "== Mixed-dtype checkpoint round trip\n";
    RoundTrip();
    std::cout << "All checkpoint I/O checks passed.\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
  return 0;
}
