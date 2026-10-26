// Causal decoder language model (port of tflow's common/ml/transformer_lm.h)
// plus the small harness the 5.2 language-Transformer ports share: key=value
// options, the character_rnn data, the trained-GPT weights used by kv_cache
// and token_sampling, and the comparison with TF's .npz fixtures.
//
// tflow -> datamunge: the session graph becomes eager calls; AddSymbolicGradients
// -> nn::grad; the Adam train_op -> nn::Optimizer (Adam, lr .003); SaveV2
// checkpoints -> .npz files of the parameter store. Weights initialise from nn's
// generator (seed 9001) unless TF's saved arrays are replayed (LoadArrays).
#pragma once

#include <datamunge/nn/nn.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "common/attention.hpp"
#include "common/checks.hpp"
#include "common/decoder_transformer.hpp"
#include "common/paths.hpp"

namespace lm {

namespace fs = std::filesystem;
using Outputs = std::map<std::string, Tensor>;

// ---- small tensor helpers ----------------------------------------------------------------------

inline nn::NamedTensors Named(const Outputs& out) { return {out.begin(), out.end()}; }

inline Tensor Rows(const Tensor& x, const std::vector<std::int64_t>& rows) { return nn::take_rows(x, rows); }

/// Copies of every parameter, by name (creation order).
inline nn::NamedTensors Arrays(const nn::ParameterStore& store) {
  nn::NamedTensors out;
  for (const auto& name : store.names()) out.emplace_back(name, store.at(name).detach().clone());
  return out;
}

/// True when both stores' arrays have the same names and bits.
inline bool SameBits(const nn::NamedTensors& a, const nn::NamedTensors& b) {
  if (a.size() != b.size()) return false;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (a[i].first != b[i].first || a[i].second.shape() != b[i].second.shape()) return false;
    const auto x = a[i].second.to_vector<float>(), y = b[i].second.to_vector<float>();
    if (std::memcmp(x.data(), y.data(), x.size() * sizeof(float)) != 0) return false;
  }
  return true;
}

/// Overwrites every parameter of `store` from an .npz written by tflow (or by
/// store.save); the archive must cover exactly the store's parameters.
inline void LoadArrays(nn::ParameterStore& store, const fs::path& file) {
  const auto arrays = nn::load_npz(file.string());
  check::Require(arrays.size() == store.size(), "saved arrays cover exactly the model's parameters: " + file.string());
  for (const auto& [name, value] : arrays) {
    check::Require(store.contains(name), "saved array " + name + " is a model parameter");
    store.set(name, nn::cast(value, store.at(name).dtype()));
  }
}

inline void RequireFinite(const Outputs& out, const std::string& what) {
  for (const auto& [k, v] : out)
    if (v.dtype() == nn::DType::Float32)
      for (float x : v.to_vector<float>()) check::Require(std::isfinite(x), what + " (" + k + ")");
}

// ---- options -----------------------------------------------------------------------------------

/// epochs=N (training examples), out=DIR, tf=DIR (TF fixtures), init=auto|tf|nn
/// (auto: replay TF's initial_arrays.npz when present), gpt=auto|ours|tf|PATH
/// (trained GPT for kv_cache/token_sampling), threads=N, verbose=0|1.
struct Options {
  int         epochs{3};
  fs::path    out, tf;
  std::string init{"auto"}, gpt{"auto"};
  int         threads{4};
  bool        verbose{false};
};

inline Options ParseOptions(int argc, char** argv, const std::string& name, bool training) {
  Options o;
  o.tf = attn::DefaultTfDir(name);
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    const auto        e = a.find('=');
    check::Require(e != std::string::npos, "expected key=value option, got " + a);
    const std::string key = a.substr(0, e), value = a.substr(e + 1);
    if (key == "epochs" && training) o.epochs = std::stoi(value);
    else if (key == "out") o.out = value;
    else if (key == "tf") o.tf = value;
    else if (key == "init" && training) o.init = value;
    else if (key == "gpt" && !training) o.gpt = value;
    else if (key == "threads") o.threads = std::stoi(value);
    else if (key == "verbose") o.verbose = value != "0";
    else
      throw std::invalid_argument("unknown option " + key + (training ? " (epochs=, out=, tf=, init=, threads=, verbose=)"
                                                                      : " (out=, tf=, gpt=, threads=, verbose=)"));
  }
  check::Require(o.epochs > 0 && o.epochs <= 100, "positive bounded epochs");
  check::Require(o.init == "auto" || o.init == "tf" || o.init == "nn", "init must be auto, tf or nn");
  if (o.out.empty()) o.out = paths::OutputDir("models/" + name);
  fs::create_directories(o.out);
  if (o.threads > 0) nn::set_num_threads(o.threads);
  return o;
}

/// Resolves init=: the TF initial arrays to replay, or empty for nn's generator.
inline fs::path InitArrays(const Options& o) {
  const fs::path file = o.tf / "initial_arrays.npz";
  if (o.init == "nn") return {};
  if (o.init == "tf") check::Require(fs::exists(file), "init=tf needs " + file.string());
  return fs::exists(file) ? file : fs::path();
}

/// Compares our record with TF's fixture of the same name (when present):
/// prints the max difference over the shared arrays and returns it (-1 if
/// the fixture is missing).
inline double CompareRecord(const Options& o, const std::string& file, const nn::NamedTensors& ours,
                            const std::function<bool(const std::string&)>& include = {}) {
  const double d = attn::CompareWithTf(o.tf / file, ours, o.verbose, include);
  if (d < 0) std::printf("  vs TF %-26s (fixture missing)\n", file.c_str());
  else std::printf("  vs TF %-26s max |ours - TF| = %.3g\n", file.c_str(), d);
  return d;
}

// ---- data ------------------------------------------------------------------------------------------

inline std::string ReadText(const fs::path& file) {
  std::ifstream in(file, std::ios::binary);
  check::Require(bool(in), "cannot read " + file.string());
  return {std::istreambuf_iterator<char>(in), {}};
}

inline std::vector<int> Encode(const std::string& vocab, const std::string& text) {
  std::vector<int> codes;
  for (char c : text) {
    const auto at = vocab.find(c);
    check::Require(at != std::string::npos, "prompt character in vocabulary");
    codes.push_back(static_cast<int>(at));
  }
  return codes;
}

// ---- the model ---------------------------------------------------------------------------------------

/// tflow's ml::TransformerLM: one-hot input [N, 64] int32 -> causal decoder;
/// loss = mean next-token cross-entropy over every position.
class TransformerLM {
 public:
  DecoderConfig                  config;
  nn::ParameterStore             store{9001};
  std::unique_ptr<nn::Optimizer> optimizer;

  explicit TransformerLM(bool training = true, DecoderConfig c = {}) : config(c) {
    {
      nn::NoGradGuard guard;
      DecoderTransformer(store, nn::zeros({1, c.context, c.input_classes}), c);
    }
    if (training) Train();
  }
  /// (Re)creates the Adam state (after replaying initial arrays).
  void Train() {
    nn::OptimizerConfig oc;
    oc.kind   = nn::OptimizerKind::Adam;
    optimizer = std::make_unique<nn::Optimizer>(store, .003, oc);
  }

  Tensor Encoded(const Tensor& x) const { return nn::one_hot(x, config.input_classes); }

  Tensor Loss(const DecoderPath& p, const Tensor& y) const {
    const Tensor logp = nn::log_softmax(p.logits, -1);
    return nn::neg(nn::mean(nn::sum(nn::mul(nn::one_hot(y, config.output_classes), logp), {2})));
  }

  Outputs Evaluate(const Tensor& x, const Tensor& y) {
    nn::NoGradGuard guard;
    const auto      p = DecoderTransformer(store, Encoded(x), config);
    Outputs out{{"logits", p.logits}, {"hidden", p.hidden}, {"attention", p.attention}, {"embedded", p.embedded},
                {"loss", Loss(p, y)}};
    RequireFinite(out, "finite Transformer fields");
    return out;
  }

  float LossValue(const Tensor& x, const Tensor& y) {
    nn::NoGradGuard guard;
    return Loss(DecoderTransformer(store, Encoded(x), config), y).item<float>();
  }

  void ValidateTokens(const Tensor& x) const {
    check::Require(x.dim() == 2 && x.size(1) == config.context, "full context token matrix");
    for (auto v : nn::cast(x, nn::DType::Int64).to_vector<std::int64_t>())
      check::Require(v >= 0 && v < config.input_classes, "valid context token IDs");
  }

  /// Context-only inference (no labels).
  Tensor Predict(const Tensor& x) {
    ValidateTokens(x);
    nn::NoGradGuard guard;
    return DecoderTransformer(store, Encoded(x), config).logits;
  }

  /// d sum(logits[:, context/2, 0]) / d one-hot input (tflow slices {0, T/2, 0} size {-1, 1, 1}).
  Tensor InputGradient(const Tensor& x) {
    Tensor encoded = Encoded(x);
    encoded.requires_grad_(true);
    const auto   p         = DecoderTransformer(store, encoded, config);
    const Tensor objective = nn::sum(nn::slice_axis(nn::slice_axis(p.logits, 1, config.context / 2, config.context / 2 + 1), 2, 0, 1));
    return nn::grad(objective, {encoded})[0].detach();
  }

  std::vector<Tensor> RawGradients(const Tensor& x, const Tensor& y) {
    const auto      p = DecoderTransformer(store, Encoded(x), config);
    nn::GradOptions go;
    go.allow_unused = true;
    auto g          = nn::grad(Loss(p, y), store.parameters(), go);
    const auto ps   = store.parameters();
    for (std::size_t i = 0; i < g.size(); ++i) g[i] = g[i].defined() ? g[i].detach() : nn::zeros_like(ps[i]);
    return g;
  }

  nn::NamedTensors Gradient(const Tensor& x, const Tensor& y) {
    const auto       g = RawGradients(x, y);
    nn::NamedTensors out;
    for (std::size_t i = 0; i < g.size(); ++i) out.emplace_back(store.names()[i], g[i]);
    return out;
  }

  void Update(const Tensor& x, const Tensor& y) { optimizer->step(RawGradients(x, y)); }

  nn::NamedTensors Arrays() const { return lm::Arrays(store); }

  /// tflow's Sample: context-cropped (last 64 tokens, left-aligned, zero
  /// padded) full-model predictions, softmax(z / temperature) inverse-CDF draw
  /// with u = (mt19937() + .5) / 2^32.
  Outputs Sample(const std::vector<int>& prompt, int count, std::uint32_t seed, double temperature) {
    check::Require(!prompt.empty() && count > 0 && temperature > 0 && std::isfinite(temperature),
                   "valid Transformer prompt/count/temperature");
    const std::int64_t       V = config.output_classes, T = config.context;
    std::vector<int>         codes(prompt.begin(), prompt.end());
    std::vector<double>      uniform;
    std::vector<float>       logits;
    std::vector<std::int32_t> contexts, lengths;
    for (int p : prompt) check::Require(p >= 0 && p < config.input_classes, "valid prompt token");
    std::mt19937 rng(seed);
    for (int t = 0; t < count; ++t) {
      const int end = static_cast<int>(codes.size()), start = std::max<int>(0, end - static_cast<int>(T)),
                length = end - start;
      std::vector<std::int32_t> context(static_cast<std::size_t>(T), 0);
      for (int j = start; j < end; ++j) context[static_cast<std::size_t>(j - start)] = codes[static_cast<std::size_t>(j)];
      contexts.insert(contexts.end(), context.begin(), context.end());
      lengths.push_back(length);
      const auto scores = Predict(nn::from_vector(context, {1, T})).to_vector<float>();
      double     maximum = -INFINITY, mass = 0;
      for (std::int64_t c = 0; c < V; ++c) {
        const float z = scores[static_cast<std::size_t>((length - 1) * V + c)];
        logits.push_back(z);
        maximum = std::max(maximum, double(z) / temperature);
      }
      const float* z = logits.data() + t * V;
      for (std::int64_t c = 0; c < V; ++c) mass += std::exp(double(z[c]) / temperature - maximum);
      const double u = (double(rng()) + .5) / 4294967296.;
      uniform.push_back(u);
      double sum    = 0;
      int    chosen = static_cast<int>(V) - 1;
      for (std::int64_t c = 0; c < V; ++c) {
        sum += std::exp(double(z[c]) / temperature - maximum) / mass;
        if (u < sum) {
          chosen = static_cast<int>(c);
          break;
        }
      }
      codes.push_back(chosen);
    }
    return {{"codes", nn::from_vector(std::vector<std::int32_t>(codes.begin(), codes.end()),
                                      {static_cast<std::int64_t>(codes.size())})},
            {"uniform", nn::from_vector(uniform, {count})},
            {"logits", nn::from_vector(logits, {count, V})},
            {"contexts", nn::from_vector(contexts, {count, T})},
            {"context_lengths", nn::from_vector(lengths, {count})}};
  }
};

/// The trained GPT (gpt_lm's final weights) used by kv_cache and token_sampling.
/// gpt=auto: TF's output/models/gpt_lm/model_arrays.npz when present (so the
/// decisions can be compared with TF's exactly), else our gpt_lm's final.npz;
/// gpt=tf / gpt=ours force one; anything else is a path.
inline fs::path TrainedGpt(const std::string& choice) {
  const fs::path tf   = attn::DefaultTfDir("gpt_lm") / "model_arrays.npz";
  const fs::path ours = paths::OutputRoot() / "models" / "gpt_lm" / "final.npz";
  fs::path       file = choice == "tf" ? tf : choice == "ours" ? ours : choice == "auto" ? (fs::exists(tf) ? tf : ours) : fs::path(choice);
  check::Require(fs::exists(file), "trained GPT weights not found: " + file.string() + " (run gpt_lm first)");
  return file;
}

inline fs::path GptVocabulary() {
  const fs::path file = paths::Dataset("character_rnn") / "vocabulary.txt";
  check::Require(fs::exists(file), "character_rnn vocabulary not found: " + file.string());
  return file;
}

}  // namespace lm
