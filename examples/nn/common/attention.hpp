// Attention helpers for the 5.x example ports (port of the parts of tflow's
// common/nn/attention.h that datamunge/nn/attention.hpp does not already
// provide), plus the small harness the 5.1 attention-fundamentals examples
// share: tflow's random input streams, float32 central differences and the
// comparison against the .npz fixtures the TF examples write.
//
// tflow -> datamunge:
//   nn::ScaledDotProductAttention(s, q, k, v, d, mask, allowed)
//       -> nn::scaled_dot_product_attention(q, k, v, mask) (scale 1/sqrt(q.size(-1)),
//          weights returned; `mask` is either an additive float bias or a bool keep mask)
//   nn::MultiHeadAttention(s, p, name, xq, xkv, d_model, heads, mask, allowed)
//       -> nn::multi_head_attention(store, name, xq, xkv, heads, options)
//   nn::AttentionAllowed(s, key_lengths, query_lengths, queries, keys, causal, offset)
//       -> attn::AttentionAllowed below (the library's causal_keep_mask is aligned
//          bottom-right instead of taking an explicit query offset, and its
//          padding_keep_mask masks keys only).
#pragma once

#include <datamunge/nn/attention.hpp>
#include <datamunge/nn/io.hpp>
#include <datamunge/nn/nn.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace attn {

namespace nn = datamunge::nn;

/// tflow's AttentionAllowed: bool [N, queries, keys], true where query t is
/// valid (t < query_lengths[n]), key j is valid (j < key_lengths[n]) and, when
/// `causal`, j <= t + query_offset (queries sit after `query_offset` cached
/// keys during incremental decoding). Lengths are integer tensors [N]. Add a
/// head axis (expand_dims(.., 1)) for [N, heads, Tq, Tk] scores.
inline nn::Tensor AttentionAllowed(const nn::Tensor& key_lengths, const nn::Tensor& query_lengths, std::int64_t queries,
                                   std::int64_t keys, bool causal = false, std::int64_t query_offset = 0) {
  if (queries < 1 || keys < 1 || query_offset < 0)
    throw std::invalid_argument("positive attention lengths and nonnegative query offset required");
  if (key_lengths.dim() != 1 || query_lengths.dim() != 1 || key_lengths.size(0) != query_lengths.size(0))
    throw std::invalid_argument("AttentionAllowed: key and query lengths must be vectors of the batch size");
  const std::int64_t batch = key_lengths.size(0);
  nn::Tensor         out   = nn::zeros({batch, queries, keys}, nn::DType::Bool);
  bool*              p     = out.data_ptr<bool>();
  for (std::int64_t b = 0; b < batch; ++b) {
    const double kl = key_lengths.at({b}), ql = query_lengths.at({b});
    for (std::int64_t t = 0; t < queries; ++t)
      for (std::int64_t j = 0; j < keys; ++j)
        p[(b * queries + t) * keys + j] = t < ql && j < kl && (!causal || j <= t + query_offset);
  }
  return out;
}

/// tflow's ScaledDotProductAttention(q, k, v, d, mask): scores scaled by
/// 1/sqrt(d), plain (weight-materialising) path, weights returned. `mask` is an
/// additive float bias or a bool keep mask (tflow's `allowed`); rows with no
/// allowed key give zero weights and output.
inline nn::Attended ScaledDotProductAttention(const nn::Tensor& q, const nn::Tensor& k, const nn::Tensor& v,
                                              std::int64_t d, const nn::Tensor& mask = nn::Tensor()) {
  if (d < 1) throw std::invalid_argument("attention key width must be positive");
  nn::AttentionOptions o;
  o.mask         = mask;
  o.scale        = 1.0 / std::sqrt(static_cast<double>(d));
  o.need_weights = true;
  o.impl         = nn::AttentionImpl::Plain;
  return nn::scaled_dot_product_attention(q, k, v, o);
}

/// tflow's MultiHeadAttention(p, name, x_q, x_kv, d_model, heads, mask,
/// allowed): bias-free projections "<name>/Wq|Wk|Wv|Wo" [d_model, d_model]
/// (U(+-sqrt(3/d_model))), weights [N, heads, Tq, Tk] returned. `mask`: keep or
/// additive, broadcastable to [N, heads, Tq, Tk]. d_model is explicit, as in tflow,
/// and checked against the input width.
inline nn::Attended MultiHeadAttention(nn::ParameterStore& store, const std::string& name, const nn::Tensor& x_q,
                                       const nn::Tensor& x_kv, std::int64_t d_model, std::int64_t heads,
                                       const nn::Tensor& mask = nn::Tensor()) {
  if (d_model < 1 || heads < 1 || d_model % heads != 0)
    throw std::invalid_argument("positive model/head counts and divisible head width required");
  if (x_q.dim() != 3 || x_q.size(2) != d_model || x_kv.dim() != 3 || x_kv.size(2) != d_model)
    throw std::invalid_argument("multi-head inputs must be [N, T, d_model]");
  nn::MultiHeadOptions o;
  o.mask         = mask;
  o.need_weights = true;
  o.impl         = nn::AttentionImpl::Plain;
  return nn::multi_head_attention(store, name, x_q, x_kv, heads, o);
}

/// Initializer that returns a fixed array (replays weights saved by TF).
inline nn::Initializer FixedInit(const nn::Tensor& value) {
  return [value](const nn::Shape& shape, nn::DType, nn::Generator&) {
    if (shape != value.shape()) throw std::invalid_argument("saved weight has the wrong shape");
    return value.clone();
  };
}

/// Integer vector tensor.
inline nn::Tensor Ints(const std::vector<int>& values) {
  return nn::from_vector(values, {static_cast<std::int64_t>(values.size())});
}

/// tflow's `random` lambdas: float32 tensors filled with scale * N(0, 1) drawn
/// from one std::mt19937 and ONE std::normal_distribution<float> (it caches
/// the second value of each pair, so a fresh distribution per tensor would
/// diverge after an odd-sized tensor). With the same seed and the same
/// standard library (both builds use libc++ here) the values equal TF's.
struct TfRandom {
  explicit TfRandom(unsigned seed) : rng(seed) {}
  nn::Tensor operator()(const nn::Shape& shape, float scale = 1.f) {
    nn::Tensor t = nn::empty(shape);
    float*     p = t.data_ptr<float>();
    for (std::int64_t i = 0; i < t.numel(); ++i) p[i] = scale * normal(rng);
    return t;
  }
  std::mt19937                    rng;
  std::normal_distribution<float> normal{0, 1};
};

/// Max |a - b| (inf on a shape mismatch).
inline double MaxAbsDiff(const nn::Tensor& a, const nn::Tensor& b) {
  if (a.shape() != b.shape()) return INFINITY;
  const auto x = nn::cast(a, nn::DType::Float64).to_vector<double>();
  const auto y = nn::cast(b, nn::DType::Float64).to_vector<double>();
  double     m = 0;
  for (std::size_t i = 0; i < x.size(); ++i) m = std::max(m, std::abs(x[i] - y[i]));
  return m;
}

/// tflow's native finite-difference probe: for `samples` coordinates spread
/// evenly over `value` (i * (numel - 1) / (samples - 1)), compares the float32
/// central difference (f(x + h) - f(x - h)) / 2h, h = 0.002, with the analytic
/// gradient. `value` is modified in place and restored. Returns the max error.
inline double CentralDifferenceError(nn::Tensor& value, const nn::Tensor& gradient, int samples,
                                     const std::function<float()>& objective, int* count = nullptr) {
  float*     p = value.data_ptr<float>();
  const auto g = gradient.to_vector<float>();
  double     error = 0;
  for (int i = 0; i < samples; ++i) {
    const std::int64_t c   = samples == 1 ? 0 : i * (value.numel() - 1) / (samples - 1);
    const float        old = p[c];
    p[c]                   = old + .002f;
    const float plus       = objective();
    p[c]                   = old - .002f;
    const float minus      = objective();
    p[c]                   = old;
    error                  = std::max(error, std::abs((plus - minus) / .004 - g[c]));
    if (count) ++*count;
  }
  return error;
}

/// Reads one array of an .npz archive (throws when missing).
inline nn::Tensor Named(const nn::NamedTensors& arrays, const std::string& name) {
  for (const auto& [key, value] : arrays)
    if (key == name) return value;
  throw std::runtime_error("npz has no array '" + name + "'");
}

inline bool Has(const nn::NamedTensors& arrays, const std::string& name) {
  for (const auto& [key, value] : arrays)
    if (key == name) return true;
  return false;
}

/// Default location of the TF example's fixtures: tflow's output/models/<name>.
inline std::filesystem::path DefaultTfDir(const std::string& name) {
  const char* home = std::getenv("HOME");
  return std::filesystem::path(home ? home : ".") / "projects" / "tflow_examples" / "output" / "models" / name;
}

/// Compares every array of `ours` that TF also saved (same key) with TF's
/// .npz `tf_file`. Returns the largest difference over the arrays whose names
/// pass `include` (all by default), or -1 when the file is missing. Prints one
/// line per compared array when `verbose`.
inline double CompareWithTf(const std::filesystem::path& tf_file, const nn::NamedTensors& ours, bool verbose = false,
                            const std::function<bool(const std::string&)>& include = {}) {
  if (!std::filesystem::exists(tf_file)) return -1;
  const auto ref   = nn::load_npz(tf_file.string());
  double     worst = 0;
  for (const auto& [name, value] : ours) {
    if (!Has(ref, name) || (include && !include(name))) continue;
    const double d = MaxAbsDiff(value, Named(ref, name));
    if (verbose) std::printf("    %-22s max |ours - TF| = %.2e\n", name.c_str(), d);
    worst = std::max(worst, d);
  }
  return worst;
}

/// Compares `ours` with TF's fixture `file` under `tf_dir`. When the arrays
/// listed in `inputs` are bit-identical to TF's (same random streams), every
/// other shared array except the finite-difference metadata ("metadata",
/// "native_fd") must agree within `tolerance` (a failed check throws). Returns
/// the max difference, or -1 when the fixture is missing or the inputs differ.
inline double CheckAgainstTf(const std::filesystem::path& tf_dir, const std::string& file, const nn::NamedTensors& ours,
                             const std::vector<std::string>& inputs, double tolerance, bool verbose = false) {
  const auto path = tf_dir / file;
  if (!std::filesystem::exists(path)) {
    std::printf("  (TF fixture %s not found: TF comparison skipped)\n", path.string().c_str());
    return -1;
  }
  auto is_input = [&](const std::string& n) { return std::find(inputs.begin(), inputs.end(), n) != inputs.end(); };
  const double din = CompareWithTf(path, ours, false, is_input);
  if (din != 0) {
    std::printf("  (inputs differ from TF's %s by %.2e: different random stream, outputs not compared)\n",
                file.c_str(), din);
    return -1;
  }
  const double d = CompareWithTf(path, ours, verbose, [&](const std::string& n) {
    return !is_input(n) && n != "metadata" && n != "native_fd";
  });
  std::printf("  vs TF %s (identical inputs): max |ours - TF| = %.2e over all outputs and gradients\n", file.c_str(), d);
  if (!(d <= tolerance))
    throw std::runtime_error(file + ": outputs differ from TF's by " + std::to_string(d) + " > " +
                             std::to_string(tolerance));
  return d;
}

/// key=value options shared by the 5.1 examples: out=DIR (default
/// output/nn/<name>), tf=DIR (TF fixtures, default tflow's output/models/<name>),
/// init=auto|tf|nn for examples with learned weights (auto: TF's saved
/// weights when present), verbose=0|1 (per-array TF comparison).
struct Options {
  std::filesystem::path out, tf;
  std::string           init{"auto"};
  bool                  verbose{false};
};

inline Options ParseOptions(int argc, char** argv, const std::string& name,
                            const std::filesystem::path& out_default) {
  Options o;
  o.out = out_default;
  o.tf  = DefaultTfDir(name);
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    const auto        e = a.find('=');
    if (e == std::string::npos) throw std::invalid_argument("expected key=value option, got " + a);
    const std::string key = a.substr(0, e), value = a.substr(e + 1);
    if (key == "out") o.out = value;
    else if (key == "tf") o.tf = value;
    else if (key == "init") o.init = value;
    else if (key == "verbose") o.verbose = value != "0";
    else throw std::invalid_argument("unknown option " + key + " (out=, tf=, init=, verbose=)");
  }
  if (o.init != "auto" && o.init != "tf" && o.init != "nn") throw std::invalid_argument("init must be auto, tf or nn");
  std::filesystem::create_directories(o.out);
  return o;
}

}  // namespace attn
