// Real MR sentiment training for Kim CNN-rand; short native demonstration
// (port of tflow's common/ml/sentence.h). Also holds the small generic
// helpers the 1.9 (1-D convolution) ports share, as tflow's sentence.h does:
// row selection, stable binary CE, metrics, fixture parameter assignment and
// host finite-difference comparison.
#pragma once

#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <set>
#include <string>
#include <vector>

#include <datamunge/nn/nn.hpp>

#include "common/checks.hpp"
#include "common/kim_cnn.hpp"
#include "common/paths.hpp"
#include "common/tf_replay.hpp"

namespace sentence {

using Rows    = std::vector<std::int64_t>;
using Weights = std::vector<std::vector<double>>;

// ---- generic helpers --------------------------------------------------------------------

inline Tensor Select(const Tensor& x, const Rows& rows) { return nn::take_rows(x, rows); }
inline Rows   Range(std::int64_t begin, std::int64_t end) {
  Rows r(static_cast<std::size_t>(end - begin));
  std::iota(r.begin(), r.end(), begin);
  return r;
}

/// Overwrites every parameter of `p` (creation order v, flat index i) with
/// value(v, i, shape) rounded to the store's dtype, and returns the values as doubles.
inline Weights FixParameters(nn::ParameterStore& p, const std::function<float(std::size_t, std::int64_t, const nn::Shape&)>& value) {
  Weights weights;
  for (std::size_t v = 0; v < p.size(); ++v) {
    const auto         shape = p.parameters()[v].shape();
    std::vector<float> data(static_cast<std::size_t>(nn::shape_numel(shape)));
    for (std::size_t i = 0; i < data.size(); ++i) data[i] = value(v, static_cast<std::int64_t>(i), shape);
    p.set(p.names()[v], nn::cast(nn::from_vector(data, shape), p.dtype()));
    weights.emplace_back(data.begin(), data.end());
  }
  return weights;
}

/// max |central difference of `objective` (step 1e-5) - gradient| over every
/// coordinate of `weights`; `coordinates` receives the count.
inline double HostGradientError(const Weights& weights, const std::vector<Tensor>& gradients,
                                const std::function<double(const Weights&)>& objective, int& coordinates) {
  double error = 0;
  coordinates  = 0;
  for (std::size_t v = 0; v < weights.size(); ++v) {
    const auto g = gradients[v].to_vector<double>();
    for (std::size_t i = 0; i < weights[v].size(); ++i) {
      auto plus = weights, minus = weights;
      plus[v][i] += 1e-5;
      minus[v][i] -= 1e-5;
      error = std::max(error, std::abs((objective(plus) - objective(minus)) / 2e-5 - g[i]));
      ++coordinates;
    }
  }
  return error;
}

/// Sum of squares of each gradient.
inline std::vector<double> SquaredNorms(const std::vector<Tensor>& gradients) {
  std::vector<double> out;
  for (const auto& g : gradients) {
    double s = 0;
    for (double x : g.to_vector<double>()) s += x * x;
    out.push_back(s);
  }
  return out;
}

/// Every gradient coordinate finite and every tensor nonzero (optionally
/// excused by `may_vanish(name)`); returns the number of nonzero tensors.
inline int RequireLiveGradients(const nn::ParameterStore& p, const std::vector<Tensor>& gradients, const std::string& what,
                                const std::function<bool(const std::string&)>& may_vanish = {}, double floor = 0) {
  int nonzero = 0;
  for (std::size_t v = 0; v < gradients.size(); ++v) {
    double norm = 0;
    for (double x : gradients[v].to_vector<double>()) {
      check::Require(std::isfinite(x), "finite " + what + " gradient coordinate");
      norm += x * x;
    }
    if (!(may_vanish && may_vanish(p.names()[v]))) check::Require(norm > floor, "nonzero " + what + " gradient: " + p.names()[v]);
    nonzero += norm > floor;
  }
  return nonzero;
}

/// -mean_n sum_c onehot(labels) log_softmax(logits) (tflow's SentenceCE).
inline Tensor SentenceCE(const Tensor& logits, const Tensor& labels) {
  const Tensor target = nn::one_hot(labels, logits.size(-1), 1.0, 0.0, logits.dtype());
  return nn::neg(nn::mean(nn::sum(nn::mul(target, nn::log_softmax(logits)), {1})));
}

inline nn::Tensor Npz(const nn::NamedTensors& a, const std::string& key) { return replay::Get(a, key); }

/// Bytes of every tensor (tflow's Snapshot); compare with SameBits.
inline replay::Map Snapshot(const std::vector<const nn::ParameterStore*>& stores) {
  replay::Map out;
  int         k = 0;
  for (const auto* s : stores) {
    for (const auto& name : s->names()) out[std::to_string(k) + ":" + name] = s->at(name).detach().clone();
    ++k;
  }
  return out;
}

/// Saves parameters (and optional state) of a store pair as one .npz with "p:"/"s:" prefixes.
inline void SaveStores(const std::string& path, const nn::ParameterStore& p, const nn::ParameterStore* s = nullptr) {
  nn::NamedTensors arrays;
  for (const auto& n : p.names()) arrays.emplace_back("p:" + n, p.at(n).detach());
  if (s)
    for (const auto& n : s->names()) arrays.emplace_back("s:" + n, s->at(n).detach());
  nn::save_npz(path, arrays);
}
/// Restores what SaveStores wrote; every array must match an existing tensor.
inline void RestoreStores(const std::string& path, nn::ParameterStore& p, nn::ParameterStore* s = nullptr) {
  const auto arrays = nn::load_npz(path);
  check::Require(arrays.size() == p.size() + (s ? s->size() : 0), "checkpoint covers exactly the model tensors");
  for (const auto& [key, value] : arrays) {
    auto& store = key.rfind("p:", 0) == 0 ? p : *s;
    const auto name = key.substr(2);
    check::Require(store.contains(name), "checkpoint tensor " + name + " belongs to the model");
    store.set(name, nn::cast(value, store.dtype()));
  }
}

/// Exports parameters as "p<i>" and state as "s<i>" (tflow's model_arrays.npz layout).
inline void ExportArrays(const std::filesystem::path& path, const nn::ParameterStore& p, const nn::ParameterStore* s = nullptr) {
  nn::NamedTensors arrays;
  for (std::size_t i = 0; i < p.size(); ++i) arrays.emplace_back("p" + std::to_string(i), p.parameters()[i].detach());
  if (s)
    for (std::size_t i = 0; i < s->size(); ++i) arrays.emplace_back("s" + std::to_string(i), s->parameters()[i].detach());
  nn::save_npz(path.string(), arrays);
}

inline void CopyIfExists(const std::filesystem::path& from, const std::filesystem::path& to) {
  if (std::filesystem::exists(from)) std::filesystem::copy_file(from, to, std::filesystem::copy_options::overwrite_existing);
}

// ---- MR sentence data ---------------------------------------------------------------------

struct SentenceData {
  Tensor tokens, lengths, labels, source_ids;
  [[nodiscard]] std::int64_t size() const { return labels.size(0); }
};
struct SentenceSplits {
  std::vector<SentenceData> pools;
  KimConfig                 config;
};

inline SentenceData SentenceRows(const SentenceData& d, const Rows& rows) {
  return {Select(d.tokens, rows), Select(d.lengths, rows), Select(d.labels, rows), Select(d.source_ids, rows)};
}

inline SentenceSplits LoadSentences(const std::filesystem::path& root) {
  const auto a      = nn::load_npz((root / "sentence.npz").string());
  const auto config = Npz(a, "config");
  check::Require(config.dtype() == nn::DType::Int32 && config.shape() == nn::Shape{2}, "sentence vocabulary metadata");
  SentenceSplits out;
  const auto     cfg  = config.to_vector<int>();
  out.config.vocabulary = cfg[0];
  out.config.time       = cfg[1];
  out.config.embedding  = 32;
  out.config.filters    = 32;
  check::Require(out.config.vocabulary >= 2 && out.config.vocabulary <= 8000 && out.config.time == 64,
                 "sentence fit vocabulary/time bounds");
  std::set<int> all;
  for (const std::string split : {"fit", "validation", "test"}) {
    SentenceData d{Npz(a, split + "_tokens"), Npz(a, split + "_lengths"), Npz(a, split + "_labels"), Npz(a, split + "_source_ids")};
    const std::int64_t n = d.tokens.size(0);
    check::Require(n > 0 && d.tokens.dtype() == nn::DType::Int32 && d.tokens.shape() == nn::Shape{n, 64}, "sentence token matrix");
    for (const auto* t : {&d.lengths, &d.labels, &d.source_ids})
      check::Require(t->dtype() == nn::DType::Int32 && t->shape() == nn::Shape{n}, "aligned sentence metadata");
    const auto tokens = d.tokens.to_vector<int>(), lengths = d.lengths.to_vector<int>(), labels = d.labels.to_vector<int>(),
               sources = d.source_ids.to_vector<int>();
    int positive = 0;
    for (std::int64_t i = 0; i < n; ++i) {
      const int length = lengths[i], label = labels[i], source = sources[i];
      check::Require(length >= 1 && length <= 64 && label >= 0 && label < 2 && source >= 0 && source < 10662 && all.insert(source).second,
                     "valid unique sentence ID/label/length");
      positive += label;
      for (int j = 0; j < 64; ++j) {
        const int token = tokens[i * 64 + j];
        check::Require(j < length ? (token >= 1 && token < out.config.vocabulary) : token == 0, "sentence legal tokens/right PAD");
      }
    }
    std::cout << " MR " << split << ":" << n << " sentences,positive=" << positive << std::endl;
    out.pools.push_back(d);
  }
  return out;
}

// ---- metrics and baselines --------------------------------------------------------------------

inline void SentenceMetrics(const std::string& name, const Tensor& labels_tensor, const Tensor& logits) {
  const std::int64_t n = labels_tensor.size(0);
  check::Require(logits.shape() == nn::Shape{n, 2}, "sentiment score shape");
  const auto z = logits.to_vector<double>();
  const auto labels = labels_tensor.to_vector<int>();
  int        correct = 0, confusion[2][2]{};
  double     nll = 0;
  for (std::int64_t i = 0; i < n; ++i) {
    const double a = z[2 * i], b = z[2 * i + 1];
    check::Require(std::isfinite(a) && std::isfinite(b), "finite binary logits");
    const int label = labels[i], prediction = b > a;
    ++confusion[label][prediction];
    correct += label == prediction;
    const double m = std::max(a, b);
    nll += m + std::log(std::exp(a - m) + std::exp(b - m)) - (label ? b : a);
  }
  std::cout << " " << name << ":accuracy=" << 100. * correct / n << "%,CE=" << nll / n << ",confusion=" << confusion[0][0] << "/"
            << confusion[0][1] << "/" << confusion[1][0] << "/" << confusion[1][1] << '\n';
}

inline Tensor SentenceNB(const SentenceData& fit, const SentenceData& test, int vocabulary) {
  std::vector<std::vector<double>> count(2, std::vector<double>(vocabulary, 1));
  double                           total[2] = {double(vocabulary), double(vocabulary)}, prior[2] = {1, 1};
  const auto ft = fit.tokens.to_vector<int>(), fl = fit.lengths.to_vector<int>(), fy = fit.labels.to_vector<int>();
  const std::int64_t T = fit.tokens.size(1);
  for (std::int64_t n = 0; n < fit.size(); ++n) {
    const int label = fy[n];
    ++prior[label];
    for (int t = 0; t < fl[n]; ++t) {
      ++count[label][ft[n * T + t]];
      ++total[label];
    }
  }
  const auto         tt = test.tokens.to_vector<int>(), tl = test.lengths.to_vector<int>();
  std::vector<float> out(static_cast<std::size_t>(2 * test.size()));
  for (std::int64_t n = 0; n < test.size(); ++n)
    for (int c = 0; c < 2; ++c) {
      double score = std::log(prior[c] / (prior[0] + prior[1]));
      for (int t = 0; t < tl[n]; ++t) score += std::log(count[c][tt[n * T + t]] / total[c]);
      out[2 * n + c] = static_cast<float>(score);
    }
  return nn::from_vector(out, {test.size(), 2});
}

/// confusion[truth * classes + prediction] counts.
inline std::vector<int> Confusion(const Tensor& labels_tensor, const std::vector<int>& predictions, int classes) {
  const auto       labels = labels_tensor.to_vector<int>();
  std::vector<int> cells(static_cast<std::size_t>(classes * classes));
  check::Require(labels.size() == predictions.size(), "confusion labels/predictions alignment");
  for (std::size_t n = 0; n < labels.size(); ++n) ++cells[labels[n] * classes + predictions[n]];
  return cells;
}

/// Prints a confusion matrix (rows = truth, columns = prediction).
inline void PrintConfusion(const std::string& name, const std::vector<int>& cells, int classes) {
  std::cout << " Confusion matrix " << name << " (rows=truth, columns=prediction):\n truth/pred";
  for (int c = 0; c < classes; ++c) std::cout << ' ' << std::setw(5) << c;
  std::cout << '\n';
  for (int r = 0; r < classes; ++r) {
    std::cout << ' ' << std::setw(10) << r;
    for (int c = 0; c < classes; ++c) std::cout << ' ' << std::setw(5) << cells[r * classes + c];
    std::cout << '\n';
  }
}

/// 1 when logits[n,1] > logits[n,0].
inline std::vector<int> Positive(const Tensor& logits) {
  const auto z = logits.to_vector<float>();
  std::vector<int> out(z.size() / 2);
  for (std::size_t n = 0; n < out.size(); ++n) out[n] = z[2 * n + 1] > z[2 * n];
  return out;
}

// ---- Kim CNN-rand model ---------------------------------------------------------------------

class SentenceModel {
 public:
  nn::ParameterStore           parameters{317};
  KimConfig                    config;
  std::unique_ptr<KimAdadelta> optimizer;
  nn::Generator                dropout{313};

  explicit SentenceModel(const KimConfig& c, bool training = true) : config(c) {
    {
      nn::NoGradGuard guard;
      KimCnn(parameters, nn::zeros({1, c.time}, nn::DType::Int32), nn::full({1}, 1, nn::DType::Int32), c);
    }
    if (training) optimizer = std::make_unique<KimAdadelta>(parameters);
  }
  KimOutput Forward(const SentenceData& d, bool training) {
    return KimCnn(parameters, d.tokens, d.lengths, config, training, .5, &dropout);
  }
  /// Inference (dropout-free, no gradient) CE.
  double Loss(const SentenceData& d) {
    nn::NoGradGuard guard;
    const double    value = SentenceCE(Forward(d, false).logits, d.labels).item<double>();
    check::Require(std::isfinite(value) && value >= 0, "finite sentence CE");
    return value;
  }
  Tensor Predict(const SentenceData& d) {
    nn::NoGradGuard guard;
    return Forward(d, false).logits.detach();
  }
  /// Predictions in batches of 50.
  Tensor PredictAll(const SentenceData& d) {
    std::vector<Tensor> parts;
    for (std::int64_t start = 0; start < d.size(); start += 50) parts.push_back(Predict(SentenceRows(d, Range(start, std::min(start + 50, d.size())))));
    return nn::concat(parts, 0);
  }
  replay::Map Snapshot() const { return sentence::Snapshot({&parameters}); }
  void        Save(const std::string& path) const { SaveStores(path, parameters); }
  void        Restore(const std::string& path) { RestoreStores(path, parameters); }
};

}  // namespace sentence
