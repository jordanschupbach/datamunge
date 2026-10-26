// WaveNet-style tanh/sigmoid gated dilated residual/skip stack with
// next-code logits, and short speaker-disjoint raw-audio next-code training
// with genuine sampled feedback (port of tflow's common/nn/wavenet_stack.h and
// common/ml/wavenet_stack.h). tflow zero-spaces dilated taps by hand to dodge
// a TF SDK gradient bug; nn::conv1d takes the dilation directly.
#pragma once

#include <algorithm>
#include <random>

#include "common/sentence.hpp"

namespace wavestack {

namespace nn = datamunge::nn;
using nn::Tensor;
using sentence::Range;
using sentence::Rows;

inline int MuLawEncode(double sample) {
  sample = std::clamp(sample, -1., 1.);
  const double y = std::copysign(std::log1p(255 * std::abs(sample)) / std::log(256.), sample);
  return std::clamp(static_cast<int>(std::floor((y + 1) * 127.5 + .5)), 0, 255);
}
inline double MuLawDecode(int code) {
  if (code < 0 || code > 255) throw std::invalid_argument("mu-law code");
  const double y = code / 127.5 - 1;
  return std::copysign(std::expm1(std::abs(y) * std::log(256.)) / 255, y);
}

struct WaveStackConfig {
  int residual = 32, skip = 64, levels = 6, cycles = 2, classes = 256;
};
inline int WaveStackField(const WaveStackConfig& config) {
  if (config.residual < 1 || config.skip < 1 || config.levels < 1 || config.levels > 12 || config.cycles < 1 || config.cycles > 4 ||
      config.classes < 2)
    throw std::invalid_argument("WaveNet stack configuration");
  return 2 + config.cycles * ((1 << config.levels) - 1);
}

/// Causal (left-padded) dilated convolution: "<name>/W" [1, k, in, out] ~
/// U(+-sqrt(6 / (k (in + out)))), "<name>/b" = 0.
inline Tensor WaveStackConv(nn::ParameterStore& p, const std::string& name, const Tensor& x, int out, int kernel = 1, int dilation = 1) {
  const std::int64_t in = x.size(-1);
  const Tensor w = p.get(name + "/W", {1, kernel, in, out}, nn::init::uniform(std::sqrt(6.f / (kernel * (in + out)))));
  const Tensor b = p.get(name + "/b", {out}, nn::init::zeros());
  const Tensor padded = nn::pad(x, {{0, 0}, {(kernel - 1) * dilation, 0}, {0, 0}});
  return nn::bias_add(nn::conv1d(padded, nn::reshape(w, {kernel, in, out}), 1, nn::Padding::Valid, dilation), b);
}

struct WaveStackOutput {
  Tensor              logits, skip;
  std::vector<Tensor> gates;
};

/// encoded [N, T, classes] (one-hot codes) -> next-code logits [N, T, classes].
inline WaveStackOutput WaveStack(nn::ParameterStore& p, const Tensor& encoded, const WaveStackConfig& config = {}) {
  (void)WaveStackField(config);
  WaveStackOutput     result;
  Tensor              x = WaveStackConv(p, "input", encoded, config.residual, 2);
  std::vector<Tensor> skips;
  const int           blocks = config.levels * config.cycles;
  for (int block = 0; block < blocks; ++block) {
    const auto   name = "block" + std::to_string(block + 1);
    const Tensor joined = WaveStackConv(p, name + "/gate", x, 2 * config.residual, 2, 1 << (block % config.levels));
    const auto   split = nn::split(joined, 2, 2);
    const Tensor gate = nn::mul(nn::tanh(split[0]), nn::sigmoid(split[1]));
    result.gates.push_back(gate);
    skips.push_back(WaveStackConv(p, name + "/skip", gate, config.skip));
    if (block + 1 < blocks) x = nn::add(x, WaveStackConv(p, name + "/residual", gate, config.residual));
  }
  result.skip = skips[0];
  for (std::size_t i = 1; i < skips.size(); ++i) result.skip = nn::add(result.skip, skips[i]);
  Tensor hidden = nn::relu(result.skip);
  hidden = nn::relu(WaveStackConv(p, "post", hidden, config.skip));
  result.logits = WaveStackConv(p, "output", hidden, config.classes);
  return result;
}

// ---- data ------------------------------------------------------------------------------------

struct WaveData {
  Tensor input, target, source, speaker, start;
  [[nodiscard]] std::int64_t size() const { return input.size(0); }
};
struct WaveSplits {
  std::vector<WaveData> pools;
  Tensor                prompts, prompt_source_ids;
};

inline WaveData WaveRows(const WaveData& d, const Rows& rows) {
  using sentence::Select;
  return {Select(d.input, rows), Select(d.target, rows), Select(d.source, rows), Select(d.speaker, rows), Select(d.start, rows)};
}

inline WaveSplits LoadWaveStack(const std::filesystem::path& root) {
  const auto                   arrays = nn::load_npz((root / "wavenet_stack.npz").string());
  WaveSplits                   result;
  std::set<int>                previous_sources, previous_speakers;
  std::set<std::pair<int, int>> windows;
  for (const std::string split : {"fit", "validation", "test"}) {
    using sentence::Npz;
    WaveData d{Npz(arrays, split + "_input"), Npz(arrays, split + "_target"), Npz(arrays, split + "_source_id"),
               Npz(arrays, split + "_speaker_id"), Npz(arrays, split + "_start")};
    const std::int64_t n = d.input.size(0);
    check::Require(d.input.dtype() == nn::DType::Int32 && d.target.dtype() == nn::DType::Int32 && d.input.shape() == nn::Shape{n, 256} &&
                       d.target.shape() == d.input.shape(),
                   "audio input/next-code target matrices");
    for (const auto* t : {&d.source, &d.speaker, &d.start})
      check::Require(t->dtype() == nn::DType::Int32 && t->shape() == nn::Shape{n}, "audio source metadata alignment");
    const auto    input = d.input.to_vector<int>(), target = d.target.to_vector<int>(), source = d.source.to_vector<int>(),
               speaker = d.speaker.to_vector<int>(), start = d.start.to_vector<int>();
    std::set<int> sources, speakers;
    for (std::int64_t i = 0; i < n; ++i) {
      check::Require(source[i] >= 0 && source[i] < 27 && speaker[i] >= 0 && speaker[i] < 6 && start[i] >= 0 && start[i] <= 768 &&
                         start[i] % 256 == 0 && !previous_sources.count(source[i]) && !previous_speakers.count(speaker[i]) &&
                         windows.insert({source[i], start[i]}).second,
                     "audio source/speaker disjointness and unique fixed windows");
      sources.insert(source[i]);
      speakers.insert(speaker[i]);
      for (int t = 0; t < 256; ++t) {
        const int x = input[i * 256 + t], y = target[i * 256 + t];
        check::Require(x >= 0 && x < 256 && y >= 0 && y < 256, "mu-law code range");
        if (t > 0) check::Require(x == target[i * 256 + t - 1], "strict one-sample teacher-forcing shift");
      }
    }
    previous_sources.insert(sources.begin(), sources.end());
    previous_speakers.insert(speakers.begin(), speakers.end());
    std::cout << " FSDD " << split << ":" << n << " windows/" << sources.size() << " clips/" << speakers.size() << " disjoint speakers\n";
    result.pools.push_back(d);
  }
  check::Require(result.pools[0].size() == 72 && result.pools[1].size() == 12 && result.pools[2].size() == 24, "pinned tiny audio pool counts");
  result.prompts = sentence::Npz(arrays, "prompts");
  result.prompt_source_ids = sentence::Npz(arrays, "prompt_source_ids");
  check::Require(result.prompts.dtype() == nn::DType::Int32 && result.prompts.shape() == nn::Shape{3, 384} &&
                     result.prompt_source_ids.dtype() == nn::DType::Int32 && result.prompt_source_ids.shape() == nn::Shape{3},
                 "three held-out 128-seed/256-reference prompts");
  for (int v : result.prompts.to_vector<int>()) check::Require(v >= 0 && v < 256, "prompt code range");
  return result;
}

/// Mean next-code NLL over positions t >= start (warmup masked out).
inline Tensor WaveNll(const Tensor& logits, const Tensor& target, int classes = 256, int start = 127) {
  const std::int64_t batch = target.size(0), time = target.size(1);
  const Tensor       scores = nn::reshape(logits, {-1, classes});
  const Tensor       onehot = nn::one_hot(nn::reshape(target, {-1}), classes, 1.0, 0.0, logits.dtype());
  const Tensor       values = nn::reshape(nn::neg(nn::sum(nn::mul(onehot, nn::log_softmax(scores)), {1})), {batch, time});
  const Tensor       mask = nn::cast(nn::greater_equal(nn::arange(0, static_cast<double>(time)), static_cast<double>(start)), logits.dtype());
  const double       count = std::max<double>(0, time - start);
  return nn::div(nn::sum(nn::mul(values, mask)), static_cast<double>(batch) * count);
}

class WaveStackModel {
 public:
  nn::ParameterStore             parameters{541};
  WaveStackConfig                config;
  int                            loss_start;
  std::unique_ptr<nn::Optimizer> optimizer;

  explicit WaveStackModel(WaveStackConfig c = {8, 16, 6, 2, 256}, bool training = true, int start = 127) : config(c), loss_start(start) {
    {
      nn::NoGradGuard guard;
      Logits(nn::zeros({1, 2}, nn::DType::Int32));
    }
    if (training) optimizer = std::make_unique<nn::Optimizer>(parameters, .001, replay::Adam());
  }
  Tensor Logits(const Tensor& codes) {
    return WaveStack(parameters, nn::one_hot(codes, config.classes, 1.0, 0.0, parameters.dtype()), config).logits;
  }
  std::vector<Tensor> Gradients(const WaveData& d) {
    return nn::grad(WaveNll(Logits(d.input), d.target, config.classes, loss_start), parameters.parameters());
  }
  double Loss(const WaveData& d, bool update = false) {
    double value;
    if (update) {
      const Tensor loss = WaveNll(Logits(d.input), d.target, config.classes, loss_start);
      optimizer->step(nn::grad(loss, parameters.parameters()));
      value = loss.item<double>();
    } else {
      nn::NoGradGuard guard;
      value = WaveNll(Logits(d.input), d.target, config.classes, loss_start).item<double>();
    }
    check::Require(std::isfinite(value) && value >= 0, "finite audio next-code NLL");
    return value;
  }
  /// All-time logits [N, T, classes], or the last step's [N, classes].
  Tensor Predict(const Tensor& input, bool next = false) {
    nn::NoGradGuard guard;
    const Tensor    logits = Logits(input);
    return (next ? nn::select(logits, 1, input.size(1) - 1) : logits).detach().contiguous();
  }
  replay::Map Snapshot() const { return sentence::Snapshot({&parameters}); }
  void        Save(const std::string& path) const { sentence::SaveStores(path, parameters); }
  void        Restore(const std::string& path) { sentence::RestoreStores(path, parameters); }
  void        Export(const std::filesystem::path& path) const { sentence::ExportArrays(path, parameters); }
};

struct WaveScore {
  double nll = 0, mse = 0, exact = 0;
  int    samples = 0;
};

inline WaveScore WaveMetrics(const std::string& name, const Tensor& logits, const WaveData& d) {
  check::Require(logits.shape() == nn::Shape{d.size(), 256, 256}, "audio categorical logit shape");
  const auto z = logits.to_vector<float>();
  const auto target = d.target.to_vector<int>();
  WaveScore  result;
  for (std::int64_t n = 0; n < d.size(); ++n)
    for (int t = 127; t < 256; ++t) {
      const float* row = z.data() + (n * 256 + t) * 256;
      double       maximum = -INFINITY, sum = 0, expectation = 0;
      int          prediction = 0;
      for (int c = 0; c < 256; ++c) {
        const double x = row[c];
        check::Require(std::isfinite(x), "finite audio categorical logits");
        maximum = std::max(maximum, x);
        if (x > row[prediction]) prediction = c;
      }
      for (int c = 0; c < 256; ++c) {
        const double p = std::exp(row[c] - maximum);
        sum += p;
        expectation += p * MuLawDecode(c);
      }
      const int y = target[n * 256 + t];
      result.nll += maximum + std::log(sum) - row[y];
      result.exact += prediction == y;
      result.mse += std::pow(expectation / sum - MuLawDecode(y), 2);
      ++result.samples;
    }
  result.nll /= result.samples;
  result.mse /= result.samples;
  result.exact /= result.samples;
  std::cout << " " << name << ":NLL=" << result.nll << ",bits/sample=" << result.nll / std::log(2.) << ",code exact=" << 100 * result.exact
            << "%,teacher-forced amplitude MSE=" << result.mse << " over" << result.samples << " scored samples\n";
  return result;
}

/// Add-one log counts of fit targets (unigram [256]) and of (input, target) pairs ([256, 256]), positions >= 127.
inline std::pair<Tensor, Tensor> WavePriors(const WaveData& fit) {
  std::vector<float> unigram(256, 1), transition(256 * 256, 1);
  const auto         input = fit.input.to_vector<int>(), target = fit.target.to_vector<int>();
  for (std::int64_t n = 0; n < fit.size(); ++n)
    for (int t = 127; t < 256; ++t) {
      ++unigram[target[n * 256 + t]];
      ++transition[input[n * 256 + t] * 256 + target[n * 256 + t]];
    }
  for (auto& v : unigram) v = std::log(v);
  for (auto& v : transition) v = std::log(v);
  return {nn::from_vector(unigram, {256}), nn::from_vector(transition, {256, 256})};
}

inline Tensor WavePriorLogits(const WaveData& d, const Tensor& prior) {
  const auto         p = prior.to_vector<float>();
  const auto         input = d.input.to_vector<int>();
  std::vector<float> out(static_cast<std::size_t>(d.size() * 256 * 256));
  for (std::int64_t n = 0; n < d.size(); ++n)
    for (int t = 0; t < 256; ++t)
      for (int c = 0; c < 256; ++c) out[(n * 256 + t) * 256 + c] = prior.dim() == 1 ? p[c] : p[input[n * 256 + t] * 256 + c];
  return nn::from_vector(out, {d.size(), 256, 256});
}

inline void WaveWrite(const std::filesystem::path& path, const Tensor& codes, int row) {
  const std::int64_t length = codes.size(1);
  const auto         c = codes.to_vector<int>();
  std::vector<float> samples(static_cast<std::size_t>(length));
  for (std::int64_t t = 0; t < length; ++t) samples[t] = static_cast<float>(MuLawDecode(c[row * length + t]));
  nn::write_wav(path.string(), nn::from_vector(samples, {length, 1}), 8000);
}

/// Three 128-code prompts continued by 256 codes sampled from the model
/// (std::minstd_rand(547) inverse-CDF draws, as in tflow), feeding samples back.
inline void WaveGenerate(WaveStackModel& model, const WaveSplits& data, const std::filesystem::path& directory) {
  const auto         prompts = data.prompts.to_vector<int>();
  std::vector<int>   context(3 * 128), combined(3 * 384);
  std::vector<float> logits(3 * 256 * 256);
  std::vector<double> draws(3 * 256);
  for (int n = 0; n < 3; ++n)
    for (int t = 0; t < 128; ++t) context[n * 128 + t] = combined[n * 384 + t] = prompts[n * 384 + t];
  std::minstd_rand rng(547);
  for (int step = 0; step < 256; ++step) {
    const auto next = model.Predict(nn::from_vector(context, {3, 128}), true).to_vector<float>();
    for (int n = 0; n < 3; ++n) {
      const double u = (double(rng()) - 1) / 2147483646.;
      draws[n * 256 + step] = u;
      double maximum = -INFINITY, sum = 0;
      for (int c = 0; c < 256; ++c) {
        const double z = next[n * 256 + c];
        check::Require(std::isfinite(z), "finite generation logits");
        maximum = std::max(maximum, z);
        logits[(n * 256 + step) * 256 + c] = next[n * 256 + c];
      }
      for (int c = 0; c < 256; ++c) sum += std::exp(next[n * 256 + c] - maximum);
      double cumulative = 0;
      int    code = 255;
      for (int c = 0; c < 256; ++c) {
        cumulative += std::exp(next[n * 256 + c] - maximum);
        if (cumulative > u * sum) {
          code = c;
          break;
        }
      }
      combined[n * 384 + 128 + step] = code;
      for (int t = 0; t < 127; ++t) context[n * 128 + t] = context[n * 128 + t + 1];
      context[n * 128 + 127] = code;
    }
  }
  const Tensor codes = nn::from_vector(combined, {3, 384});
  nn::save_npz((directory / "rollouts.npz").string(),
               {{"codes", codes}, {"reference", data.prompts}, {"source_ids", data.prompt_source_ids},
                {"step_logits", nn::from_vector(logits, {3, 256, 256})}, {"uniform_draws", nn::from_vector(draws, {3, 256})}});
  for (int n = 0; n < 3; ++n) {
    WaveWrite(directory / ("generated_" + std::to_string(n) + ".wav"), codes, n);
    WaveWrite(directory / ("reference_" + std::to_string(n) + ".wav"), data.prompts, n);
  }
  std::cout << " generated3 actual sampled-feedback continuations:128 observed codes+256 sampled codes each,8kHz/48ms total;recompute RF128context,no ground-truth feedback\n";
}

}  // namespace wavestack
