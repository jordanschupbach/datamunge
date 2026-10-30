#pragma once
#include "common/sentence.hpp"
#include <datamunge/nn/wavenet_stack_data.hpp>
namespace wavestack {
namespace nn = datamunge::nn;
using datamunge::nn::models::LoadWaveStack;
using datamunge::nn::models::WaveData;
using datamunge::nn::models::WaveNll;
using datamunge::nn::models::WavePriorLogits;
using datamunge::nn::models::WaveRows;
using datamunge::nn::models::WaveSplits;
using datamunge::nn::models::WaveStack;
using datamunge::nn::models::WaveStackConfig;
using datamunge::nn::models::WaveStackConv;
using datamunge::nn::models::WaveStackField;
using datamunge::nn::models::WaveStackOutput;
using nn::Tensor;
using sentence::Range;
using sentence::Rows;
inline int MuLawEncode(double sample) {
  sample = std::clamp(sample, -1., 1.);
  const double y = std::copysign(std::log1p(255 * std::abs(sample)) / std::log(256.), sample);
  return std::clamp(static_cast<int>(std::floor((y + 1) * 127.5 + .5)), 0, 255);
}
inline double MuLawDecode(int code) {
  if (code < 0 || code > 255)
    throw std::invalid_argument("mu-law code");
  const double y = code / 127.5 - 1;
  return std::copysign(std::expm1(std::abs(y) * std::log(256.)) / 255, y);
}

class WaveStackModel {
public:
  nn::ParameterStore parameters{541};
  WaveStackConfig config;
  int loss_start;
  std::unique_ptr<nn::Optimizer> optimizer;

  explicit WaveStackModel(WaveStackConfig c = {8, 16, 6, 2, 256}, bool training = true,
                          int start = 127)
      : config(c), loss_start(start) {
    {
      nn::NoGradGuard guard;
      Logits(nn::zeros({1, 2}, nn::DType::Int32));
    }
    if (training)
      optimizer = std::make_unique<nn::Optimizer>(parameters, .001, replay::Adam());
  }
  Tensor Logits(const Tensor &codes) {
    return WaveStack(parameters,
                     nn::one_hot(codes, config.classes, 1.0, 0.0, parameters.dtype()), config)
        .logits;
  }
  std::vector<Tensor> Gradients(const WaveData &d) {
    return nn::grad(WaveNll(Logits(d.input), d.target, config.classes, loss_start),
                    parameters.parameters());
  }
  double Loss(const WaveData &d, bool update = false) {
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
  Tensor Predict(const Tensor &input, bool next = false) {
    nn::NoGradGuard guard;
    const Tensor logits = Logits(input);
    return (next ? nn::select(logits, 1, input.size(1) - 1) : logits).detach().contiguous();
  }
  replay::Map Snapshot() const {
    return sentence::Snapshot({&parameters});
  }
  void Save(const std::string &path) const {
    sentence::SaveStores(path, parameters);
  }
  void Restore(const std::string &path) {
    sentence::RestoreStores(path, parameters);
  }
  void Export(const std::filesystem::path &path) const {
    sentence::ExportArrays(path, parameters);
  }
};

struct WaveScore {
  double nll = 0, mse = 0, exact = 0;
  int samples = 0;
};
inline WaveScore WaveMetrics(const std::string &name, const Tensor &logits,
                             const WaveData &d) {
  const auto values = datamunge::nn::models::ScoreWave(logits, d).to_vector<double>();
  std::cout << name << ":NLL=" << values[0] << ",MSE=" << values[1] << ",exact=" << values[2]
            << '\n';
  return {values[0], values[1], values[2], int(d.size() * 129)};
}
inline std::pair<Tensor, Tensor> WavePriors(const WaveData &fit) {
  const auto counts = datamunge::nn::models::WaveCountTargets(fit);
  return {nn::log(counts.first).to(nn::DType::Float32),
          nn::log(counts.second).to(nn::DType::Float32)};
}
inline void WaveWrite(const std::filesystem::path &path, const Tensor &codes, int row) {
  nn::write_wav(path.string(),
                nn::mu_law_decode(nn::select(codes, 0, row)).reshape({codes.size(1), 1}),
                8000);
}
/// Three 128-code prompts continued by 256 codes sampled from the model
/// (std::minstd_rand(547) inverse-CDF draws, as in tflow), feeding samples back.
inline void WaveGenerate(WaveStackModel &model, const WaveSplits &data,
                         const std::filesystem::path &directory) {
  auto context = nn::slice_axis(data.prompts, 1, 0, 128);
  auto codes = context;
  std::vector<Tensor> step_logits, uniform_draws;
  std::minstd_rand rng(547);
  for (int step = 0; step < 256; ++step) {
    const auto next = model.Predict(context, true);
    std::vector<double> raw(3);
    for (double &u : raw)
      u = (double(rng()) - 1) / 2147483646.;
    const auto draws = nn::from_vector(raw, {3, 1});
    const auto cumulative = nn::cumsum(nn::softmax(next.to(nn::DType::Float64)), 1);
    const auto sampled =
        nn::minimum(nn::sum(nn::less_equal(cumulative, draws).to(nn::DType::Int32), {1}),
                    nn::full({3}, 255., nn::DType::Int32))
            .reshape({3, 1});
    codes = nn::concat({codes, sampled}, 1);
    context = nn::concat({nn::slice_axis(context, 1, 1, 128), sampled}, 1);
    step_logits.push_back(next);
    uniform_draws.push_back(draws);
  }
  const auto logits = nn::stack(step_logits, 1), draws = nn::concat(uniform_draws, 1);
  nn::save_npz((directory / "rollouts.npz").string(), {{"codes", codes},
                                                       {"reference", data.prompts},
                                                       {"source_ids", data.prompt_source_ids},
                                                       {"step_logits", logits},
                                                       {"uniform_draws", draws}});
  for (int n = 0; n < 3; ++n) {
    WaveWrite(directory / ("generated_" + std::to_string(n) + ".wav"), codes, n);
    WaveWrite(directory / ("reference_" + std::to_string(n) + ".wav"), data.prompts, n);
  }
  std::cout
      << " generated3 actual sampled-feedback continuations:128 observed codes+256 sampled "
         "codes each,8kHz/48ms total;recompute RF128context,no ground-truth feedback\n";
}

} // namespace wavestack
