#pragma once
// Retained independent original host metrics/count/scoring oracle.
namespace wave_reference {namespace nn=datamunge::nn;using nn::Tensor;using datamunge::nn::models::WaveData;using wavestack::MuLawDecode;
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

}
