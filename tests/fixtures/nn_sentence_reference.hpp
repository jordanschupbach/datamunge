#pragma once
// Retained independent original host multinomial-count scoring oracle.
namespace word_reference {namespace nn=datamunge::nn;using nn::Tensor;using datamunge::nn::models::SentenceData;
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

}
