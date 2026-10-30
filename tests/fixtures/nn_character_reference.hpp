#pragma once
// Retained independent original host raw-character count/scoring oracle.
namespace character_reference {namespace nn=datamunge::nn;using nn::Tensor;using datamunge::nn::models::CharacterData;
inline Tensor CharacterNB(const CharacterData& fit, const CharacterData& test) {
  std::vector<std::vector<double>> counts(2, std::vector<double>(69, 1));
  double                           totals[2] = {69, 69}, prior[2] = {1, 1};
  const auto ft = fit.tokens.to_vector<int>(), fl = fit.lengths.to_vector<int>(), fy = fit.labels.to_vector<int>();
  const std::int64_t T = fit.tokens.size(1);
  for (std::int64_t n = 0; n < fit.size(); ++n) {
    const int label = fy[n];
    ++prior[label];
    for (int t = 0; t < fl[n]; ++t) {
      const int c = ft[n * T + t];
      if (c >= 0) {
        ++counts[label][c];
        ++totals[label];
      }
    }
  }
  const auto         tt = test.tokens.to_vector<int>(), tl = test.lengths.to_vector<int>();
  std::vector<float> result(static_cast<std::size_t>(2 * test.size()));
  for (std::int64_t n = 0; n < test.size(); ++n)
    for (int label = 0; label < 2; ++label) {
      double score = std::log(prior[label] / (prior[0] + prior[1]));
      for (int t = 0; t < tl[n]; ++t) {
        const int c = tt[n * T + t];
        if (c >= 0) score += std::log(counts[label][c] / totals[label]);
      }
      result[2 * n + label] = static_cast<float>(score);
    }
  return nn::from_vector(result, {test.size(), 2});
}

}
