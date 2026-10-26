// Exact entity spans with explicit I-after-O/type-change repair; padding
// excluded by valid lengths (port of tflow's common/ml/tagging_metrics.h).
// Tags are IOB2 with 0 = O, odd = B-type, even = I-type, type = (tag + 1) / 2.
#pragma once

#include <algorithm>
#include <cstdint>
#include <set>
#include <tuple>
#include <vector>

#include <datamunge/nn/tensor.hpp>

namespace tagging {

namespace nn = datamunge::nn;

using EntitySpan = std::tuple<int, int, int>;  // [start, end), type

inline std::set<EntitySpan> EntitySpans(const std::vector<int>& tags) {
  std::set<EntitySpan> result;
  int                  start = -1, type = 0;
  for (int t = 0; t <= static_cast<int>(tags.size()); ++t) {
    const int  tag   = t < static_cast<int>(tags.size()) ? tags[t] : 0;
    const int  kind  = tag ? (tag + 1) / 2 : 0;
    const bool begin = tag % 2 == 1;
    if (start >= 0 && (tag == 0 || kind != type || begin)) {
      result.emplace(start, t, type);
      start = -1;
    }
    if (tag && start < 0) {
      start = t;
      type  = kind;
    }
  }
  return result;
}

struct TaggingMetrics {
  int    tokens = 0, ner_correct = 0, true_positive = 0, predicted_entities = 0, gold_entities = 0;
  double NerAccuracy() const { return tokens ? double(ner_correct) / tokens : 0; }
  double EntityPrecision() const { return predicted_entities ? double(true_positive) / predicted_entities : 0; }
  double EntityRecall() const { return gold_entities ? double(true_positive) / gold_entities : 0; }
  double EntityF1() const {
    return predicted_entities + gold_entities ? 2. * true_positive / (predicted_entities + gold_entities) : 0;
  }
};

/// Scores predicted against gold NER tags [B, T] over the first lengths[b] positions.
inline TaggingMetrics ScoreTagging(const nn::Tensor& predicted, const nn::Tensor& gold, const nn::Tensor& lengths) {
  TaggingMetrics out;
  const auto     p = predicted.to_vector<int>(), g = gold.to_vector<int>(), n = lengths.to_vector<int>();
  const int      T = static_cast<int>(predicted.shape()[1]);
  for (std::size_t b = 0; b < n.size(); ++b) {
    std::vector<int> a, e;
    for (int t = 0; t < n[b]; ++t) {
      ++out.tokens;
      out.ner_correct += p[b * T + t] == g[b * T + t];
      a.push_back(p[b * T + t]);
      e.push_back(g[b * T + t]);
    }
    const auto sa = EntitySpans(a), sg = EntitySpans(e);
    out.predicted_entities += static_cast<int>(sa.size());
    out.gold_entities += static_cast<int>(sg.size());
    for (const auto& s : sa) out.true_positive += static_cast<int>(sg.count(s));
  }
  return out;
}

/// Per-token argmax of logits [B, T, K] on valid positions, -1 elsewhere (int32 [B, T]).
inline nn::Tensor TaggingArgmax(const nn::Tensor& logits, const nn::Tensor& lengths) {
  const auto B = logits.shape()[0], T = logits.shape()[1], K = logits.shape()[2];
  const auto v = logits.to_vector<float>();
  const auto n = lengths.to_vector<int>();
  std::vector<std::int32_t> out(static_cast<std::size_t>(B * T), -1);
  for (std::int64_t b = 0; b < B; ++b)
    for (std::int64_t t = 0; t < n[b]; ++t) {
      const float* row = v.data() + (b * T + t) * K;
      out[b * T + t]   = static_cast<std::int32_t>(std::max_element(row, row + K) - row);
    }
  return nn::from_vector(out, {B, T});
}

}  // namespace tagging
