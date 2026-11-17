#pragma once
#include "common/multilabel_data.hpp"
#include "datamunge/nn/multilabel_metrics.hpp"
namespace multilabel {
inline double MultilabelAP(const std::vector<double> &scores, const std::vector<int> &labels) {
  return nn::models::MultilabelAP(
             nn::from_vector(scores),
             nn::from_vector(std::vector<std::int64_t>(labels.begin(), labels.end())))
      .item();
}
struct MultilabelMetrics {
  double nll = 0, brier = 0, map = 0, micro_ap = 0, micro_f1 = 0, macro_f1 = 0, hamming = 0,
         exact = 0;
  int ap_classes = 0, absent_classes = 0;
  std::int64_t tp = 0, fp = 0, fn = 0, tn = 0;
};

inline MultilabelMetrics ScoreMultilabel(const nn::Tensor &probability,
                                         const nn::Tensor &labels, double threshold = .5) {
  const auto s = nn::models::ScoreMultilabel(probability, labels, threshold);
  return {s.nll.item(),
          s.brier.item(),
          s.map.item(),
          s.micro_ap.item(),
          s.micro_f1.item(),
          s.macro_f1.item(),
          s.hamming.item(),
          s.exact.item(),
          int(s.ap_classes.item<std::int64_t>()),
          int(s.absent_classes.item<std::int64_t>()),
          s.tp.item<std::int64_t>(),
          s.fp.item<std::int64_t>(),
          s.fn.item<std::int64_t>(),
          s.tn.item<std::int64_t>()};
}
inline double CalibrateMultilabel(const nn::Tensor &probability, const nn::Tensor &labels) {
  return nn::models::CalibrateMultilabel(probability, labels).item();
}
} // namespace multilabel
