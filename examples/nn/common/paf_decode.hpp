#pragma once
#include "common/paf_data.hpp"
#include <datamunge/nn/paf_decode.hpp>
namespace pose {
using datamunge::nn::models::AssemblePafPeople;
using datamunge::nn::models::DecodePafs;
using datamunge::nn::models::PafDecoded;
using datamunge::nn::models::PafLink;
using datamunge::nn::models::PafMetrics;
using datamunge::nn::models::PafPeak;
using datamunge::nn::models::PafPerson;
using datamunge::nn::models::ScorePafs;
struct PafField {
  std::vector<float> v;
  std::int64_t n = 0, h = 0, w = 0, c = 0;
  PafField() = default;
  PafField(const nn::Tensor &t)
      : v(t.to_vector<float>()) { // NOLINT(google-explicit-constructor)
    check::Require(t.dim() == 4, "NHWC field");
    n = t.size(0), h = t.size(1), w = t.size(2), c = t.size(3);
  }
  float operator()(std::int64_t b, std::int64_t y, std::int64_t x, std::int64_t ch) const {
    return v[std::size_t(((b * h + y) * w + x) * c + ch)];
  }
};

inline datamunge::nn::models::PafTensorField TensorField(const PafField &f) {
  return datamunge::nn::models::PafTensorField(nn::from_vector(f.v, {f.n, f.h, f.w, f.c}));
}
inline float PafSample(const PafField &f, int batch, float x, float y, int channel) {
  return nn::models::PafSample(TensorField(f), batch, x, y, channel);
}
inline std::vector<PafPeak> PafPeaks(const PafField &f, int batch, float threshold = .05f,
                                     int cap = 20) {
  return nn::models::PafPeaks(TensorField(f), batch, threshold, cap);
}
inline std::vector<PafLink> MatchPafPeaks(const std::vector<PafPeak> &p, const PafField &f,
                                          int batch, float projection = .05f,
                                          float fraction = .8f, int samples = 10) {
  return nn::models::MatchPafPeaks(p, TensorField(f), batch, projection, fraction, samples);
}
inline PafDecoded DecodePafs(const PafField &h, const PafField &f, int batch) {
  return nn::models::DecodePafs(TensorField(h), TensorField(f), batch);
}
} // namespace pose
