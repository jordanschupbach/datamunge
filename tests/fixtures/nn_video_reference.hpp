#pragma once
// Independent original host metrics,all48features and nearest-neighbor oracle.
namespace video_reference{namespace nn=datamunge::nn;using nn::Tensor;using datamunge::nn::models::VideoData;
struct Score {
  double              accuracy = 0, macro_f1 = 0, nll = 0;
  std::vector<int>    confusion;
  std::vector<double> recall;
};

/// tflow's ScoreSensor: accuracy, macro F1, mean CE, recalls and confusion of [N, classes] logits.
inline Score ScoreLogits(const Tensor& logits, const Tensor& labels, int classes) {
  const std::int64_t count = labels.numel();
  check::Require(logits.shape() == nn::Shape({count, classes}), "classification score shape");
  const auto z = logits.to_vector<float>();
  const auto y = labels.to_vector<int>();
  Score      out;
  out.confusion.resize(static_cast<std::size_t>(classes * classes));
  out.recall.resize(static_cast<std::size_t>(classes));
  for (std::int64_t n = 0; n < count; ++n) {
    const int label = y[static_cast<std::size_t>(n)];
    check::Require(label >= 0 && label < classes, "metric source class range");
    const float* row = z.data() + n * classes;
    int          prediction = 0;
    double       maximum = -INFINITY, sum = 0;
    for (int c = 0; c < classes; ++c) {
      check::Require(std::isfinite(row[c]), "finite logit");
      maximum = std::max<double>(maximum, row[c]);
      if (row[c] > row[prediction]) prediction = c;
    }
    for (int c = 0; c < classes; ++c) sum += std::exp(row[c] - maximum);
    out.nll += maximum + std::log(sum) - row[label];
    ++out.confusion[static_cast<std::size_t>(label * classes + prediction)];
    out.accuracy += label == prediction;
  }
  for (int c = 0; c < classes; ++c) {
    int truth = 0, predicted = 0;
    for (int j = 0; j < classes; ++j) {
      truth += out.confusion[static_cast<std::size_t>(c * classes + j)];
      predicted += out.confusion[static_cast<std::size_t>(j * classes + c)];
    }
    check::Require(truth > 0, "every test class has support");
    const double tp = out.confusion[static_cast<std::size_t>(c * classes + c)];
    out.recall[static_cast<std::size_t>(c)] = tp / truth;
    out.macro_f1 += truth + predicted ? 2 * tp / (truth + predicted) : 0;
  }
  out.accuracy /= count;
  out.nll /= count;
  out.macro_f1 /= classes;
  return out;
}

inline void PrintVideo(const std::string& name, const VideoData& d, const Tensor& logits) {
  const auto r = ScoreLogits(logits, d.labels, 3);
  std::cout << " " << name << ":accuracy=" << 100 * r.accuracy << "%,macroF1=" << r.macro_f1 << ",CE=" << r.nll
            << ",class recalls=";
  for (double x : r.recall) std::cout << x << ',';
  std::cout << ",confusion=";
  for (int x : r.confusion) std::cout << x << ',';
  std::cout << '\n';
}

/// 4x4 spatial grid of mean colour (or mean absolute frame difference with `motion`) per clip: [N, 48].
inline std::vector<std::vector<double>> VideoFeatures(const VideoData& d, bool motion) {
  const auto                       rgb = d.rgb.to_vector<float>();
  std::vector<std::vector<double>> out(static_cast<std::size_t>(d.size()), std::vector<double>(48, 0.0));
  const auto at = [&](std::int64_t n, int t, int y, int x, int c) {
    return static_cast<double>(rgb[static_cast<std::size_t>((((n * 16 + t) * 32 + y) * 32 + x) * 3 + c)]);
  };
  for (std::int64_t n = 0; n < d.size(); ++n)
    for (int gy = 0; gy < 4; ++gy)
      for (int gx = 0; gx < 4; ++gx)
        for (int c = 0; c < 3; ++c) {
          double sum = 0;
          for (int t = motion ? 1 : 0; t < 16; ++t)
            for (int y = gy * 8; y < (gy + 1) * 8; ++y)
              for (int x = gx * 8; x < (gx + 1) * 8; ++x)
                sum += motion ? std::abs(at(n, t, y, x, c) - at(n, t - 1, y, x, c)) : at(n, t, y, x, c);
          // Stored as float, as tflow's feature tensor is.
          out[static_cast<std::size_t>(n)][static_cast<std::size_t>((gy * 4 + gx) * 3 + c)] =
              static_cast<float>(sum / ((motion ? 15 : 16) * 64));
        }
  return out;
}

/// Standardised 1-nearest-neighbour one-hot "logits" from fit clips.
inline Tensor VideoNearest(const VideoData& fit, const VideoData& test, bool motion) {
  const auto          reference = video_reference::VideoFeatures(fit, motion), query = video_reference::VideoFeatures(test, motion);
  const auto          labels    = fit.labels.to_vector<int>();
  std::vector<double> deviation(48);
  for (std::size_t c = 0; c < 48; ++c) {
    double mean = 0, variance = 0;
    for (const auto& r : reference) mean += r[c];
    mean /= static_cast<double>(reference.size());
    for (const auto& r : reference) variance += std::pow(r[c] - mean, 2);
    deviation[c] = std::max(1e-6, std::sqrt(variance / static_cast<double>(reference.size())));
  }
  std::vector<float> out(query.size() * 3, 0.f);
  for (std::size_t n = 0; n < query.size(); ++n) {
    double best       = INFINITY;
    int    prediction = 0;
    for (std::size_t i = 0; i < reference.size(); ++i) {
      double distance = 0;
      for (std::size_t c = 0; c < 48; ++c) distance += std::pow((query[n][c] - reference[i][c]) / deviation[c], 2);
      if (distance < best) {
        best       = distance;
        prediction = labels[i];
      }
    }
    out[n * 3 + static_cast<std::size_t>(prediction)] = 1;
  }
  return nn::from_vector(out, {static_cast<std::int64_t>(query.size()), 3});
}

}
