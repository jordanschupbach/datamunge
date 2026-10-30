#pragma once
// Retained independent original host block-feature and nearest-neighbor oracle.
namespace medical_reference{namespace nn=datamunge::nn;using nn::Tensor;using datamunge::nn::models::MedicalData;
inline Tensor MedicalBaseline(const MedicalData& fit, const MedicalData& test) {
  const auto features = [](const Tensor& x) {
    const auto                       v = x.to_vector<float>();
    const std::int64_t               n = x.size(0);
    std::vector<std::vector<double>> out(static_cast<std::size_t>(n), std::vector<double>(64));
    for (std::int64_t b = 0; b < n; ++b)
      for (int z = 0; z < 28; ++z)
        for (int y = 0; y < 28; ++y)
          for (int a = 0; a < 28; ++a)
            out[static_cast<std::size_t>(b)][static_cast<std::size_t>((z / 7 * 4 + y / 7) * 4 + a / 7)] +=
                v[static_cast<std::size_t>(((b * 28 + z) * 28 + y) * 28 + a)] / 343;
    return out;
  };
  const auto         a = features(fit.voxels), b = features(test.voxels);
  const auto         labels = fit.labels.to_vector<int>();
  std::vector<float> out(b.size() * 2, 0.f);
  for (std::size_t i = 0; i < b.size(); ++i) {
    double best  = INFINITY;
    int    label = 0;
    for (std::size_t j = 0; j < a.size(); ++j) {
      double distance = 0;
      for (int k = 0; k < 64; ++k) distance += std::pow(a[j][static_cast<std::size_t>(k)] - b[i][static_cast<std::size_t>(k)], 2);
      if (distance < best) {
        best  = distance;
        label = labels[j];
      }
    }
    out[i * 2 + static_cast<std::size_t>(label)] = 1;
  }
  return nn::from_vector(out, {static_cast<std::int64_t>(b.size()), 2});
}

}
