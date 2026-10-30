#pragma once
// Independent retained original host descriptor and memory1NN oracle.
namespace two_stream_reference {
namespace nn = datamunge::nn;
using nn::Tensor;
using A = datamunge::nn::models::TwoStreamData;
inline Tensor GridFeatures(const A &d, bool motion) {
  int channels = motion ? 2 : 3, n = d.at("rgb").size(0);
  std::vector<float> out(n * 16 * channels);
  auto input = d.at(motion ? "flow" : "rgb").to_vector<float>();
  for (int b = 0; b < n; ++b)
    for (int gy = 0; gy < 4; ++gy)
      for (int gx = 0; gx < 4; ++gx)
        for (int c = 0; c < channels; ++c) {
          double sum = 0;
          for (int t = 0; t < (motion ? 10 : 1); ++t)
            for (int y = gy * 24; y < (gy + 1) * 24; ++y)
              for (int x = gx * 24; x < (gx + 1) * 24; ++x) {
                auto i = motion ? ((((b * 10 + t) * 96 + y) * 96 + x) * 2 + c)
                                : ((b * 96 + y) * 96 + x) * 3 + c;
                sum += motion ? std::abs(input[i]) : input[i];
              }
          out[(b * 16 + gy * 4 + gx) * channels + c] = sum / (576 * (motion ? 10 : 1));
        }
  return nn::from_vector(out, {n, 16 * channels});
}
inline Tensor Nearest(const A &fit, const A &test, bool motion) {
  auto a = GridFeatures(fit, motion), b = GridFeatures(test, motion);
  int dims = a.size(1);
  auto av = a.to_vector<float>(), bv = b.to_vector<float>();
  auto labels = fit.at("label").to_vector<int>();
  std::vector<double> scale(dims);
  for (int c = 0; c < dims; ++c) {
    double mean = 0, var = 0;
    for (int i = 0; i < a.size(0); ++i)
      mean += av[i * dims + c] / double(a.size(0));
    for (int i = 0; i < a.size(0); ++i)
      var += std::pow(av[i * dims + c] - mean, 2) / a.size(0);
    scale[c] = std::max(1e-6, std::sqrt(var));
  }
  std::vector<float> out(b.size(0) * 3);
  for (int n = 0; n < b.size(0); ++n) {
    double best = INFINITY;
    int label = 0;
    for (int i = 0; i < a.size(0); ++i) {
      double distance = 0;
      for (int c = 0; c < dims; ++c)
        distance += std::pow((av[i * dims + c] - bv[n * dims + c]) / scale[c], 2);
      if (distance < best) {
        best = distance;
        label = labels[i];
      }
    }
    out[n * 3 + label] = 1;
  }
  return nn::from_vector(out, {b.size(0), 3});
}

} // namespace two_stream_reference
