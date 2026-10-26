// Original-width VoxNet: 32-cube, two convolutions, max pool, FC128, classifier.
// Port of tflow's common/{nn,ml}/voxnet.h; native seeded dropout/initialization
// gives statistical training parity; TF-trained weights are checked separately.
#pragma once
#include <set>

#include "common/ecg_sensor.hpp"
#include "common/tf_replay.hpp"
namespace vox {
namespace nn = datamunge::nn;
using nn::Tensor;
using replay::Map;
inline Tensor Rotate(Tensor x, int turn) {
  if (turn < 0 || turn > 3)
    throw std::invalid_argument("quarter turn0..3");
  for (int i = 0; i < turn; ++i)
    x = nn::reverse(nn::transpose(x, {0, 2, 1, 3, 4}), {1});
  return x;
}
inline std::vector<Tensor>
Regularized(const nn::ParameterStore& p, const std::vector<Tensor>& ce, double lambda = .001) {
  if (ce.size() != p.size() || lambda < 0)
    throw std::invalid_argument("VoxNet regularized gradients");
  auto out = ce;
  for (std::size_t i = 0; i < p.size(); ++i) {
    const auto& name = p.names()[i];
    if (name.size() >= 2 && name.substr(name.size() - 2) == "/W")
      out[i] = ce[i] + p.at(name) * (2 * lambda);
  }
  return out;
}
struct Path {
  Tensor conv1, conv2, pool, flat, fc1, logits;
};
struct Dropout {
  nn::Generator first{881}, second{883}, last{887};
};
inline Path Forward(
    nn::ParameterStore& p,
    const Tensor&       occupancy,
    int                 classes  = 40,
    bool                training = false,
    Dropout*            dropout  = nullptr) {
  if (classes < 2 || (training && !dropout))
    throw std::invalid_argument("VoxNet configuration");
  Path                 out;
  nn::ConvLayerOptions o;
  o.padding = nn::Padding::Valid;
  o.stride  = 2;
  o.init    = nn::init::he_normal();
  out.conv1 = nn::leaky_relu(nn::conv3d(p, "conv1", occupancy * 2. - 1., 32, 5, o), .1);
  auto x    = out.conv1;
  if (training)
    x = nn::dropout(x, .2, true, dropout->first);
  o.stride  = 1;
  out.conv2 = nn::leaky_relu(nn::conv3d(p, "conv2", x, 32, 3, o), .1);
  out.pool  = nn::max_pool3d(out.conv2, 2, 2, nn::Padding::Valid);
  x         = out.pool;
  if (training)
    x = nn::dropout(x, .3, true, dropout->second);
  out.flat = nn::reshape(x, {x.size(0), 6912});
  nn::DenseOptions dense;
  dense.init       = nn::init::normal(.01);
  dense.activation = nn::Activation::Relu;
  out.fc1          = nn::dense(p, "fc1", out.flat, 128, dense);
  x                = out.fc1;
  if (training)
    x = nn::dropout(x, .4, true, dropout->last);
  dense.activation = nn::Activation::None;
  out.logits       = nn::dense(p, "head", x, classes, dense);
  return out;
}
struct Data {
  Tensor voxels, labels, source;
};
inline Data Rows(const Data& d, const std::vector<std::int64_t>& rows) {
  return {
      nn::take_rows(d.voxels, rows), nn::take_rows(d.labels, rows), nn::take_rows(d.source, rows)};
}
inline std::vector<Data> Load(const std::filesystem::path& root) {
  const auto            arrays = replay::ToMap(nn::load_npz((root / "voxnet.npz").string()));
  std::vector<Data>     pools;
  std::set<int>         ids;
  std::set<std::string> hashes;
  for (const std::string split : {"fit", "validation", "test"}) {
    Data d{
        arrays.at(split + "_voxels"), arrays.at(split + "_label"), arrays.at(split + "_source_id")};
    const auto n = d.voxels.size(0);
    check::Require(
        n == (split == "fit"          ? 24
              : split == "validation" ? 6
                                      : 12) &&
            d.voxels.dtype() == nn::DType::Float32 &&
            d.voxels.shape() == nn::Shape({n, 32, 32, 32, 1}),
        "published 32-cube occupancy/object count");
    check::Require(
        d.labels.dtype() == nn::DType::Int32 && d.source.dtype() == nn::DType::Int32 &&
            d.labels.shape() == nn::Shape({n}) && d.source.shape() == nn::Shape({n}),
        "voxel metadata alignment");
    std::vector<int> counts(3);
    const auto       labels = d.labels.to_vector<int>(), source = d.source.to_vector<int>();
    const auto       values = d.voxels.to_vector<float>();
    for (int i = 0; i < n; ++i) {
      check::Require(
          labels[i] >= 0 && labels[i] < 3 && ids.insert(source[i]).second,
          "disjoint held-out object IDs");
      ++counts[labels[i]];
      double occupied = 0;
      for (int x = 0; x < 32; ++x)
        for (int y = 0; y < 32; ++y)
          for (int z = 0; z < 32; ++z) {
            float v = values[i * 32768 + (x * 32 + y) * 32 + z];
            check::Require(v == 0 || v == 1, "binary occupancy");
            if (x == 0 || x == 31 || y == 0 || y == 31 || z == 0 || z == 31)
              check::Near(v, 0, 0, "empty one-voxel border");
            occupied += v;
          }
      check::Require(occupied > 0 && occupied < 27000, "nonempty genuine CAD occupancy");
      check::Require(
          hashes
              .insert(
                  std::string(
                      reinterpret_cast<const char*>(values.data() + i * 32768),
                      32768 * sizeof(float)))
              .second,
          "no duplicate occupancy across objects/splits");
    }
    for (int count : counts)
      check::Require(count == n / 3, "balanced pinned classes");
    std::cout << "ModelNet10 " << split << ":" << n
              << " distinct objects,balanced chair/sofa/toilet\n";
    pools.push_back(d);
  }
  return pools;
}
class Model {
 public:
  nn::ParameterStore             parameters{907};
  Dropout                        dropout;
  std::unique_ptr<nn::Optimizer> optimizer;
  explicit Model(bool training = true) {
    {
      nn::NoGradGuard guard;
      Forward(parameters, nn::zeros({1, 32, 32, 32, 1}), 3);
    }
    if (training) {
      nn::OptimizerConfig o;
      o.kind     = nn::OptimizerKind::Momentum;
      o.momentum = .9;
      optimizer  = std::make_unique<nn::Optimizer>(parameters, .001, o);
    }
  }
  Tensor Predict(const Data& d, bool vote = true) {
    nn::NoGradGuard guard;
    if (!vote)
      return Forward(parameters, d.voxels, 3).logits;
    std::vector<Tensor> values;
    for (int k = 0; k < 4; ++k)
      values.push_back(Forward(parameters, Rotate(d.voxels, k), 3).logits);
    auto out = values[0];
    for (int k = 1; k < 4; ++k)
      out = out + values[k];
    return out / 4.;
  }
  Tensor Classification(const Data& d) {
    return nn::sparse_softmax_cross_entropy(
        Forward(parameters, d.voxels, 3, true, &dropout).logits, d.labels);
  }
  double Update(const Data& d) {
    auto   ce      = Classification(d);
    double penalty = 0;
    {
      nn::NoGradGuard guard;
      for (const auto& name : parameters.names())
        if (name.find("/W") != std::string::npos)
          penalty += nn::sum(nn::square(parameters.at(name))).item();
    }
    const double value = ce.item() + .001 * penalty;
    auto         g     = nn::grad(ce, parameters.parameters());
    optimizer->step(Regularized(parameters, g));
    check::Require(std::isfinite(value), "finite regularized voxel objective");
    return value;
  }
  Map  Arrays() const { return replay::Arrays(parameters); }
  void Export(const std::filesystem::path& file) const {
    nn::NamedTensors out;
    for (std::size_t i = 0; i < parameters.size(); ++i)
      out.emplace_back("p" + std::to_string(i), parameters.at(parameters.names()[i]));
    nn::save_npz(file.string(), out);
  }
  void LoadExport(const std::filesystem::path& file) {
    const auto a = replay::ToMap(nn::load_npz(file.string()));
    check::Require(a.size() == parameters.size(), "all8 exported parameters");
    for (std::size_t i = 0; i < parameters.size(); ++i)
      parameters.set(parameters.names()[i], a.at("p" + std::to_string(i)));
  }
};
inline Tensor Predict(Model& model, const Data& d, bool vote = true) {
  std::vector<Tensor> out;
  for (std::int64_t start = 0; start < d.voxels.size(0); start += 4) {
    std::vector<std::int64_t> rows;
    for (auto i = start; i < std::min(start + 4, d.voxels.size(0)); ++i)
      rows.push_back(i);
    out.push_back(model.Predict(Rows(d, rows), vote));
  }
  return nn::concat(out, 0);
}
inline Tensor Features(const Data& d) {
  const auto         n = d.voxels.size(0);
  std::vector<float> out(n * 64);
  for (int k = 0; k < 4; ++k) {
    const auto v = Rotate(d.voxels, k).to_vector<float>();
    for (int b = 0; b < n; ++b)
      for (int gx = 0; gx < 4; ++gx)
        for (int gy = 0; gy < 4; ++gy)
          for (int gz = 0; gz < 4; ++gz) {
            double sum = 0;
            for (int x = gx * 8; x < (gx + 1) * 8; ++x)
              for (int y = gy * 8; y < (gy + 1) * 8; ++y)
                for (int z = gz * 8; z < (gz + 1) * 8; ++z)
                  sum += v[b * 32768 + (x * 32 + y) * 32 + z];
            out[b * 64 + (gx * 4 + gy) * 4 + gz] += sum / (4 * 512);
          }
  }
  return nn::from_vector(out, {n, 64});
}
inline Tensor Nearest(const Data& fit, const Data& test) {
  const auto          a = Features(fit).to_vector<float>(), b = Features(test).to_vector<float>();
  const auto          labels = fit.labels.to_vector<int>();
  const auto          n = fit.voxels.size(0), m = test.voxels.size(0);
  std::vector<double> scale(64);
  for (int c = 0; c < 64; ++c) {
    double mean = 0, var = 0;
    for (int i = 0; i < n; ++i)
      mean += a[i * 64 + c] / n;
    for (int i = 0; i < n; ++i)
      var += std::pow(a[i * 64 + c] - mean, 2) / n;
    scale[c] = std::max(1e-6, std::sqrt(var));
  }
  std::vector<float> out(m * 3);
  for (int row = 0; row < m; ++row) {
    double best  = INFINITY;
    int    label = 0;
    for (int i = 0; i < n; ++i) {
      double distance = 0;
      for (int c = 0; c < 64; ++c)
        distance += std::pow((a[i * 64 + c] - b[row * 64 + c]) / scale[c], 2);
      if (distance < best) {
        best  = distance;
        label = labels[i];
      }
    }
    out[row * 3 + label] = 1;
  }
  return nn::from_vector(out, {m, 3});
}
inline void Metrics(const std::string& name, const Data& d, const Tensor& logits) {
  auto s = sensor::ScoreSensor(logits, d.labels, 3);
  std::cout << name << ":accuracy=" << 100 * s.accuracy << "%,macroF1=" << s.macro_f1
            << ",NLL=" << s.nll << ",confusion=";
  for (int c : s.confusion)
    std::cout << c << ',';
  std::cout << std::endl;
}

} // namespace vox
