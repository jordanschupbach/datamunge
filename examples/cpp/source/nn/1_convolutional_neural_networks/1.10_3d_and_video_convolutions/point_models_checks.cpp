// CMD: just run-nn point_models_checks
// Supporting architecture checks for the combined PointNet/PointNet++ work.
// Both complete network graphs; untrained real-surface forward checks only.
// Training and independent entire-model audits are separate targets.
//
// Port of tflow's point_models_checks. Checks: the feature-transform
// orthogonality penalty and its 9 derivatives against host finite
// differences; full (40-class, 1024-point) and compact (width/4, 3-class,
// 256-point) PointNet and PointNet++ SSG endpoint shapes, per-layer parameter
// inventories and all learned gradients; the full canonical networks forward
// and backward on a real surface sample (all derivatives finite, saved as a
// fixture); untrained compact forwards on real OFF surface samples (exact
// identity T-Nets, PointNet permutation invariance).
//
// TF -> datamunge: no graph or session. TF's static shape inference becomes
// executed shapes: the graph checks run each network on a random cloud in
// [-1, 1]^3 (batch 1). Weights come from datamunge's generator.
// Data: output/datasets/point_clouds/point_clouds.npz (tflow's cache; prepare
// with `just point-cloud-assets` in tflow_examples), or data=PATH.
// Options: data=PATH threads=N.
// Artifacts: output/nn/models/{pointnet,pointnet_plus_plus}/full_runtime_fixture.npz.

#include <cmath>
#include <datamunge/nn/nn.hpp>
#include <filesystem>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "common/checks.hpp"
#include "common/paths.hpp"
#include "common/point_models.hpp"

namespace nn = datamunge::nn;
namespace pm = point_models;
using nn::Tensor;

namespace {

std::int64_t Inventory(bool plus, int divisor, int classes) {
  const auto   width = [&](int c) { return std::max(1, c / divisor); };
  const auto   unit  = [](int a, int b) { return std::int64_t{a} * b + 3 * b; };
  std::int64_t count = 0;
  if (!plus) {
    const auto transform = [&](int in, int dimension) {
      return unit(in, width(64)) + unit(width(64), width(128)) + unit(width(128), width(1024)) +
             unit(width(1024), width(512)) + unit(width(512), width(256)) +
             std::int64_t{width(256)} * dimension * dimension + dimension * dimension;
    };
    count += transform(3, 3) + transform(width(64), width(64));
    count += unit(3, width(64)) + unit(width(64), width(64)) + unit(width(64), width(64)) + unit(width(64), width(128)) +
             unit(width(128), width(1024));
  } else {
    count += unit(3, width(64)) + unit(width(64), width(64)) + unit(width(64), width(128));
    count += unit(3 + width(128), width(128)) + unit(width(128), width(128)) + unit(width(128), width(256));
    count += unit(3 + width(256), width(256)) + unit(width(256), width(512)) + unit(width(512), width(1024));
  }
  count += unit(width(1024), width(512)) + unit(width(512), width(256)) + std::int64_t{width(256)} * classes + classes;
  return count;
}

void PenaltyChecks() {
  const std::vector<float> input{1.1f, .2f, -.1f, .05f, .9f, .15f, 0.f, -.1f, 1.05f};
  Tensor                   a = nn::from_vector(input, {1, 3, 3});
  a.requires_grad_(true);
  const Tensor loss = pm::PointTransformPenalty(a, 3);
  const auto   g    = nn::grad(loss, {a})[0].to_vector<float>();
  std::vector<double> values(input.begin(), input.end());
  const auto host = [](const std::vector<double>& v) {
    double sum = 0;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) {
        double product = 0;
        for (int k = 0; k < 3; ++k) product += v[static_cast<std::size_t>(3 * i + k)] * v[static_cast<std::size_t>(3 * j + k)];
        const double d = product - (i == j ? 1 : 0);
        sum += .0005 * d * d;
      }
    return sum;
  };
  check::Near(loss.item<double>(), host(values), 1e-9, "reference.001*L2Loss(AA^T-I)");
  double error = 0;
  for (std::size_t i = 0; i < 9; ++i) {
    auto plus = values, minus = values;
    plus[i] += 1e-5;
    minus[i] -= 1e-5;
    error = std::max(error, std::abs((host(plus) - host(minus)) / 2e-5 - g[i]));
  }
  check::Near(error, 0, 1e-8, "ALL9 transform-entry host finite differences");
  const Tensor rotation = nn::from_vector(std::vector<float>{0, -1, 0, 1, 0, 0, 0, 0, 1}, {1, 3, 3});
  check::Near(pm::PointTransformPenalty(rotation, 3).item<double>(), 0, 0, "orthogonal feature matrix accepted");
  std::cout << "PointNet transform penalty:all9hostFD error=" << error << ",orthogonal matrix zero passed\n";
}

void GraphChecks() {
  for (bool plus : {false, true})
    for (bool compact : {false, true}) {
      const pm::PointConfig cfg = compact ? pm::PointConfig{4, 3, 256, 64, 16, 16, 32} : pm::PointConfig{};
      nn::ParameterStore    p(953), state(955);
      nn::Generator         gen(7);
      const Tensor          xyz = nn::uniform({1, cfg.points, 3}, -1.0, 1.0, gen);
      const auto net = plus ? pm::PointNetPlusPlus(p, state, xyz, cfg) : pm::PointNet(p, state, xyz, cfg);
      const auto shape = [&](const Tensor& value, const nn::Shape& expected) {
        check::Require(value.shape() == expected, "reference full/compact point model endpoint dimension");
      };
      shape(net.logits, {1, cfg.classes});
      shape(net.global, {1, 1024 / cfg.divisor});
      if (plus) {
        shape(net.centroids1, {1, cfg.first_centers, 3});
        shape(net.centroids2, {1, cfg.second_centers, 3});
        shape(net.neighbors1, {1, cfg.first_centers, cfg.first_neighbors});
        shape(net.neighbors2, {1, cfg.second_centers, cfg.second_neighbors});
      } else {
        shape(net.input_transform, {1, 3, 3});
        shape(net.feature_transform, {1, 64 / cfg.divisor, 64 / cfg.divisor});
      }
      check::Require(p.count() == Inventory(plus, cfg.divisor, cfg.classes) &&
                         p.size() == static_cast<std::size_t>(plus ? 46 : 74) &&
                         state.size() == static_cast<std::size_t>(plus ? 22 : 34),
                     "independent per-layer W/b/BN/transform/classifier inventory");
      const auto gradients = nn::grad(nn::mean(net.logits * net.logits), p.parameters());
      check::Require(gradients.size() == p.size(), "all learned point model derivatives");
      for (const auto& g : gradients) check::Require(g.defined(), "no disconnected point model parameter");
      std::cout << (plus ? "PointNet++SSG" : "PointNet two T-Nets") << (compact ? " compact" : " full reference") << ":"
                << p.count() << " parameters/" << p.size() << "learned+" << state.size()
                << "moving tensors;shape/count/all gradients passed\n";
    }
}

std::map<std::string, Tensor> LoadClouds(const std::filesystem::path& root) {
  const auto file = root / "point_clouds.npz";
  check::Require(std::filesystem::exists(file), "missing " + file.string() +
                                                    "; run `just point-cloud-assets` in tflow_examples, or pass data=PATH");
  std::map<std::string, Tensor> arrays;
  for (auto& [name, t] : nn::load_npz(file.string())) arrays[name] = t;
  return arrays;
}

void FullRuntime(const std::map<std::string, Tensor>& arrays) {
  const Tensor points = nn::slice(arrays.at("fit_xyz"), {0, 0, 0}, {1, 1024, 3}).contiguous();
  for (bool plus : {false, true}) {
    nn::ParameterStore p(989), state(991);
    const auto   net       = plus ? pm::PointNetPlusPlus(p, state, points) : pm::PointNet(p, state, points);
    const Tensor objective = nn::mean(net.logits * net.logits);
    const auto   gradient  = nn::grad(objective, p.parameters());
    nn::NamedTensors fixture{{"xyz", points}, {"logits", net.logits.detach()}, {"global", net.global.detach()}};
    const auto       ps = p.parameters(), ss = state.parameters();
    for (std::size_t i = 0; i < ps.size(); ++i) fixture.emplace_back(p.names()[i], ps[i]);
    for (std::size_t i = 0; i < ss.size(); ++i) fixture.emplace_back(state.names()[i], ss[i]);
    for (const auto& g : gradient)
      for (float v : g.to_vector<float>()) check::Require(std::isfinite(v), "all full reference derivative coordinates finite");
    const auto dir = paths::OutputDir(std::string("models/") + (plus ? "pointnet_plus_plus" : "pointnet"));
    nn::save_npz((dir / "full_runtime_fixture.npz").string(), fixture);
    std::cout << (plus ? "PointNet++" : "PointNet") << " actual full canonical40/1024point forward/all" << gradient.size()
              << " finite derivative tensors;identity T-Net zero-head upstream zeros expected at initialization\n";
  }
}

void RealForward(const std::map<std::string, Tensor>& arrays) {
  const Tensor points   = nn::slice(arrays.at("fit_xyz"), {0, 0, 0}, {2, 256, 3}).contiguous();
  const Tensor permuted = nn::reverse(points, {1}).contiguous();
  for (bool plus : {false, true}) {
    nn::ParameterStore    p(967), state(969);
    const pm::PointConfig cfg{4, 3, 256, 64, 16, 16, 32};
    nn::NoGradGuard       guard;
    const auto net = plus ? pm::PointNetPlusPlus(p, state, points, cfg) : pm::PointNet(p, state, points, cfg);
    const auto original = net.logits.to_vector<float>();
    for (float v : original) check::Require(std::isfinite(v), "finite untrained real-geometry logits");
    if (!plus)
      for (const Tensor* t : {&net.input_transform, &net.feature_transform}) {
        const std::int64_t dim = t->size(1);
        const auto         v   = t->to_vector<float>();
        for (std::int64_t b = 0; b < 2; ++b)
          for (std::int64_t i = 0; i < dim; ++i)
            for (std::int64_t j = 0; j < dim; ++j)
              check::Near(v[static_cast<std::size_t>((b * dim + i) * dim + j)], i == j ? 1 : 0, 0,
                          "both actual T-Nets initialize EXACTidentity");
      }
    const auto reversed = (plus ? pm::PointNetPlusPlus(p, state, permuted, cfg) : pm::PointNet(p, state, permuted, cfg))
                              .logits.to_vector<float>();
    double change = 0;
    for (std::size_t i = 0; i < 6; ++i) change = std::max(change, std::abs(double(original[i]) - reversed[i]));
    if (!plus) check::Near(change, 0, 1e-6, "PointNet learned-transform/globalmax permutation invariance at initialization");
    std::cout << (plus ? "PointNet++SSG" : "PointNet") << " untrained real surface forward finite;reversed-point maxlogit change="
              << change
              << (plus ? " (reference first-seed FPS/first-K ball query is order-dependent)"
                       : ";both identity T-Nets/permutation invariance passed")
              << '\n';
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::filesystem::path root = paths::Dataset("point_clouds");
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a.rfind("data=", 0) == 0)
        root = a.substr(5);
      else if (a.rfind("threads=", 0) == 0)
        nn::set_num_threads(std::stoi(a.substr(8)));
      else
        throw std::invalid_argument("expected data=PATH threads=N");
    }
    PenaltyChecks();
    GraphChecks();
    const auto arrays = LoadClouds(root);
    FullRuntime(arrays);
    RealForward(arrays);
    std::cout << "Supporting architecture/geometry checks only. Classifier training and independent entire-model audits "
                 "run separately via pointnet and pointnet_plus_plus.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
