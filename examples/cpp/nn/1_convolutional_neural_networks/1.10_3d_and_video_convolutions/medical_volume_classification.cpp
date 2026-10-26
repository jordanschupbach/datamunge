// CMD: just run-nn medical_volume_classification
// Real NoduleMNIST3D CT classification with a custom compact residual 3D CNN.
// Default 3 epochs; fit-only normalization, original-split balanced 64/16/64 subset.
//
// Port of tflow's medical_volume_classification. Checks: a residual 3D block
// with odd/even SAME stride-2 padding and a projection against a host loop,
// all 646 conv/projection/bias/input derivatives against host finite
// differences; endpoint shapes 28->14->7->4, GAP 16, two logits and the
// 13,782-parameter inventory. Then three epochs on real CT cubes with
// validation-NLL checkpoint selection, a fresh intensity-only restore and
// depth-reversal/depth-collapse controls.
//
// TF -> datamunge: no graph or session; TF's static shape inference becomes
// executed shapes. Weights come from datamunge's generator, so training
// matches TF only statistically. replay=DIR loads a TF run's model_arrays.npz
// (named arrays) into our model and compares the logits on its saved test cubes.
// Prepare (in tflow_examples): just medical-volume-assets.
// Options: epochs=3 data=PATH checks_only=0|1 replay=DIR threads=N.
// Artifacts: output/nn/models/medical_volume_classification/.

#include <algorithm>
#include <cmath>
#include <datamunge/nn/conv.hpp>
#include <datamunge/nn/layers.hpp>
#include <datamunge/nn/nn.hpp>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "common/checks.hpp"
#include "common/medical_volume.hpp"
#include "common/paths.hpp"

// Experiment workflow from examples/nn/common/medical_volume.hpp.
namespace medical_volume {
using namespace ::medical_volume;
inline void TrainMedical(const std::filesystem::path& root, int epochs) {
  check::Require(epochs > 0, "positive short CT epochs");
  const auto  pools = LoadMedical(root);
  const auto& fit = pools[0];
  const auto& validation = pools[1];
  const auto& test = pools[2];
  double      mean = 0, variance = 0;
  const auto  fv = fit.voxels.to_vector<float>();
  for (float v : fv) mean += v;
  mean /= static_cast<double>(fv.size());
  for (float v : fv) variance += std::pow(v - mean, 2);
  variance /= static_cast<double>(fv.size());
  MedicalModel model;
  // TF stores the moments in float variables.
  model.Normalize(static_cast<float>(mean), static_cast<float>(std::sqrt(variance)));
  check::Require(model.parameters.count() == 13782 && model.parameters.size() == 20 && model.state.size() == 2,
                 "independent three-stage parameter inventory");
  const auto   directory = paths::OutputDir("models/medical_volume_classification");
  const auto   prefix    = directory / "best";
  const double initial   = c3d::ScoreLogits(model.Predict(fit), fit.labels, 2).nll;
  const Tensor untrained = model.Predict(test);
  {
    const auto probe = MedicalRows(fit, {0, 1, 2, 3});
    const auto grads = nn::grad(nn::sparse_softmax_cross_entropy(model.Logits(probe.voxels, true), probe.labels),
                                model.parameters.parameters());
    for (const auto& g : grads) {
      double norm = 0;
      for (float v : g.to_vector<float>()) {
        check::Require(std::isfinite(v), "every CT derivative coordinate finite");
        norm += static_cast<double>(v) * v;
      }
      check::Require(norm > 0, "nonzero all6conv/3projection/head parameter tensors");
    }
  }
  std::cout << "medical_volume_classification:13782 parameters/20learned+2fit-normalization tensors;three3D residual "
               "blocks4/8/16,28->14->7->4,GAP/head;"
            << epochs
            << " epochs,batch4,Adam.001,headkeep.8;64fit/16validation/64test balanced CT subset\nfit-only mean=" << mean
            << ",stddev=" << std::sqrt(variance) << ",initialfitNLL=" << initial << ";all20 actual gradients finite/nonzero\n";
  nn::OptimizerConfig oc;
  oc.kind = nn::OptimizerKind::Adam;
  nn::Optimizer  optimizer(model.parameters, .001, oc);
  nn::BatchOrder order(64, 4, 1061);
  std::ofstream  history(directory / "history.tsv");
  history << "epoch\ttrain_dropout_ce\tvalidation_nll\n";
  double     best = INFINITY;
  int        selected = 0;
  const auto started = std::chrono::steady_clock::now();
  for (int epoch = 1; epoch <= epochs; ++epoch) {
    order.begin_epoch();
    double      sum  = 0;
    std::size_t seen = 0;
    for (;;) {
      const auto rows = order.next();
      if (rows.empty()) break;
      const auto   batch = MedicalRows(fit, rows);
      const Tensor loss  = nn::sparse_softmax_cross_entropy(model.Logits(batch.voxels, true), batch.labels);
      const double value = loss.item<double>();
      check::Require(std::isfinite(value) && value >= 0, "finite CT objective");
      optimizer.minimize(loss);
      sum += static_cast<double>(rows.size()) * value;
      seen += rows.size();
    }
    check::Require(seen == 64, "each fitting CT once per epoch");
    const double score = c3d::ScoreLogits(model.Predict(validation), validation.labels, 2).nll;
    if (score < best) {
      best     = score;
      selected = epoch;
      model.Save(prefix);
    }
    history << epoch << '\t' << sum / seen << '\t' << score << '\n';
    history.flush();
    std::cout << "epoch " << epoch << ":trainCE=" << sum / seen << ",validationNLL=" << score << std::endl;
  }
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  model.Restore(prefix);
  check::Near(c3d::ScoreLogits(model.Predict(validation), validation.labels, 2).nll, best, 1e-6,
              "selected CT validation score");
  // A fresh model restored from the checkpoint sees only intensities.
  MedicalModel deployed;
  deployed.Restore(prefix);
  const auto   before = deployed.Snapshot();
  const Tensor logits = deployed.Predict(test), reference = model.Predict(test), repeat = deployed.Predict(test);
  const auto   lv = logits.to_vector<float>(), rv = reference.to_vector<float>(), pv = repeat.to_vector<float>();
  for (std::size_t i = 0; i < lv.size(); ++i) {
    check::Near(lv[i], rv[i], 1e-6, "fresh22tensor restoration");
    check::Near(lv[i], pv[i], 0, "dropout-free deterministic CT inference");
  }
  const auto one = deployed.Predict(MedicalRows(test, {0})).to_vector<float>();
  for (std::size_t c = 0; c < 2; ++c) check::Near(one[c], lv[c], 1e-6, "single/batch CT inference");
  // Reverse depth: preserves every per-voxel intensity histogram and mean projection,
  // but changes true 3D neighborhoods. Also collapse depth to its mean projection.
  MedicalData reversed = test, projection = test;
  reversed.voxels      = nn::reverse(test.voxels, {1}).contiguous();
  projection.voxels    = nn::tile(nn::mean(test.voxels, {1}, true), {1, 28, 1, 1, 1}).contiguous();
  const Tensor depth_reversed = deployed.Predict(reversed), depth_collapsed = deployed.Predict(projection);
  double       delta = 0, collapse = 0;
  const auto   dr = depth_reversed.to_vector<float>(), dc = depth_collapsed.to_vector<float>();
  for (std::size_t i = 0; i < lv.size(); ++i) {
    delta += std::abs(static_cast<double>(lv[i]) - dr[i]);
    collapse += std::abs(static_cast<double>(lv[i]) - dc[i]);
  }
  check::Require(delta > 1e-6 && collapse > 1e-6, "actual trained logits depend on volumetric depth neighborhoods");
  check::Require(before == deployed.Snapshot(), "all22 learned/normalization tensors immutable across controls");
  const Tensor nearest  = MedicalBaseline(fit, test);
  const Tensor majority = nn::zeros({64, 2});
  MedicalMetrics("trained3D CNN", test, logits);
  MedicalMetrics("untrained3D CNN", test, untrained);
  MedicalMetrics("fit-only4cube intensity1NN", test, nearest);
  MedicalMetrics("class0 majority/tie", test, majority);
  MedicalMetrics("depth-reversed control", test, depth_reversed);
  MedicalMetrics("depth-collapsed control", test, depth_collapsed);
  {
    nn::NamedTensors arrays;
    const auto       ps = deployed.parameters.parameters(), ss = deployed.state.parameters();
    for (std::size_t i = 0; i < ps.size(); ++i) arrays.emplace_back(deployed.parameters.names()[i], ps[i]);
    for (std::size_t i = 0; i < ss.size(); ++i) arrays.emplace_back(deployed.state.names()[i], ss[i]);
    nn::save_npz((directory / "model_arrays.npz").string(), arrays);
  }
  nn::save_npz((directory / "test_outputs.npz").string(),
               {{"voxels", test.voxels},
                {"labels", test.labels},
                {"source_ids", test.source},
                {"logits", logits},
                {"untrained", untrained},
                {"nearest", nearest},
                {"reversed", depth_reversed},
                {"collapsed", depth_collapsed},
                {"fit_voxels", fit.voxels},
                {"fit_labels", fit.labels},
                {"fit_source_ids", fit.source}});
  for (const char* name : {"manifest.json", "ATTRIBUTION.txt", "test_contact.png"})
    if (std::filesystem::exists(root / name))
      std::filesystem::copy_file(root / name,
                                 directory / (std::string(name) == "manifest.json" ? "data_manifest.json" : name),
                                 std::filesystem::copy_options::overwrite_existing);
  check::Require(bool(history), "write CT history");
  std::cout << "selected=" << selected << ",fitNLL " << initial << " -> "
            << c3d::ScoreLogits(model.Predict(fit), fit.labels, 2).nll
            << ",depth reversal/collapse total logit changes=" << delta << '/' << collapse
            << ";intensity-only/freshrestore/single-batch/determinism/immutable22tensor checks passed\n";
  std::printf("training time %.1f s (%.2f s/epoch)\n", seconds, seconds / epochs);
  std::cout << "Artifacts:" << directory.string()
            << "\nSmall balanced original-split CT subset;no local patient IDs or clinical diagnosis validation,no full "
               "benchmark claim.\n";
}
}  // namespace medical_volume



namespace nn = datamunge::nn;
namespace mv = medical_volume;
using nn::Tensor;

namespace {

struct HostVolume {
  int                 d, h, w, c;
  std::vector<double> v;
};

HostVolume Conv(const HostVolume& x, const std::vector<double>& weight, const std::vector<double>& bias, int kernel,
                int stride) {
  HostVolume out{(x.d + stride - 1) / stride, (x.h + stride - 1) / stride, (x.w + stride - 1) / stride,
                 static_cast<int>(bias.size()), {}};
  out.v.resize(static_cast<std::size_t>(out.d * out.h * out.w * out.c));
  const int pd = std::max(0, (out.d - 1) * stride + kernel - x.d) / 2, ph = std::max(0, (out.h - 1) * stride + kernel - x.h) / 2,
            pw = std::max(0, (out.w - 1) * stride + kernel - x.w) / 2;
  for (int z = 0; z < out.d; ++z)
    for (int y = 0; y < out.h; ++y)
      for (int a = 0; a < out.w; ++a)
        for (int o = 0; o < out.c; ++o) {
          double value = bias[static_cast<std::size_t>(o)];
          for (int kz = 0; kz < kernel; ++kz)
            for (int ky = 0; ky < kernel; ++ky)
              for (int kx = 0; kx < kernel; ++kx) {
                const int iz = z * stride + kz - pd, iy = y * stride + ky - ph, ix = a * stride + kx - pw;
                if (iz < 0 || iy < 0 || ix < 0 || iz >= x.d || iy >= x.h || ix >= x.w) continue;
                for (int c = 0; c < x.c; ++c)
                  value += x.v[static_cast<std::size_t>(((iz * x.h + iy) * x.w + ix) * x.c + c)] *
                           weight[static_cast<std::size_t>((((kz * kernel + ky) * kernel + kx) * x.c + c) * out.c + o)];
              }
          out.v[static_cast<std::size_t>(((z * out.h + y) * out.w + a) * out.c + o)] = value;
        }
  return out;
}

void NumericalChecks() {
  nn::ParameterStore p(1063);
  Tensor             input = nn::zeros({1, 5, 6, 7, 2});
  {
    nn::NoGradGuard guard;
    mv::MedicalResidual(p, "block", input, 2);
  }
  std::vector<std::vector<double>> values;
  for (std::size_t index = 0; index < p.size(); ++index) {
    const auto&        shape = p.parameters()[index].shape();
    const std::int64_t n     = p.parameters()[index].numel();
    std::vector<float> t(static_cast<std::size_t>(n));
    for (std::int64_t i = 0; i < n; ++i)
      t[static_cast<std::size_t>(i)] = index % 2 ? static_cast<float>(.04 * std::cos(.7 * i + index))
                                                 : static_cast<float>(.07 * std::sin(.19 * i + .3 * index));
    p.set(p.names()[index], nn::from_vector(t, shape));
    values.emplace_back(t.begin(), t.end());
  }
  std::vector<float> x(420), coefficient(72);
  for (int i = 0; i < 420; ++i) x[static_cast<std::size_t>(i)] = static_cast<float>(.4f * std::sin(.37 * i));
  for (int i = 0; i < 72; ++i) coefficient[static_cast<std::size_t>(i)] = static_cast<float>(.1f * std::cos(.43 * i));
  values.emplace_back(x.begin(), x.end());
  input = nn::from_vector(x, {1, 5, 6, 7, 2});
  input.requires_grad_(true);
  const Tensor y         = mv::MedicalResidual(p, "block", input, 2);
  const Tensor objective = nn::sum(y * nn::from_vector(coefficient, {1, 3, 3, 4, 2}));
  auto         vars      = p.parameters();
  vars.push_back(input);
  const auto gradients = nn::grad(objective, vars);
  const auto host = [&](const std::vector<std::vector<double>>& v, std::vector<double>* result = nullptr) {
    const HostVolume in{5, 6, 7, 2, v[6]};
    auto             first = Conv(in, v[0], v[1], 3, 2);
    for (auto& q : first.v) q = std::max(0., q);
    const auto second = Conv(first, v[2], v[3], 3, 1), skip = Conv(in, v[4], v[5], 1, 2);
    double     total  = 0;
    for (std::size_t i = 0; i < second.v.size(); ++i) {
      const double q = std::max(0., second.v[i] + skip.v[i]);
      total += q * coefficient[i];
      if (result) result->push_back(q);
    }
    return total;
  };
  std::vector<double> expected;
  host(values, &expected);
  const auto yv = y.to_vector<float>();
  double     error = 0, derivative = 0;
  int        coordinates = 0;
  for (std::size_t i = 0; i < expected.size(); ++i) error = std::max(error, std::abs(expected[i] - yv[i]));
  for (std::size_t v = 0; v < values.size(); ++v) {
    const auto g = gradients[v].to_vector<float>();
    for (std::size_t i = 0; i < values[v].size(); ++i) {
      auto plus = values, minus = values;
      plus[v][i] += 1e-5;
      minus[v][i] -= 1e-5;
      derivative = std::max(derivative, std::abs((host(plus) - host(minus)) / 2e-5 - g[i]));
      ++coordinates;
    }
  }
  check::Require(coordinates == 646 && expected.size() == 72, "all2163D+4projection weights/6biases/420inputs and72outputs");
  check::Near(error, 0, 2e-7, "all72 odd/even SAME3D residual outputs");
  check::Near(derivative, 0, 3e-7, "ALL646 independent host finite differences");
  std::cout << "medical residual3D fixture:all72outputs error=" << error
            << ",ALL646 conv/projection/bias/input hostFD error=" << derivative << ";odd/even SAME stride2 padding passed\n";
}

void GraphChecks() {
  nn::ParameterStore p(1069), state(1071);
  nn::Generator      gen(5);
  const auto net = mv::MedicalVolume(p, state, nn::uniform({2, 28, 28, 28, 1}, 0.0, 1.0, gen));
  check::Require(net.stages[0].shape() == nn::Shape({2, 14, 14, 14, 4}) &&
                     net.stages[1].shape() == nn::Shape({2, 7, 7, 7, 8}) &&
                     net.stages[2].shape() == nn::Shape({2, 4, 4, 4, 16}) && net.global.shape() == nn::Shape({2, 16}) &&
                     net.logits.shape() == nn::Shape({2, 2}),
                 "independent medical endpoint shape");
  std::int64_t count = 0;
  int          in    = 1;
  for (int out : {4, 8, 16}) {
    count += 27 * in * out + out + 27 * out * out + out + in * out + out;
    in = out;
  }
  count += 16 * 2 + 2;
  check::Require(count == 13782 && p.count() == count && p.size() == 20 && state.count() == 2,
                 "independent sixconv/threeprojection/head inventory");
  std::cout << "custom compact medical3D CNN:28->14->7->4,GAP16/twologits,13782 parameters/20learned+2normalization "
               "tensors;endpoint/count checks passed\n";
}

/// Loads TF's trained named arrays into our model and replays its saved test cubes.
void Replay(const std::filesystem::path& dir) {
  std::map<std::string, Tensor> arrays, outputs;
  for (auto& [name, t] : nn::load_npz((dir / "model_arrays.npz").string())) arrays[name] = t;
  for (auto& [name, t] : nn::load_npz((dir / "test_outputs.npz").string())) outputs[name] = t;
  mv::MedicalModel model;
  check::Require(arrays.size() == model.parameters.size() + model.state.size(), "TF/ours CT tensor count");
  for (nn::ParameterStore* store : {&model.parameters, &model.state})
    for (const auto& name : store->names()) {
      const Tensor& value = arrays.at(name);
      check::Require(value.shape() == store->at(name).shape(), "TF/ours CT tensor shape " + name);
      store->set(name, value);
    }
  double worst = 0;
  for (const char* key : {"logits", "reversed", "collapsed"}) {
    mv::MedicalData d;
    const Tensor&   voxels = outputs.at("voxels");
    d.voxels = std::string(key) == "logits"     ? voxels
               : std::string(key) == "reversed" ? nn::reverse(voxels, {1}).contiguous()
                                                : nn::tile(nn::mean(voxels, {1}, true), {1, 28, 1, 1, 1}).contiguous();
    const auto ours = model.Predict(d).to_vector<float>(), theirs = outputs.at(key).to_vector<float>();
    double     error = 0;
    for (std::size_t i = 0; i < ours.size(); ++i) error = std::max(error, std::abs(double(ours[i]) - theirs[i]));
    std::cout << "  replay " << key << " max|ours-TF|=" << error << "\n";
    worst = std::max(worst, error);
  }
  check::Near(worst, 0, 1e-5, "TF-trained CT arrays reproduce TF test/reversed/collapsed logits");
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int                   epochs = 3;
    bool                  checks = false;
    std::filesystem::path root   = paths::Dataset("medical_volume"), replay;
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a.rfind("epochs=", 0) == 0)
        epochs = std::stoi(a.substr(7));
      else if (a.rfind("data=", 0) == 0)
        root = a.substr(5);
      else if (a == "checks_only=1" || a == "checks_only=0")
        checks = a == "checks_only=1";
      else if (a.rfind("replay=", 0) == 0)
        replay = a.substr(7);
      else if (a.rfind("threads=", 0) == 0)
        nn::set_num_threads(std::stoi(a.substr(8)));
      else
        throw std::invalid_argument("expected epochs=N data=PATH checks_only=0|1 replay=DIR threads=N");
    }
    NumericalChecks();
    GraphChecks();
    if (!replay.empty()) {
      std::cout << "medical CNN replay of TF arrays " << replay.string() << ":\n";
      Replay(replay);
    }
    if (!checks) mv::TrainMedical(root, epochs);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
