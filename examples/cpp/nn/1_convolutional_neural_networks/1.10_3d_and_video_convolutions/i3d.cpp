// CMD: just run-nn i3d
// RGB Inception-v1 I3D: 57 normalized 3D convolutions, nine four-branch Mixed
// modules, temporal-spatial average pool/dropout/400-class reference head.
//
// Port of tflow's i3d. Checks: kernel inflation (every slice = W2D / T), an
// inflated SAME convolution against a host loop, the repeated-image response
// with its 2/3 temporal zero-padding borders, all 196 source-kernel/input
// derivatives against host finite differences; the Sonnet-style unit (offset
// only, population moments) forward, all 14 derivatives, moving-moment update
// and stored-statistic inference; full 64x224 and compact endpoint tables,
// parameter inventory from the published branch table and all 116 gradients.
// Then the compact 16x32/divisor-8 three-class model trains on real KTH clips.
//
// TF -> datamunge: no graph or session. TF checks the full graph by static
// shape inference without allocating it; here the full-width network is
// built and executed (forward + all gradients) on a 16x32x32 clip, which
// checks the real parameter inventory and every endpoint at that size, while
// the 64x224x224 endpoint table is derived from the SAME padding rules with
// nn::conv_output_size; full=1 also executes the full network at 64x224x224
// (about 2 GB, a few seconds). Weights come from datamunge's generator, so training matches
// TF only statistically (the 6-clip test set is noisy in both).
// replay=DIR loads a TF run's trained arrays (DIR/model_arrays.npz) into our
// compact model and compares the logits on TF's saved test clips.
// Prepare (in tflow_examples): just action-video-assets.
// Options: epochs=3 data=PATH checks_only=0|1 full=0|1 replay=DIR threads=N.
// Artifacts: output/nn/models/i3d/.

#include <algorithm>
#include <cmath>
#include <datamunge/nn/conv.hpp>
#include <datamunge/nn/layers.hpp>
#include <datamunge/nn/nn.hpp>
#include <iostream>
#include <string>
#include <vector>

#include "common/c3d.hpp"
#include "common/checks.hpp"
#include "common/i3d.hpp"
#include "common/paths.hpp"

// Experiment workflow from examples/nn/common/c3d.hpp.
namespace c3d {
using namespace ::c3d;
template <class Model>
void TrainVideoModel(const std::filesystem::path& root, int epochs, const VideoTrainSpec& spec,
                     const std::function<void(Model&)>& on_init = {}) {
  check::Require(epochs > 0, "positive short video epochs");
  const auto  pools = LoadVideo(root);
  const auto& fit = pools[0];
  const auto& validation = pools[1];
  const auto& test = pools[2];
  Model       model;
  if (on_init) on_init(model);
  const auto   directory  = paths::OutputDir("models/" + spec.directory);
  const auto   checkpoint = (directory / "best.npz").string(), stats_checkpoint = (directory / "best_state.npz").string();
  const auto   probe      = VideoRows(fit, {0, 1});
  const double initial    = ScoreLogits(VideoModelPredict(model, fit), fit.labels, 3).nll;
  const Tensor untrained  = VideoModelPredict(model, test);
  {
    const auto params = model.parameters.parameters();
    const auto grads =
        nn::grad(nn::sparse_softmax_cross_entropy(model.Logits(probe.rgb, true, false), probe.labels), params);
    check::Require(grads.size() == params.size(), "all compact video parameter derivatives");
    for (const Tensor& g : grads) {
      double norm = 0;
      for (float x : g.to_vector<float>()) {
        check::Require(std::isfinite(x), "finite every video kernel/offset gradient coordinate");
        norm += static_cast<double>(x) * x;
      }
      check::Require(norm > 0, "nonzero every video kernel/offset parameter tensor derivative");
    }
  }
  std::cout << spec.title << ":" << model.parameters.count() << " parameters/" << model.parameters.size()
            << " learned tensors+" << model.state.size() << " moving-stat tensors,16x32x32RGB;" << epochs
            << " epochs,batch2,Adam.001,all learned gradients finite/nonzero,initial fitCE=" << initial << std::endl;
  nn::OptimizerConfig oc;
  oc.kind = nn::OptimizerKind::Adam;
  nn::Optimizer  optimizer(model.parameters, .001, oc);
  nn::BatchOrder batch_order(static_cast<std::size_t>(fit.size()), 2, 647);
  double         best = INFINITY;
  int            selected = 0;
  std::ofstream  history(directory / "history.tsv");
  history << "epoch\ttrain_dropout_ce\tvalidation_ce\n";
  const auto started = std::chrono::steady_clock::now();
  for (int epoch = 1; epoch <= epochs; ++epoch) {
    batch_order.begin_epoch();
    double      sum  = 0;
    std::size_t seen = 0;
    for (;;) {
      const auto rows = batch_order.next();
      if (rows.empty()) break;
      const auto   batch = VideoRows(fit, rows);
      const Tensor loss  = nn::sparse_softmax_cross_entropy(model.Logits(batch.rgb, true, true), batch.labels);
      const double value = loss.item<double>();
      check::Require(std::isfinite(value) && value >= 0, "finite video CE");
      optimizer.minimize(loss);
      sum += static_cast<double>(rows.size()) * value;
      seen += rows.size();
    }
    check::Require(seen == static_cast<std::size_t>(fit.size()), "all fitting real clips once per epoch");
    const double value = ScoreLogits(VideoModelPredict(model, validation), validation.labels, 3).nll;
    if (value < best) {
      best     = value;
      selected = epoch;
      model.parameters.save(checkpoint);
      model.state.save(stats_checkpoint);
    }
    history << epoch << '\t' << sum / seen << '\t' << value << '\n';
    history.flush();
    std::cout << " epoch " << epoch << ":trainCE=" << sum / seen << ",validationCE=" << value << std::endl;
  }
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  model.parameters.load(checkpoint);
  model.state.load(stats_checkpoint);
  check::Near(ScoreLogits(VideoModelPredict(model, validation), validation.labels, 3).nll, best, 1e-7,
              "restored video validation CE");
  const double learned = ScoreLogits(VideoModelPredict(model, fit), fit.labels, 3).nll;
  if (spec.require_improvement)
    check::Require(learned < initial, "short-run fixed fitting inference CE improves");
  else
    check::Require(std::isfinite(learned), "finite restored fitting inference CE;quality is reported without threshold");

  // A fresh model restored from the checkpoint sees only RGB clips.
  Model deployed;
  deployed.parameters.load(checkpoint);
  deployed.state.load(stats_checkpoint);
  const auto   before = VideoModelSnapshot(deployed);
  const Tensor logits = VideoModelPredict(deployed, test), reference = VideoModelPredict(model, test),
               repeat = VideoModelPredict(deployed, test);
  const auto lv = logits.to_vector<float>(), rv = reference.to_vector<float>(), pv = repeat.to_vector<float>();
  for (std::size_t i = 0; i < lv.size(); ++i) {
    check::Near(lv[i], rv[i], 1e-6, "independent video checkpoint parity");
    check::Near(lv[i], pv[i], 0, "dropout-free deterministic video inference");
  }
  const auto single = VideoModelPredict(deployed, VideoRows(test, {0})).template to_vector<float>();
  for (std::size_t c = 0; c < 3; ++c) check::Near(single[c], lv[c], 1e-6, "single/batch clip inference");

  const std::vector<std::int64_t> all{0, 1, 2, 3, 4, 5};
  VideoData reverse = VideoRows(test, all), still = VideoRows(test, all);
  reverse.rgb = nn::reverse(test.rgb, {1}).contiguous();
  still.rgb   = nn::tile(nn::slice_axis(test.rgb, 1, 0, 1), {1, 16, 1, 1, 1}).contiguous();
  const Tensor reversed = VideoModelPredict(deployed, reverse), repeated_frame = VideoModelPredict(deployed, still);
  double       order = 0, motion = 0;
  const auto   rev = reversed.to_vector<float>(), rep = repeated_frame.to_vector<float>();
  for (std::size_t i = 0; i < lv.size(); ++i) {
    order += std::abs(static_cast<double>(lv[i]) - rev[i]);
    motion += std::abs(static_cast<double>(lv[i]) - rep[i]);
  }
  check::Require(order > 1e-7 && motion > 1e-7, "video model depends on temporal order and genuine multi-frame input");
  check::Require(before == VideoModelSnapshot(deployed), "all learned and moving tensors immutable");
  {
    nn::NamedTensors arrays;
    const auto       ps = deployed.parameters.parameters(), ss = deployed.state.parameters();
    std::ofstream    mapping;
    if (spec.tsv) {
      mapping.open(directory / "model_arrays.tsv");
      mapping << "key\tname\tkind\n";
    }
    for (std::size_t i = 0; i < ps.size() + ss.size(); ++i) {
      const bool learned_tensor = i < ps.size();
      const std::string key = "p" + std::to_string(i);
      arrays.emplace_back(key, learned_tensor ? ps[i] : ss[i - ps.size()]);
      if (spec.tsv)
        mapping << key << '\t'
                << (learned_tensor ? deployed.parameters.names()[i] : deployed.state.names()[i - ps.size()]) << '\t'
                << (learned_tensor ? "learned" : "moving") << '\n';
    }
    nn::save_npz((directory / "model_arrays.npz").string(), arrays);
    if (spec.tsv) check::Require(bool(mapping), "export named learned/moving inventory");
  }
  const Tensor appearance = VideoNearest(fit, test, false), energy = VideoNearest(fit, test, true);
  const Tensor prior      = nn::zeros({6, 3});
  PrintVideo("trained " + spec.title, test, logits);
  PrintVideo("untrained", test, untrained);
  PrintVideo("fit static mean-frame1NN", test, appearance);
  PrintVideo("fit absolute-motion1NN", test, energy);
  PrintVideo("balanced prior tie class0", test, prior);
  PrintVideo("reversed-frame control", test, reversed);
  PrintVideo("repeated-first-frame control", test, repeated_frame);

  std::ofstream predictions(directory / "predictions.tsv");
  predictions << "video_id\tsegment\tperson\tlabel\tprediction\n";
  const auto vid = test.video.to_vector<int>(), seg = test.segment.to_vector<int>(), per = test.person.to_vector<int>(),
             lab = test.labels.to_vector<int>();
  for (std::size_t n = 0; n < 6; ++n) {
    std::size_t best_c = 0;
    for (std::size_t c = 1; c < 3; ++c)
      if (lv[n * 3 + c] > lv[n * 3 + best_c]) best_c = c;
    predictions << vid[n] << '\t' << seg[n] << '\t' << per[n] << '\t' << lab[n] << '\t' << best_c << '\n';
  }
  nn::save_npz((directory / "test_outputs.npz").string(),
               {{"rgb", test.rgb},           {"labels", test.labels},     {"person", test.person},
                {"video_id", test.video},    {"segment", test.segment},   {"logits", logits},
                {"untrained", untrained},    {"appearance", appearance},  {"motion", energy},
                {"reverse", reversed},       {"repeat_first", repeated_frame}, {"fit_rgb", fit.rgb},
                {"fit_labels", fit.labels},  {"fit_person", fit.person},  {"fit_video_id", fit.video}});
  for (const char* name : {"manifest.json", "ATTRIBUTION.txt", "test_contact.png"})
    if (std::filesystem::exists(root / name))
      std::filesystem::copy_file(root / name, directory / (std::string(name) == "manifest.json" ? "data_manifest.json" : name),
                                 std::filesystem::copy_options::overwrite_existing);
  check::Require(bool(history) && bool(predictions), "save real-video predictions/histories");
  std::cout << " selected=" << selected << ",fixed fitCE " << initial << " -> " << learned
            << ",reverse/repeat-first total logit changes=" << order << "/" << motion
            << ";standalone RGB-only restore/single-batch/determinism/immutable model passed\n";
  std::printf(" training time %.1f s (%.2f s/epoch)\n", seconds, seconds / epochs);
  std::cout << "Artifacts:" << directory.string() << "\n" << spec.note << "\n";
}
}  // namespace c3d

// Experiment workflow from examples/nn/common/i3d.hpp.
namespace i3d {
using namespace ::i3d;
inline void TrainI3d(const std::filesystem::path& root, int epochs) {
  c3d::VideoTrainSpec spec;
  spec.directory = "i3d";
  spec.title     = "I3D compact";
  spec.note =
      "Tiny3-action/6-person KTH subset,one test person,six clips,not full KTH/Kinetics benchmark accuracy. Random "
      "initialization;no ImageNet/Kinetics pretraining. Fullwidth execution excluded from default CPU demo.";
  c3d::TrainVideoModel<I3dModel>(root, epochs, spec, [](I3dModel& model) {
    // The actual initial filter bank: every temporal slice equals the inflated 2D source.
    int units = 0;
    for (std::size_t i = 0; i < model.parameters.size(); ++i) {
      const std::string& name = model.parameters.names()[i];
      if (name.size() < 2 || name.compare(name.size() - 2, 2, "/W") != 0 || name == "Logits/W") continue;
      const Tensor w = model.parameters.parameters()[i];
      check::Require(w.dim() == 5, "all57 initial normalized units have THWIO kernels");
      const auto         v     = w.to_vector<float>();
      const std::int64_t slice = w.numel() / w.size(0);
      for (std::int64_t t = 1; t < w.size(0); ++t)
        for (std::int64_t j = 0; j < slice; ++j)
          check::Near(v[static_cast<std::size_t>(t * slice + j)], v[static_cast<std::size_t>(j)], 0,
                      "actual initial temporal filter slices equal inflated2D source");
      ++units;
    }
    check::Require(units == 57, "all57 normalized unit kernels use actual spatial inflation");
  });
}
}  // namespace i3d



namespace nn = datamunge::nn;
using nn::Padding;
using nn::Tensor;

namespace {

void InflationChecks() {
  std::vector<float> spatial(36), input(160);
  for (int i = 0; i < 36; ++i) spatial[static_cast<std::size_t>(i)] = static_cast<float>(.09f * std::sin(.37 * i));
  for (int t = 0; t < 5; ++t)
    for (int i = 0; i < 32; ++i) input[static_cast<std::size_t>(t * 32 + i)] = static_cast<float>(.3f * std::cos(.29 * i));
  const auto inflated = i3d::InflateI3d(nn::from_vector(spatial, {3, 3, 2, 2}), 3).to_vector<float>();
  check::Require(inflated.size() == 108, "inflated [3,3,3,2,2] filter");
  for (int t = 0; t < 3; ++t)
    for (int i = 0; i < 36; ++i)
      check::Near(inflated[static_cast<std::size_t>(36 * t + i)], spatial[static_cast<std::size_t>(i)] / 3, 0,
                  "all inflated temporal slices divided by T");
  // Inflate (tile then divide) inside the differentiated computation, as TF does.
  const auto run = [](const std::vector<float>& x_values, const std::vector<float>& w_values, std::vector<Tensor>* grads) {
    Tensor x = nn::from_vector(x_values, {1, 5, 4, 4, 2}), w = nn::from_vector(w_values, {3, 3, 2, 2});
    x.requires_grad_(true);
    w.requires_grad_(true);
    const Tensor w3 = nn::tile(nn::expand_dims(w, 0), {3, 1, 1, 1, 1}) / 3.0;
    const Tensor y  = nn::conv3d(x, w3, nn::conv3d_options(1, Padding::Same));
    if (grads) *grads = nn::grad(nn::sum(y * y), {w, x});
    return y.to_vector<float>();
  };
  std::vector<Tensor> g;
  const auto          out = run(input, spatial, &g);
  const auto host = [&](const std::vector<float>& a, const std::vector<float>& b, std::vector<double>* output = nullptr) {
    double sum = 0;
    for (int t = 0; t < 5; ++t)
      for (int h = 0; h < 4; ++h)
        for (int z = 0; z < 4; ++z)
          for (int c = 0; c < 2; ++c) {
            double value = 0;
            for (int kt = 0; kt < 3; ++kt)
              for (int kh = 0; kh < 3; ++kh)
                for (int kw = 0; kw < 3; ++kw)
                  for (int ci = 0; ci < 2; ++ci) {
                    const int it = t + kt - 1, ih = h + kh - 1, iw = z + kw - 1;
                    if (it >= 0 && it < 5 && ih >= 0 && ih < 4 && iw >= 0 && iw < 4)
                      value += double(a[static_cast<std::size_t>(((it * 4 + ih) * 4 + iw) * 2 + ci)]) *
                               b[static_cast<std::size_t>(((kh * 3 + kw) * 2 + ci) * 2 + c)] / 3.;
                  }
            sum += value * value;
            if (output) output->push_back(value);
          }
    return sum;
  };
  std::vector<double> expected;
  host(input, spatial, &expected);
  double forward = 0, gradient = 0, constant = 0;
  for (std::size_t i = 0; i < 160; ++i) forward = std::max(forward, std::abs(expected[i] - out[i]));
  // Independent spatial-only correlation; SAME temporal edge attenuation is 2/3.
  for (int t = 0; t < 5; ++t)
    for (int h = 0; h < 4; ++h)
      for (int z = 0; z < 4; ++z)
        for (int c = 0; c < 2; ++c) {
          double value = 0;
          for (int kh = 0; kh < 3; ++kh)
            for (int kw = 0; kw < 3; ++kw)
              for (int ci = 0; ci < 2; ++ci) {
                const int ih = h + kh - 1, iw = z + kw - 1;
                if (ih >= 0 && ih < 4 && iw >= 0 && iw < 4)
                  value += double(input[static_cast<std::size_t>((ih * 4 + iw) * 2 + ci)]) *
                           spatial[static_cast<std::size_t>(((kh * 3 + kw) * 2 + ci) * 2 + c)];
              }
          if (t == 0 || t == 4) value *= 2. / 3;
          constant = std::max(constant, std::abs(value - out[static_cast<std::size_t>(((t * 4 + h) * 4 + z) * 2 + c)]));
        }
  const auto gw = g[0].to_vector<float>(), gx = g[1].to_vector<float>();
  for (int variable = 0; variable < 2; ++variable) {
    const auto& original = variable == 0 ? spatial : input;
    for (std::size_t i = 0; i < original.size(); ++i) {
      auto plus = original, minus = original;
      plus[i] += .001f;
      minus[i] -= .001f;
      const double denominator = double(plus[i]) - minus[i];
      const double derivative  = variable == 0 ? (host(input, plus) - host(input, minus)) / denominator
                                               : (host(plus, spatial) - host(minus, spatial)) / denominator;
      gradient = std::max(gradient, std::abs(derivative - (variable == 0 ? gw[i] : gx[i])));
    }
  }
  check::Near(forward, 0, 1e-7, "all160 independent inflated output coordinates");
  check::Near(constant, 0, 1e-7, "repeated-image spatial response and explicit temporal zero-padding borders");
  check::Near(gradient, 0, 2e-6, "all36source-kernel+160input finite differences");
  std::cout << "inflation:all160outputs error=" << forward << ",constant-video/interior and border error=" << constant
            << ",all196source/input FD error=" << gradient << std::endl;
}

void UnitChecks() {
  nn::ParameterStore p(691), state(693);
  const std::vector<float> input{-.7f, .2f, .3f, -.6f, .9f, .4f, -.2f, .8f}, weight{.4f, -.3f, .2f, .7f},
      beta{.13f, -.09f};
  std::vector<float> coefficient(8);
  for (int i = 0; i < 8; ++i) coefficient[static_cast<std::size_t>(i)] = static_cast<float>(.1f * std::cos(.7 * i));
  Tensor x = nn::from_vector(input, {2, 2, 1, 1, 2});
  {
    nn::NoGradGuard guard;
    i3d::I3dUnit(p, state, "fixture", x, 2, 1, 1, false, .5, false);
  }
  p.set("fixture/W", nn::from_vector(weight, {1, 1, 1, 2, 2}));
  p.set("fixture/beta", nn::from_vector(beta, {2}));
  x.requires_grad_(true);
  const Tensor y = i3d::I3dUnit(p, state, "fixture", x, 2, 1, 1, true, .5, false);
  const auto   gradients =
      nn::grad(nn::sum(y * nn::from_vector(coefficient, {2, 2, 1, 1, 2})), {p.at("fixture/W"), p.at("fixture/beta"), x});
  std::vector<std::vector<double>> values(3);
  values[0].assign(weight.begin(), weight.end());
  values[1].assign(beta.begin(), beta.end());
  values[2].assign(input.begin(), input.end());
  const auto host = [&](const std::vector<std::vector<double>>& v, std::vector<double>* output = nullptr,
                        std::vector<double>* moments = nullptr) {
    double z[8], mu[2] = {}, var[2] = {};
    for (int n = 0; n < 4; ++n)
      for (int c = 0; c < 2; ++c) {
        z[2 * n + c] = v[2][2 * n] * v[0][c] + v[2][2 * n + 1] * v[0][2 + c];
        mu[c] += z[2 * n + c] / 4;
      }
    for (int n = 0; n < 4; ++n)
      for (int c = 0; c < 2; ++c) var[c] += std::pow(z[2 * n + c] - mu[c], 2) / 4;
    double result = 0;
    for (int n = 0; n < 4; ++n)
      for (int c = 0; c < 2; ++c) {
        const double value = std::max(0., (z[2 * n + c] - mu[c]) / std::sqrt(var[c] + .001) + v[1][c]);
        result += value * coefficient[static_cast<std::size_t>(2 * n + c)];
        if (output) output->push_back(value);
      }
    if (moments) *moments = {mu[0], mu[1], var[0], var[1]};
    return result;
  };
  std::vector<double> expected, moments;
  host(values, &expected, &moments);
  const auto yv = y.to_vector<float>();
  double     error = 0, derivative_error = 0;
  for (std::size_t i = 0; i < 8; ++i) error = std::max(error, std::abs(expected[i] - yv[i]));
  for (std::size_t variable = 0; variable < 3; ++variable) {
    const auto g = gradients[variable].to_vector<float>();
    for (std::size_t i = 0; i < values[variable].size(); ++i) {
      auto plus = values, minus = values;
      plus[variable][i] += 1e-5;
      minus[variable][i] -= 1e-5;
      derivative_error = std::max(derivative_error, std::abs((host(plus) - host(minus)) / 2e-5 - g[i]));
    }
  }
  {
    nn::NoGradGuard guard;  // the training step's moving-moment update
    i3d::I3dUnit(p, state, "fixture", x, 2, 1, 1, true, .5, true);
  }
  const auto inference = i3d::I3dUnit(p, state, "fixture", x, 2, 1, 1, false, .5, false).to_vector<float>();
  const auto moving_mean = state.at("fixture/mean").to_vector<float>(), moving_var = state.at("fixture/variance").to_vector<float>();
  for (std::size_t c = 0; c < 2; ++c) {
    check::Near(moving_mean[c], moments[c] * .5, 1e-7, "moving mean exact population update");
    check::Near(moving_var[c], .5 + .5 * moments[2 + c], 1e-7, "moving variance population convention");
    for (std::size_t n = 0; n < 4; ++n) {
      const double z = values[2][2 * n] * values[0][c] + values[2][2 * n + 1] * values[0][2 + c];
      const double e = std::max(0., (z - .5 * moments[c]) / std::sqrt(.5 + .5 * moments[2 + c] + .001) + values[1][c]);
      check::Near(inference[2 * n + c], e, 1e-6, "stored-stat inference independent of test batch moments");
    }
  }
  check::Near(error, 0, 1e-6, "3D Unit train output versus independent population normalization");
  check::Near(derivative_error, 0, 1e-6, "all14 kernel/offset/input populationBN-ReLU host finite differences");
  std::cout << "I3D Unit populationBN/ReLU forward error=" << error << ",all14hostFD error=" << derivative_error
            << ";moving population moments/stored-stat inference passed\n";
}

/// Endpoint shapes [T, S, C] from the SAME padding rules alone.
std::vector<nn::Shape> ShapeTrace(const i3d::I3dConfig& cfg, const int channels[16]) {
  const auto same = [](std::int64_t in, std::int64_t k, std::int64_t s) {
    return nn::conv_output_size(in, k, s, 1, Padding::Same);
  };
  // (temporal window/stride, spatial window/stride) for each pool; convolutions keep sizes except Conv3d_1a (stride 2).
  std::vector<nn::Shape> shapes;
  std::int64_t           t = same(cfg.frames, 7, 2), s = same(cfg.image, 7, 2);
  const auto push = [&](int i) { shapes.push_back({1, t, s, s, channels[i] / cfg.divisor}); };
  push(0);
  s = same(s, 3, 2);  // MaxPool 1x3x3 / 1x2x2
  push(1);
  push(2);
  push(3);
  s = same(s, 3, 2);
  push(4);
  for (int block = 0, i = 5; block < 9; ++block) {
    if (block == 2) {
      t = same(t, 3, 2), s = same(s, 3, 2);
      push(i++);
    }
    if (block == 7) {
      t = same(t, 2, 2), s = same(s, 2, 2);
      push(i++);
    }
    push(i++);
  }
  return shapes;
}

void GraphChecks(bool run_full) {
  const int time[] = {32, 32, 32, 32, 32, 32, 32, 16, 16, 16, 16, 16, 16, 8, 8, 8},
            space[] = {112, 56, 56, 56, 28, 28, 28, 14, 14, 14, 14, 14, 14, 7, 7, 7},
            channels[] = {64, 64, 64, 192, 192, 256, 480, 480, 512, 512, 512, 528, 832, 832, 832, 1024};
  for (bool compact : {false, true}) {
    const i3d::I3dConfig reference = compact ? i3d::I3dConfig{8, 3, 16, 32} : i3d::I3dConfig{};
    // Full widths execute on 16x32x32 by default (full=1: at 64x224x224).
    const i3d::I3dConfig cfg = compact || run_full ? reference : i3d::I3dConfig{1, 400, 16, 32};
    const bool           small = cfg.frames == 16;
    const auto table = [&](bool at_small, int divisor) {
      std::vector<nn::Shape> shapes;
      for (int i = 0; i < 16; ++i)
        shapes.push_back({1, at_small ? time[i] / 4 : time[i], at_small ? (space[i] + 6) / 7 : space[i],
                          at_small ? (space[i] + 6) / 7 : space[i], channels[i] / divisor});
      return shapes;
    };
    check::Require(ShapeTrace(reference, channels) == table(compact, reference.divisor),
                   "I3D original or compact endpoint dimensions from the SAME padding rules");
    nn::ParameterStore p(683), state(685);
    nn::Generator      gen(5);
    const Tensor       rgb = nn::uniform({1, cfg.frames, cfg.image, cfg.image, 3}, 0.0, 1.0, gen);
    const auto         net = i3d::I3d(p, state, rgb, cfg);
    check::Require(net.stages.size() == 16, "three stem convs/four pools/nine Mixed endpoints");
    const auto expected_shapes = table(small, cfg.divisor);
    for (std::size_t i = 0; i < 16; ++i)
      check::Require(net.stages[i].shape() == expected_shapes[i], "I3D executed endpoint dimension");
    check::Require(net.pooled.shape() == nn::Shape({1, small ? 1 : 7, 1, 1, 1024 / cfg.divisor}) &&
                       net.logits.shape() == nn::Shape({1, cfg.classes}),
                   "I3D pooled/logit shapes");
    check::Require(p.size() == 116 && state.size() == 114, "57W+57beta+headW/b and114 moving moments");
    // Independent per-unit parameter inventory from the published branch table.
    const int table9[9][6] = {{64, 96, 128, 16, 32, 32},     {128, 128, 192, 32, 96, 64},   {192, 96, 208, 16, 48, 64},
                              {160, 112, 224, 24, 64, 64},   {128, 128, 256, 24, 64, 64},   {112, 144, 288, 32, 64, 64},
                              {256, 160, 320, 32, 128, 128}, {256, 160, 320, 32, 128, 128}, {384, 192, 384, 48, 128, 128}};
    const auto w     = [&](int x) { return x / cfg.divisor; };
    const auto count = [](int a, int b, int k) { return std::int64_t{a} * b * k * k * k + b; };
    std::int64_t expected = count(3, w(64), 7) + count(w(64), w(64), 1) + count(w(64), w(192), 3);
    int          in       = w(192);
    for (const auto& c : table9) {
      expected += count(in, w(c[0]), 1) + count(in, w(c[1]), 1) + count(w(c[1]), w(c[2]), 3) + count(in, w(c[3]), 1) +
                  count(w(c[3]), w(c[4]), 3) + count(in, w(c[5]), 1);
      in = w(c[0]) + w(c[2]) + w(c[4]) + w(c[5]);
    }
    expected += std::int64_t{in} * cfg.classes + cfg.classes;
    check::Require(expected == p.count(), "independent57-unit plushead parameter inventory");
    const auto gradients = nn::grad(nn::mean(nn::square(net.logits)), p.parameters());
    check::Require(gradients.size() == 116, "all116 I3D learned gradients");
    for (const auto& g : gradients) {
      check::Require(g.defined(), "no missing I3D parameter gradient");
      check::Require(nn::all(nn::is_finite(g)).item<double>() != 0, "finite I3D gradient");
    }
    std::cout << (compact ? "compact" : "full") << " I3D graph:" << p.count()
              << " learned parameters/116 tensors,114 moving tensors;all16 endpoints/9Mixed shapes,independent "
                 "inventory and all gradients passed;"
              << (compact ? "executed 16x32x32"
                          : small ? "full width executed on 16x32x32 (64x224 table from padding rules; full=1 executes it)"
                                  : "full network executed at 64x224x224")
              << "\n";
  }
}

void Replay(const std::filesystem::path& dir) {
  i3d::I3dModel model;
  std::cout << "I3D compact replay of TF arrays " << dir.string() << ":\n";
  check::Near(c3d::ReplayTfVideo(dir, model), 0, 1e-5, "TF-trained arrays reproduce TF test/reverse/repeat logits");
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int                   epochs = 3;
    bool                  checks = false, full = false;
    std::filesystem::path root   = paths::Dataset("action_video"), replay;
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg.rfind("epochs=", 0) == 0)
        epochs = std::stoi(arg.substr(7));
      else if (arg.rfind("data=", 0) == 0)
        root = arg.substr(5);
      else if (arg == "checks_only=1" || arg == "checks_only=0")
        checks = arg == "checks_only=1";
      else if (arg == "full=1" || arg == "full=0")
        full = arg == "full=1";
      else if (arg.rfind("replay=", 0) == 0)
        replay = arg.substr(7);
      else if (arg.rfind("threads=", 0) == 0)
        nn::set_num_threads(std::stoi(arg.substr(8)));
      else
        throw std::invalid_argument("expected epochs=N data=PATH checks_only=0|1 full=0|1 replay=DIR threads=N");
    }
    InflationChecks();
    UnitChecks();
    GraphChecks(full);
    if (!replay.empty()) Replay(replay);
    if (!checks) i3d::TrainI3d(root, epochs);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
