// CMD: just run-nn c3d
// C3D: eight 3x3x3 convolutions, five original-ceil pools, FC6/FC7/dropout/head.
//
// Port of tflow's c3d. Real KTH video frames, person-disjoint subset; three
// compact epochs by default. Checks: a signed 3x3x3 multichannel convolution
// against a host loop, with all 164 kernel/bias/input derivatives against
// host finite differences; odd-edge Caffe-ceil pool values and gradients;
// full Sports-1M and compact C3D stage shapes, parameter census (79,991,015
// and 114,963) and all 22 gradients.
//
// TF -> datamunge: no graph or session. TF's static shape inference of the
// full 80M-parameter graph becomes, by default, the same stage-shape tables
// and census derived from nn::conv_output_size and the layer shapes, while
// the compact network is executed and differentiated for real; full=1 also
// runs the full Sports-1M network forward and backward on one clip (about
// 1.3 GB, several seconds). Weights come from datamunge's generator, so the
// training trajectory matches TF only statistically (the tiny 6-clip test
// set makes the accuracies noisy in both).
// Prepare (in tflow_examples): just action-video-assets.
// Options: epochs=3 data=PATH checks_only=0|1 full=0|1 threads=N.
// Artifacts: output/nn/models/c3d/.

#include <cmath>
#include <cstdio>
#include <datamunge/nn/conv.hpp>
#include <datamunge/nn/layers.hpp>
#include <datamunge/nn/nn.hpp>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "common/c3d.hpp"
#include "common/checks.hpp"
#include "common/paths.hpp"

// Experiment workflow from examples/nn/common/c3d.hpp.
namespace c3d {
using namespace ::c3d;
inline void TrainC3d(const std::filesystem::path& root, int epochs) {
  check::Require(epochs > 0, "positive short C3D epochs");
  const auto  pools = LoadVideo(root);
  const auto& fit = pools[0];
  const auto& validation = pools[1];
  const auto& test = pools[2];
  C3dModel    model;
  const auto  directory = paths::OutputDir("models/c3d");
  const auto  checkpoint = (directory / "best.npz").string();
  const auto  probe = VideoRows(fit, {0, 1});
  const double initial = ScoreLogits(model.Predict(fit), fit.labels, 3).nll;
  const Tensor untrained = model.Predict(test);
  {
    const auto params = model.parameters.parameters();
    const auto grads  = nn::grad(model.Loss(probe, true), params);
    check::Require(grads.size() == 22, "all 22 compact C3D parameter derivatives");
    for (const auto& g : grads) {
      const auto v    = g.to_vector<float>();
      double     norm = 0;
      for (float x : v) {
        check::Require(std::isfinite(x), "finite every compact3D/dense gradient coordinate");
        norm += static_cast<double>(x) * x;
      }
      check::Require(norm > 0, "nonzero every compact3D/dense parameter tensor derivative");
    }
  }
  std::cout << "C3D compact:" << model.parameters.count() << " parameters/" << model.parameters.size()
            << "tensors;8three-dimensional convs,5ceilpools,FC64/64,dropout.5,16x32x32RGB;" << epochs
            << " epochs,batch2,Adam.001,all22 gradients finite/nonzero,initial fitCE=" << initial << std::endl;
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
      const Tensor loss  = model.Loss(VideoRows(fit, rows), true);
      const double value = loss.item<double>();
      check::Require(std::isfinite(value) && value >= 0, "finite C3D CE");
      optimizer.minimize(loss);
      sum += static_cast<double>(rows.size()) * value;
      seen += rows.size();
    }
    check::Require(seen == static_cast<std::size_t>(fit.size()), "all fitting real clips once per epoch");
    const double value = ScoreLogits(model.Predict(validation), validation.labels, 3).nll;
    if (value < best) {
      best     = value;
      selected = epoch;
      model.parameters.save(checkpoint);
    }
    history << epoch << '\t' << sum / seen << '\t' << value << '\n';
    history.flush();
    std::cout << " epoch " << epoch << ":trainCE=" << sum / seen << ",validationCE=" << value << std::endl;
  }
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  model.parameters.load(checkpoint);
  check::Near(ScoreLogits(model.Predict(validation), validation.labels, 3).nll, best, 1e-7, "restored C3D validation CE");
  const double learned = ScoreLogits(model.Predict(fit), fit.labels, 3).nll;
  check::Require(learned < initial, "short-run fixed fitting inference CE improves");

  // A fresh model restored from the checkpoint sees only RGB clips.
  C3dModel deployed;
  deployed.parameters.load(checkpoint);
  const auto   before    = deployed.parameters.flatten();
  const Tensor logits    = deployed.Predict(test), reference = model.Predict(test), repeat = deployed.Predict(test);
  const auto   lv = logits.to_vector<float>(), rv = reference.to_vector<float>(), pv = repeat.to_vector<float>();
  for (std::size_t i = 0; i < lv.size(); ++i) {
    check::Near(lv[i], rv[i], 1e-6, "independent C3D checkpoint parity");
    check::Near(lv[i], pv[i], 0, "dropout-free deterministic video inference");
  }
  const auto single = deployed.Predict(VideoRows(test, {0})).to_vector<float>();
  for (std::size_t c = 0; c < 3; ++c) check::Near(single[c], lv[c], 1e-6, "single/batch clip inference");

  // Controls: frames reversed in time, and the first frame repeated.
  const std::vector<std::int64_t> all{0, 1, 2, 3, 4, 5};
  VideoData reverse = VideoRows(test, all), still = VideoRows(test, all);
  reverse.rgb = nn::reverse(test.rgb, {1}).contiguous();
  still.rgb   = nn::tile(nn::slice_axis(test.rgb, 1, 0, 1), {1, 16, 1, 1, 1}).contiguous();
  const Tensor reversed = deployed.Predict(reverse), repeated_frame = deployed.Predict(still);
  double       order = 0, motion = 0;
  const auto   rev = reversed.to_vector<float>(), rep = repeated_frame.to_vector<float>();
  for (std::size_t i = 0; i < lv.size(); ++i) {
    order += std::abs(static_cast<double>(lv[i]) - rev[i]);
    motion += std::abs(static_cast<double>(lv[i]) - rep[i]);
  }
  check::Require(order > 1e-7 && motion > 1e-7, "C3D depends on temporal order and genuine multi-frame input");
  check::Require(before == deployed.parameters.flatten(), "all22 video model tensors immutable");
  {
    nn::NamedTensors arrays;
    const auto       ps = deployed.parameters.parameters();
    for (std::size_t i = 0; i < ps.size(); ++i) arrays.emplace_back("p" + std::to_string(i), ps[i]);
    nn::save_npz((directory / "model_arrays.npz").string(), arrays);
  }
  const Tensor appearance = VideoNearest(fit, test, false), energy = VideoNearest(fit, test, true);
  const Tensor prior      = nn::zeros({6, 3});
  PrintVideo("trained C3D", test, logits);
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
  std::cout << "Artifacts:" << directory.string()
            << "\nTiny3-action/6-person KTH subset,one test person,six clips,not full KTH/Sports1M benchmark accuracy.\n";
}
}  // namespace c3d



namespace nn = datamunge::nn;
using nn::Padding;
using nn::Tensor;

namespace {

std::vector<double> HostConv(const std::vector<double>& x, const std::vector<double>& w, const std::vector<double>& b) {
  std::vector<double> out(54);
  for (int t = 0; t < 3; ++t)
    for (int y = 0; y < 3; ++y)
      for (int z = 0; z < 3; ++z)
        for (int c = 0; c < 2; ++c) {
          double value = b[static_cast<std::size_t>(c)];
          for (int kt = 0; kt < 3; ++kt)
            for (int ky = 0; ky < 3; ++ky)
              for (int kz = 0; kz < 3; ++kz) {
                const int it = t + kt - 1, iy = y + ky - 1, iz = z + kz - 1;
                if (it < 0 || it >= 3 || iy < 0 || iy >= 3 || iz < 0 || iz >= 3) continue;
                for (int in = 0; in < 2; ++in)
                  value += x[static_cast<std::size_t>(((it * 3 + iy) * 3 + iz) * 2 + in)] *
                           w[static_cast<std::size_t>((((kt * 3 + ky) * 3 + kz) * 2 + in) * 2 + c)];
              }
          out[static_cast<std::size_t>(((t * 3 + y) * 3 + z) * 2 + c)] = value;
        }
  return out;
}

double Objective(const std::vector<std::vector<double>>& values, const std::vector<double>& coefficient) {
  const auto out   = HostConv(values[2], values[0], values[1]);
  double     total = 0;
  for (std::size_t i = 0; i < 54; ++i) total += coefficient[i] * out[i];
  return total;
}

void ConvChecks() {
  std::vector<float>               input(54), coefficient(54), weight(108);
  std::vector<std::vector<double>> values(3);
  std::vector<double>              host_coefficient;
  for (int i = 0; i < 54; ++i) {
    input[static_cast<std::size_t>(i)]       = static_cast<float>(.2 * std::sin(.31 * i));
    coefficient[static_cast<std::size_t>(i)] = static_cast<float>(.1 * std::cos(.37 * i));
    values[2].push_back(input[static_cast<std::size_t>(i)]);
    host_coefficient.push_back(coefficient[static_cast<std::size_t>(i)]);
  }
  for (int i = 0; i < 108; ++i) {
    weight[static_cast<std::size_t>(i)] = static_cast<float>(.07 * std::sin(.23 * i + .1));
    values[0].push_back(weight[static_cast<std::size_t>(i)]);
  }
  const std::vector<float> bias{.1f, -.2f};
  values[1] = {bias[0], bias[1]};
  nn::ParameterStore p(653);
  Tensor             x = nn::from_vector(input, {1, 3, 3, 3, 2});
  {
    nn::NoGradGuard guard;
    nn::conv3d(p, "conv", x, 2, 3);  // create, then assign the asymmetric coefficients
  }
  p.set("conv/W", nn::from_vector(weight, {3, 3, 3, 2, 2}));
  p.set("conv/b", nn::from_vector(bias, {2}));
  x.requires_grad_(true);
  const Tensor y         = nn::conv3d(p, "conv", x, 2, 3);
  const Tensor objective = nn::sum(y * nn::from_vector(coefficient, {1, 3, 3, 3, 2}));
  const auto   gradient  = nn::grad(objective, {p.at("conv/W"), p.at("conv/b"), x});
  const auto   expected  = HostConv(values[2], values[0], values[1]);
  const auto   yv        = y.to_vector<float>();
  double       value_error = 0, gradient_error = 0;
  for (std::size_t i = 0; i < 54; ++i) value_error = std::max(value_error, std::abs(expected[i] - yv[i]));
  int coordinates = 0;
  for (std::size_t v = 0; v < 3; ++v) {
    const auto g = gradient[v].to_vector<float>();
    for (std::size_t i = 0; i < values[v].size(); ++i) {
      auto plus = values, minus = values;
      plus[v][i] += 1e-5;
      minus[v][i] -= 1e-5;
      const double derivative = (Objective(plus, host_coefficient) - Objective(minus, host_coefficient)) / 2e-5;
      gradient_error          = std::max(gradient_error, std::abs(derivative - g[i]));
      ++coordinates;
    }
  }
  check::Require(coordinates == 164, "all108kernel+2bias+54input coordinates covered");
  check::Near(value_error, 0, 1e-7, "independent signed time-height-width cross correlation");
  check::Near(gradient_error, 0, 1e-7, "ALL164signed3D coordinates versus independent host finite differences");
  std::cout << " signed3x3x3 multichannel forward error=" << value_error
            << ",ALL164 kernel/bias/input host FD error=" << gradient_error << std::endl;
}

void PoolChecks() {
  std::vector<float> input(18);
  for (int i = 0; i < 18; ++i) input[static_cast<std::size_t>(i)] = static_cast<float>(-100 - i);
  Tensor x = nn::from_vector(input, {1, 2, 3, 3, 1});
  x.requires_grad_(true);
  const Tensor pooled = c3d::C3dPool(x, false), first = c3d::C3dPool(x, true);
  const Tensor gradient = nn::grad(nn::sum(pooled), {x})[0];
  check::Require(pooled.shape() == nn::Shape({1, 1, 2, 2, 1}) && first.shape() == nn::Shape({1, 2, 2, 2, 1}),
                 "ceil odd spatial pool and pool1preserves time");
  const double expected[] = {-100, -102, -106, -108};
  const auto   pv         = pooled.to_vector<float>();
  for (std::size_t i = 0; i < 4; ++i)
    check::Near(pv[i], expected[i], 0, "right/bottom partial window excludes negative-infinity padding");
  const auto gv = gradient.to_vector<float>();
  for (int t = 0; t < 2; ++t)
    for (int y = 0; y < 3; ++y)
      for (int z = 0; z < 3; ++z)
        check::Near(gv[static_cast<std::size_t>((t * 3 + y) * 3 + z)], float(t == 0 && y % 2 == 0 && z % 2 == 0), 0,
                    "unique-max odd-edge gradient onlyactualsamples");
  // tflow's formulation: PadV2 with -inf on the odd trailing edges, then VALID pooling.
  const Tensor padded = nn::pad(x, {{0, 0}, {0, 0}, {0, 1}, {0, 1}, {0, 0}}, nn::PadMode::Constant, -INFINITY);
  const auto   tf_way = nn::max_pool3d(padded, 2, 2, Padding::Valid).to_vector<float>();
  for (std::size_t i = 0; i < 4; ++i) check::Near(tf_way[i], pv[i], 0, "SAME pool equals -inf PadV2 + VALID pool");
  std::cout << " Caffe-ceil3Dpool boundary values/all input gradients/temporal pool1 fixtures passed\n";
}

std::int64_t Census(const c3d::C3dConfig& cfg) {
  const int    multiples[] = {1, 2, 4, 4, 8, 8, 8, 8};
  std::int64_t total = 0, in = 3;
  for (int m : multiples) {
    const std::int64_t out = cfg.base * m;
    total += 27 * in * out + out;
    in = out;
  }
  std::int64_t time = 16, image = cfg.image;
  for (int i = 0; i < 5; ++i) {
    if (i > 0) time = nn::conv_output_size(time, 2, 2, 1, Padding::Same);
    image = nn::conv_output_size(image, 2, 2, 1, Padding::Same);
  }
  const std::int64_t flat = time * image * image * in;
  return total + flat * cfg.dense + cfg.dense + std::int64_t{cfg.dense} * cfg.dense + cfg.dense +
         std::int64_t{cfg.dense} * cfg.classes + cfg.classes;
}

void GraphChecks(bool run_full) {
  for (bool compact : {false, true}) {
    const c3d::C3dConfig cfg = compact ? c3d::kCompact : c3d::C3dConfig{};
    const int time[] = {16, 16, 8, 8, 4, 4, 2, 2}, full_space[] = {112, 56, 28, 28, 14, 14, 7, 7},
              small_space[] = {32, 16, 8, 8, 4, 4, 2, 2}, width[] = {1, 2, 4, 4, 8, 8, 8, 8};
    const int pt[] = {16, 8, 4, 2, 1}, ps[] = {56, 28, 14, 7, 4}, cs[] = {16, 8, 4, 2, 1}, pc[] = {1, 2, 4, 8, 8};
    const std::int64_t expected_params = compact ? 114963 : 79991015;
    // Stage sizes from the TF padding rules: SAME convolutions keep every axis, ceil pools halve (rounding up).
    std::int64_t t = 16, s = cfg.image;
    int          pool = 0;
    for (int i = 0; i < 8; ++i) {
      t = nn::conv_output_size(t, 3, 1, 1, Padding::Same);
      s = nn::conv_output_size(s, 3, 1, 1, Padding::Same);
      check::Require(t == time[i] && s == (compact ? small_space[i] : full_space[i]), "original C3D full/compact stage dimensions");
      if (i == 0 || i == 1 || i == 3 || i == 5 || i == 7) {
        if (i != 0) t = nn::conv_output_size(t, 2, 2, 1, Padding::Same);
        s = nn::conv_output_size(s, 2, 2, 1, Padding::Same);
        check::Require(t == pt[pool] && s == (compact ? cs[pool] : ps[pool]), "original C3D ceil-pool dimensions");
        ++pool;
      }
    }
    check::Require(Census(cfg) == expected_params, "independent original-eight-conv/twoFC/head parameter count");
    if (compact || run_full) {
      // Execute: every stage shape, the census and all 22 gradients.
      nn::ParameterStore p(compact ? 659 : 661);
      nn::Generator      gen(5);
      const Tensor       x   = nn::uniform({1, 16, cfg.image, cfg.image, 3}, 0.0, 1.0, gen);
      const auto         net = c3d::C3d(p, x, cfg);
      for (int i = 0; i < 8; ++i)
        check::Require(net.conv[static_cast<std::size_t>(i)].shape() ==
                           nn::Shape({1, time[i], compact ? small_space[i] : full_space[i], compact ? small_space[i] : full_space[i],
                                      cfg.base * width[i]}),
                       "executed C3D conv stage shapes");
      for (int i = 0; i < 5; ++i)
        check::Require(net.pools[static_cast<std::size_t>(i)].shape() ==
                           nn::Shape({1, pt[i], compact ? cs[i] : ps[i], compact ? cs[i] : ps[i], cfg.base * pc[i]}),
                       "executed C3D pool stage shapes");
      check::Require(net.flat.shape() == nn::Shape({1, compact ? 32 : 8192}) && net.fc6.shape() == nn::Shape({1, cfg.dense}) &&
                         net.fc7.shape() == nn::Shape({1, cfg.dense}) && net.logits.shape() == nn::Shape({1, cfg.classes}),
                     "flatten/FC shapes");
      check::Require(p.count() == expected_params && p.size() == 22, "executed parameter census");
      const auto gradients = nn::grad(nn::mean(nn::square(net.logits)), p.parameters());
      check::Require(gradients.size() == 22, "all22 graph gradients constructed");
      for (const auto& g : gradients) {
        check::Require(g.defined(), "no missing C3D learned gradient");
        for (float v : g.to_vector<float>()) check::Require(std::isfinite(v), "finite C3D gradient");
      }
    }
    std::cout << (compact ? " compact" : " full Sports1M") << " C3D graph:" << expected_params
              << " parameters/22tensors;all8conv/5ceilpool/FC shapes and gradients verified;"
              << (compact ? "executed" : run_full ? "full-width forward/backward executed" : "full-width execution NOT part of default CPU demo (full=1)")
              << "\n";
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int                   epochs = 3;
    bool                  checks = false, full = false;
    std::filesystem::path root   = paths::Dataset("action_video");
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
      else if (arg.rfind("threads=", 0) == 0)
        nn::set_num_threads(std::stoi(arg.substr(8)));
      else
        throw std::invalid_argument("expected epochs=N data=PATH checks_only=0|1 full=0|1 threads=N");
    }
    ConvChecks();
    PoolChecks();
    GraphChecks(full);
    if (!checks) c3d::TrainC3d(root, epochs);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
