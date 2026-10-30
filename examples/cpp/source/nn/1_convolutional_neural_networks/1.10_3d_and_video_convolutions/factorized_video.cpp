// CMD: just run-nn factorized_video
// R(2+1)D-18 and P3D-199 with serial/parallel/shortcut factorized 3D blocks.
// Explicit symmetric borders, parameter-matched intermediate widths, all P3D A/B/C.
//
// Port of tflow's factorized_video. Checks: R(2+1)D stride-2 and P3D-A/B/C
// signed fixtures against an independent host convolution/topology with all
// 530 factor/kernel/input derivatives against host finite differences;
// parameter-balance widths; full-reference and compact endpoint shapes,
// parameter inventories (R(2+1)D-18 31,505,325 = torchvision, P3D-199
// 66,492,240 = the authors' Caffe Kinetics model) and all gradients. Then the
// compact base-4 models train three epochs EACH on real person-disjoint KTH clips.
//
// TF -> datamunge: no graph or session. TF checks the full graphs by static
// shape inference without allocating them; here the full-width networks are
// built and executed (forward + all gradients) on a 16x32x32 clip, which
// checks the real parameter inventory and every endpoint at that size, and the
// reference-size (16x112x112 / 16x160x160) endpoint table is derived from the
// padding rules with nn::conv_output_size. full=1 also executes both full
// networks at the reference size (about 1.6 GB, a few seconds). Weights come from datamunge's
// generator, so training matches TF only statistically (the 6-clip test set is noisy in both). No
// pretrained weights or benchmark-quality claim. Prepare (in tflow_examples): just
// action-video-assets. replay=DIR loads a TF run's trained arrays
// (DIR/{r2plus1d,p3d}/model_arrays.npz) into our compact models and compares the logits on TF's
// saved test clips. Options: epochs=3 data=PATH checks_only=0|1 full=0|1 replay=DIR threads=N.
// Artifacts: output/nn/models/{r2plus1d,p3d}/.

#include <datamunge/datasets/assets.hpp>
#include "common/factorized_video.hpp"

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
#include "common/paths.hpp"

namespace nn = datamunge::nn;
namespace fv = factorized_video;
using nn::Padding;
using nn::Tensor;

namespace {

void TrainR2Plus1d(const std::filesystem::path& root, int epochs) {
  using namespace c3d;
  using namespace factorized_video;
  c3d::VideoTrainSpec spec;
  spec.directory = false ? "p3d" : "r2plus1d";
  spec.title     = false ? "P3D-199 compact" : "R(2+1)D-18 compact";
  spec.tsv       = true;
  spec.note =
      "Tiny3-action/6-person KTH subset,one test person,six clips,not full KTH/Kinetics benchmark "
      "accuracy. Random "
      "fan-out Kaiming initialization,RGB[-1,1],all BN trained;no ImageNet/Kinetics pretraining. "
      "Fullwidth execution "
      "excluded from default CPU demo.";

  check::Require(epochs > 0, "positive short video epochs");
  if (!std::filesystem::exists(root / "action_video.npz"))
    datamunge::datasets::prepare_example_data("video", (root).string());
  const auto    pools      = LoadVideo(root);
  const auto&   fit        = pools[0];
  const auto&   validation = pools[1];
  const auto&   test       = pools[2];
  R2Plus1dModel model;

  const auto directory        = paths::OutputDir("models/" + spec.directory);
  const auto checkpoint       = (directory / "best.npz").string(),
             stats_checkpoint = (directory / "best_state.npz").string();
  const auto   probe          = VideoRows(fit, {0, 1});
  const double initial        = ScoreLogits(VideoModelPredict(model, fit), fit.labels, 3).nll;
  const Tensor untrained      = VideoModelPredict(model, test);
  {
    const auto params = model.parameters.parameters();
    const auto grads  = nn::grad(
        nn::sparse_softmax_cross_entropy(model.Logits(probe.rgb, true, false), probe.labels),
        params);
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
  std::cout << spec.title << ":" << model.parameters.count() << " parameters/"
            << model.parameters.size() << " learned tensors+" << model.state.size()
            << " moving-stat tensors,16x32x32RGB;" << epochs
            << " epochs,batch2,Adam.001,all learned gradients finite/nonzero,initial fitCE="
            << initial << std::endl;
  nn::OptimizerConfig oc;
  oc.kind = nn::OptimizerKind::Adam;
  nn::Optimizer  optimizer(model.parameters, .001, oc);
  nn::BatchOrder batch_order(static_cast<std::size_t>(fit.size()), 2, 647);
  double         best     = INFINITY;
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
      if (rows.empty())
        break;
      const auto   batch = VideoRows(fit, rows);
      const Tensor loss =
          nn::sparse_softmax_cross_entropy(model.Logits(batch.rgb, true, true), batch.labels);
      const double value = loss.item<double>();
      check::Require(std::isfinite(value) && value >= 0, "finite video CE");
      optimizer.minimize(loss);
      sum += static_cast<double>(rows.size()) * value;
      seen += rows.size();
    }
    check::Require(
        seen == static_cast<std::size_t>(fit.size()), "all fitting real clips once per epoch");
    const double value =
        ScoreLogits(VideoModelPredict(model, validation), validation.labels, 3).nll;
    if (value < best) {
      best     = value;
      selected = epoch;
      model.parameters.save(checkpoint);
      model.state.save(stats_checkpoint);
    }
    history << epoch << '\t' << sum / seen << '\t' << value << '\n';
    history.flush();
    std::cout << " epoch " << epoch << ":trainCE=" << sum / seen << ",validationCE=" << value
              << std::endl;
  }
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  model.parameters.load(checkpoint);
  model.state.load(stats_checkpoint);
  check::Near(
      ScoreLogits(VideoModelPredict(model, validation), validation.labels, 3).nll,
      best,
      1e-7,
      "restored video validation CE");
  const double learned = ScoreLogits(VideoModelPredict(model, fit), fit.labels, 3).nll;
  if (spec.require_improvement)
    check::Require(learned < initial, "short-run fixed fitting inference CE improves");
  else
    check::Require(
        std::isfinite(learned),
        "finite restored fitting inference CE;quality is reported without threshold");

  // A fresh model restored from the checkpoint sees only RGB clips.
  R2Plus1dModel deployed;
  deployed.parameters.load(checkpoint);
  deployed.state.load(stats_checkpoint);
  const auto   before    = VideoModelSnapshot(deployed);
  const Tensor logits    = VideoModelPredict(deployed, test),
               reference = VideoModelPredict(model, test),
               repeat    = VideoModelPredict(deployed, test);
  const auto lv = logits.to_vector<float>(), rv = reference.to_vector<float>(),
             pv = repeat.to_vector<float>();
  for (std::size_t i = 0; i < lv.size(); ++i) {
    check::Near(lv[i], rv[i], 1e-6, "independent video checkpoint parity");
    check::Near(lv[i], pv[i], 0, "dropout-free deterministic video inference");
  }
  const auto single = VideoModelPredict(deployed, VideoRows(test, {0})).to_vector<float>();
  for (std::size_t c = 0; c < 3; ++c)
    check::Near(single[c], lv[c], 1e-6, "single/batch clip inference");

  const std::vector<std::int64_t> all{0, 1, 2, 3, 4, 5};
  VideoData                       reverse = VideoRows(test, all), still = VideoRows(test, all);
  reverse.rgb = nn::reverse(test.rgb, {1}).contiguous();
  still.rgb   = nn::tile(nn::slice_axis(test.rgb, 1, 0, 1), {1, 16, 1, 1, 1}).contiguous();
  const Tensor reversed       = VideoModelPredict(deployed, reverse),
               repeated_frame = VideoModelPredict(deployed, still);
  double     order = 0, motion = 0;
  const auto rev = reversed.to_vector<float>(), rep = repeated_frame.to_vector<float>();
  for (std::size_t i = 0; i < lv.size(); ++i) {
    order += std::abs(static_cast<double>(lv[i]) - rev[i]);
    motion += std::abs(static_cast<double>(lv[i]) - rep[i]);
  }
  check::Require(
      order > 1e-7 && motion > 1e-7,
      "video model depends on temporal order and genuine multi-frame input");
  check::Require(
      before == VideoModelSnapshot(deployed), "all learned and moving tensors immutable");
  {
    nn::NamedTensors arrays;
    const auto       ps = deployed.parameters.parameters(), ss = deployed.state.parameters();
    std::ofstream    mapping;
    if (spec.tsv) {
      mapping.open(directory / "model_arrays.tsv");
      mapping << "key\tname\tkind\n";
    }
    for (std::size_t i = 0; i < ps.size() + ss.size(); ++i) {
      const bool        learned_tensor = i < ps.size();
      const std::string key            = "p" + std::to_string(i);
      arrays.emplace_back(key, learned_tensor ? ps[i] : ss[i - ps.size()]);
      if (spec.tsv)
        mapping << key << '\t'
                << (learned_tensor ? deployed.parameters.names()[i]
                                   : deployed.state.names()[i - ps.size()])
                << '\t' << (learned_tensor ? "learned" : "moving") << '\n';
    }
    nn::save_npz((directory / "model_arrays.npz").string(), arrays);
    if (spec.tsv)
      check::Require(bool(mapping), "export named learned/moving inventory");
  }
  const Tensor appearance = VideoNearest(fit, test, false), energy = VideoNearest(fit, test, true);
  const Tensor prior = nn::zeros({6, 3});
  PrintVideo("trained " + spec.title, test, logits);
  PrintVideo("untrained", test, untrained);
  PrintVideo("fit static mean-frame1NN", test, appearance);
  PrintVideo("fit absolute-motion1NN", test, energy);
  PrintVideo("balanced prior tie class0", test, prior);
  PrintVideo("reversed-frame control", test, reversed);
  PrintVideo("repeated-first-frame control", test, repeated_frame);

  std::ofstream predictions(directory / "predictions.tsv");
  predictions << "video_id\tsegment\tperson\tlabel\tprediction\n";
  const auto vid = test.video.to_vector<int>(), seg = test.segment.to_vector<int>(),
             per = test.person.to_vector<int>(), lab = test.labels.to_vector<int>();
  for (std::size_t n = 0; n < 6; ++n) {
    std::size_t best_c = 0;
    for (std::size_t c = 1; c < 3; ++c)
      if (lv[n * 3 + c] > lv[n * 3 + best_c])
        best_c = c;
    predictions << vid[n] << '\t' << seg[n] << '\t' << per[n] << '\t' << lab[n] << '\t' << best_c
                << '\n';
  }
  nn::save_npz(
      (directory / "test_outputs.npz").string(),
      {{"rgb", test.rgb},
       {"labels", test.labels},
       {"person", test.person},
       {"video_id", test.video},
       {"segment", test.segment},
       {"logits", logits},
       {"untrained", untrained},
       {"appearance", appearance},
       {"motion", energy},
       {"reverse", reversed},
       {"repeat_first", repeated_frame},
       {"fit_rgb", fit.rgb},
       {"fit_labels", fit.labels},
       {"fit_person", fit.person},
       {"fit_video_id", fit.video}});
  for (const char* name : {"manifest.json", "ATTRIBUTION.txt", "test_contact.png"})
    if (std::filesystem::exists(root / name))
      std::filesystem::copy_file(
          root / name,
          directory / (std::string(name) == "manifest.json" ? "data_manifest.json" : name),
          std::filesystem::copy_options::overwrite_existing);
  check::Require(bool(history) && bool(predictions), "save real-video predictions/histories");
  std::cout << " selected=" << selected << ",fixed fitCE " << initial << " -> " << learned
            << ",reverse/repeat-first total logit changes=" << order << "/" << motion
            << ";standalone RGB-only restore/single-batch/determinism/immutable model passed\n";
  std::printf(" training time %.1f s (%.2f s/epoch)\n", seconds, seconds / epochs);
  std::cout << "Artifacts:" << directory.string() << "\n" << spec.note << "\n";
}

void TrainP3d(const std::filesystem::path& root, int epochs) {
  using namespace c3d;
  using namespace factorized_video;
  c3d::VideoTrainSpec spec;
  spec.directory = true ? "p3d" : "r2plus1d";
  spec.title     = true ? "P3D-199 compact" : "R(2+1)D-18 compact";
  spec.tsv       = true;
  spec.note =
      "Tiny3-action/6-person KTH subset,one test person,six clips,not full KTH/Kinetics benchmark "
      "accuracy. Random "
      "fan-out Kaiming initialization,RGB[-1,1],all BN trained;no ImageNet/Kinetics pretraining. "
      "Fullwidth execution "
      "excluded from default CPU demo.";

  check::Require(epochs > 0, "positive short video epochs");
  if (!std::filesystem::exists(root / "action_video.npz"))
    datamunge::datasets::prepare_example_data("video", (root).string());
  const auto  pools      = LoadVideo(root);
  const auto& fit        = pools[0];
  const auto& validation = pools[1];
  const auto& test       = pools[2];
  P3dModel    model;

  const auto directory        = paths::OutputDir("models/" + spec.directory);
  const auto checkpoint       = (directory / "best.npz").string(),
             stats_checkpoint = (directory / "best_state.npz").string();
  const auto   probe          = VideoRows(fit, {0, 1});
  const double initial        = ScoreLogits(VideoModelPredict(model, fit), fit.labels, 3).nll;
  const Tensor untrained      = VideoModelPredict(model, test);
  {
    const auto params = model.parameters.parameters();
    const auto grads  = nn::grad(
        nn::sparse_softmax_cross_entropy(model.Logits(probe.rgb, true, false), probe.labels),
        params);
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
  std::cout << spec.title << ":" << model.parameters.count() << " parameters/"
            << model.parameters.size() << " learned tensors+" << model.state.size()
            << " moving-stat tensors,16x32x32RGB;" << epochs
            << " epochs,batch2,Adam.001,all learned gradients finite/nonzero,initial fitCE="
            << initial << std::endl;
  nn::OptimizerConfig oc;
  oc.kind = nn::OptimizerKind::Adam;
  nn::Optimizer  optimizer(model.parameters, .001, oc);
  nn::BatchOrder batch_order(static_cast<std::size_t>(fit.size()), 2, 647);
  double         best     = INFINITY;
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
      if (rows.empty())
        break;
      const auto   batch = VideoRows(fit, rows);
      const Tensor loss =
          nn::sparse_softmax_cross_entropy(model.Logits(batch.rgb, true, true), batch.labels);
      const double value = loss.item<double>();
      check::Require(std::isfinite(value) && value >= 0, "finite video CE");
      optimizer.minimize(loss);
      sum += static_cast<double>(rows.size()) * value;
      seen += rows.size();
    }
    check::Require(
        seen == static_cast<std::size_t>(fit.size()), "all fitting real clips once per epoch");
    const double value =
        ScoreLogits(VideoModelPredict(model, validation), validation.labels, 3).nll;
    if (value < best) {
      best     = value;
      selected = epoch;
      model.parameters.save(checkpoint);
      model.state.save(stats_checkpoint);
    }
    history << epoch << '\t' << sum / seen << '\t' << value << '\n';
    history.flush();
    std::cout << " epoch " << epoch << ":trainCE=" << sum / seen << ",validationCE=" << value
              << std::endl;
  }
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  model.parameters.load(checkpoint);
  model.state.load(stats_checkpoint);
  check::Near(
      ScoreLogits(VideoModelPredict(model, validation), validation.labels, 3).nll,
      best,
      1e-7,
      "restored video validation CE");
  const double learned = ScoreLogits(VideoModelPredict(model, fit), fit.labels, 3).nll;
  if (spec.require_improvement)
    check::Require(learned < initial, "short-run fixed fitting inference CE improves");
  else
    check::Require(
        std::isfinite(learned),
        "finite restored fitting inference CE;quality is reported without threshold");

  // A fresh model restored from the checkpoint sees only RGB clips.
  P3dModel deployed;
  deployed.parameters.load(checkpoint);
  deployed.state.load(stats_checkpoint);
  const auto   before    = VideoModelSnapshot(deployed);
  const Tensor logits    = VideoModelPredict(deployed, test),
               reference = VideoModelPredict(model, test),
               repeat    = VideoModelPredict(deployed, test);
  const auto lv = logits.to_vector<float>(), rv = reference.to_vector<float>(),
             pv = repeat.to_vector<float>();
  for (std::size_t i = 0; i < lv.size(); ++i) {
    check::Near(lv[i], rv[i], 1e-6, "independent video checkpoint parity");
    check::Near(lv[i], pv[i], 0, "dropout-free deterministic video inference");
  }
  const auto single = VideoModelPredict(deployed, VideoRows(test, {0})).to_vector<float>();
  for (std::size_t c = 0; c < 3; ++c)
    check::Near(single[c], lv[c], 1e-6, "single/batch clip inference");

  const std::vector<std::int64_t> all{0, 1, 2, 3, 4, 5};
  VideoData                       reverse = VideoRows(test, all), still = VideoRows(test, all);
  reverse.rgb = nn::reverse(test.rgb, {1}).contiguous();
  still.rgb   = nn::tile(nn::slice_axis(test.rgb, 1, 0, 1), {1, 16, 1, 1, 1}).contiguous();
  const Tensor reversed       = VideoModelPredict(deployed, reverse),
               repeated_frame = VideoModelPredict(deployed, still);
  double     order = 0, motion = 0;
  const auto rev = reversed.to_vector<float>(), rep = repeated_frame.to_vector<float>();
  for (std::size_t i = 0; i < lv.size(); ++i) {
    order += std::abs(static_cast<double>(lv[i]) - rev[i]);
    motion += std::abs(static_cast<double>(lv[i]) - rep[i]);
  }
  check::Require(
      order > 1e-7 && motion > 1e-7,
      "video model depends on temporal order and genuine multi-frame input");
  check::Require(
      before == VideoModelSnapshot(deployed), "all learned and moving tensors immutable");
  {
    nn::NamedTensors arrays;
    const auto       ps = deployed.parameters.parameters(), ss = deployed.state.parameters();
    std::ofstream    mapping;
    if (spec.tsv) {
      mapping.open(directory / "model_arrays.tsv");
      mapping << "key\tname\tkind\n";
    }
    for (std::size_t i = 0; i < ps.size() + ss.size(); ++i) {
      const bool        learned_tensor = i < ps.size();
      const std::string key            = "p" + std::to_string(i);
      arrays.emplace_back(key, learned_tensor ? ps[i] : ss[i - ps.size()]);
      if (spec.tsv)
        mapping << key << '\t'
                << (learned_tensor ? deployed.parameters.names()[i]
                                   : deployed.state.names()[i - ps.size()])
                << '\t' << (learned_tensor ? "learned" : "moving") << '\n';
    }
    nn::save_npz((directory / "model_arrays.npz").string(), arrays);
    if (spec.tsv)
      check::Require(bool(mapping), "export named learned/moving inventory");
  }
  const Tensor appearance = VideoNearest(fit, test, false), energy = VideoNearest(fit, test, true);
  const Tensor prior = nn::zeros({6, 3});
  PrintVideo("trained " + spec.title, test, logits);
  PrintVideo("untrained", test, untrained);
  PrintVideo("fit static mean-frame1NN", test, appearance);
  PrintVideo("fit absolute-motion1NN", test, energy);
  PrintVideo("balanced prior tie class0", test, prior);
  PrintVideo("reversed-frame control", test, reversed);
  PrintVideo("repeated-first-frame control", test, repeated_frame);

  std::ofstream predictions(directory / "predictions.tsv");
  predictions << "video_id\tsegment\tperson\tlabel\tprediction\n";
  const auto vid = test.video.to_vector<int>(), seg = test.segment.to_vector<int>(),
             per = test.person.to_vector<int>(), lab = test.labels.to_vector<int>();
  for (std::size_t n = 0; n < 6; ++n) {
    std::size_t best_c = 0;
    for (std::size_t c = 1; c < 3; ++c)
      if (lv[n * 3 + c] > lv[n * 3 + best_c])
        best_c = c;
    predictions << vid[n] << '\t' << seg[n] << '\t' << per[n] << '\t' << lab[n] << '\t' << best_c
                << '\n';
  }
  nn::save_npz(
      (directory / "test_outputs.npz").string(),
      {{"rgb", test.rgb},
       {"labels", test.labels},
       {"person", test.person},
       {"video_id", test.video},
       {"segment", test.segment},
       {"logits", logits},
       {"untrained", untrained},
       {"appearance", appearance},
       {"motion", energy},
       {"reverse", reversed},
       {"repeat_first", repeated_frame},
       {"fit_rgb", fit.rgb},
       {"fit_labels", fit.labels},
       {"fit_person", fit.person},
       {"fit_video_id", fit.video}});
  for (const char* name : {"manifest.json", "ATTRIBUTION.txt", "test_contact.png"})
    if (std::filesystem::exists(root / name))
      std::filesystem::copy_file(
          root / name,
          directory / (std::string(name) == "manifest.json" ? "data_manifest.json" : name),
          std::filesystem::copy_options::overwrite_existing);
  check::Require(bool(history) && bool(predictions), "save real-video predictions/histories");
  std::cout << " selected=" << selected << ",fixed fitCE " << initial << " -> " << learned
            << ",reverse/repeat-first total logit changes=" << order << "/" << motion
            << ";standalone RGB-only restore/single-batch/determinism/immutable model passed\n";
  std::printf(" training time %.1f s (%.2f s/epoch)\n", seconds, seconds / epochs);
  std::cout << "Artifacts:" << directory.string() << "\n" << spec.note << "\n";
}

struct HostVolume {
  int                 t, h, w, c;
  std::vector<double> x;
};

HostVolume HostFactorConv(
    const HostVolume&          input,
    const std::vector<double>& kernel,
    int                        kt,
    int                        kh,
    int                        kw,
    int                        out,
    int                        st = 1,
    int                        sh = 1,
    int                        sw = 1) {
  HostVolume y{(input.t + st - 1) / st, (input.h + sh - 1) / sh, (input.w + sw - 1) / sw, out, {}};
  y.x.resize(static_cast<std::size_t>(y.t * y.h * y.w * out));
  for (int t = 0; t < y.t; ++t)
    for (int h = 0; h < y.h; ++h)
      for (int w = 0; w < y.w; ++w)
        for (int c = 0; c < out; ++c) {
          double sum = 0;
          for (int a = 0; a < kt; ++a)
            for (int b = 0; b < kh; ++b)
              for (int d = 0; d < kw; ++d)
                for (int ci = 0; ci < input.c; ++ci) {
                  const int it = t * st + a - kt / 2, ih = h * sh + b - kh / 2,
                            iw = w * sw + d - kw / 2;
                  if (it >= 0 && it < input.t && ih >= 0 && ih < input.h && iw >= 0 && iw < input.w)
                    sum += input.x[static_cast<std::size_t>(
                               ((it * input.h + ih) * input.w + iw) * input.c + ci)] *
                           kernel[static_cast<std::size_t>(
                               (((a * kh + b) * kw + d) * input.c + ci) * out + c)];
                }
          y.x[static_cast<std::size_t>(((t * y.h + h) * y.w + w) * out + c)] = sum;
        }
  return y;
}

std::vector<float> Fill(std::size_t n, double scale, double frequency, double phase, bool cosine) {
  std::vector<float> v(n);
  for (std::size_t i = 0; i < n; ++i) {
    const double a = frequency * static_cast<double>(i) + phase;
    v[i]           = static_cast<float>(scale * (cosine ? std::cos(a) : std::sin(a)));
  }
  return v;
}

void FactorChecks() {
  int    total         = 0;
  double worst_forward = 0, worst_derivative = 0;
  for (int variant = 0; variant < 4; ++variant) {
    const bool r2   = variant == 0;
    const int  side = r2 ? 4 : 3, mid = r2 ? fv::FactorMidplanes(2, 2) : 2, stride = r2 ? 2 : 1;
    const auto kind    = static_cast<fv::P3dKind>(r2 ? 0 : variant - 1);
    const auto input   = Fill(static_cast<std::size_t>(side * side * side * 2), .3, .37, .2, false);
    const auto spatial = Fill(static_cast<std::size_t>(9 * 2 * mid), .13, .23, .4, true);
    const auto temporal = Fill(static_cast<std::size_t>(3 * mid * 2), .17, .41, .9, false);
    std::vector<std::vector<double>> values(3);
    values[0].assign(spatial.begin(), spatial.end());
    values[1].assign(temporal.begin(), temporal.end());
    values[2].assign(input.begin(), input.end());
    Tensor x = nn::from_vector(input, {1, side, side, side, 2}),
           a = nn::from_vector(spatial, {1, 3, 3, 2, mid}),
           b = nn::from_vector(temporal, {3, 1, 1, mid, 2});
    for (Tensor* t : {&x, &a, &b})
      t->requires_grad_(true);
    const auto sf = [&](const Tensor& v) {
      return nn::relu(fv::FactorConv(v, a, 1, stride, stride));
    };
    const auto tfun = [&](const Tensor& v) {
      const Tensor z = fv::FactorConv(v, b, stride, 1, 1);
      return kind == fv::P3dKind::A ? nn::relu(z) : z;
    };
    const Tensor y       = fv::P3dMix(x, sf, tfun, kind);
    const int    outside = (side + stride - 1) / stride;
    const auto   coefficient =
        Fill(static_cast<std::size_t>(outside * outside * outside * 2), .2, .19, 0, true);
    const Tensor loss =
        nn::sum(y * nn::from_vector(coefficient, {1, outside, outside, outside, 2}));
    const auto g    = nn::grad(loss, {a, b, x});
    const auto host = [&](const std::vector<std::vector<double>>& v,
                          std::vector<double>*                    out    = nullptr,
                          bool                                    linear = false) {
      const HostVolume volume{side, side, side, 2, v[2]};
      auto             z = HostFactorConv(volume, v[0], 1, 3, 3, mid, 1, stride, stride);
      if (!linear)
        for (auto& q : z.x)
          q = std::max(0., q);
      auto t = HostFactorConv(kind == fv::P3dKind::B ? volume : z, v[1], 3, 1, 1, 2, stride, 1, 1);
      if (kind == fv::P3dKind::A) {
        if (!linear)
          for (auto& q : t.x)
            q = std::max(0., q);
      } else {
        for (std::size_t i = 0; i < t.x.size(); ++i)
          t.x[i] = std::max(0., (z.x[i] + t.x[i]) * (kind == fv::P3dKind::C ? .5 : 1));
      }
      double value = 0;
      for (std::size_t i = 0; i < t.x.size(); ++i)
        value += t.x[i] * coefficient[i];
      if (out)
        *out = t.x;
      return value;
    };
    std::vector<double> expected;
    host(values, &expected);
    const auto yv = y.to_vector<float>();
    check::Require(yv.size() == expected.size(), "factorized fixture output size");
    double forward = 0, derivative = 0;
    for (std::size_t i = 0; i < expected.size(); ++i)
      forward = std::max(forward, std::abs(expected[i] - yv[i]));
    int covered = 0;
    for (std::size_t v = 0; v < 3; ++v) {
      const auto gv = g[v].to_vector<float>();
      for (std::size_t i = 0; i < values[v].size(); ++i) {
        auto plus = values, minus = values;
        plus[v][i] += 1e-6;
        minus[v][i] -= 1e-6;
        derivative = std::max(derivative, std::abs((host(plus) - host(minus)) / 2e-6 - gv[i]));
        ++covered;
      }
    }
    check::Near(
        forward,
        0,
        1e-7,
        "every signed factorized output versus independent host convolution/topology");
    check::Near(
        derivative,
        0,
        1e-7,
        "ALL spatial/temporal/input coordinates independent host finite differences");
    if (r2) {
      std::vector<double> linear;
      host(values, &linear, true);
      double difference = 0;
      for (std::size_t i = 0; i < linear.size(); ++i)
        difference += std::abs(linear[i] - expected[i]);
      check::Require(
          difference > .001,
          "intermediate ReLU makes R(2+1)D more than a collapsed linear3D kernel");
    }
    total += covered;
    worst_forward    = std::max(worst_forward, forward);
    worst_derivative = std::max(worst_derivative, derivative);
    std::cout << (r2             ? "R(2+1)D stride2"
                  : variant == 1 ? "P3D-A"
                  : variant == 2 ? "P3D-B"
                                 : "P3D-C")
              << " signed fixture:all" << covered
              << " factor/input FD coordinates,forward=" << forward << ",gradient=" << derivative
              << std::endl;
  }
  check::Require(total == 530, "224R2+3x102P3D coordinates covered");
  check::Require(
      fv::FactorMidplanes(64, 64) == 144 && fv::FactorMidplanes(64, 128) == 230,
      "independent parameter-balance widths including floor");
  std::cout << "all530 factor/kernel/input derivatives passed;explicit symmetric stride2 borders "
               "and nonlinear routing "
               "verified;worst="
            << worst_forward << "/" << worst_derivative << std::endl;
}

/// Endpoint shapes from the padding rules alone (explicit k/2 borders; P3D's -inf ceil and temporal
/// pools).
std::vector<nn::Shape> ShapeTrace(bool p3d, int base, int image) {
  const auto conv = [](std::int64_t in, std::int64_t k, std::int64_t s) {
    return nn::conv_output_size(in, k, s, 1, Padding::Explicit, k / 2, k / 2);
  };
  std::vector<nn::Shape> shapes;
  std::int64_t           t = 16, s = conv(image, 7, 2), c = base;
  if (!p3d) {
    shapes.push_back({1, t, s, s, c});
    for (int stage = 0; stage < 4; ++stage)
      for (int block = 0; block < 2; ++block) {
        const int stride = stage && block == 0 ? 2 : 1;
        s                = conv(s, 3, stride);
        t                = conv(t, 3, stride);
        c                = std::int64_t{base} << stage;
        shapes.push_back({1, t, s, s, c});
      }
  } else {
    t = nn::conv_output_size(t, 2, 2, 1, Padding::Valid);
    s = nn::conv_output_size(s + 1, 3, 2, 1, Padding::Valid);
    shapes.push_back({1, t, s, s, c});
    const int depth[] = {3, 8, 36, 3};
    for (int stage = 0; stage < 4; ++stage) {
      if (stage)
        t = nn::conv_output_size(t, 2, 2, 1, Padding::Valid);
      for (int block = 0; block < depth[stage]; ++block) {
        if (stage && block == 0)
          s = conv(s, 1, 2);
        c = 4 * (std::int64_t{base} << stage);
        shapes.push_back({1, t, s, s, c});
      }
    }
  }
  return shapes;
}

void GraphChecks(bool run_full) {
  for (bool p3d : {false, true})
    for (bool compact : {false, true}) {
      const int base = compact ? 4 : 64, classes = compact ? 3 : 400,
                reference = compact ? 32
                            : p3d   ? 160
                                    : 112;
      const int image =
          compact || run_full ? reference : 32; // full widths execute on a 16x32x32 clip by default
      const int depths[] = {3, 8, 36, 3};
      // The published endpoint table, at the executed size and at the reference size.
      const auto expected_endpoints = [&](int side) {
        std::vector<nn::Shape> shapes;
        shapes.push_back({1, p3d ? 8 : 16, side / (p3d ? 4 : 2), side / (p3d ? 4 : 2), base});
        for (int stage = 0; stage < 4; ++stage)
          for (int block = 0; block < (p3d ? depths[stage] : 2); ++block)
            shapes.push_back(
                {1,
                 p3d ? (8 >> stage) : (16 >> stage),
                 side / (p3d ? (4 << stage) : (2 << stage)),
                 side / (p3d ? (4 << stage) : (2 << stage)),
                 base * (p3d ? 4 : 1) * (1 << stage)});
        return shapes;
      };
      check::Require(
          ShapeTrace(p3d, base, reference) == expected_endpoints(reference),
          "published reference/compact residual endpoint shapes from the padding rules");
      nn::ParameterStore p(733), state(735);
      nn::Generator      gen(5);
      const Tensor       input = nn::uniform({1, 16, image, image, 3}, 0.0, 1.0, gen);
      const auto         net   = fv::FactorVideo(p, state, input, {p3d, base, classes});
      const auto         table = expected_endpoints(image);
      check::Require(
          net.stages.size() == table.size() &&
              net.stages.size() == static_cast<std::size_t>(p3d ? 51 : 9),
          "all50P3D/eightR2 residual endpoints plusstem");
      for (std::size_t i = 0; i < table.size(); ++i)
        check::Require(net.stages[i].shape() == table[i], "executed residual endpoint shape");
      check::Require(
          net.pooled.shape() == nn::Shape({1, base * (p3d ? 32 : 8)}) &&
              net.logits.shape() == nn::Shape({1, classes}),
          "pooled/logit shapes");
      std::int64_t expected = 0;
      int          units    = 0;
      const auto   count    = [&](int in, int out, int volume) {
        expected += std::int64_t{in} * out * volume + 2 * out;
        ++units;
      };
      int incoming = base;
      if (!p3d) {
        const int middle = std::max(1, 45 * base / 64);
        count(3, middle, 49);
        count(middle, base, 3);
        for (int stage = 0; stage < 4; ++stage)
          for (int block = 0; block < 2; ++block) {
            const int out = base << stage;
            const int mid =
                static_cast<int>(std::int64_t{27} * incoming * out / (9 * incoming + 3 * out));
            count(incoming, mid, 9);
            count(mid, out, 3);
            count(out, mid, 9);
            count(mid, out, 3);
            if (stage && block == 0)
              count(incoming, out, 1);
            incoming = out;
          }
      } else {
        count(3, base, 49);
        for (int stage = 0; stage < 4; ++stage)
          for (int block = 0; block < depths[stage]; ++block) {
            const int width = base << stage, out = 4 * width;
            count(incoming, width, 1);
            count(width, width, 9);
            if (stage < 3)
              count(width, width, 3);
            count(width, out, 1);
            if (incoming != out)
              count(incoming, out, 1);
            incoming = out;
          }
      }
      expected += std::int64_t{incoming} * classes + classes;
      check::Require(
          expected == p.count() && units == (p3d ? 202 : 37) &&
              p.size() == static_cast<std::size_t>(3 * units + 2) &&
              state.size() == static_cast<std::size_t>(2 * units),
          "independent per-kernel/learnedBN/head parameter inventory");
      if (!p3d && !compact)
        check::Require(expected == 31505325, "Torchvision published R(2+1)D-18 parameter count");
      if (p3d && !compact)
        check::Require(
            expected == 66492240,
            "independently parsed authors Caffe Kinetics P3D-199 parameter count");
      const auto gradients = nn::grad(nn::mean(nn::square(net.logits)), p.parameters());
      check::Require(
          gradients.size() == p.size(), "all full/compact factorized-video learned gradients");
      for (const auto& g : gradients) {
        check::Require(g.defined(), "no missing factorized video parameter gradient");
        check::Require(
            nn::all(nn::is_finite(g)).item<double>() != 0, "finite factorized video gradient");
      }
      std::cout << (p3d ? "P3D-199" : "R(2+1)D-18") << (compact ? " compact" : " full reference")
                << " graph:" << expected << " learned parameters/" << p.size() << " learned+"
                << state.size() << " state tensors;all" << table.size()
                << " endpoint shapes/count/all gradients passed;"
                << (compact ? "executed 16x32x32"
                    : image == reference
                        ? "full width executed at reference size"
                        : "full width executed on 16x32x32 (reference-size table from padding "
                          "rules; full=1 executes it)")
                << "\n";
    }
}

/// Loads TF's trained compact arrays (DIR/{r2plus1d,p3d}/model_arrays.npz) into our
/// models and replays TF's saved test clips: an exact cross-framework forward check.
void Replay(const std::filesystem::path& dir) {
  for (bool p3d : {false, true}) {
    const auto run = dir / (p3d ? "p3d" : "r2plus1d");
    if (!std::filesystem::exists(run / "model_arrays.npz")) {
      std::cout << "replay: no " << run.string() << "/model_arrays.npz, skipped\n";
      continue;
    }
    double error = 0;
    std::cout << (p3d ? "P3D-199" : "R(2+1)D-18") << " compact replay of TF arrays " << run.string()
              << ":\n";
    if (p3d) {
      fv::P3dModel model;
      error = c3d::ReplayTfVideo(run, model);
    } else {
      fv::R2Plus1dModel model;
      error = c3d::ReplayTfVideo(run, model);
    }
    check::Near(error, 0, 1e-5, "TF-trained arrays reproduce TF test/reverse/repeat logits");
  }
}

} // namespace

int main(int argc, char** argv) {
  try {
    int                   epochs = 3;
    bool                  checks = false, full = false;
    std::filesystem::path root = paths::Dataset("action_video"), replay;
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
        throw std::invalid_argument(
            "expected epochs=N data=PATH checks_only=0|1 full=0|1 replay=DIR threads=N");
    }
    FactorChecks();
    GraphChecks(full);
    if (!replay.empty())
      Replay(replay);
    if (!checks) {
      TrainR2Plus1d(root, epochs);
      TrainP3d(root, epochs);
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
