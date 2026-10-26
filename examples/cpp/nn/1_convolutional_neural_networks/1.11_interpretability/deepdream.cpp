// CMD: just run-nn deepdream
// DeepDream amplifies frozen learned feature activations in existing digit images.
// Three short octaves20/24/28,5steps each by default;zero classifier training.
//
// Port of tflow's deepdream. Prepare: run-nn vanilla_saliency (writes
// output/nn/models/vanilla_saliency), or pass source=tflow to use tflow's recorded
// classifier (then the dream trajectories are comparable with tflow's dream.npz).
// Data: saliency_data.npz (paths::Dataset("saliency"), data=PATH). Ascent and
// octave resize: common/dream.hpp. Only the convolutional part of the classifier
// is evaluated (as TF's graph prunes the dense head), so 20x20/24x24 octaves work.
// Outputs: output/nn/models/deepdream/.
// Options:steps=N,rate=F,source=PATH|tflow,data=PATH,out=PATH,threads=N.

#include <datamunge/nn/nn.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "common/checks.hpp"
#include "common/dream.hpp"
#include "common/paths.hpp"
#include "common/saliency.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;

namespace {

/// Mean of ALL post-ReLU conv1 (layer 0) or conv2 (layer 1) activations and its ordinary input derivative.
dream::DreamEvaluation Evaluate(sal::SaliencyModel& model, const Tensor& image, int layer) {
  Tensor x = image.detach().clone();
  x.requires_grad_(true);
  const auto   f = sal::SaliencyConvFeatures(model.parameters, x);
  const Tensor objective = nn::mean(layer == 0 ? f.conv1 : f.conv2);
  const Tensor g = nn::grad(objective, {x})[0].detach().contiguous();
  return {objective.item<double>(), g};
}

void Checks() {
  auto evaluate = [](const Tensor& point) {
    Tensor x = point.detach().clone();
    x.requires_grad_(true);
    const Tensor objective = nn::neg(nn::mean(nn::square(nn::sub(x, double(.7f)))));
    return dream::DreamEvaluation{objective.item<double>(), nn::grad(objective, {x})[0].detach().contiguous()};
  };
  Tensor     point = nn::full({1, 2, 2, 1}, double(.6f));
  const auto step = dream::DreamAscent(point, 2., evaluate);
  check::Require(step.backtracks == 4 && step.after > step.before, "overshooting ascent backtracks to genuine increase");
  const auto g = sal::F(step.gradient), im = sal::F(step.image);
  for (int i = 0; i < 4; ++i) {
    check::Near(g[i], (.7f - .6f) / 2, 1e-8, "all analytic quadratic gradients");
    check::Near(im[i], .725, 1e-7, "all RMS normalized accepted pixels");
  }
  const auto zero = dream::DreamAscent(point, 0, evaluate);
  check::Require(sal::Bytes(zero.image) == sal::Bytes(point) && zero.step == 0, "zero rate preserves every input bit");
  point = nn::full({1, 2, 2, 1}, double(.7f));
  const auto stationary = dream::DreamAscent(point, .02, evaluate);
  check::Require(stationary.rms == 0 && sal::Bytes(stationary.image) == sal::Bytes(point), "zero gradient safe/no division by zero");
  const Tensor corners = sal::FromF({0, 1, .5f, .25f}, {1, 2, 2, 1});
  const auto   resized = sal::F(dream::DreamResize(corners, 3, 3));
  const double expected[] = {0, .5, 1, .25, .4375, .625, .5, .375, .25};
  for (int i = 0; i < 9; ++i) check::Near(resized[i], expected[i], 0, "all half-pixel bilinear octave pixels");
  check::Require(sal::Bytes(dream::DreamResize(corners, 2, 2)) == sal::Bytes(corners), "identity resize exact");
  check::Throws([&] { dream::DreamAscent(point, -1, evaluate); }, "negative input rate rejected");
  std::cout
      << "analytic quadratic gradients/RMS pixels/4backtracks/zero-rate/zero-gradient/all9half-pixel resize/identity/invalidrate passed\n";
}

struct DreamResult {
  Tensor image;
  double first = 0, last = 0;
};

DreamResult Optimize(sal::SaliencyModel& model, const Tensor& input, const std::vector<int>& sizes, int steps, double rate, int layer,
                     const std::string& name, sal::Arrays& arrays, std::ofstream& history) {
  Tensor      current = input;
  DreamResult result;
  bool        first = true;
  for (std::size_t octave = 0; octave < sizes.size(); ++octave) {
    current = dream::DreamResize(current, sizes[octave], sizes[octave]);
    const std::string prefix = name + "_o" + std::to_string(octave);
    arrays.emplace(prefix + "_start", current);
    const double start = Evaluate(model, current, layer).score;
    if (first) {
      result.first = start;
      first = false;
    }
    for (int k = 0; k < steps; ++k) {
      const auto step = dream::DreamAscent(current, rate, [&](const Tensor& x) { return Evaluate(model, x, layer); });
      check::Require(step.after >= step.before, "accepted input steps never lower objective within an octave");
      const std::string key = prefix + "_s" + std::to_string(k);
      arrays.emplace(key + "_gradient", step.gradient);
      arrays.emplace(key + "_image", step.image);
      arrays.emplace(key + "_before", sal::Scalar(step.before));
      arrays.emplace(key + "_after", sal::Scalar(step.after));
      arrays.emplace(key + "_rms", sal::Scalar(step.rms));
      arrays.emplace(key + "_rate", sal::Scalar(step.step));
      arrays.emplace(key + "_backtracks", sal::Scalar(step.backtracks));
      history << name << '\t' << octave << '\t' << sizes[octave] << '\t' << k << '\t' << step.before << '\t' << step.after << '\t'
              << step.rms << '\t' << step.step << '\t' << step.backtracks << '\n';
      current = step.image;
    }
    const double end = Evaluate(model, current, layer).score;
    std::cout << name << " octave" << sizes[octave] << ":" << start << " -> " << end << " (" << steps << "input steps)\n";
    check::Require(rate == 0 || end > start, "nonzero short demonstration genuinely amplifies selected learned features");
  }
  result.image = current;
  result.last = Evaluate(model, current, layer).score;
  arrays.emplace(name + "_final", current);
  return result;
}

void Run(const std::filesystem::path& source, const std::filesystem::path& data, const std::filesystem::path& directory, int steps,
         double rate) {
  const auto  pools = ae::LoadSaliency(data);
  const auto& test = pools[2];
  const auto  examples = sal::Pick(test, sal::FirstPerDigit(test));
  sal::SaliencyModel model(false), initial(false);
  const auto         w = sal::RestoreSource(source, model, initial);  // verified three-epoch classifier + actual initial tensors
  const auto         before = model.Snapshot(), initial_before = initial.Snapshot();
  std::filesystem::create_directories(directory);
  sal::Arrays arrays{{"images", examples.images},
                     {"labels", examples.labels},
                     {"source_ids", examples.source},
                     {"steps", sal::Scalar(steps)},
                     {"rate", sal::Scalar(rate)}};
  std::ofstream history(directory / "history.tsv");
  history.precision(17);
  history << "variant\toctave\tsize\tstep\tbefore\tafter\trms\taccepted_rate\tbacktracks\n";
  const auto main = Optimize(model, examples.images, {20, 24, 28}, steps, rate, 1, "main", arrays, history);
  const auto shallow = Optimize(model, examples.images, {20, 24, 28}, steps, rate, 0, "conv1", arrays, history);
  const auto random = Optimize(initial, examples.images, {20, 24, 28}, steps, rate, 1, "initial", arrays, history);
  const auto single = Optimize(model, examples.images, {28}, steps * 3, rate, 1, "single", arrays, history);
  const auto zero = Optimize(model, examples.images, {20, 24, 28}, steps, 0, 1, "zero", arrays, history);
  Tensor     resampled = examples.images;
  for (int size : {20, 24, 28}) resampled = dream::DreamResize(resampled, size, size);
  check::Require(sal::Bytes(zero.image) == sal::Bytes(resampled),
                 "zero-rate multi-octave result equals resize-only control,not falsely original input");
  sal::Arrays   repeat_arrays;
  std::ofstream sink(directory / "repeat.tsv");
  const auto    repeat = Optimize(model, examples.images, {20, 24, 28}, steps, rate, 1, "repeat", repeat_arrays, sink);
  check::Require(sal::Bytes(main.image) == sal::Bytes(repeat.image), "full octave optimization deterministic");
  const auto evaluation = Evaluate(model, examples.images, 1);
  const auto g = sal::F(evaluation.gradient), image = sal::F(examples.images);
  std::vector<int> ranked(7840);
  std::iota(ranked.begin(), ranked.end(), 0);
  std::stable_sort(ranked.begin(), ranked.end(), [&](int a, int b) { return std::abs(g[a]) > std::abs(g[b]); });
  double fd_error = 0;
  for (int k = 0; k < 8; ++k) {
    const int i = ranked[k];
    auto      plus = image, minus = image;
    plus[i] += .001f;
    minus[i] -= .001f;
    const double fd = (Evaluate(model, sal::FromF(plus, examples.images.shape()), 1).score -
                       Evaluate(model, sal::FromF(minus, examples.images.shape()), 1).score) /
                      .002;
    fd_error = std::max(fd_error, std::abs(fd - g[i]));
  }
  check::Require(fd_error < 5e-5, "eight actual feature-objective finite differences agree with input derivatives");
  check::Require(before == model.Snapshot() && initial_before == initial.Snapshot(),
                 "all model weights frozen during every dream/control/finite difference");
  sal::SaveArrays(directory / "dream.npz", arrays);
  sal::SaveArrays(directory / "model_arrays.npz", w.trained);
  sal::SaveArrays(directory / "initial_arrays.npz", w.initial);
  sal::CopyManifests(directory, {{data / "manifest.json", "data_manifest.json"},
                                 {paths::Dataset("dream") / "source_manifest.json", "source_manifest.json"}});
  std::ofstream html(directory / "index.html");
  html << "<!doctype html><meta charset='utf-8'><title>DeepDream on learned digit features</title><style>body{font:16px "
          "system-ui;max-width:1000px;margin:30px auto}img{max-width:100%;image-rendering:pixelated}section{margin:28px "
          "0}</style><h1>DeepDream: amplify features in existing images</h1><p>Left to right:original,conv2 mean dream,conv1 mean "
          "control,actual initial weights conv2,single octave conv2,zero-step resize-only control. All pixels share fixed[0,1] "
          "grayscale;nearest4 enlargement only. This is input optimization of ALL post-ReLU feature channels,not a target class or "
          "synthesized noise image. The frozen classifier was trained for3epochs;ZERO new training.</p><p>Half-pixel bilinear "
          "octaves20,24,28;"
       << steps << "steps per octave,RMS-normalized rate" << rate
       << " with projection to[0,1] and monotonic backtracking. Single octave has equal total input steps. Cross-octave scores need not be "
          "monotonic because resizing changes the objective. Zero-step resize control can differ from the original image.</p><p>Feature "
          "amplification is model-specific and can distort digit identity. These small digit textures are not photorealistic "
          "synthesis,causal explanations or classification improvements. Grayscale MNIST model differs from ImageNet RGB DeepDream.</p>";
  const std::vector<std::vector<float>> panels{image, sal::F(main.image), sal::F(shallow.image), sal::F(random.image),
                                               sal::F(single.image), sal::F(zero.image)};
  const auto src = sal::I(examples.source), labels = sal::I(examples.labels);
  for (int b = 0; b < 10; ++b) {
    sal::Canvas pixels(112, 692);
    for (std::size_t p = 0; p < panels.size(); ++p)
      for (int y = 0; y < 112; ++y)
        for (int x = 0; x < 112; ++x) pixels.Gray(y, int(p) * 116 + x, 255 * panels[p][b * 784 + (y / 4) * 28 + x / 4]);
    const std::string file = "case" + std::to_string(b) + ".png";
    sal::SaliencyPng(directory / file, pixels);
    html << "<section><h2>Source" << src[b] << ",true digit" << labels[b] << "</h2><img src='" << file << "'></section>";
  }
  check::Require(bool(html) && bool(history), "write dream gallery and complete step trace");
  std::cout << "DeepDream feature amplification,not model training;5variants and repeat completed,all2x8weights "
               "immutable,finiteDifferenceMaxError="
            << fd_error << "\nmain first/last DIFFERENT scale scores=" << main.first << '/' << main.last
            << ";compare only each octave start/end above\nArtifacts:" << directory.string() << "/index.html\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    nn::set_num_threads(4);
    int                   steps = 5;
    double                rate = .02;
    std::filesystem::path source = sal::DefaultSource(), data = paths::Dataset("saliency"), out = paths::OutputRoot() / "models/deepdream";
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a.rfind("steps=", 0) == 0)
        steps = std::stoi(a.substr(6));
      else if (a.rfind("rate=", 0) == 0)
        rate = std::stod(a.substr(5));
      else if (a.rfind("source=", 0) == 0)
        source = sal::SourceDir(a.substr(7));
      else if (a.rfind("data=", 0) == 0)
        data = a.substr(5);
      else if (a.rfind("out=", 0) == 0)
        out = a.substr(4);
      else if (a.rfind("threads=", 0) == 0)
        nn::set_num_threads(std::stoi(a.substr(8)));
      else
        throw std::invalid_argument("expected steps=N rate=F source=PATH|tflow data=PATH out=PATH threads=N");
    }
    check::Require(steps > 0 && steps <= 100 && rate >= 0 && std::isfinite(rate), "steps1..100 and finite nonnegative rate");
    Checks();
    Run(source, data, out, steps, rate);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
