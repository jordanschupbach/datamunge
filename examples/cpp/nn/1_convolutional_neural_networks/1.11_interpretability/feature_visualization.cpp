// CMD: just run-nn feature_visualization
// Synthesize selected convolution-channel preferences from seeded uniform noise.
// Frozen three-epoch model;12short input steps/default,L2/squared-TV priors explicit.
//
// Port of tflow's feature_visualization. Prepare: run-nn vanilla_saliency (writes
// output/nn/models/vanilla_saliency), or pass source=tflow to use tflow's recorded
// classifier (then every trajectory is comparable with tflow's features.npz; the
// noise images are bit-identical). Objective: common/feature_visualization.hpp,
// ascent: common/dream.hpp. Outputs: output/nn/models/feature_visualization/.
// Options:steps=N,rate=F,source=PATH|tflow,out=PATH,threads=N.

#include <datamunge/nn/nn.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "common/checks.hpp"
#include "common/dream.hpp"
#include "common/feature_visualization.hpp"
#include "common/paths.hpp"
#include "common/saliency.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;

namespace {

struct Evaluation {
  double             score, feature, l2, tv;
  Tensor             gradient;
  std::vector<float> responses;  // [B, 16] spatial means of every conv2 channel
};

/// Selected post-ReLU conv2 channel mean and regularized input derivative.
Evaluation Evaluate(sal::SaliencyModel& model, const Tensor& image, int channel, float a, float b) {
  check::Require(channel >= 0 && channel < 16, "selected channel0..15");
  Tensor x = image.detach().clone();
  x.requires_grad_(true);
  const auto conv2 = sal::SaliencyConvFeatures(model.parameters, x).conv2;
  const auto objective =
      featurevis::VisualizeFeature(x, conv2, sal::FromI(std::vector<int>(std::size_t(image.size(0)), channel)), 16, a, b);
  const Tensor g = nn::grad(objective.mean, {x})[0].detach().contiguous();
  nn::NoGradGuard guard;
  return {objective.mean.item<double>(), sal::F(objective.feature)[0], sal::F(objective.l2)[0], sal::F(objective.tv)[0], g,
          sal::F(nn::mean(conv2, {1, 2}))};
}

Tensor Row(const Tensor& x, int b) { return nn::slice_axis(x, 0, b, b + 1).contiguous(); }

void Checks() {
  const Tensor noise = featurevis::FeatureNoise(1, 1433);
  Tensor       image = noise.clone();
  image.requires_grad_(true);
  const Tensor features = nn::concat({image, nn::mul(image, 2.0)}, 3);
  const auto   objective = featurevis::VisualizeFeature(image, features, sal::FromI({1}), 2, .3f, .7f);
  const auto   gradient_out = sal::F(nn::grad(objective.mean, {image})[0]);
  const auto   x = sal::F(noise);
  double       activation = 0, l2 = 0, tv = 0;
  std::vector<double> gradient(784, 2. / 784);
  for (int y = 0; y < 28; ++y)
    for (int a = 0; a < 28; ++a) {
      const int    i = y * 28 + a;
      const double v = x[i];
      activation += 2 * v / 784;
      l2 += (v - .5) * (v - .5) / 784;
      gradient[i] -= 2 * double(.3f) * (v - .5) / 784;
      for (int j : {y < 27 ? i + 28 : -1, a < 27 ? i + 1 : -1})
        if (j >= 0) {
          const double difference = v - x[j];
          tv += difference * difference / 756;
          gradient[i] -= 2 * double(.7f) * difference / 756;
          gradient[j] += 2 * double(.7f) * difference / 756;
        }
    }
  check::Near(objective.mean.item<double>(), activation - double(.3f) * l2 - double(.7f) * tv, 3e-7,
              "known full objective includes separate normalized priors");
  check::Near(sal::F(objective.feature)[0], activation, 2e-7, "selected channel rather than all-channel average");
  check::Near(sal::F(objective.l2)[0], l2, 2e-8, "all centered L2 pixels");
  check::Near(sal::F(objective.tv)[0], tv, 1e-7, "all1512squared TV edges");
  for (int i = 0; i < 784; ++i)
    check::Near(gradient_out[i], gradient[i], 2e-9, "ALL784analytic selected-feature/L2/TV derivatives including boundaries");
  check::Require(sal::Bytes(noise) == sal::Bytes(featurevis::FeatureNoise(1, 1433)) &&
                     sal::Bytes(noise) != sal::Bytes(featurevis::FeatureNoise(1, 1434)),
                 "seeded noise exact repeat and secondseed difference");
  std::cout << "known selectedchannel objective/L2/1512TVedges/ALL784gradient boundaries and seeded noise fixtures passed\n";
}

struct Result {
  std::vector<float>  images, gradients, initial_responses, final_responses;
  std::vector<double> terms, rms, rates;
  std::vector<int>    backtracks;
};

Result Optimize(sal::SaliencyModel& model, const Tensor& start, int steps, double rate, float l2, float tv, const std::string& name,
                std::ofstream& history) {
  Result result;
  result.images.resize(std::size_t(steps + 1) * 16 * 784);
  result.gradients.resize(std::size_t(steps) * 16 * 784);
  result.terms.resize(std::size_t(steps + 1) * 16 * 4);
  result.rms.resize(std::size_t(steps) * 16);
  result.rates.resize(std::size_t(steps) * 16);
  result.backtracks.resize(std::size_t(steps) * 16);
  result.initial_responses.resize(256);
  result.final_responses.resize(256);
  int    improved = 0, feature_improved = 0;
  double objective_gain = 0;
  for (int channel = 0; channel < 16; ++channel) {
    Tensor current = Row(start, channel);
    double first = 0, first_feature = 0;
    auto   record = [&](int k) {
      auto         value = Evaluate(model, current, channel, l2, tv);
      const double components[] = {value.score, value.feature, value.l2, value.tv};
      const auto   pixels = sal::F(current);
      std::copy_n(pixels.data(), 784, result.images.data() + (k * 16 + channel) * 784);
      for (int j = 0; j < 4; ++j) result.terms[(k * 16 + channel) * 4 + j] = components[j];
      if (k == 0) {
        first = value.score;
        first_feature = value.feature;
        std::copy_n(value.responses.data(), 16, result.initial_responses.data() + channel * 16);
      }
      if (k == steps) {
        std::copy_n(value.responses.data(), 16, result.final_responses.data() + channel * 16);
        objective_gain += value.score - first;
        improved += value.score > first;
        feature_improved += value.feature > first_feature;
      }
      return value;
    };
    record(0);
    for (int k = 0; k < steps; ++k) {
      const auto update = dream::DreamAscent(current, rate, [&](const Tensor& x) {
        auto e = Evaluate(model, x, channel, l2, tv);
        return dream::DreamEvaluation{e.score, e.gradient};
      });
      check::Require(update.after >= update.before, "per-channel complete regularized objective never decreases");
      const auto g = sal::F(update.gradient);
      std::copy_n(g.data(), 784, result.gradients.data() + (k * 16 + channel) * 784);
      result.rms[k * 16 + channel] = update.rms;
      result.rates[k * 16 + channel] = update.step;
      result.backtracks[k * 16 + channel] = update.backtracks;
      current = update.image;
      const auto e = record(k + 1);
      history << name << '\t' << channel << '\t' << k << '\t' << update.before << '\t' << update.after << '\t' << e.feature << '\t' << e.l2
              << '\t' << e.tv << '\t' << update.rms << '\t' << update.step << '\t' << update.backtracks << '\n';
    }
  }
  check::Require(rate == 0 || objective_gain > 0, "actual objective gain across selected features");
  std::cout << name << ":objective-improved channels=" << improved << "/16,raw-feature-improved=" << feature_improved
            << "/16,mean objective gain=" << objective_gain / 16 << "\n";
  return result;
}

void Run(const std::filesystem::path& source, const std::filesystem::path& directory, int steps, double rate) {
  sal::SaliencyModel model(false), initial(false);
  const auto         w = sal::RestoreSource(source, model, initial);  // all trained/actual initial source bytes match
  const auto         before = model.Snapshot(), initial_before = initial.Snapshot();
  const Tensor       noise = featurevis::FeatureNoise(16, 1423), noise2 = featurevis::FeatureNoise(16, 1429);
  std::filesystem::create_directories(directory);
  std::ofstream history(directory / "history.tsv");
  history.precision(17);
  history << "variant\tchannel\tstep\tbefore\tafter\tfeature\tl2\ttv\trms\trate\tbacktracks\n";
  std::map<std::string, Result> groups;
  groups.emplace("main", Optimize(model, noise, steps, rate, .01f, .2f, "main", history));
  groups.emplace("unregularized", Optimize(model, noise, steps, rate, 0, 0, "unregularized", history));
  groups.emplace("initial", Optimize(initial, noise, steps, rate, .01f, .2f, "initial", history));
  groups.emplace("second_seed", Optimize(model, noise2, steps, rate, .01f, .2f, "second_seed", history));
  groups.emplace("zero", Optimize(model, noise, steps, 0, .01f, .2f, "zero", history));
  std::ofstream sink(directory / "repeat.tsv");
  const auto    repeat = Optimize(model, noise, steps, rate, .01f, .2f, "repeat", sink);
  check::Require(repeat.images == groups.at("main").images, "all per-channel optimization trajectories repeat bitwise");
  const auto& zero = groups.at("zero");
  const auto  noise_v = sal::F(noise);
  for (int k = 0; k <= steps; ++k)
    check::Require(std::equal(noise_v.begin(), noise_v.end(), zero.images.begin() + std::size_t(k) * 16 * 784),
                   "zero step every channel/iteration leaves original noise exact");
  const auto single = Evaluate(model, Row(noise, 0), 0, .01f, .2f);
  const auto doubled = nn::concat({Row(noise, 0), Row(noise, 0)}, 0).contiguous();
  const auto batch = sal::F(Evaluate(model, doubled, 0, .01f, .2f).gradient), sg = sal::F(single.gradient);
  for (int b = 0; b < 2; ++b)
    for (int i = 0; i < 784; ++i)
      check::Near(batch[b * 784 + i] * 2, sg[i], 2e-8, "batch objective mean gives exact1/B derivative factor");
  int chosen = 0;
  for (int i = 1; i < 784; ++i)
    if (std::abs(sg[i]) > std::abs(sg[chosen])) chosen = i;
  auto plus = sal::F(Row(noise, 0)), minus = plus;
  plus[chosen] += .001f;
  minus[chosen] -= .001f;
  const double fd = (Evaluate(model, sal::FromF(plus, {1, 28, 28, 1}), 0, .01f, .2f).score -
                     Evaluate(model, sal::FromF(minus, {1, 28, 28, 1}), 0, .01f, .2f).score) /
                    .002;
  check::Near(fd, sg[chosen], 2e-4, "real full regularized selected-feature finite difference");
  check::Require(before == model.Snapshot() && initial_before == initial.Snapshot(), "all2x8model parameter tensors immutable");
  sal::Arrays arrays{{"noise", noise},
                     {"noise_second", noise2},
                     {"steps", sal::Scalar(steps)},
                     {"rate", sal::Scalar(rate)},
                     {"seed", sal::Scalar(1423)},
                     {"second_seed", sal::Scalar(1429)},
                     {"l2_weight", sal::Scalar(double(.01f))},
                     {"tv_weight", sal::Scalar(double(.2f))}};
  for (const auto& [name, g] : groups) {
    arrays[name + "_images"] = sal::FromF(g.images, {steps + 1, 16, 28, 28, 1});
    arrays[name + "_gradients"] = sal::FromF(g.gradients, {steps, 16, 28, 28, 1});
    arrays[name + "_terms"] = nn::from_vector(g.terms, {steps + 1, 16, 4});
    arrays[name + "_rms"] = nn::from_vector(g.rms, {steps, 16});
    arrays[name + "_rates"] = nn::from_vector(g.rates, {steps, 16});
    arrays[name + "_backtracks"] = nn::from_vector(g.backtracks, {steps, 16});
    arrays[name + "_initial_responses"] = sal::FromF(g.initial_responses, {16, 16});
    arrays[name + "_final_responses"] = sal::FromF(g.final_responses, {16, 16});
  }
  sal::SaveArrays(directory / "features.npz", arrays);
  sal::SaveArrays(directory / "model_arrays.npz", w.trained);
  sal::SaveArrays(directory / "initial_arrays.npz", w.initial);
  sal::CopyManifests(directory, {{source / "data_manifest.json", "data_manifest.json"},
                                 {paths::Dataset("feature_visualization") / "source_manifest.json", "source_manifest.json"}});
  int         target_strongest = 0;
  const auto& responses = groups.at("main").final_responses;
  for (int c = 0; c < 16; ++c) {
    int best = 0;
    for (int j = 1; j < 16; ++j)
      if (responses[c * 16 + j] > responses[c * 16 + best]) best = j;
    target_strongest += best == c;
  }
  std::cout << "Main selected target is strongest raw channel response in " << target_strongest
            << "/16 images;maximizing one channel does not suppress other channels.\n";
  std::string inactive;
  for (int c = 0; c < 16; ++c)
    if (groups.at("initial").initial_responses[c * 16 + c] == 0 && groups.at("initial").final_responses[c * 16 + c] == 0)
      inactive += (inactive.empty() ? "" : ",") + std::to_string(c);
  if (inactive.empty()) inactive = "none";
  std::ofstream html(directory / "index.html");
  html << "<!doctype html><meta charset='utf-8'><title>Selected channel visualization</title><style>body{font:16px "
          "system-ui;max-width:1000px;margin:30px auto}img{max-width:100%;image-rendering:pixelated}section{margin:25px "
          "0}</style><h1>Feature visualization by input optimization</h1><p>Every conv2 channel0..15 is shown without selection by "
          "response. Independent28x28 uniform noise per channel,seed1423. Left to right:starting noise,regularized trained,unregularized "
          "trained,actual initial weights,second seed1429,zero-step control. All use fixed[0,1]grayscale,nearest4 enlargement. Second seed "
          "starts from a DIFFERENT saved noise image.</p><p>Target is spatial mean of ONE post-ReLU conv2 channel,not a digit class or ALL "
          "channels. Maximize feature minus0.01*mean((x-.5)^2) minus0.2*(mean vertical squared differences+mean horizontal squared "
          "differences). Each channel optimized separately with"
       << steps << " RMS-normalized bounded input steps,rate" << rate
       << ",monotonic backtracking of the COMPLETE objective. A regularized objective gain can come from smaller penalties even when a raw "
          "feature is inactive;all objective components and16x16response matrices are saved.</p><p>ZERO new classifier training;reuses "
          "three-epoch MNIST weights. These are short-run local model preferences,not global optima,semantic detectors,causal explanations "
          "or natural digit synthesis. Priors,seed,weights and selected channel affect results. No per-image contrast stretching or "
          "winning-seed selection.</p><p>Selected target is the strongest raw channel response in "
       << target_strongest
       << "/16 main final images. This objective does not suppress other channels;raw response scales differ. Initial-model channels "
          "inactive at both endpoints: "
       << inactive << ". For an inactive channel,objective improvement need not mean feature amplification.</p>";
  const std::vector<std::string> names{"main", "unregularized", "initial", "second_seed", "zero"};
  for (int c = 0; c < 16; ++c) {
    sal::Canvas pixels(112, 692);
    for (int y = 0; y < 112; ++y)
      for (int x = 0; x < 112; ++x) {
        const int i = (y / 4) * 28 + x / 4;
        pixels.Gray(y, x, 255 * noise_v[c * 784 + i]);
        for (int p = 0; p < 5; ++p)
          pixels.Gray(y, (p + 1) * 116 + x, 255 * groups.at(names[p]).images[(std::size_t(steps) * 16 + c) * 784 + i]);
      }
    const std::string file = "channel" + std::to_string(c) + ".png";
    sal::SaliencyPng(directory / file, pixels);
    const auto& terms = groups.at("main").terms;
    html << "<section><h2>Conv2 channel" << c << "</h2><p>Complete objective:" << terms[c * 4] << " -> " << terms[(steps * 16 + c) * 4]
         << ";raw feature:" << terms[c * 4 + 1] << " -> " << terms[(steps * 16 + c) * 4 + 1]
         << ". L2/TV and all-channel responses in features.npz.</p><img src='" << file << "'></section>";
  }
  check::Require(bool(html) && bool(history), "write full gallery and step history");
  std::cout << "16selected channels/no cherry-picking;5variants+bitwise repeat,zero rate,1/Bbatchgradient,real FDerror="
            << std::abs(fd - sg[chosen]) << ",immutable2x8weights passed\nArtifacts:" << directory.string() << "/index.html\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    nn::set_num_threads(4);
    int                   steps = 12;
    double                rate = .05;
    std::filesystem::path source = sal::DefaultSource(), out = paths::OutputRoot() / "models/feature_visualization";
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a.rfind("steps=", 0) == 0)
        steps = std::stoi(a.substr(6));
      else if (a.rfind("rate=", 0) == 0)
        rate = std::stod(a.substr(5));
      else if (a.rfind("source=", 0) == 0)
        source = sal::SourceDir(a.substr(7));
      else if (a.rfind("out=", 0) == 0)
        out = a.substr(4);
      else if (a.rfind("threads=", 0) == 0)
        nn::set_num_threads(std::stoi(a.substr(8)));
      else
        throw std::invalid_argument("expected steps=N rate=F source=PATH|tflow out=PATH threads=N");
    }
    check::Require(steps > 0 && steps <= 100 && rate >= 0 && std::isfinite(rate), "steps1..100 and finite nonnegative rate");
    Checks();
    Run(source, out, steps, rate);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
