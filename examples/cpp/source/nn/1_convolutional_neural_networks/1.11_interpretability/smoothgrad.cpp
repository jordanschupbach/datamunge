// CMD: just run-nn smoothgrad
// Signed SmoothGrad mean,explicit seeded Gaussian noise/no clipping by default.
// Zero/noise-scale/seed/clipping/target/initial controls;0newtraining.
//
// Port of tflow's smoothgrad. Prepare: run-nn vanilla_saliency (writes
// output/nn/models/vanilla_saliency), or pass source=tflow to use tflow's recorded
// classifier (then every map is comparable with tflow's smoothgrad.npz; the noise
// banks are bit-identical). Data: saliency_data.npz (paths::Dataset("saliency"),
// data=PATH). Estimator: common/smoothgrad.hpp. Outputs: output/nn/models/smoothgrad/.
// Options:samples=N,sigma=S,source=PATH|tflow,data=PATH,out=PATH,threads=N.

#include <datamunge/nn/nn.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "common/checks.hpp"
#include "common/paths.hpp"
#include "common/saliency.hpp"
#include "common/smoothgrad.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;

namespace {

void EstimatorChecks() {
  const Tensor c = sal::FromF({1.f, -2.f, 0.f}, {3});
  auto         callback = [](std::function<Tensor(const Tensor&)> f) {
    return [f](const Tensor& points, const Tensor&) {
      Tensor x = points.detach().clone();
      x.requires_grad_(true);
      return nn::grad(f(x), {x})[0].detach().contiguous();
    };
  };
  auto linear = [&](const Tensor& x) { return nn::sum(nn::mul(x, c)); };
  auto quadratic = [&](const Tensor& x) { return nn::sum(nn::mul(nn::mul(x, x), nn::mul(c, .5))); };
  const Tensor input = sal::FromF({.1f, .3f, .5f, -.2f, .7f, 1.2f}, {2, 3}), targets = sal::FromI({0, 0});
  const auto   noise = smooth::GaussianNoise(input.shape(), 32, 1171), same = smooth::GaussianNoise(input.shape(), 32, 1171);
  check::Require(sal::Bytes(noise) == sal::Bytes(same) && noise.to_vector<double>() == same.to_vector<double>(), "exactseeded noise replay");
  const auto constant = smooth::SmoothGrad(input, targets, noise, .5f, false, 7, callback(linear)),
             q = smooth::SmoothGrad(input, targets, noise, .5f, false, 1, callback(quadratic)),
             qbatch = smooth::SmoothGrad(input, targets, noise, .5f, false, 11, callback(quadratic)),
             zero = smooth::SmoothGrad(input, targets, noise, 0, false, 5, callback(quadratic));
  const auto cm = sal::F(constant.mean), cv = sal::F(constant.sample_variance), qm = sal::F(q.mean), qn = sal::F(q.noisy_input),
             qb = sal::F(qbatch.mean), zm = sal::F(zero.mean), qa = sal::F(q.mean_absolute), x = sal::F(input);
  double difference = 0;
  for (int i = 0; i < 6; ++i) {
    const float coefficient = std::array<float, 3>{1, -2, 0}[i % 3];
    check::Near(cm[i], coefficient, 0, "linear SmoothGrad invariant to noise");
    check::Near(cv[i], 0, 0, "constant-gradient variancezero");
    double expected = 0;
    for (int k = 0; k < 32; ++k) expected += coefficient * qn[k * 6 + i] / 32.;
    check::Near(qm[i], expected, 2e-7, "quadratic exact FINITE-noise-bank mean,not assuming noise mean0");
    check::Near(qb[i], qm[i], 0, "noise-point batching invariance");
    check::Near(zm[i], coefficient * x[i], 0, "sigma0recovers ordinary derivative");
    difference += qa[i] - std::abs(qm[i]);
  }
  check::Require(difference > 1e-3, "meanabs differs from absmean;no silent magnitude variant");
  std::cout << "SmoothGrad linear/quadratic finitebank/sigma0/seed/batching/dummy/variance tests passed;meanabs-vsabsmean difference="
            << difference << "\n";
}

struct Group {
  smooth::SmoothGradResult result;
  float                    sigma;
  bool                     clipped;
  int                      seed;
};

Group Compute(sal::SaliencyModel& model, const sal::SaliencyData& examples, const Tensor& classes, int samples, float sigma, bool clip,
              int seed) {
  const auto noise = smooth::GaussianNoise(examples.images.shape(), samples, std::uint32_t(seed));
  return {smooth::SmoothGrad(examples.images, classes, noise, sigma, clip, 32,
                             [&](const Tensor& points, const Tensor& target) { return model.Gradient(points, target); }),
          sigma, clip, seed};
}

double Roughness(const std::vector<float>& map, int b) {
  double variation = 0, magnitude = 0;
  for (int y = 0; y < 28; ++y)
    for (int x = 0; x < 28; ++x) {
      const double value = map[b * 784 + y * 28 + x];
      magnitude += std::abs(value);
      if (y < 27) variation += std::abs(value - map[b * 784 + (y + 1) * 28 + x]);
      if (x < 27) variation += std::abs(value - map[b * 784 + y * 28 + x + 1]);
    }
  return magnitude > 0 ? variation / magnitude : 0;
}

void Run(const std::filesystem::path& source, const std::filesystem::path& data, const std::filesystem::path& directory, int samples,
         float sigma) {
  check::Require(samples >= 8 && samples <= 128 && std::isfinite(sigma) && sigma > 0 && sigma <= 1, "samples8..128 and sigma(0,1]");
  const auto  pools = ae::LoadSaliency(data);
  const auto& test = pools[2];
  sal::SaliencyModel model(false), initial(false);
  sal::RestoreSource(source, model, initial);  // same8trained/actualinitial weights
  const auto before = model.Snapshot(), initial_before = initial.Snapshot();
  const auto test_logits = model.Predict(test);
  const auto examples = sal::Pick(test, sal::FirstPerDigit(test));
  const auto logits = model.Predict(examples);
  const auto [targets, runners] = sal::TopTwo(logits);
  const auto ordinary_t = model.Gradient(examples, targets);
  const auto ordinary = sal::F(ordinary_t);
  std::map<std::string, Group> groups;
  groups.emplace("main", Compute(model, examples, targets, samples, sigma, false, 1171));
  groups.emplace("prefix", Compute(model, examples, targets, 8, sigma, false, 1171));
  groups.emplace("second_seed", Compute(model, examples, targets, samples, sigma, false, 1173));
  groups.emplace("large_sigma", Compute(model, examples, targets, samples, 2 * sigma, false, 1171));
  groups.emplace("clipped", Compute(model, examples, targets, samples, sigma, true, 1171));
  groups.emplace("runner", Compute(model, examples, runners, samples, sigma, false, 1171));
  groups.emplace("initial", Compute(initial, examples, targets, samples, sigma, false, 1171));
  groups.emplace("zero", Compute(model, examples, targets, 2, 0, false, 1171));
  const auto& main = groups.at("main").result;
  const auto  repeat = Compute(model, examples, targets, samples, sigma, false, 1171);
  auto        same = [](const Tensor& a, const Tensor& b) {
    return a.dtype() == b.dtype() && a.shape() == b.shape() && a.contiguous().to_vector<double>() == b.contiguous().to_vector<double>();
  };
  check::Require(same(main.standard_noise, repeat.result.standard_noise) && sal::Bytes(main.mean) == sal::Bytes(repeat.result.mean) &&
                     sal::Bytes(main.sample_gradient) == sal::Bytes(repeat.result.sample_gradient),
                 "exact seeded gradient/noise replay");
  {
    const auto zero = sal::F(groups.at("zero").result.mean);
    for (int i = 0; i < 7840; ++i) check::Near(zero[i], ordinary[i], 2e-6, "real sigma0 recovers vanilla");
  }
  const auto& prefix = groups.at("prefix").result;
  {
    const auto p = prefix.standard_noise.contiguous().to_vector<double>(), m = main.standard_noise.contiguous().to_vector<double>();
    for (std::size_t i = 0; i < p.size(); ++i) check::Near(p[i], m[i], 0, "small sample bank is exactprefix");
  }
  for (const std::string name : {"large_sigma", "clipped", "runner", "initial"})
    check::Require(same(groups.at(name).result.standard_noise, main.standard_noise), "paired controls use SAMEstandardnoise bank");
  check::Require(!same(groups.at("second_seed").result.standard_noise, main.standard_noise), "independentseedbankdiffers");
  const auto noisy = sal::F(main.noisy_input), mean = sal::F(main.mean), mean_abs = sal::F(main.mean_absolute),
             clipped = sal::F(groups.at("clipped").result.mean), runner = sal::F(groups.at("runner").result.mean),
             initial_mean = sal::F(groups.at("initial").result.mean), prefix_mean = sal::F(prefix.mean),
             second = sal::F(groups.at("second_seed").result.mean), standard_error = sal::F(main.standard_error);
  int    outside = 0, positive = 0, negative = 0;
  double peak = 0, vanilla_difference = 0, order_difference = 0;
  for (float v : noisy) outside += v < 0 || v > 1;
  for (int i = 0; i < 7840; ++i) {
    const double value = mean[i];
    positive += value > 0;
    negative += value < 0;
    vanilla_difference += std::abs(value - ordinary[i]);
    order_difference += mean_abs[i] - std::abs(value);
    peak = std::max({peak, std::abs(value), std::abs(double(ordinary[i])), double(mean_abs[i]), std::abs(double(clipped[i])),
                     std::abs(double(runner[i])), std::abs(double(initial_mean[i]))});
  }
  check::Require(outside > 0 && vanilla_difference > 1e-4 && order_difference > 1e-4 && positive > 0 && negative > 0,
                 "unclippednoise and signedmean differ from vanilla/magnitude averaging");
  const auto one = sal::Pick(examples, {0});
  std::vector<double> one_noise(std::size_t(samples) * 784);
  {
    const auto bank = main.standard_noise.contiguous().to_vector<double>();
    for (int k = 0; k < samples; ++k) std::copy_n(bank.data() + k * 7840, 784, one_noise.data() + k * 784);
  }
  const auto single = sal::F(smooth::SmoothGrad(one.images, sal::FromI({sal::I(targets)[0]}),
                                                nn::from_vector(one_noise, {samples, 1, 28, 28, 1}), sigma, false, 1,
                                                [&](const Tensor& points, const Tensor& classes) {
                                                  return model.Gradient(points, classes);
                                                })
                                 .mean);
  for (int i = 0; i < 784; ++i) check::Near(single[i], mean[i], 2e-6, "same-bank single/batch SmoothGrad parity");
  check::Require(before == model.Snapshot() && initial_before == initial.Snapshot(),
                 "all2x8weights immutable across noise/target/clip/seed controls");
  std::filesystem::create_directories(directory);
  sal::Arrays arrays = {{"images", examples.images}, {"labels", examples.labels},  {"source_ids", examples.source},
                        {"targets", targets},        {"runners", runners},         {"logits", logits},
                        {"vanilla", ordinary_t},     {"test_logits", test_logits}, {"test_labels", test.labels},
                        {"test_source_ids", test.source}};
  std::ofstream stats(directory / "diagnostics.tsv");
  stats << "case\tvanilla_normalized_TV\tSmoothGrad_normalized_TV\tprefix_main_L1\tsecond_seed_L1\tmean_standard_error\n";
  double vanilla_tv = 0, smooth_tv = 0;
  for (int b = 0; b < 10; ++b) {
    const double first = Roughness(ordinary, b), second_tv = Roughness(mean, b);
    double       prefix_difference = 0, seed_difference = 0, se = 0;
    vanilla_tv += first;
    smooth_tv += second_tv;
    for (int i = 0; i < 784; ++i) {
      const int j = b * 784 + i;
      prefix_difference += std::abs(double(mean[j]) - prefix_mean[j]);
      seed_difference += std::abs(double(mean[j]) - second[j]);
      se += standard_error[j] / 784.;
    }
    stats << b << '\t' << first << '\t' << second_tv << '\t' << prefix_difference << '\t' << seed_difference << '\t' << se << '\n';
  }
  for (const auto& [name, g] : groups) {
    const auto& r = g.result;
    for (const auto& [key, value] : std::vector<std::pair<std::string, Tensor>>{{"mean", r.mean},
                                                                                {"mean_absolute", r.mean_absolute},
                                                                                {"mean_square", r.mean_square},
                                                                                {"sample_variance", r.sample_variance},
                                                                                {"standard_error", r.standard_error},
                                                                                {"standard_noise", r.standard_noise},
                                                                                {"noisy_input", r.noisy_input},
                                                                                {"sample_gradient", r.sample_gradient}})
      arrays[name + "_" + key] = value;
    arrays[name + "_config"] =
        nn::from_vector(std::vector<double>{double(r.standard_noise.size(0)), double(g.sigma), double(g.clipped), double(g.seed)}, {4});
  }
  arrays["shared_peak"] = nn::scalar(float(peak));
  sal::SaveArrays(directory / "smoothgrad.npz", arrays);
  model.Export(directory / "model_arrays.npz");
  initial.Export(directory / "initial_arrays.npz");
  sal::CopyManifests(directory, {{source / "data_manifest.json", "data_manifest.json"},
                                 {paths::Dataset("smoothgrad") / "source_manifest.json", "source_manifest.json"}});
  std::ofstream html(directory / "index.html");
  html << "<!doctype html><meta charset='utf-8'><title>SmoothGrad</title><style>body{font:16px system-ui;max-width:1100px;margin:30px "
          "auto}img{max-width:100%;image-rendering:pixelated}section{margin:25px 0}</style><h1>SmoothGrad:signed mean of ordinary "
          "raw-logit gradients</h1><p>Fixed unperturbed trained target,Gaussian noise sigma="
       << sigma << " in[0,1]pixel coordinates(range1),samples=" << samples
       << ",seed1171. NO clipping by default;noisy points may leave the training pixel range. Mean signed gradients are averaged BEFORE "
          "absolute display. Meanabsolute and meansquare are separate variants,not silently substituted.</p><p>Columns:input,absolute "
          "vanilla,signed SmoothGrad,absolute SmoothGrad,meanabsolute gradient,absolute clipped-point control,absolute runner-up "
          "SmoothGrad,absolute initial-weight SmoothGrad. ONEshared absolute scale across all cases/maps;redpositive/bluenegative/"
          "whitezero signed,blackzero magnitude. Clipped control averages dF/d(noisypoint) at clippedpoints;it is NOT the derivative of "
          "F(clip(x+noise)) with a clip Jacobian.</p><p>Rawbank/noisyinputs/everygradient/samplevariance/MonteCarlo standarderrors and "
          "sample-count/second-seed/large-sigma controls are saved. Neither smooth appearance nor estimated sampling error proves causal "
          "importance or explanation faithfulness. The cached three-epoch classifier needs0newtraining.</p>";
  const auto image = sal::F(examples.images);
  const auto src = sal::I(examples.source), labels = sal::I(examples.labels), t = sal::I(targets), r = sal::I(runners);
  for (int b = 0; b < 10; ++b) {
    sal::Canvas pixels(112, 924);
    for (int y = 0; y < 112; ++y)
      for (int x = 0; x < 112; ++x) {
        const int    i = b * 784 + y / 4 * 28 + x / 4;
        const double v = mean[i] / peak;
        const auto   color = sal::SignedColor(v);
        for (int c = 0; c < 3; ++c) {
          pixels.Set(y, x, c, 255 * image[i]);
          pixels.Set(y, 116 + x, c, 255 * std::abs(double(ordinary[i])) / peak);
          pixels.Set(y, 232 + x, c, color[c]);
          pixels.Set(y, 348 + x, c, 255 * std::abs(v));
          pixels.Set(y, 464 + x, c, 255 * double(mean_abs[i]) / peak);
          pixels.Set(y, 580 + x, c, 255 * std::abs(double(clipped[i])) / peak);
          pixels.Set(y, 696 + x, c, 255 * std::abs(double(runner[i])) / peak);
          pixels.Set(y, 812 + x, c, 255 * std::abs(double(initial_mean[i])) / peak);
        }
      }
    const std::string filename = "case" + std::to_string(b) + ".png";
    sal::SaliencyPng(directory / filename, pixels);
    html << "<section><h2>Source" << src[b] << ",truth" << labels[b] << ",pred" << t[b] << ",runner" << r[b] << "</h2><img src='"
         << filename << "'></section>";
  }
  check::Require(bool(html) && bool(stats), "save SmoothGrad gallery/diagnostics");
  sal::SaliencyMetrics("unchanged3epochclassifier", test, test_logits);
  std::cout << "SmoothGrad default signed averaging:samples=" << samples << ",sigma=" << sigma
            << ",seed1171,NOclip;all8configurationscomputed,0newtraining\nvanillaL1difference=" << vanilla_difference
            << ",meanabs-vsabsmean L1gap=" << order_difference << ",signedpositive/negative=" << positive << '/' << negative
            << ",outofrange noisyfraction=" << double(outside) / double(noisy.size()) << "\nmean normalizedTV vanilla/SmoothGrad="
            << vanilla_tv / 10 << '/' << smooth_tv / 10
            << " (measured,not universal sharpness/faithfulness claim)\nSeedreplay/pairednoise/prefix/sigma0/single-batch/immutable2x8weights "
               "passed. Clipping/sigma/seed/target/model dependence and finiteMonteCarlo error explicit.\nArtifacts:"
            << directory.string() << "/index.html\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    nn::set_num_threads(4);
    int                   samples = 32;
    float                 sigma = .15f;
    std::filesystem::path source = sal::DefaultSource(), data = paths::Dataset("saliency"), out = paths::OutputRoot() / "models/smoothgrad";
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a.rfind("source=", 0) == 0)
        source = sal::SourceDir(a.substr(7));
      else if (a.rfind("data=", 0) == 0)
        data = a.substr(5);
      else if (a.rfind("out=", 0) == 0)
        out = a.substr(4);
      else if (a.rfind("samples=", 0) == 0)
        samples = std::stoi(a.substr(8));
      else if (a.rfind("sigma=", 0) == 0)
        sigma = std::stof(a.substr(6));
      else if (a.rfind("threads=", 0) == 0)
        nn::set_num_threads(std::stoi(a.substr(8)));
      else
        throw std::invalid_argument("expected source=PATH|tflow data=PATH out=PATH samples=N sigma=S threads=N");
    }
    EstimatorChecks();
    Run(source, data, out, samples, sigma);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
