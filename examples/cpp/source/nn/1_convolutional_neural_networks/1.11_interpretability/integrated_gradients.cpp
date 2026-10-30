// CMD: just run-nn integrated_gradients
// Ordinary raw-logit straight-line integrated gradients,baseline/completeness checks.
// Reuses3epochMNISTclassifier;black/darkgray/runner/initial controls,no newtraining.
//
// Port of tflow's integrated_gradients. Prepare: run-nn vanilla_saliency (writes
// output/nn/models/vanilla_saliency), or pass source=tflow to use tflow's recorded
// classifier (then every attribution is comparable with tflow's attributions.npz).
// Data: saliency_data.npz (paths::Dataset("saliency"), data=PATH). Integration
// helper: common/integrated_gradients.hpp. Outputs: output/nn/models/integrated_gradients/.
// Options:steps=N,source=PATH|tflow,data=PATH,out=PATH,threads=N.

#include <datamunge/nn/nn.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "common/checks.hpp"
#include "common/integrated_gradients.hpp"
#include "common/paths.hpp"
#include "common/saliency.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;

namespace {

void AxiomChecks() {
  auto first = [](const Tensor& x) {
    const Tensor a = nn::slice_axis(x, 1, 0, 1), b = nn::slice_axis(x, 1, 1, 2);
    return nn::add(nn::sub(nn::relu(a), nn::relu(nn::sub(a, 1.0))), nn::mul(b, 2.0));
  };
  auto second = [](const Tensor& x) {
    const Tensor a = nn::slice_axis(x, 1, 0, 1), b = nn::slice_axis(x, 1, 1, 2);
    return nn::add(nn::sub(1.0, nn::relu(nn::sub(1.0, nn::relu(a)))), nn::mul(b, 2.0));
  };
  auto callback = [](std::function<Tensor(const Tensor&)> f) {
    return [f](const Tensor& points, const Tensor&) {
      Tensor x = points.detach().clone();
      x.requires_grad_(true);
      return nn::grad(nn::sum(f(x)), {x})[0].detach().contiguous();
    };
  };
  const Tensor input = sal::FromF({2, 3, 9, .5f, -1, 4}, {2, 3}), baseline = nn::zeros({2, 3}), classes = sal::FromI({0, 0});
  const auto   ig = ig::IntegratedGradients(input, baseline, classes, 64, 7, callback(first)),
             equivalent = ig::IntegratedGradients(input, baseline, classes, 64, 1, callback(second)),
             zero = ig::IntegratedGradients(input, input, classes, 64, 5, callback(first));
  const auto a = sal::F(ig.attribution), e = sal::F(equivalent.attribution), z = sal::F(zero.attribution);
  const std::vector<double> expected = {1, 6, 0, .5, -2, 0};
  for (int i = 0; i < 6; ++i) {
    check::Near(a[i], expected[i], 1e-7, "ALL6 analytic saturated/linear/dummy IG entries");
    check::Near(e[i], expected[i], 1e-7, "equivalent implementation and different batching give same IG");
    check::Near(z[i], 0, 0, "input equals baseline gives exactzero attribution");
  }
  const auto endpoint = sal::F(callback(first)(input, classes));
  check::Near(endpoint[0], 0, 0, "saturated endpoint derivative0");
  check::Near(a[0], 1, 0, "nonzero IG despite saturated endpoint gradient0");
  check::Near(a[0] + a[1], 7, 1e-7, "positive completeness exact");
  check::Near(a[3] + a[4], -1.5, 1e-7, "negative completeness exact");
  std::cout << "IG analytic saturation/sensitivity/linear/dummy/zero-delta/equivalent-implementation/batching tests passed;ALL6 "
               "exactattributions,completeness7/-1.5,endpointgradient0butIG1\n";
}

struct Evaluation {
  ig::IntegratedResult integral;
  Tensor               baseline;
  std::vector<float>   score, base_score;
  std::vector<double>  delta, sum, residual;
};

Evaluation Evaluate(sal::SaliencyModel& model, const sal::SaliencyData& examples, const Tensor& baseline, const Tensor& targets,
                    int steps) {
  const auto z = sal::F(model.Predict(examples.images)), zb = sal::F(model.Predict(baseline));
  Evaluation out;
  out.baseline = baseline;
  out.integral = ig::IntegratedGradients(examples.images, baseline, targets, steps, 32,
                                         [&](const Tensor& points, const Tensor& classes) { return model.Gradient(points, classes); });
  const auto attribution = sal::F(out.integral.attribution);
  const auto t = sal::I(targets);
  for (int b = 0; b < 10; ++b) {
    const int c = t[b];
    out.score.push_back(z[b * 10 + c]);
    out.base_score.push_back(zb[b * 10 + c]);
    double delta = double(z[b * 10 + c]) - zb[b * 10 + c], sum = 0;
    for (int i = 0; i < 784; ++i) sum += attribution[b * 784 + i];
    out.delta.push_back(delta);
    out.sum.push_back(sum);
    out.residual.push_back(sum - delta);
  }
  return out;
}

bool Complete(const Evaluation& x) {
  for (int b = 0; b < 10; ++b)
    if (std::abs(x.residual[b]) > .01 + .01 * std::abs(x.delta[b])) return false;
  return true;
}

double MaxResidual(const Evaluation& x) {
  double maximum = 0;
  for (double r : x.residual) maximum = std::max(maximum, std::abs(r));
  return maximum;
}

void Run(const std::filesystem::path& source, const std::filesystem::path& data, const std::filesystem::path& directory,
         int requested) {
  check::Require(requested >= 2 && requested <= 256, "requested midpoint steps2..256");
  const auto  pools = ae::LoadSaliency(data);
  const auto& test = pools[2];
  sal::SaliencyModel model(false), initial(false);
  const auto         w = sal::RestoreSource(source, model, initial);  // same8 trained/initial weights as pinned source
  const auto         before = model.Snapshot(), initial_before = initial.Snapshot();
  const auto         logits = model.Predict(test);
  const auto         examples = sal::Pick(test, sal::FirstPerDigit(test));
  const auto         z = model.Predict(examples);
  const auto [targets, runners] = sal::TopTwo(z);
  const Tensor black = nn::zeros(examples.images.shape()), gray = nn::full(examples.images.shape(), double(.1f));
  const auto   coarse = Evaluate(model, examples, black, targets, std::max(1, requested / 2));
  int          steps = requested;
  auto         primary = Evaluate(model, examples, black, targets, steps), runner = Evaluate(model, examples, black, runners, steps),
       gray_result = Evaluate(model, examples, gray, targets, steps), initial_result = Evaluate(initial, examples, black, targets, steps);
  while (!(Complete(primary) && Complete(runner) && Complete(gray_result) && Complete(initial_result)) && steps < 512) {
    steps          = std::min(512, steps * 2);
    primary        = Evaluate(model, examples, black, targets, steps);
    runner         = Evaluate(model, examples, black, runners, steps);
    gray_result    = Evaluate(model, examples, gray, targets, steps);
    initial_result = Evaluate(initial, examples, black, targets, steps);
  }
  check::Require(Complete(primary) && Complete(runner) && Complete(gray_result) && Complete(initial_result),
                 "ALL40case/path completeness residuals <=.01+.01*abs(scorechange);increase quadrature if needed");
  const auto fine = Evaluate(model, examples, black, targets, 2 * steps);
  const auto endpoint = model.Gradient(examples, targets);
  const auto image = sal::F(examples.images), e = sal::F(endpoint), p = sal::F(primary.integral.attribution),
             gr = sal::F(gray_result.integral.attribution), ru = sal::F(runner.integral.attribution),
             in = sal::F(initial_result.integral.attribution);
  std::vector<float> times_input(7840);
  double endpoint_difference = 0, baseline_difference = 0, target_difference = 0, parameter_difference = 0, peak = 0;
  int    positive = 0, negative = 0;
  for (int i = 0; i < 7840; ++i) {
    times_input[i] = image[i] * e[i];
    endpoint_difference += std::abs(double(p[i]) - times_input[i]);
    baseline_difference += std::abs(double(p[i]) - gr[i]);
    target_difference += std::abs(double(p[i]) - ru[i]);
    parameter_difference += std::abs(double(p[i]) - in[i]);
    positive += p[i] > 0;
    negative += p[i] < 0;
    peak = std::max({peak, std::abs(double(p[i])), std::abs(double(gr[i])), std::abs(double(ru[i])), std::abs(double(in[i])),
                     std::abs(double(times_input[i]))});
  }
  check::Require(endpoint_difference > 1e-4 && baseline_difference > 1e-4 && target_difference > 1e-4 &&
                     parameter_difference > 1e-4 && positive > 0 && negative > 0,
                 "actual path integration and baseline/target/weight dependence with signed entries");
  const auto one = sal::Pick(examples, {0});
  const auto single = ig::IntegratedGradients(one.images, nn::zeros(one.images.shape()), sal::FromI({sal::I(targets)[0]}), steps,
                                              1, [&](const Tensor& points, const Tensor& classes) {
                                                return model.Gradient(points, classes);
                                              });
  const auto s = sal::F(single.attribution);
  for (int i = 0; i < 784; ++i) check::Near(s[i], p[i], 2e-6, "single-image/path batching invariance");
  check::Require(before == model.Snapshot() && initial_before == initial.Snapshot(),
                 "all2x8weights immutable across integration/baselines/targets");
  std::filesystem::create_directories(directory);
  sal::Arrays arrays = {{"images", examples.images},
                        {"labels", examples.labels},
                        {"source_ids", examples.source},
                        {"targets", targets},
                        {"runners", runners},
                        {"logits", z},
                        {"endpoint_gradient", endpoint},
                        {"input_times_gradient", sal::FromF(times_input, examples.images.shape())},
                        {"test_logits", logits},
                        {"test_labels", test.labels},
                        {"test_source_ids", test.source}};
  const std::vector<std::pair<std::string, const Evaluation*>> evaluations = {
      {"black", &primary}, {"coarse", &coarse}, {"fine", &fine}, {"runner", &runner}, {"gray", &gray_result}, {"initial", &initial_result}};
  std::ofstream stats(directory / "completeness.tsv");
  stats << "variant\tcase\tsteps\ttargetscore\tbaselinescore\tscoredelta\tIGsum\tresidual\n";
  for (const auto& [name, ev] : evaluations) {
    arrays[name + "_attribution"]    = ev->integral.attribution;
    arrays[name + "_mean_gradient"]  = ev->integral.mean_gradient;
    arrays[name + "_path_gradient"]  = ev->integral.path_gradient;
    arrays[name + "_nodes"]          = ev->integral.nodes;
    arrays[name + "_baseline"]       = ev->baseline;
    arrays[name + "_score"]          = sal::FromF(ev->score, {10});
    arrays[name + "_baseline_score"] = sal::FromF(ev->base_score, {10});
    arrays[name + "_delta"]          = nn::from_vector(ev->delta, {10});
    arrays[name + "_sum"]            = nn::from_vector(ev->sum, {10});
    arrays[name + "_residual"]       = nn::from_vector(ev->residual, {10});
    for (int b = 0; b < 10; ++b)
      stats << name << '\t' << b << '\t' << ev->integral.steps << '\t' << ev->score[b] << '\t' << ev->base_score[b] << '\t'
            << ev->delta[b] << '\t' << ev->sum[b] << '\t' << ev->residual[b] << '\n';
  }
  arrays["shared_peak"] = nn::scalar(float(peak));
  sal::SaveArrays(directory / "attributions.npz", arrays);
  model.Export(directory / "model_arrays.npz");
  initial.Export(directory / "initial_arrays.npz");
  sal::CopyManifests(directory, {{source / "data_manifest.json", "data_manifest.json"},
                                 {paths::Dataset("integrated_gradients") / "source_manifest.json", "source_manifest.json"}});
  std::ofstream html(directory / "index.html");
  html << "<!doctype html><meta charset='utf-8'><title>Integrated gradients</title><style>body{font:16px "
          "system-ui;max-width:1100px;margin:30px auto}img{max-width:100%;image-rendering:pixelated}section{margin:28px "
          "0}</style><h1>Integrated gradients of fixed raw class logits</h1><p>Ordinary input gradients integrated on x0+alpha*(x-x0),then "
          "multiplied by (x-x0). Midpointquadrature,not endpoint input-times-gradient. Targetclass stays fixed along each path. "
          "Blackbaseline0 and alternative darkgray0.1 are shown explicitly;neither guarantees a semantically neutral reference. SumIG "
          "approximates F(x)-F(x0),not F(x).</p><p>Columns:input,signedblackIG,absoluteblackIG,signedgrayIG,absoluterunnerIG,"
          "absoluteinitial-weightIG,absoluteendpointinput-times-gradient. ONEShared absolute scale for all maps/cases. "
          "Redpositive/bluenegative/whitezero signed;blackzero absolute. Baseline dependence and comparison panels are not proof of "
          "causal importance or segmentation.</p><p>Finalmidpointsteps="
       << steps << ",coarse=" << coarse.integral.steps << ",finecheck=" << fine.integral.steps
       << ". All40primary/control completeness tolerances checked individually;see completeness.tsv and raw path traces. Truth and "
          "selected classes remain visible. Model reused from3epochrun,0newtraining.</p>";
  const auto src = sal::I(examples.source), labels = sal::I(examples.labels), t = sal::I(targets), r = sal::I(runners);
  for (int b = 0; b < 10; ++b) {
    sal::Canvas pixels(112, 808);
    for (int y = 0; y < 112; ++y)
      for (int x = 0; x < 112; ++x) {
        const int    i = b * 784 + y / 4 * 28 + x / 4;
        const double v = p[i] / peak, grayv = gr[i] / peak;
        const auto   c1 = sal::SignedColor(v), c2 = sal::SignedColor(grayv);
        for (int c = 0; c < 3; ++c) {
          pixels.Set(y, x, c, 255 * image[i]);
          pixels.Set(y, 116 + x, c, c1[c]);
          pixels.Set(y, 232 + x, c, 255 * std::abs(v));
          pixels.Set(y, 348 + x, c, c2[c]);
          pixels.Set(y, 464 + x, c, 255 * std::abs(double(ru[i])) / peak);
          pixels.Set(y, 580 + x, c, 255 * std::abs(double(in[i])) / peak);
          pixels.Set(y, 696 + x, c, 255 * std::abs(double(times_input[i])) / peak);
        }
      }
    const std::string filename = "case" + std::to_string(b) + ".png";
    sal::SaliencyPng(directory / filename, pixels);
    html << "<section><h2>Source" << src[b] << ",truth" << labels[b] << ",pred" << t[b] << ",runner" << r[b]
         << "</h2><p>Black target/baseline=" << primary.score[b] << '/' << primary.base_score[b] << ",scorechange=" << primary.delta[b]
         << ",sumIG=" << primary.sum[b] << ",residual=" << primary.residual[b] << "</p><img src='" << filename << "'></section>";
  }
  for (const auto& [name, value] : std::vector<std::pair<std::string, float>>{{"black_baseline", 0.f}, {"gray_baseline", .1f}}) {
    sal::Canvas pixels(112, 112, sal::SaliencyByte(255 * value));
    sal::SaliencyPng(directory / (name + ".png"), pixels);
    html << "<p>" << name << ",unitpixel=" << value << "</p><img src='" << name << ".png'>";
  }
  check::Require(bool(html) && bool(stats), "write gallery and per-case completeness evidence");
  sal::SaliencyMetrics("unchanged3epochclassifier", test, logits);
  std::cout << "integrated gradients:requested/finalsteps=" << requested << '/' << steps
            << ",ordinary midpointpath/no newtraining;10realcases,black/gray/runner/initial controls\nmaxblack completeness residual "
               "coarse/main/fine="
            << MaxResidual(coarse) << '/' << MaxResidual(primary) << '/' << MaxResidual(fine) << ",gray/runner/initial="
            << MaxResidual(gray_result) << '/' << MaxResidual(runner) << '/' << MaxResidual(initial_result)
            << "\nL1differences endpoint-times-input/baseline/target/initial=" << endpoint_difference << '/' << baseline_difference << '/'
            << target_difference << '/' << parameter_difference << ",signedpositive/negative=" << positive << '/' << negative
            << "\nAll40completeness residuals within.01+.01*abs(scorechange);single/batch and immutable2x8weights passed. "
               "Finitequadrature/baseline/model dependence explicit,no causal-importance claim.\nArtifacts:"
            << directory.string() << "/index.html\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    nn::set_num_threads(4);
    int                   steps = 64;
    std::filesystem::path source = sal::DefaultSource(), data = paths::Dataset("saliency"),
                          out = paths::OutputRoot() / "models/integrated_gradients";
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a.rfind("source=", 0) == 0)
        source = sal::SourceDir(a.substr(7));
      else if (a.rfind("data=", 0) == 0)
        data = a.substr(5);
      else if (a.rfind("out=", 0) == 0)
        out = a.substr(4);
      else if (a.rfind("steps=", 0) == 0)
        steps = std::stoi(a.substr(6));
      else if (a.rfind("threads=", 0) == 0)
        nn::set_num_threads(std::stoi(a.substr(8)));
      else
        throw std::invalid_argument("expected source=PATH|tflow data=PATH out=PATH steps=N threads=N");
    }
    AxiomChecks();
    Run(source, data, out, steps);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
