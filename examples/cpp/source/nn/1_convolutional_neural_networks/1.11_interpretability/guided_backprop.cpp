// CMD: just run-nn guided_backprop
// Isolated guided-ReLU backward rule on the verified trained digit classifier.
// Same logits;vanilla/target/full-initial/head-initial controls,no retraining.
//
// Port of tflow's guided_backprop. Prepare: run-nn vanilla_saliency (writes
// output/nn/models/vanilla_saliency), or pass source=tflow to use tflow's recorded
// classifier (then every signal is comparable with tflow's signals.npz). Data:
// saliency_data.npz (paths::Dataset("saliency"), data=PATH). TF registers a custom
// TflowGuidedRelu op kernel and gradient; here it is an nn::Function
// (common/guided_backprop.hpp), and the "all three forward gates isolated" check
// counts its applications. Outputs: output/nn/models/guided_backprop/.
// Options:source=PATH|tflow,data=PATH,out=PATH,threads=N.

#include <datamunge/nn/nn.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "common/checks.hpp"
#include "common/guided_backprop.hpp"
#include "common/paths.hpp"
#include "common/saliency.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;

namespace {

void GateChecks() {
  std::vector<float> values(9), incoming(9);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      values[i * 3 + j]   = std::array<float, 3>{-2, 0, 2}[i];
      incoming[i * 3 + j] = std::array<float, 3>{-3, 0, 4}[j];
    }
  const Tensor up = sal::FromF(incoming, {9});
  auto objective_of = [&](const Tensor& x) { return nn::sum(nn::mul(guided::GuidedRelu(x), up)); };
  Tensor x = sal::FromF(values, {9});
  x.requires_grad_(true);
  const auto guided_out = sal::F(guided::GuidedRelu(x)), ordinary_out = sal::F(nn::relu(x));
  const auto g = sal::F(nn::grad(objective_of(x), {x})[0]);
  const auto v = sal::F(nn::grad(nn::sum(nn::mul(nn::relu(x), up)), {x})[0]);
  const auto deconv = sal::F(nn::mul(up, nn::cast(nn::greater(up, 0.0), nn::DType::Float32)));
  auto       rule = [](float a, float b) {
    return sal::F(guided::GuidedReluBackward(sal::FromF({a}, {1}), sal::FromF({b}, {1})))[0];
  };
  const float a = rule(2, -3), b = rule(2, 4), sum = rule(2, 1);
  for (int i = 0; i < 9; ++i) {
    const double forward = std::max(0.f, values[i]), u = incoming[i];
    check::Near(guided_out[i], forward, 0, "guided forward exactly ReLU");
    check::Near(ordinary_out[i], forward, 0, "ordinary forward exactly ReLU");
    check::Near(g[i], values[i] > 0 && u > 0 ? u : 0, 0, "ALL9 guided gate combinations including zeros");
    check::Near(v[i], values[i] > 0 ? u : 0, 0, "ALL9 ordinary gates unchanged");
    check::Near(deconv[i], u > 0 ? u : 0, 0, "ALL9 incoming-only deconv gates");
  }
  check::Near(g[6], 0, 0, "guided blocks negative upstream at active unit");
  check::Near(v[6], -3, 0, "ordinary gradient retains negative upstream");
  check::Near(g[2], 0, 0, "guided blocks inactive forward unit");
  check::Near(deconv[2], 4, 0, "incoming-only rule differs from guided");
  check::Near(a + b, 4, 0, "modified backward separate signals");
  check::Near(sum, 1, 0, "modified backward sum differs,NOT a linear VJP");
  auto plus = values, minus = values;
  plus[6] += .001f;
  minus[6] -= .001f;
  nn::NoGradGuard guard;
  const double    fd = (objective_of(sal::FromF(plus, {9})).item<double>() - objective_of(sal::FromF(minus, {9})).item<double>()) / .002;
  check::Near(fd, -3, .001, "true derivative -3 while guided signal0;do NOT validate guided as true gradient");
  std::cout << "guidedReLU ALL9sign/zero gates exact;ordinary/deconv distinctions and unchangedforward passed;nonlinear backward4!=1,trueFD="
            << fd << " vs guided0 explicitly verified\n";
}

/// tflow's GuidedModel: the saliency CNN with GuidedRelu gates and exact source weights.
struct GuidedModel {
  sal::SaliencyModel model{false, 1103, guided::GuidedRelu};
  explicit GuidedModel(const sal::Arrays& weights) {
    model.Assign(weights);
    const long before = guided::GuidedReluCount();
    model.Predict(nn::zeros({1, 28, 28, 1}));
    check::Require(guided::GuidedReluCount() - before == 3, "ALL3 forward ReLUs isolated in guided clone,none missed");
  }
  Tensor Predict(const sal::SaliencyData& d) { return model.Predict(d); }
  /// Guided raw-class backward signal, no final clipping.
  Tensor Backward(const sal::SaliencyData& d, const Tensor& classes) { return model.Gradient(d, classes); }
  std::vector<std::string> Snapshot() const { return model.Snapshot(); }
};

double Cosine(const std::vector<float>& a, const std::vector<float>& b, int row) {
  double dot = 0, aa = 0, bb = 0;
  for (int i = 0; i < 784; ++i) {
    const double x = a[row * 784 + i], y = b[row * 784 + i];
    dot += x * y;
    aa += x * x;
    bb += y * y;
  }
  return aa > 0 && bb > 0 ? dot / std::sqrt(aa * bb) : 0;
}

void Run(const std::filesystem::path& source, const std::filesystem::path& data, const std::filesystem::path& directory) {
  const auto  pools = ae::LoadSaliency(data);
  const auto& test = pools[2];
  sal::SaliencyModel vanilla(false), fresh(false);
  const auto         w = sal::RestoreSource(source, vanilla, fresh);  // restore previously validated 3epoch classifier
  auto               head = w.trained;
  head.at("head/W") = w.initial.at("head/W");
  head.at("head/b") = w.initial.at("head/b");
  GuidedModel guided(w.trained), random(w.initial), head_random(head);
  const auto  ordinary_before = vanilla.Snapshot(), before = guided.Snapshot(), random_before = random.Snapshot(),
             head_before = head_random.Snapshot();
  check::Require(ordinary_before == before, "guided model arrays match all8 restored checkpoint tensors");
  const auto logits = sal::SaliencyPredict(vanilla, test), guided_logits = sal::SaliencyPredict(guided.model, test);
  {
    const auto l = sal::F(logits), gl = sal::F(guided_logits);
    for (std::size_t i = 0; i < l.size(); ++i)
      check::Near(l[i], gl[i], 1e-6, "all2000 forward logits identical despite modified backward rule");
  }
  const auto examples = sal::Pick(test, sal::FirstPerDigit(test));
  const auto z = guided.Predict(examples);
  const auto [targets, runners] = sal::TopTwo(z);
  const auto ordinary_t = vanilla.Gradient(examples, targets), g_t = guided.Backward(examples, targets),
             runner_t = guided.Backward(examples, runners), random_t = random.Backward(examples, targets),
             head_t = head_random.Backward(examples, targets), repeat_t = guided.Backward(examples, targets);
  const auto ordinary = sal::F(ordinary_t), g = sal::F(g_t), runner = sal::F(runner_t), random_signal = sal::F(random_t),
             head_signal = sal::F(head_t), repeat = sal::F(repeat_t);
  double vanilla_delta = 0, target_delta = 0, random_delta = 0, peak = 0;
  int    positive = 0, negative = 0;
  for (std::size_t i = 0; i < g.size(); ++i) {
    check::Near(g[i], repeat[i], 0, "repeat guided determinism");
    vanilla_delta += std::abs(double(g[i]) - ordinary[i]);
    target_delta += std::abs(double(g[i]) - runner[i]);
    random_delta += std::abs(double(g[i]) - random_signal[i]);
    peak = std::max({peak, std::abs(double(g[i])), std::abs(double(ordinary[i])), std::abs(double(runner[i])),
                     std::abs(double(random_signal[i])), std::abs(double(head_signal[i]))});
    positive += g[i] > 0;
    negative += g[i] < 0;
  }
  check::Require(vanilla_delta > 0 && target_delta > 0 && random_delta > 0 && positive > 0 && negative > 0,
                 "guided differs from vanilla/targets/initialweights,negative input signals retained");
  const auto one = sal::Pick(examples, {0});
  const auto single = sal::F(guided.Backward(one, sal::FromI({sal::I(targets)[0]})));
  for (int i = 0; i < 784; ++i) check::Near(single[i], g[i], 1e-6, "single/batch guided signal parity");
  check::Require(ordinary_before == vanilla.Snapshot() && before == guided.Snapshot() && random_before == random.Snapshot() &&
                     head_before == head_random.Snapshot(),
                 "all4models/every8tensor immutable across all controls");
  std::filesystem::create_directories(directory);
  std::ofstream metrics(directory / "similarity.tsv");
  metrics << "case\tguided_runner_cosine\tguided_initial_cosine\tguided_head_initial_cosine\tvanilla_guided_cosine\n";
  double runner_cos = 0, initial_cos = 0, head_cos = 0;
  for (int b = 0; b < 10; ++b) {
    const double a = Cosine(g, runner, b), c = Cosine(g, random_signal, b), d = Cosine(g, head_signal, b);
    runner_cos += a;
    initial_cos += c;
    head_cos += d;
    metrics << b << '\t' << a << '\t' << c << '\t' << d << '\t' << Cosine(ordinary, g, b) << '\n';
  }
  sal::SaveArrays(directory / "signals.npz", {{"images", examples.images},
                                              {"labels", examples.labels},
                                              {"source_ids", examples.source},
                                              {"logits", z},
                                              {"targets", targets},
                                              {"runners", runners},
                                              {"vanilla", ordinary_t},
                                              {"guided", g_t},
                                              {"runner_guided", runner_t},
                                              {"initial_guided", random_t},
                                              {"head_initial_guided", head_t},
                                              {"shared_peak", nn::scalar(float(peak))},
                                              {"test_logits", logits},
                                              {"test_labels", test.labels},
                                              {"test_source_ids", test.source}});
  sal::SaveArrays(directory / "model_arrays.npz", w.trained);
  sal::SaveArrays(directory / "initial_arrays.npz", w.initial);
  sal::SaveArrays(directory / "head_initial_arrays.npz", head);
  sal::CopyManifests(directory, {{source / "data_manifest.json", "data_manifest.json"},
                                 {paths::Dataset("guided_backprop") / "source_manifest.json", "source_manifest.json"}});
  std::ofstream html(directory / "index.html");
  html << "<!doctype html><meta charset='utf-8'><title>Guided backpropagation</title><style>body{font:16px "
          "system-ui;max-width:1100px;margin:30px auto}img{max-width:100%;image-rendering:pixelated}section{margin:25px "
          "0}</style><h1>Guided backpropagation:modified backward visualization</h1><p>Forward scores are unchanged. Each ReLU passes "
          "incoming g only when activation&gt;0 AND g&gt;0. This nonlinear backward rule is NOT the true input "
          "derivative;finite-difference validation would be incorrect. Signed input signals can still be negative because convolution "
          "weights are signed.</p><p>Columns:input,absolute vanilla,signed guided,absolute guided,absolute runner-up guided,absolute fully "
          "initialized guided,absolute head-only initialized guided. All use ONE shared absolute scale across all "
          "images/models/targets;redpositive/bluenegative/whitezero signed,blackzero absolute. Initial controls keep the trained predicted "
          "class FIXED. Classifier reused from three-epoch MNIST run;no training here.</p><p>Guided maps can look sharp yet show weak "
          "class/parameter specificity. Similarity.tsv and rawsignals expose the measured controls;nonidentical maps do not prove "
          "causal importance or class explanation.</p>";
  const auto image = sal::F(examples.images);
  const auto src = sal::I(examples.source), labels = sal::I(examples.labels), t = sal::I(targets), r = sal::I(runners);
  for (int b = 0; b < 10; ++b) {
    sal::Canvas pixels(112, 808);
    for (int y = 0; y < 112; ++y)
      for (int x = 0; x < 112; ++x) {
        const int  i = b * 784 + y / 4 * 28 + x / 4;
        const auto color = sal::SignedColor(g[i] / peak);
        for (int c = 0; c < 3; ++c) {
          pixels.Set(y, x, c, 255 * image[i]);
          pixels.Set(y, 116 + x, c, 255 * std::abs(double(ordinary[i])) / peak);
          pixels.Set(y, 232 + x, c, color[c]);
          pixels.Set(y, 348 + x, c, 255 * std::abs(double(g[i])) / peak);
          pixels.Set(y, 464 + x, c, 255 * std::abs(double(runner[i])) / peak);
          pixels.Set(y, 580 + x, c, 255 * std::abs(double(random_signal[i])) / peak);
          pixels.Set(y, 696 + x, c, 255 * std::abs(double(head_signal[i])) / peak);
        }
      }
    const std::string filename = "case" + std::to_string(b) + ".png";
    sal::SaliencyPng(directory / filename, pixels);
    html << "<section><h2>Source" << src[b] << ",true" << labels[b] << ",pred" << t[b] << ",runner" << r[b] << "</h2><img src='"
         << filename << "'></section>";
  }
  check::Require(bool(html) && bool(metrics), "write guided gallery/similaritycontrols");
  sal::SaliencyMetrics("same3epochclassifier", test, logits);
  std::cout << "guided backprop:ALL3 ReLUgates isolated;all2000forwardlogits "
               "equalordinary;10realheldoutcases,fixedtargets;0newtrainingepochs\nvanilla/runner/initial signalL1changes="
            << vanilla_delta << '/' << target_delta << '/' << random_delta << ",guidedinputpositive/negative=" << positive << '/'
            << negative << "\nmean guided cosine runner/fullinitial/headinitial=" << runner_cos / 10 << '/' << initial_cos / 10 << '/'
            << head_cos / 10
            << " (similarity controls,not explanation-quality proof)\nSnapshotall4models/8tensors,single-batch/deterministic backward "
               "passed;guided signal NOT ordinary derivative or causal explanation.\nArtifacts:"
            << directory.string() << "/index.html\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    nn::set_num_threads(4);
    std::filesystem::path source = sal::DefaultSource(), data = paths::Dataset("saliency"),
                          out = paths::OutputRoot() / "models/guided_backprop";
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a.rfind("source=", 0) == 0)
        source = sal::SourceDir(a.substr(7));
      else if (a.rfind("data=", 0) == 0)
        data = a.substr(5);
      else if (a.rfind("out=", 0) == 0)
        out = a.substr(4);
      else if (a.rfind("threads=", 0) == 0)
        nn::set_num_threads(std::stoi(a.substr(8)));
      else
        throw std::invalid_argument("expected source=PATH|tflow data=PATH out=PATH threads=N");
    }
    GateChecks();
    Run(source, data, out);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
