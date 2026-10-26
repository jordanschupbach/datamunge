// CMD: just run-nn vanilla_saliency
// Raw-class-logit gradients on ten real held-out MNIST digits;three-epoch CNN.
// Signed/absolute maps,runner-up and fresh-weight controls,independent derivatives.
//
// Port of tflow's vanilla_saliency. Data: saliency_data.npz of tflow's
// `just saliency-assets` (found via paths::Dataset("saliency"), or data=PATH).
// Trains the small saliency CNN (common/saliency.hpp) and writes
// output/nn/models/vanilla_saliency/{best,model_arrays,initial_arrays,saliency,
// test_outputs}.npz, history.tsv, case<k>.png and index.html; the other 1.11 ports
// restore this classifier. Weights initialise with datamunge's generator, so the
// trained numbers match TF only statistically; init=PATH (init=tflow = tflow's
// recorded initial_arrays.npz) replays TF's run from its initial weights with the
// identical batch order. TF's graph shape-inference check becomes an eager shape
// check of every endpoint.
// Options:epochs=N,data=PATH,checks_only=1,init=PATH|tflow,threads=N.

#include <datamunge/nn/nn.hpp>
#include <filesystem>
#include <iostream>

#include "common/checks.hpp"
#include "common/paths.hpp"
#include "common/saliency.hpp"

// Experiment workflow from examples/nn/common/saliency.hpp.
namespace sal {
using namespace ::sal;
inline void TrainVanillaSaliency(const std::filesystem::path& root, int epochs, const std::filesystem::path& init = {}) {
  check::Require(epochs > 0, "positive short classifier epochs");
  const auto  pools = ae::LoadSaliency(root);
  const auto& fit = pools[0];
  const auto& validation = pools[1];
  const auto& test = pools[2];
  SaliencyModel model;
  if (!init.empty()) model.Assign(LoadArrays(init));
  check::Require(model.parameters.count() == 26826 && model.parameters.size() == 8, "independent two-conv/two-dense inventory");
  const auto directory = paths::OutputDir("models/vanilla_saliency");
  const auto prefix = (directory / "best.npz").string();
  const auto initial_arrays = model.Export();
  const double initial = sensor::ScoreSensor(SaliencyPredict(model, fit), fit.labels, 10).nll;
  const auto   untrained = SaliencyPredict(model, test);
  {
    const auto probe = Pick(fit, {0, 1, 2, 3, 4, 5, 6, 7});
    ae::CheckGradients(model.parameters, model.ParameterGradients(probe), "finite every classifier derivative",
                       "all8 classifier tensors receive CE derivatives");
  }
  std::cout << "vanilla saliency classifier:26826parameters/8tensors,conv5x5x8/avg2/conv3x3x16/avg2/FC32/10rawlogits;" << epochs
            << "epochs,batch32,Adam.001,noBN/dropout;1000fit/200validation/200test balanced original MNIST pools;all8 "
               "derivatives finite/nonzero"
            << (init.empty() ? "" : ";initial weights from " + init.string()) << "\n";
  nn::BatchOrder order(1000, 32, 1109);
  std::ofstream  history(directory / "history.tsv");
  history << "epoch\ttrain_ce\tvalidation_nll\n";
  double best = INFINITY;
  int    selected = 0;
  for (int epoch = 1; epoch <= epochs; ++epoch) {
    order.begin_epoch();
    double sum = 0;
    int    seen = 0;
    for (;;) {
      const auto rows = order.next();
      if (rows.empty()) break;
      sum += double(rows.size()) * model.Update(Pick(fit, rows));
      seen += int(rows.size());
    }
    check::Require(seen == 1000, "all fitting records once per epoch");
    const double value = sensor::ScoreSensor(SaliencyPredict(model, validation), validation.labels, 10).nll;
    if (value < best) {
      best = value;
      selected = epoch;
      model.parameters.save(prefix);  // validation-only NLL selects classifier
    }
    history << epoch << '\t' << sum / seen << '\t' << value << '\n';
    history.flush();
    std::cout << "epoch" << epoch << ":trainCE=" << sum / seen << ",validationNLL=" << value << std::endl;
  }
  model.parameters.load(prefix);
  check::Near(sensor::ScoreSensor(SaliencyPredict(model, validation), validation.labels, 10).nll, best, 1e-6,
              "selected validation score restored");
  SaliencyModel deployed(false), random(false);
  deployed.parameters.load(prefix);  // fresh eight-tensor classifier deployment
  if (!init.empty()) random.Assign(initial_arrays);
  const auto before = deployed.Snapshot(), random_before = random.Snapshot();
  const auto logits = SaliencyPredict(deployed, test), reference = SaliencyPredict(model, test),
             random_logits = SaliencyPredict(random, test);
  {
    const auto l = F(logits), ref = F(reference), rl = F(random_logits), u = F(untrained);
    for (std::size_t i = 0; i < l.size(); ++i) {
      check::Near(l[i], ref[i], 1e-6, "fresh classifier restore");
      check::Near(rl[i], u[i], 1e-6, "fresh control reproduces actual initialization");
    }
  }
  const auto examples = Pick(test, FirstPerDigit(test));
  const auto z = deployed.Predict(examples);
  const auto [targets, runners] = TopTwo(z);
  const auto gradients = deployed.Gradient(examples, targets), alternative = deployed.Gradient(examples, runners),
             random_gradient = random.Gradient(examples, targets), repeat = deployed.Gradient(examples, targets);
  double target_delta = 0, random_delta = 0, peak = 0;
  int    positive = 0, negative = 0;
  {
    const auto g = F(gradients), a = F(alternative), rg = F(random_gradient), rep = F(repeat);
    for (std::size_t i = 0; i < g.size(); ++i) {
      check::Near(g[i], rep[i], 0, "repeat deterministic vanilla gradients");
      target_delta += std::abs(double(g[i]) - a[i]);
      random_delta += std::abs(double(g[i]) - rg[i]);
      peak = std::max({peak, std::abs(double(g[i])), std::abs(double(a[i])), std::abs(double(rg[i]))});
      positive += g[i] > 0;
      negative += g[i] < 0;
    }
  }
  check::Require(target_delta > 1e-4 && random_delta > 1e-4 && positive > 0 && negative > 0,
                 "target/weight controls change actual signed derivatives");
  const auto one = Pick(examples, {0});
  const auto one_target = FromI({I(targets)[0]});
  const auto single = F(deployed.Gradient(one, one_target));
  {
    const auto g = F(gradients);
    for (int i = 0; i < 784; ++i)
      check::Near(single[i], g[i], 1e-6, "sum-batch score derivative equals per-image derivative,not batchmean");
  }
  // Full model finite difference at the largest-gradient coordinate;targets stay fixed.
  int coordinate = 0;
  for (int i = 1; i < 784; ++i)
    if (std::abs(single[i]) > std::abs(single[coordinate])) coordinate = i;
  auto plus = F(one.images), minus = plus;
  plus[coordinate] += .001f;
  minus[coordinate] -= .001f;
  const int    tc = I(targets)[0];
  const auto   zp = F(deployed.Predict(FromF(plus, {1, 28, 28, 1}))), zm = F(deployed.Predict(FromF(minus, {1, 28, 28, 1})));
  const double fd = (double(zp[tc]) - zm[tc]) / .002;
  const double fd_error = std::abs(fd - single[coordinate]);
  check::Require(fd_error < .02 + .02 * std::abs(single[coordinate]),
                 "real whole-network central difference agrees with input derivative within piecewise/f32 tolerance");
  check::Require(before == deployed.Snapshot() && random_before == random.Snapshot(),
                 "inference/gradients/controls leave all model weights immutable");
  deployed.Export(directory / "model_arrays.npz");
  random.Export(directory / "initial_arrays.npz");
  const auto nearest = SaliencyNearest(fit, test);
  SaliencyMetrics("trained", test, logits);
  SaliencyMetrics("untrained", test, untrained);
  SaliencyMetrics("fit-only7x7intensity1NN", test, nearest);
  SaveArrays(directory / "saliency.npz", {{"images", examples.images},
                                          {"labels", examples.labels},
                                          {"source_ids", examples.source},
                                          {"targets", targets},
                                          {"runners", runners},
                                          {"logits", z},
                                          {"gradient", gradients},
                                          {"runner_gradient", alternative},
                                          {"initial_gradient", random_gradient},
                                          {"shared_peak", nn::scalar(float(peak))}});
  SaveArrays(directory / "test_outputs.npz", {{"images", test.images},
                                              {"labels", test.labels},
                                              {"source_ids", test.source},
                                              {"logits", logits},
                                              {"untrained", untrained},
                                              {"nearest", nearest},
                                              {"fit_images", fit.images},
                                              {"fit_labels", fit.labels},
                                              {"fit_source_ids", fit.source}});
  SaliencyGallery(directory, examples, targets, runners, gradients, alternative, random_gradient, peak);
  CopyManifests(directory, {{root / "manifest.json", "data_manifest.json"}});
  check::Require(bool(history), "write classifier history");
  std::cout << "selected=" << selected << ",fitNLL " << initial << " -> "
            << sensor::ScoreSensor(SaliencyPredict(model, fit), fit.labels, 10).nll
            << ";10heldoutcases,target/initial-weight gradient L1changes=" << target_delta << '/' << random_delta
            << ",signedpositive/negative=" << positive << '/' << negative << ",realFDcoordinate=" << coordinate
            << ",error=" << fd_error
            << "\nFreshrestore/gradientdeterminism/single-batch/immutable8weights passed;rawlogit gradient wrt[0,1]pixels,no "
               "softmax/loss/guided gates/gradient-times-input/smoothing.\nArtifacts:"
            << directory.string() << "/index.html\nSaliency is local model sensitivity,not causal importance/segmentation;small "
                                     "short-trained classifier and chosen targets limit interpretation.\n";
}
}  // namespace sal



namespace nn = datamunge::nn;
using nn::Tensor;

namespace {

void DerivativeChecks() {
  std::vector<float> input(16), kernel(8), bias{.03f, -.02f}, head(54);
  for (int i = 0; i < 16; ++i) input[i] = .3f * std::cos(.41 * i);
  for (int i = 0; i < 8; ++i) kernel[i] = .4f * std::sin(.53 * i + .2);
  for (int i = 0; i < 54; ++i) head[i] = .2f * std::sin(.71 * i + .4);
  const Tensor k = sal::FromF(kernel, {2, 2, 1, 2}), b = sal::FromF(bias, {2}), h = sal::FromF(head, {18, 3});
  auto         logits_of = [&](const Tensor& x) {
    auto feature = nn::relu(nn::bias_add(nn::conv2d(x, k, nn::conv_options(1, nn::Padding::Valid)), b));
    return nn::matmul(nn::reshape(feature, {1, 18}), h);
  };
  const Tensor x = sal::FromF(input, {1, 4, 4, 1});
  const auto   logits = sal::F(logits_of(x));
  const auto   gradient = sal::F(sal::VanillaInputGradient(logits_of, x, sal::FromI({1}), 3));
  const auto   other = sal::F(sal::VanillaInputGradient(logits_of, x, sal::FromI({0}), 3));
  std::vector<double> values(input.begin(), input.end());
  auto host = [&](const std::vector<double>& v, int target) {
    double score = 0;
    for (int y = 0; y < 3; ++y)
      for (int a = 0; a < 3; ++a)
        for (int c = 0; c < 2; ++c) {
          double pre = bias[c];
          for (int ky = 0; ky < 2; ++ky)
            for (int kx = 0; kx < 2; ++kx) pre += v[(y + ky) * 4 + a + kx] * kernel[(ky * 2 + kx) * 2 + c];
          score += std::max(pre, 0.) * head[((y * 3 + a) * 2 + c) * 3 + target];
        }
    return score;
  };
  double error = 0, delta = 0;
  int    positive = 0, negative = 0;
  for (int i = 0; i < 16; ++i) {
    auto plus = values, minus = values;
    plus[i] += 1e-5;
    minus[i] -= 1e-5;
    error = std::max(error, std::abs((host(plus, 1) - host(minus, 1)) / 2e-5 - gradient[i]));
    delta += std::abs(double(gradient[i]) - other[i]);
    positive += gradient[i] > 0;
    negative += gradient[i] < 0;
  }
  check::Near(logits[1], host(values, 1), 1e-7, "raw target score independent host reference");
  check::Near(error, 0, 2e-7, "ALL16 raw-score input finite differences");
  check::Require(positive > 0 && negative > 0 && delta > 1e-4, "signed upstream derivatives preserved,target changes derivative");
  std::cout << "vanilla class-score fixture:ALL16 independent hostFD error=" << error
            << ",positive/negative derivatives=" << positive << '/' << negative << ",targetdelta=" << delta
            << ";rawscore and standard signed backprop passed\n";
}

/// TF's ShapeRefiner pass becomes eager shapes of every endpoint for a batch of two.
void GraphChecks() {
  nn::ParameterStore p(1117);
  Tensor             x = nn::zeros({2, 28, 28, 1});
  x.requires_grad_(true);
  const auto net = sal::SaliencyCnn(p, x);
  const auto grad = nn::grad(nn::sum(sal::SaliencyScore(net.logits, sal::FromI({0, 1}))), {x})[0];
  auto shape = [](const Tensor& t, const nn::Shape& dims) {
    check::Require(t.dim() == std::int64_t(dims.size()), "classifier endpoint rank");
    check::Require(t.shape() == dims, "classifier/gradient endpoint dimensions");
  };
  shape(net.conv1, {2, 28, 28, 8});
  shape(net.pool1, {2, 14, 14, 8});
  shape(net.conv2, {2, 14, 14, 16});
  shape(net.pool2, {2, 7, 7, 16});
  shape(net.hidden, {2, 32});
  shape(net.logits, {2, 10});
  shape(grad, {2, 28, 28, 1});
  check::Require(p.count() == 5 * 5 * 8 + 8 + 3 * 3 * 8 * 16 + 16 + 784 * 32 + 32 + 32 * 10 + 10 && p.count() == 26826 &&
                     p.size() == 8,
                 "independent parameter inventory");
  std::cout << "saliency CNN all6endpoint/input-gradient shapes,26826parameters/8tensors passed\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    nn::set_num_threads(4);
    int                   epochs = 3;
    bool                  checks = false;
    std::filesystem::path root = paths::Dataset("saliency"), init;
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a.rfind("epochs=", 0) == 0)
        epochs = std::stoi(a.substr(7));
      else if (a.rfind("data=", 0) == 0)
        root = a.substr(5);
      else if (a.rfind("init=", 0) == 0)
        init = a.substr(5) == "tflow" ? replay::TfModels("vanilla_saliency") / "initial_arrays.npz"
                                      : std::filesystem::path(a.substr(5));
      else if (a.rfind("threads=", 0) == 0)
        nn::set_num_threads(std::stoi(a.substr(8)));
      else if (a == "checks_only=1")
        checks = true;
      else if (a != "checks_only=0")
        throw std::invalid_argument("expected epochs=N data=PATH checks_only=0|1 init=PATH|tflow threads=N");
    }
    DerivativeChecks();
    GraphChecks();
    if (!checks) sal::TrainVanillaSaliency(root, epochs, init);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
