// CMD: just run-nn multilabel_classification
// Multi-label category presence with ResNet18,80 independent sigmoid heads and BCE.
// Real COCO128 box-category OR targets;custom disjoint80fit/24validation/24test.
// Three default epochs,fit-only capped positive weights and all BN state restored.
// Global thresholds choose validation microF1 only;grouped-score-tie AP avoids
// optimistic constant-score ranking. Report absent classes and class imbalance.
// Independent stable BCE/weights/AP/F1/calibration and all model gradient checks.
//
// Port of tflow's multilabel_classification. The data is tflow's
// multilabel.npz + manifest.json, read from $DATAMUNGE_DATA_DIR/multilabel
// (prepared by tflow's `just multilabel-assets`); the batch order is tflow's
// (nn::BatchOrder seed 263). Weights come from datamunge's generator, so
// trained numbers match TF statistically. TF builds the full ResNet-18 (11.2M
// parameters) as a graph only; here it also runs forward/backward once. TF's
// "20 BN update ops" have no counterpart (BN updates are eager). The
// checkpoint is best.npz (weights + moving BN). Outputs go to
// output/nn/models/multilabel_classification.
// Added option seed=251 (weight initialization; used only to show the seed spread).
// Options: epochs=3 data=PATH checks_only=0|1 seed=251 threads=N.

#include <datamunge/datasets/assets.hpp>
#include <cmath>
#include <iostream>

#include "common/multilabel.hpp"
#include "common/paths.hpp"

// Experiment workflow from examples/nn/common/multilabel.hpp.
namespace multilabel {
using namespace ::multilabel;
inline void TrainMultilabel(const std::filesystem::path& root, int epochs, const std::filesystem::path& directory, std::uint64_t seed = 251) {
  check::Require(epochs >= 2, "two or more short multi-label epochs");
  const auto      data    = LoadMultilabel(root);
  const auto      weights = MultilabelWeights(data.fit.labels);
  MultilabelModel model(true, weights, seed);
  const auto      probe   = MultilabelRows(data.fit, {0, 1, 2, 3, 4, 5, 6, 7});
  const double    initial = model.Loss(probe, false, true);
  const auto      untrained_validation = MultilabelPredict(model, data.validation), untrained = MultilabelPredict(model, data.test);
  std::filesystem::create_directories(directory);
  const auto    prefix = (directory / "best.npz").string();
  std::ofstream history(directory / "history.tsv");
  history << "epoch\tweighted_training_BCE\tvalidation_unweighted_BCE\tvalidation_mAP\n";
  nn::BatchOrder order(80, 8, 263);
  double         best     = INFINITY;
  int            selected = 0;
  std::cout << "multilabel_classification:" << model.parameters.count() << " parameters/" << model.parameters.size() << " tensors,BN="
            << model.state.count() << "/" << model.state.size() << ",ResNet18 smallstem/div8/GAP/80sigmoids;" << epochs << " epochs,batch8,Adam.001\n"
            << " real COCO128 custom80fit/24validation/24test unique-image pools;all80 categories,no crop/augmentation/pretraining\n"
            << " fit-only positive weights clamp[(N-pos+1)/(pos+1),1,20];fixed training-mode fit=" << initial << std::endl;
  const auto started = std::chrono::steady_clock::now();
  for (int epoch = 1; epoch <= epochs; ++epoch) {
    order.begin_epoch();
    std::size_t seen    = 0;
    double      average = 0;
    for (;;) {
      const auto ids = order.next();
      if (ids.empty()) break;
      const std::vector<std::size_t> rows(ids.begin(), ids.end());
      average += double(rows.size()) * model.Loss(MultilabelRows(data.fit, rows), true);
      seen += rows.size();
    }
    check::Require(seen == 80, "every fitting image once per epoch");
    average /= double(seen);
    const auto score = ScoreMultilabel(MultilabelPredict(model, data.validation), data.validation.labels);
    if (score.nll < best) {
      best = score.nll, selected = epoch;
      model.Save(prefix);
    }
    history << epoch << '\t' << average << '\t' << score.nll << '\t' << score.map << '\n';
    history.flush();
    std::cout << " epoch " << epoch << ":weightedtrainBCE=" << average << ",validationBCE=" << score.nll << ",mAP=" << score.map << std::endl;
  }
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  model.Restore(prefix);
  const auto validation = MultilabelPredict(model, data.validation);
  check::Near(ScoreMultilabel(validation, data.validation.labels).nll, best, 1e-8, "restored validation agreement");
  const double learned = model.Loss(probe, false, true);
  check::Require(learned < initial, "fixed training-mode weighted BCE improves");
  const double threshold = CalibrateMultilabel(validation, data.validation.labels),
               untrained_threshold = CalibrateMultilabel(untrained_validation, data.validation.labels);
  const auto   prior_validation = MultilabelPrior(data.fit.labels, 24), prior = MultilabelPrior(data.fit.labels, 24);
  const double prior_threshold  = CalibrateMultilabel(prior_validation, data.validation.labels);
  MultilabelModel deployed(false);
  deployed.Restore(prefix);
  check::Require(!deployed.optimizer && !deployed.weights.defined(), "only RGB plus checkpoint filename for extraction (no targets/class weights/optimizer)");
  const auto before      = deployed.Snapshot();
  const auto probability = MultilabelPredict(deployed, data.test), reference = MultilabelPredict(model, data.test),
             repeat      = MultilabelPredict(deployed, data.test);
  const auto pv = probability.to_vector<float>(), rv = reference.to_vector<float>(), qv = repeat.to_vector<float>();
  double     parity = 0;
  for (std::size_t i = 0; i < pv.size(); ++i) {
    parity = std::max(parity, std::abs(double(pv[i]) - rv[i]));
    check::Near(pv[i], qv[i], 0, "deterministic independent sigmoid inference");
  }
  check::Near(parity, 0, 1e-7, "fresh model exact checkpoint probability parity");
  const auto one = deployed.Predict(MultilabelRows(data.test.rgb, {3})).to_vector<float>();
  for (int c = 0; c < 80; ++c) check::Near(one[std::size_t(c)], pv[std::size_t(3 * 80 + c)], 2e-6, "single/batch BN inference agreement");
  const auto a = deployed.Predict(MultilabelRows(data.test.rgb, {0})).to_vector<float>(), b = deployed.Predict(MultilabelRows(data.test.rgb, {1})).to_vector<float>();
  double     sensitivity = 0;
  for (int c = 0; c < 80; ++c) sensitivity += std::abs(double(a[std::size_t(c)]) - b[std::size_t(c)]);
  sensitivity /= 80;
  check::Require(sensitivity > 1e-6, "image-dependent multi-label probabilities");
  check::Require(before == deployed.Snapshot(), "all learned weights and BN state immutable during inference");
  std::cout << " selected=" << selected << ",restored validationBCE=" << best << ",fixed training fit " << initial << " -> " << learned
            << ",validation-only global thresholds learned/untrained/prior=" << threshold << "/" << untrained_threshold << "/" << prior_threshold << "\n";
  PrintMultilabel("learned fixed.5", probability, data.test.labels, .5);
  PrintMultilabel("learned calibrated", probability, data.test.labels, threshold);
  PrintMultilabel("untrained calibrated", untrained, data.test.labels, untrained_threshold);
  PrintMultilabel("fit-only priors calibrated", prior, data.test.labels, prior_threshold);
  PrintMultilabel("all negative", nn::zeros({24, 80}), data.test.labels, .5);
  std::vector<std::int32_t> decisions(24 * 80);
  std::ofstream             predictions(directory / "predictions.tsv");
  predictions << "image_id\tclass_id\ttruth\tprobability\tprediction\n";
  const auto ids = data.test.ids.to_vector<int>();
  const auto tv  = data.test.labels.to_vector<float>();
  for (int n = 0; n < 24; ++n)
    for (int c = 0; c < 80; ++c) {
      const std::size_t i        = std::size_t(n * 80 + c);
      const bool        decision = pv[i] >= threshold;
      decisions[i]               = decision;
      predictions << ids[std::size_t(n)] << '\t' << c << '\t' << tv[i] << '\t' << pv[i] << '\t' << decision << '\n';
    }
  std::ofstream calibration(directory / "calibration.tsv");
  calibration << "model\tvalidation_only_threshold\nlearned\t" << threshold << "\nuntrained\t" << untrained_threshold << "\nfit_prior\t" << prior_threshold << '\n';
  nn::save_npz((directory / "test_outputs.npz").string(),
               {{"rgb", data.test.rgb},
                {"truth", data.test.labels},
                {"ids", data.test.ids},
                {"probability", probability},
                {"untrained", untrained},
                {"fit_prior", prior},
                {"decisions", nn::from_vector(decisions, {24, 80})},
                {"thresholds", nn::from_vector(std::vector<double>{threshold, untrained_threshold, prior_threshold}, {3})},
                {"weights", weights},
                {"validation_truth", data.validation.labels},
                {"validation_probability", validation},
                {"untrained_validation", untrained_validation},
                {"fit_truth", data.fit.labels},
                {"fit_ids", data.fit.ids},
                {"validation_ids", data.validation.ids}});
  MultilabelPreview(data.test, probability, threshold, directory / "comparison.png");
  std::filesystem::copy_file(root / "manifest.json", directory / "data_manifest.json", std::filesystem::copy_options::overwrite_existing);
  check::Require(bool(history) && bool(predictions) && bool(calibration), "save multi-label history/predictions/calibration");
  std::cout << " image-only parity=" << parity << ",image sensitivity=" << sensitivity << ",immutable weights/BN,single/batch pass;artifacts:" << directory
            << " (training " << seconds << " s)\nComplete short multi-label demonstration passed;custom COCO128 annotation-presence subset,not official COCO benchmark.\n";
}
}  // namespace multilabel



namespace nn = datamunge::nn;
using nn::Tensor;
using namespace multilabel;

namespace {

void LossChecks() {
  const std::vector<float> input = {-1000, 1000, 0, -2, 2, 0, -.5f, .5f}, truth = {1, 0, 1, 0, 0, 1, 1, 0}, weights = {2, 3, 4, 5};
  Tensor       x    = nn::from_vector(input, {2, 4}).requires_grad_();
  const Tensor loss = MultilabelObjective(x, nn::from_vector(truth, {2, 4}), nn::from_vector(weights, {4}));
  const auto   g    = nn::grad(loss, {x})[0].to_vector<float>();
  const auto   probabilities = nn::sigmoid(x.detach()).to_vector<float>();
  double       expected      = 0;
  for (int i = 0; i < 8; ++i) {
    const double z = input[std::size_t(i)], y = truth[std::size_t(i)], weight = y ? weights[std::size_t(i % 4)] : 1,
                 p = z >= 0 ? 1 / (1 + std::exp(-z)) : std::exp(z) / (1 + std::exp(z));
    expected += weight * (std::max(z, 0.) - z * y + std::log1p(std::exp(-std::abs(z)))) / 8;
    check::Near(g[std::size_t(i)], weight * (p - y) / 8, 1e-7, "every BCE logit derivative including exactzero/extreme logits and positive class factors");
  }
  check::Near(loss.item(), expected, 4e-5, "known finite weighted mean BCE");
  check::Require(probabilities[4] + probabilities[7] > 1, "independent sigmoid probabilities do not compete via softmax");
  const Tensor labels     = nn::from_vector(std::vector<float>{1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0}, {3, 4});
  const auto   calculated = MultilabelWeights(labels).to_vector<float>(), prior = MultilabelPrior(labels, 1).to_vector<float>();
  const double expected_weight[] = {1, 4, 4, 1.5}, expected_prior[] = {.6, .2, .2, .4};
  for (int c = 0; c < 4; ++c) {
    check::Near(calculated[std::size_t(c)], expected_weight[c], 0, "fit-only Laplace/capped class weight formula");
    check::Near(prior[std::size_t(c)], expected_prior[c], 3e-8, "fit-only smoothed class prior");
  }
  check::Near(MultilabelWeights(nn::zeros({80, 1})).to_vector<float>()[0], 20, 0, "rare/absent class positive weights capped20");
  std::cout << " all independent weighted BCE values/every logit gradient at extreme/zero,sigmoid noncompetition,fit-only class weight/prior/clipping fixtures passed\n";
}

void MetricChecks() {
  check::Near(MultilabelAP({.9, .7, .5}, {1, 0, 1}), 5. / 6, 1e-12, "known noninterpolated AP5/6");
  check::Near(MultilabelAP({.5, .5}, {1, 0}), .5, 0, "constant-score tie group AP equals prevalence");
  check::Near(MultilabelAP({.5, .5}, {0, 1}), .5, 0, "tie AP invariant to positive row ordering");
  check::Near(MultilabelAP({.9, .1}, {0, 0}), -1, 0, "absent-positive AP explicitly undefined");
  const Tensor probability = nn::from_vector(std::vector<float>{.9f, .8f, .4f, .7f, .6f, .2f, .5f, .3f, .1f}, {3, 3});
  const Tensor truth       = nn::from_vector(std::vector<float>{1, 1, 0, 0, 1, 0, 1, 0, 0}, {3, 3});
  const auto   score       = ScoreMultilabel(probability, truth, .5);
  check::Require(score.tp == 4 && score.fp == 1 && score.fn == 0 && score.tn == 4 && score.ap_classes == 2 && score.absent_classes == 1,
                 "all known multilabel confusion counts/absent class support");
  check::Near(score.map, 11. / 12, 1e-12, "mean AP over only positive-support classes");
  check::Near(score.micro_ap, .8875, 1e-12, "independent pooled-bit AP");
  check::Near(score.micro_f1, 8. / 9, 1e-12, "micro F1 pooled bits");
  check::Near(score.macro_f1, .6, 1e-12, "macro F1 includes absent class zero contribution");
  check::Near(score.hamming, 1. / 9, 1e-12, "Hamming bit error");
  check::Near(score.exact, 2. / 3, 1e-12, "subset exact-match accuracy");
  check::Near(CalibrateMultilabel(probability, truth), .5, 1e-12, "validation global micro-F1 best tie chooses highest threshold");
  const Tensor empty = nn::zeros({3, 3});
  check::Near(CalibrateMultilabel(probability, empty), 1.01, 0, "validation with no positives chooses predict-none sentinel on ties");
  const auto allnegative = ScoreMultilabel(empty, empty);
  check::Require(allnegative.exact == 1 && allnegative.micro_f1 == 0 && allnegative.ap_classes == 0 && allnegative.absent_classes == 3,
                 "all-negative empty-label metric conventions");
  std::cout << " known AP5/6/grouped ties/order invariance,absent-support handling,all confusion/F1/Hamming/exact metrics and validation-only threshold fixtures passed\n";
}

std::pair<std::int64_t, std::int64_t> Census(int divisor) {
  std::int64_t parameters = 0, state = 0;
  auto         conv       = [&](int in, int out, int kernel) {
    parameters += std::int64_t(kernel) * kernel * in * out + 2 * out;
    state += 2 * out;
  };
  int in = 64 / divisor;
  conv(3, in, 3);
  for (int stage = 0; stage < 4; ++stage) {
    const int width = (64 << stage) / divisor;
    for (int block = 0; block < 2; ++block) {
      conv(in, width, 3);
      conv(width, width, 3);
      if (in != width) conv(in, width, 1);
      in = width;
    }
  }
  parameters += std::int64_t(in) * 80 + 80;
  return {parameters, state};
}

void ModelChecks() {
  for (int divisor : {1, 8}) {
    nn::ParameterStore p(251), state(257);
    std::vector<float> input(2 * 64 * 64 * 3);
    for (std::size_t i = 0; i < input.size(); ++i) input[i] = float(.5 + .4 * std::sin(i * .17));
    Tensor     image = nn::from_vector(input, {2, 64, 64, 3}).requires_grad_();
    const auto net   = resnet::ResNet(p, state, image - .5, MultilabelConfig(divisor), true);
    auto       wrt   = p.parameters();
    wrt.push_back(image);
    const auto gradients = nn::grad(nn::mean(nn::square(net.logits)), wrt);
    const auto census    = Census(divisor);
    check::Require(p.count() == census.first && state.count() == census.second && p.size() == 62 && state.size() == 40 && net.blocks.size() == 8,
                   "independent full/compact ResNet/head/BN census");
    std::cout << " " << (divisor == 1 ? "full" : "compact") << " multilabel ResNet:" << p.count() << " parameters/62 derivatives,BN=" << state.count()
              << "/40,8blocks (executed)\n";
    if (divisor == 1) continue;
    std::vector<Tensor> stages{net.stem};
    for (const auto& s : net.stages) stages.push_back(s);
    for (const auto& b : net.blocks) stages.push_back(b);
    stages.push_back(net.logits);
    std::vector<nn::Shape> expected{{2, 64, 64, 8}, {2, 64, 64, 8}, {2, 32, 32, 16}, {2, 16, 16, 32}, {2, 8, 8, 64}};
    for (int stage = 0; stage < 4; ++stage)
      for (int block = 0; block < 2; ++block) expected.push_back({2, 64 / (1 << stage), 64 / (1 << stage), 8 << stage});
    expected.push_back({2, 80});
    for (int i = 0; i < 14; ++i) check::Require(stages[std::size_t(i)].shape() == expected[std::size_t(i)], "every residual stage/block and 80-logit geometry");
    double minimum = INFINITY, maximum = 0;
    for (const auto& g : gradients) {
      double norm = 0;
      for (float value : g.to_vector<float>()) {
        check::Require(std::isfinite(value), "finite multi-label model/RGB derivative");
        norm += double(value) * value;
      }
      norm = std::sqrt(norm);
      check::Require(norm > 0, "every62 learned tensor and observedRGB receive gradients");
      minimum = std::min(minimum, norm), maximum = std::max(maximum, norm);
    }
    std::cout << " compact all62parameter and RGB gradients finite/nonzero,norm range[" << minimum << "," << maximum << "],14exact residual/head shapes passed\n";
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int                   epochs      = 3;
    bool                  checks_only = false;
    std::uint64_t         seed        = 251;
    std::filesystem::path root        = paths::Dataset("multilabel");
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg.rfind("epochs=", 0) == 0)
        epochs = std::stoi(arg.substr(7));
      else if (arg.rfind("data=", 0) == 0)
        root = arg.substr(5);
      else if (arg == "checks_only=0" || arg == "checks_only=1")
        checks_only = arg == "checks_only=1";
      else if (arg.rfind("seed=", 0) == 0)
        seed = std::stoull(arg.substr(5));
      else if (arg.rfind("threads=", 0) == 0)
        nn::set_num_threads(std::stoi(arg.substr(8)));
      else
        throw std::invalid_argument("expected epochs=N data=PATH checks_only=0|1 seed=N threads=N");
    }
    LossChecks();
    MetricChecks();
    ModelChecks();
    if (checks_only) return 0;
    if (!std::filesystem::exists(root / "multilabel.npz"))
      datamunge::datasets::prepare_example_data("multilabel", root.string());
    TrainMultilabel(root, epochs, paths::OutputDir("models/multilabel_classification"), seed);
    std::cout << "All multi-label classification checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
