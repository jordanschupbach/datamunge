// CMD: just run-nn kim_sentence_cnn
// Kim2014 CNN-rand: learned embeddings,parallel3/4/5-ReLU word convolutions,
// max-over-time pooling,dropout.5,softmax and Adadelta with head maxnorm3.
// Real pinned MR v1.0 sentences;fit-only vocabulary,duplicate-group holdout.
// Full architecture and independent every-coordinate host gradient fixtures.
// Three compact default epochs;untrained/majority/fit-only unigram NB controls.
//
// Port of tflow's kim_sentence_cnn. Data: tflow's `just sentence-assets`
// (output/datasets/sentence/sentence.npz, found via paths::Dataset).
// Deviations: checkpoints are .npz (best.npz); tflow's GraphDef placeholder
// audit has no eager counterpart (the inference model is built without labels
// or an optimizer instead). Weights initialise with nn's generator, so trained
// numbers match TF statistically.
// Options: epochs=3 data=PATH checks_only=0|1 threads=4.
// Output: output/nn/models/kim_sentence_cnn/.

#include <datamunge/datasets/assets.hpp>
#include <array>

#include "common/sentence.hpp"

// Experiment workflow from examples/nn/common/sentence.hpp.
namespace sentence {
using namespace ::sentence;
inline void TrainSentence(const std::filesystem::path& root, int epochs) {
  check::Require(epochs > 0, "positive short sentiment epochs");
  if (!std::filesystem::exists(root / "sentence.npz"))
    datamunge::datasets::prepare_example_data("text", (root.parent_path()).string());
  const auto  data = LoadSentences(root);
  const auto& fit = data.pools[0];
  const auto& validation = data.pools[1];
  const auto& test = data.pools[2];
  SentenceModel model(data.config);
  const auto    directory = paths::OutputDir("models/kim_sentence_cnn");
  const auto    prefix = (directory / "best.npz").string();
  const auto    probe = SentenceRows(fit, Range(0, 50));
  const double  initial = model.Loss(probe);
  const Tensor  untrained = model.PredictAll(test);
  std::cout << "CNN-rand compact:" << model.parameters.count() << " parameters/" << model.parameters.size()
            << " learned tensors,18 Adadelta slots;E32,F32,windows3/4/5,dropout.5,maxnorm3;" << epochs
            << " epochs,batch50,rho.95,eps1e-6,initial fixed CE=" << initial << std::endl;
  check::Require(model.optimizer->slots.size() == 18, "two Adadelta slots per learned tensor");
  RequireLiveGradients(model.parameters, nn::grad(SentenceCE(model.Forward(probe, true).logits, probe.labels), model.parameters.parameters()),
                       "embedding/CNN/head");
  std::cout << " all9 actual training gradient tensors finite/nonzero\n";
  std::ofstream history(directory / "history.tsv");
  history << "epoch\ttrain_dropout_ce\tvalidation_ce\n";
  nn::BatchOrder order(fit.size(), 50, 331);
  double         best = INFINITY;
  int            selected = 0;
  for (int epoch = 1; epoch <= epochs; ++epoch) {
    order.begin_epoch();
    double sum = 0;
    int    seen = 0;
    for (;;) {
      const auto rows = order.next();
      if (rows.empty()) break;
      // Dropout training graph, CE, all gradients and one Adadelta/max-norm step.
      const auto   batch = SentenceRows(fit, rows);
      const Tensor loss = SentenceCE(model.Forward(batch, true).logits, batch.labels);
      model.optimizer->Step(nn::grad(loss, model.parameters.parameters()));
      const double value = loss.item<double>();
      check::Require(std::isfinite(value) && value >= 0, "finite sentence CE");
      sum += rows.size() * value;
      seen += static_cast<int>(rows.size());
    }
    check::Require(seen == fit.size(), "each fitting sentence once per epoch");
    const double score = model.Loss(validation);
    if (score < best) {
      best = score;
      selected = epoch;
      model.Save(prefix);
    }
    history << epoch << '\t' << sum / seen << '\t' << score << '\n';
    history.flush();
    std::cout << " epoch " << epoch << ":trainCE=" << sum / seen << ",validationCE=" << score << std::endl;
  }
  model.Restore(prefix);
  check::Near(model.Loss(validation), best, 1e-6, "restored selected validation CE");
  const double learned = model.Loss(probe);
  check::Require(learned < initial, "short-run fixed fitting sentiment CE improves");
  // The deployed model is a fresh inference-only model (no labels, no
  // optimizer); tflow's GraphDef placeholder audit has no eager counterpart.
  SentenceModel deployed(data.config, false);
  deployed.Restore(prefix);
  check::Require(deployed.parameters.size() == 9 && !deployed.optimizer, "inference model holds only the nine learned tensors");
  const auto   before = deployed.Snapshot();
  const Tensor logits = deployed.PredictAll(test), reference = model.PredictAll(test), repeat = deployed.PredictAll(test);
  check::Near(replay::MaxAbsDiff(logits, reference), 0, 1e-6, "training/inference independent restore parity");
  check::Require(replay::MaxAbsDiff(logits, repeat) == 0, "dropout-free deterministic inference");
  auto         one = SentenceRows(test, {0});
  const Tensor single = deployed.Predict(one);
  check::Near(replay::MaxAbsDiff(single, nn::slice_axis(logits, 0, 0, 1)), 0, 1e-6, "single/batch sentence agreement");
  const int length = one.lengths.to_vector<int>()[0];
  auto      tokens = one.tokens.to_vector<int>();
  for (int t = length; t < 64; ++t) tokens[t] = 1;
  one.tokens = nn::from_vector(tokens, {1, 64});
  check::Require(replay::MaxAbsDiff(deployed.Predict(one), single) == 0, "legal out-of-length token substitution invariant");
  one = SentenceRows(test, {0});
  tokens = one.tokens.to_vector<int>();
  for (int t = 0; t < length; ++t) tokens[t] = 1;
  one.tokens = nn::from_vector(tokens, {1, 64});
  const double sensitivity = nn::sum(nn::abs(nn::sub(nn::cast(single, nn::DType::Float64), nn::cast(deployed.Predict(one), nn::DType::Float64)))).item<double>();
  check::Require(sensitivity > 1e-6, "prediction depends on observed word IDs");
  check::Require(replay::SameBits(before, deployed.Snapshot()), "inference leaves all nine learned tensors unchanged");
  const auto head = deployed.parameters.at("classifier/W").to_vector<double>();
  for (int c = 0; c < 2; ++c) {
    double norm = 0;
    for (int i = 0; i < 3 * data.config.filters; ++i) norm += head[2 * i + c] * head[2 * i + c];
    check::Require(norm <= 9 + 1e-5, "classifier column norm bounded by3");
  }
  const Tensor nb = SentenceNB(fit, test, data.config.vocabulary);
  const Tensor majority = nn::zeros({test.size(), 2});
  SentenceMetrics("trained CNN-rand", test.labels, logits);
  SentenceMetrics("untrained", test.labels, untrained);
  SentenceMetrics("fit unigram multinomial NB", test.labels, nb);
  SentenceMetrics("majority/tie class0", test.labels, majority);
  PrintConfusion("trained CNN-rand test", Confusion(test.labels, Positive(logits), 2), 2);
  std::ofstream predictions(directory / "predictions.tsv");
  predictions << "source_id\tlabel\tprediction\tnb_prediction\tnegative_logit\tpositive_logit\n";
  const auto ids = test.source_ids.to_vector<int>(), labels = test.labels.to_vector<int>();
  const auto z = logits.to_vector<float>();
  const auto p = Positive(logits), q = Positive(nb);
  for (std::int64_t n = 0; n < test.size(); ++n)
    predictions << ids[n] << '\t' << labels[n] << '\t' << p[n] << '\t' << q[n] << '\t' << z[2 * n] << '\t' << z[2 * n + 1] << '\n';
  nn::save_npz((directory / "test_outputs.npz").string(),
               {{"tokens", test.tokens}, {"lengths", test.lengths}, {"labels", test.labels}, {"source_ids", test.source_ids},
                {"logits", logits}, {"untrained", untrained}, {"nb_logits", nb}});
  CopyIfExists(root / "manifest.json", directory / "data_manifest.json");
  check::Require(bool(history) && bool(predictions), "save sentence histories/predictions");
  std::cout << " selected=" << selected << ",fixed fitting CE " << initial << " -> " << learned
            << ",word substitution logit change=" << sensitivity
            << ";restore/padding/single-batch/determinism/immutable/maxnorm passed\nArtifacts:" << directory
            << "\nCustom MR holdout,CNN-rand,not pretrained or published cross-validation accuracy.\n";
}
}  // namespace sentence



namespace nn = datamunge::nn;
using nn::Tensor;
using sentence::Weights;

namespace {

std::array<double, 4> Host(const Weights& weights) {
  const int             ids[2][5] = {{1, 2, 3, 6, 5}, {2, 4, 1, 5, 3}}, length[2] = {3, 5};
  std::array<double, 4> result{};
  for (int n = 0; n < 2; ++n) {
    std::array<double, 6> pooled{};
    for (int branch = 0; branch < 3; ++branch) {
      const int kernel = branch + 3;
      for (int f = 0; f < 2; ++f) {
        double maximum = 0;
        for (int t = 0; t < 5 + 8 - kernel + 1; ++t) {
          double value = weights[2 + 2 * branch][f];
          for (int k = 0; k < kernel; ++k) {
            const int position = t + k - 4;
            if (position < 0 || position >= length[n]) continue;
            for (int e = 0; e < 2; ++e) value += weights[0][(ids[n][position] - 1) * 2 + e] * weights[1 + 2 * branch][(k * 2 + e) * 2 + f];
          }
          maximum = std::max(maximum, value);
        }
        pooled[2 * branch + f] = maximum;
      }
    }
    for (int c = 0; c < 2; ++c) {
      result[2 * n + c] = weights[8][c];
      for (int f = 0; f < 6; ++f) result[2 * n + c] += pooled[f] * weights[7][2 * f + c];
    }
  }
  return result;
}
double Objective(const Weights& weights) {
  const auto z = Host(weights);
  return .3 * z[0] - .2 * z[1] - .1 * z[2] + .4 * z[3];
}

void HostChecks() {
  nn::ParameterStore      p(337);
  const sentence::KimConfig config{7, 5, 2, 2};
  Tensor                  tokens = nn::from_vector(std::vector<int>{1, 2, 3, 6, 5, 2, 4, 1, 5, 3}, {2, 5});
  const Tensor            lengths = nn::from_vector(std::vector<int>{3, 5}, {2});
  const Tensor            weight = nn::from_vector(std::vector<float>{.3f, -.2f, -.1f, .4f}, {2, 2});
  sentence::KimCnn(p, tokens, lengths, config);
  const auto weights = sentence::FixParameters(p, [&](std::size_t v, std::int64_t i, const nn::Shape& shape) {
    return shape.size() == 1 && v != 8 ? .17f + .03f * static_cast<float>(std::sin(i + v))
                                       : static_cast<float>(.23 * std::sin(.7 * i + 1.3 * v + .2));
  });
  const auto   net = sentence::KimCnn(p, tokens, lengths, config);
  const auto   gradients = nn::grad(nn::sum(nn::mul(net.logits, weight)), p.parameters());
  const auto   expected = Host(weights);
  const auto   z = net.logits.to_vector<double>();
  double       value_error = 0;
  for (int i = 0; i < 4; ++i) value_error = std::max(value_error, std::abs(expected[i] - z[i]));
  const auto pad = nn::slice_axis(net.embedding, 0, 0, 1).to_vector<double>();
  for (double e : pad) check::Near(e, 0, 0, "structural PAD embedding exactly zero");
  int          coordinates = 0;
  const double gradient_error = sentence::HostGradientError(weights, gradients, Objective, coordinates);
  check::Require(coordinates == 80 && p.size() == 9, "all80 tiny parameter coordinates covered");
  check::Near(value_error, 0, 1e-7, "independent host wide convolution/ReLU/max/concat/head");
  check::Near(gradient_error, 0, 2e-6, "EVERY embedding/filter/bias/head derivative versus host FD");
  auto t = tokens.to_vector<int>();
  t[3]   = 4;
  t[4]   = 1;
  tokens = nn::from_vector(t, {2, 5});
  auto out = sentence::KimCnn(p, tokens, lengths, config).logits.to_vector<double>();
  for (int c = 0; c < 2; ++c) check::Near(out[c], expected[c], 1e-7, "padding values cannot affect sentence features");
  std::swap(t[5], t[9]);
  tokens = nn::from_vector(t, {2, 5});
  out    = sentence::KimCnn(p, tokens, lengths, config).logits.to_vector<double>();
  double difference = 0;
  for (int c = 0; c < 2; ++c) difference += std::abs(out[2 + c] - expected[2 + c]);
  check::Require(difference > 1e-5, "window features depend on word order");
  std::cout << " independent wide3/4/5 conv/ReLU/max/head error=" << value_error
            << ",ALL80 host finite-difference derivative error=" << gradient_error << ";zeroPAD/padding/order passed\n";
}

void OptimizerChecks() {
  nn::ParameterStore p(347);
  p.get("classifier/W", {2, 2}, nn::init::zeros());
  p.set("classifier/W", nn::from_vector(std::vector<float>{3.f, .1f, 4.f, .2f}, {2, 2}));
  sentence::KimAdadelta    step(p);
  std::array<double, 4> host = {3, .1, 4, .2}, eg{}, ed{};
  double                error = 0;
  for (int t = 0; t < 3; ++t) {
    std::vector<float> gradient(4);
    for (int i = 0; i < 4; ++i) {
      const float value = (i + 1) * .2f * (t == 1 ? -1 : 1);
      gradient[i]       = value;
      eg[i]             = .95 * eg[i] + .05 * value * value;
      const double delta = std::sqrt(ed[i] + 1e-6) / std::sqrt(eg[i] + 1e-6) * value;
      ed[i]             = .95 * ed[i] + .05 * delta * delta;
      host[i] -= delta;
    }
    for (int c = 0; c < 2; ++c) {
      const double scale = std::max(1., std::sqrt(host[c] * host[c] + host[2 + c] * host[2 + c]) / 3);
      host[c] /= scale;
      host[2 + c] /= scale;
    }
    step.Step({nn::from_vector(gradient, {2, 2})});
    const auto w = p.at("classifier/W").to_vector<double>(), g = step.slots.at("gradient0").to_vector<double>(),
               d = step.slots.at("update0").to_vector<double>();
    for (int i = 0; i < 4; ++i)
      error = std::max({error, std::abs(host[i] - w[i]), std::abs(eg[i] - g[i]), std::abs(ed[i] - d[i])});
  }
  check::Near(error, 0, 1e-6, "independent three-step Adadelta/both-slot/head-column projection");
  std::cout << " three signed Adadelta steps/all slots/maxnorm3 host error=" << error << std::endl;
}

void DropoutChecks() {
  nn::Generator gen(359);
  Tensor        x = nn::full({4096}, 1.0);
  x.requires_grad_(true);
  const Tensor training  = nn::dropout(x, .5, true, gen);
  const Tensor inference = nn::dropout(x, 0, true, gen);
  const auto   gradient  = nn::grad(nn::sum(training), {x})[0].to_vector<float>();
  const auto   values = training.to_vector<float>(), identity = inference.to_vector<float>();
  int          retained = 0;
  for (int i = 0; i < 4096; ++i) {
    check::Require(values[i] == 0 || values[i] == 2, "dropout.5 mask has only0 or2");
    retained += values[i] == 2;
    check::Near(gradient[i], values[i], 0, "dropout derivative uses SAME sampled mask");
    check::Near(identity[i], 1, 0, "keep1 dropout identity");
  }
  check::Require(retained > 1800 && retained < 2300, "seeded half-drop mask retention");
  std::cout << " dropout.5 values/shared-mask derivatives and keep1 identity passed,retained=" << retained << "/4096\n";
}

void LossChecks() {
  Tensor z = nn::from_vector(std::vector<float>{1000, 999, -2, 3}, {2, 2});
  z.requires_grad_(true);
  const Tensor loss = sentence::SentenceCE(z, nn::from_vector(std::vector<int>{1, 0}, {2}));
  const auto   g = nn::grad(loss, {z})[0].to_vector<double>();
  const auto   logits = z.to_vector<double>();
  double       expected = 0;
  for (int n = 0; n < 2; ++n) {
    const double a = logits[2 * n], b = logits[2 * n + 1], m = std::max(a, b), sum = std::exp(a - m) + std::exp(b - m);
    const int    label = 1 - n;
    expected += (m + std::log(sum) - (label ? b : a)) / 2;
    for (int c = 0; c < 2; ++c)
      check::Near(g[2 * n + c], (std::exp((c ? b : a) - m) / sum - (c == label)) / 2, 1e-7, "all stable mean CE logit derivatives");
  }
  check::Near(loss.item<double>(), expected, 2e-7, "independent stable mean binary CE");
  std::cout << " extreme-logit stable CE/batch mean/all logit derivatives passed\n";
}

void FullChecks() {
  nn::ParameterStore p(353);
  std::vector<int>   ids(128);
  for (int i = 0; i < 128; ++i) ids[i] = 2 + (i * 17) % 7997;
  const auto model = sentence::KimCnn(p, nn::from_vector(ids, {2, 64}), nn::from_vector(std::vector<int>{64, 37}, {2}));
  const auto norms = sentence::SquaredNorms(nn::grad(nn::sum(nn::square(model.logits)), p.parameters()));
  check::Require(p.count() == 2760602 && p.size() == 9, "independent full CNN-rand parameter formula");
  for (int i = 0; i < 3; ++i) check::Require(model.maps[i].shape() == nn::Shape{2, 70 - i, 100}, "window3/4/5 map shape");
  check::Require(model.features.shape() == nn::Shape{2, 300} && model.logits.shape() == nn::Shape{2, 2},
                 "pooled concatenation and binary output");
  for (double n : norms) check::Require(std::isfinite(n) && n > 0, "all9 full parameter tensor gradients finite/nonzero");
  std::cout << " full CNN-rand2760602 parameters/9 tensors,E300,F100,map70/69/68,concat300;every learned tensor gradient passed\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int                   epochs = 3;
    bool                  checks = false;
    std::filesystem::path root = paths::Dataset("sentence");
    nn::set_num_threads(4);
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg.rfind("epochs=", 0) == 0) epochs = std::stoi(arg.substr(7));
      else if (arg.rfind("data=", 0) == 0) root = arg.substr(5);
      else if (arg.rfind("threads=", 0) == 0) nn::set_num_threads(std::stoi(arg.substr(8)));
      else if (arg == "checks_only=1") checks = true;
      else if (arg != "checks_only=0") throw std::invalid_argument("expected epochs=N data=PATH checks_only=0|1 threads=N");
    }
    HostChecks();
    OptimizerChecks();
    DropoutChecks();
    LossChecks();
    FullChecks();
    if (!checks) sentence::TrainSentence(root, epochs);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
