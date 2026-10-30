// CMD: just run-nn crnn_ctc
// Seven-convolution CRNN with two projected bidirectional LSTMs and a native
// CTC loss/adjoint (nn::ctc_loss, TF CTCLoss semantics, blank = 36). Human
// COCO-Text grayscale oracle word crops. Three compact epochs; report exact
// words and character edit rate honestly.
//
// Port of tflow's crnn_ctc. Preparation (in ~/projects/tflow_examples):
// just ocr-assets (output/datasets/ocr/ocr.npz: 43 fit / 7 validation / 6 test crops).
// Full model architecture and independent exhaustive-path CTC value/gradient
// checks run before the demonstration.
// Options: epochs=3 data=PATH checks_only=0|1.
// Output: output/nn/models/crnn_ctc/.

#include <chrono>
#include <fstream>
#include <memory>

#include <datamunge/nn/image_io.hpp>
#include <datamunge/nn/sequence.hpp>

#include "common/checks.hpp"
#include "common/crnn.hpp"
#include "common/ocr.hpp"
#include "common/paths.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;

namespace {

Tensor Ints(const std::vector<std::int64_t>& v, nn::Shape shape = {}) { return nn::from_vector(v, shape); }

// Probabilities of a [T][C] block of logits.
std::vector<std::vector<double>> Softmax(const std::vector<float>& z, int T, int C, int B, int n) {
  std::vector<std::vector<double>> p(T, std::vector<double>(C));
  for (int t = 0; t < T; ++t) {
    double sum = 0;
    for (int c = 0; c < C; ++c) sum += std::exp(z[(t * B + n) * C + c]);
    for (int c = 0; c < C; ++c) p[t][c] = std::exp(z[(t * B + n) * C + c]) / sum;
  }
  return p;
}

void CtcChecks() {
  const std::vector<float> values{.2f, -.3f, .4f, -.1f, .5f, .1f, .7f, -.2f, .3f};
  const Tensor             logits = nn::from_vector(values, {3, 1, 3});
  const auto out = nn::ctc_loss_with_gradient(logits, Ints({0, 0}, {1, 2}), Ints({2}), Ints({3}));
  const auto p   = Softmax(values, 3, 3, 1, 0);
  double     total = 0, occupation[3][3]{};
  int        paths = 0;
  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b)
      for (int c = 0; c < 3; ++c) {
        const std::vector<std::int64_t> path{a, b, c};
        if (nn::ctc_collapse(path, 2) != std::vector<std::int64_t>{0, 0}) continue;
        ++paths;
        const double pr = p[0][a] * p[1][b] * p[2][c];
        total += pr;
        for (int t = 0; t < 3; ++t) occupation[t][path[t]] += pr;
      }
  check::Require(paths == 1, "repeated label needs separating blank,unique minimal path");
  check::Near(out.loss.item<double>(), -std::log(total), 5e-7, "CTC sum over all 27 paths");
  const auto g     = out.gradient.to_vector<double>();
  double     error = 0;
  for (int t = 0; t < 3; ++t)
    for (int c = 0; c < 3; ++c) error = std::max(error, std::abs(p[t][c] - occupation[t][c] / total - g[t * 3 + c]));
  check::Near(error, 0, 1e-6, "EVERY CTC logit derivative versus exhaustive path posterior");
  check::Require(ocr::CtcCollapse({1, 1, 36, 1, 36, 1}) == "111" && ocr::CtcCollapse({36, 36}).empty(),
                 "merge repeats BEFORE removing blank");
  check::Require(ocr::OcrDistance("coffee", "cofee") == 1 && ocr::OcrDistance("abc", "xyz") == 3 &&
                     ocr::OcrDistance("", "abc") == 3,
                 "independent edit distance cases");
  std::cout << " exhaustive repeated-label CTC value/all-logit adjoints,blank/repeat order/edit-distance fixtures passed,error="
            << error << std::endl;
}

void VariableCtcChecks() {
  std::vector<float> values(18);
  for (int i = 0; i < 18; ++i) values[i] = static_cast<float>(.3 * std::sin(i * .7));
  Tensor     z = nn::from_vector(values, {3, 2, 3});
  const auto labels = Ints({0, -1, 0, 0}, {2, 2}), label_lengths = Ints({1, 2}), lengths = Ints({2, 3});
  z.requires_grad_(true);
  const Tensor mean    = nn::mean(nn::ctc_loss(z, labels, label_lengths, lengths));
  const auto   adjoint = nn::grad(mean, {z})[0].to_vector<double>();
  double       expected = 0, error = 0;
  int          accepted = 0;
  for (int n = 0; n < 2; ++n) {
    const int length = n ? 3 : 2;
    const auto p = Softmax(values, length, 3, 2, n);
    double posterior[3][3]{}, total = 0;
    int    combinations = 1;
    for (int t = 0; t < length; ++t) combinations *= 3;
    const std::vector<std::int64_t> target = n ? std::vector<std::int64_t>{0, 0} : std::vector<std::int64_t>{0};
    for (int k = 0; k < combinations; ++k) {
      int                       number = k;
      std::vector<std::int64_t> path;
      double                    pr = 1;
      for (int t = 0; t < length; ++t) {
        const int c = number % 3;
        number /= 3;
        path.push_back(c);
        pr *= p[t][c];
      }
      if (nn::ctc_collapse(path, 2) != target) continue;
      ++accepted;
      total += pr;
      for (int t = 0; t < length; ++t) posterior[t][path[t]] += pr;
    }
    expected -= std::log(total) / 2;
    for (int t = 0; t < 3; ++t)
      for (int c = 0; c < 3; ++c) {
        const double want = t < length ? (p[t][c] - posterior[t][c] / total) / 2 : 0;
        error             = std::max(error, std::abs(want - adjoint[(t * 2 + n) * 3 + c]));
      }
  }
  check::Require(accepted == 4, "multiple alternative alignments retained plus repeated-character path");
  check::Near(mean.item<double>(), expected, 1e-6, "batch mean CTC exhaustive-path NLL");
  check::Near(error, 0, 1e-6, "all mean CTC derivatives and zero padded-time gradient");
  // Independent finite differences of the scalar mean verify the reduction too.
  double      fd_error = 0;
  const float delta    = .002f;
  for (int i = 0; i < 18; ++i) {
    auto plus = values, minus = values;
    plus[i] += delta;
    minus[i] -= delta;
    nn::NoGradGuard guard;
    const double    hi = nn::mean(nn::ctc_loss(nn::from_vector(plus, {3, 2, 3}), labels, label_lengths, lengths)).item<double>();
    const double    lo = nn::mean(nn::ctc_loss(nn::from_vector(minus, {3, 2, 3}), labels, label_lengths, lengths)).item<double>();
    fd_error = std::max(fd_error, std::abs(static_cast<double>(static_cast<float>(hi) - static_cast<float>(lo)) / (2 * delta) - adjoint[i]));
  }
  check::Near(fd_error, 0, 2e-4, "every variable-time batch-mean CTC logit finite difference");
  std::cout << " multiple-alignment/variable-time/mean reduction/padded-gradient exhaustive error=" << error
            << ",all18 finite differences error=" << fd_error << std::endl;
}

void FullChecks() {
  nn::ParameterStore p(317), state(319);
  std::vector<float> input(32 * 100);
  for (std::size_t i = 0; i < input.size(); ++i) input[i] = static_cast<float>(.5 + .4 * std::sin(i * .17));
  const auto   model     = crnn::Crnn(p, state, nn::from_vector(input, {1, 32, 100, 1}), {}, false);
  const Tensor objective = nn::sum(nn::square(model.logits));
  const auto   gradients = nn::grad(objective, p.parameters());
  const std::vector<nn::Shape> shapes{{1, 32, 100, 64}, {1, 16, 50, 128}, {1, 8, 25, 256}, {1, 8, 25, 256},
                                      {1, 4, 26, 512},  {1, 4, 26, 512},  {1, 1, 26, 512}};
  for (int i = 0; i < 7; ++i) check::Require(model.stages[i].shape() == shapes[i], "official CRNN stage shape");
  check::Require(model.logits.shape() == nn::Shape{26, 1, 37}, "official CRNN logits shape");
  check::Require(p.count() == 8327205, "independent sum of conv,BN,LSTM and projection parameters");
  check::Require(p.size() == 36 && state.size() == 6, "seven biased convs,three BN,two projected BLSTMs");
  for (const auto& g : gradients) {
    const double norm = nn::sum(nn::square(g)).item<double>();
    check::Require(std::isfinite(norm) && norm > 0, "every full-width learned tensor has nonzero derivative");
  }
  std::cout << " full CRNN:" << p.count()
            << " parameters/36 tensors,6 BN state,all seven convolution shapes/T26/two BLSTMs and all-tensor gradients passed\n";
}

class OcrModel {
 public:
  nn::ParameterStore              parameters{307}, state{311};
  crnn::CrnnConfig                config{8, 32};
  std::unique_ptr<nn::Optimizer>  optimizer;

  explicit OcrModel(bool training = true) {
    nn::NoGradGuard guard;
    crnn::Crnn(parameters, state, nn::zeros({1, 32, 100, 1}), config, false);
    if (training) {
      nn::OptimizerConfig opt;
      opt.kind  = nn::OptimizerKind::Adam;
      optimizer = std::make_unique<nn::Optimizer>(parameters, .001, opt);
    }
  }
  static Tensor LogitLengths(std::int64_t n) { return nn::full({n}, ocr::kSteps, nn::DType::Int32); }
  /// Mean CTC objective; `update` runs a training-mode step (batch statistics,
  /// moving-average update, Adam), otherwise inference mode.
  double Loss(const ocr::OcrData& d, bool update = false) {
    double value;
    if (update) {
      const auto   out  = crnn::Crnn(parameters, state, d.pixels, config, true);
      const Tensor loss = nn::mean(nn::ctc_loss(out.logits, d.labels, d.lengths, LogitLengths(d.lengths.numel())));
      optimizer->step(nn::grad(loss, parameters.parameters()));
      value = loss.item<double>();
    } else {
      nn::NoGradGuard guard;
      const auto      out = crnn::Crnn(parameters, state, d.pixels, config, false);
      value = nn::mean(nn::ctc_loss(out.logits, d.labels, d.lengths, LogitLengths(d.lengths.numel()))).item<double>();
    }
    check::Require(std::isfinite(value) && value >= 0, "finite CTC NLL");
    return value;
  }
  /// Training-graph parameter gradients without updating anything.
  std::vector<Tensor> Gradients(const ocr::OcrData& d) {
    const auto   out  = crnn::Crnn(parameters, state, d.pixels, config, true, false);
    const Tensor loss = nn::mean(nn::ctc_loss(out.logits, d.labels, d.lengths, LogitLengths(d.lengths.numel())));
    return nn::grad(loss, parameters.parameters());
  }
  Tensor Predict(const ocr::OcrData& d) {
    nn::NoGradGuard guard;
    return crnn::Crnn(parameters, state, d.pixels, config, false).logits;
  }
  void Save(const std::filesystem::path& prefix) {
    parameters.save(prefix.string() + "_params.npz");
    state.save(prefix.string() + "_state.npz");
  }
  void Restore(const std::filesystem::path& prefix) {
    parameters.load(prefix.string() + "_params.npz");
    state.load(prefix.string() + "_state.npz");
  }
  std::vector<double> Snapshot() const {
    auto a = parameters.flatten(), b = state.flatten();
    a.insert(a.end(), b.begin(), b.end());
    return a;
  }
};

double MaxAbsDiff(const Tensor& a, const Tensor& b) {
  const auto x = a.to_vector<double>(), y = b.to_vector<double>();
  double     m = 0;
  for (std::size_t i = 0; i < x.size(); ++i) m = std::max(m, std::abs(x[i] - y[i]));
  return m;
}

void TrainOcr(const std::filesystem::path& root, int epochs) {
  check::Require(epochs > 0, "positive OCR epochs");
  const auto start     = std::chrono::steady_clock::now();
  auto       data      = ocr::LoadOcr(root);
  OcrModel   model;
  const auto directory = paths::OutputDir("models/crnn_ctc");
  const auto prefix    = directory / "best";
  const auto probe     = ocr::OcrRows(data[0], {0, 1, 2, 3});
  const double initial = model.Loss(probe);
  const auto   untrained = ocr::OcrDecode(model.Predict(data[2]));
  std::ofstream history(directory / "history.tsv");
  history << "epoch\ttrain_ctc\tvalidation_ctc\n";
  nn::BatchOrder order(static_cast<std::size_t>(data[0].pixels.shape()[0]), 8, 313);
  double         best     = INFINITY;
  int            selected = 0;
  std::cout << "CRNN compact: " << model.parameters.count() << " parameters/" << model.parameters.size()
            << " learned tensors + " << model.state.size() << " BN state tensors; " << epochs
            << " epochs,batch8,Adam.001,T26,C37,2 bidirectional LSTMs H32\n";
  // Every parameter's CTC adjoint must be finite, with nonzero evidence for
  // each convolution weight, BN affine map, direction and join projection.
  const auto gradients = model.Gradients(probe);
  int        active    = 0;
  for (const auto& g : gradients) {
    double norm = 0;
    for (double v : g.to_vector<double>()) {
      check::Require(std::isfinite(v), "finite CTC every-coordinate adjoint");
      norm += v * v;
    }
    active += norm > 1e-18;
  }
  check::Require(active >= 30, "CTC reaches both directions/layers and convolution stack");
  std::cout << " finite CTC gradients through all " << gradients.size() << " tensors;nonzero=" << active << std::endl;
  for (int epoch = 1; epoch <= epochs; ++epoch) {
    order.begin_epoch();
    double sum   = 0;
    int    count = 0;
    for (;;) {
      const auto rows = order.next();
      if (rows.empty()) break;
      sum += static_cast<double>(rows.size()) * model.Loss(ocr::OcrRows(data[0], rows), true);
      count += static_cast<int>(rows.size());
    }
    check::Require(count == data[0].pixels.shape()[0], "each fit crop once per epoch");
    const double validation = model.Loss(data[1]);
    if (validation < best) {
      best     = validation;
      selected = epoch;
      model.Save(prefix);
    }
    history << epoch << '\t' << sum / count << '\t' << validation << '\n';
    history.flush();
    std::cout << " epoch " << epoch << ":trainCTC=" << sum / count << ",validationCTC=" << validation << std::endl;
  }
  model.Restore(prefix);
  check::Near(model.Loss(data[1]), best, 1e-6, "restored validation CTC");
  const double learned = model.Loss(probe);
  check::Require(learned < initial, "short-run fixed fitting CTC improves");
  OcrModel deployed(false);
  deployed.Restore(prefix);
  const auto   before = deployed.Snapshot();
  const Tensor logits = deployed.Predict(data[2]), repeat = deployed.Predict(data[2]),
               training_inference = model.Predict(data[2]);
  check::Near(MaxAbsDiff(logits, training_inference), 0, 1e-7, "training-to-independent-inference checkpoint parity");
  check::Require(MaxAbsDiff(logits, repeat) == 0, "deterministic OCR inference");
  OcrModel reference(false);
  reference.Restore(prefix);
  check::Near(MaxAbsDiff(logits, reference.Predict(data[2])), 0, 1e-7, "independent restore logit parity");
  const Tensor single = deployed.Predict(ocr::OcrRows(data[2], {0}));
  check::Near(MaxAbsDiff(single, nn::slice_axis(logits, 1, 0, 1)), 0, 2e-6, "single/batch frozen BN+BLSTM inference");
  auto inverted   = ocr::OcrRows(data[2], {0});
  inverted.pixels = nn::sub(1.0, inverted.pixels);
  const double sensitivity = nn::mean(nn::abs(nn::sub(deployed.Predict(inverted), single))).item<double>();
  check::Require(sensitivity > 1e-7, "restored logits depend on observed image pixels despite blank greedy outputs");
  std::cout << " image inversion mean logit change=" << sensitivity << '\n';
  check::Require(before == deployed.Snapshot(), "all learned and moving BN tensors immutable");
  const auto predicted = ocr::OcrDecode(logits);
  ocr::OcrMetrics("trained greedy", data[2], predicted);
  ocr::OcrMetrics("untrained greedy", data[2], untrained);
  ocr::OcrMetrics("empty baseline", data[2], std::vector<std::string>(predicted.size()));
  std::ofstream words(directory / "predictions.tsv");
  words << "annotation_id\timage_id\treference\tpredicted\tuntrained\n";
  const auto notes = data[2].annotation_ids.to_vector<int>(), images = data[2].image_ids.to_vector<int>();
  for (std::size_t n = 0; n < predicted.size(); ++n) {
    words << notes[n] << '\t' << images[n] << '\t' << ocr::OcrText(data[2], static_cast<int>(n)) << '\t' << predicted[n]
          << '\t' << untrained[n] << '\n';
    std::cout << " test:" << ocr::OcrText(data[2], static_cast<int>(n)) << " -> [" << predicted[n] << "]\n";
  }
  nn::save_image_grid(data[2].pixels, (directory / "test_crops.png").string(), 3);
  nn::save_npz((directory / "test_outputs.npz").string(),
               {{"pixels", data[2].pixels}, {"labels", data[2].labels}, {"lengths", data[2].lengths},
                {"image_ids", data[2].image_ids}, {"annotation_ids", data[2].annotation_ids}, {"logits", logits}});
  if (std::filesystem::exists(root / "manifest.json"))
    std::filesystem::copy_file(root / "manifest.json", directory / "data_manifest.json",
                               std::filesystem::copy_options::overwrite_existing);
  check::Require(bool(history) && bool(words), "save OCR history/predictions");
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  std::cout << " selected=" << selected << ",fixed fitting CTC " << initial << " -> " << learned
            << "; independent restore/single-batch/determinism/immutable-state checks passed\nArtifacts:" << directory
            << " (training+audit " << seconds << " s)\nTiny oracle-box recognition demo,not benchmark quality.\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int                   epochs      = 3;
    bool                  checks_only = false;
    std::filesystem::path root        = paths::Dataset("ocr");
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg.rfind("epochs=", 0) == 0) epochs = std::stoi(arg.substr(7));
      else if (arg.rfind("data=", 0) == 0) root = arg.substr(5);
      else if (arg == "checks_only=1") checks_only = true;
      else if (arg != "checks_only=0") throw std::invalid_argument("expected epochs=N data=PATH checks_only=0|1");
    }
    CtcChecks();
    VariableCtcChecks();
    FullChecks();
    if (!checks_only) TrainOcr(root, epochs);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
