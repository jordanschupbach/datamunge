// CMD: just run-nn ctc_handwriting
// CTC loss for unaligned real handwritten digit sequences, three compact epochs.
//
// Port of tflow's ctc_handwriting. Data: tflow's output/datasets/ctc_handwriting
// (data.npz: 1-3 real MNIST digits downsampled to 7x7 frames, written left to
// right as 8 frames of 49 pixels with a per-sequence frame count; 128 fit / 32
// validation / 32 test; labels are the digit strings without alignment).
// A dense+tanh frame encoder, an 8-unit length-masked LSTM and a linear head give
// 11 classes (blank = 10); nn::ctc_loss is TF's CTCLoss (merge_repeated, gradient
// from the forward-backward pass). 1043 parameters, Adam .01, batch 8, 48 updates.
// Checks: a variable-length / repeated-label CTC fixture (compared with TF's
// native fixture.npz when present); when tflow's recorded run is present, the
// initial outputs, first-batch gradients, history and final weights are replayed
// from TF's initial weights; then the own-initialisation run audits restore,
// label-free repeat inference, padding invariance of loss/logits/gradients and
// immutability, and reports greedy-decoding exact matches and CER.
// Options: epochs=3 data=PATH tf=PATH.
// Output: output/nn/models/ctc_handwriting/.

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>

#include <datamunge/nn/sequence.hpp>

#include "common/checks.hpp"
#include "common/ctc_handwriting.hpp"
#include "common/ocr.hpp"
#include "common/paths.hpp"
#include "common/tf_replay.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;
using replay::Map;
using namespace ctc_handwriting;

namespace {

struct Model {
  nn::ParameterStore p;
  explicit Model(std::uint64_t seed) : p(seed) {
    nn::NoGradGuard guard;
    CtcHandwriting(p, nn::zeros({1, kFrames, kPixels}), nn::ones({1}, nn::DType::Int32));
    check::Require(p.count() == 1043, "1043 handwriting parameters");
  }
  Tensor Loss(const Map& b, const Tensor& logits) {
    return nn::mean(nn::ctc_loss(logits, b.at("labels"), b.at("label_lengths"), b.at("lengths")));
  }
  Tensor Predict(const Map& b) {
    nn::NoGradGuard guard;
    return CtcHandwriting(p, b.at("input"), b.at("lengths"));
  }
  /// logits, mean CTC, the CTC adjoint d loss / d logits and d loss / d input.
  Map Evaluate(const Map& b) {
    Tensor x = b.at("input").detach().clone();
    x.requires_grad_(true);
    const Tensor logits = CtcHandwriting(p, x, b.at("lengths"));
    const Tensor loss   = Loss(b, logits);
    const auto   g      = nn::grad(loss, {logits, x});
    return {{"logits", logits.detach()}, {"loss", loss.detach()}, {"adjoint", g[0]}, {"input_gradient", g[1]}};
  }
  std::vector<Tensor> Gradients(const Map& b) {
    return nn::grad(Loss(b, CtcHandwriting(p, b.at("input"), b.at("lengths"))), p.parameters());
  }
};

std::vector<double> Train(Model& m, const Map& fit, const Map& validation, int epochs, std::ofstream* batches,
                          const std::filesystem::path* root) {
  nn::Optimizer       opt(m.p, .01, replay::Adam());
  nn::BatchOrder      order(128, 8, 44707);
  std::vector<double> history;
  int                 updates = 0;
  for (int e = 1; e <= epochs; ++e) {
    order.begin_epoch();
    for (;;) {
      const auto rows = order.next();
      if (rows.empty()) break;
      const auto b = replay::Rows(fit, rows);
      if (!updates && root) {
        nn::save_npz((*root / "first_batch.npz").string(), replay::ToNamed(b));
        nn::save_npz((*root / "initial_gradients.npz").string(), replay::ToNamed(replay::Named(m.p, m.Gradients(b))));
      }
      opt.step(m.Gradients(b));
      if (batches)
        for (auto r : rows) *batches << e << '\t' << updates << '\t' << r << '\n';
      ++updates;
    }
    history.push_back(m.Evaluate(validation).at("loss").item<double>());
  }
  return history;
}

void Fixture(const std::filesystem::path& root, const std::filesystem::path& tf) {
  std::vector<float> lv(48);
  for (int i = 0; i < 48; ++i) lv[i] = static_cast<float>(.4 * std::sin(i * .7));
  const Tensor logits  = nn::from_vector(lv, {4, 4, 3});
  const Tensor labels  = nn::from_vector(std::vector<std::int32_t>{0, -1, 0, 0, 0, 1, 1, -1}, {4, 2});
  const Tensor llen    = nn::from_vector(std::vector<std::int32_t>{1, 2, 2, 1});
  const Tensor lengths = nn::from_vector(std::vector<std::int32_t>{1, 3, 4, 2});
  const auto   r       = nn::ctc_loss_with_gradient(logits, labels, llen, lengths);
  // Hand check of the length-1 case: frame 0 must emit label 0, P = softmax(logits[0, 0])[0].
  {
    const double a = lv[0], b = lv[1], c = lv[2];
    const double nll = std::log(std::exp(a) + std::exp(b) + std::exp(c)) - a;
    check::Near(r.loss.to_vector<double>()[0], nll, 1e-6, "single-frame CTC equals -log softmax of the label");
  }
  for (std::size_t i = 0; i < 4; ++i) {
    const auto g = nn::select(r.gradient, 1, static_cast<std::int64_t>(i)).to_vector<double>();
    const int  n = lengths.to_vector<int>()[i];
    for (int t = 0; t < 4; ++t) {
      double s = 0;
      for (int c = 0; c < 3; ++c) s += g[t * 3 + c];
      check::Near(s, 0, 1e-6, "CTC logit gradient sums to zero per frame");
      if (t >= n)
        for (int c = 0; c < 3; ++c) check::Require(g[t * 3 + c] == 0, "frames past the length get zero gradient");
    }
  }
  nn::save_npz((root / "fixture.npz").string(), {{"logits", logits}, {"labels", labels}, {"label_lengths", llen},
                                                 {"lengths", lengths}, {"loss", r.loss.detach()}, {"gradient", r.gradient}});
  std::cout << " CTC fixture: variable time, repeated labels (\"00\" needs a blank), zero-gradient padding frames";
  if (std::filesystem::exists(tf / "fixture.npz")) {
    const auto ref = nn::load_npz((tf / "fixture.npz").string());
    const double dl = replay::MaxAbsDiff(r.loss, replay::Get(ref, "loss")),
                 dg = replay::MaxAbsDiff(r.gradient, replay::Get(ref, "gradient"));
    check::Near(std::max(dl, dg), 0, 2e-6, "CTC fixture loss and gradient match TF's native CTCLoss");
    std::cout << "; vs TF native loss diff=" << dl << " gradient diff=" << dg;
  }
  std::cout << "\n";
}

void TfParity(const std::filesystem::path& tf, const Map& fit, const Map& validation, const Map& test, int epochs) {
  if (!std::filesystem::exists(tf / "initial_arrays.npz")) {
    std::cout << " (tflow run not found at " << tf << "; TF parity skipped)\n";
    return;
  }
  Model m(0);
  replay::LoadInto(m.p, tf / "initial_arrays.npz");
  const double out = replay::MaxDiff(m.Evaluate(test), tf / "initial_outputs.npz");
  const auto   fb  = replay::ToMap(nn::load_npz((tf / "first_batch.npz").string()));
  const double gd  = replay::MaxDiff(replay::Named(m.p, m.Gradients(fb)), tf / "initial_gradients.npz");
  check::Near(out, 0, 2e-5, "initial test logits/CTC/adjoint/input gradients match TF");
  check::Near(gd, 0, 2e-5, "first-batch parameter gradients match TF");
  const auto hist = Train(m, fit, validation, epochs, nullptr, nullptr);
  const auto tfh  = replay::History(tf / "history.tsv");
  double     worst = 0;
  for (std::size_t e = 0; e < hist.size() && e < tfh.size(); ++e) {
    std::cout << "  replay epoch=" << e + 1 << " validationCTC=" << hist[e] << " (TF " << tfh[e] << ")\n";
    worst = std::max(worst, std::abs(hist[e] - tfh[e]));
  }
  std::cout << " TF parity from TF's initial weights: initial outputs max diff=" << out
            << ", first-batch gradients max diff=" << gd;
  if (epochs == 3) {
    const double w = replay::MaxDiff(replay::Arrays(m.p), tf / "model_arrays.npz");
    std::cout << ", final weights max diff=" << w;
    check::Near(worst, 0, 1e-3, "validation CTC history tracks TF's recorded run");
    check::Near(w, 0, 1e-3, "final weights track TF's after 48 Adam updates");
  }
  std::cout << "\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto            start = std::chrono::steady_clock::now();
    int                   epochs = 3;
    std::filesystem::path data = paths::Dataset("ctc_handwriting"), tf = replay::TfModels("ctc_handwriting");
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a.rfind("epochs=", 0) == 0) epochs = std::stoi(a.substr(7));
      else if (a.rfind("data=", 0) == 0) data = a.substr(5);
      else if (a.rfind("tf=", 0) == 0) tf = a.substr(3);
      else throw std::invalid_argument("epochs=N data=PATH tf=PATH");
    }
    check::Require(epochs > 0 && epochs <= 100, "positive epochs");
    const auto root = paths::OutputDir("models/ctc_handwriting");
    replay::CopyAssets(data, root, {"data.npz", "manifest.json"});
    const auto all = nn::load_npz((data / "data.npz").string());
    const std::vector<std::string> keys{"input", "labels", "lengths", "label_lengths", "source_ids"};
    const Map fit = replay::Split(all, "fit", keys), validation = replay::Split(all, "validation", keys),
              test = replay::Split(all, "test", keys);
    Fixture(root, tf);
    TfParity(tf, fit, validation, test, epochs);

    Model m(44703);
    m.p.save((root / "initial_arrays.npz").string());
    nn::save_npz((root / "initial_outputs.npz").string(), replay::ToNamed(m.Evaluate(test)));
    std::ofstream history(root / "history.tsv"), batches(root / "batches.tsv");
    history << "epoch\tvalidation_ctc\n";
    batches << "epoch\tstep\trow\n";
    const auto hist = Train(m, fit, validation, epochs, &batches, &root);
    for (std::size_t e = 0; e < hist.size(); ++e) {
      history << e + 1 << '\t' << hist[e] << '\n';
      std::cout << "epoch=" << e + 1 << " validationCTC=" << hist[e] << std::endl;
    }
    const Map first = replay::ToMap(nn::load_npz((root / "first_batch.npz").string()));
    nn::save_npz((root / "final_gradients.npz").string(), replay::ToNamed(replay::Named(m.p, m.Gradients(first))));
    m.p.save((root / "final.npz").string());
    Model deployed(0);
    deployed.p.load((root / "final.npz").string());
    const Map weights = replay::Arrays(deployed.p);
    check::Require(replay::SameBits(replay::Arrays(m.p), weights), "restore bitexact");
    const Map    out  = deployed.Evaluate(test);
    const Tensor pred = deployed.Predict(test);
    check::Require(replay::SameBits(pred, out.at("logits")) && replay::SameBits(pred, deployed.Predict(test)),
                   "label-free repeat inference");
    Map pad = replay::Rows(test, {0, 1});
    {
      auto       x = pad.at("input").to_vector<float>();
      const auto n = pad.at("lengths").to_vector<int>();
      for (int b = 0; b < 2; ++b)
        for (int t = n[b]; t < kFrames; ++t)
          for (int k = 0; k < kPixels; ++k) x[(b * kFrames + t) * kPixels + k] = .77f;
      pad.at("input") = nn::from_vector(x, {2, kFrames, kPixels});
    }
    const Map normal = deployed.Evaluate(replay::Rows(test, {0, 1})), padded = deployed.Evaluate(pad);
    check::Require(!replay::SameBits(pad.at("input"), replay::Rows(test, {0, 1}).at("input")), "nonVacuousPadding");
    check::Require(replay::SameBits(normal, padded), "padding loss/logits/gradients invariant");
    nn::save_npz((root / "padding_inputs.npz").string(), replay::ToNamed(pad));
    nn::save_npz((root / "padding_outputs.npz").string(), replay::ToNamed(padded));
    nn::save_npz((root / "outputs.npz").string(), replay::ToNamed(out));
    nn::save_npz((root / "model_arrays.npz").string(), replay::ToNamed(weights));
    nn::CtcOptions greedy;
    const auto     decoded = nn::ctc_greedy_decode(pred, test.at("lengths"), greedy);
    const auto     labels = test.at("labels").to_vector<int>(), llen = test.at("label_lengths").to_vector<int>();
    int            edits = 0, chars = 0, exact = 0;
    std::ofstream  predictions(root / "predictions.tsv");
    predictions << "case\tgold\tprediction\n";
    for (int b = 0; b < 32; ++b) {
      std::string text, gold;
      for (auto c : decoded.sequences[b]) text += char('0' + c);
      for (int j = 0; j < llen[b]; ++j) gold += char('0' + labels[b * 3 + j]);
      edits += ocr::OcrDistance(gold, text);
      chars += static_cast<int>(gold.size());
      exact += text == gold;
      predictions << b << '\t' << gold << '\t' << text << '\n';
    }
    check::Require(replay::SameBits(replay::Arrays(deployed.p), weights), "immutable inference");
    const double  test_ctc = out.at("loss").item<double>();
    std::ofstream metrics(root / "metrics.json");
    metrics << "{\"parameters\":1043,\"epochs\":" << epochs << ",\"updates\":" << epochs * 16 << ",\"test_ctc\":" << test_ctc
            << ",\"exact\":" << exact << ",\"edits\":" << edits << ",\"characters\":" << chars << "}\n";
    std::cout << "testCTC=" << test_ctc << " exact=" << exact << "/32 CER=" << double(edits) / chars
              << " (TF recorded testCTC=6.55194 exact=0/32 CER=1);restore/padding/labelFree/immutable PASS\n";
    std::cout << "wall " << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() << " s\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
