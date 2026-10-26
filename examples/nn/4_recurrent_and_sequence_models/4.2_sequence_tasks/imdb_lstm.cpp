// CMD: just run-nn imdb_lstm
// Real IMDB sentiment classification with an embedded length-masked LSTM; three cheap epochs.
//
// Port of tflow's imdb_lstm. Data: tflow's output/datasets/imdb_lstm
// (imdb_lstm.npz: pinned Keras IMDB reviews, 64 fit / 32 validation / 32 test,
// balanced labels, first 48 tokens re-indexed into a fit-only 256-word
// vocabulary with PAD 0 / UNK 1; vocabulary.txt).
// A learned 8-D embedding, an 8-unit LSTM (forget bias 1) whose states are
// frozen past each review's length, and an 8 -> 2 head; 2610 parameters,
// Adam .01, batch 16, three epochs (12 updates) on the softmax cross-entropy.
// When tflow's recorded run (output/models/imdb_lstm) is present, the port
// replays it from TF's initial weights (initial outputs incl. every BPTT
// partial, first-batch forward and gradients, history, final weights).
// The own-initialisation run then audits restore, label-free prediction,
// padding/zero-length/short-length invariance, causality of the forward
// states, invalid-id rejection, the frozen PAD embedding row and
// immutability, and compares with a fit-only naive Bayes baseline.
// Options: epochs=3 data=PATH out=PATH tf=PATH.
// Output: output/nn/models/imdb_lstm/.

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>

#include "common/checks.hpp"
#include "common/imdb_lstm.hpp"
#include "common/paths.hpp"
#include "common/tf_replay.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;
using replay::Map;
using namespace imdb;

namespace {

const std::vector<std::string> kPredictFields{"logits", "probability", "states", "final_state", "cell_states", "final_cell", "gates"};

struct Model {
  nn::ParameterStore p;
  explicit Model(std::uint64_t seed) : p(seed) {
    nn::NoGradGuard guard;
    const auto      z = nn::zeros({1, kUnits});
    ImdbLstm(p, nn::zeros({1, kSteps, kVocab}), nn::ones({1}, nn::DType::Int32), z, z);
    check::Require(p.count() == 2610 && p.size() == 6, "2610WordEmbeddingLstmHeadParameters");
  }
  static Tensor Encode(const Map& x) {
    for (int v : x.at("input").to_vector<int>()) check::Require(v >= 0 && v < kVocab, "validLocalWordID");
    return nn::one_hot(x.at("input"), kVocab);
  }
  static Tensor Zero(std::int64_t n) { return nn::zeros({n, kUnits}); }
  static Map Fields(const Path& path) {
    return {{"logits", path.logits.detach()},           {"probability", nn::softmax(path.logits, -1).detach()},
            {"states", path.states.detach()},           {"final_state", path.final_state.detach()},
            {"cell_states", path.cell_states.detach()}, {"final_cell", path.final_cell.detach()},
            {"gates", path.gates.detach()}};
  }
  static Tensor Loss(const Tensor& logits, const Tensor& labels) {
    return nn::neg(nn::mean(nn::sum(nn::mul(nn::one_hot(labels, 2), nn::log_softmax(logits, -1)), {1})));
  }
  /// Words and lengths only (no labels).
  Map Predict(const Map& x) {
    nn::NoGradGuard guard;
    const auto      n = x.at("input").shape()[0];
    return Fields(ImdbLstm(p, Encode(x), x.at("lengths"), Zero(n), Zero(n)));
  }
  Map Evaluate(const Map& x) {
    const auto n   = x.at("input").shape()[0];
    Tensor     enc = Encode(x), h0 = Zero(n), c0 = Zero(n);
    for (Tensor* t : {&enc, &h0, &c0}) t->requires_grad_(true);
    const Path   path   = ImdbLstm(p, enc, x.at("lengths"), h0, c0);
    const Tensor chosen = nn::sum(nn::mul(path.logits, nn::one_hot(x.at("labels"), 2)));
    nn::GradOptions o;
    o.allow_unused = true;
    const auto g   = nn::grad(std::vector<Tensor>{chosen}, {enc, h0, path.embeddings, c0}, {}, o);
    auto       def = [](const Tensor& t, const Tensor& like) { return t.defined() ? t : nn::zeros_like(like).detach(); };
    Map        out = Fields(path);
    out["loss"]                   = Loss(path.logits, x.at("labels")).detach();
    out["encoded_input_gradient"] = def(g[0], enc);
    out["initial_gradient"]       = def(g[1], h0);
    out["embedding_gradient"]     = def(g[2], path.embeddings);
    out["initial_cell_gradient"]  = def(g[3], c0);
    return out;
  }
  std::vector<Tensor> Gradients(const Map& x) {
    const auto n = x.at("input").shape()[0];
    return nn::grad(Loss(ImdbLstm(p, Encode(x), x.at("lengths"), Zero(n), Zero(n)).logits, x.at("labels")), p.parameters());
  }
};

double Accuracy(const Tensor& logits, const Tensor& labels) {
  const auto l = logits.to_vector<float>();
  const auto y = labels.to_vector<int>();
  int        correct = 0;
  for (std::size_t b = 0; b < y.size(); ++b) correct += (l[2 * b + 1] > l[2 * b]) == (y[b] == 1);
  return double(correct) / y.size();
}

/// Trains; returns {validation CE, accuracy} per epoch.
std::vector<std::pair<double, double>> Train(Model& m, const Map& fit, const Map& validation, int epochs,
                                             std::ofstream* batches, const std::filesystem::path* root) {
  nn::Optimizer  opt(m.p, .01, replay::Adam());
  nn::BatchOrder order(64, 16, 43019);
  std::vector<std::pair<double, double>> history;
  int            updates = 0;
  for (int e = 1; e <= epochs; ++e) {
    order.begin_epoch();
    for (;;) {
      const auto rows = order.next();
      if (rows.empty()) break;
      const auto b = replay::Rows(fit, rows);
      if (!updates && root) {
        nn::save_npz((*root / "first_batch.npz").string(), replay::ToNamed(b));
        nn::save_npz((*root / "first_forward.npz").string(), replay::ToNamed(m.Evaluate(b)));
        nn::save_npz((*root / "initial_gradients.npz").string(), replay::ToNamed(replay::Named(m.p, m.Gradients(b))));
      }
      opt.step(m.Gradients(b));
      if (!updates && root) m.p.save((*root / "first_after.npz").string());
      if (batches)
        for (auto r : rows) *batches << e << '\t' << updates << '\t' << r << '\n';
      ++updates;
    }
    const auto val = m.Evaluate(validation);
    history.emplace_back(val.at("loss").item<double>(), Accuracy(val.at("logits"), validation.at("labels")));
    if (root) {
      m.p.save((*root / ("epoch" + std::to_string(e) + "_arrays.npz")).string());
      nn::save_npz((*root / ("epoch" + std::to_string(e) + "_validation.npz")).string(), replay::ToNamed(val));
    }
  }
  return history;
}

void TfParity(const std::filesystem::path& tf, const Map& fit, const Map& validation, const Map& test, int epochs) {
  if (!std::filesystem::exists(tf / "initial_arrays.npz")) {
    std::cout << " (tflow run not found at " << tf << "; TF parity skipped)\n";
    return;
  }
  Model m(0);
  replay::LoadInto(m.p, tf / "initial_arrays.npz");
  const double out = replay::MaxDiff(m.Evaluate(test), tf / "initial_outputs.npz");
  auto         fb  = replay::ToMap(nn::load_npz((tf / "first_batch.npz").string()));
  const double fwd = replay::MaxDiff(m.Evaluate(fb), tf / "first_forward.npz");
  const double gd  = replay::MaxDiff(replay::Named(m.p, m.Gradients(fb)), tf / "initial_gradients.npz");
  check::Near(std::max(out, fwd), 0, 2e-6, "initial outputs, every BPTT partial and first-batch forward match TF");
  check::Near(gd, 0, 2e-6, "first-batch parameter gradients match TF");
  const auto hist = Train(m, fit, validation, epochs, nullptr, nullptr);
  const auto tfh = replay::History(tf / "history.tsv", 1), tfa = replay::History(tf / "history.tsv", 2);
  double     worst = 0;
  for (std::size_t e = 0; e < hist.size() && e < tfh.size(); ++e) {
    std::cout << "  replay epoch" << e + 1 << " validationCE=" << hist[e].first << " accuracy=" << hist[e].second
              << " (TF " << tfh[e] << ", " << tfa[e] << ")\n";
    worst = std::max(worst, std::abs(hist[e].first - tfh[e]));
  }
  std::cout << " TF parity from TF's initial weights: initial outputs/partials max diff=" << out
            << ", first-batch forward max diff=" << fwd << ", gradients max diff=" << gd;
  if (epochs == 3) {
    const double w = replay::MaxDiff(replay::Arrays(m.p), tf / "model_arrays.npz");
    std::cout << ", final weights max diff=" << w;
    check::Near(worst, 0, 1e-4, "validation history tracks TF's recorded run");
    check::Near(w, 0, 1e-4, "final weights track TF's after 12 Adam updates");
  }
  std::cout << "\n";
}

void SetIds(Map& x, const std::vector<std::int32_t>& ids) { x.at("input") = nn::from_vector(ids, x.at("input").shape()); }

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto            start = std::chrono::steady_clock::now();
    int                   epochs = 3;
    std::filesystem::path data = paths::Dataset("imdb_lstm"), root = paths::OutputRoot() / "models/imdb_lstm",
                          tf = replay::TfModels("imdb_lstm");
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a.rfind("epochs=", 0) == 0) epochs = std::stoi(a.substr(7));
      else if (a.rfind("data=", 0) == 0) data = a.substr(5);
      else if (a.rfind("out=", 0) == 0) root = a.substr(4);
      else if (a.rfind("tf=", 0) == 0) tf = a.substr(3);
      else throw std::invalid_argument("epochs=N data=PATH out=PATH tf=PATH");
    }
    check::Require(epochs > 0 && epochs <= 100, "threeCheapDefaultEpochs");
    std::filesystem::create_directories(root);
    const auto arrays = nn::load_npz((data / "imdb_lstm.npz").string());
    std::filesystem::copy_file(data / "imdb_lstm.npz", root / "data.npz", std::filesystem::copy_options::overwrite_existing);
    replay::CopyAssets(data, root, {"vocabulary.txt", "manifest.json"});
    std::vector<std::string> vocabulary;
    {
      std::ifstream vf(data / "vocabulary.txt");
      std::string   word;
      while (std::getline(vf, word)) vocabulary.push_back(word);
    }
    check::Require(vocabulary.size() == 256 && vocabulary[0] == "<pad>" && vocabulary[1] == "<unk>", "fitOnly256WordVocabulary");
    const std::vector<std::string> keys{"input", "lengths", "labels", "source_ids"};
    const Map fit = replay::Split(arrays, "fit", keys), validation = replay::Split(arrays, "validation", keys),
              test = replay::Split(arrays, "test", keys);
    TfParity(tf, fit, validation, test, epochs);

    Model m(43003);
    m.p.save((root / "initial_arrays.npz").string());
    nn::save_npz((root / "initial_outputs.npz").string(), replay::ToNamed(m.Evaluate(test)));
    std::ofstream history(root / "history.tsv"), batches(root / "batches.tsv");
    history.precision(17);
    history << "epoch\tvalidation_ce\tvalidation_accuracy\n";
    batches << "epoch\tstep\trow\n";
    const auto hist = Train(m, fit, validation, epochs, &batches, &root);
    for (std::size_t e = 0; e < hist.size(); ++e) {
      history << e + 1 << '\t' << hist[e].first << '\t' << hist[e].second << '\n';
      std::cout << "epoch" << e + 1 << " validationCE=" << hist[e].first << " accuracy=" << hist[e].second << std::endl;
    }
    const Map first = replay::ToMap(nn::load_npz((root / "first_batch.npz").string()));
    nn::save_npz((root / "final_gradients.npz").string(), replay::ToNamed(replay::Named(m.p, m.Gradients(first))));
    m.p.save((root / "final.npz").string());
    Model deployed(0);
    deployed.p.load((root / "final.npz").string());
    check::Require(replay::SameBits(replay::Arrays(m.p), replay::Arrays(deployed.p)), "allWordLstmWeightsRestoreBitexact");
    const Map before = replay::Arrays(deployed.p);
    const Map out = deployed.Evaluate(test), pred = deployed.Predict(test), repeat = deployed.Predict(test);
    for (const auto& [k, v] : pred) {
      check::Require(replay::SameBits(v, out.at(k)), "labelsNeverFedToSentimentPrediction");
      check::Require(replay::SameBits(v, repeat.at(k)), "sentimentPredictionRepeatBitexact");
    }
    nn::save_npz((root / "outputs.npz").string(), replay::ToNamed(out));
    nn::save_npz((root / "model_arrays.npz").string(), replay::ToNamed(before));
    {
      const auto single = deployed.Predict(replay::Rows(test, {0})).at("logits").to_vector<float>();
      const auto batch  = out.at("logits").to_vector<float>();
      for (int c = 0; c < 2; ++c) check::Near(single[c], batch[c], 2e-6, "singleAndBatchSentimentParity");
    }
    std::vector<std::int64_t> all(32);
    std::iota(all.begin(), all.end(), 0);
    Map        padded = replay::Rows(test, all), reversed = padded, last = padded;
    const auto ids = test.at("input").to_vector<std::int32_t>();
    const auto lengths = test.at("lengths").to_vector<int>();
    const auto enc_g = out.at("encoded_input_gradient").to_vector<float>();
    const auto states = out.at("states").to_vector<float>(), cells = out.at("cell_states").to_vector<float>();
    {
      auto pi = ids, ri = ids, li = ids;
      for (int b = 0; b < 32; ++b) {
        const int length = lengths[b];
        for (int t = 0; t < kSteps; ++t) {
          if (t >= length) {
            pi[b * kSteps + t] = 2 + (b + t) % 254;
            for (int c = 0; c < kVocab; ++c)
              check::Require(enc_g[(b * kSteps + t) * kVocab + c] == 0, "allPaddedEncodedWordPartialsExactlyZero");
            for (int c = 0; c < kUnits; ++c) {
              check::Require(states[(b * 49 + t + 1) * kUnits + c] == states[(b * 49 + t) * kUnits + c], "hiddenStateRetainedOnPaddedWords");
              check::Require(cells[(b * 49 + t + 1) * kUnits + c] == cells[(b * 49 + t) * kUnits + c], "cellStateRetainedOnPaddedWords");
            }
          } else {
            ri[b * kSteps + t] = ids[b * kSteps + length - 1 - t];
          }
        }
        if (length) li[b * kSteps + length - 1] = 1;
      }
      SetIds(padded, pi);
      SetIds(reversed, ri);
      SetIds(last, li);
    }
    const Map padout = deployed.Evaluate(padded), reverseout = deployed.Evaluate(reversed), lastout = deployed.Evaluate(last);
    check::Require(replay::SameBits(out, padout), "allSentimentOutputsAndPartialsIgnorePaddingIDs");
    {
      const auto ls = lastout.at("states").to_vector<float>();
      for (int b = 0; b < 32; ++b)
        for (int t = 0; t < lengths[b]; ++t)
          for (int c = 0; c < kUnits; ++c)
            check::Require(ls[(b * 49 + t) * kUnits + c] == states[(b * 49 + t) * kUnits + c], "changingLastWordCannotAffectEarlierStates");
    }
    for (const auto& [name, x] : std::vector<std::pair<std::string, Map*>>{{"padding", &padded}, {"reverse", &reversed}, {"last_word", &last}})
      nn::save_npz((root / (name + "_inputs.npz")).string(), replay::ToNamed(*x));
    nn::save_npz((root / "padding_outputs.npz").string(), replay::ToNamed(padout));
    nn::save_npz((root / "reverse_outputs.npz").string(), replay::ToNamed(reverseout));
    nn::save_npz((root / "last_word_outputs.npz").string(), replay::ToNamed(lastout));
    Map empty = replay::Rows(test, {0, 1});
    empty.at("lengths") = nn::zeros({2}, nn::DType::Int32);
    const Map emptyout = deployed.Evaluate(empty);
    for (auto key : {"states", "cell_states", "encoded_input_gradient", "embedding_gradient", "initial_cell_gradient"})
      for (float v : emptyout.at(key).to_vector<float>()) check::Require(v == 0, "zeroLengthReviewStateAndInputPartials");
    nn::save_npz((root / "zero_length_inputs.npz").string(), replay::ToNamed(empty));
    nn::save_npz((root / "zero_length_outputs.npz").string(), replay::ToNamed(emptyout));
    Map       shortened = replay::Rows(test, {0, 1, 2, 3}), short_padding = shortened;
    const int fixture_lengths[4] = {1, 7, 19, 31};
    {
      auto sp = shortened.at("input").to_vector<std::int32_t>();
      for (int b = 0; b < 4; ++b)
        for (int t = fixture_lengths[b]; t < kSteps; ++t) sp[b * kSteps + t] = 2 + (b + t) % 254;
      const Tensor fl = nn::from_vector(std::vector<std::int32_t>(fixture_lengths, fixture_lengths + 4));
      shortened.at("lengths") = fl;
      short_padding.at("lengths") = fl;
      SetIds(short_padding, sp);
    }
    const Map shortout = deployed.Evaluate(shortened), shortpadout = deployed.Evaluate(short_padding);
    check::Require(replay::SameBits(shortout, shortpadout), "nonVacuousShortReviewPaddingOutputAndGradientInvariance");
    {
      const auto g = shortout.at("encoded_input_gradient").to_vector<float>();
      const auto s = shortout.at("states").to_vector<float>(), c = shortout.at("cell_states").to_vector<float>();
      for (int b = 0; b < 4; ++b)
        for (int t = fixture_lengths[b]; t < kSteps; ++t) {
          for (int k = 0; k < kVocab; ++k) check::Require(g[(b * kSteps + t) * kVocab + k] == 0, "shortFixtureAllPaddingPartialsZero");
          for (int k = 0; k < kUnits; ++k)
            check::Require(s[(b * 49 + t + 1) * kUnits + k] == s[(b * 49 + t) * kUnits + k] &&
                               c[(b * 49 + t + 1) * kUnits + k] == c[(b * 49 + t) * kUnits + k],
                           "shortFixtureBothStatePathsRetained");
        }
    }
    nn::save_npz((root / "shortened_inputs.npz").string(), replay::ToNamed(shortened));
    nn::save_npz((root / "short_padding_inputs.npz").string(), replay::ToNamed(short_padding));
    nn::save_npz((root / "shortened_outputs.npz").string(), replay::ToNamed(shortout));
    nn::save_npz((root / "short_padding_outputs.npz").string(), replay::ToNamed(shortpadout));
    {
      Map  invalid = replay::Rows(test, {0});
      auto iv      = invalid.at("input").to_vector<std::int32_t>();
      iv[0]        = 256;
      SetIds(invalid, iv);
      check::Throws([&] { deployed.Predict(invalid); }, "invalidLocalWordIDRejected");
    }
    {
      const auto ia = nn::select(replay::Get(nn::load_npz((root / "initial_arrays.npz").string()), "embedding/W"), 0, 0);
      check::Require(replay::SameBits(ia, nn::select(deployed.p.at("embedding/W"), 0, 0)), "maskedPADembeddingRowNeverUpdated");
    }
    // fit-only multinomial naive Bayes over the valid words (PAD excluded)
    std::vector<double> nb(2 * kVocab, 1.0);
    {
      const auto fi = fit.at("input").to_vector<int>(), fl = fit.at("lengths").to_vector<int>(), fy = fit.at("labels").to_vector<int>();
      for (int b = 0; b < 64; ++b)
        for (int t = 0; t < fl[b]; ++t) nb[fy[b] * kVocab + fi[b * kSteps + t]] += 1;
      for (int c = 0; c < 2; ++c) {
        nb[c * kVocab] = 0;
        const double sum = std::accumulate(nb.begin() + c * kVocab + 1, nb.begin() + (c + 1) * kVocab, 0.0);
        for (int v = 1; v < kVocab; ++v) nb[c * kVocab + v] /= sum;
      }
    }
    nn::save_npz((root / "baseline.npz").string(), {{"class_word_probability", nn::from_vector(nb, {2, kVocab})}});
    std::ofstream predictions(root / "predictions.tsv");
    predictions << "source_id\tlength\tlabel\tpredicted\tnaive_bayes\n";
    int        nb_correct = 0;
    const auto logits = out.at("logits").to_vector<float>();
    const auto labels = test.at("labels").to_vector<int>(), sources = test.at("source_ids").to_vector<int>();
    for (int b = 0; b < 32; ++b) {
      double score[2] = {0, 0};
      for (int t = 0; t < lengths[b]; ++t)
        for (int c = 0; c < 2; ++c) score[c] += std::log(nb[c * kVocab + ids[b * kSteps + t]]);
      const int prediction = score[1] > score[0];
      nb_correct += prediction == labels[b];
      predictions << sources[b] << '\t' << lengths[b] << '\t' << labels[b] << '\t' << (logits[2 * b + 1] > logits[2 * b]) << '\t'
                  << prediction << '\n';
    }
    {  // embedding-gradient norms of the first 16 test reviews, one global scale, white = zero
      const auto          eg = out.at("embedding_gradient").to_vector<float>();
      std::vector<double> norm(16 * kSteps);
      double              peak = 1e-12;
      for (int b = 0; b < 16; ++b)
        for (int t = 0; t < kSteps; ++t) {
          double s = 0;
          for (int c = 0; c < 8; ++c) s += std::pow(double(eg[(b * kSteps + t) * 8 + c]), 2);
          norm[b * kSteps + t] = std::sqrt(s);
          peak                 = std::max(peak, norm[b * kSteps + t]);
        }
      std::vector<float> image(16 * 12 * kSteps * 10 * 3, 220.f);
      for (int b = 0; b < 16; ++b)
        for (int t = 0; t < kSteps; ++t) {
          const float shade = std::floor(255 * (1 - norm[b * kSteps + t] / peak) + .5);
          for (int yy = 0; yy < 11; ++yy)
            for (int xx = 0; xx < 9; ++xx)
              for (int ch = 0; ch < 3; ++ch) image[((b * 12 + yy) * kSteps * 10 + t * 10 + xx) * 3 + ch] = shade;
        }
      nn::save_png(nn::from_vector(image, {16 * 12, kSteps * 10, 3}), (root / "embedding_gradients.png").string(), 0, 255);
    }
    const double test_ce = out.at("loss").item<double>(), test_acc = Accuracy(out.at("logits"), test.at("labels"));
    std::ofstream metrics(root / "metrics.json");
    metrics.precision(17);
    metrics << "{\"parameters\":2610,\"parameter_tensors\":6,\"epochs\":" << epochs << ",\"optimizer_updates\":" << epochs * 4
            << ",\"test_ce\":" << test_ce << ",\"test_accuracy\":" << test_acc
            << ",\"reversed_prefix_accuracy\":" << Accuracy(reverseout.at("logits"), test.at("labels"))
            << ",\"last_word_unknown_accuracy\":" << Accuracy(lastout.at("logits"), test.at("labels"))
            << ",\"fit_naive_bayes_accuracy\":" << nb_correct / 32. << ",\"majority_accuracy\":0.5}\n";
    std::ofstream html(root / "index.html");
    html << "<!doctype html><meta charset='utf-8'><title>IMDB LSTM</title><style>body{font:16px system-ui;margin:30px;"
            "max-width:1100px}img{max-width:100%;image-rendering:pixelated}</style><h1>IMDB sentiment with a small LSTM</h1>"
            "<p>64 fit / 32 validation / 32 test pinned Keras IMDB reviews, first 48 tokens, fit-only 256-word vocabulary. "
            "Learned 8-D embedding, 8-unit length-masked LSTM, 8->2 head (2610 parameters), three epochs of Adam .01. "
            "No full-review benchmark or sentiment-quality claim.</p><h2>Embedding-gradient norms</h2><p>First 16 test "
            "reviews x 48 word positions; darker = larger true-target-logit gradient norm (one global scale, white zero).</p>"
            "<img src='embedding_gradients.png'><p>See metrics.json and predictions.tsv.</p>";
    check::Require(replay::SameBits(before, replay::Arrays(deployed.p)), "allSentimentInferenceControlsLeaveWeightsImmutable");
    check::Require(bool(metrics) && bool(html) && bool(history) && bool(batches) && bool(predictions), "completeImdbLstmRecords");
    std::cout << "testCE=" << test_ce << " accuracy=" << test_acc << " fitNaiveBayes=" << nb_correct / 32.
              << " (TF recorded testCE=0.690584 accuracy=0.625 NB=0.5625)"
              << ";LSTMprefix/BPTT/padding/causality/restore/labelFree/immutable PASS\nArtifacts:" << (root / "index.html").string()
              << "\nwall " << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() << " s\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
