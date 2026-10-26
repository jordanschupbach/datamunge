// CMD: just run-nn lstm_forecast
// Chronological rolling-origin sunspot forecasts with fitting-only scaling.
//
// Port of tflow's lstm_forecast. Data: tflow's output/datasets/lstm_forecast
// (data.npz: yearly sunspot numbers cut into 12-year windows -> next 3 years,
// chronological fit/validation/test origins, normalization mean/sd from the
// fitting years only; fetched by tflow's scripts/fetch_datasets.py).
// An 8-unit LSTM over the normalized window and a dense 8 -> 3 head predict
// the three horizons directly (347 parameters); Adam .01, batch 8, three epochs
// (78 updates) on the normalized MSE.
// When tflow's recorded run (output/models/lstm_forecast) is present, the port
// first replays it from TF's initial weights: initial test outputs, first-batch
// gradients, the validation history and the final weights are compared.
// Then it trains from its own initialisation and audits restore, past-only
// prediction, future-target independence, temporal order and immutability
// like the TF version. The forecast plot (forecast.png) is not reproduced.
// Options: epochs=3 data=PATH tf=PATH.
// Output: output/nn/models/lstm_forecast/.

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>

#include "common/checks.hpp"
#include "common/lstm_forecast.hpp"
#include "common/paths.hpp"
#include "common/tf_replay.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;
using replay::Map;

namespace {

struct Model {
  nn::ParameterStore p;
  double             mean, sd;
  Model(std::uint64_t seed, double m, double s) : p(seed), mean(m), sd(s) {
    nn::NoGradGuard guard;
    forecast::LstmForecast(p, nn::zeros({1, 12, 1}));
    check::Require(p.count() == 347, "347 parameters");
  }
  Tensor Predict(const Map& b) {
    nn::NoGradGuard guard;
    return forecast::Predict(p, b.at("input"), mean, sd);
  }
  Tensor Loss(const Map& b) {
    return forecast::NormalizedMse(forecast::Predict(p, b.at("input"), mean, sd), b.at("target"), sd);
  }
  /// prediction, loss and d loss / d input (tflow's Evaluate fetches).
  Map Evaluate(const Map& b) {
    Tensor x = b.at("input").detach().clone();
    x.requires_grad_(true);
    const Tensor pred = forecast::Predict(p, x, mean, sd);
    const Tensor loss = forecast::NormalizedMse(pred, b.at("target"), sd);
    const auto   g    = nn::grad(loss, {x});
    return {{"prediction", pred.detach()}, {"loss", loss.detach()}, {"input_gradient", g[0]}};
  }
  std::vector<Tensor> Gradients(const Map& b) { return nn::grad(Loss(b), p.parameters()); }
};

/// Trains `epochs` epochs; returns the validation history.
std::vector<double> Train(Model& m, const Map& fit, const Map& validation, int epochs, std::ofstream* batches,
                          const std::filesystem::path* root) {
  nn::Optimizer       opt(m.p, .01, replay::Adam());
  nn::BatchOrder      order(fit.at("input").shape()[0], 8, 44807);
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
    nn::NoGradGuard guard;
    history.push_back(m.Loss(validation).item<double>());
  }
  return history;
}

void TfParity(const std::filesystem::path& tf, double mean, double sd, const Map& fit, const Map& validation,
              const Map& test, int epochs) {
  if (!std::filesystem::exists(tf / "initial_arrays.npz")) {
    std::cout << " (tflow run not found at " << tf << "; TF parity skipped)\n";
    return;
  }
  Model m(0, mean, sd);
  replay::LoadInto(m.p, tf / "initial_arrays.npz");
  const double out = replay::MaxDiff(m.Evaluate(test), tf / "initial_outputs.npz");
  const auto   fb  = replay::ToMap(nn::load_npz((tf / "first_batch.npz").string()));
  const double gd  = replay::MaxDiff(replay::Named(m.p, m.Gradients(fb)), tf / "initial_gradients.npz");
  check::Near(out, 0, 1e-3, "initial test predictions/loss/input gradients match TF");
  check::Near(gd, 0, 1e-5, "first-batch parameter gradients match TF");
  const auto hist = Train(m, fit, validation, epochs, nullptr, nullptr);
  const auto tfh  = replay::History(tf / "history.tsv");
  double     worst = 0;
  for (std::size_t e = 0; e < hist.size() && e < tfh.size(); ++e) {
    std::cout << "  replay epoch=" << e + 1 << " validationNormalizedMSE=" << hist[e] << " (TF " << tfh[e] << ")\n";
    worst = std::max(worst, std::abs(hist[e] - tfh[e]));
  }
  std::cout << " TF parity from TF's initial weights: initial outputs max diff=" << out
            << " (predictions in sunspot units), first-batch gradients max diff=" << gd;
  if (epochs == 3) {
    const double w = replay::MaxDiff(replay::Arrays(m.p), tf / "model_arrays.npz");
    std::cout << ", final weights max diff=" << w;
    check::Near(worst, 0, 1e-3, "validation history tracks TF's recorded run");
    check::Near(w, 0, 1e-3, "final weights track TF's after 78 Adam updates");
  }
  std::cout << "\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto            start = std::chrono::steady_clock::now();
    int                   epochs = 3;
    std::filesystem::path data = paths::Dataset("lstm_forecast"), tf = replay::TfModels("lstm_forecast");
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a.rfind("epochs=", 0) == 0) epochs = std::stoi(a.substr(7));
      else if (a.rfind("data=", 0) == 0) data = a.substr(5);
      else if (a.rfind("tf=", 0) == 0) tf = a.substr(3);
      else throw std::invalid_argument("epochs=N data=PATH tf=PATH");
    }
    check::Require(epochs > 0 && epochs <= 100, "positive epochs");
    const auto root = paths::OutputDir("models/lstm_forecast");
    replay::CopyAssets(data, root, {"data.npz", "manifest.json", "sunspots.csv"});
    const auto   all  = nn::load_npz((data / "data.npz").string());
    const auto   norm = replay::Get(all, "normalization").to_vector<double>();
    const double mean = static_cast<float>(norm[0]), sd = static_cast<float>(norm[1]);  // TF feeds float32 constants
    const std::vector<std::string> keys{"input", "target", "origins"};
    const Map fit = replay::Split(all, "fit", keys), validation = replay::Split(all, "validation", keys),
              test = replay::Split(all, "test", keys);
    TfParity(tf, mean, sd, fit, validation, test, epochs);

    Model m(44803, mean, sd);
    m.p.save((root / "initial_arrays.npz").string());
    nn::save_npz((root / "initial_outputs.npz").string(), replay::ToNamed(m.Evaluate(test)));
    std::ofstream history(root / "history.tsv"), batches(root / "batches.tsv");
    history << "epoch\tvalidation_normalized_mse\n";
    batches << "epoch\tstep\trow\n";
    const auto hist = Train(m, fit, validation, epochs, &batches, &root);
    for (std::size_t e = 0; e < hist.size(); ++e) {
      history << e + 1 << '\t' << hist[e] << '\n';
      std::cout << "epoch=" << e + 1 << " validationNormalizedMSE=" << hist[e] << std::endl;
    }
    const int updates = epochs * 26;
    const Map first   = replay::ToMap(nn::load_npz((root / "first_batch.npz").string()));
    nn::save_npz((root / "final_gradients.npz").string(), replay::ToNamed(replay::Named(m.p, m.Gradients(first))));
    m.p.save((root / "final.npz").string());
    Model deployed(0, mean, sd);
    deployed.p.load((root / "final.npz").string());
    const Map weights = replay::Arrays(deployed.p);
    check::Require(replay::SameBits(replay::Arrays(m.p), weights), "restore parity");
    const Map    out  = deployed.Evaluate(test);
    const Tensor pred = deployed.Predict(test);
    check::Require(replay::SameBits(pred, out.at("prediction")) && replay::SameBits(pred, deployed.Predict(test)),
                   "past-only repeat prediction");
    Map corrupt = replay::Rows(test, {0});
    corrupt.at("target") = nn::full_like(corrupt.at("target"), -1000);
    check::Require(replay::SameBits(deployed.Predict(corrupt), deployed.Predict(replay::Rows(test, {0}))),
                   "future targets cannot enter prediction");
    Map  reverse = replay::Rows(test, {0});
    auto x       = reverse.at("input").to_vector<float>();
    std::reverse(x.begin(), x.end());
    reverse.at("input") = nn::from_vector(x, {1, 12, 1});
    const auto change = deployed.Predict(reverse).to_vector<float>(),
               base   = deployed.Predict(replay::Rows(test, {0})).to_vector<float>();
    double effect = 0;
    for (int j = 0; j < 3; ++j) effect += std::abs(change[j] - base[j]);
    check::Require(effect > 1e-4, "temporal order matters");
    check::Require(replay::SameBits(replay::Arrays(deployed.p), weights), "immutable forecast");
    nn::save_npz((root / "outputs.npz").string(), replay::ToNamed(out));
    nn::save_npz((root / "model_arrays.npz").string(), replay::ToNamed(weights));
    const double test_mse = out.at("loss").item<double>();
    std::ofstream metrics(root / "metrics.json");
    metrics << "{\"parameters\":347,\"epochs\":" << epochs << ",\"updates\":" << updates
            << ",\"test_normalized_mse\":" << test_mse << ",\"reverse_effect\":" << effect << "}\n";
    std::cout << "testNormalizedMSE=" << test_mse << " (TF recorded 0.79072);pastOnly/restore/immutable/order PASS\n";
    std::cout << "wall " << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() << " s\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
