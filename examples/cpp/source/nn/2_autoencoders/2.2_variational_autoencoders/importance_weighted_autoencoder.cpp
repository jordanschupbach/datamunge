// CMD: just run-nn importance_weighted_autoencoder
// Matched K1/K5 IWAE training with sampled density ratios and common K20 selection.
// Three default epochs.
//
// Port of tflow's importance_weighted_autoencoder. Preparation (in ~/projects/tflow_examples):
// just beta-vae-assets; data are read from $DATAMUNGE_DATA_DIR/beta_vae. Outputs go to
// output/nn/models/importance_weighted_autoencoder (k1/ and k5/ subdirectories, index.html,
// case<k>.png). Checkpoints are best.npz instead of SaveV2 bundles. Options: epochs=N data=PATH
// out=PATH init=PATH (shared initial parameters, e.g. tflow's k1/initial_arrays.npz, to follow
// its trajectory exactly).

#include <chrono>
#include <iostream>
#include <limits>

#include "common/importance_weighted_autoencoder.hpp"

namespace {

namespace nn = datamunge::nn;

nn::Tensor Prefix(const nn::Tensor& t, std::int64_t n) { return nn::slice_axis(t, 0, 0, n).contiguous(); }

void Checks(const std::filesystem::path& root) {
  const std::vector<float> wv{1000, -1000, 1001, -1001, 999, -999};
  auto w        = nn::from_vector(wv, {3, 2}).requires_grad_();
  auto bound    = -ae::IwaeLogMeanExp(w, 3);
  auto gradient = nn::grad(nn::mean(bound, {0}), {w})[0];
  auto repeated = ae::IwaeLogMeanExp(nn::tensor({{-700., -800.}, {-700., -800.}, {-700., -800.}}), 3);
  auto single   = ae::IwaeLogMeanExp(nn::tensor({{-700., -800.}}), 1);
  const auto bv = bound.to_vector<float>(), gv = gradient.to_vector<float>(), rv = repeated.to_vector<float>(),
             sv = single.to_vector<float>();
  for (int b = 0; b < 2; ++b) {
    double maximum = -std::numeric_limits<double>::infinity();
    for (int k = 0; k < 3; ++k) maximum = std::max(maximum, double(wv[k * 2 + b]));
    double sum = 0;
    for (int k = 0; k < 3; ++k) sum += std::exp(wv[k * 2 + b] - maximum);
    check::Near(bv[b], -maximum - std::log(sum / 3), 5e-5, "stable importance bound");
    for (int k = 0; k < 3; ++k)
      check::Near(gv[k * 2 + b], -.5 * std::exp(wv[k * 2 + b] - maximum) / sum, 1e-7, "normalized importance gradient");
    check::Near(rv[b], sv[b], 0, "duplicated identical samples leave bound unchanged");
  }
  nn::save_npz((root / "algebra.npz").string(), {{"log_weights", w.detach()},
                                                 {"negative_bound", bound.detach()},
                                                 {"gradient", gradient},
                                                 {"duplicate", repeated},
                                                 {"single", single}});
  std::cout << "IWAE stable extremes/K1/duplicate samples/all normalized partials passed" << std::endl;
}

std::vector<double> Score(ae::ImportanceWeightedModel& model, const nn::Tensor& images, const ae::VaeNoiseBank& noise) {
  auto   out = model.Evaluate(images, noise.epsilon);
  double ess = 0;
  for (float v : out[10].to_vector<float>()) ess += v;
  return {double(out[6].item<float>()), ess / double(images.size(0))};
}

double Train(int samples, int epochs, const std::vector<ae::BetaFactorData>& data,
             const std::vector<ae::VaeNoiseBank>& training, const std::vector<ae::VaeNoiseBank>& noise,
             const ae::VaeNoiseBank& prior, const std::filesystem::path& root, const nn::NamedTensors& initial) {
  const auto path = root / ("k" + std::to_string(samples));
  std::filesystem::create_directories(path);
  ae::ImportanceWeightedModel model;
  model.Load(initial);
  check::Require(model.parameters.count() == 51416 && model.parameters.size() == 10, "same compact VAE census");
  nn::save_npz((path / "initial_arrays.npz").string(), ae::Arrays(model.parameters));
  nn::BatchOrder order(210, 32, 2243);
  std::ofstream  history(path / "history.tsv"), batches(path / "batches.tsv");
  history.precision(17);
  history << "epoch\ttrain_preupdate_negative_bound\tvalidation_k20_negative_bound\tvalidation_k20_ess\n";
  batches << "epoch\tstep\trow\tsource_id\n";
  const auto fit_source = data[0].source.to_vector<int>();
  const auto best_path  = (path / "best.npz").string();
  double     best       = std::numeric_limits<double>::infinity();
  int        selected = 0, step = 0;
  const auto started = std::chrono::steady_clock::now();
  for (int epoch = 1; epoch <= epochs; ++epoch) {
    order.begin_epoch();
    double sum  = 0;
    int    seen = 0;
    for (;;) {
      auto rows = order.next();
      if (rows.empty()) break;
      auto images  = nn::take_rows(data[0].images, rows);
      auto epsilon = ae::IwaeNoiseRows(training[epoch - 1], rows, samples);
      if (step == 0) {
        nn::save_npz((path / "first_batch.npz").string(), {{"images", images}, {"epsilon", epsilon}});
        auto g = ae::CheckGradients(model.parameters, model.Gradient(images, epsilon), "all IWAE derivatives finite",
                                    "all ten tensors receive sampled-density IWAE gradients");
        nn::save_npz((path / "first_gradients.npz").string(), g);
        auto f = model.Evaluate(images, epsilon);
        nn::save_npz((path / "first_forward.npz").string(),
                     {{"log_weights", f[8]}, {"normalized_weights", f[9]}, {"negative_bound", f[7]}, {"ess", f[10]}});
      }
      sum += double(rows.size()) * model.Update(images, epsilon);
      seen += int(rows.size());
      for (auto row : rows) batches << epoch << '\t' << step << '\t' << row << '\t' << fit_source[row] << '\n';
      if (step == 0) nn::save_npz((path / "first_update_arrays.npz").string(), ae::Arrays(model.parameters));
      ++step;
    }
    check::Require(seen == 210, "all fitting images once per epoch");
    auto score = Score(model, data[1].images, noise[1]);
    history << epoch << '\t' << sum / seen << '\t' << score[0] << '\t' << score[1] << '\n';
    nn::save_npz((path / ("epoch" + std::to_string(epoch) + "_arrays.npz")).string(), ae::Arrays(model.parameters));
    if (score[0] < best) {
      best     = score[0];
      selected = epoch;
      model.parameters.save(best_path);  // COMMON fixed K20 validation bound selection
    }
    std::cout << "trainK" << samples << " epoch" << epoch << " common validationK20negativebound=" << score[0]
              << ",ESS=" << score[1] << std::endl;
  }
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  ae::ImportanceWeightedModel deployed(false);
  deployed.parameters.load(best_path);
  check::Near(Score(deployed, data[1].images, noise[1])[0], best, 1e-8, "fresh common K20 selected score");
  const auto       before = ae::Snapshot(deployed.parameters);
  nn::NamedTensors output;
  for (int split = 0; split < 3; ++split) {
    const std::int64_t n    = data[split].images.size(0);
    const std::string  name = split == 0 ? "fit" : split == 1 ? "validation" : "test";
    auto mean = deployed.Evaluate(data[split].images, nn::zeros({1, n, 4})),
         draw = deployed.Evaluate(data[split].images, noise[split].epsilon);
    check::Require(ae::Bytes(mean[0]) == ae::Bytes(mean[2]), "zero epsilon gives posterior mean");
    auto       mu = Prefix(draw[0], n), lv = Prefix(draw[1], n);
    const auto a = mu.to_vector<float>(), b = lv.to_vector<float>(), ma = mean[0].to_vector<float>(),
               mb = mean[1].to_vector<float>();
    for (std::int64_t i = 0; i < n * 4; ++i) {
      check::Near(a[i], ma[i], 4e-6, "mu independent of noise/K up to GEMM roundoff");
      check::Near(b[i], mb[i], 4e-6, "logvariance independent of noise/K up to GEMM roundoff");
    }
    output.emplace_back(name + "_mu", mu);
    output.emplace_back(name + "_log_variance", lv);
    output.emplace_back(name + "_mean_prediction", mean[3]);
    output.emplace_back(name + "_sample_z", draw[2]);
    output.emplace_back(name + "_sample_prediction", draw[3]);
    output.emplace_back(name + "_sample_nll", draw[4]);
    output.emplace_back(name + "_kl", Prefix(draw[5], n));
    output.emplace_back(name + "_log_weights", draw[8]);
    output.emplace_back(name + "_normalized_weights", draw[9]);
    output.emplace_back(name + "_negative_bound", draw[7]);
    output.emplace_back(name + "_ess", draw[10]);
  }
  auto first_noise = ae::IwaeNoiseRows(noise[2], ae::IwaeRows(42), 1);
  auto a = deployed.Evaluate(data[2].images, first_noise), again = deployed.Evaluate(data[2].images, first_noise);
  check::Require(ae::Bytes(a[3]) == ae::Bytes(again[3]) && ae::Bytes(a[3]) == ae::Bytes(deployed.Decode(a[2])),
                 "repeat noise and target-free latent decode exact");
  auto single = deployed.Evaluate(nn::take_rows(data[2].images, {0}), ae::IwaeNoiseRows(noise[2], {0}, 1));
  const double parity = ae::MaxAbsDiff(single[3], a[3], 784);
  check::Require(parity < 2e-6, "single/batch decode parity");
  auto other = deployed.Evaluate(data[2].images, nn::zeros(first_noise.shape()));
  check::Require(ae::Bytes(a[2]) != ae::Bytes(other[2]) && ae::Bytes(a[3]) != ae::Bytes(other[3]),
                 "noise changes samples and probabilities");
  auto prior_z    = ae::BetaVaeNoiseAt(prior, 0);
  auto prediction = deployed.Decode(prior_z);
  nn::save_npz((path / "prior.npz").string(), {{"z", prior_z}, {"probability", prediction}});
  ae::Mosaic(path / "prior_means.png", prediction, 5, 3);
  auto r2 = ae::BetaFactorProbe(path, ae::Find(output, "fit_mu"), ae::Find(output, "test_mu"), data[0].factors,
                                data[2].factors, ae::Find(output, "fit_log_variance"));
  check::Require(before == ae::Snapshot(deployed.parameters), "all inference and diagnostics immutable");
  nn::save_npz((path / "outputs.npz").string(), output);
  nn::save_npz((path / "model_arrays.npz").string(), ae::Arrays(deployed.parameters));
  nn::NamedTensors    estimates;
  std::vector<double> scores;
  for (int k : {1, 5, 20}) {
    auto f = deployed.Evaluate(data[2].images, ae::IwaeNoiseRows(noise[2], ae::IwaeRows(42), k));
    scores.push_back(f[6].item<float>());
    estimates.emplace_back("negative_bound_k" + std::to_string(k), f[7]);
    estimates.emplace_back("ess_k" + std::to_string(k), f[10]);
  }
  nn::save_npz((path / "test_estimates.npz").string(), estimates);
  const double  mse = ae::Mse(ae::Find(output, "test_mean_prediction"), data[2].images);
  std::ofstream meta(path / "metrics.json");
  meta.precision(17);
  meta << "{\"training_samples\":" << samples << ",\"evaluation_samples\":20,\"parameters\":51416,\"epochs\":" << epochs
       << ",\"updates\":" << step << ",\"selected_epoch\":" << selected << ",\"test_negative_bound_k1\":" << scores[0]
       << ",\"test_negative_bound_k5\":" << scores[1] << ",\"test_negative_bound_k20\":" << scores[2]
       << ",\"mean_decode_test_mse\":" << mse << ",\"single_batch_max_error\":" << parity
       << ",\"training_seconds\":" << seconds << "}\n";
  check::Require(bool(meta) && bool(history) && bool(batches), "complete IWAE evidence");
  std::cout << "trainK" << samples << " selected" << selected << ",test negative bounds K1/5/20=" << scores[0] << '/'
            << scores[1] << '/' << scores[2] << ",decode(mu)MSE=" << mse << ",ridge R2 shape/size/x/y=" << r2[1][0]
            << '/' << r2[1][1] << '/' << r2[1][2] << '/' << r2[1][3] << std::endl;
  return seconds;
}

void Run(const std::filesystem::path& data, const std::filesystem::path& root, int epochs, const std::string& init) {
  auto pools = ae::LoadBetaFactors(data);
  std::filesystem::create_directories(root);
  Checks(root);
  ae::SaveFactorData(root, pools);
  std::vector<ae::VaeNoiseBank> training, noise{ae::BetaVaeNoise(210, 20, 2411), ae::BetaVaeNoise(42, 20, 2417),
                                                ae::BetaVaeNoise(42, 20, 2423)};
  auto             prior = ae::BetaVaeNoise(20, 1, 2429);
  nn::NamedTensors banks{{"prior_standard", prior.standard}, {"prior_epsilon", prior.epsilon}};
  for (int i = 0; i < 3; ++i) {
    const std::string n = i == 0 ? "fit" : i == 1 ? "validation" : "test";
    banks.emplace_back(n + "_standard", noise[i].standard);
    banks.emplace_back(n + "_epsilon", noise[i].epsilon);
  }
  nn::save_npz((root / "fixed_noise.npz").string(), banks);
  for (int e = 1; e <= epochs; ++e) {
    training.push_back(ae::BetaVaeNoise(210, 5, 2437 + e));
    nn::save_npz((root / ("train_noise_epoch" + std::to_string(e) + ".npz")).string(),
                 {{"standard", training.back().standard}, {"epsilon", training.back().epsilon}});
  }
  ae::BetaVariationalAutoencoderModel base(1, false);
  if (!init.empty()) base.parameters.load(init);
  const auto initial = ae::Arrays(base.parameters);
  double     seconds = Train(1, epochs, pools, training, noise, prior, root, initial);
  seconds += Train(5, epochs, pools, training, noise, prior, root, initial);
  auto one = nn::load_npz((root / "k1/outputs.npz").string()), five = nn::load_npz((root / "k5/outputs.npz").string());
  auto mean = ae::PixelMean(pools[0].images);
  std::ofstream html(root / "index.html");
  html << "<!doctype html><meta charset='utf-8'><title>IWAE K1 versus K5</title><style>body{font:16px "
          "system-ui;margin:30px;max-width:1000px}img{max-width:100%;image-rendering:pixelated}</style><h1>Importance-"
          "weighted autoencoder</h1><p>Matched K1/K5 training:51416parameters,three epochs by default,21updates EACH. "
          "Tiny294-image known-factor binary lattice,210fit/42validation/42test. Factors ONLY used in post-training "
          "probes. logw=logp(x|z)+logp(z)-logq(z|x);loss=-mean_cases(logsumexp_samples(logw)-logK). Sampled log density "
          "ratio,not analytic KL. Ordinary full pathwise gradients,not DReG. <a href='https://arxiv.org/abs/1509.00519'>"
          "Original IWAE paper</a>.</p><p>Same initialization,source order and first Gaussian draw;K5 gets four "
          "additional draws. BOTH checkpoints use COMMON fixed K20 validation importance bound. Finite Monte Carlo "
          "estimates are not exact likelihoods. Expected bounds tighten with K;one nested finite sample need not be "
          "monotonic as K changes. ESS and normalized weights are saved.</p><p>Original,trainK1 decode(mu),trainK5 "
          "decode(mu),fit-only mean. Ten evenly spaced heldout indices,unfiltered. Probabilities,not pixel draws. "
          "Decode(mu) is not the posterior predictive mean. Three epochs do not establish superior reconstruction or "
          "disentanglement.</p>";
  const auto source = pools[2].source.to_vector<int>();
  for (int k = 0; k < 10; ++k) {
    const int  b = k * 41 / 9;
    const auto file = "case" + std::to_string(k) + ".png";
    ae::Mosaic(root / file,
               nn::stack({ae::Row(pools[2].images, b), ae::Row(ae::Find(one, "test_mean_prediction"), b),
                          ae::Row(ae::Find(five, "test_mean_prediction"), b), mean},
                         0),
               4, 4);
    html << "<h2>Case" << k << ",source" << source[b] << "</h2><img src='" << file << "'>";
  }
  for (const std::string name : {"k1", "k5"})
    html << "<h2>" << name << ":same20prior codes</h2><img src='" << name << "/prior_means.png'><p><a href='" << name
         << "/factor_probes.html'>Heldout factor diagnostics</a></p>";
  check::Require(bool(html), "IWAE gallery written");
  std::cout << "training time " << seconds << " s\nArtifacts:" << (root / "index.html").string() << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int                   epochs = 3;
    std::string           init;
    std::filesystem::path data = paths::Dataset("beta_vae"),
                          out  = paths::OutputRoot() / "models/importance_weighted_autoencoder";
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a.rfind("epochs=", 0) == 0)
        epochs = std::stoi(a.substr(7));
      else if (a.rfind("data=", 0) == 0)
        data = a.substr(5);
      else if (a.rfind("out=", 0) == 0)
        out = a.substr(4);
      else if (a.rfind("init=", 0) == 0)
        init = a.substr(5);
      else
        throw std::invalid_argument("expected epochs=N data=PATH out=PATH init=PATH");
    }
    check::Require(epochs > 0 && epochs <= 100, "three default epochs or explicit1..100");
    Run(data, out, epochs, init);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
