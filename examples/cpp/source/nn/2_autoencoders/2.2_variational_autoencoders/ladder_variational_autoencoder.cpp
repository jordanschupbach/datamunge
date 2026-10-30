// CMD: just run-nn ladder_variational_autoencoder
// Two-level precision-fusion Ladder VAE with conditional lower prior and ancestral generation.
// THREE default epochs, 21 updates.
//
// Port of tflow's ladder_variational_autoencoder. Preparation (in ~/projects/tflow_examples):
// just beta-vae-assets; data are read from $DATAMUNGE_DATA_DIR/beta_vae. Outputs go to
// output/nn/models/ladder_variational_autoencoder. Checkpoints are best.npz instead of SaveV2
// bundles; metrics.json gains training_seconds. Options: epochs=N data=PATH out=PATH init=PATH
// (tflow's initial_arrays.npz, to follow its trajectory exactly).

#include <chrono>
#include <iostream>
#include <limits>

#include "common/ladder_variational_autoencoder.hpp"

namespace {

namespace nn = datamunge::nn;

void Checks(const std::filesystem::path& root) {
  const std::vector<float> u{2, -1, .3f, .8f}, v{float(std::log(4)), 0, -.2f, .4f}, p{-1, 1, .1f, -.4f},
      r{float(std::log(9)), float(std::log(4)), .6f, -.3f};
  auto U = nn::from_vector(u, {2, 2}).requires_grad_(), V = nn::from_vector(v, {2, 2}).requires_grad_(),
       P = nn::from_vector(p, {2, 2}).requires_grad_(), R = nn::from_vector(r, {2, 2}).requires_grad_();
  auto q         = ae::LadderPrecisionFusion(U, V, P, R);
  auto kl        = ae::LadderConditionalKl(q.mu, q.log_variance, P, R);
  auto objective = nn::sum(nn::square(q.mu) + nn::square(q.log_variance), {0, 1}) + nn::sum(kl, {0});
  auto g         = nn::grad(objective, {U, V, P, R});
  nn::Tensor extreme_mu, extreme_lv, zero;
  {
    nn::NoGradGuard guard;
    auto extreme = ae::LadderPrecisionFusion(U, nn::from_vector(std::vector<float>{-80, 80, 80, -80}, {2, 2}), P,
                                             nn::from_vector(std::vector<float>{80, -80, -80, 80}, {2, 2}));
    extreme_mu = extreme.mu.contiguous();
    extreme_lv = extreme.log_variance.contiguous();
    zero       = ae::LadderConditionalKl(U, V, U, V).contiguous();
  }
  std::vector<nn::Tensor> out{q.mu.detach().contiguous(), q.log_variance.detach().contiguous(),
                              kl.detach().contiguous(), g[0], g[1], g[2], g[3], extreme_mu, extreme_lv, zero};
  for (const auto& t : out)
    for (float x : t.contiguous().to_vector<float>()) check::Require(std::isfinite(x), "finite fusion and all gradients");
  for (const auto& t : {g[0], g[1], g[2], g[3]}) {
    double norm = 0;
    for (float x : t.contiguous().to_vector<float>()) norm += double(x) * x;
    check::Require(norm > 0, "fusion derivatives include evidence AND prior mean/logvariance");
  }
  const auto mu = out[0].to_vector<float>(), lv = out[1].to_vector<float>(), z = out[9].to_vector<float>();
  for (int i = 0; i < 4; ++i) {
    const double a = std::exp(-double(v[i])), b = std::exp(-double(r[i]));
    check::Near(mu[i], (a * u[i] + b * p[i]) / (a + b), 3e-7, "known precision weighted mean");
    check::Near(lv[i], -std::log(a + b), 3e-7, "known harmonic posterior variance");
  }
  for (int b = 0; b < 2; ++b) check::Near(z[b], 0, 0, "conditional KL(q||q) exactly zero");
  nn::save_npz((root / "fusion_algebra.npz").string(),
               {{"evidence_mu", U.detach()}, {"evidence_lv", V.detach()}, {"prior_mu", P.detach()},
                {"prior_lv", R.detach()},    {"mu", out[0]},          {"lv", out[1]},
                {"kl", out[2]},              {"du", out[3]},          {"dv", out[4]},
                {"dp", out[5]},              {"dr", out[6]},          {"extreme_mu", out[7]},
                {"extreme_lv", out[8]}});
  std::cout << "precision fusion/evidence+prior gradients/conditional KL zero/extreme stable precisions passed"
            << std::endl;
}

nn::Tensor Columns(const nn::Tensor& x, std::int64_t begin, std::int64_t end) {
  return nn::slice_axis(x, 1, begin, end).contiguous();
}

void Run(const std::filesystem::path& data, const std::filesystem::path& root, int epochs, const std::string& init) {
  auto pools = ae::LoadBetaFactors(data);
  std::filesystem::create_directories(root);
  Checks(root);
  ae::SaveFactorData(root, pools);
  std::vector<ae::VaeNoiseBank> noise{ae::LadderNoise(210, 4, 2521), ae::LadderNoise(42, 4, 2531),
                                      ae::LadderNoise(42, 4, 2539)},
      training;
  auto             prior_noise = ae::LadderNoise(20, 1, 2543);
  nn::NamedTensors fixed{{"prior_standard", prior_noise.standard}, {"prior_epsilon", prior_noise.epsilon}};
  for (int split = 0; split < 3; ++split) {
    const std::string name = split == 0 ? "fit" : split == 1 ? "validation" : "test";
    fixed.emplace_back(name + "_standard", noise[split].standard);
    fixed.emplace_back(name + "_epsilon", noise[split].epsilon);
  }
  nn::save_npz((root / "fixed_noise.npz").string(), fixed);
  for (int e = 1; e <= epochs; ++e) {
    training.push_back(ae::LadderNoise(210, 1, 2551 + e));
    nn::save_npz((root / ("train_noise_epoch" + std::to_string(e) + ".npz")).string(),
                 {{"standard", training.back().standard}, {"epsilon", training.back().epsilon}});
  }
  ae::LadderVaeModel model;
  if (!init.empty()) model.Load(nn::load_npz(init));
  check::Require(model.parameters.size() == 22 && model.parameters.count() == 52196, "two-level ladder parameter census");
  nn::save_npz((root / "initial_arrays.npz").string(), ae::Arrays(model.parameters));
  nn::BatchOrder order(210, 32, 2507);
  std::ofstream  history(root / "history.tsv"), batches(root / "batches.tsv");
  history.precision(17);
  history << "epoch\ttrain_preupdate_negative_elbo\tvalidation_nll\tvalidation_lower_kl\tvalidation_top_kl\t"
             "validation_negative_elbo\n";
  batches << "epoch\tstep\trow\tsource_id\n";
  const auto fit_source = pools[0].source.to_vector<int>();
  const auto best_path  = (root / "best.npz").string();
  double     best       = std::numeric_limits<double>::infinity();
  int        selected = 0, step = 0;
  const auto started = std::chrono::steady_clock::now();
  for (int epoch = 1; epoch <= epochs; ++epoch) {
    order.begin_epoch();
    double sum     = 0;
    int    seen    = 0;
    auto   epsilon = ae::LadderNoiseAt(training[epoch - 1], 0);
    for (;;) {
      auto rows = order.next();
      if (rows.empty()) break;
      auto x = nn::take_rows(pools[0].images, rows), e = nn::take_rows(epsilon, rows);
      if (step == 0) {
        nn::save_npz((root / "first_batch.npz").string(), {{"images", x}, {"epsilon", e}});
        auto g = ae::CheckGradients(model.parameters, model.Gradient(x, e), "all ladder gradients finite",
                                    "all22tensors receive gradients");
        nn::save_npz((root / "first_gradients.npz").string(), g);
        nn::save_npz((root / "first_forward.npz").string(), model.Evaluate(x, e));
      }
      sum += double(rows.size()) * model.Update(x, e);
      seen += int(rows.size());
      for (auto row : rows) batches << epoch << '\t' << step << '\t' << row << '\t' << fit_source[row] << '\n';
      if (step == 0) nn::save_npz((root / "first_update_arrays.npz").string(), ae::Arrays(model.parameters));
      ++step;
    }
    check::Require(seen == 210, "all210fitting cases once per epoch");
    auto score = ae::LadderScore(model, pools[1].images, noise[1]);
    history << epoch << '\t' << sum / seen;
    for (auto value : score) history << '\t' << value;
    history << '\n';
    nn::save_npz((root / ("epoch" + std::to_string(epoch) + "_arrays.npz")).string(), ae::Arrays(model.parameters));
    if (score[3] < best) {
      best     = score[3];
      selected = epoch;
      model.parameters.save(best_path);  // four-sample validation ELBO selects checkpoint
    }
    std::cout << "epoch" << epoch << ":validation NLL=" << score[0] << ",lowerKL=" << score[1]
              << ",topKL=" << score[2] << ",negativeELBO=" << score[3] << std::endl;
  }
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();

  ae::LadderVaeModel deployed(false);
  deployed.parameters.load(best_path);  // fresh ladder model-only restore
  check::Near(ae::LadderScore(deployed, pools[1].images, noise[1])[3], best, 1e-8, "fresh selected validation score");
  const auto       before = ae::Snapshot(deployed.parameters);
  nn::NamedTensors output;
  for (int split = 0; split < 3; ++split) {
    const std::int64_t n    = pools[split].images.size(0);
    const std::string  name = split == 0 ? "fit" : split == 1 ? "validation" : "test";
    auto               greedy = deployed.Evaluate(pools[split].images, nn::zeros({n, 6}));
    check::Require(ae::SameBytes(ae::Find(greedy, "lower_z"), ae::Find(greedy, "lower_mu")) &&
                       ae::SameBytes(ae::Find(greedy, "top_z"), ae::Find(greedy, "top_mu")),
                   "both zero-noise samples equal their recursive means");
    for (const auto& [key, value] : greedy) output.emplace_back(name + "_greedy_" + key, value);
    for (int k = 0; k < 4; ++k) {
      auto f = deployed.Evaluate(pools[split].images, ae::LadderNoiseAt(noise[split], k));
      for (const auto& [key, value] : f) output.emplace_back(name + "_draw" + std::to_string(k) + "_" + key, value);
    }
  }
  auto test   = deployed.Evaluate(pools[2].images, ae::LadderNoiseAt(noise[2], 0)),
       repeat = deployed.Evaluate(pools[2].images, ae::LadderNoiseAt(noise[2], 0));
  check::Require(ae::SameBytes(ae::Find(test, "probability"), ae::Find(repeat, "probability")) &&
                     ae::SameBytes(ae::Find(test, "probability"), deployed.Decode(ae::Find(test, "lower_z"))),
                 "repeat and target-free lower latent decode BITEXACT");
  auto single = deployed.Evaluate(nn::take_rows(pools[2].images, {0}),
                                  nn::take_rows(ae::LadderNoiseAt(noise[2], 0), {0}));
  const double parity = ae::MaxAbsDiff(ae::Find(single, "probability"), ae::Find(test, "probability"), 784);
  check::Require(parity < 2e-6, "single/batch parity");
  std::vector<float> lower(42 * 6, 0.f), upper(42 * 6, 0.f);
  for (int b = 0; b < 42; ++b) {
    for (int j = 0; j < 4; ++j) lower[b * 6 + j] = .5f;
    for (int j = 4; j < 6; ++j) upper[b * 6 + j] = .5f;
  }
  auto base = deployed.Evaluate(pools[2].images, nn::zeros({42, 6})),
       low  = deployed.Evaluate(pools[2].images, nn::from_vector(lower, {42, 6})),
       top  = deployed.Evaluate(pools[2].images, nn::from_vector(upper, {42, 6}));
  check::Require(ae::SameBytes(ae::Find(base, "lower_mu"), ae::Find(low, "lower_mu")) &&
                     ae::SameBytes(ae::Find(base, "prior_mu"), ae::Find(low, "prior_mu")) &&
                     ae::MaxAbsDiff(ae::Find(base, "lower_z"), ae::Find(low, "lower_z"), 42 * 4) > 1e-4,
                 "lower noise changes sample only,not hierarchy heads");
  const double prior_effect  = ae::MaxAbsDiff(ae::Find(base, "prior_mu"), ae::Find(top, "prior_mu"), 42 * 4),
               fusion_effect = ae::MaxAbsDiff(ae::Find(base, "lower_mu"), ae::Find(top, "lower_mu"), 42 * 4);
  check::Require(prior_effect > 1e-5 && fusion_effect > 1e-5,
                 "top noise changes conditional prior AND fused lower posterior");
  nn::save_npz((root / "top_noise_intervention.npz").string(), top);
  nn::save_npz((root / "lower_noise_intervention.npz").string(), low);
  auto prior_bank = ae::LadderNoiseAt(prior_noise, 0);
  auto prior      = deployed.Prior(Columns(prior_bank, 4, 6), Columns(prior_bank, 0, 4));
  check::Require(ae::SameBytes(ae::Find(prior, "probability"), deployed.Decode(ae::Find(prior, "lower_z"))),
                 "ancestral hierarchical sample decodes exactly");
  nn::save_npz((root / "prior.npz").string(), prior);
  ae::Mosaic(root / "prior_means.png", ae::Find(prior, "probability"), 5, 3);
  check::Require(before == ae::Snapshot(deployed.parameters), "all inference/prior/intervention calls immutable");
  nn::save_npz((root / "outputs.npz").string(), output);
  nn::save_npz((root / "model_arrays.npz").string(), ae::Arrays(deployed.parameters));
  auto         score    = ae::LadderScore(deployed, pools[2].images, noise[2]);
  const double mse      = ae::Mse(ae::Find(output, "test_greedy_probability"), pools[2].images);
  auto         mean     = ae::Constant(ae::PixelMean(pools[0].images), 42);
  const double mean_mse = ae::Mse(mean, pools[2].images);
  std::ofstream meta(root / "metrics.json");
  meta.precision(17);
  meta << "{\"parameters\":52196,\"tensors\":22,\"epochs\":" << epochs << ",\"updates\":" << step
       << ",\"selected_epoch\":" << selected << ",\"mc_samples\":4,\"test_nll\":" << score[0]
       << ",\"test_lower_kl\":" << score[1] << ",\"test_top_kl\":" << score[2]
       << ",\"test_negative_elbo\":" << score[3] << ",\"greedy_decode_test_mse\":" << mse
       << ",\"fit_mean_test_mse\":" << mean_mse << ",\"single_batch_max_error\":" << parity
       << ",\"top_noise_prior_mu_max_effect\":" << prior_effect
       << ",\"top_noise_fused_mu_max_effect\":" << fusion_effect << ",\"training_seconds\":" << seconds << "}\n";
  std::ofstream html(root / "index.html");
  html << "<!doctype html><meta charset='utf-8'><title>Two-level Ladder VAE</title><style>body{font:16px "
          "system-ui;max-width:1000px;margin:30px}img{max-width:100%;image-rendering:pixelated}</style><h1>Ladder "
          "variational autoencoder</h1><p>Compact two-level precision-fusion inference:encoder image784->32->16;top "
          "posterior2D,lower Gaussian evidence4D. Learned p(z1|z2) and precision-weighted posterior combine top-down "
          "prior with bottom-up evidence. Decoder4->32->784Bernoulli logits.52196parameters/22tensors. THREE epochs "
          "by default,21updates. Tiny294-image generated factor lattice,210fit42validation42test. Factor values are "
          "never training inputs or checkpoint selectors. <a href='https://arxiv.org/abs/1602.02282'>Ladder VAE "
          "paper</a>. Educational core mechanism,not the paper's large batch-normalized,warm-up benchmark "
          "recipe.</p><p>Loss:SUMpixelNLL+conditionalKL(q1||p1)+KL(q2||N0I),MEANcases. Four explicit Gaussian draws "
          "select validation ELBO;conditional lower KL is averaged over top samples. No KL annealing,free "
          "bits,normalization or variance clipping. Fresh model-only checkpoint restore;optimizer resume "
          "unsupported.</p><p>Original,recursive greedy means,one posterior probability draw,fit-only mean. Ten "
          "evenly spaced heldout indices,unfiltered,fixed0..1gray. Greedy means are NOT posterior predictive means. "
          "Test negativeELBO="
       << score[3] << " (NLL=" << score[0] << ",lowerKL=" << score[1] << ",topKL=" << score[2]
       << "),greedyMSE=" << mse << ",fitmeanMSE=" << mean_mse
       << ". Short training does not establish reconstruction quality,semantic hierarchy or disentanglement.</p>";
  const auto source = pools[2].source.to_vector<int>();
  const std::vector<nn::Tensor> values{ae::Flatten(pools[2].images), ae::Find(output, "test_greedy_probability"),
                                       ae::Find(output, "test_draw0_probability"), mean};
  for (int k = 0; k < 10; ++k) {
    const int               b = k * 41 / 9;
    std::vector<nn::Tensor> panels;
    for (const auto& v : values) panels.push_back(nn::reshape(ae::Row(v, b), {1, 784}));
    const std::string file = "case" + std::to_string(k) + ".png";
    ae::Mosaic(root / file, nn::concat(panels, 0), 4, 4);
    html << "<h2>Case" << k << ",source" << source[b] << "</h2><img src='" << file << "'>";
  }
  html << "<h2>20ancestral prior decoder means</h2><p>Sample z2~N0I,then z1~p(z1|z2),then show Bernoulli "
          "probabilities. No image or encoder input.</p><img src='prior_means.png'>";
  check::Require(bool(meta) && bool(html) && bool(history) && bool(batches), "complete ladder evidence");
  std::cout << "selected" << selected << ",test NLL=" << score[0] << ",lowerKL=" << score[1] << ",topKL=" << score[2]
            << ",negativeELBO=" << score[3] << ",greedyMSE=" << mse << ",fitmeanMSE=" << mean_mse
            << ";restore/decode/noise hierarchy/batch/immutable passed\ntraining time " << seconds
            << " s\nArtifacts:" << (root / "index.html").string() << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int                   epochs = 3;
    std::string           init;
    std::filesystem::path data = paths::Dataset("beta_vae"),
                          out  = paths::OutputRoot() / "models/ladder_variational_autoencoder";
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
