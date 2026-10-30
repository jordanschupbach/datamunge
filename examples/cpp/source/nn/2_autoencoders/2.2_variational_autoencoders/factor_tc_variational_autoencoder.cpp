// CMD: just run-nn factor_tc_variational_autoencoder
// Matched FactorVAE and full-reference betaTCVAE on a tiny known-factor lattice.
// Three default epochs.
//
// Port of tflow's factor_tc_variational_autoencoder. Preparation (in ~/projects/tflow_examples):
// just beta-vae-assets; data are read from $DATAMUNGE_DATA_DIR/beta_vae. Outputs go to
// output/nn/models/factor_tc_variational_autoencoder (factor_vae/ and beta_tcvae/, index.html,
// case<k>.png, tc_algebra.npz). Checkpoints are best.npz / best_discriminator.npz instead of SaveV2
// bundles. Options: epochs=N data=PATH out=PATH init=DIR (a tflow output directory: loads
// DIR/factor_vae/initial_arrays.npz as the shared VAE start and DIR/factor_vae/initial_discriminator.npz
// as the FactorVAE discriminator start, to follow tflow's trajectories exactly).

#include <chrono>
#include <iostream>
#include <limits>
#include <numeric>

#include "common/total_correlation_autoencoder.hpp"

namespace {

namespace nn = datamunge::nn;

void Checks(const std::filesystem::path& root) {
  auto z  = nn::from_vector(std::vector<float>{.3f, -.2f, 1.f, .4f}, {2, 2}).requires_grad_();
  auto mu = nn::from_vector(std::vector<float>{0.f, 0.f, 1.f, -.3f, -.5f, .8f}, {3, 2}).requires_grad_();
  auto lv = nn::from_vector(std::vector<float>{.2f, -.3f, .4f, .1f, -.6f, .5f}, {3, 2}).requires_grad_();
  auto tc = ae::EmpiricalTotalCorrelation(z, mu, lv, 3);
  auto g  = nn::grad(tc.mean, {z, mu, lv});
  nn::Tensor zero, extreme;
  {
    nn::NoGradGuard guard;
    zero    = ae::EmpiricalTotalCorrelation(z, nn::tensor({{0.f, 0.f}}), nn::tensor({{.2f, -.3f}}), 1).per_case;
    extreme = ae::EmpiricalTotalCorrelation(nn::tensor({{1000.f, -1000.f}}), mu, lv, 3).per_case;
  }
  std::vector<nn::Tensor> out{tc.per_case, tc.log_joint, tc.log_product, g[0], g[1], g[2], zero, extreme};
  for (auto& t : out) {
    t = t.detach().contiguous();
    for (float v : t.to_vector<float>()) check::Require(std::isfinite(v), "finite TC algebra and extreme log densities");
  }
  const auto zv = out[6].to_vector<float>();
  for (int i = 0; i < 2; ++i) check::Near(zv[i], 0, 1e-6, "one diagonal Gaussian has factorized density");
  nn::save_npz((root / "tc_algebra.npz").string(),
               {{"z", z.detach()}, {"mu", mu.detach()}, {"log_variance", lv.detach()}, {"tc", out[0]},
                {"log_joint", out[1]}, {"log_product", out[2]}, {"dz", out[3]}, {"dmu", out[4]}, {"dlv", out[5]},
                {"single_component_tc", out[6]}, {"extreme_tc", out[7]}});
  std::cout << "TC algebra/sample+reference gradients/single-component zero/extreme stable density passed" << std::endl;
}

/// tflow's Permute: each latent column independently shuffled with one std::mt19937(seed).
nn::Tensor Permute(const nn::Tensor& joint, std::uint32_t seed) {
  const int          n = int(joint.size(0));
  const auto         j = joint.contiguous().to_vector<float>();
  std::vector<float> product(j.size());
  std::mt19937       random(seed);
  std::vector<int>   rows(n);
  for (int d = 0; d < 4; ++d) {
    std::iota(rows.begin(), rows.end(), 0);
    std::shuffle(rows.begin(), rows.end(), random);
    for (int i = 0; i < n; ++i) product[i * 4 + d] = j[rows[i] * 4 + d];
  }
  return nn::from_vector(product, joint.shape());
}

double Train(bool factor, const std::vector<ae::BetaFactorData>& pools, const std::vector<ae::VaeNoiseBank>& train_noise,
             const std::vector<ae::VaeNoiseBank>& eval_noise, const ae::VaeNoiseBank& prior_noise, int epochs,
             const std::filesystem::path& root, const nn::NamedTensors& initial, const nn::NamedTensors& initial_disc) {
  const auto directory = root / (factor ? "factor_vae" : "beta_tcvae");
  std::filesystem::create_directories(directory);
  ae::TotalCorrelationModel model(factor, pools[0].images);
  model.Load(initial);
  if (factor && !initial_disc.empty()) ae::LoadArrays(model.discriminator, initial_disc);
  if (factor) nn::save_npz((directory / "initial_discriminator.npz").string(), model.DiscArrays());
  check::Require(model.parameters.size() == 10 && model.parameters.count() == 51416,
                 "compact four-dimensional betaVAE census");
  nn::save_npz((directory / "initial_arrays.npz").string(), ae::Arrays(model.parameters));
  auto           initial_score = ae::TotalCorrelationScore(model, pools[2].images, eval_noise[2]);
  nn::BatchOrder order(210, 32, 2243), disc_order(210, 32, 2303);
  std::ofstream  disc_batches(directory / "discriminator_batches.tsv"), history(directory / "history.tsv"),
      batches(directory / "batches.tsv");
  disc_batches << "epoch\tstep\trow\tsource_id\n";
  history.precision(17);
  history << "epoch\tfit_preupdate_weighted_objective\tvalidation_nll\tvalidation_kl\tvalidation_negative_elbo\t"
             "validation_weighted_objective\n";
  batches << "epoch\tstep\trow\tsource_id\n";
  const auto fit_source = pools[0].source.to_vector<int>();
  const auto best_path = (directory / "best.npz").string(), best_disc = (directory / "best_discriminator.npz").string();
  double     best = std::numeric_limits<double>::infinity();
  int        selected = 0, step = 0;
  const auto started = std::chrono::steady_clock::now();
  for (int epoch = 1; epoch <= epochs; ++epoch) {
    auto epsilon    = ae::BetaVaeNoiseAt(train_noise[epoch - 1], 0);
    auto disc_noise = ae::BetaVaeNoise(210, 1, 2333 + epoch);
    nn::save_npz((directory / ("discriminator_noise_epoch" + std::to_string(epoch) + ".npz")).string(),
                 {{"standard", disc_noise.standard}, {"epsilon", disc_noise.epsilon}});
    order.begin_epoch();
    disc_order.begin_epoch();
    double sum  = 0;
    int    seen = 0;
    for (;;) {
      auto rows = order.next();
      if (rows.empty()) break;
      auto input = nn::take_rows(pools[0].images, rows), e = nn::take_rows(epsilon, rows);
      if (step == 0) {
        auto g = ae::CheckGradients(model.parameters, model.Gradient(input, e), "all TC-augmented derivatives finite",
                                    "all10posterior/decoder tensors receive gradients");
        nn::save_npz((directory / "first_gradients.npz").string(), g);
        nn::save_npz((directory / "first_batch.npz").string(), {{"images", input}, {"epsilon", e}});
      }
      const auto disc_before = factor ? ae::Snapshot(model.discriminator) : std::vector<std::string>{};
      sum += double(rows.size()) * model.Update(input, e);
      if (factor) {
        check::Require(disc_before == ae::Snapshot(model.discriminator),
                       "VAE update freezes ALL discriminator parameters but preserves z gradient");
        const auto vae_before = ae::Snapshot(model.parameters);
        auto       disc_rows  = disc_order.next();
        check::Require(disc_rows.size() == rows.size(), "independent fitting minibatch matches discriminator batch size");
        for (auto row : disc_rows) disc_batches << epoch << '\t' << step << '\t' << row << '\t' << fit_source[row] << '\n';
        auto disc_input   = nn::take_rows(pools[0].images, disc_rows);
        auto disc_epsilon = nn::take_rows(ae::BetaVaeNoiseAt(disc_noise, 0), disc_rows);
        auto joint        = model.Evaluate(disc_input, disc_epsilon)[2];
        auto permuted     = Permute(joint, 2311 + step);
        if (step == 0) {
          nn::save_npz((directory / "first_discriminator_batch.npz").string(),
                       {{"joint", joint}, {"product", permuted}, {"images", disc_input}, {"epsilon", disc_epsilon}});
          auto             dg = model.DiscGradient(joint, permuted);
          nn::NamedTensors gs;
          for (std::size_t j = 0; j < dg.size(); ++j) gs.emplace_back(model.discriminator.names()[j], dg[j]);
          nn::save_npz((directory / "first_discriminator_gradients.npz").string(), gs);
        }
        const double dl = model.DiscUpdate(joint, permuted);
        check::Require(std::isfinite(dl) && dl >= 0, "finite discriminator cross entropy");
        check::Require(vae_before == ae::Snapshot(model.parameters), "detached discriminator update freezes ALL VAE tensors");
        if (step == 0) nn::save_npz((directory / "first_discriminator_update.npz").string(), model.DiscArrays());
      }
      seen += int(rows.size());
      for (auto row : rows) batches << epoch << '\t' << step << '\t' << row << '\t' << fit_source[row] << '\n';
      if (step == 0) nn::save_npz((directory / "first_update_arrays.npz").string(), ae::Arrays(model.parameters));
      ++step;
    }
    check::Require(seen == 210, "all210fitting images once per epoch");
    auto score = ae::TotalCorrelationScore(model, pools[1].images, eval_noise[1]);
    history << epoch << '\t' << sum / seen << '\t' << score[0] << '\t' << score[1] << '\t' << score[2] << '\t'
            << score[3] << '\n';
    nn::save_npz((directory / ("epoch" + std::to_string(epoch) + "_arrays.npz")).string(), ae::Arrays(model.parameters));
    if (factor)
      nn::save_npz((directory / ("epoch" + std::to_string(epoch) + "_discriminator.npz")).string(), model.DiscArrays());
    if (score[2] < best) {
      best     = score[2];
      selected = epoch;
      model.parameters.save(best_path);  // COMMON unweighted 4-sample validation negative ELBO selects
      if (factor) model.discriminator.save(best_disc);
    }
    std::cout << (factor ? "FactorVAE" : "BetaTCVAE") << " epoch" << epoch << ":NLL=" << score[0] << ",KL=" << score[1]
              << ",commonnegativeELBO=" << score[2] << ",weightedobjective=" << score[3] << std::endl;
  }
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  ae::TotalCorrelationModel deployed(factor, pools[0].images, false);
  deployed.parameters.load(best_path);
  if (factor) {
    deployed.discriminator.load(best_disc);
    nn::save_npz((directory / "model_discriminator.npz").string(), deployed.DiscArrays());
  }
  const auto before = ae::Snapshot(deployed.parameters);
  check::Near(ae::TotalCorrelationScore(deployed, pools[1].images, eval_noise[1])[2], best, 1e-8,
              "fresh common selected validation score");
  nn::NamedTensors output;
  for (int split = 0; split < 3; ++split) {
    const std::string  name = split == 0 ? "fit" : split == 1 ? "validation" : "test";
    const std::int64_t n    = pools[split].images.size(0);
    auto               mean = deployed.Evaluate(pools[split].images, nn::zeros({n, 4}));
    check::Require(ae::Bytes(mean[2]) == ae::Bytes(mean[0]), "epsilon0 yields EXACT posterior mean");
    std::vector<nn::Tensor> zs, prob, nll;
    std::vector<float>      tcs;
    for (int k = 0; k < 4; ++k) {
      auto draw = deployed.Evaluate(pools[split].images, ae::BetaVaeNoiseAt(eval_noise[split], k));
      check::Require(ae::Bytes(draw[0]) == ae::Bytes(mean[0]) && ae::Bytes(draw[1]) == ae::Bytes(mean[1]),
                     "noise changes samples,not posterior parameters");
      tcs.push_back(draw[8].item<float>());
      zs.push_back(draw[2]);
      prob.push_back(draw[3]);
      nll.push_back(draw[4]);
    }
    output.emplace_back(name + "_sample_tc", nn::from_vector(tcs, {4}));
    output.emplace_back(name + "_mu", mean[0]);
    output.emplace_back(name + "_log_variance", mean[1]);
    output.emplace_back(name + "_mean_prediction", mean[3]);
    output.emplace_back(name + "_kl", mean[5]);
    output.emplace_back(name + "_sample_z", nn::stack(zs, 0));
    output.emplace_back(name + "_sample_prediction", nn::stack(prob, 0));
    output.emplace_back(name + "_sample_nll", nn::stack(nll, 0));
  }
  auto test   = deployed.Evaluate(pools[2].images, ae::BetaVaeNoiseAt(eval_noise[2], 0)),
       repeat = deployed.Evaluate(pools[2].images, ae::BetaVaeNoiseAt(eval_noise[2], 0));
  check::Require(ae::Bytes(test[2]) == ae::Bytes(repeat[2]) && ae::Bytes(test[3]) == ae::Bytes(repeat[3]) &&
                     ae::Bytes(deployed.Decode(test[2])) == ae::Bytes(test[3]),
                 "fixednoise repeat and exact targetfree sampled decode");
  auto other = deployed.Evaluate(pools[2].images, ae::BetaVaeNoiseAt(eval_noise[2], 1));
  check::Require(ae::Bytes(test[2]) != ae::Bytes(other[2]) && ae::Bytes(test[3]) != ae::Bytes(other[3]),
                 "changed epsilon changes sample AND probability");
  auto single = deployed.Evaluate(nn::take_rows(pools[2].images, {0}),
                                  nn::take_rows(ae::BetaVaeNoiseAt(eval_noise[2], 0), {0}));
  const double parity = ae::MaxAbsDiff(single[3], test[3], 784);
  check::Require(parity < 2e-6, "betaVAE single/batch parity");
  output.emplace_back("zero_prediction", deployed.Decode(nn::zeros({42, 4})));
  auto       prior_z = ae::BetaVaeNoiseAt(prior_noise, 0), prior = deployed.Decode(prior_z);
  const auto fit_mu  = ae::Find(output, "fit_mu").to_vector<float>();
  std::vector<float> center(4), traversal(36 * 4);
  for (int d = 0; d < 4; ++d) {
    double s = 0;
    for (int b = 0; b < 210; ++b) s += fit_mu[b * 4 + d];
    center[d] = float(s / 210);
  }
  for (int d = 0; d < 4; ++d)
    for (int k = 0; k < 9; ++k)
      for (int j = 0; j < 4; ++j) traversal[(d * 9 + k) * 4 + j] = center[j] + (j == d ? -2.f + .5f * k : 0);
  auto traversal_z = nn::from_vector(traversal, {36, 4}), traversal_p = deployed.Decode(traversal_z);
  nn::save_npz((directory / "prior.npz").string(), {{"z", prior_z},
                                                    {"probability", prior},
                                                    {"traversal_center", nn::from_vector(center, {4})},
                                                    {"traversal_z", traversal_z},
                                                    {"traversal_probability", traversal_p}});
  ae::Mosaic(directory / "prior_means.png", prior, 5, 3);
  ae::Mosaic(directory / "traversals.png", traversal_p, 9, 3);
  auto r2 = ae::BetaFactorProbe(directory, ae::Find(output, "fit_mu"), ae::Find(output, "test_mu"), pools[0].factors,
                                pools[2].factors, ae::Find(output, "fit_log_variance"));
  check::Require(before == ae::Snapshot(deployed.parameters),
                 "prior/traversal/factor diagnostics leave all10model tensors immutable");
  nn::save_npz((directory / "outputs.npz").string(), output);
  nn::save_npz((directory / "model_arrays.npz").string(), ae::Arrays(deployed.parameters));
  auto          score = ae::TotalCorrelationScore(deployed, pools[2].images, eval_noise[2]);
  const double  mse   = ae::Mse(ae::Find(output, "test_mean_prediction"), pools[2].images);
  std::ofstream meta(directory / "metrics.json");
  meta.precision(17);
  meta << "{\"extra_tc_weight\":3,\"parameters\":51416,\"epochs\":" << epochs << ",\"updates\":" << step
       << ",\"selected_epoch\":" << selected
       << ",\"mc_samples\":4,\"checkpoint_criterion\":\"common_unweighted_validation_negative_elbo\","
          "\"initial_test_negative_elbo\":"
       << initial_score[2] << ",\"test_nll\":" << score[0] << ",\"test_kl\":" << score[1]
       << ",\"test_negative_elbo\":" << score[2] << ",\"test_weighted_objective\":" << score[3]
       << ",\"mean_decode_test_mse\":" << mse << ",\"single_batch_max_error\":" << parity
       << ",\"training_seconds\":" << seconds << "}\n";
  check::Require(bool(meta) && bool(history) && bool(batches), "complete TC-VAE evidence");
  std::cout << (factor ? "FactorVAE" : "BetaTCVAE") << ":selected" << selected << ",testNLL=" << score[0]
            << ",KL=" << score[1] << ",commonnegativeELBO=" << score[2] << ",weighted=" << score[3]
            << ",meanMSE=" << mse << ";ridge R2 shape/size/x/y=" << r2[1][0] << "/" << r2[1][1] << "/" << r2[1][2]
            << "/" << r2[1][3] << ";restore/decode/noise/batch/immutable passed\n";
  return seconds;
}

void Run(const std::filesystem::path& data, const std::filesystem::path& root, int epochs, const std::string& init) {
  auto pools = ae::LoadBetaFactors(data);
  std::filesystem::create_directories(root);
  Checks(root);
  ae::SaveFactorData(root, pools);
  std::vector<ae::VaeNoiseBank> training,
      noise{ae::BetaVaeNoise(210, 4, 2273), ae::BetaVaeNoise(42, 4, 2267), ae::BetaVaeNoise(42, 4, 2269)};
  auto             prior = ae::BetaVaeNoise(20, 1, 2281);
  nn::NamedTensors noise_arrays{{"prior_standard", prior.standard}, {"prior_epsilon", prior.epsilon}};
  for (int i = 0; i < 3; ++i) {
    const std::string n = i == 0 ? "fit" : i == 1 ? "validation" : "test";
    noise_arrays.emplace_back(n + "_standard", noise[i].standard);
    noise_arrays.emplace_back(n + "_epsilon", noise[i].epsilon);
  }
  nn::save_npz((root / "fixed_noise.npz").string(), noise_arrays);
  for (int epoch = 1; epoch <= epochs; ++epoch) {
    training.push_back(ae::BetaVaeNoise(210, 1, 2251 + epoch));
    nn::save_npz((root / ("train_noise_epoch" + std::to_string(epoch) + ".npz")).string(),
                 {{"standard", training.back().standard}, {"epsilon", training.back().epsilon}});
  }
  ae::BetaVariationalAutoencoderModel base(1, false);
  nn::NamedTensors                    initial_disc;
  if (!init.empty()) {
    base.parameters.load((std::filesystem::path(init) / "factor_vae" / "initial_arrays.npz").string());
    initial_disc = nn::load_npz((std::filesystem::path(init) / "factor_vae" / "initial_discriminator.npz").string());
  }
  auto   initial = ae::Arrays(base.parameters);
  double seconds = Train(true, pools, training, noise, prior, epochs, root, initial, initial_disc);
  seconds += Train(false, pools, training, noise, prior, epochs, root, initial, {});
  auto a = nn::load_npz((root / "factor_vae" / "outputs.npz").string()),
       b = nn::load_npz((root / "beta_tcvae" / "outputs.npz").string());
  auto mean = ae::Constant(ae::PixelMean(pools[0].images), 42);
  const std::vector<nn::Tensor> panels{ae::Flatten(pools[2].images), ae::Find(a, "test_mean_prediction"),
                                       ae::Find(b, "test_mean_prediction"), mean};
  std::ofstream html(root / "index.html");
  html << "<!doctype html><meta charset='utf-8'><title>FactorVAE and betaTCVAE</title><style>body{font:16px "
          "system-ui;margin:30px;max-width:1000px}img{max-width:100%;image-rendering:pixelated}</style><h1>FactorVAE / "
          "betaTCVAE</h1><p>Tiny educational 294-image binary factor lattice:210 fit,42 validation,42 test. Three epochs "
          "by default,7updates per epoch,51416 VAE parameters. Factors are used ONLY in post-training diagnostics. "
          "Matched initialization,source order and VAE noise. Checkpoints use four-sample COMMON validation NLL+analytic "
          "KL.</p><p>FactorVAE: NLL+KL+3*mean discriminator logit;4->16 tanh->1 density ratio network,97parameters. One "
          "VAE Adam update then one discriminator Adam update on detached fresh posterior draws on independent fitting "
          "minibatches and independently permuted columns. Separate optimizers; opposite parameter stores checked "
          "immutable every step. Logit positive means joint,negative means product.</p><p>BetaTCVAE: NLL+analytic "
          "KL+3*TC (beta_TC=4). Log joint mixture minus sum log marginal mixtures,ALL210 fitting posterior components; "
          "gradients include reference encoder heads. Exact finite empirical densities,Monte Carlo expectations;this is "
          "an educational adaptation of the paper's minibatch density estimator. Negative sample TC is retained. No "
          "validation/test mixture components. No optimizer-resume claim. <a "
          "href='https://proceedings.mlr.press/v80/kim18b'>FactorVAE paper</a>,<a "
          "href='https://arxiv.org/abs/1802.04942'>betaTCVAE paper</a>.</p><p>Original,FactorVAE decode(mu),betaTCVAE "
          "decode(mu),fit-only mean. Ten evenly spaced heldout indices,unfiltered. Probabilities,not sampled binary "
          "images. Three epochs do not establish disentanglement;inspect the factor probes,negative R2 and mixing.</p>";
  const auto source = pools[2].source.to_vector<int>();
  for (int k = 0; k < 10; ++k) {
    const int               row = k * 41 / 9;
    std::vector<nn::Tensor> contact;
    for (const auto& p : panels) contact.push_back(ae::Row(p, row));
    const std::string file = "case" + std::to_string(k) + ".png";
    ae::Mosaic(root / file, nn::stack(contact, 0), 4, 4);
    html << "<h2>Case" << k << ",source" << source[row] << "</h2><img src='" << file << "'>";
  }
  for (const std::string name : {"factor_vae", "beta_tcvae"})
    html << "<h2>" << name << "</h2><a href='" << name
         << "/factor_probes.html'>Heldout factor diagnostics</a><p>20 identical standard-normal codes across "
            "models</p><img src='"
         << name << "/prior_means.png'><p>4 rows,one coordinate varied -2..2 around fit mean</p><img src='" << name
         << "/traversals.png'>";
  check::Require(bool(html), "gallery written");
  std::cout << "training time " << seconds << " s\nArtifacts:" << root.string() << "/index.html" << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int                   epochs = 3;
    std::string           init;
    std::filesystem::path data = paths::Dataset("beta_vae"),
                          out  = paths::OutputRoot() / "models/factor_tc_variational_autoencoder";
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
        throw std::invalid_argument("expected epochs=N data=PATH out=PATH init=DIR");
    }
    check::Require(epochs > 0 && epochs <= 100, "three default epochs or explicit1..100");
    Run(data, out, epochs, init);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
