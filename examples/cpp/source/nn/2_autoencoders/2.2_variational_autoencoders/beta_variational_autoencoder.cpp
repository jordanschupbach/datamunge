// CMD: just run-nn beta_variational_autoencoder
// Matched beta1/beta4 Gaussian VAEs on a tiny known-factor binary lattice.
// THREE default epochs, 21 updates/model.
//
// Port of tflow's beta_variational_autoencoder. Preparation (in ~/projects/tflow_examples):
// just beta-vae-assets; data are read from $DATAMUNGE_DATA_DIR/beta_vae. Outputs go to
// output/nn/models/beta_variational_autoencoder (beta1/ and beta4/ subdirectories, index.html,
// case<k>.png). Checkpoints are best.npz instead of SaveV2 bundles. Options: epochs=N
// data=PATH out=PATH init=PATH (shared initial parameters, e.g. tflow's beta1/initial_arrays.npz,
// to follow its trajectory exactly).

#include <chrono>
#include <iostream>
#include <limits>

#include "common/beta_variational_autoencoder.hpp"

namespace {

namespace nn = datamunge::nn;

void Checks() { ae::VaeKnownChecks(4, "beta4"); }

double Train(float beta, const std::vector<ae::BetaFactorData>& pools, const std::vector<ae::VaeNoiseBank>& train_noise,
             const std::vector<ae::VaeNoiseBank>& eval_noise, const ae::VaeNoiseBank& prior_noise, int epochs,
             const std::filesystem::path& root, const nn::NamedTensors& initial) {
  const auto directory = root / (beta == 1 ? "beta1" : "beta4");
  std::filesystem::create_directories(directory);
  ae::BetaVariationalAutoencoderModel model(beta);
  model.Load(initial);
  check::Require(model.parameters.size() == 10 && model.parameters.count() == 51416,
                 "compact four-dimensional betaVAE census");
  nn::save_npz((directory / "initial_arrays.npz").string(), ae::Arrays(model.parameters));
  auto           initial_score = ae::BetaVaeScore(model, pools[2].images, eval_noise[2], beta);
  nn::BatchOrder order(210, 32, 2243);
  std::ofstream  history(directory / "history.tsv"), batches(directory / "batches.tsv");
  history.precision(17);
  history << "epoch\tfit_preupdate_weighted_objective\tvalidation_nll\tvalidation_kl\tvalidation_negative_elbo\t"
             "validation_weighted_objective\n";
  batches << "epoch\tstep\trow\tsource_id\n";
  const auto fit_source = pools[0].source.to_vector<int>();
  const auto best_path  = (directory / "best.npz").string();
  double     best       = std::numeric_limits<double>::infinity();
  int        selected = 0, step = 0;
  const auto started = std::chrono::steady_clock::now();
  for (int epoch = 1; epoch <= epochs; ++epoch) {
    auto epsilon = ae::BetaVaeNoiseAt(train_noise[epoch - 1], 0);
    order.begin_epoch();
    double sum  = 0;
    int    seen = 0;
    for (;;) {
      auto rows = order.next();
      if (rows.empty()) break;
      auto input = nn::take_rows(pools[0].images, rows), e = nn::take_rows(epsilon, rows);
      if (step == 0) {
        auto g = ae::CheckGradients(model.parameters, model.Gradient(input, e), "all beta-weighted derivatives finite",
                                    "all10posterior/decoder tensors receive gradients");
        nn::save_npz((directory / "first_gradients.npz").string(), g);
        nn::save_npz((directory / "first_batch.npz").string(), {{"images", input}, {"epsilon", e}});
      }
      sum += double(rows.size()) * model.Update(input, e);
      seen += int(rows.size());
      for (auto row : rows) batches << epoch << '\t' << step << '\t' << row << '\t' << fit_source[row] << '\n';
      if (step == 0) nn::save_npz((directory / "first_update_arrays.npz").string(), ae::Arrays(model.parameters));
      ++step;
    }
    check::Require(seen == 210, "all210fitting images once per epoch");
    auto score = ae::BetaVaeScore(model, pools[1].images, eval_noise[1], beta);
    history << epoch << '\t' << sum / seen << '\t' << score[0] << '\t' << score[1] << '\t' << score[2] << '\t'
            << score[3] << '\n';
    nn::save_npz((directory / ("epoch" + std::to_string(epoch) + "_arrays.npz")).string(),
                 ae::Arrays(model.parameters));
    if (score[2] < best) {
      best     = score[2];
      selected = epoch;
      model.parameters.save(best_path);  // COMMON unweighted 4-sample validation negative ELBO selects
    }
    std::cout << "beta" << beta << " epoch" << epoch << ":NLL=" << score[0] << ",KL=" << score[1]
              << ",commonnegativeELBO=" << score[2] << ",weightedobjective=" << score[3] << std::endl;
  }
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  ae::BetaVariationalAutoencoderModel deployed(beta, false);
  deployed.parameters.load(best_path);  // fresh betaVAE inference restore
  const auto before = ae::Snapshot(deployed.parameters);
  check::Near(ae::BetaVaeScore(deployed, pools[1].images, eval_noise[1], beta)[2], best, 1e-8,
              "fresh common selected validation score");
  nn::NamedTensors output;
  for (int split = 0; split < 3; ++split) {
    const std::string  name = split == 0 ? "fit" : split == 1 ? "validation" : "test";
    const std::int64_t n    = pools[split].images.size(0);
    auto               mean = deployed.Evaluate(pools[split].images, nn::zeros({n, 4}));
    check::Require(ae::Bytes(mean[2]) == ae::Bytes(mean[0]), "epsilon0 yields EXACT posterior mean");
    std::vector<nn::Tensor> zs, prob, nll;
    for (int k = 0; k < 4; ++k) {
      auto draw = deployed.Evaluate(pools[split].images, ae::BetaVaeNoiseAt(eval_noise[split], k));
      check::Require(ae::Bytes(draw[0]) == ae::Bytes(mean[0]) && ae::Bytes(draw[1]) == ae::Bytes(mean[1]),
                     "noise changes samples,not posterior parameters");
      zs.push_back(draw[2]);
      prob.push_back(draw[3]);
      nll.push_back(draw[4]);
    }
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
    double sum = 0;
    for (int b = 0; b < 210; ++b) sum += fit_mu[b * 4 + d];
    center[d] = float(sum / 210);
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
  auto          score = ae::BetaVaeScore(deployed, pools[2].images, eval_noise[2], beta);
  const double  mse   = ae::Mse(ae::Find(output, "test_mean_prediction"), pools[2].images);
  std::ofstream meta(directory / "metrics.json");
  meta.precision(17);
  meta << "{\"beta\":" << beta << ",\"parameters\":51416,\"epochs\":" << epochs << ",\"updates\":" << step
       << ",\"selected_epoch\":" << selected
       << ",\"mc_samples\":4,\"checkpoint_criterion\":\"common_unweighted_validation_negative_elbo\","
          "\"initial_test_negative_elbo\":"
       << initial_score[2] << ",\"test_nll\":" << score[0] << ",\"test_kl\":" << score[1]
       << ",\"test_negative_elbo\":" << score[2] << ",\"test_weighted_objective\":" << score[3]
       << ",\"mean_decode_test_mse\":" << mse << ",\"single_batch_max_error\":" << parity
       << ",\"training_seconds\":" << seconds << "}\n";
  check::Require(bool(meta) && bool(history) && bool(batches), "complete betaVAE evidence");
  std::cout << "beta" << beta << ":selected" << selected << ",testNLL=" << score[0] << ",KL=" << score[1]
            << ",commonnegativeELBO=" << score[2] << ",weighted=" << score[3] << ",meanMSE=" << mse
            << ";ridge R2 shape/size/x/y=" << r2[1][0] << "/" << r2[1][1] << "/" << r2[1][2] << "/" << r2[1][3]
            << ";restore/decode/noise/batch/immutable passed\n";
  return seconds;
}

void Run(const std::filesystem::path& data, const std::filesystem::path& root, int epochs, const std::string& init) {
  auto pools = ae::LoadBetaFactors(data);
  std::filesystem::create_directories(root);
  ae::SaveFactorData(root, pools);
  if (std::filesystem::exists(data / "manifest.json"))
    std::filesystem::copy_file(data / "manifest.json", root / "data_manifest.json",
                               std::filesystem::copy_options::overwrite_existing);
  std::vector<ae::VaeNoiseBank> train_noise,
      eval_noise{ae::BetaVaeNoise(210, 4, 2273), ae::BetaVaeNoise(42, 4, 2267), ae::BetaVaeNoise(42, 4, 2269)};
  auto             prior_noise = ae::BetaVaeNoise(20, 1, 2281);
  nn::NamedTensors noise_arrays{{"prior_standard", prior_noise.standard}, {"prior_epsilon", prior_noise.epsilon}};
  for (int split = 0; split < 3; ++split) {
    const std::string name = split == 0 ? "fit" : split == 1 ? "validation" : "test";
    noise_arrays.emplace_back(name + "_standard", eval_noise[split].standard);
    noise_arrays.emplace_back(name + "_epsilon", eval_noise[split].epsilon);
  }
  nn::save_npz((root / "fixed_noise.npz").string(), noise_arrays);
  for (int epoch = 1; epoch <= epochs; ++epoch) {
    train_noise.push_back(ae::BetaVaeNoise(210, 1, 2251 + epoch));
    nn::save_npz((root / ("train_noise_epoch" + std::to_string(epoch) + ".npz")).string(),
                 {{"standard", train_noise.back().standard}, {"epsilon", train_noise.back().epsilon}});
  }
  ae::BetaVariationalAutoencoderModel first(1, false), second(4, false);
  if (!init.empty()) first.parameters.load(init);
  auto initial = ae::Arrays(first.parameters);
  second.Load(initial);
  const std::vector<std::int64_t> eight{0, 1, 2, 3, 4, 5, 6, 7};
  auto images = nn::take_rows(pools[0].images, eight), epsilon = nn::take_rows(ae::BetaVaeNoiseAt(eval_noise[0], 0), eight);
  auto a = first.Evaluate(images, epsilon), b = second.Evaluate(images, epsilon);
  for (int i = 0; i < 6; ++i)
    check::Require(ae::Bytes(a[i]) == ae::Bytes(b[i]), "initial posterior/sample/probabilities/NLL/KL BITEXACT across betas");
  double mean_kl = 0;
  for (float v : a[5].to_vector<float>()) mean_kl += v;
  mean_kl /= double(a[5].numel());
  check::Near(b[6].item<float>() - a[6].item<float>(), 3 * mean_kl, 2e-4,
              "beta changes weighted objective by EXACT3KL up to scalar reduction roundoff");
  nn::save_npz((root / "initial_forward.npz").string(), {{"images", images},
                                                         {"epsilon", epsilon},
                                                         {"mu", a[0]},
                                                         {"log_variance", a[1]},
                                                         {"z", a[2]},
                                                         {"probability", a[3]},
                                                         {"nll", a[4]},
                                                         {"kl", a[5]}});
  double seconds = Train(1, pools, train_noise, eval_noise, prior_noise, epochs, root, initial);
  seconds += Train(4, pools, train_noise, eval_noise, prior_noise, epochs, root, initial);
  auto one = nn::load_npz((root / "beta1" / "outputs.npz").string()),
       four = nn::load_npz((root / "beta4" / "outputs.npz").string());
  auto mean = ae::Constant(ae::PixelMean(pools[0].images), 42);
  auto draw = nn::select(ae::Find(four, "test_sample_prediction"), 0, 0);
  auto original = ae::Flatten(pools[2].images), p1 = ae::Find(one, "test_mean_prediction"),
       p4 = ae::Find(four, "test_mean_prediction");
  const std::vector<nn::Tensor> panels{original, p1, p4, mean, nn::abs(p1 - original), nn::abs(p4 - original), draw};
  std::ofstream html(root / "index.html");
  html << "<!doctype html><meta charset='utf-8'><title>Tiny known-factor betaVAE</title><style>body{font:16px "
          "system-ui;max-width:1000px;margin:30px auto}img{max-width:100%;image-rendering:pixelated}</style><h1>Beta1 "
          "versus beta4:known-factor diagnostics</h1><p>Generated28x28binary circle/square x3sizes x7xpositions "
          "x7ypositions=294unique images.210fit/42validation/42test disjoint combinations;each factor balanced. This "
          "is a tiny educational lattice,not MNIST/dSprites or a disentanglement benchmark. Factors are NOT fed to the "
          "VAE;only after training they fit recovery probes and evaluate heldout axis alignment.</p><p>Matched784->32"
          "ReLU/mu+logvariance4heads,explicit4DGaussian samples,decoder4->32ReLU->784Bernoulli "
          "probabilities,51416parameters. TrainSUM784NLL+beta*SUM4KL,MEANbatch. Both start BITEXACT same "
          "posterior/decoder mapping and use SAME source order/noise. "
       << epochs
       << "epochs EACH,only7updates/epoch. COMMON fixed4sample validation unweightedNLL+KL selects each "
          "checkpoint;weighted objectives reported separately. No capacity annealing or factor-based selection. <a "
          "href='https://openreview.net/forum?id=Sy2fzU9gl'>BetaVAE</a>.</p><p>Original,beta1decode(mu),beta4decode(mu),"
          "fit-only mean,abs beta1error,abs beta4error,beta4posterior draw probabilities. Ten evenly-spaced heldout "
          "indices,unfiltered by quality. Fixed[0,1]gray/nearest4/no error amplification. Decode(mu) is NOT posterior "
          "predictive mean. TestMSE beta1="
       << ae::Mse(p1, pools[2].images) << ",beta4=" << ae::Mse(p4, pools[2].images)
       << ",mean=" << ae::Mse(mean, pools[2].images)
       << ".</p><p>Higher beta changes regularization;it does NOT guarantee useful or disentangled factors in a "
          "three-epoch single-seed run. See <a href='beta1/factor_probes.html'>beta1heldout factor probes</a> and <a "
          "href='beta4/factor_probes.html'>beta4probes</a>. Fit-only standardization,ridge.001,best-axis selection "
          "use210fit records;R2/testcorrelations use42heldout combinations. These are correlation/linear-recovery "
          "diagnostics,not MIG/DCI or the paper's classifier metric. Negative R2 and latent mixing remain visible.</p>";
  const auto source = pools[2].source.to_vector<int>(), factors = pools[2].factors.to_vector<int>();
  for (int k = 0; k < 10; ++k) {
    const int         b    = k * 41 / 9;
    const std::string file = "case" + std::to_string(k) + ".png";
    ae::CaseStrip(root / file, panels, b);
    html << "<h2>Case" << k << ",source" << source[b] << ",factors";
    for (int j = 0; j < 4; ++j) html << ' ' << factors[b * 4 + j];
    html << "</h2><img src='" << file << "'>";
  }
  for (const std::string name : {"beta1", "beta4"})
    html << "<h2>" << name
         << ":20unfiltered prior decoder means</h2><p>Same standard-normal prior codes across models. "
            "Probabilities,not Bernoulli pixel draws.</p><img src='"
         << name << "/prior_means.png'><h2>" << name
         << ":one-latent traversals</h2><p>Four rows vary one coordinate around the FIT-only mean code by-2..2,step.5;"
            "other coordinates fixed. No per-image normalization. Visually changing outputs is not proof of semantic "
            "disentanglement.</p><img src='"
         << name << "/traversals.png'>";
  check::Require(bool(html), "write beta comparison gallery");
  std::cout << "Pairedinitialmapping/source/noise/21defaultupdates each/knownfactor diagnostics complete.\ntraining time "
            << seconds << " s\nArtifacts:" << root.string() << "/index.html\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int                   epochs = 3;
    std::string           init;
    std::filesystem::path data = paths::Dataset("beta_vae"),
                          out  = paths::OutputRoot() / "models/beta_variational_autoencoder";
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
    Checks();
    Run(data, out, epochs, init);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
