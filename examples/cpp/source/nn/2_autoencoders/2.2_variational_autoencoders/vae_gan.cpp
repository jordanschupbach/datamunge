// CMD: just run-nn vae_gan
// Compact VAE-GAN on the tiny known-factor lattice. THREE epochs, isolated D->E->G Adam updates.
//
// Port of tflow's vae_gan. Preparation (in ~/projects/tflow_examples): just beta-vae-assets; data
// are read from $DATAMUNGE_DATA_DIR/beta_vae. Outputs go to output/nn/models/vae_gan. Checkpoints
// are best.npz (all three groups, "<k>/<name>") instead of SaveV2 bundles; metrics.json gains
// training_seconds. Options: epochs=N data=PATH out=PATH init=PATH (tflow's initial_arrays.npz,
// to follow its trajectory exactly).

#include <chrono>
#include <iostream>
#include <limits>

#include "common/vae_gan.hpp"

namespace {

namespace nn = datamunge::nn;

void Run(const std::filesystem::path& data, const std::filesystem::path& root, int epochs, const std::string& init) {
  auto pools = ae::LoadBetaFactors(data);
  std::filesystem::create_directories(root);
  ae::SaveFactorData(root, pools);
  std::vector<ae::VaeNoiseBank> noise, priors;
  for (int k = 0; k < 3; ++k) {
    noise.push_back(ae::BetaVaeNoise(pools[k].images.size(0), 4, 3323 + k * 6));
    priors.push_back(ae::BetaVaeNoise(pools[k].images.size(0), 4, 3347 + k * 6));
  }
  nn::NamedTensors banks;
  for (int k = 0; k < 3; ++k) {
    const std::string split = k == 0 ? "fit" : k == 1 ? "validation" : "test";
    banks.emplace_back(split + "_epsilon", noise[k].epsilon);
    banks.emplace_back(split + "_standard", noise[k].standard);
    banks.emplace_back(split + "_prior_epsilon", priors[k].epsilon);
    banks.emplace_back(split + "_prior_standard", priors[k].standard);
  }
  auto prior_noise = ae::BetaVaeNoise(20, 1, 3371);
  banks.emplace_back("generation_standard", prior_noise.standard);
  banks.emplace_back("generation_epsilon", prior_noise.epsilon);
  nn::save_npz((root / "fixed_noise.npz").string(), banks);
  ae::VaeGanModel model;
  if (!init.empty()) model.Load(nn::load_npz(init));
  const std::int64_t parameters = model.Count();
  const std::size_t  tensors    = model.Tensors();
  check::Require(parameters == 32393 && tensors == 14, "compact VAE-GAN census");
  nn::save_npz((root / "initial_arrays.npz").string(), model.Arrays());
  nn::BatchOrder order(210, 32, 3373);
  std::ofstream  history(root / "history.tsv"), batches(root / "batches.tsv");
  history.precision(17);
  history << "epoch\tvalidation_mc_mse\tvalidation_pixel_nll\tvalidation_feature_nll\tvalidation_kl\n";
  batches << "epoch\tstep\trow\tsource_id\n";
  const auto fit_source = pools[0].source.to_vector<int>();
  const auto best_path  = (root / "best.npz").string();
  double     best       = std::numeric_limits<double>::infinity();
  int        selected = 0, steps = 0;
  const auto started = std::chrono::steady_clock::now();
  for (int epoch = 1; epoch <= epochs; ++epoch) {
    auto train = ae::BetaVaeNoise(210, 1, 3389 + epoch), prior = ae::BetaVaeNoise(210, 1, 3407 + epoch);
    nn::save_npz((root / ("train_noise_epoch" + std::to_string(epoch) + ".npz")).string(),
                 {{"standard", train.standard},
                  {"epsilon", train.epsilon},
                  {"prior_standard", prior.standard},
                  {"prior_epsilon", prior.epsilon}});
    auto e = ae::BetaVaeNoiseAt(train, 0), p = ae::BetaVaeNoiseAt(prior, 0);
    order.begin_epoch();
    int seen = 0;
    for (;;) {
      auto rows = order.next();
      if (rows.empty()) break;
      auto x = nn::take_rows(pools[0].images, rows), epsilon = nn::take_rows(e, rows), z = nn::take_rows(p, rows);
      if (!steps) {
        nn::save_npz((root / "first_batch.npz").string(), {{"images", x}, {"epsilon", epsilon}, {"prior", z}});
        nn::save_npz((root / "first_forward.npz").string(), model.Evaluate(x, epsilon, z));
        for (int group = 0; group < 3; ++group)
          nn::save_npz((root / ("initial_gradients_group" + std::to_string(group) + ".npz")).string(),
                       model.Gradient(group, x, epsilon, z));
      }
      for (int group : {2, 0, 1}) {
        const auto before = steps ? nn::NamedTensors{} : model.Arrays();
        if (!steps)
          nn::save_npz((root / ("first_actual_gradients_group" + std::to_string(group) + ".npz")).string(),
                       model.Gradient(group, x, epsilon, z));
        const double loss = model.Update(group, x, epsilon, z);
        check::Require(std::isfinite(loss) && loss >= 0, "all three nonnegative finite training objectives");
        if (!steps) {
          auto after   = model.Arrays();
          bool changed = false;
          for (const auto& [key, value] : before) {
            const bool same = ae::Bytes(value) == ae::Bytes(ae::Find(after, key));
            if (key[0] - '0' == group)
              changed |= !same;
            else
              check::Require(same, "isolated update changes ONLY intended variable group");
          }
          check::Require(changed, "intended group actually updates");
          nn::save_npz((root / ("first_after_group" + std::to_string(group) + ".npz")).string(), after);
        }
      }
      for (auto row : rows) batches << epoch << '\t' << steps << '\t' << row << '\t' << fit_source[row] << '\n';
      ++steps;
      seen += int(rows.size());
    }
    check::Require(seen == 210, "all fit cases once per epoch");
    auto score = ae::VaeGanScore(model, pools[1].images, noise[1], priors[1]);
    history << epoch;
    for (auto v : score) history << '\t' << v;
    history << '\n';
    nn::save_npz((root / ("epoch" + std::to_string(epoch) + "_arrays.npz")).string(), model.Arrays());
    if (score[0] < best) {
      best     = score[0];
      selected = epoch;
      nn::save_npz(best_path, model.Arrays());  // validation MC pixel MSE selects all THREE groups
    }
    std::cout << "epoch" << epoch << ":validation MCpixelMSE=" << score[0] << ",diagnostic pixelNLL=" << score[1]
              << ",featureNLL=" << score[2] << ",KL=" << score[3] << std::endl;
  }
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();

  ae::VaeGanModel deployed(false);
  deployed.Load(nn::load_npz(best_path));  // fresh all-group model-only restore
  check::Near(ae::VaeGanScore(deployed, pools[1].images, noise[1], priors[1])[0], best, 1e-9,
              "restored selection score");
  const auto       before = deployed.Snapshot();
  nn::NamedTensors outputs;
  for (int k = 0; k < 3; ++k) {
    const std::string  split = k == 0 ? "fit" : k == 1 ? "validation" : "test";
    const std::int64_t n     = pools[k].images.size(0);
    auto mean = deployed.Evaluate(pools[k].images, nn::zeros({n, 4}), ae::BetaVaeNoiseAt(priors[k], 0));
    check::Require(ae::SameBytes(ae::Find(mean, "mu"), ae::Find(mean, "z")), "zero epsilon gives posterior mu");
    for (const auto& [key, v] : mean) outputs.emplace_back(split + "_zero_" + key, v);
    for (int draw = 0; draw < 4; ++draw) {
      auto f = deployed.Evaluate(pools[k].images, ae::BetaVaeNoiseAt(noise[k], draw), ae::BetaVaeNoiseAt(priors[k], draw));
      for (const auto& [key, v] : f) outputs.emplace_back(split + "_draw" + std::to_string(draw) + "_" + key, v);
    }
  }
  auto e = ae::BetaVaeNoiseAt(noise[2], 0), p = ae::BetaVaeNoiseAt(priors[2], 0);
  auto test = deployed.Evaluate(pools[2].images, e, p), repeat = deployed.Evaluate(pools[2].images, e, p),
       other = deployed.Evaluate(pools[2].images, ae::BetaVaeNoiseAt(noise[2], 1), p);
  check::Require(ae::SameBytes(ae::Find(test, "probability"), ae::Find(repeat, "probability")) &&
                     ae::SameBytes(ae::Find(test, "probability"), deployed.Decode(ae::Find(test, "z"))),
                 "repeat and code-only decode BITEXACT");
  check::Require(!ae::SameBytes(ae::Find(test, "z"), ae::Find(other, "z")) &&
                     !ae::SameBytes(ae::Find(test, "probability"), ae::Find(other, "probability")) &&
                     ae::SameBytes(ae::Find(test, "mu"), ae::Find(other, "mu")),
                 "noise affects latent/probability,not encoder heads");
  auto single = deployed.Evaluate(nn::take_rows(pools[2].images, {0}), nn::take_rows(e, {0}), nn::take_rows(p, {0}));
  const double parity = ae::MaxAbsDiff(ae::Find(single, "probability"), ae::Find(test, "probability"), 784);
  check::Require(parity < 2e-6, "single/batch parity");
  auto prior_z = ae::BetaVaeNoiseAt(prior_noise, 0), prior = deployed.Decode(prior_z);
  nn::save_npz((root / "prior.npz").string(), {{"z", prior_z}, {"probability", prior}});
  ae::Mosaic(root / "prior_means.png", prior, 5, 3);
  check::Require(before == deployed.Snapshot(), "all inference leaves THREE groups unchanged");
  nn::save_npz((root / "model_arrays.npz").string(), deployed.Arrays());
  nn::save_npz((root / "outputs.npz").string(), outputs);
  auto         score    = ae::VaeGanScore(deployed, pools[2].images, noise[2], priors[2]);
  auto         mean     = ae::Constant(ae::PixelMean(pools[0].images), 42);
  const double mse      = ae::Mse(ae::Find(outputs, "test_zero_probability"), pools[2].images),
               baseline = ae::Mse(mean, pools[2].images);
  std::ofstream meta(root / "metrics.json");
  meta.precision(17);
  meta << "{\"parameters\":" << parameters << ",\"tensors\":" << tensors << ",\"epochs\":" << epochs
       << ",\"steps_per_group\":" << steps << ",\"total_optimizer_updates\":" << 3 * steps
       << ",\"selected_epoch\":" << selected << ",\"mc_samples\":4,\"test_mc_mse\":" << score[0]
       << ",\"test_diagnostic_pixel_nll\":" << score[1] << ",\"test_feature_nll\":" << score[2]
       << ",\"test_kl\":" << score[3] << ",\"decode_mu_mse\":" << mse << ",\"fit_mean_mse\":" << baseline
       << ",\"single_batch_max_error\":" << parity << ",\"training_seconds\":" << seconds << "}\n";
  std::ofstream html(root / "index.html");
  html << "<!doctype html><meta charset='utf-8'><title>Compact VAE-GAN</title><style>body{font:16px "
          "system-ui;max-width:1100px;margin:30px}img{max-width:100%;image-rendering:pixelated}</style><h1>VAE-GAN:"
          "learned feature reconstruction</h1><p><a href='https://proceedings.mlr.press/v48/larsen16.html'>Larsen et "
          "al.</a> inspired compact teaching variant:encoder784-16tanh-4DGaussian,decoder4-16tanh-784sigmoid,"
          "discriminator784-8tanh-1logit. Separate E/G/D parameter groups,three Adam optimizers,explicit D then E then "
          "G recomputation. Three default epochs,21steps EACH,63optimizer updates. FeatureNLL=.5SUM8(stop(realfeature)"
          "-reconstructionfeature)^2,unit covariance and additiveconstant omitted. Encoder minimizes featureNLL+KL;"
          "generator minimizes featureNLL+non-saturating adversarial loss on posterior and prior decodes;"
          "discriminator uses realCE+.5(posteriorFakeCE+priorFakeCE). Discriminator weights are fixed during E/G steps "
          "but its input Jacobian stays live. No pixel reconstruction term in training;non-saturating choice differs "
          "from original minimax recipe. These objectives are NOT a pixel-space ELBO.</p><p>Reuse294generated binary "
          "circle/square images,210fit42validation42test. Four fixed Gaussian validation draws select pixelMSE,"
          "independent of changing discriminator feature metric. PixelNLL is an auxiliary diagnostic,NOT the optimized "
          "objective or a feature-space likelihood normalized over images. Prior generation receives ONLY z~N0I. Fresh "
          "all-group checkpoint restore,isolated updates,repeat/noise/batch/immutable checks pass.</p><p>Test "
          "MCpixelMSE="
       << score[0] << ",diagnostic pixelNLL=" << score[1] << ",learnedfeatureNLL=" << score[2] << ",KL=" << score[3]
       << ",decode(mu)MSE=" << mse << ",fit-onlymeanMSE=" << baseline
       << ". A small moving feature loss does not establish perceptual similarity. No matched trained VAE "
          "control,quality improvement or realistic-generation claim. Three-epoch weaknesses retained.</p><h2>Ten "
          "evenly spaced unfiltered heldout cases</h2><p>Original/decode(mu)/one posterior probability draw/fit-only "
          "mean,fixed0..1gray. Decode(mu) is not a posterior predictive mean;images are output probabilities,not "
          "Bernoulli pixel samples.</p>";
  const auto                    source = pools[2].source.to_vector<int>();
  const std::vector<nn::Tensor> values{ae::Flatten(pools[2].images), ae::Find(outputs, "test_zero_probability"),
                                       ae::Find(test, "probability"), mean};
  for (int k = 0; k < 10; ++k) {
    const int               b = k * 41 / 9;
    std::vector<nn::Tensor> panels;
    for (const auto& v : values) panels.push_back(nn::reshape(ae::Row(v, b), {1, 784}));
    const std::string name = "case_" + std::to_string(k) + ".png";
    ae::Mosaic(root / name, nn::concat(panels, 0), 4, 3);
    html << "<p>Source" << source[b] << "<br><img src='" << name << "'></p>";
  }
  html << "<h2>20unfiltered prior decoder probabilities</h2><img src='prior_means.png'><p>Single seed,tiny synthetic "
          "data;no semantic or perceptual-quality certification. Discriminator feature geometry evolves during "
          "fitting. All poor outputs remain visible.</p>";
  check::Require(bool(meta) && bool(html) && bool(history) && bool(batches), "complete VAE-GAN artifacts");
  std::cout << "parameters=" << parameters << ",tensors=" << tensors << ",selected" << selected
            << ",testMCmse=" << score[0] << ",decodeMuMSE=" << mse << ",fitMeanMSE=" << baseline
            << ";isolatedD/E/G/restore/code-only/noise/batch/immutable PASS\ntraining time " << seconds
            << " s\nArtifacts:" << (root / "index.html").string() << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int                   epochs = 3;
    std::string           init;
    std::filesystem::path data = paths::Dataset("beta_vae"), out = paths::OutputRoot() / "models/vae_gan";
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
