// CMD: just run-nn info_variational_autoencoder
// Prepare: just beta-vae-assets. Matched InfoVAE/MMD-VAE, THREE default epochs EACH.
//
// Port of tflow's info_variational_autoencoder. Preparation (in ~/projects/tflow_examples):
// just beta-vae-assets; data are read from $DATAMUNGE_DATA_DIR/beta_vae (the 294-image
// known-factor lattice). Outputs go to output/nn/models/info_variational_autoencoder
// (coefficient_algebra.npz, info_vae/ and mmd_vae/ subdirectories, index.html).
// Checkpoints are best.npz instead of SaveV2 bundles; metrics.json gains training_seconds.
// Options: epochs=N data=PATH out=PATH init=PATH (shared initial parameters for both variants,
// e.g. tflow's info_vae/initial_arrays.npz, to follow its trajectory exactly).

#include <chrono>
#include <iostream>
#include <limits>

#include "common/info_variational_autoencoder.hpp"

namespace {

namespace nn = datamunge::nn;

double Difference(const nn::Tensor& a, const nn::Tensor& b) {
  check::Require(a.shape() == b.shape(), "matching outputs");
  const auto x = a.contiguous().to_vector<float>(), y = b.contiguous().to_vector<float>();
  double     d = 0;
  for (size_t i = 0; i < x.size(); ++i) d = std::max(d, std::abs(double(x[i]) - y[i]));
  return d;
}
float Scalar(const nn::NamedTensors& f, const std::string& name) { return ae::Find(f, name).item<float>(); }

/// {NLL, KL, MMD, objective, negative ELBO} averaged over 4 fixed posterior/prior draws.
std::vector<double> Score(ae::InfoVariationalAutoencoderModel& model, const nn::Tensor& x, const ae::VaeNoiseBank& noise,
                          const ae::VaeNoiseBank& prior) {
  std::vector<double> sum(5);
  const auto          n = x.size(0);
  for (int k = 0; k < 4; ++k) {
    auto       f   = model.Evaluate(x, ae::BetaVaeNoiseAt(noise, k), ae::BetaVaeNoiseAt(prior, k));
    const auto nll = ae::Find(f, "nll").to_vector<float>(), kl = ae::Find(f, "kl").to_vector<float>();
    double     a = 0, b = 0;
    for (std::int64_t i = 0; i < n; ++i) a += nll[i], b += kl[i];
    sum[0] += a / n;
    sum[1] += b / n;
    sum[2] += Scalar(f, "mmd");
    sum[3] += Scalar(f, "loss");
    sum[4] += Scalar(f, "negative_elbo");
  }
  for (auto& v : sum) v /= 4;
  return sum;
}

void Algebra(const std::filesystem::path& root) {
  auto nll = nn::tensor({4., 6.}).requires_grad_(), kl = nn::tensor({2., 4.}).requires_grad_(),
       mmd = nn::scalar(.25).requires_grad_();
  auto info = ae::InfoVaeObjective(nll, kl, mmd, .5f, 10.f), mmdonly = ae::InfoVaeObjective(nll, kl, mmd, 1.f, 10.f),
       vae = ae::InfoVaeObjective(nll, kl, mmd, 0.f, 1.f);
  auto g  = nn::grad(info, {nll, kl, mmd});
  check::Near(info.item<float>(), 8.875, 0, "InfoVAE .5KL+9.5MMD");
  check::Near(mmdonly.item<float>(), 7.5, 0, "MMDVAE0KL+10MMD");
  check::Near(vae.item<float>(), 8, 0, "alpha0lambda1 EXACTbaseVAE");
  const auto gn = g[0].to_vector<float>(), gk = g[1].to_vector<float>();
  check::Require(gn == std::vector<float>{.5f, .5f} && gk == std::vector<float>{.25f, .25f} &&
                     g[2].item<float>() == 9.5f,
                 "InfoVAE coefficient partials");
  nn::NoGradGuard guard;
  nn::save_npz((root / "coefficient_algebra.npz").string(), {{"info", info.detach()},
                                                             {"mmd_only", mmdonly.detach()},
                                                             {"vae", vae.detach()},
                                                             {"gradient_nll", g[0]},
                                                             {"gradient_kl", g[1]},
                                                             {"gradient_mmd", g[2]}});
}

double Run(const std::vector<ae::BetaFactorData>& pools, const std::filesystem::path& root, int epochs, float alpha,
           const std::string& init) {
  std::filesystem::create_directories(root);
  std::vector<ae::VaeNoiseBank> priors, noise;
  nn::NamedTensors              bank;
  for (int k = 0; k < 3; ++k) {
    const std::string split = k == 0 ? "fit" : k == 1 ? "validation" : "test";
    noise.push_back(ae::BetaVaeNoise(pools[k].images.size(0), 4, 3911 + k * 6));
    priors.push_back(ae::BetaVaeNoise(pools[k].images.size(0), 4, 3943 + k * 6));
    bank.emplace_back(split + "_standard", noise[k].standard);
    bank.emplace_back(split + "_epsilon", noise[k].epsilon);
    bank.emplace_back(split + "_prior_standard", priors[k].standard);
    bank.emplace_back(split + "_prior_epsilon", priors[k].epsilon);
  }
  auto generation = ae::BetaVaeNoise(20, 1, 3967);
  bank.emplace_back("generation_standard", generation.standard);
  bank.emplace_back("generation_epsilon", generation.epsilon);
  ae::SaveFactorData(root, pools);
  nn::save_npz((root / "fixed_noise.npz").string(), bank);
  ae::InfoVariationalAutoencoderModel model(alpha, 10);
  if (!init.empty()) model.parameters.load(init);
  check::Require(model.parameters.count() == 26104 && model.parameters.size() == 10, "small InfoVAE census");
  nn::save_npz((root / "initial_arrays.npz").string(), ae::Arrays(model.parameters));
  nn::BatchOrder order(210, 32, 3989);
  std::ofstream  history(root / "history.tsv"), batches(root / "batches.tsv");
  history.precision(17);
  history << "epoch\tvalidation_nll\tvalidation_kl\tvalidation_mmd\tvalidation_objective\tvalidation_negative_elbo\n";
  batches << "epoch\tstep\trow\tsource_id\n";
  const auto fit_source = pools[0].source.to_vector<int>();
  const auto best_path  = (root / "best.npz").string();
  double     best       = std::numeric_limits<double>::infinity();
  int        selected = 0, steps = 0;
  const auto started = std::chrono::steady_clock::now();
  for (int epoch = 1; epoch <= epochs; ++epoch) {
    auto e = ae::BetaVaeNoise(210, 1, 4001 + epoch), p = ae::BetaVaeNoise(210, 1, 4013 + epoch);
    nn::save_npz((root / ("train_noise_epoch" + std::to_string(epoch) + ".npz")).string(),
                 {{"standard", e.standard},
                  {"epsilon", e.epsilon},
                  {"prior_standard", p.standard},
                  {"prior_epsilon", p.epsilon}});
    auto epsilon = ae::BetaVaeNoiseAt(e, 0), prior = ae::BetaVaeNoiseAt(p, 0);
    order.begin_epoch();
    int seen = 0;
    for (;;) {
      auto rows = order.next();
      if (rows.empty()) break;
      auto x = nn::take_rows(pools[0].images, rows), ep = nn::take_rows(epsilon, rows), pr = nn::take_rows(prior, rows);
      if (steps == 0) {
        nn::save_npz((root / "first_batch.npz").string(), {{"images", x}, {"epsilon", ep}, {"prior", pr}});
        nn::save_npz((root / "first_forward.npz").string(), model.Evaluate(x, ep, pr));
        nn::save_npz((root / "first_gradients.npz").string(),
                     ae::CheckGradients(model.parameters, model.Gradient(x, ep, pr), "all InfoVAE derivatives finite",
                                        "all10tensors receive InfoVAE gradients"));
      }
      model.Update(x, ep, pr);
      if (steps == 0) nn::save_npz((root / "first_update_arrays.npz").string(), ae::Arrays(model.parameters));
      for (auto row : rows) batches << epoch << '\t' << steps << '\t' << row << '\t' << fit_source[row] << '\n';
      ++steps;
      seen += int(rows.size());
    }
    check::Require(seen == 210, "all fit cases once per epoch");
    auto score = Score(model, pools[1].images, noise[1], priors[1]);
    history << epoch;
    for (auto v : score) history << '\t' << v;
    history << '\n';
    nn::save_npz((root / ("epoch" + std::to_string(epoch) + "_arrays.npz")).string(), ae::Arrays(model.parameters));
    if (score[3] < best) {
      best     = score[3];
      selected = epoch;
      model.parameters.save(best_path);  // fixed 4-draw validation InfoVAE objective selects checkpoint
    }
    std::cout << "alpha" << alpha << ",epoch" << epoch << ":valNLL=" << score[0] << ",KL=" << score[1]
              << ",MMD=" << score[2] << ",objective=" << score[3] << std::endl;
  }
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  ae::InfoVariationalAutoencoderModel deployed(alpha, 10, false);
  deployed.parameters.load(best_path);  // fresh InfoVAE model-only restore
  check::Near(Score(deployed, pools[1].images, noise[1], priors[1])[3], best, 1e-8,
              "restored fixednoise validation objective");
  const auto       before = ae::Snapshot(deployed.parameters);
  nn::NamedTensors outputs;
  for (int k = 0; k < 3; ++k) {
    const std::string split = k == 0 ? "fit" : k == 1 ? "validation" : "test";
    auto mean = deployed.Evaluate(pools[k].images, nn::zeros({pools[k].images.size(0), 4}), ae::BetaVaeNoiseAt(priors[k], 0));
    check::Require(ae::Bytes(ae::Find(mean, "z")) == ae::Bytes(ae::Find(mean, "mu")), "epsilon0 gives mu");
    for (const auto& [key, v] : mean) outputs.emplace_back(split + "_zero_" + key, v);
    for (int draw = 0; draw < 4; ++draw) {
      auto f = deployed.Evaluate(pools[k].images, ae::BetaVaeNoiseAt(noise[k], draw), ae::BetaVaeNoiseAt(priors[k], draw));
      for (const auto& [key, v] : f) outputs.emplace_back(split + "_draw" + std::to_string(draw) + "_" + key, v);
    }
  }
  auto ep = ae::BetaVaeNoiseAt(noise[2], 0), pr = ae::BetaVaeNoiseAt(priors[2], 0);
  auto test = deployed.Evaluate(pools[2].images, ep, pr), repeat = deployed.Evaluate(pools[2].images, ep, pr),
       other = deployed.Evaluate(pools[2].images, ae::BetaVaeNoiseAt(noise[2], 1), pr);
  const auto probability = ae::Find(test, "probability");
  check::Require(ae::SameBytes(probability, ae::Find(repeat, "probability")) &&
                     ae::SameBytes(probability, deployed.Decode(ae::Find(test, "z"))),
                 "repeat/code-only decoder BITEXACT");
  check::Require(ae::SameBytes(ae::Find(test, "mu"), ae::Find(other, "mu")) &&
                     !ae::SameBytes(ae::Find(test, "z"), ae::Find(other, "z")) &&
                     !ae::SameBytes(probability, ae::Find(other, "probability")),
                 "Gaussiannoise affects z/decoder,not posteriorheads");
  auto small = deployed.Evaluate(nn::take_rows(pools[2].images, {0, 1}), nn::take_rows(ep, {0, 1}),
                                 nn::take_rows(pr, {0, 1, 2}));
  const double parity = Difference(ae::Find(small, "probability"), nn::take_rows(probability, {0, 1}));
  check::Require(parity < 2e-6, "posterior reconstruction smallbatch/unequalprior parity");
  auto prior_z = ae::BetaVaeNoiseAt(generation, 0), prior = deployed.Decode(prior_z);
  nn::save_npz((root / "prior.npz").string(), {{"z", prior_z}, {"probability", prior}});
  nn::save_npz((root / "outputs.npz").string(), outputs);
  nn::save_npz((root / "model_arrays.npz").string(), ae::Arrays(deployed.parameters));
  check::Require(before == ae::Snapshot(deployed.parameters), "InfoVAE all inference immutable");
  auto         score = Score(deployed, pools[2].images, noise[2], priors[2]);
  auto         mean  = ae::Constant(ae::PixelMean(pools[0].images), 42);
  const auto   zero_probability = ae::Find(outputs, "test_zero_probability");
  const double mse = ae::Mse(zero_probability, pools[2].images), baseline = ae::Mse(mean, pools[2].images);
  std::ofstream meta(root / "metrics.json");
  meta.precision(17);
  meta << "{\"parameters\":26104,\"tensors\":10,\"epochs\":" << epochs << ",\"updates\":" << steps
       << ",\"alpha\":" << alpha << ",\"lambda\":10,\"local_kl_weight\":" << 1 - alpha
       << ",\"aggregate_mmd_weight\":" << alpha + 9 << ",\"selected_epoch\":" << selected
       << ",\"mc_samples\":4,\"test_nll\":" << score[0] << ",\"test_kl\":" << score[1] << ",\"test_mmd\":" << score[2]
       << ",\"test_objective\":" << score[3] << ",\"test_diagnostic_negative_elbo\":" << score[4]
       << ",\"decode_mu_mse\":" << mse << ",\"fit_mean_mse\":" << baseline << ",\"small_batch_max_error\":" << parity
       << ",\"training_seconds\":" << seconds << "}\n";
  ae::Mosaic(root / "prior_means.png", prior, 5, 3);
  std::ofstream html(root / "index.html");
  html << "<!doctype html><meta charset='utf-8'><title>InfoVAE / MMD-VAE</title><style>body{font:16px "
          "system-ui;margin:30px;max-width:1100px}img{max-width:100%;image-rendering:pixelated}</style><h1>InfoVAE / "
          "MMD-VAE,alpha="
       << alpha
       << "</h1><p><a href='https://arxiv.org/abs/1706.02262'>InfoVAE</a> compact Gaussian4D teaching variant. Train "
          "mean SUM784BernoulliNLL+(1-alpha)mean analyticKL+(alpha+lambda-1)unbiased aggregateIMQMMD. lambda10,alpha"
       << alpha << ",localKLweight=" << 1 - alpha << ",aggregateMMDweight=" << alpha + 9
       << ". alpha1 gives MMD-VAE with ZEROlocalKL,alpha.5 retains .5KL. Matched initialization/data/order/posterior+"
          "prior noise;THREEepochs21Adam EACH. Encoder784-16tanh-Gaussian4,decoder4-16tanh-784Bernoulli,26104params10"
          "tensors.</p><p>Aggregate regularizer uses sampled posterior z,not mu;averagedIMQc1/4/16,within diagonals "
          "excluded,cross ALLpairs,negative unbiased estimates retained,independent priorN0I. Fixed4draw validation "
          "OWNobjective selects epoch"
       << selected
       << ". Objective not an ordinary ELBO;NLL+KL reported separately as diagnostic. No mutual-information estimator "
          "or semantic/disentanglement certification. Synthetic294shapes210fit42validation42test,not an image "
          "benchmark.</p><p>TestNLL="
       << score[0] << ",KL=" << score[1] << ",MMD=" << score[2] << ",OWNobjective=" << score[3]
       << ",diagnostic negativeELBO=" << score[4] << ",decode(mu)MSE=" << mse << ",fit-onlymeanMSE=" << baseline
       << ". A single tiny short run does not establish InfoVAE superiority or prior matching.</p><p>Ten evenly "
          "spaced unfiltered heldout cases:original/decode(mu)/one posterior probability draw/fit-onlymean,fixed0..1"
          "gray. Decode(mu)is not posterior predictive mean. Outputs are probabilities,not sampled pixels.</p>";
  const auto source = pools[2].source.to_vector<int>();
  const std::vector<nn::Tensor> values{ae::Flatten(pools[2].images), zero_probability, probability, mean};
  for (int k = 0; k < 10; ++k) {
    const int               b = k * 41 / 9;
    std::vector<nn::Tensor> panels;
    for (const auto& v : values) panels.push_back(nn::take_rows(v, {b}));
    const std::string name = "case_" + std::to_string(k) + ".png";
    ae::Mosaic(root / name, nn::concat(panels, 0), 4, 3);
    html << "<p>Source" << source[b] << "<br><img src='" << name << "'></p>";
  }
  html << "<h2>20unfiltered prior decoder probabilities</h2><img src='prior_means.png'><p>Short-run weaknesses "
          "retained,no latent-information/quality claim.</p>";
  check::Require(bool(meta) && bool(html) && bool(history) && bool(batches), "complete InfoVAE artifacts");
  std::cout << "alpha" << alpha << ",selected" << selected << ",testobjective=" << score[3]
            << ",diagnosticELBO=" << score[4] << ",decodeMuMSE=" << mse << ",mean=" << baseline
            << ";coefficients/restore/noise/codeOnly/batch/immutable PASS\nArtifacts:" << (root / "index.html").string()
            << std::endl;
  return seconds;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int                   epochs = 3;
    std::string           init;
    std::filesystem::path data = paths::Dataset("beta_vae"),
                          out  = paths::OutputRoot() / "models/info_variational_autoencoder";
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
    auto pools = ae::LoadBetaFactors(data);
    std::filesystem::create_directories(out);
    Algebra(out);
    double seconds = Run(pools, out / "info_vae", epochs, .5f, init);
    seconds += Run(pools, out / "mmd_vae", epochs, 1.f, init);
    auto a = nn::load_npz((out / "info_vae" / "initial_arrays.npz").string()),
         b = nn::load_npz((out / "mmd_vae" / "initial_arrays.npz").string());
    for (const auto& [key, value] : a)
      check::Require(ae::SameBytes(value, ae::Find(b, key)), "both variants EXACTmatched initialization");
    std::ofstream html(out / "index.html");
    html << "<!doctype html><meta charset='utf-8'><title>Paired InfoVAE / MMD-VAE</title><h1>Paired InfoVAE / "
            "MMD-VAE</h1><p>Matched26104parameter initialization,data,batch order,posterior/prior draws;THREEepochs "
            "EACH,42totalAdam updates. lambda10. <a href='info_vae/index.html'>InfoVAE alpha.5:.5KL+9.5MMD</a> / <a "
            "href='mmd_vae/index.html'>MMD-VAE alpha1:0KL+10MMD</a>. Own fixed4draw validation objective selects each "
            "model;not comparable optimized ELBOs. Tiny single seed,unfiltered failures retained,no mutual-information "
            "or superiority claim.</p>";
    std::cout << "training time " << seconds << " s\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
