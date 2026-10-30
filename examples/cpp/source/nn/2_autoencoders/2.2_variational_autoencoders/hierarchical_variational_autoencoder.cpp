// CMD: just run-nn hierarchical_variational_autoencoder
// Spatial hierarchical VAE and miniature NVAE with residual Gaussian posteriors and BN/SE cell.
// THREE default epochs each.
//
// Port of tflow's hierarchical_variational_autoencoder. Preparation (in ~/projects/tflow_examples):
// just beta-vae-assets; data are read from $DATAMUNGE_DATA_DIR/beta_vae. Outputs go to
// output/nn/models/hierarchical_variational_autoencoder/{hierarchical,miniature_nvae}. Gaussian
// noise banks are bit-identical to tflow's; checkpoints are best.npz (weights + BN statistics)
// instead of SaveV2 bundles. Options: epochs=N data=PATH out=PATH init=DIR (a directory holding
// hierarchical/initial_arrays.npz and miniature_nvae/initial_arrays.npz, e.g. tflow's output
// directory, to follow its trajectories exactly).

#include <chrono>
#include <iostream>
#include <limits>

#include "common/hierarchical_variational_autoencoder.hpp"

namespace {

namespace nn = datamunge::nn;

void Checks(const std::filesystem::path& root) {
  auto pm = nn::tensor({{1.f, -2.f, .5f}}), pl = nn::tensor({{.2f, -.4f, .6f}}), zero = nn::zeros({1, 3});
  auto q  = ae::HierarchyPosterior(pm, pl, zero, zero, true);
  auto kl = ae::LadderConditionalKl(q.mu, q.log_variance, pm, pl);
  check::Near(kl.to_vector<float>()[0], 0, 0, "NVAE zero-residual KL exactly zero");
  nn::save_npz((root / "residual_algebra.npz").string(), {{"prior_mu", q.mu}, {"prior_lv", q.log_variance}, {"kl", kl}});
  std::cout << "residual posterior zero correction gives prior/KL0 passed" << std::endl;
}

double Difference(const nn::Tensor& a, const nn::Tensor& b) {
  check::Require(a.shape() == b.shape(), "matching comparison shapes");
  return ae::MaxAbsDiff(a, b, a.numel());
}

double Train(const std::filesystem::path& data, const std::filesystem::path& root, int epochs, bool nvae,
             const std::string& init) {
  auto pools = ae::LoadBetaFactors(data);
  std::filesystem::create_directories(root);
  Checks(root);
  ae::SaveFactorData(root, pools);
  std::vector<ae::VaeNoiseBank> noise{ae::HierarchyNoise(210, 4, 2621), ae::HierarchyNoise(42, 4, 2633),
                                      ae::HierarchyNoise(42, 4, 2647)},
      training;
  auto             prior_noise = ae::HierarchyNoise(20, 1, 2657);
  nn::NamedTensors fixed{{"prior_standard", prior_noise.standard}, {"prior_epsilon", prior_noise.epsilon}};
  for (int split = 0; split < 3; ++split) {
    const std::string name = split == 0 ? "fit" : split == 1 ? "validation" : "test";
    fixed.emplace_back(name + "_standard", noise[split].standard);
    fixed.emplace_back(name + "_epsilon", noise[split].epsilon);
  }
  nn::save_npz((root / "fixed_noise.npz").string(), fixed);
  for (int e = 1; e <= epochs; ++e) {
    training.push_back(ae::HierarchyNoise(210, 1, 2663 + e));
    nn::save_npz((root / ("train_noise_epoch" + std::to_string(e) + ".npz")).string(),
                 {{"standard", training.back().standard}, {"epsilon", training.back().epsilon}});
  }
  ae::HierarchicalVaeModel model(nvae);
  if (!init.empty()) model.Load(init);
  check::Require(model.parameters.size() == (nvae ? 37u : 26u) && model.parameters.count() == (nvae ? 2027 : 1977) &&
                     model.state.size() == (nvae ? 6u : 0u),
                 "spatial hierarchy/NVAE parameter and BN census");
  nn::save_npz((root / "initial_arrays.npz").string(), model.Arrays());
  nn::BatchOrder order(210, 32, 2617);
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
    auto   epsilon = ae::HierarchyNoiseAt(training[epoch - 1], 0);
    for (;;) {
      auto rows = order.next();
      if (rows.empty()) break;
      auto x = nn::take_rows(pools[0].images, rows), e = nn::take_rows(epsilon, rows);
      if (step == 0) {
        nn::save_npz((root / "first_batch.npz").string(), {{"images", x}, {"epsilon", e}});
        auto g = ae::CheckGradients(model.parameters, model.Gradient(x, e), "all hierarchy gradients finite",
                                    "all hierarchy/NVAE trainable tensors receive gradients");
        nn::save_npz((root / "first_gradients.npz").string(), g);
        nn::save_npz((root / "first_forward.npz").string(), model.Evaluate(x, e, true));
        if (nvae) nn::save_npz((root / "first_regularizer_gradients.npz").string(), model.RegularizerGradients(x, e));
      }
      sum += double(rows.size()) * model.Update(x, e);
      seen += int(rows.size());
      for (auto row : rows) batches << epoch << '\t' << step << '\t' << row << '\t' << fit_source[row] << '\n';
      if (step == 0) nn::save_npz((root / "first_update_arrays.npz").string(), model.Arrays());
      ++step;
    }
    check::Require(seen == 210, "all210fitting cases once per epoch");
    auto score = ae::HierarchyScore(model, pools[1].images, noise[1]);
    history << epoch << '\t' << sum / seen;
    for (auto value : score) history << '\t' << value;
    history << '\n';
    nn::save_npz((root / ("epoch" + std::to_string(epoch) + "_arrays.npz")).string(), model.Arrays());
    if (score[3] < best) {
      best     = score[3];
      selected = epoch;
      model.Save(best_path);  // four-sample validation ELBO selects checkpoint (weights + BN statistics)
    }
    std::cout << "epoch" << epoch << ":validation NLL=" << score[0] << ",lowerKL=" << score[1]
              << ",topKL=" << score[2] << ",negativeELBO=" << score[3] << std::endl;
  }
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  ae::HierarchicalVaeModel deployed(nvae, false);
  deployed.Load(best_path);
  check::Near(ae::HierarchyScore(deployed, pools[1].images, noise[1])[3], best, 1e-8, "fresh selected validation score");
  const auto       before = deployed.Snapshot();
  nn::NamedTensors output;
  for (int split = 0; split < 3; ++split) {
    const std::int64_t n      = pools[split].images.size(0);
    const std::string  name   = split == 0 ? "fit" : split == 1 ? "validation" : "test";
    auto               greedy = deployed.Evaluate(pools[split].images, nn::zeros({n, 51}));
    check::Require(ae::Bytes(ae::Find(greedy, "lower_z")) == ae::Bytes(ae::Find(greedy, "lower_mu")) &&
                       ae::Bytes(ae::Find(greedy, "top_z")) == ae::Bytes(ae::Find(greedy, "top_mu")),
                   "both zero-noise samples equal their recursive means");
    for (const auto& [key, value] : greedy) output.emplace_back(name + "_greedy_" + key, value);
    for (int k = 0; k < 4; ++k) {
      auto f = deployed.Evaluate(pools[split].images, ae::HierarchyNoiseAt(noise[split], k));
      for (const auto& [key, value] : f) output.emplace_back(name + "_draw" + std::to_string(k) + "_" + key, value);
    }
  }
  auto test   = deployed.Evaluate(pools[2].images, ae::HierarchyNoiseAt(noise[2], 0)),
       repeat = deployed.Evaluate(pools[2].images, ae::HierarchyNoiseAt(noise[2], 0));
  check::Require(ae::Bytes(ae::Find(test, "probability")) == ae::Bytes(ae::Find(repeat, "probability")) &&
                     ae::Bytes(ae::Find(test, "probability")) ==
                         ae::Bytes(deployed.Decode(ae::Find(test, "lower_z"), ae::Find(test, "top_z"))),
                 "repeat and target-free lower latent decode BITEXACT");
  auto single = deployed.Evaluate(nn::take_rows(pools[2].images, {0}), nn::take_rows(ae::HierarchyNoiseAt(noise[2], 0), {0}));
  const double parity = ae::MaxAbsDiff(ae::Find(single, "probability"), ae::Find(test, "probability"), 784);
  check::Require(parity < 2e-6, "single/batch parity");
  std::vector<float> lower_v(42 * 51, 0.f), top_v(42 * 51, 0.f);
  for (int b = 0; b < 42; ++b) {
    for (int j = 0; j < 49; ++j) lower_v[b * 51 + j] = .5f;
    for (int j = 49; j < 51; ++j) top_v[b * 51 + j] = .5f;
  }
  auto base = deployed.Evaluate(pools[2].images, nn::zeros({42, 51})),
       low  = deployed.Evaluate(pools[2].images, nn::from_vector(lower_v, {42, 51})),
       top  = deployed.Evaluate(pools[2].images, nn::from_vector(top_v, {42, 51}));
  check::Require(ae::Bytes(ae::Find(base, "lower_mu")) == ae::Bytes(ae::Find(low, "lower_mu")) &&
                     ae::Bytes(ae::Find(base, "prior_mu")) == ae::Bytes(ae::Find(low, "prior_mu")) &&
                     Difference(ae::Find(base, "lower_z"), ae::Find(low, "lower_z")) > 1e-4,
                 "lower noise changes sample only,not hierarchy heads");
  const double prior_effect  = Difference(ae::Find(base, "prior_mu"), ae::Find(top, "prior_mu")),
               fusion_effect = Difference(ae::Find(base, "lower_mu"), ae::Find(top, "lower_mu"));
  check::Require(prior_effect > 1e-5 && fusion_effect > 1e-5,
                 "top noise changes conditional prior AND conditional lower posterior");
  nn::save_npz((root / "top_noise_intervention.npz").string(), top);
  nn::save_npz((root / "lower_noise_intervention.npz").string(), low);
  auto prior_bank = ae::HierarchyNoiseAt(prior_noise, 0);
  auto e1 = nn::slice_axis(prior_bank, 1, 0, 49).contiguous(), z2 = nn::slice_axis(prior_bank, 1, 49, 51).contiguous();
  auto prior = deployed.Prior(z2, e1);
  check::Require(ae::Bytes(ae::Find(prior, "probability")) ==
                     ae::Bytes(deployed.Decode(ae::Find(prior, "lower_z"), ae::Find(prior, "top_z"))),
                 "ancestral hierarchical sample decodes exactly");
  nn::save_npz((root / "prior.npz").string(), prior);
  ae::Mosaic(root / "prior_means.png", ae::Find(prior, "probability"), 5, 3);
  check::Require(before == deployed.Snapshot(), "all inference/prior/intervention calls immutable");
  nn::save_npz((root / "outputs.npz").string(), output);
  nn::save_npz((root / "model_arrays.npz").string(), deployed.Arrays());
  auto         score    = ae::HierarchyScore(deployed, pools[2].images, noise[2]);
  const double mse      = ae::Mse(ae::Find(output, "test_greedy_probability"), pools[2].images);
  auto         mean     = ae::Constant(ae::PixelMean(pools[0].images), 42);
  const double mean_mse = ae::Mse(mean, pools[2].images);
  std::ofstream meta(root / "metrics.json");
  meta.precision(17);
  meta << "{\"miniature_nvae\":" << (nvae ? "true" : "false") << ",\"parameters\":" << model.parameters.count()
       << ",\"tensors\":" << model.parameters.size() << ",\"state_tensors\":" << model.state.size()
       << ",\"epochs\":" << epochs << ",\"updates\":" << step << ",\"selected_epoch\":" << selected
       << ",\"mc_samples\":4,\"test_nll\":" << score[0] << ",\"test_lower_kl\":" << score[1]
       << ",\"test_top_kl\":" << score[2] << ",\"test_negative_elbo\":" << score[3]
       << ",\"greedy_decode_test_mse\":" << mse << ",\"fit_mean_test_mse\":" << mean_mse
       << ",\"single_batch_max_error\":" << parity << ",\"top_noise_prior_mu_max_effect\":" << prior_effect
       << ",\"top_noise_posterior_mu_max_effect\":" << fusion_effect << ",\"training_seconds\":" << seconds << "}\n";
  std::ofstream html(root / "index.html");
  html << "<!doctype html><meta charset='utf-8'><title>Spatial hierarchical VAE / miniature NVAE</title><style>body{font:"
          "16px system-ui;max-width:1000px;margin:30px}img{max-width:100%;image-rendering:pixelated}</style><h1>"
       << (nvae ? "Miniature NVAE" : "Spatial hierarchical VAE")
       << "</h1><p>Global2D top group and7x7lower spatial Gaussian group. Learned p(z1|z2) and bidirectional "
          "q(z1|x,z2);decoder uses both sampled latent groups. Ordinary hierarchy uses direct posterior heads;miniature "
          "NVAE uses residual mean/logvariance corrections,one expansion/depthwise/BN/Swish/projection/SE residual cell "
          "and spectral penalty on the TWO4x1prior kernels (exact sole singular values). Tiny two-group teaching "
          "adaptation,not full NVAE's deep multiscale architecture or all-kernel spectral/warmup recipe. <a "
          "href='https://arxiv.org/abs/2007.03898'>NVAE paper</a>.</p><p>THREE default epochs,21Adam "
          "updates,210fit42validation42test from generated294binaryimages. Factors NEVER fed to the networks. BN uses "
          "batch moments for fitting,moving statistics for evaluation;both weights and statistics restored. Loss "
          "SUM784BernoulliNLL+conditionalKL49+topKL2,mean cases,plus1e-4spectral term for miniature NVAE. Four fixed "
          "inference draws select UNREGULARIZED validation ELBO. No free bits,KLannealing,posterior flow or variance "
          "clipping.</p><p>Original,recursive greedy means,one posterior probability draw,fit-only mean. Ten evenly "
          "spaced heldout cases,unfiltered,fixed0..1gray. Greedy means are not posterior predictive means. "
          "TestnegativeELBO="
       << score[3] << ",NLL=" << score[0] << ",lowerKL=" << score[1] << ",topKL=" << score[2] << ",greedyMSE=" << mse
       << ",fitmeanMSE=" << mean_mse
       << ". Short training does not establish superior quality or semantic hierarchy.</p>";
  const std::vector<nn::Tensor> values{ae::Flatten(pools[2].images), ae::Find(output, "test_greedy_probability"),
                                       ae::Find(output, "test_draw0_probability"), mean};
  const auto                    source = pools[2].source.to_vector<int>();
  for (int k = 0; k < 10; ++k) {
    const int         b    = k * 41 / 9;
    const std::string file = "case" + std::to_string(k) + ".png";
    ae::CaseStrip(root / file, values, b);
    html << "<h2>Case" << k << ",source" << source[b] << "</h2><img src='" << file << "'>";
  }
  html << "<h2>20ancestral prior decoder means</h2><p>Sample z2~N0I,then z1~p(z1|z2),then show Bernoulli "
          "probabilities. No image or encoder input.</p><img src='prior_means.png'>";
  check::Require(bool(meta) && bool(html) && bool(history) && bool(batches), "complete hierarchy evidence");
  std::cout << "selected" << selected << ",test NLL=" << score[0] << ",lowerKL=" << score[1] << ",topKL=" << score[2]
            << ",negativeELBO=" << score[3] << ",greedyMSE=" << mse << ",fitmeanMSE=" << mean_mse
            << ";restore/decode/noise hierarchy/batch/immutable passed" << std::endl;
  return seconds;
}

void Run(const std::filesystem::path& data, const std::filesystem::path& root, int epochs, const std::string& init) {
  std::filesystem::create_directories(root);
  auto initial = [&](const char* variant) {
    return init.empty() ? std::string() : (std::filesystem::path(init) / variant / "initial_arrays.npz").string();
  };
  double seconds = Train(data, root / "hierarchical", epochs, false, initial("hierarchical"));
  seconds += Train(data, root / "miniature_nvae", epochs, true, initial("miniature_nvae"));
  std::ofstream html(root / "index.html");
  html << "<!doctype html><meta charset='utf-8'><title>Hierarchical VAE / miniature NVAE</title><h1>Two spatial "
          "hierarchical VAE examples</h1><p>Three epochs each. These are different architectures,not a controlled "
          "quality comparison.</p><p><a href='hierarchical/index.html'>Direct hierarchical posterior</a></p><p><a "
          "href='miniature_nvae/index.html'>Miniature NVAE residual posterior and depthwise BN/SE cell</a></p>";
  check::Require(bool(html), "hierarchy comparison index written");
  std::cout << "training time " << seconds << " s\nArtifacts:" << (root / "index.html").string() << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int                   epochs = 3;
    std::string           init;
    std::filesystem::path data = paths::Dataset("beta_vae"),
                          out  = paths::OutputRoot() / "models/hierarchical_variational_autoencoder";
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
