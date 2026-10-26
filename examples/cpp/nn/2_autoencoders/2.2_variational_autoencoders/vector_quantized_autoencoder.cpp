// CMD: just run-nn vector_quantized_autoencoder
// Spatial VQ-VAE with separate straight-through/codebook/commitment gradients and learned Markov prior.
// THREE default epochs.
//
// Port of tflow's vector_quantized_autoencoder. Preparation (in ~/projects/tflow_examples):
// just beta-vae-assets; data are read from $DATAMUNGE_DATA_DIR/beta_vae. Outputs go to
// output/nn/models/vector_quantized_autoencoder. Checkpoints are best.npz instead of SaveV2
// bundles; metrics.json adds training_seconds. Options: epochs=N data=PATH out=PATH init=PATH
// (start from saved parameters, e.g. tflow's initial_arrays.npz, to follow its trajectory exactly).

#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>

#include "common/vector_quantized_autoencoder.hpp"

namespace {

namespace nn = datamunge::nn;

void Checks(const std::filesystem::path& root) {
  const std::vector<float> zv{-.5f, 0.f, .9f, .2f}, bv{-1, 0, 0, 0, 1, 0};
  auto Z = nn::from_vector(zv, {2, 2}).requires_grad_(), B = nn::from_vector(bv, {3, 2}).requires_grad_();
  auto q          = ae::VectorQuantize(Z, B, 3, 2);
  auto recon      = .5 * nn::sum(nn::square(q.straight_through), {0, 1}) + 0.0 * nn::sum(B, {0, 1});
  auto dictionary = q.codebook_loss + 0.0 * nn::sum(Z, {0, 1});
  auto commit     = .25 * q.commitment_loss + 0.0 * nn::sum(B, {0, 1});
  auto gr = nn::grad(recon, {Z, B}), gb = nn::grad(dictionary, {Z, B}), gc = nn::grad(commit, {Z, B});
  const auto ids = q.indices.to_vector<int>();
  check::Require(ids[0] == 0 && ids[1] == 2, "nearest argmin tie chooses lower index");
  check::Require(ae::SameBytes(q.quantized.detach(), q.straight_through.detach()) && ae::SameBytes(q.quantized.detach(), gr[0]),
                 "ST forward exact quantized code and identity encoder reconstruction derivative");
  for (float v : gr[1].to_vector<float>()) check::Near(v, 0, 0, "reconstruction does not update codebook");
  for (float v : gc[1].to_vector<float>()) check::Near(v, 0, 0, "commitment does not update codebook");
  for (float v : gb[0].to_vector<float>()) check::Near(v, 0, 0, "dictionary loss does not update encoder");
  nn::save_npz((root / "quantizer_algebra.npz").string(), {{"encoded", Z.detach()},
                                                           {"codebook", B.detach()},
                                                           {"indices", q.indices},
                                                           {"quantized", q.quantized.detach()},
                                                           {"straight_through", q.straight_through.detach()},
                                                           {"reconstruction_dencoded", gr[0]},
                                                           {"reconstruction_dbook", gr[1]},
                                                           {"dictionary_dencoded", gb[0]},
                                                           {"dictionary_dbook", gb[1]},
                                                           {"commitment_dencoded", gc[0]},
                                                           {"commitment_dbook", gc[1]}});
  std::cout << "nearest tie/ST exact forward/reconstruction-codebook-commitment gradient separation passed" << std::endl;
}

float Scalar(const nn::NamedTensors& f, const std::string& key) { return ae::Find(f, key).item<float>(); }

void Run(const std::filesystem::path& data, const std::filesystem::path& root, int epochs, const std::string& init) {
  auto pools = ae::LoadBetaFactors(data);
  std::filesystem::create_directories(root);
  Checks(root);
  ae::SaveFactorData(root, pools);
  ae::VectorQuantizedModel model;
  if (!init.empty()) model.parameters.load(init);
  check::Require(model.parameters.count() == 805 && model.parameters.size() == 9, "compact spatial VQ census");
  nn::save_npz((root / "initial_arrays.npz").string(), ae::Arrays(model.parameters));
  nn::BatchOrder order(210, 32, 2711);
  std::ofstream  history(root / "history.tsv"), batches(root / "batches.tsv");
  history.precision(17);
  history << "epoch\ttrain_preupdate_objective\tvalidation_nll\tvalidation_codebook_loss\tvalidation_commitment_loss\t"
             "validation_objective\n";
  batches << "epoch\tstep\trow\tsource_id\n";
  const auto fit_source = pools[0].source.to_vector<int>();
  const auto best_path  = (root / "best.npz").string();
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
      auto images = nn::take_rows(pools[0].images, rows);
      if (step == 0) {
        nn::save_npz((root / "first_batch.npz").string(), {{"images", images}});
        auto g = ae::CheckGradients(model.parameters, model.Gradient(images), "all VQ gradients finite",
                                    "all nine tensors receive VQ gradients");
        nn::save_npz((root / "first_gradients.npz").string(), g);
        nn::save_npz((root / "first_forward.npz").string(), model.Evaluate(images));
      }
      sum += double(rows.size()) * model.Update(images);
      seen += int(rows.size());
      for (auto row : rows) batches << epoch << '\t' << step << '\t' << row << '\t' << fit_source[row] << '\n';
      if (step == 0) nn::save_npz((root / "first_update_arrays.npz").string(), ae::Arrays(model.parameters));
      ++step;
    }
    check::Require(seen == 210, "all fitting cases exactly once per epoch");
    auto         validation = model.Evaluate(pools[1].images);
    const double nll        = Scalar(validation, "reconstruction_loss");
    history << epoch << '\t' << sum / seen << '\t' << nll << '\t' << Scalar(validation, "codebook_loss") << '\t'
            << Scalar(validation, "commitment_loss") << '\t' << Scalar(validation, "loss") << '\n';
    nn::save_npz((root / ("epoch" + std::to_string(epoch) + "_arrays.npz")).string(), ae::Arrays(model.parameters));
    if (nll < best) {
      best     = nll;
      selected = epoch;
      model.parameters.save(best_path);  // validation reconstruction NLL selects VQ checkpoint
    }
    std::cout << "epoch" << epoch << ":validation NLL=" << nll << ",dictionary=" << Scalar(validation, "codebook_loss")
              << ",commitment=" << Scalar(validation, "commitment_loss") << std::endl;
  }
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  ae::VectorQuantizedModel deployed(false);
  deployed.parameters.load(best_path);  // fresh VQ model-only restore
  check::Near(Scalar(deployed.Evaluate(pools[1].images), "reconstruction_loss"), best, 1e-8, "restored selected NLL");
  const auto          before = ae::Snapshot(deployed.parameters);
  nn::NamedTensors    output;
  std::vector<double> perplexity;
  for (int split = 0; split < 3; ++split) {
    const std::string name = split == 0 ? "fit" : split == 1 ? "validation" : "test";
    auto              f    = deployed.Evaluate(pools[split].images);
    check::Require(ae::SameBytes(ae::Find(f, "straight_through"), ae::Find(f, "quantized")),
                   "ST forward bitexact learned code vectors");
    auto ids = ae::VectorQuantizedModel::IndexGrid(ae::Find(f, "indices"), pools[split].images.size(0));
    check::Require(ae::SameBytes(ae::Find(f, "probability"), deployed.Decode(ids)),
                   "code-index-only decode exactly matches image encoding");
    auto usage = ae::VqUsage(ids);
    perplexity.push_back(usage.second);
    output.emplace_back(name + "_code_counts", usage.first);
    output.emplace_back(name + "_index_grid", ids);
    for (const auto& [key, value] : f) output.emplace_back(name + "_" + key, value);
  }
  const auto test_probability = ae::Find(output, "test_probability");
  auto       repeated         = deployed.Evaluate(pools[2].images);
  check::Require(ae::SameBytes(ae::Find(repeated, "probability"), test_probability), "deterministic repeat bitexact");
  auto         single = deployed.Evaluate(nn::take_rows(pools[2].images, {0}));
  const double parity = ae::VqDifference(ae::Find(single, "probability"), nn::take_rows(test_probability, {0}));
  check::Require(parity < 2e-6, "VQ single/batch parity");
  auto         zero = nn::zeros({1, 49}, nn::DType::Int32), one = nn::full({1, 49}, 1.0, nn::DType::Int32);
  const double symbol_effect = ae::VqDifference(deployed.Decode(zero), deployed.Decode(one));
  check::Require(symbol_effect > 1e-5, "decoder responds to discrete symbol changes");
  for (int bad : {-1, 16}) {
    bool rejected = false;
    try {
      deployed.Decode(nn::full({1, 49}, double(bad), nn::DType::Int32));
    } catch (const std::exception&) {
      rejected = true;
    }
    check::Require(rejected, "invalid negative or too-large symbols rejected before graph execution");
  }
  ae::VqMarkovPrior prior(ae::Find(output, "fit_index_grid"));
  auto              sampled           = prior.Sample(20, 2729);
  auto              prior_probability = deployed.Decode(sampled.first);
  const double prior_fit_nll = prior.Nll(ae::Find(output, "fit_index_grid")),
               prior_test_nll = prior.Nll(ae::Find(output, "test_index_grid"));
  nn::save_npz((root / "prior.npz").string(), {{"counts", prior.counts()},
                                               {"probabilities", prior.probabilities()},
                                               {"sample_ids", sampled.first},
                                               {"uniform", sampled.second},
                                               {"probability", prior_probability}});
  ae::Mosaic(root / "prior_means.png", prior_probability, 5, 3);
  check::Require(before == ae::Snapshot(deployed.parameters),
                 "prior fitting,decoding and diagnostics leave VQ parameters immutable");
  nn::save_npz((root / "model_arrays.npz").string(), ae::Arrays(deployed.parameters));
  nn::save_npz((root / "outputs.npz").string(), output);
  const double mse      = ae::Mse(test_probability, pools[2].images);
  auto         mean     = ae::Constant(ae::PixelMean(pools[0].images), 42);
  const double mean_mse = ae::Mse(mean, pools[2].images);
  const auto   fit_counts = ae::Find(output, "fit_code_counts").to_vector<int>();
  int          used       = 0;
  for (int k = 0; k < 16; ++k) used += fit_counts[k] > 0;
  const double test_nll = Scalar(output, "test_reconstruction_loss");
  std::ofstream meta(root / "metrics.json");
  meta.precision(17);
  meta << "{\"parameters\":805,\"tensors\":9,\"epochs\":" << epochs << ",\"updates\":" << step
       << ",\"selected_epoch\":" << selected
       << ",\"vocabulary\":16,\"embedding_dimensions\":4,\"tokens_per_image\":49,\"commitment_beta\":0.25,\"test_nll\":"
       << test_nll << ",\"test_codebook_loss\":" << Scalar(output, "test_codebook_loss")
       << ",\"test_commitment_loss\":" << Scalar(output, "test_commitment_loss")
       << ",\"test_objective\":" << Scalar(output, "test_loss") << ",\"test_mse\":" << mse
       << ",\"fit_mean_test_mse\":" << mean_mse << ",\"fit_used_codes\":" << used
       << ",\"fit_code_perplexity\":" << perplexity[0] << ",\"test_code_perplexity\":" << perplexity[2]
       << ",\"prior_fit_nll_nats_per_grid\":" << prior_fit_nll << ",\"prior_test_nll_nats_per_grid\":" << prior_test_nll
       << ",\"single_batch_max_error\":" << parity << ",\"symbol_change_max_effect\":" << symbol_effect
       << ",\"training_seconds\":" << seconds << "}\n";
  std::ofstream html(root / "index.html");
  html << "<!doctype html><meta charset='utf-8'><title>Vector quantized autoencoder</title><style>body{font:16px "
          "system-ui;max-width:1100px;margin:30px}img{max-width:100%;image-rendering:pixelated}</style><h1>VQ-VAE with a "
          "learned raster Markov prior</h1><p>28x28binary image ->conv1->8ReLU/avg2 ->conv8->4ReLU/avg2 ->7x7x4encoder "
          "codes. Each4vector snaps to one of16learned embeddings. Exact-forward straight-through derivative routes "
          "reconstruction to encoder,not codebook. Dictionary loss moves codebook only;beta.25commitment moves encoder "
          "only. Decoderrepeat2/conv4->8ReLU/repeat2/conv8->1Bernoulli.805parameters/9tensors. THREEepochs/21Adam "
          "updates. <a href='https://arxiv.org/abs/1711.00937'>VQ-VAE paper</a>.</p><p>Train mean SUM784BernoulliNLL "
          "+mean latent-element dictionary square +.25mean latent-element commitment square. No EMA updates,code resets "
          "or Gaussian KL. Validation reconstructionNLL selects checkpoint. Tiny294syntheticimages210fit42val42test;"
          "factors never feed fitting or selection. TestNLL="
       << test_nll << ",MSE=" << mse << ",fitmeanMSE=" << mean_mse << ". Fitting usedcodes=" << used
       << "/16,perplexity=" << perplexity[0]
       << ";dead codes/collapse are retained.</p><p>Original,reconstruction,absolute error,fit-only mean,code-ID "
          "grid(0black..15white). Ten evenly spaced heldout cases,unfiltered,fixed gray scales;IDs are categorical,their "
          "brightness is not an ordering of semantics. Short-run quality may remain weak. Quantized reconstruction is "
          "deterministic;probabilities,not binary draws.</p>";
  const auto x = pools[2].images.contiguous().to_vector<float>(), y = test_probability.to_vector<float>(),
             m = mean.to_vector<float>();
  const auto grid = ae::Find(output, "test_index_grid").to_vector<int>();
  const auto src  = pools[2].source.to_vector<int>();
  for (int k = 0; k < 10; ++k) {
    const int          b = k * 41 / 9;
    std::vector<float> panels(5 * 784);
    for (int i = 0; i < 784; ++i) {
      const float xi = x[b * 784 + i], yi = y[b * 784 + i];
      panels[i]           = xi;
      panels[784 + i]     = yi;
      panels[2 * 784 + i] = std::abs(yi - xi);
      panels[3 * 784 + i] = m[b * 784 + i];
      const int row = i / 28, col = i % 28;
      panels[4 * 784 + i] = grid[b * 49 + (row / 4) * 7 + col / 4] / 15.f;
    }
    const auto file = "case" + std::to_string(k) + ".png";
    ae::Mosaic(root / file, nn::from_vector(panels, {5, 784}), 5, 4);
    html << "<h2>Case" << k << ",source" << src[b] << "</h2><img src='" << file << "'>";
  }
  html << "<h2>20unfiltered learned-prior decoder means</h2><p>After selecting the autoencoder,fit smoothed raster "
          "first-order Markov categorical transitions on FIT codes only,BOS16,Dirichlet pseudocount.5. Sample49symbols "
          "sequentially and decode IDs without images. This is a cheap separate autoregressive prior baseline,not the "
          "paper's PixelCNN. Prior code-grid NLL fit="
       << prior_fit_nll << ",test=" << prior_test_nll
       << "nats/grid.49x4=196rawfixed-width symbol bits excludes codebook/decoder and framing;no finished codec or "
          "image-likelihood claim.</p><img src='prior_means.png'>";
  check::Require(bool(meta) && bool(html) && bool(history) && bool(batches), "complete VQ evidence");
  std::cout << "selected" << selected << ",testNLL=" << test_nll << ",MSE=" << mse << ",fitmeanMSE=" << mean_mse
            << ",used=" << used << "/16,fitperplexity=" << perplexity[0] << ",prior testNLL=" << prior_test_nll
            << ";restore/IDs-only decode/repeat/batch/invalid IDs/immutable passed\ntraining time " << seconds
            << " s\nArtifacts:" << root.string() << "/index.html\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int                   epochs = 3;
    std::string           init;
    std::filesystem::path data = paths::Dataset("beta_vae"),
                          out  = paths::OutputRoot() / "models/vector_quantized_autoencoder";
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
