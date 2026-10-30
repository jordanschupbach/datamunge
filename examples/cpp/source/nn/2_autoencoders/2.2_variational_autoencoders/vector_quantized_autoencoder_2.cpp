// CMD: just run-nn vector_quantized_autoencoder_2
// Two-scale VQ-VAE-2 with top-conditioned bottom encoding and hierarchical autoregressive priors.
// THREE default epochs.
//
// Port of tflow's vector_quantized_autoencoder_2. Preparation (in ~/projects/tflow_examples):
// just beta-vae-assets; data are read from $DATAMUNGE_DATA_DIR/beta_vae. Outputs go to
// output/nn/models/vector_quantized_autoencoder_2. Checkpoints are best.npz instead of SaveV2
// bundles; metrics.json adds training_seconds. Options: epochs=N data=PATH out=PATH init=PATH
// (start from saved parameters, e.g. tflow's initial_arrays.npz, to follow its trajectory exactly).

#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>

#include "common/vector_quantized_autoencoder_2.hpp"

namespace {

namespace nn = datamunge::nn;

float Scalar(const nn::NamedTensors& f, const std::string& key) { return ae::Find(f, key).item<float>(); }

void Run(const std::filesystem::path& data, const std::filesystem::path& root, int epochs, const std::string& init) {
  auto pools = ae::LoadBetaFactors(data);
  std::filesystem::create_directories(root);
  ae::SaveFactorData(root, pools);
  ae::VectorQuantizedModel2 model;
  if (!init.empty()) model.parameters.load(init);
  check::Require(model.parameters.count() == 1157 && model.parameters.size() == 14, "compact spatial VQ census");
  nn::save_npz((root / "initial_arrays.npz").string(), ae::Arrays(model.parameters));
  nn::BatchOrder order(210, 32, 2803);
  std::ofstream  history(root / "history.tsv"), batches(root / "batches.tsv");
  history.precision(17);
  history << "epoch\ttrain_preupdate_objective\tvalidation_nll\tvalidation_top_codebook_loss\tvalidation_bottom_codebook_"
             "loss\tvalidation_top_commitment_loss\tvalidation_bottom_commitment_loss\tvalidation_objective\n";
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
                                    "all fourteen tensors receive VQ2 gradients");
        nn::save_npz((root / "first_gradients.npz").string(), g);
        nn::save_npz((root / "first_forward.npz").string(), model.Evaluate(images));
        auto   hierarchy     = model.HierarchyGradients(images);
        double coupling_norm = 0;
        for (float v : ae::Find(hierarchy, "bottom_encoded_square_dtop_encoded").to_vector<float>())
          coupling_norm += std::abs(v);
        check::Require(coupling_norm > 0, "upper straight-through code has nonzero bottom-encoder derivative");
        for (const std::string key : {"reconstruction_dtop_book", "reconstruction_dbottom_book"})
          for (float v : ae::Find(hierarchy, key).to_vector<float>())
            check::Near(v, 0, 0, "neither codebook receives reconstruction gradients");
        nn::save_npz((root / "first_hierarchy_gradients.npz").string(), hierarchy);
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
    history << epoch << '\t' << sum / seen << '\t' << nll;
    for (const std::string key :
         {"top_codebook_loss", "bottom_codebook_loss", "top_commitment_loss", "bottom_commitment_loss", "loss"})
      history << '\t' << Scalar(validation, key);
    history << '\n';
    nn::save_npz((root / ("epoch" + std::to_string(epoch) + "_arrays.npz")).string(), ae::Arrays(model.parameters));
    if (nll < best) {
      best     = nll;
      selected = epoch;
      model.parameters.save(best_path);  // validation reconstruction NLL selects VQ checkpoint
    }
    std::cout << "epoch" << epoch << ":validation NLL=" << nll << ",top_dictionary=" << Scalar(validation, "top_codebook_loss")
              << ",bottom_dictionary=" << Scalar(validation, "bottom_codebook_loss") << std::endl;
  }
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  ae::VectorQuantizedModel2 deployed(false);
  deployed.parameters.load(best_path);  // fresh VQ model-only restore
  check::Near(Scalar(deployed.Evaluate(pools[1].images), "reconstruction_loss"), best, 1e-8, "restored selected NLL");
  const auto          before = ae::Snapshot(deployed.parameters);
  nn::NamedTensors    output;
  std::vector<double> top_perplexity, bottom_perplexity;
  for (int split = 0; split < 3; ++split) {
    const std::string name = split == 0 ? "fit" : split == 1 ? "validation" : "test";
    auto              f    = deployed.Evaluate(pools[split].images);
    for (const std::string level : {"top", "bottom"})
      check::Require(ae::SameBytes(ae::Find(f, level + "_straight_through"), ae::Find(f, level + "_quantized")),
                     "both ST values bitexact code vectors");
    const std::int64_t n = pools[split].images.size(0);
    auto top    = ae::VectorQuantizedModel2::IndexGrid(ae::Find(f, "top_indices"), n, 49),
         bottom = ae::VectorQuantizedModel2::IndexGrid(ae::Find(f, "bottom_indices"), n, 196);
    check::Require(ae::SameBytes(ae::Find(f, "probability"), deployed.Decode(top, bottom)),
                   "both-grid code-only decode exactly matches reconstruction");
    auto tu = ae::VqUsage(top), bu = ae::VqUsage(bottom);
    top_perplexity.push_back(tu.second);
    bottom_perplexity.push_back(bu.second);
    output.emplace_back(name + "_top_code_counts", tu.first);
    output.emplace_back(name + "_bottom_code_counts", bu.first);
    output.emplace_back(name + "_top_index_grid", top);
    output.emplace_back(name + "_bottom_index_grid", bottom);
    for (const auto& [key, value] : f) output.emplace_back(name + "_" + key, value);
  }
  const auto test_probability = ae::Find(output, "test_probability");
  auto       repeated         = deployed.Evaluate(pools[2].images);
  check::Require(ae::SameBytes(ae::Find(repeated, "probability"), test_probability), "deterministic repeat bitexact");
  auto         single = deployed.Evaluate(nn::take_rows(pools[2].images, {0}));
  const double parity = ae::VqDifference(ae::Find(single, "probability"), nn::take_rows(test_probability, {0}));
  check::Require(parity < 2e-6, "VQ single/batch parity");
  auto zero_top = nn::zeros({1, 49}, nn::DType::Int32), one_top = nn::full({1, 49}, 1.0, nn::DType::Int32),
       zero_bottom = nn::zeros({1, 196}, nn::DType::Int32), one_bottom = nn::full({1, 196}, 1.0, nn::DType::Int32);
  auto         zero_decoded  = deployed.Decode(zero_top, zero_bottom);
  const double top_effect    = ae::VqDifference(zero_decoded, deployed.Decode(one_top, zero_bottom)),
               bottom_effect = ae::VqDifference(zero_decoded, deployed.Decode(zero_top, one_bottom));
  check::Require(top_effect > 1e-5 && bottom_effect > 1e-5, "decoder responds to BOTH discrete groups");
  auto probe_image = nn::take_rows(pools[2].images, {0});
  auto c0 = deployed.EncodeWithTop(probe_image, zero_top), c1 = deployed.EncodeWithTop(probe_image, one_top);
  const double conditioning_effect = ae::VqDifference(ae::Find(c0, "encoded_bottom"), ae::Find(c1, "encoded_bottom"));
  check::Require(conditioning_effect > 1e-5, "bottom encoder changes with supplied top symbols");
  nn::save_npz((root / "top_conditioning_intervention.npz").string(),
               {{"images", probe_image},
                {"zero_top_encoded_bottom", ae::Find(c0, "encoded_bottom")},
                {"one_top_encoded_bottom", ae::Find(c1, "encoded_bottom")},
                {"zero_top_bottom_indices", ae::Find(c0, "bottom_indices")},
                {"one_top_bottom_indices", ae::Find(c1, "bottom_indices")}});
  for (int bad : {-1, 16}) {
    auto bad_top = nn::full({1, 49}, double(bad), nn::DType::Int32),
         bad_bottom = nn::full({1, 196}, double(bad), nn::DType::Int32);
    for (int which = 0; which < 2; ++which) {
      bool rejected = false;
      try {
        deployed.Decode(which ? zero_top : bad_top, which ? bad_bottom : zero_bottom);
      } catch (const std::exception&) {
        rejected = true;
      }
      check::Require(rejected, "invalid symbol rejected in either vocabulary");
    }
  }
  const auto fit_top = ae::Find(output, "fit_top_index_grid"), fit_bottom = ae::Find(output, "fit_bottom_index_grid"),
             test_top = ae::Find(output, "test_top_index_grid"), test_bottom = ae::Find(output, "test_bottom_index_grid");
  ae::VqMarkovPrior       top_prior(fit_top);
  ae::Vq2ConditionalPrior bottom_prior(fit_top, fit_bottom);
  auto top_sample        = top_prior.Sample(20, 2827);
  auto bottom_sample     = bottom_prior.Sample(top_sample.first, 2833);
  auto prior_probability = deployed.Decode(top_sample.first, bottom_sample.first);
  const double prior_top_fit = top_prior.Nll(fit_top), prior_top_test = top_prior.Nll(test_top),
               prior_bottom_fit = bottom_prior.Nll(fit_top, fit_bottom),
               prior_bottom_test = bottom_prior.Nll(test_top, test_bottom);
  nn::save_npz((root / "prior.npz").string(), {{"top_counts", top_prior.counts()},
                                               {"top_probabilities", top_prior.probabilities()},
                                               {"bottom_counts", bottom_prior.counts()},
                                               {"bottom_probabilities", bottom_prior.probabilities()},
                                               {"sample_top_ids", top_sample.first},
                                               {"sample_bottom_ids", bottom_sample.first},
                                               {"top_uniform", top_sample.second},
                                               {"bottom_uniform", bottom_sample.second},
                                               {"probability", prior_probability}});
  ae::Mosaic(root / "prior_means.png", prior_probability, 5, 3);
  check::Require(before == ae::Snapshot(deployed.parameters),
                 "hierarchical prior/decoding/conditioning diagnostics preserve all parameters");
  nn::save_npz((root / "model_arrays.npz").string(), ae::Arrays(deployed.parameters));
  nn::save_npz((root / "outputs.npz").string(), output);
  const double mse      = ae::Mse(test_probability, pools[2].images);
  auto         mean     = ae::Constant(ae::PixelMean(pools[0].images), 42);
  const double mean_mse = ae::Mse(mean, pools[2].images);
  const auto   tc = ae::Find(output, "fit_top_code_counts").to_vector<int>(),
             bc = ae::Find(output, "fit_bottom_code_counts").to_vector<int>();
  int top_used = 0, bottom_used = 0;
  for (int k = 0; k < 16; ++k) {
    top_used += tc[k] > 0;
    bottom_used += bc[k] > 0;
  }
  const double  test_nll = Scalar(output, "test_reconstruction_loss");
  std::ofstream meta(root / "metrics.json");
  meta.precision(17);
  meta << "{\"parameters\":1157,\"tensors\":14,\"epochs\":" << epochs << ",\"updates\":" << step
       << ",\"selected_epoch\":" << selected
       << ",\"vocabulary_each\":16,\"embedding_dimensions\":4,\"top_tokens\":49,\"bottom_tokens\":196,\"commitment_"
          "beta\":0.25,\"test_nll\":"
       << test_nll << ",\"test_objective\":" << Scalar(output, "test_loss") << ",\"test_mse\":" << mse
       << ",\"fit_mean_test_mse\":" << mean_mse << ",\"fit_top_used\":" << top_used
       << ",\"fit_bottom_used\":" << bottom_used << ",\"fit_top_perplexity\":" << top_perplexity[0]
       << ",\"fit_bottom_perplexity\":" << bottom_perplexity[0] << ",\"test_top_perplexity\":" << top_perplexity[2]
       << ",\"test_bottom_perplexity\":" << bottom_perplexity[2] << ",\"prior_top_fit_nll\":" << prior_top_fit
       << ",\"prior_top_test_nll\":" << prior_top_test << ",\"prior_bottom_fit_nll\":" << prior_bottom_fit
       << ",\"prior_bottom_test_nll\":" << prior_bottom_test << ",\"single_batch_max_error\":" << parity
       << ",\"top_symbol_max_effect\":" << top_effect << ",\"bottom_symbol_max_effect\":" << bottom_effect
       << ",\"top_conditioning_bottom_encoded_max_effect\":" << conditioning_effect
       << ",\"training_seconds\":" << seconds << "}\n";
  std::ofstream html(root / "index.html");
  html << "<!doctype html><meta charset='utf-8'><title>Two-scale VQ-VAE-2</title><style>body{font:16px "
          "system-ui;max-width:1100px;margin:30px}img{max-width:100%;image-rendering:pixelated}</style><h1>Two-scale "
          "VQ-VAE-2</h1><p>7x7upper and14x14lower grids,SEPARATE16x4codebooks. Lower encoder is conditioned on decoded "
          "upper discrete codes;decoder consumes BOTH quantized groups. Exact-forward straight-through reconstruction "
          "gradients reach encoders,not either codebook. Dictionary/commitment(beta.25) losses at both "
          "levels.1157parameters/14tensors,THREEepochs/21Adam updates. Compact teaching adaptation with first-order "
          "Markov priors,not the paper's large residual networks or PixelCNN priors. <a "
          "href='https://arxiv.org/abs/1906.00446'>VQ-VAE-2 paper</a>.</p><p>Tiny294binaryimages210fit42val42test,"
          "factors never feed fitting or checkpoint selection. Validation conditional reconstruction NLL selects "
          "checkpoint. TestNLL="
       << test_nll << ",MSE=" << mse << ",fitmeanMSE=" << mean_mse << ". FIT usedcodes upper=" << top_used
       << "/16,lower=" << bottom_used << "/16;perplexities=" << top_perplexity[0] << '/' << bottom_perplexity[0]
       << ". Dead codes remain visible. No quality or semantic coarse/fine separation claim.</p><p>Original,"
          "reconstruction,absolute error,fit-only mean,uppercode-ID grid,lowercode-ID grid. Ten evenly spaced unfiltered "
          "heldout cases,fixed0..1gray,code IDs categorical0black..15white,not semantic ordering. Deterministic "
          "probabilities,not binary samples.</p>";
  const auto x = pools[2].images.contiguous().to_vector<float>(), y = test_probability.to_vector<float>(),
             m = mean.to_vector<float>();
  const auto tg = test_top.to_vector<int>(), bg = test_bottom.to_vector<int>();
  const auto src = pools[2].source.to_vector<int>();
  for (int k = 0; k < 10; ++k) {
    const int          b = k * 41 / 9;
    std::vector<float> panels(6 * 784);
    for (int i = 0; i < 784; ++i) {
      const float xi = x[b * 784 + i], yi = y[b * 784 + i];
      panels[i]           = xi;
      panels[784 + i]     = yi;
      panels[2 * 784 + i] = std::abs(yi - xi);
      panels[3 * 784 + i] = m[b * 784 + i];
      const int row = i / 28, col = i % 28;
      panels[4 * 784 + i] = tg[b * 49 + (row / 4) * 7 + col / 4] / 15.f;
      panels[5 * 784 + i] = bg[b * 196 + (row / 2) * 14 + col / 2] / 15.f;
    }
    const auto file = "case" + std::to_string(k) + ".png";
    ae::Mosaic(root / file, nn::from_vector(panels, {6, 784}), 6, 4);
    html << "<h2>Case" << k << ",source" << src[b] << "</h2><img src='" << file << "'>";
  }
  html << "<h2>20unfiltered hierarchical-prior decoder means</h2><p>After model selection,FIT ONLY upper raster Markov "
          "p(top),then lower conditional raster Markov p(bottom|top,parent-aligned upper symbol,previouslower). Both "
          "tables use pseudocount.5. Sample uppergrid first,then lowergrid,then decode both without images. This cheap "
          "two-stage baseline is not PixelCNN. Joint CODEGRID testNLL="
       << prior_top_test + prior_bottom_test << "nats (upper=" << prior_top_test
       << ",lowerconditional=" << prior_bottom_test
       << ").49+196=245symbols x4=980rawfixed-width bits,excluding tables/decoder/framing;no finished codec,image "
          "likelihood or compression superiority claim.</p><img src='prior_means.png'>";
  check::Require(bool(meta) && bool(html) && bool(history) && bool(batches), "complete VQ2 evidence");
  std::cout << "selected" << selected << ",testNLL=" << test_nll << ",MSE=" << mse << ",fitmeanMSE=" << mean_mse
            << ",used top/bottom=" << top_used << '/' << bottom_used << ",perplexities=" << top_perplexity[0] << '/'
            << bottom_perplexity[0] << ",priorjoint testNLL=" << prior_top_test + prior_bottom_test
            << ";restore/two-ID-grid decode/cross-levelgradient/conditioning/batch/invalid IDs/immutable passed\n"
               "training time "
            << seconds << " s\nArtifacts:" << root.string() << "/index.html\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int                   epochs = 3;
    std::string           init;
    std::filesystem::path data = paths::Dataset("beta_vae"),
                          out  = paths::OutputRoot() / "models/vector_quantized_autoencoder_2";
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
