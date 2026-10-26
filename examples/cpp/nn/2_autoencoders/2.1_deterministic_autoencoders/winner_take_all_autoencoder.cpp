// CMD: just run-nn winner_take_all_autoencoder
// Lifetime WTA: training-only per-feature batch winners, full ReLU deployment.
// Three default epochs per model; matched plain control.
//
// Port of tflow's winner_take_all_autoencoder (the model is
// examples/nn/common/competitive_autoencoder.hpp, shared with k_sparse_autoencoder).
// Preparation (in ~/projects/tflow_examples): just autoencoder-assets; data are read from
// $DATAMUNGE_DATA_DIR/saliency. Outputs (competitive/ and plain/ training records, data.npz,
// index.html and digit<k>.png galleries) go to output/nn/models/winner_take_all_autoencoder.
// Checkpoints are best.npz files instead of TF SaveV2 bundles. Options: epochs=N data=PATH
// out=PATH init=PATH (start both models from saved parameters, e.g. tflow's
// competitive/initial_arrays.npz, to follow its trajectories exactly; by default nn's own
// initialiser is used, identically for both models).

#include "common/competitive_autoencoder.hpp"

// Experiment workflow from examples/nn/common/competitive_autoencoder.hpp.
namespace ae {
using namespace ::ae;
inline double TrainCompetitiveAutoencoder(bool lifetime, bool plain, const std::vector<SaliencyData>& pools, int epochs,
                                          const std::filesystem::path& root, const nn::ParameterStore& initial) {
  const auto directory = root / (plain ? "plain" : "competitive");
  std::filesystem::create_directories(directory);
  CompetitiveAutoencoderModel model(lifetime, plain);
  model.Load(initial);
  check::Require(model.parameters.size() == 4 && model.parameters.count() == 101200,
                 "hard-competition and matched plain parameter census");
  nn::save_npz((directory / "initial_arrays.npz").string(), Arrays(model.parameters));
  auto initial_output = model.Evaluate(pools[2].images);
  nn::save_npz((directory / "initial_outputs.npz").string(),
               {{"code", initial_output[0]}, {"reconstruction", initial_output[1]}});
  nn::BatchOrder order(1000, 32, 1933);
  std::ofstream  history(directory / "history.tsv"), batches(directory / "batches.tsv");
  history.precision(17);
  history << "epoch\tfit_preupdate_mse\tvalidation_mse\n";
  batches << "epoch\tstep\trow\tsource_id\n";
  const auto fit_source = pools[0].source.to_vector<int>();
  const auto best_path  = (directory / "best.npz").string();
  const char* label     = plain ? "plain" : "competitive";
  double      best      = std::numeric_limits<double>::infinity();
  int         selected = 0, step = 0;
  const auto  started = std::chrono::steady_clock::now();
  for (int epoch = 1; epoch <= epochs; ++epoch) {
    order.begin_epoch();
    double sum  = 0;
    int    seen = 0;
    for (;;) {
      auto rows = order.next();
      if (rows.empty()) break;
      auto input = nn::take_rows(pools[0].images, rows);
      // Training forward (competition mask applied), reconstruction MSE against the input.
      auto x    = Flatten(input);
      auto net  = CompetitiveAutoencoder(model.parameters, x, lifetime, plain);
      auto loss = ReconstructionMse(net.training_reconstruction, x);
      if (step == 0) {
        auto probe = model.Probe(input);
        CheckCompetitiveSupport(probe, lifetime, plain);
        nn::save_npz((directory / "first_probe.npz").string(),
                     {{"raw", probe[0]}, {"mask", probe[1]}, {"training_code", probe[2]}, {"reconstruction", probe[3]}});
        nn::GradOptions keep;
        keep.retain_graph = true;
        auto gradients    = CheckGradients(model.parameters, nn::grad(loss, model.parameters.parameters(), keep),
                                           "all model MSE derivatives finite",
                                           "all model tensors receive full MSE gradients");
        nn::save_npz((directory / "first_gradients.npz").string(), gradients);
        nn::save_npz((directory / "first_batch.npz").string(), {{"images", input}});
      }
      if (step == 31) {
        auto probe = model.Probe(input);
        CheckCompetitiveSupport(probe, lifetime, plain);
        nn::save_npz((directory / "partial_probe.npz").string(), {{"images", input},
                                                                  {"raw", probe[0]},
                                                                  {"mask", probe[1]},
                                                                  {"training_code", probe[2]},
                                                                  {"reconstruction", probe[3]}});
      }
      const double loss_value = loss.item<float>();
      check::Require(std::isfinite(loss_value) && loss_value >= 0, "finite pre-update MSE");
      model.optimizer->minimize(loss);
      sum += double(rows.size()) * loss_value;
      seen += int(rows.size());
      for (auto row : rows) batches << epoch << '\t' << step << '\t' << row << '\t' << fit_source[row] << '\n';
      if (step == 0) nn::save_npz((directory / "first_update_arrays.npz").string(), Arrays(model.parameters));
      ++step;
    }
    check::Require(seen == 1000, "all fitting records exactly once per epoch");
    auto         value = model.Evaluate(pools[1].images);
    const double mse   = Mse(value[1], pools[1].images);
    history << epoch << '\t' << sum / seen << '\t' << mse << '\n';
    nn::save_npz((directory / ("epoch" + std::to_string(epoch) + "_arrays.npz")).string(), Arrays(model.parameters));
    if (mse < best) {
      best     = mse;
      selected = epoch;
      model.parameters.save(best_path);  // common validation reconstruction MSE selects each model
    }
    std::cout << label << " epoch" << epoch << ":validationMSE=" << mse << std::endl;
  }
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  CompetitiveAutoencoderModel deployed(lifetime, plain, false);
  deployed.parameters.load(best_path);  // fresh competitive/plain inference restore
  const auto              before = Snapshot(deployed.parameters);
  std::vector<std::int64_t> first32(32);
  std::iota(first32.begin(), first32.end(), 0);
  auto probe_images = nn::take_rows(pools[2].images, first32);
  auto probe        = deployed.Probe(probe_images);
  CheckCompetitiveSupport(probe, lifetime, plain);
  nn::save_npz((directory / "selected_probe.npz").string(), {{"images", probe_images},
                                                             {"raw", probe[0]},
                                                             {"mask", probe[1]},
                                                             {"training_code", probe[2]},
                                                             {"reconstruction", probe[3]}});
  nn::NamedTensors arrays;
  for (int split = 0; split < 3; ++split) {
    const std::string name = split == 0 ? "fit" : split == 1 ? "validation" : "test";
    auto              out  = deployed.Evaluate(pools[split].images);
    arrays.emplace_back(name + "_code", out[0]);
    arrays.emplace_back(name + "_reconstruction", out[1]);
    if (split == 1) check::Near(Mse(out[1], pools[split].images), best, 1e-8, "fresh selected validation MSE");
  }
  auto test = deployed.Evaluate(pools[2].images), repeat = deployed.Evaluate(pools[2].images);
  check::Require(Bytes(test[1]) == Bytes(repeat[1]) && Bytes(deployed.Decode(test[0])) == Bytes(test[1]),
                 "deterministic exact encode-decode");
  auto zero    = deployed.Decode(nn::zeros({200, 64}));
  auto shifted = deployed.Decode(nn::concat({nn::slice_axis(test[0], 0, 1, 200), nn::slice_axis(test[0], 0, 0, 1)}, 0));
  arrays.emplace_back("zero_reconstruction", zero);
  arrays.emplace_back("shuffled_reconstruction", shifted);
  auto         one   = deployed.Evaluate(nn::take_rows(pools[2].images, {0}));
  const double error = MaxAbsDiff(one[1], test[1], 784);
  check::Require(error < 1e-6, "single/batch competitive/plain inference parity");
  check::Require(before == Snapshot(deployed.parameters), "all deployed parameters immutable through code controls");
  nn::save_npz((directory / "outputs.npz").string(), arrays);
  nn::save_npz((directory / "model_arrays.npz").string(), Arrays(deployed.parameters));
  const double  test_mse = Mse(test[1], pools[2].images);
  std::ofstream meta(directory / "metrics.json");
  meta.precision(17);
  meta << "{\"lifetime\":" << (lifetime ? "true" : "false") << ",\"plain\":" << (plain ? "true" : "false")
       << ",\"parameters\":" << model.parameters.count() << ",\"epochs\":" << epochs << ",\"updates\":" << step
       << ",\"selected_epoch\":" << selected << ",\"test_mse\":" << test_mse
       << ",\"zero_mse\":" << Mse(zero, pools[2].images) << ",\"shuffled_mse\":" << Mse(shifted, pools[2].images)
       << ",\"single_batch_max_error\":" << error << ",\"training_seconds\":" << seconds << "}\n";
  check::Require(bool(meta) && bool(history) && bool(batches), "write complete competitive/plain evidence");
  std::cout << label << ":selected" << selected << ",parameters=" << model.parameters.count() << ",testMSE=" << test_mse
            << ",freshrestore/decode/repeat/immutable parameters passed (training " << seconds << " s)\n";
  return seconds;
}
}  // namespace ae

// Experiment workflow from examples/nn/common/competitive_autoencoder.hpp.
namespace ae {
using namespace ::ae;
inline void RunCompetitiveAutoencoder(bool lifetime, const std::filesystem::path& data, const std::filesystem::path& root,
                                      int epochs, const std::string& init = "") {
  auto pools = LoadSaliency(data);
  std::filesystem::create_directories(root);
  CompetitiveAutoencoderModel factory(lifetime, false, false);
  if (!init.empty()) factory.parameters.load(init);
  double seconds = TrainCompetitiveAutoencoder(lifetime, false, pools, epochs, root, factory.parameters);
  seconds += TrainCompetitiveAutoencoder(lifetime, true, pools, epochs, root, factory.parameters);
  SaveData(root / "data.npz", pools);
  CopyManifests(data, root);
  const auto t = nn::load_npz((root / "competitive" / "outputs.npz").string()),
             u = nn::load_npz((root / "plain" / "outputs.npz").string());
  const auto mean = Constant(PixelMean(pools[0].images), 200);
  const auto competitive = Find(t, "test_reconstruction"), plain = Find(u, "test_reconstruction");
  const double competitive_mse = Mse(competitive, pools[2].images), plain_mse = Mse(plain, pools[2].images),
               mean_mse        = Mse(mean, pools[2].images);
  std::ofstream html(root / "index.html");
  html << "<!doctype html><meta charset='utf-8'><title>Hard competition autoencoder</title><style>body{font:16px "
          "system-ui;max-width:1000px;margin:30px auto}img{max-width:100%;image-rendering:pixelated}section{margin:"
          "25px 0}</style><h1>"
       << (lifetime ? "Lifetime winner-take-all" : "Per-image k-sparse")
       << " autoencoder</h1><p>Original,competitive,plain control,fit-only mean,absolute competitive error,zero-code,"
          "shuffled-code. Fixed[0,1]gray/nearest4/no error amplification. First test record of each digit,unfiltered."
          "</p><p>784->64->784sigmoid,101200parameters. "
       << (lifetime ? "ReLU encoder,TRAINING ONLY lifetime winners:each feature keeps ceil(batch/8) examples,4of32 "
                      "or1of8. Deployed encoder is full ReLU with NO lifetime gate;deployed code is NOT guaranteed "
                      "sparse. This is fully-connected WTA,not spatial/convolutional WTA."
                    : "LINEAR encoder,top8largest SIGNED activities per image,not top absolute magnitude;mask is "
                      "retained in training AND inference. Exactly8selected coordinates,at most8nonzero values;"
                      "negative winners remain signed. No ReLU,no sparsity penalty or straight-through gradient.")
       << " TopK ties select lower indices;derivatives pass ONLY selected coordinates. The discrete ranking has no "
          "derivative;at ties this is one deterministic subgradient convention.</p><p>Matched initial parameter bytes "
          "and source orders;plain control removes competition. "
       << epochs
       << "epochs EACH,Adam.003,batch32/final8. Separate validation reconstruction MSE selects each deployed model. "
          "Sigmoid decoder/raw pixels/compact width are educational adaptations of the <a href='https://arxiv.org/abs/"
       << (lifetime ? "1409.2752" : "1312.5663")
       << "'>hard sparsity approach</a>,not the original scale/decoder/classification experiment. Same weights do not "
          "imply same gated initial mapping. Labels only balance pools/captions. Blurry errors/weak quality remain "
          "visible;no bit-rate or semantic/classification claim.</p><p>TestMSE competitive="
       << competitive_mse << ",plain=" << plain_mse << ",mean=" << mean_mse << ".</p>";
  DigitGallery(root, pools[2],
               {pools[2].images, competitive, plain, mean, nn::abs(competitive - Flatten(pools[2].images)),
                Find(t, "zero_reconstruction"), Find(t, "shuffled_reconstruction")},
               html);
  check::Require(bool(html), "write hard-competition gallery");
  std::cout << "Hard winner competition/matched plain controls/fresh deployment verified.\nTestMSE competitive="
            << competitive_mse << ",plain=" << plain_mse << ",mean=" << mean_mse << "\ntraining time " << seconds
            << " s\nArtifacts:" << root.string() << "/index.html\n";
}
}  // namespace ae



int main(int argc, char** argv) {
  try {
    int                   epochs = 3;
    std::string           init;
    std::filesystem::path data = paths::Dataset("saliency"), out = paths::OutputRoot() / "models/winner_take_all_autoencoder";
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
    ae::CompetitiveChecks();
    ae::RunCompetitiveAutoencoder(true, data, out, epochs, init);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
