// CMD: just run-nn recurrent_autoencoder
// Sequence-to-sequence tanh recurrent autoencoder with a final-state-only bottleneck.
//
// Port of tflow's recurrent_autoencoder. Preparation (in ~/projects/tflow_examples):
// just autoencoder-assets; data are read from $DATAMUNGE_DATA_DIR/saliency. Outputs
// (history, npz arrays, states.npz, metrics.json, index.html and digit<k>.png galleries)
// go to output/nn/models/recurrent_autoencoder. Checkpoints are best.npz files instead of
// TF SaveV2 bundles. Options: epochs=N data=PATH out=PATH init=PATH (start from saved
// parameters, e.g. tflow's initial_arrays.npz, to follow its trajectory exactly; by
// default nn's own initialiser is used).

#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>

#include "common/recurrent_autoencoder.hpp"

namespace {

namespace nn = datamunge::nn;

double Train(const std::vector<ae::SaliencyData>& pools, int epochs, const std::filesystem::path& directory,
             const std::string& init) {
  std::filesystem::create_directories(directory);
  ae::RecurrentAutoencoderModel model;
  if (!init.empty()) model.parameters.load(init);
  check::Require(model.parameters.size() == 8 && model.parameters.count() == 4956,
                 "shared-over-time recurrent parameter census");
  nn::save_npz((directory / "initial_arrays.npz").string(), ae::Arrays(model.parameters));
  auto initial = model.Evaluate(pools[2].images);
  nn::save_npz((directory / "initial_outputs.npz").string(), {{"code", initial[0]}, {"reconstruction", initial[1]}});
  nn::BatchOrder order(1000, 32, 1873);
  std::ofstream  history(directory / "history.tsv"), batches(directory / "batches.tsv");
  history.precision(17);
  history << "epoch\tfit_preupdate_mse\tvalidation_mse\n";
  batches << "epoch\tstep\trow\tsource_id\n";
  const auto fit_source = pools[0].source.to_vector<int>();
  const auto best_path  = (directory / "best.npz").string();
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
      auto input = nn::take_rows(pools[0].images, rows);
      // Full backpropagation through all 28 encoder + 28 decoder steps.
      auto sequence = ae::RecurrentAutoencoderModel::Sequence(input);
      auto loss     = ae::RecurrentAutoencoderModel::Loss(ae::RecurrentAutoencoder(model.parameters, sequence).reconstruction,
                                                          sequence);
      if (step == 0) {
        nn::GradOptions keep;
        keep.retain_graph = true;
        auto gradients    = ae::CheckGradients(model.parameters, nn::grad(loss, model.parameters.parameters(), keep),
                                               "all model MSE derivatives finite",
                                               "all model tensors receive full MSE gradients");
        nn::save_npz((directory / "first_gradients.npz").string(), gradients);
        nn::save_npz((directory / "first_batch.npz").string(), {{"images", input}});
      }
      const double value = loss.item<float>();
      check::Require(std::isfinite(value) && value >= 0, "finite pre-update MSE");
      model.optimizer->minimize(loss);
      sum += double(rows.size()) * value;
      seen += int(rows.size());
      for (auto row : rows) batches << epoch << '\t' << step << '\t' << row << '\t' << fit_source[row] << '\n';
      if (step == 0) nn::save_npz((directory / "first_update_arrays.npz").string(), ae::Arrays(model.parameters));
      ++step;
    }
    check::Require(seen == 1000, "all fitting records exactly once per epoch");
    auto         value = model.Evaluate(pools[1].images);
    const double mse   = ae::Mse(value[1], pools[1].images);
    history << epoch << '\t' << sum / seen << '\t' << mse << '\n';
    nn::save_npz((directory / ("epoch" + std::to_string(epoch) + "_arrays.npz")).string(),
                 ae::Arrays(model.parameters));
    if (mse < best) {
      best     = mse;
      selected = epoch;
      model.parameters.save(best_path);  // common validation reconstruction MSE selects the model
    }
    std::cout << "recurrent epoch" << epoch << ":validationMSE=" << mse << std::endl;
  }
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  ae::RecurrentAutoencoderModel deployed(false);
  deployed.parameters.load(best_path);  // fresh recurrent inference restore
  const auto before = ae::Snapshot(deployed.parameters);
  auto       probe  = ae::Rows(pools[2], {0, 1, 2, 3, 4, 5, 6, 7}).images;
  auto       states = deployed.States(probe);
  nn::save_npz((directory / "states.npz").string(),
               {{"images", probe}, {"encoder_states", states[0]}, {"decoder_states", states[1]}});
  nn::NamedTensors arrays;
  for (int split = 0; split < 3; ++split) {
    const std::string name = split == 0 ? "fit" : split == 1 ? "validation" : "test";
    auto              out  = deployed.Evaluate(pools[split].images);
    arrays.emplace_back(name + "_code", out[0]);
    arrays.emplace_back(name + "_reconstruction", out[1]);
    if (split == 1) check::Near(ae::Mse(out[1], pools[split].images), best, 1e-8, "fresh selected validation MSE");
  }
  auto test = deployed.Evaluate(pools[2].images), repeat = deployed.Evaluate(pools[2].images);
  check::Require(ae::Bytes(test[1]) == ae::Bytes(repeat[1]) && ae::Bytes(deployed.Decode(test[0])) == ae::Bytes(test[1]),
                 "deterministic exact encode-decode");
  auto zero     = deployed.Decode(nn::zeros({200, 32}));
  auto shuffled = deployed.Decode(nn::concat({nn::slice_axis(test[0], 0, 1, 200), nn::slice_axis(test[0], 0, 0, 1)}, 0));
  arrays.emplace_back("zero_reconstruction", zero);
  arrays.emplace_back("shuffled_reconstruction", shuffled);
  // Rows in reverse order (y -> 27 - y): the ordered input is disrupted, output positions unchanged.
  std::vector<std::int64_t> reversed_rows(28);
  for (int y = 0; y < 28; ++y) reversed_rows[y] = 27 - y;
  auto reversed    = nn::gather(ae::RecurrentAutoencoderModel::Sequence(pools[2].images), nn::from_vector(reversed_rows, {28}), 1)
                      .contiguous();
  auto reverse_out = deployed.Evaluate(reversed);
  arrays.emplace_back("reversed_code", reverse_out[0]);
  arrays.emplace_back("reversed_reconstruction", reverse_out[1]);
  auto         one   = deployed.Evaluate(nn::take_rows(pools[2].images, {0}));
  const double error = ae::MaxAbsDiff(one[1], test[1], 784);
  check::Require(error < 1e-6, "single/batch recurrent inference parity");
  check::Require(before == ae::Snapshot(deployed.parameters), "all deployed parameters immutable through code controls");
  nn::save_npz((directory / "outputs.npz").string(), arrays);
  nn::save_npz((directory / "model_arrays.npz").string(), ae::Arrays(deployed.parameters));
  const double  test_mse = ae::Mse(test[1], pools[2].images);
  std::ofstream meta(directory / "metrics.json");
  meta.precision(17);
  meta << "{\"parameters\":" << model.parameters.count() << ",\"epochs\":" << epochs << ",\"updates\":" << step
       << ",\"selected_epoch\":" << selected << ",\"test_mse\":" << test_mse
       << ",\"zero_mse\":" << ae::Mse(zero, pools[2].images) << ",\"shuffled_mse\":" << ae::Mse(shuffled, pools[2].images)
       << ",\"single_batch_max_error\":" << error << ",\"training_seconds\":" << seconds << "}\n";
  check::Require(bool(meta) && bool(history) && bool(batches), "write complete recurrent evidence");
  std::cout << "recurrent:selected" << selected << ",parameters=" << model.parameters.count() << ",testMSE=" << test_mse
            << ",freshrestore/decode/repeat/immutable parameters passed\n";
  return seconds;
}

void Run(const std::filesystem::path& data, const std::filesystem::path& root, int epochs, const std::string& init) {
  auto pools = ae::LoadSaliency(data);
  std::filesystem::create_directories(root);
  const double seconds = Train(pools, epochs, root, init);
  ae::SaveData(root / "data.npz", pools);
  ae::CopyManifests(data, root);
  const auto t    = nn::load_npz((root / "outputs.npz").string());
  const auto mean = ae::Constant(ae::PixelMean(pools[0].images), 200);
  const auto reconstruction = ae::Find(t, "test_reconstruction");
  std::ofstream html(root / "index.html");
  html << "<!doctype html><meta charset='utf-8'><title>Recurrent sequence autoencoder</title><style>body{font:16px "
          "system-ui;max-width:1000px;margin:30px auto}img{max-width:100%;image-rendering:pixelated}section{margin:"
          "25px 0}</style><h1>Sequence-to-sequence recurrent autoencoder</h1><p>Original,reconstruction,fit-only mean,"
          "absolute error,zero-code,shuffled-code,reversed encoder input. Fixed[0,1] grayscale,nearest4,no error "
          "amplification. First test record per digit,unfiltered.</p><p>MNIST is a sequence of28rows of28pixels. Tanh "
          "encoder32,final-state-only code32,repeated into tanh decoder32 with zero initial state,shared sigmoid row "
          "projection. Distinct encoder/decoder weights,shared across time;4956parameters. No teacher forcing,"
          "original-input bypass,attention or target rows in decoder. Full backpropagation through all28encoder+"
          "28decoder steps. This is fixed-length row-sequence reconstruction,not language translation or "
          "variable-length prediction.</p><p>"
       << epochs
       << " epochs,Adam.003,batch32,1000fit/200validation/200test. Validation selects the completed epoch. Labels only "
          "balance/captions. Reversing encoder rows deliberately disrupts the ordered input while output positions stay "
          "unchanged;not an equivalent reconstruction task. Continuous code is not a compressed bitstream. Short "
          "training may produce generic blurry rows or lose digit identity.</p><p>TestMSE="
       << ae::Mse(reconstruction, pools[2].images) << ",fitmean=" << ae::Mse(mean, pools[2].images) << ".</p>";
  ae::DigitGallery(root, pools[2],
                   {pools[2].images, reconstruction, mean,
                    nn::abs(nn::reshape(reconstruction, {200, 784}) - ae::Flatten(pools[2].images)),
                    ae::Find(t, "zero_reconstruction"), ae::Find(t, "shuffled_reconstruction"),
                    ae::Find(t, "reversed_reconstruction")},
                   html);
  check::Require(bool(html), "write recurrent gallery");
  std::cout << "Full recurrent sequence reconstruction/fresh restore/latent controls verified.\ntraining time " << seconds
            << " s\nArtifacts:" << root.string() << "/index.html\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int                   epochs = 3;
    std::string           init;
    std::filesystem::path data = paths::Dataset("saliency"),
                          out  = paths::OutputRoot() / "models/recurrent_autoencoder";
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
