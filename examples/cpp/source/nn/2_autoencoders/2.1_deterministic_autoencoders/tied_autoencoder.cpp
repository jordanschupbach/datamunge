// CMD: just run-nn tied_autoencoder
// ONE live 784x32 parameter for encoder AND transposed decoder; two independent biases.
// Three default epochs, matched-initial untied control, and explicit gradient-path checks.
//
// Port of tflow's tied_autoencoder (model: examples/nn/common/tied_autoencoder.hpp).
// Preparation (in ~/projects/tflow_examples): just autoencoder-assets; data are read from
// $DATAMUNGE_DATA_DIR/saliency. Outputs (tied/ and untied/ training records, initial_forward.npz,
// data.npz, index.html and digit<k>.png galleries) go to output/nn/models/tied_autoencoder.
// Checkpoints are best.npz files instead of TF SaveV2 bundles; metrics.json gains
// "training_seconds". In eager mode the tie is the decoder weight being recomputed as
// transpose(encoder/W) on every forward pass (no decoder parameter exists); it is checked
// coordinate-by-coordinate after every update as in tflow. Options: epochs=N data=PATH out=PATH
// init=PATH (tied initial parameters, e.g. tflow's tied/initial_arrays.npz, to follow its
// trajectories exactly; the untied control always starts from their exact transpose).

#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>

#include "common/tied_autoencoder.hpp"

namespace {

namespace nn = datamunge::nn;

void Checks() {
  nn::ParameterStore p(1861);
  {
    nn::NoGradGuard guard;
    ae::TiedAutoencoder(p, nn::zeros({1, 3}), 3, 2, true);
  }
  const std::vector<float> w{.2f, .1f, -.3f, .4f, .5f, -.2f}, eb{.1f, .2f}, db{-.1f, .05f, .2f},
      in{.1f, .2f, .7f, .8f, .3f, .4f};
  p.set("encoder/W", nn::from_vector(w, {3, 2}));
  p.set("encoder/b", nn::from_vector(eb, {2}));
  p.set("decoder/b", nn::from_vector(db, {3}));
  auto       input    = nn::from_vector(in, {2, 3});
  auto       net      = ae::TiedAutoencoder(p, input, 3, 2, true);
  auto       loss     = ae::ReconstructionMse(net.reconstruction, input);
  auto       gradient = nn::grad(loss, p.parameters());
  const auto code = net.code.to_vector<float>(), recon = net.reconstruction.to_vector<float>();
  const auto gw = gradient[0].to_vector<float>(), geb = gradient[1].to_vector<float>(),
             gdb = gradient[2].to_vector<float>();
  double     encoder[3][2] = {}, decoder[3][2] = {}, ge[2] = {}, gd[3] = {}, mse = 0;
  for (int b = 0; b < 2; ++b) {
    double c[2] = {eb[0], eb[1]}, delta[3];
    for (int j = 0; j < 2; ++j) {
      for (int i = 0; i < 3; ++i) c[j] += in[b * 3 + i] * double(w[i * 2 + j]);
      check::Require(c[j] > 0, "toy avoids ReLU kink");
      check::Near(code[b * 2 + j], c[j], 5e-8, "all4toy codes");
    }
    for (int i = 0; i < 3; ++i) {
      double value = db[i];
      for (int j = 0; j < 2; ++j) value += c[j] * w[i * 2 + j];
      const double y = 1 / (1 + std::exp(-value)), error = y - in[b * 3 + i];
      check::Near(recon[b * 3 + i], y, 8e-8, "all6toy sigmoid predictions");
      mse += error * error / 6;
      delta[i] = error / 3 * y * (1 - y);
      gd[i] += delta[i];
      for (int j = 0; j < 2; ++j) decoder[i][j] += delta[i] * c[j];
    }
    for (int j = 0; j < 2; ++j) {
      double dz = 0;
      for (int i = 0; i < 3; ++i) dz += delta[i] * w[i * 2 + j];
      ge[j] += dz;
      for (int i = 0; i < 3; ++i) encoder[i][j] += in[b * 3 + i] * dz;
    }
  }
  check::Require(p.size() == 3 && p.count() == 11, "one sharedmatrix plus two biases,not a copied decoder parameter");
  check::Near(loss.item<float>(), mse, 2e-8, "known tied MSE");
  double encoder_norm = 0, decoder_norm = 0;
  for (int i = 0; i < 3; ++i) {
    check::Near(gdb[i], gd[i], 3e-8, "all3decoder bias derivatives");
    for (int j = 0; j < 2; ++j) {
      check::Near(gw[i * 2 + j], encoder[i][j] + decoder[i][j], 3e-8,
                  "ALL6shared derivatives equal encoder-path PLUS decoder-path");
      encoder_norm += std::abs(encoder[i][j]);
      decoder_norm += std::abs(decoder[i][j]);
    }
  }
  for (int j = 0; j < 2; ++j) check::Near(geb[j], ge[j], 3e-8, "all2encoder bias derivatives");
  check::Require(encoder_norm > 1e-3 && decoder_norm > 1e-3, "neither shared gradient contribution is silently missing");
  std::cout << "known tied toy4codes/6predictions/MSE/ALL11derivatives/BOTHsharedmatrix contributions passed\n";
}

nn::Tensor Shift(const nn::Tensor& codes) {
  std::vector<std::int64_t> rows;
  for (std::int64_t b = 0; b < codes.size(0); ++b) rows.push_back((b + 1) % codes.size(0));
  return nn::take_rows(codes, rows).contiguous();
}

double Train(bool tied, const std::vector<ae::SaliencyData>& pools, int epochs, const std::filesystem::path& root,
             const nn::NamedTensors& initial) {
  const std::string name      = tied ? "tied" : "untied";
  const auto        directory = root / name;
  std::filesystem::create_directories(directory);
  ae::TiedAutoencoderModel model(tied);
  model.Load(initial);
  check::Require(model.parameters.size() == (tied ? 3u : 4u) && model.parameters.count() == (tied ? 25904 : 50992),
                 "independent tied/untied parameter census");
  if (tied) model.CheckTie();
  nn::save_npz((directory / "initial_arrays.npz").string(), ae::Arrays(model.parameters));
  nn::BatchOrder order(1000, 32, 1867);
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
      // Forward (live transposed decoder when tied), reconstruction MSE, gradients through BOTH paths.
      auto net  = model.Net(input);
      auto loss = ae::ReconstructionMse(net.reconstruction, ae::Flatten(input));
      if (step == 0) {
        nn::GradOptions keep;
        keep.retain_graph = true;
        nn::save_npz((directory / "first_gradients.npz").string(),
                     ae::CheckGradients(model.parameters, nn::grad(loss, model.parameters.parameters(), keep),
                                        "all model MSE derivatives finite",
                                        "all model tensors receive full MSE gradients"));
        nn::save_npz((directory / "first_batch.npz").string(), {{"images", input}});
      }
      const double value = loss.item<float>();
      check::Require(std::isfinite(value) && value >= 0, "finite pre-update MSE");
      model.optimizer->minimize(loss);
      sum += double(rows.size()) * value;
      seen += int(rows.size());
      for (auto row : rows) batches << epoch << '\t' << step << '\t' << row << '\t' << fit_source[row] << '\n';
      if (tied) model.CheckTie();
      if (step == 0) nn::save_npz((directory / "first_update_arrays.npz").string(), ae::Arrays(model.parameters));
      ++step;
    }
    check::Require(seen == 1000, "all fitting records exactly once per epoch");
    const double mse = ae::Mse(model.Evaluate(pools[1].images)[1], pools[1].images);
    history << epoch << '\t' << sum / seen << '\t' << mse << '\n';
    nn::save_npz((directory / ("epoch" + std::to_string(epoch) + "_arrays.npz")).string(),
                 ae::Arrays(model.parameters));
    if (mse < best) {
      best     = mse;
      selected = epoch;
      model.parameters.save(best_path);  // common validation reconstruction MSE selects each model
    }
    std::cout << name << " epoch" << epoch << ":validationMSE=" << mse << std::endl;
  }
  const double             seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  ae::TiedAutoencoderModel deployed(tied, false);
  deployed.parameters.load(best_path);  // fresh tied/untied inference restore
  if (tied) deployed.CheckTie();
  const auto       before = ae::Snapshot(deployed.parameters);
  nn::NamedTensors arrays;
  for (int split = 0; split < 3; ++split) {
    const std::string s   = split == 0 ? "fit" : split == 1 ? "validation" : "test";
    auto              out = deployed.Evaluate(pools[split].images);
    arrays.emplace_back(s + "_code", out[0]);
    arrays.emplace_back(s + "_reconstruction", out[1]);
    if (split == 1) check::Near(ae::Mse(out[1], pools[split].images), best, 1e-8, "fresh selected validation MSE");
  }
  auto test = deployed.Evaluate(pools[2].images), repeat = deployed.Evaluate(pools[2].images);
  check::Require(ae::Bytes(test[1]) == ae::Bytes(repeat[1]) && ae::Bytes(deployed.Decode(test[0])) == ae::Bytes(test[1]),
                 "deterministic exact encode-decode");
  arrays.emplace_back("zero_reconstruction", deployed.Decode(nn::zeros({200, 32})));
  arrays.emplace_back("shuffled_reconstruction", deployed.Decode(Shift(test[0])));
  auto         one   = deployed.Evaluate(ae::Rows(pools[2], {0}).images);
  const double error = ae::MaxAbsDiff(one[1], test[1], 784);
  check::Require(error < 1e-6, "single/batch tied or untied inference parity");
  check::Require(before == ae::Snapshot(deployed.parameters), "all deployed parameters immutable through code controls");
  nn::save_npz((directory / "outputs.npz").string(), arrays);
  nn::save_npz((directory / "model_arrays.npz").string(), ae::Arrays(deployed.parameters));
  nn::save_npz((directory / "decoder_weight.npz").string(), {{"decoder_weight", deployed.DecoderWeight()}});
  const double  test_mse = ae::Mse(test[1], pools[2].images);
  std::ofstream meta(directory / "metrics.json");
  meta.precision(17);
  meta << "{\"tied\":" << (tied ? "true" : "false") << ",\"parameters\":" << model.parameters.count()
       << ",\"epochs\":" << epochs << ",\"updates\":" << step << ",\"selected_epoch\":" << selected
       << ",\"test_mse\":" << test_mse << ",\"zero_mse\":" << ae::Mse(ae::Find(arrays, "zero_reconstruction"), pools[2].images)
       << ",\"shuffled_mse\":" << ae::Mse(ae::Find(arrays, "shuffled_reconstruction"), pools[2].images)
       << ",\"single_batch_max_error\":" << error << ",\"training_seconds\":" << seconds << "}\n";
  check::Require(bool(meta) && bool(history) && bool(batches), "write complete tied/untied evidence");
  std::cout << name << ":selected" << selected << ",parameters=" << model.parameters.count() << ",testMSE=" << test_mse
            << ",freshrestore/decode/repeat/immutable parameters passed\n";
  return seconds;
}

void Run(const std::filesystem::path& data, const std::filesystem::path& root, int epochs, const std::string& init) {
  auto pools = ae::LoadSaliency(data);
  std::filesystem::create_directories(root);
  ae::TiedAutoencoderModel factory(true, false);
  if (!init.empty()) factory.parameters.load(init);
  const auto tied_initial   = ae::Arrays(factory.parameters);
  auto       untied_initial = tied_initial;
  const auto transposed     = nn::transpose(ae::Find(tied_initial, "encoder/W"), {1, 0}).contiguous();
  untied_initial.emplace_back("decoder/W", transposed);
  ae::TiedAutoencoderModel initial_control(false, false);
  initial_control.Load(untied_initial);
  auto initial_probe = ae::Rows(pools[0], {0, 1, 2, 3, 4, 5, 6, 7}).images;
  auto initial_t = factory.Evaluate(initial_probe), initial_u = initial_control.Evaluate(initial_probe);
  check::Require(ae::Bytes(initial_t[0]) == ae::Bytes(initial_u[0]) && ae::Bytes(initial_t[1]) == ae::Bytes(initial_u[1]),
                 "native matched initial codes AND reconstructions are BITEXACT");
  nn::save_npz((root / "initial_forward.npz").string(), {{"images", initial_probe},
                                                         {"tied_code", initial_t[0]},
                                                         {"untied_code", initial_u[0]},
                                                         {"tied_reconstruction", initial_t[1]},
                                                         {"untied_reconstruction", initial_u[1]}});
  double seconds = Train(true, pools, epochs, root, tied_initial);
  seconds += Train(false, pools, epochs, root, untied_initial);
  const auto t_initial = nn::load_npz((root / "tied" / "initial_arrays.npz").string()),
             u_initial = nn::load_npz((root / "untied" / "initial_arrays.npz").string());
  for (const auto& [key, value] : t_initial)
    check::Require(ae::Bytes(ae::Find(u_initial, key)) == ae::Bytes(value), "matched initial encoder/bias bytes");
  check::Require(ae::Bytes(ae::Find(u_initial, "decoder/W")) == ae::Bytes(transposed),
                 "untied starts EXACTLY same numerical transpose,then independently trains");
  ae::SaveData(root / "data.npz", pools);
  ae::CopyManifests(data, root);
  const auto t = nn::load_npz((root / "tied" / "outputs.npz").string()),
             u = nn::load_npz((root / "untied" / "outputs.npz").string());
  const auto mean     = ae::Constant(ae::PixelMean(pools[0].images), 200);
  const auto t_recon  = ae::Find(t, "test_reconstruction");
  const double tied_mse = ae::Mse(t_recon, pools[2].images),
               untied_mse = ae::Mse(ae::Find(u, "test_reconstruction"), pools[2].images),
               mean_mse   = ae::Mse(mean, pools[2].images);
  std::ofstream html(root / "index.html");
  html << "<!doctype html><meta charset='utf-8'><title>Actual tied-weight autoencoder</title><style>body{font:16px "
          "system-ui;max-width:1000px;margin:30px auto}img{max-width:100%;image-rendering:pixelated}section{margin:"
          "25px 0}</style><h1>Tied weights:ONE parameter,two gradient paths</h1><p>Original,tied,untied,fit-only mean,"
          "absolute tied error,tied zero-code,tied shuffled-code. Fixed[0,1]grayscale/nearest4,no error amplification "
          "or quality filtering.</p><p>784->32ReLU->784sigmoid. Tied decoder is the LIVE transpose of the SAME encoder "
          "parameter,never a copied decoder parameter. Separate encoder/decoder biases;25904parameters versus50992"
          "untied. After EVERY tied update all25088decoder coordinates are checked EXACTLY against current encoder "
          "transpose. Shared MSE gradient sums encoder and decoder contributions.</p><p>Both models start numerically "
          "IDENTICAL,including untied decoder initialized to encoder transpose. Same source/order/Adam.003/"
       << epochs
       << "epochs;each selects on common validation MSE. Parameter counts differ,so this is a shared-constraint "
          "comparison,not equal capacity or statistical proof of superiority. Continuous32float code is not bit-rate "
          "compression. Blurry/distorted shapes can persist.</p><p>TestMSE tied="
       << tied_mse << ",untied=" << untied_mse << ",fitmean=" << mean_mse
       << ". Zero/shuffle controls expose input-specific encoding;no classification/denoising/generative claim.</p>";
  ae::DigitGallery(root, pools[2],
                   {pools[2].images, t_recon, ae::Find(u, "test_reconstruction"), mean,
                    nn::abs(t_recon - ae::Flatten(pools[2].images)), ae::Find(t, "zero_reconstruction"),
                    ae::Find(t, "shuffled_reconstruction")},
                   html);
  check::Require(bool(html), "write tied weight gallery");
  std::cout << "Actualsharedparameter/transpose afterEVERYupdate/BOTHgradientpaths/numericallymatched initial "
               "untiedcontrol verified.\ntraining time "
            << seconds << " s\nArtifacts:" << root.string() << "/index.html\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int                   epochs = 3;
    std::string           init;
    std::filesystem::path data = paths::Dataset("saliency"), out = paths::OutputRoot() / "models/tied_autoencoder";
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
