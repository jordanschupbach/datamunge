// CMD: just run-nn denoising_autoencoder
// Actual corrupted-input/CLEAN-target autoencoder with fresh epoch noise and matched ordinary AE.
// Three epochs/default; fixed validation/test Gaussian banks, clean/low/strong/secondseed controls.
//
// Port of tflow's denoising_autoencoder. Preparation (in ~/projects/tflow_examples):
// just autoencoder-assets; data are read from $DATAMUNGE_DATA_DIR/saliency. Outputs
// (denoising/ and ordinary/ training records, data.npz with every noise bank, the
// per-epoch training corruptions, test_metrics.tsv, outputs.npz, index.html and
// digit<k>.png galleries) go to output/nn/models/denoising_autoencoder. The Gaussian
// banks (MT19937/Box-Muller) and therefore every corrupted image are bit-identical to
// tflow's. Checkpoints are best.npz files instead of TF SaveV2 bundles. Options:
// epochs=N data=PATH out=PATH init=PATH (start both models from saved parameters, e.g.
// tflow's denoising/initial_arrays.npz, to follow its trajectories exactly; by default
// nn's own initialiser is used, identically for both models).

#include <array>
#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>

#include "common/denoising_autoencoder.hpp"

namespace {

namespace nn = datamunge::nn;

constexpr float sigma = .35f;

void Checks() {
  auto clean = nn::from_vector(std::vector<float>{0, .2f, .8f, 1}, {2, 2});
  auto z     = nn::from_vector(std::vector<double>{-2., 1., -1., 2.}, {1, 2, 2});
  auto noisy = ae::CorruptForDenoising(clean, z, .3f);
  const float expected[] = {0, .5f, .5f, 1};
  const auto  nv         = noisy.to_vector<float>();
  for (int i = 0; i < 4; ++i) check::Near(nv[i], expected[i], 6e-8, "known additiveGaussian/float32/clamp actual pixels");
  check::Require(ae::Bytes(ae::CorruptForDenoising(clean, z, 0)) == ae::Bytes(clean), "zero sigma exact identity");
  auto a = ae::GaussianNoise(clean.shape(), 1, 1673), b = ae::GaussianNoise(clean.shape(), 1, 1673),
       c = ae::GaussianNoise(clean.shape(), 1, 1674);
  check::Require(a.to_vector<double>() == b.to_vector<double>() && a.to_vector<double>() != c.to_vector<double>(),
                 "Gaussian seed repeat/secondseed distinction");
  check::Throws([&] { ae::CorruptForDenoising(clean, z, -1); }, "negative corruption sigma rejected");
  auto prediction = noisy.clone().requires_grad_();
  auto loss       = ae::ReconstructionMse(prediction, clean);
  auto gradient   = nn::grad(loss, {prediction})[0].to_vector<float>();
  check::Near(loss.item<float>(), .045, 2e-8, "nonzero error to CLEAN target despite prediction exactly noisy input");
  const auto cv = clean.to_vector<float>();
  for (int i = 0; i < 4; ++i)
    check::Near(gradient[i], (double(nv[i]) - cv[i]) / 2, 2e-8, "all4clean-target MSE derivatives");
  std::cout << "corruption/clamp/zero/seed/invalidsigma and CLEAN-vs-noisy target/MSE/4derivative fixtures passed\n";
}

double Train(const std::string& name, const std::vector<ae::SaliencyData>& pools, const nn::Tensor& validation_noisy,
             int epochs, const std::filesystem::path& root, const std::string& init) {
  const auto directory = root / name;
  std::filesystem::create_directories(directory);
  ae::DenoisingModel model;
  if (!init.empty()) model.parameters.load(init);
  check::Require(model.parameters.count() == 50992 && model.parameters.size() == 4, "784->32ReLU->784sigmoid inventory");
  nn::save_npz((directory / "initial_arrays.npz").string(), ae::Arrays(model.parameters));
  nn::BatchOrder order(1000, 32, 1663);
  std::ofstream  history(directory / "history.tsv"), batches(directory / "batches.tsv");
  history.precision(17);
  history << "epoch\tfit_preupdate_clean_target_mse\tvalidation_denoising_mse\n";
  batches << "epoch\tstep\trow\tsource_id\n";
  const auto fit_source = pools[0].source.to_vector<int>();
  const auto best_path  = (directory / "best.npz").string();
  double     best       = std::numeric_limits<double>::infinity();
  int        selected = 0, step = 0;
  const auto started = std::chrono::steady_clock::now();
  for (int epoch = 1; epoch <= epochs; ++epoch) {
    nn::Tensor corruption;
    if (name == "denoising") {
      auto bank  = ae::GaussianNoise(pools[0].images.shape(), 1, std::uint32_t(1671 + epoch));
      corruption = ae::CorruptForDenoising(pools[0].images, bank, sigma);
      nn::save_npz((root / ("train_corruption_epoch" + std::to_string(epoch) + ".npz")).string(),
                   {{"standard_noise", bank}, {"corrupted", corruption}});
    }
    order.begin_epoch();
    double sum  = 0;
    int    seen = 0;
    for (;;) {
      auto rows = order.next();
      if (rows.empty()) break;
      auto clean = nn::take_rows(pools[0].images, rows);
      auto input = name == "ordinary" ? clean : nn::take_rows(corruption, rows);
      // Forward the (corrupted or clean) input, score it against the CLEAN target.
      auto net  = ae::UndercompleteAutoencoder(model.parameters, ae::Flatten(input));
      auto loss = ae::ReconstructionMse(net.reconstruction, ae::Flatten(clean));
      if (step == 0) {
        nn::GradOptions keep;
        keep.retain_graph = true;
        auto gradients    = ae::CheckGradients(model.parameters, nn::grad(loss, model.parameters.parameters(), keep),
                                               "every actual clean-target parameter derivative finite",
                                               "all4parameters receive CLEAN-target loss gradients");
        nn::save_npz((directory / "first_gradients.npz").string(), gradients);
        nn::save_npz((directory / "first_batch.npz").string(), {{"input", input}, {"clean_target", clean}});
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
    check::Require(seen == 1000, "all1000fitting clean images once per epoch");
    const double value = model.Loss(validation_noisy, pools[1].images);
    history << epoch << '\t' << sum / seen << '\t' << value << '\n';
    nn::save_npz((directory / ("epoch" + std::to_string(epoch) + "_arrays.npz")).string(),
                 ae::Arrays(model.parameters));
    if (value < best) {
      best     = value;
      selected = epoch;
      model.parameters.save(best_path);  // both variants select on SAME fixed noisy validation-to-clean MSE
    }
    std::cout << name << " epoch" << epoch << ":fixed-noisy validation CLEAN-target MSE=" << value << std::endl;
  }
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  nn::save_npz((directory / "last_arrays.npz").string(), ae::Arrays(model.parameters));
  model.parameters.load(best_path);  // restore selected denoising-task checkpoint
  nn::save_npz((directory / "model_arrays.npz").string(), ae::Arrays(model.parameters));
  std::ofstream meta(directory / "metrics.json");
  meta.precision(17);
  meta << "{\"epochs\":" << epochs << ",\"updates\":" << step << ",\"selected_epoch\":" << selected
       << ",\"validation_clean_target_mse\":" << best
       << ",\"training_corrupted_inputs\":" << (name == "denoising" ? "true" : "false")
       << ",\"training_seconds\":" << seconds << "}\n";
  check::Require(bool(meta) && bool(history) && bool(batches), "write all training evidence");
  return seconds;
}

void Run(const std::filesystem::path& data, const std::filesystem::path& root, int epochs, const std::string& init) {
  auto pools = ae::LoadSaliency(data);
  std::filesystem::create_directories(root);
  auto validation_bank = ae::GaussianNoise(pools[1].images.shape(), 1, 1709),
       test_bank       = ae::GaussianNoise(pools[2].images.shape(), 1, 1711),
       second_bank     = ae::GaussianNoise(pools[2].images.shape(), 1, 1717);
  auto validation_noisy = ae::CorruptForDenoising(pools[1].images, validation_bank, sigma),
       test_noisy       = ae::CorruptForDenoising(pools[2].images, test_bank, sigma),
       test_low         = ae::CorruptForDenoising(pools[2].images, test_bank, .15f),
       test_strong      = ae::CorruptForDenoising(pools[2].images, test_bank, .6f),
       test_second      = ae::CorruptForDenoising(pools[2].images, second_bank, sigma);
  ae::SaveData(root / "data.npz", pools,
               {{"validation_standard_noise", validation_bank},
                {"test_standard_noise", test_bank},
                {"second_standard_noise", second_bank},
                {"validation_corrupted", validation_noisy},
                {"test_corrupted", test_noisy},
                {"test_low", test_low},
                {"test_strong", test_strong},
                {"test_second", test_second}});
  double seconds = Train("denoising", pools, validation_noisy, epochs, root, init);
  seconds += Train("ordinary", pools, validation_noisy, epochs, root, init);
  const auto initial_d = nn::load_npz((root / "denoising" / "initial_arrays.npz").string()),
             initial_o = nn::load_npz((root / "ordinary" / "initial_arrays.npz").string());
  for (const auto& [key, value] : initial_d)
    check::Require(ae::Bytes(ae::Find(initial_o, key)) == ae::Bytes(value),
                   "matched ordinary and denoising initial parameters");
  ae::DenoisingModel deployed(false), ordinary(false);
  deployed.parameters.load((root / "denoising" / "best.npz").string());  // fresh denoising inference restore
  ordinary.parameters.load((root / "ordinary" / "best.npz").string());   // fresh ordinary control restore
  const auto before = ae::Snapshot(deployed.parameters), ordinary_before = ae::Snapshot(ordinary.parameters);
  nn::NamedTensors                              outputs;
  const std::map<std::string, nn::Tensor>       cases{{"noisy", test_noisy},
                                                      {"clean", pools[2].images},
                                                      {"low", test_low},
                                                      {"strong", test_strong},
                                                      {"second", test_second}};
  std::map<std::string, std::array<double, 3>> scores;
  std::ofstream                                 metrics(root / "test_metrics.tsv");
  metrics.precision(17);
  metrics << "case\tcorrupted_to_clean_mse\tdenoising_to_clean_mse\tordinary_to_clean_mse\n";
  for (const auto& [name, input] : cases) {
    auto d = deployed.Predict(input), o = ordinary.Predict(input);
    outputs.emplace_back(name + "_denoising_code", d[0]);
    outputs.emplace_back(name + "_denoising_reconstruction", d[1]);
    outputs.emplace_back(name + "_ordinary_reconstruction", o[1]);
    const double raw = ae::Mse(input, pools[2].images), denoised = ae::Mse(d[1], pools[2].images),
                 control = ae::Mse(o[1], pools[2].images);
    scores.emplace(name, std::array<double, 3>{raw, denoised, control});
    metrics << name << '\t' << raw << '\t' << denoised << '\t' << control << '\n';
    std::cout << name << ":raw/denoised/ordinary CLEAN-target MSE=" << raw << '/' << denoised << '/' << control << '\n';
  }
  auto result = deployed.Predict(test_noisy), repeat = deployed.Predict(test_noisy);
  check::Require(ae::Bytes(result[1]) == ae::Bytes(repeat[1]), "deterministic target-free denoising inference");
  check::Require(ae::Bytes(deployed.Decode(result[0])) == ae::Bytes(result[1]), "explicit encode-decode exact");
  auto         one   = deployed.Predict(nn::take_rows(test_noisy, {0}));
  const double error = ae::MaxAbsDiff(one[1], result[1], 784);
  check::Require(error < 1e-6, "single/batch denoising parity");
  check::Require(before == ae::Snapshot(deployed.parameters) && ordinary_before == ae::Snapshot(ordinary.parameters),
                 "all2x4model parameters immutable through test controls");
  nn::save_npz((root / "outputs.npz").string(), outputs);
  ae::CopyManifests(data, root);
  const auto   mean     = ae::Constant(ae::PixelMean(pools[0].images), 200);
  const double baseline = ae::Mse(mean, pools[2].images);
  std::ofstream html(root / "index.html");
  html << "<!doctype html><meta charset='utf-8'><title>Clean-target denoising autoencoder</title><style>body{font:"
          "16px system-ui;max-width:1100px;margin:30px auto}img{max-width:100%;image-rendering:pixelated}section{"
          "margin:25px 0}</style><h1>Denoising:corrupted inputs,CLEAN targets</h1><p>Clean original,corrupted "
          "sigma.35,denoised,ordinary AE on SAME corruption,absolute denoising error,fit-only mean,denoiser on clean "
          "input,denoiser on sigma.6. Fixed[0,1]grayscale/nearest4,no error amplification or quality filtering.</p>"
          "<p>784->32ReLU->784sigmoid/50992params,Adam.003,batch32,"
       << epochs
       << "epochs each,matched initial parameters and order1663. Denoiser receives NEW epoch-specific Gaussian "
          "draws,ordinary control trains clean->clean. BOTH select on SAME fixed noisy-validation-to-clean MSE. No "
          "labels enter training or selection;NO clean target is required for inference.</p><p>Noise MT19937/"
          "Box-Muller standardGaussian,sigma.35 in original unit-pixel coordinates,castfloat32 THEN clamp[0,1]. "
          "Clipping changes the corruption distribution and is NOT inverted. Epoch seeds1672onward;validation1709/"
          "test1711/second1717 independent. Low.15 and strong.6 share test standard draws for paired amplitude "
          "controls. Clean-input test has sigma0. Raw corruptions,allnoise banks and clean targets saved.</p><p>"
          "Fit-only mean MSE="
       << baseline
       << ". All raw/denoised/ordinary CLEAN-target MSE values,including LOW-noise failure cases,in "
          "test_metrics.tsv. A short undercomplete model may blur digit identity or be WORSE than leaving mild noise "
          "unchanged. One chosen corruption family and tiny data budget do not demonstrate universal denoising,"
          "classification,generative quality or bit-rate compression.</p>";
  html << "<table><tr><th>Case</th><th>Leave input unchanged</th><th>Noise-trained AE</th><th>Ordinary AE</th></tr>";
  for (const auto& [name, value] : scores)
    html << "<tr><td>" << name << "</td><td>" << value[0] << "</td><td>" << value[1] << "</td><td>" << value[2]
         << "</td></tr>";
  html << "</table><p>All values compare against CLEAN targets. Noise-trained versus ordinary on main noise: "
       << (scores.at("noisy")[1] < scores.at("noisy")[2] ? "noise-trained lower MSE" : "ordinary lower MSE in this short run")
       << ". This baseline comparison remains visible regardless of result.</p>";
  const auto denoised = ae::Find(outputs, "noisy_denoising_reconstruction");
  ae::DigitGallery(root, pools[2],
                   {pools[2].images, test_noisy, denoised, ae::Find(outputs, "noisy_ordinary_reconstruction"),
                    nn::abs(denoised - ae::Flatten(pools[2].images)), mean,
                    ae::Find(outputs, "clean_denoising_reconstruction"),
                    ae::Find(outputs, "strong_denoising_reconstruction")},
                   html);
  check::Require(bool(html) && bool(metrics), "write actual denoising metrics and gallery");
  std::cout << "Freshrestore/target-freeinference/encode-decode/repeat/single-batch/immutable2x4weights passed,"
               "maxbatcherror="
            << error << "\nFit-onlymean CLEAN-target MSE=" << baseline
            << ";weak/mildnoise failures retained.\ntraining time " << seconds << " s\nArtifacts:" << root.string()
            << "/index.html\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int                   epochs = 3;
    std::string           init;
    std::filesystem::path data = paths::Dataset("saliency"),
                          out  = paths::OutputRoot() / "models/denoising_autoencoder";
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
