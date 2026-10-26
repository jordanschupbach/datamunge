// CMD: just run-nn undercomplete_autoencoder
// Dense 784->32 ReLU->784 sigmoid reconstruction, three inexpensive default epochs.
// Fit-only pixel mean/black/initial/zero-code/shuffled-code baselines; no class targets.
//
// Port of tflow's undercomplete_autoencoder (model: ae::UndercompleteAutoencoder in
// examples/nn/common/autoencoder.hpp). Preparation (in ~/projects/tflow_examples): just
// autoencoder-assets; data are read from $DATAMUNGE_DATA_DIR/saliency. Outputs (history.tsv,
// batches.tsv, per-epoch arrays, autoencoder.npz, metrics.json, index.html and digit<k>.png
// galleries) go to output/nn/models/undercomplete_autoencoder. Checkpoints are best.npz files
// instead of TF SaveV2 bundles; metrics.json gains "training_seconds". Options: epochs=N
// data=PATH out=PATH init=PATH (start from saved parameters, e.g. tflow's initial_arrays.npz,
// to follow its trajectory exactly; by default nn's own initialiser is used).

#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>

#include "common/autoencoder.hpp"

namespace {

namespace nn = datamunge::nn;

class UndercompleteModel {
 public:
  nn::ParameterStore             parameters{1471};
  std::unique_ptr<nn::Optimizer> optimizer;

  explicit UndercompleteModel(bool training = true) {
    {
      nn::NoGradGuard guard;
      ae::UndercompleteAutoencoder(parameters, nn::zeros({1, 784}));
    }
    if (training) optimizer = ae::MakeAdam(parameters);
  }
  /// {code, reconstruction, loss}
  std::vector<nn::Tensor> Evaluate(const nn::Tensor& images) {
    nn::NoGradGuard guard;
    auto            x = ae::Flatten(images);
    auto            n = ae::UndercompleteAutoencoder(parameters, x);
    return {n.code, n.reconstruction, ae::ReconstructionMse(n.reconstruction, x)};
  }
  nn::Tensor Decode(const nn::Tensor& codes) {
    nn::NoGradGuard guard;
    return nn::dense(parameters, "decoder", codes, 784, nn::Activation::Sigmoid);
  }
  nn::Tensor Loss(const nn::Tensor& images) {
    auto x = ae::Flatten(images);
    return ae::ReconstructionMse(ae::UndercompleteAutoencoder(parameters, x).reconstruction, x);
  }
  std::vector<nn::Tensor> Gradient(const nn::Tensor& images) { return nn::grad(Loss(images), parameters.parameters()); }
  double                  Update(const nn::Tensor& images) {
    auto         loss  = Loss(images);
    const double value = loss.item<float>();
    check::Require(std::isfinite(value) && value >= 0, "finite pre-update MSE");
    optimizer->minimize(loss);
    return value;
  }
};

void Checks() {
  nn::ParameterStore p(1481);
  {
    nn::NoGradGuard guard;
    ae::UndercompleteAutoencoder(p, nn::zeros({1, 2}), 2, 1);
  }
  const std::vector<std::vector<float>> fixed{{.3f, .2f}, {.1f}, {.4f, -.5f}, {-.2f, .1f}};
  for (std::size_t i = 0; i < 4; ++i) p.set(p.names()[i], nn::from_vector(fixed[i], p.at(p.names()[i]).shape()));
  const std::vector<float> in{.1f, .8f, .7f, .2f};
  auto                     input = nn::from_vector(in, {2, 2});
  auto                     net   = ae::UndercompleteAutoencoder(p, input, 2, 1);
  auto                     loss  = ae::ReconstructionMse(net.reconstruction, input);
  auto                     g     = nn::grad(loss, p.parameters());
  const auto               code = net.code.to_vector<float>(), recon = net.reconstruction.to_vector<float>();
  double                   expected[7] = {}, mse = 0;
  for (int b = 0; b < 2; ++b) {
    double c = .1f;
    for (int j = 0; j < 2; ++j) c += in[b * 2 + j] * double(fixed[0][j]);
    check::Near(code[b], c, 3e-8, "all toy bottleneck values");
    double dz = 0;
    for (int j = 0; j < 2; ++j) {
      const double prediction = 1 / (1 + std::exp(-(c * fixed[2][j] + fixed[3][j])));
      check::Near(recon[b * 2 + j], prediction, 5e-8, "all toy sigmoid reconstructions");
      const double error = prediction - in[b * 2 + j];
      mse += error * error / 4;
      const double delta = error / 2 * prediction * (1 - prediction);
      expected[3 + j] += c * delta;
      expected[5 + j] += delta;
      dz += delta * fixed[2][j];
    }
    for (int j = 0; j < 2; ++j) expected[j] += in[b * 2 + j] * dz;
    expected[2] += dz;
  }
  check::Near(loss.item<float>(), mse, 2e-8, "MSE averaged over both examples and both pixels");
  int offset = 0;
  for (int t = 0; t < 4; ++t)
    for (float v : g[t].to_vector<float>())
      check::Near(v, expected[offset++], 2e-8, "ALL7analytic nonlinear reconstruction parameter gradients");
  check::Throws([&] { ae::UndercompleteAutoencoder(p, input, 2, 2); }, "equal-size latent rejected as not undercomplete");
  std::cout << "known nonlinear toy:2codes/4sigmoidpixels/MSE/all7parameter derivatives and strict bottleneck checks "
               "passed\n";
}

/// Rows of `codes` [n, k] cyclically shifted by one record.
nn::Tensor Shift(const nn::Tensor& codes) {
  const auto         n = codes.size(0);
  std::vector<std::int64_t> rows;
  for (std::int64_t b = 0; b < n; ++b) rows.push_back((b + 1) % n);
  return nn::take_rows(codes, rows).contiguous();
}

void Run(const std::filesystem::path& data, const std::filesystem::path& directory, int epochs,
         const std::string& init) {
  auto        pools = ae::LoadSaliency(data);
  const auto& fit = pools[0];
  const auto& validation = pools[1];
  const auto& test = pools[2];
  UndercompleteModel model;
  if (!init.empty()) model.parameters.load(init);
  check::Require(model.parameters.size() == 4 && model.parameters.count() == 50992,
                 "independent untied784x32+32+32x784+784 inventory");
  std::filesystem::create_directories(directory);
  nn::save_npz((directory / "initial_arrays.npz").string(), ae::Arrays(model.parameters));
  auto         initial_fit = model.Evaluate(fit.images), initial_test = model.Evaluate(test.images);
  auto         mean            = ae::PixelMean(fit.images);
  auto         mean_prediction = ae::Constant(mean, 200);
  auto         black           = nn::zeros({200, 784});
  const double initial_mse     = ae::Mse(initial_fit[1], fit.images);

  nn::BatchOrder order(1000, 32, 1483);
  std::ofstream  history(directory / "history.tsv"), batches(directory / "batches.tsv");
  history.precision(17);
  history << "epoch\ttrain_preupdate_mse\tvalidation_mse\n";
  batches << "epoch\tstep\trow\tsource_id\n";
  const auto  best_path = (directory / "best.npz").string();
  double      best      = std::numeric_limits<double>::infinity();
  int         selected = 0, step = 0;
  nn::Tensor  first_images, first_ids;
  const auto  fit_source = fit.source.to_vector<int>();
  const auto  started    = std::chrono::steady_clock::now();
  for (int epoch = 1; epoch <= epochs; ++epoch) {
    order.begin_epoch();
    int    seen = 0;
    double sum  = 0;
    for (;;) {
      auto rows = order.next();
      if (rows.empty()) break;
      auto batch = ae::Rows(fit, rows);
      if (step == 0) {
        first_images = batch.images;
        first_ids    = batch.source;
        nn::save_npz((directory / "first_gradients.npz").string(),
                     ae::CheckGradients(model.parameters, model.Gradient(batch.images),
                                        "every actual parameter derivative finite",
                                        "all four tensors receive reconstruction gradients"));
      }
      sum += double(rows.size()) * model.Update(batch.images);
      seen += int(rows.size());
      for (auto row : rows) batches << epoch << '\t' << step << '\t' << row << '\t' << fit_source[row] << '\n';
      if (step == 0) nn::save_npz((directory / "first_update_arrays.npz").string(), ae::Arrays(model.parameters));
      ++step;
    }
    check::Require(seen == 1000, "every fit-only image appears once per epoch including final short batch");
    const double mse = ae::Mse(model.Evaluate(validation.images)[1], validation.images);
    history << epoch << '\t' << sum / seen << '\t' << mse << '\n';
    nn::save_npz((directory / ("epoch" + std::to_string(epoch) + "_arrays.npz")).string(),
                 ae::Arrays(model.parameters));
    if (mse < best) {
      best     = mse;
      selected = epoch;
      model.parameters.save(best_path);  // validation-only reconstruction MSE selects checkpoint
    }
    std::cout << "epoch" << epoch << ":preupdate fitMSE=" << sum / seen << ",validationMSE=" << mse << std::endl;
  }
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  model.parameters.load(best_path);  // restore validation-selected model-only checkpoint
  UndercompleteModel deployed(false);
  deployed.parameters.load(best_path);  // fresh inference-only deploy restores encoder and decoder
  const auto before     = ae::Snapshot(deployed.parameters);
  auto       fit_result = deployed.Evaluate(fit.images), val_result = deployed.Evaluate(validation.images),
       result = deployed.Evaluate(test.images), repeat = deployed.Evaluate(test.images);
  check::Near(ae::Mse(val_result[1], validation.images), best, 1e-8, "fresh restore reproduces selected validation MSE");
  check::Require(result[0].shape() == nn::Shape({200, 32}) && result[1].shape() == nn::Shape({200, 784}),
                 "actual undercomplete code/reconstruction shapes");
  check::Require(ae::Bytes(result[1]) == ae::Bytes(repeat[1]), "deterministic image-only reconstruction");
  check::Require(ae::Bytes(model.Evaluate(test.images)[1]) == ae::Bytes(result[1]),
                 "fresh restore exact all156800test pixels");
  check::Require(ae::Bytes(deployed.Decode(result[0])) == ae::Bytes(result[1]),
                 "encode then standalone decode reproduces direct reconstruction");
  auto         one         = deployed.Evaluate(ae::Rows(test, {0}).images);
  const double batch_error = ae::MaxAbsDiff(one[1], result[1], 784);
  check::Require(batch_error < 1e-6, "single/batch reconstruction parity");
  auto shifted = Shift(result[0]), zeros = nn::zeros({200, 32});
  auto shuffled = deployed.Decode(shifted), zero_decode = deployed.Decode(zeros);
  const double fit_mse = ae::Mse(fit_result[1], fit.images), test_mse = ae::Mse(result[1], test.images),
               mean_mse = ae::Mse(mean_prediction, test.images), black_mse = ae::Mse(black, test.images),
               shuffled_mse = ae::Mse(shuffled, test.images), zero_mse = ae::Mse(zero_decode, test.images),
               initial_test_mse = ae::Mse(initial_test[1], test.images);
  check::Require(fit_mse < initial_mse * .8, "actual fitting lowers reconstruction MSE substantially");
  check::Require(before == ae::Snapshot(deployed.parameters),
                 "all four deployed weights immutable through inference/code controls");

  nn::save_npz((directory / "autoencoder.npz").string(),
               {{"fit_images", fit.images},
                {"fit_source_ids", fit.source},
                {"validation_images", validation.images},
                {"validation_source_ids", validation.source},
                {"test_images", test.images},
                {"test_labels", test.labels},
                {"test_source_ids", test.source},
                {"fit_code", fit_result[0]},
                {"fit_reconstruction", fit_result[1]},
                {"validation_code", val_result[0]},
                {"validation_reconstruction", val_result[1]},
                {"test_code", result[0]},
                {"test_reconstruction", result[1]},
                {"initial_test_reconstruction", initial_test[1]},
                {"fit_pixelmean", mean},
                {"mean_reconstruction", mean_prediction},
                {"black_reconstruction", black},
                {"shifted_codes", shifted},
                {"shuffled_reconstruction", shuffled},
                {"zero_reconstruction", zero_decode},
                {"first_batch_images", first_images},
                {"first_batch_source_ids", first_ids}});
  nn::save_npz((directory / "model_arrays.npz").string(), ae::Arrays(deployed.parameters));
  ae::CopyManifests(data, directory);

  std::ofstream metrics(directory / "metrics.json");
  metrics.precision(17);
  metrics << "{\"epochs\":" << epochs << ",\"selected_epoch\":" << selected << ",\"updates\":" << step
          << ",\"fit_initial_mse\":" << initial_mse << ",\"fit_mse\":" << fit_mse << ",\"validation_mse\":" << best
          << ",\"test_mse\":" << test_mse << ",\"test_initial_mse\":" << initial_test_mse
          << ",\"test_mean_mse\":" << mean_mse << ",\"test_black_mse\":" << black_mse
          << ",\"test_shuffled_mse\":" << shuffled_mse << ",\"test_zero_code_mse\":" << zero_mse
          << ",\"single_batch_max_error\":" << batch_error << ",\"training_seconds\":" << seconds << "}\n";

  std::ofstream html(directory / "index.html");
  html << "<!doctype html><meta charset='utf-8'><title>Undercomplete MNIST autoencoder</title><style>body{font:16px "
          "system-ui;max-width:1000px;margin:30px auto}img{max-width:100%;image-rendering:pixelated}section{margin:"
          "25px 0}</style><h1>Undercomplete autoencoder:784 to32 to784</h1><p>Left to right:original,trained "
          "reconstruction,absolute error,fitting-only pixel-mean baseline,initial model,zero-code decoder,cyclically "
          "shuffled test code. Fixed[0,1]intensity for every tile,nearest4 enlargement;error is absolute pixel "
          "difference without contrast amplification.</p><p>"
       << epochs
       << "epochs,1000fit/200validation/200test original MNIST,untied dense32ReLU code/sigmoid decoder,50992parameters,"
          "Adam.003,batch32. Targets are original inputs;no labels fed to fitting or checkpoint selection. Validation "
          "MSE alone selects epoch"
       << selected
       << ". All32 code coordinates are float32;24.5x fewer coordinates is NOT a claim of24.5x compressed file "
          "size.</p><p>Test per-pixel MSE:trained="
       << test_mse << ",fit-only mean=" << mean_mse << ",black=" << black_mse << ",initial=" << initial_test_mse
       << ",zero-code=" << zero_mse << ",shuffled-code=" << shuffled_mse
       << ". Simple baselines and code controls expose blurry averages or weak encoding. This short-run "
          "reconstruction is not classification,denoising or a generative prior.</p>";
  ae::DigitGallery(directory, test,
                   {test.images, result[1], nn::abs(result[1] - ae::Flatten(test.images)), mean_prediction,
                    initial_test[1], zero_decode, shuffled},
                   html);
  check::Require(bool(metrics) && bool(html) && bool(history) && bool(batches),
                 "write metrics/gallery/history/fit-only batch provenance");
  std::cout << "undercomplete50992parameters/4tensors;" << epochs << "epochs," << step << "updates,selected=" << selected
            << ",fitMSE " << initial_mse << " -> " << fit_mse << "\ntest MSE=" << test_mse << ",mean=" << mean_mse
            << ",black=" << black_mse << ",shuffled=" << shuffled_mse << ",zero-code=" << zero_mse
            << "\nfresh restore/explicit encode-decode/determinism/single-batch/immutable4weights/all4finite nonzero "
               "gradients passed;no labels used in training\ntraining time "
            << seconds << " s\nArtifacts:" << directory.string() << "/index.html\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int                   epochs = 3;
    std::string           init;
    std::filesystem::path data = paths::Dataset("saliency"),
                          out  = paths::OutputRoot() / "models/undercomplete_autoencoder";
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
    check::Require(epochs > 0 && epochs <= 100, "short default or explicit1..100epochs");
    Checks();
    Run(data, out, epochs, init);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
