// CMD: just run-nn convolutional_autoencoder
// Actual spatial 7x7x4 code / nearest-repeat convolutional decoder, 741 learned parameters.
// Three default epochs, mean/initial/pool-only/zero/shuffled-code controls.
//
// Port of tflow's convolutional_autoencoder. Preparation (in ~/projects/tflow_examples):
// just autoencoder-assets; data are read from $DATAMUNGE_DATA_DIR/saliency. Outputs
// (history, npz arrays, metrics.json, index.html and digit<k>.png galleries) go to
// output/nn/models/convolutional_autoencoder. Options: epochs=N data=PATH out=PATH
// init=PATH (start from saved parameters, e.g. tflow's initial_arrays.npz, to
// follow its trajectory exactly; by default nn's own initialiser is used).

#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>

#include "common/autoencoder.hpp"

namespace {

namespace nn = datamunge::nn;

void Checks() {
  std::vector<float> pixels(16);
  for (int i = 0; i < 16; ++i) pixels[i] = float(i) / 15;
  auto x        = nn::from_vector(pixels, {1, 4, 4, 1}).requires_grad_();
  auto code     = nn::avg_pool2d(x, 2, 2);
  auto enlarged = nn::upsample_nearest(code, 2);
  auto loss     = nn::mean(nn::square(enlarged - x));
  auto gradient = nn::grad(loss, {x})[0];
  double mse    = 0;
  for (int y = 0; y < 4; ++y)
    for (int a = 0; a < 4; ++a) {
      double mean = 0;
      for (int dy = 0; dy < 2; ++dy)
        for (int dx = 0; dx < 2; ++dx) mean += pixels[((y / 2) * 2 + dy) * 4 + (a / 2) * 2 + dx] / 4.;
      check::Near(code.at({0, y / 2, a / 2, 0}), mean, 4e-8, "all4pool means");
      check::Near(enlarged.at({0, y, a, 0}), mean, 4e-8, "all16nearestrepeat pixels,not tile-whole-image");
      const double error = pixels[y * 4 + a] - mean;
      mse += error * error / 16;
      check::Near(gradient.at({0, y, a, 0}), error / 8, 2e-8, "all16analytic pool+repeat MSE input derivatives");
    }
  check::Near(loss.item<float>(), mse, 2e-8, "all16spatial reconstruction MSE terms");
  std::cout << "known4x4->2x2->4x4 allpool/repeatpixels/MSE/ALL16analytic gradients passed\n";
}

void Run(const std::filesystem::path& data, const std::filesystem::path& root, int epochs, const std::string& init) {
  auto pools = ae::LoadSaliency(data);
  ae::ConvolutionalAutoencoderModel model;
  if (!init.empty()) model.parameters.load(init);
  check::Require(model.parameters.size() == 8 && model.parameters.count() == 741,
                 "independent3x3kernels+biases741params/8tensors");
  std::filesystem::create_directories(root);
  nn::save_npz((root / "initial_arrays.npz").string(), ae::Arrays(model.parameters));
  auto initial = model.Evaluate(pools[2].images);
  nn::BatchOrder order(1000, 32, 1807);
  std::ofstream  history(root / "history.tsv"), batches(root / "batches.tsv");
  history.precision(17);
  history << "epoch\tfit_preupdate_mse\tvalidation_mse\n";
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
      auto input = nn::take_rows(pools[0].images, rows);
      if (step == 0) {
        auto gradients = ae::CheckGradients(model.parameters, model.Gradient(input),
                                            "all convolutional parameter gradients finite",
                                            "all8convolutional tensors receive reconstruction gradients");
        nn::save_npz((root / "first_gradients.npz").string(), gradients);
        nn::save_npz((root / "first_batch.npz").string(), {{"images", input}});
      }
      sum += double(rows.size()) * model.Update(input);
      seen += int(rows.size());
      for (auto row : rows) batches << epoch << '\t' << step << '\t' << row << '\t' << fit_source[row] << '\n';
      if (step == 0) nn::save_npz((root / "first_update_arrays.npz").string(), ae::Arrays(model.parameters));
      ++step;
    }
    check::Require(seen == 1000, "each fitting record once per epoch including last8");
    auto         validation = model.Evaluate(pools[1].images);
    const double value      = ae::Mse(validation[1], pools[1].images);
    history << epoch << '\t' << sum / seen << '\t' << value << '\n';
    nn::save_npz((root / ("epoch" + std::to_string(epoch) + "_arrays.npz")).string(), ae::Arrays(model.parameters));
    if (value < best) {
      best     = value;
      selected = epoch;
      model.parameters.save(best_path);  // validation MSE alone selects the checkpoint
    }
    std::cout << "epoch" << epoch << ":fitPreMSE=" << sum / seen << ",validationMSE=" << value << std::endl;
  }
  const double train_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  model.parameters.load(best_path);  // restore selected weights for the checkpoint-gradient probes
  ae::ConvolutionalAutoencoderModel deployed(false);
  deployed.parameters.load(best_path);  // fresh spatial inference restore
  const auto       before = ae::Snapshot(deployed.parameters);
  nn::NamedTensors arrays;
  for (int split = 0; split < 3; ++split) {
    const std::string name = split == 0 ? "fit" : split == 1 ? "validation" : "test";
    auto              out  = deployed.Evaluate(pools[split].images);
    check::Require(out[0].shape() == nn::Shape({pools[split].images.size(0), 7, 7, 4}),
                   "actual spatial196-coordinate bottleneck");
    check::Require(out[1].shape() == pools[split].images.shape(), "decoder restores original28x28x1 geometry");
    arrays.emplace_back(name + "_images", pools[split].images);
    arrays.emplace_back(name + "_source_ids", pools[split].source);
    arrays.emplace_back(name + "_code", out[0]);
    arrays.emplace_back(name + "_reconstruction", out[1]);
    if (split == 1) check::Near(ae::Mse(out[1], pools[split].images), best, 1e-8, "fresh restored validation MSE");
  }
  arrays.emplace_back("test_labels", pools[2].labels);
  arrays.emplace_back("initial_test_reconstruction", initial[1]);
  auto test = deployed.Evaluate(pools[2].images), repeat = deployed.Evaluate(pools[2].images);
  check::Require(ae::Bytes(test[1]) == ae::Bytes(repeat[1]), "deterministic spatial reconstruction");
  check::Require(ae::Bytes(deployed.Decode(test[0])) == ae::Bytes(test[1]),
                 "standalone convolutional decoder EXACT encode/decode");
  auto probe           = ae::Rows(pools[2], {0, 1, 2, 3, 4, 5, 6, 7});
  auto stages          = deployed.Activations(probe.images);
  auto probe_gradients = model.Gradient(probe.images);
  nn::NamedTensors probe_arrays;
  for (std::size_t k = 0; k < probe_gradients.size(); ++k)
    probe_arrays.emplace_back(model.parameters.names()[k], probe_gradients[k]);
  nn::save_npz((root / "probe_gradients.npz").string(), probe_arrays);
  const nn::Shape expected[] = {{8, 28, 28, 8}, {8, 14, 14, 8}, {8, 14, 14, 4},
                                {8, 7, 7, 4},   {8, 14, 14, 8}, {8, 28, 28, 1}};
  for (int k = 0; k < 6; ++k) {
    check::Require(stages[k].shape() == expected[k], "all six actual intermediate spatial shapes");
    arrays.emplace_back("probe_stage" + std::to_string(k), stages[k]);
  }
  arrays.emplace_back("probe_images", probe.images);
  auto         single      = deployed.Evaluate(ae::Rows(pools[2], {0}).images);
  const double batch_error = ae::MaxAbsDiff(single[1], test[1], 784);
  check::Require(batch_error < 2e-6, "single/batch convolutional inference parity");
  // Decode an all-zero code and each image's code cyclically shifted by one record.
  auto zero     = deployed.Decode(nn::zeros({200, 7, 7, 4}));
  auto shuffled = deployed.Decode(nn::concat({nn::slice_axis(test[0], 0, 1, 200), nn::slice_axis(test[0], 0, 0, 1)}, 0));
  arrays.emplace_back("zero_reconstruction", zero);
  arrays.emplace_back("shuffled_reconstruction", shuffled);
  auto mean4 = nn::reshape(ae::Constant(ae::PixelMean(pools[0].images), 200), {200, 28, 28, 1});
  arrays.emplace_back("mean_reconstruction", mean4);
  // 4x4 average then nearest repeat: a 49-coordinate input-dependent baseline.
  auto pooled = nn::upsample_nearest(nn::avg_pool2d(pools[2].images, 4, 4), 4).contiguous();
  arrays.emplace_back("pool_repeat_reconstruction", pooled);
  check::Require(before == ae::Snapshot(deployed.parameters),
                 "all8parameters immutable through inference/zero/shuffledcode controls");
  nn::save_npz((root / "convolutional.npz").string(), arrays);
  nn::save_npz((root / "model_arrays.npz").string(), ae::Arrays(deployed.parameters));
  ae::CopyManifests(data, root);
  const double mse = ae::Mse(test[1], pools[2].images), mean_mse = ae::Mse(mean4, pools[2].images),
               pool_mse = ae::Mse(pooled, pools[2].images);
  std::ofstream metrics(root / "metrics.json");
  metrics.precision(17);
  metrics << "{\"epochs\":" << epochs << ",\"updates\":" << step << ",\"selected_epoch\":" << selected
          << ",\"parameters\":741,\"test_mse\":" << mse << ",\"mean_mse\":" << mean_mse
          << ",\"pool_repeat_mse\":" << pool_mse << ",\"initial_mse\":" << ae::Mse(initial[1], pools[2].images)
          << ",\"zero_mse\":" << ae::Mse(zero, pools[2].images)
          << ",\"shuffled_mse\":" << ae::Mse(shuffled, pools[2].images)
          << ",\"single_batch_max_error\":" << batch_error << "}\n";
  std::ofstream html(root / "index.html");
  html << "<!doctype html><meta charset='utf-8'><title>Convolutional autoencoder</title><style>body{font:16px "
          "system-ui;max-width:1100px;margin:30px auto}img{max-width:100%;image-rendering:pixelated}section{margin:"
          "25px 0}</style><h1>Convolutional autoencoder: spatial bottleneck</h1><p>Original, trained, absolute "
          "error, fit-only mean, 4x4 average/nearest repeat, untrained, zero code, cyclic shuffled code. Fixed [0,1] "
          "grayscale/nearest 4x, no error amplification or quality filtering.</p><p>Conv3x3 same 1->8 ReLU, AvgPool2, "
          "Conv3x3 same 8->4 ReLU, AvgPool2 gives a 7x7x4 code. Decoder nearest-repeat 2, Conv3x3 same 4->8 ReLU, "
          "nearest-repeat 2, Conv3x3 same 8->1 sigmoid. No skips/dense bypass; 741 untied parameters, "
       << epochs
       << " epochs, batch 32, Adam .003. Clean inputs are targets; no labels enter training or validation-MSE "
          "selection. The 196 continuous code values are 4x fewer coordinates, not a bit-rate/file compression "
          "claim.</p><p>Held-out MSE trained="
       << mse << ", fit mean=" << mean_mse << ", pool/repeat=" << pool_mse
       << ". Pool/repeat is a 49-coordinate input-dependent baseline, not a same-capacity learned comparator. "
          "Latent zero/shuffle and initial-model controls expose whether learned spatial information matters. MSE "
          "alone does not establish shape fidelity, classification, denoising or a generative prior.</p>";
  const auto error  = nn::abs(test[1] - pools[2].images);
  const auto labels = pools[2].labels.to_vector<int>(), source = pools[2].source.to_vector<int>();
  for (int digit = 0; digit < 10; ++digit) {
    int b = 0;
    while (labels[b] != digit) ++b;
    std::vector<nn::Tensor> panels;
    for (const nn::Tensor& t : {pools[2].images, test[1], error, mean4, pooled, initial[1], zero, shuffled})
      panels.push_back(ae::Row(t, b));
    const std::string file = "digit" + std::to_string(digit) + ".png";
    ae::PanelRow(root / file, panels);
    html << "<section><h2>Digit " << digit << ", source " << source[b] << "</h2><img src='" << file
         << "'></section>";
  }
  check::Require(bool(metrics) && bool(html) && bool(history) && bool(batches), "write spatial autoencoder artifacts");
  std::cout << "Convolutional741params/8tensors,selectedepoch=" << selected << ",testMSE=" << mse
            << ",fitmean=" << mean_mse << ",pool/repeat=" << pool_mse
            << "\nSpatialall6shapes/freshrestore/exactdecode/repeat/single-batch/immutable8weights/"
               "all8gradienttensors passed.\ntraining time "
            << train_seconds << " s\nArtifacts:" << root.string() << "/index.html\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int                   epochs = 3;
    std::string           init;
    std::filesystem::path data = paths::Dataset("saliency"), out = paths::OutputRoot() / "models/convolutional_autoencoder";
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
