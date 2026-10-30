// CMD: just run-nn masked_autoencoder
// Visible-only transformer MAE with 75 percent masking and a smaller mask-token decoder.
//
// Port of tflow's masked_autoencoder. Preparation (in ~/projects/tflow_examples):
// just autoencoder-assets; data are read from $DATAMUNGE_DATA_DIR/saliency. Outputs
// (history, per-epoch masks and arrays, probe.npz, outputs.npz, metrics.json, index.html
// and digit<k>.png galleries) go to output/nn/models/masked_autoencoder. The patch masks
// are the same hash-derived selections as tflow's (bit-identical). Checkpoints are
// best.npz files instead of TF SaveV2 bundles. Options: epochs=N data=PATH out=PATH
// init=PATH (start from saved parameters, e.g. tflow's initial_arrays.npz, to follow its
// trajectory exactly; by default nn's own initialiser is used).

#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>

#include "common/masked_autoencoder.hpp"

namespace {

namespace nn = datamunge::nn;

void LossChecks() {
  std::vector<float> mask(32);
  for (int b = 0; b < 2; ++b)
    for (int p = 0; p < 16; ++p) mask[b * 16 + p] = p < 4 ? 0.f : 1.f;
  auto patches    = nn::full({2, 16, 49}, .7f);
  auto y          = std::vector<float>(2 * 16 * 49, .3f);
  auto prediction = nn::from_vector(y, {2, 16, 49}).requires_grad_();
  auto m          = nn::from_vector(mask, {2, 16});
  auto loss       = ae::MaeLoss(prediction, patches, m);
  auto gradient   = nn::grad(loss, {prediction})[0].to_vector<float>();
  check::Near(loss.item<float>(), .16, 3e-7, "loss denominator counts ONLY12hidden patches per image");
  for (int i = 0; i < 2 * 16 * 49; ++i)
    check::Near(gradient[i], ((i / 49) % 16 < 4) ? 0 : -.8 / (2 * 12 * 49), 2e-10,
                "all1568prediction derivatives;visible EXACTzero");
  const double before = loss.item<float>();
  for (int b = 0; b < 2; ++b)
    for (int p = 0; p < 4; ++p)
      for (int i = 0; i < 49; ++i) y[(b * 16 + p) * 49 + i] = .99f;
  check::Near(ae::MaeLoss(nn::from_vector(y, {2, 16, 49}), patches, m).item<float>(), before, 0,
              "visible predictions excluded exactly");
  std::cout << "masked loss/denominator/ALL1568derivatives/visible perturbation invariance passed\n";
}

void Run(const std::filesystem::path& data, const std::filesystem::path& root, int epochs, const std::string& init) {
  auto pools = ae::LoadSaliency(data);
  std::filesystem::create_directories(root);
  ae::MaskedAutoencoderModel model;
  if (!init.empty()) model.parameters.load(init);
  check::Require(model.parameters.size() == 35 && model.parameters.count() == 13649,
                 "independent tiny transformer parameter census");
  nn::save_npz((root / "initial_arrays.npz").string(), ae::Arrays(model.parameters));
  const auto test_selection = ae::MaeSelection(pools[2].source);
  auto       initial        = model.Evaluate(pools[2].images, test_selection);
  nn::save_npz((root / "initial_outputs.npz").string(), {{"code", initial[0]}, {"prediction", initial[1]}});
  nn::BatchOrder order(1000, 32, 1889);
  std::ofstream  history(root / "history.tsv"), batches(root / "batches.tsv");
  history.precision(17);
  history << "epoch\tfit_preupdate_masked_mse\tvalidation_masked_mse\n";
  batches << "epoch\tstep\trow\tsource_id\n";
  const auto fit_source = pools[0].source.to_vector<int>();
  const auto best_path  = (root / "best.npz").string();
  double     best       = std::numeric_limits<double>::infinity();
  int        selected = 0, step = 0;
  const auto started = std::chrono::steady_clock::now();
  for (int epoch = 1; epoch <= epochs; ++epoch) {
    order.begin_epoch();
    const auto epoch_selection = ae::MaeSelection(pools[0].source, epoch);
    nn::save_npz((root / ("epoch" + std::to_string(epoch) + "_mask.npz")).string(), {{"selection", epoch_selection}});
    double sum  = 0;
    int    seen = 0;
    for (;;) {
      auto rows = order.next();
      if (rows.empty()) break;
      auto input = nn::take_rows(pools[0].images, rows), selection = nn::take_rows(epoch_selection, rows);
      // Visible-only encode, mask-token decode, hidden-patch-only loss.
      auto net  = ae::MaskedAutoencoder(model.parameters, input, selection);
      auto loss = ae::MaeLoss(net.prediction, net.patches, net.mask);
      if (step == 0) {
        nn::GradOptions keep;
        keep.retain_graph = true;
        auto gradients    = ae::CheckGradients(model.parameters, nn::grad(loss, model.parameters.parameters(), keep),
                                               "all transformer gradients finite",
                                               "ALL35tensors receive nonzero masked-loss gradients");
        nn::save_npz((root / "first_gradients.npz").string(), gradients);
        nn::save_npz((root / "first_batch.npz").string(), {{"images", input}, {"selection", selection}});
      }
      const double value = loss.item<float>();
      check::Require(std::isfinite(value) && value >= 0, "finite masked MSE");
      model.optimizer->minimize(loss);
      sum += double(rows.size()) * value;
      seen += int(rows.size());
      for (auto row : rows) batches << epoch << '\t' << step << '\t' << row << '\t' << fit_source[row] << '\n';
      if (step == 0) nn::save_npz((root / "first_update_arrays.npz").string(), ae::Arrays(model.parameters));
      ++step;
    }
    check::Require(seen == 1000, "every fitting record once per epoch");
    const auto   val_selection = ae::MaeSelection(pools[1].source);
    auto         value         = model.Evaluate(pools[1].images, val_selection);
    const double mse           = ae::MaeMse(value[1], pools[1].images, val_selection);
    check::Near(value[2].item<float>(), mse, 2e-7, "graph and independent native hidden-only MSE");
    history << epoch << '\t' << sum / seen << '\t' << mse << '\n';
    nn::save_npz((root / ("epoch" + std::to_string(epoch) + "_arrays.npz")).string(), ae::Arrays(model.parameters));
    if (mse < best) {
      best     = mse;
      selected = epoch;
      model.parameters.save(best_path);  // fixed masked validation MSE selects epoch
    }
    std::cout << "MAE epoch" << epoch << ":validation masked MSE=" << mse << std::endl;
  }
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  ae::MaskedAutoencoderModel deployed(false);
  deployed.parameters.load(best_path);  // fresh inference MAE restore
  const auto       before = ae::Snapshot(deployed.parameters);
  nn::NamedTensors saved;
  for (int split = 0; split < 3; ++split) {
    const std::string name      = split == 0 ? "fit" : split == 1 ? "validation" : "test";
    const auto        selection = ae::MaeSelection(pools[split].source);
    auto              out       = deployed.Evaluate(pools[split].images, selection);
    saved.emplace_back(name + "_code", out[0]);
    saved.emplace_back(name + "_prediction", out[1]);
    saved.emplace_back(name + "_selection", selection);
    if (split == 1)
      check::Near(ae::MaeMse(out[1], pools[split].images, selection), best, 1e-8, "fresh selected validation metric");
  }
  auto test = deployed.Evaluate(pools[2].images, test_selection), repeat = deployed.Evaluate(pools[2].images, test_selection);
  check::Require(ae::Bytes(test[1]) == ae::Bytes(repeat[1]) &&
                     ae::Bytes(deployed.Decode(test[0], test_selection)) == ae::Bytes(test[1]),
                 "deterministic exact target-free code decoding");
  const auto         images = pools[2].images.to_vector<float>(), sel = test_selection.to_vector<float>();
  std::vector<float> altered(images.size()), shifted(images.size()), permuted(sel.size());
  for (int b = 0; b < 200; ++b) {
    std::copy_n(images.data() + ((b + 1) % 200) * 784, 784, shifted.data() + b * 784);
    for (int p = 0; p < 16; ++p)
      for (int i = 0; i < 49; ++i) {
        const int j = b * 784 + ae::MaePixel(p, i);
        altered[j]  = ae::MaeHidden(sel, b, p) ? 1 - images[j] : images[j];
      }
    for (int k = 0; k < 4; ++k)
      for (int p = 0; p < 16; ++p) permuted[b * 64 + k * 16 + p] = sel[b * 64 + (3 - k) * 16 + p];
  }
  const nn::Shape image_shape = pools[2].images.shape();
  auto hidden = deployed.Evaluate(nn::from_vector(altered, image_shape), test_selection);
  check::Require(ae::Bytes(hidden[0]) == ae::Bytes(test[0]) && ae::Bytes(hidden[1]) == ae::Bytes(test[1]),
                 "changing EVERY hidden input pixel leaves codes/predictions BITEXACT;no target leakage");
  auto shuffled    = deployed.Evaluate(nn::from_vector(shifted, image_shape), test_selection);
  auto permutation = deployed.Evaluate(pools[2].images, nn::from_vector(permuted, {200, 4, 16}));
  const auto second_selection = ae::MaeSelection(pools[2].source, 0, 1913);
  auto       second           = deployed.Evaluate(pools[2].images, second_selection);
  const auto zero_prediction  = deployed.Decode(nn::zeros({200, 4, 32}), test_selection);
  saved.emplace_back("zero_prediction", zero_prediction);
  saved.emplace_back("shuffled_prediction", shuffled[1]);
  saved.emplace_back("second_selection", second_selection);
  saved.emplace_back("second_prediction", second[1]);
  saved.emplace_back("permuted_prediction", permutation[1]);
  auto single = deployed.Evaluate(nn::take_rows(pools[2].images, {0}), nn::take_rows(test_selection, {0}));
  const double single_error      = ae::MaxAbsDiff(single[1], test[1], 784),
               permutation_error = ae::MaxAbsDiff(permutation[1], test[1], 200 * 784);
  check::Require(single_error < 2e-6 && permutation_error < 2e-6,
                 "batch parity and visible token permutation/restoration equivariance");
  check::Require(before == ae::Snapshot(deployed.parameters), "inference and controls leave ALL35parameter tensors immutable");
  const std::vector<std::int64_t> first8{0, 1, 2, 3, 4, 5, 6, 7};
  auto probe = nn::take_rows(pools[2].images, first8), probe_selection = nn::take_rows(test_selection, first8);
  auto stages = deployed.Probe(probe, probe_selection);
  nn::save_npz((root / "probe.npz").string(), {{"images", probe},
                                              {"selection", probe_selection},
                                              {"patches", stages[0]},
                                              {"visible", stages[1]},
                                              {"mask", stages[2]},
                                              {"encoder_attention", stages[3]},
                                              {"decoder_attention", stages[4]}});
  nn::save_npz((root / "outputs.npz").string(), saved);
  nn::save_npz((root / "model_arrays.npz").string(), ae::Arrays(deployed.parameters));
  ae::SaveData(root / "data.npz", pools);
  ae::CopyManifests(data, root);
  const double test_mse = ae::MaeMse(test[1], pools[2].images, test_selection);
  std::ofstream meta(root / "metrics.json");
  meta.precision(17);
  meta << "{\"parameters\":13649,\"epochs\":" << epochs << ",\"updates\":" << step << ",\"selected_epoch\":" << selected
       << ",\"test_masked_mse\":" << test_mse
       << ",\"zero_masked_mse\":" << ae::MaeMse(zero_prediction, pools[2].images, test_selection)
       << ",\"shuffled_masked_mse\":" << ae::MaeMse(shuffled[1], pools[2].images, test_selection)
       << ",\"second_masked_mse\":" << ae::MaeMse(second[1], pools[2].images, second_selection)
       << ",\"single_batch_max_error\":" << single_error
       << ",\"visible_permutation_max_error\":" << permutation_error << ",\"training_seconds\":" << seconds << "}\n";
  // Gallery panels: original, visible input (gray hidden), raw prediction, visible-copy composite,
  // fit-only mean, hidden-only absolute error, zero-code raw prediction.
  const auto         predicted = ae::MaeImage(test[1]).to_vector<float>();
  std::vector<float> visible(images.size()), composite(images.size()), error(images.size(), 0.f);
  for (int b = 0; b < 200; ++b)
    for (int p = 0; p < 16; ++p)
      for (int i = 0; i < 49; ++i) {
        const int  j      = b * 784 + ae::MaePixel(p, i);
        const bool hidden_patch = ae::MaeHidden(sel, b, p);
        visible[j]   = hidden_patch ? .5f : images[j];
        composite[j] = hidden_patch ? predicted[j] : images[j];
        error[j]     = hidden_patch ? std::abs(predicted[j] - images[j]) : 0.f;
      }
  std::ofstream html(root / "index.html");
  html << "<!doctype html><meta charset='utf-8'><title>Tiny transformer MAE</title><style>body{font:16px system-ui;"
          "max-width:1000px;margin:30px auto}img{max-width:100%;image-rendering:pixelated}</style><h1>Visible-only "
          "transformer masked autoencoder</h1><p>Original,visible input(gray hidden),raw prediction,visible-copy "
          "composite,fit-only mean,hidden-only absolute error,zero-code raw prediction. Fixed[0,1]gray/nearest4/no "
          "error amplification. First heldout example per digit,unfiltered. Composite copies known pixels;raw "
          "prediction does not.</p><p>16nonoverlapping7x7patches;75percent masked,encoder ONLY4visible tokens,width32/"
          "one2-head transformer block. Decoder width16/one2-head block receives restored projected codes and learned "
          "mask tokens plus fixed2D sin/cos positions. MLP ratio2,tanh-GELU,prenorm;no CLS token or dropout. Sigmoid "
          "raw-pixel targets;loss ONLY12hidden patches. Small educational adaptation of <a "
          "href='https://arxiv.org/abs/2111.06377'>He et al.MAE</a>,not an ImageNet reproduction or transfer-learning "
          "benchmark.</p><p>"
       << epochs
       << "epochs,13649parameters,Adam.003,batch32. New reproducible per-example masks each training epoch;fixed "
          "disjoint validation/test masks. Test masked MSE="
       << test_mse
       << ". Zero/shuffled content and second mask diagnostics remain in metrics. Hidden-input perturbation leaves "
          "codes/predictions BITEXACT. Short training can produce blurry generic patches;no classification/semantic or "
          "compression claim.</p>";
  ae::DigitGallery(root, pools[2],
                   {pools[2].images, nn::from_vector(visible, {200, 784}), nn::from_vector(predicted, {200, 784}),
                    nn::from_vector(composite, {200, 784}), ae::Constant(ae::PixelMean(pools[0].images), 200),
                    nn::from_vector(error, {200, 784}), ae::MaeImage(zero_prediction)},
                   html);
  check::Require(bool(meta) && bool(history) && bool(batches) && bool(html), "write complete masked transformer evidence");
  std::cout << "MAE selected epoch" << selected << ",test masked MSE=" << test_mse
            << ",zero=" << ae::MaeMse(zero_prediction, pools[2].images, test_selection)
            << ",shuffled=" << ae::MaeMse(shuffled[1], pools[2].images, test_selection)
            << ",second mask=" << ae::MaeMse(second[1], pools[2].images, second_selection)
            << ";fresh restore/exact decode/hidden leakage/permutation/batch/immutable checks passed\ntraining time "
            << seconds << " s\nArtifacts:" << root.string() << "/index.html\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int                   epochs = 3;
    std::string           init;
    std::filesystem::path data = paths::Dataset("saliency"), out = paths::OutputRoot() / "models/masked_autoencoder";
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
    LossChecks();
    Run(data, out, epochs, init);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
