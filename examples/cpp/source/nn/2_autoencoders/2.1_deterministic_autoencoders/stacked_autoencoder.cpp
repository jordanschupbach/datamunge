// CMD: just run-nn stacked_autoencoder
// Greedy 784->64 and 64->16 pretraining, exact transfer into 784->64->16->64->784.
// Three epochs per stage/default; validation-only selection and same-stack scratch control.
//
// Port of tflow's stacked_autoencoder (model: examples/nn/common/stacked_autoencoder.hpp).
// Preparation (in ~/projects/tflow_examples): just autoencoder-assets; data are read from
// $DATAMUNGE_DATA_DIR/saliency. Outputs (stage1/, stage2/, finetune/, scratch/ training records,
// transferred_arrays.npz, stacked.npz, metrics.json, index.html and digit<k>.png galleries) go
// to output/nn/models/stacked_autoencoder. Checkpoints are best.npz files instead of TF SaveV2
// bundles; metrics.json gains "training_seconds". Options: epochs=N data=PATH out=PATH
// init=DIR (a tflow stacked_autoencoder output directory: stage1/, stage2/ and scratch/
// initial_arrays.npz are loaded into stage1, stage2 and both full stacks, to follow tflow's
// trajectories exactly; by default nn's own initialiser is used, with the full stacks
// identical because they share a seed).

#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>
#include <tuple>

#include "common/stacked_autoencoder.hpp"

namespace {

namespace nn = datamunge::nn;
using Model  = ae::StackedAutoencoderModel;

void Checks() {
  Model     toy({4, 3, 2, 3, 4}, 1543);
  const int widths[] = {4, 3, 2, 3, 4};
  std::vector<std::vector<float>> w(4), b(4);
  for (int l = 0; l < 4; ++l) {
    for (int i = 0; i < widths[l] * widths[l + 1]; ++i) w[l].push_back(float(.1 * std::sin(i + l + .3)));
    for (int i = 0; i < widths[l + 1]; ++i) b[l].push_back(float(.01 * (i - 1)));
    toy.parameters.set("layer" + std::to_string(l) + "/W", nn::from_vector(w[l], {widths[l], widths[l + 1]}));
    toy.parameters.set("layer" + std::to_string(l) + "/b", nn::from_vector(b[l], {widths[l + 1]}));
  }
  std::vector<float> in;
  for (int i = 0; i < 8; ++i) in.push_back(.1f * float(i + 1));
  auto   input    = nn::from_vector(in, {2, 4});
  auto   actual   = toy.Evaluate(input);
  // Explicit forward through every sigmoid layer, full-stack MSE, then reverse-mode gradients.
  auto   toy_layers = ae::SigmoidAutoencoder(toy.parameters, input, toy.widths);
  auto   gradient   = nn::grad(ae::ReconstructionMse(toy_layers.back(), input), toy.parameters.parameters());
  double forward_error = 0;
  std::vector<std::vector<double>> activations(5);
  activations[0].assign(in.begin(), in.end());
  for (int l = 0; l < 4; ++l) {
    auto&      next = activations[l + 1];
    const auto got  = actual[l].to_vector<float>();
    next.resize(2 * widths[l + 1]);
    for (int n = 0; n < 2; ++n)
      for (int j = 0; j < widths[l + 1]; ++j) {
        double value = b[l][j];
        for (int i = 0; i < widths[l]; ++i) value += activations[l][n * widths[l] + i] * w[l][i * widths[l + 1] + j];
        next[n * widths[l + 1] + j] = 1 / (1 + std::exp(-value));
        forward_error = std::max(forward_error, std::abs(got[n * widths[l + 1] + j] - next[n * widths[l + 1] + j]));
      }
  }
  std::cout << "toy float32 sigmoid versus float64 maxerror=" << forward_error << "\n";
  check::Near(forward_error, 0, 2e-7, "all24toy sigmoid activations across four layers,float32 tolerance");
  std::vector<double> delta(8);
  double              loss = 0;
  for (int i = 0; i < 8; ++i) {
    const double value = activations[4][i], error = value - activations[0][i];
    loss += error * error / 8;
    delta[i] = 2 * error / 8 * value * (1 - value);
  }
  check::Near(actual.back().item<float>(), loss, 2e-8, "known full-stack MSE");
  int checked = 0;
  for (int l = 3; l >= 0; --l) {
    std::vector<double> prior(2 * widths[l], 0);
    const auto          gw = gradient[2 * l].to_vector<float>(), gb = gradient[2 * l + 1].to_vector<float>();
    for (int j = 0; j < widths[l + 1]; ++j) {
      double db = 0;
      for (int n = 0; n < 2; ++n) db += delta[n * widths[l + 1] + j];
      check::Near(gb[j], db, 2e-8, "all toy bias chain derivatives");
      ++checked;
      for (int i = 0; i < widths[l]; ++i) {
        double dw = 0;
        for (int n = 0; n < 2; ++n) {
          dw += activations[l][n * widths[l] + i] * delta[n * widths[l + 1] + j];
          prior[n * widths[l] + i] += delta[n * widths[l + 1] + j] * w[l][i * widths[l + 1] + j];
        }
        check::Near(gw[i * widths[l + 1] + j], dw, 2e-8, "all toy weight chain derivatives");
        ++checked;
      }
    }
    for (std::size_t i = 0; i < prior.size(); ++i) prior[i] *= activations[l][i] * (1 - activations[l][i]);
    delta = prior;
  }
  check::Require(checked == 48, "independent toy gradient inventory48");
  check::Require(ae::Bytes(toy.Decode(actual[1])) == ae::Bytes(actual[3]),
                 "two-layer standalone decoder has same full result");
  check::Throws([] { Model bad({4, 4, 4}, 1543); }, "non-undercomplete stack rejected");
  std::cout << "toy four-layer sigmoid forward/MSE/ALL48analytic derivatives/transfer/decode/strictshape checks passed\n";
}

struct Stage {
  int    selected;
  double initial_fit, fit, validation, seconds;
};

Stage Train(Model& model, const nn::Tensor& fit, const nn::Tensor& validation, const nn::Tensor& source, int epochs,
            const std::filesystem::path& directory, bool include_zero) {
  std::filesystem::create_directories(directory);
  nn::save_npz((directory / "initial_arrays.npz").string(), ae::Arrays(model.parameters));
  auto         initial     = model.Evaluate(fit);
  const double initial_fit = ae::Mse(initial[initial.size() - 2], fit);
  auto         v           = model.Evaluate(validation);
  const double epoch0      = ae::Mse(v[v.size() - 2], validation);
  double       best        = epoch0;
  int          selected    = 0;
  const auto   best_path   = (directory / "best.npz").string();
  if (include_zero)
    model.parameters.save(best_path);  // epoch0 pretrained/scratch baseline eligible for validation selection
  else
    best = std::numeric_limits<double>::infinity();
  nn::save_npz((directory / "epoch0_arrays.npz").string(), ae::Arrays(model.parameters));
  std::ofstream history(directory / "history.tsv"), batches(directory / "batches.tsv");
  history.precision(17);
  history << "epoch\ttrain_preupdate_mse\tvalidation_mse\n";
  batches << "epoch\tstep\trow\tsource_id\n";
  history << 0 << '\t' << initial_fit << '\t' << epoch0 << '\n';
  const auto     ids = source.to_vector<int>();
  nn::BatchOrder order(1000, 32, 1559);
  int            step    = 0;
  const auto     started = std::chrono::steady_clock::now();
  for (int epoch = 1; epoch <= epochs; ++epoch) {
    order.begin_epoch();
    double sum  = 0;
    int    seen = 0;
    for (;;) {
      auto rows = order.next();
      if (rows.empty()) break;
      auto x = nn::take_rows(fit, rows);
      // Forward the stage, reconstruction MSE against its own input.
      auto layers = ae::SigmoidAutoencoder(model.parameters, x, model.widths);
      auto loss   = ae::ReconstructionMse(layers.back(), x);
      if (step == 0) {
        nn::GradOptions keep;
        keep.retain_graph = true;
        nn::save_npz((directory / "first_gradients.npz").string(),
                     ae::CheckGradients(model.parameters, nn::grad(loss, model.parameters.parameters(), keep),
                                        "all stage derivatives finite",
                                        "every stage tensor receives reconstruction gradient"));
        nn::save_npz((directory / "first_batch.npz").string(), {{"images", x}});
      }
      const double value = loss.item<float>();
      check::Require(std::isfinite(value) && value >= 0, "finite stage MSE");
      model.optimizer->minimize(loss);
      sum += double(rows.size()) * value;
      seen += int(rows.size());
      for (auto row : rows) batches << epoch << '\t' << step << '\t' << row << '\t' << ids[row] << '\n';
      if (step == 0) nn::save_npz((directory / "first_update_arrays.npz").string(), ae::Arrays(model.parameters));
      ++step;
    }
    check::Require(seen == 1000, "each fitting record once per stage epoch including short last batch");
    auto         out = model.Evaluate(validation);
    const double mse = ae::Mse(out[out.size() - 2], validation);
    history << epoch << '\t' << sum / seen << '\t' << mse << '\n';
    nn::save_npz((directory / ("epoch" + std::to_string(epoch) + "_arrays.npz")).string(),
                 ae::Arrays(model.parameters));
    if (mse < best) {
      best     = mse;
      selected = epoch;
      model.parameters.save(best_path);  // validation-only stage selection
    }
    std::cout << directory.filename().string() << " epoch" << epoch << ":fitPreMSE=" << sum / seen
              << ",validationMSE=" << mse << std::endl;
  }
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  model.parameters.load(best_path);  // restore selected greedy/joint model
  auto         out     = model.Evaluate(fit);
  const double fit_mse = ae::Mse(out[out.size() - 2], fit);
  check::Require(fit_mse < initial_fit || (include_zero && selected == 0),
                 "greedy stage learns target or eligible epoch0 is explicitly retained");
  nn::save_npz((directory / "model_arrays.npz").string(), ae::Arrays(model.parameters));
  std::ofstream meta(directory / "stage.json");
  meta.precision(17);
  meta << "{\"epochs\":" << epochs << ",\"updates\":" << step << ",\"selected_epoch\":" << selected
       << ",\"include_epoch0\":" << (include_zero ? "true" : "false") << ",\"initial_fit_mse\":" << initial_fit
       << ",\"fit_mse\":" << fit_mse << ",\"validation_mse\":" << best << ",\"training_seconds\":" << seconds << "}\n";
  check::Require(bool(meta) && bool(history) && bool(batches), "write complete stage evidence");
  return {selected, initial_fit, fit_mse, best, seconds};
}

nn::Tensor Shift(const nn::Tensor& codes) {
  std::vector<std::int64_t> rows;
  for (std::int64_t b = 0; b < codes.size(0); ++b) rows.push_back((b + 1) % codes.size(0));
  return nn::take_rows(codes, rows).contiguous();
}

void Run(const std::filesystem::path& data, const std::filesystem::path& directory, int epochs,
         const std::filesystem::path& init) {
  auto pools = ae::LoadSaliency(data);
  auto fit = ae::Flatten(pools[0].images).contiguous(), val = ae::Flatten(pools[1].images).contiguous(),
       test = ae::Flatten(pools[2].images).contiguous();
  const auto& ids = pools[0].source;
  std::filesystem::create_directories(directory);
  Model stage1({784, 64, 784}, 1549), stage2({64, 16, 64}, 1553), joint({784, 64, 16, 64, 784}, 1553),
      scratch({784, 64, 16, 64, 784}, 1553);
  if (!init.empty()) {
    stage1.parameters.load((init / "stage1" / "initial_arrays.npz").string());
    stage2.parameters.load((init / "stage2" / "initial_arrays.npz").string());
    joint.parameters.load((init / "scratch" / "initial_arrays.npz").string());
    scratch.parameters.load((init / "scratch" / "initial_arrays.npz").string());
  }
  check::Require(stage1.parameters.count() == 101200 && stage2.parameters.count() == 2128 &&
                     joint.parameters.count() == 103328,
                 "independent greedy/full parameter inventories");
  check::Require(ae::Snapshot(joint.parameters) == ae::Snapshot(scratch.parameters),
                 "same full graph/initial seed before transfer");

  auto       first  = Train(stage1, fit, val, ids, epochs, directory / "stage1", false);
  const auto frozen = ae::Snapshot(stage1.parameters);
  auto encoded_fit = stage1.Evaluate(fit)[0], encoded_val = stage1.Evaluate(val)[0],
       encoded_test = stage1.Evaluate(test)[0];
  auto second       = Train(stage2, encoded_fit, encoded_val, ids, epochs, directory / "stage2", false);
  check::Require(frozen == ae::Snapshot(stage1.parameters), "stage2 cannot mutate frozen stage1 encoder/decoder");
  check::Require(ae::Bytes(stage1.Evaluate(fit)[0]) == ae::Bytes(encoded_fit),
                 "cached stage2 fitting targets remain exact frozen encoder outputs");

  const auto       a = ae::Arrays(stage1.parameters), b = ae::Arrays(stage2.parameters);
  nn::NamedTensors transfer;
  for (const auto& [target, source_name, source] : std::vector<std::tuple<std::string, std::string, int>>{
           {"layer0", "layer0", 1}, {"layer3", "layer1", 1}, {"layer1", "layer0", 2}, {"layer2", "layer1", 2}})
    for (const std::string suffix : {"/W", "/b"})
      transfer.emplace_back(target + suffix, ae::Find(source == 1 ? a : b, source_name + suffix));
  joint.Load(transfer);
  const auto loaded = ae::Arrays(joint.parameters);
  for (const auto& [key, value] : transfer)
    check::Require(ae::Bytes(ae::Find(loaded, key)) == ae::Bytes(value),
                   "ALL8pretrained arrays transferred exactly including both decoders");
  nn::save_npz((directory / "transferred_arrays.npz").string(), loaded);
  auto pretrained = joint.Evaluate(test);
  auto nested1    = stage1.Decode(stage2.Evaluate(encoded_test)[1]);
  check::Require(ae::Bytes(pretrained[3]) == ae::Bytes(nested1),
                 "composed selected greedy decoders exactly match transferred full graph");

  auto  fine  = Train(joint, fit, val, ids, epochs, directory / "finetune", true);
  auto  plain = Train(scratch, fit, val, ids, epochs, directory / "scratch", true);
  auto  final = joint.Evaluate(test), scratch_result = scratch.Evaluate(test);
  Model deployed({784, 64, 16, 64, 784}, 1553, false);
  deployed.parameters.load((directory / "finetune" / "best.npz").string());  // fresh inference restore all8 tensors
  const auto before   = ae::Snapshot(deployed.parameters);
  auto       restored = deployed.Evaluate(test);
  check::Require(ae::Bytes(restored[3]) == ae::Bytes(final[3]) && ae::Bytes(restored[1]) == ae::Bytes(final[1]),
                 "fresh restore exact all test reconstructions/codes");
  check::Require(ae::Bytes(deployed.Decode(restored[1])) == ae::Bytes(restored[3]), "full encode then two-layer decoder exact");
  check::Require(before == ae::Snapshot(deployed.parameters), "deployed inference leaves all8parameters immutable");
  check::Require(frozen == ae::Snapshot(stage1.parameters), "stage1 remains frozen during all later stages");

  auto shifted = Shift(final[1]);
  auto shuffled = deployed.Decode(shifted), zero = deployed.Decode(nn::zeros({200, 16}));
  auto mean = ae::Constant(ae::PixelMean(fit), 200);
  check::Require(before == ae::Snapshot(deployed.parameters),
                 "zero/shuffled code controls leave all8deployed parameters immutable");
  auto         first_reconstruction = stage1.Evaluate(test)[1];
  const double pre_mse = ae::Mse(pretrained[3], test), fine_mse = ae::Mse(final[3], test),
               scratch_mse = ae::Mse(scratch_result[3], test), mean_mse = ae::Mse(mean, test);
  nn::save_npz((directory / "stacked.npz").string(), {{"fit_images", fit},
                                                     {"validation_images", val},
                                                     {"test_images", test},
                                                     {"fit_source_ids", ids},
                                                     {"validation_source_ids", pools[1].source},
                                                     {"test_source_ids", pools[2].source},
                                                     {"test_labels", pools[2].labels},
                                                     {"stage1_fit_code", encoded_fit},
                                                     {"stage1_validation_code", encoded_val},
                                                     {"stage1_test_code", encoded_test},
                                                     {"stage1_test_reconstruction", first_reconstruction},
                                                     {"pretrained_test_code", pretrained[1]},
                                                     {"pretrained_test_reconstruction", pretrained[3]},
                                                     {"finetuned_test_code", final[1]},
                                                     {"finetuned_test_reconstruction", final[3]},
                                                     {"scratch_test_code", scratch_result[1]},
                                                     {"scratch_test_reconstruction", scratch_result[3]},
                                                     {"mean_reconstruction", mean},
                                                     {"shuffled_reconstruction", shuffled},
                                                     {"zero_reconstruction", zero}});
  ae::CopyManifests(data, directory);
  const double seconds = first.seconds + second.seconds + fine.seconds + plain.seconds;
  std::ofstream metrics(directory / "metrics.json");
  metrics.precision(17);
  metrics << "{\"epochs_per_stage\":" << epochs << ",\"staged_epochs\":" << epochs * 3
          << ",\"scratch_epochs\":" << epochs << ",\"staged_updates\":" << epochs * 96
          << ",\"scratch_updates\":" << epochs * 32 << ",\"pretrained_test_mse\":" << pre_mse
          << ",\"finetuned_test_mse\":" << fine_mse << ",\"scratch_test_mse\":" << scratch_mse
          << ",\"mean_test_mse\":" << mean_mse << ",\"shuffled_test_mse\":" << ae::Mse(shuffled, test)
          << ",\"zero_test_mse\":" << ae::Mse(zero, test) << ",\"training_seconds\":" << seconds << "}\n";

  std::ofstream html(directory / "index.html");
  html << "<!doctype html><meta charset='utf-8'><title>Greedy stacked autoencoder</title><style>body{font:16px "
          "system-ui;max-width:1100px;margin:30px auto}img{max-width:100%;image-rendering:pixelated}section{margin:"
          "25px 0}</style><h1>Stacked autoencoder with actual layer-wise pretraining</h1><p>Original,stage1 "
          "reconstruction,composed pretrained stack,pretrained plus fine-tuning,scratch joint training,fit-only mean,"
          "cyclic shuffled code,zero-code decoder. Fixed[0,1]grayscale/nearest4,all ten first heldout examples per "
          "true digit,no quality filtering.</p><p>784->64->16->64->784,ALL sigmoid layers,103328untied parameters. "
          "Stage1 learns784->64->784;selected encoder1 is frozen while stage2 learns64->16->64 from its actual cached "
          "representations. All8arrays including both decoders transfer EXACTLY before joint fine-tuning. No labels "
          "enter any loss or selection.</p><p>"
       << epochs << "epochs PER stage;staged total" << 3 * epochs << "epochs/" << 96 * epochs
       << "updates across different-sized models,versus scratch" << epochs << "epochs/" << 32 * epochs
       << "full-model updates. Same full architecture and fitting order;equal joint phase,NOT equal total work or a "
          "causal proof that pretraining is better. Each phase selects by its own validation MSE;epoch0 is eligible "
          "for fine-tuning/scratch.</p><p>Heldout MSE pretrained="
       << pre_mse << ",fine-tuned=" << fine_mse << ",scratch=" << scratch_mse << ",fit-mean=" << mean_mse
       << ". Greedy stage2 reconstruction quality need not preserve raw-image detail. Blurry average-looking "
          "reconstructions and weak short-run results remain visible. This16float code is not a bit-rate compression "
          "claim,classification or a generative prior.</p>";
  ae::DigitGallery(directory, pools[2],
                   {test, first_reconstruction, pretrained[3], final[3], scratch_result[3], mean, shuffled, zero}, html);
  check::Require(bool(metrics) && bool(html), "write stacked numerical report/gallery");
  std::cout << "Stage selections=" << first.selected << '/' << second.selected << '/' << fine.selected
            << ",scratch=" << plain.selected << "\nHeldoutMSE pretrained/fine/scratch/mean=" << pre_mse << '/'
            << fine_mse << '/' << scratch_mse << '/' << mean_mse
            << "\nFrozen stage1/cachedtargets/ALL8exacttransfer/nesteddecoder/freshrestore/encode-decode/"
               "immutable8weights passed. Staged work3x scratch epochs;no superiority claim.\ntraining time "
            << seconds << " s\nArtifacts:" << directory.string() << "/index.html\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int                   epochs = 3;
    std::filesystem::path init;
    std::filesystem::path data = paths::Dataset("saliency"), out = paths::OutputRoot() / "models/stacked_autoencoder";
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
    check::Require(epochs > 0 && epochs <= 100, "three default epochs PER stage or explicit1..100");
    Checks();
    Run(data, out, epochs, init);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
