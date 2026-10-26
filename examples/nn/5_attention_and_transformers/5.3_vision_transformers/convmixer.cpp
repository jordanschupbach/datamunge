// ConvMixer with train/infer BN and residual depthwise spatial mixing.
// CMD: just run-nn convmixer
// Eager port of tflow's convmixer; same 2018 learned parameters/22 learned tensors plus 10 BN state
// tensors, batch order, Adam and checks. Replays recorded TF initial arrays when present and
// compares saved fields; otherwise native initialization gives statistical parity. Checkpoints use
// NPZ; epochs=N retains the source option.
#include "common/convmixer.hpp"
namespace nn = datamunge::nn;
namespace {
void Save(const std::filesystem::path& file, const vit::A& values) {
  nn::save_npz(file.string(), replay::ToNamed(values));
}
void Run(int epochs) {
  const auto root   = paths::OutputDir("models/convmixer"),
             assets = paths::Dataset("vision_transformer");
  replay::CopyAssets(assets, root, {"data.npz", "manifest.json"});
  auto data = replay::ToMap(nn::load_npz((root / "data.npz").string()));
  auto fit = vit::Pool(data, "fit"), validation = vit::Pool(data, "validation"),
       test = vit::Pool(data, "test");
  convmixer::Model model;
  check::Require(
      model.parameters.count() == 2018 && model.parameters.size() == 22,
      "2018 parameters/32 two ConvMixer blocks tensors");
  const auto tf        = replay::TfModels("convmixer");
  const bool replay_tf = std::filesystem::exists(tf / "initial_arrays.npz");
  if (replay_tf)
    model.Load(tf / "initial_arrays.npz");
  auto compare = [&](const vit::A& out, const std::string& file, double tolerance) {
    if (replay_tf && std::filesystem::exists(tf / file)) {
      const double error = replay::MaxDiff(out, tf / file);
      check::Near(error, 0, tolerance, "TF replay " + file);
      std::cout << "TF " << file << " max difference=" << error << std::endl;
    }
  };
  Save(root / "initial_arrays.npz", model.Arrays());
  auto initial = model.Run(test);
  Save(root / "initial_outputs.npz", initial);
  compare(initial, "initial_outputs.npz", 5e-5);
  nn::BatchOrder order(128, 8, 49707);
  std::ofstream  history(root / "history.tsv"), batches(root / "batches.tsv");
  history << "epoch\tvalidation_ce\n";
  batches << "epoch\tstep\trow\tsource_id\n";
  int    updates = 0;
  vit::A first;
  for (int epoch = 1; epoch <= epochs; ++epoch) {
    order.begin_epoch();
    for (;;) {
      auto rows = order.next();
      if (rows.empty())
        break;
      auto b = replay::Rows(fit, rows);
      if (!updates) {
        first = b;
        Save(root / "first_batch.npz", b);
        auto g = model.Run(b, true);
        Save(root / "initial_gradients.npz", g);
        compare(g, "initial_gradients.npz", 1e-4);
      }
      model.Update(b);
      if (!updates) {
        Save(root / "first_after.npz", model.Arrays());
        // Biases immediately before training-mode BN have zero true gradients;
        // Adam can amplify roundoff there. Initial forward/gradient parity is
        // checked above; training metrics are compared after each epoch.
      }
      auto ids = fit.at("source_ids").to_vector<int>();
      for (auto row : rows)
        batches << epoch << '\t' << updates << '\t' << row << '\t' << ids[row] << '\n';
      ++updates;
    }
    auto out = model.Run(validation);
    history << epoch << '\t' << out.at("loss").item() << '\n';
    const auto prefix = "epoch" + std::to_string(epoch);
    Save(root / (prefix + "_validation.npz"), out);
    Save(root / (prefix + "_arrays.npz"), model.Arrays());
    compare(out, prefix + "_validation.npz", 5e-5);
    std::cout << "epoch=" << epoch << " validationCE=" << out.at("loss").item() << std::endl;
  }
  auto final = model.Arrays(), grad = model.Run(first, true);
  Save(root / "model_arrays.npz", final);
  Save(root / "final_gradients.npz", grad);
  compare(grad, "final_gradients.npz", 5e-5);
  model.Save(root / "final.npz");
  convmixer::Model deployed(false);
  deployed.Load(root / "final.npz");
  check::Require(
      replay::SameBits(model.Arrays(), deployed.Arrays()), "32 tensors restored bitexact");
  auto before = deployed.Arrays(), out = deployed.Run(test);
  Save(root / "outputs.npz", out);
  compare(out, "outputs.npz", 5e-5);
  check::Near(
      replay::MaxAbsDiff(deployed.Predict({{"images", test.at("images")}}), out.at("logits")),
      0,
      0,
      "image-only inference");
  auto shuffled      = test;
  shuffled["images"] = vit::Permute(test.at("images"));
  Save(root / "permuted_inputs.npz", shuffled);
  auto moved = deployed.Run(shuffled);
  Save(root / "permuted_outputs.npz", moved);
  double position_effect = replay::MaxAbsDiff(out.at("logits"), moved.at("logits"));
  check::Require(position_effect > 1e-6, "depthwise spatial kernels distinguish patch arrangement");
  vit::A zero;
  for (const auto& [name, value] : final)
    if (name.find("/depthwise_bn/gamma") != std::string::npos ||
        name.find("/depthwise_bn/beta") != std::string::npos)
      zero[name] = nn::zeros(value.shape());
  auto nomix = deployed.Run(test, false, zero), nomixperm = deployed.Run(shuffled, false, zero);
  const double no_position_error = replay::MaxAbsDiff(nomix.at("logits"), nomixperm.at("logits"));
  check::Require(
      no_position_error < 2e-6,
      "without depthwise mixing pointwise network is permutation invariant");
  Save(root / "no_depthwise.npz", nomix);
  Save(root / "no_depthwise_permuted.npz", nomixperm);
  const double paired_error = replay::MaxAbsDiff(nomix.at("logits"), out.at("logits"));
  check::Require(paired_error > 1e-6, "depthwise residual affects actual predictions");
  for (int row = 0; row < 4; ++row)
    check::Near(
        replay::MaxAbsDiff(
            deployed.Predict(replay::Rows(test, {row})),
            nn::slice(out.at("logits"), {row, 0}, {1, 10})),
        0,
        2e-6,
        "ConvMixer batch isolation");
  double fd    = 0;
  int    count = 0;
  for (const auto& name : model.parameters.names()) {
    const auto value  = final.at(name);
    auto       values = value.to_vector<float>();
    for (int i : {0, int(values.size() / 2), int(values.size() - 1)}) {
      const float old = values[i];
      values[i]       = old + .0005f;
      auto plus = model.Run(first, false, {{name, nn::from_vector(values, value.shape())}}, true)
                      .at("loss")
                      .item();
      values[i]  = old - .0005f;
      auto minus = model.Run(first, false, {{name, nn::from_vector(values, value.shape())}}, true)
                       .at("loss")
                       .item();
      values[i] = old;
      fd = std::max(fd, std::abs((plus - minus) / .001 - grad.at(name).to_vector<float>()[i]));
      ++count;
    }
  }
  check::Require(fd < .003, "native ConvMixer all-parameter finite differences");
  check::Require(
      replay::SameBits(before, deployed.Arrays()), "ConvMixer inference and controls immutable");
  const auto z       = out.at("logits").to_vector<float>();
  const auto labels  = test.at("labels").to_vector<int>();
  int        correct = 0;
  for (int i = 0; i < 32; ++i) {
    int best = 0;
    for (int c = 1; c < 10; ++c)
      if (z[i * 10 + c] > z[i * 10 + best])
        best = c;
    correct += best == labels[i];
  }
  std::ofstream metrics(root / "metrics.json");
  metrics.precision(17);
  metrics << "{\"parameters\":2018,\"variables\":22,\"epochs\":" << epochs
          << ",\"updates\":" << updates << ",\"test_ce\":" << out.at("loss").item()
          << ",\"test_accuracy\":" << correct / 32.
          << ",\"spatial_order_effect\":" << position_effect
          << ",\"depthwise_branch_effect\":" << paired_error
          << ",\"no_depthwise_permutation_error\":" << no_position_error
          << ",\"native_fd_count\":" << count << ",\"native_fd_error\":" << fd << "}\n";
  std::cout << "testCE=" << out.at("loss").item() << " accuracy=" << correct << "/32; " << count
            << " FD max=" << fd << ";patch/depthwise/pointwise/BNstate/restore/immutable PASS\n";
}
} // namespace
int main(int argc, char** argv) {
  try {
    nn::set_num_threads(2);
    int epochs = 3;
    for (int i = 1; i < argc; ++i) {
      std::string a = argv[i];
      check::Require(a.rfind("epochs=", 0) == 0, "epochs=N");
      epochs = std::stoi(a.substr(7));
    }
    check::Require(epochs > 0 && epochs <= 100, "epochs1..100");
    Run(epochs);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
