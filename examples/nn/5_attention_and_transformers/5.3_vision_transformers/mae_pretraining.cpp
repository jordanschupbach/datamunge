// Visible-only MAE pretraining on real CIFAR RGB, with masked reconstruction.
// CMD: just run-nn mae_pretraining
// Eager port of tflow's mae_pretraining; same visible-only encoder and full-grid decoder,
// batch order, Adam and checks. Replays recorded TF initial arrays when present
// and compares saved fields; otherwise native initialization gives statistical
// parity. Checkpoints use NPZ; epochs=N retains the source option.
#include "common/mae_pretraining.hpp"
namespace nn = datamunge::nn;
namespace {
void Save(const std::filesystem::path& file, const vit::A& values) {
  nn::save_npz(file.string(), replay::ToNamed(values));
}
void Run(int epochs) {
  const auto root   = paths::OutputDir("models/mae_pretraining"),
             assets = paths::Dataset("vision_transformer");
  replay::CopyAssets(assets, root, {"data.npz", "manifest.json"});
  auto data = replay::ToMap(nn::load_npz((root / "data.npz").string()));
  auto fit = vit::Pool(data, "fit"), validation = mae::Masked(vit::Pool(data, "validation"), 49111),
       test = mae::Masked(vit::Pool(data, "test"), 49113);
  mae::Model model;
  std::cout << "parameters=" << model.parameters.count() << " variables=" << model.parameters.size()
            << std::endl;
  const auto tf        = replay::TfModels("mae_pretraining");
  const bool replay_tf = std::filesystem::exists(tf / "initial_arrays.npz");
  if (replay_tf)
    replay::LoadInto(model.parameters, tf / "initial_arrays.npz");
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
  std::vector<float> ramp(3072);
  for (int i = 0; i < 3072; ++i)
    ramp[i] = i / 3071.f;
  auto image   = nn::from_vector(ramp, {1, 32, 32, 3});
  auto fixture = model.Run(
      mae::Masked({{"images", image}, {"labels", nn::zeros({1}, nn::DType::Int32)}}, 49301));
  auto patches = fixture.at("patches").to_vector<float>();
  for (int p = 0; p < 16; ++p)
    for (int y = 0; y < 8; ++y)
      for (int x = 0; x < 8; ++x)
        for (int c = 0; c < 3; ++c)
          check::Near(
              patches[p * 192 + (y * 8 + x) * 3 + c],
              ramp[(((p / 4) * 8 + y) * 32 + (p % 4) * 8 + x) * 3 + c],
              0,
              "exact RGB patch indexing");
  Save(root / "patch_fixture.npz", {{"images", image}, {"patches", fixture.at("patches")}});
  Save(root / "test_inputs.npz", test);
  Save(root / "validation_inputs.npz", validation);
  nn::BatchOrder order(128, 8, 49107);
  std::ofstream  history(root / "history.tsv"), batches(root / "batches.tsv");
  history << "epoch\tvalidation_mse\n";
  batches << "epoch\tstep\trow\tsource_id\n";
  int    updates = 0;
  vit::A first;
  for (int epoch = 1; epoch <= epochs; ++epoch) {
    order.begin_epoch();
    for (;;) {
      auto rows = order.next();
      if (rows.empty())
        break;
      auto b = mae::Masked(replay::Rows(fit, rows), 49119 + updates);
      Save(root / ("batch" + std::to_string(updates) + "_inputs.npz"), b);
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
    std::cout << "epoch=" << epoch << " validationMaskedMSE=" << out.at("loss").item() << std::endl;
  }
  auto final = model.Arrays(), grad = model.Run(first, true);
  Save(root / "model_arrays.npz", final);
  Save(root / "final_gradients.npz", grad);
  compare(grad, "final_gradients.npz", 1e-4);
  model.parameters.save((root / "final.npz").string());
  mae::Model deployed(false);
  deployed.parameters.load((root / "final.npz").string());
  check::Require(
      replay::SameBits(model.Arrays(), deployed.Arrays()), "MAE model tensors restored bitexact");
  auto before = deployed.Arrays(), out = deployed.Run(test);
  Save(root / "outputs.npz", out);
  compare(out, "outputs.npz", 5e-5);
  auto target_free = test;
  target_free.erase("mask");
  target_free.erase("labels");
  check::Near(
      replay::MaxAbsDiff(deployed.Predict(target_free), out.at("prediction")),
      0,
      0,
      "MAE inference requires no reconstruction mask or labels");
  auto hidden_change = test;
  auto pixels = test.at("images").to_vector<float>(), mask = test.at("mask").to_vector<float>();
  for (int b = 0; b < 32; ++b)
    for (int p = 0; p < 16; ++p)
      if (mask[b * 16 + p])
        for (int y = 0; y < 8; ++y)
          for (int x = 0; x < 8; ++x)
            for (int c = 0; c < 3; ++c) {
              auto i    = ((b * 32 + p / 4 * 8 + y) * 32 + p % 4 * 8 + x) * 3 + c;
              pixels[i] = 1 - pixels[i];
            }
  hidden_change["images"] = nn::from_vector(pixels, test.at("images").shape());
  auto noleak             = deployed.Run(hidden_change);
  check::Near(
      replay::MaxAbsDiff(noleak.at("latent"), out.at("latent")),
      0,
      0,
      "masked pixels never enter encoder");
  check::Near(
      replay::MaxAbsDiff(noleak.at("prediction"), out.at("prediction")),
      0,
      0,
      "masked targets cannot leak into decoder");
  check::Require(
      std::abs(noleak.at("loss").item() - out.at("loss").item()) > 1e-6,
      "masked targets affect reconstruction loss");
  Save(root / "hidden_changed_inputs.npz", hidden_change);
  Save(root / "hidden_changed_outputs.npz", noleak);
  for (int row = 0; row < 4; ++row)
    check::Near(
        replay::MaxAbsDiff(
            deployed.Predict(replay::Rows(test, {row})),
            nn::slice(out.at("prediction"), {row, 0, 0}, {1, 16, 192})),
        0,
        2e-6,
        "MAE batch isolation");
  double fd    = 0;
  int    count = 0;
  for (const auto& name : model.parameters.names()) {
    const auto value  = final.at(name);
    auto       values = value.to_vector<float>();
    for (int i : {0, int(values.size() / 2), int(values.size() - 1)}) {
      const float old = values[i];
      values[i]       = old + .0005f;
      auto plus       = model.Run(first, false, {{name, nn::from_vector(values, value.shape())}})
                      .at("loss")
                      .item();
      values[i]  = old - .0005f;
      auto minus = model.Run(first, false, {{name, nn::from_vector(values, value.shape())}})
                       .at("loss")
                       .item();
      values[i] = old;
      fd = std::max(fd, std::abs((plus - minus) / .001 - grad.at(name).to_vector<float>()[i]));
      ++count;
    }
  }
  check::Require(fd < .003, "native MAE all-parameter finite differences");
  check::Require(
      replay::SameBits(before, deployed.Arrays()), "MAE inference and controls immutable");
  std::ofstream metrics(root / "metrics.json");
  metrics.precision(17);
  metrics << "{\"parameters\":" << model.parameters.count()
          << ",\"variables\":" << model.parameters.size() << ",\"epochs\":" << epochs
          << ",\"updates\":" << updates << ",\"test_mse\":" << out.at("loss").item()
          << ",\"native_fd_count\":" << count << ",\"native_fd_error\":" << fd << "}\n";
  std::cout << "testMaskedMSE=" << out.at("loss").item() << "; " << count << " FD max=" << fd
            << ";visible-only/75pctmask/noLeak/restore/immutable PASS\n";
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
