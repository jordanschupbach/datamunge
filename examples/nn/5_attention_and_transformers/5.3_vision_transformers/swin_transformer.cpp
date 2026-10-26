// Hierarchical Swin on real CIFAR pixels; shifted masks and numerical checks.
// CMD: just run-nn swin_transformer
// Eager port of tflow's swin_transformer; same 8058 parameters/63 tensors,
// batch order, Adam and checks. Replays recorded TF initial arrays when present
// and compares saved fields; otherwise native initialization gives statistical
// parity. Checkpoints use NPZ; epochs=N retains the source option.
#include "common/swin_transformer.hpp"
namespace nn = datamunge::nn;
namespace {
void Save(const std::filesystem::path& file, const vit::A& values) {
  nn::save_npz(file.string(), replay::ToNamed(values));
}
void Run(int epochs) {
  const auto root   = paths::OutputDir("models/swin_transformer"),
             assets = paths::Dataset("vision_transformer");
  replay::CopyAssets(assets, root, {"data.npz", "manifest.json"});
  auto data = replay::ToMap(nn::load_npz((root / "data.npz").string()));
  auto fit = vit::Pool(data, "fit"), validation = vit::Pool(data, "validation"),
       test = vit::Pool(data, "test");
  swin::Model model;
  check::Require(
      model.parameters.count() == 8058 && model.parameters.size() == 63,
      "8058 parameters/63 hierarchical shifted-window tensors");
  const auto tf        = replay::TfModels("swin_transformer");
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
  auto fixture = model.Run({{"images", image}, {"labels", nn::zeros({1}, nn::DType::Int32)}});
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
  nn::BatchOrder order(128, 8, 48707);
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
  compare(grad, "final_gradients.npz", 1e-4);
  model.parameters.save((root / "final.npz").string());
  swin::Model deployed(false);
  deployed.parameters.load((root / "final.npz").string());
  check::Require(
      replay::SameBits(model.Arrays(), deployed.Arrays()), "63 tensors restored bitexact");
  auto before = deployed.Arrays(), out = deployed.Run(test);
  Save(root / "outputs.npz", out);
  compare(out, "outputs.npz", 5e-5);
  check::Near(
      replay::MaxAbsDiff(deployed.Predict({{"images", test.at("images")}}), out.at("logits")),
      0,
      0,
      "image-only inference");
  for (int block = 0; block < 4; ++block) {
    const auto mask      = out.at("mask" + std::to_string(block)).to_vector<float>();
    const auto attention = out.at("attention" + std::to_string(block)).to_vector<float>();
    const int  nw        = block < 2 ? 4 : 1;
    int        blocked = 0, allowed = 0;
    for (int w = 0; w < nw; ++w)
      for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
          const float m = mask[w * 16 + i * 4 + j];
          if (m < 0)
            ++blocked;
          else
            ++allowed;
          for (int b = 0; b < 32; ++b)
            for (int head = 0; head < 2; ++head) {
              const float a = attention[((b * nw + w) * 2 + head) * 16 + i * 4 + j];
              check::Require(
                  m < 0 ? a < 1e-35 : a > 0, "mask forbids cyclic wrap but retains real neighbors");
            }
        }
        check::Near(mask[w * 16 + i * 4 + i], 0, 0, "self keys never masked");
      }
    check::Require(
        block == 1 ? (blocked == 28 && allowed == 36) : blocked == 0,
        "exact shifted mask pair counts");
  }
  auto control = deployed.Run(test, false, {{"mask1", nn::zeros({1, 4, 1, 4, 4})}});
  check::Require(
      replay::MaxAbsDiff(control.at("logits"), out.at("logits")) > 1e-7,
      "shift mask affects actual predictions");
  Save(
      root / "unmasked_outputs.npz",
      {{"logits", control.at("logits")},
       {"loss", control.at("loss")},
       {"input_gradient", control.at("input_gradient")}});
  vit::A zero_bias;
  for (const auto& [name, value] : final)
    if (name.find("relative_bias") != std::string::npos)
      zero_bias[name] = nn::zeros(value.shape());
  auto nobias = deployed.Run(test, false, zero_bias);
  check::Require(
      replay::MaxAbsDiff(nobias.at("logits"), out.at("logits")) > 1e-7,
      "relative biases affect actual predictions");
  Save(root / "zero_bias_outputs.npz", nobias);
  for (int row = 0; row < 4; ++row)
    check::Near(
        replay::MaxAbsDiff(
            deployed.Predict(replay::Rows(test, {row})),
            nn::slice(out.at("logits"), {row, 0}, {1, 10})),
        0,
        2e-6,
        "Swin batch isolation");
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
  check::Require(fd < .003, "native Swin all-parameter finite differences");
  check::Require(
      replay::SameBits(before, deployed.Arrays()), "Swin inference and controls immutable");
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
  metrics << "{\"parameters\":8058,\"variables\":63,\"epochs\":" << epochs
          << ",\"updates\":" << updates << ",\"test_ce\":" << out.at("loss").item()
          << ",\"test_accuracy\":" << correct / 32. << ",\"native_fd_count\":" << count
          << ",\"native_fd_error\":" << fd << "}\n";
  std::cout << "testCE=" << out.at("loss").item() << " accuracy=" << correct << "/32; " << count
            << " FD max=" << fd << ";patch/shifted-masks/merging/restore/immutable PASS\n";
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
