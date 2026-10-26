// Scratch ViT on actual CIFAR pixels; patch/position and numerical checks.
// CMD: just run-nn vision_transformer
// Eager port of tflow's vision_transformer; same 2994 parameters/32 tensors,
// batch order, Adam and checks. Replays recorded TF initial arrays when present
// and compares saved fields; otherwise native initialization gives statistical
// parity. Checkpoints use NPZ; epochs=N retains the source option.
#include "common/vision_transformer.hpp"
namespace nn = datamunge::nn;
namespace {
void Save(const std::filesystem::path& file, const vit::A& values) {
  nn::save_npz(file.string(), replay::ToNamed(values));
}
void Run(int epochs) {
  const auto root   = paths::OutputDir("models/vision_transformer"),
             assets = paths::Dataset("vision_transformer");
  replay::CopyAssets(assets, root, {"data.npz", "manifest.json"});
  auto data = replay::ToMap(nn::load_npz((root / "data.npz").string()));
  auto fit = vit::Pool(data, "fit"), validation = vit::Pool(data, "validation"),
       test = vit::Pool(data, "test");
  vit::Model model;
  check::Require(
      model.parameters.count() == 2994 && model.parameters.size() == 32,
      "2994 parameters/32 patch CLS two-block tensors");
  const auto tf        = replay::TfModels("vision_transformer");
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
  nn::BatchOrder order(128, 8, 48307);
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
        // Softmax is invariant to a shared key bias. Its true derivative is zero,
        // but ~1e-9 reduction noise differs between kernels and Adam amplifies
        // it relative to epsilon=1e-8. Check meaningful updates tightly.
        const auto reference = tf / "first_after.npz";
        if (replay_tf && std::filesystem::exists(reference)) {
          const auto saved  = replay::ToMap(nn::load_npz(reference.string()));
          const auto actual = model.Arrays();
          for (const auto& [key, value] : actual) {
            const double error    = replay::MaxAbsDiff(value, saved.at(key));
            const bool   qkv_bias = key == "block0/qkv/b" || key == "block1/qkv/b";
            check::Near(error, 0, qkv_bias ? .0005 : 5e-5, "TF first Adam update " + key);
            if (qkv_bias) {
              const auto a = value.to_vector<float>(), b = saved.at(key).to_vector<float>();
              for (int i = 0; i < 24; ++i)
                if (i < 8 || i >= 16)
                  check::Near(a[i], b[i], 5e-5, "TF query/value bias update");
            }
          }
        }
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
  vit::Model deployed(false);
  deployed.parameters.load((root / "final.npz").string());
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
  check::Require(position_effect > 1e-6, "learned positions make patch order observable");
  auto positions = final.at("position").to_vector<float>(), permuted_positions = positions;
  for (int p = 1; p < 17; ++p)
    for (int d = 0; d < 8; ++d)
      permuted_positions[p * 8 + d] = positions[(17 - p) * 8 + d];
  auto   position = nn::from_vector(permuted_positions, {1, 17, 8}), zero = nn::zeros({1, 17, 8});
  auto   aligned      = deployed.Run(shuffled, false, {{"position", position}});
  double paired_error = replay::MaxAbsDiff(out.at("logits"), aligned.at("logits"));
  check::Require(paired_error < 5e-6, "CLS invariance with paired patch/position permutation");
  const auto a = aligned.at("hidden").to_vector<float>(), h = out.at("hidden").to_vector<float>();
  for (int b = 0; b < 32; ++b)
    for (int p = 0; p < 17; ++p)
      for (int d = 0; d < 8; ++d)
        check::Near(
            a[(b * 17 + p) * 8 + d],
            h[(b * 17 + (p ? 17 - p : 0)) * 8 + d],
            5e-6,
            "patch-state permutation covariance");
  Save(root / "paired_position.npz", {{"position", position}});
  Save(root / "paired_outputs.npz", aligned);
  auto nopos               = deployed.Run(test, false, {{"position", zero}}),
       noposperm           = deployed.Run(shuffled, false, {{"position", zero}});
  double no_position_error = replay::MaxAbsDiff(nopos.at("logits"), noposperm.at("logits"));
  check::Require(no_position_error < 5e-6, "without positions CLS depends on patch set");
  Save(root / "no_position_outputs.npz", nopos);
  Save(root / "no_position_permuted_outputs.npz", noposperm);
  for (int row = 0; row < 4; ++row)
    check::Near(
        replay::MaxAbsDiff(
            deployed.Predict(replay::Rows(test, {row})),
            nn::slice(out.at("logits"), {row, 0}, {1, 10})),
        0,
        2e-6,
        "ViT batch isolation");
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
  check::Require(fd < .003, "native ViT all-parameter finite differences");
  check::Require(
      replay::SameBits(before, deployed.Arrays()), "ViT inference and controls immutable");
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
  metrics << "{\"parameters\":2994,\"variables\":32,\"epochs\":" << epochs
          << ",\"updates\":" << updates << ",\"test_ce\":" << out.at("loss").item()
          << ",\"test_accuracy\":" << correct / 32.
          << ",\"position_intervention\":" << position_effect
          << ",\"paired_permutation_error\":" << paired_error
          << ",\"no_position_permutation_error\":" << no_position_error
          << ",\"native_fd_count\":" << count << ",\"native_fd_error\":" << fd << "}\n";
  std::cout << "testCE=" << out.at("loss").item() << " accuracy=" << correct << "/32; " << count
            << " FD max=" << fd << ";patch/position/CLS/permutation/restore/immutable PASS\n";
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
