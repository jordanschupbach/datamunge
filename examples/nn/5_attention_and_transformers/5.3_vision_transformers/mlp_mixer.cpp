// Attention-free MLP-Mixer on real CIFAR pixels, with spatial token MLPs.
// CMD: just run-nn mlp_mixer
// Eager port of tflow's mlp_mixer; same 2834 parameters/30 tensors,
// batch order, Adam and checks. Replays recorded TF initial arrays when present
// and compares saved fields; otherwise native initialization gives statistical
// parity. Checkpoints use NPZ; epochs=N retains the source option.
#include "common/mlp_mixer.hpp"
namespace nn = datamunge::nn;
namespace {
void Save(const std::filesystem::path& file, const vit::A& values) {
  nn::save_npz(file.string(), replay::ToNamed(values));
}
void Run(int epochs) {
  const auto root   = paths::OutputDir("models/mlp_mixer"),
             assets = paths::Dataset("vision_transformer");
  replay::CopyAssets(assets, root, {"data.npz", "manifest.json"});
  auto data = replay::ToMap(nn::load_npz((root / "data.npz").string()));
  auto fit = vit::Pool(data, "fit"), validation = vit::Pool(data, "validation"),
       test = vit::Pool(data, "test");
  mixer::Model model;
  check::Require(
      model.parameters.count() == 2834 && model.parameters.size() == 30,
      "2834 parameters/32 patch token/channel two-block tensors");
  const auto tf        = replay::TfModels("mlp_mixer");
  const bool replay_tf = std::filesystem::exists(tf / "initial_arrays.npz");
  if (replay_tf)
    replay::LoadInto(model.parameters, tf / "initial_arrays.npz");
  auto compare = [&](const vit::A& out, const std::string& file, double tolerance) {
    if (replay_tf && std::filesystem::exists(tf / file)) {
      const auto saved = replay::ToMap(nn::load_npz((tf / file).string()));
      double     error = 0, raw_shift = 0;
      for (const auto& [key, value] : out) {
        const auto& reference = saved.at(key);
        const bool  trained =
            file.rfind("epoch", 0) == 0 || file == "outputs.npz" || file == "final_gradients.npz";
        if (trained && (key.rfind("state", 0) == 0 || key.rfind("token_state", 0) == 0)) {
          // token2/b's channel-constant residual shifts are removed by the
          // following LN. Compare states in the same zero-channel-mean gauge.
          raw_shift = std::max(raw_shift, replay::MaxAbsDiff(value, reference));
          error     = std::max(
              error,
              replay::MaxAbsDiff(
                  value - nn::mean(value, {-1}, true),
                  reference - nn::mean(reference, {-1}, true)));
        } else
          error = std::max(error, replay::MaxAbsDiff(value, reference));
      }
      if (raw_shift)
        std::cout << "removed channel-constant residual shift=" << raw_shift << std::endl;
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
  nn::BatchOrder order(128, 8, 49507);
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
        // token2/b adds a channel-constant shift per token; subsequent layer
        // norms remove it. Adam magnifies kernel-dependent ~1e-9 roundoff
        // in these mathematically zero derivatives (epsilon=1e-8).
        if (replay_tf && std::filesystem::exists(tf / "first_after.npz")) {
          const auto saved = replay::ToMap(nn::load_npz((tf / "first_after.npz").string()));
          for (const auto& [key, value] : model.Arrays()) {
            const bool null_bias = key == "block0/token2/b" || key == "block1/token2/b";
            check::Near(
                replay::MaxAbsDiff(value, saved.at(key)),
                0,
                null_bias ? .0005 : 5e-5,
                "TF first Adam update " + key);
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
  mixer::Model deployed(false);
  deployed.parameters.load((root / "final.npz").string());
  check::Require(
      replay::SameBits(model.Arrays(), deployed.Arrays()), "30 tensors restored bitexact");
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
  check::Require(
      position_effect > 1e-6,
      "token mixing distinguishes spatial order without position embeddings");
  vit::A zero;
  for (const auto& [name, value] : final)
    if (name.find("/token2/") != std::string::npos)
      zero[name] = nn::zeros(value.shape());
  auto nomix = deployed.Run(test, false, zero), nomixperm = deployed.Run(shuffled, false, zero);
  const double no_position_error = replay::MaxAbsDiff(nomix.at("logits"), nomixperm.at("logits"));
  check::Require(
      no_position_error < 2e-6, "without token mixing pooled channel MLP is permutation invariant");
  Save(root / "no_token_mixing.npz", nomix);
  Save(root / "no_token_mixing_permuted.npz", nomixperm);
  const double paired_error = replay::MaxAbsDiff(nomix.at("logits"), out.at("logits"));
  check::Require(paired_error > 1e-6, "token branch affects actual predictions");
  for (int row = 0; row < 4; ++row)
    check::Near(
        replay::MaxAbsDiff(
            deployed.Predict(replay::Rows(test, {row})),
            nn::slice(out.at("logits"), {row, 0}, {1, 10})),
        0,
        2e-6,
        "MLP-Mixer batch isolation");
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
  check::Require(fd < .003, "native MLP-Mixer all-parameter finite differences");
  check::Require(
      replay::SameBits(before, deployed.Arrays()), "MLP-Mixer inference and controls immutable");
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
  metrics << "{\"parameters\":2834,\"variables\":30,\"epochs\":" << epochs
          << ",\"updates\":" << updates << ",\"test_ce\":" << out.at("loss").item()
          << ",\"test_accuracy\":" << correct / 32. << ",\"token_order_effect\":" << position_effect
          << ",\"token_branch_effect\":" << paired_error
          << ",\"no_token_mixing_permutation_error\":" << no_position_error
          << ",\"native_fd_count\":" << count << ",\"native_fd_error\":" << fd << "}\n";
  std::cout << "testCE=" << out.at("loss").item() << " accuracy=" << correct << "/32; " << count
            << " FD max=" << fd << ";patch/token/channel/permutation/restore/immutable PASS\n";
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
