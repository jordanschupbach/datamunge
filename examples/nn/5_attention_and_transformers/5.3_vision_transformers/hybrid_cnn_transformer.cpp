// Local learned CNN features feed global relative-position Transformer layers.
// CMD: just run-nn hybrid_cnn_transformer
// Eager port of tflow's hybrid_cnn_transformer; same 1910 parameters/34 tensors,
// batch order, Adam and checks. Replays recorded TF initial arrays when present
// and compares saved fields; otherwise native initialization gives statistical
// parity. Checkpoints use NPZ; epochs=N retains the source option.
#include "common/hybrid_cnn_transformer.hpp"
namespace nn = datamunge::nn;
namespace {
void Save(const std::filesystem::path& file, const vit::A& values) {
  nn::save_npz(file.string(), replay::ToNamed(values));
}
void Run(int epochs) {
  const auto root   = paths::OutputDir("models/hybrid_cnn_transformer"),
             assets = paths::Dataset("vision_transformer");
  replay::CopyAssets(assets, root, {"data.npz", "manifest.json"});
  auto data = replay::ToMap(nn::load_npz((root / "data.npz").string()));
  auto fit = vit::Pool(data, "fit"), validation = vit::Pool(data, "validation"),
       test = vit::Pool(data, "test");
  hybrid::Model model;
  check::Require(
      model.parameters.count() == 1910 && model.parameters.size() == 34,
      "1910 parameters/32 patch CLS two-block tensors");
  const auto tf        = replay::TfModels("hybrid_cnn_transformer");
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
  nn::BatchOrder order(128, 8, 49907);
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
  hybrid::Model deployed(false);
  deployed.parameters.load((root / "final.npz").string());
  check::Require(
      replay::SameBits(model.Arrays(), deployed.Arrays()), "34 tensors restored bitexact");
  auto before = deployed.Arrays(), out = deployed.Run(test);
  Save(root / "outputs.npz", out);
  compare(out, "outputs.npz", 5e-5);
  check::Near(
      replay::MaxAbsDiff(deployed.Predict({{"images", test.at("images")}}), out.at("logits")),
      0,
      0,
      "image-only inference");
  vit::A no_attention, zero_bias;
  for (const auto& [name, value] : final) {
    if (name.find("/out/") != std::string::npos)
      no_attention[name] = nn::zeros(value.shape());
    if (name.find("/relative_bias") != std::string::npos)
      zero_bias[name] = nn::zeros(value.shape());
  }
  auto local  = deployed.Run(test, false, no_attention),
       nobias = deployed.Run(test, false, zero_bias);
  Save(root / "no_attention.npz", local);
  Save(root / "zero_bias.npz", nobias);
  double position_effect = replay::MaxAbsDiff(local.at("logits"), out.at("logits"));
  double paired_error    = replay::MaxAbsDiff(nobias.at("logits"), out.at("logits"));
  check::Require(
      position_effect > 1e-6 && paired_error > 1e-7,
      "global attention and learned relative geometry affect predictions");
  auto changed = test;
  auto pixels  = test.at("images").to_vector<float>();
  for (int b = 0; b < 32; ++b)
    for (int y = 24; y < 32; ++y)
      for (int x = 24; x < 32; ++x)
        for (int c = 0; c < 3; ++c) {
          auto i    = ((b * 32 + y) * 32 + x) * 3 + c;
          pixels[i] = 1 - pixels[i];
        }
  changed["images"] = nn::from_vector(pixels, test.at("images").shape());
  auto moved = deployed.Run(changed), localmoved = deployed.Run(changed, false, no_attention);
  auto first_token = [](const nn::Tensor& x) { return nn::slice(x, {0, 0, 0}, {x.size(0), 1, 8}); };
  check::Near(
      replay::MaxAbsDiff(first_token(localmoved.at("hidden")), first_token(local.at("hidden"))),
      0,
      0,
      "CNN-only first token cannot see opposite corner");
  double no_position_error =
      replay::MaxAbsDiff(first_token(moved.at("hidden")), first_token(out.at("hidden")));
  check::Require(
      no_position_error > 1e-6, "global attention carries remote CNN features into first token");
  Save(root / "remote_inputs.npz", changed);
  Save(root / "remote_outputs.npz", moved);
  Save(root / "remote_local_outputs.npz", localmoved);
  for (int row = 0; row < 4; ++row)
    check::Near(
        replay::MaxAbsDiff(
            deployed.Predict(replay::Rows(test, {row})),
            nn::slice(out.at("logits"), {row, 0}, {1, 10})),
        0,
        2e-6,
        "hybrid CNN-transformer batch isolation");
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
  check::Require(fd < .003, "native hybrid CNN-transformer all-parameter finite differences");
  check::Require(
      replay::SameBits(before, deployed.Arrays()),
      "hybrid CNN-transformer inference and controls immutable");
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
  metrics << "{\"parameters\":1910,\"variables\":34,\"epochs\":" << epochs
          << ",\"updates\":" << updates << ",\"test_ce\":" << out.at("loss").item()
          << ",\"test_accuracy\":" << correct / 32. << ",\"attention_effect\":" << position_effect
          << ",\"relative_bias_effect\":" << paired_error
          << ",\"remote_attention_effect\":" << no_position_error
          << ",\"native_fd_count\":" << count << ",\"native_fd_error\":" << fd << "}\n";
  std::cout << "testCE=" << out.at("loss").item() << " accuracy=" << correct << "/32; " << count
            << " FD max=" << fd
            << ";CNN/attention/relative-position/remote-context/restore/immutable PASS\n";
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
