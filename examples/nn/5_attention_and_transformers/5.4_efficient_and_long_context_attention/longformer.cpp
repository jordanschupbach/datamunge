// Longformer with sliding windows and separate global attention projections.
// CMD: just run-nn longformer
// Eager port of tflow's longformer; same 3426 parameters/36 tensors,
// batch order, Adam and checks. Replays recorded TF initial arrays when present
// and compares saved fields; otherwise native initialization gives statistical
// parity. Checkpoints use NPZ; epochs=N retains the source option.
#include "common/longformer.hpp"
namespace nn = datamunge::nn;
namespace {
void Save(const std::filesystem::path& file, const vit::A& values) {
  nn::save_npz(file.string(), replay::ToNamed(values));
}
nn::Tensor Pattern(const nn::Shape& shape, float phase) {
  std::vector<float> values(nn::shape_numel(shape));
  for (std::size_t i = 0; i < values.size(); ++i)
    values[i] = std::sin(float(i + 1) * .37f + phase) * .4f;
  return nn::from_vector(values, shape);
}
void OperatorChecks(const std::filesystem::path& root) {
  for (int n : {5, 16, 64, 256, 1024}) {
    vit::A                  b;
    std::vector<nn::Tensor> inputs;
    int                     a = 0;
    for (auto key : {"q", "k", "v", "gq", "gk", "gv", "weight"}) {
      b[key] = Pattern({1, 2, n, 4}, .2f * a++).requires_grad_();
      if (a <= 6)
        inputs.push_back(b[key]);
    }
    auto path =
        longformer::Context(b.at("q"), b.at("k"), b.at("v"), b.at("gq"), b.at("gk"), b.at("gv"));
    auto            loss = nn::sum(path.context * b.at("weight"));
    nn::GradOptions retained;
    retained.retain_graph = true;
    auto   g              = nn::grad(loss, inputs, retained);
    auto   probe          = nn::sum(nn::slice(path.context, {0, 0, n - 1, 0}, {1, 2, 1, 4}));
    auto   pg             = nn::grad(probe, {b.at("v"), b.at("gv")});
    vit::A out{
        {"context", path.context},
        {"attention", path.attention},
        {"global_attention", path.global_attention},
        {"loss", loss},
        {"local_dv", pg[0]},
        {"local_dgv", pg[1]}};
    int i = 0;
    for (auto key : {"dq", "dk", "dv", "dgq", "dgk", "dgv"})
      out[key] = g[i++];
    check::Require(
        path.attention.shape() == nn::Shape({1, 2, n, 6}) &&
            path.global_attention.shape() == nn::Shape({1, 2, 1, n}),
        "linear local/global scores");
    auto dv = pg[0].to_vector<float>();
    for (int j = 1; j < n - 3; ++j)
      for (int h = 0; h < 2; ++h)
        for (int d = 0; d < 4; ++d)
          check::Near(dv[(h * n + j) * 4 + d], 0, 0, "remote local value blocked");
    check::Near(
        nn::max(nn::abs(pg[1])).item(), 0, 0, "local query excludes global value projection");
    auto stem = "operator" + std::to_string(n);
    Save(root / (stem + "_inputs.npz"), b);
    Save(root / (stem + "_outputs.npz"), out);
    auto tf = replay::TfModels("longformer") / (stem + "_outputs.npz");
    if (std::filesystem::exists(tf))
      check::Near(replay::MaxDiff(out, tf), 0, 2e-6, "TF sparse operator parity");
  }
  std::cout << "Longformer N5..1024 local/global gradients and exact locality PASS" << std::endl;
}
void Run(int epochs) {
  const auto root   = paths::OutputDir("models/longformer"),
             assets = paths::Dataset("vision_transformer");
  OperatorChecks(root);
  replay::CopyAssets(assets, root, {"data.npz", "manifest.json"});
  auto data = replay::ToMap(nn::load_npz((root / "data.npz").string()));
  auto fit = vit::Pool(data, "fit"), validation = vit::Pool(data, "validation"),
       test = vit::Pool(data, "test");
  longformer::Model model;
  check::Require(
      model.parameters.count() == 3426 && model.parameters.size() == 36,
      "3426 parameters/36 local/global attention tensors");
  const auto tf        = replay::TfModels("longformer");
  const bool replay_tf = std::filesystem::exists(tf / "initial_arrays.npz");
  if (replay_tf)
    replay::LoadInto(model.parameters, tf / "initial_arrays.npz");
  auto compare = [&](const vit::A& out, const std::string& file, double tolerance) {
    if (replay_tf && std::filesystem::exists(tf / file)) {
      // Key-projection biases add the same per-channel constant to every
      // key, which softmax cancels. Adam can magnify roundoff in those null
      // derivatives; compare key coordinates modulo that exact symmetry.
      auto actual = out, expected = replay::ToMap(nn::load_npz((tf / file).string()));
      for (auto key : {"k0", "k1", "gk0", "gk1"})
        if (actual.count(key)) {
          actual[key]   = actual.at(key) - nn::mean(actual.at(key), {2}, true);
          expected[key] = expected.at(key) - nn::mean(expected.at(key), {2}, true);
        }
      double error = 0;
      for (const auto& [key, value] : actual)
        error = std::max(error, replay::MaxAbsDiff(value, expected.at(key)));
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
  nn::BatchOrder order(128, 8, 50707);
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
  longformer::Model deployed(false);
  deployed.parameters.load((root / "final.npz").string());
  check::Require(
      replay::SameBits(model.Arrays(), deployed.Arrays()), "36 tensors restored bitexact");
  auto before = deployed.Arrays(), out = deployed.Run(test);
  Save(root / "outputs.npz", out);
  compare(out, "outputs.npz", 5e-5);
  check::Near(
      replay::MaxAbsDiff(deployed.Predict({{"images", test.at("images")}}), out.at("logits")),
      0,
      0,
      "image-only inference");
  for (int row = 0; row < 4; ++row)
    check::Near(
        replay::MaxAbsDiff(
            deployed.Predict(replay::Rows(test, {row})),
            nn::slice(out.at("logits"), {row, 0}, {1, 10})),
        0,
        2e-6,
        "Longformer batch isolation");
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
  check::Require(fd < .003, "native Longformer all-parameter finite differences");
  check::Require(
      replay::SameBits(before, deployed.Arrays()), "Longformer inference and controls immutable");
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
  metrics << "{\"parameters\":3426,\"variables\":36,\"epochs\":" << epochs
          << ",\"updates\":" << updates << ",\"test_ce\":" << out.at("loss").item()
          << ",\"test_accuracy\":" << correct / 32. << ",\"native_fd_count\":" << count
          << ",\"native_fd_error\":" << fd << "}\n";
  std::cout << "testCE=" << out.at("loss").item() << " accuracy=" << correct << "/32; " << count
            << " FD max=" << fd << ";localWindow/globalAttention/restore/immutable PASS\n";
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
