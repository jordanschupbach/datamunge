// Performer FAVOR+ with fixed orthogonal features and linear attention checks.
// CMD: just run-nn performer
// Eager port of tflow's performer; same 2994 parameters/32 trainables plus two fixed projections,
// batch order, Adam and checks. Replays recorded TF initial arrays when present
// and compares saved fields; otherwise native initialization gives statistical
// parity. Checkpoints use NPZ; epochs=N retains the source option.
#include "common/performer.hpp"
namespace nn = datamunge::nn;
namespace {
void Save(const std::filesystem::path& file, const vit::A& values) {
  nn::save_npz(file.string(), replay::ToNamed(values));
}
void OperatorChecks(const std::filesystem::path& root) {
  auto make = [](int n, float phase) {
    std::vector<float> values(2 * n * 4);
    for (std::size_t i = 0; i < values.size(); ++i)
      values[i] = .4f * std::sin(.37f * float(i + 1) + phase);
    return nn::from_vector(values, {1, 2, n, 4});
  };
  for (int n : {5, 16, 64, 256, 1024}) {
    vit::A inputs{
        {"q", make(n, .1f)},
        {"k", make(n, .3f)},
        {"v", make(n, .7f)},
        {"weight", make(n, .9f)},
        {"omega", performer::Projection(16, 4, 50509)}};
    auto q = inputs.at("q").detach().requires_grad_(), k = inputs.at("k").detach().requires_grad_(),
         v      = inputs.at("v").detach().requires_grad_();
    auto   path = performer::Context(q, k, v, inputs.at("omega"));
    auto   loss = nn::sum(path.context * inputs.at("weight"));
    auto   g    = nn::grad(loss, {q, k, v});
    vit::A out{
        {"context", path.context},
        {"query_features", path.query_features},
        {"key_features", path.key_features},
        {"summary", path.summary},
        {"denominator", path.denominator},
        {"loss", loss},
        {"dq", g[0]},
        {"dk", g[1]},
        {"dv", g[2]}};
    check::Require(
        path.query_features.shape() == nn::Shape({1, 2, n, 16}) &&
            path.summary.shape() == nn::Shape({1, 2, 16, 4}),
        "linear feature/constant summary shapes");
    for (const auto& derivative : g)
      check::Require(
          nn::all(nn::is_finite(derivative)).item<bool>(), "finite FAVOR+ QKV derivatives");
    check::Require(
        nn::min(path.query_features).item() > 0 && nn::min(path.key_features).item() > 0,
        "positive random features");
    const auto prefix = "operator" + std::to_string(n);
    Save(root / (prefix + "_inputs.npz"), inputs);
    Save(root / (prefix + "_outputs.npz"), out);
    const auto reference = replay::TfModels("performer") / (prefix + "_outputs.npz");
    if (n <= 16 && std::filesystem::exists(reference))
      check::Near(
          replay::MaxDiff(out, reference), 0, 2e-6, "TF FAVOR+ operator values/QKV derivatives");
  }
  std::cout << "FAVOR+ QKV derivatives,N5..1024 linear feature/constant summary fixtures PASS\n";
}
void Run(int epochs) {
  const auto root   = paths::OutputDir("models/performer"),
             assets = paths::Dataset("vision_transformer");
  OperatorChecks(root);
  replay::CopyAssets(assets, root, {"data.npz", "manifest.json"});
  auto data = replay::ToMap(nn::load_npz((root / "data.npz").string()));
  auto fit = vit::Pool(data, "fit"), validation = vit::Pool(data, "validation"),
       test = vit::Pool(data, "test");
  performer::Model model;
  check::Require(
      model.parameters.count() == 2994 && model.parameters.size() == 32,
      "2994 parameters/32 patch CLS two-block tensors");
  const auto tf        = replay::TfModels("performer");
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
  nn::BatchOrder order(128, 8, 50507);
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
  model.Save(root / "final.npz");
  performer::Model deployed(false);
  deployed.Load(root / "final.npz");
  check::Require(
      replay::SameBits(model.Arrays(), deployed.Arrays()),
      "all34 learned/fixed tensors restored bitexact");
  auto before = deployed.Arrays(), out = deployed.Run(test);
  Save(root / "outputs.npz", out);
  compare(out, "outputs.npz", 5e-5);
  check::Near(
      replay::MaxAbsDiff(deployed.Predict({{"images", test.at("images")}}), out.at("logits")),
      0,
      0,
      "image-only inference");
  const auto initial_arrays = replay::ToMap(nn::load_npz((root / "initial_arrays.npz").string()));
  for (const auto& [name, value] : final)
    if (name.find("/omega") != std::string::npos)
      check::Near(
          replay::MaxAbsDiff(value, initial_arrays.at(name)),
          0,
          0,
          "fixed projection survives every optimizer update");
  for (int row = 0; row < 4; ++row)
    check::Near(
        replay::MaxAbsDiff(
            deployed.Predict(replay::Rows(test, {row})),
            nn::slice(out.at("logits"), {row, 0}, {1, 10})),
        0,
        2e-6,
        "Performer batch isolation");
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
  check::Require(fd < .003, "native Performer all-parameter finite differences");
  check::Require(
      replay::SameBits(before, deployed.Arrays()), "Performer inference and controls immutable");
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
          << ",\"test_accuracy\":" << correct / 32. << ",\"native_fd_count\":" << count
          << ",\"native_fd_error\":" << fd << "}\n";
  std::cout << "testCE=" << out.at("loss").item() << " accuracy=" << correct << "/32; " << count
            << " FD max=" << fd << ";FAVOR+/fixedORF/linearAttention/restore/immutable PASS\n";
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
