// Linformer with learned sequence projections and compressed attention checks.
// CMD: just run-nn linformer
// Eager port of tflow's linformer; same 3266 parameters/36 tensors,
// batch order, Adam and checks. Replays recorded TF initial arrays when present
// and compares saved fields; otherwise native initialization gives statistical
// parity. Checkpoints use NPZ; epochs=N retains the source option.
#include "common/linformer.hpp"
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
vit::A Operator(vit::A b) {
  std::vector<nn::Tensor> inputs;
  for (const auto key : {"q", "k", "v", "E", "F"}) {
    b[key] = b.at(key).detach().clone().requires_grad_();
    inputs.push_back(b.at(key));
  }
  auto path = linformer::Context(b.at("q"), b.at("k"), b.at("v"), b.at("E"), b.at("F"));
  auto g    = nn::grad(nn::sum(path.context * b.at("weight")), inputs);
  return {
      {"context", path.context.detach()},
      {"attention", path.attention.detach()},
      {"dq", g[0]},
      {"dk", g[1]},
      {"dv", g[2]},
      {"dE", g[3]},
      {"dF", g[4]}};
}
void OperatorChecks(const std::filesystem::path& root) {
  vit::A b{
      {"q", Pattern({1, 2, 5, 4}, .1f)},
      {"k", Pattern({1, 2, 5, 4}, .3f)},
      {"v", Pattern({1, 2, 5, 4}, .7f)},
      {"weight", Pattern({1, 2, 5, 4}, .9f)},
      {"E", nn::eye(5)},
      {"F", nn::eye(5)}};
  auto identity = Operator(b);
  auto q = b.at("q").detach().requires_grad_(), k = b.at("k").detach().requires_grad_(),
       v     = b.at("v").detach().requires_grad_();
  auto dense = nn::matmul(nn::softmax(nn::matmul(q, k, false, true) * .5, -1), v);
  auto full  = nn::grad(nn::sum(dense * b.at("weight")), {q, k, v});
  check::Near(
      replay::MaxAbsDiff(identity.at("context"), dense),
      0,
      2e-6,
      "identity E/F recovers full attention");
  for (int i = 0; i < 3; ++i)
    check::Near(
        replay::MaxAbsDiff(
            identity.at(
                i == 0   ? "dq"
                : i == 1 ? "dk"
                         : "dv"),
            full[i]),
        0,
        2e-6,
        "identity Q/K/V derivatives match full attention");
  auto identity_out = identity;
  identity_out.erase("attention");
  identity_out["full_context"] = dense;
  identity_out["full_dq"]      = full[0];
  identity_out["full_dk"]      = full[1];
  identity_out["full_dv"]      = full[2];
  Save(root / "identity_inputs.npz", b);
  Save(root / "identity_outputs.npz", identity_out);
  b["E"]     = Pattern({5, 3}, .4f);
  b["F"]     = Pattern({5, 3}, .8f);
  auto rank3 = Operator(b);
  Save(root / "operator_inputs.npz", b);
  Save(root / "operator_outputs.npz", rank3);
  const auto tf = replay::TfModels("linformer");
  if (std::filesystem::exists(tf / "operator_outputs.npz"))
    check::Near(
        replay::MaxDiff(rank3, tf / "operator_outputs.npz"),
        0,
        2e-6,
        "TF rank3 forward/all operator derivatives");
  b["F"]    = nn::zeros({5, 3});
  auto zero = Operator(b);
  for (const auto key : {"context", "dq", "dk", "dv", "dE"})
    check::Near(
        nn::max(nn::abs(zero.at(key))).item(), 0, 0, "zeroF blocks context/Q/K/V/E derivatives");
  check::Require(
      nn::sum(nn::abs(zero.at("dF"))).item() > 1e-6, "F receives value-path gradient at zero");
  std::ofstream scaling(root / "scaling.tsv");
  scaling << "length\trank\theads\tactual_score_elements\ttheoretical_full_score_elements\n";
  nn::NoGradGuard guard;
  for (int n : {16, 64, 256, 1024}) {
    vit::A a{
        {"q", Pattern({1, 2, n, 4}, .1f)},
        {"k", Pattern({1, 2, n, 4}, .3f)},
        {"v", Pattern({1, 2, n, 4}, .7f)},
        {"weight", Pattern({1, 2, n, 4}, .9f)},
        {"E", Pattern({n, 4}, .4f)},
        {"F", Pattern({n, 4}, .8f)}};
    auto out = linformer::Context(a.at("q"), a.at("k"), a.at("v"), a.at("E"), a.at("F"));
    check::Require(
        out.attention.shape() == nn::Shape({1, 2, n, 4}) &&
            out.context.shape() == nn::Shape({1, 2, n, 4}),
        "linear score/context shapes at rank4");
    scaling << n << "\t4\t2\t" << out.attention.numel() << '\t' << 2 * n * n << '\n';
    Save(root / ("scaling" + std::to_string(n) + "_inputs.npz"), a);
    Save(
        root / ("scaling" + std::to_string(n) + "_outputs.npz"),
        {{"attention", out.attention}, {"context", out.context}});
  }
  std::cout << "identity/full QKV gradients,rank3/zeroF,N16..1024 fixed-rank shapes PASS\n";
}
void Run(int epochs) {
  const auto root   = paths::OutputDir("models/linformer"),
             assets = paths::Dataset("vision_transformer");
  OperatorChecks(root);
  replay::CopyAssets(assets, root, {"data.npz", "manifest.json"});
  auto data = replay::ToMap(nn::load_npz((root / "data.npz").string()));
  auto fit = vit::Pool(data, "fit"), validation = vit::Pool(data, "validation"),
       test = vit::Pool(data, "test");
  linformer::Model model;
  check::Require(
      model.parameters.count() == 3266 && model.parameters.size() == 36,
      "3266 parameters/36 length-projection/attention tensors");
  const auto tf        = replay::TfModels("linformer");
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
  nn::BatchOrder order(128, 8, 50307);
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
  linformer::Model deployed(false);
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
  vit::A zero_e, zero_f;
  for (const auto& [name, value] : final)
    if (name.size() > 2 &&
        (name.substr(name.size() - 2) == "/E" || name.substr(name.size() - 2) == "/F"))
      (name.back() == 'E' ? zero_e : zero_f)[name] = nn::zeros(value.shape());
  auto noe = deployed.Run(test, false, zero_e), nof = deployed.Run(test, false, zero_f);
  Save(root / "zero_E_outputs.npz", noe);
  Save(root / "zero_F_outputs.npz", nof);
  const double e_effect = replay::MaxAbsDiff(noe.at("logits"), out.at("logits")),
               f_effect = replay::MaxAbsDiff(nof.at("logits"), out.at("logits"));
  check::Require(
      e_effect > 1e-6 && f_effect > 1e-6, "both learned key/value projections affect predictions");
  for (int layer = 0; layer < 2; ++layer) {
    auto attention = noe.at("attention" + std::to_string(layer));
    check::Require(
        attention.shape() == nn::Shape({32, 2, 17, 4}), "17query by4compressedkey scores");
    for (float value : attention.to_vector<float>())
      check::Near(value, .25, 0, "zeroE makes four basis scores equal");
  }
  const auto nof_logits = nof.at("logits").to_vector<float>();
  for (int row = 1; row < 32; ++row)
    for (int c = 0; c < 10; ++c)
      check::Near(nof_logits[row * 10 + c], nof_logits[c], 0, "zeroF cuts image content from CLS");
  check::Near(
      nn::max(nn::abs(nof.at("input_gradient"))).item(), 0, 0, "zeroF pixel derivative exact zero");
  for (int row = 0; row < 4; ++row)
    check::Near(
        replay::MaxAbsDiff(
            deployed.Predict(replay::Rows(test, {row})),
            nn::slice(out.at("logits"), {row, 0}, {1, 10})),
        0,
        2e-6,
        "Linformer batch isolation");
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
  check::Require(fd < .003, "native Linformer all-parameter finite differences");
  check::Require(
      replay::SameBits(before, deployed.Arrays()), "Linformer inference and controls immutable");
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
  metrics << "{\"parameters\":3266,\"variables\":36,\"epochs\":" << epochs
          << ",\"updates\":" << updates << ",\"test_ce\":" << out.at("loss").item()
          << ",\"test_accuracy\":" << correct / 32. << ",\"zero_E_effect\":" << e_effect
          << ",\"zero_F_effect\":" << f_effect << ",\"native_fd_count\":" << count
          << ",\"native_fd_error\":" << fd << "}\n";
  std::cout << "testCE=" << out.at("loss").item() << " accuracy=" << correct << "/32; " << count
            << " FD max=" << fd << ";sequenceProjection/E/F/compression/restore/immutable PASS\n";
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
