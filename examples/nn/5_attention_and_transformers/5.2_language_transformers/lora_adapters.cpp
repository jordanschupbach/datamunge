// LoRA and bottleneck adapters with genuinely frozen pretrained BERT weights.
// CMD: just run-nn lora_adapters
// Eager port: separate task optimizer, full/delta NPZ restoration, all source
// freeze, merge, padding, module-intervention and finite-difference controls.
#include <fstream>

#include "common/peft_bert.hpp"
namespace nn = datamunge::nn;
namespace {
using A     = peft::A;
using Model = peft::Model;
void Save(const std::filesystem::path& path, const A& a) {
  nn::save_npz(path.string(), replay::ToNamed(a));
}
A Pool(const A& data, const std::string& split) {
  A b;
  for (auto k : {"input", "lengths", "labels", "source_ids"})
    b[k] = data.at(split + "_" + k);
  return b;
}
void Task(const std::string& mode, const A& data, int epochs, const std::filesystem::path& root) {
  std::filesystem::create_directories(root);
  Model model(mode);
  check::Require(
      model.base.count() == 2216 && model.base.size() == 21,
      "2216 frozen base parameters/21 tensors");
  int expected = mode == "lora" ? 82 : mode == "adapter" ? 102 : 18;
  check::Require(model.task.count() == expected, "exact trainable parameter count");
  Save(root / "fresh_base.npz", model.Arrays(true));
  auto pretrained = replay::TfModels("bert_mlm") / "model_arrays.npz";
  std::filesystem::copy_file(
      pretrained,
      root / "pretrained_arrays.npz",
      std::filesystem::copy_options::overwrite_existing);
  model.Import(replay::ToMap(nn::load_npz(pretrained.string())));
  auto tf     = replay::TfModels("lora_adapters") / mode;
  bool parity = std::filesystem::exists(tf / "initial_task.npz");
  if (parity) {
    model.Import(replay::ToMap(nn::load_npz((tf / "frozen_base.npz").string())));
    model.Import(replay::ToMap(nn::load_npz((tf / "initial_task.npz").string())), false);
  }
  auto compare = [&](const A& out, const std::string& file) {
    if (parity && std::filesystem::exists(tf / file)) {
      double e = replay::MaxDiff(out, tf / file);
      std::cout << mode << " TF " << file << " difference=" << e << std::endl;
      check::Near(e, 0, 5e-5, "TF PEFT parity " + file);
    }
  };
  auto base = model.Arrays(true), initial = model.Arrays();
  Save(root / "frozen_base.npz", base);
  Save(root / "initial_task.npz", initial);
  auto fit = Pool(data, "fit"), validation = Pool(data, "validation"), test = Pool(data, "test");
  auto init = model.Run(test);
  Save(root / "initial_outputs.npz", init);
  compare(init, "initial_outputs.npz");
  nn::BatchOrder order(64, 8, 47911);
  std::ofstream  batches(root / "batches.tsv"), history(root / "history.tsv");
  batches << "epoch\tstep\trow\tsource_id\n";
  history << "epoch\tvalidation_ce\n";
  int updates = 0;
  A   first;
  for (int e = 1; e <= epochs; ++e) {
    order.begin_epoch();
    for (;;) {
      auto rows = order.next();
      if (rows.empty())
        break;
      auto b = bert::Rows(fit, rows);
      if (!updates) {
        first = b;
        Save(root / "first_batch.npz", b);
        auto g = model.Run(b, true);
        Save(root / "initial_gradients.npz", g);
        compare(g, "initial_gradients.npz");
      }
      model.Update(b);
      check::Require(
          replay::SameBits(base, model.Arrays(true)),
          "every base tensor bitexact after EACH Adam step");
      if (!updates)
        Save(root / "first_after.npz", model.Arrays());
      auto ids = fit.at("source_ids").to_vector<int>();
      for (auto row : rows)
        batches << e << '\t' << updates << '\t' << row << '\t' << ids[row] << '\n';
      ++updates;
    }
    auto out = model.Run(validation);
    history << e << '\t' << out.at("loss").item() << '\n';
    Save(root / ("epoch" + std::to_string(e) + "_validation.npz"), out);
    Save(root / ("epoch" + std::to_string(e) + "_arrays.npz"), model.Arrays());
    compare(out, "epoch" + std::to_string(e) + "_validation.npz");
    std::cout << mode << " epoch=" << e << " validationCE=" << out.at("loss").item() << std::endl;
  }
  auto final = model.Arrays(), grad = model.Run(first, true);
  Save(root / "task_arrays.npz", final);
  Save(root / "final_base.npz", model.Arrays(true));
  Save(root / "final_gradients.npz", grad);
  compare(grad, "final_gradients.npz");
  model.Save(root);
  Model deployed(mode, false);
  deployed.Load(root);
  check::Require(
      replay::SameBits(final, deployed.Arrays()) && replay::SameBits(base, deployed.Arrays(true)),
      "full PEFT restore bitexact");
  Model delta(mode, false);
  delta.Import(base);
  delta.task.load((root / "delta.npz").string());
  check::Require(
      replay::SameBits(final, delta.Arrays()) && replay::SameBits(base, delta.Arrays(true)),
      "delta plus base restore bitexact");
  auto out = deployed.Run(test);
  Save(root / "outputs.npz", out);
  compare(out, "outputs.npz");
  check::Near(
      replay::MaxAbsDiff(out.at("logits"), delta.Predict(test)),
      0,
      0,
      "delta restored predictions exact");
  A disabled;
  for (const auto& [name, value] : final)
    if (name != "head/W" && name != "head/b") {
      if (mode == "lora" && name.substr(name.size() - 2) != "/B")
        continue;
      if (mode == "adapter" && name.find("/up/") == std::string::npos)
        continue;
      disabled[name] = nn::zeros(value.shape());
    }
  auto off = deployed.Run(test, false, disabled);
  Save(root / "disabled.npz", {{"logits", off.at("logits")}});
  double effect = replay::MaxAbsDiff(out.at("logits"), off.at("logits"));
  if (mode != "head")
    check::Require(effect > 1e-8, "learned PEFT modules change logits");
  double merge_error = 0;
  if (mode == "lora") {
    A merged = base;
    for (auto projection : {"attention/q", "attention/v"}) {
      auto prefix = std::string("lora/") + projection;
      auto key    = std::string(projection) + "/W";
      merged[key] =
          base.at(key) + nn::matmul(final.at(prefix + "/A"), final.at(prefix + "/B")) * 2.;
    }
    Model plain("head", false);
    plain.Import(merged);
    plain.Import(final, false);
    auto result = plain.Run(test);
    merge_error = replay::MaxAbsDiff(out.at("logits"), result.at("logits"));
    check::Near(merge_error, 0, 2e-6, "merged LoRA plain graph predictions");
    check::Near(
        replay::MaxAbsDiff(out.at("input_gradient"), result.at("input_gradient")),
        0,
        2e-6,
        "merged LoRA input adjoint");
    Save(root / "merged_base.npz", merged);
    Save(root / "merged_outputs.npz", result);
    std::filesystem::create_directories(root / "merged");
    plain.Save(root / "merged");
    Model restored("head", false);
    restored.Load(root / "merged");
    check::Near(
        replay::MaxAbsDiff(result.at("logits"), restored.Predict(test)),
        0,
        0,
        "plain merged checkpoint exact");
  }
  auto padded = bert::Rows(test, {0, 1, 2}), original = deployed.Run(padded);
  auto x = padded.at("input").to_vector<int>(), lengths = padded.at("lengths").to_vector<int>();
  for (int b = 0; b < 3; ++b)
    for (int t = lengths[b]; t < 128; ++t)
      x[b * 128 + t] = 5 + (b + t) % 65;
  padded["input"] = nn::from_vector(x, {3, 128});
  auto invariant  = deployed.Run(padded);
  for (auto key : {"logits", "hidden", "attention", "embedded", "loss", "input_gradient"})
    check::Near(
        replay::MaxAbsDiff(original.at(key), invariant.at(key)), 0, 0, "PEFT padding invariance");
  Save(root / "padding_inputs.npz", padded);
  Save(root / "padding_outputs.npz", invariant);
  double fd    = 0;
  int    count = 0;
  for (auto name : model.task.names()) {
    auto value  = final.at(name);
    auto values = value.to_vector<float>();
    for (int i : {0, int(values.size() / 2), int(values.size() - 1)}) {
      float old = values[i];
      values[i] = old + .001f;
      auto plus = model.Run(first, false, {{name, nn::from_vector(values, value.shape())}})
                      .at("loss")
                      .item();
      values[i]  = old - .001f;
      auto minus = model.Run(first, false, {{name, nn::from_vector(values, value.shape())}})
                       .at("loss")
                       .item();
      values[i] = old;
      fd = std::max(fd, std::abs((plus - minus) / .002 - grad.at(name).to_vector<float>()[i]));
      ++count;
    }
  }
  check::Require(fd < .002, "native PEFT trainable derivatives");
  check::Require(
      replay::SameBits(final, deployed.Arrays()) && replay::SameBits(base, deployed.Arrays(true)) &&
          replay::SameBits(base, model.Arrays(true)),
      "frozen and deployed weights immutable");
  auto z       = out.at("logits").to_vector<float>();
  auto labels  = test.at("labels").to_vector<int>();
  int  correct = 0;
  for (int i = 0; i < 16; ++i)
    correct += (z[i * 2 + 1] > z[i * 2]) == labels[i];
  std::ofstream metrics(root / "metrics.json");
  metrics.precision(17);
  metrics << "{\"mode\":\"" << mode
          << "\",\"frozen_parameters\":2216,\"frozen_variables\":21,\"trainable_parameters\":"
          << expected << ",\"epochs\":" << epochs << ",\"updates\":" << updates
          << ",\"test_ce\":" << out.at("loss").item() << ",\"test_accuracy\":" << correct / 16.
          << ",\"module_effect\":" << effect << ",\"merge_error\":" << merge_error
          << ",\"native_fd_count\":" << count << ",\"native_fd_error\":" << fd << "}\n";
  std::cout << mode << " accuracy=" << correct << "/16 trainable=" << expected << " frozen=2216; "
            << count << " FD max=" << fd << " mergeError=" << merge_error
            << "; freeze/deltaRestore/padding/immutable PASS\n";
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
    auto root = paths::OutputDir("models/lora_adapters");
    replay::CopyAssets(
        paths::Dataset("bert_finetune/classification"), root, {"data.npz", "manifest.json"});
    auto data = replay::ToMap(nn::load_npz((root / "data.npz").string()));
    for (auto mode : {"head", "lora", "adapter"})
      Task(mode, data, epochs, root / mode);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
