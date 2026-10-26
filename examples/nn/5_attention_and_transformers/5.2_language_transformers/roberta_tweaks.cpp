// Controlled static/dynamic masking and batch-size changes for RoBERTa-style
// MLM.
//
// CMD: just run-nn roberta_tweaks
// Eager port preserving the four controlled comparisons and all source checks.
#include <fstream>

#include "common/compact_bert.hpp"
namespace nn = datamunge::nn;
namespace {
using A     = compact_bert::A;
using Model = compact_bert::Model;
void Save(const std::filesystem::path& file, const A& a) {
  nn::save_npz(file.string(), replay::ToNamed(a));
}
A Pool(const A& a, const std::string& split, int epoch = 1) {
  A b;
  for (auto k : {"target", "segments", "lengths", "source_start"})
    b[k] = a.at(split + "_" + k);
  for (auto k : {"input", "selected", "actions"})
    b[k] = a.at((split == "fit" ? "epoch" + std::to_string(epoch) : split) + "_" + k);
  return b;
}
void Run(const std::string& name, bool dynamic, int batch, int epochs, const A& data) {
  compact_bert::Config config{8, 2, false, false};
  std::vector<A>       pools{Pool(data, "fit"), Pool(data, "validation"), Pool(data, "test")};
  auto                 root = paths::OutputDir("models/roberta_tweaks/" + name),
       tf                   = replay::TfModels("roberta_tweaks") / name;
  replay::CopyAssets(
      paths::Dataset("bert_mlm"), root, {"data.npz", "manifest.json", "vocabulary.json"});
  bool  distill = false;
  Model model(config);
  int   expected = 2190;
  check::Require(model.parameters.size() == 41, "segment-free two-block MLM41 tensors");
  check::Require(
      model.parameters.count() == expected,
      "actual factorization/sharing/student compression count");
  bool parity = std::filesystem::exists(tf / "initial_arrays.npz");
  if (parity)
    replay::LoadInto(model.parameters, tf / "initial_arrays.npz");
  auto compare = [&](const A& out, const std::string& file) {
    if (parity && std::filesystem::exists(tf / file)) {
      double error = replay::MaxDiff(out, tf / file);
      std::cout << name << " TF " << file << " difference=" << error << std::endl;
      check::Near(error, 0, 2e-4, "compact TF " + file);
    }
  };
  Save(root / "initial_arrays.npz", model.Arrays());
  auto firstout = model.Run(pools[2]);
  Save(root / "initial_outputs.npz", firstout);
  compare(firstout, "initial_outputs.npz");
  if (distill)
    for (int i = 0; i < 3; ++i)
      Save(
          root / (std::string(
                      i == 0   ? "fit"
                      : i == 1 ? "validation"
                               : "test") +
                  "_teacher.npz"),
          {{"logits", pools[i].at("teacher_logits")}, {"hidden", pools[i].at("teacher_hidden")}});
  nn::BatchOrder order(128, batch, 46707);
  std::ofstream  history(root / "history.tsv"), batches(root / "batches.tsv");
  history << "epoch\tvalidation_mlm\tvalidation_kd\tvalidation_cosine\tvalidation_total\n";
  batches << "epoch\tstep\trow\n";
  int steps = 0;
  A   first;
  for (int e = 1; e <= epochs; ++e) {
    pools[0] = Pool(data, "fit", dynamic ? e : 1);
    Save(root / ("epoch" + std::to_string(e) + "_training.npz"), pools[0]);
    order.begin_epoch();
    for (;;) {
      auto rows = order.next();
      if (rows.empty())
        break;
      auto b = bert::Rows(pools[0], rows);
      if (steps % (128 / batch) == 0)
        Save(root / ("epoch" + std::to_string(e) + "_first_batch.npz"), b);
      if (!steps) {
        first = b;
        Save(root / "first_batch.npz", b);
        Save(root / "first_forward.npz", model.Run(b));
        auto out = model.Run(b, true);
        A    g;
        for (auto k : model.parameters.names())
          g[k] = out.at(k);
        Save(root / "initial_gradients.npz", g);
        compare(g, "initial_gradients.npz");
      }
      model.Update(b);
      if (!steps)
        Save(root / "first_after.npz", model.Arrays());
      for (auto r : rows)
        batches << e << '\t' << steps << '\t' << r << '\t' << (dynamic ? e : 1) << '\n';
      ++steps;
    }
    Save(root / ("epoch" + std::to_string(e) + "_arrays.npz"), model.Arrays());
    auto v = model.Run(pools[1]);
    history << e;
    for (auto k : {"mlm_loss", "kd_loss", "cos_loss", "loss"})
      history << '\t' << v.at(k).item();
    history << '\n';
    std::cout << name << " epoch=" << e << " validationMLM=" << v.at("mlm_loss").item()
              << " total=" << v.at("loss").item() << std::endl;
  }
  auto g = model.Run(first, true);
  A    gradients;
  for (auto key : model.parameters.names())
    gradients[key] = g.at(key);
  Save(root / "final_gradients.npz", gradients);
  compare(gradients, "final_gradients.npz");
  model.parameters.save((root / "final.npz").string());
  Model deployed(config, distill, false);
  deployed.parameters.load((root / "final.npz").string());
  auto final = deployed.Arrays();
  check::Require(replay::SameBits(model.Arrays(), final), "compact bitexact restore");
  auto out = deployed.Run(pools[2]);
  Save(root / "model_arrays.npz", final);
  Save(root / "outputs.npz", out);
  compare(out, "outputs.npz");
  auto b              = bert::Rows(pools[2], {0, 1});
  b["lengths"]        = nn::from_vector(std::vector<int>{24, 20}, {2});
  auto selected_short = b.at("selected").to_vector<float>();
  for (int n = 0; n < 2; ++n)
    for (int t = n == 0 ? 24 : 20; t < 32; ++t)
      selected_short[n * 32 + t] = 0;
  b["selected"] = nn::from_vector(selected_short, {2, 32});
  auto base = deployed.Run(b), padded = bert::Rows(b, {0, 1});
  auto lengths = padded.at("lengths").to_vector<int>();
  for (auto key : {"input", "segments"}) {
    auto x = padded.at(key).to_vector<int>();
    for (int n = 0; n < 2; ++n)
      for (int t = lengths[n]; t < 32; ++t)
        x[n * 32 + t] = std::string(key) == "input" ? 17 : 1;
    padded[key] = nn::from_vector(x, {2, 32});
  }
  if (distill)
    for (auto key : {"teacher_logits", "teacher_hidden"}) {
      auto t     = padded.at(key);
      auto x     = t.to_vector<float>();
      int  width = t.size(2);
      for (int n = 0; n < 2; ++n)
        for (int i = lengths[n] * width; i < 32 * width; ++i)
          x[n * 32 * width + i] = width == 70 ? 1000 : -1000;
      padded[key] = nn::from_vector(x, t.shape());
    }
  auto po = deployed.Run(padded);
  for (const auto& [k, v] : base)
    check::Near(
        replay::MaxAbsDiff(v, po.at(k)), 0, 0, "padding cannot affect compact paths/loss/partials");
  Save(root / "padding_inputs.npz", padded);
  Save(root / "padding_outputs.npz", po);
  auto no_gold        = b;
  no_gold["target"]   = nn::full({2, 32}, 69, nn::DType::Int32);
  no_gold["selected"] = nn::zeros({2, 32});
  if (distill) {
    no_gold["teacher_logits"] = nn::full({2, 32, 70}, 1000);
    no_gold["teacher_hidden"] = nn::full({2, 32, 8}, -1000);
  }
  check::Near(
      replay::MaxAbsDiff(deployed.Predict(b), deployed.Predict(no_gold)),
      0,
      0,
      "student inference excludes gold and teacher");
  double fd       = 0;
  int    fd_count = 0;
  for (auto key : model.parameters.names()) {
    auto value = final.at(key);
    auto x     = value.to_vector<float>();
    for (int i : {0, int(x.size() / 2), int(x.size() - 1)}) {
      float old = x[i];
      x[i]      = old + .001f;
      auto plus =
          model.Run(first, false, {{key, nn::from_vector(x, value.shape())}}).at("loss").item();
      x[i] = old - .001f;
      auto minus =
          model.Run(first, false, {{key, nn::from_vector(x, value.shape())}}).at("loss").item();
      x[i] = old;
      fd   = std::max(fd, std::abs((plus - minus) / .002 - g.at(key).to_vector<float>()[i]));
      ++fd_count;
    }
  }
  check::Require(fd < .004, "shared/tied/triple-loss native central differences");
  check::Require(replay::SameBits(final, deployed.Arrays()), "compact inference immutable");
  int  correct = 0, count = 0;
  auto logits   = out.at("logits").to_vector<float>(),
       selected = pools[2].at("selected").to_vector<float>();
  auto target   = pools[2].at("target").to_vector<int>();
  for (int i = 0; i < 32 * 32; ++i)
    if (selected[i]) {
      int best = 0;
      for (int c = 1; c < 70; ++c)
        if (logits[i * 70 + c] > logits[i * 70 + best])
          best = c;
      correct += best == target[i];
      ++count;
    }
  std::ofstream meta(root / "metrics.json");
  meta.precision(17);
  meta << "{\"parameters\":" << expected << ",\"variables\":" << model.parameters.size()
       << ",\"embedding\":" << config.embedding << ",\"layers\":" << config.layers
       << ",\"shared\":" << config.shared << ",\"segments\":" << config.segments
       << ",\"distilled\":" << distill << ",\"epochs\":" << epochs << ",\"updates\":" << steps
       << ",\"test_mlm_ce\":" << out.at("mlm_loss").item()
       << ",\"test_total\":" << out.at("loss").item() << ",\"test_kd\":" << out.at("kd_loss").item()
       << ",\"test_cosine\":" << out.at("cos_loss").item()
       << ",\"selected_accuracy\":" << double(correct) / count
       << ",\"native_fd_count\":" << fd_count << ",\"native_fd_error\":" << fd << "}\n";
  std::cout << name << " params=" << expected << " accuracy=" << correct << '/' << count << " "
            << fd_count << "nativeFD max=" << fd
            << "; padding/teacherFree/restore/immutable PASS\n";
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
    check::Require(epochs > 0 && epochs <= 100, "positive bounded epochs");
    auto data =
        replay::ToMap(nn::load_npz((paths::Dataset("roberta_tweaks") / "data.npz").string()));
    check::Require(
        data.count("epoch" + std::to_string(epochs) + "_input"),
        "prepare_roberta_tweaks.py --epochs=N for requested epochs");
    Run("static_b8", false, 8, epochs, data);
    Run("dynamic_b8", true, 8, epochs, data);
    Run("static_b16", false, 16, epochs, data);
    Run("dynamic_b16", true, 16, epochs, data);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
