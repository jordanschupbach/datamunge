// BEiT masked token pretraining with a separately trained categorical
// tokenizer. CMD: just run-nn beit_pretraining
// Eager port: separate stores/optimizers, same batches, losses and controls.
// NPZ checkpoints replace TF checkpoints. Recorded TF initial arrays are replayed
// when present; otherwise initialization parity is statistical. Option epochs=N.
#include "common/beit_pretraining.hpp"
namespace nn = datamunge::nn;
namespace {
using beit::Model;
using vit::A;
void Save(const std::filesystem::path& path, const A& a) {
  nn::save_npz(path.string(), replay::ToNamed(a));
}
A WithTeacher(Model& teacher, A b) {
  b["tokens"] = teacher.TokenIDs(b);
  return b;
}
void FiniteDifferences(
    Model&                       model,
    const A&                     batch,
    const A&                     gradients,
    const std::filesystem::path& root) {
  const auto arrays = model.Arrays();
  double     error  = 0;
  int        count  = 0;
  for (const auto& name : model.parameters.names()) {
    const auto v      = arrays.at(name);
    auto       values = v.to_vector<float>();
    for (int c : {0, int(values.size() / 2), int(values.size() - 1)}) {
      float old = values[c];
      values[c] = old + .0005f;
      double plus =
          model.Run(batch, false, {{name, nn::from_vector(values, v.shape())}}).at("loss").item();
      values[c] = old - .0005f;
      double minus =
          model.Run(batch, false, {{name, nn::from_vector(values, v.shape())}}).at("loss").item();
      values[c] = old;
      error     = std::max(
          error, std::abs((plus - minus) / .001 - gradients.at(name).to_vector<float>()[c]));
      ++count;
    }
  }
  check::Require(error < .004, "BEiT/CNN whole-model finite differences");
  std::ofstream file(root / "finite_differences.json");
  file.precision(17);
  file << "{\"count\":" << count << ",\"error\":" << error << "}\n";
  std::cout << (model.tokenizer ? "tokenizer" : "student") << " nativeFD=" << count
            << " max=" << error << std::endl;
}
void Train(
    Model&                       model,
    Model*                       frozen,
    const A&                     data,
    int                          epochs,
    const std::filesystem::path& root) {
  std::filesystem::create_directories(root);
  auto fit = vit::Pool(data, "fit"), validation = vit::Pool(data, "validation"),
       test = vit::Pool(data, "test");
  if (frozen) {
    fit        = WithTeacher(*frozen, fit);
    validation = WithTeacher(*frozen, validation);
    test       = WithTeacher(*frozen, test);
    Save(
        root / "teacher_targets.npz",
        {{"fit", fit.at("tokens")},
         {"validation", validation.at("tokens")},
         {"test", test.at("tokens")}});
  }
  const auto tf =
      replay::TfModels("beit_pretraining") / (model.tokenizer ? "tokenizer" : "student");
  const bool replay_tf = std::filesystem::exists(tf / "initial_arrays.npz");
  if (replay_tf)
    replay::LoadInto(model.parameters, tf / "initial_arrays.npz");
  auto compare = [&](const A& out, const std::string& file, double tolerance) {
    if (replay_tf && std::filesystem::exists(tf / file)) {
      auto reference = replay::ToMap(nn::load_npz((tf / file).string()));
      // The hard teacher targets must match before comparing student losses.
      const auto targets = tf / "teacher_targets.npz";
      if (!model.tokenizer && std::filesystem::exists(targets)) {
        const auto expected = replay::ToMap(nn::load_npz(targets.string()));
        if (!replay::SameBits(expected.at("fit"), fit.at("tokens"))) {
          std::cout << "TF teacher hard targets differ; training comparison is statistical\n";
          return;
        }
      }
      double error = replay::MaxDiff(out, reference);
      // Small CNN activation errors accumulate through 48 Adam updates; check
      // predictions/loss/input partials more tightly than internal feature maps.
      check::Near(
          error, 0, model.tokenizer ? std::max(tolerance, 2e-4) : tolerance, "TF BEiT " + file);
      if (model.tokenizer)
        for (const auto key : {"logits", "loss", "input_gradient"})
          check::Near(
              replay::MaxAbsDiff(out.at(key), reference.at(key)),
              0,
              5e-5,
              "TF CNN " + std::string(key));
      std::cout << (model.tokenizer ? "tokenizer" : "student") << " TF " << file
                << " difference=" << error << std::endl;
    }
  };
  validation = beit::Augment(validation, 49311, false);
  test       = beit::Augment(test, 49313, false);
  Save(root / "validation_inputs.npz", validation);
  Save(root / "test_inputs.npz", test);
  auto frozen_before = frozen ? frozen->Arrays() : A{};
  Save(root / "initial_arrays.npz", model.Arrays());
  auto initial = model.Run(test);
  Save(root / "initial_outputs.npz", initial);
  compare(initial, "initial_outputs.npz", 5e-5);
  nn::BatchOrder order(128, 8, 49307);
  std::ofstream  batches(root / "batches.tsv"), history(root / "history.tsv");
  batches << "epoch\tstep\trow\tsource_id\n";
  history << "epoch\tvalidation_loss\n";
  int updates = 0;
  A   first;
  for (int epoch = 1; epoch <= epochs; ++epoch) {
    order.begin_epoch();
    for (;;) {
      auto rows = order.next();
      if (rows.empty())
        break;
      auto batch = beit::Augment(replay::Rows(fit, rows), 49319 + updates, model.tokenizer);
      Save(root / ("batch" + std::to_string(updates) + "_inputs.npz"), batch);
      if (frozen)
        check::Require(
            replay::SameBits(frozen->TokenIDs(batch), batch.at("tokens")),
            "cached hard targets equal frozen image teacher");
      if (!updates) {
        first = batch;
        Save(root / "first_batch.npz", batch);
        auto g = model.Run(batch, true);
        Save(root / "initial_gradients.npz", g);
        compare(g, "initial_gradients.npz", 1e-4);
      }
      model.Update(batch);
      if (frozen)
        check::Require(
            replay::SameBits(frozen_before, frozen->Arrays()),
            "teacher immutable after EVERY student update");
      if (!updates)
        Save(root / "first_after.npz", model.Arrays());
      const auto ids = fit.at("source_ids").to_vector<int>();
      for (auto row : rows)
        batches << epoch << '\t' << updates << '\t' << row << '\t' << ids[row] << '\n';
      ++updates;
    }
    auto out = model.Run(validation);
    history << epoch << '\t' << out.at("loss").item() << '\n';
    const auto prefix = "epoch" + std::to_string(epoch);
    Save(root / (prefix + "_arrays.npz"), model.Arrays());
    Save(root / (prefix + "_validation.npz"), out);
    compare(out, prefix + "_validation.npz", 1e-4);
    std::cout << (model.tokenizer ? "tokenizer" : "student") << " epoch=" << epoch
              << " validationLoss=" << out.at("loss").item() << std::endl;
  }
  Save(root / "model_arrays.npz", model.Arrays());
  auto gradients = model.Run(first, true);
  Save(root / "final_gradients.npz", gradients);
  FiniteDifferences(model, first, gradients, root);
  model.parameters.save((root / "final.npz").string());
  Model deployed(model.tokenizer, false);
  deployed.parameters.load((root / "final.npz").string());
  check::Require(
      replay::SameBits(model.Arrays(), deployed.Arrays()), "teacher/student checkpoints bitexact");
  auto before = deployed.Arrays(), out = deployed.Run(test);
  Save(root / "outputs.npz", out);
  compare(out, "outputs.npz", 1e-4);
  auto targets_free = test;
  targets_free.erase("tokens");
  targets_free.erase("labels");
  check::Near(
      replay::MaxAbsDiff(deployed.Predict(targets_free), out.at("logits")),
      0,
      0,
      "inference requires neither teacher nor gold targets");
  if (!model.tokenizer) {
    auto changed = test;
    auto pixels = test.at("images").to_vector<float>(), mask = test.at("mask").to_vector<float>();
    for (int b = 0; b < 32; ++b)
      for (int patch = 0; patch < 16; ++patch)
        if (mask[b * 16 + patch])
          for (int y = 0; y < 8; ++y)
            for (int x = 0; x < 8; ++x)
              for (int c = 0; c < 3; ++c) {
                auto i    = ((b * 32 + patch / 4 * 8 + y) * 32 + patch % 4 * 8 + x) * 3 + c;
                pixels[i] = 1 - pixels[i];
              }
    changed["images"] = nn::from_vector(pixels, test.at("images").shape());
    auto hidden       = deployed.Run(changed);
    check::Near(
        replay::MaxAbsDiff(hidden.at("logits"), out.at("logits")),
        0,
        0,
        "masked pixel content cannot leak into student");
    Save(root / "hidden_changed_inputs.npz", changed);
    Save(root / "hidden_changed_outputs.npz", hidden);
    auto targets = test;
    auto ids     = test.at("tokens").to_vector<int>();
    for (std::size_t i = 0; i < ids.size(); ++i)
      if (!mask[i])
        ids[i] = (ids[i] + 1) % 16;
    targets["tokens"] = nn::from_vector(ids, test.at("tokens").shape());
    auto visible      = deployed.Run(targets);
    check::Near(
        visible.at("loss").item(), out.at("loss").item(), 0, "unmasked targets excluded from CE");
    Save(root / "visible_targets_inputs.npz", targets);
    Save(root / "visible_targets_outputs.npz", visible);
  } else
    Save(root / "hard_outputs.npz", deployed.HardDecode(test));
  for (int row = 0; row < 3; ++row)
    check::Near(
        replay::MaxAbsDiff(
            deployed.Predict(replay::Rows(test, {row})),
            nn::slice(out.at("logits"), {row, 0, 0}, {1, 16, 16})),
        0,
        3e-6,
        "teacher/student batch isolation");
  check::Require(
      replay::SameBits(before, deployed.Arrays()), "teacher/student inference immutable");
  int correct = 0, total = 0;
  if (!model.tokenizer) {
    const auto z    = out.at("logits").to_vector<float>();
    const auto ids  = test.at("tokens").to_vector<int>();
    const auto mask = test.at("mask").to_vector<float>();
    for (int i = 0; i < 32 * 16; ++i)
      if (mask[i]) {
        int best = 0;
        for (int c = 1; c < 16; ++c)
          if (z[i * 16 + c] > z[i * 16 + best])
            best = c;
        correct += best == ids[i];
        ++total;
      }
  }
  std::ofstream meta(root / "metrics.json");
  meta.precision(17);
  meta << "{\"teacher\":" << (model.tokenizer ? "true" : "false")
       << ",\"parameters\":" << model.parameters.count()
       << ",\"variables\":" << model.parameters.size() << ",\"epochs\":" << epochs
       << ",\"updates\":" << updates << ",\"test_loss\":" << out.at("loss").item()
       << ",\"test_accuracy\":" << (total ? double(correct) / total : 0.) << "}\n";
  std::cout << (model.tokenizer ? "teacher" : "student") << " testLoss=" << out.at("loss").item()
            << " accuracy=" << correct << "/" << total
            << ";restore/tokenizerFreeze/labelFree/maskedCE/immutable PASS\n";
}
void Run(int epochs) {
  const auto root = paths::OutputDir("models/beit_pretraining");
  replay::CopyAssets(paths::Dataset("vision_transformer"), root, {"data.npz", "manifest.json"});
  auto  data = replay::ToMap(nn::load_npz((root / "data.npz").string()));
  Model teacher(true);
  Train(teacher, nullptr, data, epochs, root / "tokenizer");
  Model frozen(true, false);
  frozen.parameters.load((root / "tokenizer/final.npz").string());
  check::Require(
      replay::SameBits(teacher.Arrays(), frozen.Arrays()), "frozen teacher exact transfer");
  const auto before = frozen.Arrays();
  Model      student(false);
  Train(student, &frozen, data, epochs, root / "student");
  check::Require(
      replay::SameBits(before, frozen.Arrays()),
      "entire student training leaves frozen teacher unchanged");
  Save(root / "frozen_teacher_after.npz", frozen.Arrays());
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
