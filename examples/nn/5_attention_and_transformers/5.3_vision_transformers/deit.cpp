// DeiT hard distillation with an actual frozen CNN teacher and dual tokens.
// CMD: just run-nn deit
// Eager port: separate stores/optimizers, same batches, losses and controls.
// NPZ checkpoints replace TF checkpoints. Recorded TF initial arrays are replayed
// when present; otherwise initialization parity is statistical. Option epochs=N.
#include "common/deit.hpp"
namespace nn = datamunge::nn;
namespace {
using deit::Model;
using vit::A;
void Save(const std::filesystem::path& path, const A& a) {
  nn::save_npz(path.string(), replay::ToNamed(a));
}
A WithTeacher(Model& teacher, A b) {
  b["teacher_labels"] = teacher.TeacherLabels(b);
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
      float old   = values[c];
      values[c]   = old + .0005f;
      double plus = model.Run(batch, false, .5, {{name, nn::from_vector(values, v.shape())}})
                        .at("loss")
                        .item();
      values[c]    = old - .0005f;
      double minus = model.Run(batch, false, .5, {{name, nn::from_vector(values, v.shape())}})
                         .at("loss")
                         .item();
      values[c] = old;
      error     = std::max(
          error, std::abs((plus - minus) / .001 - gradients.at(name).to_vector<float>()[c]));
      ++count;
    }
  }
  check::Require(error < .004, "DeiT/CNN whole-model finite differences");
  std::ofstream file(root / "finite_differences.json");
  file.precision(17);
  file << "{\"count\":" << count << ",\"error\":" << error << "}\n";
  std::cout << (model.teacher ? "teacher" : "student") << " nativeFD=" << count << " max=" << error
            << std::endl;
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
        {{"fit", fit.at("teacher_labels")},
         {"validation", validation.at("teacher_labels")},
         {"test", test.at("teacher_labels")}});
  }
  const auto tf        = replay::TfModels("deit") / (model.teacher ? "teacher" : "student");
  const bool replay_tf = std::filesystem::exists(tf / "initial_arrays.npz");
  if (replay_tf)
    replay::LoadInto(model.parameters, tf / "initial_arrays.npz");
  auto compare = [&](const A& out, const std::string& file, double tolerance) {
    if (replay_tf && std::filesystem::exists(tf / file)) {
      auto reference = replay::ToMap(nn::load_npz((tf / file).string()));
      // The hard teacher targets must match before comparing student losses.
      const auto targets = tf / "teacher_targets.npz";
      if (!model.teacher && std::filesystem::exists(targets)) {
        const auto expected = replay::ToMap(nn::load_npz(targets.string()));
        if (!replay::SameBits(expected.at("fit"), fit.at("teacher_labels"))) {
          std::cout << "TF teacher hard targets differ; training comparison is statistical\n";
          return;
        }
      }
      double error = replay::MaxDiff(out, reference);
      // Small CNN activation errors accumulate through 48 Adam updates; check
      // predictions/loss/input partials more tightly than internal feature maps.
      check::Near(
          error, 0, model.teacher ? std::max(tolerance, 2e-4) : tolerance, "TF DeiT " + file);
      if (model.teacher)
        for (const auto key : {"logits", "loss", "input_gradient"})
          check::Near(
              replay::MaxAbsDiff(out.at(key), reference.at(key)),
              0,
              5e-5,
              "TF CNN " + std::string(key));
      std::cout << (model.teacher ? "teacher" : "student") << " TF " << file
                << " difference=" << error << std::endl;
    }
  };
  auto frozen_before = frozen ? frozen->Arrays() : A{};
  Save(root / "initial_arrays.npz", model.Arrays());
  auto initial = model.Run(test);
  Save(root / "initial_outputs.npz", initial);
  compare(initial, "initial_outputs.npz", 5e-5);
  nn::BatchOrder order(128, 8, 48507);
  std::ofstream  batches(root / "batches.tsv"), history(root / "history.tsv");
  batches << "epoch\tstep\trow\tsource_id\n";
  history << "epoch\tvalidation_loss\tclassification_ce\tdistillation_ce\n";
  int updates = 0;
  A   first;
  for (int epoch = 1; epoch <= epochs; ++epoch) {
    order.begin_epoch();
    for (;;) {
      auto rows = order.next();
      if (rows.empty())
        break;
      auto batch = replay::Rows(fit, rows);
      if (frozen)
        check::Require(
            replay::SameBits(frozen->TeacherLabels(batch), batch.at("teacher_labels")),
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
    history << epoch << '\t' << out.at("loss").item() << '\t'
            << out.at("classification_loss").item() << '\t'
            << (model.teacher ? 0 : out.at("distillation_loss").item()) << '\n';
    const auto prefix = "epoch" + std::to_string(epoch);
    Save(root / (prefix + "_arrays.npz"), model.Arrays());
    Save(root / (prefix + "_validation.npz"), out);
    compare(out, prefix + "_validation.npz", 1e-4);
    std::cout << (model.teacher ? "teacher" : "student") << " epoch=" << epoch
              << " validationLoss=" << out.at("loss").item() << std::endl;
  }
  Save(root / "model_arrays.npz", model.Arrays());
  auto gradients = model.Run(first, true);
  Save(root / "final_gradients.npz", gradients);
  FiniteDifferences(model, first, gradients, root);
  model.parameters.save((root / "final.npz").string());
  Model deployed(model.teacher, false);
  deployed.parameters.load((root / "final.npz").string());
  check::Require(
      replay::SameBits(model.Arrays(), deployed.Arrays()), "teacher/student checkpoints bitexact");
  auto before = deployed.Arrays(), out = deployed.Run(test);
  Save(root / "outputs.npz", out);
  compare(out, "outputs.npz", 1e-4);
  check::Near(
      replay::MaxAbsDiff(deployed.Predict({{"images", test.at("images")}}), out.at("logits")),
      0,
      0,
      "image-only inference requires neither teacher nor gold targets");
  if (!model.teacher) {
    auto cls = model.Run(first, true, 0), dist = model.Run(first, true, 1);
    Save(root / "classification_only.npz", cls);
    Save(root / "distillation_only.npz", dist);
    for (const auto& name : model.parameters.names())
      check::Near(
          replay::MaxAbsDiff(gradients.at(name), (cls.at(name) + dist.at(name)) * .5),
          0,
          2e-6,
          "dual-loss gradient decomposition");
    for (const auto name : {"distill_head/W", "distill_head/b"})
      check::Near(
          nn::max(nn::abs(cls.at(name))).item(),
          0,
          0,
          "classification does not supervise DIST head");
    for (const auto name : {"head/W", "head/b"})
      check::Near(
          nn::max(nn::abs(dist.at(name))).item(), 0, 0, "teacher CE does not supervise CLS head");
    auto changed = first;
    auto targets = first.at("teacher_labels").to_vector<int>();
    for (auto& t : targets)
      t = (t + 1) % 10;
    changed["teacher_labels"] = nn::from_vector(targets, first.at("teacher_labels").shape());
    auto control              = model.Run(changed, true);
    check::Near(
        replay::MaxAbsDiff(control.at("logits"), gradients.at("logits")),
        0,
        0,
        "teacher target cannot leak into prediction");
    check::Near(
        control.at("classification_loss").item(),
        gradients.at("classification_loss").item(),
        0,
        "teacher affects only distillation loss");
    check::Require(
        replay::MaxAbsDiff(control.at("distill_head/W"), gradients.at("distill_head/W")) > .001,
        "teacher supervision trains DIST head");
    Save(root / "changed_teacher_inputs.npz", changed);
    Save(root / "changed_teacher_outputs.npz", control);
    auto ablated = deployed.Run(test, false, .5, {{"distillation_token", nn::zeros({1, 1, 8})}});
    Save(root / "zero_dist_token.npz", ablated);
    check::Require(
        replay::MaxAbsDiff(out.at("logits"), ablated.at("logits")) > 1e-6,
        "trained DIST token affects logits");
    check::Near(
        replay::MaxAbsDiff(
            out.at("logits"), (out.at("class_logits") + out.at("distill_logits")) * .5),
        0,
        0,
        "inference mean of classifier logits");
  }
  for (int row = 0; row < 3; ++row)
    check::Near(
        replay::MaxAbsDiff(
            deployed.Predict(replay::Rows(test, {row})),
            nn::slice(out.at("logits"), {row, 0}, {1, 10})),
        0,
        3e-6,
        "teacher/student batch isolation");
  check::Require(
      replay::SameBits(before, deployed.Arrays()), "teacher/student inference immutable");
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
  std::ofstream meta(root / "metrics.json");
  meta.precision(17);
  meta << "{\"teacher\":" << (model.teacher ? "true" : "false")
       << ",\"parameters\":" << model.parameters.count()
       << ",\"variables\":" << model.parameters.size() << ",\"epochs\":" << epochs
       << ",\"updates\":" << updates << ",\"test_loss\":" << out.at("loss").item()
       << ",\"test_accuracy\":" << correct / 32. << "}\n";
  std::cout << (model.teacher ? "teacher" : "student") << " testLoss=" << out.at("loss").item()
            << " accuracy=" << correct
            << "/32;restore/teacherFreeze/labelFree/dualHeads/immutable PASS\n";
}
void Run(int epochs) {
  const auto root = paths::OutputDir("models/deit");
  replay::CopyAssets(paths::Dataset("vision_transformer"), root, {"data.npz", "manifest.json"});
  auto  data = replay::ToMap(nn::load_npz((root / "data.npz").string()));
  Model teacher(true);
  check::Require(
      teacher.parameters.count() == 498 && teacher.parameters.size() == 6,
      "actual 498-parameter CNN teacher");
  Train(teacher, nullptr, data, epochs, root / "teacher");
  Model frozen(true, false);
  frozen.parameters.load((root / "teacher/final.npz").string());
  check::Require(
      replay::SameBits(teacher.Arrays(), frozen.Arrays()), "frozen teacher exact transfer");
  const auto before = frozen.Arrays();
  Model      student(false);
  check::Require(
      student.parameters.count() == 3100 && student.parameters.size() == 35,
      "3100 parameters/35 dual-token tensors");
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
