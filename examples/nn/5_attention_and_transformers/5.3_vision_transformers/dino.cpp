// DINO self-distillation with multi-crop views, detached teacher and EMA state.
// CMD: just run-nn dino
// Eager port with NPZ checkpoints and recorded TF initial-array replay.
// Same seeded crop construction, student-only Adam, EMA/center schedule and checks.
#include "common/dino.hpp"
namespace nn = datamunge::nn;
namespace {
using vit::A;
void Save(const std::filesystem::path& path, const A& a) {
  nn::save_npz(path.string(), replay::ToNamed(a));
}
void Run(int epochs) {
  const auto root = paths::OutputDir("models/dino");
  replay::CopyAssets(paths::Dataset("vision_transformer"), root, {"data.npz", "manifest.json"});
  auto data = replay::ToMap(nn::load_npz((root / "data.npz").string()));
  auto fit = vit::Pool(data, "fit"), validation = dino::Views(vit::Pool(data, "validation"), 50111),
       test = dino::Views(vit::Pool(data, "test"), 50113);
  dino::Model model;
  check::Require(
      model.student.count() == 3452 && model.student.size() == 37 && model.teacher.count() == 3452,
      "matching 3452-parameter37-tensor DINO networks");
  const auto tf        = replay::TfModels("dino");
  const bool replay_tf = std::filesystem::exists(tf / "initial_arrays.npz");
  if (replay_tf)
    model.Assign(replay::ToMap(nn::load_npz((tf / "initial_arrays.npz").string())));
  auto init = model.Arrays();
  for (const auto& [name, value] : init)
    if (name.rfind("student/", 0) == 0)
      check::Require(
          replay::SameBits(value, init.at("teacher/" + name.substr(8))),
          "teacher initialized exactly from student");
  auto compare = [&](const A& out, const std::string& file, double tolerance) {
    if (replay_tf && std::filesystem::exists(tf / file)) {
      const auto error = replay::MaxDiff(out, tf / file);
      check::Near(error, 0, tolerance, "TF DINO " + file);
      std::cout << "TF " << file << " difference=" << error << std::endl;
    }
  };
  Save(root / "initial_arrays.npz", init);
  Save(root / "validation_inputs.npz", validation);
  Save(root / "test_inputs.npz", test);
  auto initial = model.Run(test);
  Save(root / "initial_outputs.npz", initial);
  compare(initial, "initial_outputs.npz", 5e-5);
  auto fit_original  = dino::Views(fit, 0, true),
       test_original = dino::Views(vit::Pool(data, "test"), 0, true);
  Save(root / "initial_fit_original.npz", model.Run(fit_original));
  Save(root / "initial_test_original.npz", model.Run(test_original));
  nn::BatchOrder order(128, 8, 50107);
  std::ofstream  batches(root / "batches.tsv"), history(root / "history.tsv");
  batches << "epoch\tstep\trow\tsource_id\n";
  history << "epoch\tvalidation_loss\n";
  int step = 0;
  A   first;
  for (int epoch = 1; epoch <= epochs; ++epoch) {
    order.begin_epoch();
    for (;;) {
      auto ids = order.next();
      if (ids.empty())
        break;
      auto b = dino::Views(replay::Rows(fit, ids), 50119 + step);
      Save(root / ("step" + std::to_string(step) + "_inputs.npz"), b);
      Save(root / ("step" + std::to_string(step) + "_before.npz"), model.Arrays());
      if (!step) {
        first = b;
        Save(root / "first_batch.npz", first);
        auto g = model.Run(first, true);
        Save(root / "initial_gradients.npz", g);
        compare(g, "initial_gradients.npz", 1e-4);
      }
      model.Update(b, step, epochs * 16, epoch == 1);
      auto after = model.Arrays();
      Save(root / ("step" + std::to_string(step) + "_after.npz"), after);
      if (!step)
        Save(root / "first_after.npz", after);
      const auto sources = fit.at("source_ids").to_vector<int>();
      for (auto row : ids)
        batches << epoch << '\t' << step << '\t' << row << '\t' << sources[row] << '\n';
      ++step;
    }
    auto result = model.Run(validation);
    history << epoch << '\t' << result.at("loss").item() << '\n';
    const auto prefix = "epoch" + std::to_string(epoch);
    Save(root / (prefix + "_arrays.npz"), model.Arrays());
    Save(root / (prefix + "_validation.npz"), result);
    compare(result, prefix + "_validation.npz", 5e-4);
    std::cout << "epoch=" << epoch << " crossViewValidationLoss=" << result.at("loss").item()
              << std::endl;
  }
  const auto arrays    = model.Arrays();
  auto       gradients = model.Run(first, true);
  Save(root / "final_gradients.npz", gradients);
  for (const auto& name : model.teacher.names())
    check::Near(
        nn::max(nn::abs(gradients.at("teacher/" + name))).item(),
        0,
        0,
        "teacher detached loss gradient exact zero");
  double fd    = 0;
  int    count = 0;
  for (const auto& name : model.student.names()) {
    const auto value  = arrays.at("student/" + name);
    auto       values = value.to_vector<float>();
    for (int i : {0, int(values.size() / 2), int(values.size() - 1)}) {
      const float old = values[i];
      values[i]       = old + .00005f;
      model.Assign({{"student/" + name, nn::from_vector(values, value.shape())}});
      const auto plus = model.Run(first).at("loss").item();
      values[i]       = old - .00005f;
      model.Assign({{"student/" + name, nn::from_vector(values, value.shape())}});
      const auto minus = model.Run(first).at("loss").item();
      values[i]        = old;
      model.Assign({{"student/" + name, nn::from_vector(values, value.shape())}});
      fd = std::max(
          fd,
          std::abs((plus - minus) / .0001 - gradients.at("student/" + name).to_vector<float>()[i]));
      ++count;
    }
  }
  check::Require(fd < .02, "DINO whole-student finite differences");
  Save(root / "model_arrays.npz", model.Arrays());
  Save(root / "final.npz", model.Arrays());
  dino::Model deployed(false);
  deployed.Assign(replay::ToMap(nn::load_npz((root / "final.npz").string())));
  auto before = deployed.Arrays();
  check::Require(
      replay::SameBits(arrays, before), "all student/teacher/center tensors restored bitexact");
  auto result = deployed.Run(test);
  Save(root / "outputs.npz", result);
  compare(result, "outputs.npz", 5e-4);
  Save(root / "fit_original.npz", deployed.Run(fit_original));
  Save(root / "test_original.npz", deployed.Run(test_original));
  auto      local_changed = test;
  auto      pixels        = test.at("views").to_vector<float>();
  const int start         = 2 * 32 * 32 * 32 * 3;
  for (std::size_t i = start; i < pixels.size(); ++i)
    pixels[i] = 1 - pixels[i];
  local_changed["views"] = nn::from_vector(pixels, test.at("views").shape());
  auto control           = deployed.Run(local_changed);
  check::Near(
      replay::MaxAbsDiff(control.at("teacher_logits"), result.at("teacher_logits")),
      0,
      0,
      "teacher sees only two global views");
  check::Require(
      replay::MaxAbsDiff(control.at("student_logits"), result.at("student_logits")) > 1e-7,
      "student responds to local crops");
  Save(root / "local_changed_inputs.npz", local_changed);
  Save(root / "local_changed_outputs.npz", control);
  for (int row = 0; row < 3; ++row) {
    auto one =
        nn::gather(test.at("views"), nn::from_vector(std::vector<std::int32_t>{row}, {1}), 1);
    auto single = deployed.Run({{"views", one}});
    check::Near(
        replay::MaxAbsDiff(
            single.at("student_logits"),
            nn::slice(result.at("student_logits"), {0, row, 0}, {4, 1, 16})),
        0,
        5e-6,
        "crop/image batch isolation");
  }
  check::Require(
      replay::SameBits(before, deployed.Arrays()),
      "inference leaves teacher/student/center immutable");
  std::ofstream meta(root / "metrics.json");
  meta.precision(17);
  meta << "{\"parameters_per_network\":3452,\"variables_per_network\":37,\"epochs\":" << epochs
       << ",\"updates\":" << step << ",\"test_loss\":" << result.at("loss").item()
       << ",\"native_fd_count\":" << count << ",\"native_fd_error\":" << fd << "}\n";
  std::cout << "DINO testCrossViewLoss=" << result.at("loss").item() << "; " << count
            << " FD max=" << fd << ";detachedTeacher/EMA/center/multicrop/restore/immutable PASS\n";
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
