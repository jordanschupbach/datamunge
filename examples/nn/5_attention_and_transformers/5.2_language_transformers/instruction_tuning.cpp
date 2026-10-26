// Supervised instruction tuning: prompt-conditioned, completion-only causal CE.
// CMD: just run-nn instruction_tuning
// Eager port preserving completion masking, transfer and all source controls.
#include "common/instruction_tuning.hpp"

#include <fstream>
#include <numeric>
namespace nn = datamunge::nn;
namespace {
using A     = instruction::A;
using Model = instruction::Model;
void Save(const std::filesystem::path& path, const A& a) {
  nn::save_npz(path.string(), replay::ToNamed(a));
}
A Pool(const A& data, const std::string& split) {
  A b;
  for (auto key :
       {"input", "target", "selected", "lengths", "prompt", "prompt_lengths", "source_ids"})
    b[key] = data.at(split + "_" + key);
  return b;
}
void Run(int epochs) {
  auto root = paths::OutputDir("models/instruction_tuning");
  replay::CopyAssets(
      paths::Dataset("instruction_tuning"), root, {"data.npz", "manifest.json", "vocabulary.json"});
  auto  data = replay::ToMap(nn::load_npz((root / "data.npz").string()));
  Model model;
  check::Require(
      model.parameters.count() == 5491 && model.parameters.size() == 18,
      "5491 params18 tensors with addedPAD/EOS classes");
  Save(root / "fresh_arrays.npz", model.Arrays());
  auto pretrained = replay::TfModels("gpt_lm") / "model_arrays.npz";
  std::filesystem::copy_file(
      pretrained,
      root / "pretrained_arrays.npz",
      std::filesystem::copy_options::overwrite_existing);
  model.Import(replay::ToMap(nn::load_npz(pretrained.string())));
  auto tf     = replay::TfModels("instruction_tuning");
  bool parity = std::filesystem::exists(tf / "initial_arrays.npz");
  if (parity)
    replay::LoadInto(model.parameters, tf / "initial_arrays.npz");
  auto compare = [&](const A& out, const std::string& file) {
    if (parity && std::filesystem::exists(tf / file)) {
      double e = replay::MaxDiff(out, tf / file);
      std::cout << "TF " << file << " difference=" << e << std::endl;
      auto expected = replay::ToMap(nn::load_npz((tf / file).string()));
      for (const auto& [key, value] : out)
        check::Near(
            replay::MaxAbsDiff(value, expected.at(key)),
            0,
            key == "hidden_a" || key == "hidden_b" ? 2e-4 : 5e-5,
            "TF sentence " + file + "/" + key);
    }
  };
  auto initial = model.Arrays();
  Save(root / "initial_arrays.npz", initial);
  auto fit = Pool(data, "fit"), validation = Pool(data, "validation"), test = Pool(data, "test");
  auto init = model.Run(test);
  Save(root / "initial_outputs.npz", init);
  compare(init, "initial_outputs.npz");
  Save(
      root / "initial_generation.npz",
      model.Generate(test.at("prompt"), test.at("prompt_lengths")));
  nn::BatchOrder order(64, 8, 48107);
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
    std::cout << "epoch=" << e << " validationCE=" << out.at("loss").item() << std::endl;
  }
  auto final = model.Arrays(), grad = model.Run(first, true);
  Save(root / "model_arrays.npz", final);
  Save(root / "final_gradients.npz", grad);
  compare(grad, "final_gradients.npz");
  model.parameters.save((root / "final.npz").string());
  Model deployed(false);
  deployed.parameters.load((root / "final.npz").string());
  check::Require(replay::SameBits(final, deployed.Arrays()), "all21 sentence tensors bitexact");
  auto out = deployed.Run(test);
  Save(root / "outputs.npz", out);
  compare(out, "outputs.npz");
  Save(root / "generation.npz", deployed.Generate(test.at("prompt"), test.at("prompt_lengths")));
  auto control  = bert::Rows(first, {0, 1, 2, 3, 4, 5, 6, 7});
  auto labels   = control.at("target").to_vector<int>();
  auto selected = first.at("selected").to_vector<float>();
  for (size_t i = 0; i < labels.size(); ++i)
    if (selected[i] == 0)
      labels[i] = (labels[i] + 19) % 67;
  control["target"] = nn::from_vector(labels, control.at("target").shape());
  auto changed      = model.Run(control, true);
  for (const auto& [key, value] : grad)
    check::Near(
        replay::MaxAbsDiff(value, changed.at(key)),
        0,
        0,
        "masked prompt/padding labels cannot affect objective/derivatives");
  Save(root / "masked_labels_inputs.npz", control);
  Save(root / "masked_labels_outputs.npz", changed);
  double prompt_grad = 0;
  auto   ig          = grad.at("input_gradient").to_vector<float>(),
       lg            = grad.at("logit_gradient").to_vector<float>();
  auto pl            = first.at("prompt_lengths").to_vector<int>();
  for (int b = 0; b < 8; ++b)
    for (int t = 0; t < 64; ++t)
      for (int c = 0; c < 67; ++c) {
        int i = (b * 64 + t) * 67 + c;
        if (t < pl[b])
          prompt_grad = std::max(prompt_grad, double(std::abs(ig[i])));
        if (!selected[b * 64 + t])
          check::Near(lg[i], 0, 0, "prompt/padding prediction logits zero direct loss derivative");
      }
  check::Require(prompt_grad > 1e-8, "completion loss depends on prompt INPUT context");
  auto zero        = first;
  zero["selected"] = nn::zeros(first.at("selected").shape());
  auto empty       = model.Run(zero, true);
  check::Near(empty.at("loss").item(), 0, 0, "zero completion mask loss0");
  for (auto name : model.parameters.names())
    check::Near(
        nn::max(nn::abs(empty.at(name))).item(),
        0,
        0,
        "no completion tokens=>every parameter gradient0");
  Save(root / "zero_mask_outputs.npz", empty);
  auto all        = first;
  all["selected"] = nn::squeeze(bert::LengthMask(first.at("lengths"), 64), {2});
  auto full       = model.Run(all, true);
  check::Require(
      std::abs(full.at("loss").item() - grad.at("loss").item()) > 1e-4,
      "all-token objective different task");
  Save(root / "all_tokens_outputs.npz", full);
  auto one = bert::Rows(test, {0}), original = deployed.Run(one);
  int  prompt = one.at("prompt_lengths").to_vector<int>()[0],
      length  = one.at("lengths").to_vector<int>()[0];
  auto ids    = one.at("input").to_vector<int>();
  for (int q = prompt - 1; q < length; ++q) {
    auto b = one;
    auto x = ids;
    if (q + 1 < length)
      x[q + 1] = (x[q + 1] + 7) % 65;
    b["input"] = nn::from_vector(x, {1, 64});
    auto z     = deployed.Predict(b);
    check::Near(
        replay::MaxAbsDiff(
            nn::slice(z, {0, 0, 0}, {1, q + 1, 67}),
            nn::slice(original.at("logits"), {0, 0, 0}, {1, q + 1, 67})),
        0,
        0,
        "shifted completion target cannot affect its prediction");
  }
  auto future = one;
  auto x      = ids;
  for (int t = 33; t < length; ++t)
    x[t] = (x[t] + 11) % 65;
  future["input"] = nn::from_vector(x, {1, 64});
  auto fut        = deployed.Run(future);
  check::Near(
      replay::MaxAbsDiff(
          nn::slice(original.at("logits"), {0, 0, 0}, {1, 33, 67}),
          nn::slice(fut.at("logits"), {0, 0, 0}, {1, 33, 67})),
      0,
      0,
      "future edits preserve prefix logits");
  check::Near(
      nn::max(nn::abs(nn::slice(original.at("prefix_gradient"), {0, 33, 0}, {1, 31, 67}))).item(),
      0,
      0,
      "future encoded partial0");
  Save(root / "causal_input.npz", one);
  Save(root / "causal_outputs.npz", original);
  Save(root / "future_input.npz", future);
  Save(root / "future_outputs.npz", fut);
  auto padding = bert::Rows(test, {0, 1, 2}), base = deployed.Run(padding);
  auto px = padding.at("input").to_vector<int>(), lens = padding.at("lengths").to_vector<int>();
  for (int b = 0; b < 3; ++b)
    for (int t = lens[b]; t < 64; ++t)
      px[b * 64 + t] = (t + b) % 65;
  padding["input"] = nn::from_vector(px, {3, 64});
  auto padded      = deployed.Run(padding);
  for (const auto& [key, value] : base)
    check::Near(
        replay::MaxAbsDiff(value, padded.at(key)),
        0,
        0,
        "instruction padding fields/partials invariant");
  Save(root / "padding_inputs.npz", padding);
  Save(root / "padding_outputs.npz", padded);
  double fd    = 0;
  int    count = 0;
  for (auto name : model.parameters.names()) {
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
  check::Require(fd < .003, "native PEFT trainable derivatives");
  check::Require(
      replay::SameBits(final, deployed.Arrays()), "single tower/cached retrieval immutable");
  std::ofstream metrics(root / "metrics.json");
  metrics.precision(17);
  metrics << "{\"parameters\":5491,\"variables\":18,\"epochs\":" << epochs
          << ",\"updates\":" << updates << ",\"test_completion_ce\":" << out.at("loss").item()
          << ",\"prompt_input_gradient\":" << prompt_grad << ",\"native_fd_count\":" << count
          << ",\"native_fd_error\":" << fd << "}\n";
  std::cout << "testCompletionCE=" << out.at("loss").item() << "; " << count << " FD max=" << fd
            << "; transfer/completionMask/shift/causal/padding/labelFree/restore/immutable PASS\n";
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
