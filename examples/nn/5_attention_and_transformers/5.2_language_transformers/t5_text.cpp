// T5 span-denoising pretraining and sentiment expressed with the same text
// decoder.
//
// CMD: just run-nn t5_text
// Eager port preserving relative RMS attention, tied text head and all controls.
#include "common/t5_text.hpp"

#include <fstream>
namespace nn = datamunge::nn;
namespace {
using A     = t5::A;
using Model = t5::Model;
void Save(const std::filesystem::path& file, const A& a) {
  nn::save_npz(file.string(), replay::ToNamed(a));
}
void Run(const std::string& name, int epochs) {
  auto root = paths::OutputDir("models/t5_text/" + name), tf = replay::TfModels("t5_text") / name;
  replay::CopyAssets(
      paths::Dataset("t5_text/" + name), root, {"data.npz", "manifest.json", "vocabulary.json"});
  auto           data = replay::ToMap(nn::load_npz((root / "data.npz").string()));
  std::vector<A> pools;
  for (auto split : {"fit", "validation", "test"}) {
    A b;
    for (auto key : {"input", "target", "teacher", "input_lengths", "output_lengths", "source_ids"})
      b[key] = data.at(std::string(split) + "_" + key);
    pools.push_back(b);
  }
  Model model;
  int   expected = 2040;
  check::Require(
      model.parameters.count() == expected && model.parameters.size() == 26,
      "shared table,bias-free attention/FF,seven RMS scales,two32x2relative tables");
  Save(root / "random_arrays.npz", model.Arrays());
  if (name == "sentiment") {
    auto source = paths::OutputDir("models/t5_text/denoise");
    model.parameters.load((source / "final.npz").string());
    auto weights = replay::ToMap(nn::load_npz((source / "model_arrays.npz").string()));
    check::Require(
        replay::SameBits(weights, model.Arrays()), "all pretrained T5 tensors transferred exactly");
  }
  bool parity = std::filesystem::exists(tf / "initial_arrays.npz");
  if (parity)
    replay::LoadInto(model.parameters, tf / "initial_arrays.npz");
  auto compare = [&](const A& out, const std::string& file) {
    if (parity && std::filesystem::exists(tf / file)) {
      double error = replay::MaxDiff(out, tf / file);
      std::cout << name << " TF " << file << " difference=" << error << std::endl;
      check::Near(error, 0, 2e-4, "T5 TF " + file);
    }
  };
  Save(root / "initial_arrays.npz", model.Arrays());
  auto firstout = model.Run(pools[2]);
  Save(root / "initial_outputs.npz", firstout);
  compare(firstout, "initial_outputs.npz");
  nn::BatchOrder order(pools[0].at("input").size(0), 8, 46807);
  std::ofstream  history(root / "history.tsv"), batches(root / "batches.tsv");
  history << "epoch\tvalidation_loss\n";
  batches << "epoch\tstep\trow\n";
  int steps = 0;
  A   first;
  for (int e = 1; e <= epochs; ++e) {
    order.begin_epoch();
    for (;;) {
      auto rows = order.next();
      if (rows.empty())
        break;
      auto b = bert::Rows(pools[0], rows);
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
        batches << e << '\t' << steps << '\t' << r << '\n';
      ++steps;
    }
    auto v = model.Run(pools[1]);
    history << e;
    for (auto k : {"loss"})
      history << '\t' << v.at(k).item();
    history << '\n';
    std::cout << name << " epoch=" << e << " validationTokenCE=" << v.at("loss").item()
              << " total=" << v.at("loss").item() << std::endl;
  }
  auto g = model.Run(first, true);
  A    gradients;
  for (auto key : model.parameters.names())
    gradients[key] = g.at(key);
  Save(root / "final_gradients.npz", gradients);
  compare(gradients, "final_gradients.npz");
  model.parameters.save((root / "final.npz").string());
  Model deployed(false);
  deployed.parameters.load((root / "final.npz").string());
  auto final = deployed.Arrays();
  check::Require(replay::SameBits(model.Arrays(), final), "compact bitexact restore");
  auto out = deployed.Run(pools[2]);
  Save(root / "model_arrays.npz", final);
  Save(root / "outputs.npz", out);
  compare(out, "outputs.npz");
  nn::Tensor trace;
  auto       greedy = deployed.Greedy(pools[2], &trace);
  check::Near(
      replay::MaxAbsDiff(
          nn::cast(greedy, nn::DType::Float32),
          nn::cast(deployed.Greedy(pools[2]), nn::DType::Float32)),
      0,
      0,
      "deterministic text generation");
  Save(root / "decoded.npz", {{"greedy", greedy}, {"cross_attention", trace}});
  auto b = bert::Rows(pools[2], {0, 1}), base = deployed.Run(b), corrupt = bert::Rows(b, {0, 1});
  corrupt["target"]         = nn::full({2, 48}, 17, nn::DType::Int32);
  corrupt["teacher"]        = nn::full({2, 48}, 17, nn::DType::Int32);
  corrupt["output_lengths"] = nn::zeros({2}, nn::DType::Int32);
  check::Require(
      replay::SameBits({{"greedy", deployed.Greedy(b)}}, {{"greedy", deployed.Greedy(corrupt)}}),
      "generation excludes gold tokens/target lengths");
  auto padded = bert::Rows(b, {0, 1});
  for (auto key : {"input", "teacher"}) {
    auto x       = padded.at(key).to_vector<int>(),
         lengths = padded.at(std::string(key) == "input" ? "input_lengths" : "output_lengths")
                       .to_vector<int>();
    for (int n = 0; n < 2; ++n)
      for (int t = lengths[n]; t < 48; ++t)
        x[n * 48 + t] = 17;
    padded[key] = nn::from_vector(x, {2, 48});
  }
  auto po = deployed.Run(padded);
  for (const auto& [k, v] : base)
    check::Near(replay::MaxAbsDiff(v, po.at(k)), 0, 0, "all T5 paths/partials padding invariant");
  Save(root / "padding_inputs.npz", padded);
  Save(root / "padding_outputs.npz", po);
  auto future   = bert::Rows(b, {0, 1});
  auto teachers = future.at("teacher").to_vector<int>();
  for (int n = 0; n < 2; ++n)
    for (int t = 2; t < 48; ++t)
      teachers[n * 48 + t] = 17;
  future["teacher"] = nn::from_vector(teachers, {2, 48});
  auto   fo         = deployed.Run(future);
  double later      = 0;
  auto   x = base.at("logits").to_vector<float>(), y = fo.at("logits").to_vector<float>(),
       pg = base.at("prefix_gradient").to_vector<float>();
  for (int n = 0; n < 2; ++n)
    for (int t = 0; t < 48; ++t)
      for (int c = 0; c < 72; ++c) {
        int i = (n * 48 + t) * 72 + c;
        if (t < 2)
          check::Near(x[i], y[i], 0, "future token cannot affect earlier logits");
        else {
          later = std::max(later, double(std::abs(x[i] - y[i])));
          check::Near(pg[i], 0, 0, "prefix derivative into future teacher0");
        }
      }
  check::Require(later > 1e-5, "future perturbation affects later logits");
  Save(root / "future_inputs.npz", future);
  Save(root / "future_outputs.npz", fo);
  auto empty             = bert::Rows(b, {0, 1});
  empty["input_lengths"] = nn::zeros({2}, nn::DType::Int32);
  auto eo                = deployed.Run(empty);
  for (auto key : {"memory", "encoder_attention", "cross_attention", "partial0", "partial2"})
    check::Near(nn::max(nn::abs(eo.at(key))).item(), 0, 0, "empty source paths/gradients0");
  Save(root / "empty_inputs.npz", empty);
  Save(root / "empty_outputs.npz", eo);
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
  int  total = 0, correct = 0, exact = 0;
  auto z         = out.at("logits").to_vector<float>();
  auto generated = greedy.to_vector<int>(), target = pools[2].at("target").to_vector<int>(),
       lengths = pools[2].at("output_lengths").to_vector<int>();
  for (int n = 0; n < int(lengths.size()); ++n) {
    bool match = true;
    for (int t = 0; t < 48; ++t) {
      match &= generated[n * 48 + t] == target[n * 48 + t];
      if (t < lengths[n]) {
        int best = 0;
        for (int c = 1; c < 72; ++c)
          if (z[(n * 48 + t) * 72 + c] > z[(n * 48 + t) * 72 + best])
            best = c;
        correct += best == target[n * 48 + t];
        ++total;
      }
    }
    exact += match;
  }
  std::ofstream meta(root / "metrics.json");
  meta.precision(17);
  meta << "{\"parameters\":2040,\"variables\":26,\"epochs\":" << epochs << ",\"updates\":" << steps
       << ",\"test_token_ce\":" << out.at("loss").item()
       << ",\"teacher_token_accuracy\":" << double(correct) / total
       << ",\"greedy_exact_accuracy\":" << double(exact) / lengths.size()
       << ",\"native_fd_count\":" << fd_count << ",\"native_fd_error\":" << fd
       << ",\"nonvacuous_later_logit_effect\":" << later << "}\n";
  std::cout << name << " testCE=" << out.at("loss").item()
            << " teacherAccuracy=" << double(correct) / total << " greedyExact=" << exact << '/'
            << lengths.size() << "; " << fd_count << "nativeFD max=" << fd
            << "; relativeRMS/tiedTextHead/causal/padding/labelFree/restore/immutable PASS\n";
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
    for (auto task : {"denoise", "sentiment"})
      Run(task, epochs);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
