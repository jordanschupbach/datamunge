// Genuine Transformer-XL relative attention and detached segment recurrence.
// CMD: just run-nn transformer_xl
// Eager port preserving segment recurrence, truncated BPTT and source controls.
#include "common/transformer_xl.hpp"

#include <fstream>
namespace nn = datamunge::nn;
namespace {
using Model  = xl::Model;
using A      = xl::A;
using Memory = xl::Memory;
nn::Tensor Step(const nn::Tensor& all, int step) {
  return nn::squeeze(nn::slice(all, {step, 0, 0}, {1, all.size(1), all.size(2)}), {0});
}
void Save(const std::filesystem::path& file, A out) {
  nn::save_npz(file.string(), replay::ToNamed(out));
}
void Save(
    const std::filesystem::path& file,
    const nn::Tensor&            x,
    const nn::Tensor&            y,
    const Memory&                m,
    A                            out) {
  out["input"]   = x;
  out["target"]  = y;
  out["memory0"] = m[0];
  out["memory1"] = m[1];
  Save(file, out);
}
void Compare(const A& out, const std::string& name) {
  auto file = replay::TfModels("transformer_xl") / name;
  if (std::filesystem::exists(file)) {
    double e = replay::MaxDiff(out, file);
    std::cout << "TF " << name << " max difference=" << e << std::endl;
    check::Near(e, 0, 5e-5, "TF XL " + name);
  }
}
double Evaluate(
    Model&                       model,
    const A&                     data,
    const std::string&           split,
    const std::filesystem::path& root,
    const std::string&           prefix) {
  auto   memory   = Model::Empty(4);
  double nll      = 0;
  int    segments = data.at(split + "_input").size(0);
  for (int s = 0; s < segments; ++s) {
    auto x = Step(data.at(split + "_input"), s), y = Step(data.at(split + "_target"), s);
    auto out = model.Run(x, memory, &y);
    nll += out.at("loss").item();
    auto file = prefix + "_step" + std::to_string(s) + ".npz";
    Save(root / file, x, y, memory, out);
    auto with       = out;
    with["input"]   = x;
    with["target"]  = y;
    with["memory0"] = memory[0];
    with["memory1"] = memory[1];
    Compare(with, file);
    memory = Model::Advance(memory, out);
  }
  return nll / segments;
}
void Run(int epochs) {
  auto root = paths::OutputDir("models/transformer_xl");
  replay::CopyAssets(
      paths::Dataset("transformer_xl"), root, {"data.npz", "vocabulary.txt", "manifest.json"});
  auto  data = replay::ToMap(nn::load_npz((root / "data.npz").string()));
  Model model;
  auto  tf = replay::TfModels("transformer_xl");
  if (std::filesystem::exists(tf / "initial_arrays.npz"))
    replay::LoadInto(model.parameters, tf / "initial_arrays.npz");
  Save(root / "initial_arrays.npz", model.Arrays());
  double        initial = Evaluate(model, data, "test", root, "initial_test");
  int           updates = 0;
  std::ofstream history(root / "history.tsv"), batches(root / "batches.tsv");
  history << "epoch\tvalidation_nll\n";
  batches << "epoch\tstep\tstream\tsource_start\tmemory_length\n";
  for (int e = 1; e <= epochs; ++e) {
    auto memory = Model::Empty(4);
    for (int s = 0; s < 16; ++s) {
      auto x = Step(data.at("fit_input"), s), y = Step(data.at("fit_target"), s);
      auto out = model.Run(x, memory, &y, e == 1 && s < 2);
      if (e == 1 && s < 2)
        Save(root / ("initial_train_step" + std::to_string(s) + ".npz"), x, y, memory, out);
      auto start = data.at("fit_start").to_vector<int>();
      for (int b = 0; b < 4; ++b)
        batches << e << '\t' << s << '\t' << b << '\t' << start[b] + s * 16 << '\t'
                << memory[0].size(1) << '\n';
      model.Update(x, y, memory);
      if (e == 1 && s < 2)
        Save(root / ("after_step" + std::to_string(s) + "_arrays.npz"), model.Arrays());
      memory = Model::Advance(memory, out);
      ++updates;
    }
    Save(root / ("epoch" + std::to_string(e) + "_arrays.npz"), model.Arrays());
    double v =
        Evaluate(model, data, "validation", root, "epoch" + std::to_string(e) + "_validation");
    history << e << '\t' << v << '\n';
    std::cout << "epoch=" << e << " validationNLL=" << v << std::endl;
  }
  auto final = model.Arrays();
  Save(root / "model_arrays.npz", final);
  model.parameters.save((root / "final.npz").string());
  Model deployed(false);
  deployed.parameters.load((root / "final.npz").string());
  check::Require(replay::SameBits(final, deployed.Arrays()), "XL checkpoint bitexact");
  double test = Evaluate(deployed, data, "test", root, "test");
  auto   x = Step(data.at("fit_input"), 0), y = Step(data.at("fit_target"), 1);
  auto   empty = Model::Empty(4);
  auto   first = deployed.Run(x, empty);
  auto   mem   = Model::Advance(empty, first);
  x            = Step(data.at("fit_input"), 1);
  Save(root / "final_train_step1.npz", x, y, mem, model.Run(x, mem, &y, true));
  auto out = deployed.Run(x, mem, &y), reset = deployed.Run(x, empty, &y);
  Save(root / "reset.npz", x, y, empty, reset);
  double memory_effect = replay::MaxAbsDiff(out.at("logits"), reset.at("logits"));
  check::Require(memory_effect > 1e-5, "prior segment changes current logits");
  for (int l = 0; l < 2; ++l)
    check::Near(
        nn::max(nn::abs(out.at("memory_gradient" + std::to_string(l)))).item(),
        0,
        0,
        "recurrent memory derivative exactly zero");
  auto   full  = nn::concat({Step(data.at("fit_input"), 0), x}, 1);
  auto   whole = deployed.Run(full, empty);
  double parity =
      replay::MaxAbsDiff(nn::slice(whole.at("logits"), {0, 16, 0}, {4, 16, 65}), out.at("logits"));
  check::Require(parity < 2e-5, "full32 vs two segments full past parity");
  Save(root / "full32.npz", {{"input", full}, {"logits", whole.at("logits")}});
  auto changed = x.to_vector<int>();
  for (int b = 0; b < 4; ++b)
    for (int t = 8; t < 16; ++t)
      changed[b * 16 + t] = (changed[b * 16 + t] + 11) % 65;
  auto changed_input = nn::from_vector(changed, {4, 16});
  auto causal        = deployed.Run(changed_input, mem);
  check::Near(
      replay::MaxAbsDiff(
          nn::slice(causal.at("logits"), {0, 0, 0}, {4, 8, 65}),
          nn::slice(out.at("logits"), {0, 0, 0}, {4, 8, 65})),
      0,
      0,
      "future tokens cannot alter prefix");
  Save(root / "future.npz", {{"input", changed_input}, {"logits", causal.at("logits")}});
  for (int b = 0; b < 4; ++b) {
    auto   one = nn::slice(x, {b, 0}, {1, 16});
    Memory isolated;
    for (int l = 0; l < 2; ++l)
      isolated.push_back(nn::slice(mem[l], {b, 0, 0}, {1, 16, 8}));
    auto z = deployed.Run(one, isolated);
    check::Near(
        replay::MaxAbsDiff(z.at("logits"), nn::slice(out.at("logits"), {b, 0, 0}, {1, 16, 65})),
        0,
        2e-6,
        "batch streams independent");
  }
  auto   grad  = model.Run(x, mem, &y, true);
  double fd    = 0;
  int    count = 0;
  for (auto name : model.parameters.names()) {
    auto value  = final.at(name);
    auto values = value.to_vector<float>();
    for (int i : {0, int(values.size() / 2), int(values.size() - 1)}) {
      float old = values[i];
      values[i] = old + .002f;
      auto plus = model.Run(x, mem, &y, false, {{name, nn::from_vector(values, value.shape())}})
                      .at("loss")
                      .item();
      values[i]  = old - .002f;
      auto minus = model.Run(x, mem, &y, false, {{name, nn::from_vector(values, value.shape())}})
                       .at("loss")
                       .item();
      values[i] = old;
      fd = std::max(fd, std::abs((plus - minus) / .004 - grad.at(name).to_vector<float>()[i]));
      ++count;
    }
  }
  check::Require(fd < .003, "XL native parameter finite differences");
  std::ifstream      vf(root / "vocabulary.txt");
  std::string        vocab((std::istreambuf_iterator<char>(vf)), {});
  std::vector<int>   codes(129);
  std::vector<float> scores(128 * 65);
  codes[0]                 = vocab.find('\n');
  auto          generation = Model::Empty(1);
  std::ofstream sample(root / "sample.txt");
  sample << '\n';
  for (int t = 0; t < 128; ++t) {
    auto token  = nn::from_vector(std::vector<int>{codes[t]}, {1, 1});
    auto z      = deployed.Run(token, generation);
    auto logits = z.at("logits").to_vector<float>();
    int  best   = 0;
    for (int c = 0; c < 65; ++c) {
      scores[t * 65 + c] = logits[c];
      if (logits[c] > logits[best])
        best = c;
    }
    codes[t + 1] = best;
    sample << vocab.at(best);
    generation = Model::Advance(generation, z);
    check::Require(generation[0].size(1) == std::min(t + 1, 16), "bounded generation memory");
  }
  Save(
      root / "generation.npz",
      {{"codes", nn::from_vector(codes, {129})}, {"logits", nn::from_vector(scores, {128, 65})}});
  check::Require(replay::SameBits(final, deployed.Arrays()), "XL inference immutable");
  std::vector<double> unigram(65, 1);
  for (int i : data.at("fit_target").to_vector<int>())
    ++unigram[i];
  double baseline = 0;
  auto   testids  = data.at("test_target").to_vector<int>();
  for (int i : testids)
    baseline -= std::log(unigram[i] / (1024. + 65));
  baseline /= testids.size();
  std::ofstream metrics(root / "metrics.json");
  metrics.precision(17);
  metrics << "{\"parameters\":" << model.parameters.count()
          << ",\"variables\":" << model.parameters.size() << ",\"epochs\":" << epochs
          << ",\"updates\":" << updates << ",\"initial_test_nll\":" << initial
          << ",\"test_nll\":" << test << ",\"fit_unigram_nll\":" << baseline
          << ",\"full_segment_error\":" << parity << ",\"memory_effect\":" << memory_effect
          << ",\"native_fd_count\":" << count << ",\"native_fd_error\":" << fd << "}\n";
  std::cout << "testNLL=" << test << " initial=" << initial << " unigram=" << baseline
            << "; full/segment error=" << parity << " memory effect=" << memory_effect << "; "
            << count << " FD max=" << fd << "; restore/immutable/generation PASS\n";
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
    Run(epochs);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
