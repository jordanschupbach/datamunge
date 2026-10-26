// XLNet-style permutation LM: two streams, target exclusion and segment memory.
// CMD: just run-nn xlnet
// Eager port preserving physical token order and all source controls.
#include "common/xlnet.hpp"

#include <fstream>
namespace nn = datamunge::nn;
namespace {
using Model  = xlnet::Model;
using A      = xlnet::A;
using Memory = xlnet::Memory;
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
  out["rank"]  = y;
  out["memory0"] = m[0];
  out["memory1"] = m[1];
  Save(file, out);
}
void Compare(const A& out, const std::string& name) {
  auto file = replay::TfModels("xlnet") / name;
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
    auto x = Step(data.at(split + "_input"), s), y = Step(data.at(split + "_rank0"), s);
    auto out = model.Run(x, memory, y, &x);
    nll += out.at("loss").item();
    auto file = prefix + "_step" + std::to_string(s) + ".npz";
    Save(root / file, x, y, memory, out);
    auto with       = out;
    with["input"]   = x;
    with["rank"]    = y;
    with["memory0"] = memory[0];
    with["memory1"] = memory[1];
    Compare(with, file);
    memory = Model::Advance(memory, out);
  }
  return nll / segments;
}
void Run(int epochs) {
  auto root = paths::OutputDir("models/xlnet");
  replay::CopyAssets(
      paths::Dataset("xlnet"), root, {"data.npz", "vocabulary.txt", "manifest.json"});
  auto data = replay::ToMap(nn::load_npz((root / "data.npz").string()));
  check::Require(
      data.count("fit_rank" + std::to_string(epochs)), "prepare_xlnet.py --epochs=N first");
  Model model;
  check::Require(
      model.parameters.count() == 1873 && model.parameters.size() == 31,
      "XLNet tied head/truly shared streams");
  auto tf = replay::TfModels("xlnet");
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
      auto x = Step(data.at("fit_input"), s), y = Step(data.at("fit_rank" + std::to_string(e)), s);
      auto out = model.Run(x, memory, y, &x, e == 1 && s < 2);
      if (e == 1 && s < 2)
        Save(root / ("initial_train_step" + std::to_string(s) + ".npz"), x, y, memory, out);
      auto start = data.at("fit_start").to_vector<int>();
      for (int b = 0; b < 4; ++b)
        batches << e << '\t' << s << '\t' << b << '\t' << start[b] + s * 16 << '\t'
                << memory[0].size(1) << '\n';
      model.Update(x, x, memory, y);
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
  double test  = Evaluate(deployed, data, "test", root, "test");
  auto   empty = Model::Empty(4);
  auto   x0 = Step(data.at("fit_input"), 0), r0 = Step(data.at("fit_rank1"), 0);
  auto   first = deployed.Run(x0, empty, r0);
  auto   mem   = Model::Advance(empty, first);
  auto   x = Step(data.at("fit_input"), 1), rank = Step(data.at("fit_rank1"), 1);
  auto   result = model.Run(x, mem, rank, &x, true);
  Save(root / "final_train_step1.npz", x, rank, mem, result);
  double context_effect = 0;
  auto   ids = x.to_vector<int>(), ranks = rank.to_vector<int>();
  auto   original_logits = result.at("logits").to_vector<float>();
  for (int pos = 0; pos < 16; ++pos) {
    auto changed = ids;
    for (int b = 0; b < 4; ++b)
      changed[b * 16 + pos] = (changed[b * 16 + pos] + 17) % 65;
    auto input = nn::from_vector(changed, {4, 16});
    auto z     = deployed.Run(input, mem, rank);
    Save(root / ("intervention" + std::to_string(pos) + ".npz"), input, rank, mem, z);
    auto logits = z.at("logits").to_vector<float>();
    for (int b = 0; b < 4; ++b)
      for (int q = 0; q < 16; ++q)
        for (int c = 0; c < 65; ++c) {
          double delta =
              std::abs(logits[(b * 16 + q) * 65 + c] - original_logits[(b * 16 + q) * 65 + c]);
          if (ranks[b * 16 + pos] >= ranks[b * 16 + q])
            check::Near(delta, 0, 0, "no own/later-rank target leakage");
          else
            context_effect = std::max(context_effect, delta);
        }
  }
  check::Require(context_effect > 1e-4, "earlier permutation context affects prediction");
  auto qg = result.at("query_gradient").to_vector<float>();
  for (int b = 0; b < 4; ++b)
    for (int t = 0; t < 16; ++t)
      if (ranks[b * 16 + t] >= ranks[b * 16 + 7])
        for (int c = 0; c < 65; ++c)
          check::Near(qg[(b * 16 + t) * 65 + c], 0, 0, "exact zero own/later query derivative");
  std::vector<int> aware(32), arank(32);
  for (int b = 0; b < 2; ++b) {
    int r = 0;
    for (int t = 0; t < 16; ++t) {
      aware[b * 16 + t] = ids[t];
      if (t != 7 && t != 8)
        arank[b * 16 + t] = r++;
    }
    for (int t = 0; t < 16; ++t)
      if (t != 7 && t != 8 && arank[b * 16 + t] >= 8)
        arank[b * 16 + t] += 2;
    arank[b * 16 + 7] = b ? 9 : 8;
    arank[b * 16 + 8] = b ? 8 : 9;
  }
  auto aware_x = nn::from_vector(aware, {2, 16}), aware_rank = nn::from_vector(arank, {2, 16});
  auto awareness = deployed.Run(aware_x, Model::Empty(2), aware_rank);
  Save(root / "target_aware.npz", aware_x, aware_rank, Model::Empty(2), awareness);
  Save(root / "reset.npz", x, rank, empty, deployed.Run(x, empty, rank, &x));
  auto   grad  = result;
  double fd    = 0;
  int    count = 0;
  for (auto name : model.parameters.names()) {
    auto value  = final.at(name);
    auto values = value.to_vector<float>();
    for (int i : {0, int(values.size() / 2), int(values.size() - 1)}) {
      float old = values[i];
      values[i] = old + .002f;
      auto plus =
          model.Run(x, mem, rank, &x, false, {{name, nn::from_vector(values, value.shape())}})
              .at("loss")
              .item();
      values[i] = old - .002f;
      auto minus =
          model.Run(x, mem, rank, &x, false, {{name, nn::from_vector(values, value.shape())}})
              .at("loss")
              .item();
      values[i] = old;
      fd = std::max(fd, std::abs((plus - minus) / .004 - grad.at(name).to_vector<float>()[i]));
      ++count;
    }
  }
  check::Require(fd < .003, "XL native parameter finite differences");
  for (int row = 0; row < 2; ++row) {
    auto original = nn::slice(Step(data.at("test_input"), 0), {row, 0}, {1, 16}),
         r        = nn::slice(Step(data.at("test_rank0"), 0), {row, 0}, {1, 16});
    auto rv = r.to_vector<int>(), known = original.to_vector<int>();
    for (int t = 0; t < 16; ++t)
      if (rv[t] >= 8)
        known[t] = 0;
    std::vector<int>   trace(128), positions(8);
    std::vector<float> scores(8 * 65);
    for (int step = 0; step < 8; ++step) {
      int pos = -1;
      for (int t = 0; t < 16; ++t)
        if (rv[t] == step + 8)
          pos = t;
      check::Require(pos >= 0, "infill permutation target exists");
      positions[step] = pos;
      std::copy(known.begin(), known.end(), trace.begin() + step * 16);
      auto z = deployed.Run(nn::from_vector(known, {1, 16}), Model::Empty(1), r)
                   .at("logits")
                   .to_vector<float>();
      int best = 0;
      for (int c = 0; c < 65; ++c) {
        scores[step * 65 + c] = z[pos * 65 + c];
        if (z[pos * 65 + c] > z[pos * 65 + best])
          best = c;
      }
      known[pos] = best;
    }
    Save(
        root / ("infill" + std::to_string(row) + ".npz"),
        {{"original", original},
         {"rank", r},
         {"result", nn::from_vector(known, {1, 16})},
         {"trace", nn::from_vector(trace, {8, 16})},
         {"positions", nn::from_vector(positions, {8})},
         {"logits", nn::from_vector(scores, {8, 65})}});
  }
  check::Require(replay::SameBits(final, deployed.Arrays()), "XLNet inference immutable");
  std::vector<double> unigram(65, 1);
  auto                fitids = data.at("fit_input").to_vector<int>();
  for (int id : fitids)
    ++unigram[id];
  double baseline = 0;
  int    held = 0, correct = 0;
  for (int step = 0; step < 4; ++step) {
    auto z = replay::ToMap(
        nn::load_npz((root / ("test_step" + std::to_string(step) + ".npz")).string()));
    auto r = z.at("rank").to_vector<int>(), input = z.at("input").to_vector<int>();
    auto logits = z.at("logits").to_vector<float>();
    for (int i = 0; i < 64; ++i)
      if (r[i] >= 8) {
        baseline -= std::log(unigram[input[i]] / (fitids.size() + 65.));
        ++held;
        int best = 0;
        for (int c = 1; c < 65; ++c)
          if (logits[i * 65 + c] > logits[i * 65 + best])
            best = c;
        correct += best == input[i];
      }
  }
  baseline /= held;
  std::ofstream metrics(root / "metrics.json");
  metrics.precision(17);
  metrics << "{\"parameters\":1873,\"variables\":31,\"epochs\":" << epochs
          << ",\"updates\":" << updates << ",\"initial_test_partial_ce\":" << initial
          << ",\"test_partial_ce\":" << test << ",\"fit_unigram_partial_ce\":" << baseline
          << ",\"selected_accuracy\":" << double(correct) / held
          << ",\"context_effect\":" << context_effect << ",\"native_fd_count\":" << count
          << ",\"native_fd_error\":" << fd << "}\n";
  std::cout << "testPartialCE=" << test << " initial=" << initial << " unigram=" << baseline
            << " selectedAccuracy=" << double(correct) / held << " contextEffect=" << context_effect
            << "; " << count << " FD max=" << fd
            << "; twoStreams/noLeak/memory/infill/restore/immutable PASS\n";
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
