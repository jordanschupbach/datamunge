// SBERT-style cosine regression and dual-encoder in-batch-negative retrieval.
// CMD: just run-nn sentence_embeddings
// Eager port preserving shared towers, transfer and all source controls.
#include "common/sentence_embeddings.hpp"

#include <fstream>
#include <numeric>
namespace nn = datamunge::nn;
namespace {
using A     = sentence_embeddings::A;
using Model = sentence_embeddings::Model;
void Save(const std::filesystem::path& path, const A& a) {
  nn::save_npz(path.string(), replay::ToNamed(a));
}
A Pool(const A& data, const std::string& split) {
  A b;
  for (auto k : {"input_a", "input_b", "lengths_left", "lengths_right", "score", "source_id"})
    b[k] = data.at(split + "_" + k);
  return b;
}
void Task(const std::string& mode, const A& data, int epochs, const std::filesystem::path& root) {
  std::filesystem::create_directories(root);
  bool  dual = mode == "dual";
  Model model(dual);
  check::Require(
      model.parameters.count() == 2216 && model.parameters.size() == 21,
      "shared2216parameter21tensor encoder");
  Save(root / "fresh_arrays.npz", model.Arrays());
  auto pretrained = replay::TfModels("bert_mlm") / "model_arrays.npz";
  std::filesystem::copy_file(
      pretrained,
      root / "pretrained_arrays.npz",
      std::filesystem::copy_options::overwrite_existing);
  model.Import(replay::ToMap(nn::load_npz(pretrained.string())));
  auto tf     = replay::TfModels("sentence_embeddings") / mode;
  bool parity = std::filesystem::exists(tf / "initial_arrays.npz");
  if (parity)
    replay::LoadInto(model.parameters, tf / "initial_arrays.npz");
  auto compare = [&](const A& out, const std::string& file) {
    if (parity && std::filesystem::exists(tf / file)) {
      double e = replay::MaxDiff(out, tf / file);
      std::cout << mode << " TF " << file << " difference=" << e << std::endl;
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
  auto fit = Pool(data, mode + "_fit"), validation = Pool(data, mode + "_validation"),
       test = Pool(data, mode + "_test");
  auto init = model.Run(test);
  Save(root / "initial_outputs.npz", init);
  compare(init, "initial_outputs.npz");
  nn::BatchOrder order(64, 8, 47507);
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
      auto ids = fit.at("source_id").to_vector<int>();
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
  Save(root / "model_arrays.npz", final);
  Save(root / "final_gradients.npz", grad);
  compare(grad, "final_gradients.npz");
  model.parameters.save((root / "final.npz").string());
  Model deployed(dual, false);
  deployed.parameters.load((root / "final.npz").string());
  check::Require(replay::SameBits(final, deployed.Arrays()), "all21 sentence tensors bitexact");
  auto out = deployed.Run(test);
  Save(root / "outputs.npz", out);
  compare(out, "outputs.npz");
  auto docs    = deployed.Encode(test.at("input_b"), test.at("lengths_right")),
       queries = deployed.Encode(test.at("input_a"), test.at("lengths_left"));
  auto q = queries.to_vector<float>(), d = docs.to_vector<float>(),
       scores = out.at("scores").to_vector<float>();
  std::vector<float> cache(256);
  std::vector<int>   ranking(256);
  for (int i = 0; i < 16; ++i) {
    std::vector<int> ids(16);
    std::iota(ids.begin(), ids.end(), 0);
    for (int j = 0; j < 16; ++j) {
      float dot = 0;
      for (int k = 0; k < 8; ++k)
        dot += q[i * 8 + k] * d[j * 8 + k];
      cache[i * 16 + j] = dot;
      check::Near(dot, scores[i * 16 + j], 2e-6, "cached dot product equals pair graph");
    }
    std::stable_sort(ids.begin(), ids.end(), [&](int a, int b) {
      return cache[i * 16 + a] > cache[i * 16 + b];
    });
    std::copy(ids.begin(), ids.end(), ranking.begin() + i * 16);
    auto one      = bert::Rows(test, {i});
    auto isolated = deployed.Encode(one.at("input_a"), one.at("lengths_left"));
    check::Near(
        replay::MaxAbsDiff(isolated, nn::slice(queries, {i, 0}, {1, 8})),
        0,
        2e-6,
        "independent single sentence encoding");
  }
  Save(
      root / "retrieval.npz",
      {{"queries", queries},
       {"documents", docs},
       {"scores", nn::from_vector(cache, {16, 16})},
       {"ranking", nn::from_vector(ranking, {16, 16})}});
  auto padding = bert::Rows(test, {0, 1, 2});
  for (auto key : {"input_a", "input_b"}) {
    auto x       = padding.at(key).to_vector<int>(),
         lengths = padding.at(std::string(key) == "input_a" ? "lengths_left" : "lengths_right")
                       .to_vector<int>();
    for (int i = 0; i < 3; ++i)
      for (int t = lengths[i]; t < 128; ++t)
        x[i * 128 + t] = 5 + (i + t) % 65;
    padding[key] = nn::from_vector(x, {3, 128});
  }
  auto baseline = deployed.Run(bert::Rows(test, {0, 1, 2})), changed = deployed.Run(padding);
  for (const auto& [k, v] : baseline)
    check::Near(
        replay::MaxAbsDiff(v, changed.at(k)),
        0,
        0,
        "all sentence fields/partials padding invariant");
  Save(root / "padding_inputs.npz", padding);
  Save(root / "padding_outputs.npz", changed);
  auto empty = deployed.Encode(padding.at("input_a"), nn::zeros({3}, nn::DType::Int32));
  check::Near(nn::max(nn::abs(empty)).item(), 0, 0, "empty sentence zero embedding");
  Save(root / "empty.npz", {{"embedding", empty}});
  double fd    = 0;
  int    count = 0;
  for (auto name : model.parameters.names()) {
    auto value  = final.at(name);
    auto values = value.to_vector<float>();
    for (int i : {0, int(values.size() / 2), int(values.size() - 1)}) {
      float old = values[i];
      values[i] = old + .0005f;
      auto plus = model.Run(first, false, {{name, nn::from_vector(values, value.shape())}})
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
  check::Require(fd < .004, "native PEFT trainable derivatives");
  check::Require(
      replay::SameBits(final, deployed.Arrays()), "single tower/cached retrieval immutable");
  std::ofstream metrics(root / "metrics.json");
  metrics.precision(17);
  metrics << "{\"dual\":" << (dual ? "true" : "false")
          << ",\"parameters\":2216,\"variables\":21,\"epochs\":" << epochs
          << ",\"updates\":" << updates << ",\"initial_test_loss\":" << init.at("loss").item()
          << ",\"test_loss\":" << out.at("loss").item() << ",\"native_fd_count\":" << count
          << ",\"native_fd_error\":" << fd << "}\n";
  std::cout << mode << " testLoss=" << out.at("loss").item()
            << " initial=" << init.at("loss").item() << "; " << count << " nativeFD max=" << fd
            << "; shared/transfer/cache/padding/empty/restore/immutable PASS\n";
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
    auto root = paths::OutputDir("models/sentence_embeddings");
    replay::CopyAssets(
        paths::Dataset("sentence_embeddings"),
        root,
        {"data.npz", "manifest.json", "vocabulary.json"});
    auto data = replay::ToMap(nn::load_npz((root / "data.npz").string()));
    for (auto mode : {"sbert", "dual"})
      Task(mode, data, epochs, root / mode);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
