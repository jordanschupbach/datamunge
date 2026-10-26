// BERT fine-tuning for sentiment, NER and extractive QA.
//
// CMD: just run-nn bert_finetune
// Eager port preserving encoder transfer, task heads and all source controls.
#include "common/bert_finetune.hpp"

#include <fstream>
namespace nn = datamunge::nn;
namespace {
using A     = finebert::A;
using Model = finebert::Model;
void Save(const std::filesystem::path& file, const A& a) {
  nn::save_npz(file.string(), replay::ToNamed(a));
}
void Run(const std::string& name, int epochs) {
  auto root = paths::OutputDir("models/bert_finetune/" + name),
       tf   = replay::TfModels("bert_finetune") / name;
  replay::CopyAssets(paths::Dataset("bert_finetune/" + name), root, {"data.npz", "manifest.json"});
  auto           data = replay::ToMap(nn::load_npz((root / "data.npz").string()));
  std::vector<A> pools;
  for (auto split : {"fit", "validation", "test"}) {
    A b;
    for (auto key : {"input", "segments", "lengths", "labels", "weights", "source_ids"})
      b[key] = data.at(std::string(split) + "_" + key);
    pools.push_back(b);
  }
  Model model(name);
  int   expected = name == "classification" ? 2306 : name == "ner" ? 2333 : 3258;
  check::Require(model.parameters.count() == expected, "transferred encoder plus task head count");
  auto source =
      replay::ToMap(nn::load_npz((replay::TfModels("bert_mlm") / "model_arrays.npz").string()));
  Save(root / "source_arrays.npz", source);
  auto random = model.Arrays();
  Save(root / "random_arrays.npz", random);
  model.ImportEncoder(source);
  auto initial = model.Arrays();
  for (const auto& [key, value] : source)
    if (initial.count(key)) {
      auto x = initial.at(key).to_vector<float>(), y = value.to_vector<float>();
      for (size_t i = 0; i < y.size(); ++i)
        check::Near(x[i], y[i], 0, "pretrained values exact transfer");
    }
  auto position = initial.at("position/W").to_vector<float>(),
       fresh    = random.at("position/W").to_vector<float>();
  for (size_t i = 256; i < position.size(); ++i)
    check::Near(position[i], fresh[i], 0, "extended positions retain fresh initialization");
  bool parity = std::filesystem::exists(tf / "initial_arrays.npz");
  if (parity)
    replay::LoadInto(model.parameters, tf / "initial_arrays.npz");
  initial      = model.Arrays();
  auto compare = [&](const A& out, const std::string& file) {
    if (parity && std::filesystem::exists(tf / file)) {
      auto actual = out, expected = replay::ToMap(nn::load_npz((tf / file).string()));
      // QA normalizes start/end logits over positions. Per-row constant
      // shifts (including final-norm beta/head bias null derivatives) cancel.
      // Compare those coordinates modulo their exact likelihood symmetry.
      if (name == "qa" && actual.count("logits")) {
        actual["logits"]   = nn::log_softmax(nn::transpose(actual.at("logits"), {0, 2, 1}), -1);
        expected["logits"] = nn::log_softmax(nn::transpose(expected.at("logits"), {0, 2, 1}), -1);
        auto mask          = bert::LengthMask(pools[2].at("lengths"), model.time);
        for (auto* values : {&actual, &expected})
          if (values->count("hidden")) {
            auto h              = values->at("hidden");
            auto mean           = nn::sum(h, {1}, true) / nn::sum(mask, {1}, true);
            (*values)["hidden"] = (h - mean) * mask;
          }
      }
      double error = 0;
      for (const auto& [key, value] : actual)
        error = std::max(error, replay::MaxAbsDiff(value, expected.at(key)));
      std::cout << name << " TF " << file << " difference=" << error << std::endl;
      check::Near(error, 0, 2e-4, "fine-tuning TF " + file);
    }
  };
  Save(root / "initial_arrays.npz", model.Arrays());
  auto firstout = model.Run(pools[2]);
  Save(root / "initial_outputs.npz", firstout);
  compare(firstout, "initial_outputs.npz");
  nn::BatchOrder order(64, 8, 46507);
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
    std::cout << name << " epoch=" << e << " validationLoss=" << v.at("loss").item()
              << " total=" << v.at("loss").item() << std::endl;
  }
  auto g = model.Run(first, true);
  A    gradients;
  for (auto key : model.parameters.names())
    gradients[key] = g.at(key);
  Save(root / "final_gradients.npz", gradients);
  compare(gradients, "final_gradients.npz");
  model.parameters.save((root / "final.npz").string());
  Model deployed(name, false);
  deployed.parameters.load((root / "final.npz").string());
  auto final = deployed.Arrays();
  check::Require(replay::SameBits(model.Arrays(), final), "compact bitexact restore");
  auto out = deployed.Run(pools[2]);
  Save(root / "model_arrays.npz", final);
  Save(root / "outputs.npz", out);
  compare(out, "outputs.npz");
  for (auto key : {"token/W", "attention/v/W", "ff1/W"})
    check::Require(
        !replay::SameBits({{key, initial.at(key)}}, {{key, final.at(key)}}),
        "pretrained encoder genuinely fine-tuned");
  auto             z = out.at("logits").to_vector<float>();
  std::vector<int> pred;
  auto             weights = pools[2].at("weights").to_vector<float>();
  for (int b = 0; b < 16; ++b) {
    if (name == "classification")
      pred.push_back(z[b * 2 + 1] > z[b * 2]);
    else if (name == "ner") {
      for (int t = 0; t < model.time; ++t) {
        int best = 0;
        for (int c = 1; c < 13; ++c)
          if (z[(b * model.time + t) * 13 + c] > z[(b * model.time + t) * 13 + best])
            best = c;
        pred.push_back(best);
      }
    } else {
      double best  = -INFINITY;
      int    start = -1, end = -1;
      for (int i = 0; i < model.time; ++i)
        if (weights[b * model.time + i] > 0)
          for (int j = i; j < model.time; ++j)
            if (weights[b * model.time + j] > 0) {
              double score =
                  double(z[(b * model.time + i) * 2]) + double(z[(b * model.time + j) * 2 + 1]);
              if (score > best) {
                best  = score;
                start = i;
                end   = j;
              }
            }
      check::Require(start >= 0 && end >= start, "legal context-only span");
      pred.push_back(start);
      pred.push_back(end);
    }
  }
  nn::Shape predshape = name == "classification" ? nn::Shape{16}
                        : name == "ner"          ? nn::Shape{16, model.time}
                                                 : nn::Shape{16, 2};
  Save(root / "predictions.npz", {{"prediction", nn::from_vector(pred, predshape)}});
  auto b = bert::Rows(pools[2], {0, 1}), base = deployed.Run(b), padded = bert::Rows(b, {0, 1});
  auto lengths = padded.at("lengths").to_vector<int>();
  for (auto key : {"input", "segments"}) {
    auto x = padded.at(key).to_vector<int>();
    for (int n = 0; n < 2; ++n)
      for (int t = lengths[n]; t < model.time; ++t)
        x[n * model.time + t] = std::string(key) == "input" ? 17 : 1;
    padded[key] = nn::from_vector(x, {2, model.time});
  }
  auto po = deployed.Run(padded);
  for (const auto& [k, v] : base)
    check::Near(
        replay::MaxAbsDiff(v, po.at(k)), 0, 0, "all task paths/input partials padding invariant");
  Save(root / "padding_inputs.npz", padded);
  Save(root / "padding_outputs.npz", po);
  auto bad      = b;
  bad["labels"] = nn::zeros(b.at("labels").shape(), nn::DType::Int32);
  check::Near(
      replay::MaxAbsDiff(deployed.Predict(b), deployed.Predict(bad)),
      0,
      0,
      "task inference no labels");
  if (name == "ner") {
    auto ignored = bert::Rows(b, {0, 1});
    auto labels  = ignored.at("labels").to_vector<int>();
    auto w       = ignored.at("weights").to_vector<float>();
    for (size_t i = 0; i < labels.size(); ++i)
      if (!w[i])
        labels[i] = 12;
    ignored["labels"] = nn::from_vector(labels, {2, model.time});
    check::Near(
        deployed.Run(ignored).at("loss").item(),
        base.at("loss").item(),
        0,
        "ignored wordpiece labels no effect");
  }
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
  std::ofstream meta(root / "metrics.json");
  meta.precision(17);
  meta << "{\"parameters\":" << expected << ",\"variables\":" << model.parameters.size()
       << ",\"epochs\":" << epochs << ",\"updates\":" << steps << ",\"time\":" << model.time
       << ",\"test_loss\":" << out.at("loss").item() << ",\"native_fd_count\":" << fd_count
       << ",\"native_fd_error\":" << fd << "}\n";
  std::cout
      << name << " testLoss=" << out.at("loss").item() << " " << fd_count << "nativeFD max=" << fd
      << "; 21encoderTransfers/newPositions/taskGrad/padding/labelFree/restore/immutable PASS\n";
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
    for (auto task : {"classification", "ner", "qa"})
      Run(task, epochs);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
