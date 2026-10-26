// CMD: just run-nn bert_mlm
// Compact bidirectional BERT masked-language pretraining on real pinned text.
//
// Port of tflow's bert_mlm (common/{nn,ml}/bert_mlm.h -> examples/nn/common/bert_mlm.hpp).
// Data: the pinned bert_mlm split (paths::Dataset("bert_mlm")/data.npz: 128
// fitting, 32 validation, 32 test sentence pairs of 32 tokens with segment
// ids, lengths and the BERT 80/10/10 corruption already applied; `selected`
// marks the positions the MLM loss scores). One post-norm encoder block, tied
// MLM head, 1606 parameters in 26 tensors, Adam .003, batch 8 from
// nn::BatchOrder(128, 8, 46407) (tflow's order), three epochs = 48 updates.
//
// Checks (as tflow): parameter count; bit-exact checkpoint restore; exact
// padding invariance of every output and input partial (padded tokens and
// segments changed); labels at unselected positions cannot change the loss and
// prediction needs no labels; no selected position gives zero loss and zero
// partials; changing right context changes earlier predictions (bidirectional);
// whole-model float32 central differences (78 coordinates) < .004; inference
// leaves the weights unchanged. Reports the selected-position test accuracy.
//
// Deviations: weights replay TF's saved initial arrays by default (init=auto:
// tflow's output/models/bert_mlm/initial_arrays.npz when present, then every
// record is compared with TF's; init=nn: nn's generator, statistical parity).
// The checkpoint is final.npz. Options: epochs=3 out=DIR tf=DIR init=auto|tf|nn
// threads=4 verbose=0. Output: output/nn/models/bert_mlm/.
#include "common/bert_mlm.hpp"
#include "common/transformer_lm.hpp"

namespace {

using bert::Batch;
using lm::Tensor;
namespace nn = datamunge::nn;
namespace fs = std::filesystem;

nn::NamedTensors Named(const Batch& b) { return {b.begin(), b.end()}; }

/// Applies f(row, t, length, values...) to int32 [N, 32] fields of a batch.
Tensor Edited(const Tensor& x, const std::function<void(std::vector<std::int32_t>&)>& f) {
  auto v = x.to_vector<std::int32_t>();
  f(v);
  return nn::from_vector(v, x.shape());
}

int Run(const lm::Options& o) {
  const auto     started = std::chrono::steady_clock::now();
  const fs::path root = o.out, assets = paths::Dataset("bert_mlm");
  check::Require(fs::exists(assets / "data.npz"), "bert_mlm dataset missing: " + assets.string());
  for (auto f : {"data.npz", "manifest.json", "vocabulary.json"})
    if (fs::exists(assets / f)) fs::copy_file(assets / f, root / f, fs::copy_options::overwrite_existing);
  const auto         data = nn::load_npz((root / "data.npz").string());
  std::vector<Batch> pools;
  for (std::string split : {"fit", "validation", "test"}) {
    Batch b;
    for (std::string k : {"input", "target", "segments", "selected", "lengths", "source_start", "actions"})
      b[k] = attn::Named(data, split + "_" + k);
    pools.push_back(b);
  }
  bert::BertMlmModel model;
  check::Require(model.store.count() == 1606 && model.store.size() == 26, "1606 tied BERT parameters26 tensors");
  const fs::path init = lm::InitArrays(o);
  if (!init.empty()) {
    lm::LoadArrays(model.store, init);
    model.Train();
    std::cout << "replaying TF's initial arrays " << init.string() << "\n";
  } else {
    std::cout << "weights from nn's generator (statistical parity with TF only)\n";
  }
  nn::save_npz((root / "initial_arrays.npz").string(), model.Arrays());
  nn::save_npz((root / "initial_outputs.npz").string(), Named(model.Evaluate(pools[2])));
  nn::BatchOrder order(128, 8, 46407);
  std::ofstream  history(root / "history.tsv"), batches(root / "batches.tsv");
  history << "epoch\tvalidation_mlm_ce\n";
  batches << "epoch\tstep\trow\n";
  int   steps = 0;
  Batch first;
  for (int e = 1; e <= o.epochs; ++e) {
    order.begin_epoch();
    for (;;) {
      const auto rows = order.next();
      if (rows.empty()) break;
      const auto b = bert::Rows(pools[0], rows);
      if (!steps) {
        first = b;
        nn::save_npz((root / "first_batch.npz").string(), Named(b));
        nn::save_npz((root / "first_forward.npz").string(), Named(model.Evaluate(b)));
        nn::save_npz((root / "initial_gradients.npz").string(), model.Arrays(&b));
      }
      model.Update(b);
      if (!steps) nn::save_npz((root / "first_after.npz").string(), model.Arrays());
      for (auto row : rows) batches << e << '\t' << steps << '\t' << row << '\n';
      ++steps;
    }
    const float ce = model.Evaluate(pools[1]).at("loss").item<float>();
    history << e << '\t' << ce << '\n';
    std::cout << "epoch=" << e << " validationMLMCE=" << ce << std::endl;
  }
  nn::save_npz((root / "final_gradients.npz").string(), model.Arrays(&first));
  model.store.save((root / "final.npz").string());
  bert::BertMlmModel deployed(false);
  deployed.store.load((root / "final.npz").string(), true);
  const auto weights = deployed.Arrays();
  check::Require(lm::SameBits(model.Arrays(), weights), "BERT bitexact restore");
  const auto outputs = deployed.Evaluate(pools[2]);
  nn::save_npz((root / "outputs.npz").string(), Named(outputs));
  nn::save_npz((root / "model_arrays.npz").string(), weights);

  const auto original = bert::Rows(pools[2], {0, 1});
  const auto base     = deployed.Evaluate(original);
  const auto lengths  = original.at("lengths").to_vector<std::int32_t>();
  auto       padded   = original;
  auto pad = [&](std::int32_t value) {
    return [&, value](std::vector<std::int32_t>& v) {
      for (int b = 0; b < 2; ++b)
        for (int t = lengths[static_cast<std::size_t>(b)]; t < 32; ++t) v[static_cast<std::size_t>(b * 32 + t)] = value;
    };
  };
  padded["input"]    = Edited(padded.at("input"), pad(17));
  padded["segments"] = Edited(padded.at("segments"), pad(1));
  const auto po      = deployed.Evaluate(padded);
  for (const auto& [k, v] : base) {
    const auto a = v.to_vector<float>(), b = po.at(k).to_vector<float>();
    for (std::size_t i = 0; i < a.size(); ++i) check::Near(a[i], b[i], 0, "BERT exact numerical padding invariance");
  }
  nn::save_npz((root / "padding_inputs.npz").string(), Named(padded));
  nn::save_npz((root / "padding_outputs.npz").string(), Named(po));

  auto       unlabeled = original;
  const auto sel       = original.at("selected").to_vector<float>();
  unlabeled["target"]  = Edited(unlabeled.at("target"), [&](std::vector<std::int32_t>& v) {
    for (std::size_t i = 0; i < v.size(); ++i)
      if (!sel[i]) v[i] = 69;
  });
  check::Near(base.at("loss").item<float>(), deployed.Evaluate(unlabeled).at("loss").item<float>(), 0,
              "unselected labels cannot affect MLM loss");
  unlabeled["target"] = nn::full({2, 32}, 69, nn::DType::Int32);
  check::Require(deployed.Predict(original).to_vector<float>() == deployed.Predict(unlabeled).to_vector<float>(),
                 "label-free MLM prediction");
  auto empty        = original;
  empty["selected"] = nn::zeros({2, 32});
  const auto eo     = deployed.Evaluate(empty);
  check::Near(eo.at("loss").item<float>(), 0, 0, "no selected positions zero loss");
  for (float g : eo.at("input_gradient").to_vector<float>()) check::Near(g, 0, 0, "no selected positions zero partials");

  auto changed     = original;
  changed["input"] = Edited(changed.at("input"), [&](std::vector<std::int32_t>& v) {
    for (int b = 0; b < 2; ++b)
      for (int t = 10; t < lengths[static_cast<std::size_t>(b)] - 1; ++t) {
        auto& x = v[static_cast<std::size_t>(b * 32 + t)];
        x       = 5 + (x + 17) % 65;
      }
  });
  const auto future = deployed.Evaluate(changed);
  const auto bl = base.at("logits").to_vector<float>(), fl = future.at("logits").to_vector<float>();
  double     effect = 0;
  for (int b = 0; b < 2; ++b)
    for (int t = 1; t < 8; ++t)
      for (int c = 0; c < 70; ++c) {
        const auto i = static_cast<std::size_t>((b * 32 + t) * 70 + c);
        effect       = std::max(effect, std::abs(double(bl[i] - fl[i])));
      }
  check::Require(effect > 1e-6, "bidirectional right context changes earlier predictions");
  nn::save_npz((root / "right_context_inputs.npz").string(), Named(changed));
  nn::save_npz((root / "right_context_outputs.npz").string(), Named(future));

  double     fd_error = 0;
  int        fd_count = 0;
  const auto gradients = model.Arrays(&first);
  for (std::size_t k = 0; k < model.store.size(); ++k) {
    Tensor             value = model.store.at(model.store.names()[k]);
    float*             p     = value.data_ptr<float>();
    const auto         g     = gradients[k].second.to_vector<float>();
    const std::int64_t n     = value.numel();
    for (std::int64_t i : {std::int64_t(0), n / 2, n - 1}) {
      const float old = p[i];
      p[i]            = old + .001f;
      const float plus = model.LossValue(first);
      p[i]             = old - .001f;
      const float minus = model.LossValue(first);
      p[i]              = old;
      fd_error          = std::max(fd_error, std::abs((plus - minus) / .002 - g[static_cast<std::size_t>(i)]));
      ++fd_count;
    }
  }
  check::Require(lm::SameBits(model.Arrays(), weights), "finite-difference probes restore the weights");
  check::Require(fd_error < .004, "native tied encoder and MLM finite differences");
  int        correct = 0, selected = 0;
  const auto ol = outputs.at("logits").to_vector<float>();
  const auto ts = pools[2].at("selected").to_vector<float>();
  const auto tt = pools[2].at("target").to_vector<std::int32_t>();
  for (int b = 0; b < 32; ++b)
    for (int t = 0; t < 32; ++t)
      if (ts[static_cast<std::size_t>(b * 32 + t)]) {
        const float* z    = ol.data() + (b * 32 + t) * 70;
        int          best = 0;
        for (int c = 1; c < 70; ++c)
          if (z[c] > z[best]) best = c;
        correct += best == tt[static_cast<std::size_t>(b * 32 + t)];
        ++selected;
      }
  check::Require(lm::SameBits(deployed.Arrays(), weights), "BERT inference immutable");
  const float   ce = outputs.at("loss").item<float>();
  std::ofstream metrics(root / "metrics.json");
  metrics.precision(17);
  metrics << "{\"parameters\":1606,\"variables\":26,\"epochs\":" << o.epochs << ",\"updates\":" << steps
          << ",\"test_mlm_ce\":" << ce << ",\"test_selected_accuracy\":" << double(correct) / selected
          << ",\"test_selected_positions\":" << selected << ",\"right_context_logit_effect\":" << effect
          << ",\"native_fd_count\":" << fd_count << ",\"native_fd_error\":" << fd_error << "}\n";
  std::cout << "testSelectedCE=" << ce << " accuracy=" << correct << '/' << selected << " rightContextEffect=" << effect
            << "; " << fd_count << " nativeFD max=" << fd_error << " padding/selectedOnly/labelFree/restore/immutable PASS\n";

  if (!init.empty() && o.epochs == 3) {
    double firsts = 0;
    for (auto f : {"initial_arrays.npz", "initial_outputs.npz", "first_batch.npz", "first_forward.npz",
                   "initial_gradients.npz", "first_after.npz"})
      firsts = std::max(firsts, lm::CompareRecord(o, f, nn::load_npz((root / f).string())));
    check::Require(firsts >= 0 && firsts <= 2e-5, "replayed initial forward/gradients/first update match TF");
    for (auto f : {"model_arrays.npz", "final_gradients.npz", "outputs.npz", "padding_outputs.npz",
                   "right_context_outputs.npz"})
      lm::CompareRecord(o, f, nn::load_npz((root / f).string()));
  }
  std::cout << "wall " << std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() << " s\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return Run(lm::ParseOptions(argc, argv, "bert_mlm", true));
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
