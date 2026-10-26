// CMD: just run-nn transformer_nmt
// Scratch post-norm Transformer NMT on pinned real English-French pairs.
//
// Port of tflow's transformer_nmt (common/{nn,ml}/transformer_nmt.h ->
// examples/nn/common/transformer_nmt.hpp). Data: the pinned translation split
// (paths::Dataset("translation")/data.npz: 128 fitting, 32 validation, 32 test
// pairs of 6 byte tokens with BOS 1 / EOS 2, teacher-forced decoder inputs and
// lengths). One encoder and one decoder layer (D 8, 2 heads, FF 16), 5760
// parameters in 33 tensors, Adam .003, batch 8 from nn::BatchOrder(128, 8,
// 46007) (tflow's order), three epochs = 48 updates.
//
// Checks (as tflow): parameter count; bit-exact checkpoint restore;
// deterministic greedy decoding that ignores gold targets/teacher/lengths;
// exact padding invariance of every output and partial; future teacher tokens
// cannot change earlier logits and the prefix gradient to them is 0; an empty
// source leaks nothing through memory, attentions or partials; whole-model
// float32 central differences (99 coordinates) < .003; inference leaves the
// weights unchanged. Reports test token CE, teacher-forced token accuracy and
// greedy exact-sequence accuracy.
//
// Deviations: weights replay TF's saved initial arrays by default (init=auto:
// tflow's output/models/transformer_nmt/initial_arrays.npz when present, then
// every record is compared with TF's; init=nn: nn's generator, statistical
// parity). The checkpoint is final.npz. Options: epochs=3 out=DIR tf=DIR
// init=auto|tf|nn threads=4 verbose=0. Output: output/nn/models/transformer_nmt/.
#include "common/transformer_lm.hpp"
#include "common/transformer_nmt.hpp"

namespace {

using lm::Tensor;
using nmt::SeqBatch;
namespace nn = datamunge::nn;
namespace fs = std::filesystem;

Tensor Edited(const Tensor& x, const std::function<void(std::vector<std::int32_t>&)>& f) {
  auto v = x.to_vector<std::int32_t>();
  f(v);
  return nn::from_vector(v, x.shape());
}

int Run(const lm::Options& o) {
  const auto     started = std::chrono::steady_clock::now();
  const fs::path root = o.out, assets = paths::Dataset("translation");
  check::Require(fs::exists(assets / "data.npz"), "translation dataset missing: " + assets.string());
  for (auto f : {"data.npz", "manifest.json", "vocabulary.txt", "attribution.tsv", "_about.txt"})
    if (fs::exists(assets / f)) fs::copy_file(assets / f, root / f, fs::copy_options::overwrite_existing);
  const auto            data = nn::load_npz((root / "data.npz").string());
  auto                  at   = [&](const std::string& k) { return attn::Named(data, k); };
  std::vector<SeqBatch> pools;
  for (std::string s : {"fit", "validation", "test"})
    pools.push_back({at(s + "_input"), at(s + "_target"), at(s + "_teacher"), at(s + "_input_lengths"),
                     at(s + "_output_lengths"), at(s + "_source_ids")});
  nmt::TransformerNmtModel m;
  check::Require(m.store.count() == 5760 && m.store.size() == 33,
                 "5760 parameters,33 variables: sharedembedding/threeattention/twoFF/fiveLN/head");
  const fs::path init = lm::InitArrays(o);
  if (!init.empty()) {
    lm::LoadArrays(m.store, init);
    m.Train();
    std::cout << "replaying TF's initial arrays " << init.string() << "\n";
  } else {
    std::cout << "weights from nn's generator (statistical parity with TF only)\n";
  }
  nn::save_npz((root / "initial_arrays.npz").string(), m.Arrays());
  nn::save_npz((root / "initial_outputs.npz").string(), lm::Named(m.Evaluate(pools[2])));
  nn::BatchOrder order(128, 8, 46007);
  std::ofstream  batches(root / "batches.tsv"), history(root / "history.tsv");
  batches << "epoch\tstep\trow\n";
  history << "epoch\tvalidation_token_ce\n";
  int      updates = 0;
  SeqBatch first;
  for (int e = 1; e <= o.epochs; ++e) {
    order.begin_epoch();
    for (;;) {
      const auto rows = order.next();
      if (rows.empty()) break;
      const auto b = nmt::SeqRows(pools[0], rows);
      if (!updates) {
        first = b;
        nn::save_npz((root / "first_batch.npz").string(), nmt::Named(b));
        nn::save_npz((root / "initial_gradients.npz").string(), m.Arrays(&b));
        nn::save_npz((root / "first_forward.npz").string(), lm::Named(m.Evaluate(b)));
      }
      m.Update(b);
      if (!updates) nn::save_npz((root / "first_after.npz").string(), m.Arrays());
      for (auto row : rows) batches << e << '\t' << updates << '\t' << row << '\n';
      ++updates;
    }
    const float ce = m.Evaluate(pools[1]).at("loss").item<float>();
    history << e << '\t' << ce << '\n';
    std::cout << "epoch=" << e << " validationTokenCE=" << ce << std::endl;
  }
  nn::save_npz((root / "final_gradients.npz").string(), m.Arrays(&first));
  m.store.save((root / "final.npz").string());
  nmt::TransformerNmtModel deployed(false);
  deployed.store.load((root / "final.npz").string(), true);
  const auto weights = deployed.Arrays();
  check::Require(lm::SameBits(m.Arrays(), weights), "bit exact model restore");
  const auto out = deployed.Evaluate(pools[2]);
  Tensor     trace;
  const auto greedy = deployed.Greedy(pools[2], &trace);
  check::Require(greedy.to_vector<std::int32_t>() == deployed.Greedy(pools[2]).to_vector<std::int32_t>(),
                 "deterministic greedy");
  const auto b       = nmt::SeqRows(pools[2], {0, 1});
  const auto regular = deployed.Evaluate(b);
  auto       corrupt = b;
  corrupt.teacher        = nn::full({2, 6}, 13, nn::DType::Int32);
  corrupt.target         = nn::full({2, 6}, 13, nn::DType::Int32);
  corrupt.output_lengths = nn::zeros({2}, nn::DType::Int32);
  check::Require(deployed.Greedy(b).to_vector<std::int32_t>() == deployed.Greedy(corrupt).to_vector<std::int32_t>(),
                 "decode independent of gold/teacher/target lengths");
  const auto il = b.input_lengths.to_vector<std::int32_t>(), ol = b.output_lengths.to_vector<std::int32_t>();
  auto       padded = b;
  padded.input      = Edited(padded.input, [&](std::vector<std::int32_t>& v) {
    for (int j = 0; j < 2; ++j)
      for (int t = il[static_cast<std::size_t>(j)]; t < 6; ++t) v[static_cast<std::size_t>(j * 6 + t)] = 13;
  });
  padded.teacher = Edited(padded.teacher, [&](std::vector<std::int32_t>& v) {
    for (int j = 0; j < 2; ++j)
      for (int t = ol[static_cast<std::size_t>(j)]; t < 6; ++t) v[static_cast<std::size_t>(j * 6 + t)] = 13;
  });
  const auto po = deployed.Evaluate(padded);
  for (const auto& [k, v] : regular) {
    const auto x = v.to_vector<float>(), y = po.at(k).to_vector<float>();
    for (std::size_t i = 0; i < x.size(); ++i) check::Near(x[i], y[i], 0, "all path and derivative padding invariance: " + k);
  }
  nn::save_npz((root / "padding_inputs.npz").string(), nmt::Named(padded));
  nn::save_npz((root / "padding_outputs.npz").string(), lm::Named(po));
  auto future    = b;
  future.teacher = Edited(future.teacher, [](std::vector<std::int32_t>& v) {
    for (int j = 0; j < 2; ++j)
      for (int t = 2; t < 6; ++t) v[static_cast<std::size_t>(j * 6 + t)] = 13;
  });
  const auto fo = deployed.Evaluate(future);
  const auto rl = regular.at("logits").to_vector<float>(), fl = fo.at("logits").to_vector<float>();
  const auto pg = regular.at("prefix_gradient").to_vector<float>();
  for (int j = 0; j < 2; ++j)
    for (int t = 0; t < 6; ++t)
      for (int k = 0; k < 256; ++k) {
        const auto i = static_cast<std::size_t>((j * 6 + t) * 256 + k);
        if (t < 2) check::Near(rl[i], fl[i], 0, "future token cannot affect earlier logits");
        else check::Near(pg[i], 0, 0, "prefix derivative to future teacher zero");
      }
  nn::save_npz((root / "future_inputs.npz").string(), nmt::Named(future));
  nn::save_npz((root / "future_outputs.npz").string(), lm::Named(fo));
  auto empty          = b;
  empty.input_lengths = nn::zeros({2}, nn::DType::Int32);
  const auto eo       = deployed.Evaluate(empty);
  for (auto key : {"memory", "encoder_attention", "cross_attention", "partial0", "partial2"})
    for (float x : eo.at(key).to_vector<float>()) check::Near(x, 0, 0, "empty source cannot leak through attention or partials");
  nn::save_npz((root / "empty_inputs.npz").string(), nmt::Named(empty));
  nn::save_npz((root / "empty_outputs.npz").string(), lm::Named(eo));

  double     fd_error = 0;
  int        fd_count = 0;
  const auto grads    = m.Arrays(&first);
  for (std::size_t k = 0; k < m.store.size(); ++k) {
    Tensor             value = m.store.at(m.store.names()[k]);
    float*             p     = value.data_ptr<float>();
    const auto         g     = grads[k].second.to_vector<float>();
    const std::int64_t n     = value.numel();
    for (std::int64_t c : {std::int64_t(0), n / 2, n - 1}) {
      const float old = p[c];
      p[c]            = old + .002f;
      const float plus = m.LossValue(first);
      p[c]             = old - .002f;
      const float minus = m.LossValue(first);
      p[c]              = old;
      fd_error          = std::max(fd_error, std::abs((plus - minus) / .004 - g[static_cast<std::size_t>(c)]));
      ++fd_count;
    }
  }
  check::Require(lm::SameBits(m.Arrays(), weights), "finite-difference probes restore the weights");
  check::Require(fd_error < .003, "native whole-model FD including LN and three attentions");
  nn::save_npz((root / "model_arrays.npz").string(), weights);
  nn::save_npz((root / "outputs.npz").string(), lm::Named(out));
  nn::save_npz((root / "decoded.npz").string(), {{"greedy", greedy}, {"attention", trace}});
  int        tokens = 0, correct = 0, exact = 0;
  const auto gv = greedy.to_vector<std::int32_t>(), tv = pools[2].target.to_vector<std::int32_t>();
  const auto lv = out.at("logits").to_vector<float>();
  const auto tl = pools[2].output_lengths.to_vector<std::int32_t>();
  for (int j = 0; j < 32; ++j) {
    bool match = true;
    for (int t = 0; t < 6; ++t) {
      const auto i = static_cast<std::size_t>(j * 6 + t);
      match &= gv[i] == tv[i];
      if (t < tl[static_cast<std::size_t>(j)]) {
        const float* z    = lv.data() + i * 256;
        int          best = 0;
        for (int k = 1; k < 256; ++k)
          if (z[k] > z[best]) best = k;
        correct += best == tv[i];
        ++tokens;
      }
    }
    exact += match;
  }
  check::Require(lm::SameBits(deployed.Arrays(), weights), "inference immutable");
  const float   ce = out.at("loss").item<float>();
  std::ofstream metrics(root / "metrics.json");
  metrics << "{\"parameters\":5760,\"epochs\":" << o.epochs << ",\"updates\":" << updates << ",\"test_token_ce\":" << ce
          << ",\"teacher_token_accuracy\":" << double(correct) / tokens
          << ",\"encoded_greedy_exact_accuracy\":" << exact / 32.0 << ",\"native_fd_count\":" << fd_count
          << ",\"native_fd_error\":" << fd_error << "}\n";
  std::cout << "testTokenCE=" << ce << " teacherTokenAccuracy=" << double(correct) / tokens << " greedyExact=" << exact
            << "/32; " << fd_count << " nativeFD maxError=" << fd_error
            << " padding/future/empty/restore/labelFree/immutable PASS\n";

  if (!init.empty() && o.epochs == 3) {
    double firsts = 0;
    for (auto f : {"initial_arrays.npz", "initial_outputs.npz", "first_batch.npz", "first_forward.npz",
                   "initial_gradients.npz", "first_after.npz"})
      firsts = std::max(firsts, lm::CompareRecord(o, f, nn::load_npz((root / f).string())));
    check::Require(firsts >= 0 && firsts <= 2e-5, "replayed initial forward/gradients/first update match TF");
    for (auto f : {"model_arrays.npz", "final_gradients.npz", "outputs.npz", "decoded.npz", "padding_outputs.npz",
                   "future_outputs.npz", "empty_outputs.npz"})
      lm::CompareRecord(o, f, nn::load_npz((root / f).string()));
  }
  std::cout << "wall " << std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() << " s\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return Run(lm::ParseOptions(argc, argv, "transformer_nmt", true));
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
