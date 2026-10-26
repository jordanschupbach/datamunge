// CMD: just run-nn gpt_lm
// Scratch GPT-style character language model with pre-norm causal attention.
//
// Port of tflow's gpt_lm (common/ml/transformer_lm.h, common/nn/decoder_transformer.h
// -> examples/nn/common/{transformer_lm,decoder_transformer}.hpp). Data: the
// pinned character_rnn split (paths::Dataset("character_rnn"): 128 fitting,
// 32 validation, 32 test windows of 64 characters, 65-character vocabulary).
// One pre-norm block (D 16, 2 heads, FF 32, GELU-tanh), 5425 parameters in 18
// tensors, Adam .003, batch 8 from nn::BatchOrder(128, 8, 46107) (tflow's
// order), three epochs = 48 updates.
//
// Checks (as tflow): parameter count; bit-exact checkpoint restore; the input
// Jacobian of position 32's logits is exactly zero for every later position;
// flipping future tokens leaves the prefix logits bit-identical; flipping any
// later target never leaks to its own prediction; whole-model float32 central
// differences (54 coordinates) < .003; inference and sampling leave the weights
// unchanged. Also: fit unigram/bigram baselines and four seeded samples.
//
// Deviations: weights replay TF's saved initial arrays by default (init=auto
// uses tflow's output/models/gpt_lm/initial_arrays.npz when present; init=nn
// uses nn's generator, statistical parity only), and every record is then
// compared with TF's. The checkpoint is final.npz (not a SaveV2 bundle).
// Options: epochs=3 out=DIR tf=DIR init=auto|tf|nn threads=4 verbose=0.
// Output: output/nn/models/gpt_lm/.
#include "common/transformer_lm.hpp"

namespace {

using lm::Tensor;
namespace nn = datamunge::nn;
namespace fs = std::filesystem;

int Run(const lm::Options& o) {
  const auto     started = std::chrono::steady_clock::now();
  const fs::path root    = o.out, assets = paths::Dataset("character_rnn");
  check::Require(fs::exists(assets / "character_rnn.npz"), "character_rnn dataset missing: " + assets.string());
  fs::copy_file(assets / "character_rnn.npz", root / "data.npz", fs::copy_options::overwrite_existing);
  for (auto f : {"manifest.json", "vocabulary.txt"})
    if (fs::exists(assets / f)) fs::copy_file(assets / f, root / f, fs::copy_options::overwrite_existing);
  const auto  data  = nn::load_npz((root / "data.npz").string());
  auto        at    = [&](const std::string& k) { return attn::Named(data, k); };
  std::string vocab = lm::ReadText(root / "vocabulary.txt");
  check::Require(vocab.size() == 65, "65 fitting-partition characters");

  lm::TransformerLM m;
  check::Require(m.store.count() == 5425 && m.store.size() == 18, "5425 parameters18 tensors LN-GELU GPT");
  const fs::path init = lm::InitArrays(o);
  if (!init.empty()) {
    lm::LoadArrays(m.store, init);
    m.Train();
    std::cout << "replaying TF's initial arrays " << init.string() << "\n";
  } else {
    std::cout << "weights from nn's generator (statistical parity with TF only)\n";
  }
  nn::save_npz((root / "initial_arrays.npz").string(), m.Arrays());
  nn::save_npz((root / "initial_outputs.npz").string(), lm::Named(m.Evaluate(at("test_input"), at("test_target"))));

  nn::BatchOrder order(128, 8, 46107);
  std::ofstream  batches(root / "batches.tsv"), history(root / "history.tsv");
  batches << "epoch\tstep\trow\tstart\n";
  history << "epoch\tvalidation_nll\tvalidation_perplexity\n";
  const auto starts  = nn::cast(at("fit_start"), nn::DType::Int64).to_vector<std::int64_t>();
  int        updates = 0;
  Tensor     firstx, firsty;
  for (int e = 1; e <= o.epochs; ++e) {
    order.begin_epoch();
    for (;;) {
      const auto rows = order.next();
      if (rows.empty()) break;
      const Tensor x = lm::Rows(at("fit_input"), rows), y = lm::Rows(at("fit_target"), rows);
      if (!updates) {
        firstx = x;
        firsty = y;
        nn::save_npz((root / "first_batch.npz").string(), {{"input", x}, {"target", y}});
        nn::save_npz((root / "initial_gradients.npz").string(), m.Gradient(x, y));
        nn::save_npz((root / "first_forward.npz").string(), lm::Named(m.Evaluate(x, y)));
      }
      m.Update(x, y);
      if (!updates) nn::save_npz((root / "first_after.npz").string(), m.Arrays());
      for (auto r : rows) batches << e << '\t' << updates << '\t' << r << '\t' << starts[static_cast<std::size_t>(r)] << '\n';
      ++updates;
    }
    const auto   v   = m.Evaluate(at("validation_input"), at("validation_target"));
    const double nll = v.at("loss").item<float>();
    history << e << '\t' << nll << '\t' << std::exp(nll) << '\n';
    nn::save_npz((root / ("epoch" + std::to_string(e) + "_arrays.npz")).string(), m.Arrays());
    nn::save_npz((root / ("epoch" + std::to_string(e) + "_validation.npz")).string(), lm::Named(v));
    std::cout << "epoch=" << e << " validationNLL=" << nll << " perplexity=" << std::exp(nll) << std::endl;
  }
  nn::save_npz((root / "final_gradients.npz").string(), m.Gradient(firstx, firsty));
  m.store.save((root / "final.npz").string());
  lm::TransformerLM deployed(false);
  deployed.store.load((root / "final.npz").string(), true);
  check::Require(lm::SameBits(m.Arrays(), deployed.Arrays()), "18 tensors restored bitexact");
  const auto before = deployed.Arrays();
  const auto out    = deployed.Evaluate(at("test_input"), at("test_target"));
  nn::save_npz((root / "model_arrays.npz").string(), before);
  nn::save_npz((root / "outputs.npz").string(), lm::Named(out));

  // Causality: the Jacobian of position 32's logits w.r.t. later inputs is 0.
  const Tensor one    = lm::Rows(at("test_input"), {0});
  const Tensor logits = deployed.Predict(one), jac = deployed.InputGradient(one);
  nn::save_npz((root / "causality.npz").string(), {{"input", one}, {"logits", logits}, {"input_gradient", jac}});
  const auto jv = jac.to_vector<float>();
  for (int t = 33; t < 64; ++t)
    for (int c = 0; c < 65; ++c) check::Near(jv[static_cast<std::size_t>(t * 65 + c)], 0, 0, "future encoded token derivative0");
  auto         ids = one.to_vector<std::int32_t>();
  const auto   lv  = logits.to_vector<float>();
  auto         modified = ids;
  for (int t = 32; t < 64; ++t) modified[static_cast<std::size_t>(t)] = (modified[static_cast<std::size_t>(t)] + 17) % 65;
  const Tensor modified_t = nn::from_vector(modified, {1, 64});
  const Tensor altered    = deployed.Predict(modified_t);
  const auto   av         = altered.to_vector<float>();
  for (int i = 0; i < 32 * 65; ++i) check::Near(lv[static_cast<std::size_t>(i)], av[static_cast<std::size_t>(i)], 0, "future token flip prefix invariant");
  nn::save_npz((root / "future_control.npz").string(), {{"input", modified_t}, {"logits", altered}});
  // Every current TARGET flip leaves its prediction prefix unchanged because
  // the actual input is shifted one character behind the target.
  for (int t = 0; t < 64; ++t) {
    auto changed = ids;
    if (t + 1 < 64) changed[static_cast<std::size_t>(t + 1)] = (changed[static_cast<std::size_t>(t + 1)] + 17) % 65;
    const auto z = deployed.Predict(nn::from_vector(changed, {1, 64})).to_vector<float>();
    for (int i = 0; i < (t + 1) * 65; ++i)
      check::Near(z[static_cast<std::size_t>(i)], lv[static_cast<std::size_t>(i)], 0, "shifted target cannot leak to own prediction");
  }

  // Whole-model central differences at the final weights (first batch).
  double     fd_error = 0;
  int        fd_count = 0;
  const auto grad     = m.Gradient(firstx, firsty);
  for (std::size_t k = 0; k < m.store.size(); ++k) {
    Tensor             value = m.store.at(m.store.names()[k]);
    float*             p     = value.data_ptr<float>();
    const auto         g     = grad[k].second.to_vector<float>();
    const std::int64_t n     = value.numel();
    for (std::int64_t c : {std::int64_t(0), n / 2, n - 1}) {
      const float old = p[c];
      p[c]            = old + .002f;
      const float plus = m.LossValue(firstx, firsty);
      p[c]             = old - .002f;
      const float minus = m.LossValue(firstx, firsty);
      p[c]              = old;
      fd_error          = std::max(fd_error, std::abs((plus - minus) / .004 - g[static_cast<std::size_t>(c)]));
      ++fd_count;
    }
  }
  check::Require(lm::SameBits(m.Arrays(), before), "finite-difference probes restore the weights");
  check::Require(fd_error < .003, "native GPT whole-model finite differences");

  int generated = 0;
  for (int i = 0; i < 4; ++i) {
    const std::string prompt = i == 0 ? "\n" : "First Citizen:\n";
    const double      temp   = i == 2 ? .8 : i == 3 ? 1.2 : 1.;
    const auto        sample = deployed.Sample(lm::Encode(vocab, prompt), 128, 46109, temp);
    nn::save_npz((root / ("sample" + std::to_string(i) + ".npz")).string(), lm::Named(sample));
    std::ofstream text(root / ("sample" + std::to_string(i) + ".txt"));
    for (auto c : sample.at("codes").to_vector<std::int32_t>()) text << vocab.at(static_cast<std::size_t>(c));
    generated += 128;
  }

  // Add-one unigram / bigram baselines from the first 8192 fitting characters.
  const auto          fx = at("fit_input").to_vector<std::int32_t>(), fy = at("fit_target").to_vector<std::int32_t>();
  const auto          tx = at("test_input").to_vector<std::int32_t>(), ty = at("test_target").to_vector<std::int32_t>();
  std::vector<double> unigram(65, 1), bigram(65 * 65, 1);
  for (int i = 0; i < 8192; ++i) {
    ++unigram[static_cast<std::size_t>(fy[i])];
    ++bigram[static_cast<std::size_t>(fx[i] * 65 + fy[i])];
  }
  for (auto& u : unigram) u /= 8192 + 65;
  for (int c = 0; c < 65; ++c) {
    double sum = 0;
    for (int j = 0; j < 65; ++j) sum += bigram[static_cast<std::size_t>(c * 65 + j)];
    for (int j = 0; j < 65; ++j) bigram[static_cast<std::size_t>(c * 65 + j)] /= sum;
  }
  double unll = 0, bnll = 0;
  for (int i = 0; i < 2048; ++i) {
    unll -= std::log(unigram[static_cast<std::size_t>(ty[i])]);
    bnll -= std::log(bigram[static_cast<std::size_t>(tx[i] * 65 + ty[i])]);
  }
  unll /= 2048;
  bnll /= 2048;
  nn::save_npz((root / "baseline.npz").string(),
               {{"unigram_probability", nn::from_vector(unigram, {65})}, {"bigram_probability", nn::from_vector(bigram, {65, 65})}});
  check::Require(lm::SameBits(before, deployed.Arrays()), "all inference and draws immutable");

  const double  nll = out.at("loss").item<float>();
  std::ofstream metrics(root / "metrics.json");
  metrics.precision(17);
  metrics << "{\"parameters\":5425,\"parameter_tensors\":18,\"epochs\":" << o.epochs << ",\"optimizer_updates\":" << updates
          << ",\"test_nll\":" << nll << ",\"test_perplexity\":" << std::exp(nll)
          << ",\"test_bits_per_character\":" << nll / std::log(2.) << ",\"fit_unigram_nll\":" << unll
          << ",\"fit_bigram_nll\":" << bnll << ",\"native_fd_count\":" << fd_count << ",\"native_fd_error\":" << fd_error
          << ",\"generated_characters\":" << generated << "}\n";
  std::cout << "testNLL=" << nll << " perplexity=" << std::exp(nll) << " unigram/bigram=" << unll << '/' << bnll << "; "
            << fd_count << " nativeFD max=" << fd_error << "; " << generated
            << " generated chars; future/targetShift/restore/immutable PASS\n";

  // Comparison with TF's records (exact replay when TF's initial arrays were loaded).
  if (!init.empty() && o.epochs == 3) {
    double first = 0;
    for (auto f : {"initial_arrays.npz", "initial_outputs.npz", "first_batch.npz", "first_forward.npz",
                   "initial_gradients.npz", "first_after.npz"}) {
      const auto ours = nn::load_npz((root / f).string());
      first           = std::max(first, lm::CompareRecord(o, f, ours));
    }
    check::Require(first >= 0 && first <= 2e-5, "replayed initial forward/gradients/first update match TF");
    for (auto f : {"epoch3_validation.npz", "outputs.npz", "model_arrays.npz", "final_gradients.npz", "causality.npz",
                   "sample0.npz", "sample1.npz", "sample2.npz", "sample3.npz", "baseline.npz"})
      lm::CompareRecord(o, f, nn::load_npz((root / f).string()));
  }
  std::cout << "wall " << std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() << " s\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return Run(lm::ParseOptions(argc, argv, "gpt_lm", true));
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
