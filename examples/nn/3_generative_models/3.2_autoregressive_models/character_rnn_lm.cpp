// CMD: just run-nn character_rnn_lm
// Actual Elman character language model: tinyShakespeare, three epochs,
// streaming hidden-state generation.
//
// Port of tflow's character_rnn_lm (common/ml/character_rnn_lm.h,
// common/nn/character_rnn_lm.h -> examples/nn/common/character_rnn_lm.hpp).
// Data: the pinned tinyShakespeare character_rnn split
// (paths::Dataset("character_rnn"): 128 fit / 32 validation / 32 test windows
// of 64 characters, 65-character fit-only vocabulary). Learned 65x8 embedding,
// tanh hidden 16, 16x16 recurrence, 65-class head: 2025 parameters in 6
// tensors; full 64-step BPTT from a zero state, Adam .003, batch 8 in tflow's
// BatchOrder(128, 8, 8911), three epochs = 48 updates.
//
// Checks (as tflow): data windows/shift; census; restore bit-exact; the
// one-step stateful graph reproduces all 64 states/logits of the full unroll
// (< 2e-6); encoded-input Jacobian of position 32 exactly zero for every
// later position; future intervention leaves earlier logits bit-identical;
// invalid character id rejected; four seeded 256-character streaming draws
// (sampled feedback) repeat bit-exactly; temperature controls share uniforms
// and first raw logits; inference immutable. Fit-only unigram/bigram baselines.
// Added: init=auto replays TF's initial arrays and compares every record; TF's
// final weights must reproduce TF's test outputs and seeded samples.
// Deviations: final.npz checkpoint, no independent_audit.json. Options:
// epochs=3 data=DIR out=DIR tf=DIR init=auto|tf|nn threads=4 verbose=0.
// Output: output/nn/models/character_rnn_lm.
#include "common/binary_pixels.hpp"
#include "common/character_rnn_lm.hpp"
#include "common/transformer_lm.hpp"

namespace {

using lm::Tensor;
namespace nn = datamunge::nn;
namespace fs = std::filesystem;

std::string Escape(const std::string& text) {
  std::string out;
  for (char c : text) out += c == '&' ? "&amp;" : c == '<' ? "&lt;" : c == '>' ? "&gt;" : std::string(1, c);
  return out;
}

int Run(const lm::Options& o, const fs::path& data) {
  const auto     started = std::chrono::steady_clock::now();
  const fs::path root    = o.out;
  check::Require(fs::exists(data / "character_rnn.npz"), "character_rnn dataset missing: " + data.string());
  const auto  corpus = replay::ToMap(nn::load_npz((data / "character_rnn.npz").string()));
  std::string vocab  = lm::ReadText(data / "vocabulary.txt");
  check::Require(vocab.size() == 65 && std::is_sorted(vocab.begin(), vocab.end()) && std::set<char>(vocab.begin(), vocab.end()).size() == 65,
                 "65 sorted unique fit-only ASCII characters");
  fs::copy_file(data / "character_rnn.npz", root / "data.npz", fs::copy_options::overwrite_existing);
  for (auto f : {"manifest.json", "vocabulary.txt"})
    if (fs::exists(data / f)) fs::copy_file(data / f, root / f, fs::copy_options::overwrite_existing);
  for (std::string split : {"fit", "validation", "test"}) {
    const auto& x = corpus.at(split + "_input");
    const auto& y = corpus.at(split + "_target");
    const int   n = split == "fit" ? 128 : 32;
    check::Require(x.dtype() == nn::DType::Int32 && x.shape() == nn::Shape({n, 64}) && y.shape() == x.shape(), "fixed 64-char next-token windows");
    const auto xv = x.to_vector<int>(), yv = y.to_vector<int>();
    for (int i = 0; i < n * 64; ++i) {
      check::Require(xv[i] >= 0 && xv[i] < 65 && yv[i] >= 0 && yv[i] < 65, "valid characters");
      if (i % 64 < 63) check::Require(yv[i] == xv[i + 1], "exact next-character shift");
    }
  }
  auto at = [&](const std::string& k) { return corpus.at(k); };

  charlm::CharacterRnnLM m;
  check::Require(m.store.count() == 2025 && m.store.size() == 6, "2025 parameters six shared embedding-recurrent-head tensors");
  const fs::path init = lm::InitArrays(o);
  if (!init.empty()) {
    replay::LoadInto(m.store, init);
    m.Train();
    std::cout << "replaying TF's initial arrays " << init.string() << "\n";
  } else {
    std::cout << "weights from nn's generator (statistical parity with TF only)\n";
  }
  nn::save_npz((root / "initial_arrays.npz").string(), lm::Named(m.Arrays()));
  nn::save_npz((root / "initial_outputs.npz").string(), lm::Named(m.Evaluate(at("test_input"), at("test_target"))));
  nn::BatchOrder order(128, 8, 8911);
  std::ofstream  history(root / "history.tsv"), batches(root / "batches.tsv");
  history.precision(17);
  history << "epoch\tvalidation_nll\tvalidation_bits_per_character\tvalidation_perplexity\n";
  batches << "epoch\tstep\trow\tstart\n";
  const auto starts = at("fit_start").to_vector<int>();
  int        steps  = 0;
  Tensor     firstx, firsty;
  for (int e = 1; e <= o.epochs; ++e) {
    order.begin_epoch();
    for (;;) {
      const auto rows = order.next();
      if (rows.empty()) break;
      const Tensor x = lm::Rows(at("fit_input"), rows), y = lm::Rows(at("fit_target"), rows);
      if (!steps) {
        firstx = x, firsty = y;
        nn::save_npz((root / "first_batch.npz").string(), {{"input", x}, {"target", y}});
        nn::save_npz((root / "first_forward.npz").string(), lm::Named(m.Evaluate(x, y)));
        nn::save_npz((root / "initial_gradients.npz").string(), lm::Named(m.Gradient(x, y)));
      }
      m.Update(x, y);
      if (!steps) nn::save_npz((root / "first_after.npz").string(), lm::Named(m.Arrays()));
      for (auto r : rows) batches << e << '\t' << steps << '\t' << r << '\t' << starts[r] << '\n';
      ++steps;
    }
    const auto   v   = m.Evaluate(at("validation_input"), at("validation_target"));
    const double nll = v.at("loss").item<float>();
    history << e << '\t' << nll << '\t' << nll / std::log(2.) << '\t' << std::exp(nll) << '\n';
    nn::save_npz((root / ("epoch" + std::to_string(e) + "_arrays.npz")).string(), lm::Named(m.Arrays()));
    nn::save_npz((root / ("epoch" + std::to_string(e) + "_validation.npz")).string(), lm::Named(v));
    std::cout << "epoch" << e << ":validationbits/char=" << nll / std::log(2.) << ",perplexity=" << std::exp(nll) << std::endl;
  }
  const double train_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  m.store.save((root / "final.npz").string());
  nn::save_npz((root / "final_gradients.npz").string(), lm::Named(m.Gradient(firstx, firsty)));
  charlm::CharacterRnnLM deployed(false);
  deployed.store.load((root / "final.npz").string(), true);
  check::Require(bpix::SameBits(m.Arrays(), deployed.Arrays()), "character RNN all parameter restore BITEXACT");
  const auto before = deployed.Arrays();
  const auto output = deployed.Evaluate(at("test_input"), at("test_target"));
  nn::save_npz((root / "outputs.npz").string(), lm::Named(output));
  nn::save_npz((root / "model_arrays.npz").string(), lm::Named(before));

  // The stateful one-step graph reproduces the full 64-step unroll.
  const Tensor single = lm::Rows(at("test_input"), {0});
  const auto   sv = single.to_vector<std::int32_t>();
  const auto   ol = output.at("logits").to_vector<float>(), oh = output.at("hidden").to_vector<float>();
  Tensor             state = nn::zeros({1, 16});
  std::vector<float> step_scores(64 * 65), step_states(64 * 16);
  double             parity = 0;
  for (int t = 0; t < 64; ++t) {
    const auto r = deployed.Step(nn::from_vector(std::vector<std::int32_t>{sv[t]}, {1}), state);
    state        = r.second;
    const auto z = r.first.to_vector<float>(), h = state.to_vector<float>();
    for (int c = 0; c < 65; ++c) step_scores[t * 65 + c] = z[c], parity = std::max(parity, std::abs(double(z[c]) - ol[t * 65 + c]));
    for (int j = 0; j < 16; ++j) step_states[t * 16 + j] = h[j], parity = std::max(parity, std::abs(double(h[j]) - oh[t * 16 + j]));
  }
  check::Require(parity < 2e-6, "stateful one-step matches full 64 BPTT graph / all states and logits");
  const Tensor jac = deployed.InputGradient(single);
  const auto   jv  = jac.to_vector<float>();
  for (int t = 33; t < 64; ++t)
    for (int c = 0; c < 65; ++c) check::Require(jv[t * 65 + c] == 0, "ALL future characters have exact zero input partials");
  nn::save_npz((root / "state_parity.npz").string(), {{"input", single},
                                                      {"step_logits", nn::from_vector(step_scores, {1, 64, 65})},
                                                      {"step_states", nn::from_vector(step_states, {1, 64, 16})},
                                                      {"input_gradient", jac}});
  auto changed = sv;
  for (int t = 32; t < 64; ++t) changed[t] = (changed[t] + 17) % 65;
  const Tensor changed_t = nn::from_vector(changed, {1, 64});
  const auto   altered   = deployed.Evaluate(changed_t, lm::Rows(at("test_target"), {0}));
  const auto   av        = altered.at("logits").to_vector<float>();
  for (int i = 0; i < 32 * 65; ++i) check::Require(av[i] == step_scores[i], "future intervention does not change earlier character logits");
  nn::save_npz((root / "future_control.npz").string(), {{"input", changed_t}, {"logits", altered.at("logits")}});
  check::Throws([&] { deployed.Step(nn::from_vector(std::vector<std::int32_t>{65}, {1}), state); }, "invalid character ID rejected");

  const std::vector<std::string> prompts{"\n", "First Citizen:\n", "First Citizen:\n", "First Citizen:\n"};
  const double                   temperatures[] = {1, 1, .8, 1.2};
  std::vector<lm::Outputs>       samples;
  std::ofstream                  html(root / "index.html");
  html << "<!doctype html><meta charset='utf-8'><title>Character RNN language model</title><style>body{font:16px system-ui;margin:30px;"
          "max-width:1000px}pre{white-space:pre-wrap;background:#eee;padding:12px}</style><h1>Character-level RNN language model</h1>"
          "<p>Vanilla Elman RNN, learned 65x8 character embedding, tanh hidden 16, recurrent 16x16, next-character 65-way head, 2025 "
          "parameters in 6 tensors. Full 64-step BPTT with a zero hidden state per training window; three epochs, 48 Adam .003 "
          "updates, fixed final. Pinned tinyShakespeare, fit-only 65-character alphabet, 128/32/32 disjoint windows.</p><p>The "
          "shared ONE-STEP graph uses only the current observed character and the explicit previous hidden state; generation streams "
          "the state beyond 64 positions with actual SAMPLED feedback. Same MT19937 uniforms across temperatures; temperature divides "
          "the LOGITS. Outputs unfiltered. <a href='https://github.com/karpathy/char-rnn'>char-rnn</a>.</p>";
  for (int k = 0; k < 4; ++k) {
    const auto ids  = lm::Encode(vocab, prompts[k]);
    auto       draw = deployed.Sample(ids, 256, 8929, temperatures[k]);
    check::Require(bpix::SameBits(draw, deployed.Sample(ids, 256, 8929, temperatures[k])), "character ancestral seed repeat BITEXACT");
    samples.push_back(draw);
    nn::save_npz((root / ("sample" + std::to_string(k) + ".npz")).string(), lm::Named(draw));
    std::string text;
    for (int c : draw.at("codes").to_vector<int>()) {
      check::Require(c >= 0 && c < 65, "valid generated character id");
      text += vocab[c];
    }
    std::ofstream generated(root / ("sample" + std::to_string(k) + ".txt"), std::ios::binary);
    generated << text;
    check::Require(bool(generated), "character text artifact");
    html << "<h2>Sample" << k << ", temperature " << temperatures[k] << "</h2><p>Visible prompt:" << Escape(prompts[k]) << "</p><pre>"
         << Escape(text) << "</pre>";
  }
  for (int k = 2; k < 4; ++k) {
    check::Require(bpix::Bits(samples[1].at("uniform")) == bpix::Bits(samples[k].at("uniform")), "same uniform controls for temperature");
    const auto a = samples[1].at("logits").to_vector<float>(), b = samples[k].at("logits").to_vector<float>();
    for (int c = 0; c < 65; ++c) check::Require(a[c] == b[c], "temperature control SAME prompt first RAW logits");
  }
  // Add-one unigram / bigram baselines from the 8192 fit characters.
  const auto          fx = at("fit_input").to_vector<int>(), fy = at("fit_target").to_vector<int>();
  const auto          tx = at("test_input").to_vector<int>(), ty = at("test_target").to_vector<int>();
  std::vector<double> unigram(65, 1), bigram(65 * 65, 1);
  for (int i = 0; i < 8192; ++i) ++unigram[fy[i]], ++bigram[fx[i] * 65 + fy[i]];
  for (auto& u : unigram) u /= 8192 + 65;
  for (int c = 0; c < 65; ++c) {
    double sum = 0;
    for (int j = 0; j < 65; ++j) sum += bigram[c * 65 + j];
    for (int j = 0; j < 65; ++j) bigram[c * 65 + j] /= sum;
  }
  double unll = 0, bnll = 0;
  for (int i = 0; i < 2048; ++i) unll -= std::log(unigram[ty[i]]), bnll -= std::log(bigram[tx[i] * 65 + ty[i]]);
  unll /= 2048, bnll /= 2048;
  nn::save_npz((root / "baseline.npz").string(),
               {{"unigram_probability", nn::from_vector(unigram, {65})}, {"bigram_probability", nn::from_vector(bigram, {65, 65})}});
  const double  nll = output.at("loss").item<float>();
  std::ofstream meta(root / "metrics.json");
  meta.precision(17);
  meta << "{\"parameters\":2025,\"parameter_tensors\":6,\"vocabulary\":65,\"embedding\":8,\"hidden\":16,\"sequence\":64,\"epochs\":"
       << o.epochs << ",\"optimizer_updates\":" << steps << ",\"selected_epoch_fixed_final\":" << o.epochs << ",\"test_nll\":" << nll
       << ",\"test_bits_per_character\":" << nll / std::log(2.) << ",\"test_perplexity\":" << std::exp(nll) << ",\"fit_unigram_nll\":" << unll
       << ",\"fit_bigram_nll\":" << bnll << ",\"uniform_nll\":" << std::log(65.) << ",\"full_step_parity_max_error\":" << parity
       << ",\"sampled_characters\":1024,\"train_seconds\":" << train_seconds << "}\n";
  check::Require(bpix::SameBits(before, deployed.Arrays()), "ALL language eval / stateful generation immutable parameters");
  check::Require(bool(meta) && bool(html) && bool(history) && bool(batches), "complete character RNN records");
  std::cout << "testNLL/perplexity=" << nll << '/' << std::exp(nll) << ", fitting unigram/bigram=" << unll << '/' << bnll
            << "; BPTT/state-parity/causal/restore/1024 sampled chars/immutable PASS\n";

  if (!init.empty() && init == o.tf / "initial_arrays.npz") {
    std::cout << "replay vs TF's recorded run (" << o.tf.string() << "):\n";
    double first = 0;
    for (auto f : {"initial_arrays.npz", "initial_outputs.npz", "first_batch.npz", "first_forward.npz", "initial_gradients.npz", "first_after.npz"})
      first = std::max(first, lm::CompareRecord(o, f, nn::load_npz((root / f).string())));
    check::Require(first >= 0 && first <= 2e-5, "replayed initial forward/gradients/first update match TF");
    if (o.epochs == 3) {
      const auto ours_h = replay::History(root / "history.tsv"), tf_h = replay::History(o.tf / "history.tsv");
      double     hd     = 0;
      for (std::size_t i = 0; i < std::min(ours_h.size(), tf_h.size()); ++i) hd = std::max(hd, std::abs(ours_h[i] - tf_h[i]));
      std::printf("  vs TF %-26s max |ours - TF| = %.3g\n", "history.tsv", hd);
      check::Require(hd < 1e-4, "replayed validation history follows TF's");
      for (auto f : {"epoch3_validation.npz", "model_arrays.npz", "outputs.npz", "final_gradients.npz", "state_parity.npz", "future_control.npz",
                     "sample0.npz", "sample1.npz", "sample2.npz", "sample3.npz", "baseline.npz"})
        lm::CompareRecord(o, f, nn::load_npz((root / f).string()));
    }
  }
  if (fs::exists(o.tf / "model_arrays.npz") && fs::exists(o.tf / "outputs.npz")) {
    charlm::CharacterRnnLM fresh(false);
    replay::LoadInto(fresh.store, o.tf / "model_arrays.npz");
    const double d = lm::CompareRecord(o, "outputs.npz", lm::Named(fresh.Evaluate(at("test_input"), at("test_target"))));
    check::Require(d >= 0 && d <= 2e-5, "TF's final weights reproduce TF's test outputs");
    int same = 0;
    for (int k = 0; k < 4; ++k) {
      const auto draw = fresh.Sample(lm::Encode(vocab, prompts[k]), 256, 8929, temperatures[k]);
      const auto tf_codes = replay::Get(nn::load_npz((o.tf / ("sample" + std::to_string(k) + ".npz")).string()), "codes");
      same += bpix::Bits(draw.at("codes")) == bpix::Bits(tf_codes);
    }
    std::printf("  from TF's final weights: %d/4 seeded 256-character samples identical to TF's\n", same);
    check::Require(same == 4, "TF's final weights reproduce TF's seeded samples");
  }
  std::cout << "training " << train_seconds << " s, wall " << std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count()
            << " s\nArtifacts:" << (root / "index.html").string() << std::endl;
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    fs::path                 data = paths::Dataset("character_rnn");
    std::vector<char*>       rest{argv[0]};
    for (int i = 1; i < argc; ++i) {
      if (std::string(argv[i]).rfind("data=", 0) == 0) data = std::string(argv[i]).substr(5);
      else rest.push_back(argv[i]);
    }
    return Run(lm::ParseOptions(int(rest.size()), rest.data(), "character_rnn_lm", true), data);
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
