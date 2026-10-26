// CMD: just run-nn transformer_lm
// Small decoder Transformer LM: causal multi-head attention, next characters and
// context-only ancestral generation.
//
// Port of tflow's transformer_lm (common/ml/transformer_lm.h,
// common/nn/decoder_transformer.h -> examples/nn/common/{transformer_lm,
// decoder_transformer}.hpp, shared with 5.2's gpt_lm). Data: the pinned
// tinyShakespeare character_rnn split (paths::Dataset("character_rnn"): 128 fit
// / 32 validation / 32 test windows of 64 characters, 65-character fit-only
// vocabulary). One pre-norm block (D 16, 2 heads, FF 32, GELU-tanh), learned
// token + position embeddings, final LayerNorm, untied 65-class head: 5425
// parameters in 18 tensors; Adam .003, batch 8 in tflow's BatchOrder(128, 8,
// 9011), three epochs = 48 updates.
//
// Checks (as tflow): data windows/shift; census; restore bit-exact;
// single/batch logit parity; the input Jacobian of position 32's logits is
// exactly zero for every later position; future-token intervention leaves
// earlier logits bit-identical; every future attention weight exactly 0 and
// every row sums to 1; invalid ids rejected; four seeded 256-character
// context-cropped draws repeat bit-exactly; temperature controls share
// uniforms and first raw logits; inference immutable. Fit-only unigram/bigram
// baselines; attention.png (mean held-out attention, 2 heads).
// Added: init=auto replays TF's initial_arrays.npz (the whole run then follows
// TF's; every record is compared, first-step records must match), and TF's
// final weights must reproduce TF's test outputs and samples. Deviations:
// final.npz checkpoint, no independent_audit.json. Options: epochs=3 data=DIR
// out=DIR tf=DIR init=auto|tf|nn threads=4 verbose=0. Output: output/nn/models/transformer_lm.
#include "common/binary_pixels.hpp"
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

  lm::TransformerLM m;
  check::Require(m.store.count() == 5425 && m.store.size() == 18, "5425 parameters 18 attention-LN-FF-embedding tensors");
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
  nn::BatchOrder order(128, 8, 9011);
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
        nn::save_npz((root / "initial_gradients.npz").string(), m.Gradient(x, y));
      }
      m.Update(x, y);
      if (!steps) nn::save_npz((root / "first_after.npz").string(), m.Arrays());
      for (auto r : rows) batches << e << '\t' << steps << '\t' << r << '\t' << starts[r] << '\n';
      ++steps;
    }
    const auto   v   = m.Evaluate(at("validation_input"), at("validation_target"));
    const double nll = v.at("loss").item<float>();
    history << e << '\t' << nll << '\t' << nll / std::log(2.) << '\t' << std::exp(nll) << '\n';
    nn::save_npz((root / ("epoch" + std::to_string(e) + "_arrays.npz")).string(), m.Arrays());
    nn::save_npz((root / ("epoch" + std::to_string(e) + "_validation.npz")).string(), lm::Named(v));
    std::cout << "epoch" << e << ":validationbits/char=" << nll / std::log(2.) << ",perplexity=" << std::exp(nll) << std::endl;
  }
  const double train_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  m.store.save((root / "final.npz").string());
  nn::save_npz((root / "final_gradients.npz").string(), m.Gradient(firstx, firsty));
  lm::TransformerLM deployed(false);
  deployed.store.load((root / "final.npz").string(), true);
  check::Require(lm::SameBits(m.Arrays(), deployed.Arrays()), "Transformer all parameter restore BITEXACT");
  const auto before = deployed.Arrays();
  const auto output = deployed.Evaluate(at("test_input"), at("test_target"));
  nn::save_npz((root / "outputs.npz").string(), lm::Named(output));
  nn::save_npz((root / "model_arrays.npz").string(), before);

  const Tensor single = lm::Rows(at("test_input"), {0});
  const Tensor full   = deployed.Predict(single);
  const auto   fv = full.to_vector<float>(), ov = output.at("logits").to_vector<float>();
  double       parity = 0;
  for (int j = 0; j < 64 * 65; ++j) parity = std::max(parity, std::abs(double(fv[j]) - ov[j]));
  check::Require(parity < 2e-6, "single/batch Transformer parity");
  const Tensor jac = deployed.InputGradient(single);
  const auto   jv  = jac.to_vector<float>();
  for (int t = 33; t < 64; ++t)
    for (int c = 0; c < 65; ++c) check::Require(jv[t * 65 + c] == 0, "ALL future token input partials exact zero");
  nn::save_npz((root / "causality.npz").string(), {{"input", single}, {"logits", full}, {"input_gradient", jac}});
  auto changed = single.to_vector<std::int32_t>();
  for (int t = 32; t < 64; ++t) changed[t] = (changed[t] + 17) % 65;
  const Tensor changed_t = nn::from_vector(changed, {1, 64});
  const Tensor altered   = deployed.Predict(changed_t);
  const auto   av        = altered.to_vector<float>();
  for (int i = 0; i < 32 * 65; ++i) check::Require(av[i] == fv[i], "future token intervention cannot change earlier logits");
  nn::save_npz((root / "future_control.npz").string(), {{"input", changed_t}, {"logits", altered}});
  const auto attention = output.at("attention").to_vector<float>();  // [32, 2, 64, 64]
  for (int b = 0; b < 32; ++b)
    for (int h = 0; h < 2; ++h)
      for (int q = 0; q < 64; ++q) {
        double mass = 0;
        for (int k = 0; k < 64; ++k) {
          const float w = attention[((b * 2 + h) * 64 + q) * 64 + k];
          if (k > q) check::Require(w == 0, "future attention weights EXACT ZERO");
          mass += w;
        }
        check::Near(mass, 1, 2e-6, "all causal attention rows normalized");
      }
  check::Throws([&] { deployed.Predict(nn::full({1, 64}, 65, nn::DType::Int32)); }, "invalid Transformer character IDs rejected");

  const std::vector<std::string> prompts{"\n", "First Citizen:\n", "First Citizen:\n", "First Citizen:\n"};
  const double                   temperatures[] = {1, 1, .8, 1.2};
  std::vector<lm::Outputs>       samples;
  std::ofstream                  html(root / "index.html");
  html << "<!doctype html><meta charset='utf-8'><title>Transformer language model</title><style>body{font:16px system-ui;margin:30px;"
          "max-width:1000px}pre{white-space:pre-wrap;background:#eee;padding:12px}</style><h1>Transformer language model</h1><p><a "
          "href='https://arxiv.org/abs/1706.03762'>Transformer attention</a>, compact decoder-only character LM: ONE pre-norm block, "
          "d16 / two heads of 8 / FF 32, tanh-approx GELU, learned 65x16 token + 64x16 position embeddings, final LayerNorm, untied "
          "65-class head; 5425 parameters in 18 tensors. No recurrence, dropout or KV cache.</p><p>Pinned tinyShakespeare, fit-only "
          "65-char vocabulary, 128/32/32 disjoint windows, 64 inputs / next targets. Three epochs, 48 Adam .003 updates, fixed final. "
          "Ancestral prefix-only generation crops the last 64 tokens and uses the LAST VALID position's logits.</p><h2>Mean held-out "
          "attention, all 32 sequences</h2><img src='attention.png'>";
  for (int k = 0; k < 4; ++k) {
    const auto ids  = lm::Encode(vocab, prompts[k]);
    auto       draw = deployed.Sample(ids, 256, 9029, temperatures[k]);
    check::Require(bpix::SameBits(draw, deployed.Sample(ids, 256, 9029, temperatures[k])), "character ancestral seed repeat BITEXACT");
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
  std::vector<float> heatmap(256 * 512 * 3);
  for (int h = 0; h < 2; ++h)
    for (int y = 0; y < 64; ++y)
      for (int x = 0; x < 64; ++x) {
        double mean = 0;
        for (int b = 0; b < 32; ++b) mean += attention[((b * 2 + h) * 64 + y) * 64 + x] / 32.;
        const float value = float(std::floor(255 * std::clamp(mean, 0., 1.) + .5));
        for (int dy = 0; dy < 4; ++dy)
          for (int dx = 0; dx < 4; ++dx)
            for (int c = 0; c < 3; ++c) heatmap[((y * 4 + dy) * 512 + h * 256 + x * 4 + dx) * 3 + c] = value;
      }
  bpix::PngBytes(root / "attention.png", heatmap, 256, 512);
  const double  nll = output.at("loss").item<float>();
  std::ofstream meta(root / "metrics.json");
  meta.precision(17);
  meta << "{\"parameters\":5425,\"parameter_tensors\":18,\"vocabulary\":65,\"dimensions\":16,\"heads\":2,\"feed_forward\":32,\"blocks\":1,"
          "\"sequence\":64,\"epochs\":"
       << o.epochs << ",\"optimizer_updates\":" << steps << ",\"selected_epoch_fixed_final\":" << o.epochs << ",\"test_nll\":" << nll
       << ",\"test_bits_per_character\":" << nll / std::log(2.) << ",\"test_perplexity\":" << std::exp(nll) << ",\"fit_unigram_nll\":" << unll
       << ",\"fit_bigram_nll\":" << bnll << ",\"uniform_nll\":" << std::log(65.) << ",\"single_batch_max_error\":" << parity
       << ",\"sampled_characters\":1024,\"train_seconds\":" << train_seconds << "}\n";
  check::Require(lm::SameBits(before, deployed.Arrays()), "ALL Transformer likelihood-and-cropped generation immutable parameters");
  check::Require(bool(meta) && bool(html) && bool(history) && bool(batches), "complete Transformer LM records");
  std::cout << "testNLL/perplexity=" << nll << '/' << std::exp(nll) << ", fitting unigram/bigram=" << unll << '/' << bnll
            << "; attention-LN-GELU/causal/restore/1024 sampled chars/immutable PASS\n";

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
      for (auto f : {"epoch3_validation.npz", "model_arrays.npz", "outputs.npz", "final_gradients.npz", "causality.npz", "future_control.npz",
                     "sample0.npz", "sample1.npz", "sample2.npz", "sample3.npz", "baseline.npz"})
        lm::CompareRecord(o, f, nn::load_npz((root / f).string()));
    }
  }
  if (fs::exists(o.tf / "model_arrays.npz") && fs::exists(o.tf / "outputs.npz")) {
    lm::TransformerLM fresh(false);
    lm::LoadArrays(fresh.store, o.tf / "model_arrays.npz");
    const double d = lm::CompareRecord(o, "outputs.npz", lm::Named(fresh.Evaluate(at("test_input"), at("test_target"))));
    check::Require(d >= 0 && d <= 2e-5, "TF's final weights reproduce TF's test outputs");
    int same = 0;
    for (int k = 0; k < 4; ++k) {
      const auto draw = fresh.Sample(lm::Encode(vocab, prompts[k]), 256, 9029, temperatures[k]);
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
    return Run(lm::ParseOptions(int(rest.size()), rest.data(), "transformer_lm", true), data);
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
