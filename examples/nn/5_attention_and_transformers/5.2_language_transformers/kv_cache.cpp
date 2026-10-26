// CMD: just run-nn kv_cache
// Actual incremental Transformer K/V reuse, chunk prefill and safe context
// rebuild.
//
// Port of tflow's kv_cache (common/ml/kv_cache.h, common/nn/cached_decoder.h ->
// examples/nn/common/{kv_cache,decoder_transformer}.hpp). The trained gpt_lm
// model runs in three cached schedules over two 64-character test windows -
// 64 single-token appends, chunks {3,1,7,4,13,1,20,15} and one 64-token
// prefill - and every logit must equal the uncached full causal model (< 3e-6);
// old keys/values stay bit-identical, within-chunk future attention is 0, a
// fresh one-row cache equals its batch row, and eight malformed inputs are
// rejected (empty batch/chunk, batch mismatch, K/V shape mismatch, context
// overflow, bad token, NaN cache). Then 80 characters are sampled after
// "First Citizen:\n" (std::mt19937(46313)) with the cache; once the 64-token
// window slides the cache is rebuilt (absolute positions shift), and every
// cached decision must equal the context-cropped full model's.
//
// The trained GPT: gpt=auto (default) loads TF's gpt_lm model_arrays.npz when
// tflow's outputs are present (then every record is compared with TF's),
// gpt=ours our gpt_lm run's final.npz, gpt=PATH any compatible .npz.
// Options: out=DIR tf=DIR gpt=auto threads=4 verbose=0. Output: output/nn/models/kv_cache/.
#include "common/kv_cache.hpp"
#include "common/token_sampling.hpp"
#include "common/transformer_lm.hpp"

namespace {

using lm::Tensor;
namespace nn = datamunge::nn;
namespace fs = std::filesystem;

int Run(const lm::Options& o) {
  const auto     started = std::chrono::steady_clock::now();
  const fs::path root    = o.out;
  lm::CachedTransformerLM model;
  check::Require(model.store.count() == 5425 && model.store.size() == 18,
                 "cache path reuses all eighteen variables without new parameters");
  const fs::path weights_file = lm::TrainedGpt(o.gpt);
  std::cout << "trained GPT weights: " << weights_file.string() << "\n";
  lm::LoadArrays(model.store, weights_file);
  check::Require(model.store.size() == 18, "cache path adds no parameters");
  const auto weights = lm::Arrays(model.store);
  nn::save_npz((root / "model_arrays.npz").string(), weights);
  const auto data   = nn::load_npz((paths::Dataset("character_rnn") / "character_rnn.npz").string());
  const Tensor tokens = lm::Rows(attn::Named(data, "test_input"), {0, 1});
  const Tensor full   = model.Predict(tokens);
  nn::save_npz((root / "full.npz").string(), {{"input", tokens}, {"logits", full}});
  const auto tok = tokens.to_vector<std::int32_t>();
  const auto fv  = full.to_vector<float>();
  double     maxerror = 0;
  int        appends  = 0;
  std::int64_t reused = 0, pairs = 0, projected = 0;
  std::vector<std::string> records{"full.npz", "model_arrays.npz"};
  for (std::string name : {"single", "chunks", "prefill"}) {
    auto             state  = model.Empty(2);
    int              offset = 0, index = 0;
    std::vector<int> sizes  = name == "single" ? std::vector<int>(64, 1)
                              : name == "chunks" ? std::vector<int>{3, 1, 7, 4, 13, 1, 20, 15}
                                                 : std::vector<int>{64};
    for (int count : sizes) {
      std::vector<std::int32_t> chunk;
      for (int b = 0; b < 2; ++b)
        for (int t = 0; t < count; ++t) chunk.push_back(tok[static_cast<std::size_t>(b * 64 + offset + t)]);
      const Tensor ids = nn::from_vector(chunk, {2, count});
      const auto   old = state;
      auto         out = model.Append(ids, state);
      const auto   lv  = out.at("logits").to_vector<float>();
      for (int b = 0; b < 2; ++b)
        for (int t = 0; t < count; ++t)
          for (int c = 0; c < 65; ++c) {
            const double error = std::abs(lv[static_cast<std::size_t>((b * count + t) * 65 + c)] -
                                          fv[static_cast<std::size_t>((b * 64 + offset + t) * 65 + c)]);
            maxerror = std::max(maxerror, error);
            check::Require(error < 3e-6, "cached every-position logits equal full causal model");
          }
      if (offset > 0) {
        for (const auto& [now, then] : {std::pair{state.keys, old.keys}, std::pair{state.values, old.values}}) {
          const auto a = nn::slice_axis(now, 2, 0, offset).contiguous().to_vector<float>(), b = then.to_vector<float>();
          check::Require(a == b, "old keys/values unchanged");
        }
      }
      const auto av = out.at("attention").to_vector<float>();  // [2, 2, count, offset + count]
      for (int b = 0; b < 2; ++b)
        for (int h = 0; h < 2; ++h)
          for (int q = 0; q < count; ++q)
            for (int k = offset + q + 1; k < offset + count; ++k)
              check::Near(av[static_cast<std::size_t>(((b * 2 + h) * count + q) * (offset + count) + k)], 0, 0,
                          "within-chunk future attention0");
      out["input"]      = ids;
      const auto record = name + "_" + std::to_string(index++) + ".npz";
      nn::save_npz((root / record).string(), lm::Named(out));
      records.push_back(record);
      if (name == "single") {
        reused += offset;
        projected += count;
        pairs += std::int64_t(count) * (offset + count);
      }
      offset += count;
      ++appends;
    }
    check::Require(offset == 64, "all context positions consumed");
  }
  // Independent batch rows and a fresh stream reproduce their batched cache.
  for (int b = 0; b < 2; ++b) {
    auto       s   = model.Empty(1);
    const auto out = model.Append(lm::Rows(tokens, {b}), s).at("logits").to_vector<float>();
    for (int i = 0; i < 64 * 65; ++i)
      check::Near(out[static_cast<std::size_t>(i)], fv[static_cast<std::size_t>(b * 64 * 65 + i)], 3e-6,
                  "fresh one-row cache equals batch row");
  }
  int  rejected = 0;
  auto reject   = [&](auto f) {
    try {
      f();
    } catch (const std::invalid_argument&) {
      ++rejected;
    }
  };
  const Tensor one = nn::zeros({1, 1}, nn::DType::Int32);
  reject([&] { model.Empty(0); });
  reject([&] {
    auto s = model.Empty(1);
    model.Append(nn::zeros({1, 0}, nn::DType::Int32), s);
  });
  reject([&] {
    auto s = model.Empty(2);
    model.Append(one, s);
  });
  reject([&] {
    auto s   = model.Empty(1);
    s.values = nn::zeros({1, 2, 1, 8});
    model.Append(one, s);
  });
  reject([&] {
    auto s = model.Empty(1);
    model.Append(nn::zeros({1, 65}, nn::DType::Int32), s);
  });
  reject([&] {
    auto s = model.Empty(1);
    model.Append(nn::full({1, 1}, 65, nn::DType::Int32), s);
  });
  reject([&] {
    auto s = model.Empty(1);
    model.Append(nn::zeros({1, 64}, nn::DType::Int32), s);
    model.Append(one, s);
  });
  reject([&] {
    auto s   = model.Empty(1);
    s.keys   = nn::full({1, 2, 1, 8}, NAN);
    s.values = nn::zeros({1, 2, 1, 8});
    model.Append(one, s);
  });
  check::Require(rejected == 8, "cache shape/length/token/NaN validation");

  fs::copy_file(lm::GptVocabulary(), root / "vocabulary.txt", fs::copy_options::overwrite_existing);
  const std::string vocab   = lm::ReadText(root / "vocabulary.txt");
  std::vector<int>  history = lm::Encode(vocab, "First Citizen:\n");
  const int         prompt  = static_cast<int>(history.size());
  auto              state   = model.Empty(1);
  std::mt19937      rng(46313);
  std::vector<float>        logits, uncached;
  std::vector<double>       uniform;
  std::vector<std::int32_t> rebuilds, lengths;
  int          rebuild_count        = 0;
  std::int64_t generation_projected = 0;
  for (int t = 0; t < 80; ++t) {
    bool       rebuilt = false;
    const auto out     = model.Advance(history, state, &rebuilt);
    rebuilds.push_back(rebuilt);
    rebuild_count += rebuilt;
    generation_projected += rebuilt ? std::min(static_cast<int>(history.size()), 64) : 1;
    lengths.push_back(static_cast<std::int32_t>(state.keys.size(2)));
    const auto                q     = out.at("logits").size(1) - 1;
    const int                 count = std::min(static_cast<int>(history.size()), 64);
    std::vector<std::int32_t> input(64, 0);
    for (int j = 0; j < count; ++j) input[static_cast<std::size_t>(j)] = history[history.size() - static_cast<std::size_t>(count - j)];
    const auto          expected = model.Predict(nn::from_vector(input, {1, 64})).to_vector<float>();
    const auto          ov       = out.at("logits").to_vector<float>();
    std::vector<double> z(65), base(65);
    for (int c = 0; c < 65; ++c) {
      z[static_cast<std::size_t>(c)]    = ov[static_cast<std::size_t>(q * 65 + c)];
      base[static_cast<std::size_t>(c)] = expected[static_cast<std::size_t>((count - 1) * 65 + c)];
      logits.push_back(static_cast<float>(z[static_cast<std::size_t>(c)]));
      uncached.push_back(static_cast<float>(base[static_cast<std::size_t>(c)]));
      check::Near(z[static_cast<std::size_t>(c)], base[static_cast<std::size_t>(c)], 3e-6,
                  "cached generation equals context-cropped full model");
    }
    const double u = (double(rng()) + .5) / 4294967296.;
    uniform.push_back(u);
    const int chosen    = lm::DrawToken(lm::FilterTokens(z).probability, u),
              reference = lm::DrawToken(lm::FilterTokens(base).probability, u);
    check::Require(chosen == reference, "all actual seeded cached/full decisions equal");
    history.push_back(chosen);
    if (t == 0 || t == 49 || t == 50 || t == 79) {
      const auto record = "generation_cache_" + std::to_string(t) + ".npz";
      nn::save_npz((root / record).string(), {{"keys", state.keys}, {"values", state.values}});
      records.push_back(record);
    }
  }
  std::ofstream text(root / "generated.txt");
  for (int c : history) text << vocab.at(static_cast<std::size_t>(c));
  nn::save_npz((root / "generation.npz").string(),
               {{"codes", nn::from_vector(std::vector<std::int32_t>(history.begin(), history.end()), {prompt + 80})},
                {"logits", nn::from_vector(logits, {80, 65})},
                {"uncached_logits", nn::from_vector(uncached, {80, 65})},
                {"uniform", nn::from_vector(uniform, {80})},
                {"rebuilt", nn::cast(nn::from_vector(rebuilds, {80}), nn::DType::Bool)},
                {"lengths", nn::from_vector(lengths, {80})}});
  records.push_back("generation.npz");
  check::Require(lm::SameBits(lm::Arrays(model.store), weights), "full/cache/generation weights immutable");
  std::ofstream metrics(root / "metrics.json");
  metrics.precision(17);
  metrics << "{\"parameters\":5425,\"cache_parameters\":0,\"append_runs\":" << appends
          << ",\"max_native_logit_error\":" << maxerror << ",\"single_stream_projected_token_positions\":" << projected
          << ",\"single_stream_reused_prefix_positions\":" << reused
          << ",\"single_stream_attention_pairs_per_head\":" << pairs
          << ",\"uncached_variable_prefix_projected_positions\":2080,\"uncached_"
             "variable_prefix_attention_pairs_per_head\":89440,\"rejected_inputs\":"
             "8,\"generated_characters\":80,\"generation_rebuilds\":"
          << rebuild_count << ",\"generation_projected_positions\":" << generation_projected << "}\n";
  std::cout << appends << " native empty/prefill/chunk/single cache appends; max logit error=" << maxerror
            << "; old KV unchanged, batch independent, 8 guards; 80 cached/full sample decisions with " << rebuild_count
            << " correct absolute-position rebuilds; immutable PASS\n";

  if (lm::CompareRecord(o, "model_arrays.npz", weights) == 0) {
    double worst = 0, codes = 0;
    for (const auto& r : records) {
      const auto ours = nn::load_npz((root / r).string());
      if (o.verbose) std::printf("  %s\n", r.c_str());
      worst = std::max(worst, attn::CompareWithTf(o.tf / r, ours, o.verbose));
    }
    codes = attn::CompareWithTf(o.tf / "generation.npz", nn::load_npz((root / "generation.npz").string()), false,
                                [](const std::string& k) { return k == "codes" || k == "rebuilt" || k == "lengths"; });
    std::printf("  vs TF: %zu records (logits, K/V, attention, hidden, generation): max |ours - TF| = %.3g; generated codes/rebuilds %s\n",
                records.size(), worst, codes == 0 ? "identical" : "DIFFER");
    check::Require(codes == 0 && worst <= 1e-5, "cache records and generated text equal TF's with TF's weights");
  }
  std::cout << "wall " << std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() << " s\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return Run(lm::ParseOptions(argc, argv, "kv_cache", false));
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
