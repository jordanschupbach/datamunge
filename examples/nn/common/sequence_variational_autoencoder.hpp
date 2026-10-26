// Port of tflow's common/nn/sequence_variational_autoencoder.h and
// common/ml/sequence_variational_autoencoder.h: the compact sentence VAE (masked GRU16
// encoder over separate 8-D embeddings, Gaussian 4-D sentence code, causal teacher-forced GRU16
// decoder whose initial state is tanh(dense(z))), its generated-grammar data (tflow's `just
// sequence-vae-assets`, $DATAMUNGE_DATA_DIR/sequence_vae: 432 unique sentences, 288 fit /
// 72 validation / 72 test, PAD0/BOS1/EOS2, 19 words, 7 steps) with exhaustive regeneration
// checks, target-free greedy/ancestral decoding and the Monte Carlo score. posterior_collapse
// reuses the model with a KL weight (Update/Gradient `beta`).
#pragma once

#include <map>

#include "common/beta_variational_autoencoder.hpp"

namespace ae {

constexpr int kSequenceVocab = 19, kSequenceSteps = 7;

// ---- model ------------------------------------------------------------------------------

/// one_hot(ids) x "<name>" [19, 8] (U(-.2, .2)): the TF graph's MatMul(OneHot, book).
inline nn::Tensor SequenceEmbedding(nn::ParameterStore& p, const std::string& name, const nn::Tensor& ids) {
  auto book = p.get(name, {kSequenceVocab, 8}, nn::init::uniform(.2));
  return nn::matmul(nn::one_hot(ids, kSequenceVocab), book);
}

struct SequenceEncoderOutput {
  nn::Tensor state, states;
};
/// GRU16 over every column of tokens [B, T]; PAD (id 0) leaves the state unchanged.
inline SequenceEncoderOutput SequenceEncoder(nn::ParameterStore& p, const nn::Tensor& tokens) {
  nn::Tensor              h = nn::zeros({tokens.size(0), 16});
  std::vector<nn::Tensor> states;
  for (std::int64_t t = 0; t < tokens.size(1); ++t) {
    auto ids      = nn::select(tokens, 1, t).contiguous();
    auto proposed = nn::gru_step(p, "encoder_gru", SequenceEmbedding(p, "encoder_embedding", ids), h, 1.0);
    auto mask     = nn::expand_dims(nn::cast(nn::not_equal(ids, 0.0), nn::DType::Float32), 1);
    h             = mask * proposed + (1.0 - mask) * h;
    states.push_back(h);
  }
  return {h, nn::stack_time(states)};
}
inline nn::Tensor SequenceDecoderInitial(nn::ParameterStore& p, const nn::Tensor& z) {
  return nn::dense(p, "decoder_initial", z, 16, nn::Activation::Tanh);
}

/// BOS followed by the first six targets (teacher forcing with the PREVIOUS target only).
inline nn::Tensor SequenceDecoderInputs(const nn::Tensor& tokens) {
  const auto         v = tokens.to_vector<int>();
  const std::int64_t n = tokens.size(0);
  std::vector<int>   out(v.size());
  for (std::int64_t b = 0; b < n; ++b)
    for (int t = 0; t < kSequenceSteps; ++t) out[b * 7 + t] = t ? v[b * 7 + t - 1] : 1;
  return nn::from_vector(out, {n, 7});
}

struct SequenceVaeOutput {
  nn::Tensor decoder_inputs, encoder_state, encoder_states, mu, log_variance, z, decoder_initial, decoder_states,
      logits, probability, predicted_ids, mask, token_nll, nll, kl, correct, valid_tokens, loss;
};

/// The causal decoder from a fixed z over explicit prefixes `inputs` [B, 7]; fills logits,
/// decoder states, probabilities and (when targets are given) the masked token NLL fields.
inline void SequenceTeacher(nn::ParameterStore& p, SequenceVaeOutput& o, const nn::Tensor& inputs,
                            const nn::Tensor* tokens) {
  o.decoder_initial = SequenceDecoderInitial(p, o.z);
  nn::Tensor              h = o.decoder_initial;
  std::vector<nn::Tensor> logits, states, probabilities, predictions, losses;
  for (int t = 0; t < kSequenceSteps; ++t) {
    auto input = nn::select(inputs, 1, t).contiguous();
    h          = nn::gru_step(p, "decoder_gru", SequenceEmbedding(p, "decoder_embedding", input), h, 1.0);
    auto score = nn::dense(p, "token_output", h, kSequenceVocab);
    logits.push_back(score);
    states.push_back(h);
    probabilities.push_back(nn::softmax(score, 1));
    predictions.push_back(nn::cast(nn::argmax(score, 1), nn::DType::Int32));
    if (tokens) {
      auto target = nn::select(*tokens, 1, t).contiguous();
      auto ce     = -nn::sum(nn::one_hot(target, kSequenceVocab) * nn::log_softmax(score, 1), {1});
      losses.push_back(ce * nn::cast(nn::not_equal(target, 0.0), nn::DType::Float32));
    }
  }
  o.logits         = nn::stack_time(logits);
  o.decoder_states = nn::stack_time(states);
  o.probability    = nn::stack_time(probabilities);
  o.predicted_ids  = nn::stack_time(predictions);
  if (tokens) o.token_nll = nn::stack_time(losses);
}

inline SequenceVaeOutput SequenceVariationalAutoencoder(nn::ParameterStore& p, const nn::Tensor& tokens,
                                                        const nn::Tensor& epsilon) {
  SequenceVaeOutput o;
  auto              encoded = SequenceEncoder(p, tokens);
  o.encoder_state           = encoded.state;
  o.encoder_states          = encoded.states;
  o.mu                      = nn::dense(p, "mu", o.encoder_state, 4);
  o.log_variance            = nn::dense(p, "log_variance", o.encoder_state, 4);
  o.z                       = VaeReparameterize(o.mu, o.log_variance, epsilon);
  o.decoder_inputs          = SequenceDecoderInputs(tokens);
  SequenceTeacher(p, o, o.decoder_inputs, &tokens);
  o.mask         = nn::cast(nn::not_equal(tokens, 0.0), nn::DType::Float32);
  o.nll          = nn::sum(o.token_nll, {1});
  o.valid_tokens = nn::sum(o.mask, {1});
  o.correct      = nn::sum(o.mask * nn::cast(nn::equal(o.predicted_ids, tokens), nn::DType::Float32), {1});
  o.kl           = VaeKl(o.mu, o.log_variance);
  o.loss         = nn::mean(o.nll + o.kl, {0});
  return o;
}

/// tflow's AnnealedSequenceObjective: mean(NLL + beta KL).
inline nn::Tensor AnnealedSequenceObjective(const SequenceVaeOutput& net, double beta) {
  return nn::mean(net.nll + beta * net.kl, {0});
}

/// The TF field names of an evaluation, in tflow's order.
inline nn::NamedTensors SequenceFields(const SequenceVaeOutput& o) {
  return {{"decoder_inputs", o.decoder_inputs}, {"encoder_state", o.encoder_state}, {"encoder_states", o.encoder_states},
          {"mu", o.mu},
          {"log_variance", o.log_variance},
          {"z", o.z},
          {"decoder_initial", o.decoder_initial},
          {"decoder_states", o.decoder_states},
          {"logits", o.logits},
          {"probability", o.probability},
          {"predicted_ids", o.predicted_ids},
          {"mask", o.mask},
          {"token_nll", o.token_nll},
          {"nll", o.nll},
          {"kl", o.kl},
          {"correct", o.correct},
          {"valid_tokens", o.valid_tokens},
          {"loss", o.loss}};
}

struct SequenceGeneration {
  nn::Tensor ids, logits, probability, ended;
};
/// Target-free autoregressive decoding from z: greedy argmax, or ancestral sampling with one
/// explicit float64 uniform per step (token = min(18, #{cdf <= u})). After EOS every output is PAD.
inline SequenceGeneration SequenceGenerate(nn::ParameterStore& p, const nn::Tensor& z, bool sample,
                                           const nn::Tensor& uniforms) {
  const std::int64_t      n = z.size(0);
  std::vector<int>        previous(n, 1), ids(n * 7, 0), ended(n, 0);
  std::vector<double>     u;
  if (sample) u = uniforms.to_vector<double>();
  nn::Tensor              h = SequenceDecoderInitial(p, z);
  std::vector<nn::Tensor> logits, probabilities;
  for (int t = 0; t < kSequenceSteps; ++t) {
    h                = nn::gru_step(p, "decoder_gru",
                                    SequenceEmbedding(p, "decoder_embedding", nn::from_vector(previous, {n})), h, 1.0);
    auto score       = nn::dense(p, "token_output", h, kSequenceVocab);
    auto probability = nn::softmax(score, 1);
    std::vector<int> chosen(n);
    if (sample) {
      auto p64        = nn::cast(probability, nn::DType::Float64);
      auto cdf        = nn::cumsum(p64 / nn::sum(p64, {1}, true), 1).to_vector<double>();
      for (std::int64_t b = 0; b < n; ++b) {
        int count = 0;
        for (int w = 0; w < kSequenceVocab; ++w) count += u[b * 7 + t] >= cdf[b * kSequenceVocab + w];
        chosen[b] = std::min(18, count);
      }
    } else {
      const auto best = nn::argmax(score, 1).to_vector<int>();
      std::copy(best.begin(), best.end(), chosen.begin());
    }
    for (std::int64_t b = 0; b < n; ++b) {
      const int emitted = ended[b] ? 0 : chosen[b];
      ended[b]          = ended[b] || emitted == 2;
      previous[b]       = emitted;
      ids[b * 7 + t]    = emitted;
    }
    logits.push_back(score);
    probabilities.push_back(probability);
  }
  return {nn::from_vector(ids, {n, 7}), nn::stack_time(logits).contiguous(),
          nn::stack_time(probabilities).contiguous(), nn::from_vector(ended, {n})};
}

class SequenceVariationalModel {
 public:
  nn::ParameterStore             parameters{3109};
  std::unique_ptr<nn::Optimizer> optimizer;

  explicit SequenceVariationalModel(bool training = true) {
    {
      nn::NoGradGuard guard;
      SequenceVariationalAutoencoder(parameters, nn::from_vector(std::vector<int>(7, 3), {1, 7}), nn::zeros({1, 4}));
    }
    if (training) optimizer = MakeAdam(parameters);
  }
  void Load(const nn::NamedTensors& arrays) { LoadArrays(parameters, arrays); }
  /// Every field (contiguous), all float fields checked finite.
  SequenceVaeOutput Evaluate(const nn::Tensor& tokens, const nn::Tensor& noise) {
    nn::NoGradGuard guard;
    auto            o = SequenceVariationalAutoencoder(parameters, tokens, noise);
    for (auto* t : {&o.decoder_inputs, &o.encoder_state, &o.encoder_states, &o.mu, &o.log_variance, &o.z,
                    &o.decoder_initial, &o.decoder_states, &o.logits, &o.probability, &o.predicted_ids, &o.mask,
                    &o.token_nll, &o.nll, &o.kl, &o.correct, &o.valid_tokens, &o.loss}) {
      *t = t->contiguous();
      if (t->dtype() == nn::DType::Float32)
        for (float v : t->to_vector<float>()) check::Require(std::isfinite(v), "all sequence fields finite");
    }
    return o;
  }
  SequenceGeneration Generate(const nn::Tensor& z, bool sample = false, const nn::Tensor& uniforms = {}) {
    if (sample) {
      check::Require(uniforms.dtype() == nn::DType::Float64 && uniforms.shape() == nn::Shape{z.size(0), 7},
                     "one explicit token-sampling uniform per step");
      for (double v : uniforms.to_vector<double>()) check::Require(v > 0 && v < 1, "open-interval generation uniforms");
    }
    nn::NoGradGuard guard;
    return SequenceGenerate(parameters, z, sample, uniforms);
  }
  /// Fixed-z prefix-only teacher decoder (no target sequence): {logits, probability}.
  std::pair<nn::Tensor, nn::Tensor> Teacher(const nn::Tensor& z, const nn::Tensor& inputs) {
    nn::NoGradGuard   guard;
    SequenceVaeOutput o;
    o.z = z;
    SequenceTeacher(parameters, o, inputs, nullptr);
    return {o.logits.contiguous(), o.probability.contiguous()};
  }
  nn::Tensor ExtraPaddingState(const nn::Tensor& tokens) {
    nn::NoGradGuard guard;
    return SequenceEncoder(parameters, tokens).state.contiguous();
  }
  std::vector<nn::Tensor> Gradient(const nn::Tensor& tokens, const nn::Tensor& noise, double beta = 1) {
    return nn::grad(AnnealedSequenceObjective(SequenceVariationalAutoencoder(parameters, tokens, noise), beta),
                    parameters.parameters());
  }
  /// One Adam step on mean(NLL + beta KL); returns the pre-update objective.
  double Update(const nn::Tensor& tokens, const nn::Tensor& noise, double beta = 1) {
    check::Require(std::isfinite(beta) && beta >= 0 && beta <= 1, "KLweight in0..1");
    auto loss = AnnealedSequenceObjective(SequenceVariationalAutoencoder(parameters, tokens, noise), beta);
    const double value = loss.item<float>();
    check::Require(std::isfinite(value) && value >= 0, "finite sequence negativeELBO");
    optimizer->minimize(loss);
    return value;
  }
};

/// Monte Carlo {sequence NLL, KL, negative ELBO, teacher token perplexity, teacher token accuracy}.
inline std::vector<double> SequenceVaeScore(SequenceVariationalModel& model, const nn::Tensor& tokens,
                                            const VaeNoiseBank& noise) {
  double             nll = 0, kl = 0, correct = 0, valid = 0;
  const std::int64_t n = tokens.size(0), samples = noise.epsilon.size(0);
  for (std::int64_t k = 0; k < samples; ++k) {
    auto       f = model.Evaluate(tokens, BetaVaeNoiseAt(noise, k));
    const auto a = f.nll.to_vector<float>(), c = f.correct.to_vector<float>(), v = f.valid_tokens.to_vector<float>(),
               d = f.kl.to_vector<float>();
    for (std::int64_t b = 0; b < n; ++b) {
      nll += a[b];
      correct += c[b];
      valid += v[b];
      if (k == 0) kl += d[b];
    }
  }
  const double m = double(n) * double(samples);
  return {nll / m, kl / n, nll / m + kl / n, std::exp(nll / valid), correct / valid};
}

// ---- generated-grammar data ----------------------------------------------------------------

inline const std::vector<std::string>& SequenceVocabulary() {
  static const std::vector<std::string> words = {"[pad]", "[bos]", "[eos]",  "a",      "the",      "red",   "green",
                                                 "blue",  "circle", "square", "triangle", "moves", "rolls", "left",
                                                 "right", "up",     "down",   "quickly",  "slowly"};
  return words;
}

struct SequenceVaeData {
  nn::Tensor tokens, inputs, lengths, factors, source;
};

/// Loads sentences.npz + vocabulary.txt and regenerates EVERY sentence from its factors.
inline std::vector<SequenceVaeData> LoadSequenceVaeData(const std::filesystem::path& root) {
  const auto    arrays = nn::load_npz((root / "sentences.npz").string());
  std::ifstream text(root / "vocabulary.txt");
  std::vector<std::string> words;
  std::string              word;
  while (std::getline(text, word)) words.push_back(word);
  check::Require(words == SequenceVocabulary(), "exact fixed grammar vocabulary PAD0/BOS1/EOS2");
  const auto config = Find(arrays, "config");
  check::Require(config.dtype() == nn::DType::Int32 && config.numel() == 2 && config.to_vector<int>()[0] == 19 &&
                     config.to_vector<int>()[1] == 7,
                 "19word vocabulary and7steps");
  std::vector<SequenceVaeData> pools;
  std::set<int>                sources;
  std::set<std::string>        sequences;
  for (const std::string split : {"fit", "validation", "test"}) {
    const std::int64_t n = split == "fit" ? 288 : 72;
    SequenceVaeData    d{Find(arrays, split + "_tokens"), Find(arrays, split + "_decoder_inputs"),
                      Find(arrays, split + "_lengths"), Find(arrays, split + "_factors"),
                      Find(arrays, split + "_source_ids")};
    auto is = [](const nn::Tensor& t, const nn::Shape& s) { return t.dtype() == nn::DType::Int32 && t.shape() == s; };
    check::Require(is(d.tokens, {n, 7}) && is(d.inputs, {n, 7}) && is(d.lengths, {n}) && is(d.factors, {n, 6}) &&
                       is(d.source, {n}),
                   "exact generated sentence schema");
    const auto tokens = d.tokens.to_vector<int>(), inputs = d.inputs.to_vector<int>(),
               lengths = d.lengths.to_vector<int>(), factors = d.factors.to_vector<int>(),
               source = d.source.to_vector<int>();
    int       counts[6][4] = {}, shorts = 0, total = 0;
    const int card[]       = {2, 3, 3, 2, 4, 3};
    for (std::int64_t b = 0; b < n; ++b) {
      int f[6];
      for (int j = 0; j < 6; ++j) {
        f[j] = factors[b * 6 + j];
        check::Require(f[j] >= 0 && f[j] < card[j], "valid grammar factor");
        ++counts[j][f[j]];
      }
      const int id      = (((((f[0] * 3 + f[1]) * 3 + f[2]) * 2 + f[3]) * 4 + f[4]) * 3 + f[5]),
                residue = 2 * ((f[1] + f[2] + f[5]) % 3) + (f[0] + f[3] + f[4]) % 2;
      check::Require(source[b] == id && sources.insert(id).second, "unique exact sentence sourceID");
      check::Require(split == "test" ? residue == 0 : split == "validation" ? residue == 1 : residue >= 2,
                     "heldout grammar combination split");
      std::vector<int> expected{3 + f[0], 5 + f[1], 8 + f[2], 11 + f[3], 13 + f[4]};
      if (f[5]) expected.push_back(16 + f[5]);
      expected.push_back(2);
      const int length = int(expected.size());
      check::Require(lengths[b] == length, "EOS-inclusive6/7length");
      shorts += length == 6;
      total += length;
      expected.resize(7, 0);
      std::string key;
      for (int t = 0; t < 7; ++t) {
        const int token = tokens[b * 7 + t];
        check::Require(token == expected[t], "ALL432sentences regenerate every token/EOS/PAD");
        check::Require(inputs[b * 7 + t] == (t ? expected[t - 1] : 1),
                       "teacher forcing uses BOS then PREVIOUS target only");
        key += std::to_string(token) + ",";
      }
      check::Require(sequences.insert(key).second, "no duplicate sentence across splits");
    }
    for (int j = 0; j < 6; ++j)
      for (int k = 0; k < card[j]; ++k)
        check::Require(counts[j][k] == n / card[j], "each grammar factor balanced in every split");
    check::Require(shorts == n / 3 && total == n * 20 / 3, "balanced optional-adverb lengths and valid-token count");
    pools.push_back(d);
  }
  check::Require(sources.size() == 432 && sequences.size() == 432, "exhaustive432unique generated sentences");
  return pools;
}

inline std::string SequenceText(const nn::Tensor& tokens, std::int64_t row) {
  const auto  v = nn::select(tokens, 0, row).to_vector<int>();
  std::string text;
  for (int id : v) {
    check::Require(id >= 0 && id < 19, "display token in vocabulary");
    if (!text.empty()) text += ' ';
    text += SequenceVocabulary()[id];
    if (id == 2) break;
  }
  return text;
}

inline bool SequenceGrammarValid(const nn::Tensor& tokens, std::int64_t row) {
  const auto       v = nn::select(tokens, 0, row).to_vector<int>();
  std::vector<int> ids;
  bool             eos = false;
  for (int id : v) {
    if (id == 2) {
      eos = true;
      break;
    }
    ids.push_back(id);
  }
  if (!eos || (ids.size() != 5 && ids.size() != 6)) return false;
  return ids[0] >= 3 && ids[0] <= 4 && ids[1] >= 5 && ids[1] <= 7 && ids[2] >= 8 && ids[2] <= 10 && ids[3] >= 11 &&
         ids[3] <= 12 && ids[4] >= 13 && ids[4] <= 16 && (ids.size() == 5 || (ids[5] >= 17 && ids[5] <= 18));
}

/// fit/validation/test arrays under tflow's data.npz names.
inline nn::NamedTensors SequenceDataArrays(const std::vector<SequenceVaeData>& pools) {
  nn::NamedTensors out;
  for (int split = 0; split < 3; ++split) {
    const std::string name = split == 0 ? "fit" : split == 1 ? "validation" : "test";
    out.emplace_back(name + "_tokens", pools[split].tokens);
    out.emplace_back(name + "_decoder_inputs", pools[split].inputs);
    out.emplace_back(name + "_lengths", pools[split].lengths);
    out.emplace_back(name + "_factors", pools[split].factors);
    out.emplace_back(name + "_source_ids", pools[split].source);
  }
  return out;
}

}  // namespace ae
