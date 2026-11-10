#pragma once
/// @file sequence_variational_autoencoder.hpp
/// @brief Explicit masked-GRU sentence VAE, causal prefix decoding and EOS-aware generation.
#include <limits>

#include "datamunge/nn/gaussian_autoencoder.hpp"

namespace datamunge::nn::models {
/// @brief Checked fixed sentence-model geometry and private shared teacher-forcing kernel.
namespace sequence_vae_detail {
/// @brief Check representable activation products for the fixed nineteen-token GRU model.
/// @param batch Positive batch size.
/// @param steps Positive time dimension.
/// @throws std::invalid_argument For nonpositive dimensions or int64 product overflow.
inline void Geometry(std::int64_t batch, std::int64_t steps) {
  const auto limit = std::numeric_limits<std::int64_t>::max();
  if (batch <= 0 || steps <= 0 || steps > limit / 19 || batch > limit / (steps * 19))
    throw std::invalid_argument("sentence VAE requires representable positive batch/time products");
}
/// @brief Validate a nonempty integer token matrix, optionally with exactly seven steps.
/// @param ids Int32/Int64 [B,T], with every ID in [0,18]; strides accepted.
/// @param seven Require T=7 when true.
/// @throws std::invalid_argument For invalid shape, dtype, IDs or products.
inline void Tokens(const Tensor& ids, bool seven) {
  if (!ids.defined() || ids.dim() != 2 ||
      (ids.dtype() != DType::Int32 && ids.dtype() != DType::Int64))
    throw std::invalid_argument("sentence VAE requires Int32/64 token matrices");
  Geometry(ids.size(0), ids.size(1));
  if (seven && ids.size(1) != 7)
    throw std::invalid_argument("sentence VAE teacher targets/prefixes require seven steps");
  for (auto id : ids.contiguous().to_vector<std::int64_t>())
    if (id < 0 || id >= 19)
      throw std::invalid_argument("sentence VAE token IDs must lie in [0,18]");
}
/// @brief Check that a store uses supported floating arithmetic before creating any leaves.
/// @param p Mutable or const store whose dtype is checked.
/// @throws std::invalid_argument For a dtype other than Float32/Float64.
inline void Store(const ParameterStore& p) {
  if (p.dtype() != DType::Float32 && p.dtype() != DType::Float64)
    throw std::invalid_argument("sentence VAE requires a Float32/64 store");
}
/// @brief Apply the source one-hot embedding with its independent uniform(-.2,.2) book.
/// @param p Mutable floating store, synchronized by the caller.
/// @param name Exact embedding parameter name.
/// @param ids Valid integer row IDs, already checked by the public entry point.
/// @return Owning differentiable [B,8] values; IDs themselves are discrete.
inline Tensor Embedding(ParameterStore& p, const std::string& name, const Tensor& ids) {
  const auto book = p.get(name, {19, 8}, init::uniform(.2));
  return matmul(cast(one_hot(ids, 19), p.dtype()), book);
}
} // namespace sequence_vae_detail

/// @brief Own final and complete PAD-masked encoder states of the fixed sentence GRU.
///
/// Both handles own their lifetimes and parameter-gradient paths. No model,
/// optimizer, RNG state or input references are retained in this value.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore p(3109);
/// const auto out=n::models::SequenceEncoder(p,n::from_vector(std::vector<int>{3,5,2,0},{1,4}));
/// // out.state is [1,16]; its last two recorded states are equal because PAD is ignored.
/// @endcode
/// @see SequenceEncoder(), SequenceVariationalAutoencoder()
struct SequenceEncoderOutput {
  Tensor state;  ///< Final state[B,16]; PAD columns preserve the preceding state exactly.
  Tensor states; ///< Every post-mask state[B,T,16], including repeated PAD states.
};

/// @brief Encode integer sentences with a shared reset-before GRU16 and separate 8-D embeddings.
///
/// Fixed vocabulary19: PAD0/BOS1/EOS2; all IDs in[0,18] are accepted, without
/// grammar/EOS parsing. Int32/64 tokens[B,T] require B,T>0. Starting at zero,
/// each proposed gru_step(update_bias=1) is combined as mask*proposed+(1-mask)*h.
/// PAD alone is masked; EOS and BOS are ordinary nonzero inputs. Appending PAD
/// leaves the final state unchanged. T may differ from seven for padding probes.
/// Creates/reuses encoder_embedding[19,8] with uniform(-.2,.2) then the six
/// encoder_gru leaves:7 leaves/1,352 scalars. Store must be Float32/64. Strides
/// accepted; all continuous parameter paths remain live, without detach/BPTT
/// truncation or normalization. Inputs unchanged, outputs own lifetimes.
/// Checks IDs/products before leaves; incompatible existing leaves may fail later.
/// No RNG beyond lazy initialization; synchronize shared stores. Work O(B*T)
/// for this fixed architecture, activation storage O(B*T*16).
/// @param parameters Mutable floating store; original model seed3109.
/// @param tokens Integer token matrix[B,T], T>0.
/// @return Owning final state and complete masked state history.
/// @throws std::invalid_argument For invalid inputs, store dtype or incompatible leaves.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore p(3109);
/// const auto out=n::models::SequenceEncoder(p,n::zeros({2,8},n::DType::Int32));
/// // Every state is exactly zero: all eight columns are PAD.
/// @endcode
/// @par References
/// - Cho et al. (2014), Learning Phrase Representations using RNN Encoder-Decoder
///   for Statistical Machine Translation, https://arxiv.org/abs/1406.1078 .
///   Uses the library's reset-before GRU with update bias1 and explicit PAD mask.
/// @see SequenceEncoderOutput, gru_step(), SequenceVariationalAutoencoder()
inline SequenceEncoderOutput SequenceEncoder(ParameterStore& parameters, const Tensor& tokens) {
  sequence_vae_detail::Store(parameters);
  sequence_vae_detail::Tokens(tokens, false);
  Tensor              h = zeros({tokens.size(0), 16}, parameters.dtype());
  std::vector<Tensor> states;
  for (std::int64_t t = 0; t < tokens.size(1); ++t) {
    const auto ids      = select(tokens, 1, t).contiguous();
    const auto proposed = gru_step(
        parameters,
        "encoder_gru",
        sequence_vae_detail::Embedding(parameters, "encoder_embedding", ids),
        h,
        1.0);
    const auto mask = expand_dims(cast(not_equal(ids, 0.0), parameters.dtype()), 1);
    h               = mask * proposed + (1.0 - mask) * h;
    states.push_back(h);
  }
  return {h, stack_time(states)};
}

/// @brief Construct causal teacher prefixes: BOS followed by the first six target tokens.
///
/// Accepts Int32/64[B,7], B>0, all IDs in[0,18], including arbitrary nongrammatical
/// strings. Output is independent owning Int32[B,7]. No EOS masking, mutation,
/// gradients, model state or random draws; work/storage O(B*7). A short sentence's
/// final prefix token is EOS, preceding its final PAD target.
/// @param tokens Nonempty seven-column integer target matrix, strides accepted.
/// @return Owning Int32 previous-target prefixes.
/// @throws std::invalid_argument For invalid shapes, dtype, IDs or products.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// const auto
/// prefixes=n::models::SequenceDecoderInputs(n::from_vector(std::vector<int>{3,5,8,11,13,2,0},{1,7}));
/// // prefixes is [1,3,5,8,11,13,2], never the current target.
/// @endcode
/// @see SequenceTeacher(), SequenceVariationalAutoencoder()
inline Tensor SequenceDecoderInputs(const Tensor& tokens) {
  sequence_vae_detail::Tokens(tokens, true);
  const auto       v = tokens.contiguous().to_vector<int>();
  std::vector<int> out(v.size());
  for (std::int64_t b = 0; b < tokens.size(0); ++b)
    for (int t = 0; t < 7; ++t)
      out[b * 7 + t] = t ? v[b * 7 + t - 1] : 1;
  return from_vector(out, {tokens.size(0), 7});
}

/// @brief Own every endpoint of a seven-step masked-GRU Gaussian sentence VAE.
///
/// All eighteen fields own lifetime, including discrete Int32 decoder inputs and
/// predictions. Continuous fields retain parameter/noise gradients, except hard
/// argmax accuracy. The loss is the unweighted negative ELBO, in nats per sentence;
/// use AnnealedSequenceObjective for a different KL weight. No optimizer is hidden.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore p(3109);
/// const auto
/// out=n::models::SequenceVariationalAutoencoder(p,n::full({1,7},3,n::DType::Int32),n::zeros({1,4}));
/// // out.z equals out.mu; loss is mean(nll+kl), with PAD targets excluded from NLL.
/// @endcode
/// @see SequenceVariationalAutoencoder(), SequenceTeacher(), AnnealedSequenceObjective()
struct SequenceVaeOutput {
  Tensor decoder_inputs;  ///< Int32[B,7], BOS then previous target.
  Tensor encoder_state;   ///< Final masked GRU state[B,16].
  Tensor encoder_states;  ///< Complete masked encoder history[B,7,16].
  Tensor mu;              ///< Diagonal posterior means[B,4].
  Tensor log_variance;    ///< Posterior log variances[B,4], unclipped.
  Tensor z;               ///< Explicit Gaussian posterior sample[B,4].
  Tensor decoder_initial; ///< tanh(dense(z))[B,16], the decoder's latent input.
  Tensor decoder_states;  ///< Unmasked teacher decoder history[B,7,16].
  Tensor logits;          ///< Shared token projection[B,7,19].
  Tensor probability;     ///< Token softmax[B,7,19]; PAD steps still have raw probabilities.
  Tensor predicted_ids;   ///< Hard argmax Int32[B,7], including predictions at PAD targets.
  Tensor mask;            ///< Floating non-PAD target indicator[B,7].
  Tensor token_nll;       ///< Categorical cross entropy times target mask[B,7], nats.
  Tensor nll;             ///< Time-summed masked NLL[B], including EOS.
  Tensor kl;              ///< Analytic standard-normal KL[B], summed over four coordinates.
  Tensor correct;         ///< Number of correct non-PAD teacher predictions[B].
  Tensor valid_tokens;    ///< Number of non-PAD targets[B].
  Tensor loss;            ///< Scalar batch mean(nll+kl), not a token average.
};
namespace sequence_vae_detail {
/// @brief Shared source-ordered teacher kernel, optionally recording masked target losses.
/// @param p Mutable floating store with validated input geometry.
/// @param o Owning output with its z field already populated; other decoder fields are replaced.
/// @param inputs Valid integer prefixes[B,7].
/// @param tokens Optional validated targets[B,7], or null for prefix-only decoding.
inline void
Teacher(ParameterStore& p, SequenceVaeOutput& o, const Tensor& inputs, const Tensor* tokens) {
  o.decoder_initial     = dense(p, "decoder_initial", o.z, 16, Activation::Tanh);
  Tensor              h = o.decoder_initial;
  std::vector<Tensor> logits, states, probabilities, predictions, losses;
  for (int t = 0; t < 7; ++t) {
    const auto input = select(inputs, 1, t).contiguous();
    h                = gru_step(p, "decoder_gru", Embedding(p, "decoder_embedding", input), h, 1.0);
    const auto score = dense(p, "token_output", h, 19);
    logits.push_back(score);
    states.push_back(h);
    probabilities.push_back(softmax(score, 1));
    predictions.push_back(cast(argmax(score, 1), DType::Int32));
    if (tokens) {
      const auto target = select(*tokens, 1, t).contiguous();
      const auto ce     = -sum(cast(one_hot(target, 19), p.dtype()) * log_softmax(score, 1), {1});
      losses.push_back(ce * cast(not_equal(target, 0.0), p.dtype()));
    }
  }
  o.logits         = stack_time(logits);
  o.decoder_states = stack_time(states);
  o.probability    = stack_time(probabilities);
  o.predicted_ids  = stack_time(predictions);
  if (tokens)
    o.token_nll = stack_time(losses);
}
} // namespace sequence_vae_detail

/// @brief Own a target-free teacher decoder's initial state, full history and token scores.
///
/// Floating fields retain continuous z/parameter paths. Prefix IDs and hard
/// predictions have no derivative. Handles own lifetime, without a reference to
/// a model, input or fitting state.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore p(3109);
/// const auto out=n::models::SequenceTeacher(p,n::zeros({2,4}),n::ones({2,7},n::DType::Int32));
/// // Owns logits/probability[2,7,19], states[2,7,16] and initial[2,16].
/// @endcode
/// @see SequenceTeacher(), SequenceVaeOutput
struct SequenceTeacherOutput {
  Tensor initial;       ///< Latent-derived initial decoder state[B,16].
  Tensor states;        ///< Every decoder state[B,7,16].
  Tensor logits;        ///< Raw token scores[B,7,19].
  Tensor probability;   ///< Token softmax[B,7,19].
  Tensor predicted_ids; ///< Hard argmax Int32[B,7].
};

/// @brief Decode explicit previous-token prefixes causally from a fixed sentence code.
///
/// Fixed z[B,4] must be finite Float32/64 matching the store; prefixes are
/// Int32/64[B,7] IDs in[0,18]. No target tensor, encoder, loss, EOS stopping or RNG
/// is used. tanh(dense(z)) initializes a GRU16 with separate embedding[19,8].
/// Every step updates the state, including PAD inputs. Future prefixes cannot
/// change earlier logits at FIXED z. Creates11 decoder leaves/1,755 scalars.
/// IDs/products/batches/dtype are checked before leaves; incompatible leaves can
/// fail later. Strides and continuous gradients are retained, inputs unchanged,
/// outputs own lifetime. Synchronize shared stores. Fixed-architecture O(B*7)
/// work and activation storage; numerical overflow is not repaired.
/// @param parameters Mutable Float32/64 store.
/// @param z Finite continuous sentence codes[B,4], B>0.
/// @param inputs Explicit previous-token prefixes[B,7], already including BOS if wanted.
/// @return Owning initial state, decoder history, logits, probabilities and hard predictions.
/// @throws std::invalid_argument For invalid input/store or incompatible leaves.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore p(3109);
/// const auto out=n::models::SequenceTeacher(p,n::ones({1,4}),n::ones({1,7},n::DType::Int32));
/// const auto dz=n::grad(n::sum(out.logits),{out.initial});
/// // The decoder consumes supplied prefixes only; no target sequence is available.
/// @endcode
/// @par References
/// - Bowman et al. (2016), Generating Sentences from a Continuous Space,
///   https://arxiv.org/abs/1511.06349 . Compact GRU teaching variant, not the paper's recipe.
/// @see SequenceTeacherOutput, SequenceGenerate(), SequenceDecoderInputs()
inline SequenceTeacherOutput
SequenceTeacher(ParameterStore& parameters, const Tensor& z, const Tensor& inputs) {
  sequence_vae_detail::Store(parameters);
  sequence_vae_detail::Tokens(inputs, true);
  aae_detail::Matrix(z, 4, parameters.dtype(), false);
  if (z.size(0) != inputs.size(0))
    throw std::invalid_argument("sentence prefix/code batches must match");
  SequenceVaeOutput out;
  out.z = z;
  sequence_vae_detail::Teacher(parameters, out, inputs, nullptr);
  return {out.decoder_initial, out.decoder_states, out.logits, out.probability, out.predicted_ids};
}

/// @brief Evaluate the complete explicit-noise masked-GRU sentence variational autoencoder.
///
/// Int32/64 tokens[B,7] are vocabulary IDs[0,18], not necessarily grammatical or
/// EOS-terminated; all-PAD rows are legal and have zero reconstruction NLL.
/// Finite epsilon[B,4] matches the Float32/64 store. The PAD-masked GRU16 encoder
/// uses embedding8, two linear Gaussian heads4 and explicit reparameterization.
/// A separate embedding8/GRU16 decoder starts at tanh(dense(z)) and receives BOS
/// then previous targets. Cross entropy is computed with log_softmax, includes
/// EOS, masks PAD and sums time. Loss is mean(nll+kl), without token averaging,
/// clipping, warmup, word dropout, normalization or hidden fitting. Creates22
/// leaves/3,243 scalars in original order; original seed3109. Source Float32
/// operations and reduction order are retained. All continuous paths, including
/// BPTT and epsilon, stay live; input checks precede leaves. No mutation or RNG
/// beyond lazy initialization; outputs own lifetime. Shared stores require
/// synchronization. Work/storage O(B*7) for the fixed architecture. Computed
/// overflow/nonfinite parameters are not repaired; incompatible leaves may fail later.
/// @param parameters Mutable Float32/64 store.
/// @param tokens Nonempty seven-column Int32/64 target matrix.
/// @param epsilon Finite matching explicit Gaussian draws[B,4].
/// @return Eighteen owning endpoints, including unweighted scalar negative ELBO.
/// @throws std::invalid_argument For invalid shapes, dtype, IDs, values or incompatible leaves.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore p(3109);
/// const auto
/// out=n::models::SequenceVariationalAutoencoder(p,n::full({2,7},3,n::DType::Int32),n::ones({2,4}));
/// const auto gradients=n::grad(out.loss,p.parameters());
/// // All22 leaves are explicit; caller owns Adam and checkpoint selection.
/// @endcode
/// @par References
/// - Bowman et al. (2016), Generating Sentences from a Continuous Space,
///   https://arxiv.org/abs/1511.06349 . Small GRU/global-Gaussian teaching variant.
/// @see SequenceVaeOutput, SequenceEncoder(), SequenceTeacher(), GaussianReparameterize()
inline SequenceVaeOutput SequenceVariationalAutoencoder(
    ParameterStore& parameters,
    const Tensor&   tokens,
    const Tensor&   epsilon) {
  sequence_vae_detail::Store(parameters);
  sequence_vae_detail::Tokens(tokens, true);
  aae_detail::Matrix(epsilon, 4, parameters.dtype(), false);
  if (tokens.size(0) != epsilon.size(0))
    throw std::invalid_argument("sentence target/noise batches must match");
  SequenceVaeOutput out;
  const auto        encoded = SequenceEncoder(parameters, tokens);
  out.encoder_state         = encoded.state;
  out.encoder_states        = encoded.states;
  out.mu                    = dense(parameters, "mu", out.encoder_state, 4);
  out.log_variance          = dense(parameters, "log_variance", out.encoder_state, 4);
  out.z                     = GaussianReparameterize(out.mu, out.log_variance, epsilon);
  out.decoder_inputs        = SequenceDecoderInputs(tokens);
  sequence_vae_detail::Teacher(parameters, out, out.decoder_inputs, &tokens);
  out.mask         = cast(not_equal(tokens, 0.0), parameters.dtype());
  out.nll          = sum(out.token_nll, {1});
  out.valid_tokens = sum(out.mask, {1});
  out.correct = sum(out.mask * cast(equal(out.predicted_ids, tokens), parameters.dtype()), {1});
  out.kl      = DiagonalGaussianKl(out.mu, out.log_variance);
  out.loss    = mean(out.nll + out.kl, {0});
  return out;
}

/// @brief Form mean(NLL+beta*KL) for an explicit sentence-VAE KL annealing weight.
///
/// Reads only finite matching nonempty Float32/64 nll and kl vectors in net.
/// Other fields may be undefined. beta in[0,1] is cast by ordinary tensor scalar
/// arithmetic, preserving source Float32 rounding. Both tensor paths remain
/// connected at beta0, with zero KL derivative. No clipping, mutation, RNG or
/// optimizer. Owning scalar, O(B) work/storage; callers schedule beta themselves.
/// @param net Output with finite same-shape/dtype nll[B] and kl[B], B>0.
/// @param beta Finite annealing weight in[0,1].
/// @return Differentiable scalar batch-mean annealed objective, in nats per sentence.
/// @throws std::invalid_argument For invalid vectors, dtype, values or beta.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::models::SequenceVaeOutput out; out.nll=n::ones({2}); out.kl=n::full({2},2.);
/// const auto loss=n::models::AnnealedSequenceObjective(out,.5); // 2 nats.
/// @endcode
/// @par References
/// - Bowman et al. (2016), https://arxiv.org/abs/1511.06349 , KL cost annealing.
/// @see SequenceVariationalAutoencoder(), CollapseBeta()
inline Tensor AnnealedSequenceObjective(const SequenceVaeOutput& net, double beta) {
  if (!net.nll.defined() || !net.kl.defined() || net.nll.dim() != 1 ||
      net.kl.shape() != net.nll.shape() || !std::isfinite(beta) || beta < 0 || beta > 1)
    throw std::invalid_argument("sentence objective requires matching vectors and beta in[0,1]");
  gaussian_detail::Match(
      reshape(net.nll, {net.nll.numel(), 1}), reshape(net.kl, {net.kl.numel(), 1}));
  return mean(net.nll + beta * net.kl, {0});
}

/// @brief Own emitted sentence IDs, all raw decoder scores and per-row EOS flags.
///
/// Hard token choices are discrete and stop their own gradient paths. Floating
/// logits/probabilities retain continuous z/parameter paths along the chosen
/// prefixes; they are not a straight-through estimator. No input references or RNG state.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore p(3109);
/// const auto out=n::models::SequenceGenerate(p,n::zeros({2,4}));
/// // IDs[2,7] are PAD after first EOS; raw logits/probabilities still describe every step.
/// @endcode
/// @see SequenceGenerate(), SequenceTeacher()
struct SequenceGeneration {
  Tensor ids;         ///< Owning Int32[B,7], with PAD suffix after first EOS2.
  Tensor logits;      ///< Raw differentiable decoder scores[B,7,19], even after EOS.
  Tensor probability; ///< Raw differentiable token softmax[B,7,19], even after EOS.
  Tensor ended;       ///< Int32[B] indicator that EOS was emitted within seven steps.
};

/// @brief Generate seven target-free greedy or ancestral tokens from explicit sentence codes.
///
/// Starts decoder at tanh(dense(z)), previous token BOS1, then feeds each emitted
/// token back. Greedy uses first-index argmax. Sampling renormalizes Float64-cast
/// probabilities and chooses min(18,count(cdf<=u)) with supplied open-interval
/// Float64 uniforms[B,7]. z is finite matching Float32/64[B,4], B>0. After EOS2,
/// emitted IDs become PAD0 and remain PAD; raw states/scores still evolve for all
/// seven steps. No targets, encoder, hidden draws, NoGrad guard or fitting. Greedy
/// ignores uniforms. Sampling comparisons and hard prefixes have no derivative;
/// raw scores retain continuous paths for fixed chosen prefixes. Creates/reuses
/// the same11 decoder leaves/1,755 scalars as SequenceTeacher. Checks precede
/// leaves; incompatible leaves can fail later. Strides accepted, owning outputs,
/// inputs unchanged. Synchronize shared stores. Fixed O(B*7) work/storage.
/// @param parameters Mutable Float32/64 store.
/// @param z Finite matching explicit latent codes[B,4].
/// @param sample False for greedy, true for ancestral categorical sampling.
/// @param uniforms Explicit Float64[B,7] values strictly between0 and1 when sampling; ignored
/// otherwise.
/// @return Owning emitted IDs, raw logits/probabilities and EOS flags.
/// @throws std::invalid_argument For invalid inputs/store, uniforms or incompatible leaves.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore p(3109);
/// const auto
/// out=n::models::SequenceGenerate(p,n::zeros({2,4}),true,n::full({2,7},.5,n::DType::Float64));
/// // Every categorical choice uses the supplied uniform; no RNG is hidden.
/// @endcode
/// @par References
/// - Bowman et al. (2016), https://arxiv.org/abs/1511.06349 . Autoregressive decoding from a global
/// code.
/// @see SequenceGeneration, SequenceTeacher(), SequenceVariationalAutoencoder()
inline SequenceGeneration SequenceGenerate(
    ParameterStore& parameters,
    const Tensor&   z,
    bool            sample   = false,
    const Tensor&   uniforms = {}) {
  sequence_vae_detail::Store(parameters);
  aae_detail::Matrix(z, 4, parameters.dtype(), false);
  sequence_vae_detail::Geometry(z.size(0), 7);
  const auto          n = z.size(0);
  std::vector<double> u;
  if (sample) {
    aae_detail::Matrix(uniforms, 7, DType::Float64, false);
    if (uniforms.size(0) != n)
      throw std::invalid_argument("sentence sampling batches must match");
    u = uniforms.contiguous().to_vector<double>();
    for (double v : u)
      if (v <= 0 || v >= 1)
        throw std::invalid_argument("sentence sampling requires open-interval uniforms");
  }
  std::vector<int>    previous(n, 1), ids(n * 7, 0), ended(n, 0);
  Tensor              h = dense(parameters, "decoder_initial", z, 16, Activation::Tanh);
  std::vector<Tensor> logits, probabilities;
  for (int t = 0; t < 7; ++t) {
    h = gru_step(
        parameters,
        "decoder_gru",
        sequence_vae_detail::Embedding(parameters, "decoder_embedding", from_vector(previous, {n})),
        h,
        1.0);
    const auto       score       = dense(parameters, "token_output", h, 19);
    const auto       probability = softmax(score, 1);
    std::vector<int> chosen(n);
    if (sample) {
      const auto p64 = cast(probability, DType::Float64);
      const auto cdf = cumsum(p64 / sum(p64, {1}, true), 1).to_vector<double>();
      for (std::int64_t b = 0; b < n; ++b) {
        int count = 0;
        for (int w = 0; w < 19; ++w)
          count += u[b * 7 + t] >= cdf[b * 19 + w];
        chosen[b] = std::min(18, count);
      }
    } else
      chosen = argmax(score, 1).to_vector<int>();
    for (std::int64_t b = 0; b < n; ++b) {
      const int emitted = ended[b] ? 0 : chosen[b];
      ended[b]          = ended[b] || emitted == 2;
      previous[b]       = emitted;
      ids[b * 7 + t]    = emitted;
    }
    logits.push_back(score);
    probabilities.push_back(probability);
  }
  return {
      from_vector(ids, {n, 7}),
      stack_time(logits).contiguous(),
      stack_time(probabilities).contiguous(),
      from_vector(ended, {n})};
}
} // namespace datamunge::nn::models
