#pragma once

/// @file sequence.hpp
/// @brief Structured sequence losses and decoders: CTC (OCR, speech) and the
/// linear-chain CRF (BiLSTM-CRF tagging).
///
/// CTC follows TensorFlow's CTCLoss kernel (and tf.nn.ctc_loss with dense
/// labels): log-space forward-backward, `ctc_merge_repeated` on, and the
/// gradient with respect to the logits computed in the forward pass.
///
/// The CRF functions follow tensorflow_addons' tfa.text.crf_* semantics,
/// with optional start/end score vectors (as in tflow's LinearChainCrf):
///   score(y) = sum_t U[t, y_t] + sum_{t>0} A[y_{t-1}, y_t] + start[y_0] + end[y_{n-1}]
/// with `transitions` indexed [from, to]. A sequence of length 0 has score 0
/// and log-normaliser 0.

#include <cstdint>
#include <vector>

#include "datamunge/nn/tensor.hpp"

namespace datamunge::nn {

// ---- CTC ------------------------------------------------------------------------------------

struct CtcOptions {
  /// Blank class. Negative values count from the number of classes C, so the
  /// default -1 is C - 1 as in TF's CTCLoss op (tf.compat.v1.nn.ctc_loss);
  /// tf.nn.ctc_loss (v2) defaults to blank 0, so pass 0 to match it.
  std::int64_t blank{-1};
  /// Logits are [T, B, C] when true (TF default), [B, T, C] otherwise.
  bool time_major{true};
  /// TF `ignore_longer_outputs_than_inputs`: when a label cannot be aligned
  /// in its logit length (L + repeats > T), give loss 0 and gradient 0
  /// instead of throwing.
  bool ignore_longer_outputs_than_inputs{false};
};

/// CTC negative log-likelihood per example, shape [B], in the logits' dtype.
///
/// `logits` are unnormalised scores (a log-softmax over the class axis is
/// applied inside). `labels` is a dense int tensor [B, L_max]; entries past
/// `label_lengths[b]` are ignored (any padding value). `logit_lengths[b]`
/// (<= T) is the number of valid frames; later frames get zero gradient.
/// Labels must lie in [0, C) and differ from the blank.
///
/// A recorded primitive: like TF's CTCLoss, the gradient (softmax minus the
/// alignment posterior) is computed with the loss and saved, so the backward
/// pass is first order only. Gradients with respect to the upstream gradient
/// are exact, but second derivatives with respect to the logits are not
/// (they come out as zero), matching TF, whose CTCLoss is not twice
/// differentiable either.
Tensor ctc_loss(const Tensor& logits, const Tensor& labels, const Tensor& label_lengths, const Tensor& logit_lengths,
                const CtcOptions& options = {});

/// Loss and TF CTCLoss's second output: d loss[b] / d logits, in the logits'
/// layout, before any reduction over the batch. The loss is recorded as in
/// `ctc_loss`; the gradient has no history.
struct CtcLossResult {
  Tensor loss, gradient;
};
CtcLossResult ctc_loss_with_gradient(const Tensor& logits, const Tensor& labels, const Tensor& label_lengths,
                                     const Tensor& logit_lengths, const CtcOptions& options = {});

/// Collapses a frame-level path: merge repeated classes (when
/// `merge_repeated`), then drop blanks (TF's order: "a-a" keeps both a's).
std::vector<std::int64_t> ctc_collapse(const std::vector<std::int64_t>& path, std::int64_t blank,
                                       bool merge_repeated = true);

struct CtcDecoded {
  /// Decoded label sequence per batch element.
  std::vector<std::vector<std::int64_t>> sequences;
  /// The same as an int64 [B, max_len] tensor padded with -1.
  Tensor dense;
  /// [B, 1]: minus the sum over valid frames of the largest raw logit (what
  /// TF's CTCGreedyDecoder returns as "log_probability", i.e. neg_sum_logits).
  Tensor neg_sum_logits;
  /// [B]: log-probability of the greedy frame path under log_softmax(logits).
  Tensor path_log_prob;
};

/// Best-path (greedy) CTC decoding (TF ctc_greedy_decoder): the argmax class
/// at each of the first `logit_lengths[b]` frames (first maximum on ties),
/// then `ctc_collapse`. An undefined `logit_lengths` uses every frame.
/// `options.blank` and `options.time_major` are honoured.
CtcDecoded ctc_greedy_decode(const Tensor& logits, const Tensor& logit_lengths = Tensor(),
                             const CtcOptions& options = {}, bool merge_repeated = true);

// ---- linear-chain CRF ------------------------------------------------------------------------

/// Unnormalised score of the tag sequences `tags` [B, T] (int) under unary
/// `potentials` [B, T, K] and `transitions` [K, K] ([from, to]), counting
/// only the first `lengths[b]` positions (tfa.text.crf_sequence_score).
/// `start` / `end` [K] are optional (undefined = none). Differentiable in
/// the potentials, transitions, start and end; returns [B].
Tensor crf_sequence_score(const Tensor& potentials, const Tensor& tags, const Tensor& lengths,
                          const Tensor& transitions, const Tensor& start = Tensor(), const Tensor& end = Tensor());

/// log Z: log-sum-exp of the score over all K^len tag sequences, by the
/// forward algorithm (tfa.text.crf_log_norm). Built from differentiable ops,
/// so gradients of any order flow; the gradient with respect to the
/// potentials is the posterior marginals. Returns [B]; 0 where lengths is 0.
Tensor crf_log_norm(const Tensor& potentials, const Tensor& lengths, const Tensor& transitions,
                    const Tensor& start = Tensor(), const Tensor& end = Tensor());

/// Forward variables of `crf_log_norm` (log alpha) for every position, [B, T, K];
/// positions past a sequence's length repeat its last alpha, and an empty
/// sequence's alphas are the start scores (zeros without `start`).
Tensor crf_forward(const Tensor& potentials, const Tensor& lengths, const Tensor& transitions,
                   const Tensor& start = Tensor());

/// tfa.text.crf_log_likelihood: crf_sequence_score - crf_log_norm, [B].
/// The negative mean is the usual BiLSTM-CRF training loss.
Tensor crf_log_likelihood(const Tensor& potentials, const Tensor& tags, const Tensor& lengths,
                          const Tensor& transitions, const Tensor& start = Tensor(), const Tensor& end = Tensor());

struct CrfDecoded {
  /// Best tag sequence [B, T] (int32); positions past the length hold `pad_tag`.
  Tensor tags;
  /// Score of the best sequence [B] (float64; 0 for empty sequences).
  Tensor scores;
};

/// Viterbi decoding (tfa.text.crf_decode), computed in double precision.
/// Ties go to the smallest tag index. tfa leaves 0 in padded positions
/// (the default); tflow's decoder uses -1.
CrfDecoded crf_decode(const Tensor& potentials, const Tensor& lengths, const Tensor& transitions,
                      const Tensor& start = Tensor(), const Tensor& end = Tensor(), std::int32_t pad_tag = 0);

}  // namespace datamunge::nn
