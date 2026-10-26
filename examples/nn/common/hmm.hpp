// datamunge::nn port of tflow_examples src/common/ml/hmm.h.
// Batched categorical hidden Markov models with float64 nn ops.
// Fixed sequence length per model, any batch size. Scaled forward/backward
// recurrences avoid products of tiny probabilities. Viterbi uses log scores;
// the host only follows integer backpointers. Baum-Welch pools expected counts
// across sequences, normalizes rows, and keeps any zero-count row unchanged.
// No pseudocounts: the update maximizes ordinary observation likelihood.
// (nn::sequence.hpp's CRF routines work in log space with unnormalized
// potentials; this keeps tflow's probability-space scaled recurrences.)
#pragma once

#include <cmath>
#include <cstdint>
#include <datamunge/nn/nn.hpp>
#include <stdexcept>
#include <vector>

namespace ml {
namespace nn = datamunge::nn;

/// initial [K], transition [K,K], emission [K,V]; all float64 probability rows.
struct HmmState {
  nn::Tensor initial, transition, emission;
};
struct HmmResult {
  HmmState   next;
  nn::Tensor log_likelihood, posterior, viterbi_score;  // [B], [B,T,K], [B]
  nn::Tensor path;                                      // int32 [B,T]
};

class HiddenMarkovModel {
 public:
  HiddenMarkovModel(int states, int symbols, int length) : k_(states), v_(symbols), t_(length) {
    if (k_ <= 0 || v_ <= 0 || t_ <= 0) throw std::invalid_argument("positive HMM dimensions required");
  }

  /// observations: int32 [B,T] symbols in [0,V).
  HmmResult Run(const nn::Tensor& observations, const HmmState& state) const {
    Validate(observations, state);
    nn::NoGradGuard guard;
    const auto& transition = state.transition;
    const auto  emission_probs = nn::gather(nn::transpose(state.emission), observations);  // [B,T,K]
    const auto  log_transition = nn::log(transition);
    std::vector<nn::Tensor> evidence, alpha, beta(t_), scale, log_scale, posterior, transitions, pointers;
    nn::Tensor              delta;
    for (int t = 0; t < t_; ++t) {
      evidence.push_back(nn::select(emission_probs, 1, t));  // [B,K]
      const auto prediction = t == 0 ? state.initial : nn::matmul(alpha.back(), transition);
      const auto joint      = prediction * evidence.back();
      scale.push_back(nn::sum(joint, {1}, true));
      alpha.push_back(joint / scale.back());
      log_scale.push_back(nn::log(scale.back()));
      if (t == 0) {
        delta = nn::log(state.initial) + nn::log(evidence.back());
        pointers.push_back(nn::zeros(delta.shape(), nn::DType::Int64));
      } else {
        const auto candidates = nn::expand_dims(delta, 2) + log_transition;  // [B,from,to]
        pointers.push_back(nn::argmax(candidates, 1));
        delta = nn::max(candidates, {1}) + nn::log(evidence.back());
      }
    }
    HmmResult out;
    out.log_likelihood     = nn::reshape(nn::sum(nn::concat(log_scale, 1), {1}), {-1});
    const auto final_state = nn::argmax(delta, 1);
    out.viterbi_score      = nn::max(delta, {1});
    const auto ptrs        = nn::stack(pointers, 1);  // [B,T,K]
    beta.back()            = nn::ones_like(alpha.back());
    for (int t = t_ - 2; t >= 0; --t)
      beta[t] = nn::matmul(evidence[t + 1] * beta[t + 1], transition, false, true) / scale[t + 1];
    for (int t = 0; t < t_; ++t) {
      const auto joint = alpha[t] * beta[t];
      posterior.push_back(joint / nn::sum(joint, {1}, true));
      if (t + 1 < t_) {
        const auto pair = nn::expand_dims(alpha[t], 2) * transition * nn::expand_dims(evidence[t + 1] * beta[t + 1], 1);
        transitions.push_back(nn::sum(pair / nn::sum(pair, {1, 2}, true), {0}));
      }
    }
    out.posterior     = nn::stack(posterior, 1);
    out.next.initial  = nn::mean(posterior.front(), {0});
    const auto keep   = [](const nn::Tensor& counts, const nn::Tensor& old) {
      const auto total = nn::sum(counts, {1}, true);
      return nn::where(nn::greater(total, 0.0), counts / nn::maximum(total, nn::scalar_like(1e-300, total)), old);
    };
    if (t_ > 1) {
      nn::Tensor sum = transitions[0];
      for (size_t i = 1; i < transitions.size(); ++i) sum = sum + transitions[i];
      out.next.transition = keep(sum, transition);
    } else {
      out.next.transition = transition.clone();
    }
    const auto one_hot_symbols = nn::one_hot(nn::reshape(observations, {-1}), v_, 1.0, 0.0, nn::DType::Float64);
    const auto emission_counts = nn::matmul(nn::reshape(out.posterior, {-1, k_}), one_hot_symbols, true, false);
    out.next.emission          = keep(emission_counts, state.emission);

    const auto  batch = observations.size(0);
    const auto  ll = out.log_likelihood.to_vector<double>(), vs = out.viterbi_score.to_vector<double>();
    const auto  last = final_state.to_vector<std::int64_t>(), back = ptrs.to_vector<std::int64_t>();
    std::vector<std::int32_t> path(static_cast<size_t>(batch * t_));
    for (std::int64_t b = 0; b < batch; ++b) {
      if (!std::isfinite(ll[b]) || !std::isfinite(vs[b]))
        throw std::runtime_error("observation sequence has zero or invalid HMM probability");
      std::int64_t s = last[b];
      for (int t = t_ - 1; t >= 0; --t) {
        path[b * t_ + t] = static_cast<std::int32_t>(s);
        if (t) s = back[(b * t_ + t) * k_ + s];
      }
    }
    out.path = nn::from_vector(path, {batch, t_});
    return out;
  }

 private:
  void Validate(const nn::Tensor& observations, const HmmState& state) const {
    const auto f64 = nn::DType::Float64;
    if (observations.dtype() != nn::DType::Int32 || observations.dim() != 2 || observations.size(0) < 1 ||
        observations.size(1) != t_ || state.initial.dtype() != f64 || state.initial.shape() != nn::Shape{k_} ||
        state.transition.dtype() != f64 || state.transition.shape() != nn::Shape{k_, k_} ||
        state.emission.dtype() != f64 || state.emission.shape() != nn::Shape{k_, v_})
      throw std::invalid_argument("HMM observation/state shape or dtype mismatch");
    for (auto symbol : observations.to_vector<std::int32_t>())
      if (symbol < 0 || symbol >= v_) throw std::invalid_argument("observation symbol out of range");
    const auto validate_rows = [](const nn::Tensor& p, int width) {
      const auto values = p.to_vector<double>();
      for (size_t start = 0; start < values.size(); start += width) {
        double sum = 0;
        for (int j = 0; j < width; ++j) {
          const double probability = values[start + j];
          if (!std::isfinite(probability) || probability < 0 || probability > 1)
            throw std::invalid_argument("invalid HMM probability");
          sum += probability;
        }
        if (std::abs(sum - 1) > 1e-10) throw std::invalid_argument("HMM probability row must sum to one");
      }
    };
    validate_rows(state.initial, k_);
    validate_rows(state.transition, k_);
    validate_rows(state.emission, v_);
  }

  int k_, v_, t_;
};

}  // namespace ml
