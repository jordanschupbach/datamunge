#pragma once

/// @file rnn_transducer.hpp
/// @brief Differentiable RNN-T lattice loss and spoken-digit transducer.

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/ops.hpp"

namespace datamunge::nn {

/// @brief Compute mean negative log likelihood over monotone RNN-T paths.
/// A blank advances time. A label advances transcript position at the same
/// time. A final blank advances beyond the last frame. Batches have fixed
/// input and transcript lengths, with zero-length transcripts supported.
/// @param log_probs Log probabilities [batch,time,labels+1,classes].
/// @param labels Integer nonblank class IDs [batch,labels].
/// @param blank Blank class index.
/// @return Differentiable scalar negative log likelihood.
/// @throws std::invalid_argument For invalid dimensions, dtype or labels.
/// @par Example
/// @code
/// auto loss = dm::nn::rnnt_loss(dm::nn::log_softmax(dm::nn::zeros({1,2,2,3})),
///                             dm::nn::zeros({1,1},dm::nn::DType::Int32),2);
/// @endcode
/// @par References
/// - Graves (2012), Sequence Transduction with Recurrent Neural Networks. https://arxiv.org/abs/1211.3711
inline Tensor rnnt_loss(const Tensor& log_probs, const Tensor& labels, std::int64_t blank) {
  if (!log_probs.defined() || !labels.defined() || log_probs.dim()!=4 || labels.dim()!=2 ||
      log_probs.size(0)<1 || log_probs.size(1)<1 || log_probs.size(0)!=labels.size(0) ||
      log_probs.size(2)-1!=labels.size(1) || blank<0 || blank>=log_probs.size(3) ||
      (log_probs.dtype()!=DType::Float32 && log_probs.dtype()!=DType::Float64) ||
      (labels.dtype()!=DType::Int32 && labels.dtype()!=DType::Int64) ||
      (labels.numel()>0 && (min(labels).item()<0 || max(labels).item()>=log_probs.size(3) ||
                            sum(equal(labels,blank)).item()!=0))) {
    throw std::invalid_argument("Invalid RNN-T lattice or labels");
  }
  const auto batch=labels.size(0), length=labels.size(1);
  const auto ids=labels.to(DType::Int64);
  const auto rows=arange(0,batch,1,DType::Int64);
  std::vector<Tensor> previous(length+1), current(length+1);
  for (std::int64_t t=0;t<log_probs.size(1);++t) {
    const auto row=select(log_probs,1,t);
    current[0]=t==0 ? zeros({batch},log_probs.dtype()) : previous[0];
    for (std::int64_t u=1;u<=length;++u) {
      const auto symbol=stack({rows,select(ids,1,u-1)},1);
      const auto from_label=current[u-1]+gather_nd(select(row,1,u-1),symbol);
      current[u]=t==0 ? from_label : logsumexp(stack({previous[u],from_label}),{0});
    }
    for (std::int64_t u=0;u<=length;++u)
      previous[u]=current[u]+select(select(row,1,u),1,blank);
  }
  return -mean(previous.back());
}

/// @brief Compact GRU encoder, label predictor and joint network for spoken digits.
/// @par Example
/// @code
/// dm::nn::ParameterStore weights(67231);
/// dm::nn::DigitTransducer model(weights);
/// auto tokens = model.decode(dm::nn::zeros({1,28,24}));
/// @endcode
class DigitTransducer {
 public:
  /// @brief Borrow a parameter store that outlives this model.
  /// @param model Mutable store with lazily initialized parameters.
  explicit DigitTransducer(ParameterStore& model) : model_(model) {}

  /// @brief Build normalized alignment scores with reference-history conditioning.
  /// @param audio Floating log-mel input [batch, even frames, bands].
  /// @param labels Digit IDs [batch,length].
  /// @return Log probabilities [batch, frames/2, length+1, 11], blank at index 10.
  /// @throws std::invalid_argument For invalid audio, labels or parameters.
  /// @see rnnt_loss()
  Tensor lattice(const Tensor& audio, const Tensor& labels) {
    if (!audio.defined() || audio.dim()!=3 || !labels.defined() || labels.dim()!=2 ||
        labels.size(0)!=audio.size(0) ||
        (labels.dtype()!=DType::Int32 && labels.dtype()!=DType::Int64) ||
        (labels.numel()>0 && (min(labels).item()<0 || max(labels).item()>9))) {
      throw std::invalid_argument("Invalid transducer labels");
    }
    const auto encoded=encode(audio);
    const auto batch=audio.size(0);
    auto state=predict(full({batch},11,DType::Int64),zeros({batch,20},audio.dtype()));
    std::vector<Tensor> history{state};
    for (std::int64_t u=0;u<labels.size(1);++u) {
      state=predict(select(labels,1,u),state);
      history.push_back(state);
    }
    return log_softmax(join(expand_dims(encoded,2),expand_dims(stack(history,1),1)));
  }

  /// @brief Decode digits from audio using greedy blank and label transitions.
  /// @param audio Floating log-mel input [batch, even frames, bands].
  /// @param limit Maximum emitted digits per example, default four.
  /// @return Integer [batch,limit] output padded with -1.
  /// @throws std::invalid_argument For invalid audio, cap or parameters.
  /// @see lattice()
  Tensor decode(const Tensor& audio, int limit = 4) {
    if (limit<1) throw std::invalid_argument("Transducer output cap must be positive");
    NoGradGuard guard;
    const auto encoded=encode(audio);
    std::vector<std::int32_t> output(encoded.size(0)*limit,-1);
    for (std::int64_t row=0;row<encoded.size(0);++row) {
      auto state=predict(full({1},11,DType::Int64),zeros({1,20},audio.dtype()));
      int emitted=0;
      for (std::int64_t t=0;t<encoded.size(1) && emitted<limit;++t) {
        const auto frame=select(slice_axis(encoded,0,row,row+1),1,t);
        while (emitted<limit) {
          const auto best=argmax(join(frame,state));
          const auto token=static_cast<int>(best.item());
          if (token==10) break;
          output[row*limit+emitted++]=token;
          state=predict(best,state);
        }
      }
    }
    return from_vector(output,{encoded.size(0),limit});
  }

 private:
  ParameterStore& model_; ///< Borrowed trainable model parameters.

  /// @brief Pair adjacent log-mel frames and encode them with a causal GRU.
  /// @param audio Log-mel input with positive axes and an even frame count.
  /// @return Encoded tensor [batch, frames/2, 20].
  Tensor encode(const Tensor& audio) {
    if (!audio.defined() || audio.dim()!=3 || audio.size(0)<1 || audio.size(1)<2 ||
        audio.size(1)%2!=0 || audio.size(2)<1 || audio.dtype()!=model_.dtype()) {
      throw std::invalid_argument("Invalid transducer audio");
    }
    const auto frames=reshape((audio+6)*.2,{audio.size(0),audio.size(1)/2,2*audio.size(2)});
    auto state=zeros({audio.size(0),20},audio.dtype());
    std::vector<Tensor> output;
    for (const auto& frame : time_steps(frames)) {
      state=gru_step(model_,"encoder",frame,state);
      output.push_back(state);
    }
    return stack(output,1);
  }

  /// @brief Update the prediction GRU with a token.
  /// @param token Batched token IDs including the BOS symbol.
  /// @param state Previous prediction state.
  /// @return Next prediction state.
  Tensor predict(const Tensor& token,const Tensor& state) {
    return gru_step(model_,"predictor",one_hot(token,12,1,0,model_.dtype()),state);
  }

  /// @brief Combine broadcast-compatible acoustic and predictor states.
  /// @param frame Acoustic state or sequence with 20 features.
  /// @param state Predictor state or sequence with 20 features.
  /// @return Digit and blank logits.
  Tensor join(const Tensor& frame,const Tensor& state) {
    const auto hidden=tanh(dense(model_,"acoustic",frame,20)+dense(model_,"history",state,20));
    return dense(model_,"head",hidden,11);
  }
};

}  // namespace datamunge::nn
