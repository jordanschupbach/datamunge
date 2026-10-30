#pragma once
/// @file character_cnn_models.hpp
/// @brief Source raw-byte metadata encoder and complete six-convolution character classifier.
#include "datamunge/nn/kim_cnn_models.hpp"
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
using nn::Tensor;
/// @brief Expose the source69-slot raw-character inventory,including its repeated hyphen.
///
/// String/ID metadata only. Lookup uses the last duplicate-hyphen position,
/// so the earlier slot is structurally unused by the source byte encoder.
/// Space and unknown bytes are absent and map to-1,not a learned UNK token.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto index = m::CharacterAlphabet.rfind('-');
/// @endcode
/// @par References
/// - Zhang,Zhao and LeCun(2015),Character-level Convolutional Networks for Text Classification, https://arxiv.org/abs/1509.01626. Source binary MR adaptation keeps six7/7/3/3/3/3 convolutions,three VALID3pools,two dense/ReLU/dropout layers and the69-slot alphabet. Teaching source uses He-uniform conv/Glorot dense initialization,not a claim of exact paper training or benchmark accuracy.
/// @see CharacterEncode()
inline const std::string CharacterAlphabet =
    "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}";

/// @brief Encode reversed final source bytes with ASCII lowercase and structural zero slots.
///
/// Reverse the original byte sequence,retain its last time bytes in reversed
/// order,lowercase ASCII A..Z only,and look up the LAST matching alphabet slot.
/// Spaces,unknown bytes and padding map-1,which becomes an all-zero onehot.
/// Empty text yields all-1. This is categorical string metadata encoding;
/// all numerical onehot/convolution/objective computations remain tensors.
///
/// @param text Raw byte string;MR source uses Latin1/ASCII,not Unicode grapheme processing.
/// @param time Positive number of output metadata positions.
/// @return Length-time integer vector,known alphabet IDs0..68 and unknown/right PAD=-1.
/// @throws std::invalid_argument For nonpositive requested output time.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto ids = m::CharacterEncode("Ab -?", 7);
/// @endcode
/// @par References
/// - Zhang,Zhao and LeCun(2015),Character-level Convolutional Networks for Text Classification, https://arxiv.org/abs/1509.01626. Source binary MR adaptation keeps six7/7/3/3/3/3 convolutions,three VALID3pools,two dense/ReLU/dropout layers and the69-slot alphabet. Teaching source uses He-uniform conv/Glorot dense initialization,not a claim of exact paper training or benchmark accuracy.
/// @see CharacterAlphabet, CharacterCnn()
inline std::vector<int> CharacterEncode(const std::string &text, int time) {
  if (time < 1)
    throw std::invalid_argument("positive character metadata time required");
  std::vector<int> out(time, -1);
  for (int i = 0; i < std::min<int>(time, static_cast<int>(text.size())); ++i) {
    unsigned char c = text[text.size() - 1 - i];
    if (c >= 'A' && c <= 'Z')
      c += 32;
    const auto position = CharacterAlphabet.rfind(static_cast<char>(c));
    if (position != std::string::npos)
      out[i] = static_cast<int>(position);
  }
  return out;
}

/// @brief Configure the six-convolution raw-character classifier and fixed onehot depth.
///
/// Metadata only. Minimum supported time123 gives final length1. The full
/// original-small shape uses1014/256/1024/69,with a source-adapted binary head.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// m::CharacterConfig c{256, 16, 64, 69};
/// @endcode
/// @par References
/// - Zhang,Zhao and LeCun(2015),Character-level Convolutional Networks for Text Classification, https://arxiv.org/abs/1509.01626. Source binary MR adaptation keeps six7/7/3/3/3/3 convolutions,three VALID3pools,two dense/ReLU/dropout layers and the69-slot alphabet. Teaching source uses He-uniform conv/Glorot dense initialization,not a claim of exact paper training or benchmark accuracy.
/// @see CharacterFinalLength(), CharacterCnn()
struct CharacterConfig {
  int time = 1014;    ///< Fixed raw-character positions;original-small1014,compact256.
  int channels = 256; ///< Positive width for all six convs;compact16.
  int dense = 1024;   ///< Positive width of both dense/ReLU layers;compact64.
  int alphabet = 69;  ///< Positive onehot feature depth,source69 including repeated hyphen.
};

/// @brief Compute the exact retained length after six VALID convolutions and three pools.
///
/// Convolution kernels7/7/3/3/3/3;VALID3stride3 pooling follows conv1,2,6.
/// Integer shape arithmetic only,no numerical signal or learned state.
///
/// @param length Original positive number of input character positions.
/// @return Final temporal length;1014->34,256->5,123->1.
/// @throws std::invalid_argument For an input too short to retain any required convolution/pool position.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto length = m::CharacterFinalLength(256); // 5
/// @endcode
/// @par References
/// - Zhang,Zhao and LeCun(2015),Character-level Convolutional Networks for Text Classification, https://arxiv.org/abs/1509.01626. Source binary MR adaptation keeps six7/7/3/3/3/3 convolutions,three VALID3pools,two dense/ReLU/dropout layers and the69-slot alphabet. Teaching source uses He-uniform conv/Glorot dense initialization,not a claim of exact paper training or benchmark accuracy.
/// @see CharacterCnn()
inline int CharacterFinalLength(int length) {
  if(length<1)throw std::invalid_argument("positive original character time required");
  for (int layer = 0; layer < 6; ++layer) {
    length -= layer < 2 ? 6 : 2;
    if (length < 1)
      throw std::invalid_argument("character CNN input too short for six convolutions");
    if (layer == 0 || layer == 1 || layer == 5) {
      if (length < 3)
        throw std::invalid_argument("character CNN input too short for pool");
      length = (length - 3) / 3 + 1;
    }
    if (length < 1)
      throw std::invalid_argument("character CNN input too short for pool");
  }
  return length;
}

/// @brief Own raw-character onehot ingress,nine temporal activations and binary logits.
///
/// Tensor handles own graph/storage lifetimes. IDs and lengths are metadata;
/// there is no learned character embedding or additional normalization state.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p(2);
/// const auto out =
///     m::CharacterCnn(p, n::zeros({2, 256}, n::DType::Int32),
///                     n::full({2}, 200., n::DType::Int32), m::CharacterConfig{256, 16, 64, 69});
/// @endcode
/// @par References
/// - Zhang,Zhao and LeCun(2015),Character-level Convolutional Networks for Text Classification, https://arxiv.org/abs/1509.01626. Source binary MR adaptation keeps six7/7/3/3/3/3 convolutions,three VALID3pools,two dense/ReLU/dropout layers and the69-slot alphabet. Teaching source uses He-uniform conv/Glorot dense initialization,not a claim of exact paper training or benchmark accuracy.
/// @see CharacterCnn()
struct CharacterOutput {
  Tensor logits;    ///< Raw binary[N,2] head scores.
  Tensor quantized; ///< Floating[N,time,alphabet] observed-length-masked onehot metadata.
  Tensor flat;      ///< Flattened final pool[N,final_length*channels] before dense layers.
  std::vector<Tensor>
      stages; ///< Six convolution and three pooling activations,in execution order.
};

/// @brief Classify raw observed character IDs with the complete six-convolution source topology.
///
/// Tensor one_hot maps-1 to an all-zero vector;observed-length masking also
/// zeros every padded position regardless its legal ID. Six He-uniform VALID
/// convolutions/ReLU use7/7/3/3/3/3 with3stride3 pools after1,2,6. Flatten
/// in time/channel order,two Glorot dense/ReLU layers with explicit-generator
/// dropout,then Glorot linear2class head. No learned embedding,BN,source text,
/// labels or optimizer enter forward. Fully unknown inputs still follow learned
/// bias paths;no fabricated character is inserted. Width scaling preserves
/// all six convs/three pools/both hidden layers and18 learned leaves.
///
/// @param p Mutable learned store;warm all18 leaves before optimizer creation.
/// @param ids Int32/64[N,time] categorical IDs-1..alphabet-1,N>0;unknown/PAD=-1.
/// @param lengths Int32/64[N] observed lengths0..time;outside-length positions mask to0.
/// @param config Valid retained time,positive conv/dense widths and onehot depth.
/// @param training False by default;true activates caller-generator dropout after both dense ReLUs.
/// @param keep Finite dropout keep probability(0,1],default1;source fit.5.
/// @param gen Required mutable generator only when training&&keep<1;otherwise unused.
/// @return Raw binary logits,masked onehot inputs,flattened pool and all9temporal stages.
/// @throws std::invalid_argument For invalid architecture,input IDs/lengths,dropout policy or stored leaves.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p(2);
/// const auto out =
///     m::CharacterCnn(p, n::zeros({2, 256}, n::DType::Int32),
///                     n::full({2}, 200., n::DType::Int32), m::CharacterConfig{256, 16, 64, 69});
/// @endcode
/// @par References
/// - Zhang,Zhao and LeCun(2015),Character-level Convolutional Networks for Text Classification, https://arxiv.org/abs/1509.01626. Source binary MR adaptation keeps six7/7/3/3/3/3 convolutions,three VALID3pools,two dense/ReLU/dropout layers and the69-slot alphabet. Teaching source uses He-uniform conv/Glorot dense initialization,not a claim of exact paper training or benchmark accuracy.
/// @see CharacterEncode(), CharacterFinalLength()
inline CharacterOutput CharacterCnn(nn::ParameterStore &p, const Tensor &ids,
                                    const Tensor &lengths, const CharacterConfig &config = {},
                                    bool training = false, double keep = 1.0,
                                    nn::Generator *gen = nullptr) {
  const int final_length = CharacterFinalLength(config.time);
  if (config.channels < 1 || config.dense < 1 || config.alphabet < 1)
    throw std::invalid_argument("character CNN dimensions");
  if (ids.dim() != 2 || ids.size(0) < 1 || ids.size(1) != config.time ||
      lengths.shape() != nn::Shape{ids.size(0)} ||
      (ids.dtype() != nn::DType::Int32 && ids.dtype() != nn::DType::Int64) ||
      !nn::all(nn::logical_and(nn::greater_equal(ids, -1.),
                               nn::less(ids, double(config.alphabet))))
           .item<bool>() ||
      !std::isfinite(keep) || keep <= 0 || keep > 1 || (training && keep < 1 && !gen))
    throw std::invalid_argument(
        "bounded character IDs,lengths and explicit dropout policy required");
  CharacterOutput out;
  const Tensor onehot = nn::one_hot(ids, config.alphabet, 1.0, 0.0, p.dtype());
  out.quantized =
      nn::mul(onehot, nn::expand_dims(LengthMask(lengths, config.time, p.dtype()), 2));
  Tensor x = out.quantized;
  for (int layer = 0; layer < 6; ++layer) {
    nn::ConvLayerOptions opt;
    opt.padding = nn::Padding::Valid;
    opt.activation = nn::Activation::Relu;
    opt.init = nn::init::he_uniform();
    x = nn::conv1d(p, "conv" + std::to_string(layer + 1), x, config.channels,
                   layer < 2 ? 7 : 3, opt);
    out.stages.push_back(x);
    if (layer == 0 || layer == 1 || layer == 5) {
      x = nn::max_pool1d(x, 3, 3, nn::Padding::Valid);
      out.stages.push_back(x);
    }
  }
  out.flat = nn::reshape(x, {-1, static_cast<std::int64_t>(final_length) * config.channels});
  x = nn::dense(p, "dense1", out.flat, config.dense, nn::Activation::Relu);
  if (training && keep < 1)
    x = nn::dropout(x, 1 - keep, true, *gen);
  x = nn::dense(p, "dense2", x, config.dense, nn::Activation::Relu);
  if (training && keep < 1)
    x = nn::dropout(x, 1 - keep, true, *gen);
  out.logits = nn::dense(p, "head", x, 2);
  return out;
}

} // namespace datamunge::nn::models
