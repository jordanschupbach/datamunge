#pragma once
/// @file vqa_models.hpp
/// @brief Observed-input CNN,masked LSTM and joint/unimodal VQA graphs.
#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/model_plot.hpp"
#include "datamunge/nn/ops.hpp"
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Configure the source CNN,masked LSTM and finite-answer VQA teaching model.
///
/// Architecture metadata only;loader replaces vocabulary/answer/time metadata
/// from fitting-only artifacts and selects compact8/32/32/64 learned widths.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// m::VqaConfig c;
/// c.vocabulary = 8;
/// c.answers = 4;
/// c.time = 3;
/// c.base = 2;
/// c.embedding = 3;
/// c.hidden = 3;
/// c.fusion = 4;
/// n::ParameterStore p(277);
/// @endcode
/// @par References
/// - Antol et al.(2015),VQA:Visual Question Answering, https://arxiv.org/abs/1505.00468;official
/// VQA-v2 assets, https://visualqa.org/download.html. This source teaching variant is a
/// from-scratch six-convolution CNN,one masked LSTM and concatenation/product fusion,not the paper
/// pretrained network or benchmark result.
/// @see VqaBaseline()
struct VqaConfig {
  int vocabulary = 342; ///< Fit-only token classes including PAD0/UNK1.
  int answers    = 64;  ///< Fit-only answer classes including always-incorrect UNK0.
  int time       = 24;  ///< Positive padded question time steps.
  int base       = 64;  ///< Positive initial CNN channels,doubled per block.
  int embedding  = 300; ///< Positive learned token embedding width.
  int hidden     = 256; ///< Positive LSTM hidden/cell width.
  int fusion     = 512; ///< Positive per-modality projection and fusion hidden width.
};
/// @brief Select jointly trained or separately trained unimodal prediction graphs.
///
/// Unimodal variants omit the other encoder and have independent learned stores.
/// They are fitted separately,not evaluated by zeroing a jointly trained input.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// m::VqaConfig c;
/// c.vocabulary = 8;
/// c.answers = 4;
/// c.time = 3;
/// c.base = 2;
/// c.embedding = 3;
/// c.hidden = 3;
/// c.fusion = 4;
/// n::ParameterStore p(277);
/// const auto mode = m::VqaMode::kJoint;
/// @endcode
/// @par References
/// - Antol et al.(2015),VQA:Visual Question Answering, https://arxiv.org/abs/1505.00468;official
/// VQA-v2 assets, https://visualqa.org/download.html. This source teaching variant is a
/// from-scratch six-convolution CNN,one masked LSTM and concatenation/product fusion,not the paper
/// pretrained network or benchmark result.
/// @see VqaBaseline()
enum class VqaMode {
  kJoint,   ///< RGB plus length-masked question encoder and product fusion.
  kImage,   ///< RGB-only encoder;question arguments are ignored.
  kQuestion ///< Question-only encoder;RGB argument is ignored.
};

/// @brief Retain final recurrent memory and each masked hidden activation.
///
/// Tensor handles own graph lifetimes;no padding updates the returned memory.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// m::VqaConfig c;
/// c.vocabulary = 8;
/// c.answers = 4;
/// c.time = 3;
/// c.base = 2;
/// c.embedding = 3;
/// c.hidden = 3;
/// c.fusion = 4;
/// n::ParameterStore p(277);
/// const auto q =
///     m::VqaLstm(p, n::ones({2, 3}, n::DType::Int32), n::full({2}, 2., n::DType::Int32), c);
/// @endcode
/// @par References
/// - Antol et al.(2015),VQA:Visual Question Answering, https://arxiv.org/abs/1505.00468;official
/// VQA-v2 assets, https://visualqa.org/download.html. This source teaching variant is a
/// from-scratch six-convolution CNN,one masked LSTM and concatenation/product fusion,not the paper
/// pretrained network or benchmark result.
/// @see VqaLstm()
struct VqaQuestion {
  nn::Tensor              h;     ///< Final length-masked hidden state[N,hidden].
  nn::Tensor              c;     ///< Final length-masked cell memory[N,hidden].
  std::vector<nn::Tensor> steps; ///< All time hidden states,freezing after each row length.
};

/// @brief Encode padded token IDs with a shared learned embedding and length-masked LSTM.
///
/// Architecture tracing groups the complete masked recurrence at the embedded
/// sequence boundary; fixed integer token selections are outside the gradient trace.
///
/// Gate packing is i,f,g,o with learned Glorot input/recurrent kernels and
/// forget-bias1 at initialization. Each row freezes BOTH h and c after length,
/// so masked embeddings receive zero gradients. Parameter leaves are shared
/// across time. No optimization or RNG beyond initial learned leaf creation.
///
/// @param p Mutable learned store;creates embedding and three recurrent parameter leaves.
/// @param tokens Int32/64[N,time],N>0,bounded0..vocabulary-1;PAD content beyond length is ignored.
/// @param lengths Int32/64[N],0..time inclusive;zero length retains zero hidden/cell.
/// @param config Positive vocabulary/time/embedding/hidden widths.
/// @return Final h/c and time hidden states,all[N,hidden].
/// @throws std::invalid_argument For invalid token/length geometry,IDs,bounds or stored parameter
/// shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// m::VqaConfig c;
/// c.vocabulary = 8;
/// c.answers = 4;
/// c.time = 3;
/// c.base = 2;
/// c.embedding = 3;
/// c.hidden = 3;
/// c.fusion = 4;
/// n::ParameterStore p(277);
/// const auto q =
///     m::VqaLstm(p, n::ones({2, 3}, n::DType::Int32), n::full({2}, 2., n::DType::Int32), c);
/// @endcode
/// @par References
/// - Antol et al.(2015),VQA:Visual Question Answering, https://arxiv.org/abs/1505.00468;official
/// VQA-v2 assets, https://visualqa.org/download.html. This source teaching variant is a
/// from-scratch six-convolution CNN,one masked LSTM and concatenation/product fusion,not the paper
/// pretrained network or benchmark result.
/// @see VqaQuestion, VqaBaseline()
inline VqaQuestion VqaLstm(
    nn::ParameterStore& p,
    const nn::Tensor&   tokens,
    const nn::Tensor&   lengths,
    const VqaConfig&    config) {
  if (config.vocabulary < 2 || config.time < 1 || config.embedding < 1 || config.hidden < 1 ||
      tokens.dim() != 2 || tokens.size(0) < 1 || tokens.size(1) != config.time ||
      lengths.shape() != nn::Shape{tokens.size(0)} ||
      (tokens.dtype() != nn::DType::Int32 && tokens.dtype() != nn::DType::Int64) ||
      (lengths.dtype() != nn::DType::Int32 && lengths.dtype() != nn::DType::Int64) ||
      !nn::all(
           nn::logical_and(
               nn::greater_equal(tokens, 0.), nn::less(tokens, double(config.vocabulary))))
           .item<bool>() ||
      !nn::all(
           nn::logical_and(
               nn::greater_equal(lengths, 0.), nn::less_equal(lengths, double(config.time))))
           .item<bool>())
    throw std::invalid_argument("bounded padded VQA token IDs and lengths required");
  const nn::Tensor embedded =
      nn::embedding(p, "question/embedding", tokens, config.vocabulary, config.embedding);
  const auto    inputs = nn::time_steps(embedded);
  nn::LstmState state  = nn::lstm_zero_state(tokens.size(0), config.hidden, embedded.dtype());
  VqaQuestion   result;
  for (int time = 0; time < config.time; ++time) {
    const auto proposed = nn::lstm_step(p, "question/lstm", inputs[std::size_t(time)], state, 1.0);
    const nn::Tensor active = nn::expand_dims(nn::greater(lengths, double(time)), 1);
    state = {nn::where(active, proposed.h, state.h), nn::where(active, proposed.c, state.c)};
    result.steps.push_back(state.h);
  }
  result.h = state.h;
  if (nn::model_plot_detail::recording())
    nn::model_stage("Length-masked question LSTM", embedded, result.h);
  result.c = state.c;
  return result;
}

/// @brief Retain logits,active modality encodings and all staged VQA activations.
///
/// Undefined tensor fields denote intentionally omitted modality encoders.
/// All active handles retain differentiation graphs;no mutable running state.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// m::VqaConfig c;
/// c.vocabulary = 8;
/// c.answers = 4;
/// c.time = 3;
/// c.base = 2;
/// c.embedding = 3;
/// c.hidden = 3;
/// c.fusion = 4;
/// n::ParameterStore p(277);
/// const auto out = m::VqaBaseline(p, n::zeros({2, 64, 64, 3}), n::ones({2, 3}, n::DType::Int32),
///                                 n::full({2}, 2., n::DType::Int32), c);
/// @endcode
/// @par References
/// - Antol et al.(2015),VQA:Visual Question Answering, https://arxiv.org/abs/1505.00468;official
/// VQA-v2 assets, https://visualqa.org/download.html. This source teaching variant is a
/// from-scratch six-convolution CNN,one masked LSTM and concatenation/product fusion,not the paper
/// pretrained network or benchmark result.
/// @see VqaBaseline()
struct VqaResult {
  nn::Tensor logits;        ///< Unbounded answer logits[N,answers].
  nn::Tensor image_code;    ///< Tanh projected image[N,fusion],undefined in question-only mode.
  nn::Tensor question_code; ///< Tanh projected question[N,fusion],undefined in image-only mode.
  nn::Tensor question_h;    ///< Masked raw LSTM[N,hidden],undefined in image-only mode.
  std::vector<nn::Tensor> stages; ///< Ordered CNN/projection/fusion/head activations,14 for joint.
  std::vector<nn::Tensor>
      question_steps; ///< All time masked LSTM hidden states,empty for image-only.
};

/// @brief Predict finite-vocabulary answers from observed RGB and/or a masked question sequence.
///
/// CNN centers RGB by.5 and applies three blocks of two3x3 SAME He/ReLU
/// convolutions followed by VALID2x2 pooling. Global average features project
/// with tanh. Question embedding/shared LSTM projects with tanh. Joint input
/// is concatenated image,question and elementwise product. A tanh fusion dense
/// layer precedes the linear answer head. Joint has24 learned leaves/14 stages
/// and time shared recurrent steps. No pretrained weights,BN,dropout,target
/// input or optimizer;caller shows soft-answer objective and update explicitly.
///
/// @param p Mutable learned store;warm the complete selected graph before optimizer construction.
/// @param rgb Floating[N,64,64,3],matching p dtype;ignored only in question mode.
/// @param tokens Int32/64[N,time] observed IDs;ignored only in image mode.
/// @param lengths Int32/64[N] observed lengths;ignored only in image mode.
/// @param config Positive dimensions,default full64/300/256/512.
/// @param mode Joint by default;unimodal graphs omit the other encoder.
/// @return Raw logits[N,answers],active modality codes and ordered retained activations.
/// @throws std::invalid_argument For invalid dimensions,mode,active input geometry/dtype or stored
/// leaves.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// m::VqaConfig c;
/// c.vocabulary = 8;
/// c.answers = 4;
/// c.time = 3;
/// c.base = 2;
/// c.embedding = 3;
/// c.hidden = 3;
/// c.fusion = 4;
/// n::ParameterStore p(277);
/// const auto out = m::VqaBaseline(p, n::zeros({2, 64, 64, 3}), n::ones({2, 3}, n::DType::Int32),
///                                 n::full({2}, 2., n::DType::Int32), c);
/// @endcode
/// @par References
/// - Antol et al.(2015),VQA:Visual Question Answering, https://arxiv.org/abs/1505.00468;official
/// VQA-v2 assets, https://visualqa.org/download.html. This source teaching variant is a
/// from-scratch six-convolution CNN,one masked LSTM and concatenation/product fusion,not the paper
/// pretrained network or benchmark result.
/// @see VqaLstm(), VqaSoftCE()
inline VqaResult VqaBaseline(
    nn::ParameterStore& p,
    const nn::Tensor&   rgb,
    const nn::Tensor&   tokens,
    const nn::Tensor&   lengths,
    const VqaConfig&    config = {},
    VqaMode             mode   = VqaMode::kJoint) {
  if (config.vocabulary < 2 || config.answers < 2 || config.time < 1 || config.base < 1 ||
      config.embedding < 1 || config.hidden < 1 || config.fusion < 1)
    throw std::invalid_argument("positive VQA dimensions");
  if (mode != VqaMode::kJoint && mode != VqaMode::kImage && mode != VqaMode::kQuestion)
    throw std::invalid_argument("known VQA mode required");
  if (mode != VqaMode::kQuestion &&
      (rgb.dim() != 4 || rgb.size(0) < 1 || rgb.size(1) != 64 || rgb.size(2) != 64 ||
       rgb.size(3) != 3 || rgb.dtype() != p.dtype()))
    throw std::invalid_argument("64px floating observed RGB required");
  if (mode == VqaMode::kJoint && (tokens.dim() != 2 || tokens.size(0) != rgb.size(0)))
    throw std::invalid_argument("aligned VQA modalities required");
  VqaResult result;
  if (mode != VqaMode::kQuestion) {
    nn::Tensor           image = rgb - .5;
    nn::ConvLayerOptions conv;
    conv.init       = nn::init::he_normal();
    conv.activation = nn::Activation::Relu;
    for (int block = 0; block < 3; ++block) {
      const int width = config.base * (1 << block);
      for (int layer = 0; layer < 2; ++layer) {
        image = nn::conv2d(
            p,
            "image/block" + std::to_string(block) + "/conv" + std::to_string(layer),
            image,
            width,
            3,
            conv);
        result.stages.push_back(image);
      }
      image = nn::max_pool2d(image, 2, 2);
      result.stages.push_back(image);
    }
    result.image_code = nn::dense(
        p, "image/project", nn::global_avg_pool2d(image), config.fusion, nn::Activation::Tanh);
    result.stages.push_back(result.image_code);
  }
  if (mode != VqaMode::kImage) {
    const auto question   = VqaLstm(p, tokens, lengths, config);
    result.question_h     = question.h;
    result.question_steps = question.steps;
    result.question_code =
        nn::dense(p, "question/project", question.h, config.fusion, nn::Activation::Tanh);
    result.stages.push_back(result.question_code);
  }
  nn::Tensor fused;
  if (mode == VqaMode::kJoint)
    fused = nn::concat(
        {result.image_code, result.question_code, result.image_code * result.question_code}, 1);
  else
    fused = mode == VqaMode::kImage ? result.image_code : result.question_code;
  result.stages.push_back(fused);
  const nn::Tensor hidden = nn::dense(p, "fusion", fused, config.fusion, nn::Activation::Tanh);
  result.stages.push_back(hidden);
  result.logits = nn::dense(p, "answer", hidden, config.answers);
  result.stages.push_back(result.logits);
  return result;
}

/// @brief Return the stable source artifact label for a VQA modality.
///
/// Unknown enum values are rejected;no tensor computation or fitting.
///
/// @param mode Joint,image or question graph selector.
/// @return joint,image_only or question_only.
/// @throws std::invalid_argument For unknown mode.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto name = m::VqaModeName(m::VqaMode::kJoint);
/// @endcode
/// @see VqaMode
inline std::string VqaModeName(VqaMode mode) {
  if (mode != VqaMode::kJoint && mode != VqaMode::kImage && mode != VqaMode::kQuestion)
    throw std::invalid_argument("known VQA mode required");
  return mode == VqaMode::kJoint   ? "joint"
         : mode == VqaMode::kImage ? "image_only"
                                   : "question_only";
}

} // namespace datamunge::nn::models
