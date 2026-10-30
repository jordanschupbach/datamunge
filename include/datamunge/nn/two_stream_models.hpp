#pragma once
/// @file two_stream_models.hpp
/// @brief Complete RGB/flow CNN-M branches,probability fusion and explicit variational-flow objective.
#include "datamunge/nn/io.hpp"
#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/ops.hpp"
#include "datamunge/nn/training.hpp"
#include <cmath>
#include <filesystem>
#include <iostream>
#include <map>
#include <set>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
using nn::Tensor;
/// @brief Configure both complete CNN-M branches and their categorical heads.
///
/// Full224px/width1/FC4096/2048/101classes defaults;compact96/divisor16/64/32/3
/// retains all five convolutions,three pools,two hidden layers and32learned leaves.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// m::TwoStreamConfig c{96, 16, 64, 32, 3};
/// @endcode
/// @par References
/// - Simonyan and Zisserman(2014),Two-Stream Convolutional Networks for Action Recognition in Videos, https://arxiv.org/abs/1406.2199. Source CNN-M adaptation retains five convolutions/two hidden layers per stream and averageSOFTMAX fusion. Compact96px/divisor16/FC64/32/threeaction KTH example uses declared adjacent-frame variational flow,not published pretrained/benchmark flow.
/// @see TwoStream(),TwoStreamFlowStack()
struct TwoStreamConfig {
  int image = 224;   ///< Exact square input side;compact96.
  int divisor = 1;   ///< Positive convolution-width divisor<=96;compact16.
  int fc6 = 4096;    ///< Positive first hidden width;compact64.
  int fc7 = 2048;    ///< Positive second hidden width;compact32.
  int classes = 101; ///< Categorical count>=2;compact3.
};
/// @brief Compute the final pooled spatial side of the source CNN-M geometry.
///
/// Pure geometry metadata,no tensor or parameter allocation.
///
/// @param image Input square side,caller validates a positive resulting side.
/// @return Final integer spatial side after source valid convolutions and explicit pool borders.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const int side = m::TwoStreamFinalSide(96);
/// @endcode
/// @par References
/// - Simonyan and Zisserman(2014),Two-Stream Convolutional Networks for Action Recognition in Videos, https://arxiv.org/abs/1406.2199. Source CNN-M adaptation retains five convolutions/two hidden layers per stream and averageSOFTMAX fusion. Compact96px/divisor16/FC64/32/threeaction KTH example uses declared adjacent-frame variational flow,not published pretrained/benchmark flow.
/// @see TwoStream(),TwoStreamFlowStack()
inline int TwoStreamFinalSide(int image) {
  int side = (image - 7) / 2 + 1;
  side = (side - 3) / 2 + 1;
  side = (side + 2 - 5) / 2 + 1;
  side = (side + 1 - 3) / 2 + 1;
  side = (side - 3) / 2 + 1;
  return side;
}
/// @brief Center ten signed dense fields and interleave their dx/dy channels.
///
/// Center independently over spatial axes2/3,not time or batch. Tensor transpose
/// then reshape orders dx0,dy0,dx1,dy1,...;the input graph remains live.
///
/// @param flow Float32[N,fields,image,image,2] observed displacement.
/// @param image Exact square observed field side.
/// @param fields Positive temporal displacement count,default10.
/// @return Tensor[N,image,image,2*fields],spatially mean-centered per field/component,clipped[-10,10]/10.
/// @throws std::invalid_argument For nonpositive geometry or rank/shape mismatch.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto stacked = m::TwoStreamFlowStack(n::zeros({1, 10, 96, 96, 2}), 96);
/// @endcode
/// @par References
/// - Simonyan and Zisserman(2014),Two-Stream Convolutional Networks for Action Recognition in Videos, https://arxiv.org/abs/1406.2199. Source CNN-M adaptation retains five convolutions/two hidden layers per stream and averageSOFTMAX fusion. Compact96px/divisor16/FC64/32/threeaction KTH example uses declared adjacent-frame variational flow,not published pretrained/benchmark flow.
/// @see TwoStream(),TwoStreamFlowStack()
inline Tensor TwoStreamFlowStack(const Tensor &flow, int image, int fields = 10) {
  if (image < 1 || fields < 1 || flow.dim() != 5 ||
      flow.shape() != nn::Shape{flow.size(0), fields, image, image, 2})
    throw std::invalid_argument("aligned dense flow field geometry required");
  auto centered = flow - nn::mean(flow, {2, 3}, true);
  auto scaled = nn::clip(centered, -10, 10) / 10.;
  return nn::reshape(nn::transpose(scaled, {0, 2, 3, 1, 4}),
                     {flow.size(0), image, image, 2 * fields});
}
/// @brief Average independently normalized branch SOFTMAX probabilities.
///
/// This averages probabilities,not logits. Both branch graphs remain live.
/// Caller computes log-scores separately with its declared floor.
///
/// @param a Raw[N,classes] RGB logits.
/// @param b Matching raw temporal logits.
/// @return Normalized categorical probabilities,each stream weighted.5.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto q = m::TwoStreamFusion(n::zeros({2, 3}), n::ones({2, 3}));
/// @endcode
/// @par References
/// - Simonyan and Zisserman(2014),Two-Stream Convolutional Networks for Action Recognition in Videos, https://arxiv.org/abs/1406.2199. Source CNN-M adaptation retains five convolutions/two hidden layers per stream and averageSOFTMAX fusion. Compact96px/divisor16/FC64/32/threeaction KTH example uses declared adjacent-frame variational flow,not published pretrained/benchmark flow.
/// @see TwoStream(),TwoStreamFlowStack()
inline Tensor TwoStreamFusion(const Tensor &a, const Tensor &b) {
  return (nn::softmax(a, -1) + nn::softmax(b, -1)) * .5;
}
/// @brief Own every convolution,pool,hidden and head endpoint of one CNN-M stream.
///
/// Owning tensor handles retain graph/storage. Conv endpoints are before LRN;
/// RGB has LRN after conv1/2,temporal after conv1only.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p(853);
/// const auto out = m::TwoStream(p, n::zeros({1, 96, 96, 3}), n::zeros({1, 10, 96, 96, 2}),
///                               {96, 16, 64, 32, 3});
/// @endcode
/// @par References
/// - Simonyan and Zisserman(2014),Two-Stream Convolutional Networks for Action Recognition in Videos, https://arxiv.org/abs/1406.2199. Source CNN-M adaptation retains five convolutions/two hidden layers per stream and averageSOFTMAX fusion. Compact96px/divisor16/FC64/32/threeaction KTH example uses declared adjacent-frame variational flow,not published pretrained/benchmark flow.
/// @see TwoStream(),TwoStreamFlowStack()
struct TwoStreamBranch {
  Tensor logits;            ///< Raw categorical branch scores[N,classes].
  Tensor flat;              ///< Last pooled map flattened per row.
  Tensor fc6;               ///< First ReLU hidden activation before dropout.
  Tensor fc7;               ///< Second ReLU hidden activation before dropout.
  std::vector<Tensor> conv; ///< All five pre-LRN convolution ReLU endpoints.
  std::vector<Tensor>
      pool; ///< All three pooled endpoints,including explicit second-pool border.
};
/// @brief Apply the complete five-convolution CNN-M stream with source borders and LRN.
///
/// Convolutions96/256/512/512/512divisor with7/5/3/3/3kernels,strides2/2/1/1/1,
/// explicit one-pixel borders after first;maxpool3stride2 afterconv1/2/5,
/// second pool pads bottom/right negativeinfinity. LRN depth2,bias1,alpha.0001/5,
/// beta.75;HeUniform/ReLU and bias.01. Two HeUniform/ReLU hidden layers then
/// linear Glorot head. No loss,fitting,optimizer or state update hidden.
///
/// @param p Mutable Float32 store;name prefixes all16learned branch leaves.
/// @param name Unique stream prefix,source rgb or flow.
/// @param h Float32[N,image,image,3or20],already scaled RGB or stacked flow.
/// @param cfg Source widths/spatial/hidden/class configuration.
/// @param temporal True expects20flowchannels and omits second LRN;false expects3RGBchannels.
/// @param training Apply both keep.5hidden dropouts iftrue.
/// @param drop6 Explicit mutable first-hidden generator,required in training.
/// @param drop7 Explicit mutable second-hidden generator,required in training.
/// @return Complete branch endpoints and raw categorical logits.
/// @throws std::invalid_argument For invalid config/input dtype/shape or missing training generators.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p;
/// const auto out =
///     m::TwoStreamCnnM(p, "rgb", n::zeros({1, 96, 96, 3}), {96, 16, 64, 32, 3}, false);
/// @endcode
/// @par References
/// - Simonyan and Zisserman(2014),Two-Stream Convolutional Networks for Action Recognition in Videos, https://arxiv.org/abs/1406.2199. Source CNN-M adaptation retains five convolutions/two hidden layers per stream and averageSOFTMAX fusion. Compact96px/divisor16/FC64/32/threeaction KTH example uses declared adjacent-frame variational flow,not published pretrained/benchmark flow.
/// @see TwoStream(),TwoStreamFlowStack()
inline TwoStreamBranch TwoStreamCnnM(nn::ParameterStore &p, const std::string &name, Tensor h,
                                     const TwoStreamConfig &cfg, bool temporal,
                                     bool training = false, nn::Generator *drop6 = nullptr,
                                     nn::Generator *drop7 = nullptr) {
  if (cfg.divisor < 1 || cfg.divisor > 96 || cfg.fc6 < 1 || cfg.fc7 < 1 || cfg.classes < 2 ||
      TwoStreamFinalSide(cfg.image) < 1 || h.dim() != 4 ||
      h.shape() != nn::Shape{h.size(0), cfg.image, cfg.image, temporal ? 20 : 3} ||
      h.dtype() != nn::DType::Float32 || p.dtype() != h.dtype() ||
      (training && (!drop6 || !drop7)))
    throw std::invalid_argument(
        "Float32 CNN-M geometry and explicit dropout streams required");
  TwoStreamBranch out;
  const int widths[] = {96, 256, 512, 512, 512}, kernel[] = {7, 5, 3, 3, 3};
  for (int l = 0; l < 5; ++l) {
    int width = widths[l] / cfg.divisor;

    if (l)
      h = nn::pad(h, {{0, 0}, {1, 1}, {1, 1}, {0, 0}});
    auto prefix = name + "/conv" + std::to_string(l + 1);
    p.get(prefix + "/b", {width}, nn::init::constant(.01));
    nn::ConvLayerOptions opt;
    opt.padding = nn::Padding::Valid;
    opt.stride = l < 2 ? 2 : 1;
    opt.init = nn::init::he_uniform();
    opt.activation = nn::Activation::Relu;
    h = nn::conv2d(p, prefix, h, width, kernel[l], opt);
    out.conv.push_back(h);
    if (l == 0 || (l == 1 && !temporal))
      h = nn::local_response_normalization(h, 2, 1, .0001 / 5, .75);
    if (l == 0 || l == 1 || l == 4) {
      if (l == 1)
        h = nn::pad(h, {{0, 0}, {0, 1}, {0, 1}, {0, 0}}, nn::PadMode::Constant, -INFINITY);
      h = nn::max_pool2d(h, {3, 3}, {2, 2}, nn::Padding::Valid);
      out.pool.push_back(h);
    }
  }
  out.flat =
      nn::reshape(h, {h.size(0), TwoStreamFinalSide(cfg.image) *
                                     TwoStreamFinalSide(cfg.image) * (512 / cfg.divisor)});
  nn::DenseOptions dense;
  dense.init = nn::init::he_uniform();
  dense.activation = nn::Activation::Relu;
  dense.bias_value = .01;
  out.fc6 = nn::dense(p, name + "/fc6", out.flat, cfg.fc6, dense);
  h = out.fc6;
  if (training)
    h = nn::dropout(h, .5, true, *drop6);
  out.fc7 = nn::dense(p, name + "/fc7", h, cfg.fc7, dense);
  h = out.fc7;
  if (training)
    h = nn::dropout(h, .5, true, *drop7);
  out.logits = nn::dense(p, name + "/fc8", h, cfg.classes);
  return out;
}
/// @brief Own both CNN-M branches,preprocessed flow and late-fusion outputs.
///
/// Tensor handles retain both graphs. No learned fusion weight;each softmax
/// probability vector contributes exactly one half.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p(853);
/// const auto out = m::TwoStream(p, n::zeros({1, 96, 96, 3}), n::zeros({1, 10, 96, 96, 2}),
///                               {96, 16, 64, 32, 3});
/// @endcode
/// @par References
/// - Simonyan and Zisserman(2014),Two-Stream Convolutional Networks for Action Recognition in Videos, https://arxiv.org/abs/1406.2199. Source CNN-M adaptation retains five convolutions/two hidden layers per stream and averageSOFTMAX fusion. Compact96px/divisor16/FC64/32/threeaction KTH example uses declared adjacent-frame variational flow,not published pretrained/benchmark flow.
/// @see TwoStream(),TwoStreamFlowStack()
struct TwoStreamOutput {
  TwoStreamBranch rgb;  ///< Complete observed-RGB branch.
  TwoStreamBranch flow; ///< Complete signed ten-field temporal branch.
  Tensor stack;         ///< Per-field centered/clipped/scaled[N,image,image,20]flow channels.
  Tensor
      probabilities; ///< Arithmetic average of independently normalized branch probabilities.
  Tensor log_scores; ///< Log(max(probabilities,1e-8)),for categorical scoring.
};
/// @brief Classify observed RGB and ten genuine adjacent-frame dense displacement fields.
///
/// RGB*2-1;flow center/clip/10 perfield and stackdx/dy. RGB LRN afterconv1/2,
/// temporal afterconv1only. Training objective must be explicit separate CE sum;
/// validation/inference averageSOFTMAX probabilities,not logits. No pretrained
/// loading or fitting hidden. Inference neither consumes dropout nor mutates leaves.
///
/// @param p Mutable Float32 learned store;warm all32leaves before optimizer construction.
/// @param rgb Float32[N,image,image,3],normally[0,1].
/// @param displacement Float32[N,10,image,image,2]signed observed adjacent-frame displacement.
/// @param cfg Full224px defaults;compact96/divisor16/FC64/32/classes3.
/// @param training Apply all four hidden dropouts iftrue,defaultfalse.
/// @param drops Exactly four mutable generators required during training.
/// @return Both complete streams,stacked displacement,fused probabilities and floored log-scores.
/// @throws std::invalid_argument For invalid observed geometry,dtype/config,missing four generators or stored tensors.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p(853);
/// const auto out = m::TwoStream(p, n::zeros({1, 96, 96, 3}), n::zeros({1, 10, 96, 96, 2}),
///                               {96, 16, 64, 32, 3});
/// @endcode
/// @par References
/// - Simonyan and Zisserman(2014),Two-Stream Convolutional Networks for Action Recognition in Videos, https://arxiv.org/abs/1406.2199. Source CNN-M adaptation retains five convolutions/two hidden layers per stream and averageSOFTMAX fusion. Compact96px/divisor16/FC64/32/threeaction KTH example uses declared adjacent-frame variational flow,not published pretrained/benchmark flow.
/// @see TwoStream(),TwoStreamFlowStack()
inline TwoStreamOutput TwoStream(nn::ParameterStore &p, const Tensor &rgb,
                                 const Tensor &displacement, const TwoStreamConfig &cfg = {},
                                 bool training = false,
                                 std::vector<nn::Generator> *drops = nullptr) {
  if (training && (!drops || drops->size() != 4))
    throw std::invalid_argument("four explicit training dropout streams required");
  TwoStreamOutput out;
  out.stack = TwoStreamFlowStack(displacement, cfg.image);
  out.rgb = TwoStreamCnnM(p, "rgb", rgb * 2. - 1., cfg, false, training,
                          drops ? &(*drops)[0] : nullptr, drops ? &(*drops)[1] : nullptr);
  out.flow = TwoStreamCnnM(p, "flow", out.stack, cfg, true, training,
                           drops ? &(*drops)[2] : nullptr, drops ? &(*drops)[3] : nullptr);
  out.probabilities = TwoStreamFusion(out.rgb.logits, out.flow.logits);
  out.log_scores = nn::log(nn::maximum(out.probabilities, nn::scalar(1e-8)));
  return out;
}

/// @brief Compute classical2x2 Horn-Schunck brightness derivatives from adjacent grayscale images.
///
/// Horn and Schunck(1981),Determining Optical Flow,Artificial Intelligence17,185-203,
/// https://people.csail.mit.edu/bkph/papers/Optical_Flow_OPT.pdf.
/// Declared single-scale quadratic-flow preparation,not Farneback or groundtruth.
/// Spatial derivative kernels divide by4and apply to first+second;temporal2x2
/// mean applies to second-first. All values remain tensors;no fitting occurs here.
///
/// @param first Float32[N,H,W,1] first actual grayscale observation.
/// @param second Matching next actual observation.
/// @return Ix,Iy,It tensor triplet with symmetric bottom/right borders.
/// @throws std::invalid_argument For rank/channel/shape mismatch.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto derivatives =
///     m::TwoStreamFlowDerivatives(n::zeros({1, 4, 4, 1}), n::ones({1, 4, 4, 1}));
/// @endcode
/// @par References
/// - Simonyan and Zisserman(2014),Two-Stream Convolutional Networks for Action Recognition in Videos, https://arxiv.org/abs/1406.2199. Source CNN-M adaptation retains five convolutions/two hidden layers per stream and averageSOFTMAX fusion. Compact96px/divisor16/FC64/32/threeaction KTH example uses declared adjacent-frame variational flow,not published pretrained/benchmark flow.
/// @see TwoStream(),TwoStreamFlowStack()
inline std::vector<Tensor> TwoStreamFlowDerivatives(const Tensor &first,
                                                    const Tensor &second) {
  if (first.dim() != 4 || first.size(3) != 1 || first.shape() != second.shape())
    throw std::invalid_argument("matching grayscale NHWC adjacent images required");
  const auto average =
      nn::pad(first + second, {{0, 0}, {0, 1}, {0, 1}, {0, 0}}, nn::PadMode::Symmetric);
  const auto delta =
      nn::pad(second - first, {{0, 0}, {0, 1}, {0, 1}, {0, 0}}, nn::PadMode::Symmetric);
  const auto x = nn::from_vector(std::vector<float>{-1, 1, -1, 1}, {2, 2, 1, 1}) / 4.;
  const auto y = nn::from_vector(std::vector<float>{-1, -1, 1, 1}, {2, 2, 1, 1}) / 4.;
  return {nn::conv2d(average, x, nn::Conv2DOptions{.padding=nn::Padding::Valid}),
          nn::conv2d(average, y, nn::Conv2DOptions{.padding=nn::Padding::Valid}),
          nn::conv2d(delta, nn::ones({2, 2, 1, 1}) / 4.,
                     nn::Conv2DOptions{.padding=nn::Padding::Valid})};
}

/// @brief Evaluate the explicit single-scale variational dense-flow quadratic objective.
///
/// No optimizer hidden. Caller shows explicit TensorSGD steps on u/v.
/// Forward differences include each horizontal/vertical neighbor once,without wrap.
/// This is a linearized single-scale brightness model,not large-motion groundtruth.
///
/// @param u Float32[N,H,W,1] horizontal displacement leaves.
/// @param v Matching vertical displacement leaves.
/// @param derivatives Ix,Iy,It from TwoStreamFlowDerivatives.
/// @param alpha Finite positive smoothness coefficient,default.05.
/// @return Scalar half SUM brightness residual squared plus alpha squared times first-difference smoothness.
/// @throws std::invalid_argument For invalid derivative count,rank/geometry,or nonpositive/nonfinite alpha.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore flow;
/// const auto u = flow.get("u", {1, 4, 4, 1}, n::init::zeros()),
///            v = flow.get("v", {1, 4, 4, 1}, n::init::zeros());
/// const auto d = m::TwoStreamFlowDerivatives(n::zeros({1, 4, 4, 1}), n::ones({1, 4, 4, 1}));
/// const auto loss = m::TwoStreamFlowEnergy(u, v, d);
/// @endcode
/// @par References
/// - Simonyan and Zisserman(2014),Two-Stream Convolutional Networks for Action Recognition in Videos, https://arxiv.org/abs/1406.2199. Source CNN-M adaptation retains five convolutions/two hidden layers per stream and averageSOFTMAX fusion. Compact96px/divisor16/FC64/32/threeaction KTH example uses declared adjacent-frame variational flow,not published pretrained/benchmark flow.
/// @see TwoStream(),TwoStreamFlowStack()
inline Tensor TwoStreamFlowEnergy(const Tensor &u, const Tensor &v,
                                  const std::vector<Tensor> &derivatives, double alpha = .05) {
  if (u.dim() != 4 || u.size(1) < 2 || u.size(2) < 2 || u.shape() != v.shape() ||
      derivatives.size() != 3 || !std::isfinite(alpha) || alpha <= 0)
    throw std::invalid_argument("positive variational flow geometry and smoothness required");
  const auto residual = derivatives[0] * u + derivatives[1] * v + derivatives[2];
  auto energy = nn::sum(nn::square(residual)) * .5;
  for (const auto &field : {u, v})
    for (int axis : {1, 2}) {
      const auto difference = nn::slice_axis(field, axis, 1, field.size(axis)) -
                              nn::slice_axis(field, axis, 0, field.size(axis) - 1);
      energy = energy + nn::sum(nn::square(difference)) * (.5 * alpha * alpha);
    }
  return energy;
}

/// @brief Own named aligned RGB,flow,label,person,video and segment tensor handles.
///
/// String keys are metadata;all observation arrays remain tensors. Query labels
/// and source IDs are never read by the graph or retained-memory predictors.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// m::TwoStreamData d;
/// @endcode
/// @par References
/// - Simonyan and Zisserman(2014),Two-Stream Convolutional Networks for Action Recognition in Videos, https://arxiv.org/abs/1406.2199. Source CNN-M adaptation retains five convolutions/two hidden layers per stream and averageSOFTMAX fusion. Compact96px/divisor16/FC64/32/threeaction KTH example uses declared adjacent-frame variational flow,not published pretrained/benchmark flow.
/// @see TwoStream(),TwoStreamFlowStack()
using TwoStreamData = std::map<std::string, Tensor>;
/// @brief Internal prepared-two-stream validation utilities.
namespace two_stream_detail {
/// @brief Reject an invalid prepared data contract.
/// @param ok Required condition.
/// @param message Failure context.
inline void require(bool ok, const std::string &message) {
  if (!ok)
    throw std::invalid_argument(message);
}
} // namespace two_stream_detail
/// @brief Load aligned prepared RGB and ten-field flow pools with original source provenance.
///
/// RequiresFloat32RGB[N,96,96,3]in[0,1],finite signed flow[N,10,96,96,2],
/// Int32labels/person/video/segment,uniqueclipIDs and person/video-disjoint pools.
/// No download,flow estimation or fitting hidden.
///
/// @param root Local directory containing two_stream.npz.
/// @return 24fitting/6validation/6held-out pairs with disjoint people and source videos.
/// @throws std::invalid_argument For missing files/arrays or invalid observation/metadata contracts.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto pools = m::LoadTwoStream("build/cnn-datasets/two_stream");
/// @endcode
/// @par References
/// - Simonyan and Zisserman(2014),Two-Stream Convolutional Networks for Action Recognition in Videos, https://arxiv.org/abs/1406.2199. Source CNN-M adaptation retains five convolutions/two hidden layers per stream and averageSOFTMAX fusion. Compact96px/divisor16/FC64/32/threeaction KTH example uses declared adjacent-frame variational flow,not published pretrained/benchmark flow.
/// @see TwoStream(),TwoStreamFlowStack()
inline std::vector<TwoStreamData> LoadTwoStream(const std::filesystem::path &root) {
  TwoStreamData a;
  for (const auto &[key, value] : nn::load_npz((root / "two_stream.npz").string()))
    a.emplace(key, value);
  std::vector<TwoStreamData> pools;
  std::set<int> prior_people, prior_videos;
  std::set<std::pair<int, int>> clips;
  for (const std::string split : {"fit", "validation", "test"}) {
    TwoStreamData d;
    for (auto key : {"rgb", "flow", "label", "person", "video_id", "segment"})
      d[key] = a.at(split + "_" + key);
    int n = d.at("rgb").size(0);
    two_stream_detail::require(n == (split == "fit" ? 24 : 6) &&
                                   d.at("rgb").shape() == nn::Shape({n, 96, 96, 3}) &&
                                   d.at("flow").shape() == nn::Shape({n, 10, 96, 96, 2}),
                               "actual aligned RGB/ten displacement fields");
    for (auto key : {"rgb", "flow"}) {
      auto x = d.at(key);
      two_stream_detail::require(x.dtype() == nn::DType::Float32, "float observed modality");
      for (float v : x.to_vector<float>())
        two_stream_detail::require(std::isfinite(v) &&
                                       ((std::string(key) != "rgb") || (v >= 0 && v <= 1)),
                                   "finite modality/RGB[0,1]");
    }
    for (auto key : {"label", "person", "video_id", "segment"})
      two_stream_detail::require(d.at(key).dtype() == nn::DType::Int32 &&
                                     d.at(key).shape() == nn::Shape({n}),
                                 "annotation alignment");
    auto labels = d.at("label").to_vector<int>(), people = d.at("person").to_vector<int>(),
         videos = d.at("video_id").to_vector<int>(),
         segments = d.at("segment").to_vector<int>();
    std::set<int> persons, vids;
    for (int i = 0; i < n; ++i) {
      two_stream_detail::require(
          labels[i] >= 0 && labels[i] < 3 && !prior_people.count(people[i]) &&
              !prior_videos.count(videos[i]) && clips.emplace(videos[i], segments[i]).second,
          "unique clips/person/source-video heldout pools");
      persons.insert(people[i]);
      vids.insert(videos[i]);
    }
    prior_people.insert(persons.begin(), persons.end());
    prior_videos.insert(vids.begin(), vids.end());
    std::cout << "KTH " << split << ":" << n << " RGB+flow pairs," << persons.size()
              << " people," << vids.size() << " source AVIs\n";
    pools.push_back(d);
  }
  return pools;
}

/// @brief Compute4x4cell RGB or absolute ten-field displacement summaries with tensors.
///
/// Float64cell sums preserve source host accumulation scale;outputFloat32.
/// No learned parameters,query labels or IDs used.
///
/// @param d Named aligned observed modality tensors.
/// @param motion False uses RGB;true averages absolute flow over ten fields.
/// @return Float32[N,48]RGB or[N,32]motion features.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// m::TwoStreamData d{{"rgb", n::zeros({1, 96, 96, 3})}};
/// const auto f = m::TwoStreamGridFeatures(d, false);
/// @endcode
/// @par References
/// - Simonyan and Zisserman(2014),Two-Stream Convolutional Networks for Action Recognition in Videos, https://arxiv.org/abs/1406.2199. Source CNN-M adaptation retains five convolutions/two hidden layers per stream and averageSOFTMAX fusion. Compact96px/divisor16/FC64/32/threeaction KTH example uses declared adjacent-frame variational flow,not published pretrained/benchmark flow.
/// @see TwoStream(),TwoStreamFlowStack()
inline Tensor TwoStreamGridFeatures(const TwoStreamData &d, bool motion) {
  const auto x = d.at(motion ? "flow" : "rgb");
  const auto count = x.size(0);
  if (motion)
    return nn::mean(nn::abs(x).to(nn::DType::Float64).reshape({count, 10, 4, 24, 4, 24, 2}),
                    {1, 3, 5})
        .reshape({count, 32})
        .to(nn::DType::Float32);
  return nn::mean(x.to(nn::DType::Float64).reshape({count, 4, 24, 4, 24, 3}), {2, 4})
      .reshape({count, 48})
      .to(nn::DType::Float32);
}

/// @brief Predict categories by fitting-only standardized retained-memory modality1NN.
///
/// No fitted learnable parameters or optimizer. Float64fitpopulation SD clamps
/// at1e-6;Float32subtraction precedes Float64distance as in the original reference.
///
/// @param fit Named fitting modality/label tensors.
/// @param test Named query observations;query labels and IDs unused.
/// @param motion Choose absolute displacement features iftrue,RGB iffalse.
/// @return Float32[Nquery,3]one-hot selected-neighbor scores,first fitting row wins ties.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// m::TwoStreamData d{{"rgb", n::zeros({2, 96, 96, 3})},
///                    {"label", n::zeros({2}, n::DType::Int32)}};
/// const auto z = m::TwoStreamNearest(d, d, false);
/// @endcode
/// @par References
/// - Simonyan and Zisserman(2014),Two-Stream Convolutional Networks for Action Recognition in Videos, https://arxiv.org/abs/1406.2199. Source CNN-M adaptation retains five convolutions/two hidden layers per stream and averageSOFTMAX fusion. Compact96px/divisor16/FC64/32/threeaction KTH example uses declared adjacent-frame variational flow,not published pretrained/benchmark flow.
/// @see TwoStream(),TwoStreamFlowStack()
inline Tensor TwoStreamNearest(const TwoStreamData &fit, const TwoStreamData &test,
                               bool motion) {
  const auto a = TwoStreamGridFeatures(fit, motion), b = TwoStreamGridFeatures(test, motion),
             wide = a.to(nn::DType::Float64);
  const auto scale =
      nn::maximum(nn::sqrt(nn::mean(nn::square(wide - nn::mean(wide, {0})), {0})),
                  nn::full({a.size(1)}, 1e-6, nn::DType::Float64));
  const auto delta =
      (nn::expand_dims(a, 0) - nn::expand_dims(b, 1)).to(nn::DType::Float64) / scale;
  const auto ids = nn::argmin(nn::sum(nn::square(delta), {2}), 1);
  return nn::one_hot(nn::gather(fit.at("label"), ids, 0), 3);
}
} // namespace datamunge::nn::models
