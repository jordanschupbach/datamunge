#pragma once
/// @file video_data.hpp
/// @brief Genuine person-disjoint RGB video ingress and tensor nonparametric reference features.
#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/io.hpp"
#include "datamunge/nn/ops.hpp"
#include "datamunge/nn/training.hpp"
#include <filesystem>
#include <map>
#include <set>
#include <stdexcept>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
using nn::Tensor;
/// @brief Own genuine RGB clips and aligned source-video/person identities.
///
/// Integer metadata retains original annotation identities. Every numeric
/// image validation and feature statistic uses tensors;no auto-download.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto pools = m::LoadVideo("build/cnn-datasets/action_video");
/// const auto fit = pools[0];
/// @endcode
/// @par References
/// - Tran et al.(2015),Learning Spatiotemporal Features with 3D Convolutional Networks, https://arxiv.org/abs/1412.0767. Source teaching C3D uses RGB*2-1,He-uniform convolution,library Glorot dense,ceil pools and a compact32px/base4/FC64/three-action head;not pretrained Sports1M or benchmark training.
/// @see C3d()
struct VideoData {
  Tensor rgb;     ///< Float32[N,16,32,32,3],original sampled RGB in[0,1].
  Tensor labels;  ///< Int32[N],boxing0/handclapping1/handwaving2.
  Tensor person;  ///< Int32[N],original source subject.
  Tensor video;   ///< Int32[N],unique original video0..17.
  Tensor segment; ///< Int32[N],first or second original annotated segment.

  /// @brief Return original clip count.
  /// @return RGB leading-axis length.
  std::int64_t size() const {
    return rgb.size(0);
  }
};
/// @brief Gather aligned original video clips and source identities.
///
/// No fitting or new clip generation.
///
/// @param d Genuine video container.
/// @param rows Bounded original row indices;duplicates allowed.
/// @return Owning gathered tensor handles for all five aligned fields.
/// @throws std::invalid_argument For invalid row indices.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto pools = m::LoadVideo("build/cnn-datasets/action_video");
/// const auto d = m::VideoRows(pools[0], {0, 1});
/// @endcode
/// @par References
/// - Tran et al.(2015),Learning Spatiotemporal Features with 3D Convolutional Networks, https://arxiv.org/abs/1412.0767. Source teaching C3D uses RGB*2-1,He-uniform convolution,library Glorot dense,ceil pools and a compact32px/base4/FC64/three-action head;not pretrained Sports1M or benchmark training.
/// @see C3d()
inline VideoData VideoRows(const VideoData &d, const std::vector<std::int64_t> &rows) {
  return {nn::take_rows(d.rgb, rows), nn::take_rows(d.labels, rows),
          nn::take_rows(d.person, rows), nn::take_rows(d.video, rows),
          nn::take_rows(d.segment, rows)};
}
/// @brief Load genuine annotated KTH clips with person/video/segment separation.
///
/// Custom scenario1/person1..4fit,5validation,6test,first2annotated segments
/// pervideo;16evenly spaced actual frames per segment. Mirrors retain original
/// AVI bytes/annotations. Not original KTH benchmark protocol. No download,
/// synthetic frames or parameter fitting.
///
/// @param root Directory containing action_video.npz and original provenance manifest.
/// @return Fit24/validation6/test6clips,each16RGB32px frames.
/// @throws std::invalid_argument For missing arrays,invalid shape/dtype/range,zero motion or leaked source identities.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto pools = m::LoadVideo("build/cnn-datasets/action_video");
/// @endcode
/// @par References
/// - Tran et al.(2015),Learning Spatiotemporal Features with 3D Convolutional Networks, https://arxiv.org/abs/1412.0767. Source teaching C3D uses RGB*2-1,He-uniform convolution,library Glorot dense,ceil pools and a compact32px/base4/FC64/three-action head;not pretrained Sports1M or benchmark training.
/// @see C3d()
inline std::vector<VideoData> LoadVideo(const std::filesystem::path &root) {
  nn::NoGradGuard guard;
  std::map<std::string, Tensor> a;
  for (const auto &[key, value] : nn::load_npz((root / "action_video.npz").string()))
    if (!a.emplace(key, value).second)
      throw std::invalid_argument("unique video archive keys required");
  std::vector<VideoData> pools;
  std::set<int> prior_people, prior_videos;
  std::set<std::pair<int, int>> clips;
  for (const std::string split : {"fit", "validation", "test"}) {
    VideoData d{a.at(split + "_rgb"), a.at(split + "_label"), a.at(split + "_person"),
                a.at(split + "_video_id"), a.at(split + "_segment")};
    const auto n = d.rgb.size(0);
    if (n != (split == "fit" ? 24 : 6) || d.rgb.shape() != nn::Shape{n, 16, 32, 32, 3} ||
        d.rgb.dtype() != nn::DType::Float32 || !nn::all(nn::is_finite(d.rgb)).item<bool>() ||
        !nn::all(nn::logical_and(nn::greater_equal(d.rgb, 0.), nn::less_equal(d.rgb, 1.)))
             .item<bool>())
      throw std::invalid_argument("bounded finite original16frame RGB clips required");
    for (const auto *t : {&d.labels, &d.person, &d.video, &d.segment})
      if (t->shape() != nn::Shape{n} || t->dtype() != nn::DType::Int32)
        throw std::invalid_argument("aligned original video metadata required");
    const auto labels = d.labels.to_vector<int>(), people = d.person.to_vector<int>(),
               videos = d.video.to_vector<int>(), segments = d.segment.to_vector<int>();
    std::set<int> new_people, new_videos;
    for (int row = 0; row < n; ++row) {
      if (labels[row] < 0 || labels[row] >= 3 || people[row] < 1 || people[row] > 25 ||
          videos[row] < 0 || videos[row] >= 18 || segments[row] < 0 || segments[row] >= 2 ||
          prior_people.count(people[row]) || prior_videos.count(videos[row]) ||
          !clips.insert({videos[row], segments[row]}).second)
        throw std::invalid_argument(
            "bounded disjoint original video/person/segment identities required");
      new_people.insert(people[row]);
      new_videos.insert(videos[row]);
    }
    const auto motion =
        nn::sum(nn::abs(nn::slice_axis(d.rgb, 1, 1, 16) - nn::slice_axis(d.rgb, 1, 0, 15))
                    .to(nn::DType::Float64),
                {1, 2, 3, 4});
    if (!nn::all(nn::greater(motion, 1e-6)).item<bool>())
      throw std::invalid_argument("actual motion required in every source clip");
    const auto counts = nn::sum(nn::one_hot(d.labels, 3), {0});
    if (!nn::all(nn::equal(counts, double(n / 3))).item<bool>())
      throw std::invalid_argument("balanced original source actions required");
    prior_people.insert(new_people.begin(), new_people.end());
    prior_videos.insert(new_videos.begin(), new_videos.end());
    pools.push_back(d);
  }
  return pools;
}
/// @brief Extract complete4x4spatial mean-color or temporal-difference features.
///
/// Double tensor accumulation then source Float32 rounding. No learned
/// parameters or held-out population statistics;all16selected frames contribute.
///
/// @param d Genuine video clips with16frames/32px RGB.
/// @param motion False uses all16mean colors;true15absolute consecutive selected-frame differences.
/// @return Float32[N,48] features in source grid-row/grid-column/channel order.
/// @throws std::invalid_argument For invalid RGB geometry.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto pools = m::LoadVideo("build/cnn-datasets/action_video");
/// const auto features = m::VideoFeatures(pools[0], true);
/// @endcode
/// @par References
/// - Tran et al.(2015),Learning Spatiotemporal Features with 3D Convolutional Networks, https://arxiv.org/abs/1412.0767. Source teaching C3D uses RGB*2-1,He-uniform convolution,library Glorot dense,ceil pools and a compact32px/base4/FC64/three-action head;not pretrained Sports1M or benchmark training.
/// @see C3d()
inline Tensor VideoFeatures(const VideoData &d, bool motion) {
  if (d.rgb.shape() != nn::Shape{d.size(), 16, 32, 32, 3})
    throw std::invalid_argument("16frame32px video required");
  auto x = d.rgb.to(nn::DType::Float64);
  if (motion)
    x = nn::abs(nn::slice_axis(x, 1, 1, 16) - nn::slice_axis(x, 1, 0, 15));
  return nn::mean(x.reshape({d.size(), motion ? 15 : 16, 4, 8, 4, 8, 3}), {1, 3, 5})
      .reshape({d.size(), 48})
      .to(nn::DType::Float32);
}
/// @brief Predict a fitting-population-standardized parameter-free video1NN reference.
///
/// Retain fitting observations as memory. Double tensor population stddev
/// clamped1e-6,standardized squared distances,first argmin. No optimizer
/// or fitted neural leaves;fitting population scale is explicitly documented.
///
/// @param fit Fitting clips and labels only.
/// @param test Query RGB clips;labels/identities ignored for inference.
/// @param motion Select static or temporal-difference48feature view.
/// @return Float32[M,3] onehot reference logits;first fitting row wins exact ties.
/// @throws std::invalid_argument For invalid geometry or empty fitting data.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto pools = m::LoadVideo("build/cnn-datasets/action_video");
/// const auto z = m::VideoNearest(pools[0], pools[2], false);
/// @endcode
/// @par References
/// - Tran et al.(2015),Learning Spatiotemporal Features with 3D Convolutional Networks, https://arxiv.org/abs/1412.0767. Source teaching C3D uses RGB*2-1,He-uniform convolution,library Glorot dense,ceil pools and a compact32px/base4/FC64/three-action head;not pretrained Sports1M or benchmark training.
/// @see C3d()
inline Tensor VideoNearest(const VideoData &fit, const VideoData &test, bool motion) {
  nn::NoGradGuard guard;
  const auto x = VideoFeatures(fit, motion).to(nn::DType::Float64),
             y = VideoFeatures(test, motion).to(nn::DType::Float64);
  const auto deviation =
      nn::maximum(nn::sqrt(nn::variance(x, {0})), nn::full({48}, 1e-6, nn::DType::Float64));
  const auto distances =
      nn::sum(nn::square((nn::expand_dims(y, 1) - nn::expand_dims(x, 0)) / deviation), {2});
  return nn::one_hot(nn::gather(fit.labels, nn::argmin(distances, 1), 0), 3);
}
} // namespace datamunge::nn::models
