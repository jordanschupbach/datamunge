#pragma once
/// @file paf_decode.hpp
/// @brief Tensor peak ranking,bilinear limb scoring and grouped metrics with discrete person/link metadata.
#include "datamunge/nn/paf_data.hpp"
#include <set>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Present one tensor-selected local confidence peak as assembly metadata.
///
/// Numeric peak selection/ranking is tensor-based. Host scalar fields serialize
/// selected coordinates/scores for discrete link ownership and user presentation.
///
/// @throws std::invalid_argument For invalid field geometry,nonfinite candidates,indices or decoder configuration.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const m::PafPeak p{1, 2, 3, .9f};
/// @endcode
/// @par References
/// - Cao,Simon,Wei and Sheikh(2017),Realtime Multi-Person2D Pose Estimation using Part Affinity Fields, https://arxiv.org/abs/1611.08050. Source teaching decoder uses local3x3 NMS,ten clamped bilinear samples,distance prior,stable greedy limb matching and conflict-safe person assembly. Custom grouped.2-box diagnostics exclude derived neck and are not COCO AP.
/// @see PafPeaks(), MatchPafPeaks(), AssemblePafPeople(), DecodePafs()
struct PafPeak {
  int joint;   ///< COCO18 joint-type metadata,0..17.
  float x;     ///< Field-grid x coordinate,Float32.
  float y;     ///< Field-grid y coordinate,Float32.
  float score; ///< Raw linear confidence,not a probability.
};
/// @brief Present a directed matched peak pair and its tensor-computed score.
///
/// Indices/topology are metadata;all numeric sample/projection/mean/prior
/// computations occur in tensor operations before this presentation value.
///
/// @throws std::invalid_argument For invalid field geometry,nonfinite candidates,indices or decoder configuration.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const m::PafLink link{0, 1, 0, .8f};
/// @endcode
/// @par References
/// - Cao,Simon,Wei and Sheikh(2017),Realtime Multi-Person2D Pose Estimation using Part Affinity Fields, https://arxiv.org/abs/1611.08050. Source teaching decoder uses local3x3 NMS,ten clamped bilinear samples,distance prior,stable greedy limb matching and conflict-safe person assembly. Custom grouped.2-box diagnostics exclude derived neck and are not COCO AP.
/// @see PafPeaks(), MatchPafPeaks(), AssemblePafPeople(), DecodePafs()
struct PafLink {
  int a;       ///< Start peak index in the decoded peak list.
  int b;       ///< End peak index in the decoded peak list.
  int edge;    ///< Directed COCO18 limb type0..18.
  float score; ///< Tensor line-integral/distance-prior score,Float32 presentation.
};
/// @brief Own conflict-safe person-subset peak/link ownership metadata.
///
/// Joint index arrays reference PafDecoded.peaks;link indices reference its
/// link list. Copying owns metadata without duplicating or changing field values.
///
/// @throws std::invalid_argument For invalid field geometry,nonfinite candidates,indices or decoder configuration.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const m::PafPerson person;
/// @endcode
/// @par References
/// - Cao,Simon,Wei and Sheikh(2017),Realtime Multi-Person2D Pose Estimation using Part Affinity Fields, https://arxiv.org/abs/1611.08050. Source teaching decoder uses local3x3 NMS,ten clamped bilinear samples,distance prior,stable greedy limb matching and conflict-safe person assembly. Custom grouped.2-box diagnostics exclude derived neck and are not COCO AP.
/// @see PafPeaks(), MatchPafPeaks(), AssemblePafPeople(), DecodePafs()
struct PafPerson {
  std::array<int, 18> joints; ///< One peak index per joint type,-1 if absent.
  std::vector<int> links;     ///< Indices into the decoded global limb-link list.

  /// @brief Initialize an empty person subset with every joint absent.
  ///
  /// Pure discrete ownership metadata;no coordinates,score fitting or optimizer.
  ///
  /// @par Example
  /// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const m::PafPerson person;
/// @endcode
  /// @see AssemblePafPeople()
  PafPerson() {
    joints.fill(-1);
  }
};
/// @brief Own one full-image bottom-up pose decoding.
///
/// No GT boxes/person count enter decoding. Thresholds and grouping rules are
/// fixed source teaching policies;empty fields can produce no people.
///
/// @throws std::invalid_argument For invalid field geometry,nonfinite candidates,indices or decoder configuration.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const m::PafDecoded result{};
/// @endcode
/// @par References
/// - Cao,Simon,Wei and Sheikh(2017),Realtime Multi-Person2D Pose Estimation using Part Affinity Fields, https://arxiv.org/abs/1611.08050. Source teaching decoder uses local3x3 NMS,ten clamped bilinear samples,distance prior,stable greedy limb matching and conflict-safe person assembly. Custom grouped.2-box diagnostics exclude derived neck and are not COCO AP.
/// @see PafPeaks(), MatchPafPeaks(), AssemblePafPeople(), DecodePafs()
struct PafDecoded {
  std::vector<PafPeak> peaks;    ///< Tensor-selected local maxima,grouped by joint type.
  std::vector<PafLink> links;    ///< Stable greedily matched directed limbs.
  std::vector<PafPerson> people; ///< Retained conflict-safe person subsets.
};
/// @brief Own detached tensor field values and dimension metadata for bottom-up decoding.
///
/// Canonical numerical helpers consume tensors directly. A separate legacy
/// example adapter preserves old mutable host .v fields for fixture compatibility;
/// this library type never computes with host pixel arrays.
///
/// @throws std::invalid_argument For invalid field geometry,nonfinite candidates,indices or decoder configuration.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const m::PafTensorField field(n::ones({1, 4, 5, 38}));
/// @endcode
/// @par References
/// - Cao,Simon,Wei and Sheikh(2017),Realtime Multi-Person2D Pose Estimation using Part Affinity Fields, https://arxiv.org/abs/1611.08050. Source teaching decoder uses local3x3 NMS,ten clamped bilinear samples,distance prior,stable greedy limb matching and conflict-safe person assembly. Custom grouped.2-box diagnostics exclude derived neck and are not COCO AP.
/// @see PafPeaks(), MatchPafPeaks(), AssemblePafPeople(), DecodePafs()
struct PafTensorField {
  nn::Tensor values;  ///< Detached owning Float32 NHWC snapshot.
  std::int64_t n = 0; ///< Image count.
  std::int64_t h = 0; ///< Field height.
  std::int64_t w = 0; ///< Field width.
  std::int64_t c = 0; ///< Field channels.

  /// @brief Snapshot one positive NHWC numeric field as detached Float32 values.
  ///
  /// Retains an independent cloned snapshot,not a dangling host pointer. Source
  /// Float32 ingress is explicit. Public values may be replaced by caller;
  /// decoder methods validate dimensions/finiteness before numerical use.
  ///
  /// @param input Positive-batch/height/width/channel tensor.
  /// @throws std::invalid_argument For invalid NHWC geometry.
  /// @par Example
  /// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const m::PafTensorField field(n::ones({1, 2, 3, 4}));
/// @endcode
  /// @see PafSamples()
  explicit PafTensorField(const nn::Tensor &input) {
    if (input.dim() != 4 || input.numel() < 1)
      throw std::invalid_argument("positive NHWC field required");
    nn::NoGradGuard guard;
    values = input.detach().to(nn::DType::Float32).contiguous().clone();
    n = input.size(0);
    h = input.size(1);
    w = input.size(2);
    c = input.size(3);
  }
  /// @brief Read one bounded field element for scalar presentation.
  ///
  /// Tensor gather/select performs numeric access;host arithmetic is index
  /// metadata only. This accessor is not used by vectorized line scoring.
  ///
  /// @param batch Bounded image index.
  /// @param row Bounded field row.
  /// @param column Bounded field column.
  /// @param channel Bounded field channel.
  /// @return Float32 selected value.
  /// @throws std::invalid_argument For out-of-range indices.
  /// @par Example
  /// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const m::PafTensorField f(n::ones({1, 2, 3, 4}));
/// const auto v = f(0, 1, 2, 3);
/// @endcode
  /// @see PafSamples()
  float operator()(std::int64_t batch, std::int64_t row, std::int64_t column,
                   std::int64_t channel) const {
    if (batch < 0 || batch >= n || row < 0 || row >= h || column < 0 || column >= w ||
        channel < 0 || channel >= c)
      throw std::invalid_argument("bounded PAF field element required");
    return nn::select(nn::select(nn::select(nn::select(values, 0, batch), 0, row), 0, column),
                      0, channel)
        .item<float>();
  }
};
/// @brief Sample one scalar limb channel at arbitrary tensor positions with clamped bilinear weights.
///
/// Source Float32 clamp/floor/weight/weighted-sum order is preserved. Borders
/// clamp to the nearest field pixel. Tensor indices gather all samples at once;
/// no host pixel arithmetic or gradient-based image fitting.
///
/// @param field Detached Float32 NHWC field snapshot.
/// @param batch Bounded image index.
/// @param x Numeric nonempty sampling x positions.
/// @param y Matching-shape y positions.
/// @param channel Bounded field channel.
/// @return Detached Float32 values with x.shape().
/// @throws std::invalid_argument For invalid field geometry,nonfinite candidates,indices or decoder configuration.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const m::PafTensorField f(n::ones({1, 4, 5, 38}));
/// const auto y = m::PafSamples(f, 0, n::tensor({1.25, -3.}), n::tensor({2.5, 9.}), 0);
/// @endcode
/// @par References
/// - Cao,Simon,Wei and Sheikh(2017),Realtime Multi-Person2D Pose Estimation using Part Affinity Fields, https://arxiv.org/abs/1611.08050. Source teaching decoder uses local3x3 NMS,ten clamped bilinear samples,distance prior,stable greedy limb matching and conflict-safe person assembly. Custom grouped.2-box diagnostics exclude derived neck and are not COCO AP.
/// @see PafPeaks(), MatchPafPeaks(), AssemblePafPeople(), DecodePafs()
inline nn::Tensor PafSamples(const PafTensorField &field, int batch, const nn::Tensor &x,
                             const nn::Tensor &y, int channel) {
  nn::NoGradGuard guard;
  if (batch < 0 || batch >= field.n || channel < 0 || channel >= field.c ||
      x.shape() != y.shape() || x.numel() < 1)
    throw std::invalid_argument("aligned bounded PAF sampling metadata required");
  const auto xx = nn::clip(x.to(nn::DType::Float32), 0., double(field.w - 1)),
             yy = nn::clip(y.to(nn::DType::Float32), 0., double(field.h - 1));
  if (!nn::all(nn::logical_and(nn::is_finite(xx), nn::is_finite(yy))).item<bool>())
    throw std::invalid_argument("finite PAF sampling positions required");
  const auto x0 = nn::floor(xx).to(nn::DType::Int64), y0 = nn::floor(yy).to(nn::DType::Int64),
             x1 = nn::minimum(x0 + 1., nn::full({}, double(field.w - 1), nn::DType::Int64)),
             y1 = nn::minimum(y0 + 1., nn::full({}, double(field.h - 1), nn::DType::Int64)),
             ax = xx - x0.to(nn::DType::Float32), ay = yy - y0.to(nn::DType::Float32),
             image = nn::select(nn::select(field.values, 0, batch), 2, channel)
                         .reshape({field.h * field.w});
  /// @brief Gather scalar channel values for vectorized source pixel indices.
  /// @param row Int64 source rows.
  /// @param column Int64 source columns.
  /// @return Float32 sampled channel values with the same index shape.
  auto at = [&](const nn::Tensor &row, const nn::Tensor &column) {
    return nn::gather(image, row * double(field.w) + column);
  };
  return (1. - ay) * ((1. - ax) * at(y0, x0) + ax * at(y0, x1)) +
         ay * ((1. - ax) * at(y1, x0) + ax * at(y1, x1));
}
/// @brief Present one tensor bilinear field sample as a Float32 scalar.
///
/// Legacy scalar presentation delegates all numeric work to PafSamples.
///
/// @param field Detached field snapshot.
/// @param batch Bounded image index.
/// @param x Finite field x position,clamped at borders.
/// @param y Finite field y position,clamped at borders.
/// @param channel Bounded field channel.
/// @return Source-ordered Float32 bilinear value.
/// @throws std::invalid_argument For invalid field geometry,nonfinite candidates,indices or decoder configuration.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const m::PafTensorField f(n::ones({1, 4, 5, 38}));
/// const auto v = m::PafSample(f, 0, 1.25f, 2.5f, 0);
/// @endcode
/// @par References
/// - Cao,Simon,Wei and Sheikh(2017),Realtime Multi-Person2D Pose Estimation using Part Affinity Fields, https://arxiv.org/abs/1611.08050. Source teaching decoder uses local3x3 NMS,ten clamped bilinear samples,distance prior,stable greedy limb matching and conflict-safe person assembly. Custom grouped.2-box diagnostics exclude derived neck and are not COCO AP.
/// @see PafPeaks(), MatchPafPeaks(), AssemblePafPeople(), DecodePafs()
inline float PafSample(const PafTensorField &field, int batch, float x, float y, int channel) {
  return PafSamples(field, batch, nn::full({}, x), nn::full({}, y), channel).item<float>();
}
/// @brief Select stable3x3 local confidence maxima with tensor comparisons and ranking.
///
/// Neighbors outside the image do not suppress peaks. Within a plateau,an
/// equal earlier raster neighbor suppresses the later pixel;stable argsort
/// retains source ties. Only18 joint channels participate,not background.
/// Host loops serialize joint/selected-peak metadata only;all pixel comparisons,
/// coordinates and score ranking are tensor computations. No GT input.
///
/// @param heat Float32 confidence field with19 channels.
/// @param batch Bounded image index.
/// @param threshold Finite strict peak cutoff,default.05f.
/// @param cap Positive maximum retained peaks per joint,default20.
/// @return Joint-major peaks,descending score with row-major ties.
/// @throws std::invalid_argument For invalid field geometry,nonfinite candidates,indices or decoder configuration.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const m::PafTensorField h(n::zeros({1, 4, 5, 19}));
/// const auto peaks = m::PafPeaks(h, 0);
/// @endcode
/// @par References
/// - Cao,Simon,Wei and Sheikh(2017),Realtime Multi-Person2D Pose Estimation using Part Affinity Fields, https://arxiv.org/abs/1611.08050. Source teaching decoder uses local3x3 NMS,ten clamped bilinear samples,distance prior,stable greedy limb matching and conflict-safe person assembly. Custom grouped.2-box diagnostics exclude derived neck and are not COCO AP.
/// @see PafPeaks(), MatchPafPeaks(), AssemblePafPeople(), DecodePafs()
inline std::vector<PafPeak> PafPeaks(const PafTensorField &heat, int batch,
                                     float threshold = .05f, int cap = 20) {
  nn::NoGradGuard guard;
  if (heat.c != 19 || batch < 0 || batch >= heat.n || cap < 1 || !std::isfinite(threshold))
    throw std::invalid_argument("COCO18 confidence field/peak policy invalid");
  const auto values = nn::slice_axis(nn::select(heat.values, 0, batch), 2, 0, 18);
  if (!nn::all(nn::is_finite(values)).item<bool>())
    throw std::invalid_argument("finite joint confidence values required");
  auto keep = nn::greater(values, double(threshold));
  const auto padded =
      nn::pad(values, {{1, 1}, {1, 1}, {0, 0}}, nn::PadMode::Constant, -INFINITY);
  for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx) {
      if (dy == 0 && dx == 0)
        continue;
      const auto neighbor = nn::slice(padded, {dy + 1, dx + 1, 0}, {heat.h, heat.w, 18});
      const bool earlier = dy < 0 || (dy == 0 && dx < 0);
      keep = nn::logical_and(keep, earlier ? nn::greater(values, neighbor)
                                           : nn::greater_equal(values, neighbor));
    }
  std::vector<PafPeak> result;
  for (int joint = 0; joint < 18; ++joint) {
    const auto candidate = nn::select(values, 2, joint).reshape({heat.h * heat.w}),
               valid = nn::select(keep, 2, joint).reshape({heat.h * heat.w}),
               scores = nn::where(valid, candidate, -INFINITY),
               order = nn::slice_axis(nn::argsort(scores, true), 0, 0,
                                      std::min<std::int64_t>(cap, scores.numel())),
               picked = nn::gather(scores, order), position = order.to(nn::DType::Float32),
               y = nn::floor(position / double(heat.w)), x = position - y * double(heat.w);
    const auto table = nn::stack({x, y, picked}, 1).to_vector<float>();
    for (std::size_t i = 0; i < table.size() / 3; ++i)
      if (std::isfinite(table[i * 3 + 2]))
        result.push_back({joint, table[i * 3], table[i * 3 + 1], table[i * 3 + 2]});
  }
  return result;
}
/// @brief Score directed candidate limbs with tensor line integrals and greedily assign endpoints.
///
/// Source Float32 endpoint differences and sampling fractions precede Double
/// direction/projection/mean/distance-prior calculations. Prior is
/// min(0,.5*field_height/length-1). Require enough strict projections and
/// positive combined score;zero-length limbs reject. Stable Float32 score
/// ranking preserves source pair-enumeration ties. Host loops manage only
/// class/index candidates and one-to-one ownership,not numeric field math.
///
/// @param peaks Selected finite joint-type/coordinate/confidence metadata.
/// @param fields Detached38-channel directed limb field.
/// @param batch Bounded image index.
/// @param minimum_projection Strict positive projection cutoff,default.05f.
/// @param minimum_fraction Fraction of samples exceeding cutoff,default.8f,in[0,1].
/// @param samples At least two bilinear samples,default10.
/// @return Edge-major stable descending-score one-to-one directed links.
/// @throws std::invalid_argument For invalid field geometry,nonfinite candidates,indices or decoder configuration.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const m::PafTensorField f(n::zeros({1, 16, 16, 38}));
/// const auto links = m::MatchPafPeaks({{1, 2, 3, 1}, {2, 5, 3, 1}}, f, 0);
/// @endcode
/// @par References
/// - Cao,Simon,Wei and Sheikh(2017),Realtime Multi-Person2D Pose Estimation using Part Affinity Fields, https://arxiv.org/abs/1611.08050. Source teaching decoder uses local3x3 NMS,ten clamped bilinear samples,distance prior,stable greedy limb matching and conflict-safe person assembly. Custom grouped.2-box diagnostics exclude derived neck and are not COCO AP.
/// @see PafPeaks(), MatchPafPeaks(), AssemblePafPeople(), DecodePafs()
inline std::vector<PafLink> MatchPafPeaks(const std::vector<PafPeak> &peaks,
                                          const PafTensorField &fields, int batch,
                                          float minimum_projection = .05f,
                                          float minimum_fraction = .8f, int samples = 10) {
  nn::NoGradGuard guard;
  if (fields.c != 38 || batch < 0 || batch >= fields.n || samples < 2 ||
      !std::isfinite(minimum_projection) || !std::isfinite(minimum_fraction) ||
      minimum_fraction < 0 || minimum_fraction > 1 ||
      !nn::all(nn::is_finite(fields.values)).item<bool>())
    throw std::invalid_argument("finite directed PAF field and line policy required");
  std::vector<float> coordinates;
  for (const auto &p : peaks) {
    if (p.joint < 0 || p.joint >= 18 || !std::isfinite(p.x) || !std::isfinite(p.y) ||
        !std::isfinite(p.score))
      throw std::invalid_argument("finite bounded-joint peak metadata required");
    coordinates.insert(coordinates.end(), {p.x, p.y});
  }
  if (peaks.empty())
    return {};
  const auto xy = nn::from_vector(coordinates, {std::int64_t(peaks.size()), 2}),
             steps = nn::arange(0, samples).reshape({1, samples}) / double(samples - 1);
  std::vector<PafLink> links;
  for (int edge = 0; edge < 19; ++edge) {
    std::vector<std::int64_t> starts, ends;
    for (std::size_t a = 0; a < peaks.size(); ++a)
      if (peaks[a].joint == kPoseEdges[edge][0])
        for (std::size_t b = 0; b < peaks.size(); ++b)
          if (peaks[b].joint == kPoseEdges[edge][1]) {
            starts.push_back(a);
            ends.push_back(b);
          }
    if (starts.empty())
      continue;
    const auto ai = nn::from_vector(starts, {std::int64_t(starts.size())}),
               bi = nn::from_vector(ends, {std::int64_t(ends.size())}),
               pa = nn::gather(xy, ai), pb = nn::gather(xy, bi),
               dx = (nn::select(pb, 1, 0) - nn::select(pa, 1, 0)).to(nn::DType::Float64),
               dy = (nn::select(pb, 1, 1) - nn::select(pa, 1, 1)).to(nn::DType::Float64),
               length = nn::sqrt(dx * dx + dy * dy), present = nn::greater_equal(length, 1e-9),
               safe_length = nn::where(present, length, 1.);
    const auto sx = (nn::select(pa, 1, 0).to(nn::DType::Float64).reshape({-1, 1}) +
                     steps.to(nn::DType::Float64) * dx.reshape({-1, 1}))
                        .to(nn::DType::Float32),
               sy = (nn::select(pa, 1, 1).to(nn::DType::Float64).reshape({-1, 1}) +
                     steps.to(nn::DType::Float64) * dy.reshape({-1, 1}))
                        .to(nn::DType::Float32);
    const auto px = PafSamples(fields, batch, sx, sy, 2 * edge).to(nn::DType::Float64),
               py = PafSamples(fields, batch, sx, sy, 2 * edge + 1).to(nn::DType::Float64),
               projection = px * dx.reshape({-1, 1}) / safe_length.reshape({-1, 1}) +
                            py * dy.reshape({-1, 1}) / safe_length.reshape({-1, 1}),
               good = nn::sum(
                   nn::greater(projection, double(minimum_projection)).to(nn::DType::Float64),
                   {1}),
               prior = nn::minimum(nn::zeros_like(length),
                                   .5 * double(fields.h) / safe_length - 1.),
               score = nn::sum(projection, {1}) / double(samples) + prior,
               required =
                   (nn::full({}, minimum_fraction) * double(samples)).to(nn::DType::Float64),
               accepted =
                   nn::logical_and(present, nn::logical_and(nn::greater_equal(good, required),
                                                            nn::greater(score, 0.)));
    const auto ranked = nn::where(accepted, score.to(nn::DType::Float32), -INFINITY);
    const auto order = nn::argsort(ranked, true).to_vector<std::int64_t>();
    const auto sorted_scores =
                   nn::gather(ranked, nn::from_vector(order, {std::int64_t(order.size())}))
                       .to_vector<float>();
    std::set<int> used_a, used_b;
    for (std::size_t i = 0; i < order.size(); ++i) {
      if (!std::isfinite(sorted_scores[i]))
        break;
      const int a = int(starts[order[i]]), b = int(ends[order[i]]);
      if (used_a.contains(a) || used_b.contains(b))
        continue;
      links.push_back({a, b, edge, sorted_scores[i]});
      used_a.insert(a);
      used_b.insert(b);
    }
  }
  return links;
}
/// @brief Assemble conflict-safe person subsets using discrete peak/link ownership.
///
/// Grow/merge link by link;never merge two different peaks of the same
/// joint type. Host vectors/sets manage topology/ownership metadata only.
/// Retention score is a Float64 tensor mean of source Float32 peak/link
/// scores. No GT person count,annotations,box or fitting informs grouping.
///
/// @param peaks Selected peak metadata with finite confidence values.
/// @param links Matched directed link metadata.
/// @param minimum_joints Required retained joint types,default3,in1..18.
/// @param minimum_average Minimum tensor mean of retained peak/link scores,default.05f.
/// @return Retained person subsets with original global peak/link indices.
/// @throws std::invalid_argument For invalid field geometry,nonfinite candidates,indices or decoder configuration.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto people = m::AssemblePafPeople({{1, 0, 0, 1}, {2, 1, 0, 1}, {3, 2, 0, 1}},
///                                          {{0, 1, 0, 1}, {1, 2, 2, 1}});
/// @endcode
/// @par References
/// - Cao,Simon,Wei and Sheikh(2017),Realtime Multi-Person2D Pose Estimation using Part Affinity Fields, https://arxiv.org/abs/1611.08050. Source teaching decoder uses local3x3 NMS,ten clamped bilinear samples,distance prior,stable greedy limb matching and conflict-safe person assembly. Custom grouped.2-box diagnostics exclude derived neck and are not COCO AP.
/// @see PafPeaks(), MatchPafPeaks(), AssemblePafPeople(), DecodePafs()
inline std::vector<PafPerson> AssemblePafPeople(const std::vector<PafPeak> &peaks,
                                                const std::vector<PafLink> &links,
                                                int minimum_joints = 3,
                                                float minimum_average = .05f) {
  if (minimum_joints < 1 || minimum_joints > 18 || !std::isfinite(minimum_average))
    throw std::invalid_argument("valid person retention policy required");
  for (const auto &p : peaks)
    if (p.joint < 0 || p.joint >= 18 || !std::isfinite(p.score))
      throw std::invalid_argument("valid peak ownership metadata required");
  for (const auto &l : links)
    if (l.a < 0 || l.b < 0 || l.a >= int(peaks.size()) || l.b >= int(peaks.size()) ||
        l.edge < 0 || l.edge >= 19 || !std::isfinite(l.score))
      throw std::invalid_argument("valid link ownership metadata required");
  std::vector<PafPerson> people;
  for (int index = 0; index < int(links.size()); ++index) {
    const auto &link = links[std::size_t(index)];
    const int ja = peaks.at(std::size_t(link.a)).joint,
              jb = peaks.at(std::size_t(link.b)).joint;
    int owner_a = -1, owner_b = -1;
    for (int i = 0; i < int(people.size()); ++i) {
      if (people[std::size_t(i)].joints[std::size_t(ja)] == link.a)
        owner_a = i;
      if (people[std::size_t(i)].joints[std::size_t(jb)] == link.b)
        owner_b = i;
    }
    if (owner_a < 0 && owner_b < 0) {
      PafPerson person;
      person.joints[std::size_t(ja)] = link.a;
      person.joints[std::size_t(jb)] = link.b;
      person.links.push_back(index);
      people.push_back(person);
    } else if (owner_a == owner_b) {
      people[std::size_t(owner_a)].links.push_back(index);
    } else if (owner_a < 0 || owner_b < 0) {
      const int owner = owner_a >= 0 ? owner_a : owner_b, joint = owner_a >= 0 ? jb : ja,
                peak = owner_a >= 0 ? link.b : link.a;
      auto &person = people[std::size_t(owner)];
      if (person.joints[std::size_t(joint)] < 0 || person.joints[std::size_t(joint)] == peak) {
        person.joints[std::size_t(joint)] = peak;
        person.links.push_back(index);
      }
    } else {
      auto &pa = people[std::size_t(owner_a)];
      auto &pb = people[std::size_t(owner_b)];
      bool conflict = false;
      for (std::size_t j = 0; j < 18; ++j)
        if (pa.joints[j] >= 0 && pb.joints[j] >= 0 && pa.joints[j] != pb.joints[j])
          conflict = true;
      if (conflict)
        continue;
      for (std::size_t j = 0; j < 18; ++j)
        if (pa.joints[j] < 0)
          pa.joints[j] = pb.joints[j];
      pa.links.insert(pa.links.end(), pb.links.begin(), pb.links.end());
      pa.links.push_back(index);
      people.erase(people.begin() + owner_b);
    }
  }
  std::vector<PafPerson> kept;
  for (const auto &person : people) {
    int count = 0;
    std::vector<double> scores;
    for (int peak : person.joints)
      if (peak >= 0) {
        ++count;
        scores.push_back(peaks[std::size_t(peak)].score);
      }
    for (int link : person.links)
      scores.push_back(links[std::size_t(link)].score);
    if (count >= minimum_joints &&
        nn::mean(nn::from_vector(scores, {std::int64_t(scores.size())})).item() >=
            minimum_average)
      kept.push_back(person);
  }
  return kept;
}

/// @brief Decode one observed-image confidence/PAF pair without GT inputs.
///
/// Uses fixed default local-NMS,line-integral,matching and retention policies.
/// No optimizer,person boxes or annotations enter decoding.
///
/// @param heat Detached19-channel confidence snapshot.
/// @param fields Aligned38-channel directed limb snapshot.
/// @param batch Bounded image index.
/// @return Selected peaks,matched limbs and grouped people.
/// @throws std::invalid_argument For invalid field geometry,nonfinite candidates,indices or decoder configuration.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const m::PafTensorField h(n::zeros({1, 4, 5, 19})), f(n::zeros({1, 4, 5, 38}));
/// const auto result = m::DecodePafs(h, f, 0);
/// @endcode
/// @par References
/// - Cao,Simon,Wei and Sheikh(2017),Realtime Multi-Person2D Pose Estimation using Part Affinity Fields, https://arxiv.org/abs/1611.08050. Source teaching decoder uses local3x3 NMS,ten clamped bilinear samples,distance prior,stable greedy limb matching and conflict-safe person assembly. Custom grouped.2-box diagnostics exclude derived neck and are not COCO AP.
/// @see PafPeaks(), MatchPafPeaks(), AssemblePafPeople(), DecodePafs()
inline PafDecoded DecodePafs(const PafTensorField &heat, const PafTensorField &fields,
                             int batch) {
  auto peaks = PafPeaks(heat, batch);
  auto links = MatchPafPeaks(peaks, fields, batch);
  auto people = AssemblePafPeople(peaks, links);
  return {peaks, links, people};
}
/// @brief Snapshot tensor fields and decode one observed image.
///
/// Convenience wrapper;all numerical work delegates to tensor field helpers.
///
/// @param heat Numeric NHWC19 confidence fields.
/// @param fields Aligned numeric NHWC38 directed limb fields.
/// @param batch Bounded image index.
/// @return Complete bottom-up decoding metadata.
/// @throws std::invalid_argument For invalid field geometry,nonfinite candidates,indices or decoder configuration.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto result = m::DecodePafs(n::zeros({1, 4, 5, 19}), n::zeros({1, 4, 5, 38}), 0);
/// @endcode
/// @par References
/// - Cao,Simon,Wei and Sheikh(2017),Realtime Multi-Person2D Pose Estimation using Part Affinity Fields, https://arxiv.org/abs/1611.08050. Source teaching decoder uses local3x3 NMS,ten clamped bilinear samples,distance prior,stable greedy limb matching and conflict-safe person assembly. Custom grouped.2-box diagnostics exclude derived neck and are not COCO AP.
/// @see PafPeaks(), MatchPafPeaks(), AssemblePafPeople(), DecodePafs()
inline PafDecoded DecodePafs(const nn::Tensor &heat, const nn::Tensor &fields, int batch) {
  return DecodePafs(PafTensorField(heat), PafTensorField(fields), batch);
}
/// @brief Own custom grouped annotated-joint counts and tensor-derived precision/recall.
///
/// Counts are discrete matching metadata. Derived neck1 is excluded.
/// These diagnostics are custom greedy.2-box matching,not COCO OKS/AP.
///
/// @throws std::invalid_argument For invalid field geometry,nonfinite candidates,indices or decoder configuration.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const m::PafMetrics score{};
/// @endcode
/// @par References
/// - Cao,Simon,Wei and Sheikh(2017),Realtime Multi-Person2D Pose Estimation using Part Affinity Fields, https://arxiv.org/abs/1611.08050. Source teaching decoder uses local3x3 NMS,ten clamped bilinear samples,distance prior,stable greedy limb matching and conflict-safe person assembly. Custom grouped.2-box diagnostics exclude derived neck and are not COCO AP.
/// @see PafPeaks(), MatchPafPeaks(), AssemblePafPeople(), DecodePafs()
struct PafMetrics {
  int correct = 0;   ///< Correct annotated joints after greedy person matching,excluding neck.
  int predicted = 0; ///< Predicted grouped non-neck joint count.
  int annotated = 0; ///< Original labeled non-neck joint count.
  int people = 0;    ///< Retained predicted person count.
  int true_people = 0; ///< Original annotated person count.

  /// @brief Return grouped-joint precision,zero without predictions.
  ///
  /// Tensor Float64 division of discrete matching counts;not COCO AP.
  ///
  /// @return Double correct/predicted,or0 for zero denominator.
  /// @par Example
  /// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// m::PafMetrics s;
/// s.correct = 2;
/// s.predicted = 4;
/// s.annotated = 5;
/// const auto v = s.precision();
/// @endcode
  /// @see ScorePafs()
  double precision() const {
    return predicted ? (nn::full({}, double(correct), nn::DType::Float64) /
                        nn::full({}, double(predicted), nn::DType::Float64))
                           .item()
                     : 0.;
  }
  /// @brief Return grouped-joint recall,zero without annotations.
  ///
  /// Tensor Float64 division of discrete matching counts;not COCO AP.
  ///
  /// @return Double correct/annotated,or0 for zero denominator.
  /// @par Example
  /// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// m::PafMetrics s;
/// s.correct = 2;
/// s.predicted = 4;
/// s.annotated = 5;
/// const auto v = s.recall();
/// @endcode
  /// @see ScorePafs()
  double recall() const {
    return annotated ? (nn::full({}, double(correct), nn::DType::Float64) /
                        nn::full({}, double(annotated), nn::DType::Float64))
                           .item()
                     : 0.;
  }
};
/// @brief Score grouped annotated joints with tensor distances and stable greedy person matching.
///
/// Exclude derived neck1 from all joint counts. Predicted field coordinates
/// map to128px with source Float32 half-pixel arithmetic. Distances follow
/// Float32 subtraction then Double hypot-like norm cast once to Float32.
/// Correct means distance<=.2*max(original box width,height). Candidate
/// correct counts and stable descending matching order use tensors;host sets
/// manage only one-to-one person ownership. Missing joint NaNs select safe
/// coordinates before geometry. This is a custom diagnostic,not COCO AP.
///
/// @param decoded One complete bottom-up decoding per image.
/// @param data Original COCO18 joints,boxes and person counts.
/// @param side Positive predicted field side,default16.
/// @return Discrete correct/predicted/annotated/grouped-person counts.
/// @throws std::invalid_argument For invalid field geometry,nonfinite candidates,indices or decoder configuration.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// m::PafData d{n::zeros({1, 128, 128, 3}), n::zeros({1, 1, 18, 3}),
///              n::tensor({64., 64., 80., 80.}, {1, 1, 4}), n::ones({1}, n::DType::Int32),
///              n::zeros({1}, n::DType::Int32)};
/// const auto s = m::ScorePafs({m::PafDecoded{}}, d);
/// @endcode
/// @par References
/// - Cao,Simon,Wei and Sheikh(2017),Realtime Multi-Person2D Pose Estimation using Part Affinity Fields, https://arxiv.org/abs/1611.08050. Source teaching decoder uses local3x3 NMS,ten clamped bilinear samples,distance prior,stable greedy limb matching and conflict-safe person assembly. Custom grouped.2-box diagnostics exclude derived neck and are not COCO AP.
/// @see PafPeaks(), MatchPafPeaks(), AssemblePafPeople(), DecodePafs()
inline PafMetrics ScorePafs(const std::vector<PafDecoded> &decoded, const PafData &data,
                            int side = 16) {
  nn::NoGradGuard guard;
  if (data.joints.dim() != 4 || data.joints.size(2) != 18 || data.joints.size(3) != 3 ||
      decoded.size() != std::size_t(data.joints.size(0)) ||
      data.boxes.shape() != nn::Shape{data.joints.size(0), data.joints.size(1), 4} ||
      data.counts.shape() != nn::Shape{data.joints.size(0)} || side < 1)
    throw std::invalid_argument("one grouped decode per aligned COCO18 image required");
  PafMetrics result;
  const auto counts = data.counts.to_vector<std::int32_t>();
  const auto nonneck = nn::not_equal(nn::arange(0, 18), 1.);
  for (std::size_t batch = 0; batch < decoded.size(); ++batch) {
    const auto &output = decoded[batch];
    const int people = counts[batch];
    if (people < 1 || people > data.joints.size(1))
      throw std::invalid_argument("bounded original people required");
    result.people += int(output.people.size());
    result.true_people += people;
    const auto truth = nn::slice_axis(nn::select(data.joints, 0, batch), 0, 0, people)
                           .to(nn::DType::Float32),
               boxes = nn::slice_axis(nn::select(data.boxes, 0, batch), 0, 0, people)
                           .to(nn::DType::Float32),
               known = nn::logical_and(nn::greater(nn::select(truth, 2, 2), 0.), nonneck);
    result.annotated += nn::count_nonzero(known).item<int>();
    std::vector<std::int64_t> prediction_ids, truth_ids, correct;
    for (std::size_t index = 0; index < output.people.size(); ++index) {
      const auto &person = output.people[index];
      std::vector<float> xy;
      std::vector<std::int32_t> present;
      for (int j = 0; j < 18; ++j) {
        const int peak = person.joints[j];
        if (peak >= 0) {
          if (peak >= int(output.peaks.size()))
            throw std::invalid_argument("grouped peak index out of bounds");
          xy.insert(xy.end(), {output.peaks[peak].x, output.peaks[peak].y});
          present.push_back(j != 1);
        } else {
          xy.insert(xy.end(), {0.f, 0.f});
          present.push_back(0);
        }
      }
      const auto active = nn::from_vector(present, {18}).to(nn::DType::Bool),
                 coordinates =
                     (nn::from_vector(xy, {1, 18, 2}) + .5) * 128. / double(side) - .5;
      result.predicted += nn::count_nonzero(active).item<int>();
      const auto support = nn::logical_and(known, active),
                 p = nn::where(nn::expand_dims(support, 2), coordinates, 0.),
                 t = nn::where(nn::expand_dims(support, 2), nn::slice_axis(truth, 2, 0, 2),
                               0.);
      const auto difference = (p - t).to(nn::DType::Float64),
                 distance =
                     nn::sqrt(nn::sum(nn::square(difference), {2})).to(nn::DType::Float32),
                 scale = nn::max(nn::slice_axis(boxes, 1, 2, 4), {1})
                             .to(nn::DType::Float64)
                             .reshape({people, 1}),
                 matches = nn::logical_and(
                     support, nn::less_equal(distance.to(nn::DType::Float64), .2 * scale)),
                 per = nn::sum(matches.to(nn::DType::Int64), {1});
      const auto values = per.to_vector<std::int64_t>();
      for (int person_index = 0; person_index < people; ++person_index)
        if (values[person_index] > 0) {
          prediction_ids.push_back(index);
          truth_ids.push_back(person_index);
          correct.push_back(values[person_index]);
        }
    }
    if (!correct.empty()) {
      const auto order =
          nn::argsort(nn::from_vector(correct, {std::int64_t(correct.size())}), true)
              .to_vector<std::int64_t>();
      std::set<std::int64_t> used_predictions, used_truth;
      for (auto i : order)
        if (!used_predictions.contains(prediction_ids[i]) &&
            !used_truth.contains(truth_ids[i])) {
          result.correct += int(correct[i]);
          used_predictions.insert(prediction_ids[i]);
          used_truth.insert(truth_ids[i]);
        }
    }
  }
  return result;
}
} // namespace datamunge::nn::models
