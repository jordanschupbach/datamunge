// Multi-person COCO18 heatmaps, averaged directed limb fields and unknown
// masks: a port of tflow's common/ml/paf_data.h (host-side target
// construction on std::vector, nn tensors in and out).
#pragma once

#include <array>
#include <cmath>
#include <filesystem>
#include <set>
#include <string>
#include <vector>

#include "common/pose_data.hpp"

namespace paf_reference {
namespace nn=datamunge::nn;
using pose::PoseBatch;using pose::PoseArrays;

inline constexpr int kPoseEdges[19][2] = {{1, 2},  {1, 5},   {2, 3},   {3, 4},  {5, 6},   {6, 7},   {1, 8},
                                          {8, 9},  {9, 10},  {1, 11},  {11, 12}, {12, 13}, {1, 0},   {0, 14},
                                          {14, 16}, {0, 15}, {15, 17}, {2, 16},  {5, 17}};

/// rgb [N,128,128,3], joints [N,P,18,3] (x, y, visibility), boxes [N,P,4]
/// (centre x, centre y, w, h), counts/image_ids int32 [N].
struct PafData {
  nn::Tensor rgb, joints, boxes, counts, image_ids;
};
struct PafSplits {
  PafData fit, validation, test;
};

inline PafData PafBatch(const PafData& data, const std::vector<std::size_t>& rows) {
  return {PoseBatch(data.rgb, rows), PoseBatch(data.joints, rows), PoseBatch(data.boxes, rows), PoseBatch(data.counts, rows),
          PoseBatch(data.image_ids, rows)};
}

/// COCO17 [N,P,17,3] -> OpenPose COCO18 order with a neck at the midpoint of
/// two labeled shoulders (visibility = min flag). Unlabeled joints are zero.
inline nn::Tensor Coco18(const nn::Tensor& input) {
  check::Require(input.dim() == 4 && input.size(2) == 17 && input.size(3) == 3, "COCO17 person annotations");
  const int          mapping[] = {0, -1, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3};
  const std::int64_t N = input.size(0), P = input.size(1);
  const auto         in = input.to_vector<float>();
  std::vector<float> out(std::size_t(N * P * 18 * 3), 0.f);
  auto               src = [&](std::int64_t n, std::int64_t p, int j, int c) { return in[std::size_t(((n * P + p) * 17 + j) * 3 + c)]; };
  for (std::int64_t n = 0; n < N; ++n)
    for (std::int64_t person = 0; person < P; ++person)
      for (int joint = 0; joint < 18; ++joint) {
        float* o = &out[std::size_t(((n * P + person) * 18 + joint) * 3)];
        if (joint == 1) {
          const float a = src(n, person, 5, 2), b = src(n, person, 6, 2);
          if (a > 0 && b > 0) {
            for (int c = 0; c < 2; ++c) o[c] = .5f * (src(n, person, 5, c) + src(n, person, 6, c));
            o[2] = std::min(a, b);
          }
        } else if (src(n, person, mapping[joint], 2) > 0)
          for (int c = 0; c < 3; ++c) o[c] = src(n, person, mapping[joint], c);
      }
  return nn::from_vector(out, {N, P, 18, 3});
}

inline PafSplits LoadPafImages(const std::filesystem::path& root) {
  const auto    arrays = PoseArrays(root / "poses.npz");
  std::set<int> previous;
  auto          read = [&](const std::string& split, std::int64_t count) {
    PafData result{arrays.at(split + "_full_rgb"), Coco18(arrays.at(split + "_full_joints")), arrays.at(split + "_full_boxes"),
                   arrays.at(split + "_full_counts"), arrays.at(split + "_full_image_ids")};
    const std::vector<std::pair<nn::Tensor, nn::Shape>> shapes = {{result.rgb, {count, 128, 128, 3}}, {result.joints, {count, 8, 18, 3}},
                                                                  {result.boxes, {count, 8, 4}},      {result.counts, {count}},
                                                                  {result.image_ids, {count}}};
    for (const auto& item : shapes) check::Require(item.first.shape() == item.second, "full-person pose array geometry");
    check::Require(result.rgb.dtype() == nn::DType::Float32 && result.joints.dtype() == nn::DType::Float32 &&
                       result.boxes.dtype() == nn::DType::Float32 && result.counts.dtype() == nn::DType::Int32 &&
                       result.image_ids.dtype() == nn::DType::Int32,
                   "full-person pose numeric types");
    const auto    counts = result.counts.to_vector<std::int32_t>(), image_ids = result.image_ids.to_vector<std::int32_t>();
    const auto    boxes = result.boxes.to_vector<float>(), joints = result.joints.to_vector<float>();
    std::set<int> ids;
    for (std::int64_t n = 0; n < count; ++n) {
      const int people = counts[std::size_t(n)], id = image_ids[std::size_t(n)];
      check::Require(people > 0 && people <= 8 && !previous.count(id), "person count and disjoint source-image IDs");
      ids.insert(id);
      for (int person = 0; person < people; ++person) {
        for (int c = 0; c < 4; ++c) {
          const float b = boxes[std::size_t((n * 8 + person) * 4 + c)];
          check::Require(std::isfinite(b) && (c < 2 || b > 0), "finite positive person boxes");
        }
        for (int j = 0; j < 18; ++j) {
          const float* o = &joints[std::size_t(((n * 8 + person) * 18 + j) * 3)];
          check::Require(o[2] == 0 || o[2] == 1 || o[2] == 2, "COCO18 visibility");
          if (o[2] > 0)
            for (int c = 0; c < 2; ++c) check::Require(std::isfinite(o[c]), "finite annotated full-image joints");
        }
      }
    }
    for (float v : result.rgb.to_vector<float>()) check::Require(std::isfinite(v) && v >= 0 && v <= 1, "real full-image RGB values");
    previous.insert(ids.begin(), ids.end());
    return result;
  };
  auto fit = read("fit", 6), validation = read("validation", 1), test = read("test", 4);
  return {fit, validation, test};
}

/// heatmap/heatmask [N,side,side,19] (18 joints + background), paf/pafmask [N,side,side,38].
struct PafTargets {
  nn::Tensor heatmap, paf, heatmask, pafmask;
};

/// Max-combined Gaussians, background = 1 - max, limb strips of half-width
/// `thickness` averaged over overlapping people; unknown joints/limbs zero the
/// mask inside that person's box (conservative, not COCO crowd segmentation).
inline PafTargets PreparePafTargets(const PafData& data, int side = 16, int input_size = 128, float sigma = 1.f, float thickness = 1.f) {
  check::Require(side > 0 && input_size > 0 && sigma > 0 && thickness > 0, "PAF target configuration");
  const std::int64_t batch = data.joints.size(0), P = data.joints.size(1);
  const auto         jv = data.joints.to_vector<float>(), bv = data.boxes.to_vector<float>();
  const auto         cv = data.counts.to_vector<std::int32_t>();
  const std::size_t  pixels = std::size_t(batch * side * side);
  std::vector<float> heat(pixels * 19, 0.f), paf(pixels * 38, 0.f), heatmask(pixels * 19, 1.f), pafmask(pixels * 38, 1.f), counts(pixels * 19, 0.f);
  auto               at = [&](std::int64_t n, int y, int x) { return std::size_t((n * side + y) * side + x); };
  for (std::int64_t n = 0; n < batch; ++n)
    for (int person = 0; person < cv[std::size_t(n)]; ++person) {
      std::array<bool, 18>                  annotated{};
      std::array<std::array<double, 2>, 18> points{};
      for (int j = 0; j < 18; ++j) {
        const float* o = &jv[std::size_t(((n * P + person) * 18 + j) * 3)];
        annotated[std::size_t(j)] = o[2] > 0;
        if (annotated[std::size_t(j)])
          for (int c = 0; c < 2; ++c) {
            const float value = o[c];
            annotated[std::size_t(j)] = annotated[std::size_t(j)] && std::isfinite(value) && value >= 0 && value <= input_size - 1;
            points[std::size_t(j)][std::size_t(c)] = (value + .5) * side / input_size - .5;
          }
      }
      const float* box = &bv[std::size_t((n * P + person) * 4)];
      const double bx = box[0], by = box[1], bw = box[2], bh = box[3];
      for (int y = 0; y < side; ++y)
        for (int x = 0; x < side; ++x) {
          const double      ix = (x + .5) * input_size / side - .5, iy = (y + .5) * input_size / side - .5;
          const bool        in_box = std::abs(ix - bx) <= bw / 2 && std::abs(iy - by) <= bh / 2;
          const std::size_t q      = at(n, y, x);
          for (int j = 0; j < 18; ++j) {
            if (!annotated[std::size_t(j)]) {
              if (in_box) heatmask[q * 19 + std::size_t(j)] = 0;
              continue;
            }
            const double dx = x - points[std::size_t(j)][0], dy = y - points[std::size_t(j)][1];
            if (std::abs(dx) <= 3 * sigma && std::abs(dy) <= 3 * sigma)
              heat[q * 19 + std::size_t(j)] =
                  std::max(heat[q * 19 + std::size_t(j)], float(std::exp(-(dx * dx + dy * dy) / (2 * sigma * sigma))));
          }
          for (int edge = 0; edge < 19; ++edge) {
            const int a = kPoseEdges[edge][0], b = kPoseEdges[edge][1];
            if (!annotated[std::size_t(a)] || !annotated[std::size_t(b)]) {
              if (in_box)
                for (int c = 0; c < 2; ++c) pafmask[q * 38 + std::size_t(2 * edge + c)] = 0;
              continue;
            }
            const auto&  pa = points[std::size_t(a)];
            const auto&  pb = points[std::size_t(b)];
            const double dx = pb[0] - pa[0], dy = pb[1] - pa[1], length = std::hypot(dx, dy);
            if (length < 1e-9) continue;
            const double ux = dx / length, uy = dy / length, rx = x - pa[0], ry = y - pa[1], along = rx * ux + ry * uy,
                         across = std::abs(rx * uy - ry * ux);
            if (along >= 0 && along <= length && across <= thickness) {
              paf[q * 38 + std::size_t(2 * edge)] += float(ux);
              paf[q * 38 + std::size_t(2 * edge + 1)] += float(uy);
              counts[q * 19 + std::size_t(edge)] += 1;
            }
          }
        }
    }
  for (std::size_t q = 0; q < pixels; ++q) {
    float maximum = 0, valid = 1;
    for (int j = 0; j < 18; ++j) {
      maximum = std::max(maximum, heat[q * 19 + std::size_t(j)]);
      valid   = std::min(valid, heatmask[q * 19 + std::size_t(j)]);
    }
    heat[q * 19 + 18]     = 1 - maximum;
    heatmask[q * 19 + 18] = valid;
    for (int edge = 0; edge < 19; ++edge)
      if (counts[q * 19 + std::size_t(edge)] > 0)
        for (int c = 0; c < 2; ++c) paf[q * 38 + std::size_t(2 * edge + c)] /= counts[q * 19 + std::size_t(edge)];
  }
  const nn::Shape s19{batch, side, side, 19}, s38{batch, side, side, 38};
  return {nn::from_vector(heat, s19), nn::from_vector(paf, s38), nn::from_vector(heatmask, s19), nn::from_vector(pafmask, s38)};
}

/// Per-image mean over masked entries, averaged over images with any mask;
/// masked-out predictions (even NaN) get exactly zero derivative.
inline nn::Tensor PafMaskedMSE(const nn::Tensor& prediction, const nn::Tensor& truth, const nn::Tensor& mask) {
  const nn::Tensor active  = nn::greater(mask, .5);
  const nn::Tensor p       = nn::where(active, prediction, 0.0), t = nn::where(active, truth, 0.0);
  const nn::Tensor count   = nn::sum(mask, {1, 2, 3});
  const nn::Tensor present = nn::cast(nn::greater(count, 0.0), prediction.dtype());
  const nn::Tensor mean    = nn::sum(nn::squared_difference(p, t), {1, 2, 3}) / nn::maximum(count, nn::scalar_like(1.0, count));
  return nn::sum(mean * present) / nn::maximum(nn::sum(present), nn::scalar_like(1.0, count));
}

} // namespace paf_reference

// Bottom-up local peaks, directed PAF line scores, matching and person
// assembly: a port of tflow's common/ml/paf_decode.h. Fields are read on the
// host through PafField (one copy of an NHWC nn tensor).
#pragma once

#include <algorithm>
#include <numeric>



namespace paf_reference {
namespace nn=datamunge::nn;

struct PafPeak {
  int   joint;
  float x, y, score;
};
struct PafLink {
  int   a, b, edge;
  float score;
};
struct PafPerson {
  std::array<int, 18> joints;
  std::vector<int>    links;
  PafPerson() { joints.fill(-1); }
};
struct PafDecoded {
  std::vector<PafPeak>   peaks;
  std::vector<PafLink>   links;
  std::vector<PafPerson> people;
};

/// Host copy of an NHWC float field.
struct PafField {
  std::vector<float> v;
  std::int64_t       n = 0, h = 0, w = 0, c = 0;
  PafField() = default;
  PafField(const nn::Tensor& t) : v(t.to_vector<float>()) {  // NOLINT(google-explicit-constructor)
    check::Require(t.dim() == 4, "NHWC field");
    n = t.size(0), h = t.size(1), w = t.size(2), c = t.size(3);
  }
  float operator()(std::int64_t b, std::int64_t y, std::int64_t x, std::int64_t ch) const {
    return v[std::size_t(((b * h + y) * w + x) * c + ch)];
  }
};

/// Bilinear sample with border clamping.
inline float PafSample(const PafField& field, int n, float x, float y, int channel) {
  const int height = int(field.h), width = int(field.w);
  x = std::clamp(x, 0.f, float(width - 1));
  y = std::clamp(y, 0.f, float(height - 1));
  const int   x0 = int(std::floor(x)), y0 = int(std::floor(y)), x1 = std::min(x0 + 1, width - 1), y1 = std::min(y0 + 1, height - 1);
  const float ax = x - x0, ay = y - y0;
  return (1 - ay) * ((1 - ax) * field(n, y0, x0, channel) + ax * field(n, y0, x1, channel)) +
         ay * ((1 - ax) * field(n, y1, x0, channel) + ax * field(n, y1, x1, channel));
}

/// 3x3 local maxima above `threshold` per joint (plateau ties keep the first
/// pixel in raster order), the best `cap` per joint.
inline std::vector<PafPeak> PafPeaks(const PafField& heat, int n, float threshold = .05f, int cap = 20) {
  check::Require(heat.c == 19 && cap > 0, "COCO18 plus background heatmaps");
  std::vector<PafPeak> peaks;
  const int            h = int(heat.h), w = int(heat.w);
  for (int joint = 0; joint < 18; ++joint) {
    std::vector<PafPeak> local;
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        const float value = heat(n, y, x, joint);
        check::Require(std::isfinite(value), "finite confidence field");
        if (value <= threshold) continue;
        bool maximum = true;
        for (int dy = -1; dy <= 1; ++dy)
          for (int dx = -1; dx <= 1; ++dx) {
            const int yy = y + dy, xx = x + dx;
            if (yy < 0 || yy >= h || xx < 0 || xx >= w || (dx == 0 && dy == 0)) continue;
            const float neighbor = heat(n, yy, xx, joint);
            if (neighbor > value || (neighbor == value && yy * w + xx < y * w + x)) maximum = false;
          }
        if (maximum) local.push_back({joint, float(x), float(y), value});
      }
    std::stable_sort(local.begin(), local.end(), [](const auto& a, const auto& b) { return a.score > b.score; });
    if (int(local.size()) > cap) local.resize(std::size_t(cap));
    peaks.insert(peaks.end(), local.begin(), local.end());
  }
  return peaks;
}

/// Directed line integral of each limb field between candidate peaks with a
/// distance prior, then greedy one-to-one matching by score.
inline std::vector<PafLink> MatchPafPeaks(const std::vector<PafPeak>& peaks, const PafField& fields, int n, float minimum_projection = .05f,
                                          float minimum_fraction = .8f, int samples = 10) {
  check::Require(fields.c == 38 && samples >= 2, "directed19-limb field and line integration samples");
  for (float v : fields.v) check::Require(std::isfinite(v), "finite PAF vectors");
  std::vector<PafLink> links;
  for (int edge = 0; edge < 19; ++edge) {
    std::vector<PafLink> candidates;
    for (int a = 0; a < int(peaks.size()); ++a)
      if (peaks[std::size_t(a)].joint == kPoseEdges[edge][0])
        for (int b = 0; b < int(peaks.size()); ++b)
          if (peaks[std::size_t(b)].joint == kPoseEdges[edge][1]) {
            const auto&  pa = peaks[std::size_t(a)];
            const auto&  pb = peaks[std::size_t(b)];
            const double dx = pb.x - pa.x, dy = pb.y - pa.y, length = std::hypot(dx, dy);
            if (length < 1e-9) continue;
            double sum  = 0;
            int    good = 0;
            for (int i = 0; i < samples; ++i) {
              const float  t = float(i) / float(samples - 1), x = float(pa.x + t * dx), y = float(pa.y + t * dy);
              const double value = PafSample(fields, n, x, y, edge * 2) * dx / length + PafSample(fields, n, x, y, edge * 2 + 1) * dy / length;
              sum += value;
              good += value > minimum_projection;
            }
            const double score = sum / samples + std::min(0., .5 * double(fields.h) / length - 1);
            if (good >= minimum_fraction * samples && score > 0) candidates.push_back({a, b, edge, float(score)});
          }
    std::stable_sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) { return a.score > b.score; });
    std::set<int> used_a, used_b;
    for (const auto& candidate : candidates)
      if (!used_a.count(candidate.a) && !used_b.count(candidate.b)) {
        links.push_back(candidate);
        used_a.insert(candidate.a);
        used_b.insert(candidate.b);
      }
  }
  return links;
}

/// Grows person subsets link by link; merges two subsets only without a
/// conflicting same-type peak; keeps people with enough joints and score.
inline std::vector<PafPerson> AssemblePafPeople(const std::vector<PafPeak>& peaks, const std::vector<PafLink>& links, int minimum_joints = 3,
                                                float minimum_average = .05f) {
  std::vector<PafPerson> people;
  for (int index = 0; index < int(links.size()); ++index) {
    const auto& link = links[std::size_t(index)];
    const int   ja = peaks.at(std::size_t(link.a)).joint, jb = peaks.at(std::size_t(link.b)).joint;
    int         owner_a = -1, owner_b = -1;
    for (int i = 0; i < int(people.size()); ++i) {
      if (people[std::size_t(i)].joints[std::size_t(ja)] == link.a) owner_a = i;
      if (people[std::size_t(i)].joints[std::size_t(jb)] == link.b) owner_b = i;
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
      const int owner = owner_a >= 0 ? owner_a : owner_b, joint = owner_a >= 0 ? jb : ja, peak = owner_a >= 0 ? link.b : link.a;
      auto&     person = people[std::size_t(owner)];
      if (person.joints[std::size_t(joint)] < 0 || person.joints[std::size_t(joint)] == peak) {
        person.joints[std::size_t(joint)] = peak;
        person.links.push_back(index);
      }
    } else {
      auto& pa = people[std::size_t(owner_a)];
      auto& pb = people[std::size_t(owner_b)];
      bool  conflict = false;
      for (std::size_t j = 0; j < 18; ++j)
        if (pa.joints[j] >= 0 && pb.joints[j] >= 0 && pa.joints[j] != pb.joints[j]) conflict = true;
      if (conflict) continue;
      for (std::size_t j = 0; j < 18; ++j)
        if (pa.joints[j] < 0) pa.joints[j] = pb.joints[j];
      pa.links.insert(pa.links.end(), pb.links.begin(), pb.links.end());
      pa.links.push_back(index);
      people.erase(people.begin() + owner_b);
    }
  }
  std::vector<PafPerson> kept;
  for (const auto& person : people) {
    int    count = 0;
    double score = 0;
    for (int peak : person.joints)
      if (peak >= 0) {
        ++count;
        score += peaks[std::size_t(peak)].score;
      }
    for (int link : person.links) score += links[std::size_t(link)].score;
    if (count >= minimum_joints && score / double(std::size_t(count) + person.links.size()) >= minimum_average) kept.push_back(person);
  }
  return kept;
}

inline PafDecoded DecodePafs(const PafField& heat, const PafField& fields, int n) {
  auto peaks  = PafPeaks(heat, n);
  auto links  = MatchPafPeaks(peaks, fields, n);
  auto people = AssemblePafPeople(peaks, links);
  return {peaks, links, people};
}

struct PafMetrics {
  int    correct = 0, predicted = 0, annotated = 0, people = 0, true_people = 0;
  double precision() const { return predicted ? double(correct) / predicted : 0; }
  double recall() const { return annotated ? double(correct) / annotated : 0; }
};

/// Custom grouped metric: annotated joints (excluding the derived neck) within
/// .2 x the larger box side, greedy one-to-one person matching by correct count.
inline PafMetrics ScorePafs(const std::vector<PafDecoded>& decoded, const PafData& data, int side = 16) {
  check::Require(decoded.size() == std::size_t(data.rgb.size(0)), "one bottom-up decode per image");
  const std::int64_t P = data.joints.size(1);
  const auto         jv = data.joints.to_vector<float>(), bv = data.boxes.to_vector<float>();
  const auto         cv = data.counts.to_vector<std::int32_t>();
  auto joint = [&](int n, int person, int j, int c) { return jv[std::size_t(((n * P + person) * 18 + j) * 3 + c)]; };
  PafMetrics result;
  for (int n = 0; n < int(decoded.size()); ++n) {
    const auto& output = decoded[std::size_t(n)];
    result.people += int(output.people.size());
    const int truth_count = cv[std::size_t(n)];
    result.true_people += truth_count;
    for (int person = 0; person < truth_count; ++person)
      for (int j = 0; j < 18; ++j)
        if (j != 1 && joint(n, person, j, 2) > 0) ++result.annotated;
    for (const auto& person : output.people)
      for (int j = 0; j < 18; ++j)
        if (j != 1 && person.joints[std::size_t(j)] >= 0) ++result.predicted;
    struct Candidate {
      int prediction, truth, correct;
    };
    std::vector<Candidate> candidates;
    for (int i = 0; i < int(output.people.size()); ++i)
      for (int person = 0; person < truth_count; ++person) {
        int         correct = 0;
        const float scale   = std::max(bv[std::size_t((n * P + person) * 4 + 2)], bv[std::size_t((n * P + person) * 4 + 3)]);
        for (int j = 0; j < 18; ++j)
          if (j != 1 && joint(n, person, j, 2) > 0 && output.people[std::size_t(i)].joints[std::size_t(j)] >= 0) {
            const auto& peak = output.peaks[std::size_t(output.people[std::size_t(i)].joints[std::size_t(j)])];
            const float x = (peak.x + .5f) * 128 / float(side) - .5f, y = (peak.y + .5f) * 128 / float(side) - .5f;
            correct += std::hypot(x - joint(n, person, j, 0), y - joint(n, person, j, 1)) <= .2 * scale;
          }
        if (correct > 0) candidates.push_back({i, person, correct});
      }
    std::stable_sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) { return a.correct > b.correct; });
    std::set<int> predictions, truths;
    for (const auto& candidate : candidates)
      if (!predictions.count(candidate.prediction) && !truths.count(candidate.truth)) {
        result.correct += candidate.correct;
        predictions.insert(candidate.prediction);
        truths.insert(candidate.truth);
      }
  }
  return result;
}

} // namespace paf_reference
