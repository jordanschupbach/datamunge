// Bottom-up local peaks, directed PAF line scores, matching and person
// assembly: a port of tflow's common/ml/paf_decode.h. Fields are read on the
// host through PafField (one copy of an NHWC nn tensor).
#pragma once

#include <algorithm>
#include <numeric>

#include "common/paf_data.hpp"

namespace pose {

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

}  // namespace pose
