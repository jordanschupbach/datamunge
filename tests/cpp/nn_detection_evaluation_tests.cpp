#include <dm/dm.hpp>
#include <gtest/gtest.h>
#include <random>
namespace n = dm::nn;
namespace m = n::models;
namespace oracle {
using Box = n::Box;
using EvaluationTruth = m::DetectionEvaluationTruth;
using Detection = m::DetectionEvaluationPrediction;
double area(const Box &b) {
  return std::max(0.f, b[2] - b[0]) * std::max(0.f, b[3] - b[1]);
}
double iou(const Box &a, const Box &b) {
  const double intersection = std::max(0.f, std::min(a[2], b[2]) - std::max(a[0], b[0])) *
                              std::max(0.f, std::min(a[3], b[3]) - std::max(a[1], b[1]));
  return intersection ? intersection / (area(a) + area(b) - intersection) : 0;
}
inline double EvaluationOverlap(const Box &detection, const EvaluationTruth &truth,
                                bool coco) {
  if (!coco || !truth.crowd)
    return iou(detection, truth.box);
  const double h = std::max(0.f, std::min(detection[2], truth.box[2]) -
                                     std::max(detection[0], truth.box[0]));
  const double w = std::max(0.f, std::min(detection[3], truth.box[3]) -
                                     std::max(detection[1], truth.box[1]));
  return h * w / std::max(double(area(detection)), 1e-12);
}

/// {interpolated AP (VOC07 11-point or COCO 101-point), precision-envelope integral}; {-1, -1} without positives.
inline std::pair<double, double> DatasetClassAp(const std::vector<Detection> &predictions,
                                                const std::vector<EvaluationTruth> &truth,
                                                const std::vector<int> &image_ids, int label,
                                                double threshold, bool coco) {
  struct Scored {
    float score;
    bool positive, ignored;
  };
  std::vector<Scored> outcomes;
  int positives = 0;
  for (int image : image_ids) {
    std::vector<EvaluationTruth> gt;
    for (const auto &g : truth)
      if (g.image == image && g.label == label)
        gt.push_back(g);
    auto ignored = [&](const EvaluationTruth &g) {
      return coco ? g.crowd : g.difficult;
    };
    if (coco)
      std::stable_sort(gt.begin(), gt.end(), [&](const auto &a, const auto &b) {
        return ignored(a) < ignored(b);
      });
    for (const auto &g : gt)
      if (!ignored(g))
        ++positives;
    std::vector<Detection> detections;
    for (const auto &p : predictions)
      if (p.image == image && p.label == label)
        detections.push_back(p);
    std::stable_sort(detections.begin(), detections.end(), [](const auto &a, const auto &b) {
      return a.score > b.score;
    });
    if (coco && detections.size() > 100)
      detections.resize(100);
    std::vector<bool> matched(gt.size(), false);
    for (const auto &detection : detections) {
      int best = -1;
      double overlap = coco ? std::min(threshold, 1 - 1e-10) : threshold;
      // VOC chooses the maximum-overlap GT even when it was already matched;
      // a repeated detection of that object is a false positive.
      for (std::size_t j = 0; j < gt.size(); ++j) {
        if (coco && matched[j] && !gt[j].crowd)
          continue;
        if (coco && best >= 0 && !ignored(gt[std::size_t(best)]) && ignored(gt[j]))
          break;
        const double candidate = EvaluationOverlap(detection.box, gt[j], coco);
        if (coco ? candidate < overlap : candidate <= overlap)
          continue;
        best = int(j);
        overlap = candidate;
      }
      if (best < 0)
        outcomes.push_back({detection.score, false, false});
      else if (ignored(gt[std::size_t(best)])) {
        outcomes.push_back({detection.score, false, true});
        matched[std::size_t(best)] = true;
      } else {
        outcomes.push_back({detection.score, !matched[std::size_t(best)], false});
        matched[std::size_t(best)] = true;
      }
    }
  }
  if (!positives)
    return {-1, -1};
  std::stable_sort(outcomes.begin(), outcomes.end(), [](const auto &a, const auto &b) {
    return a.score > b.score;
  });
  std::vector<double> recall, precision;
  int tp = 0, fp = 0;
  for (const auto &r : outcomes) {
    if (r.ignored)
      continue;
    tp += r.positive;
    fp += !r.positive;
    recall.push_back(double(tp) / positives);
    precision.push_back(double(tp) / std::max(1, tp + fp));
  }
  for (int i = int(precision.size()) - 2; i >= 0; --i)
    precision[std::size_t(i)] =
        std::max(precision[std::size_t(i)], precision[std::size_t(i) + 1]);
  const int samples = coco ? 101 : 11;
  double interpolated = 0;
  for (int k = 0; k < samples; ++k) {
    const double r = k * (1. / (samples - 1));
    const auto found = std::lower_bound(recall.begin(), recall.end(), r);
    if (found != recall.end())
      interpolated += precision[std::size_t(found - recall.begin())];
  }
  interpolated /= samples;
  double integral = 0, previous = 0;
  for (std::size_t i = 0; i < recall.size(); ++i) {
    integral += (recall[i] - previous) * precision[i];
    previous = recall[i];
  }
  return {interpolated, integral};
}

} // namespace oracle

TEST(NnDetectionEvaluation, DifficultAndDuplicateRules) {
  const std::vector<m::DetectionEvaluationTruth> truth = {
      {1, 1, {0, 0, 10, 10}, false, false}, {1, 1, {20, 20, 40, 40}, true, false}};
  const std::vector<m::DetectionEvaluationPrediction> pred = {{1, 1, .99f, {20, 20, 40, 40}},
                                                              {1, 1, .9f, {0, 0, 10, 10}},
                                                              {1, 1, .8f, {0, 0, 10, 10}}};
  const auto r = m::EvaluateDetectionClass(pred, truth, {1}, 1, .5, false);
  EXPECT_EQ(r.positives, 1);
  EXPECT_EQ(r.interpolated.item(), 1);
  EXPECT_EQ(r.integral.item(), 1);
  EXPECT_EQ(r.recall.to_vector<double>(), std::vector<double>({1, 1}));
}
TEST(NnDetectionEvaluation, RepeatedCrowdHitsAndRegularTruthPreference) {
  const std::vector<m::DetectionEvaluationTruth> truth = {
      {1, 1, {0, 0, 20, 20}, false, true}, {1, 1, {30, 30, 40, 40}, false, false}};
  const std::vector<m::DetectionEvaluationPrediction> pred = {
      {1, 1, .99f, {1, 1, 3, 3}}, {1, 1, .98f, {4, 4, 8, 8}}, {1, 1, .9f, {30, 30, 40, 40}}};
  const auto r = m::EvaluateDetectionClass(pred, truth, {1}, 1, .5, true);
  EXPECT_EQ(r.positives, 1);
  EXPECT_EQ(r.recall.numel(), 1);
  EXPECT_EQ(r.interpolated.item(), 1);
  const auto preferred = m::EvaluateDetectionClass(
      {{1, 1, .9f, {0, 0, 10, 10}}},
      {{1, 1, {0, 0, 10, 10}, false, true}, {1, 1, {0, 0, 10, 20}, false, false}}, {1}, 1, .5,
      true);
  EXPECT_EQ(preferred.interpolated.item(), 1);
}
TEST(NnDetectionEvaluation, ExactThresholdDiffersAndEmptyPredictionsAreZero) {
  const std::vector<m::DetectionEvaluationTruth> t = {{1, 1, {0, 0, 10, 10}}};
  const std::vector<m::DetectionEvaluationPrediction> p = {{1, 1, .9f, {0, 0, 10, 20}}};
  EXPECT_EQ(m::EvaluateDetectionClass(p, t, {1}, 1, .5, true).interpolated.item(), 1);
  EXPECT_EQ(m::EvaluateDetectionClass(p, t, {1}, 1, .5, false).interpolated.item(), 0);
  EXPECT_EQ(m::EvaluateDetectionClass({}, t, {1}, 1, .5, true).interpolated.item(), 0);
  EXPECT_EQ(m::EvaluateDetectionClass({}, t, {2}, 1, .5, true).interpolated.item(), -1);
}
TEST(NnDetectionEvaluation, Cap100AndAbsentCategoryExclusion) {
  std::vector<m::DetectionEvaluationPrediction> p;
  for (int i = 0; i < 100; ++i)
    p.push_back({1, 1, 1.f - i * .001f, {30, 30, 40, 40}});
  p.push_back({1, 1, .1f, {0, 0, 10, 10}});
  const std::vector<m::DetectionEvaluationTruth> t = {{1, 1, {0, 0, 10, 10}},
                                                      {1, 99, {0, 0, 20, 20}, false, true}};
  EXPECT_EQ(m::EvaluateDetectionClass(p, t, {1}, 1, .5, true).interpolated.item(), 0);
  const auto r = m::EvaluateDetectionDataset({}, t, {1}, true);
  EXPECT_EQ(r.classes, 1);
  EXPECT_EQ(r.ap.item(), 0);
  EXPECT_EQ(r.per_class50.size(), 1);
}
TEST(NnDetectionEvaluation, RecallGridAndThreeApDefinitions) {
  std::vector<m::DetectionEvaluationTruth> t;
  std::vector<m::DetectionEvaluationPrediction> p;
  for (int i = 0; i < 10; ++i) {
    n::Box b = {float(i * 20), 0, float(i * 20 + 10), 10};
    t.push_back({1, 1, b});
    if (i < 3)
      p.push_back({1, 1, .9f - .1f * i, b});
  }
  const auto v = m::EvaluateDetectionClass(p, t, {1}, 1, .5, false),
             c = m::EvaluateDetectionClass(p, t, {1}, 1, .5, true);
  EXPECT_NEAR(v.interpolated.item(), 3. / 11, 1e-12);
  EXPECT_NEAR(v.integral.item(), .3, 1e-12);
  EXPECT_NEAR(c.interpolated.item(), 31. / 101, 1e-12);
}
TEST(NnDetectionEvaluation, ExplicitSyntheticVocInclusiveOption) {
  const std::vector<m::DetectionEvaluationTruth> t = {{1, 1, {0, 0, 10, 10}}};
  const std::vector<m::DetectionEvaluationPrediction> p = {{1, 1, .9f, {0, 0, 10, 20}}};
  EXPECT_EQ(m::EvaluateDetectionClass(p, t, {1}, 1, .5, false, true).integral.item(), 1);
  EXPECT_EQ(m::EvaluateDetectionDataset(p, t, {1}, false, true).ap50.item(), 1);
}
TEST(NnDetectionEvaluation, StableScoreAndIouTieRules) {
  const std::vector<m::DetectionEvaluationTruth> t = {{1, 1, {0, 0, 10, 10}},
                                                      {1, 1, {0, 0, 10, 10}}};
  const std::vector<m::DetectionEvaluationPrediction> p = {{1, 1, .9f, {0, 0, 10, 10}},
                                                           {1, 1, .9f, {0, 0, 10, 10}}};
  EXPECT_NEAR(m::EvaluateDetectionClass(p, t, {1}, 1, .5, false).integral.item(), .5, 1e-12);
  EXPECT_NEAR(m::EvaluateDetectionClass(p, t, {1}, 1, .5, true).integral.item(), 1, 1e-12);
}
TEST(NnDetectionEvaluation, RandomizedTensorApMatchesIndependentSourceOracle) {
  std::mt19937 rng(31);
  std::uniform_int_distribution<int> coord(0, 20), side(1, 15), flag(0, 4);
  for (int trial = 0; trial < 12; ++trial) {
    std::vector<m::DetectionEvaluationTruth> t;
    std::vector<m::DetectionEvaluationPrediction> p;
    for (int i = 0; i < 12; ++i) {
      const float y = coord(rng), x = coord(rng);
      n::Box box = {y, x, y + side(rng), x + side(rng)};
      t.push_back({i % 3, i % 2, box, flag(rng) == 0, flag(rng) == 0});
      if (i % 3)
        p.push_back({i % 3, i % 2, float(1 - i * .05), box});
    }
    for (int i = 0; i < 15; ++i) {
      const float y = coord(rng), x = coord(rng);
      p.push_back({i % 3, i % 2, float(.95 - i * .04), {y, x, y + side(rng), x + side(rng)}});
    }
    for (bool coco : {false, true})
      for (int label : {0, 1})
        for (double cutoff : {.5, .75, .95}) {
          const auto expected = oracle::DatasetClassAp(p, t, {0, 1, 2}, label, cutoff, coco);
          const auto actual = m::EvaluateDetectionClass(p, t, {0, 1, 2}, label, cutoff, coco);
          EXPECT_NEAR(actual.interpolated.item(), expected.first, 1e-12);
          EXPECT_NEAR(actual.integral.item(), expected.second, 1e-12);
        }
  }
}
TEST(NnDetectionEvaluation, InvalidRecordsAndSelectionFailEvenWithoutTruth) {
  EXPECT_THROW(m::EvaluateDetectionDataset({}, {}, {1, 1}, true), std::invalid_argument);
  EXPECT_THROW((m::EvaluateDetectionDataset({{1, 0, NAN, {0, 0, 1, 1}}}, {}, {1}, true)),
               std::invalid_argument);
  EXPECT_THROW((m::EvaluateDetectionClass({}, {{1, 0, {1, 0, 0, 1}}}, {1}, 0, .5, true)),
               std::invalid_argument);
  EXPECT_THROW(m::EvaluateDetectionClass({}, {}, {1}, 0, NAN, true), std::invalid_argument);
  EXPECT_THROW(m::EvaluateDetectionClass({}, {}, {1}, -1, .5, true), std::invalid_argument);
  EXPECT_THROW(m::EvaluateDetectionClass({}, {}, {-1}, 0, .5, true), std::invalid_argument);
}
