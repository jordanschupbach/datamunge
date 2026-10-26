// Dataset-aware VOC2007/COCO box AP (port of tflow's
// common/ml/detection_evaluation.h): difficult/crowd ignore rules, greedy
// matching per image/class/IoU, stable score order and absent-class
// exclusion. VOC2007: strict IoU > .5 and 11-point interpolation, difficult
// hits ignored. COCO: IoU >= threshold, crowds match repeatedly using
// intersection/detection area, 101 recall samples, max100
// detections/image/class and .50:.05:.95. Reports all-area box AP; no
// segmentation/area-specific metrics are implied. The interpolated sums are
// divided once at the end (tflow adds precision/samples per sample).
// https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py
// https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2007/index.html
#pragma once

#include <set>

#include "common/detection_scenes.hpp"

namespace detection_evaluation {

using detector::Box;
using detector::Detection;

struct EvaluationTruth {
  int  image, label;
  Box  box;
  bool difficult = false, crowd = false;
};
struct EvaluationImage {
  int                   id, height, width;
  std::filesystem::path filename;
};
struct DatasetAp {
  double             ap50 = 0, ap75 = 0, ap = 0, voc07 = 0;
  int                classes = 0;
  std::map<int, double> per_class50;
};

inline double EvaluationOverlap(const Box& detection, const EvaluationTruth& truth, bool coco) {
  if (!coco || !truth.crowd) return detector::BoxIoU(detection, truth.box);
  const double h = std::max(0.f, std::min(detection[2], truth.box[2]) - std::max(detection[0], truth.box[0]));
  const double w = std::max(0.f, std::min(detection[3], truth.box[3]) - std::max(detection[1], truth.box[1]));
  return h * w / std::max(double(detector::BoxArea(detection)), 1e-12);
}

/// {interpolated AP (VOC07 11-point or COCO 101-point), precision-envelope integral}; {-1, -1} without positives.
inline std::pair<double, double> DatasetClassAp(const std::vector<Detection>& predictions,
                                                const std::vector<EvaluationTruth>& truth, const std::vector<int>& image_ids,
                                                int label, double threshold, bool coco) {
  struct Scored {
    float score;
    bool  positive, ignored;
  };
  std::vector<Scored> outcomes;
  int                 positives = 0;
  for (int image : image_ids) {
    std::vector<EvaluationTruth> gt;
    for (const auto& g : truth)
      if (g.image == image && g.label == label) gt.push_back(g);
    auto ignored = [&](const EvaluationTruth& g) { return coco ? g.crowd : g.difficult; };
    if (coco) std::stable_sort(gt.begin(), gt.end(), [&](const auto& a, const auto& b) { return ignored(a) < ignored(b); });
    for (const auto& g : gt)
      if (!ignored(g)) ++positives;
    std::vector<Detection> detections;
    for (const auto& p : predictions)
      if (p.image == image && p.label == label) detections.push_back(p);
    std::stable_sort(detections.begin(), detections.end(), [](const auto& a, const auto& b) { return a.score > b.score; });
    if (coco && detections.size() > 100) detections.resize(100);
    std::vector<bool> matched(gt.size(), false);
    for (const auto& detection : detections) {
      int    best    = -1;
      double overlap = coco ? std::min(threshold, 1 - 1e-10) : threshold;
      // VOC chooses the maximum-overlap GT even when it was already matched;
      // a repeated detection of that object is a false positive.
      for (std::size_t j = 0; j < gt.size(); ++j) {
        if (coco && matched[j] && !gt[j].crowd) continue;
        if (coco && best >= 0 && !ignored(gt[std::size_t(best)]) && ignored(gt[j])) break;
        const double candidate = EvaluationOverlap(detection.box, gt[j], coco);
        if (coco ? candidate < overlap : candidate <= overlap) continue;
        best    = int(j);
        overlap = candidate;
      }
      if (best < 0) outcomes.push_back({detection.score, false, false});
      else if (ignored(gt[std::size_t(best)])) {
        outcomes.push_back({detection.score, false, true});
        matched[std::size_t(best)] = true;
      } else {
        outcomes.push_back({detection.score, !matched[std::size_t(best)], false});
        matched[std::size_t(best)] = true;
      }
    }
  }
  if (!positives) return {-1, -1};
  std::stable_sort(outcomes.begin(), outcomes.end(), [](const auto& a, const auto& b) { return a.score > b.score; });
  std::vector<double> recall, precision;
  int                 tp = 0, fp = 0;
  for (const auto& r : outcomes) {
    if (r.ignored) continue;
    tp += r.positive;
    fp += !r.positive;
    recall.push_back(double(tp) / positives);
    precision.push_back(double(tp) / std::max(1, tp + fp));
  }
  for (int i = int(precision.size()) - 2; i >= 0; --i)
    precision[std::size_t(i)] = std::max(precision[std::size_t(i)], precision[std::size_t(i) + 1]);
  const int samples = coco ? 101 : 11;
  double    interpolated = 0;
  for (int k = 0; k < samples; ++k) {
    const double r     = k * (1. / (samples - 1));
    const auto   found = std::lower_bound(recall.begin(), recall.end(), r);
    if (found != recall.end()) interpolated += precision[std::size_t(found - recall.begin())];
  }
  interpolated /= samples;
  double integral = 0, previous = 0;
  for (std::size_t i = 0; i < recall.size(); ++i) {
    integral += (recall[i] - previous) * precision[i];
    previous = recall[i];
  }
  return {interpolated, integral};
}

inline DatasetAp EvaluateDataset(const std::vector<Detection>& predictions, const std::vector<EvaluationTruth>& truth,
                                 const std::vector<int>& images, bool coco) {
  std::set<int> labels;
  for (const auto& g : truth) labels.insert(g.label);
  DatasetAp result;
  for (int label : labels) {
    const auto first = DatasetClassAp(predictions, truth, images, label, .5, coco);
    if (first.first < 0) continue;
    ++result.classes;
    result.ap50 += coco ? first.first : first.second;
    result.voc07 += coco ? 0 : first.first;
    result.per_class50[label] = coco ? first.first : first.second;
    if (coco)
      for (int k = 0; k < 10; ++k) {
        const auto ap = DatasetClassAp(predictions, truth, images, label, .5 + ((.95 - .5) / 9) * k, true).first;
        result.ap += ap / 10;
        if (k == 5) result.ap75 += ap;
      }
  }
  if (result.classes) {
    result.ap50 /= result.classes;
    result.ap75 /= result.classes;
    result.ap /= result.classes;
    result.voc07 /= result.classes;
  }
  return result;
}

inline std::vector<EvaluationImage> LoadEvaluationImages(const std::filesystem::path& directory) {
  std::ifstream stream(directory / "images.tsv");
  if (!stream) throw std::runtime_error("missing evaluation images.tsv; run python3 scripts/fetch_datasets.py detection_eval");
  std::vector<EvaluationImage> result;
  EvaluationImage              row;
  std::string                  filename;
  while (stream >> row.id >> row.height >> row.width >> filename) {
    if (row.height < 1 || row.width < 1) throw std::runtime_error("invalid evaluation image dimensions");
    row.filename = directory / filename;
    result.push_back(row);
  }
  if (!stream.eof() || result.empty()) throw std::runtime_error("malformed/empty evaluation image list");
  return result;
}
inline std::vector<EvaluationTruth> LoadEvaluationTruth(const std::filesystem::path& directory) {
  std::ifstream stream(directory / "annotations.tsv");
  if (!stream) throw std::runtime_error("missing evaluation annotations.tsv");
  std::vector<EvaluationTruth> result;
  EvaluationTruth              row;
  int                          difficult, crowd;
  double                       area;
  while (stream >> row.image >> row.label >> row.box[0] >> row.box[1] >> row.box[2] >> row.box[3] >> difficult >> crowd >> area) {
    if (detector::BoxArea(row.box) <= 0 || !(area >= 0)) throw std::runtime_error("invalid evaluation box");
    row.difficult = difficult;
    row.crowd     = crowd;
    result.push_back(row);
  }
  if (!stream.eof()) throw std::runtime_error("malformed evaluation annotations");
  return result;
}

/// Reads a COCO-style detection list [{"image_id":..,"category_id":..,"score":..,"bbox":[x,y,w,h]},...]
/// (the format detection_evaluation writes) into detections with [ymin, xmin, ymax, xmax) boxes.
inline std::vector<Detection> LoadPredictionsJson(const std::filesystem::path& path) {
  std::ifstream stream(path);
  if (!stream) throw std::runtime_error("missing " + path.string());
  const std::string      text((std::istreambuf_iterator<char>(stream)), {});
  std::vector<Detection> result;
  auto number = [&](std::size_t& at, const std::string& key) {
    at = text.find("\"" + key + "\":", at);
    if (at == std::string::npos) throw std::runtime_error("malformed predictions.json");
    at += key.size() + 3;
    std::size_t used = 0;
    const double v   = std::stod(text.substr(at, 40), &used);
    at += used;
    return v;
  };
  for (std::size_t at = text.find('{'); at != std::string::npos; at = text.find('{', at)) {
    Detection d;
    d.image          = int(number(at, "image_id"));
    d.label          = int(number(at, "category_id"));
    d.score          = float(number(at, "score"));
    std::size_t open = text.find('[', at), close = text.find(']', open);
    std::stringstream values(text.substr(open + 1, close - open - 1));
    double            x, y, w, h;
    char              comma;
    values >> x >> comma >> y >> comma >> w >> comma >> h;
    d.box = {float(y), float(x), float(y + h), float(x + w)};
    result.push_back(d);
    at = close;
  }
  return result;
}

}  // namespace detection_evaluation
