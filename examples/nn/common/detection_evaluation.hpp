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
  int image, label;
  Box box;
  bool difficult = false, crowd = false;
};
struct EvaluationImage {
  int id, height, width;
  std::filesystem::path filename;
};
struct DatasetAp {
  double ap50 = 0, ap75 = 0, ap = 0, voc07 = 0;
  int classes = 0;
  std::map<int, double> per_class50;
};

inline std::vector<datamunge::nn::models::DetectionEvaluationPrediction>
TensorPredictions(const std::vector<Detection> &predictions) {
  std::vector<datamunge::nn::models::DetectionEvaluationPrediction> out;
  for (const auto &p : predictions)
    out.push_back({p.image, p.label, p.score, p.box});
  return out;
}
inline std::vector<datamunge::nn::models::DetectionEvaluationTruth>
TensorTruth(const std::vector<EvaluationTruth> &truth) {
  std::vector<datamunge::nn::models::DetectionEvaluationTruth> out;
  for (const auto &t : truth)
    out.push_back({t.image, t.label, t.box, t.difficult, t.crowd});
  return out;
}
inline std::pair<double, double> DatasetClassAp(const std::vector<Detection> &predictions,
                                                const std::vector<EvaluationTruth> &truth,
                                                const std::vector<int> &image_ids, int label,
                                                double threshold, bool coco) {
  const auto result = datamunge::nn::models::EvaluateDetectionClass(
      TensorPredictions(predictions), TensorTruth(truth), image_ids, label, threshold, coco);
  return {result.interpolated.item(), result.integral.item()};
}
inline DatasetAp EvaluateDataset(const std::vector<Detection> &predictions,
                                 const std::vector<EvaluationTruth> &truth,
                                 const std::vector<int> &images, bool coco) {
  const auto result = datamunge::nn::models::EvaluateDetectionDataset(
      TensorPredictions(predictions), TensorTruth(truth), images, coco);
  DatasetAp out;
  out.ap50 = result.ap50.item();
  out.ap75 = result.ap75.item();
  out.ap = result.ap.item();
  out.voc07 = result.voc07.item();
  out.classes = result.classes;
  for (const auto &[label, ap] : result.per_class50)
    out.per_class50[label] = ap.item();
  return out;
}

inline std::vector<EvaluationImage>
LoadEvaluationImages(const std::filesystem::path &directory) {
  std::ifstream stream(directory / "images.tsv");
  if (!stream)
    throw std::runtime_error(
        "missing evaluation images.tsv; run python3 scripts/fetch_datasets.py detection_eval");
  std::vector<EvaluationImage> result;
  EvaluationImage row;
  std::string filename;
  while (stream >> row.id >> row.height >> row.width >> filename) {
    if (row.height < 1 || row.width < 1)
      throw std::runtime_error("invalid evaluation image dimensions");
    row.filename = directory / filename;
    result.push_back(row);
  }
  if (!stream.eof() || result.empty())
    throw std::runtime_error("malformed/empty evaluation image list");
  return result;
}
inline std::vector<EvaluationTruth>
LoadEvaluationTruth(const std::filesystem::path &directory) {
  std::ifstream stream(directory / "annotations.tsv");
  if (!stream)
    throw std::runtime_error("missing evaluation annotations.tsv");
  std::vector<EvaluationTruth> result;
  EvaluationTruth row;
  int difficult, crowd;
  double area;
  while (stream >> row.image >> row.label >> row.box[0] >> row.box[1] >> row.box[2] >>
         row.box[3] >> difficult >> crowd >> area) {
    if (detector::BoxArea(row.box) <= 0 || !(area >= 0))
      throw std::runtime_error("invalid evaluation box");
    row.difficult = difficult;
    row.crowd = crowd;
    result.push_back(row);
  }
  if (!stream.eof())
    throw std::runtime_error("malformed evaluation annotations");
  return result;
}

/// Reads a COCO-style detection list [{"image_id":..,"category_id":..,"score":..,"bbox":[x,y,w,h]},...]
/// (the format detection_evaluation writes) into detections with [ymin, xmin, ymax, xmax) boxes.
inline std::vector<Detection> LoadPredictionsJson(const std::filesystem::path &path) {
  std::ifstream stream(path);
  if (!stream)
    throw std::runtime_error("missing " + path.string());
  const std::string text((std::istreambuf_iterator<char>(stream)), {});
  std::vector<Detection> result;
  auto number = [&](std::size_t &at, const std::string &key) {
    at = text.find("\"" + key + "\":", at);
    if (at == std::string::npos)
      throw std::runtime_error("malformed predictions.json");
    at += key.size() + 3;
    std::size_t used = 0;
    const double v = std::stod(text.substr(at, 40), &used);
    at += used;
    return v;
  };
  for (std::size_t at = text.find('{'); at != std::string::npos; at = text.find('{', at)) {
    Detection d;
    d.image = int(number(at, "image_id"));
    d.label = int(number(at, "category_id"));
    d.score = float(number(at, "score"));
    std::size_t open = text.find('[', at), close = text.find(']', open);
    std::stringstream values(text.substr(open + 1, close - open - 1));
    double x, y, w, h;
    char comma;
    values >> x >> comma >> y >> comma >> w >> comma >> h;
    d.box = {float(y), float(x), float(y + h), float(x + w)};
    result.push_back(d);
    at = close;
  }
  return result;
}

} // namespace detection_evaluation
