// CMD: just run-nn detection_evaluation
// Real Pascal VOC / COCO subset box mAP of a published detector's predictions.
//
// Port of tflow's detection_evaluation (partial). Preparation, in
// ~/projects/tflow_examples: python3 scripts/fetch_datasets.py detection_eval --count 64,
// then run tflow's detection_evaluation, which writes predictions.json next
// to each subset's annotations.
//
// tflow loads the official TensorFlow SSD MobileNet-v2 320x320 COCO17
// SavedModel and runs it on seeded 64-image VOC2007-test/COCO2017-val
// subsets. Loading SavedModels has no datamunge counterpart (out of scope),
// so this port keeps the evaluator: it scores the predictions.json the TF run
// exported (original COCO category ids, [x, y, w, h] boxes) against the same
// annotations.tsv, preserving difficult/crowd flags. VOC classes map to their
// COCO category IDs; VOC boxes convert one-based inclusive to zero-based
// half-open (done by the preparation helper). Report VOC07 11-point AP50 and
// precision-envelope integral AP50; COCO uses 101 recall points, IoU
// .50:.05:.95, all areas, max100/image/class and crowd
// intersection/detection-area matching. Classes absent from nonignored truth
// are omitted from the mean. These small-subset numbers are not
// full-benchmark claims.
//
// Fixtures check difficult hits, repeated crowd hits, duplicate detections,
// empty predictions, absent classes, threshold equality and max-detections.
// Writes metrics_nn.tsv beside each subset's annotations. When the prepared
// subsets or the TF predictions are missing, only the fixtures run.
// Options: data=PATH (default <datasets>/detection_eval).

#include <iomanip>
#include <iostream>

#include "common/detection_evaluation.hpp"

namespace ev = detection_evaluation;

namespace {
void EvaluationChecks() {
  using ev::EvaluationTruth;
  using detector::Detection;
  const std::vector<EvaluationTruth> truth = {
      {1, 1, {0, 0, 10, 10}, false, false}, {1, 1, {20, 20, 40, 40}, true, false}, {2, 2, {0, 0, 20, 20}, false, true}};
  const std::vector<Detection> predictions = {
      {1, 1, .99f, {20, 20, 40, 40}}, {1, 1, .9f, {0, 0, 10, 10}}, {1, 1, .8f, {0, 0, 10, 10}}, {2, 2, .9f, {1, 1, 3, 3}}, {2, 2, .8f, {4, 4, 8, 8}}};
  const auto voc = ev::DatasetClassAp(predictions, truth, {1}, 1, .5, false);
  check::Near(voc.first, 1, 1e-12, "difficult hit ignored before true positive");
  check::Near(voc.second, 1, 1e-12, "duplicate after full recall cannot increase AP");
  check::Near(ev::DatasetClassAp(predictions, truth, {2}, 2, .5, true).first, -1, 0, "crowd-only category absent from positive mean");
  const std::vector<EvaluationTruth> crowd      = {{1, 1, {0, 0, 20, 20}, false, true}, {1, 1, {30, 30, 40, 40}, false, false}};
  const std::vector<Detection>       crowd_hits = {{1, 1, .99f, {1, 1, 3, 3}}, {1, 1, .98f, {4, 4, 8, 8}}, {1, 1, .9f, {30, 30, 40, 40}}};
  check::Near(ev::DatasetClassAp(crowd_hits, crowd, {1}, 1, .5, true).first, 1, 1e-12,
              "multiple interior crowd detections ignored with detection-area denominator");
  const std::vector<EvaluationTruth> one  = {{1, 1, {0, 0, 10, 10}, false, false}};
  const std::vector<Detection>       half = {{1, 1, .9f, {0, 0, 10, 20}}};
  check::Near(ev::DatasetClassAp(half, one, {1}, 1, .5, true).first, 1, 1e-12, "COCO inclusive IoU equality");
  check::Near(ev::DatasetClassAp(half, one, {1}, 1, .5, false).first, 0, 0, "VOC strict IoU threshold");
  check::Near(ev::DatasetClassAp({}, one, {1}, 1, .5, true).first, 0, 0, "empty predictions AP zero");
  std::vector<Detection> many;
  for (int i = 0; i < 100; ++i) many.push_back({1, 1, 1 - i * .001f, {30, 30, 40, 40}});
  many.push_back({1, 1, .1f, {0, 0, 10, 10}});
  check::Near(ev::DatasetClassAp(many, one, {1}, 1, .5, true).first, 0, 0, "COCO cap100 applies before matching");
  std::vector<EvaluationTruth> ten;
  std::vector<Detection>       three;
  for (int i = 0; i < 10; ++i) {
    const detector::Box box = {float(20 * i), 0, float(20 * i + 10), 10};
    ten.push_back({1, 1, box, false, false});
    if (i < 3) three.push_back({1, 1, .9f - .1f * i, box});
  }
  check::Near(ev::DatasetClassAp(three, ten, {1}, 1, .5, false).first, 3. / 11, 1e-12, "VOC recall grid at .3");
  const auto all = ev::EvaluateDataset({}, truth, {1, 2}, true);
  check::Require(all.classes == 1, "crowd-only categories excluded from mean");
  std::cout << "  VOC difficult/duplicate/strict threshold and COCO crowd/threshold/cap/empty/absent-class fixtures passed"
            << std::endl;
}

bool Evaluate(const std::filesystem::path& directory, bool coco) {
  if (!std::filesystem::exists(directory / "annotations.tsv") || !std::filesystem::exists(directory / "predictions.json")) {
    std::cout << "  " << directory.string() << ": prepared annotations or TF predictions.json missing; skipped" << std::endl;
    return false;
  }
  const auto       images = ev::LoadEvaluationImages(directory);
  const auto       truth  = ev::LoadEvaluationTruth(directory);
  const auto       detections = ev::LoadPredictionsJson(directory / "predictions.json");
  std::vector<int> ids;
  for (const auto& image : images) ids.push_back(image.id);
  const auto result = ev::EvaluateDataset(detections, truth, ids, coco);
  std::cout << std::fixed << std::setprecision(6) << "== " << directory.filename().string() << ": images=" << images.size()
            << ", boxes=" << truth.size() << ", predictions=" << detections.size() << ", evaluated classes=" << result.classes
            << ", integral/COCO AP50=" << result.ap50;
  if (coco) std::cout << ", AP75=" << result.ap75 << ", AP50:95=" << result.ap;
  else std::cout << ", VOC07 AP50=" << result.voc07;
  std::cout << std::defaultfloat << std::endl;
  std::ofstream metrics(directory / "metrics_nn.tsv");
  metrics << std::setprecision(12) << "ap50\t" << result.ap50 << "\nap75\t" << result.ap75 << "\nap\t" << result.ap << "\nvoc07\t"
          << result.voc07 << "\nclasses\t" << result.classes << '\n';
  check::Require(result.classes > 0 && result.ap50 > .05, "published detector has measurable real-dataset AP50");
  return true;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    std::filesystem::path directory = paths::Dataset("detection_eval");
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg.rfind("data=", 0) == 0) directory = arg.substr(5);
      else throw std::invalid_argument("expected data=PATH");
    }
    EvaluationChecks();
    const bool voc = Evaluate(directory / "voc2007", false), coco = Evaluate(directory / "coco2017", true);
    if (voc && coco) std::cout << "All real-dataset AP checks passed.\n";
    else std::cout << "Evaluator fixtures passed; real-dataset evaluation needs the prepared subsets and TF predictions.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
