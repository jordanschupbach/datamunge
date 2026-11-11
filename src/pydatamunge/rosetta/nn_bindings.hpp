// Generated from prebindings/nn/src/nn.i and examples/bindings/nn_components.i, then maintained by hand.
#pragma once

#include "datamunge/nn/nn.hpp"
#include "datamunge/nn/mt19937.hpp"
namespace datamunge::nn {
using EarlyStoppingDecision = EarlyStopping::Decision;
}
#include <datamunge/datasets/assets.hpp>
#include "common/runtime.hpp"
#include "common/paths.hpp"
#include "common/sentence_assets.hpp"
#include "common/checks.hpp"
#include "common/image_training.hpp"
#include "common/resnet.hpp"
#include "common/resnet_demo.hpp"
#include "common/architecture_checks.hpp"
#include "common/mobile_checks.hpp"
#include "common/grouped.hpp"
#include "common/lrn.hpp"
#include "common/inception.hpp"
#include "common/normalized_conv.hpp"
#include "common/deconvnet.hpp"
#include "common/alexnet.hpp"
#include "common/googlenet.hpp"
#include "common/inception23.hpp"
#include "common/inception4.hpp"
#include "common/nin.hpp"
#include "common/vgg.hpp"
#include "common/xception.hpp"
#include "common/zfnet.hpp"
#include "common/preact_resnet.hpp"
#include "common/densenet.hpp"
#include "common/dpn.hpp"
#include "common/pyramidnet.hpp"
#include "common/res2net.hpp"
#include "common/resnext.hpp"
#include "common/revnet.hpp"
#include "common/stochastic_depth.hpp"
#include "common/wide_resnet.hpp"
#include "common/mobilenet.hpp"
#include "common/mbconv.hpp"
#include "common/condensenet.hpp"
#include "common/ghostnet.hpp"
#include "common/regnet.hpp"
#include "common/shufflenet.hpp"
#include "common/squeezenet.hpp"
#include "common/convnext.hpp"
#include "common/deformable.hpp"
#include "common/non_local.hpp"
#include "common/octconv.hpp"
#include "common/repvgg.hpp"
#include "common/attention_cnn.hpp"
#include "common/mnist.hpp"
#include "common/cnn_baselines.hpp"
#include "common/memory_components.hpp"
#include "common/cnn_validation.hpp"
#include "common/residual_validation.hpp"
#include "common/autoencoder.hpp"
#include "common/segmentation.hpp"
#include "common/fcn.hpp"
#include "common/unet.hpp"
#include "common/unet_variants.hpp"
#include "common/segnet.hpp"
#include "common/hrnet.hpp"
#include "common/deeplab.hpp"
#include "common/deeplab_v3.hpp"
#include "common/pspnet.hpp"
#include "common/dense_crf.hpp"
#include "common/segmentation_losses.hpp"
#include "common/kim_cnn.hpp"
#include "common/tf_replay.hpp"
#include "common/sentence.hpp"
#include "common/time_series.hpp"
#include "common/ecg_sensor.hpp"
#include "common/character_cnn.hpp"
#include "common/character.hpp"
#include "common/tcn.hpp"
#include "common/wavenet_stack.hpp"
#include "common/saliency.hpp"
#include "common/dream.hpp"
#include "common/feature_visualization.hpp"
#include "common/grad_cam.hpp"
#include "common/occlusion.hpp"
#include "common/integrated_gradients.hpp"
#include "common/smoothgrad.hpp"
#include "common/guided_backprop.hpp"
#include "common/detection.hpp"
#include "common/detection_scenes.hpp"
#include "common/fpn.hpp"
#include "common/two_stage.hpp"
#include "common/retinanet.hpp"
#include "common/fcos.hpp"
#include "common/ssd.hpp"
#include "common/centernet.hpp"
#include "common/hungarian.hpp"
#include "common/detr.hpp"
#include "common/bifpn.hpp"
#include "common/efficientdet.hpp"
#include "common/detection_mosaic.hpp"
#include "common/detection_affine.hpp"
#include "common/word_tree.hpp"
#include "common/yolo_v1.hpp"
#include "common/yolo_v2.hpp"
#include "common/yolo_v3.hpp"
#include "common/yolo_v5.hpp"
#include "common/instance_segmentation.hpp"
#include "common/mask_rcnn.hpp"
#include "common/panoptic.hpp"
#include "common/panoptic_fpn.hpp"
#include "common/panoptic_scenes.hpp"
#include "common/yolact.hpp"
#include "common/detection_evaluation.hpp"
#include "common/detection_metrics.hpp"
#include "common/highway.hpp"
#include "common/stn.hpp"
#include "common/capsnet.hpp"
#include "common/im2col.hpp"
#include "common/receptive_field.hpp"

// ---- %inline adapters ----
namespace paths {
inline std::string DataRootString() { return DataRoot().string(); }
inline std::string DatasetString(const std::string& name) { return Dataset(name).string(); }
inline std::string OutputRootString() { return OutputRoot().string(); }
inline std::string OutputDirString(const std::string& relative) { return OutputDir(relative).string(); }
}
namespace image_training {
inline std::pair<datamunge::nn::ImageDataset, datamunge::nn::ImageDataset> LoadCifar10String(const std::string& directory) {
  return LoadCifar10(std::filesystem::path(directory));
}
}
namespace nn_components {
/// A C++ image_training::LogitsBuilder callable from R and Python:
/// model.call(params, state, images, training) -> logits.
struct LogitsModel {
  image_training::LogitsBuilder builder;
  datamunge::nn::Tensor call(datamunge::nn::ParameterStore& params, datamunge::nn::ParameterStore& state,
                             const datamunge::nn::Tensor& images, bool training) const {
    if (!builder) throw std::runtime_error("empty LogitsModel");
    return builder(params, state, images, training);
  }
};
}
namespace resnet_demo {
inline nn_components::LogitsModel CompactResNetModel(int depth) { return {CompactResNet(depth)}; }
}
namespace condensenet {
struct CondenseContextHandle {
  std::vector<mobilenet::MobileLayer> layers;
  CondenseContext context;
  CondenseContextHandle(datamunge::nn::ParameterStore& p, datamunge::nn::ParameterStore& state, bool training,
                        double momentum = .9, double progress = 0)
      : context{p, state, layers, training, momentum, progress} {}
  CondenseContext& get() { return context; }
};
inline LearnedGroupConvOutputs LearnedGroupConvHandle(CondenseContextHandle& handle, const std::string& name, const datamunge::nn::Tensor& x,
                                               std::int64_t outputs, int groups, int condense) {
  return LearnedGroupConv(handle.context, name, x, outputs, groups, condense);
}
}
namespace shufflenet {
struct ShuffleContextHandle {
  std::vector<std::pair<datamunge::nn::Shape, std::int64_t>> layers;
  ShuffleContext context;
  ShuffleContextHandle(datamunge::nn::ParameterStore& p, datamunge::nn::ParameterStore& state, bool training,
                       double momentum = .9, bool update_stats = true, bool record_layers = false)
      : context{p, state, training, momentum, update_stats, record_layers ? &layers : nullptr} {}
  ShuffleContext& get() { return context; }
  int layer_count() const { return int(layers.size()); }
  std::vector<std::int64_t> layer_shape(int index) const { return layers.at(static_cast<std::size_t>(index)).first; }
  std::int64_t layer_cost(int index) const { return layers.at(static_cast<std::size_t>(index)).second; }
};
}
namespace ae {
inline bool SameTensorBits(const nn::Tensor& first, const nn::Tensor& second) {
  return nn::equal_bits(first, second);
}
inline std::vector<SaliencyData> LoadSaliencyString(const std::string& root) {
  return LoadSaliency(root);
}
inline void SaveDataString(const std::string& file, const std::vector<SaliencyData>& pools) {
  SaveData(file, pools);
}
}
namespace seg {
inline std::string SegmentationDirectoryString(const std::string& name) { return SegmentationDirectory(name).string(); }
/// Confusion counts (truth rows, prediction columns, row-major) of logits [B, H, W, classes]
/// against labels [B, H, W]; the host example sums these over its evaluation batches and
/// passes the total to ConfusionMeanIou.
inline std::vector<std::int64_t> SegmentationConfusion(const nn::Tensor& logits, const nn::Tensor& labels, int classes) {
  nn::NoGradGuard           guard;
  std::vector<std::int64_t> confusion(std::size_t(classes) * classes, 0);
  const auto predictions = nn::argmax(logits, 3).to_vector<std::int64_t>();
  const auto targets     = labels.to_vector<std::int32_t>();
  check::Require(predictions.size() == targets.size(), "prediction/mask pixel counts agree");
  for (std::size_t i = 0; i < targets.size(); ++i) {
    const int target = targets[i], prediction = int(predictions[i]);
    check::Require(prediction >= 0 && prediction < classes, "valid segmentation prediction");
    ++confusion[std::size_t(target * classes + prediction)];
  }
  return confusion;
}
/// SegmentationPreview from the host example's own logits of rows 0..min(4, count)-1.
inline void SegmentationPreviewLogits(const nn::Tensor& logits, const SegmentationData& data, const std::string& filename) {
  const nn::Tensor fixed = logits.detach();
  SegmentationPreview([&](const nn::Tensor&) { return fixed; }, data, filename);
}
inline std::vector<double> SquaredDistance2DInt(const std::vector<int>& seeds, int height, int width) {
  std::vector<bool> flags(seeds.begin(), seeds.end());
  return SquaredDistance2D(flags, height, width);
}
}
namespace deeplab {
inline std::vector<nn::Tensor> AtrousScorePyramidVector(nn::ParameterStore& p, const std::string& name, const nn::Tensor& features,
                                                 int classes, const std::vector<int>& rates) {
  if (rates.size() != 4) throw std::invalid_argument("AtrousScorePyramid takes four rates");
  return AtrousScorePyramid(p, name, features, classes, std::array<int, 4>{rates[0], rates[1], rates[2], rates[3]});
}
}
namespace sentence {
/// RequireLiveGradients with the parameters named in `may_vanish` excused from the nonzero check.
inline int RequireLiveGradientsNamed(const nn::ParameterStore& p, const std::vector<Tensor>& gradients, const std::string& what,
                              const std::vector<std::string>& may_vanish = {}, double floor = 0) {
  if (may_vanish.empty()) return RequireLiveGradients(p, gradients, what, {}, floor);
  return RequireLiveGradients(
      p, gradients, what,
      [&](const std::string& name) { return std::find(may_vanish.begin(), may_vanish.end(), name) != may_vanish.end(); }, floor);
}
inline void ExportArraysFile(const std::string& path, const nn::ParameterStore& p) { ExportArrays(std::filesystem::path(path), p); }
inline void ExportArraysFile(const std::string& path, const nn::ParameterStore& p, const nn::ParameterStore& s) {
  ExportArrays(std::filesystem::path(path), p, &s);
}
inline void CopyIfExistsFile(const std::string& from, const std::string& to) {
  CopyIfExists(std::filesystem::path(from), std::filesystem::path(to));
}
inline SentenceSplits LoadSentencesDir(const std::string& root) { return LoadSentences(std::filesystem::path(root)); }
/// Snapshot of one or two stores as named tensors ("<k>:<name>", k = store index).
inline nn::NamedTensors SnapshotNamed(const nn::ParameterStore& p) { return replay::ToNamed(Snapshot({&p})); }
inline nn::NamedTensors SnapshotNamed(const nn::ParameterStore& p, const nn::ParameterStore& s) {
  return replay::ToNamed(Snapshot({&p, &s}));
}
}
namespace replay {
inline std::string TfModelsString(const std::string& name) { return TfModels(name).string(); }
inline bool SameBitsTensor(const Tensor& a, const Tensor& b) { return SameBits(a, b); }
/// Bit-identical named tensors (same keys, same bytes).
inline bool SameBitsTensor(const nn::NamedTensors& a, const nn::NamedTensors& b) { return SameBits(ToMap(a), ToMap(b)); }
inline void LoadIntoFile(nn::ParameterStore& p, const std::string& file) { LoadInto(p, std::filesystem::path(file)); }
/// max |a[k] - file[k]| over the names of `a`.
inline double MaxDiffFile(const nn::NamedTensors& a, const std::string& file) { return MaxDiff(ToMap(a), std::filesystem::path(file)); }
inline double MaxDiffFile(const nn::NamedTensors& a, const nn::NamedTensors& b) { return MaxDiff(ToMap(a), ToMap(b)); }
inline std::vector<double> HistoryFile(const std::string& file, int column = 1) { return History(std::filesystem::path(file), column); }
inline void CopyAssetsDir(const std::string& from, const std::string& to, const std::vector<std::string>& files) {
  CopyAssets(std::filesystem::path(from), std::filesystem::path(to), files);
}
}
namespace series {
inline SeriesSplits LoadSeriesDir(const std::string& root) { return LoadSeries(std::filesystem::path(root)); }
}
namespace sensor {
inline SensorSplits LoadSensorDir(const std::string& root) { return LoadSensor(std::filesystem::path(root)); }
inline void CompareWithTflowDir(const std::string& root, const std::string& tf_dir) {
  CompareWithTflow(std::filesystem::path(root), std::filesystem::path(tf_dir));
}
}
namespace charcnn {
inline CharacterSplits LoadCharactersDir(const std::string& root) { return LoadCharacters(std::filesystem::path(root)); }
}
namespace wavestack {
inline WaveSplits LoadWaveStackDir(const std::string& root) { return LoadWaveStack(std::filesystem::path(root)); }
}
namespace sal {
inline SaliencyCnnOutput SaliencyConvFeaturesRelu(nn::ParameterStore& p, const Tensor& image) { return SaliencyConvFeatures(p, image); }
inline SaliencyCnnOutput SaliencyCnnRelu(nn::ParameterStore& p, const Tensor& image) { return SaliencyCnn(p, image); }
/// Convolutional features with guided_backprop's GuidedRelu gate (the full guided network is guided::GuidedSaliencyCnn).
inline SaliencyCnnOutput SaliencyConvFeaturesGuided(nn::ParameterStore& p, const Tensor& image) {
  return SaliencyConvFeatures(p, image, Gate(guided::GuidedRelu));
}
inline nn::NamedTensors LoadArraysNamed(const std::string& file) {
  const auto arrays = LoadArrays(std::filesystem::path(file));
  return nn::NamedTensors(arrays.begin(), arrays.end());
}
inline void SaveArraysNamed(const std::string& file, const nn::NamedTensors& arrays) {
  SaveArrays(std::filesystem::path(file), Arrays(arrays.begin(), arrays.end()));
}
inline void SaliencyPngFile(const std::string& path, const Canvas& canvas) { SaliencyPng(std::filesystem::path(path), canvas); }
inline std::vector<double> SignedColorVector(double g) {
  const auto rgb = SignedColor(g);
  return {rgb[0], rgb[1], rgb[2]};
}
inline std::string SourceDirString(const std::string& value) { return SourceDir(value).string(); }
inline std::string DefaultSourceString() { return DefaultSource().string(); }
/// RestoreSource; returns whether `source` is our own run (best.npz present) rather than tflow's record.
inline bool RestoreSourceDir(const std::string& source, SaliencyModel& model, SaliencyModel& initial) {
  return RestoreSource(std::filesystem::path(source), model, initial).ours;
}
inline void SaliencyGalleryDir(const std::string& directory, const SaliencyData& examples, const Tensor& targets, const Tensor& runners,
                        const Tensor& gradients, const Tensor& alternative, const Tensor& random, double peak) {
  SaliencyGallery(std::filesystem::path(directory), examples, targets, runners, gradients, alternative, random, peak);
}
}
namespace detector {
/// [ymin, xmin, ymax, xmax) from four values; BoxValues reads one back as a plain vector.
inline Box MakeBox(float ymin, float xmin, float ymax, float xmax) { return {ymin, xmin, ymax, xmax}; }
inline std::vector<float> BoxValues(const Box& box) { return {box[0], box[1], box[2], box[3]}; }
inline nn::BoxVariances MakeBoxVariances(double y, double x, double height, double width) { return {y, x, height, width}; }
/// A std::mt19937_64 the host example owns, for the samplers that draw from one
/// (SampleRois, MakeRpnTargets) and the std::shuffle / uniform draws around them.
struct Mt19937_64 {
  std::mt19937_64 engine;
  explicit Mt19937_64(std::uint64_t seed) : engine(seed) {}
  std::uint64_t next() { return engine(); }
  /// std::uniform_int_distribution<int>(low, high)(engine).
  int uniform_int(int low, int high) { return std::uniform_int_distribution<int>(low, high)(engine); }
  /// std::uniform_real_distribution<float>(low, high)(engine).
  float uniform_float(float low, float high) { return std::uniform_real_distribution<float>(low, high)(engine); }
  /// std::shuffle of the values with this engine.
  std::vector<int> shuffle(std::vector<int> values) {
    std::shuffle(values.begin(), values.end(), engine);
    return values;
  }
};
inline std::vector<double> ClassApVector(const std::vector<Detection>& detections, const DetectionScenes& scenes,
                                  const std::vector<int>& rows, int label, float threshold, bool voc) {
  const auto ap = ClassAp(detections, scenes, rows, label, threshold, voc);
  return {ap[0], ap[1], ap[2]};
}
inline Box DecodeBoxVector(const Box& anchor, const std::vector<float>& delta) {
  if (delta.size() != 4) throw std::invalid_argument("box delta takes four values");
  return DecodeBox(anchor, delta.data());
}
inline Box DecodeBoxWithVector(const Box& anchor, const std::vector<float>& delta, const nn::BoxVariances& variance) {
  if (delta.size() != 4) throw std::invalid_argument("box delta takes four values");
  return DecodeBoxWith(anchor, delta.data(), variance);
}
inline std::string TrainingDirectoryString(const std::string& name) { return TrainingDirectory(name).string(); }
inline Digits LoadDigitsDir(const std::string& directory, const std::string& split) {
  return LoadDigits(std::filesystem::path(directory), split);
}
inline std::pair<DetectionScenes, DetectionScenes> StandardScenesDir(const std::string& directory, int size = 64) {
  return StandardScenes(std::filesystem::path(directory), size);
}
/// CropScene as a [size, size] float tensor in [-1, 1].
inline nn::Tensor CropSceneTensor(const DetectionScenes& scenes, int row, const Box& box, int size) {
  std::vector<float> out(std::size_t(size) * size);
  CropScene(scenes, row, box, size, out.data());
  return nn::from_vector(out, {size, size});
}
inline void SampleRoisEngine(const DetectionScenes& scenes, int row, int index, const std::vector<Box>& candidates, int total,
                      int background, int stride, int feature_size, Mt19937_64& random, RoiBatch& batch) {
  SampleRois(scenes, row, index, candidates, total, background, stride, feature_size, random.engine, batch);
}
inline RpnTargets MakeRpnTargetsEngine(const std::vector<Box>& anchors, const DetectionScenes& scenes, const std::vector<int>& rows,
                                Mt19937_64& random, int per_image = 128) {
  return MakeRpnTargets(anchors, scenes, rows, random.engine, per_image);
}
/// RpnProposals of one image from its objectness [A] and deltas [A * 4] values.
inline std::vector<Box> RpnProposalsVector(const std::vector<float>& objectness, const std::vector<float>& deltas,
                                    const std::vector<Box>& anchors, int height, int width, int before = 300, int after = 100) {
  if (objectness.size() != anchors.size() || deltas.size() != 4 * anchors.size())
    throw std::invalid_argument("RPN proposals take one objectness and four deltas per anchor");
  return RpnProposals(objectness.data(), deltas.data(), anchors, height, width, before, after);
}
}
namespace detr {
inline detector::Box DetrCornersVector(const std::vector<float>& center_size) {
  if (center_size.size() != 4) throw std::invalid_argument("DETR boxes take four values");
  return DetrCorners(center_size.data());
}
}
namespace yolo {
/// Height/width prior from two values.
inline Dimension MakeDimension(float height, float width) { return {height, width}; }
inline std::vector<float> DimensionValues(const Dimension& d) { return {d[0], d[1]}; }
inline void WriteYoloV1BootstrapPath(const YoloV1PretrainedWeights& weights, const YoloV1Config& cfg, const std::string& prefix) {
  WriteYoloV1Bootstrap(weights, cfg, std::filesystem::path(prefix));
}
/// DecodeYoloV2 of the cell values starting at `offset` in `raw` (a prediction's to_vector values).
inline Box DecodeYoloV2Values(const std::vector<float>& raw, int offset, int row, int col, int cells, const Dimension& prior, int height, int width) {
  if (offset < 0 || std::size_t(offset) + 5 > raw.size()) throw std::out_of_range("YOLO v2 cell offset");
  return DecodeYoloV2(raw.data() + offset, row, col, cells, prior, height, width);
}
/// MakeYoloV3Targets from the three raw head tensors (coarse, middle, fine).
inline YoloV3Targets MakeYoloV3TargetsTensors(const nn::Tensor& coarse, const nn::Tensor& middle, const nn::Tensor& fine,
                                       const std::vector<Dimension>& priors, const DetectionScenes& scenes, const std::vector<int>& rows) {
  return MakeYoloV3Targets({coarse.to_vector<float>(), middle.to_vector<float>(), fine.to_vector<float>()},
                           {coarse.shape(), middle.shape(), fine.shape()}, priors, scenes, rows);
}
/// MakeYoloV5Targets from the three raw head tensors' shapes (coarse, middle, fine).
inline YoloV5Targets MakeYoloV5TargetsTensors(const nn::Tensor& coarse, const nn::Tensor& middle, const nn::Tensor& fine,
                                       const std::vector<Dimension>& priors, const DetectionScenes& scenes, const std::vector<int>& rows) {
  return MakeYoloV5Targets({coarse.shape(), middle.shape(), fine.shape()}, priors, scenes, rows);
}
inline Box DecodeYoloV5Values(const std::vector<float>& raw, int offset, int row, int col, int cells, const Dimension& prior, int stride, int height,
                       int width) {
  if (offset < 0 || std::size_t(offset) + 5 > raw.size()) throw std::out_of_range("YOLO v5 cell offset");
  return DecodeYoloV5(raw.data() + offset, row, col, cells, prior, stride, height, width);
}
/// A CspContext over the host example's stores (the struct itself holds references).
struct CspContextHandle {
  CspContext context;
  CspContextHandle(nn::ParameterStore& parameters, nn::ParameterStore& state, bool training = false, bool update_stats = true)
      : context{parameters, state, training, update_stats} {}
  CspContext& get() { return context; }
};
}
namespace instance {
/// PasteMask of the `side * side * classes` probabilities starting at `offset` in `probabilities`.
inline BinaryMask PasteMaskValues(const std::vector<float>& probabilities, int offset, int side, int classes, int label, const Box& box,
                           int height, int width) {
  if (offset < 0 || std::size_t(offset) + std::size_t(side) * side * classes > probabilities.size())
    throw std::out_of_range("mask probabilities offset");
  return PasteMask(probabilities.data() + offset, side, classes, label, box, height, width);
}
inline void ExportInstanceMasksPath(const InstanceScenes& scenes, const std::vector<MaskDetection>& detections, const std::string& filename) {
  ExportInstanceMasks(scenes, detections, std::filesystem::path(filename));
}
}
namespace panoptic {
/// [segment id, pixel area] rows of PanopticAreas.
inline std::vector<std::vector<int>> PanopticAreasList(const PanopticMap& map) {
  std::vector<std::vector<int>> rows;
  for (const auto& [id, area] : PanopticAreas(map)) rows.push_back({id, area});
  return rows;
}
/// Per-class PQ statistics (a std::map<int, PqStat> the host example accumulates with add()).
struct PqStatsHandle {
  PqStats stats;
  std::vector<int> labels() const {
    std::vector<int> result;
    for (const auto& [label, _] : stats) result.push_back(label);
    return result;
  }
  PqStat at(int label) const { return stats.at(label); }
  bool has(int label) const { return stats.count(label) != 0; }
  /// AddPanopticStats(this, sample).
  void add(const PqStatsHandle& sample) { AddPanopticStats(stats, sample.stats); }
};
inline PqStatsHandle PanopticStatsHandle(const PanopticMap& truth, const PanopticMap& prediction) {
  PqStatsHandle handle;
  handle.stats = PanopticStats(truth, prediction);
  return handle;
}
/// PanopticQuality with `things` given as 0/1 per class.
inline PqResult PanopticQualityHandle(const PqStatsHandle& stats, const std::vector<int>& things, int subset = -1) {
  return PanopticQuality(stats.stats, std::vector<bool>(things.begin(), things.end()), subset);
}
inline void PanopticPreviewPath(const PanopticScenes& data, const std::vector<std::vector<int>>& semantic, const std::vector<PanopticMap>& predicted,
                         const std::string& filename) {
  PanopticPreview(data, semantic, predicted, std::filesystem::path(filename));
}
}
namespace yolact {
/// A std::mt19937 the host example owns for MakeYolactTargets' positive-mask subsampling.
struct Mt19937 {
  std::mt19937 engine;
  explicit Mt19937(std::uint32_t seed) : engine(seed) {}
  std::uint32_t next() { return engine(); }
};
inline YolactTargets MakeYolactTargetsEngine(const YolactOutputs& net, const std::vector<Box>& anchors, const instance::InstanceScenes& scenes,
                                      const std::vector<int>& rows, int classes, Mt19937& random) {
  return MakeYolactTargets(net, anchors, scenes, rows, classes, random.engine);
}
}
namespace detection_evaluation {
inline std::vector<EvaluationImage> LoadEvaluationImagesDir(const std::string& directory) {
  return LoadEvaluationImages(std::filesystem::path(directory));
}
inline std::vector<EvaluationTruth> LoadEvaluationTruthDir(const std::string& directory) {
  return LoadEvaluationTruth(std::filesystem::path(directory));
}
inline std::vector<Detection> LoadPredictionsJsonFile(const std::string& path) { return LoadPredictionsJson(std::filesystem::path(path)); }
}

namespace nn_python_extend {
namespace nn = datamunge::nn;

inline double datamunge__nn__Tensor__item(const datamunge::nn::Tensor& self) { return self.item<double>(); }
inline std::vector<double> datamunge__nn__Tensor__values(const datamunge::nn::Tensor& self) { return self.to_vector<double>(); }
inline std::string datamunge__nn__Tensor____repr__(const datamunge::nn::Tensor& self) { return self.str(); }
inline datamunge::nn::Tensor resnet__ResNetOutputs__stage(const resnet::ResNetOutputs& self, int index) {
    if (index < 0 || index >= 4) throw std::out_of_range("ResNet stage index 0..3");
    return self.stages[static_cast<std::size_t>(index)];
  }
inline datamunge::nn::Tensor inception__InceptionOutputs__branches_at(const inception::InceptionOutputs& self, int index) {
    if (index < 0 || index >= 4) throw std::out_of_range("branches" " index must be in 0.." "4");
    return self.branches[static_cast<std::size_t>(index)];
  }
inline datamunge::nn::Tensor alexnet__AlexNetOutputs__convolution_at(const alexnet::AlexNetOutputs& self, int index) {
    if (index < 0 || index >= 5) throw std::out_of_range("convolution" " index must be in 0.." "5");
    return self.convolution[static_cast<std::size_t>(index)];
  }
inline datamunge::nn::Tensor alexnet__AlexNetOutputs__pooling_at(const alexnet::AlexNetOutputs& self, int index) {
    if (index < 0 || index >= 3) throw std::out_of_range("pooling" " index must be in 0.." "3");
    return self.pooling[static_cast<std::size_t>(index)];
  }
inline datamunge::nn::Tensor googlenet__GoogLeNetOutputs__stem_at(const googlenet::GoogLeNetOutputs& self, int index) {
    if (index < 0 || index >= 3) throw std::out_of_range("stem" " index must be in 0.." "3");
    return self.stem[static_cast<std::size_t>(index)];
  }
inline datamunge::nn::Tensor googlenet__GoogLeNetOutputs__inception_at(const googlenet::GoogLeNetOutputs& self, int index) {
    if (index < 0 || index >= 9) throw std::out_of_range("inception" " index must be in 0.." "9");
    return self.inception[static_cast<std::size_t>(index)];
  }
inline datamunge::nn::Tensor nin__NinOutputs__blocks_at(const nin::NinOutputs& self, int index) {
    if (index < 0 || index >= 3) throw std::out_of_range("blocks" " index must be in 0.." "3");
    return self.blocks[static_cast<std::size_t>(index)];
  }
inline datamunge::nn::Tensor nin__NinOutputs__pooling_at(const nin::NinOutputs& self, int index) {
    if (index < 0 || index >= 2) throw std::out_of_range("pooling" " index must be in 0.." "2");
    return self.pooling[static_cast<std::size_t>(index)];
  }
inline datamunge::nn::Tensor vgg__VggOutputs__pooling_at(const vgg::VggOutputs& self, int index) {
    if (index < 0 || index >= 5) throw std::out_of_range("pooling" " index must be in 0.." "5");
    return self.pooling[static_cast<std::size_t>(index)];
  }
inline datamunge::nn::Tensor zfnet__ZFNetOutputs__convolution_at(const zfnet::ZFNetOutputs& self, int index) {
    if (index < 0 || index >= 5) throw std::out_of_range("convolution" " index must be in 0.." "5");
    return self.convolution[static_cast<std::size_t>(index)];
  }
inline datamunge::nn::Tensor zfnet__ZFNetOutputs__pooling_at(const zfnet::ZFNetOutputs& self, int index) {
    if (index < 0 || index >= 3) throw std::out_of_range("pooling" " index must be in 0.." "3");
    return self.pooling[static_cast<std::size_t>(index)];
  }
inline datamunge::nn::Tensor zfnet__ZFNetOutputs__normalized_at(const zfnet::ZFNetOutputs& self, int index) {
    if (index < 0 || index >= 2) throw std::out_of_range("normalized" " index must be in 0.." "2");
    return self.normalized[static_cast<std::size_t>(index)];
  }
inline datamunge::nn::Tensor preact__PreActResNetOutputs__stages_at(const preact::PreActResNetOutputs& self, int index) {
    if (index < 0 || index >= 3) throw std::out_of_range("stages" " index must be in 0.." "3");
    return self.stages[static_cast<std::size_t>(index)];
  }
inline int dpn__DPNLayout__blocks_at(const dpn::DPNLayout& self, int stage) { return self.blocks.at(static_cast<std::size_t>(stage)); }
inline int dpn__DPNLayout__inc_at(const dpn::DPNLayout& self, int stage) { return self.inc.at(static_cast<std::size_t>(stage)); }
inline dpn::DPNPaths dpn__DPNOutputs__stages_at(const dpn::DPNOutputs& self, int stage) { return self.stages.at(static_cast<std::size_t>(stage)); }
inline int revnet__RevNetUnit__input_channels(const revnet::RevNetUnit& self) { return self.in; }
inline void revnet__RevNetUnit__set_input_channels(revnet::RevNetUnit& self, int value) { self.in = value; }
inline datamunge::nn::Tensor wide_resnet__WideResNetOutputs__stages_at(const wide_resnet::WideResNetOutputs& self, int index) {
    if (index < 0 || index >= 3) throw std::out_of_range("stages" " index must be in 0.." "3");
    return self.stages[static_cast<std::size_t>(index)];
  }
inline int mobilenet__MobileNetV3Block__input_channels(const mobilenet::MobileNetV3Block& self) { return self.in; }
inline void mobilenet__MobileNetV3Block__set_input_channels(mobilenet::MobileNetV3Block& self, int value) { self.in = value; }
/// Recorded convolutions: count, output shape and multiply-adds per output position of layer i.
inline int shufflenet__ShuffleNetOutputs__layer_count(const shufflenet::ShuffleNetOutputs& self) { return int(self.layers.size()); }
inline std::vector<std::int64_t> shufflenet__ShuffleNetOutputs__layer_shape(const shufflenet::ShuffleNetOutputs& self, int index) { return self.layers.at(static_cast<std::size_t>(index)).first; }
inline std::int64_t shufflenet__ShuffleNetOutputs__layer_cost(const shufflenet::ShuffleNetOutputs& self, int index) { return self.layers.at(static_cast<std::size_t>(index)).second; }
inline int repvgg__RepVGGBlockSpec__input_channels(const repvgg::RepVGGBlockSpec& self) { return self.in; }
inline void repvgg__RepVGGBlockSpec__set_input_channels(repvgg::RepVGGBlockSpec& self, int value) { self.in = value; }
inline void attention_cnn__AttentionResNetConfig__use_selective_kernel(attention_cnn::AttentionResNetConfig& self, int cardinality, int reduction, double momentum = .9, bool update_stats = true,
                            std::int64_t minimum = 32) {
    self.conv2 = attention_cnn::SelectiveKernelHook(cardinality, reduction, momentum, update_stats, minimum);
  }
inline bool attention_cnn__AttentionResNetConfig__has_conv2_hook(const attention_cnn::AttentionResNetConfig& self) { return static_cast<bool>(self.conv2); }
inline bool attention_cnn__AttentionResNetConfig__has_after_block_hook(const attention_cnn::AttentionResNetConfig& self) { return static_cast<bool>(self.after_block); }
inline int hrnet__HRNetConfig__modules_at(const hrnet::HRNetConfig& self, int index) {
    if (index < 0 || index >= 3) throw std::out_of_range("modules" " index must be in 0.." "3");
    return self.modules[static_cast<std::size_t>(index)];
  }
inline void hrnet__HRNetConfig__set_modules_at(hrnet::HRNetConfig& self, int index, int value) {
    if (index < 0 || index >= 3) throw std::out_of_range("modules" " index must be in 0.." "3");
    self.modules[static_cast<std::size_t>(index)] = value;
  }
inline int deeplab__DeepLabV2Config__rates_at(const deeplab::DeepLabV2Config& self, int index) {
    if (index < 0 || index >= 4) throw std::out_of_range("rates" " index must be in 0.." "4");
    return self.rates[static_cast<std::size_t>(index)];
  }
inline void deeplab__DeepLabV2Config__set_rates_at(deeplab::DeepLabV2Config& self, int index, int value) {
    if (index < 0 || index >= 4) throw std::out_of_range("rates" " index must be in 0.." "4");
    self.rates[static_cast<std::size_t>(index)] = value;
  }
inline int deeplab__DeepLabV3Config__rates_at(const deeplab::DeepLabV3Config& self, int index) {
    if (index < 0 || index >= 3) throw std::out_of_range("rates" " index must be in 0.." "3");
    return self.rates[static_cast<std::size_t>(index)];
  }
inline void deeplab__DeepLabV3Config__set_rates_at(deeplab::DeepLabV3Config& self, int index, int value) {
    if (index < 0 || index >= 3) throw std::out_of_range("rates" " index must be in 0.." "3");
    self.rates[static_cast<std::size_t>(index)] = value;
  }
/// Export() as an .npz file.
inline void sal__SaliencyModel__ExportFile(const sal::SaliencyModel& self, const std::string& file) { self.Export(std::filesystem::path(file)); }
/// Assign() from named arrays (nn.load_npz's result).
inline void sal__SaliencyModel__AssignNamed(sal::SaliencyModel& self, const datamunge::nn::NamedTensors& weights) {
    self.Assign(sal::Arrays(weights.begin(), weights.end()));
  }
/// Hard-negative selection mask [N * A] as 0/1 integers.
inline std::vector<int> ssd__SsdLossOutputs__selected_negatives_mask(const ssd::SsdLossOutputs& self) {
    return std::vector<int>(self.selected_negatives.begin(), self.selected_negatives.end());
  }
inline int centernet__CenterNetConfig__upsample_widths_at(const centernet::CenterNetConfig& self, int index) {
    if (index < 0 || index >= 3) throw std::out_of_range("upsample_widths" " index must be in 0.." "3");
    return self.upsample_widths[static_cast<std::size_t>(index)];
  }
inline void centernet__CenterNetConfig__set_upsample_widths_at(centernet::CenterNetConfig& self, int index, int value) {
    if (index < 0 || index >= 3) throw std::out_of_range("upsample_widths" " index must be in 0.." "3");
    self.upsample_widths[static_cast<std::size_t>(index)] = value;
  }
/// Children of `parent` (-1 for the roots); empty when it has none.
inline std::vector<int> wordtree__WordTree__group(const wordtree::WordTree& self, int parent) {
    const auto it = self.groups.find(parent);
    return it == self.groups.end() ? std::vector<int>{} : it->second;
  }
inline std::vector<std::string> yolo__YoloV1PretrainedWeights__parameter_names(const yolo::YoloV1PretrainedWeights& self) {
    std::vector<std::string> names;
    for (const auto& [name, _] : self.parameters) names.push_back(name);
    return names;
  }
inline std::vector<std::string> yolo__YoloV1PretrainedWeights__state_names(const yolo::YoloV1PretrainedWeights& self) {
    std::vector<std::string> names;
    for (const auto& [name, _] : self.state) names.push_back(name);
    return names;
  }
inline nn::Tensor yolo__YoloV1PretrainedWeights__parameter(const yolo::YoloV1PretrainedWeights& self, const std::string& name) { return self.parameters.at(name); }
inline nn::Tensor yolo__YoloV1PretrainedWeights__state_value(const yolo::YoloV1PretrainedWeights& self, const std::string& name) { return self.state.at(name); }
/// Targets of head `level` (0 coarse .. 2 fine).
inline yolo::YoloV3LevelTargets yolo__YoloV3Targets__level(const yolo::YoloV3Targets& self, int index) {
    if (index < 0 || index > 2) throw std::out_of_range("YOLO v3 has three head levels");
    return self.levels[static_cast<std::size_t>(index)];
  }
inline yolo::YoloV5LevelTargets yolo__YoloV5Targets__level(const yolo::YoloV5Targets& self, int index) {
    if (index < 0 || index > 2) throw std::out_of_range("YOLO v5 has three head levels");
    return self.levels[static_cast<std::size_t>(index)];
  }
/// Number of object supports of `scene`.
inline int instance__InstanceScenes__instance_count(const instance::InstanceScenes& self, int scene) { return int(self.instance_masks.at(static_cast<std::size_t>(scene)).size()); }
/// H x W 0/1 support of object `object` of `scene`.
inline instance::BinaryMask instance__InstanceScenes__instance_mask(const instance::InstanceScenes& self, int scene, int object) {
    return self.instance_masks.at(static_cast<std::size_t>(scene)).at(static_cast<std::size_t>(object));
  }
inline std::vector<int> panoptic__PanopticMap__segment_ids(const panoptic::PanopticMap& self) {
    std::vector<int> ids;
    for (const auto& [id, _] : self.segments) ids.push_back(id);
    return ids;
  }
inline panoptic::PanopticSegment panoptic__PanopticMap__segment(const panoptic::PanopticMap& self, int id) { return self.segments.at(id); }
inline bool panoptic__PanopticMap__has_segment(const panoptic::PanopticMap& self, int id) { return self.segments.count(id) != 0; }
/// Mined hard negatives [B * A] as 0/1 integers.
inline std::vector<int> yolact__YolactLoss__negatives_mask(const yolact::YolactLoss& self) { return std::vector<int>(self.negatives.begin(), self.negatives.end()); }
inline std::string detection_evaluation__EvaluationImage__filename_string(const detection_evaluation::EvaluationImage& self) { return self.filename.string(); }
inline std::vector<int> detection_evaluation__DatasetAp__per_class50_labels(const detection_evaluation::DatasetAp& self) {
    std::vector<int> labels;
    for (const auto& [label, _] : self.per_class50) labels.push_back(label);
    return labels;
  }
inline double detection_evaluation__DatasetAp__per_class50_value(const detection_evaluation::DatasetAp& self, int label) { return self.per_class50.at(label); }
/// Decoder output [N, 784] in (0, 1) from the digit capsules masked by `mask` [N, 10].
inline nn::Tensor capsnet__CapsNetOutputs__Reconstruct(const capsnet::CapsNetOutputs& self, const nn::Tensor& mask) { return self.reconstruct(mask); }

}  // namespace nn_python_extend

