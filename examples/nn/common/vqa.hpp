// CNN plus length-masked LSTM VQA with joint and separately trained unimodal
// modes, short training, unimodal baselines and observed-input-only restored
// inference: a port of tflow's common/nn/vqa.h and common/ml/vqa.h.
// TF -> datamunge: parameters are an nn::ParameterStore (seed 277, datamunge's
// generator); the LSTM is nn::lstm_step (same i,f,g,o packing, Glorot ranges
// and forget bias 1 as tflow's LstmStep); the checkpoint is best.npz; the
// deployment model is a second VqaModel built without an optimizer or target
// input (TF inspects GraphDef placeholders); PNGs are written with nn::save_png.
#pragma once

#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>

#include "common/vqa_data.hpp"
#include <datamunge/nn/vqa_models.hpp>
#include <datamunge/nn/vqa_visualization.hpp>

namespace vqa {

using datamunge::nn::models::VqaBaseline;
using datamunge::nn::models::VqaLstm;
using datamunge::nn::models::VqaMode;
using datamunge::nn::models::VqaModeName;
using datamunge::nn::models::VqaQuestion;
using datamunge::nn::models::VqaResult;
class VqaModel {
public:
  nn::ParameterStore parameters{277};
  VqaConfig config;
  VqaMode mode;
  std::unique_ptr<nn::Optimizer> optimizer;
  VqaModel(const VqaConfig &c, VqaMode m = VqaMode::kJoint, bool training = true)
      : config(c), mode(m) {
    {
      nn::NoGradGuard guard;
      VqaBaseline(parameters, nn::zeros({1, 64, 64, 3}),
                  nn::zeros({1, config.time}, nn::DType::Int32),
                  nn::ones({1}, nn::DType::Int32), config, mode);
    }
    if (training) {
      nn::OptimizerConfig opt;
      opt.kind = nn::OptimizerKind::Adam;
      optimizer = std::make_unique<nn::Optimizer>(parameters, .001, opt);
    }
  }
  double Loss(const VqaData &data, bool update = false) {
    check::Require(bool(optimizer), "VQA target loss requires training graph");
    std::unique_ptr<nn::NoGradGuard> guard;
    if (!update)
      guard = std::make_unique<nn::NoGradGuard>();
    const nn::Tensor loss = VqaSoftCE(
        VqaBaseline(parameters, data.rgb, data.tokens, data.lengths, config, mode).logits,
        data.target);
    const double value = loss.item();
    check::Require(std::isfinite(value) && value >= 0, "finite ten-answer CE");
    if (update)
      optimizer->step(nn::grad(loss, parameters.parameters()));
    return value;
  }
  /// Only observed RGB, question token IDs and lengths.
  nn::Tensor Predict(const nn::Tensor &rgb, const nn::Tensor &tokens,
                     const nn::Tensor &lengths) {
    nn::NoGradGuard guard;
    return nn::softmax(VqaBaseline(parameters, rgb, tokens, lengths, config, mode).logits, 1)
        .contiguous();
  }
  nn::Tensor Predict(const VqaData &observed) {
    return Predict(observed.rgb, observed.tokens, observed.lengths);
  }
  std::vector<std::vector<float>> Snapshot() const {
    std::vector<std::vector<float>> out;
    for (const auto &v : parameters.parameters())
      out.push_back(v.to_vector<float>());
    return out;
  }
  void Save(const std::string &path) const {
    nn::NamedTensors arrays;
    for (std::size_t i = 0; i < parameters.size(); ++i)
      arrays.push_back({parameters.names()[i], parameters.parameters()[i]});
    nn::save_npz(path, arrays);
  }
  void Restore(const std::string &path) {
    nn::NoGradGuard guard;
    const auto arrays = nn::load_npz(path);
    check::Require(arrays.size() == parameters.size(), "complete VQA checkpoint");
    for (const auto &[name, value] : arrays)
      parameters.set(name, value);
  }
};

inline nn::Tensor VqaPredict(VqaModel &model, const VqaData &data) {
  std::vector<nn::Tensor> parts;
  const std::int64_t n = data.rgb.size(0);
  for (std::int64_t start = 0; start < n; start += 8) {
    std::vector<std::size_t> rows;
    for (std::int64_t i = start; i < std::min<std::int64_t>(start + 8, n); ++i)
      rows.push_back(std::size_t(i));
    parts.push_back(model.Predict(VqaRows(data, rows)));
  }
  return nn::concat(parts, 0).contiguous();
}

inline void PrintVqa(const std::string &name, const nn::Tensor &probability,
                     const VqaData &data) {
  const auto score = ScoreVqa(probability, data);
  std::cout << " " << name << ":VQA-consensus=" << 100 * score.consensus
            << "%,canonical majority exact=" << 100 * score.exact << "%,softCE=" << score.nll
            << ",known-majority coverage=" << 100 * score.coverage
            << "%,vocabulary oracle=" << 100 * score.oracle << "%,type yes-no/number/other=";
  for (std::size_t type = 0; type < 3; ++type)
    std::cout << 100 * score.type_score[type] << "(" << score.type_count[type] << "),";
  std::cout << '\n';
}

using datamunge::nn::models::VqaAnswers;
using datamunge::nn::models::VqaText;
inline int VqaArgmax(const std::vector<float> &p, int row, int classes) {
  return int(datamunge::nn::models::VqaAnswerIds(
                 nn::from_vector(p, {std::int64_t(p.size() / classes), classes}))
                 .to_vector<std::int64_t>()
                 .at(row));
}
inline void VqaPreview(const VqaData &data, const nn::Tensor &probability,
                       const std::filesystem::path &file) {
  nn::save_image_grid(datamunge::nn::models::VqaPreviewTiles(data, probability), file.string(),
                      4, 0, 1, 0, 0, 1);
}
} // namespace vqa
