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

namespace vqa {

enum class VqaMode { kJoint, kImage, kQuestion };

struct VqaQuestion {
  nn::Tensor              h, c;
  std::vector<nn::Tensor> steps;
};

/// Embedding + LSTM over `config.time` steps; a row's state freezes after its length.
inline VqaQuestion VqaLstm(nn::ParameterStore& p, const nn::Tensor& tokens, const nn::Tensor& lengths, const VqaConfig& config) {
  const nn::Tensor embedded = nn::embedding(p, "question/embedding", tokens, config.vocabulary, config.embedding);
  const auto       inputs   = nn::time_steps(embedded);
  nn::LstmState    state    = nn::lstm_zero_state(tokens.size(0), config.hidden, embedded.dtype());
  VqaQuestion      result;
  for (int time = 0; time < config.time; ++time) {
    const auto       proposed = nn::lstm_step(p, "question/lstm", inputs[std::size_t(time)], state, 1.0);
    const nn::Tensor active   = nn::expand_dims(nn::greater(lengths, double(time)), 1);
    state                     = {nn::where(active, proposed.h, state.h), nn::where(active, proposed.c, state.c)};
    result.steps.push_back(state.h);
  }
  result.h = state.h, result.c = state.c;
  return result;
}

struct VqaResult {
  nn::Tensor              logits, image_code, question_code, question_h;
  std::vector<nn::Tensor> stages, question_steps;
};

inline VqaResult VqaBaseline(nn::ParameterStore& p, const nn::Tensor& rgb, const nn::Tensor& tokens, const nn::Tensor& lengths,
                             const VqaConfig& config = {}, VqaMode mode = VqaMode::kJoint) {
  if (config.vocabulary < 2 || config.answers < 2 || config.time < 1 || config.base < 1 || config.embedding < 1 || config.hidden < 1 ||
      config.fusion < 1)
    throw std::invalid_argument("positive VQA dimensions");
  VqaResult result;
  if (mode != VqaMode::kQuestion) {
    nn::Tensor           image = rgb - .5;
    nn::ConvLayerOptions conv;
    conv.init       = nn::init::he_normal();
    conv.activation = nn::Activation::Relu;
    for (int block = 0; block < 3; ++block) {
      const int width = config.base * (1 << block);
      for (int layer = 0; layer < 2; ++layer) {
        image = nn::conv2d(p, "image/block" + std::to_string(block) + "/conv" + std::to_string(layer), image, width, 3, conv);
        result.stages.push_back(image);
      }
      image = nn::max_pool2d(image, 2, 2);
      result.stages.push_back(image);
    }
    result.image_code = nn::dense(p, "image/project", nn::global_avg_pool2d(image), config.fusion, nn::Activation::Tanh);
    result.stages.push_back(result.image_code);
  }
  if (mode != VqaMode::kImage) {
    const auto question   = VqaLstm(p, tokens, lengths, config);
    result.question_h     = question.h;
    result.question_steps = question.steps;
    result.question_code  = nn::dense(p, "question/project", question.h, config.fusion, nn::Activation::Tanh);
    result.stages.push_back(result.question_code);
  }
  nn::Tensor fused;
  if (mode == VqaMode::kJoint)
    fused = nn::concat({result.image_code, result.question_code, result.image_code * result.question_code}, 1);
  else
    fused = mode == VqaMode::kImage ? result.image_code : result.question_code;
  result.stages.push_back(fused);
  const nn::Tensor hidden = nn::dense(p, "fusion", fused, config.fusion, nn::Activation::Tanh);
  result.stages.push_back(hidden);
  result.logits = nn::dense(p, "answer", hidden, config.answers);
  result.stages.push_back(result.logits);
  return result;
}

inline std::string VqaModeName(VqaMode mode) {
  return mode == VqaMode::kJoint ? "joint" : mode == VqaMode::kImage ? "image_only" : "question_only";
}

class VqaModel {
 public:
  nn::ParameterStore             parameters{277};
  VqaConfig                      config;
  VqaMode                        mode;
  std::unique_ptr<nn::Optimizer> optimizer;
  VqaModel(const VqaConfig& c, VqaMode m = VqaMode::kJoint, bool training = true) : config(c), mode(m) {
    {
      nn::NoGradGuard guard;
      VqaBaseline(parameters, nn::zeros({1, 64, 64, 3}), nn::zeros({1, config.time}, nn::DType::Int32), nn::ones({1}, nn::DType::Int32), config,
                  mode);
    }
    if (training) {
      nn::OptimizerConfig opt;
      opt.kind  = nn::OptimizerKind::Adam;
      optimizer = std::make_unique<nn::Optimizer>(parameters, .001, opt);
    }
  }
  double Loss(const VqaData& data, bool update = false) {
    check::Require(bool(optimizer), "VQA target loss requires training graph");
    std::unique_ptr<nn::NoGradGuard> guard;
    if (!update) guard = std::make_unique<nn::NoGradGuard>();
    const nn::Tensor loss  = VqaSoftCE(VqaBaseline(parameters, data.rgb, data.tokens, data.lengths, config, mode).logits, data.target);
    const double     value = loss.item();
    check::Require(std::isfinite(value) && value >= 0, "finite ten-answer CE");
    if (update) optimizer->step(nn::grad(loss, parameters.parameters()));
    return value;
  }
  /// Only observed RGB, question token IDs and lengths.
  nn::Tensor Predict(const nn::Tensor& rgb, const nn::Tensor& tokens, const nn::Tensor& lengths) {
    nn::NoGradGuard guard;
    return nn::softmax(VqaBaseline(parameters, rgb, tokens, lengths, config, mode).logits, 1).contiguous();
  }
  nn::Tensor Predict(const VqaData& observed) { return Predict(observed.rgb, observed.tokens, observed.lengths); }
  std::vector<std::vector<float>> Snapshot() const {
    std::vector<std::vector<float>> out;
    for (const auto& v : parameters.parameters()) out.push_back(v.to_vector<float>());
    return out;
  }
  void Save(const std::string& path) const {
    nn::NamedTensors arrays;
    for (std::size_t i = 0; i < parameters.size(); ++i) arrays.push_back({parameters.names()[i], parameters.parameters()[i]});
    nn::save_npz(path, arrays);
  }
  void Restore(const std::string& path) {
    nn::NoGradGuard guard;
    const auto      arrays = nn::load_npz(path);
    check::Require(arrays.size() == parameters.size(), "complete VQA checkpoint");
    for (const auto& [name, value] : arrays) parameters.set(name, value);
  }
};

inline nn::Tensor VqaPredict(VqaModel& model, const VqaData& data) {
  std::vector<nn::Tensor> parts;
  const std::int64_t      n = data.rgb.size(0);
  for (std::int64_t start = 0; start < n; start += 8) {
    std::vector<std::size_t> rows;
    for (std::int64_t i = start; i < std::min<std::int64_t>(start + 8, n); ++i) rows.push_back(std::size_t(i));
    parts.push_back(model.Predict(VqaRows(data, rows)));
  }
  return nn::concat(parts, 0).contiguous();
}

inline void PrintVqa(const std::string& name, const nn::Tensor& probability, const VqaData& data) {
  const auto score = ScoreVqa(probability, data);
  std::cout << " " << name << ":VQA-consensus=" << 100 * score.consensus << "%,canonical majority exact=" << 100 * score.exact
            << "%,softCE=" << score.nll << ",known-majority coverage=" << 100 * score.coverage << "%,vocabulary oracle=" << 100 * score.oracle
            << "%,type yes-no/number/other=";
  for (std::size_t type = 0; type < 3; ++type) std::cout << 100 * score.type_score[type] << "(" << score.type_count[type] << "),";
  std::cout << '\n';
}

inline std::vector<std::string> VqaAnswers(const std::filesystem::path& root, int count) {
  std::ifstream file(root / "answer_vocabulary.tsv");
  check::Require(bool(file), "answer text vocabulary");
  std::vector<std::string> result;
  std::string              line;
  while (std::getline(file, line)) {
    const auto tab = line.find('\t');
    check::Require(tab != std::string::npos && std::stoi(line.substr(0, tab)) == int(result.size()), "answer text indices");
    result.push_back(line.substr(tab + 1));
  }
  check::Require(result.size() == std::size_t(count), "complete answer strings");
  return result;
}

inline std::map<std::int64_t, std::pair<std::string, std::string>> VqaText(const std::filesystem::path& root) {
  std::ifstream file(root / "questions.tsv");
  check::Require(bool(file), "human question text file");
  std::string line;
  std::getline(file, line);
  std::map<std::int64_t, std::pair<std::string, std::string>> out;
  while (std::getline(file, line)) {
    std::vector<std::string> columns;
    std::size_t              begin = 0;
    for (int i = 0; i < 4; ++i) {
      const auto end = line.find('\t', begin);
      check::Require(end != std::string::npos, "question TSV columns");
      columns.push_back(line.substr(begin, end - begin));
      begin = end + 1;
    }
    columns.push_back(line.substr(begin));
    out[std::stoll(columns[0])] = {columns[3], columns[4]};
  }
  return out;
}

inline int VqaArgmax(const std::vector<float>& p, int n, int classes) {
  int answer = 0;
  for (int c = 1; c < classes; ++c)
    if (p[std::size_t(n * classes + c)] > p[std::size_t(n * classes + answer)]) answer = c;
  return answer;
}

/// Every held-out image with a green/yellow/red consensus border and the answer distribution bars.
inline void VqaPreview(const VqaData& data, const nn::Tensor& probability, const std::filesystem::path& path) {
  const int  count = int(data.rgb.size(0)), classes = int(probability.size(1)), rows = (count + 3) / 4;
  const auto p = probability.to_vector<float>(), score = data.score.to_vector<float>(), rgb = data.rgb.to_vector<float>();
  std::vector<float> image(std::size_t(rows * 80 * 256 * 3), 0.f);
  auto at = [&](int y, int x, int c) -> float& { return image[(std::size_t(y) * 256 + std::size_t(x)) * 3 + std::size_t(c)]; };
  for (int n = 0; n < count; ++n) {
    const int   row = n / 4, column = n % 4, answer = VqaArgmax(p, n, classes);
    const float s = score[std::size_t(n * classes + answer)];
    for (int y = 0; y < 64; ++y)
      for (int x = 0; x < 64; ++x)
        for (int c = 0; c < 3; ++c) {
          const bool border = x < 2 || y < 2 || x >= 62 || y >= 62;
          const bool color  = s >= .6 ? c == 1 : s > 0 ? c < 2 : c == 0;
          at(row * 80 + y, column * 64 + x, c) =
              border ? (color ? 255.f : 0.f) : float(std::clamp(int(std::lround(rgb[std::size_t(((n * 64 + y) * 64 + x) * 3 + c)] * 255)), 0, 255));
        }
    for (int c = 0; c < std::min(classes, 64); ++c) {
      const int height = int(std::lround(14 * p[std::size_t(n * classes + c)]));
      for (int y = 0; y < height; ++y) at(row * 80 + 79 - y, column * 64 + c, c == answer ? 1 : 2) = 255;
    }
  }
  nn::save_png(nn::from_vector(image, {rows * 80, 256, 3}), path.string(), 0, 255);
}

}  // namespace vqa
