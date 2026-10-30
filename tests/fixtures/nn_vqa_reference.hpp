// Real VQA-v2 human question inputs, ten-answer targets and official consensus
// scores: a port of tflow's common/ml/vqa_data.h (VqaConfig of
// common/nn/vqa.h lives here because the loader fills it).
// TF -> datamunge: vqa.npz is read with nn::load_npz (float32 RGB/targets/
// scores, int32 tokens/lengths/answers/ids/types/prefixes, int64 question ids).
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <datamunge/nn/nn.hpp>
#include <filesystem>
#include <map>
#include <numeric>
#include <set>
#include <string>
#include <vector>

#include "common/checks.hpp"

namespace vqa_reference {

namespace nn = datamunge::nn;

struct VqaConfig {
  int vocabulary = 342, answers = 64, time = 24, base = 64, embedding = 300, hidden = 256, fusion = 512;
};

struct VqaData {
  nn::Tensor rgb, tokens, lengths, target, score, hard, image_ids, question_ids, types, prefix;
};
struct VqaSplits {
  VqaData   fit, validation, test;
  VqaConfig config;
};

inline nn::Tensor VqaRows(const nn::Tensor& input, const std::vector<std::size_t>& rows) {
  std::vector<std::int64_t> ids;
  for (auto r : rows) {
    check::Require(r < std::size_t(input.size(0)), "VQA row index");
    ids.push_back(std::int64_t(r));
  }
  return nn::take_rows(input, ids).contiguous();
}
inline VqaData VqaRows(const VqaData& d, const std::vector<std::size_t>& rows) {
  return {VqaRows(d.rgb, rows),    VqaRows(d.tokens, rows),       VqaRows(d.lengths, rows),      VqaRows(d.target, rows), VqaRows(d.score, rows),
          VqaRows(d.hard, rows),   VqaRows(d.image_ids, rows),    VqaRows(d.question_ids, rows), VqaRows(d.types, rows),  VqaRows(d.prefix, rows)};
}

inline VqaSplits LoadVqa(const std::filesystem::path& root) {
  std::map<std::string, nn::Tensor> arrays;
  for (const auto& [name, value] : nn::load_npz((root / "vqa.npz").string())) arrays[name] = value;
  const auto& cfg = arrays.at("config");
  check::Require(cfg.dtype() == nn::DType::Int32 && cfg.shape() == nn::Shape({3}), "VQA vocabulary metadata");
  const auto c = cfg.to_vector<std::int32_t>();
  VqaConfig  config;
  config.vocabulary = c[0], config.answers = c[1], config.time = c[2], config.base = 8, config.embedding = 32, config.hidden = 32, config.fusion = 64;
  check::Require(config.vocabulary >= 2 && config.vocabulary <= 512 && config.answers >= 2 && config.answers <= 64 && config.time == 24,
                 "fit-only vocab/time limits");
  std::set<int>          prior_images;
  std::set<std::int64_t> questions;
  auto                   read = [&](const std::string& split) {
    VqaData out{arrays.at(split + "_rgb"),       arrays.at(split + "_tokens"),       arrays.at(split + "_lengths"), arrays.at(split + "_target"),
                arrays.at(split + "_score"),     arrays.at(split + "_hard"),         arrays.at(split + "_image_ids"),
                arrays.at(split + "_question_ids"), arrays.at(split + "_types"),     arrays.at(split + "_prefix")};
    const std::int64_t count = out.rgb.size(0);
    check::Require(count > 0 && out.rgb.dtype() == nn::DType::Float32 && out.rgb.shape() == nn::Shape({count, 64, 64, 3}) &&
                       out.tokens.dtype() == nn::DType::Int32 && out.tokens.shape() == nn::Shape({count, config.time}) &&
                       out.target.dtype() == nn::DType::Float32 && out.target.shape() == nn::Shape({count, config.answers}) &&
                       out.score.dtype() == nn::DType::Float32 && out.score.shape() == out.target.shape(),
                   "VQA image/token/target/score dimensions");
    check::Require(out.question_ids.dtype() == nn::DType::Int64 && out.question_ids.shape() == nn::Shape({count}) &&
                       out.image_ids.dtype() == nn::DType::Int32 && out.image_ids.shape() == nn::Shape({count}) &&
                       out.lengths.dtype() == nn::DType::Int32 && out.lengths.shape() == nn::Shape({count}) &&
                       out.hard.dtype() == nn::DType::Int32 && out.hard.shape() == nn::Shape({count}) && out.types.dtype() == nn::DType::Int32 &&
                       out.types.shape() == nn::Shape({count}) && out.prefix.dtype() == nn::DType::Int32 &&
                       out.prefix.shape() == nn::Shape({count, 2}),
                   "VQA source/question metadata dimensions");
    const auto image_ids = out.image_ids.to_vector<std::int32_t>(), lengths = out.lengths.to_vector<std::int32_t>(),
               hard = out.hard.to_vector<std::int32_t>(), types = out.types.to_vector<std::int32_t>(), tokens = out.tokens.to_vector<std::int32_t>();
    const auto qids = out.question_ids.to_vector<std::int64_t>();
    const auto target = out.target.to_vector<float>(), score = out.score.to_vector<float>();
    std::set<int> images;
    for (std::int64_t n = 0; n < count; ++n) {
      const std::size_t i = std::size_t(n);
      const int         id = image_ids[i], length = lengths[i];
      check::Require(!prior_images.count(id) && questions.insert(qids[i]).second, "no images cross VQA splits or repeated question IDs");
      images.insert(id);
      check::Require(length > 0 && length <= config.time && hard[i] >= 0 && hard[i] < config.answers && types[i] >= 0 && types[i] < 3,
                     "question length/answer/type bounds");
      for (int t = 0; t < config.time; ++t) {
        const int token = tokens[i * std::size_t(config.time) + std::size_t(t)];
        check::Require(token >= 0 && token < config.vocabulary && (t < length ? token > 0 : token == 0), "fit-vocabulary token range and right padding");
      }
      double total = 0;
      for (int a = 0; a < config.answers; ++a) {
        const float tv = target[i * std::size_t(config.answers) + std::size_t(a)], sv = score[i * std::size_t(config.answers) + std::size_t(a)];
        check::Require(std::isfinite(tv) && tv >= 0 && tv <= 1.000001 && std::isfinite(sv) && sv >= 0 && sv <= 1, "ten-answer target/consensus range");
        total += tv;
      }
      check::Near(total, 1, 2e-6, "all10 answer frequencies retained including OOV");
      check::Near(score[i * std::size_t(config.answers)], 0, 0, "UNK answer never counted correct");
    }
    for (float v : out.rgb.to_vector<float>()) check::Require(std::isfinite(v) && v >= 0 && v <= 1, "observed VQA RGB values");
    prior_images.insert(images.begin(), images.end());
    return out;
  };
  auto fit = read("fit"), validation = read("validation"), test = read("test");
  return {fit, validation, test, config};
}

/// Soft ten-answer cross-entropy: -mean_n sum_c freq log softmax(logits).
inline nn::Tensor VqaSoftCE(const nn::Tensor& logits, const nn::Tensor& frequencies) {
  return -nn::mean(nn::sum(frequencies * nn::log_softmax(logits, 1), {1}));
}

/// Official leave-one-annotator-out consensus for `matching` of `annotators` answers.
inline double VqaConsensusCount(int matching, int annotators = 10) {
  check::Require(annotators > 1 && matching >= 0 && matching <= annotators, "annotator consensus count");
  return (matching * std::min(1., std::max(matching - 1, 0) / 3.) + (annotators - matching) * std::min(1., matching / 3.)) / annotators;
}

struct VqaMetrics {
  double             nll = 0, consensus = 0, exact = 0, coverage = 0, oracle = 0;
  std::array<double, 3> type_score{};
  std::array<int, 3>    type_count{};
};

inline VqaMetrics ScoreVqa(const nn::Tensor& probability, const VqaData& data) {
  check::Require(probability.shape() == data.target.shape(), "VQA probability/target shape");
  VqaMetrics  result;
  const int   count = int(probability.size(0)), classes = int(probability.size(1));
  const auto  p = probability.to_vector<float>(), target = data.target.to_vector<float>(), score = data.score.to_vector<float>();
  const auto  hard = data.hard.to_vector<std::int32_t>(), types = data.types.to_vector<std::int32_t>();
  for (int n = 0; n < count; ++n) {
    int    answer = 0;
    double sum = 0, oracle = 0;
    for (int c = 0; c < classes; ++c) {
      const std::size_t j = std::size_t(n * classes + c);
      const double      v = p[j];
      check::Require(std::isfinite(v) && v >= 0 && v <= 1, "valid answer probability");
      sum += v;
      if (v > p[std::size_t(n * classes + answer)]) answer = c;
      result.nll -= target[j] * std::log(std::max(v, 1e-30));
      oracle = std::max(oracle, double(score[j]));
    }
    check::Near(sum, 1, 2e-6, "answer probability normalization");
    const double s = score[std::size_t(n * classes + answer)];
    result.consensus += s;
    result.exact += hard[std::size_t(n)] > 0 && answer == hard[std::size_t(n)];
    result.coverage += hard[std::size_t(n)] > 0;
    result.oracle += oracle;
    const int type = types[std::size_t(n)];
    result.type_score[std::size_t(type)] += s;
    ++result.type_count[std::size_t(type)];
  }
  result.nll /= count, result.consensus /= count, result.exact /= count, result.coverage /= count, result.oracle /= count;
  for (int type = 0; type < 3; ++type)
    if (result.type_count[std::size_t(type)]) result.type_score[std::size_t(type)] /= result.type_count[std::size_t(type)];
  return result;
}

/// Fit-only answer prior (add-one global counts), optionally plus the fit
/// counts of the test question's two-token prefix.
inline nn::Tensor VqaPrior(const VqaData& fitting, const VqaData& testing, bool question_prefix = false) {
  const int           classes = int(fitting.target.size(1));
  std::vector<double> global(std::size_t(classes), 1.);
  std::map<std::pair<int, int>, std::vector<double>> groups;
  const auto fp = fitting.prefix.to_vector<std::int32_t>(), tp = testing.prefix.to_vector<std::int32_t>();
  const auto ft = fitting.target.to_vector<float>();
  for (std::int64_t n = 0; n < fitting.target.size(0); ++n) {
    const auto key = std::make_pair(fp[std::size_t(2 * n)], fp[std::size_t(2 * n + 1)]);
    if (!groups.count(key)) groups[key] = std::vector<double>(std::size_t(classes));
    for (int c = 0; c < classes; ++c) {
      const float value = ft[std::size_t(n * classes + c)];
      global[std::size_t(c)] += value;
      groups[key][std::size_t(c)] += value;
    }
  }
  const double global_sum = std::accumulate(global.begin(), global.end(), 0.);
  for (auto& value : global) value /= global_sum;
  const std::int64_t rows = testing.target.size(0);
  std::vector<float> out(std::size_t(rows * classes));
  for (std::int64_t n = 0; n < rows; ++n) {
    const auto key = std::make_pair(tp[std::size_t(2 * n)], tp[std::size_t(2 * n + 1)]);
    const auto it  = groups.find(key);
    double     sum = 0;
    for (int c = 0; c < classes; ++c) {
      const double value = global[std::size_t(c)] + (question_prefix && it != groups.end() ? it->second[std::size_t(c)] : 0);
      out[std::size_t(n * classes + c)] = float(value);
      sum += value;
    }
    for (int c = 0; c < classes; ++c) out[std::size_t(n * classes + c)] = float(out[std::size_t(n * classes + c)] / sum);
  }
  return nn::from_vector(out, {rows, classes});
}

} // namespace vqa_reference
