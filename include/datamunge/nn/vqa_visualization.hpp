#pragma once
/// @file vqa_visualization.hpp
/// @brief Original human VQA text metadata and tensor consensus/probability preview tiles.
#include "datamunge/nn/image_io.hpp"
#include "datamunge/nn/vqa_data.hpp"
#include <fstream>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Read exact indexed fitting answer strings for presentation.
///
/// Reject missing,repeated,misordered or incomplete label indices. Strings
/// are the preparation normalization policy,not generated answer captions.
///
/// @param root Prepared source root with answer_vocabulary.tsv.
/// @param count Expected number of answer classes including UNK0.
/// @return String labels in class-index order;no numeric transformations or fitting.
/// @throws std::invalid_argument For missing or malformed label metadata.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto split = m::LoadVqa("build/cnn-datasets/vqa");
/// const auto labels = m::VqaAnswers("build/cnn-datasets/vqa", split.config.answers);
/// @endcode
/// @par References
/// - Official VQA-v2 human questions/answers, https://visualqa.org/download.html. Preview border uses mapped-string leave-one-out consensus,not a full official normalized benchmark evaluation.
/// @see VqaText()

inline std::vector<std::string> VqaAnswers(const std::filesystem::path &root, int count) {
  std::ifstream file(root / "answer_vocabulary.tsv");
  if (!file)
    throw std::invalid_argument("VQA answer strings required");
  std::vector<std::string> out;
  std::string line;
  while (std::getline(file, line)) {
    const auto tab = line.find('\t');
    if (tab == std::string::npos || std::stoi(line.substr(0, tab)) != int(out.size()))
      throw std::invalid_argument("indexed VQA answer strings required");
    out.push_back(line.substr(tab + 1));
  }
  if (out.size() != std::size_t(count))
    throw std::invalid_argument("complete VQA answer strings required");
  return out;
}
/// @brief Read original human questions and attach their finite-vocabulary majority label.
///
/// questions.tsv has original question ID,image ID and human question text,
/// without a header. Majority uses loaded first-maximum hard-answer metadata;
/// UNK means answer vocabulary omitted that original majority. String/ID
/// handling only;no numerical fitting or generated question templates.
///
/// @param root Prepared source root with questions.tsv,vqa.npz and answer_vocabulary.tsv.
/// @return Map exact Int64 question ID to original question text and mapped majority string.
/// @throws std::invalid_argument For malformed or duplicate text rows,missing question identity or labels.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto text = m::VqaText("build/cnn-datasets/vqa");
/// @endcode
/// @par References
/// - Official VQA-v2 human questions/answers, https://visualqa.org/download.html. Preview border uses mapped-string leave-one-out consensus,not a full official normalized benchmark evaluation.
/// @see VqaAnswers(), LoadVqa()

inline std::map<std::int64_t, std::pair<std::string, std::string>>
VqaText(const std::filesystem::path &root) {
  const auto split = LoadVqa(root);
  const auto labels = VqaAnswers(root, split.config.answers);
  std::map<std::int64_t, std::string> majority;
  for (const auto &data : {split.fit, split.validation, split.test}) {
    const auto ids = data.question_ids.to_vector<std::int64_t>();
    const auto answers = data.hard.to_vector<std::int32_t>();
    for (std::size_t i = 0; i < ids.size(); ++i)
      majority.emplace(ids[i], labels.at(answers[i]));
  }
  std::ifstream file(root / "questions.tsv");
  if (!file)
    throw std::invalid_argument("original VQA human questions required");
  std::map<std::int64_t, std::pair<std::string, std::string>> out;
  std::string line;
  while (std::getline(file, line)) {
    const auto first = line.find('\t'),
               second = line.find('\t', first == std::string::npos ? 0 : first + 1);
    if (first == std::string::npos || second == std::string::npos)
      throw std::invalid_argument("VQA question-ID,image-ID,text columns required");
    const auto id = std::stoll(line.substr(0, first));
    if (!out.emplace(id, std::pair{line.substr(second + 1), majority.at(id)}).second)
      throw std::invalid_argument("unique original VQA text IDs required");
  }
  if (out.size() != majority.size())
    throw std::invalid_argument("complete VQA human text metadata required");
  return out;
}
/// @brief Select each first-maximum answer class with a tensor argmax.
///
/// Finite scores are required. This helper is independent of target labels.
///
/// @param probability Numeric[N,C] answer probabilities or raw ranking scores,N>0,C>0.
/// @return Int64[N] first-maximum class indices;no threshold or learning.
/// @throws std::invalid_argument For empty geometry or nonfinite values.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto ids = m::VqaAnswerIds(n::tensor({.5, .5, 0.}, {1, 3})); // class0
/// @endcode
/// @see ScoreVqa()

inline nn::Tensor VqaAnswerIds(const nn::Tensor &probability) {
  if (probability.dim() != 2 || probability.size(0) < 1 || probability.size(1) < 1 ||
      !nn::all(nn::is_finite(probability)).item<bool>())
    throw std::invalid_argument("finite nonempty VQA answer scores required");
  return nn::argmax(probability, 1);
}
/// @brief Render original RGB with mapped-consensus borders and full answer probability bars.
///
/// Two-pixel borders are green for score>=.6,yellow for positive lower score,
/// red for0. Compare score in Double to retain the source Float32 boundary
/// policy. Bottom16px contain all C bars,height=round(14*probability),green
/// for first-max answer and blue otherwise. Source Float32 multiplication
/// precedes half-up rounding. Unused columns stay black. RGB is original
/// observed input,not a generated picture;targets only color evaluation borders.
///
/// @param data Aligned Float32[N,64,64,3] original images and[N,C] mapped consensus scores.
/// @param probability Finite unit-row-sum[N,C],2<=C<=64.
/// @return Float32[N,80,64,3] preview tiles in0..1;no host pixel loops.
/// @throws std::invalid_argument For invalid RGB,score/probability geometry,values or row normalization.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// m::VqaData d;
/// d.rgb = n::full({2, 64, 64, 3}, .5);
/// d.score = n::tensor({0., .6, .3, 0., 1., 0.}, {2, 3});
/// const auto p = n::tensor({.1, .8, .1, .8, .1, .1}, {2, 3});
/// const auto tiles = m::VqaPreviewTiles(d, p);
/// @endcode
/// @par References
/// - Official VQA-v2 human questions/answers, https://visualqa.org/download.html. Preview border uses mapped-string leave-one-out consensus,not a full official normalized benchmark evaluation.
/// @see VqaAnswerIds(), ScoreVqa()

inline nn::Tensor VqaPreviewTiles(const VqaData &data, const nn::Tensor &probability) {
  nn::NoGradGuard guard;
  const auto ids = VqaAnswerIds(probability);
  const auto n = probability.size(0), c = probability.size(1);
  if (c < 2 || c > 64 || data.rgb.shape() != nn::Shape{n, 64, 64, 3} ||
      data.score.shape() != probability.shape() ||
      !nn::all(nn::is_finite(data.rgb)).item<bool>() ||
      !nn::all(nn::logical_and(nn::greater_equal(probability, 0.),
                               nn::less_equal(probability, 1.)))
           .item<bool>() ||
      nn::max(nn::abs(nn::sum(probability.to(nn::DType::Float64), {1}) - 1.)).item() > 2e-6)
    throw std::invalid_argument("aligned VQA preview images/scores/probabilities required");
  const auto selected = nn::sum(nn::one_hot(ids, c).to(nn::DType::Float64) *
                                    data.score.to(nn::DType::Float32).to(nn::DType::Float64),
                                {1})
                            .reshape({n, 1, 1, 1}),
             high = nn::greater_equal(selected, .6), positive = nn::greater(selected, 0.),
             red = nn::tensor({1., 0., 0.}), green = nn::tensor({0., 1., 0.}),
             yellow = nn::tensor({1., 1., 0.}),
             color = nn::where(high, green, nn::where(positive, yellow, red)),
             x = nn::arange(0, 64).reshape({1, 1, 64, 1}),
             y = nn::arange(0, 64).reshape({1, 64, 1, 1}),
             border =
                 nn::logical_or(nn::logical_or(nn::less(x, 2.), nn::greater_equal(x, 62.)),
                                nn::logical_or(nn::less(y, 2.), nn::greater_equal(y, 62.))),
             rgb = (nn::floor((data.rgb.to(nn::DType::Float32) * 255.).to(nn::DType::Float64) +
                              .5) /
                    255.)
                       .to(nn::DType::Float32),
             top = nn::where(border, color, nn::clip(rgb, 0., 1.));
  const auto height = nn::floor(
                 (probability.to(nn::DType::Float32) * 14.).to(nn::DType::Float64) + .5),
             padded = nn::pad(height, {{0, 0}, {0, 64 - c}}),
             by = nn::arange(0, 16).reshape({1, 16, 1, 1}),
             visible = nn::greater_equal(by, 16. - padded.reshape({n, 1, 64, 1})),
             winner = nn::equal(x, ids.reshape({n, 1, 1, 1})), blue = nn::tensor({0., 0., 1.}),
             bottom = nn::where(visible, nn::where(winner, green, blue), 0.);
  return nn::concat({top, bottom}, 1).to(nn::DType::Float32);
}
} // namespace datamunge::nn::models
