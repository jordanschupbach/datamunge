#pragma once
/// @file vqa_data.hpp
/// @brief Tensor VQA-v2 inputs,ten-human-answer objectives and explicit coverage-aware diagnostics.
#include "datamunge/nn/io.hpp"
#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/training.hpp"
#include "datamunge/nn/vqa_models.hpp"
#include <array>
#include <filesystem>
#include <map>
#include <set>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Own genuine observed question/image tensors and aligned original human-answer metadata.
///
/// Each question keeps ten released human answers,not generated templates.
/// Multiple questions share photographs;source images are disjoint across splits.
/// Only rgb,tokens,lengths enter prediction. Tensor handles share storage on copy.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// m::VqaData d;
/// d.target = n::tensor({0., .7, .3}, {1, 3});
/// d.score = n::tensor({0., 1., .9}, {1, 3});
/// d.hard = n::full({1}, 1., n::DType::Int32);
/// d.types = n::zeros({1}, n::DType::Int32);
/// d.prefix = n::tensor({2, 3}, {1, 2}).to(n::DType::Int32);
/// @endcode
/// @par References
/// - Official VQA-v2 downloads, https://visualqa.org/download.html;official leave-one-annotator-out evaluation, https://github.com/GT-Vision-Lab/VQA/blob/master/PythonEvaluationTools/vqaEvaluation/vqaEval.py. Custom85-photograph subset uses released answer strings,lowercase/number-word/article mapping,not the complete official punctuation/contraction normalizer or benchmark score.
/// @see LoadVqa(), VqaRows()
struct VqaData {
  nn::Tensor rgb;     ///< Float32[N,64,64,3] original whole-photo RGB in0..1.
  nn::Tensor tokens;  ///< Int32[N,time] fitting-vocabulary question IDs,right PAD0/UNK1.
  nn::Tensor lengths; ///< Int32[N] positive observed question lengths<=time.
  nn::Tensor target;  ///< Float32[N,answers] ten-human-answer frequencies,including OOV0.
  nn::Tensor score; ///< Float32[N,answers] mapped-string leave-one-out consensus,UNK0 always0.
  nn::Tensor hard;  ///< Int32[N] first-maximum frequency answer,UNK0 is never exact-correct.
  nn::Tensor image_ids; ///< Int32[N] original COCO source image identities.
  nn::Tensor
      question_ids; ///< Int64[N] exact original question identities;never cast to floating point.
  nn::Tensor types; ///< Int32[N] original answer type:0yes/no,1number,2other.
  nn::Tensor
      prefix; ///< Int32[N,2] first two fitting-vocabulary token IDs,metadata for reference.
};
/// @brief Own source-image-disjoint VQA partitions and fitting-only model vocabulary metadata.
///
/// Prepared custom split has254fit/72validation/89test human questions over
/// 55/15/15 source photos. Config selects compact8/32/32/64 learned widths.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto split = m::LoadVqa("build/cnn-datasets/vqa");
/// @endcode
/// @par References
/// - Official VQA-v2 downloads, https://visualqa.org/download.html;official leave-one-annotator-out evaluation, https://github.com/GT-Vision-Lab/VQA/blob/master/PythonEvaluationTools/vqaEvaluation/vqaEval.py. Custom85-photograph subset uses released answer strings,lowercase/number-word/article mapping,not the complete official punctuation/contraction normalizer or benchmark score.
/// @see LoadVqa()
struct VqaSplits {
  VqaData fit; ///< Fitting questions and annotations,used for vocabularies and optimization.
  VqaData validation; ///< Separate photograph pool,used only for checkpoint selection.
  VqaData test;       ///< Held-out photograph pool,never used for fitting.
  VqaConfig config;   ///< Fitting-only vocabulary counts/time and compact architecture widths.
};
/// @brief Gather leading tensor rows without numeric host copies.
///
/// Exact Int64 question IDs retain their integer values. No normalization or fit.
///
/// @param input Nonempty-rank tensor with leading observation dimension.
/// @param rows Bounded row-index metadata;empty rows allowed by tensor gather.
/// @return Requested leading rows,retaining dtype,graph and requested order.
/// @throws std::invalid_argument For missing leading axis or out-of-range row.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto out = m::VqaRows(n::ones({3, 2}), {2, 0});
/// @endcode
/// @see VqaRows(const VqaData&,const std::vector<std::size_t>&)

inline nn::Tensor VqaRows(const nn::Tensor &input, const std::vector<std::size_t> &rows) {
  if (input.dim() < 1)
    throw std::invalid_argument("VQA leading observation axis required");
  std::vector<std::int64_t> ids;
  for (auto row : rows) {
    if (row >= std::size_t(input.size(0)))
      throw std::invalid_argument("bounded VQA row required");
    ids.push_back(std::int64_t(row));
  }
  return nn::take_rows(input, ids).contiguous();
}
/// @brief Gather all observed VQA inputs,targets and provenance with identical row metadata.
///
/// No fitting,target transformation or source-image reassignment.
///
/// @param d Aligned genuine VQA observations and targets.
/// @param rows Bounded leading row indices.
/// @return Aligned requested rows,preserving graph,dtype and exact original IDs.
/// @throws std::invalid_argument For invalid axis or row index.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto split = m::LoadVqa("build/cnn-datasets/vqa");
/// const auto d = m::VqaRows(split.test, {0, 1});
/// @endcode
/// @par References
/// - Official VQA-v2 downloads, https://visualqa.org/download.html;official leave-one-annotator-out evaluation, https://github.com/GT-Vision-Lab/VQA/blob/master/PythonEvaluationTools/vqaEvaluation/vqaEval.py. Custom85-photograph subset uses released answer strings,lowercase/number-word/article mapping,not the complete official punctuation/contraction normalizer or benchmark score.
/// @see LoadVqa()

inline VqaData VqaRows(const VqaData &d, const std::vector<std::size_t> &rows) {
  return {VqaRows(d.rgb, rows),       VqaRows(d.tokens, rows),       VqaRows(d.lengths, rows),
          VqaRows(d.target, rows),    VqaRows(d.score, rows),        VqaRows(d.hard, rows),
          VqaRows(d.image_ids, rows), VqaRows(d.question_ids, rows), VqaRows(d.types, rows),
          VqaRows(d.prefix, rows)};
}
/// @brief Load checked genuine VQA-v2 questions and source-image-disjoint cached tensors.
///
/// Checks exact dtypes/shapes,unit RGB,token lengths/right padding,frequency
/// normalization,score bounds,UNK-zero,majority and prefix agreement,unique
/// Int64 question IDs and no source-image leakage. Numeric validation uses
/// tensors;host sets retain only source IDs. Does not download or fit anything.
/// Preparation scripts retain original archive hashes and selected questions.
///
/// @param root Directory containing vqa.npz and reproducible original-source manifest.
/// @return Three nonempty partitions and compact configuration from fitting-only metadata.
/// @throws std::invalid_argument For malformed arrays,metadata,values or leaked/repeated provenance.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto split = m::LoadVqa("build/cnn-datasets/vqa");
/// @endcode
/// @par References
/// - Official VQA-v2 downloads, https://visualqa.org/download.html;official leave-one-annotator-out evaluation, https://github.com/GT-Vision-Lab/VQA/blob/master/PythonEvaluationTools/vqaEvaluation/vqaEval.py. Custom85-photograph subset uses released answer strings,lowercase/number-word/article mapping,not the complete official punctuation/contraction normalizer or benchmark score.
/// @see VqaRows(), VqaSoftCE(), ScoreVqa()

inline VqaSplits LoadVqa(const std::filesystem::path &root) {
  std::map<std::string, nn::Tensor> arrays;
  for (const auto &[name, value] : nn::load_npz((root / "vqa.npz").string())) {
    if (!arrays.emplace(name, value).second)
      throw std::invalid_argument("duplicate VQA array");
  }
  const auto &cfg = arrays.at("config");
  if (cfg.dtype() != nn::DType::Int32 || cfg.shape() != nn::Shape{3})
    throw std::invalid_argument("Int32 VQA vocabulary metadata required");
  const auto c = cfg.to_vector<std::int32_t>();
  VqaConfig config;
  config.vocabulary = c[0];
  config.answers = c[1];
  config.time = c[2];
  config.base = 8;
  config.embedding = 32;
  config.hidden = 32;
  config.fusion = 64;
  if (c[0] < 2 || c[0] > 512 || c[1] < 2 || c[1] > 64 || c[2] != 24)
    throw std::invalid_argument("fitting-only VQA vocab/time limits required");
  std::set<std::int32_t> previous;
  std::set<std::int64_t> questions;
  /// @brief Load one partition,validating numeric tensor contracts and exact provenance.
  /// @param name Prepared fit,validation or test prefix.
  /// @return Checked aligned observations;updates only metadata identity sets.
  auto read = [&](const std::string &name) {
    VqaData d{arrays.at(name + "_rgb"),       arrays.at(name + "_tokens"),
              arrays.at(name + "_lengths"),   arrays.at(name + "_target"),
              arrays.at(name + "_score"),     arrays.at(name + "_hard"),
              arrays.at(name + "_image_ids"), arrays.at(name + "_question_ids"),
              arrays.at(name + "_types"),     arrays.at(name + "_prefix")};
    if (d.rgb.dim() != 4 || d.rgb.size(0) < 1)
      throw std::invalid_argument("nonempty VQA photographs required");
    const auto n = d.rgb.size(0);
    if (d.rgb.dtype() != nn::DType::Float32 || d.rgb.shape() != nn::Shape{n, 64, 64, 3} ||
        d.target.dtype() != nn::DType::Float32 ||
        d.target.shape() != nn::Shape{n, config.answers} ||
        d.score.dtype() != nn::DType::Float32 || d.score.shape() != d.target.shape() ||
        d.tokens.dtype() != nn::DType::Int32 ||
        d.tokens.shape() != nn::Shape{n, config.time} ||
        d.prefix.dtype() != nn::DType::Int32 || d.prefix.shape() != nn::Shape{n, 2} ||
        d.question_ids.dtype() != nn::DType::Int64 || d.question_ids.shape() != nn::Shape{n})
      throw std::invalid_argument("aligned VQA array geometry/dtype required");
    for (const auto &v : {d.lengths, d.hard, d.image_ids, d.types})
      if (v.dtype() != nn::DType::Int32 || v.shape() != nn::Shape{n})
        throw std::invalid_argument("aligned Int32 VQA metadata required");
    for (const auto &v : {d.rgb, d.target, d.score})
      if (!nn::all(nn::logical_and(nn::is_finite(v), nn::logical_and(nn::greater_equal(v, 0.),
                                                                     nn::less_equal(v, 1.))))
               .item<bool>())
        throw std::invalid_argument("finite unit VQA values required");
    const auto valid_lengths = nn::logical_and(nn::greater(d.lengths, 0.),
                                               nn::less_equal(d.lengths, double(config.time))),
               valid_tokens = nn::logical_and(nn::greater_equal(d.tokens, 0.),
                                              nn::less(d.tokens, double(config.vocabulary))),
               active = nn::less(nn::arange(0, config.time).reshape({1, config.time}),
                                 d.lengths.reshape({n, 1})),
               padding = nn::where(active, nn::greater(d.tokens, 0.), nn::equal(d.tokens, 0.));
    if (!nn::all(valid_lengths).item<bool>() || !nn::all(valid_tokens).item<bool>() ||
        !nn::all(padding).item<bool>() ||
        !nn::all(nn::logical_and(nn::greater_equal(d.types, 0.), nn::less(d.types, 3.)))
             .item<bool>() ||
        nn::max(nn::abs(nn::sum(d.target.to(nn::DType::Float64), {1}) - 1.)).item() > 2e-6 ||
        !nn::all(nn::equal(nn::select(d.score, 1, 0), 0.)).item<bool>() ||
        !nn::all(nn::equal(d.hard, nn::argmax(d.target, 1))).item<bool>() ||
        !nn::all(nn::equal(d.prefix, nn::slice_axis(d.tokens, 1, 0, 2))).item<bool>())
      throw std::invalid_argument("bounded padded questions,normalized human frequencies and "
                                  "UNK-zero scores required");
    std::set<std::int32_t> current;
    for (auto id : d.image_ids.to_vector<std::int32_t>()) {
      if (previous.count(id))
        throw std::invalid_argument("VQA source photograph leakage");
      current.insert(id);
    }
    for (auto id : d.question_ids.to_vector<std::int64_t>())
      if (!questions.insert(id).second)
        throw std::invalid_argument("repeated VQA question identity");
    previous.insert(current.begin(), current.end());
    return d;
  };
  auto fit = read("fit"), validation = read("validation"), test = read("test");
  return {fit, validation, test, config};
}
/// @brief Compute stable ten-human-answer frequency cross-entropy with explicit target mass.
///
/// Max-shift log-softmax retains numerical stability for large finite logits.
/// Each row has equal weight. No argmax,normalization,optimizer or label lookup.
///
/// @param logits Floating[N,C],N>0,C>=2,raw answer logits.
/// @param frequencies Matching finite nonnegative target mass;normalization is caller policy.
/// @return Scalar-mean_n sum_c frequency*(-log_softmax),differentiable in both arguments.
/// @throws std::invalid_argument For misaligned geometry or invalid target mass.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto z = n::tensor({1000., 999., 998.}, {1, 3}).requires_grad_();
/// const auto loss = m::VqaSoftCE(z, n::tensor({.6, .3, .1}, {1, 3}));
/// const auto g = n::grad(loss, {z})[0];
/// @endcode
/// @par References
/// - Official VQA-v2 downloads, https://visualqa.org/download.html;official leave-one-annotator-out evaluation, https://github.com/GT-Vision-Lab/VQA/blob/master/PythonEvaluationTools/vqaEvaluation/vqaEval.py. Custom85-photograph subset uses released answer strings,lowercase/number-word/article mapping,not the complete official punctuation/contraction normalizer or benchmark score.
/// @see ScoreVqa()

inline nn::Tensor VqaSoftCE(const nn::Tensor &logits, const nn::Tensor &frequencies) {
  if (logits.dim() != 2 || logits.size(0) < 1 || logits.size(1) < 2 ||
      frequencies.shape() != logits.shape() ||
      !nn::all(nn::logical_and(nn::is_finite(frequencies), nn::greater_equal(frequencies, 0.)))
           .item<bool>())
    throw std::invalid_argument("aligned nonnegative VQA answer frequencies required");
  return -nn::mean(nn::sum(frequencies * nn::log_softmax(logits, 1), {1}));
}
/// @brief Compute leave-one-annotator-out consensus from a discrete matching-answer count.
///
/// Tensor Double formula gives0,.3,.6,.9,1 for matching0..4 with ten answers.
/// This is not simply min(matching/3,1). String normalization is separate and
/// the teaching preparation does not implement all official punctuation rules.
///
/// @param matching Bounded0..annotators number of equal normalized answer strings.
/// @param annotators Integer>1,default10.
/// @return Double scalar mean of min(other_matching/3,1) over held-out annotators.
/// @throws std::invalid_argument For invalid annotator or matching counts.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto score = m::VqaConsensusCount(2); // .6
/// @endcode
/// @par References
/// - Official VQA-v2 downloads, https://visualqa.org/download.html;official leave-one-annotator-out evaluation, https://github.com/GT-Vision-Lab/VQA/blob/master/PythonEvaluationTools/vqaEvaluation/vqaEval.py. Custom85-photograph subset uses released answer strings,lowercase/number-word/article mapping,not the complete official punctuation/contraction normalizer or benchmark score.
/// @see ScoreVqa()

inline double VqaConsensusCount(int matching, int annotators = 10) {
  if (annotators < 2 || matching < 0 || matching > annotators)
    throw std::invalid_argument("bounded VQA annotator count required");
  const auto m = nn::full({}, double(matching), nn::DType::Float64);
  return ((m * nn::clip((m - 1.) / 3., 0., 1.) +
           (double(annotators) - m) * nn::clip(m / 3., 0., 1.)) /
          double(annotators))
      .item();
}
/// @brief Report equal-question soft likelihood,consensus and finite-vocabulary diagnostic limits.
///
/// Scalar fields serialize tensor reductions. Type count is discrete metadata.
/// Questions sharing a photograph are dependent;metrics are not image-weighted.
/// Mapped-string consensus follows the official count formula,but custom subset
/// and incomplete string normalizer preclude a benchmark evaluation claim.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// m::VqaData d;
/// d.target = n::tensor({0., .7, .3}, {1, 3});
/// d.score = n::tensor({0., 1., .9}, {1, 3});
/// d.hard = n::full({1}, 1., n::DType::Int32);
/// d.types = n::zeros({1}, n::DType::Int32);
/// d.prefix = n::tensor({2, 3}, {1, 2}).to(n::DType::Int32);
/// const auto score = m::ScoreVqa(n::tensor({.1, .8, .1}, {1, 3}), d);
/// @endcode
/// @par References
/// - Official VQA-v2 downloads, https://visualqa.org/download.html;official leave-one-annotator-out evaluation, https://github.com/GT-Vision-Lab/VQA/blob/master/PythonEvaluationTools/vqaEvaluation/vqaEval.py. Custom85-photograph subset uses released answer strings,lowercase/number-word/article mapping,not the complete official punctuation/contraction normalizer or benchmark score.
/// @see ScoreVqa()
struct VqaMetrics {
  double nll = 0; ///< Mean ten-answer negative log likelihood with probability floor1e-30.
  double consensus =
      0;            ///< Mean mapped-string leave-one-out score at first-max predicted answer.
  double exact = 0; ///< Fraction predicted==first-max target,excluding UNK-majority matches.
  double coverage = 0; ///< Fraction whose majority target is in known answer vocabulary.
  double oracle = 0; ///< Mean maximum attainable mapped-string score within answer vocabulary.
  std::array<double, 3> type_score{}; ///< Per-type equal-question score,0 for empty type.
  std::array<int, 3> type_count{}; ///< Counts for yes/no,number,other original answer types.
};
/// @brief Score finite-answer probabilities using tensor reductions and observed human consensus.
///
/// First-class argmax resolves ties. UNK-majority is excluded from exact accuracy.
/// UNK consensus must already be0 in data;loader enforces it. Double reductions
/// preserve Float32 ingress;NLL floors probabilities at1e-30. Empty type score0.
/// No host numerical answer loops,learning or mutation.
///
/// @param probability Finite nonnegative unit-row-sum[N,C] answer distribution.
/// @param data Matching targets,scores,hard answers and original type metadata.
/// @return Equal-question NLL,consensus,exact,coverage,oracle and three type summaries.
/// @throws std::invalid_argument For invalid geometry,probabilities,type indices or row normalization.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// m::VqaData d;
/// d.target = n::tensor({0., .7, .3}, {1, 3});
/// d.score = n::tensor({0., 1., .9}, {1, 3});
/// d.hard = n::full({1}, 1., n::DType::Int32);
/// d.types = n::zeros({1}, n::DType::Int32);
/// d.prefix = n::tensor({2, 3}, {1, 2}).to(n::DType::Int32);
/// const auto score = m::ScoreVqa(n::tensor({.1, .8, .1}, {1, 3}), d);
/// @endcode
/// @par References
/// - Official VQA-v2 downloads, https://visualqa.org/download.html;official leave-one-annotator-out evaluation, https://github.com/GT-Vision-Lab/VQA/blob/master/PythonEvaluationTools/vqaEvaluation/vqaEval.py. Custom85-photograph subset uses released answer strings,lowercase/number-word/article mapping,not the complete official punctuation/contraction normalizer or benchmark score.
/// @see VqaSoftCE(), VqaConsensusCount()

inline VqaMetrics ScoreVqa(const nn::Tensor &probability, const VqaData &data) {
  nn::NoGradGuard guard;
  if (probability.dim() != 2 || probability.size(0) < 1 ||
      probability.shape() != data.target.shape() ||
      data.score.shape() != data.target.shape() ||
      data.hard.shape() != nn::Shape{probability.size(0)} ||
      data.types.shape() != data.hard.shape())
    throw std::invalid_argument("aligned VQA scoring tensors required");
  const auto p = probability.to(nn::DType::Float32).to(nn::DType::Float64);
  if (!nn::all(nn::logical_and(nn::is_finite(p), nn::logical_and(nn::greater_equal(p, 0.),
                                                                 nn::less_equal(p, 1.))))
           .item<bool>() ||
      nn::max(nn::abs(nn::sum(p, {1}) - 1.)).item() > 2e-6 ||
      !nn::all(nn::logical_and(nn::greater_equal(data.types, 0.), nn::less(data.types, 3.)))
           .item<bool>())
    throw std::invalid_argument("normalized VQA probability and bounded types required");
  const auto predicted = nn::argmax(p, 1),
             select = nn::one_hot(predicted, p.size(1), 1.,0., nn::DType::Float64),
             scores = nn::sum(
                 select * data.score.to(nn::DType::Float32).to(nn::DType::Float64), {1}),
             known = nn::greater(data.hard, 0.),
             types = nn::one_hot(data.types, 3,1.,0., nn::DType::Float64),
             counts = nn::sum(types, {0}),
             type_scores = nn::sum(types * nn::expand_dims(scores, 1), {0}) /
                           nn::maximum(counts, nn::ones_like(counts));
  VqaMetrics result;
  result.nll =
      -nn::mean(nn::sum(data.target.to(nn::DType::Float32).to(nn::DType::Float64) *
                            nn::log(nn::maximum(p, nn::full({}, 1e-30, nn::DType::Float64))),
                        {1}))
           .item();
  result.consensus = nn::mean(scores).item();
  result.exact =
      nn::mean(nn::logical_and(known, nn::equal(predicted, data.hard)).to(nn::DType::Float64))
          .item();
  result.coverage = nn::mean(known.to(nn::DType::Float64)).item();
  result.oracle =
      nn::mean(nn::max(data.score.to(nn::DType::Float32).to(nn::DType::Float64), {1})).item();
  const auto ts = type_scores.to_vector<double>();
  const auto tc = counts.to(nn::DType::Int32).to_vector<std::int32_t>();
  for (int i = 0; i < 3; ++i) {
    result.type_score[i] = ts[i];
    result.type_count[i] = tc[i];
  }
  return result;
}
/// @brief Compute the source smoothed-count prior only as an explicit-optimization parity oracle.
///
/// All frequency/count/group/normalization arithmetic uses tensors. Global
/// add-one counts normalize in Double. Matching prefixes add fitting-frequency
/// counts,then source numerator rounds Float32 before final Double division.
/// Reports explicitly fit learned count fields with SGD and use this helper
/// only to verify parity;this is not a hidden fit API.
///
/// @param fitting Fitting-only frequency targets[N,C] and Int32[N,2] token prefixes.
/// @param testing Observed token prefixes[M,2];target is read only for M and C shape.
/// @param question_prefix Add matching fitting-prefix counts to global distribution,default false.
/// @return Float32[M,C] source-rounded smoothed answer probabilities.
/// @throws std::invalid_argument For incompatible target/prefix geometry.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// m::VqaData d;
/// d.target = n::tensor({0., .7, .3}, {1, 3});
/// d.score = n::tensor({0., 1., .9}, {1, 3});
/// d.hard = n::full({1}, 1., n::DType::Int32);
/// d.types = n::zeros({1}, n::DType::Int32);
/// d.prefix = n::tensor({2, 3}, {1, 2}).to(n::DType::Int32);
/// const auto p = m::VqaPrior(d, d, true);
/// @endcode
/// @par References
/// - Official VQA-v2 downloads, https://visualqa.org/download.html;official leave-one-annotator-out evaluation, https://github.com/GT-Vision-Lab/VQA/blob/master/PythonEvaluationTools/vqaEvaluation/vqaEval.py. Custom85-photograph subset uses released answer strings,lowercase/number-word/article mapping,not the complete official punctuation/contraction normalizer or benchmark score.
/// @see VqaSoftCE(), ScoreVqa()

inline nn::Tensor VqaPrior(const VqaData &fitting, const VqaData &testing,
                           bool question_prefix = false) {
  nn::NoGradGuard guard;
  if (fitting.target.dim() != 2 || fitting.target.size(0) < 1 || testing.target.dim() != 2 ||
      testing.target.size(0) < 1 || testing.target.size(1) != fitting.target.size(1) ||
      fitting.prefix.shape() != nn::Shape{fitting.target.size(0), 2} ||
      testing.prefix.shape() != nn::Shape{testing.target.size(0), 2})
    throw std::invalid_argument("aligned VQA reference targets and prefixes required");
  const auto counts = nn::sum(fitting.target.to(nn::DType::Float32).to(nn::DType::Float64),
                              {0}, true) +
                      1.,
             global = counts / nn::sum(counts, {1}, true);
  const auto matches = nn::all(nn::equal(nn::expand_dims(testing.prefix, 1),
                                         nn::expand_dims(fitting.prefix, 0)),
                               {2})
                           .to(nn::DType::Float64),
             numerator =
                 global +
                 (question_prefix
                      ? nn::matmul(
                            matches,
                            fitting.target.to(nn::DType::Float32).to(nn::DType::Float64))
                      : nn::zeros({testing.target.size(0), fitting.target.size(1)},
                                  nn::DType::Float64));
  return (numerator.to(nn::DType::Float32).to(nn::DType::Float64) /
          nn::sum(numerator, {1}, true))
      .to(nn::DType::Float32);
}
} // namespace datamunge::nn::models
