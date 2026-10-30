#pragma once
/// @file ocr_data.hpp
/// @brief Tensor human-word crop ingress,padded CTC metadata,decoding and edit metrics.
#include "datamunge/image/read_image.hpp"
#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/checkpoint.hpp"
#include "datamunge/nn/image_io.hpp"
#include "datamunge/nn/image_ops.hpp"
#include "datamunge/nn/sequence.hpp"
#include "datamunge/nn/training.hpp"
#include <array>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Ordered ASCII class metadata for OCR nonblank character classes0..35.
inline const std::string kAlphabet = "0123456789abcdefghijklmnopqrstuvwxyz";
/// @brief Default CTC blank class after all36 character classes.
inline constexpr int kBlank = 36;
/// @brief Time steps produced by the32x100 CRNN convolution geometry.
inline constexpr int kSteps = 26;
/// @brief Number of CTC logits,including the last blank class.
inline constexpr int kClasses = 37;
/// @brief Merge repeated class IDs before removing CTC blanks.
///
/// Postprocessing operates only on class metadata. A blank separates repeated
/// characters;removing blanks before merging would change the result.
///
/// @param path Nonnegative character/blank class metadata in time order.
/// @param blank Last class ID,default36;may select a shorter alphabet prefix.
/// @return Collapsed ASCII character string.
/// @throws std::invalid_argument For invalid shape,dtype,range,metadata or missing source data.
/// @par Example
/// @code
/// const auto text = dm::nn::models::CtcCollapse({1, 1, 36, 1, 36, 1}); // "111"
/// @endcode
/// @par References
/// - Graves et al.(2006),Connectionist Temporal Classification:Labelling Unsegmented Sequence Data with Recurrent Neural Networks, https://www.cs.toronto.edu/~graves/icml_2006.pdf. CTC repeat/blank order and normalized edit-rate definitions.
/// - Veit et al.(2016),COCO-Text:Dataset and Benchmark for Text Detection and Recognition in Natural Images, https://arxiv.org/abs/1601.07140;V2 human annotation source https://github.com/bgshih/cocotext/releases/download/dl/cocotext.v2.zip. Custom source-image-disjoint43/7/6 word subset and grayscale32x100 stretch are teaching preparation,not official benchmark evaluation.
/// @see LoadOcr(), Crnn(), ScoreTensorOcr()
inline std::string CtcCollapse(const std::vector<std::int64_t> &path, int blank = kBlank) {
  if (blank < 0 || blank > kBlank)
    throw std::invalid_argument("OCR blank outside alphabet prefix");
  for (auto c : path)
    if (c < 0 || c > blank)
      throw std::invalid_argument("invalid CTC path class");
  std::string out;
  for (auto c : nn::ctc_collapse(path, blank))
    out += kAlphabet[static_cast<std::size_t>(c)];
  return out;
}
/// @brief Compute unit-cost Levenshtein distance through an Int64 tensor dynamic program.
///
/// Host loops compare character-class metadata only;all DP costs/minima use
/// Int64 tensors. O(|a||b|) time,O(|b|) row storage. Bytes are compared,not
/// Unicode graphemes. Empty inputs have the expected length distance.
///
/// @param a First ASCII/byte string,possibly empty.
/// @param b Second ASCII/byte string,possibly empty.
/// @return Integer insertion/deletion/substitution distance.
/// @throws std::invalid_argument For invalid shape,dtype,range,metadata or missing source data.
/// @par Example
/// @code
/// const int distance = dm::nn::models::OcrDistance("coffee", "cofee"); // 1
/// @endcode
/// @par References
/// - Graves et al.(2006),Connectionist Temporal Classification:Labelling Unsegmented Sequence Data with Recurrent Neural Networks, https://www.cs.toronto.edu/~graves/icml_2006.pdf. CTC repeat/blank order and normalized edit-rate definitions.
/// - Wagner and Fischer(1974),The String-to-String Correction Problem,JACM21(1),168-173, https://doi.org/10.1145/321796.321811. Unit-cost insertion/deletion/substitution DP.
/// - Veit et al.(2016),COCO-Text:Dataset and Benchmark for Text Detection and Recognition in Natural Images, https://arxiv.org/abs/1601.07140;V2 human annotation source https://github.com/bgshih/cocotext/releases/download/dl/cocotext.v2.zip. Custom source-image-disjoint43/7/6 word subset and grayscale32x100 stretch are teaching preparation,not official benchmark evaluation.
/// @see LoadOcr(), Crnn(), ScoreTensorOcr()
inline int OcrDistance(const std::string &a, const std::string &b) {
  if (a.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      b.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::invalid_argument("edit-distance string too long");
  auto row = nn::arange(0, static_cast<double>(b.size() + 1), 1, nn::DType::Int64);
  for (std::size_t i = 0; i < a.size(); ++i) {
    std::vector<nn::Tensor> next;
    next.push_back(nn::full({}, static_cast<std::int64_t>(i + 1), nn::DType::Int64));
    for (std::size_t j = 0; j < b.size(); ++j) {
      const auto deletion = nn::select(row, 0, j + 1) + 1, insertion = next.back() + 1,
                 substitution = nn::select(row, 0, j) + int(a[i] != b[j]);
      next.push_back(nn::minimum(nn::minimum(deletion, insertion), substitution));
    }
    row = nn::stack(next);
  }
  return nn::select(row, 0, b.size()).item<int>();
}
/// @brief Own grayscale word observations,padded character targets and source IDs.
///
/// Tensor handles own/share their storage. Source IDs are provenance metadata,
/// never network inputs. Prepared files are validated by LoadOcr.
///
/// @throws std::invalid_argument For invalid shape,dtype,range,metadata or missing source data.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const n::models::OcrData data{
///     n::zeros({1, 32, 100, 1}),
///     n::concat({n::zeros({1, 1}, n::DType::Int32), n::full({1, 11}, -1, n::DType::Int32)}, 1),
///     n::ones({1}, n::DType::Int32), n::zeros({1}, n::DType::Int32),
///     n::ones({1}, n::DType::Int32)};
/// @endcode
/// @par References
/// - Graves et al.(2006),Connectionist Temporal Classification:Labelling Unsegmented Sequence Data with Recurrent Neural Networks, https://www.cs.toronto.edu/~graves/icml_2006.pdf. CTC repeat/blank order and normalized edit-rate definitions.
/// - Veit et al.(2016),COCO-Text:Dataset and Benchmark for Text Detection and Recognition in Natural Images, https://arxiv.org/abs/1601.07140;V2 human annotation source https://github.com/bgshih/cocotext/releases/download/dl/cocotext.v2.zip. Custom source-image-disjoint43/7/6 word subset and grayscale32x100 stretch are teaching preparation,not official benchmark evaluation.
/// @see LoadOcr(), Crnn(), ScoreTensorOcr()
struct OcrData {
  nn::Tensor pixels;         ///< Float32[N,32,100,1] observed grayscale word crops in[0,1].
  nn::Tensor labels;         ///< Int32[N,12] character classes0..35,padded with-1.
  nn::Tensor lengths;        ///< Int32[N] true target lengths1..12.
  nn::Tensor image_ids;      ///< Int32[N] original source photograph IDs.
  nn::Tensor annotation_ids; ///< Int32[N] unique human word annotation IDs.
};
/// @brief Gather word images,labels,lengths and provenance in the same order.
///
/// Host validation reads row metadata only;image/target gathering uses tensors.
///
/// @param d Aligned prepared word tensors.
/// @param rows Bounded source rows,duplicates/empty allowed.
/// @return Aligned gathered word tensors,observation graph preserved.
/// @throws std::invalid_argument For invalid shape,dtype,range,metadata or missing source data.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const n::models::OcrData d{
///     n::zeros({2, 32, 100, 1}),
///     n::concat({n::zeros({2, 1}, n::DType::Int32), n::full({2, 11}, -1, n::DType::Int32)}, 1),
///     n::ones({2}, n::DType::Int32), n::zeros({2}, n::DType::Int32),
///     n::ones({2}, n::DType::Int32)};
/// const auto batch = n::models::OcrRows(d, {1, 0});
/// @endcode
/// @par References
/// - Graves et al.(2006),Connectionist Temporal Classification:Labelling Unsegmented Sequence Data with Recurrent Neural Networks, https://www.cs.toronto.edu/~graves/icml_2006.pdf. CTC repeat/blank order and normalized edit-rate definitions.
/// - Veit et al.(2016),COCO-Text:Dataset and Benchmark for Text Detection and Recognition in Natural Images, https://arxiv.org/abs/1601.07140;V2 human annotation source https://github.com/bgshih/cocotext/releases/download/dl/cocotext.v2.zip. Custom source-image-disjoint43/7/6 word subset and grayscale32x100 stretch are teaching preparation,not official benchmark evaluation.
/// @see LoadOcr(), Crnn(), ScoreTensorOcr()
inline OcrData OcrRows(const OcrData &d, const std::vector<std::int64_t> &rows) {
  for (auto r : rows)
    if (r < 0 || r >= d.pixels.size(0))
      throw std::invalid_argument("OCR row bounds");
  return {nn::take_rows(d.pixels, rows), nn::take_rows(d.labels, rows),
          nn::take_rows(d.lengths, rows), nn::take_rows(d.image_ids, rows),
          nn::take_rows(d.annotation_ids, rows)};
}
/// @brief Serialize one padded target row as an ASCII word.
///
/// Class metadata conversion only;no pixel computation or fitting.
///
/// @param d Prepared12-column targets and aligned length vector.
/// @param n Bounded target row.
/// @return Nonempty true target string without-1 padding.
/// @throws std::invalid_argument For invalid shape,dtype,range,metadata or missing source data.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const n::models::OcrData d{
///     n::zeros({1, 32, 100, 1}),
///     n::tensor({10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, {1, 12}, n::DType::Int32),
///     n::ones({1}, n::DType::Int32), n::zeros({1}, n::DType::Int32),
///     n::ones({1}, n::DType::Int32)};
/// const auto word = n::models::OcrText(d, 0); // "a"
/// @endcode
/// @par References
/// - Graves et al.(2006),Connectionist Temporal Classification:Labelling Unsegmented Sequence Data with Recurrent Neural Networks, https://www.cs.toronto.edu/~graves/icml_2006.pdf. CTC repeat/blank order and normalized edit-rate definitions.
/// - Veit et al.(2016),COCO-Text:Dataset and Benchmark for Text Detection and Recognition in Natural Images, https://arxiv.org/abs/1601.07140;V2 human annotation source https://github.com/bgshih/cocotext/releases/download/dl/cocotext.v2.zip. Custom source-image-disjoint43/7/6 word subset and grayscale32x100 stretch are teaching preparation,not official benchmark evaluation.
/// @see LoadOcr(), Crnn(), ScoreTensorOcr()
inline std::string OcrText(const OcrData &d, int n) {
  if (n < 0 || n >= d.labels.size(0) || d.labels.dim() != 2 || d.labels.size(1) != 12 ||
      d.lengths.shape() != nn::Shape{d.labels.size(0)})
    throw std::invalid_argument("OCR target row geometry");
  const auto length = nn::select(d.lengths, 0, n).item<int>();
  if (length < 1 || length > 12)
    throw std::invalid_argument("OCR target length");
  const auto ids = nn::select(d.labels, 0, n).to_vector<int>();
  std::string text;
  for (int i = 0; i < length; ++i) {
    if (ids[i] < 0 || ids[i] >= 36)
      throw std::invalid_argument("OCR target class");
    text += kAlphabet[ids[i]];
  }
  return text;
}
/// @brief Load and validate prepared human-word tensors with source-image-disjoint splits.
///
/// Requires finite32x100 grayscale observations,Int32[N,12] padded targets,
/// lengths1..12,unique nonnegative annotation IDs and disjoint source photograph
/// IDs between splits. Multiple words from one photograph may share a split.
/// At most12 characters plus11 repeat separators fit within26 CTC steps.
/// Validation verifies numeric/provenance structure,not correctness of annotations.
/// No normalization,training or optimizer is hidden by loading.
///
/// @param root Directory containing ocr.npz.
/// @return Three nonempty splits in fitting,validation,test order.
/// @throws std::invalid_argument For invalid shape,dtype,range,metadata or missing source data.
/// @par Example
/// @code
/// const auto root = std::filesystem::temp_directory_path() / "dm-missing-ocr-example";
/// try {
///   dm::nn::models::LoadOcr(root);
/// } catch (const std::exception &) {
/// }
/// @endcode
/// @par References
/// - Graves et al.(2006),Connectionist Temporal Classification:Labelling Unsegmented Sequence Data with Recurrent Neural Networks, https://www.cs.toronto.edu/~graves/icml_2006.pdf. CTC repeat/blank order and normalized edit-rate definitions.
/// - Veit et al.(2016),COCO-Text:Dataset and Benchmark for Text Detection and Recognition in Natural Images, https://arxiv.org/abs/1601.07140;V2 human annotation source https://github.com/bgshih/cocotext/releases/download/dl/cocotext.v2.zip. Custom source-image-disjoint43/7/6 word subset and grayscale32x100 stretch are teaching preparation,not official benchmark evaluation.
/// @see LoadOcr(), Crnn(), ScoreTensorOcr()
inline std::vector<OcrData> LoadOcr(const std::filesystem::path &root) {
  const auto arrays = nn::load_npz((root / "ocr.npz").string());
  /// @brief Retrieve one required prepared tensor by its archive key.
  /// @param key Exact split-prefixed array name.
  /// @return Matching owned tensor handle.
  auto get = [&](const std::string &key) {
    for (const auto &[name, value] : arrays)
      if (name == key)
        return value;
    throw std::runtime_error("ocr.npz missing " + key);
  };
  std::vector<OcrData> result;
  std::set<int> previous, annotations;
  for (const std::string split : {"fit", "validation", "test"}) {
    OcrData d{get(split + "_pixels"), get(split + "_labels"), get(split + "_lengths"),
              get(split + "_image_ids"), get(split + "_annotation_ids")};
    if (d.pixels.dim() != 4 || d.pixels.size(0) < 1 ||
        d.pixels.shape() != nn::Shape{d.pixels.size(0), 32, 100, 1} ||
        d.pixels.dtype() != nn::DType::Float32)
      throw std::invalid_argument("OCR pixel geometry/dtype");
    const auto count = d.pixels.size(0);
    if (d.labels.shape() != nn::Shape{count, 12} || d.lengths.shape() != nn::Shape{count} ||
        d.image_ids.shape() != nn::Shape{count} ||
        d.annotation_ids.shape() != nn::Shape{count})
      throw std::invalid_argument("OCR aligned metadata geometry");
    for (const auto *t : {&d.labels, &d.lengths, &d.image_ids, &d.annotation_ids})
      if (t->dtype() != nn::DType::Int32)
        throw std::invalid_argument("OCR Int32 metadata required");
    if (!nn::all(
             nn::logical_and(nn::greater_equal(d.pixels, 0.), nn::less_equal(d.pixels, 1.)))
             .item<bool>() ||
        !nn::all(
             nn::logical_and(nn::greater_equal(d.lengths, 1), nn::less_equal(d.lengths, 12)))
             .item<bool>())
      throw std::invalid_argument("OCR pixel/length ranges");
    const auto active = nn::less(nn::arange(0, 12, 1, nn::DType::Int32).reshape({1, 12}),
                                 d.lengths.reshape({count, 1}));
    const auto classes =
        nn::logical_and(nn::greater_equal(d.labels, 0), nn::less(d.labels, 36));
    if (!nn::all(nn::where(active, classes, nn::equal(d.labels, -1))).item<bool>())
      throw std::invalid_argument("OCR label/padding ranges");
    const auto images = d.image_ids.to_vector<int>(),
               notes = d.annotation_ids.to_vector<int>();
    std::set<int> current;
    for (std::int64_t i = 0; i < count; ++i) {
      if (images[i] < 0 || notes[i] < 0 || previous.contains(images[i]) ||
          !annotations.insert(notes[i]).second)
        throw std::invalid_argument("OCR source-image leakage or duplicate annotation");
      current.insert(images[i]);
    }
    previous.insert(current.begin(), current.end());
    result.push_back(d);
  }
  return result;
}
/// @brief Greedily decode fixed-length CRNN logits using tensor argmax.
///
/// Numerical selection is tensor based;host postprocessing only collapses
/// class metadata and serializes characters. Stable argmax chooses the first
/// class on a tie. Greedy decoding is not prefix-beam or best-word marginal decoding.
///
/// @param logits Finite floating[26,N,37],positive N,last class blank36.
/// @return N ASCII strings,possibly empty.
/// @throws std::invalid_argument For invalid shape,dtype,range,metadata or missing source data.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto words = n::models::OcrDecode(n::zeros({26, 2, 37}));
/// @endcode
/// @par References
/// - Graves et al.(2006),Connectionist Temporal Classification:Labelling Unsegmented Sequence Data with Recurrent Neural Networks, https://www.cs.toronto.edu/~graves/icml_2006.pdf. CTC repeat/blank order and normalized edit-rate definitions.
/// - Veit et al.(2016),COCO-Text:Dataset and Benchmark for Text Detection and Recognition in Natural Images, https://arxiv.org/abs/1601.07140;V2 human annotation source https://github.com/bgshih/cocotext/releases/download/dl/cocotext.v2.zip. Custom source-image-disjoint43/7/6 word subset and grayscale32x100 stretch are teaching preparation,not official benchmark evaluation.
/// @see LoadOcr(), Crnn(), ScoreTensorOcr()
inline std::vector<std::string> OcrDecode(const nn::Tensor &logits) {
  if (logits.dim() != 3 || logits.size(0) != 26 || logits.size(1) < 1 ||
      logits.size(2) != 37 ||
      !nn::all(nn::less(nn::abs(logits), std::numeric_limits<double>::infinity()))
           .item<bool>())
    throw std::invalid_argument("finite time-major26xNx37 OCR logits required");
  const auto classes = nn::argmax(logits, 2).to(nn::DType::Int64);
  std::vector<std::string> result;
  for (std::int64_t i = 0; i < logits.size(1); ++i)
    result.push_back(CtcCollapse(nn::select(classes, 1, i).to_vector<std::int64_t>()));
  return result;
}
/// @brief Own word-exact and character-edit recognition diagnostics.
///
/// Detached metrics use tensor edit-distance DP and tensor reductions. A low
/// CTC objective does not guarantee a correct greedily decoded word.
///
/// @throws std::invalid_argument For invalid shape,dtype,range,metadata or missing source data.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const n::models::OcrData d{
///     n::zeros({1, 32, 100, 1}),
///     n::tensor({10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, {1, 12}, n::DType::Int32),
///     n::ones({1}, n::DType::Int32), n::zeros({1}, n::DType::Int32),
///     n::ones({1}, n::DType::Int32)};
/// const auto metrics = n::models::ScoreTensorOcr(d, {"a"});
/// @endcode
/// @par References
/// - Graves et al.(2006),Connectionist Temporal Classification:Labelling Unsegmented Sequence Data with Recurrent Neural Networks, https://www.cs.toronto.edu/~graves/icml_2006.pdf. CTC repeat/blank order and normalized edit-rate definitions.
/// - Veit et al.(2016),COCO-Text:Dataset and Benchmark for Text Detection and Recognition in Natural Images, https://arxiv.org/abs/1601.07140;V2 human annotation source https://github.com/bgshih/cocotext/releases/download/dl/cocotext.v2.zip. Custom source-image-disjoint43/7/6 word subset and grayscale32x100 stretch are teaching preparation,not official benchmark evaluation.
/// @see LoadOcr(), Crnn(), ScoreTensorOcr()
struct TensorOcrMetrics {
  nn::Tensor errors;     ///< Int64 scalar total insertion/deletion/substitution edits.
  nn::Tensor characters; ///< Int64 scalar number of reference characters.
  nn::Tensor exact;      ///< Int64 scalar exact-match words.
  nn::Tensor words;      ///< Int64 scalar evaluated word count.
  nn::Tensor character_error_rate; ///< Float64 edits/reference characters,may exceed1.
};
/// @brief Score actual target words against decoded ASCII predictions.
///
/// Every numeric edit cost and aggregate uses tensors. Class/string metadata
/// remain host-side;no fitting or post-hoc test selection. CER can exceed1
/// when predictions add many extra characters.
///
/// @param d Prepared nonempty targets/positive lengths.
/// @param predicted One decoded string per target row,including empty strings.
/// @return Detached edit,reference-character,exact-word,count and CER tensors.
/// @throws std::invalid_argument For invalid shape,dtype,range,metadata or missing source data.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const n::models::OcrData d{
///     n::zeros({1, 32, 100, 1}),
///     n::tensor({10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, {1, 12}, n::DType::Int32),
///     n::ones({1}, n::DType::Int32), n::zeros({1}, n::DType::Int32),
///     n::ones({1}, n::DType::Int32)};
/// const auto metrics = n::models::ScoreTensorOcr(d, {""});
/// @endcode
/// @par References
/// - Graves et al.(2006),Connectionist Temporal Classification:Labelling Unsegmented Sequence Data with Recurrent Neural Networks, https://www.cs.toronto.edu/~graves/icml_2006.pdf. CTC repeat/blank order and normalized edit-rate definitions.
/// - Veit et al.(2016),COCO-Text:Dataset and Benchmark for Text Detection and Recognition in Natural Images, https://arxiv.org/abs/1601.07140;V2 human annotation source https://github.com/bgshih/cocotext/releases/download/dl/cocotext.v2.zip. Custom source-image-disjoint43/7/6 word subset and grayscale32x100 stretch are teaching preparation,not official benchmark evaluation.
/// @see LoadOcr(), Crnn(), ScoreTensorOcr()
inline TensorOcrMetrics ScoreTensorOcr(const OcrData &d,
                                       const std::vector<std::string> &predicted) {
  nn::NoGradGuard guard;
  if (predicted.empty() || static_cast<std::int64_t>(predicted.size()) != d.labels.size(0))
    throw std::invalid_argument("aligned nonempty OCR predictions required");
  std::vector<std::int64_t> edits;
  for (std::size_t i = 0; i < predicted.size(); ++i)
    edits.push_back(OcrDistance(OcrText(d, int(i)), predicted[i]));
  const auto costs = nn::from_vector(edits, {static_cast<std::int64_t>(edits.size())}),
             errors = nn::sum(costs), characters = nn::sum(d.lengths.to(nn::DType::Int64));
  return {errors, characters, nn::count_nonzero(nn::equal(costs, 0)).to(nn::DType::Int64),
          nn::full({}, static_cast<std::int64_t>(predicted.size()), nn::DType::Int64),
          errors.to(nn::DType::Float64) / characters.to(nn::DType::Float64)};
}
/// @brief Print canonical tensor OCR metrics for the legacy example.
///
/// Presentation only;ScoreTensorOcr performs all numerical work.
///
/// @param name Presentation model label.
/// @param d Aligned target words.
/// @param predicted One decoded string per row.
/// @throws std::invalid_argument For invalid shape,dtype,range,metadata or missing source data.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const n::models::OcrData d{
///     n::zeros({1, 32, 100, 1}),
///     n::tensor({10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, {1, 12}, n::DType::Int32),
///     n::ones({1}, n::DType::Int32), n::zeros({1}, n::DType::Int32),
///     n::ones({1}, n::DType::Int32)};
/// n::models::OcrMetrics("perfect", d, {"a"});
/// @endcode
/// @par References
/// - Graves et al.(2006),Connectionist Temporal Classification:Labelling Unsegmented Sequence Data with Recurrent Neural Networks, https://www.cs.toronto.edu/~graves/icml_2006.pdf. CTC repeat/blank order and normalized edit-rate definitions.
/// - Veit et al.(2016),COCO-Text:Dataset and Benchmark for Text Detection and Recognition in Natural Images, https://arxiv.org/abs/1601.07140;V2 human annotation source https://github.com/bgshih/cocotext/releases/download/dl/cocotext.v2.zip. Custom source-image-disjoint43/7/6 word subset and grayscale32x100 stretch are teaching preparation,not official benchmark evaluation.
/// @see LoadOcr(), Crnn(), ScoreTensorOcr()
inline void OcrMetrics(const std::string &name, const OcrData &d,
                       const std::vector<std::string> &predicted) {
  const auto m = ScoreTensorOcr(d, predicted);
  std::cout << " " << name << ": word exact=" << m.exact.item<std::int64_t>() << "/"
            << m.words.item<std::int64_t>()
            << ",character edit rate=" << m.character_error_rate.item() << " ("
            << m.errors.item<std::int64_t>() << " edits/" << m.characters.item<std::int64_t>()
            << " reference characters)\n";
}
/// @brief Describe one observed word crop and its original human annotation.
///
/// Host values are source metadata only. Crop bounds and all grayscale/resize
/// arithmetic are computed by tensors;the text never modifies observed pixels.
///
/// @throws std::invalid_argument For invalid shape,dtype,range,metadata or missing source data.
/// @par Example
/// @code
/// const dm::nn::models::OcrCropRecord record{9, 17, "a", {0., 0., 2., 3.}, 2, 3};
/// @endcode
/// @par References
/// - Graves et al.(2006),Connectionist Temporal Classification:Labelling Unsegmented Sequence Data with Recurrent Neural Networks, https://www.cs.toronto.edu/~graves/icml_2006.pdf. CTC repeat/blank order and normalized edit-rate definitions.
/// - Veit et al.(2016),COCO-Text:Dataset and Benchmark for Text Detection and Recognition in Natural Images, https://arxiv.org/abs/1601.07140;V2 human annotation source https://github.com/bgshih/cocotext/releases/download/dl/cocotext.v2.zip. Custom source-image-disjoint43/7/6 word subset and grayscale32x100 stretch are teaching preparation,not official benchmark evaluation.
/// @see LoadOcr(), Crnn(), ScoreTensorOcr()
struct OcrCropRecord {
  std::int32_t image_id;      ///< Original nonnegative COCO photograph ID.
  std::int32_t annotation_id; ///< Nonnegative unique human word annotation ID.
  std::string text;           ///< Lowercase ASCII human target,1..12 alphabet characters.
  std::array<double, 4> bbox; ///< Original image x,y,width,height,unrounded pixels.
  int expected_width = 0; ///< Source image width to verify,zero disables this metadata check.
  int expected_height =
      0; ///< Source image height to verify,zero disables this metadata check.
};
/// @brief Decode original photographs and prepare human-annotated grayscale word crops.
///
/// Tensor floor/ceil round box edges outward and clip to original image bounds.
/// RGB converts through Float64 encoded grayscale weights.299/.587/.114,then
/// casts to Float32 before half-pixel bilinear stretch32x100(no aspect padding,
/// rotation rectification or antialias). Grayscale files remain one channel.
/// Human word classes are metadata;no character segmentation or pixel synthesis.
/// Known boxes make this oracle-crop recognition,not end-to-end text detection.
///
/// @param image_directory Directory of original12-digit-ID.jpg images.
/// @param records Nonempty source IDs,text,xywh boxes and optional original dimensions.
/// @return Float32[N,32,100,1] crops and aligned Int32 padded targets/provenance.
/// @throws std::invalid_argument For invalid shape,dtype,range,metadata or missing source data.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto root = std::filesystem::temp_directory_path() / "dm-ocr-crop-example";
/// std::filesystem::create_directories(root);
/// n::save_png(n::ones({3, 2, 1}), (root / "000000000009.jpg").string());
/// const auto words = n::models::ReadOcrCrops(root, {{9, 17, "a", {0., 0., 2., 3.}, 2, 3}});
/// std::filesystem::remove(root / "000000000009.jpg");
/// std::filesystem::remove(root);
/// @endcode
/// @par References
/// - Graves et al.(2006),Connectionist Temporal Classification:Labelling Unsegmented Sequence Data with Recurrent Neural Networks, https://www.cs.toronto.edu/~graves/icml_2006.pdf. CTC repeat/blank order and normalized edit-rate definitions.
/// - Veit et al.(2016),COCO-Text:Dataset and Benchmark for Text Detection and Recognition in Natural Images, https://arxiv.org/abs/1601.07140;V2 human annotation source https://github.com/bgshih/cocotext/releases/download/dl/cocotext.v2.zip. Custom source-image-disjoint43/7/6 word subset and grayscale32x100 stretch are teaching preparation,not official benchmark evaluation.
/// @see LoadOcr(), Crnn(), ScoreTensorOcr()
inline OcrData ReadOcrCrops(const std::filesystem::path &image_directory,
                            const std::vector<OcrCropRecord> &records) {
  nn::NoGradGuard guard;
  if (records.empty())
    throw std::invalid_argument("nonempty OCR source records required");
  std::vector<nn::Tensor> crops;
  std::vector<std::int32_t> labels, lengths, images, annotations;
  std::set<std::int32_t> seen;
  for (const auto &r : records) {
    if (r.image_id < 0 || r.annotation_id < 0 || !seen.insert(r.annotation_id).second ||
        r.text.empty() || r.text.size() > 12)
      throw std::invalid_argument("invalid OCR source IDs/text");
    for (char c : r.text)
      if (kAlphabet.find(c) == std::string::npos)
        throw std::invalid_argument("OCR alphabet character required");
    std::ostringstream name;
    name << std::setfill('0') << std::setw(12) << r.image_id << ".jpg";
    auto pixel =
        nn::from_image(datamunge::image::read_image((image_directory / name.str()).string()));
    if (pixel.dim() != 3 || (pixel.size(2) != 1 && pixel.size(2) != 3) ||
        (r.expected_width > 0 && pixel.size(1) != r.expected_width) ||
        (r.expected_height > 0 && pixel.size(0) != r.expected_height))
      throw std::invalid_argument("OCR source image geometry mismatch");
    const auto box = nn::from_vector(std::vector<double>(r.bbox.begin(), r.bbox.end()), {4});
    if (!nn::all(nn::less(nn::abs(box), std::numeric_limits<double>::infinity())).item<bool>())
      throw std::invalid_argument("finite OCR box required");
    const auto x = nn::select(box, 0, 0), y = nn::select(box, 0, 1), w = nn::select(box, 0, 2),
               h = nn::select(box, 0, 3);
    if (!nn::logical_and(nn::greater(w, 0.), nn::greater(h, 0.)).item<bool>() ||
        !nn::logical_and(nn::logical_and(nn::less(x, double(pixel.size(1))),
                                         nn::less(y, double(pixel.size(0)))),
                         nn::logical_and(nn::greater(x + w, 0.), nn::greater(y + h, 0.)))
             .item<bool>())
      throw std::invalid_argument("positive intersecting OCR box required");
    const auto x0 = nn::clip(nn::floor(x), 0., double(pixel.size(1) - 1)),
               y0 = nn::clip(nn::floor(y), 0., double(pixel.size(0) - 1)),
               x1 = nn::clip(nn::ceil(x + w), 0., double(pixel.size(1))),
               y1 = nn::clip(nn::ceil(y + h), 0., double(pixel.size(0)));
    auto observed = nn::slice_axis(nn::slice_axis(pixel, 0, y0.item<std::int64_t>(),
                                                  y1.item<std::int64_t>()),
                                   1, x0.item<std::int64_t>(), x1.item<std::int64_t>())
                        .to(nn::DType::Float64) /
                    255.;
    if (pixel.size(2) == 3)
      observed = .299 * nn::slice_axis(observed, 2, 0, 1) +
                 .587 * nn::slice_axis(observed, 2, 1, 2) +
                 .114 * nn::slice_axis(observed, 2, 2, 3);
    crops.push_back(nn::resize_bilinear(nn::expand_dims(observed.to(nn::DType::Float32), 0),
                                        {32, 100}, false, true));
    for (char c : r.text)
      labels.push_back(static_cast<std::int32_t>(kAlphabet.find(c)));
    for (std::size_t j = r.text.size(); j < 12; ++j)
      labels.push_back(-1);
    lengths.push_back(r.text.size());
    images.push_back(r.image_id);
    annotations.push_back(r.annotation_id);
  }
  const auto count = static_cast<std::int64_t>(records.size());
  return {nn::concat(crops, 0), nn::from_vector(labels, {count, 12}),
          nn::from_vector(lengths, {count}), nn::from_vector(images, {count}),
          nn::from_vector(annotations, {count})};
}
} // namespace datamunge::nn::models
