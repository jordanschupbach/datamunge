#pragma once
/// @file multilabel_data.hpp
/// @brief Tensor COCO category-presence ingress,fit-only class weights/priors and aligned image rows.
#include "datamunge/image/read_image.hpp"
#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/checkpoint.hpp"
#include "datamunge/nn/image_io.hpp"
#include "datamunge/nn/image_ops.hpp"
#include "datamunge/nn/ops.hpp"
#include <filesystem>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Own observed RGB,independent category-presence bits and source-image IDs.
///
/// Tensor handles share owning storage. Prepared COCO subsets use Float32
/// RGB[N,64,64,3] in[0,1],Float32 binary targets[N,80],Int32 source IDs[N].
/// The container itself does not validate values or split provenance.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::MultilabelData data{n::zeros({2, 64, 64, 3}), n::zeros({2, 80}),
///                                n::arange(1, 3, 1, n::DType::Int32)};
/// @endcode
/// @par References
/// - Lin et al.(2014),Microsoft COCO:Common Objects in Context, https://arxiv.org/abs/1405.0312;COCO128 source https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco128.yaml. Custom80/24/24 annotation-presence teaching split and64px tensor resize,not official COCO evaluation.
/// @see LoadMultilabel(), MultilabelWeights()
struct MultilabelData {
  nn::Tensor rgb;    ///< Float32[N,64,64,3] full observed images.
  nn::Tensor labels; ///< Float32[N,80] independent binary category-presence targets.
  nn::Tensor ids;    ///< Int32[N] original COCO image IDs.
};
/// @brief Own custom disjoint COCO128 fitting,validation and test subsets.
///
/// LoadMultilabel validates80/24/24 rows and unique IDs;the value container
/// alone does not imply any official train/validation benchmark protocol.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::MultilabelData data{n::zeros({2, 64, 64, 3}), n::zeros({2, 80}),
///                                n::arange(1, 3, 1, n::DType::Int32)};
/// n::models::MultilabelSplits split{data, data, data};
/// @endcode
/// @par References
/// - Lin et al.(2014),Microsoft COCO:Common Objects in Context, https://arxiv.org/abs/1405.0312;COCO128 source https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco128.yaml. Custom80/24/24 annotation-presence teaching split and64px tensor resize,not official COCO evaluation.
/// @see LoadMultilabel(), MultilabelWeights()
struct MultilabelSplits {
  MultilabelData fit;        ///<80 fitting images when loaded from the prepared archive.
  MultilabelData validation; ///<24 custom held-out calibration/checkpoint images.
  MultilabelData test;       ///<24 other held-out images,never used to fit or calibrate.
};
/// @brief Internal binary-target and observed-image shape validation.
namespace multilabel_data_detail {
/// @brief Require a two-dimensional numeric binary matrix with positive class width.
/// @param labels Numeric/bool[N,C],C positive,N may be zero.
/// @throws std::invalid_argument For invalid rank,width or nonbinary values.
inline void ValidateLabels(const nn::Tensor &labels) {
  if (labels.dim() != 2 || labels.shape()[1] < 1 ||
      !nn::all(nn::logical_or(nn::equal(labels, 0.), nn::equal(labels, 1.))).item<bool>())
    throw std::invalid_argument("binary multilabel target matrix required");
}
/// @brief Require valid positive64px RGB images,80 binary bits and aligned source IDs.
/// @param data Float32 RGB/targets and Int32 IDs.
/// @throws std::invalid_argument For invalid shape,dtype or values.
inline void ValidateData(const MultilabelData &data) {
  ValidateLabels(data.labels);
  if (data.rgb.dim() != 4 || data.rgb.shape()[0] < 1 || data.rgb.shape()[1] != 64 ||
      data.rgb.shape()[2] != 64 || data.rgb.shape()[3] != 3 ||
      data.rgb.dtype() != nn::DType::Float32 || data.labels.dtype() != nn::DType::Float32 ||
      data.labels.shape() != nn::Shape{data.rgb.shape()[0], 80} ||
      data.ids.dtype() != nn::DType::Int32 ||
      data.ids.shape() != nn::Shape{data.rgb.shape()[0]} ||
      !nn::all(nn::logical_and(nn::greater_equal(data.rgb, 0.), nn::less_equal(data.rgb, 1.)))
           .item<bool>())
    throw std::invalid_argument("invalid multilabel image tensors");
}
} // namespace multilabel_data_detail
/// @brief Gather arbitrary batch rows while preserving logical image/target order.
///
/// Host vectors carry only indices. There is no pixel arithmetic,fitting
/// or input mutation.
///
/// @param input Tensor with a leading batch axis.
/// @param rows Valid requested batch-row metadata,duplicates/empty allowed.
/// @return Same-type tensor with selected leading rows and retained input gradients.
/// @throws std::invalid_argument For invalid shapes,dtypes,binary labels,metadata,rows,counts or unreadable prepared data.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto rows = n::models::MultilabelRows(n::zeros({3, 64, 64, 3}), {2, 0, 2});
/// @endcode
/// @par References
/// - Lin et al.(2014),Microsoft COCO:Common Objects in Context, https://arxiv.org/abs/1405.0312;COCO128 source https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco128.yaml. Custom80/24/24 annotation-presence teaching split and64px tensor resize,not official COCO evaluation.
/// @see MultilabelRows(const MultilabelData&, const std::vector<std::size_t>&)
inline nn::Tensor MultilabelRows(const nn::Tensor &input,
                                 const std::vector<std::size_t> &rows) {
  if (input.dim() < 1)
    throw std::invalid_argument("multilabel tensor needs a batch axis");
  std::vector<std::int64_t> indices;
  for (const auto row : rows) {
    if (row >= std::size_t(input.shape()[0]))
      throw std::invalid_argument("multilabel row out of bounds");
    indices.push_back(std::int64_t(row));
  }
  return nn::gather(input, nn::from_vector(indices)).contiguous();
}
/// @brief Gather image,presence-label and source-ID rows together.
///
/// No image arithmetic,fitting or mutation. Pixel and numeric target
/// gathering use tensors.
///
/// @param data Aligned observed-image,presence-bit and source-ID tensors.
/// @param rows Valid requested source rows,duplicates/empty allowed.
/// @return Selected owning tensor handles in one aligned order.
/// @throws std::invalid_argument For invalid shapes,dtypes,binary labels,metadata,rows,counts or unreadable prepared data.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::MultilabelData data{n::zeros({2, 64, 64, 3}), n::zeros({2, 80}),
///                                n::arange(1, 3, 1, n::DType::Int32)};
/// const auto selected = n::models::MultilabelRows(data, {1, 0, 1});
/// @endcode
/// @par References
/// - Lin et al.(2014),Microsoft COCO:Common Objects in Context, https://arxiv.org/abs/1405.0312;COCO128 source https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco128.yaml. Custom80/24/24 annotation-presence teaching split and64px tensor resize,not official COCO evaluation.
/// @see LoadMultilabel(), MultilabelWeights()
inline MultilabelData MultilabelRows(const MultilabelData &data,
                                     const std::vector<std::size_t> &rows) {
  return {MultilabelRows(data.rgb, rows), MultilabelRows(data.labels, rows),
          MultilabelRows(data.ids, rows)};
}
/// @brief Load and validate a prepared disjoint80/24/24 category-presence archive.
///
/// Each split needs at least one image with multiple present categories.
/// All128 IDs must be distinct. Validation checks numeric consistency,not
/// official annotation provenance;the provided preparation script records
/// source ZIP hash,original IDs,categories and custom split seed. No download,
/// resizing,training or archive mutation occurs in this reader.
///
/// @param root Directory containing multilabel.npz with split_rgb/labels/ids arrays.
/// @return Fitting,validation and test subsets with valid numeric values and unique image IDs.
/// @throws std::invalid_argument For invalid shapes,dtypes,binary labels,metadata,rows,counts or unreadable prepared data.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto root = std::filesystem::temp_directory_path() / "dm-multilabel-doc-example";
/// std::filesystem::create_directories(root);
/// n::NamedTensors arrays;
/// int offset = 1;
/// for (const auto &entry :
///      std::vector<std::pair<std::string, int>>{{"fit", 80}, {"validation", 24}, {"test", 24}}) {
///   const auto &[split, count] = entry;
///   arrays.push_back({split + "_rgb", n::zeros({count, 64, 64, 3})});
///   arrays.push_back(
///       {split + "_labels", n::concat({n::ones({count, 2}), n::zeros({count, 78})}, 1)});
///   arrays.push_back({split + "_ids", n::arange(offset, offset + count, 1, n::DType::Int32)});
///   offset += count;
/// }
/// n::save_npz((root / "multilabel.npz").string(), arrays);
/// const auto data = n::models::LoadMultilabel(root);
/// std::filesystem::remove(root / "multilabel.npz");
/// std::filesystem::remove(root);
/// @endcode
/// @par References
/// - Lin et al.(2014),Microsoft COCO:Common Objects in Context, https://arxiv.org/abs/1405.0312;COCO128 source https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco128.yaml. Custom80/24/24 annotation-presence teaching split and64px tensor resize,not official COCO evaluation.
/// @see LoadMultilabel(), MultilabelWeights()
inline MultilabelSplits LoadMultilabel(const std::filesystem::path &root) {
  std::map<std::string, nn::Tensor> arrays;
  for (auto &[name, value] : nn::load_npz((root / "multilabel.npz").string()))
    arrays[name] = value;
  std::set<std::int32_t> ids;
  /// @brief Read a numeric split and update the shared source-ID registry.
  /// @param split Required NPZ prefix.
  /// @param count Exact expected split row count.
  /// @return Valid observed-image/presence-bit tensor subset.
  auto read = [&](const std::string &split, std::int64_t count) {
    MultilabelData data{arrays.at(split + "_rgb"), arrays.at(split + "_labels"),
                        arrays.at(split + "_ids")};
    multilabel_data_detail::ValidateData(data);
    if (data.rgb.shape()[0] != count ||
        !nn::any(nn::greater(nn::count_nonzero(nn::equal(data.labels, 1.), {1}), 1.))
             .item<bool>())
      throw std::invalid_argument(
          "expected split size and genuine multi-object images required");
    for (const auto id : data.ids.to_vector<std::int32_t>())
      if (!ids.insert(id).second)
        throw std::invalid_argument("multilabel source IDs overlap");
    return data;
  };
  auto fit = read("fit", 80), validation = read("validation", 24), test = read("test", 24);
  return {fit, validation, test};
}
/// @brief Compute fit-only capped positive-class BCE weights from exact binary counts.
///
/// Counts use Int64 tensor reductions before Float32 division. Absent
/// classes retain a finite capped weight;empty fitting rows give weight1.
/// Only supplied fitting labels enter the calculation;no optimizer or
/// validation/test calibration is hidden here.
///
/// @param labels Numeric/bool[N,C] binary fitting targets,C positive,N may be zero.
/// @return Detached Float32[C] clamp((N-positive+1)/(positive+1),1,20).
/// @throws std::invalid_argument For invalid shapes,dtypes,binary labels,metadata,rows,counts or unreadable prepared data.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto weights = n::models::MultilabelWeights(n::tensor({1., 0., 0., 1., 1., 0.}, {3, 2}));
/// @endcode
/// @par References
/// - Lin et al.(2014),Microsoft COCO:Common Objects in Context, https://arxiv.org/abs/1405.0312;COCO128 source https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco128.yaml. Custom80/24/24 annotation-presence teaching split and64px tensor resize,not official COCO evaluation.
/// @see LoadMultilabel(), MultilabelWeights()
inline nn::Tensor MultilabelWeights(const nn::Tensor &labels) {
  nn::NoGradGuard guard;
  multilabel_data_detail::ValidateLabels(labels);
  const auto positives = nn::count_nonzero(nn::equal(labels, 1.), {0});
  const auto total = nn::from_scalar(labels.shape()[0]);
  const auto numerator = (total - positives + nn::ones_like(positives)).to(nn::DType::Float32),
             denominator = (positives + nn::ones_like(positives)).to(nn::DType::Float32);
  return nn::clip(numerator / denominator, 1., 20.);
}
/// @brief Repeat a Laplace-smoothed fit-only category-presence prior.
///
/// This closed-form reference uses one positive and one negative pseudobit
/// per class. Empty fitting rows give.5. The report also reaches these
/// probabilities with an explicit one-step SGD Brier objective;this helper
/// contains no optimizer,calibration or held-out label access.
///
/// @param fitting Numeric/bool[N,C] binary fitting targets,C positive,N may be zero.
/// @param count Nonnegative number of output rows.
/// @return Detached Float32[count,C] probabilities(positive+1)/(N+2).
/// @throws std::invalid_argument For invalid shapes,dtypes,binary labels,metadata,rows,counts or unreadable prepared data.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto prior = n::models::MultilabelPrior(n::tensor({1., 0., 0., 1., 1., 0.}, {3, 2}), 4);
/// @endcode
/// @par References
/// - Lin et al.(2014),Microsoft COCO:Common Objects in Context, https://arxiv.org/abs/1405.0312;COCO128 source https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco128.yaml. Custom80/24/24 annotation-presence teaching split and64px tensor resize,not official COCO evaluation.
/// @see LoadMultilabel(), MultilabelWeights()
inline nn::Tensor MultilabelPrior(const nn::Tensor &fitting, int count) {
  nn::NoGradGuard guard;
  multilabel_data_detail::ValidateLabels(fitting);
  if (count < 0)
    throw std::invalid_argument("nonnegative multilabel prior row count required");
  const auto positives = nn::count_nonzero(nn::equal(fitting, 1.), {0});
  const auto denominator =
      (nn::from_scalar(fitting.shape()[0]) + nn::from_scalar(std::int64_t(2)))
          .to(nn::DType::Float32);
  const auto prior =
      (positives + nn::ones_like(positives)).to(nn::DType::Float32) / denominator;
  return nn::ones({count, fitting.shape()[1]}) * prior.reshape({1, fitting.shape()[1]});
}
/// @brief Decode observed images and OR annotation-category IDs into independent presence bits.
///
/// Byte decoding is ingress. Tensor operations replicate grayscale toRGB,
/// divide by255,bilinearly stretch to64px without crop/antialias,and OR one-hot
/// category metadata into labels. Category multiplicity never increases a bit
/// above1. Signature-based decoding permits PNG test fixtures under jpg names.
/// No source box-coordinate arithmetic,fitting or label-based pixel changes.
///
/// @param image_directory Directory with12-digit original image-ID filenames plus .jpg.
/// @param ids Nonempty nonnegative Int32 original image IDs,in requested output order.
/// @param categories Same-length lists of contiguous COCO128 class IDs0..79;duplicates/empty allowed.
/// @return Float32[N,64,64,3] RGB,Float32[N,80] OR-presence labels,Int32[N] IDs.
/// @throws std::invalid_argument For invalid shapes,dtypes,binary labels,metadata,rows,counts or unreadable prepared data.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto root = std::filesystem::temp_directory_path() / "dm-multilabel-image-example";
/// std::filesystem::create_directories(root);
/// n::save_png(n::ones({2, 3, 1}), (root / "000000000009.jpg").string());
/// const auto data = n::models::ReadMultilabelImages(root, {9}, {{0, 0, 2}});
/// std::filesystem::remove(root / "000000000009.jpg");
/// std::filesystem::remove(root);
/// @endcode
/// @par References
/// - Lin et al.(2014),Microsoft COCO:Common Objects in Context, https://arxiv.org/abs/1405.0312;COCO128 source https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco128.yaml. Custom80/24/24 annotation-presence teaching split and64px tensor resize,not official COCO evaluation.
/// @see LoadMultilabel(), MultilabelWeights()
inline MultilabelData ReadMultilabelImages(const std::filesystem::path &image_directory,
                                           const std::vector<std::int32_t> &ids,
                                           const std::vector<std::vector<int>> &categories) {
  nn::NoGradGuard guard;
  if (ids.empty() || ids.size() != categories.size())
    throw std::invalid_argument("aligned nonempty multilabel image metadata required");
  std::vector<nn::Tensor> images, targets;
  for (std::size_t i = 0; i < ids.size(); ++i) {
    if (ids[i] < 0)
      throw std::invalid_argument("nonnegative COCO image ID required");
    std::ostringstream name;
    name << std::setfill('0') << std::setw(12) << ids[i] << ".jpg";
    auto pixels =
        nn::from_image(datamunge::image::read_image((image_directory / name.str()).string()));
    if (pixels.dim() != 3 || (pixels.shape()[2] != 1 && pixels.shape()[2] != 3))
      throw std::invalid_argument("grayscale or RGB observed image required");
    if (pixels.shape()[2] == 1)
      pixels = pixels * nn::ones({1, 1, 3});
    images.push_back(nn::resize_bilinear(nn::expand_dims(pixels / 255., 0), {64, 64}));
    std::vector<std::int64_t> labels;
    for (const auto category : categories[i]) {
      if (category < 0 || category >= 80)
        throw std::invalid_argument("COCO category outside0..79");
      labels.push_back(category);
    }
    const auto bits = nn::one_hot(nn::from_vector(labels), 80);
    targets.push_back(nn::greater(nn::sum(bits, {0}), 0.).to(nn::DType::Float32));
  }
  return {nn::concat(images, 0), nn::stack(targets), nn::from_vector(ids)};
}
} // namespace datamunge::nn::models
