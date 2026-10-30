#pragma once
/// @file flower_data.hpp
/// @brief Tensor preprocessing,official-subset ingress,metrics and galleries for fine-grained flower examples.
#include "datamunge/image/read_image.hpp"
#include "datamunge/nn/bilinear_models.hpp"
#include "datamunge/nn/image_io.hpp"
#include "datamunge/nn/checkpoint.hpp"
#include <filesystem>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Own observed flower RGB images,zero-based class labels and original source IDs.
///
/// Tensor handles own shared storage. Prepared subsets use48 rows,64x64RGB
/// Float32[0,1],Int32 labels0..5,and unique official IDs1..8189. Row selection
/// can return smaller or empty batches. No optimizer or learned state.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::FlowerData data{n::zeros({6, 64, 64, 3}), n::arange(0, 6, 1, n::DType::Int32),
///                            n::arange(1, 7, 1, n::DType::Int32)};
/// @endcode
/// @par References
/// - Nilsback and Zisserman(2008),Automated Flower Classification over a Large Number of Classes, https://www.robots.ox.ac.uk/~vgg/data/flowers/102/. Six-class/eight-per-class64px official-split subset is an explicit teaching adaptation;not the full102-class benchmark.
/// @see LoadFlowers(), ScoreFlowers()
struct FlowerData {
  nn::Tensor rgb;    ///< Float32[N,64,64,3] observed images in[0,1].
  nn::Tensor labels; ///< Int32[N] zero-based selected-class labels0..5.
  nn::Tensor ids;    ///< Int32[N] original image IDs1..8189.
};
/// @brief Own disjoint fitting,validation and test flower subsets.
///
/// This container alone does not validate split disjointness;LoadFlowers
/// validates all144 IDs across the three prepared subsets.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::FlowerData data{n::zeros({6, 64, 64, 3}), n::arange(0, 6, 1, n::DType::Int32),
///                            n::arange(1, 7, 1, n::DType::Int32)};
/// n::models::FlowerSplits splits{data, data, data};
/// @endcode
/// @par References
/// - Nilsback and Zisserman(2008),Automated Flower Classification over a Large Number of Classes, https://www.robots.ox.ac.uk/~vgg/data/flowers/102/. Six-class/eight-per-class64px official-split subset is an explicit teaching adaptation;not the full102-class benchmark.
/// @see LoadFlowers(), ScoreFlowers()
struct FlowerSplits {
  FlowerData fit;        ///< Fitting data,48 balanced source images when loaded.
  FlowerData validation; ///< Disjoint official validation subset,48 images.
  FlowerData test;       ///< Disjoint official test subset,48 images.
};
/// @brief Internal validation for flower tensor ingress and evaluation.
namespace flower_data_detail {
/// @brief Check positive RGB batch,aligned six-class labels and source IDs.
/// @param data Flower tensors,Float32 RGB[0,1],Int32 labels and IDs.
/// @throws std::invalid_argument For invalid shape,dtype or values.
inline void Validate(const FlowerData &data) {
  if (data.rgb.dim() != 4 || data.rgb.shape()[0] < 1 || data.rgb.shape()[1] != 64 ||
      data.rgb.shape()[2] != 64 || data.rgb.shape()[3] != 3 ||
      data.rgb.dtype() != nn::DType::Float32 ||
      data.labels.shape() != nn::Shape{data.rgb.shape()[0]} ||
      data.ids.shape() != data.labels.shape() || data.labels.dtype() != nn::DType::Int32 ||
      data.ids.dtype() != nn::DType::Int32 ||
      !nn::all(nn::logical_and(nn::greater_equal(data.rgb, 0.), nn::less_equal(data.rgb, 1.)))
           .item<bool>() ||
      !nn::all(nn::logical_and(nn::greater_equal(data.labels, 0.), nn::less(data.labels, 6.)))
           .item<bool>() ||
      !nn::all(
           nn::logical_and(nn::greater_equal(data.ids, 1.), nn::less_equal(data.ids, 8189.)))
           .item<bool>())
    throw std::invalid_argument("invalid flower tensors");
}
} // namespace flower_data_detail
/// @brief Gather arbitrary source rows of a flower tensor.
///
/// Gather retains input gradients. Host vectors carry only indices;no pixel
/// arithmetic,fitting or mutation.
///
/// @param input Tensor with a batch axis.
/// @param rows Valid batch-row metadata,duplicates/empty allowed.
/// @return Same-type tensor with rows.size() leading extent and original trailing shape.
/// @throws std::invalid_argument For invalid shapes,dtypes,rows,labels,IDs,probabilities or unreadable prepared files.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto rows = n::models::FlowerRows(n::zeros({3, 64, 64, 3}), {2, 0, 2});
/// @endcode
/// @par References
/// - Nilsback and Zisserman(2008),Automated Flower Classification over a Large Number of Classes, https://www.robots.ox.ac.uk/~vgg/data/flowers/102/. Six-class/eight-per-class64px official-split subset is an explicit teaching adaptation;not the full102-class benchmark.
/// @see FlowerRows(const FlowerData&, const std::vector<std::size_t>&)
inline nn::Tensor FlowerRows(const nn::Tensor &input, const std::vector<std::size_t> &rows) {
  if (input.dim() < 1)
    throw std::invalid_argument("flower tensor needs a batch axis");
  std::vector<std::int64_t> ids;
  for (const auto row : rows) {
    if (row >= std::size_t(input.shape()[0]))
      throw std::invalid_argument("flower row out of bounds");
    ids.push_back(std::int64_t(row));
  }
  return nn::gather(input, nn::from_vector(ids)).contiguous();
}
/// @brief Gather image,label and official-ID rows in one aligned order.
///
/// No pixel arithmetic,fitting or mutation;input gradients are retained.
///
/// @param data Flower image/label/ID tensors.
/// @param rows Valid requested source rows,duplicates/empty allowed.
/// @return Aligned owning tensor handles for the selected rows.
/// @throws std::invalid_argument For invalid shapes,dtypes,rows,labels,IDs,probabilities or unreadable prepared files.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::FlowerData data{n::zeros({6, 64, 64, 3}), n::arange(0, 6, 1, n::DType::Int32),
///                            n::arange(1, 7, 1, n::DType::Int32)};
/// const auto selected = n::models::FlowerRows(data, {5, 0, 5});
/// @endcode
/// @par References
/// - Nilsback and Zisserman(2008),Automated Flower Classification over a Large Number of Classes, https://www.robots.ox.ac.uk/~vgg/data/flowers/102/. Six-class/eight-per-class64px official-split subset is an explicit teaching adaptation;not the full102-class benchmark.
/// @see LoadFlowers(), ScoreFlowers()
inline FlowerData FlowerRows(const FlowerData &data, const std::vector<std::size_t> &rows) {
  return {FlowerRows(data.rgb, rows), FlowerRows(data.labels, rows),
          FlowerRows(data.ids, rows)};
}
/// @brief Read and validate a prepared balanced Flowers102 subset archive.
///
/// Expected prefixes fit,validation,test. Each has eight rows per class0..5.
/// Validation checks archive consistency and ID disjointness;it cannot prove
/// official split/label provenance without source metadata. Use the provided
/// dm::datasets::prepare_example_data("flowers", directory) workflow to construct that provenance. No downloading,
/// resizing,fitting or archive mutation occurs here.
///
/// @param root Directory containing flowers.npz with split_rgb/labels/ids arrays.
/// @return Three48-image subsets with valid shapes,ranges,class balance and disjoint IDs.
/// @throws std::invalid_argument For invalid shapes,dtypes,rows,labels,IDs,probabilities or unreadable prepared files.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto root = std::filesystem::temp_directory_path() / "dm-flower-doc-example";
/// std::filesystem::create_directories(root);
/// n::NamedTensors arrays;
/// int offset = 1;
/// for (const std::string split : {"fit", "validation", "test"}) {
///   const auto rows = n::arange(0, 48, 1, n::DType::Int32);
///   arrays.push_back({split + "_rgb", n::zeros({48, 64, 64, 3})});
///   arrays.push_back(
///       {split + "_labels", n::floor(rows.to(n::DType::Float32) / 8.).to(n::DType::Int32)});
///   arrays.push_back({split + "_ids", rows + n::full({}, offset, n::DType::Int32)});
///   offset += 48;
/// }
/// n::save_npz((root / "flowers.npz").string(), arrays);
/// const auto data = n::models::LoadFlowers(root);
/// std::filesystem::remove(root / "flowers.npz");
/// std::filesystem::remove(root);
/// @endcode
/// @par References
/// - Nilsback and Zisserman(2008),Automated Flower Classification over a Large Number of Classes, https://www.robots.ox.ac.uk/~vgg/data/flowers/102/. Six-class/eight-per-class64px official-split subset is an explicit teaching adaptation;not the full102-class benchmark.
/// @see LoadFlowers(), ScoreFlowers()
inline FlowerSplits LoadFlowers(const std::filesystem::path &root) {
  std::map<std::string, nn::Tensor> arrays;
  for (auto &[name, value] : nn::load_npz((root / "flowers.npz").string()))
    arrays[name] = value;
  std::set<int> previous;
  /// @brief Read one prepared split and update the shared discrete ID registry.
  /// @param split Required archive prefix.
  /// @return Valid balanced48-image tensor subset.
  auto read = [&](const std::string &split) {
    FlowerData data{arrays.at(split + "_rgb"), arrays.at(split + "_labels"),
                    arrays.at(split + "_ids")};
    flower_data_detail::Validate(data);
    if (data.rgb.shape()[0] != 48)
      throw std::invalid_argument("48 flowers per prepared split required");
    const auto counts = nn::unsorted_segment_sum(nn::ones_like(data.labels), data.labels, 6);
    if (!nn::all(nn::equal(counts, 8.)).item<bool>())
      throw std::invalid_argument("eight flowers per class required");
    for (const auto id : data.ids.to_vector<int>())
      if (!previous.insert(id).second)
        throw std::invalid_argument("flower IDs must be disjoint");
    return data;
  };
  auto fit = read("fit"), validation = read("validation"), test = read("test");
  return {fit, validation, test};
}
/// @brief Decode selected original images and resize them through tensor bilinear sampling.
///
/// Image decoding is byte ingress. Pixel conversion,division by255 and
/// bilinear64px resizing use tensors;no aspect crop or antialias filter is
/// applied. Images are stretched to a square. No fitting. read_image dispatches
/// by file signature(the example writes a tiny PNG fixture under a jpg name).
///
/// @param root Directory with jpg/image_XXXXX.jpg source images.
/// @param ids Nonempty official image IDs1..8189 in requested output order.
/// @param labels Same-length zero-based class labels0..5.
/// @return Float32[N,64,64,3] normalized RGB plus Int32 labels and IDs.
/// @throws std::invalid_argument For invalid shapes,dtypes,rows,labels,IDs,probabilities or unreadable prepared files.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto root = std::filesystem::temp_directory_path() / "dm-flower-image-example";
/// std::filesystem::create_directories(root / "jpg");
/// n::save_png(n::ones({2, 3, 3}), (root / "jpg/image_00001.jpg").string());
/// const auto data = n::models::ReadFlowerImageRows(root, {1}, {0});
/// std::filesystem::remove(root / "jpg/image_00001.jpg");
/// std::filesystem::remove(root / "jpg");
/// std::filesystem::remove(root);
/// @endcode
/// @par References
/// - Nilsback and Zisserman(2008),Automated Flower Classification over a Large Number of Classes, https://www.robots.ox.ac.uk/~vgg/data/flowers/102/. Six-class/eight-per-class64px official-split subset is an explicit teaching adaptation;not the full102-class benchmark.
/// @see LoadFlowers(), ScoreFlowers()
inline FlowerData ReadFlowerImageRows(const std::filesystem::path &root,
                                      const std::vector<int> &ids,
                                      const std::vector<int> &labels) {
  nn::NoGradGuard guard;
  if (ids.empty() || ids.size() != labels.size())
    throw std::invalid_argument("aligned nonempty flower IDs/labels required");
  std::vector<nn::Tensor> images;
  for (std::size_t i = 0; i < ids.size(); ++i) {
    if (ids[i] < 1 || ids[i] > 8189 || labels[i] < 0 || labels[i] >= 6)
      throw std::invalid_argument("invalid flower ID/label");
    std::ostringstream name;
    name << "image_" << std::setfill('0') << std::setw(5) << ids[i] << ".jpg";
    auto pixels =
        nn::from_image(datamunge::image::read_image((root / "jpg" / name.str()).string()));
    if (pixels.dim() != 3 || pixels.shape()[2] != 3)
      throw std::invalid_argument("RGB flower source image required");
    images.push_back(nn::resize_bilinear(nn::expand_dims(pixels / 255., 0), {64, 64}));
  }
  return {nn::concat(images, 0),
          nn::from_vector(std::vector<std::int32_t>(labels.begin(), labels.end())),
          nn::from_vector(std::vector<std::int32_t>(ids.begin(), ids.end()))};
}
/// @brief Own scalar probability metrics and exact class confusion tensors.
///
/// Detached scalar tensors are Float64;confusion is Int64 with true rows
/// and predicted columns. No forward pass,fitting or mutable running state.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto metrics =
///     n::models::ScoreFlowers(n::full({6, 6}, 1. / 6), n::arange(0, 6, 1, n::DType::Int32));
/// @endcode
/// @par References
/// - Nilsback and Zisserman(2008),Automated Flower Classification over a Large Number of Classes, https://www.robots.ox.ac.uk/~vgg/data/flowers/102/. Six-class/eight-per-class64px official-split subset is an explicit teaching adaptation;not the full102-class benchmark.
/// @see LoadFlowers(), ScoreFlowers()
struct TensorFlowerMetrics {
  nn::Tensor nll;      ///< Float64 scalar mean negative log labelled probability,floor1e-30.
  nn::Tensor accuracy; ///< Float64 scalar correct fraction,first maximum wins ties.
  nn::Tensor macro_accuracy; ///< Float64 scalar mean recall over present true classes.
  nn::Tensor brier;     ///< Float64 scalar mean six-class squared-error sum,not channel mean.
  nn::Tensor confusion; ///< Int64[6,6] exact true-row/prediction-column counts.
};
/// @brief Summarize normalized six-class probabilities with detached tensor reductions.
///
/// Argmax ties select class0 first. Missing true classes are excluded from
/// macro recall. NLL floors labelled probabilities at1e-30. No fitting or
/// input gradients;all probability arithmetic is promoted to Float64.
///
/// @param probability Float32/64[N,6] finite[0,1],row sums within2e-6 of one.
/// @param labels Int32/64[N] class IDs0..5,N positive.
/// @return Float64 scalar NLL,accuracy,present-class macro recall,Brier sum and Int64 confusion.
/// @throws std::invalid_argument For invalid shapes,dtypes,rows,labels,IDs,probabilities or unreadable prepared files.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto metrics =
///     n::models::ScoreFlowers(n::full({6, 6}, 1. / 6), n::arange(0, 6, 1, n::DType::Int32));
/// @endcode
/// @par References
/// - Nilsback and Zisserman(2008),Automated Flower Classification over a Large Number of Classes, https://www.robots.ox.ac.uk/~vgg/data/flowers/102/. Six-class/eight-per-class64px official-split subset is an explicit teaching adaptation;not the full102-class benchmark.
/// @see LoadFlowers(), ScoreFlowers()
inline TensorFlowerMetrics ScoreFlowers(const nn::Tensor &probability,
                                        const nn::Tensor &labels) {
  nn::NoGradGuard guard;
  angular_margin_detail::Validate(probability, labels, 6);
  const auto p = probability.to(nn::DType::Float64);
  if (!nn::all(nn::logical_and(nn::greater_equal(p, 0.), nn::less_equal(p, 1.)))
           .item<bool>() ||
      nn::max(nn::abs(nn::sum(p, {1}) - 1.)).item() > 2e-6)
    throw std::invalid_argument("normalized flower probabilities required");
  const auto target = nn::one_hot(labels, 6, 1., 0., nn::DType::Float64);
  const auto confusion = AngularClassificationConfusion(p, labels);
  const auto true_count = nn::sum(confusion.to(nn::DType::Float64), {1});
  const auto diagonal =
      nn::sum(confusion.to(nn::DType::Float64) * nn::eye(6, nn::DType::Float64), {1});
  const auto present = nn::greater(true_count, 0.);
  const auto recall = diagonal / nn::maximum(true_count, nn::ones_like(true_count));
  return {-nn::mean(nn::log(
              nn::maximum(nn::sum(p * target, {1}), nn::full({}, 1e-30, nn::DType::Float64)))),
          nn::mean(nn::equal(nn::argmax(p, 1), labels).to(nn::DType::Float64)),
          nn::sum(recall) / nn::count_nonzero(present).to(nn::DType::Float64),
          nn::mean(nn::sum(nn::square(p - target), {1})), confusion};
}
/// @brief Compute per-image mean RGB through promoted tensor reductions.
///
/// Accumulate in Float64 then cast to Float32,matching source per-image
/// mean storage. This is descriptive preprocessing,not optimization.
///
/// @param data Positive valid64px RGB flower batch.
/// @return Float32[N,3] mean pixel colors,retaining image gradients.
/// @throws std::invalid_argument For invalid shapes,dtypes,rows,labels,IDs,probabilities or unreadable prepared files.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::FlowerData data{n::zeros({6, 64, 64, 3}), n::arange(0, 6, 1, n::DType::Int32),
///                            n::arange(1, 7, 1, n::DType::Int32)};
/// const auto means = n::models::FlowerColorMeans(data);
/// @endcode
/// @par References
/// - Nilsback and Zisserman(2008),Automated Flower Classification over a Large Number of Classes, https://www.robots.ox.ac.uk/~vgg/data/flowers/102/. Six-class/eight-per-class64px official-split subset is an explicit teaching adaptation;not the full102-class benchmark.
/// @see LoadFlowers(), ScoreFlowers()
inline nn::Tensor FlowerColorMeans(const FlowerData &data) {
  flower_data_detail::Validate(data);
  return nn::mean(data.rgb.to(nn::DType::Float64), {1, 2}).to(nn::DType::Float32);
}
/// @brief Compute a fit-only mean-color prototype reference without CNN features.
///
/// This closed-form reference averages fitting colors within class. Prototype
/// accumulation and distances use Float64 tensors,followed by stable softmax
/// and Float32 output. Small rounding changes from the old host exp/sum loop
/// are intentional. No test labels fit prototypes;no optimizer or CNN training.
///
/// @param fitting Valid observed fitting images with all six labels present.
/// @param testing Valid observed test images;targets are not used in the calculation.
/// @return Float32[Ntest,6] softmax(-squared mean-color distance/.02) probabilities.
/// @throws std::invalid_argument For invalid shapes,dtypes,rows,labels,IDs,probabilities or unreadable prepared files.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::FlowerData data{n::zeros({6, 64, 64, 3}), n::arange(0, 6, 1, n::DType::Int32),
///                            n::arange(1, 7, 1, n::DType::Int32)};
/// const auto probability = n::models::FlowerColorBaseline(data, data);
/// @endcode
/// @par References
/// - Nilsback and Zisserman(2008),Automated Flower Classification over a Large Number of Classes, https://www.robots.ox.ac.uk/~vgg/data/flowers/102/. Six-class/eight-per-class64px official-split subset is an explicit teaching adaptation;not the full102-class benchmark.
/// @see LoadFlowers(), ScoreFlowers()
inline nn::Tensor FlowerColorBaseline(const FlowerData &fitting, const FlowerData &testing) {
  nn::NoGradGuard guard;
  const auto fit = FlowerColorMeans(fitting).to(nn::DType::Float64),
             test = FlowerColorMeans(testing).to(nn::DType::Float64);
  const auto counts = nn::unsorted_segment_sum(
      nn::ones({fitting.labels.numel()}, nn::DType::Float64), fitting.labels, 6);
  if (!nn::all(nn::greater(counts, 0.)).item<bool>())
    throw std::invalid_argument("all fitting color classes required");
  const auto centers =
      nn::unsorted_segment_sum(fit, fitting.labels, 6) / counts.reshape({6, 1});
  const auto distances =
      nn::sum(nn::square(nn::expand_dims(test, 1) - nn::expand_dims(centers, 0)), {2});
  return nn::softmax(-distances / .02, 1).to(nn::DType::Float32);
}
/// @brief Render actual flower predictions and probability bars through tensor masks.
///
/// Two-pixel image borders are green for correct,red for wrong. Six bars
/// below each image have rounded heights0..14;the true class is green,others
/// blue. Image pixels are clamped/rounded to source byte colors. No host pixel
/// arithmetic,model forward,fitting or parameter mutation.
///
/// @param data Valid positive64px RGB images with aligned labels/IDs.
/// @param probability Float32/64[N,6] normalized class probabilities.
/// @return Detached Float32[N,80,64,3] RGB tiles in[0,1].
/// @throws std::invalid_argument For invalid shapes,dtypes,rows,labels,IDs,probabilities or unreadable prepared files.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::FlowerData data{n::zeros({6, 64, 64, 3}), n::arange(0, 6, 1, n::DType::Int32),
///                            n::arange(1, 7, 1, n::DType::Int32)};
/// const auto tiles = n::models::FlowerPredictionTiles(data, n::full({6, 6}, 1. / 6));
/// @endcode
/// @par References
/// - Nilsback and Zisserman(2008),Automated Flower Classification over a Large Number of Classes, https://www.robots.ox.ac.uk/~vgg/data/flowers/102/. Six-class/eight-per-class64px official-split subset is an explicit teaching adaptation;not the full102-class benchmark.
/// @see LoadFlowers(), ScoreFlowers()
inline nn::Tensor FlowerPredictionTiles(const FlowerData &data,
                                        const nn::Tensor &probability) {
  nn::NoGradGuard guard;
  flower_data_detail::Validate(data);
  ScoreFlowers(probability, data.labels);
  const auto n = data.rgb.shape()[0];
  const auto prediction = nn::argmax(probability, 1);
  const auto correct = nn::equal(prediction, data.labels).reshape({n, 1, 1, 1});
  const auto green = nn::tensor({0.f, 1.f, 0.f}).reshape({1, 1, 1, 3}),
             red = nn::tensor({1.f, 0.f, 0.f}).reshape({1, 1, 1, 3}),
             blue = nn::tensor({0.f, 0.f, 1.f}).reshape({1, 1, 1, 3});
  const auto x = nn::arange(0, 64).reshape({1, 1, 64, 1}),
             y = nn::arange(0, 64).reshape({1, 64, 1, 1});
  const auto border =
      nn::logical_or(nn::logical_or(nn::less(x, 2.), nn::greater_equal(x, 62.)),
                     nn::logical_or(nn::less(y, 2.), nn::greater_equal(y, 62.)));
  const auto rgb = nn::floor((nn::clip(data.rgb, 0., 1.) * 255.).to(nn::DType::Float64) + .5).to(nn::DType::Float32) / 255.;
  const auto image = nn::where(border, nn::where(correct, green, red), rgb);
  const auto bar_class =
      nn::clip(nn::floor(nn::arange(0, 64) / 10.), 0., 5.).to(nn::DType::Int64);
  const auto heights =
      nn::floor(nn::gather(probability.to(nn::DType::Float32), bar_class, 1) * 14. + .5)
          .reshape({n, 1, 64, 1});
  const auto by = nn::arange(0, 16).reshape({1, 16, 1, 1});
  const auto inside =
      nn::logical_and(nn::less(x, 60.), nn::less(x - nn::floor(x / 10.) * 10., 8.));
  const auto drawn = nn::logical_and(inside, nn::greater_equal(by, 16. - heights));
  const auto truth =
      nn::equal(bar_class.reshape({1, 1, 64, 1}), data.labels.reshape({n, 1, 1, 1}));
  const auto bars = nn::where(drawn, nn::where(truth, green, blue), nn::zeros({n, 16, 64, 3}));
  return nn::concat({image, bars}, 1);
}
} // namespace datamunge::nn::models
