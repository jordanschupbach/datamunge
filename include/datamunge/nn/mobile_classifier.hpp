#pragma once
/// @file mobile_classifier.hpp
/// @brief Trainable compact MobileNet-v1 image classifiers with complete inference checkpoints.
#include <datamunge/nn/tensor.hpp>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace datamunge::nn {
/// @brief One epoch's owned training and validation measurements.
struct MobileClassifierEpoch;
/// @brief CPU minibatch fitting settings for MobileNetClassifier.
/// @par Example
/// @code
/// dm::nn::MobileClassifierOptions options; options.epochs=20;
/// @endcode
struct MobileClassifierOptions {
  std::size_t epochs{20};             ///< Positive epoch count.
  std::size_t batch_size{16};         ///< Positive maximum minibatch size.
  double      learning_rate{.003};    ///< Finite positive Adam learning rate.
  double      brightness_jitter{.15}; ///< Independent training-image gain jitter in [0,.5].
  double      grayscale_probability{
      0};            ///< Independent training-image luminance conversion probability in [0,1].
  std::uint64_t seed{17}; ///< Minibatch order, image-gain and grayscale augmentation seed.
  std::function<void(const MobileClassifierEpoch&)>
      progress; ///< Optional synchronous epoch-end observer; sees measured losses without changing
                ///< validation selection. Must not mutate/reenter this model; callback exceptions
                ///< propagate with partial updates retained.
};
/// @brief One measured training epoch, with saved-BN-statistic validation results.
/// @par Example
/// @code
/// dm::nn::MobileClassifierEpoch epoch;
/// std::print("{}\n",epoch.epoch); // 0
/// @endcode
struct MobileClassifierEpoch {
  std::size_t epoch{0};               ///< One-based epoch index.
  double      training_loss{0};       ///< Sample-weighted mean cross entropy before each update.
  double      validation_loss{0};     ///< Frozen-mode validation cross entropy after the epoch.
  double      validation_accuracy{0}; ///< Correct argmax fraction, without augmentation.
};
/// @brief Fitting evidence; the model restores the epoch with lowest validation loss.
/// @par Example
/// @code
/// dm::nn::MobileClassifierTraining result;
/// std::print("{}\n",result.selected_epoch); // 0
/// @endcode
struct MobileClassifierTraining {
  std::vector<MobileClassifierEpoch> history;           ///< Every requested epoch, in order.
  std::size_t                        selected_epoch{0}; ///< One-based validation-selected epoch.
};
/// @brief Full MobileNet-v1 with a configurable narrow width and native Adam/autograd training.
/// Uses all thirteen depthwise/pointwise blocks, ReLU6, moving-statistic BN and
/// global average pooling. CPU float32 inputs are NHWC RGB in [0,1]; the forward
/// normalizes them to [-1,1]. Default width .125 and side 64 make a compact model.
/// This is trained from scratch, without pretrained weights. The checkpoint
/// includes architecture, every learned tensor and every BN moving moment.
/// Single-writer, including inference; fitting is not resumable from optimizer state.
/// @par Example
/// @code
/// dm::nn::MobileNetClassifier model(2,64,.125,17);
/// auto probabilities=model.predict(dm::nn::zeros({1,64,64,3}));
/// model.save("mobile.npz");
/// auto restored=dm::nn::MobileNetClassifier::load("mobile.npz");
/// @endcode
/// @par References
/// - Howard et al. (2017), MobileNets: Efficient Convolutional Neural Networks
///   for Mobile Vision Applications. https://arxiv.org/abs/1704.04861
class MobileNetClassifier {
 public:
  /// @brief Initialize a real width-scaled MobileNet-v1 and its BN state.
  /// @param classes Output classes, 2..1000; default two.
  /// @param side RGB image side, 16..256 and divisible by 16; default 64.
  /// @param width Finite width multiplier in [.03125,1]; default .125.
  /// @param seed Parameter initialization seed, default 17.
  /// @throws std::invalid_argument Invalid architecture settings.
  /// @par Example
  /// See the class example.
  explicit MobileNetClassifier(
      int           classes = 2,
      int           side    = 64,
      double        width   = .125,
      std::uint64_t seed    = 17);
  /// @brief Release the owned parameters and moving moments.
  ~MobileNetClassifier();
  /// @brief Transfer exclusive model ownership; the source becomes empty.
  /// @param other Model to move.
  MobileNetClassifier(MobileNetClassifier&& other) noexcept;
  /// @brief Replace this model by moving exclusive ownership.
  /// @param other Model to move.
  /// @return This model.
  MobileNetClassifier& operator=(MobileNetClassifier&& other) noexcept;
  /// @brief Fit every layer using Adam and select an epoch solely on validation loss.
  /// Brightness augmentation multiplies each training image by a seeded random
  /// gain and clamps to [0,1]; optional seeded grayscale augmentation uses luminance
  /// .299R+.587G+.114B on each training image. Validation is never augmented or used for gradients.
  /// Inputs are checked before any training mutation; solver errors may leave
  /// partial training updates. The optional progress observer runs after each measured epoch;
  /// its exceptions propagate before best-checkpoint restoration. At completion both weights and BN
  /// moments are restored together to the selected epoch.
  /// @param images Nonempty CPU float32 [N,side,side,3], finite values in [0,1].
  /// @param labels CPU int64 [N], each in [0,classes).
  /// @param validation Independent nonempty images with the same pixel contract.
  /// @param validation_labels Corresponding CPU int64 labels.
  /// @param options Valid positive training settings.
  /// @return Full per-epoch evidence and the selected epoch index.
  /// @throws std::invalid_argument Invalid pixels, shapes, labels or settings, before mutation.
  /// @throws std::runtime_error Nonfinite training/validation loss.
  /// @par Example
  /// @code
  /// dm::nn::MobileNetClassifier model(2,16,.0625,17);
  /// auto x=dm::nn::zeros({2,16,16,3});
  /// auto y=dm::nn::from_vector(std::vector<std::int64_t>{0,1});
  /// dm::nn::MobileClassifierOptions options; options.epochs=1;
  /// auto fitted=model.fit(x,y,x,y,options); // Use independent validation data in real work.
  /// @endcode
  MobileClassifierTraining fit(
      const Tensor&           images,
      const Tensor&           labels,
      const Tensor&           validation,
      const Tensor&           validation_labels,
      MobileClassifierOptions options = {});
  /// @brief Predict class probabilities using only RGB pixels and saved BN moments.
  /// @param images Nonempty CPU float32 NHWC batch matching the constructor's side.
  /// @return Independent CPU float32 [N,classes] softmax probabilities, without a graph.
  /// @throws std::invalid_argument Invalid shape, dtype or pixel values.
  /// @par Example
  /// See the class example.
  Tensor predict(const Tensor& images);
  /// @brief Read the number of learned scalar parameters, excluding BN moving moments.
  /// @return Positive parameter count.
  std::int64_t parameter_count() const;
  /// @brief Write a complete, versioned inference checkpoint as a native NumPy NPZ archive.
  /// @param path Destination .npz path; parent directory must exist.
  /// @throws std::runtime_error Checkpoint I/O failure.
  /// @par Example
  /// See the class example.
  void save(const std::string& path) const;
  /// @brief Load architecture, finite weights and BN moments; reject missing/extra tensors.
  /// @param path Existing inference NPZ checkpoint produced by save().
  /// @return Independent owned model ready for deterministic prediction.
  /// @throws std::runtime_error Invalid checkpoint, tensor names/shapes/dtypes or I/O error.
  /// @par Example
  /// See the class example.
  static MobileNetClassifier load(const std::string& path);

 private:
  struct Impl;                 ///< Owned architecture, trainable parameters and inference BN state.
  std::unique_ptr<Impl> impl_; ///< Exclusive mutable model; never shared implicitly.
};
} // namespace datamunge::nn
