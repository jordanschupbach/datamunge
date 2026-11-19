#pragma once

/// @file saved_model.hpp
/// @brief Optional TensorFlow SavedModel serving for CPU Float32 tensors.

#include <memory>
#include <string>

#include <datamunge/nn/tensor.hpp>

namespace datamunge::nn {

/// @brief Report whether this Datamunge build links the TensorFlow serving backend.
/// @return True when built with DATAMUNGE_ENABLE_TENSORFLOW.
/// @par Example
/// @code
/// std::println("{}", dm::nn::saved_model_runtime_available());
/// @endcode
/// @see SavedModel
bool saved_model_runtime_available() noexcept;

/// @brief Own a serving session for a trusted local TensorFlow SavedModel.
///
/// Select exactly one named Float32 input and output from a SignatureDef.
/// Shapes use -1 for dynamic axes. Copies share the loaded inference session.
/// @par Example
/// @code
/// const dm::nn::SavedModel model("saved_model");
/// const auto y = model.predict(dm::nn::zeros({1, 2}));
/// @endcode
/// @par References
/// - TensorFlow SavedModel, https://www.tensorflow.org/guide/saved_model
/// @see saved_model_runtime_available()
class SavedModel {
 public:
  /// @brief Load the serve-tagged MetaGraph and resolve named Float32 tensors.
  /// @param directory Trusted local directory containing saved_model.pb and its data.
  /// @param signature SignatureDef key, default serving_default.
  /// @param input Logical input key, default features.
  /// @param output Logical output key, default predictions.
  /// @throws std::invalid_argument Missing signature, incompatible tensor metadata or additional required inputs.
  /// @throws std::runtime_error Backend unavailable or TensorFlow failed to load the model.
  /// @par Example
  /// @code
  /// const dm::nn::SavedModel model("saved_model", "serving_default", "features", "predictions");
  /// @endcode
  explicit SavedModel(const std::string& directory,
                      const std::string& signature = "serving_default",
                      const std::string& input = "features",
                      const std::string& output = "predictions");

  /// @brief Return the input shape declared by the serving signature.
  /// @return Copy of the input dimensions, with -1 for a dynamic dimension.
  [[nodiscard]] Shape input_shape() const;

  /// @brief Return the output shape declared by the serving signature.
  /// @return Copy of the output dimensions, with -1 for a dynamic dimension.
  [[nodiscard]] Shape output_shape() const;

  /// @brief Run the selected output and return its values as a detached CPU tensor.
  /// @param features Defined CPU Float32 input matching the signature rank and static dimensions.
  /// @return Owned Float32 output matching the serving signature.
  /// @throws std::invalid_argument Incompatible input shape, dtype or device.
  /// @throws std::runtime_error TensorFlow failure or inconsistent output.
  /// @par Example
  /// @code
  /// const dm::nn::SavedModel model("saved_model");
  /// const auto y = model.predict(dm::nn::zeros({4, 2}));
  /// @endcode
  [[nodiscard]] Tensor predict(const Tensor& features) const;

 private:
  /// @brief Shared session and metadata implementation, isolated from SDK headers.
  struct Impl;
  std::shared_ptr<Impl> impl_; ///< Retained serving session.
};

} // namespace datamunge::nn
