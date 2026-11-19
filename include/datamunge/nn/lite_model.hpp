#pragma once

/// @file lite_model.hpp
/// @brief Optional LiteRT CPU inference for dense Float32 TensorFlow Lite models.

#include <array>
#include <memory>
#include <string>

#include <datamunge/nn/tensor.hpp>

namespace datamunge::nn {

/// @brief Check whether LiteRT inference is enabled in this Datamunge build.
/// @return True if the core links LiteRT.
/// @par Example
/// @code
/// std::println("{}", dm::nn::lite_model_runtime_available());
/// @endcode
/// @see LiteModel
bool lite_model_runtime_available() noexcept;

/// @brief Compile a local TensorFlow Lite model once and reuse its CPU buffers.
///
/// Supports a named signature with one dense Float32 input and output at the
/// runtime's initial concrete shapes. Copies share the compiled model and its
/// buffers. Calls to predict() on those copies serialize access to the buffers.
/// @par Example
/// @code
/// const dm::nn::LiteModel model("model.tflite");
/// auto prediction = model.predict(dm::nn::zeros(model.input_shape()));
/// @endcode
/// @par References
/// - Google LiteRT SDK, https://developers.google.com/edge/litert/next/cpp_sdk
/// @see lite_model_runtime_available()
class LiteModel {
 public:
  /// @brief Load and compile a local FlatBuffer model for CPU inference.
  /// @param path Trusted local TensorFlow Lite model file.
  /// @param signature Logical signature key.
  /// @param input Logical input key.
  /// @param output Logical output key.
  /// @throws std::invalid_argument Unknown signature or incompatible tensor layout.
  /// @throws std::runtime_error Disabled backend, malformed model, or LiteRT failure.
  /// @par Example
  /// @code
  /// const dm::nn::LiteModel model("model.tflite", "serving_default", "features", "predictions");
  /// @endcode
  explicit LiteModel(const std::string& path, const std::string& signature = "serving_default",
                     const std::string& input = "features", const std::string& output = "predictions");

  /// @brief Get the compiled dense input shape.
  /// @return Copy of the concrete input dimensions.
  [[nodiscard]] Shape input_shape() const;

  /// @brief Get the compiled dense output shape.
  /// @return Copy of the concrete output dimensions.
  [[nodiscard]] Shape output_shape() const;

  /// @brief Get the allocated input and output buffer capacities.
  /// @return Buffer sizes in bytes, in input/output order.
  [[nodiscard]] std::array<std::size_t, 2> buffer_sizes() const;

  /// @brief Invoke the compiled signature synchronously and copy its result.
  /// @param features CPU Float32 tensor matching input_shape().
  /// @return Detached owning Float32 prediction tensor with output_shape().
  /// @throws std::invalid_argument Incompatible shape, dtype, or device.
  /// @throws std::runtime_error LiteRT buffer or execution failure.
  /// @par Example
  /// @code
  /// const dm::nn::LiteModel model("model.tflite");
  /// const auto y = model.predict(dm::nn::zeros(model.input_shape()));
  /// @endcode
  [[nodiscard]] Tensor predict(const Tensor& features) const;

 private:
  /// @brief Shared implementation owning the runtime model and reusable buffers.
  struct Impl;
  std::shared_ptr<Impl> impl_; ///< Owned executable state.
};

} // namespace datamunge::nn
