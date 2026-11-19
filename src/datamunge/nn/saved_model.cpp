/// @file saved_model.cpp
/// @brief Optional TensorFlow SavedModel loading and validated inference.

#include <datamunge/nn/saved_model.hpp>

#include <stdexcept>

#ifdef DATAMUNGE_ENABLE_TENSORFLOW
#include <algorithm>
#include <cstring>
#include <tensorflow/cc/saved_model/loader.h>
#endif

namespace datamunge::nn {

bool saved_model_runtime_available() noexcept {
#ifdef DATAMUNGE_ENABLE_TENSORFLOW
  return true;
#else
  return false;
#endif
}

#ifdef DATAMUNGE_ENABLE_TENSORFLOW
namespace {

/// @brief Resolve a dense single Float32 TensorInfo with a declared rank.
Shape signature_shape(const tensorflow::TensorInfo& tensor) {
  if (tensor.encoding_case() != tensorflow::TensorInfo::kName || tensor.name().empty() ||
      tensor.dtype() != tensorflow::DT_FLOAT || !tensor.has_tensor_shape() ||
      tensor.tensor_shape().unknown_rank())
    throw std::invalid_argument("SavedModel: expected a dense Float32 signature with known rank");
  Shape shape;
  for (const auto& dim : tensor.tensor_shape().dim()) {
    if (dim.size() < -1) throw std::invalid_argument("SavedModel: invalid signature dimension");
    shape.push_back(dim.size());
  }
  return shape;
}

/// @brief Compare a concrete shape with a signature containing wildcard axes.
bool matches(const Shape& actual, const Shape& signature) {
  return std::ranges::equal(actual, signature, [](auto size, auto spec) {
    return spec == -1 || size == spec;
  });
}

} // namespace
#endif

/// @brief Store serving signature metadata and own the restored runtime session.
struct SavedModel::Impl {
  Shape input_shape; ///< Declared input dimensions.
  Shape output_shape; ///< Declared output dimensions.
#ifdef DATAMUNGE_ENABLE_TENSORFLOW
  tensorflow::SavedModelBundle bundle; ///< TensorFlow model and restored session.
  std::string input; ///< Resolved graph tensor name for feeds.
  std::string output; ///< Resolved graph tensor name for fetches.
#endif
};

SavedModel::SavedModel(const std::string& directory, const std::string& signature,
                       const std::string& input, const std::string& output) {
#ifdef DATAMUNGE_ENABLE_TENSORFLOW
  auto model = std::make_shared<Impl>();
  tensorflow::SessionOptions options;
  (*options.config.mutable_device_count())["GPU"] = 0;
  const auto status = tensorflow::LoadSavedModel(options, {}, directory, {"serve"}, &model->bundle);
  if (!status.ok()) throw std::runtime_error("SavedModel: " + status.ToString());
  const auto& signatures = model->bundle.meta_graph_def.signature_def();
  const auto entry = signatures.find(signature);
  if (entry == signatures.end()) throw std::invalid_argument("SavedModel: unknown signature " + signature);
  const auto& inputs = entry->second.inputs();
  const auto& outputs = entry->second.outputs();
  if (inputs.size() != 1 || !inputs.contains(input) || !outputs.contains(output))
    throw std::invalid_argument("SavedModel: expected one named input and the requested output");
  model->input_shape = signature_shape(inputs.at(input));
  model->output_shape = signature_shape(outputs.at(output));
  model->input = inputs.at(input).name();
  model->output = outputs.at(output).name();
  impl_ = std::move(model);
#else
  (void)directory;
  (void)signature;
  (void)input;
  (void)output;
  throw std::runtime_error("SavedModel: rebuild with DATAMUNGE_ENABLE_TENSORFLOW=ON");
#endif
}

Shape SavedModel::input_shape() const { return impl_->input_shape; }
Shape SavedModel::output_shape() const { return impl_->output_shape; }

Tensor SavedModel::predict(const Tensor& features) const {
#ifdef DATAMUNGE_ENABLE_TENSORFLOW
  if (!features.defined() || features.device().backend != compute::Backend::CPU ||
      features.dtype() != DType::Float32 || !matches(features.shape(), impl_->input_shape))
    throw std::invalid_argument("SavedModel: incompatible input shape, dtype or device");
  tensorflow::TensorShape shape;
  for (const auto dim : features.shape()) shape.AddDim(dim);
  tensorflow::Tensor input(tensorflow::DT_FLOAT, shape);
  const auto packed = features.contiguous();
  if (packed.numel())
    std::memcpy(input.flat<float>().data(), packed.data_ptr<float>(), packed.numel() * sizeof(float));
  std::vector<tensorflow::Tensor> outputs;
  const auto status = impl_->bundle.session->Run({{impl_->input, input}}, {impl_->output}, {}, &outputs);
  if (!status.ok()) throw std::runtime_error("SavedModel: " + status.ToString());
  if (outputs.size() != 1 || outputs.front().dtype() != tensorflow::DT_FLOAT)
    throw std::runtime_error("SavedModel: incompatible output dtype or count");
  const auto& output = outputs.front();
  Shape output_shape;
  for (int axis = 0; axis < output.dims(); ++axis)
    output_shape.push_back(output.dim_size(axis));
  if (!matches(output_shape, impl_->output_shape))
    throw std::runtime_error("SavedModel: incompatible output shape");
  auto result = empty(output_shape, DType::Float32);
  if (result.numel())
    std::memcpy(result.data_ptr<float>(), output.flat<float>().data(), result.numel() * sizeof(float));
  return result;
#else
  (void)features;
  throw std::runtime_error("SavedModel: rebuild with DATAMUNGE_ENABLE_TENSORFLOW=ON");
#endif
}

} // namespace datamunge::nn
