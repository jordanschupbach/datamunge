/// @file lite_model.cpp
/// @brief LiteRT compiled CPU signature and reusable buffer ownership.

#include <datamunge/nn/lite_model.hpp>

#include <stdexcept>

#ifdef DATAMUNGE_ENABLE_LITERT
#include <algorithm>
#include <cstring>
#include <limits>
#include <mutex>
#include <string_view>

#include <litert/c/litert_compiled_model.h>
#include <litert/c/litert_environment.h>
#include <litert/c/litert_model.h>
#include <litert/c/litert_options.h>
#include <litert/c/litert_tensor_buffer.h>
#include <litert/c/litert_tensor_buffer_requirements.h>
#endif

namespace datamunge::nn {

bool lite_model_runtime_available() noexcept {
#ifdef DATAMUNGE_ENABLE_LITERT
  return true;
#else
  return false;
#endif
}

#ifdef DATAMUNGE_ENABLE_LITERT
namespace {

/// @brief Surface a LiteRT failure with the operation that triggered it.
void check(LiteRtStatus status, const char* operation) {
  if (status != kLiteRtStatusOk)
    throw std::runtime_error("LiteModel: " + std::string(operation) +
                             " failed (LiteRT status " + std::to_string(status) + ")");
}

/// @brief Own a unique SDK handle with its corresponding destroy function.
template<class Handle, void (*Destroy)(Handle)>
struct Owner {
  Handle value{}; ///< SDK handle, zero until initialized.
  Owner() = default; ///< Start without a handle.
  Owner(const Owner&) = delete; ///< Ownership is unique.
  Owner& operator=(const Owner&) = delete; ///< Ownership is unique.
  ~Owner() { if (value) Destroy(value); } ///< Release the SDK handle.
};

/// @brief Find a logical signature and retain its index for compiled calls.
std::pair<LiteRtSignature, LiteRtParamIndex> find_signature(LiteRtModel model, std::string_view key) {
  LiteRtParamIndex count = 0;
  check(LiteRtGetNumModelSignatures(model, &count), "count signatures");
  for (LiteRtParamIndex i = 0; i < count; ++i) {
    LiteRtSignature signature{};
    const char* name = nullptr;
    check(LiteRtGetModelSignature(model, i, &signature), "get signature");
    check(LiteRtGetSignatureKey(signature, &name), "get signature key");
    if (name && key == name) return {signature, i};
  }
  throw std::invalid_argument("LiteModel: unknown signature " + std::string(key));
}

/// @brief Check a ranked concrete Float32 layout and calculate its packed bytes.
std::pair<Shape, std::size_t> packed_layout(const LiteRtRankedTensorType& type) {
  if (type.element_type != kLiteRtElementTypeFloat32 || type.layout.has_strides ||
      type.layout.rank > std::size(type.layout.dimensions))
    throw std::invalid_argument("LiteModel: expected dense Float32 buffers");
  Shape dimensions;
  std::size_t bytes = sizeof(float);
  for (unsigned i = 0; i < type.layout.rank; ++i) {
    const auto dim = type.layout.dimensions[i];
    if (dim <= 0 || bytes > std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(dim))
      throw std::invalid_argument("LiteModel: invalid concrete dimension or buffer size");
    bytes *= static_cast<std::size_t>(dim);
    dimensions.push_back(dim);
  }
  return {std::move(dimensions), bytes};
}

} // namespace
#endif

/// @brief Own immutable layout metadata and opaque compiled model resources.
struct LiteModel::Impl {
  Shape input_shape; ///< Compiled input dimensions.
  Shape output_shape; ///< Compiled output dimensions.
  std::array<std::size_t, 2> capacities{}; ///< Required buffer sizes in bytes.
#ifdef DATAMUNGE_ENABLE_LITERT
  Owner<LiteRtEnvironment, LiteRtDestroyEnvironment> environment; ///< Outlives model and buffers.
  Owner<LiteRtModel, LiteRtDestroyModel> model; ///< Parsed model.
  Owner<LiteRtCompiledModel, LiteRtDestroyCompiledModel> compiled; ///< CPU executable.
  Owner<LiteRtTensorBuffer, LiteRtDestroyTensorBuffer> input; ///< Reused input buffer.
  Owner<LiteRtTensorBuffer, LiteRtDestroyTensorBuffer> output; ///< Reused output buffer.
  LiteRtParamIndex signature_index{}; ///< Signature invocation index.
  std::array<std::size_t, 2> payload{}; ///< Packed tensor sizes in bytes.
  std::mutex invocation; ///< Serializes access to the compiled model and buffers.
#endif
};

LiteModel::LiteModel(const std::string& path, const std::string& signature,
                     const std::string& input, const std::string& output) {
#ifdef DATAMUNGE_ENABLE_LITERT
  auto state = std::make_shared<Impl>();
  check(LiteRtCreateEnvironment(0, nullptr, &state->environment.value), "create environment");
  check(LiteRtCreateModelFromFile(state->environment.value, path.c_str(), &state->model.value), "load FlatBuffer");
  const auto [entry, index] = find_signature(state->model.value, signature);
  state->signature_index = index;
  LiteRtParamIndex inputs = 0, outputs = 0;
  check(LiteRtGetNumSignatureInputs(entry, &inputs), "count inputs");
  check(LiteRtGetNumSignatureOutputs(entry, &outputs), "count outputs");
  const char* input_name = nullptr;
  const char* output_name = nullptr;
  if (inputs != 1 || outputs != 1)
    throw std::invalid_argument("LiteModel: expected one input and one output");
  check(LiteRtGetSignatureInputName(entry, 0, &input_name), "get logical input name");
  check(LiteRtGetSignatureOutputName(entry, 0, &output_name), "get logical output name");
  if (!input_name || !output_name || input != input_name || output != output_name)
    throw std::invalid_argument("LiteModel: signature keys do not match");

  Owner<LiteRtOptions, LiteRtDestroyOptions> options;
  check(LiteRtCreateOptions(&options.value), "create options");
  check(LiteRtSetOptionsHardwareAccelerators(options.value, kLiteRtHwAcceleratorCpu), "select CPU");
  check(LiteRtCreateCompiledModel(state->environment.value, state->model.value,
                                  options.value, &state->compiled.value), "compile model");

  LiteRtTensor in_tensor{}, out_tensor{};
  LiteRtRankedTensorType in_type{}, out_type{};
  check(LiteRtGetSignatureInputTensorByIndex(entry, 0, &in_tensor), "get input tensor");
  check(LiteRtGetSignatureOutputTensorByIndex(entry, 0, &out_tensor), "get output tensor");
  check(LiteRtGetRankedTensorType(in_tensor, &in_type), "get input type");
  check(LiteRtGetRankedTensorType(out_tensor, &out_type), "get output type");
  check(LiteRtGetCompiledModelInputTensorLayout(state->compiled.value, index, 0, &in_type.layout), "get input layout");
  check(LiteRtGetCompiledModelOutputTensorLayouts(state->compiled.value, index, 1, &out_type.layout, true), "get output layout");
  auto [input_shape, input_bytes] = packed_layout(in_type);
  auto [output_shape, output_bytes] = packed_layout(out_type);
  state->input_shape = std::move(input_shape);
  state->output_shape = std::move(output_shape);
  state->payload = {input_bytes, output_bytes};

  LiteRtTensorBufferRequirements in_req{}, out_req{};
  check(LiteRtGetCompiledModelInputBufferRequirements(state->compiled.value, index, 0, &in_req), "get input requirements");
  check(LiteRtGetCompiledModelOutputBufferRequirements(state->compiled.value, index, 0, &out_req), "get output requirements");
  check(LiteRtGetTensorBufferRequirementsBufferSize(in_req, &state->capacities[0]), "get input capacity");
  check(LiteRtGetTensorBufferRequirementsBufferSize(out_req, &state->capacities[1]), "get output capacity");
  if (state->capacities[0] < input_bytes || state->capacities[1] < output_bytes)
    throw std::runtime_error("LiteModel: required buffer is smaller than the tensor");
  check(LiteRtCreateManagedTensorBufferFromRequirements(state->environment.value, &in_type, in_req,
                                                         &state->input.value), "allocate input buffer");
  check(LiteRtCreateManagedTensorBufferFromRequirements(state->environment.value, &out_type, out_req,
                                                         &state->output.value), "allocate output buffer");
  impl_ = std::move(state);
#else
  (void)path;
  (void)signature;
  (void)input;
  (void)output;
  throw std::runtime_error("LiteModel: rebuild with DATAMUNGE_ENABLE_LITERT=ON");
#endif
}

Shape LiteModel::input_shape() const { return impl_->input_shape; }
Shape LiteModel::output_shape() const { return impl_->output_shape; }
std::array<std::size_t, 2> LiteModel::buffer_sizes() const { return impl_->capacities; }

Tensor LiteModel::predict(const Tensor& features) const {
#ifdef DATAMUNGE_ENABLE_LITERT
  if (!features.defined() || features.device().backend != compute::Backend::CPU ||
      features.dtype() != DType::Float32 || features.shape() != impl_->input_shape)
    throw std::invalid_argument("LiteModel: incompatible input shape, dtype or device");
  const auto packed = features.contiguous();
  auto result = empty(impl_->output_shape, DType::Float32);
  const auto* input_data = packed.data_ptr<float>();
  auto* output_data = result.data_ptr<float>();
  std::lock_guard lock(impl_->invocation);
  void* data = nullptr;
  check(LiteRtLockTensorBuffer(impl_->input.value, &data, kLiteRtTensorBufferLockModeWrite), "lock input");
  std::memcpy(data, input_data, impl_->payload[0]);
  check(LiteRtUnlockTensorBuffer(impl_->input.value), "unlock input");
  check(LiteRtRunCompiledModel(impl_->compiled.value, impl_->signature_index,
                               1, &impl_->input.value, 1, &impl_->output.value), "run signature");
  LiteRtLayout layout{};
  check(LiteRtGetCompiledModelOutputTensorLayouts(impl_->compiled.value, impl_->signature_index, 1,
                                                  &layout, false), "check output layout");
  if (layout.has_strides || layout.rank != impl_->output_shape.size() ||
      !std::equal(impl_->output_shape.begin(), impl_->output_shape.end(), layout.dimensions))
    throw std::runtime_error("LiteModel: output layout changed during invocation");
  check(LiteRtLockTensorBuffer(impl_->output.value, &data, kLiteRtTensorBufferLockModeRead), "lock output");
  std::memcpy(output_data, data, impl_->payload[1]);
  check(LiteRtUnlockTensorBuffer(impl_->output.value), "unlock output");
  return result;
#else
  (void)features;
  throw std::runtime_error("LiteModel: rebuild with DATAMUNGE_ENABLE_LITERT=ON");
#endif
}

} // namespace datamunge::nn
