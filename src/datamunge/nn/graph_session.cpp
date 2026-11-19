/// @file graph_session.cpp
/// @brief Optional CPU TensorFlow graph execution with partition metadata.

#include <datamunge/nn/graph_session.hpp>

#include <fstream>
#include <locale>
#include <stdexcept>

#ifdef DATAMUNGE_ENABLE_TENSORFLOW
#include <cstring>
#include <tensorflow/c/c_api_experimental.h>
#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/public/session.h>
#endif

namespace datamunge::nn {

#ifdef DATAMUNGE_ENABLE_TENSORFLOW
namespace {

/// @brief Convert a TensorFlow failure into a Datamunge exception.
void check(const absl::Status& status) {
  if (!status.ok()) throw std::runtime_error("GraphSession: " + status.ToString());
}

/// @brief Copy graph input storage to a dense TensorFlow Float32 tensor.
tensorflow::Tensor to_float_tensor(const Tensor& tensor) {
  if (!tensor.defined() || tensor.device().backend != compute::Backend::CPU ||
      tensor.dtype() != DType::Float32)
    throw std::invalid_argument("GraphSession: expected a CPU Float32 feed");
  tensorflow::TensorShape shape;
  for (const auto dim : tensor.shape()) shape.AddDim(dim);
  tensorflow::Tensor result(tensorflow::DT_FLOAT, shape);
  const auto packed = tensor.contiguous();
  if (packed.numel())
    std::memcpy(result.flat<float>().data(), packed.data_ptr<float>(), packed.numel() * sizeof(float));
  return result;
}

/// @brief Copy a TensorFlow Float32 result to an owning CPU tensor.
Tensor from_float_tensor(const tensorflow::Tensor& tensor) {
  if (tensor.dtype() != tensorflow::DT_FLOAT)
    throw std::runtime_error("GraphSession: expected a Float32 fetch");
  Shape shape;
  for (int axis = 0; axis < tensor.dims(); ++axis) shape.push_back(tensor.dim_size(axis));
  auto result = empty(shape, DType::Float32);
  if (result.numel())
    std::memcpy(result.data_ptr<float>(), tensor.flat<float>().data(), result.numel() * sizeof(float));
  return result;
}

} // namespace
#endif

/// @brief Retain the runtime session without exposing TensorFlow headers publicly.
struct GraphSession::Impl {
#ifdef DATAMUNGE_ENABLE_TENSORFLOW
  std::unique_ptr<tensorflow::Session> session; ///< Executable TensorFlow graph.
#endif
};

void enable_tensorflow_cpu_jit() {
#ifdef DATAMUNGE_ENABLE_TENSORFLOW
  TF_SetTfXlaCpuGlobalJit(true);
  TF_SetXlaEnableLazyCompilation(false);
  TF_SetXlaMinClusterSize(2);
#else
  throw std::runtime_error("GraphSession: rebuild with DATAMUNGE_ENABLE_TENSORFLOW=ON");
#endif
}

GraphSession::GraphSession(const std::string& path, GraphOptimization mode)
    : GraphSession(path, mode, 2, 2) {}

GraphSession::GraphSession(const std::string& path, GraphOptimization mode,
                           int intra_threads, int inter_threads) {
#ifdef DATAMUNGE_ENABLE_TENSORFLOW
  if (mode != GraphOptimization::Baseline && mode != GraphOptimization::Grappler &&
      mode != GraphOptimization::Xla)
    throw std::invalid_argument("GraphSession: unknown optimization policy");
  if (intra_threads <= 0 || inter_threads <= 0)
    throw std::invalid_argument("GraphSession: thread counts must be positive");
  tensorflow::GraphDef graph;
  check(tensorflow::ReadBinaryProto(tensorflow::Env::Default(), path, &graph));
  tensorflow::SessionOptions options;
  (*options.config.mutable_device_count())["GPU"] = 0;
  options.config.set_intra_op_parallelism_threads(intra_threads);
  options.config.set_inter_op_parallelism_threads(inter_threads);
  options.config.set_use_per_session_threads(true);
  auto* settings = options.config.mutable_graph_options();
  auto* rewriter = settings->mutable_rewrite_options();
  rewriter->set_disable_meta_optimizer(mode == GraphOptimization::Baseline);
  rewriter->set_min_graph_nodes(-1);
  rewriter->set_constant_folding(tensorflow::RewriterConfig::ON);
  settings->mutable_optimizer_options()->set_global_jit_level(
      mode == GraphOptimization::Xla ? tensorflow::OptimizerOptions::ON_1
                                    : tensorflow::OptimizerOptions::OFF);
  auto state = std::make_shared<Impl>();
  state->session.reset(tensorflow::NewSession(options));
  if (!state->session) throw std::runtime_error("GraphSession: could not open session");
  check(state->session->Create(graph));
  impl_ = std::move(state);
#else
  (void)path;
  (void)mode;
  (void)intra_threads;
  (void)inter_threads;
  throw std::runtime_error("GraphSession: rebuild with DATAMUNGE_ENABLE_TENSORFLOW=ON");
#endif
}

NamedTensors GraphSession::run(const NamedTensors& feeds, const std::vector<std::string>& fetches,
    std::vector<std::pair<std::string, std::string>>* partition_nodes) const {
  return run(feeds, fetches, partition_nodes, nullptr);
}

NamedTensors GraphSession::run(const NamedTensors& feeds, const std::vector<std::string>& fetches,
    std::vector<std::pair<std::string, std::string>>* partition_nodes,
    std::vector<GraphTraceEvent>* trace) const {
#ifdef DATAMUNGE_ENABLE_TENSORFLOW
  std::vector<std::pair<std::string, tensorflow::Tensor>> inputs;
  inputs.reserve(feeds.size());
  for (const auto& [name, value] : feeds) inputs.emplace_back(name, to_float_tensor(value));
  std::vector<tensorflow::Tensor> outputs;
  tensorflow::RunMetadata metadata;
  if (partition_nodes || trace) {
    tensorflow::RunOptions options;
    options.set_output_partition_graphs(partition_nodes != nullptr);
    if (trace) options.set_trace_level(tensorflow::RunOptions::FULL_TRACE);
    check(impl_->session->Run(options, inputs, fetches, {}, &outputs, &metadata));
  } else {
    check(impl_->session->Run(inputs, fetches, {}, &outputs));
  }
  NamedTensors result;
  result.reserve(outputs.size());
  for (std::size_t i = 0; i < outputs.size(); ++i)
    result.emplace_back(fetches.at(i), from_float_tensor(outputs[i]));
  if (partition_nodes) {
    partition_nodes->clear();
    for (const auto& graph : metadata.partition_graphs())
      for (const auto& node : graph.node())
        partition_nodes->emplace_back(node.name(), node.op());
  }
  if (trace) {
    std::vector<GraphTraceEvent> events;
    std::uint32_t index = 0;
    for (const auto& device : metadata.step_stats().dev_stats()) {
      ++index;
      for (const auto& node : device.node_stats()) {
        const auto duration = node.op_end_rel_micros() - node.op_start_rel_micros();
        if (duration < 0) throw std::runtime_error("GraphSession: negative trace duration");
        events.push_back({node.node_name(), index, node.thread_id(),
                          node.all_start_micros() + node.op_start_rel_micros(), duration});
      }
    }
    *trace = std::move(events);
  }
  return result;
#else
  (void)feeds;
  (void)fetches;
  (void)partition_nodes;
  (void)trace;
  throw std::runtime_error("GraphSession: rebuild with DATAMUNGE_ENABLE_TENSORFLOW=ON");
#endif
}

void save_graph_trace(const std::string& path, const std::vector<GraphTraceEvent>& events) {
  for (const auto& event : events)
    if (event.duration_us < 0)
      throw std::invalid_argument("save_graph_trace: negative event duration");
  std::ofstream out(path);
  out.imbue(std::locale::classic());
  out << "{\"traceEvents\":[";
  for (std::size_t i = 0; i < events.size(); ++i) {
    const auto& event = events[i];
    if (i) out << ',';
    out << "\n{\"ph\":\"X\",\"cat\":\"TensorFlow\",\"name\":\"";
    for (unsigned char ch : event.name) {
      if (ch == '"' || ch == '\\') out << '\\' << static_cast<char>(ch);
      else if (ch < 32) {
        constexpr char hex[] = "0123456789abcdef";
        out << "\\u00" << hex[ch >> 4] << hex[ch & 15];
      } else out << static_cast<char>(ch);
    }
    out << "\",\"pid\":" << event.device << ",\"tid\":" << event.thread
        << ",\"ts\":" << event.start_us << ",\"dur\":" << event.duration_us << '}';
  }
  out << "\n],\"displayTimeUnit\":\"ms\"}\n";
  out.close();
  if (!out) throw std::runtime_error("save_graph_trace: could not write " + path);
}

} // namespace datamunge::nn
