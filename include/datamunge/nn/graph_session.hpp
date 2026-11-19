#pragma once

/// @file graph_session.hpp
/// @brief TensorFlow CPU GraphDef execution and optimization inspection.

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <datamunge/nn/io.hpp>

namespace datamunge::nn {

/// @brief Optimizer settings for a TensorFlow CPU graph session.
/// @par Example
/// @code
/// const auto mode = dm::nn::GraphOptimization::Grappler;
/// @endcode
enum class GraphOptimization {
  Baseline, ///< Disable the Grappler meta-optimizer and XLA JIT.
  Grappler, ///< Enable Grappler and request constant folding.
  Xla ///< Enable Grappler and session-level XLA JIT.
};

/// @brief Enable CPU XLA auto-clustering before creating TensorFlow sessions.
///
/// This sets process-wide TensorFlow flags for CPU JIT, eager compilation
/// and a minimum cluster size of two. Call before concurrent TensorFlow use.
/// Pass GraphOptimization::Xla when constructing the session.
/// @throws std::runtime_error TensorFlow backend was disabled at build time.
/// @par Example
/// @code
/// if (dm::nn::saved_model_runtime_available()) dm::nn::enable_tensorflow_cpu_jit();
/// @endcode
/// @see GraphSession, GraphOptimization
void enable_tensorflow_cpu_jit();

/// @brief One operation interval from TensorFlow RunMetadata.
/// @par Example
/// @code
/// dm::nn::GraphTraceEvent event{"product", 1, 0, 25, 10};
/// std::println("{}", event.duration_us); // 10
/// @endcode
/// @see save_graph_trace()
struct GraphTraceEvent {
  std::string name; ///< TensorFlow node name.
  std::uint32_t device; ///< One-based device group in the trace.
  std::uint32_t thread; ///< Runtime thread identifier.
  std::int64_t start_us; ///< Start timestamp in microseconds.
  std::int64_t duration_us; ///< Elapsed operation time in microseconds.
};

/// @brief Write complete operation events in Chrome Trace Event JSON format.
/// @param path Output file path. Parent directories must exist.
/// @param events Operation intervals with nonnegative durations.
/// @throws std::invalid_argument Any duration is negative.
/// @throws std::runtime_error The file could not be written.
/// @par Example
/// @code
/// dm::nn::save_graph_trace("trace.json", {{"product", 1, 0, 25, 10}});
/// @endcode
/// @see GraphTraceEvent, GraphSession
void save_graph_trace(const std::string& path, const std::vector<GraphTraceEvent>& events);

/// @brief Own a trusted local GraphDef loaded into a CPU TensorFlow session.
///
/// Build with DATAMUNGE_ENABLE_TENSORFLOW. Feeds and fetches are Float32
/// tensors, with graph-defined names and shapes. Copies share the session.
/// @par Example
/// @code
/// const dm::nn::GraphSession session("graph.pb", dm::nn::GraphOptimization::Grappler);
/// @endcode
/// @par References
/// - TensorFlow graph optimization, https://www.tensorflow.org/guide/graph_optimization
/// @see enable_tensorflow_cpu_jit(), SavedModel
class GraphSession {
 public:
  /// @brief Load a binary GraphDef and choose one CPU optimization policy.
  /// @param path Trusted local graph file.
  /// @param mode Graph optimization policy, baseline by default.
  /// @throws std::invalid_argument Unknown optimization policy.
  /// @throws std::runtime_error Disabled backend or failure to load the graph or session.
  /// @par Example
  /// @code
  /// const dm::nn::GraphSession session("graph.pb");
  /// @endcode
  explicit GraphSession(const std::string& path, GraphOptimization mode = GraphOptimization::Baseline);

  /// @brief Load a binary GraphDef with isolated TensorFlow thread pools.
  /// @param path Trusted local graph file.
  /// @param mode Graph optimization policy.
  /// @param intra_threads Positive thread limit for parallel work inside an operation.
  /// @param inter_threads Positive thread limit across independent operations.
  /// @throws std::invalid_argument Unknown policy or nonpositive thread count.
  /// @throws std::runtime_error Disabled backend or failed graph/session load.
  /// @par Example
  /// @code
  /// const dm::nn::GraphSession session("graph.pb", dm::nn::GraphOptimization::Baseline, 1, 1);
  /// @endcode
  GraphSession(const std::string& path, GraphOptimization mode, int intra_threads, int inter_threads);

  /// @brief Run requested endpoints with optional partition-graph inspection.
  /// @param feeds Named CPU Float32 tensors, copied into TensorFlow.
  /// @param fetches Output endpoint names.
  /// @param partition_nodes Optional destination for executed partition node names and operation types.
  /// @return Detached CPU Float32 outputs in fetch order, with the requested names.
  /// @throws std::invalid_argument Invalid feed type or device.
  /// @throws std::runtime_error Runtime error or non-Float32 fetch.
  /// @par Example
  /// @code
  /// const dm::nn::GraphSession session("graph.pb");
  /// const auto y = session.run({{"features", dm::nn::zeros({65536})}}, {"output"});
  /// @endcode
  [[nodiscard]] NamedTensors run(const NamedTensors& feeds, const std::vector<std::string>& fetches,
      std::vector<std::pair<std::string, std::string>>* partition_nodes = nullptr) const;

  /// @brief Run endpoints with full operation tracing and optional partitions.
  /// @param feeds Named CPU Float32 feed tensors.
  /// @param fetches Output endpoint names.
  /// @param partition_nodes Optional destination for partition node names and operations.
  /// @param trace Optional destination for complete CPU operation intervals.
  /// @return Detached Float32 outputs in fetch order.
  /// @throws std::invalid_argument Invalid feed type or device.
  /// @throws std::runtime_error TensorFlow failure or inconsistent output or trace.
  /// @par Example
  /// @code
  /// const dm::nn::GraphSession session("graph.pb");
  /// std::vector<dm::nn::GraphTraceEvent> trace;
  /// const auto out = session.run({{"features", dm::nn::zeros({65536})}}, {"output"}, nullptr, &trace);
  /// @endcode
  [[nodiscard]] NamedTensors run(const NamedTensors& feeds, const std::vector<std::string>& fetches,
      std::vector<std::pair<std::string, std::string>>* partition_nodes,
      std::vector<GraphTraceEvent>* trace) const;

 private:
  /// @brief Shared TensorFlow session owner.
  struct Impl;
  std::shared_ptr<Impl> impl_; ///< Retained executable graph.
};

} // namespace datamunge::nn
