/// @file model_plot.cpp
/// @brief Executed layer grouping, graph extraction and native SVG architecture rendering.
#include "datamunge/nn/model_plot.hpp"

#include <algorithm>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "datamunge/nn/autograd.hpp"

namespace datamunge::nn {
/// @brief Invocation-local tracing and private SVG layout helpers.
namespace {
/// @brief Operation identity including the output slot of multi-output primitives.
using Key = std::pair<std::uint64_t, std::uint32_t>;
/// @brief Metadata for one grouped stage; lives only during the trace.
struct Layer {
  std::string         name;       ///< Display name.
  std::string         kind;       ///< Layer type and options.
  std::vector<Tensor> inputs;     ///< External differentiable data boundaries.
  Tensor              output;     ///< Output boundary retaining recorded history.
  std::vector<Tensor> parameters; ///< Reachable learned leaves inside the stage boundary.
};
/// @brief Invocation-local stage registry.
struct Trace {
  std::map<Key, Layer> layers; ///< Named functional layers, keyed by output slot.
  std::map<Key, Layer> blocks; ///< Composite stages, separate from their underlying layers.
};
/// @brief Only this thread's active trace; ordinary forwards do not allocate a registry.
thread_local Trace* active = nullptr;
/// @brief Restore nested tracing state after success or exception.
struct TraceGuard {
  Trace* previous; ///< Outer registry, possibly null.

  /// @brief Install the invocation's registry.
  explicit TraceGuard(Trace& trace) : previous(active) { active = &trace; }
  /// @brief Restore the previous registry.
  ~TraceGuard() { active = previous; }
};
/// @brief Identify one edge without process-specific pointer ordering.
Key key(const Edge& edge) {
  return {edge.node->sequence_nr(), edge.input_nr};
}
/// @brief Register an already-computed stage with all external differentiable boundaries.
/// @param name Display name.
/// @param kind Layer category or composite marker.
/// @param inputs External boundaries; gradient-free constants are ignored.
/// @param output Previously computed stage output.
/// @param composite Store as a block rather than a functional layer; default false.
void capture(
    const std::string&         name,
    const std::string&         kind,
    const std::vector<Tensor>& inputs,
    const Tensor&              output,
    bool                       composite = false) {
  if (!active || !output.defined() || inputs.empty())
    return;
  const auto b = gradient_edge(output);
  if (!b.node)
    return;
  Layer         layer{name, kind, {}, output, {}};
  std::set<Key> boundaries;
  for (const auto& input : inputs) {
    if (!input.defined())
      return;
    const auto edge = gradient_edge(input);
    if (!edge.node)
      continue;
    if (key(edge) == key(b))
      return;
    if (boundaries.insert(key(edge)).second)
      layer.inputs.push_back(input);
  }
  if (boundaries.empty())
    return;
  std::vector<Edge> pending{b};
  std::set<Key>     visited;
  while (!pending.empty()) {
    const auto edge = pending.back();
    pending.pop_back();
    if (!edge.node || boundaries.contains(key(edge)) || !visited.insert(key(edge)).second)
      continue;
    if (const auto leaf = std::dynamic_pointer_cast<AccumulateGrad>(edge.node)) {
      if (const auto variable = leaf->variable())
        layer.parameters.emplace_back(variable);
    } else {
      for (const auto& upstream : edge.node->next_edges())
        pending.push_back(upstream);
    }
  }
  (composite ? active->blocks : active->layers)[key(b)] = std::move(layer);
}
} // namespace

namespace model_plot_detail {
bool recording() noexcept {
  return active != nullptr;
}
Tensor
layer(const std::string& name, const std::string& kind, const Tensor& input, const Tensor& output) {
  if (active)
    capture(name, kind, {input}, output);
  return output;
}
} // namespace model_plot_detail

Tensor model_stage(const std::string& name, const Tensor& input, const Tensor& output) {
  if (active)
    capture(name, "Composite stage", {input}, output, true);
  return output;
}

Tensor
model_stage(const std::string& name, const std::vector<Tensor>& inputs, const Tensor& output) {
  if (active)
    capture(name, "Composite stage", inputs, output, true);
  return output;
}

ModelDiagram
plot_model(const ModelForward& forward, const Tensor& example, const ModelPlotOptions& options) {
  if (!forward || !example.defined() ||
      (example.dtype() != DType::Float32 && example.dtype() != DType::Float64))
    throw std::invalid_argument(
        "plot_model: require a forward callback and a defined floating input");
  model_plot_detail::validate_options(options);
  EnableGradGuard grad;
  auto            input = example.detach().clone().requires_grad_();
  const auto      input_edge =
      gradient_edge(input); // Keep the input sink alive, including for identity.
  Trace      trace;
  TraceGuard guard(trace);
  const auto output = forward(input);
  if (!output.defined() || !output.requires_grad())
    throw std::invalid_argument(
        "plot_model: output has no differentiable history; return a recorded tensor");
  ModelDiagram graph;
  graph.options_     = options;
  const auto detail  = options.expand_layers ? ModelPlotDetail::Operations : options.detail;
  auto       grouped = trace.layers;
  if (detail == ModelPlotDetail::Blocks)
    for (const auto& [key, layer] : trace.blocks)
      grouped[key] = layer;
  std::map<Key, std::size_t>  seen;
  std::set<const TensorImpl*> counted;
  /// @brief Add a learned tensor once, including shared invocations hidden by grouping.
  const auto count_parameter = [&](const Tensor& parameter) {
    if (parameter.impl() != input.impl() && counted.insert(parameter.impl()).second)
      graph.parameters_ += parameter.numel();
  };
  /// @brief Recursively emit upstream nodes before consumers, retaining output-slot shape metadata.
  std::function<std::size_t(const Edge&)> visit = [&](const Edge& edge) -> std::size_t {
    const auto k = key(edge);
    if (const auto it = seen.find(k); it != seen.end())
      return it->second;
    ModelPlotNode node;
    if (k == key(input_edge)) {
      node.label = "Input";
      node.kind  = "Input";
      node.shape = input.shape();
    } else if (const auto it = grouped.find(k);
               detail != ModelPlotDetail::Operations && it != grouped.end()) {
      const auto& layer = it->second;
      node.label        = layer.name;
      node.kind         = layer.kind;
      node.shape        = layer.output.shape();
      for (const auto& input : layer.inputs)
        node.inputs.push_back(visit(gradient_edge(input)));
      for (const auto& p : layer.parameters) {
        node.parameters += p.numel();
        count_parameter(p);
      }
    } else if (const auto leaf = std::dynamic_pointer_cast<AccumulateGrad>(edge.node)) {
      node.label = "Parameter";
      node.kind  = "Parameter";
      if (const auto variable = leaf->variable()) {
        const Tensor parameter(variable);
        node.shape      = parameter.shape();
        node.parameters = parameter.numel();
        count_parameter(parameter);
      }
    } else {
      node.label       = edge.node->name();
      node.kind        = "Operation";
      const auto& meta = edge.node->output_meta();
      if (edge.input_nr < meta.size())
        node.shape = meta[edge.input_nr].shape;
      for (const auto& e : edge.node->next_edges())
        if (e.node)
          node.inputs.push_back(visit(e));
    }
    const auto id = graph.nodes_.size();
    graph.nodes_.push_back(std::move(node));
    seen[k] = id;
    return id;
  };
  const auto last = visit(gradient_edge(output));
  graph.nodes_.push_back({"Output", "Output", output.shape(), {last}, 0});
  return graph;
}

void ModelDiagram::save_svg(const std::string& path) const {
  std::ofstream file(path, std::ios::binary);
  if (!file || !(file << to_svg()))
    throw std::runtime_error("model diagram: cannot write SVG to " + path);
  file.close();
  if (!file)
    throw std::runtime_error("model diagram: cannot finish SVG file " + path);
}

Module::Module(ModelForward forward, std::string name)
    : forward_(std::move(forward)), name_(std::move(name)) {
  if (!forward_)
    throw std::invalid_argument("Module: forward callback is empty");
}
Tensor Module::operator()(const Tensor& input) const {
  return forward_(input);
}
ModelDiagram Module::plot(const Tensor& example, ModelPlotOptions options) const {
  if (options.title == "Neural network")
    options.title = name_;
  return plot_model(forward_, example, options);
}
} // namespace datamunge::nn
