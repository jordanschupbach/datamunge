#pragma once

/// @file model_plot.hpp
/// @brief Executed neural-network architecture diagrams and callable modules with SVG plots.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "datamunge/nn/tensor.hpp"

namespace datamunge::nn {

/// @brief One single-input, single-output tensor forward function.
///
/// Capture stores by reference or shared ownership. Captured references must outlive
/// the callback. For builders returning a structure, return its logits or other chosen tensor.
/// @par Example
/// @code
/// dm::nn::ModelForward forward = [](const dm::nn::Tensor &x) { return dm::nn::relu(x); };
/// const auto y = forward(dm::nn::ones({1, 2}));
/// @endcode
/// @see Module, plot_model()
using ModelForward = std::function<Tensor(const Tensor&)>;

/// @brief Visual conventions for a left-to-right neural-network architecture diagram.
///
/// Every style preserves the same recorded dependencies and exact shape labels.
/// Node labels rotate 90 degrees counterclockwise. Volumes, planes and neuron
/// samples describe tensor geometry, not activation values. ModelPlotSizing controls
/// the dimension-to-pixel mapping, independently of this drawing convention.
/// @par Example
/// @code
/// dm::nn::ModelPlotOptions options;
/// options.style = dm::nn::ModelPlotStyle::LeNet;
/// @endcode
/// @par References
/// - VisualTorch rendering examples (visual inspiration),
///   https://visualtorch.readthedocs.io/en/latest/usage_examples/index.html
/// @see ModelPlotOptions, ModelPlotDetail
enum class ModelPlotStyle {
  Flow,  ///< Colored volumetric feature blocks joined by funnels and dependency arrows.
  Graph, ///< Node-and-edge view with optional sampled dense-layer neurons.
  LeNet  ///< Stacked feature-map planes and narrow dense feature columns.
};

/// @brief How named composite stages and functional layers are represented.
///
/// Blocks uses model_stage() boundaries and falls back to named functional layers.
/// Layers ignores those composite boundaries. Operations expands both into recorded
/// tensor primitives. No logical node is removed merely because a drawing is large.
/// @par Example
/// @code
/// dm::nn::ModelPlotOptions options;
/// options.detail = dm::nn::ModelPlotDetail::Layers;
/// @endcode
/// @see model_stage(), ModelPlotOptions
enum class ModelPlotDetail {
  Blocks,    ///< Group named composite blocks, falling back to functional layers.
  Layers,    ///< Show named functional layers and remaining primitive operations.
  Operations ///< Show every reachable recorded tensor operation.
};

/// @brief Shared category colors for all model diagram styles.
///
/// Values must be six-digit hexadecimal SVG colors, including '#'. Category labels
/// in the legend keep the figure interpretable without relying on color alone.
/// @par Example
/// @code
/// dm::nn::ModelPlotPalette colors;
/// colors.convolution = "#3388CC";
/// dm::nn::ModelPlotOptions options;
/// options.colors = colors;
/// @endcode
/// @see ModelPlotOptions
struct ModelPlotPalette {
  std::string input{"#E69F00"};         ///< Input tensor glyphs.
  std::string convolution{"#56B4E9"};   ///< Convolution and depthwise convolution layers.
  std::string dense{"#0072B2"};         ///< Dense and classifier layers.
  std::string pooling{"#009E73"};       ///< Pooling and spatial averaging.
  std::string activation{"#F0E442"};    ///< Elementwise activations.
  std::string normalization{"#CC79A7"}; ///< Normalization layers.
  std::string block{"#8B80C9"};         ///< Named composite blocks.
  std::string merge{"#D55E00"};         ///< Additive and concatenation joins.
  std::string output{"#6C8194"};        ///< Output tensor glyphs.
  std::string operation{"#94A3B8"};     ///< Other recorded tensor operations.
  std::string parameter{"#B88644"};     ///< Explicit parameter leaves in operation views.
  std::string background{"#FFFFFF"};    ///< Canvas background.
  std::string text{"#172033"};          ///< Heading, legend and node labels.
  std::string edge{"#64748B"};          ///< Dependency arrows and neuron connections.
  std::string outline{"#475569"};       ///< Glyph outlines.
};

/// @brief Content displayed beside each node; accessible tooltips always retain its name and shape.
///
/// Full includes name, layer description, shape and learned-value count. None hides
/// visible node labels only; the title, legend and accessible node metadata remain.
/// @par Example
/// @code
/// dm::nn::ModelPlotOptions options;
/// options.labels.type = dm::nn::ModelPlotLabel::NameAndShape;
/// @endcode
/// @see ModelPlotLabels
enum class ModelPlotLabel {
  Full,         ///< Name, kind, exact shape and parameter count when nonzero.
  Name,         ///< Display name only.
  Shape,        ///< Exact output tensor shape only.
  NameAndShape, ///< Display name and exact output tensor shape.
  None          ///< No visible node labels.
};

/// @brief Visible node-label content, font size and wrapping.
///
/// Labels rotate 90 degrees counterclockwise. Layout reserves space using the
/// requested monospace font size and wrap length; labels never change the recorded graph.
/// @par Example
/// @code
/// dm::nn::ModelPlotLabels labels;
/// labels.type = dm::nn::ModelPlotLabel::NameAndShape;
/// labels.size = 15;
/// dm::nn::ModelPlotOptions options;
/// options.labels = labels;
/// @endcode
/// @see ModelPlotOptions
struct ModelPlotLabels {
  ModelPlotLabel type{ModelPlotLabel::Full}; ///< Label contents; Full by default.
  double         size{13};                   ///< Node font size in SVG pixels, 6..72.
  int            wrap{30};                   ///< UTF-8 code points per line, 8..120.
};

/// @brief Transform applied to dimension ratios before converting them to pixels.
///
/// For dimension d and diagram-wide maximum m, Linear maps d/m, SquareRoot maps
/// sqrt(d/m), and Logarithmic maps log1p(d)/log1p(m). Compression is explicit;
/// only Linear preserves dimension ratios above the configured visibility floor.
/// @par Example
/// @code
/// dm::nn::ModelPlotOptions options;
/// options.sizing.channel_scale = dm::nn::ModelPlotScale::SquareRoot;
/// @endcode
/// @see ModelPlotSizing
enum class ModelPlotScale {
  Linear,     ///< Preserve dimension ratios, subject to minimum_extent.
  SquareRoot, ///< Compress large dimension ratios with a square root.
  Logarithmic ///< Compress large dimension ratios with log1p.
};

/// @brief Map per-example tensor dimensions to glyph dimensions using shared scales.
///
/// Channels-last spatial tensors use their last dimension as channel count. The
/// preceding spatial dimensions form the face: width is the last spatial axis,
/// height is the product of earlier spatial axes (one for rank-three sequences).
/// Thus NHWC uses W by H; NDHWC unfolds D*H by W. Batch size is excluded. Rank-two
/// tensors use feature count as a vertical column; rank-one tensors use their length.
/// Spatial axes share one maximum; channel depths and vector lengths each use their
/// own diagram-wide maximum. Linear scales preserve ratios within each family;
/// flow volume is proportional to H*W*C when no dimension hits the visibility floor.
/// LeNet uses the same face/depth envelope but bounds the number of drawn sheets;
/// graph nodes use that envelope, with sampled neuron markers for dense vectors.
/// Graph neuron markers and LeNet sheets are representative, not one per element.
/// Scalar byte width and batch size do not affect geometry; exact shapes retain batch.
/// @par Example
/// @code
/// dm::nn::ModelPlotOptions options;
/// options.sizing.spatial_size = 180;
/// options.sizing.channel_size = 90;
/// options.sizing.minimum_extent = 1;
/// @endcode
/// @see ModelPlotScale, ModelPlotOptions
struct ModelPlotSizing {
  ModelPlotScale spatial_scale{ModelPlotScale::Linear}; ///< Face-axis ratio transform.
  ModelPlotScale channel_scale{ModelPlotScale::Linear}; ///< Channel and vector ratio transform.
  double         spatial_size{144}; ///< Largest face axis or vector height, in pixels, 16..2000.
  double         channel_size{64};  ///< Largest channel depth in pixels, 4..1000.
  double         minimum_extent{2}; ///< Visible dimension floor in pixels, (0,16].
  int            max_planes{6};     ///< LeNet: at most 1..32 representative feature-map sheets.
};

/// @brief Empty space around glyphs and labels, in SVG pixels.
///
/// Layer spacing separates column envelopes; lane spacing separates complete node
/// and label envelopes. The renderer expands columns and lanes to prevent overlap.
/// @par Example
/// @code
/// dm::nn::ModelPlotOptions options;
/// options.spacing.layer = 56;
/// options.spacing.lane = 48;
/// @endcode
/// @see ModelPlotOptions
struct ModelPlotSpacing {
  double layer{38};  ///< Minimum gap between adjacent columns, 8..1000.
  double lane{24};   ///< Gap between parallel node/label envelopes, 8..1000.
  double label{12};  ///< Glyph-to-label gap, 0..1000.
  double margin{24}; ///< Canvas edge padding, 16..1000.
};

/// @brief Presentation settings for an executed model graph.
///
/// Width grows with graph depth; parallel branches occupy separate vertical lanes.
/// Graphs always flow left to right, with node labels rotated counterclockwise.
/// Wide images should be viewed at their native size with horizontal scrolling.
/// No nodes are silently removed. This is an executed trace
/// for one input shape and control-flow path, rather than a static program analysis.
/// @par Example
/// @code
/// dm::nn::ModelPlotOptions options;
/// options.title = "My classifier";
/// options.expand_layers = true;
/// @endcode
/// @see plot_model(), ModelDiagram
struct ModelPlotOptions {
  std::string title{"Neural network"}; ///< Accessible SVG title; arbitrary text is XML-escaped.
  int         width{0}; ///< Minimum canvas width: zero fits content; otherwise 400..10,000 pixels.
  ModelPlotStyle   style{ModelPlotStyle::Flow};     ///< Volumetric flow, graph or LeNet view.
  ModelPlotDetail  detail{ModelPlotDetail::Blocks}; ///< Composite, layer or primitive detail.
  bool             legend{true};       ///< Show a category-color legend and schematic-scale note.
  bool             show_neurons{true}; ///< Graph style: sample rank-two input/dense feature units.
  int              max_neurons{8}; ///< Graph style: at most 2..32 sampled units per eligible node.
  ModelPlotPalette colors{};       ///< Shared category palette; every entry must be #RRGGBB.
  bool   expand_layers{false};     ///< Compatibility alias: true overrides detail with Operations.
  double opacity{0.85};            ///< Glyph fill opacity, 0..1; labels and outlines stay readable.
  double edge_opacity{0.65};       ///< Dependency and auxiliary connector opacity, 0..1.
  double funnel_opacity{0.09};     ///< Flow funnel fill opacity, 0..1.
  double line_width{1.25};         ///< Dependency/outline stroke width in SVG pixels, (0,10].
  ModelPlotLabels  labels{};       ///< Node label content, font size and wrap length.
  ModelPlotSizing  sizing{};       ///< Shared shape-to-pixel scales and visible dimension floor.
  ModelPlotSpacing spacing{};      ///< Column, branch, label and canvas spacing.
};

/// @brief One displayed forward node, with input edges pointing to earlier node IDs.
///
/// Shapes include the actual example batch size. Parameter counts are per invocation;
/// ModelDiagram::parameter_count() deduplicates shared tensors across invocations.
/// @par Example
/// @code
/// const auto graph = dm::nn::plot_model([](const auto &x) { return dm::nn::relu(x); },
///                                       dm::nn::zeros({1, 2}));
/// const auto shape = graph.nodes().back().shape; // [1,2]
/// @endcode
/// @see ModelDiagram
struct ModelPlotNode {
  std::string              label;  ///< Layer name or primitive operation name.
  std::string              kind;   ///< Layer description, Input, Output, Parameter, or operation.
  Shape                    shape;  ///< Actual forward output dimensions.
  std::vector<std::size_t> inputs; ///< Source node IDs; repeated operands retain repeated edges.
  std::int64_t             parameters{0}; ///< Learned scalar values used by this displayed layer.
};

/// @brief Owned, tensor-free architecture graph that renders itself as standalone SVG.
///
/// The diagram retains only text, shapes and edges, so it does not keep an autograd
/// graph or model storage alive. Rendering is deterministic and requires no Graphviz,
/// TeX, display server or network access. The graph describes recorded differentiable
/// operations; constants and deliberately detached paths are not expanded.
/// @par Example
/// @code
/// const auto graph = dm::nn::plot_model([](const auto &x) { return dm::nn::relu(x); },
///                                       dm::nn::zeros({1, 2}));
/// graph.save_svg("model.svg");
/// @endcode
/// @see Module, plot_model()
class ModelDiagram {
 public:
  /// @brief Borrow displayed nodes in forward topological order.
  /// @return Nodes owned by this diagram; valid for its lifetime.
  const std::vector<ModelPlotNode>& nodes() const noexcept { return nodes_; }
  /// @brief Count distinct reachable learned tensor values, excluding the example input.
  /// @return Scalar count, deduplicated by tensor identity; constants are excluded.
  std::int64_t parameter_count() const noexcept { return parameters_; }

  /// @brief Render an accessible SVG with layer names, output shapes and directed edges.
  ///
  /// All dependencies progress left to right, without folding or reversing rows.
  /// Node labels wrap and rotate 90 degrees counterclockwise; the heading and legend
  /// remain upright. Shape dimensions use the selected shared scales and visibility
  /// floor; the legend reports compression. Sampled neurons and feature-map sheet
  /// counts are representative; printed shapes are exact. Only dense consumers
  /// receive all-to-all sampled-neuron connectors.
  /// Parameter leaves sit immediately before their earliest consumer instead of
  /// accumulating in the input column. Each column packs its own glyphs and labels.
  /// Embedded XML special characters are escaped.
  /// No forward pass is performed. Concurrent const rendering is safe.
  /// @return Complete UTF-8 SVG text with a title and description.
  /// @par Example
  /// @code
  /// const auto graph = dm::nn::plot_model([](const auto &x) { return x * 2.; },
  ///                                       dm::nn::ones({1, 2}));
  /// const auto svg = graph.to_svg(); // Save text to a file, not a report result block.
  /// @endcode
  /// @see save_svg()
  std::string to_svg() const;

  /// @brief Write the diagram as SVG, replacing an existing file.
  ///
  /// Parent directories must already exist. No forward pass or model mutation occurs.
  /// @param path File name, relative to the working directory or absolute.
  /// @throws std::runtime_error If opening or writing the file fails.
  /// @par Example
  /// @code
  /// const auto graph = dm::nn::plot_model([](const auto &x) { return x + 1.; },
  ///                                       dm::nn::zeros({1, 2}));
  /// graph.save_svg("network.svg");
  /// @endcode
  /// @see to_svg()
  void save_svg(const std::string& path) const;

 private:
  std::vector<ModelPlotNode> nodes_;         ///< Owned forward graph; no tensor handles retained.
  std::int64_t               parameters_{0}; ///< Unique reachable learned scalar values.
  ModelPlotOptions           options_;       ///< Rendering settings captured at trace time.
  friend ModelDiagram plot_model(const ModelForward&, const Tensor&, const ModelPlotOptions&);
};

/// @brief Trace one forward execution and return an SVG-capable model diagram.
///
/// Executes forward exactly once on a detached clone of the floating example input,
/// with autograd enabled even inside NoGradGuard. Input values, flags and gradients
/// are unchanged. Recording mode and nested tracing state are restored on exceptions.
/// Dense/convolution/normalization layers are grouped by default; expand_layers shows
/// their primitives. Branches, joins, shared layers and actual batch dimensions are retained.
/// No optimizer or backward pass runs. The callback has its normal side effects:
/// lazy parameter creation is allowed. Select inference mode to avoid dropout draws
/// or normalization updates, and warm up before constructing an optimizer.
/// Parameter totals deduplicate reachable gradient leaves, including those inside grouped layers;
/// frozen constants or disconnected parameters are not counted.
/// @param forward Nonempty tensor forward callback; captured objects must remain alive.
/// @param example Defined Float32/Float64 example tensor, including its batch dimension.
/// @param options Style, detail, colors, opacity, spacing, label and shape-sizing settings.
/// @return Owned graph for this input shape and executed control-flow path.
/// @throws std::invalid_argument If callback/input/options are invalid, the output is
/// undefined, or the output has no recorded differentiable history. Invalid styles,
/// detail/label/scale levels, color strings, nonfinite/out-of-range numbers or sample
/// limits are rejected before the callback runs.
/// Callback errors propagate.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// const auto graph = n::plot_model([&](const auto &x) {
///   return n::dense(p, "linear", x, 3);
/// }, n::zeros({1, 4}));
/// graph.save_svg("linear.svg");
/// @endcode
/// @see Module::plot(), model_stage()
ModelDiagram plot_model(
    const ModelForward&     forward,
    const Tensor&           example,
    const ModelPlotOptions& options = {});

/// @brief Mark a composite stage for grouped model diagrams, returning its output unchanged.
///
/// Call inside a forward function after computing output from input. Outside plotting,
/// this is an identity with no allocation or tensor mutation. A grouped stage stands
/// for the differentiable subgraph between input and output; all external data inputs
/// must pass through input. For independent inputs, use the vector-input overload
/// or concatenate them before computing the stage output.
/// Named layer detail remains available with detail=Layers, and primitives with
/// detail=Operations (or expand_layers=true). A later mark of
/// the same output wins. Does not fit, detach, copy values or alter gradient flow.
/// @param name Display name for this stage.
/// @param input Tensor at the stage's boundary, preceding output in the forward graph.
/// @param output Computed stage output; must differ from input to create a group.
/// @return The same output tensor handle.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// const auto graph = n::plot_model([&](const auto &x) {
///   const auto y = n::mlp(p, "hidden", x, {4, 4});
///   return n::model_stage("Two hidden layers", x, y);
/// }, n::zeros({1, 2}));
/// @endcode
/// @see plot_model(), ModelPlotOptions::expand_layers
Tensor model_stage(const std::string& name, const Tensor& input, const Tensor& output);

/// @brief Mark a composite stage with multiple external tensor inputs.
///
/// Records one block with an incoming edge from each distinct differentiable input.
/// Supply every external data boundary used by output; constants without gradient
/// history are ignored. Inputs precede output in the recorded graph. Grouping does
/// not recompute values or change gradients. Outside a plot it returns output without
/// allocating. Layers/Operations detail expands the block as for the single-input form.
/// @param name Display name for this stage.
/// @param inputs External data boundaries; duplicates are merged. Empty/undefined boundaries
/// leave the output ungrouped, as does a boundary equal to output.
/// @param output Already-computed output depending on the supplied inputs.
/// @return The same output tensor handle.
/// @par Example
/// @code
/// namespace n=dm::nn;
/// const auto graph=n::plot_model([](const n::Tensor& x){
///   const auto a=x*2., b=x*3.;
///   return n::model_stage("Two-branch merge",std::vector<n::Tensor>{a,b},n::relu(a+b));
/// },n::ones({1,4}));
/// @endcode
/// @see model_stage(), plot_model()
Tensor
model_stage(const std::string& name, const std::vector<Tensor>& inputs, const Tensor& output);

/// @brief Callable functional neural-network module with an architecture plot method.
///
/// Wrap existing linear, deep, convolutional or other tensor forward functions. Owns
/// the callback, not the stores referenced by it. Copies share the callback's captured
/// state. Forward execution and any mutable stores require the caller's usual locking.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// n::Module model([&](const auto &x) {
///   return n::dense(p, "output", n::mlp(p, "hidden", x, {8, 4}), 2);
/// }, "Deep classifier");
/// model.plot(n::zeros({1, 3})).save_svg("deep.svg");
/// @endcode
/// @see plot_model(), ModelDiagram
class Module {
 public:
  /// @brief Own a forward callback and its default diagram title.
  /// @param forward Nonempty tensor forward function; reference captures remain caller-owned.
  /// @param name Default diagram title, "Neural network" by default.
  /// @throws std::invalid_argument If forward is empty.
  /// @par Example
  /// @code
  /// dm::nn::Module model([](const auto &x) { return dm::nn::relu(x); }, "ReLU");
  /// @endcode
  /// @see plot()
  explicit Module(ModelForward forward, std::string name = "Neural network");

  /// @brief Execute the wrapped forward function with ordinary eager semantics.
  /// @param input Tensor supplied unchanged to the callback.
  /// @return The callback's tensor, retaining its ordinary autograd history.
  /// @throws std::exception Any exception thrown by the callback propagates.
  /// @par Example
  /// @code
  /// dm::nn::Module model([](const auto &x) { return x + 1.; });
  /// const auto y = model(dm::nn::zeros({1, 2})); // [1,1]
  /// @endcode
  /// @see plot()
  Tensor operator()(const Tensor& input) const;

  /// @brief Plot this module using one forward pass on a cloned example input.
  /// @param example Floating example tensor, including batch size.
  /// @param options Rendering settings; the default title uses this module's name.
  /// @return A diagram with save_svg() and to_svg(); no model graph retained.
  /// @throws std::invalid_argument As documented for plot_model(); callback errors propagate.
  /// @par Example
  /// @code
  /// dm::nn::Module model([](const auto &x) { return dm::nn::tanh(x); }, "Tanh");
  /// model.plot(dm::nn::zeros({1, 2})).save_svg("tanh.svg");
  /// @endcode
  /// @see plot_model()
  ModelDiagram plot(const Tensor& example, ModelPlotOptions options = {}) const;

 private:
  ModelForward forward_; ///< Owned callback; reference captures remain caller-owned.
  std::string  name_;    ///< Default accessible diagram title.
};

/// @brief Internal named-layer recording support; not a separate user API.
namespace model_plot_detail {
/// @brief Validate renderer enums, width, palette values and bounded neuron sampling.
/// @param options Options to inspect without executing a forward pass.
/// @throws std::invalid_argument If a renderer option is outside its documented domain.
void validate_options(const ModelPlotOptions& options);
/// @brief Whether this thread is currently collecting a model diagram.
/// @return True inside plot_model(), false during ordinary forwards.
bool recording() noexcept;
/// @brief Register a named single-input layer only while a model trace is active.
/// @param name Layer parameter prefix and display label.
/// @param kind Operation and configuration text.
/// @param input Layer input boundary.
/// @param output Layer output boundary; returned unchanged.
/// @return The same output handle.
Tensor
layer(const std::string& name, const std::string& kind, const Tensor& input, const Tensor& output);
} // namespace model_plot_detail
} // namespace datamunge::nn
