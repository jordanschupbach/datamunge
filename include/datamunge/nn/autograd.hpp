#pragma once

/// @file autograd.hpp
/// @brief Eager (define-by-run) reverse-mode automatic differentiation.
///
/// Every op that runs while grad mode is on and has an input that requires
/// grad attaches a `Node` to its outputs. The node knows the op's inputs (as
/// edges to their own nodes) and a backward function that maps gradients of
/// the outputs to gradients of the inputs. `grad()` and `Tensor::backward()`
/// walk these nodes in reverse topological order.
///
/// Backward functions are written with ordinary `nn` ops. With
/// `create_graph = true` they run in grad mode, so the gradients they return
/// have history of their own and can be differentiated again (Hessian-vector
/// products, gradient penalties, ...).
///
/// The mapping from TensorFlow concepts:
///   AddSymbolicGradients -> nn::grad(outputs, inputs, grad_outputs)
///   StopGradient         -> nn::stop_gradient(x) or NoGradGuard
///   custom gradient      -> nn::Function / nn::apply_function
///
/// Memory: a node keeps the tensors its backward needs (`SavedTensor`
/// captures and saved outputs) until that backward has run; with
/// `retain_graph = false` the engine then frees them at once, so live memory
/// falls steadily during the backward pass. `checkpoint` trades compute for
/// memory by not keeping a segment's intermediates at all and recomputing
/// them during backward; `reversible_sequence` rebuilds activations of
/// additive coupling blocks (RevNet) from their outputs. `nn/memory.hpp`
/// reports live and peak tensor bytes.

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "datamunge/nn/tensor.hpp"

namespace datamunge::nn {

class Generator;

// ---- grad mode ------------------------------------------------------------------

/// Whether ops currently record history (thread-local; on by default).
bool is_grad_enabled();
void set_grad_enabled(bool enabled);

/// Turns recording off for its lifetime (TF: ops outside the gradient path).
class NoGradGuard {
 public:
  NoGradGuard() : previous_(is_grad_enabled()) { set_grad_enabled(false); }
  ~NoGradGuard() { set_grad_enabled(previous_); }
  NoGradGuard(const NoGradGuard&)            = delete;
  NoGradGuard& operator=(const NoGradGuard&) = delete;

 private:
  bool previous_;
};

/// Turns recording on for its lifetime (inside a NoGradGuard).
class EnableGradGuard {
 public:
  EnableGradGuard() : previous_(is_grad_enabled()) { set_grad_enabled(true); }
  ~EnableGradGuard() { set_grad_enabled(previous_); }
  EnableGradGuard(const EnableGradGuard&)            = delete;
  EnableGradGuard& operator=(const EnableGradGuard&) = delete;

 private:
  bool previous_;
};

// ---- graph ------------------------------------------------------------------------

/// Backward function of a node: gradients of its outputs (one per output,
/// zero-filled when nothing flowed into that output) and, for nodes created
/// with `save_outputs`, the outputs themselves. Returns one gradient per
/// input; an undefined Tensor means "no gradient".
using BackwardFn =
    std::function<std::vector<Tensor>(const std::vector<Tensor>& grad_outputs, const std::vector<Tensor>& outputs)>;

/// Where a gradient goes: input slot `input_nr` of `node`.
struct Edge {
  std::shared_ptr<Node> node;
  std::uint32_t         input_nr{0};
};

/// Shape and dtype of a tensor flowing through a node, used to materialise
/// zero gradients and to check returned gradient shapes.
struct TensorMeta {
  Shape shape;
  DType dtype{DType::Float32};
};

/// One recorded op.
class Node : public std::enable_shared_from_this<Node> {
 public:
  Node(std::string name, std::vector<Edge> next_edges, std::vector<TensorMeta> input_meta,
       std::vector<TensorMeta> output_meta, BackwardFn fn);
  /// Frees upstream nodes iteratively, so dropping a very deep graph (an
  /// RNN unrolled over thousands of steps) cannot overflow the stack.
  virtual ~Node();

  [[nodiscard]] const std::string&             name() const noexcept { return name_; }
  [[nodiscard]] const std::vector<Edge>&       next_edges() const noexcept { return next_edges_; }
  [[nodiscard]] const std::vector<TensorMeta>& input_meta() const noexcept { return input_meta_; }
  [[nodiscard]] const std::vector<TensorMeta>& output_meta() const noexcept { return output_meta_; }
  /// Creation order (process-wide counter): a node made later has a larger number.
  [[nodiscard]] std::uint64_t                  sequence_nr() const noexcept { return sequence_nr_; }
  /// True once `release()` has freed the backward function and saved values.
  [[nodiscard]] bool                           released() const noexcept { return released_; }

  /// Runs the backward function. Throws if the node was already released.
  virtual std::vector<Tensor> apply(const std::vector<Tensor>& grad_outputs);
  /// Frees the backward function and anything it saved.
  virtual void release();

  /// Keeps detached copies of the outputs so `apply` can hand them back to
  /// the backward function with this node as their history.
  void save_outputs(const std::vector<Tensor>& outputs);

 protected:
  std::string             name_;
  std::vector<Edge>       next_edges_;
  std::vector<TensorMeta> input_meta_;
  std::vector<TensorMeta> output_meta_;
  BackwardFn              fn_;
  bool                    released_{false};
  std::vector<Tensor>     saved_outputs_;  // detached
  std::vector<std::uint64_t> saved_versions_;
  std::uint64_t           sequence_nr_{0};
};

/// Sink node that adds incoming gradients into a leaf tensor's `grad()`.
class AccumulateGrad : public Node {
 public:
  explicit AccumulateGrad(const std::shared_ptr<TensorImpl>& variable);
  std::vector<Tensor> apply(const std::vector<Tensor>& grad_outputs) override;
  void                release() override {}
  [[nodiscard]] std::shared_ptr<TensorImpl> variable() const { return variable_.lock(); }

 private:
  std::weak_ptr<TensorImpl> variable_;
};

/// The edge a gradient for `t` arrives on (its grad_fn, or the leaf's accumulator).
Edge gradient_edge(const Tensor& t);

/// True when grad mode is on and any defined input requires grad.
bool any_requires_grad(const std::vector<Tensor>& inputs);

/// Attaches history to freshly created `outputs` when `any_requires_grad(inputs)`.
/// Use this to define new differentiable primitives. With `save_outputs`, the
/// backward function receives the outputs as its second argument.
void record(const std::string& name, const std::vector<Tensor>& inputs, const std::vector<Tensor>& outputs,
            BackwardFn fn, bool save_outputs = false);

/// A tensor captured for the backward pass that remembers the storage
/// version at capture time, so an in-place write made afterwards is reported
/// instead of silently producing a wrong gradient.
class SavedTensor {
 public:
  SavedTensor() = default;
  explicit SavedTensor(Tensor tensor);
  [[nodiscard]] const Tensor& get() const;

 private:
  Tensor        tensor_;
  std::uint64_t version_{0};
};

// ---- running reverse mode ----------------------------------------------------------

/// Options for `grad`.
struct GradOptions {
  /// Keep the graph so it can be walked again (implied by create_graph).
  bool retain_graph{false};
  /// Record the backward pass so the returned gradients are differentiable.
  bool create_graph{false};
  /// Return an undefined tensor (instead of throwing) for inputs the outputs do not depend on.
  /// Outputs with no history (undefined, or not requiring grad) are then
  /// ignored instead of rejected; if none has history every gradient is
  /// undefined. Without allow_unused such outputs throw std::invalid_argument.
  bool allow_unused{false};
};

/// Gradients of sum_i <grad_outputs[i], outputs[i]> with respect to each of
/// `inputs` (TF AddSymbolicGradients). `grad_outputs` defaults to ones for
/// one-element outputs. Does not touch `Tensor::grad()`.
std::vector<Tensor> grad(const std::vector<Tensor>& outputs, const std::vector<Tensor>& inputs,
                         const std::vector<Tensor>& grad_outputs = {}, const GradOptions& options = {});
/// Single-output form.
std::vector<Tensor> grad(const Tensor& output, const std::vector<Tensor>& inputs, const GradOptions& options = {});

/// Accumulates gradients of the outputs into every reachable leaf's `grad()`.
void backward(const std::vector<Tensor>& outputs, const std::vector<Tensor>& grad_outputs = {},
              bool retain_graph = false, bool create_graph = false);

/// A view of `x` with no history: gradients do not flow through it (TF StopGradient).
Tensor stop_gradient(const Tensor& x);

// ---- user-defined differentiable functions -----------------------------------------

/// Forward of a custom function. Runs with grad mode off on detached inputs.
using CustomForward = std::function<std::vector<Tensor>(const std::vector<Tensor>& inputs)>;
/// Backward of a custom function: receives the original inputs, the outputs
/// and the output gradients; returns one gradient per input (undefined = none).
using CustomBackward = std::function<std::vector<Tensor>(
    const std::vector<Tensor>& inputs, const std::vector<Tensor>& outputs, const std::vector<Tensor>& grad_outputs)>;

/// Runs `forward` and records `backward` as its gradient. This is how custom
/// gradients, straight-through estimators and gradient reversal are written.
std::vector<Tensor> apply_function(const std::string& name, const std::vector<Tensor>& inputs,
                                   const CustomForward& forward, const CustomBackward& backward);

/// A reusable custom function (TF: an op plus REGISTER_GRADIENT_OP).
class Function {
 public:
  Function(std::string name, CustomForward forward, CustomBackward backward)
      : name_(std::move(name)), forward_(std::move(forward)), backward_(std::move(backward)) {}

  std::vector<Tensor> operator()(const std::vector<Tensor>& inputs) const {
    return apply_function(name_, inputs, forward_, backward_);
  }
  /// One input, first output.
  Tensor operator()(const Tensor& input) const { return apply_function(name_, {input}, forward_, backward_).at(0); }

  [[nodiscard]] const std::string& name() const noexcept { return name_; }

 private:
  std::string    name_;
  CustomForward  forward_;
  CustomBackward backward_;
};

// ---- gradient checkpointing --------------------------------------------------------

/// A segment of a model for `checkpoint`: tensors in, tensors out.
using CheckpointFn = std::function<std::vector<Tensor>(const std::vector<Tensor>& inputs)>;
/// A one-tensor segment (a layer or a block).
using CheckpointFn1 = std::function<Tensor(const Tensor& input)>;

/// Options for `checkpoint`.
struct CheckpointOptions {
  /// Replay the default generator's draws (dropout, noise, ...) during the
  /// recomputation, so it sees the same random numbers as the forward pass.
  bool preserve_rng_state{true};
  /// Further generators `fn` draws from (for example a ParameterStore's
  /// `generator()`); their states are replayed the same way.
  std::vector<Generator*> generators;
};

/// Runs `fn(inputs)` without keeping its intermediate tensors, and runs it
/// again during the backward pass to rebuild them (gradient checkpointing,
/// "rematerialization"). Memory falls from every intermediate of `fn` to its
/// inputs and outputs; the price is one extra forward of `fn` per backward.
///
/// Gradients reach the inputs and every tensor `fn` reads from its closure
/// (parameters, tensors computed earlier), whichever of `grad` / `backward`
/// is used. With `create_graph` the recomputation is recorded, so second
/// order gradients go through a checkpoint, and checkpoints may be nested.
///
/// `fn` must compute the same values when it runs again: its random draws are
/// replayed from the generators in `options`, but other side effects (batch
/// norm moving averages, counters) happen twice unless `fn` skips them when
/// `in_recompute()` is true. `fn` must not call `grad` / `backward` itself.
/// When grad mode is off, or nothing `fn` uses requires grad, this is just
/// `fn(inputs)`.
std::vector<Tensor> checkpoint(const CheckpointFn& fn, const std::vector<Tensor>& inputs,
                               const CheckpointOptions& options = {});
/// One input, one output.
Tensor checkpoint(const CheckpointFn1& fn, const Tensor& input, const CheckpointOptions& options = {});

/// Runs `layers` in order, split into `segments` consecutive groups of about
/// equal length; every group but the last is a `checkpoint`. For n layers,
/// sqrt(n) segments keep about 2 sqrt(n) activations instead of n.
Tensor checkpoint_sequential(const std::vector<CheckpointFn1>& layers, std::int64_t segments, const Tensor& input,
                             const CheckpointOptions& options = {});

/// True while `checkpoint` or `reversible_sequence` re-runs a forward
/// function during the backward pass (use it to skip side effects such as
/// moving-average updates).
bool in_recompute();

/// An additive coupling block: y1 = x1 + f(x2), y2 = x2 + g(y1). It is
/// inverted exactly (up to rounding) by x2 = y2 - g(y1), x1 = y1 - f(x2).
struct ReversibleBlock {
  CheckpointFn1 f;
  CheckpointFn1 g;
};

/// Runs a chain of reversible blocks (RevNet, https://arxiv.org/abs/1707.04585)
/// on the halves (x1, x2) and returns (y1, y2) of the last block. No
/// activation inside the chain is kept: the backward pass walks the blocks in
/// reverse, rebuilds each block's input from its output, recomputes f and g
/// on it and backpropagates through that block alone, so memory does not
/// grow with the number of blocks. Gradients reach x1, x2 and every tensor
/// f and g read from their closures. Rebuilt inputs differ from the forward
/// ones by rounding, which can grow over many blocks. f and g must be
/// deterministic (see `checkpoint`); second order (`create_graph`) is not
/// supported — use `checkpoint` for that.
std::vector<Tensor> reversible_sequence(const std::vector<ReversibleBlock>& blocks, const Tensor& x1,
                                        const Tensor& x2);

}  // namespace datamunge::nn
