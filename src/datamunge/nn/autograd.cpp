/// @file autograd.cpp
/// @brief Eager gradient recording, reverse graph traversal, checkpointing, and reversible blocks.

#include "datamunge/nn/autograd.hpp"

#include <algorithm>
#include <atomic>
#include <stdexcept>
#include <unordered_set>

#include "datamunge/nn/ops.hpp"
#include "datamunge/nn/profiler.hpp"
#include "datamunge/nn/random.hpp"
#include "detail.hpp"

namespace datamunge::nn {

/// @brief Thread-local recording state and private graph discovery/execution machinery.
namespace {
/// @brief CPU contexts interoperate; GPU gradients must use the exact native execution owner.
/// @param tensor Defined tensor whose storage context is inspected.
/// @param context Expected gradient destination, retained elsewhere.
/// @return Whether the tensor can be used without an implicit device/context transfer.
bool compatible_context(const Tensor& tensor,const compute::Context& context) {
  return (tensor.device().backend==compute::Backend::CPU && context.device().backend==compute::Backend::CPU) ||
    tensor.context().shares_with(context);
}
/// @brief Whether this thread records differentiable operations; restored by grad guards.
thread_local bool grad_enabled = true;
/// @brief Nesting depth of checkpoint/reversible forward recomputations on this thread.
thread_local int recompute_depth = 0;
/// @brief Process-wide source of node creation IDs; relaxed atomic increments keep IDs unique.
std::atomic<std::uint64_t> next_sequence_nr{1};

/// @brief Non-owning identity of a gradient destination, used while graph ownership exists elsewhere.
struct EdgeKey {
  /// @brief Receiving backward node; this key does not retain it.
  Node*         node;
  /// @brief Incoming gradient slot at the receiving node.
  std::uint32_t slot;
  /// @brief Compares node identity and slot, without inspecting graph contents.
  bool          operator==(const EdgeKey& o) const { return node == o.node && slot == o.slot; }
};
/// @brief Hash function for destination identities in checkpoint boundary sets.
struct EdgeKeyHash {
  /// @brief Combines the destination pointer hash with a mixed slot index.
  std::size_t operator()(const EdgeKey& k) const {
    return std::hash<const void*>()(k.node) ^ (static_cast<std::size_t>(k.slot) * 0x9e3779b97f4a7c15ULL);
  }
};

/// While a checkpoint runs its forward function, `record` does not build a
/// graph: it notes which edges from outside the function carry gradient in
/// (the "boundary") and tags outputs with a shared marker node, so tensors
/// computed inside count as requiring grad but nothing is saved for backward.
struct Collector {
  /// @brief Placeholder attached to outputs produced inside the collected forward segment.
  std::shared_ptr<Node>                         marker;
  /// @brief Distinct outside gradient destinations encountered in first-use order.
  std::vector<Edge>                             boundary;
  /// @brief Deduplication set for destinations already in the explicit inputs or boundary.
  std::unordered_set<EdgeKey, EdgeKeyHash>      seen;

  /// @brief Records an outside tensor's gradient destination, ignoring untracked and marker tensors.
  void note(const Tensor& t) {
    if (!t.defined() || !t.requires_grad()) return;
    Edge e = gradient_edge(t);
    if (!e.node || e.node == marker) return;
    if (seen.insert({e.node.get(), e.input_nr}).second) boundary.push_back(std::move(e));
  }
};
/// @brief Active checkpoint collector on this thread, or null for ordinary graph recording.
thread_local Collector* collector = nullptr;
}  // namespace

bool is_grad_enabled() { return grad_enabled; }
void set_grad_enabled(bool enabled) { grad_enabled = enabled; }

// ---- Node ---------------------------------------------------------------------------

Node::Node(std::string name, std::vector<Edge> next_edges, std::vector<TensorMeta> input_meta,
           std::vector<TensorMeta> output_meta, BackwardFn fn)
    : name_(std::move(name)),
      next_edges_(std::move(next_edges)),
      input_meta_(std::move(input_meta)),
      output_meta_(std::move(output_meta)),
      fn_(std::move(fn)),
      sequence_nr_(next_sequence_nr.fetch_add(1, std::memory_order_relaxed)) {}

Node::~Node() {
  // Take ownership of the upstream nodes before the backward closure (which
  // may also reference them through saved tensors) is destroyed, then tear
  // the chain down with an explicit stack.
  std::vector<std::shared_ptr<Node>> pending;
  for (auto& e : next_edges_)
    if (e.node) pending.push_back(std::move(e.node));
  fn_ = nullptr;
  saved_outputs_.clear();
  while (!pending.empty()) {
    std::shared_ptr<Node> node = std::move(pending.back());
    pending.pop_back();
    if (node.use_count() != 1) continue;  // still referenced elsewhere
    for (auto& e : node->next_edges_)
      if (e.node) pending.push_back(std::move(e.node));
    node->fn_ = nullptr;
    node->saved_outputs_.clear();
  }
}

void Node::save_outputs(const std::vector<Tensor>& outputs) {
  saved_outputs_.clear();
  saved_versions_.clear();
  for (const auto& t : outputs) {
    saved_outputs_.push_back(t.detach());
    saved_versions_.push_back(t.impl()->storage->version);
  }
}

std::vector<Tensor> Node::apply(const std::vector<Tensor>& grad_outputs) {
  if (released_)
    throw std::logic_error("autograd: tried to run backward through '" + name_ +
                           "' a second time; its saved values were freed. Pass retain_graph = true the first time.");
  std::vector<Tensor> outputs;
  outputs.reserve(saved_outputs_.size());
  for (std::size_t i = 0; i < saved_outputs_.size(); ++i) {
    const Tensor& s = saved_outputs_[i];
    if (s.impl()->storage->version != saved_versions_[i])
      throw std::logic_error("autograd: an output of '" + name_ +
                             "' that its gradient needs was modified in place after it was computed");
    // Re-attach this node as the output's history so a create_graph pass can
    // differentiate through it.
    Tensor t              = s.detach();
    t.impl()->grad_fn     = shared_from_this();
    t.impl()->output_nr   = static_cast<std::uint32_t>(i);
    t.impl()->requires_grad = true;
    outputs.push_back(t);
  }
  return fn_(grad_outputs, outputs);
}

void Node::release() {
  released_ = true;
  fn_       = nullptr;
  saved_outputs_.clear();
}

AccumulateGrad::AccumulateGrad(const std::shared_ptr<TensorImpl>& variable)
    : Node("AccumulateGrad", {}, {{variable->shape, variable->dtype,variable->storage->context}}, {{variable->shape, variable->dtype,variable->storage->context}}, nullptr),
      variable_(variable) {}

std::vector<Tensor> AccumulateGrad::apply(const std::vector<Tensor>& grad_outputs) {
  auto var = variable_.lock();
  if (!var || !grad_outputs[0].defined()) return {};
  Tensor g = grad_outputs[0];
  if (!is_grad_enabled()) g = g.detach();
  if (var->grad) {
    Tensor previous(var->grad);
    var->grad = add(previous, g).impl_ptr();
  } else {
    // Own the buffer: later accumulation must not write into a tensor the
    // caller still holds.
    var->grad = (is_grad_enabled() ? g : g.clone()).impl_ptr();
  }
  return {};
}

// ---- recording ---------------------------------------------------------------------------

namespace {

/// @brief Reuses a leaf's live weakly cached sink or creates a new AccumulateGrad node.
std::shared_ptr<Node> grad_accumulator(const std::shared_ptr<TensorImpl>& impl) {
  if (auto existing = impl->grad_accumulator.lock()) return existing;
  auto node              = std::make_shared<AccumulateGrad>(impl);
  impl->grad_accumulator = node;
  return node;
}

}  // namespace

Edge gradient_edge(const Tensor& t) {
  if (!t.defined() || !t.requires_grad()) return {};
  if (t.impl()->grad_fn) return {t.impl()->grad_fn, t.impl()->output_nr};
  return {grad_accumulator(t.impl_ptr()), 0};
}

bool any_requires_grad(const std::vector<Tensor>& inputs) {
  if (!is_grad_enabled()) return false;
  for (const auto& t : inputs)
    if (t.defined() && t.requires_grad()) return true;
  return false;
}

void record(const std::string& name, const std::vector<Tensor>& inputs, const std::vector<Tensor>& outputs,
            BackwardFn fn, bool save_outputs) {
  if (!any_requires_grad(inputs)) return;
  if (collector) {
    for (const auto& t : inputs) collector->note(t);
    for (const auto& out : outputs) {
      TensorImpl* impl = out.impl();
      if (!is_floating(impl->dtype)) continue;
      if (impl->grad_fn)
        throw std::logic_error("autograd: op '" + name + "' tried to record history on a tensor that already has it");
      impl->grad_fn       = collector->marker;
      impl->output_nr     = 0;
      impl->requires_grad = true;
    }
    return;
  }
  std::vector<Edge>       edges;
  std::vector<TensorMeta> input_meta;
  for (const auto& t : inputs) {
    edges.push_back(gradient_edge(t));
    input_meta.push_back(t.defined() ? TensorMeta{t.shape(), t.dtype(),t.context()} : TensorMeta{});
  }
  std::vector<TensorMeta> output_meta;
  for (const auto& t : outputs) output_meta.push_back({t.shape(), t.dtype(),t.context()});
  auto node = std::make_shared<Node>(name, std::move(edges), std::move(input_meta), std::move(output_meta),
                                     std::move(fn));
  if (save_outputs) node->save_outputs(outputs);
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    TensorImpl* impl = outputs[i].impl();
    if (!is_floating(impl->dtype)) continue;
    if (impl->grad_fn)
      throw std::logic_error("autograd: op '" + name + "' tried to record history on a tensor that already has it");
    impl->grad_fn       = node;
    impl->output_nr     = static_cast<std::uint32_t>(i);
    impl->requires_grad = true;
  }
}

SavedTensor::SavedTensor(Tensor tensor) : tensor_(std::move(tensor)) {
  if (tensor_.defined()) version_ = tensor_.impl()->storage->version;
}

const Tensor& SavedTensor::get() const {
  if (tensor_.defined() && tensor_.impl()->storage->version != version_)
    throw std::logic_error("autograd: a tensor saved for backward was modified in place after it was saved");
  return tensor_;
}

// ---- engine --------------------------------------------------------------------------------

namespace {

/// @brief Invocation-local execution state for one reachable backward node.
/// Dense records avoid mutating the shared graph and support recursive checkpoints.
struct BackwardNode {
  /// @brief Owned graph node retained until this backward traversal finishes.
  std::shared_ptr<Node> node;
  /// @brief Start of this node's gradient slots in the traversal's flat buffer array.
  std::size_t offset{0};
  bool needed{false}; ///< Whether this node contributes to a requested gradient.
  bool captured{false}; ///< Whether an incoming gradient here is an explicit capture target.
  bool received{false}; ///< Whether any defined gradient has arrived in this node's slots.
};

/// @brief Dense node records indexed by an invocation-local open-addressed hash table.
/// The hash table retains indices rather than pointers into the resizable record vector.
class BackwardGraph {
 public:
  /// @brief Sentinel for an absent node or an unoccupied hash bucket.
  static constexpr std::size_t missing = static_cast<std::size_t>(-1);
  /// @brief Node records in first-discovery order, retained for the whole traversal.
  std::vector<BackwardNode> nodes;

  /// @brief Returns a node's record index, or missing for null/unregistered nodes.
  std::size_t find(Node* node) const {
    if (!node || buckets_.empty()) return missing;
    auto bucket = hash(node) & (buckets_.size() - 1);
    while (buckets_[bucket] != missing) {
      const auto index = buckets_[bucket];
      if (nodes[index].node.get() == node) return index;
      bucket = (bucket + 1) & (buckets_.size() - 1);
    }
    return missing;
  }

  /// @brief Registers a non-null node and returns its index plus whether it was newly inserted.
  std::pair<std::size_t, bool> insert(const std::shared_ptr<Node>& node) {
    const auto existing = find(node.get());
    if (existing != missing) return {existing, false};
    if (buckets_.empty() || (nodes.size() + 1) > buckets_.size() / 2)
      rehash(buckets_.empty() ? 64 : buckets_.size() * 2);
    const auto index = nodes.size();
    nodes.push_back({node});
    auto bucket = hash(node.get()) & (buckets_.size() - 1);
    while (buckets_[bucket] != missing) bucket = (bucket + 1) & (buckets_.size() - 1);
    buckets_[bucket] = index;
    return {index, true};
  }

  /// @brief Returns whether a registered node is marked as contributing to requested gradients.
  bool needed(Node* node) const {
    const auto index = find(node);
    return index != missing && nodes[index].needed;
  }

 private:
  /// @brief Mixes a live node's unique sequence number for use in power-of-two bucket arrays.
  static std::size_t hash(const Node* node) {
    // Sequence numbers avoid the zero low bits of aligned pointer hashes.
    auto key = node->sequence_nr();
    key = (key ^ (key >> 30)) * 0xbf58476d1ce4e5b9ULL;
    key = (key ^ (key >> 27)) * 0x94d049bb133111ebULL;
    return static_cast<std::size_t>(key ^ (key >> 31));
  }
  /// @brief Rebuilds bucket indices at a nonzero power-of-two capacity large enough for all nodes.
  void rehash(std::size_t capacity) {
    std::vector<std::size_t> buckets(capacity, missing);
    for (std::size_t index = 0; index < nodes.size(); ++index) {
      auto bucket = hash(nodes[index].node.get()) & (capacity - 1);
      while (buckets[bucket] != missing) bucket = (bucket + 1) & (capacity - 1);
      buckets[bucket] = index;
    }
    buckets_.swap(buckets);
  }
  /// @brief Power-of-two array of indices into nodes, with missing marking unused buckets.
  std::vector<std::size_t> buckets_;
};

/// @brief Orders destination identities by node pointer and then slot for capture lookup.
bool edge_less(const EdgeKey& a, const EdgeKey& b) {
  return a.node == b.node ? a.slot < b.slot : std::less<Node*>{}(a.node, b.node);
}

/// @brief Adds defined contributions, treating an undefined gradient as additive zero.
Tensor accumulate(const Tensor& a, const Tensor& b) {
  if (!a.defined()) return b;
  if (!b.defined()) return a;
  return add(a, b);
}

/// Shared implementation of grad() and backward(). In capture mode the
/// gradients arriving at `capture` edges are returned and leaf accumulators
/// are not run; otherwise every reachable accumulator runs. Nodes older than
/// `stop_before` (when nonzero) are a boundary: never entered or run, only
/// captured (checkpoint recomputations use this).
std::vector<Tensor> run_backward(const std::vector<Tensor>& outputs, const std::vector<Tensor>& grad_outputs,
                                 bool retain_graph, bool create_graph, const std::vector<Edge>* capture,
                                 bool allow_unused, std::uint64_t stop_before = 0) {
  if (collector)
    throw std::logic_error("autograd: grad/backward was called inside a checkpointed function, which is not supported");
  // Nodes created before recomputation are capture boundaries rather than nodes to execute.
  auto is_boundary = [stop_before](const Node* n) { return stop_before != 0 && n->sequence_nr() < stop_before; };
  if (!grad_outputs.empty() && grad_outputs.size() != outputs.size())
    throw std::invalid_argument("grad: got " + std::to_string(grad_outputs.size()) + " grad_outputs for " +
                                std::to_string(outputs.size()) + " outputs");

  // Roots and their seeds.
  std::vector<std::pair<Edge, Tensor>> roots;
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    const Tensor& out = outputs[i];
    if (!out.defined() || !out.requires_grad())
      throw std::invalid_argument("grad: output " + std::to_string(i) +
                                  " does not require grad (it has no history connecting it to any input)");
    Tensor seed;
    if (!grad_outputs.empty() && grad_outputs[i].defined()) {
      seed = grad_outputs[i];
      if (seed.shape() != out.shape())
        throw std::invalid_argument("grad: grad_outputs[" + std::to_string(i) + "] has shape " +
                                    shape_str(seed.shape()) + " but the output has shape " + shape_str(out.shape()));
      if(!compatible_context(seed,out.context()))
        throw std::invalid_argument("grad: explicit seed context differs from output context; transfer it explicitly");
      seed = cast(seed, out.dtype());
    } else {
      if (out.numel() != 1)
        throw std::invalid_argument("grad: output " + std::to_string(i) + " has shape " + shape_str(out.shape()) +
                                    "; non-scalar outputs need an explicit grad_output");
      seed = ones_like(out);
    }
    roots.emplace_back(gradient_edge(out), seed);
  }

  // Capture targets are usually small, immutable and may contain duplicates.
  // Sorted contiguous entries retain all duplicate input positions.
  std::vector<std::pair<EdgeKey, std::size_t>> capture_index;
  if (capture) {
    capture_index.reserve(capture->size());
    for (std::size_t i = 0; i < capture->size(); ++i) {
      const Edge& e = (*capture)[i];
      if (e.node) capture_index.push_back({{e.node.get(), e.input_nr}, i});
    }
    std::sort(capture_index.begin(), capture_index.end(),
        [](const auto& a, const auto& b) { return edge_less(a.first, b.first); });
  }

  // Discover with iterative DFS, then execute in reverse postorder. Complete
  // each root's DFS before the next so ancestor roots retain correct ordering.
  BackwardGraph graph;
  std::vector<std::size_t> postorder;
  postorder.reserve(32);
  {
    std::vector<std::pair<std::size_t, std::size_t>> stack;
    for (auto& r : roots) {
      if (!r.first.node) continue;
      const auto root = graph.insert(r.first.node);
      if (!root.second) continue;
      stack.emplace_back(root.first, 0);
      while (!stack.empty()) {
        auto& [index, next] = stack.back();
        Node* node = graph.nodes[index].node.get();
        if (next < node->next_edges().size() && !is_boundary(node)) {
          const Edge& e = node->next_edges()[next++];
          if (e.node) {
            const auto child = graph.insert(e.node);
            if (child.second) stack.emplace_back(child.first, 0);
          }
          continue;
        }
        postorder.push_back(index);
        stack.pop_back();
      }
    }
  }
  for (const auto& target : capture_index) {
    const auto index = graph.find(target.first.node);
    if (index != BackwardGraph::missing) graph.nodes[index].captured = true;
  }
  std::size_t slots = 0;
  for (auto index : postorder) {  // children come before parents
    auto& state = graph.nodes[index];
    Node* n = state.node.get();
    state.needed = !capture || state.captured;
    if (!state.needed && !is_boundary(n))
      for (const Edge& e : n->next_edges())
        if (graph.needed(e.node.get())) {
          state.needed = true;
          break;
        }
    state.offset = slots;
    if (state.needed) slots += n->output_meta().size();
  }

  // All pending gradient handles share one allocation. Reuse the call vector
  // rather than allocating a separate vector/hash node for every graph node.
  std::vector<Tensor> buffers(slots);
  // Add a contribution to its already-discovered destination and mark the node ready for work.
  auto add_to_buffer = [&](Node* node, std::uint32_t slot, const Tensor& g) {
    auto& state = graph.nodes[graph.find(node)];
    Tensor& value = buffers[state.offset + slot];
    value = accumulate(value, g);
    state.received = true;
  };
  for (auto& r : roots)
    if (graph.needed(r.first.node.get())) add_to_buffer(r.first.node.get(), r.first.input_nr, r.second);

  std::vector<Tensor> captured(capture ? capture->size() : 0);

  // Process in reverse postorder: every node runs after all its consumers.
  bool previous_mode = is_grad_enabled();
  set_grad_enabled(create_graph);
  try {
    std::vector<Tensor> grads;
    for (auto it = postorder.rbegin(); it != postorder.rend(); ++it) {
      auto& state = graph.nodes[*it];
      Node* node = state.node.get();
      if (!state.needed) continue;
      if (!state.received) {
        if (!retain_graph) node->release();
        continue;  // nothing flowed here
      }
      grads.clear();
      grads.reserve(node->output_meta().size());
      for (std::size_t slot = 0; slot < node->output_meta().size(); ++slot)
        grads.push_back(std::move(buffers[state.offset + slot]));

      if (capture) {
        if (state.captured) for (std::uint32_t slot = 0; slot < grads.size(); ++slot) {
          const EdgeKey key{node, slot};
          auto c = std::lower_bound(capture_index.begin(), capture_index.end(), key,
              [](const auto& entry, const EdgeKey& target) { return edge_less(entry.first, target); });
          for (; c != capture_index.end() && c->first == key; ++c)
            captured[c->second] = grads[slot];
        }
        if (dynamic_cast<AccumulateGrad*>(node) || is_boundary(node)) continue;
        bool upstream = false;
        for (const Edge& e : node->next_edges())
          if (e.node && graph.needed(e.node.get())) upstream = true;
        if (!upstream) {
          if (!retain_graph) node->release();
          continue;
        }
      }

      bool any = false;
      for (std::size_t s = 0; s < grads.size(); ++s) {
        if (grads[s].defined()) {
          any = true;
        } else {
          const auto& m = node->output_meta()[s];
          grads[s]      = m.context ? zeros(m.shape,m.dtype,*m.context) : zeros(m.shape,m.dtype);
        }
      }
      if (!any) continue;

      std::vector<Tensor> input_grads;
      {
        ProfileScope profile(node->name(), "backward");
        input_grads = node->apply(grads);
      }
      if (!retain_graph) node->release();
      const auto& edges = node->next_edges();
      if (!input_grads.empty() && input_grads.size() != edges.size())
        throw std::logic_error("autograd: backward of '" + node->name() + "' returned " +
                               std::to_string(input_grads.size()) + " gradients for " +
                               std::to_string(edges.size()) + " inputs");
      for (std::size_t i = 0; i < input_grads.size(); ++i) {
        const Edge& e = edges[i];
        if (!e.node || !graph.needed(e.node.get()) || !input_grads[i].defined()) continue;
        const auto& meta = node->input_meta()[i];
        Tensor      g    = input_grads[i];
        if (g.shape() != meta.shape)
          throw std::logic_error("autograd: backward of '" + node->name() + "' returned a gradient of shape " +
                                 shape_str(g.shape()) + " for input " + std::to_string(i) + " of shape " +
                                 shape_str(meta.shape));
        if(meta.context ? !compatible_context(g,*meta.context) : g.device().backend!=compute::Backend::CPU)
          throw std::logic_error("autograd: backward returned a gradient on the wrong device context");
        if (g.dtype() != meta.dtype) g = cast(g, meta.dtype);
        add_to_buffer(e.node.get(), e.input_nr, g);
      }
    }
  } catch (...) {
    set_grad_enabled(previous_mode);
    throw;
  }
  set_grad_enabled(previous_mode);

  if (capture && !allow_unused)
    for (std::size_t i = 0; i < captured.size(); ++i)
      if (!captured[i].defined())
        throw std::invalid_argument("grad: input " + std::to_string(i) +
                                    " is not used by the outputs (pass allow_unused to get an undefined gradient)");
  return captured;
}

}  // namespace

std::vector<Tensor> grad(const std::vector<Tensor>& outputs, const std::vector<Tensor>& inputs,
                         const std::vector<Tensor>& grad_outputs, const GradOptions& options) {
  std::vector<Edge> capture;
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    if (!inputs[i].defined() || !inputs[i].requires_grad())
      throw std::invalid_argument("grad: input " + std::to_string(i) + " does not require grad");
    capture.push_back(gradient_edge(inputs[i]));
  }
  if (options.allow_unused) {
    // Outputs without history contribute nothing: drop them (and their seeds)
    // so the inputs they would have reached come back undefined.
    if (!grad_outputs.empty() && grad_outputs.size() != outputs.size())
      throw std::invalid_argument("grad: got " + std::to_string(grad_outputs.size()) + " grad_outputs for " +
                                  std::to_string(outputs.size()) + " outputs");
    std::vector<Tensor> used_outputs, used_seeds;
    for (std::size_t i = 0; i < outputs.size(); ++i) {
      if (!outputs[i].defined() || !outputs[i].requires_grad()) continue;
      used_outputs.push_back(outputs[i]);
      if (!grad_outputs.empty()) used_seeds.push_back(grad_outputs[i]);
    }
    if (used_outputs.empty()) return std::vector<Tensor>(inputs.size());
    if (used_outputs.size() != outputs.size())
      return run_backward(used_outputs, used_seeds, options.retain_graph || options.create_graph,
                          options.create_graph, &capture, true);
  }
  return run_backward(outputs, grad_outputs, options.retain_graph || options.create_graph, options.create_graph,
                      &capture, options.allow_unused);
}

std::vector<Tensor> grad(const Tensor& output, const std::vector<Tensor>& inputs, const GradOptions& options) {
  return grad(std::vector<Tensor>{output}, inputs, {}, options);
}

void backward(const std::vector<Tensor>& outputs, const std::vector<Tensor>& grad_outputs, bool retain_graph,
              bool create_graph) {
  run_backward(outputs, grad_outputs, retain_graph || create_graph, create_graph, nullptr, true);
}

Tensor stop_gradient(const Tensor& x) { return x.detach(); }

// ---- custom functions ---------------------------------------------------------------------

std::vector<Tensor> apply_function(const std::string& name, const std::vector<Tensor>& inputs,
                                   const CustomForward& forward, const CustomBackward& backward) {
  std::vector<Tensor> outputs;
  {
    NoGradGuard         guard;
    std::vector<Tensor> detached;
    for (const auto& t : inputs) detached.push_back(detail::nograd(t));
    outputs = forward(detached);
  }
  // Outputs that alias an input (an identity forward) need their own impl to carry history.
  for (auto& out : outputs)
    for (const auto& in : inputs)
      if (out.defined() && in.defined() && out.impl()->storage == in.impl()->storage) {
        out = out.detach();
        break;
      }
  std::vector<SavedTensor> saved;
  for (const auto& t : inputs) saved.emplace_back(t);
  record(
      name, inputs, outputs,
      [saved, backward, name](const std::vector<Tensor>& g, const std::vector<Tensor>& outs) {
        std::vector<Tensor> ins;
        for (const auto& s : saved) ins.push_back(s.get());
        std::vector<Tensor> result = backward(ins, outs, g);
        if (result.size() != ins.size())
          throw std::logic_error("custom function '" + name + "': backward returned " +
                                 std::to_string(result.size()) + " gradients for " + std::to_string(ins.size()) +
                                 " inputs");
        return result;
      },
      true);
  return outputs;
}

// ---- gradient checkpointing ------------------------------------------------------------

bool in_recompute() { return recompute_depth > 0; }

namespace {

/// Marks this thread as recomputing, with grad mode on.
class RecomputeScope {
 public:
  /// @brief Marks entry into recomputation; the member guard has already enabled recording.
  RecomputeScope() { ++recompute_depth; }
  /// @brief Removes this recomputation level before the member guard restores recording mode.
  ~RecomputeScope() { --recompute_depth; }
  /// @brief Disallowed to keep recomputation depth increments and decrements balanced.
  RecomputeScope(const RecomputeScope&)            = delete;
  /// @brief Disallowed because an active recomputation scope cannot transfer its depth ownership.
  RecomputeScope& operator=(const RecomputeScope&) = delete;

 private:
  /// @brief Enables graph recording for recomputation and restores the previous thread mode.
  EnableGradGuard grad_;
};

/// Generator states captured before a checkpointed forward.
using RngSnapshot = std::vector<std::pair<Generator*, Generator>>;

/// @brief Copies the optional default generator and each non-null explicit generator before forward.
RngSnapshot snapshot_rng(const CheckpointOptions& options) {
  RngSnapshot snap;
  if (options.preserve_rng_state) snap.emplace_back(&default_generator(), default_generator());
  for (Generator* g : options.generators)
    if (g) snap.emplace_back(g, *g);
  return snap;
}

/// Puts the snapshot's states back for its lifetime, then restores the
/// states the generators had before.
class RngReplay {
 public:
  /// @brief Installs saved generator states and stashes the states to restore on scope exit.
  /// Generator pointers must remain valid for the entire replay scope.
  explicit RngReplay(const RngSnapshot& snap) {
    for (const auto& [gen, state] : snap) {
      stash_.emplace_back(gen, *gen);
      *gen = state;
    }
  }
  /// @brief Restores stashed generator states in reverse order, including repeated generator pointers.
  ~RngReplay() {
    for (auto it = stash_.rbegin(); it != stash_.rend(); ++it) *it->first = it->second;
  }
  /// @brief Disallowed so saved generator states are restored exactly once.
  RngReplay(const RngReplay&)            = delete;
  /// @brief Disallowed because assignment could discard an active generator restoration scope.
  RngReplay& operator=(const RngReplay&) = delete;

 private:
  /// @brief Generator states displaced by this replay, restored when it ends.
  RngSnapshot stash_;
};

/// A leaf read inside a checkpoint and its storage version at that time.
struct LeafVersion {
  /// @brief Weak reference to a captured leaf, without extending its lifetime.
  std::weak_ptr<TensorImpl> variable;
  /// @brief Expected leaf storage version before backward recomputation.
  std::uint64_t             version{0};
};

/// What a collector run of a forward function found.
struct Discovery {
  /// @brief Forward results, possibly carrying marker history; detached/reattached before return.
  std::vector<Tensor>      outputs;
  /// @brief Gradient destinations: explicit inputs first (possibly null), then outside tensors.
  std::vector<Edge>        edges;
  /// @brief Input shape/dtype information aligned with edges.
  std::vector<TensorMeta>  meta;
  /// @brief Weak leaf references and captured storage versions for mutation checks.
  std::vector<LeafVersion> leaves;
};

/// Runs `run` with the collector on: no graph is kept, and the edges through
/// which gradient enters from outside are listed (inputs first, in order).
Discovery discover(const std::function<std::vector<Tensor>()>& run, const std::vector<Tensor>& inputs) {
  Collector c;
  c.marker = std::make_shared<Node>("CheckpointMarker", std::vector<Edge>{}, std::vector<TensorMeta>{},
                                    std::vector<TensorMeta>{}, nullptr);
  Discovery d;
  for (const auto& t : inputs) {
    Edge e = gradient_edge(t);
    if (e.node) c.seen.insert({e.node.get(), e.input_nr});
    d.edges.push_back(e);
    d.meta.push_back(t.defined() ? TensorMeta{t.shape(), t.dtype(),t.context()} : TensorMeta{});
  }
  /// @brief Restores an outer collector even if the collected forward callback throws.
  struct Restore {
    /// @brief Thread-local collector that was active before this discovery run.
    Collector* previous;
    /// @brief Reinstalls the outer collector during exception unwinding or normal scope exit.
    ~Restore() { collector = previous; }
  } restore{collector};
  collector = &c;
  d.outputs = run();
  for (const auto& out : d.outputs) c.note(out);  // outside tensors returned unchanged
  collector = restore.previous;
  for (auto& e : c.boundary) {
    d.meta.push_back(e.node->output_meta().at(e.input_nr));
    d.edges.push_back(std::move(e));
  }
  for (const auto& e : d.edges)
    if (auto* acc = dynamic_cast<AccumulateGrad*>(e.node.get()))
      if (auto var = acc->variable()) d.leaves.push_back({var, var->storage->version});
  return d;
}

/// @brief Returns whether any discovered destination has a backward node.
bool has_edges(const std::vector<Edge>& edges) {
  for (const auto& e : edges)
    if (e.node) return true;
  return false;
}

/// @brief Rejects changed storage versions on still-live captured leaves; expired leaves are ignored.
/// @throws std::logic_error If a forward dependency was updated before recomputation.
void check_leaves(const std::vector<LeafVersion>& leaves, const std::string& what) {
  for (const auto& leaf : leaves)
    if (auto var = leaf.variable.lock())
      if (var->storage->version != leaf.version)
        throw std::logic_error(what + ": a tensor it reads was modified in place after the forward pass "
                                      "(for example an optimizer step before backward)");
}

/// Throws if the recomputed graph under `outputs` takes gradient from an
/// outside tensor that the forward pass did not use (a nondeterministic fn).
void check_boundary(const std::vector<Tensor>& outputs, std::uint64_t stop_before, const std::vector<Edge>& allowed,
                    const std::string& what) {
  std::unordered_set<EdgeKey, EdgeKeyHash> ok;
  for (const auto& e : allowed)
    if (e.node) ok.insert({e.node.get(), e.input_nr});
  std::unordered_set<Node*> visited;
  std::vector<Node*>        stack;
  // Validate outside destinations immediately; queue newly encountered recomputation nodes.
  auto                      visit = [&](const Edge& e) {
    if (!e.node) return;
    const bool outside = e.node->sequence_nr() < stop_before || dynamic_cast<AccumulateGrad*>(e.node.get());
    if (outside) {
      if (!ok.count({e.node.get(), e.input_nr}))
        throw std::logic_error(what + ": the recomputation reads a tensor that needs a gradient but the forward pass "
                                      "did not read (the function must compute the same thing every time)");
    } else if (visited.insert(e.node.get()).second) {
      stack.push_back(e.node.get());
    }
  };
  for (const auto& out : outputs)
    if (out.defined() && out.requires_grad()) visit(gradient_edge(out));
  while (!stack.empty()) {
    Node* n = stack.back();
    stack.pop_back();
    for (const auto& e : n->next_edges()) visit(e);
  }
}

/// Gives every float output that carries gradient a fresh handle whose
/// history is `node`; others are returned as plain (detached) tensors.
void attach(std::vector<Tensor>& outputs, const std::shared_ptr<Node>& node) {
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    Tensor& out = outputs[i];
    if (!out.defined()) continue;
    const bool carries = node && out.requires_grad() && is_floating(out.dtype());
    out                = out.detach();
    if (!carries) continue;
    out.impl()->grad_fn       = node;
    out.impl()->output_nr     = static_cast<std::uint32_t>(i);
    out.impl()->requires_grad = true;
  }
}

/// @brief Captures each tensor's metadata, using scalar Float32 defaults for undefined handles.
std::vector<TensorMeta> metas(const std::vector<Tensor>& ts) {
  std::vector<TensorMeta> m;
  for (const auto& t : ts) m.push_back(t.defined() ? TensorMeta{t.shape(), t.dtype(),t.context()} : TensorMeta{});
  return m;
}

/// Backward of `checkpoint`: re-runs the function on the saved inputs with
/// grad mode on, then backpropagates through that fresh graph alone.
class CheckpointNode : public Node {
 public:
  /// @brief Retains a discovered boundary, versioned inputs, callback, and RNG snapshot for replay.
  CheckpointNode(Discovery& d, std::vector<TensorMeta> output_meta, CheckpointFn fn, const std::vector<Tensor>& inputs,
                 RngSnapshot rng)
      : Node("Checkpoint", d.edges, d.meta, std::move(output_meta), nullptr),
        fn_ckpt_(std::move(fn)),
        rng_(std::move(rng)),
        leaves_(std::move(d.leaves)) {
    for (const auto& t : inputs) inputs_.emplace_back(t);
  }

  /// @brief Recomputes the forward with replayed RNG state and differentiates only its fresh graph.
  /// Validates captured leaf versions and boundary determinism before returning input gradients.
  std::vector<Tensor> apply(const std::vector<Tensor>& grad_outputs) override {
    if (released_)
      throw std::logic_error("autograd: tried to run backward through 'Checkpoint' a second time; its saved values "
                             "were freed. Pass retain_graph = true the first time.");
    check_leaves(leaves_, "checkpoint");
    const bool          create = is_grad_enabled();
    std::vector<Tensor> ins;
    for (const auto& s : inputs_) ins.push_back(s.get());
    const std::uint64_t stop = next_sequence_nr.load(std::memory_order_relaxed);
    std::vector<Tensor> outs;
    {
      RecomputeScope scope;
      RngReplay      replay(rng_);
      outs = fn_ckpt_(ins);
    }
    if (outs.size() != output_meta_.size())
      throw std::logic_error("checkpoint: the function returned " + std::to_string(outs.size()) +
                             " outputs when recomputed, " + std::to_string(output_meta_.size()) + " the first time");
    std::vector<Tensor> roots, seeds;
    for (std::size_t k = 0; k < outs.size(); ++k)
      if (outs[k].defined() && outs[k].requires_grad() && is_floating(outs[k].dtype()) && grad_outputs[k].defined()) {
        roots.push_back(outs[k]);
        seeds.push_back(grad_outputs[k]);
      }
    std::vector<Tensor> result(next_edges_.size());
    if (roots.empty()) return result;
    check_boundary(roots, stop, next_edges_, "checkpoint");
    outs.clear();
    result = run_backward(roots, seeds, create, create, &next_edges_, true, stop);
    // An edge listed twice (one tensor passed as two inputs) gets its gradient once.
    std::unordered_set<EdgeKey, EdgeKeyHash> seen;
    for (std::size_t i = 0; i < result.size(); ++i)
      if (next_edges_[i].node && !seen.insert({next_edges_[i].node.get(), next_edges_[i].input_nr}).second)
        result[i] = Tensor();
    return result;
  }

  /// @brief Releases replay inputs, the callback, and generator snapshots after backward.
  void release() override {
    Node::release();
    fn_ckpt_ = nullptr;
    inputs_.clear();
    rng_.clear();
  }

 private:
  /// @brief Forward segment to recompute; cleared on release().
  CheckpointFn             fn_ckpt_;
  /// @brief Explicit forward inputs with storage-version checks.
  std::vector<SavedTensor> inputs_;
  /// @brief Pre-forward generator states to replay during backward recomputation.
  RngSnapshot              rng_;
  /// @brief Captured leaves whose versions must match the initial forward.
  std::vector<LeafVersion> leaves_;
};

/// Backward of `reversible_sequence`: rebuilds each block's input from its
/// output, last block first, and backpropagates through one block at a time.
class ReversibleNode : public Node {
 public:
  /// @brief Retains coupling blocks and their discovered external gradient dependencies.
  ReversibleNode(Discovery& d, std::vector<TensorMeta> output_meta, std::vector<ReversibleBlock> blocks)
      : Node("ReversibleSequence", d.edges, d.meta, std::move(output_meta), nullptr),
        blocks_(std::move(blocks)),
        leaves_(std::move(d.leaves)) {}

  /// @brief Reconstructs each block's inputs and propagates its gradients, from last block to first.
  /// @throws std::invalid_argument If higher-order gradient recording is requested.
  /// @throws std::logic_error For released state, changed saved storage, or a changed boundary.
  std::vector<Tensor> apply(const std::vector<Tensor>& grad_outputs) override {
    if (released_)
      throw std::logic_error("autograd: tried to run backward through 'ReversibleSequence' a second time; its saved "
                             "values were freed. Pass retain_graph = true the first time.");
    if (is_grad_enabled())
      throw std::invalid_argument("reversible_sequence: create_graph (second order) is not supported; use checkpoint");
    check_leaves(leaves_, "reversible_sequence");
    for (std::size_t i = 0; i < saved_outputs_.size(); ++i)
      if (saved_outputs_[i].impl()->storage->version != saved_versions_[i])
        throw std::logic_error("reversible_sequence: an output was modified in place before backward");
    Tensor              y1 = saved_outputs_[0], y2 = saved_outputs_[1];
    Tensor              dy1 = grad_outputs[0], dy2 = grad_outputs[1];
    std::vector<Tensor> acc(next_edges_.size());
    // Backpropagates `seed` from `out` (computed from leaf `x`) and returns
    // d/dx, adding the outside gradients into `acc`.
    auto through = [&](const Tensor& out, const Tensor& seed, const Tensor& x, std::uint64_t stop) -> Tensor {
      if (!out.requires_grad()) return Tensor();
      std::vector<Edge> capture{gradient_edge(x)};
      capture.insert(capture.end(), next_edges_.begin(), next_edges_.end());
      check_boundary({out}, stop, capture, "reversible_sequence");
      auto r = run_backward({out}, {seed}, false, false, &capture, true, stop);
      for (std::size_t i = 1; i < r.size(); ++i) acc[i - 1] = accumulate(acc[i - 1], r[i]);
      return r[0];
    };
    for (std::size_t b = blocks_.size(); b-- > 0;) {
      const auto& block = blocks_[b];
      // y2 = x2 + g(y1): backprop g, then x2 = y2 - g(y1).
      std::uint64_t stop = next_sequence_nr.load(std::memory_order_relaxed);
      Tensor        z1   = y1.detach();
      z1.requires_grad_();
      Tensor gz;
      {
        RecomputeScope scope;
        gz = block.g(z1);
      }
      Tensor dz1 = accumulate(dy1, through(gz, dy2, z1, stop));
      Tensor x2  = sub(y2, gz.detach());
      gz         = Tensor();
      // y1 = x1 + f(x2): backprop f, then x1 = y1 - f(x2).
      stop       = next_sequence_nr.load(std::memory_order_relaxed);
      Tensor x2l = x2.detach();
      x2l.requires_grad_();
      Tensor fx;
      {
        RecomputeScope scope;
        fx = block.f(x2l);
      }
      Tensor dx2 = accumulate(dy2, through(fx, dz1, x2l, stop));
      Tensor x1  = sub(y1, fx.detach());
      fx         = Tensor();
      y1         = x1;
      y2         = x2;
      dy1        = dz1;
      dy2        = dx2;
    }
    acc[0] = accumulate(acc[0], dy1);
    acc[1] = accumulate(acc[1], dy2);
    return acc;
  }

  /// @brief Releases the coupling callbacks and base-node saved output state.
  void release() override {
    Node::release();
    blocks_.clear();
  }

 private:
  /// @brief Owned forward coupling callbacks, traversed in reverse during backward.
  std::vector<ReversibleBlock> blocks_;
  /// @brief Leaf storage versions captured when the reversible forward was discovered.
  std::vector<LeafVersion>     leaves_;
};

}  // namespace

std::vector<Tensor> checkpoint(const CheckpointFn& fn, const std::vector<Tensor>& inputs,
                               const CheckpointOptions& options) {
  if (!fn) throw std::invalid_argument("checkpoint: empty function");
  // Nothing to save, or an enclosing checkpoint is already collecting.
  if (!is_grad_enabled() || collector) return fn(inputs);
  RngSnapshot rng = snapshot_rng(options);
  Discovery   d   = discover([&] { return fn(inputs); }, inputs);
  if (!has_edges(d.edges)) {
    attach(d.outputs, nullptr);
    return d.outputs;
  }
  auto node = std::make_shared<CheckpointNode>(d, metas(d.outputs), fn, inputs, std::move(rng));
  attach(d.outputs, node);
  return d.outputs;
}

Tensor checkpoint(const CheckpointFn1& fn, const Tensor& input, const CheckpointOptions& options) {
  if (!fn) throw std::invalid_argument("checkpoint: empty function");
  return checkpoint([fn](const std::vector<Tensor>& in) { return std::vector<Tensor>{fn(in[0])}; },
                    std::vector<Tensor>{input}, options)
      .at(0);
}

Tensor checkpoint_sequential(const std::vector<CheckpointFn1>& layers, std::int64_t segments, const Tensor& input,
                             const CheckpointOptions& options) {
  const auto n = static_cast<std::int64_t>(layers.size());
  if (segments < 1) throw std::invalid_argument("checkpoint_sequential: segments must be >= 1");
  segments = std::min(segments, std::max<std::int64_t>(n, 1));
  Tensor x = input;
  for (std::int64_t s = 0; s < segments; ++s) {
    const std::int64_t begin = s * n / segments, end = (s + 1) * n / segments;
    // The segment owns copies of its layers: it runs again during backward.
    std::vector<CheckpointFn1> part(layers.begin() + begin, layers.begin() + end);
    auto run = [part](const Tensor& in) {
      Tensor h = in;
      for (const auto& layer : part) h = layer(h);
      return h;
    };
    x = s + 1 < segments ? checkpoint(CheckpointFn1(run), x, options) : run(x);
  }
  return x;
}

std::vector<Tensor> reversible_sequence(const std::vector<ReversibleBlock>& blocks, const Tensor& x1,
                                        const Tensor& x2) {
  for (const auto& b : blocks)
    if (!b.f || !b.g) throw std::invalid_argument("reversible_sequence: every block needs f and g");
  // Apply the additive coupling chain; discovery and ordinary evaluation use the same closure.
  auto run = [&]() {
    Tensor a = x1, b = x2;
    for (const auto& block : blocks) {
      a = add(a, block.f(b));
      b = add(b, block.g(a));
    }
    return std::vector<Tensor>{a, b};
  };
  if (!is_grad_enabled() || collector) return run();
  Discovery d = discover(run, {x1, x2});
  if (!has_edges(d.edges)) {
    attach(d.outputs, nullptr);
    return d.outputs;
  }
  for (const auto& out : d.outputs)
    if (!is_floating(out.dtype())) throw std::invalid_argument("reversible_sequence: needs float tensors");
  auto node = std::make_shared<ReversibleNode>(d, metas(d.outputs), blocks);
  attach(d.outputs, node);
  node->save_outputs(d.outputs);
  return d.outputs;
}

}  // namespace datamunge::nn
