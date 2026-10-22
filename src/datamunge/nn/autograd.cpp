#include "datamunge/nn/autograd.hpp"

#include <algorithm>
#include <map>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include "datamunge/nn/ops.hpp"
#include "detail.hpp"

namespace datamunge::nn {

namespace {
thread_local bool grad_enabled = true;
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
      fn_(std::move(fn)) {}

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
    : Node("AccumulateGrad", {}, {{variable->shape, variable->dtype}}, {{variable->shape, variable->dtype}}, nullptr),
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
  std::vector<Edge>       edges;
  std::vector<TensorMeta> input_meta;
  for (const auto& t : inputs) {
    edges.push_back(gradient_edge(t));
    input_meta.push_back(t.defined() ? TensorMeta{t.shape(), t.dtype()} : TensorMeta{});
  }
  std::vector<TensorMeta> output_meta;
  for (const auto& t : outputs) output_meta.push_back({t.shape(), t.dtype()});
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

struct EdgeKey {
  Node*         node;
  std::uint32_t slot;
  bool          operator==(const EdgeKey& o) const { return node == o.node && slot == o.slot; }
};
struct EdgeKeyHash {
  std::size_t operator()(const EdgeKey& k) const {
    return std::hash<const void*>()(k.node) ^ (static_cast<std::size_t>(k.slot) * 0x9e3779b97f4a7c15ULL);
  }
};

Tensor accumulate(const Tensor& a, const Tensor& b) {
  if (!a.defined()) return b;
  if (!b.defined()) return a;
  return add(a, b);
}

/// Shared implementation of grad() and backward(). In capture mode the
/// gradients arriving at `capture` edges are returned and leaf accumulators
/// are not run; otherwise every reachable accumulator runs.
std::vector<Tensor> run_backward(const std::vector<Tensor>& outputs, const std::vector<Tensor>& grad_outputs,
                                 bool retain_graph, bool create_graph, const std::vector<Edge>* capture,
                                 bool allow_unused) {
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
      seed = detail::as_dtype(seed, out.dtype());
    } else {
      if (out.numel() != 1)
        throw std::invalid_argument("grad: output " + std::to_string(i) + " has shape " + shape_str(out.shape()) +
                                    "; non-scalar outputs need an explicit grad_output");
      seed = ones_like(out);
    }
    roots.emplace_back(gradient_edge(out), seed);
  }

  // Capture targets.
  std::unordered_map<EdgeKey, std::vector<std::size_t>, EdgeKeyHash> capture_index;
  std::unordered_set<Node*>                                         capture_nodes;
  if (capture) {
    for (std::size_t i = 0; i < capture->size(); ++i) {
      const Edge& e = (*capture)[i];
      if (!e.node) continue;
      capture_index[{e.node.get(), e.input_nr}].push_back(i);
      capture_nodes.insert(e.node.get());
    }
  }

  // Discover the graph (iterative DFS) and keep nodes that matter: in
  // capture mode, those with a path to a capture node.
  std::unordered_map<Node*, std::shared_ptr<Node>> nodes;
  std::vector<Node*>                               postorder;
  {
    std::unordered_set<Node*>                           visited;
    std::vector<std::pair<std::shared_ptr<Node>, std::size_t>> stack;
    for (auto& r : roots)
      if (r.first.node && visited.insert(r.first.node.get()).second) stack.emplace_back(r.first.node, 0);
    while (!stack.empty()) {
      auto& [node, next] = stack.back();
      if (next < node->next_edges().size()) {
        const Edge& e = node->next_edges()[next++];
        if (e.node && visited.insert(e.node.get()).second) stack.emplace_back(e.node, 0);
        continue;
      }
      nodes[node.get()] = node;
      postorder.push_back(node.get());
      stack.pop_back();
    }
  }
  std::unordered_set<Node*> needed;
  for (Node* n : postorder) {  // children come before parents in postorder
    bool keep = capture ? capture_nodes.count(n) > 0 : true;
    if (!keep)
      for (const Edge& e : n->next_edges())
        if (e.node && needed.count(e.node.get())) {
          keep = true;
          break;
        }
    if (keep) needed.insert(n);
  }

  std::unordered_map<Node*, std::vector<Tensor>> buffers;
  auto add_to_buffer = [&](Node* node, std::uint32_t slot, const Tensor& g) {
    auto& buf = buffers[node];
    if (buf.empty()) buf.resize(node->output_meta().size());
    buf[slot] = accumulate(buf[slot], g);
  };
  for (auto& r : roots)
    if (r.first.node && needed.count(r.first.node.get())) add_to_buffer(r.first.node.get(), r.first.input_nr, r.second);

  std::vector<Tensor> captured(capture ? capture->size() : 0);

  // Process in reverse postorder: every node runs after all its consumers.
  bool previous_mode = is_grad_enabled();
  set_grad_enabled(create_graph);
  try {
    for (auto it = postorder.rbegin(); it != postorder.rend(); ++it) {
      Node* node = *it;
      if (!needed.count(node)) continue;
      auto bit = buffers.find(node);
      if (bit == buffers.end()) {
        if (!retain_graph) node->release();
        continue;  // nothing flowed here
      }
      std::vector<Tensor> grads = std::move(bit->second);
      buffers.erase(bit);

      if (capture) {
        for (std::uint32_t slot = 0; slot < grads.size(); ++slot) {
          auto c = capture_index.find({node, slot});
          if (c == capture_index.end()) continue;
          for (auto idx : c->second) captured[idx] = grads[slot];
        }
        if (dynamic_cast<AccumulateGrad*>(node)) continue;
        bool upstream = false;
        for (const Edge& e : node->next_edges())
          if (e.node && needed.count(e.node.get())) upstream = true;
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
          grads[s]      = zeros(m.shape, m.dtype);
        }
      }
      if (!any) continue;

      std::vector<Tensor> input_grads = node->apply(grads);
      if (!retain_graph) node->release();
      const auto& edges = node->next_edges();
      if (!input_grads.empty() && input_grads.size() != edges.size())
        throw std::logic_error("autograd: backward of '" + node->name() + "' returned " +
                               std::to_string(input_grads.size()) + " gradients for " +
                               std::to_string(edges.size()) + " inputs");
      for (std::size_t i = 0; i < input_grads.size(); ++i) {
        const Edge& e = edges[i];
        if (!e.node || !needed.count(e.node.get()) || !input_grads[i].defined()) continue;
        const auto& meta = node->input_meta()[i];
        Tensor      g    = input_grads[i];
        if (g.shape() != meta.shape)
          throw std::logic_error("autograd: backward of '" + node->name() + "' returned a gradient of shape " +
                                 shape_str(g.shape()) + " for input " + std::to_string(i) + " of shape " +
                                 shape_str(meta.shape));
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

}  // namespace datamunge::nn
