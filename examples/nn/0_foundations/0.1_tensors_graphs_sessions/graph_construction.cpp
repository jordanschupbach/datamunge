// CMD: just run-nn graph_construction
// Graph construction: naming, control dependencies, device placement, errors.
//
// Port of tflow's graph_construction. In TF's C++ API a graph is built through
// a tensorflow::Scope that names ops, places them on devices, attaches
// control dependencies and collects errors; the graph then runs in a Session.
// datamunge::nn is eager (plan D2): ops run when they are called, and the only
// graph is the autograd tape that tensors requiring gradients record. The
// four TF sections map as follows:
//   1. Scopes and op names -> names live where they matter in eager code: the
//      ParameterStore ("encoder/layer/w"), where a repeated name means *reuse*
//      (weight sharing) instead of TF's uniquified "block_1". The recorded
//      tape stands in for the GraphDef dump: each node prints its op name and
//      the nodes (or named leaves) its gradient flows to.
//   2. Control dependencies -> program order. The counter values match TF's
//      read_plain / read_after_inc / read_cleared exactly.
//   3. Device placement -> nn runs on the CPU only (GPU is out of scope), so
//      there is no device request to honour or to soften; the result of the
//      placed computation matches TF's soft-placement run.
//   4. Errors during construction -> a bad op throws at the call; there is no
//      sticky Scope status that skips later ops.

#include <datamunge/nn/nn.hpp>
#include <deque>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "common/checks.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;

namespace {

// Print the autograd tape that produced `out`: one line per node, newest
// first, with the nodes (or named leaf tensors) its gradient flows to. Leaves
// are named through `names`; this is the eager counterpart of DumpGraph.
void DumpTape(const Tensor& out, const std::map<const nn::TensorImpl*, std::string>& names) {
  std::map<const nn::Node*, std::string>  label;
  std::vector<std::shared_ptr<nn::Node>>  order;
  std::deque<std::shared_ptr<nn::Node>>   queue;
  std::set<const nn::Node*>               seen;
  auto                                    root = nn::gradient_edge(out).node;
  if (!root) {
    std::cout << "  (no tape: the output does not require grad)\n";
    return;
  }
  queue.push_back(root);
  while (!queue.empty()) {
    auto n = queue.front();
    queue.pop_front();
    if (!seen.insert(n.get()).second) continue;
    order.push_back(n);
    for (const auto& e : n->next_edges())
      if (e.node) queue.push_back(e.node);
  }
  auto name_of = [&](const std::shared_ptr<nn::Node>& n) -> std::string {
    if (auto acc = std::dynamic_pointer_cast<nn::AccumulateGrad>(n)) {
      auto var = acc->variable();
      auto it  = names.find(var.get());
      return it != names.end() ? it->second : std::string("<leaf>");
    }
    return n->name() + "_" + std::to_string(n->sequence_nr());
  };
  for (const auto& n : order) {
    if (std::dynamic_pointer_cast<nn::AccumulateGrad>(n)) continue;  // leaves are listed as inputs
    std::cout << "  " << name_of(n) << "  [" << n->name() << "]  inputs=";
    for (const auto& e : n->next_edges()) std::cout << (e.node ? name_of(e.node) : std::string("<const>")) << " ";
    std::cout << "\n";
  }
}

// ---------------------------------------------------------------------------
// 1. Names
// ---------------------------------------------------------------------------

void Naming() {
  Header("Names: parameters and the recorded tape");
  nn::ParameterStore store(1);
  // A parameter's name is a path; prefixes play the role of sub-scopes.
  Tensor a = store.get("a", {}, nn::init::constant(1));
  Tensor b = nn::scalar(2), c = nn::scalar(3);  // constants: no name, no history
  Tensor w = store.get("encoder/w", {2, 2}, [](const nn::Shape&, nn::DType d, nn::Generator&) {
    return nn::tensor({{1, 0}, {0, 1}}, d);
  });
  Tensor h1 = nn::add(a, b);   // "encoder/layer/Add" in TF
  Tensor h2 = nn::add(h1, c);  // "encoder/layer/Add_1"
  std::cout << "  parameter names:        ";
  for (const auto& n : store.names()) std::cout << n << " ";
  std::cout << "\n  op names on the tape:   h1 <- " << h1.grad_fn_name() << ", h2 <- " << h2.grad_fn_name() << "\n";

  // TF makes a repeated sub-scope unique ("block", "block_1"). A repeated
  // parameter name in nn is the same parameter: that is how layers share weights.
  Tensor p = store.get("block/k", {}, nn::init::constant(5));
  Tensor q = store.get("block/k", {}, nn::init::constant(7));  // initializer ignored: reused
  std::cout << "  repeated name \"block/k\": same tensor? " << (p.is_same(q) ? "yes" : "no") << ", value "
            << q.item<float>() << "\n";
  check::Require(p.is_same(q) && q.item<float>() == 5.f, "a repeated parameter name reuses the parameter");
  check::Require(store.size() == 3, "three parameters (a, encoder/w, block/k)");
  check::Near(h2.item<float>(), 6.0, 0, "h2 = a + b + c");

  Header("Tape of the computation above (the GraphDef counterpart)");
  Tensor y = nn::sum(nn::matmul(nn::reshape(nn::mul(h2, p), {1, 1}) * nn::ones({1, 2}), w));
  std::map<const nn::TensorImpl*, std::string> names;
  for (const auto& n : store.names()) names[store.at(n).impl()] = n;
  DumpTape(y, names);
  // Gradients follow the tape: dy/da = p * sum(w) = 5 * 2.
  const auto g = nn::grad(y, {a, w});
  std::cout << "  dy/da = " << g[0].item<float>() << ", dy/dw = " << g[1].values_str() << "\n";
  check::Near(g[0].item<float>(), 10.0, 0, "dy/da");
}

// ---------------------------------------------------------------------------
// 2. Control dependencies
// ---------------------------------------------------------------------------

void ControlDependencies() {
  Header("Control dependencies (program order)");
  Tensor counter = nn::scalar(0, nn::DType::Int32);
  auto   init    = [&] { counter.fill_(0); };
  auto   inc     = [&] { counter.add_(nn::scalar(1, nn::DType::Int32)); };
  // Reading has no dependency on inc...
  auto read_plain = [&] { return counter.clone(); };
  // ...unless the code runs inc first: in eager code a control dependency is
  // simply a statement placed before the read.
  auto read_after_inc = [&] {
    inc();
    return counter.clone();
  };
  // TF's WithNoControlDependencies: a read that does not run inc.
  auto read_cleared = read_plain;

  init();
  std::vector<int> got;
  for (int i = 0; i < 3; ++i) {
    got.push_back(read_plain().item<std::int32_t>());
    std::cout << "  read_plain      -> " << got.back() << "\n";
  }
  for (int i = 0; i < 3; ++i) {
    got.push_back(read_after_inc().item<std::int32_t>());
    std::cout << "  read_after_inc  -> " << got.back() << "\n";
  }
  got.push_back(read_cleared().item<std::int32_t>());
  std::cout << "  read_cleared    -> " << got.back() << " (no increment)\n";
  check::Require(got == std::vector<int>{0, 0, 0, 1, 2, 3, 3}, "counter reads match TF (0 0 0 1 2 3 3)");
}

// ---------------------------------------------------------------------------
// 3. Device placement
// ---------------------------------------------------------------------------

void DevicePlacement() {
  Header("Device placement");
  const Tensor a        = nn::tensor({{1, 2}, {3, 4}});
  const Tensor on_cpu   = nn::matmul(a, a);
  const Tensor on_gpu   = nn::matmul(a, a);  // there is no GPU device: it runs on the CPU too
  const Tensor unplaced = nn::add(on_cpu, on_gpu);
  std::cout << "  devices: /device:CPU:0 only (GPU, and so placement requests, are out of scope)\n"
            << "  strict placement:   not applicable (no device can be requested)\n"
            << "  result:             " << unplaced.values_str() << " (TF soft placement: same values)\n";
  check::Require(unplaced.to_vector<float>() == std::vector<float>{14, 20, 30, 44}, "placed matmul result");
}

// ---------------------------------------------------------------------------
// 4. Error handling
// ---------------------------------------------------------------------------

void Errors() {
  Header("Errors during construction");
  // The bad op throws immediately with the shapes involved; nothing is
  // poisoned, so later ops run normally (TF: the Scope keeps the first error
  // and skips every later op, and ToGraphDef refuses to serialise).
  const Tensor      a       = nn::tensor({{1, 2, 3}, {4, 5, 6}});  // [2,3]
  const std::string message = check::Throws([&] { (void)nn::matmul(a, a); }, "matmul [2,3] x [2,3]");
  std::cout << "  matmul([2,3],[2,3]) threw: " << message << "\n";
  const Tensor later = nn::neg(a);
  std::cout << "  op run after the error has a value: " << later.defined() << " " << later.values_str() << "\n";
  check::Require(later.defined() && later.to_vector<float>()[5] == -6.f, "ops after an error still run");
}

}  // namespace

int main() {
  try {
    Naming();
    ControlDependencies();
    DevicePlacement();
    Errors();
    std::cout << "\nAll graph construction checks passed.\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
  return 0;
}
