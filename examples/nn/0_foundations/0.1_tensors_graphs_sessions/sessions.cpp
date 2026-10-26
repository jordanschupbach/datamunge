// CMD: just run-nn sessions
// Sessions: feeds, fetches, targets, RunOptions, callables.
//
// Port of tflow's sessions. A TF Session owns a copy of the graph and runs
// subsets of it: each call names what to feed, what to fetch and which ops to
// target for their side effects. datamunge::nn is eager (plan D2), so there
// is no session: the "graph" is ordinary C++ code and each concept maps to a
// plain-language construct, shown section by section:
//   SessionOptions      -> nn::set_num_threads (intra-op); there is no
//                          inter-op pool or soft placement (CPU only)
//   feeds and fetches   -> function arguments and return values
//   feeding an          -> call the later half of the computation directly
//     intermediate
//   missing feed        -> an undefined Tensor argument throws
//   targets             -> call a function for its side effect (a counter
//                          tensor updated in place)
//   RunOptions tracing  -> nn records no per-op trace; the traced run below
//                          times each stage with std::chrono instead
//   callables           -> a std::function capturing the signature
// The numbers (y, y_sum, the step counter) are identical to TF's; the timings
// compare a direct call with a std::function call instead of Run vs
// RunCallable.

#include <chrono>
#include <cstdio>
#include <datamunge/nn/nn.hpp>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include "common/checks.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;

namespace {

void Show(const std::string& label, const Tensor& t) {
  std::cout << "  " << label << ": " << nn::shape_str(t.shape()) << " " << t.values_str(32) << "\n";
}

Tensor Batch(int rows) {
  std::vector<float> v;
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < 3; ++c) v.push_back(static_cast<float>(r + c));
  return nn::from_vector(v, {rows, 3});
}

// The "graph": y = x @ w + b, plus a step counter updated as a side effect.
struct Model {
  Tensor w    = nn::tensor({{1, 0}, {0, 1}, {1, 1}});
  Tensor b    = nn::tensor({0.5, -0.5});
  Tensor step = nn::scalar(0, nn::DType::Int64);  // TF Variable(int64)

  // A shape check plays the role of Placeholder::Shape({-1, 3}); an undefined
  // tensor is the eager analogue of an unfed placeholder.
  Tensor xw(const Tensor& x) const {
    if (!x.defined()) throw std::invalid_argument("x: no value supplied (the eager form of a missing feed)");
    if (x.dim() != 2 || x.shape()[1] != 3)
      throw std::invalid_argument("x: expected shape [?, 3], got " + nn::shape_str(x.shape()));
    return nn::matmul(x, w);
  }
  Tensor y_from_xw(const Tensor& xw_value) const { return nn::add(xw_value, b); }
  Tensor y(const Tensor& x) const { return y_from_xw(xw(x)); }
  static Tensor y_sum(const Tensor& y) { return nn::sum(y); }
  void          step_init() { step.fill_(0); }
  void          step_inc() { step.add_(nn::scalar(1, nn::DType::Int64)); }
};

}  // namespace

int main() {
  try {
    Model m;

    // ------------------------------------------------------------------------
    // SessionOptions: thread pools, placement, logging, ...
    // ------------------------------------------------------------------------
    Header("SessionOptions");
    const int default_threads = nn::get_num_threads();
    nn::set_num_threads(2);  // threads inside one op (TF intra_op_parallelism_threads)
    std::cout << "  target: in-process, eager (there is no session or remote target)\n"
              << "  config: intra_op_threads " << nn::get_num_threads() << " (default " << default_threads
              << "), gemm " << nn::gemm_backend() << "; no inter-op pool, no placement (CPU only)\n";
    check::Require(nn::get_num_threads() == 2, "set_num_threads");

    // ------------------------------------------------------------------------
    // Feeds and fetches
    // ------------------------------------------------------------------------
    Header("Feeds and fetches");
    Tensor y2 = m.y(Batch(2)), s2 = Model::y_sum(y2);
    Show("y (batch 2)", y2);
    Show("y_sum", s2);
    check::Require(y2.to_vector<float>() == std::vector<float>{2.5f, 2.5f, 4.5f, 4.5f}, "y for batch 2");
    check::Near(s2.item<float>(), 14.0, 0, "y_sum for batch 2");
    // Any batch size works with the same code.
    Tensor y5 = m.y(Batch(5));
    Show("y (batch 5)", y5);
    check::Require(y5.shape() == nn::Shape({5, 2}), "y for batch 5 has shape [5, 2]");

    // TF can feed *any* tensor, skipping the ops that produce it; eagerly that
    // is just calling the later stage with a value of our choosing.
    const Tensor yfed = m.y_from_xw(nn::tensor({{100, 200}}));
    Show("y with xw fed", yfed);
    check::Require(yfed.to_vector<float>() == std::vector<float>{100.5f, 199.5f}, "feeding an intermediate");

    // Computing y without a value for x is an error reported as an exception.
    const std::string missing = check::Throws([&] { (void)m.y(Tensor()); }, "missing feed");
    std::cout << "  missing feed -> " << missing.substr(0, 70) << "...\n";

    // ------------------------------------------------------------------------
    // Targets: run ops for their side effects, fetching nothing
    // ------------------------------------------------------------------------
    Header("Targets");
    m.step_init();
    for (int i = 0; i < 3; ++i) m.step_inc();
    std::cout << "  step after 3 target-only runs: " << m.step.item<std::int64_t>() << "\n";
    check::Require(m.step.item<std::int64_t>() == 3, "step counter after 3 increments");
    // Fetch and side effect in one "step".
    Tensor y1 = m.y(Batch(1));
    m.step_inc();
    Show("y (combined run)", y1);

    // ------------------------------------------------------------------------
    // RunOptions and RunMetadata: tracing
    // ------------------------------------------------------------------------
    Header("RunOptions / RunMetadata");
    // nn has no per-op tracer; time each stage of the computation explicitly.
    using clock = std::chrono::steady_clock;
    auto         us    = [](clock::time_point a, clock::time_point b) {
      return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count();
    };
    const Tensor x64   = Batch(64);
    const auto   t_beg = clock::now();
    const Tensor xw    = m.xw(x64);
    const auto   t_mm  = clock::now();
    const Tensor y     = m.y_from_xw(xw);
    const auto   t_add = clock::now();
    const Tensor ys    = Model::y_sum(y);
    const auto   t_sum = clock::now();
    std::cout << "  per-stage timings (std::chrono):\n    device /device:CPU:0\n"
              << "      xw (MatMul)  start+0us  op " << us(t_beg, t_mm) << "us\n"
              << "      y (Add)      start+" << us(t_beg, t_mm) << "us  op " << us(t_mm, t_add) << "us\n"
              << "      y_sum (Sum)  start+" << us(t_beg, t_add) << "us  op " << us(t_add, t_sum) << "us\n";
    check::Near(ys.item<float>(), 8384, 0, "traced y_sum");  // sum over r < 64 of 2 (2r + 2.5)

    // ------------------------------------------------------------------------
    // Callables: a pre-built feed/fetch/target signature
    // ------------------------------------------------------------------------
    Header("Callables");
    // feeds: x; fetches: y_sum; target: step_inc.
    const std::function<Tensor(const Tensor&)> callable = [&m](const Tensor& x) {
      Tensor out = Model::y_sum(m.y(x));
      m.step_inc();
      return out;
    };
    const Tensor cs = callable(Batch(2));
    Show("y_sum via callable", cs);
    check::Near(cs.item<float>(), 14.0, 0, "callable y_sum");

    const int    iters = 2000;
    const Tensor batch = Batch(8);
    Tensor       out;
    const auto   t0 = clock::now();
    for (int i = 0; i < iters; ++i) {
      out = Model::y_sum(m.y(batch));
      m.step_inc();
    }
    const auto t1 = clock::now();
    for (int i = 0; i < iters; ++i) out = callable(batch);
    const auto t2 = clock::now();
    std::cout << "  " << iters << " x direct call:   " << us(t0, t1) / iters << " us/step\n"
              << "  " << iters << " x std::function: " << us(t1, t2) / iters << " us/step\n";

    std::cout << "  final step counter: " << m.step.item<std::int64_t>() << "\n";
    // 3 + 1 (combined) + 1 (callable) + 2000 + 2000, as in TF.
    check::Require(m.step.item<std::int64_t>() == 4005, "final step counter");
    nn::set_num_threads(0);  // restore the default (TF Session::Close)
    std::cout << "\nAll session checks passed.\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
  return 0;
}
