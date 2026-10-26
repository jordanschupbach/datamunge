// CMD: just run-nn variables_initializers
// Variables and initializers (Glorot, He, orthogonal).
//
// Port of tflow's variables_initializers. A TF Variable owns a mutable buffer
// that persists across Session::Run calls and starts uninitialized; its
// initial value is a subgraph, which is where initialization schemes live.
// In datamunge::nn (eager, plan D2) a variable is just a tensor that is
// updated in place (add_, mul_, copy_), usually held in a ParameterStore, and
// an initializer is a function that returns a tensor. As in TF, Glorot, He and
// orthogonal initializers are written here from the random and linear-algebra
// ops (nn::init:: has library versions of each).
//
// Deviations: there is no "uninitialized" state (a default-constructed Tensor
// is undefined, and reading it throws); there is no session that owns
// variable copies, so "same graph, new session" becomes ParameterStore::clone,
// an independent copy; TF's two variable flavours (ref Variable, resource
// VarHandleOp) are one concept here. Random draws come from nn::Generator, so
// sampled values and statistics match TF only statistically; the expected
// standard deviations and the orthogonality errors are checked.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <datamunge/nn/nn.hpp>
#include <iostream>
#include <string>
#include <vector>

#include "common/checks.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;

namespace {

struct Stats {
  double mean = 0, stddev = 0, min = 0, max = 0;
};

Stats ComputeStats(const Tensor& t) {
  const auto f = t.to_vector<float>();
  Stats      s;
  s.min = s.max = f[0];
  double sum = 0, sq = 0;
  for (float v : f) {
    sum += v;
    sq += double(v) * v;
    s.min = std::min<double>(s.min, v);
    s.max = std::max<double>(s.max, v);
  }
  s.mean   = sum / f.size();
  s.stddev = std::sqrt(std::max(0.0, sq / f.size() - s.mean * s.mean));
  return s;
}

Stats PrintStats(const std::string& label, const Tensor& t, double expected_std) {
  const Stats s = ComputeStats(t);
  std::printf("  %-28s shape %-9s mean %+.4f  std %.4f (expected %.4f)  range [%+.3f, %+.3f]\n", label.c_str(),
              nn::shape_str(t.shape()).c_str(), s.mean, s.stddev, expected_std, s.min, s.max);
  return s;
}

// ---------------------------------------------------------------------------
// Initializers, written from the random ops. Each takes its own seed, as the
// TF version gives each random op a Seed attr.
// ---------------------------------------------------------------------------

// Uniform in [-limit, limit).
Tensor Uniform(const nn::Shape& shape, double limit, std::uint64_t seed) {
  nn::Generator g(seed);
  return nn::uniform(shape, -limit, limit, g);
}

// Normal with the given standard deviation.
Tensor Normal(const nn::Shape& shape, double stddev, std::uint64_t seed) {
  nn::Generator g(seed);
  return nn::normal(shape, 0.0, stddev, g);
}

// Truncated normal (redrawn beyond 2 sigma) with the given standard deviation.
// Keras's VarianceScaling divides by 0.8796 so that the truncated distribution
// has the requested std; we do the same.
Tensor TruncatedNormal(const nn::Shape& shape, double stddev, std::uint64_t seed) {
  nn::Generator g(seed);
  return nn::truncated_normal(shape, 0.0, stddev / 0.87962566103423978, g);
}

// Glorot (Xavier): variance 2 / (fan_in + fan_out).
Tensor GlorotUniform(int fan_in, int fan_out, std::uint64_t seed) {
  return Uniform({fan_in, fan_out}, std::sqrt(6.0 / (fan_in + fan_out)), seed);
}
Tensor GlorotNormal(int fan_in, int fan_out, std::uint64_t seed) {
  return TruncatedNormal({fan_in, fan_out}, std::sqrt(2.0 / (fan_in + fan_out)), seed);
}

// He (Kaiming): variance 2 / fan_in, for ReLU networks.
Tensor HeUniform(int fan_in, int fan_out, std::uint64_t seed) {
  return Uniform({fan_in, fan_out}, std::sqrt(6.0 / fan_in), seed);
}
Tensor HeNormal(int fan_in, int fan_out, std::uint64_t seed) {
  return TruncatedNormal({fan_in, fan_out}, std::sqrt(2.0 / fan_in), seed);
}

// Orthogonal: QR-decompose a Gaussian matrix and keep Q, with the sign of each
// column fixed by diag(R) so the distribution is uniform over the orthogonal
// group (Saxe et al. 2013). If rows < cols, build the transpose.
Tensor Orthogonal(int rows, int cols, std::uint64_t seed, double gain = 1.0) {
  const bool    flip = rows < cols;
  const int     big = std::max(rows, cols), small = std::min(rows, cols);
  nn::Generator g(seed);
  const auto    qr = nn::qr(nn::normal({big, small}, 0.0, 1.0, g));  // q [big,small], r [small,small]
  const Tensor  d  = nn::sign(nn::matrix_diag_part(qr.r));          // [small]
  Tensor        q  = nn::mul(qr.q, d);                                // broadcast over rows
  if (flip) q = nn::transpose(q, {1, 0});
  return nn::mul(q, gain);
}

// ---------------------------------------------------------------------------

void Lifecycle() {
  Header("Variable lifecycle");
  Tensor w;  // declared, no value yet: the closest thing to an uninitialized Variable
  std::cout << "  initialized before assignment? " << w.defined() << "\n";
  const std::string message = check::Throws([&] { (void)w.to_vector<float>(); }, "read before assignment");
  std::cout << "  read before assignment -> " << message.substr(0, 60) << "\n";
  w = nn::tensor({{1, 2}, {3, 4}});  // TF Assign(w, Const)
  std::cout << "  initialized after assignment?  " << w.defined() << "\n"
            << "  w = " << w.values_str() << "\n";
  check::Require(w.defined(), "w is initialized");

  // Updates mutate w in place; every alias of w sees them.
  const Tensor alias = w;
  w.add_(nn::full({2, 2}, 1.0));  // AssignAdd
  std::cout << "  after add_(1):        " << w.values_str() << "\n";
  w.copy_(nn::mul(w, 10.0));  // Assign(w, w * 10)
  std::cout << "  after copy_(w*10):    " << w.values_str() << "\n";
  // ScatterUpdate(w, {1}, {{-1, -2}}): replace row 1.
  w.copy_(nn::concat({nn::slice_axis(w, 0, 0, 1), nn::tensor({{-1, -2}})}, 0));
  std::cout << "  after row 1 update:   " << w.values_str() << "\n";
  check::Require(alias.to_vector<float>() == std::vector<float>{20, 30, -1, -2}, "in-place updates (alias sees them)");

  // A ParameterStore groups named variables; clone() is an independent copy
  // (TF: a second Session has its own copy of every variable).
  nn::ParameterStore store(1);
  store.set("w", w);
  nn::ParameterStore other = store.clone();
  {
    nn::NoGradGuard no_grad;  // parameters require grad: update them outside the tape
    other.at("w").add_(nn::full({2, 2}, 100.0));
  }
  std::cout << "  store w = " << store.at("w").values_str() << ", clone after add_(100) = " << other.at("w").values_str()
            << "\n";
  check::Require(store.at("w").to_vector<float>() == std::vector<float>{20, 30, -1, -2}, "clone is independent");
}

void Initializers() {
  Header("Initializers");
  const int fan_in = 256, fan_out = 128;

  struct Entry {
    std::string label;
    Tensor      init;
    double      expected_std;
  };
  const double       glorot_std = std::sqrt(2.0 / (fan_in + fan_out));
  const double       he_std     = std::sqrt(2.0 / fan_in);
  std::vector<Entry> entries    = {
      {"zeros", nn::zeros({fan_in, fan_out}), 0.0},
      {"constant 0.1", nn::full({fan_in, fan_out}, 0.1), 0.0},
      {"uniform [-0.05,0.05)", Uniform({fan_in, fan_out}, 0.05, 1), 0.05 / std::sqrt(3.0)},
      {"normal std 0.02", Normal({fan_in, fan_out}, 0.02, 2), 0.02},
      {"truncated normal std 0.02", TruncatedNormal({fan_in, fan_out}, 0.02, 3), 0.02},
      {"glorot uniform", GlorotUniform(fan_in, fan_out, 4), glorot_std},
      {"glorot normal", GlorotNormal(fan_in, fan_out, 5), glorot_std},
      {"he uniform", HeUniform(fan_in, fan_out, 6), he_std},
      {"he normal", HeNormal(fan_in, fan_out, 7), he_std},
      {"orthogonal", Orthogonal(fan_in, fan_out, 8), 1.0 / std::sqrt(double(fan_in))},
      {"orthogonal (wide)", Orthogonal(16, 64, 9), 1.0 / std::sqrt(64.0)},
  };
  // All variables are initialized by building them; there is no init_all op.
  for (const Entry& e : entries) {
    const Stats s = PrintStats(e.label, e.init, e.expected_std);
    if (e.expected_std == 0) {
      check::Require(s.stddev == 0, e.label + ": constant");
    } else {
      check::Near(s.stddev / e.expected_std, 1.0, 0.03, e.label + ": std / expected");
      check::Near(s.mean, 0.0, 4 * e.expected_std / std::sqrt(double(e.init.numel())) + 1e-7, e.label + ": mean");
    }
  }
  check::Require(ComputeStats(entries[4].init).max <= 2 * 0.02 / 0.87962566103423978 + 1e-6,
                 "truncated normal stays within 2 sigma");

  // Orthogonality: max |Q^T Q - I| for the tall matrix and |Q Q^T - I| for the wide one.
  Header("Orthogonality check");
  for (int which : {9, 10}) {
    const Tensor& q    = entries[static_cast<std::size_t>(which)].init;
    const bool    tall = q.shape()[0] >= q.shape()[1];
    const Tensor  gram = tall ? nn::matmul(nn::transpose(q, {1, 0}), q) : nn::matmul(q, nn::transpose(q, {1, 0}));
    const auto    n    = tall ? q.shape()[1] : q.shape()[0];
    const double  err  = nn::max(nn::abs(nn::sub(gram, nn::eye(n)))).item<float>();
    std::cout << "  " << entries[static_cast<std::size_t>(which)].label << ": max |Q" << (tall ? "^T Q" : " Q^T")
              << " - I| = " << err << "\n";
    check::Require(err < 1e-5, "orthogonal initializer is orthogonal");
  }
}

void Seeds() {
  Header("Seeds and reproducibility");
  // A seeded generator reproduces its stream every time it is created; the
  // process-wide default generator (what unseeded ops draw from) keeps
  // advancing, so each "session" sees different numbers.
  std::vector<std::vector<float>> seeded, unseeded;
  for (int run = 0; run < 2; ++run) {
    nn::Generator g(123);
    seeded.push_back(nn::normal({3}, 0.0, 1.0, g).to_vector<float>());
    unseeded.push_back(nn::normal({3}).to_vector<float>());
    std::printf("  session %d: seeded [%.5f %.5f %.5f]  unseeded [%.5f %.5f %.5f]\n", run, seeded.back()[0],
                seeded.back()[1], seeded.back()[2], unseeded.back()[0], unseeded.back()[1], unseeded.back()[2]);
  }
  check::Require(seeded[0] == seeded[1], "seeded draws repeat");
  check::Require(unseeded[0] != unseeded[1], "unseeded draws differ");
}

void ResourceVariables() {
  Header("Resource variables");
  // TF2-style variables (VarHandleOp + ReadVariableOp/AssignVariableOp) are
  // what every nn parameter already is: a handle to a buffer, read and
  // updated explicitly, with gradients taken with respect to it.
  Tensor v = nn::tensor({1, 2, 3}).requires_grad_();  // AssignVariableOp
  std::cout << "  after assignment:   " << v.values_str() << "\n";
  {
    nn::NoGradGuard no_grad;  // an in-place update is not part of any gradient
    v.add_(nn::tensor({10, 10, 10}));  // AssignAddVariableOp
  }
  std::cout << "  after add_(10):     " << v.values_str() << "\n";
  const Tensor g = nn::grad(nn::sum(nn::square(v)), {v})[0];
  std::cout << "  d sum(v^2) / dv:    " << g.values_str() << "\n";
  check::Require(v.to_vector<float>() == std::vector<float>{11, 12, 13}, "resource variable value");
  check::Require(g.to_vector<float>() == std::vector<float>{22, 24, 26}, "gradient w.r.t. the variable");
}

}  // namespace

int main() {
  try {
    Lifecycle();
    Initializers();
    Seeds();
    ResourceVariables();
    std::cout << "\nAll variable and initializer checks passed.\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
  return 0;
}
