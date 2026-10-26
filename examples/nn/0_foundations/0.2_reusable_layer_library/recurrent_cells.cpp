// CMD: just run-nn recurrent_cells
// Recurrent cells: vanilla RNN, LSTM and GRU.
//
// Port of tflow's recurrent_cells. The cells (nn::rnn_step, nn::lstm_step,
// nn::gru_step in layers.hpp; tflow's common/nn/rnn.h and nn::LstmStep, with
// the same parameter names, packing and initial ranges) map an input and the
// previous hidden state to the next state; a sequence is processed by
// unrolling the step. The program checks one step of each cell against the
// equations computed by hand, counts their parameters, measures how the
// gradient of the final state with respect to the initial state decays with
// the sequence length (the vanishing-gradient problem the gated cells were
// designed to fix), and trains all three on a latch task where the answer is
// decided by the first element of a long noisy sequence.
//
// Inputs are drawn with std::mt19937_64 exactly as in TF; weights come from
// nn's generator, so the gradient norms and accuracies match TF statistically.
// Adam (lr 1e-3) clips by global norm 1 with nn::clip_by_global_norm, which
// scales by 1 / max(norm, 1) where TF's helper uses min(1, 1 / (norm + 1e-6)).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <datamunge/nn/nn.hpp>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "common/checks.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;

namespace {

enum class Cell { kRnn, kLstm, kGru };
const char* CellName(Cell c) { return c == Cell::kRnn ? "RNN" : c == Cell::kLstm ? "LSTM" : "GRU"; }

// A cell plus the initial bias of its memory gate (LSTM forget gate, GRU update gate).
struct Variant {
  Cell        cell;
  double      gate_bias;
  std::string label;
};

// Applies the cell over `steps`, returning the final hidden state (and the
// LSTM's cell state if any).
nn::LstmState Run(nn::ParameterStore& p, const Variant& v, const std::vector<Tensor>& steps, nn::LstmState state) {
  for (const auto& x : steps) {
    if (v.cell == Cell::kRnn)
      state.h = nn::rnn_step(p, "cell", x, state.h);
    else if (v.cell == Cell::kGru)
      state.h = nn::gru_step(p, "cell", x, state.h, v.gate_bias);
    else
      state = nn::lstm_step(p, "cell", x, state, v.gate_bias);
  }
  return state;
}

Tensor Random(const nn::Shape& shape, std::mt19937_64& rng, float std = 1.f) {
  std::vector<float>              v(static_cast<std::size_t>(nn::shape_numel(shape)));
  std::normal_distribution<float> normal(0.f, std);
  for (auto& e : v) e = normal(rng);
  return nn::from_vector(v, shape);
}
double Sigmoid(double x) { return 1 / (1 + std::exp(-x)); }

// ---------------------------------------------------------------------------
// 1. One step of each cell by hand
// ---------------------------------------------------------------------------

void OneStep() {
  std::printf("== One step with in = 3, units = 4, batch 2: cell vs hand computation, and parameter counts\n");
  for (Cell cell : {Cell::kRnn, Cell::kLstm, Cell::kGru}) {
    nn::ParameterStore p(1);
    std::mt19937_64    rng(1);
    const Tensor xv = Random({2, 3}, rng), hv = Random({2, 4}, rng, 0.5f), cv = Random({2, 4}, rng, 0.5f);
    const nn::LstmState next = Run(p, {cell, cell == Cell::kLstm ? 1.0 : 0.0, ""}, {xv}, {hv, cv});
    std::map<std::string, std::vector<float>> v;
    for (const auto& name : p.names()) v[name] = p.at(name).to_vector<float>();
    const auto X = xv.to_vector<float>(), H = hv.to_vector<float>(), C = cv.to_vector<float>();
    const auto h_out = next.h.to_vector<float>();
    // (a W)_{ij} with a [2, k] row-major and W [k, cols] row-major.
    auto mm = [&](const std::vector<float>& a, int k_dim, const std::vector<float>& W, int cols, int i, int j) {
      double acc = 0;
      for (int k = 0; k < k_dim; ++k) acc += a[i * k_dim + k] * W[k * cols + j];
      return acc;
    };
    float err = 0;
    for (int i = 0; i < 2; ++i)
      for (int j = 0; j < 4; ++j) {
        double hn = 0;
        if (cell == Cell::kRnn) {
          hn = std::tanh(mm(X, 3, v["cell/Wx"], 4, i, j) + mm(H, 4, v["cell/Wh"], 4, i, j) + v["cell/b"][j]);
        } else if (cell == Cell::kGru) {
          auto zr = [&](int col) {
            return Sigmoid(mm(X, 3, v["cell/Wx_zr"], 8, i, col) + mm(H, 4, v["cell/Wh_zr"], 8, i, col) + v["cell/b_zr"][col]);
          };
          const double z      = zr(j);
          double       rh_Whn = 0;
          for (int k = 0; k < 4; ++k) rh_Whn += zr(4 + k) * H[i * 4 + k] * v["cell/Wh_n"][k * 4 + j];
          const double n = std::tanh(mm(X, 3, v["cell/Wx_n"], 4, i, j) + rh_Whn + v["cell/b_n"][j]);
          hn             = (1 - z) * n + z * H[i * 4 + j];
        } else {
          auto gate = [&](int g) {
            return mm(X, 3, v["cell/Wx"], 16, i, g * 4 + j) + mm(H, 4, v["cell/Wh"], 16, i, g * 4 + j) + v["cell/b"][g * 4 + j];
          };
          const double ig = Sigmoid(gate(0)), fg = Sigmoid(gate(1)), gg = std::tanh(gate(2)), og = Sigmoid(gate(3));
          const double cn = fg * C[i * 4 + j] + ig * gg;
          hn              = og * std::tanh(cn);
          err             = std::max<float>(err, std::abs(cn - next.c.to_vector<float>()[i * 4 + j]));
        }
        err = std::max<float>(err, std::abs(hn - h_out[i * 4 + j]));
      }
    const int in = 3, u = 4;
    const int expect = cell == Cell::kRnn   ? in * u + u * u + u
                       : cell == Cell::kGru ? 3 * (in * u + u * u + u)
                                            : 4 * (in * u + u * u + u);
    std::printf("  %-5s max |cell - hand| = %.1e, parameters %lld (= %s)\n", CellName(cell), err,
                static_cast<long long>(p.count()),
                cell == Cell::kRnn ? "in*u + u*u + u" : cell == Cell::kGru ? "3 (in*u + u*u + u)" : "4 (in*u + u*u + u)");
    check::Require(err < 1e-5f, std::string(CellName(cell)) + " step matches the hand computation");
    check::Require(p.count() == expect, std::string(CellName(cell)) + " parameter count");
  }
}

// ---------------------------------------------------------------------------
// 2. Gradient of the final state with respect to the initial state
// ---------------------------------------------------------------------------

void GradientThroughTime() {
  std::printf(
      "\n== Gradient of L = sum(h_T . r) w.r.t. the initial state, random inputs, units 32, mean over 8 "
      "initialisations\n");
  std::printf("  %-28s", "T:");
  const std::vector<int> lengths = {1, 5, 10, 20, 40, 80};
  for (int T : lengths) std::printf(" %9d", T);
  std::printf("\n");
  const std::vector<Variant> variants = {{Cell::kRnn, 0, "RNN          ||dL/dh_0||"},
                                         {Cell::kLstm, 1, "LSTM b_f=1   ||dL/dh_0||"},
                                         {Cell::kLstm, 1, "LSTM b_f=1   ||dL/dc_0||"},
                                         {Cell::kLstm, 3, "LSTM b_f=3   ||dL/dc_0||"},
                                         {Cell::kLstm, 5, "LSTM b_f=5   ||dL/dc_0||"},
                                         {Cell::kGru, 0, "GRU  b_z=0   ||dL/dh_0||"},
                                         {Cell::kGru, 3, "GRU  b_z=3   ||dL/dh_0||"}};
  std::map<std::string, std::vector<double>> table;
  for (const Variant& v : variants) {
    const bool wrt_c = v.label.find("dc_0") != std::string::npos;
    std::printf("  %-28s", v.label.c_str());
    for (int T : lengths) {
      double mean_norm = 0;
      for (std::uint64_t seed = 1; seed <= 8; ++seed) {
        nn::ParameterStore p(seed);
        std::mt19937_64    rng(seed);
        const Tensor       x  = Random({1, T, 4}, rng);
        Tensor             h0 = Random({1, 32}, rng, 0.1f).requires_grad_();
        Tensor             c0 = Random({1, 32}, rng, 0.1f).requires_grad_();
        const Tensor       r  = Random({1, 32}, rng);
        const auto         final_state = Run(p, v, nn::time_steps(x), {h0, c0});
        const Tensor       loss        = nn::sum(nn::mul(final_state.h, r));
        const Tensor       g           = nn::grad(loss, {wrt_c ? c0 : h0})[0];
        mean_norm += std::sqrt(nn::sum(nn::square(g)).item<float>()) / 8;
      }
      table[v.label].push_back(mean_norm);
      std::printf(" %9.2e", mean_norm);
    }
    std::printf("\n");
  }
  std::printf(
      "  (sigmoid(1) = 0.73, sigmoid(3) = 0.95, sigmoid(5) = 0.99: the gate bias sets the retention per step;\n"
      "   the tanh RNN has no such knob, its decay follows the recurrent weights)\n");
  // The qualitative result the TF run shows: the plain RNN's gradient vanishes,
  // and a larger memory-gate bias keeps the long-range gradient alive.
  const auto& rnn = table["RNN          ||dL/dh_0||"];
  check::Require(rnn.back() < 1e-3 * rnn.front(), "RNN gradient vanishes over 80 steps");
  check::Require(table["LSTM b_f=5   ||dL/dc_0||"].back() > 10 * table["LSTM b_f=1   ||dL/dc_0||"].back(),
                 "larger forget bias keeps the cell-state gradient");
  check::Require(table["GRU  b_z=3   ||dL/dh_0||"].back() > 10 * table["GRU  b_z=0   ||dL/dh_0||"].back(),
                 "larger update bias keeps the GRU gradient");
}

// ---------------------------------------------------------------------------
// 3. The latch task
// ---------------------------------------------------------------------------

void LatchTask() {
  const int T = 40, units = 32, batch = 64, steps = 1000;
  std::printf(
      "\n== Latch task: x_1 = +/-1 decides the label, x_2..x_%d are N(0,1) noise; units %d, %d Adam steps, batch %d\n",
      T, units, steps, batch);
  std::printf("  %-11s", "step:");
  for (int s = 100; s <= steps; s += 100) std::printf(" %7d", s);
  std::printf("   (accuracy on 512 held-out sequences)\n");
  const std::vector<Variant> variants = {{Cell::kRnn, 0, "RNN"},
                                         {Cell::kLstm, 1, "LSTM b_f=1"},
                                         {Cell::kLstm, 3, "LSTM b_f=3"},
                                         {Cell::kGru, 0, "GRU b_z=0"},
                                         {Cell::kGru, 3, "GRU b_z=3"}};
  std::map<std::string, double> final_accuracy;
  for (const Variant& v : variants) {
    nn::ParameterStore p(7);
    auto               model = [&](const Tensor& x) {
      const std::int64_t n     = x.shape()[0];
      const Tensor       h0    = nn::zeros({n, units});
      const auto         final_state = Run(p, v, nn::time_steps(x), {h0, h0});
      return nn::dense(p, "out", final_state.h, 1);
    };
    std::mt19937_64 rng(11);
    auto            make = [&](int n, Tensor* xv, Tensor* yv) {
      std::vector<float>              xs(static_cast<std::size_t>(n * T)), ys(static_cast<std::size_t>(n));
      std::normal_distribution<float> normal(0.f, 1.f);
      for (auto& e : xs) e = normal(rng);
      for (int i = 0; i < n; ++i) {
        const float bit = rng() % 2 ? 1.f : -1.f;
        xs[static_cast<std::size_t>(i * T)] = bit;
        ys[static_cast<std::size_t>(i)]     = bit > 0 ? 1.f : 0.f;
      }
      *xv = nn::from_vector(xs, {n, T, 1});
      *yv = nn::from_vector(ys, {n, 1});
    };
    Tensor xtest, ytest;
    make(512, &xtest, &ytest);
    {
      nn::NoGradGuard guard;
      (void)model(nn::zeros({1, T, 1}));  // creates the parameters
    }
    nn::OptimizerConfig config;
    config.kind = nn::OptimizerKind::Adam;
    nn::Optimizer adam(p, 1e-3, config);
    std::printf("  %-11s", v.label.c_str());
    double accuracy = 0;
    for (int step = 1; step <= steps; ++step) {
      Tensor xv, yv;
      make(batch, &xv, &yv);
      // Sigmoid cross-entropy in its numerically stable fused form.
      const Tensor loss = nn::sigmoid_cross_entropy(model(xv), yv);
      adam.step(nn::clip_by_global_norm(nn::grad(loss, p.parameters()), 1.0).first);
      if (step % 100 == 0) {
        nn::NoGradGuard guard;
        const Tensor    predicted = nn::cast(nn::greater(model(xtest), nn::scalar(0)), nn::DType::Float32);
        accuracy                  = nn::mean(nn::cast(nn::equal(predicted, ytest), nn::DType::Float32)).item<float>();
        std::printf(" %6.1f%%", 100 * accuracy);
        std::fflush(stdout);
      }
    }
    final_accuracy[v.label] = accuracy;
    std::printf("\n");
  }
  // TF's run: the gated cells with a memory bias solve the task; the plain
  // RNN cannot carry x_1 across 40 noisy steps.
  check::Require(final_accuracy["LSTM b_f=3"] > 0.95, "LSTM b_f=3 solves the latch task");
  check::Require(final_accuracy["GRU b_z=3"] > 0.95, "GRU b_z=3 solves the latch task");
}

}  // namespace

int main() {
  try {
    OneStep();
    GradientThroughTime();
    LatchTask();
    std::printf("\nAll recurrent cell checks passed.\n");
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
  return 0;
}
