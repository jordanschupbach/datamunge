// CMD: just run-nn attention
// Attention: additive, scaled dot-product and multi-head.
//
// Port of tflow's attention. The attention layers in datamunge/nn/attention.hpp
// compute a weighted average of value vectors, with weights given by a
// softmax over scores between a query and the keys. This program checks
// scaled dot-product attention against hand-written loops, shows why the
// scores are divided by sqrt(d) (without it the softmax saturates as d grows),
// applies a causal mask, checks additive (Bahdanau) attention, verifies the
// multi-head layer's head splitting against a direct computation and its
// parameter count, compares the symbolic gradient through the softmax with the
// closed form, and trains a tiny attention model on a key-value lookup task to
// show the weights becoming a one-hot pointer.
//
// TF -> datamunge: nn::ScaledDotProductAttention(q, k, v, d, mask) is
// nn::scaled_dot_product_attention(q, k, v, mask) (plain path, weights
// returned), nn::CausalMask is nn::causal_mask, nn::AdditiveAttention and
// nn::MultiHeadAttention are nn::additive_attention and
// nn::multi_head_attention (same parameter names and initialiser ranges).
// Inputs and lookup batches come from the same std::mt19937_64 streams as TF,
// so section 1 and the scaling table print the same numbers; the layers'
// parameters are drawn by datamunge's generator, so the head weights and the
// lookup trajectory match TF statistically. A datamunge extra in section 1
// checks the fused (memory-efficient) kernel against the plain path.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <datamunge/nn/attention.hpp>
#include <datamunge/nn/nn.hpp>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "common/checks.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;

namespace {

Tensor Random(const nn::Shape& shape, std::mt19937_64& rng, float std = 1.f) {
  Tensor                          t = nn::empty(shape);
  std::normal_distribution<float> normal(0.f, std);
  float*                          p = t.data_ptr<float>();
  for (std::int64_t i = 0; i < t.numel(); ++i) p[i] = normal(rng);
  return t;
}

using Rows = std::vector<std::vector<double>>;

// Reference: softmax(q k^T / sqrt(d) + mask) v for one example; q [Tq, d], k [Tk, d], v [Tk, dv].
void AttendRef(const Rows& q, const Rows& k, const Rows& v, bool causal, Rows* weights, Rows* out) {
  const int Tq = static_cast<int>(q.size()), Tk = static_cast<int>(k.size()), d = static_cast<int>(q[0].size()),
            dv = static_cast<int>(v[0].size());
  weights->assign(Tq, std::vector<double>(Tk)), out->assign(Tq, std::vector<double>(dv, 0));
  for (int i = 0; i < Tq; ++i) {
    std::vector<double> sc(Tk);
    double              mx = -1e300;
    for (int j = 0; j < Tk; ++j) {
      sc[j] = 0;
      for (int a = 0; a < d; ++a) sc[j] += q[i][a] * k[j][a];
      sc[j] /= std::sqrt(d);
      if (causal && j > i) sc[j] = -1e9;
      mx = std::max(mx, sc[j]);
    }
    double z = 0;
    for (int j = 0; j < Tk; ++j) z += std::exp(sc[j] - mx);
    for (int j = 0; j < Tk; ++j) {
      (*weights)[i][j] = std::exp(sc[j] - mx) / z;
      for (int a = 0; a < dv; ++a) (*out)[i][a] += (*weights)[i][j] * v[j][a];
    }
  }
}

Rows ToRows(const Tensor& t, int n) {  // [N, T, d] or [T, d] -> rows of example n
  const auto T = t.size(-2), d = t.size(-1);
  const auto v = t.to_vector<double>();
  Rows       r(T, std::vector<double>(d));
  for (int i = 0; i < T; ++i)
    for (int j = 0; j < d; ++j) r[i][j] = v[(n * T + i) * d + j];
  return r;
}

// ---------------------------------------------------------------------------
// 1. Scaled dot-product attention by hand, with and without a causal mask
// ---------------------------------------------------------------------------

void DotProduct() {
  std::printf("== Scaled dot-product attention, q/k [1, 4, 8], v [1, 4, 3], vs hand-written loops\n");
  std::mt19937_64 rng(1);
  const Tensor    qv = Random({1, 4, 8}, rng), kv = Random({1, 4, 8}, rng), vv = Random({1, 4, 3}, rng);
  const nn::Attended plain  = nn::scaled_dot_product_attention(qv, kv, vv, Tensor());
  const nn::Attended causal = nn::scaled_dot_product_attention(qv, kv, vv, nn::causal_mask(4));
  for (int c = 0; c < 2; ++c) {
    const nn::Attended& a = c ? causal : plain;
    Rows                w, o;
    AttendRef(ToRows(qv, 0), ToRows(kv, 0), ToRows(vv, 0), c == 1, &w, &o);
    float dw = 0, dout = 0;
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) dw = std::max<float>(dw, std::abs(w[i][j] - a.weights.at({0, i, j})));
      for (int b = 0; b < 3; ++b) dout = std::max<float>(dout, std::abs(o[i][b] - a.output.at({0, i, b})));
    }
    std::printf("  %s: max |weights - hand| = %.1e, max |output - hand| = %.1e; weights:\n", c ? "causal" : "plain ", dw,
                dout);
    for (int i = 0; i < 4; ++i) {
      std::printf("    query %d:", i);
      double sum = 0;
      for (int j = 0; j < 4; ++j) std::printf(" %6.3f", a.weights.at({0, i, j})), sum += a.weights.at({0, i, j});
      std::printf("   (sum %.3f)\n", sum);
      check::Near(sum, 1.0, 1e-5, "attention weights sum to one");
    }
    check::Require(dw < 1e-6 && dout < 1e-6, "scaled dot-product attention differs from the hand computation");
  }
  // datamunge extra: the fused kernel (online softmax over key tiles, no [Tq, Tk] matrix).
  nn::AttentionOptions fused;
  fused.impl   = nn::AttentionImpl::Fused;
  fused.causal = true;
  const float dfused = nn::max(nn::abs(nn::scaled_dot_product_attention(qv, kv, vv, fused).output - causal.output))
                           .item<float>();
  std::printf("  fused causal kernel: max |output - plain| = %.1e\n", dfused);
  check::Require(dfused < 1e-6, "fused attention differs from the plain path");
}

// ---------------------------------------------------------------------------
// 2. Why divide by sqrt(d): the variance of q . k and the softmax entropy
// ---------------------------------------------------------------------------

void Scaling() {
  std::printf("\n== q, k ~ N(0, 1) entries, 64 keys: variance of the scores and entropy of the softmax (max = log 64 = %.2f nats)\n",
              std::log(64.0));
  std::printf("  %6s %14s %14s %16s %16s\n", "d", "var(q.k)", "var(q.k/sqrt d)", "entropy unscaled", "entropy scaled");
  std::mt19937_64                  rng(2);
  std::normal_distribution<double> normal(0, 1);
  for (int d : {4, 16, 64, 256, 1024}) {
    double    var_raw = 0, var_scaled = 0, ent_raw = 0, ent_scaled = 0;
    const int trials = 200;
    for (int t = 0; t < trials; ++t) {
      std::vector<double> q(d);
      for (double& x : q) x = normal(rng);
      std::vector<double> sc(64);
      for (int j = 0; j < 64; ++j) {
        sc[j] = 0;
        for (int a = 0; a < d; ++a) sc[j] += q[a] * normal(rng);
        var_raw += sc[j] * sc[j] / (64.0 * trials), var_scaled += sc[j] * sc[j] / d / (64.0 * trials);
      }
      auto entropy = [&](double scale) {
        double mx = -1e300, z = 0, h = 0;
        for (double s : sc) mx = std::max(mx, s * scale);
        for (double s : sc) z += std::exp(s * scale - mx);
        for (double s : sc) {
          const double p = std::exp(s * scale - mx) / z;
          if (p > 0) h -= p * std::log(p);
        }
        return h;
      };
      ent_raw += entropy(1) / trials, ent_scaled += entropy(1 / std::sqrt(d)) / trials;
    }
    std::printf("  %6d %14.1f %14.2f %16.2f %16.2f\n", d, var_raw, var_scaled, ent_raw, ent_scaled);
    check::Near(var_scaled, 1.0, 0.1, "scaled score variance");
  }
  std::printf("  (unscaled, the scores grow like sqrt(d) and the softmax collapses onto one key, killing its gradient)\n");
}

// ---------------------------------------------------------------------------
// 3. Additive attention by hand
// ---------------------------------------------------------------------------

void Additive() {
  std::printf("\n== Additive attention: q [2, 5], k [2, 6, 3], v [2, 6, 2], units 4, vs hand computation\n");
  nn::ParameterStore p(3);
  std::mt19937_64    rng(3);
  const Tensor       qv = Random({2, 5}, rng), kv = Random({2, 6, 3}, rng), vv = Random({2, 6, 2}, rng);
  const nn::Attended a  = nn::additive_attention(p, "add", qv, kv, vv, 4);
  const Tensor       Wq = p.at("add/Wq"), Wk = p.at("add/Wk"), vec = p.at("add/v");
  float              dw = 0, dout = 0;
  for (int n = 0; n < 2; ++n) {
    std::vector<double> sc(6);
    double              mx = -1e300;
    for (int t = 0; t < 6; ++t) {
      sc[t] = 0;
      for (int u = 0; u < 4; ++u) {
        double h = 0;
        for (int i = 0; i < 5; ++i) h += qv.at({n, i}) * Wq.at({i, u});
        for (int i = 0; i < 3; ++i) h += kv.at({n, t, i}) * Wk.at({i, u});
        sc[t] += std::tanh(h) * vec.at({u, 0});
      }
      mx = std::max(mx, sc[t]);
    }
    double z = 0;
    for (double s : sc) z += std::exp(s - mx);
    std::vector<double> ctx(2, 0);
    for (int t = 0; t < 6; ++t) {
      const double w = std::exp(sc[t] - mx) / z;
      dw             = std::max<float>(dw, std::abs(w - a.weights.at({n, t})));
      for (int i = 0; i < 2; ++i) ctx[i] += w * vv.at({n, t, i});
    }
    for (int i = 0; i < 2; ++i) dout = std::max<float>(dout, std::abs(ctx[i] - a.output.at({n, i})));
  }
  std::printf("  weights shape %s, context shape %s; max |weights - hand| = %.1e, max |context - hand| = %.1e; parameters %lld (= 5*4 + 3*4 + 4)\n",
              nn::shape_str(a.weights.shape()).c_str(), nn::shape_str(a.output.shape()).c_str(), dw, dout,
              static_cast<long long>(p.count()));
  check::Require(dw < 1e-6 && dout < 1e-6, "additive attention differs from the hand computation");
  check::Require(p.count() == 36, "additive attention parameter count");
}

// ---------------------------------------------------------------------------
// 4. Multi-head attention: head splitting, parameters, gradient through the softmax
// ---------------------------------------------------------------------------

void MultiHead() {
  std::printf("\n== Multi-head self-attention, x [2, 5, 16], 4 heads of 4\n");
  nn::ParameterStore   p(4);
  std::mt19937_64      rng(4);
  const Tensor         xv = Random({2, 5, 16}, rng);
  nn::MultiHeadOptions options;
  options.need_weights = true;
  const nn::Attended a = nn::multi_head_attention(p, "self", xv, xv, 4, options);
  std::printf("  output shape %s, weights shape %s, parameters %lld (= 4 * 16 * 16)\n",
              nn::shape_str(a.output.shape()).c_str(), nn::shape_str(a.weights.shape()).c_str(),
              static_cast<long long>(p.count()));
  // Direct computation: per head h, q_h = x Wq[:, 4h:4h+4], etc.; concat heads; times Wo.
  const Tensor Wq = p.at("self/Wq"), Wk = p.at("self/Wk"), Wv = p.at("self/Wv"), Wo = p.at("self/Wo");
  float        dmax = 0;
  for (int n = 0; n < 2; ++n) {
    Rows concat(5, std::vector<double>(16, 0));
    for (int h = 0; h < 4; ++h) {
      Rows q(5, std::vector<double>(4, 0)), k = q, v = q, w, o;
      for (int t = 0; t < 5; ++t)
        for (int j = 0; j < 4; ++j)
          for (int i = 0; i < 16; ++i) {
            q[t][j] += xv.at({n, t, i}) * Wq.at({i, 4 * h + j});
            k[t][j] += xv.at({n, t, i}) * Wk.at({i, 4 * h + j});
            v[t][j] += xv.at({n, t, i}) * Wv.at({i, 4 * h + j});
          }
      AttendRef(q, k, v, false, &w, &o);
      for (int t = 0; t < 5; ++t)
        for (int j = 0; j < 4; ++j) concat[t][4 * h + j] = o[t][j];
    }
    for (int t = 0; t < 5; ++t)
      for (int j = 0; j < 16; ++j) {
        double acc = 0;
        for (int i = 0; i < 16; ++i) acc += concat[t][i] * Wo.at({i, j});
        dmax = std::max<float>(dmax, std::abs(acc - a.output.at({n, t, j})));
      }
  }
  std::printf("  max |layer - per-head hand computation| = %.1e\n", dmax);
  check::Require(dmax < 1e-5, "multi-head attention differs from the per-head computation");
  check::Require(p.count() == 1024, "multi-head attention parameter count");
  std::printf("  head weights for example 0, query 2 (each head is its own distribution over the 5 keys):\n");
  for (int h = 0; h < 4; ++h) {
    std::printf("    head %d:", h);
    for (int j = 0; j < 5; ++j) std::printf(" %6.3f", a.weights.at({0, h, 2, j}));
    std::printf("\n");
  }
  // Gradient through the softmax: L = sum(weights * r) for one query; dL/dscore = alpha * (r - alpha . r).
  const Tensor sc    = Random({5}, rng).requires_grad_(), rv = Random({5}, rng);
  const Tensor alpha = nn::softmax(sc);
  const Tensor g     = nn::grad(nn::sum(alpha * rv), {sc})[0];
  double       dot   = 0;
  for (int j = 0; j < 5; ++j) dot += alpha.at({j}) * rv.at({j});
  float derr = 0;
  for (int j = 0; j < 5; ++j) derr = std::max<float>(derr, std::abs(alpha.at({j}) * (rv.at({j}) - dot) - g.at({j})));
  std::printf("  softmax gradient check: max |dL/dscore - alpha (r - alpha.r)| = %.1e\n", derr);
  check::Require(derr < 1e-6, "softmax gradient");
}

// ---------------------------------------------------------------------------
// 5. Learning to look up: attention as a soft dictionary
// ---------------------------------------------------------------------------

void Lookup(int steps) {
  std::printf("\n== Lookup task: 6 (key id, value) pairs and a query id; predict the value of the matching key\n");
  const int          T = 6, ids = 10, dim = 8, batch = 64;
  nn::ParameterStore p(5);
  const Tensor       table = p.get("emb", {ids, dim}, nn::init::uniform(0.5));
  nn::Optimizer      adam(p, 0.02, {nn::OptimizerKind::Adam});
  struct Batch {
    Tensor keys, query, values, target;
  };
  std::mt19937_64                 rng(5);
  std::normal_distribution<float> normal(0.f, 1.f);
  auto                            make = [&](int n) {
    std::vector<std::int64_t> kv(n * T), qv(n);
    std::vector<float>        vv(n * T), tv(n);
    for (int i = 0; i < n; ++i) {
      std::vector<int> perm(ids);
      for (int j = 0; j < ids; ++j) perm[j] = j;
      std::shuffle(perm.begin(), perm.end(), rng);  // distinct keys
      const int which = static_cast<int>(rng() % T);
      for (int t = 0; t < T; ++t) kv[i * T + t] = perm[t], vv[i * T + t] = normal(rng);
      qv[i] = perm[which], tv[i] = vv[i * T + which];
    }
    return Batch{nn::from_vector(kv, {n, T}), nn::from_vector(qv, {n}), nn::from_vector(vv, {n, T}),
                 nn::from_vector(tv, {n})};
  };
  auto forward = [&](const Batch& b) {
    const Tensor k = nn::gather(table, b.keys, 0);                     // [B, T, dim]
    const Tensor q = nn::expand_dims(nn::gather(table, b.query, 0), 1);  // [B, 1, dim]
    const nn::Attended a    = nn::scaled_dot_product_attention(q, k, nn::expand_dims(b.values, -1), Tensor());  // [B, 1, 1]
    const Tensor       pred = nn::reshape(a.output, {-1});
    return std::make_pair(nn::mean(nn::square(pred - b.target)), a.weights);
  };
  const Batch test = make(512);
  double      final_on_target = 0, final_mse = 0;
  for (int step = 1; step <= steps; ++step) {
    const Batch b = make(batch);
    adam.step(nn::grad(forward(b).first, {table}));
    if (step == 1 || step % 100 == 0 || step == steps) {
      nn::NoGradGuard guard;
      const auto      r = forward(test);
      // Mean weight placed on the matching key.
      double on_target = 0;
      for (int i = 0; i < 512; ++i)
        for (int t = 0; t < T; ++t)
          if (test.keys.at({i, t}) == test.query.at({i})) on_target += r.second.at({i, 0, t}) / 512;
      std::printf("  step %3d  test MSE %.4f  mean attention weight on the matching key %.3f (1/6 = %.3f at random)\n",
                  step, r.first.item(), on_target, 1.0 / T);
      final_on_target = on_target, final_mse = r.first.item();
    }
  }
  // TF reaches MSE 0.0002 and 0.989 on the matching key.
  check::Require(std::isfinite(final_mse) && final_mse >= 0 && final_on_target > 0 && final_on_target <= 1,
                 "finite lookup loss and valid attention weights");
  check::Require(adam.iterations() == steps, "one Adam update per lookup batch");
  if (steps >= 600) {
    check::Require(final_mse < 0.005, "lookup test MSE did not converge");
    check::Require(final_on_target > 0.95, "attention did not learn to point at the matching key");
  }
  nn::NoGradGuard guard;
  const auto evaluated = forward(test);
  nn::ConfusionMatrix pointer(T);
  for (int i = 0; i < 512; ++i) {
    int truth = 0, prediction = 0;
    for (int t = 0; t < T; ++t) {
      if (test.keys.at({i, t}) == test.query.at({i})) truth = t;
      if (evaluated.second.at({i, 0, t}) > evaluated.second.at({i, 0, prediction})) prediction = t;
    }
    pointer.add(truth, prediction);
  }
  std::cout << "  pointer accuracy=" << pointer.accuracy() << '\n'
            << "  Confusion matrix (rows=matching key position, columns=predicted position; held-out 512 queries):\n";
  for (const auto& row : pointer.cells()) {
    std::cout << "    ";
    for (auto count : row) std::cout << count << ' ';
    std::cout << '\n';
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int steps = 600;
    for (int i = 1; i < argc; ++i) {
      const std::string argument(argv[i]);
      if (argument.rfind("epochs=", 0) == 0) steps = std::stoi(argument.substr(7));
      if (argument.rfind("steps=", 0) == 0) steps = std::stoi(argument.substr(6));
    }
    check::Require(steps > 0, "positive training budget required");
    DotProduct();
    Scaling();
    Additive();
    MultiHead();
    Lookup(steps);
  } catch (const std::exception& e) {
    std::cerr << "FAILED: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
