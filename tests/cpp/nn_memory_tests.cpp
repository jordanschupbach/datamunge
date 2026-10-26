// Autograd memory tests for datamunge::nn: saved tensors are freed as soon
// as their backward has run, peak memory during backward stays near the
// theoretical minimum (deep MLP, small ResNet, RNN unrolled over 1000 steps),
// graphs leave nothing behind, and gradient checkpointing / reversible
// sequences give exact gradients with less memory.

#include <gtest/gtest.h>

#include <cmath>
#include <datamunge/nn/memory.hpp>
#include <datamunge/nn/nn.hpp>
#include <functional>
#include <string>
#include <vector>

namespace nn = datamunge::nn;
using nn::DType;
using nn::Tensor;

namespace {

using Inputs = std::vector<Tensor>;

std::int64_t bytes_of(const std::vector<Tensor>& ts) {
  std::int64_t n = 0;
  for (const auto& t : ts)
    if (t.defined()) n += t.numel() * static_cast<std::int64_t>(nn::dtype_size(t.dtype()));
  return n;
}

double max_abs_diff(const Tensor& a, const Tensor& b) {
  const auto x = a.to_vector<double>(), y = b.to_vector<double>();
  EXPECT_EQ(x.size(), y.size());
  double worst = 0;
  for (std::size_t i = 0; i < x.size() && i < y.size(); ++i) worst = std::max(worst, std::abs(x[i] - y[i]));
  return worst;
}

void expect_close(const std::vector<Tensor>& a, const std::vector<Tensor>& b, double tol) {
  ASSERT_EQ(a.size(), b.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    ASSERT_EQ(a[i].defined(), b[i].defined()) << "gradient " << i;
    if (a[i].defined()) EXPECT_LE(max_abs_diff(a[i], b[i]), tol) << "gradient " << i;
  }
}

// ---- models used by the memory tests -------------------------------------------------

/// `layers` square weights [width, width] for a ReLU MLP.
std::vector<Tensor> mlp_weights(int layers, std::int64_t width, DType dtype = DType::Float32, std::uint64_t seed = 1) {
  nn::Generator       gen(seed);
  std::vector<Tensor> w;
  const double        limit = std::sqrt(6.0 / static_cast<double>(width));
  for (int i = 0; i < layers; ++i) w.push_back(nn::uniform({width, width}, -limit, limit, gen, dtype).requires_grad_());
  return w;
}

Tensor mlp_layer(const Tensor& h, const Tensor& w) { return nn::relu(nn::matmul(h, w)); }

/// A compact post-activation ResNet (conv/BN/ReLU stem, `blocks` basic
/// residual blocks of width `width`, global average pool, dense head).
Tensor small_resnet(nn::ParameterStore& p, nn::ParameterStore& stats, const Tensor& image, int blocks,
                    std::int64_t width) {
  nn::ConvLayerOptions conv;
  conv.bias = false;
  auto cbr  = [&](const std::string& name, const Tensor& x, bool relu) {
    Tensor y = nn::batch_norm(p, stats, name + "/bn", nn::conv2d(p, name, x, width, 3, conv), true);
    return relu ? nn::relu(y) : y;
  };
  Tensor h = cbr("stem", image, true);
  for (int b = 0; b < blocks; ++b) {
    const std::string name = "block" + std::to_string(b);
    Tensor            r    = cbr(name + "/conv2", cbr(name + "/conv1", h, true), false);
    h                      = nn::relu(r + h);
  }
  return nn::dense(p, "head", nn::mean(h, {1, 2}), 10);
}

// ---- the counters -------------------------------------------------------------------

TEST(NnMemory, CountsLiveAndPeakStorageBytes) {
  const std::int64_t base  = nn::live_memory_bytes();
  const std::int64_t count = nn::live_storage_count();
  nn::PeakMemoryScope scope;
  {
    Tensor a = nn::zeros({1000}, DType::Float32);
    EXPECT_EQ(nn::live_memory_bytes() - base, 4000);
    EXPECT_EQ(nn::live_storage_count() - count, 1);
    {
      Tensor b = nn::zeros({500}, DType::Float64);
      Tensor v = b.reshape({50, 10});  // a view shares the buffer
      EXPECT_EQ(nn::live_memory_bytes() - base, 8000);
    }
    EXPECT_EQ(nn::live_memory_bytes() - base, 4000);
  }
  EXPECT_EQ(nn::live_memory_bytes(), base);
  EXPECT_EQ(scope.peak_above_baseline(), 8000);
  EXPECT_EQ(scope.live_above_baseline(), 0);
  nn::reset_peak_memory();
  EXPECT_EQ(nn::peak_memory_bytes(), nn::live_memory_bytes());
}

// ---- saved tensors are freed when their backward has run ---------------------------

// A 100-layer ReLU MLP. Each layer keeps one activation: its matmul input is
// the previous ReLU output, which the ReLU also keeps for its backward. During
// backward every activation is freed as soon as its layer's backward ran,
// so the live bytes never exceed what was saved plus the weight gradients
// produced so far plus a few activation-sized temporaries.
TEST(NnMemory, DeepMlpFreesSavedTensorsDuringBackward) {
  const int          L = 100;
  const std::int64_t B = 32, W = 128;
  const auto         w      = mlp_weights(L, W);
  const Tensor       x      = nn::normal({B, W});
  const std::int64_t act    = B * W * 4;
  const std::int64_t wbytes = W * W * 4;
  const std::int64_t base   = nn::live_memory_bytes();

  Tensor h = x;
  for (int i = 0; i < L; ++i) h = mlp_layer(h, w[i]);
  Tensor loss = nn::mean(nn::square(h));
  h           = Tensor();
  const std::int64_t saved = nn::live_memory_bytes() - base;
  EXPECT_LE(saved, L * act + 4 * act) << "forward keeps about one activation per layer";
  EXPECT_GE(saved, L * act) << "forward keeps at least one activation per layer";

  nn::PeakMemoryScope scope;
  auto                g = nn::grad(loss, w);
  // Peak: the saved activations, all weight gradients at the end, a few temporaries.
  EXPECT_LE(scope.peak_above_baseline(), L * wbytes + 4 * act + 2 * wbytes);
  // The graph is still alive (loss holds it), but everything it saved is gone.
  EXPECT_LE(nn::live_memory_bytes() - base - bytes_of(g), 64) << "saved tensors outlived their backward";
  loss = Tensor();
  g.clear();
  EXPECT_EQ(nn::live_memory_bytes(), base);
}

TEST(NnMemory, DroppingAnUnusedGraphFreesEverything) {
  const auto         w    = mlp_weights(50, 64);
  const std::int64_t base = nn::live_memory_bytes();
  {
    Tensor h = nn::normal({8, 64});
    for (const auto& wi : w) h = mlp_layer(h, wi);
    EXPECT_GT(nn::live_memory_bytes(), base);
  }
  EXPECT_EQ(nn::live_memory_bytes(), base) << "a reference cycle keeps an unused graph alive";
}

// An Elman RNN unrolled over 1000 steps keeps one hidden state per step
// (tanh saves its output, which is also the next matmul's saved input).
TEST(NnMemory, UnrolledRnnBackwardStaysFlat) {
  const int          T = 1000;
  const std::int64_t B = 8, H = 64, D = 16;
  nn::ParameterStore p(3);
  const Tensor       xs   = nn::normal({T, B, D});
  const auto         step = [&](int t, const Tensor& h) { return nn::rnn_step(p, "rnn", nn::slice_axis(xs, 0, t, t + 1).reshape({B, D}), h); };
  Tensor             h    = step(0, nn::zeros({B, H}));  // creates the parameters
  h                       = nn::zeros({B, H});
  const std::int64_t base = nn::live_memory_bytes();
  const std::int64_t unit = B * H * 4;
  for (int t = 0; t < T; ++t) h = step(t, h);
  Tensor loss = nn::mean(nn::square(h));
  h           = Tensor();
  const std::int64_t saved = nn::live_memory_bytes() - base;
  EXPECT_LE(saved, T * (unit + B * D * 4) + 8 * unit) << "about one hidden state (plus the input slice) per step";

  nn::PeakMemoryScope scope;
  auto                g = nn::grad(loss, p.parameters());
  EXPECT_LE(scope.peak_above_baseline(), 2 * bytes_of(g) + 16 * unit) << "backward must free each step's state";
  EXPECT_LE(nn::live_memory_bytes() - base - bytes_of(g), 64);
  loss = Tensor();
  g.clear();
  EXPECT_LE(nn::live_memory_bytes(), base);  // the initial zero state was dropped too
}

TEST(NnMemory, ResNetBackwardFreesActivations) {
  nn::ParameterStore p(5), stats(6);
  const Tensor       image = nn::normal({4, 16, 16, 3});
  { small_resnet(p, stats, image, 4, 16); }  // create parameters
  const std::int64_t base  = nn::live_memory_bytes();
  Tensor             loss  = nn::mean(nn::square(small_resnet(p, stats, image, 4, 16)));
  const std::int64_t saved = nn::live_memory_bytes() - base;
  const std::int64_t act   = 4 * 16 * 16 * 16 * 4;
  EXPECT_GE(saved, 9 * act);  // at least one map per conv layer

  nn::PeakMemoryScope scope;
  auto                g = nn::grad(loss, p.parameters());
  EXPECT_LE(scope.peak_above_baseline(), bytes_of(g) + 6 * act) << "peak far above the saved activations";
  EXPECT_LE(nn::live_memory_bytes() - base - bytes_of(g), 64);
  loss = Tensor();
  g.clear();
  EXPECT_EQ(nn::live_memory_bytes(), base);
}

TEST(NnMemory, RetainGraphKeepsSavedTensorsUntilDropped) {
  const auto         w    = mlp_weights(10, 32, DType::Float64);
  const std::int64_t base = nn::live_memory_bytes();
  Tensor             h    = nn::normal({4, 32}, 0, 1, DType::Float64);
  for (const auto& wi : w) h = mlp_layer(h, wi);
  Tensor             loss  = nn::sum(h);
  const std::int64_t saved = nn::live_memory_bytes() - base;
  h                        = Tensor();
  nn::GradOptions keep;
  keep.retain_graph = true;
  auto g1           = nn::grad(loss, w, keep);
  EXPECT_GE(nn::live_memory_bytes() - base - bytes_of(g1), saved - 4 * 32 * 8);
  auto g2 = nn::grad(loss, w);
  expect_close(g1, g2, 0);
  g1.clear();
  EXPECT_LE(nn::live_memory_bytes() - base - bytes_of(g2), 64);
  loss = Tensor();
  g2.clear();
  EXPECT_EQ(nn::live_memory_bytes(), base);
}

TEST(NnMemory, CreateGraphLeavesNoCycleBehind) {
  const auto         w    = mlp_weights(5, 16, DType::Float64);
  const std::int64_t base = nn::live_memory_bytes();
  {
    Tensor h = nn::normal({4, 16}, 0, 1, DType::Float64);
    for (const auto& wi : w) h = nn::tanh(nn::matmul(h, wi));
    nn::GradOptions o;
    o.create_graph = true;
    auto   g       = nn::grad(nn::sum(nn::square(h)), w, o);
    Tensor penalty = nn::sum(nn::square(g[0]));
    auto   gg      = nn::grad(penalty, w);
  }
  EXPECT_EQ(nn::live_memory_bytes(), base);
}

// Every op's backward closure must not keep its own outputs alive through a
// reference cycle, with or without a backward pass.
TEST(NnMemory, OpGraphsDoNotLeak) {
  using Op                          = std::function<Tensor(const Tensor&)>;
  const std::vector<std::pair<const char*, Op>> ops = {
      {"relu", [](const Tensor& x) { return nn::relu(x); }},
      {"sigmoid", [](const Tensor& x) { return nn::sigmoid(x); }},
      {"tanh", [](const Tensor& x) { return nn::tanh(x); }},
      {"exp", [](const Tensor& x) { return nn::exp(x); }},
      {"sqrt", [](const Tensor& x) { return nn::sqrt(nn::abs(x) + 1.0); }},
      {"softmax", [](const Tensor& x) { return nn::softmax(x); }},
      {"log_softmax", [](const Tensor& x) { return nn::log_softmax(x); }},
      {"matmul", [](const Tensor& x) { return nn::matmul(x, nn::transpose(x)); }},
      {"max", [](const Tensor& x) { return nn::max(x, {1}); }},
      {"logsumexp", [](const Tensor& x) { return nn::logsumexp(x, {1}); }},
      {"concat/split", [](const Tensor& x) { return nn::split(nn::concat({x, x}, 1), 2, 1)[1]; }},
      {"gather", [](const Tensor& x) { return nn::gather(x, nn::from_vector<std::int64_t>({2, 0}), 0); }},
      {"conv2d", [](const Tensor& x) {
         return nn::conv2d(x.reshape({1, 4, 6, 1}), nn::ones({3, 3, 1, 2}, x.dtype()), nn::conv_options(1));
       }},
      {"max_pool", [](const Tensor& x) { return nn::max_pool2d(x.reshape({1, 4, 6, 1}), 2, 2); }},
      {"custom", [](const Tensor& x) {
         return nn::apply_function("Twice", {x}, [](const Inputs& in) { return Inputs{in[0] * 2.0}; },
                                   [](const Inputs&, const Inputs&, const Inputs& g) { return Inputs{g[0] * 2.0}; })[0];
       }},
  };
  for (const auto& [name, op] : ops) {
    const std::int64_t base = nn::live_memory_bytes();
    {
      Tensor x = nn::normal({4, 6}, 0, 1, DType::Float64).requires_grad_();
      Tensor y = op(x);
      y        = nn::sum(nn::square(y));
    }
    EXPECT_EQ(nn::live_memory_bytes(), base) << name << " leaks when its graph is dropped";
    {
      Tensor x = nn::normal({4, 6}, 0, 1, DType::Float64).requires_grad_();
      Tensor y = nn::sum(nn::square(op(x)));
      y.backward();
      x.zero_grad();
    }
    EXPECT_EQ(nn::live_memory_bytes(), base) << name << " leaks after backward";
  }
}

// A root that another root's graph reaches must wait for that root's
// gradient (grad of several outputs where one feeds another).
TEST(NnAutogradEngine, RootsThatFeedOtherRoots) {
  Tensor x = nn::tensor({0.3, -0.7}, {2}, DType::Float64).requires_grad_();
  // d/dx [sum(y1) + sum(y2)] = 2 + 2 exp(2x) for y1 = 2x, y2 = exp(y1)
  for (int order = 0; order < 3; ++order) {
    Tensor y1   = x * 2.0;
    Tensor y2   = nn::exp(y1);
    auto   outs = order == 0 ? std::vector<Tensor>{y1, y2} : order == 1 ? std::vector<Tensor>{y2, y1} : std::vector<Tensor>{x, y2};
    auto   g    = nn::grad(outs, {x}, {nn::ones_like(outs[0]), nn::ones_like(outs[1])});
    for (int i = 0; i < 2; ++i) {
      const double xi = x.at({i});
      const double expected = outs[0].is_same(x) ? 1 + 2 * std::exp(2 * xi) : 2 + 2 * std::exp(2 * xi);
      EXPECT_NEAR(g[0].at({i}), expected, 1e-12);
    }
  }
}

// ---- checkpoint -------------------------------------------------------------------------------

/// A float64 two-layer block whose weights come from the closure.
nn::CheckpointFn1 block_fn(const Tensor& w1, const Tensor& w2) {
  return [w1, w2](const Tensor& x) { return nn::tanh(nn::matmul(nn::sigmoid(nn::matmul(x, w1)), w2)); };
}

TEST(NnCheckpoint, MatchesPlainAutograd) {
  nn::Generator gen(11);
  Tensor        x  = nn::normal({3, 5}, 0, 1, gen, DType::Float64).requires_grad_();
  Tensor        w1 = nn::normal({5, 4}, 0, 1, gen, DType::Float64).requires_grad_();
  Tensor        w2 = nn::normal({4, 5}, 0, 1, gen, DType::Float64).requires_grad_();
  auto          fn = block_fn(w1, w2);

  Tensor plain = nn::sum(nn::square(fn(fn(x))));
  Tensor ckpt  = nn::sum(nn::square(nn::checkpoint(fn, nn::checkpoint(fn, x))));
  EXPECT_EQ(plain.item(), ckpt.item());
  EXPECT_EQ(ckpt.grad_fn_name(), "Sum");
  auto gp = nn::grad(plain, {x, w1, w2});
  auto gc = nn::grad(ckpt, {x, w1, w2});
  expect_close(gp, gc, 1e-12);

  // backward() accumulates into the leaves the same way.
  Tensor again = nn::sum(nn::square(nn::checkpoint(fn, nn::checkpoint(fn, x))));
  again.backward();
  expect_close({x.grad(), w1.grad(), w2.grad()}, gp, 1e-12);
}

TEST(NnCheckpoint, Gradcheck) {
  nn::Generator gen(12);
  Inputs        in = {nn::normal({2, 3}, 0, 1, gen, DType::Float64).requires_grad_(),
                      nn::normal({3, 3}, 0, 1, gen, DType::Float64).requires_grad_(),
                      nn::normal({3, 3}, 0, 1, gen, DType::Float64).requires_grad_()};
  // Weights reach the segment only through its closure.
  auto f = [](const Inputs& v) {
    Tensor h = nn::checkpoint(block_fn(v[1], v[2]), v[0]);
    return nn::checkpoint(block_fn(v[2], v[1]), h * 1.5);
  };
  auto r = nn::gradcheck(f, in);
  EXPECT_TRUE(r.ok) << r.message;
  auto rr = nn::gradgradcheck(f, in);
  EXPECT_TRUE(rr.ok) << rr.message;
}

TEST(NnCheckpoint, NestedCheckpoints) {
  nn::Generator gen(13);
  Tensor        x  = nn::normal({2, 4}, 0, 1, gen, DType::Float64).requires_grad_();
  Tensor        w1 = nn::normal({4, 4}, 0, 1, gen, DType::Float64).requires_grad_();
  Tensor        w2 = nn::normal({4, 4}, 0, 1, gen, DType::Float64).requires_grad_();
  auto          fn = block_fn(w1, w2);
  int           outer_runs = 0, inner_runs = 0;
  nn::CheckpointFn1 inner = [&](const Tensor& h) {
    ++inner_runs;
    return fn(h);
  };
  nn::CheckpointFn1 outer = [&](const Tensor& h) {
    ++outer_runs;
    return nn::checkpoint(inner, fn(h));
  };
  Tensor y     = nn::sum(nn::checkpoint(outer, x));
  Tensor plain = nn::sum(fn(fn(x)));
  EXPECT_NEAR(y.item(), plain.item(), 1e-14);
  EXPECT_EQ(outer_runs, 1);
  EXPECT_EQ(inner_runs, 1);
  auto g = nn::grad(y, {x, w1, w2});
  // Outer recomputed once; the inner function runs again inside it and once
  // more for its own backward.
  EXPECT_EQ(outer_runs, 2);
  EXPECT_EQ(inner_runs, 3);
  expect_close(g, nn::grad(plain, {x, w1, w2}), 1e-12);

  auto r = nn::gradgradcheck(
      [&](const Inputs& v) {
        auto f = block_fn(v[1], v[2]);
        return nn::checkpoint([f](const Tensor& h) { return nn::checkpoint(f, f(h)); }, v[0]);
      },
      {x, w1, w2});
  EXPECT_TRUE(r.ok) << r.message;
}

TEST(NnCheckpoint, SecondOrderThroughACheckpoint) {
  nn::Generator gen(14);
  Tensor        x  = nn::normal({3, 4}, 0, 1, gen, DType::Float64).requires_grad_();
  Tensor        w1 = nn::normal({4, 4}, 0, 1, gen, DType::Float64).requires_grad_();
  Tensor        w2 = nn::normal({4, 4}, 0, 1, gen, DType::Float64).requires_grad_();
  auto          fn = block_fn(w1, w2);
  nn::GradOptions o;
  o.create_graph = true;
  // Gradient penalty: d/dw of ||d f / dx||^2.
  auto penalty = [&](bool ckpt) {
    Tensor y  = nn::sum(nn::square(ckpt ? nn::checkpoint(fn, x) : fn(x)));
    Tensor gx = nn::grad(y, {x}, o)[0];
    return nn::grad(nn::sum(nn::square(gx)), {w1, w2, x});
  };
  expect_close(penalty(true), penalty(false), 1e-11);
}

TEST(NnCheckpoint, ReplaysRandomness) {
  nn::Generator gen(15);
  Tensor        x = nn::normal({16, 8}, 0, 1, gen, DType::Float64).requires_grad_();
  Tensor        w = nn::normal({8, 8}, 0, 1, gen, DType::Float64).requires_grad_();
  nn::Generator drop(99);
  auto          fn = [&](const Tensor& h) { return nn::dropout(nn::matmul(h, w), 0.5, true, drop); };
  auto          run = [&](bool ckpt, const nn::CheckpointOptions& opt) {
    drop.seed(7);
    nn::manual_seed(8);
    // The default generator is used too.
    auto f2 = [&](const Tensor& h) { return fn(h) * nn::uniform({16, 8}, 0.5, 1.5, DType::Float64); };
    Tensor y = nn::sum(nn::square(ckpt ? nn::checkpoint(nn::CheckpointFn1(f2), x, opt) : f2(x)));
    return nn::grad(y, {x, w});
  };
  nn::CheckpointOptions opt;
  opt.generators = {&drop};
  auto plain     = run(false, opt);
  auto ckpt      = run(true, opt);
  expect_close(ckpt, plain, 1e-12);

  // The generators continue where the forward pass left them.
  drop.seed(7);
  nn::manual_seed(8);
  Tensor y = nn::sum(nn::checkpoint(nn::CheckpointFn1(fn), x, opt));
  nn::Generator expected = drop;
  auto          g        = nn::grad(y, {w});
  EXPECT_EQ(drop.next_u64(), expected.next_u64());

  // Without replaying the dropout generator the masks differ.
  nn::CheckpointOptions no_replay;
  auto                  wrong = run(true, no_replay);
  EXPECT_GT(max_abs_diff(wrong[1], plain[1]), 1e-3);
}

TEST(NnCheckpoint, MultipleOutputsAndEdgeCases) {
  nn::Generator gen(16);
  Tensor        a = nn::normal({3}, 0, 1, gen, DType::Float64).requires_grad_();
  Tensor        b = nn::normal({3}, 0, 1, gen, DType::Float64).requires_grad_();
  Tensor        c = nn::normal({3}, 0, 1, gen, DType::Float64);  // no grad
  nn::CheckpointFn fn = [](const Inputs& in) {
    return Inputs{in[0] * in[1], in[0],  // an input returned unchanged
                  nn::argmax(in[1], 0),  // an integer output
                  in[2] * 2.0,           // does not depend on anything with grad
                  nn::exp(in[0]) + in[2]};
  };
  auto plain = fn({a, b, c});
  auto out   = nn::checkpoint(fn, {a, b, c});
  ASSERT_EQ(out.size(), 5u);
  EXPECT_FALSE(out[3].requires_grad());
  EXPECT_FALSE(out[2].requires_grad());
  auto loss = [](const Inputs& o) { return nn::sum(o[0] * 2.0) + nn::sum(nn::square(o[1])) + nn::sum(o[4]); };
  expect_close(nn::grad(loss(out), {a, b}), nn::grad(loss(plain), {a, b}), 1e-12);

  // One tensor passed as two inputs gets its gradient once.
  Tensor y  = nn::sum(nn::checkpoint(fn, {a, a, c})[0]);
  Tensor yp = nn::sum(a * a);
  expect_close(nn::grad(y, {a}), nn::grad(yp, {a}), 1e-12);

  // Grad mode off, or nothing requiring grad: just the function.
  {
    nn::NoGradGuard ng;
    EXPECT_FALSE(nn::checkpoint(fn, {a, b, c})[0].requires_grad());
  }
  EXPECT_FALSE(nn::checkpoint(fn, {c, c, c})[0].requires_grad());
}

TEST(NnCheckpoint, RetainGraphAndErrors) {
  nn::Generator gen(17);
  Tensor        x  = nn::normal({2, 3}, 0, 1, gen, DType::Float64).requires_grad_();
  Tensor        w1 = nn::normal({3, 3}, 0, 1, gen, DType::Float64).requires_grad_();
  Tensor        w2 = nn::normal({3, 3}, 0, 1, gen, DType::Float64).requires_grad_();
  Tensor        y  = nn::sum(nn::checkpoint(block_fn(w1, w2), x));
  nn::GradOptions keep;
  keep.retain_graph = true;
  auto g1           = nn::grad(y, {w1}, keep);
  auto g2           = nn::grad(y, {w1});
  expect_close(g1, g2, 0);
  EXPECT_THROW(nn::grad(y, {w1}), std::logic_error);

  // An optimizer step between forward and backward is reported.
  Tensor y2 = nn::sum(nn::checkpoint(block_fn(w1, w2), x));
  {
    nn::NoGradGuard ng;
    w2.add_(nn::ones_like(w2), 0.1);
  }
  EXPECT_THROW(nn::grad(y2, {w1}), std::logic_error);

  // A function that reads something new when recomputed is reported.
  int               calls = 0;
  nn::CheckpointFn1 fickle = [&](const Tensor& h) { return ++calls == 1 ? h * 2.0 : h * w1.sum(); };
  Tensor y3 = nn::sum(nn::checkpoint(fickle, x));
  EXPECT_THROW(nn::grad(y3, {x}), std::logic_error);

  // grad inside a checkpointed function is not supported.
  nn::CheckpointFn1 inner_grad = [&](const Tensor& h) { return nn::grad(nn::sum(h * h), {h})[0]; };
  EXPECT_THROW(nn::checkpoint(inner_grad, x), std::logic_error);
}

TEST(NnCheckpoint, InRecomputeFlag) {
  Tensor           x = nn::ones({2}, DType::Float64).requires_grad_();
  std::vector<int> seen;
  Tensor           y = nn::sum(nn::checkpoint(
      [&](const Tensor& h) {
        seen.push_back(nn::in_recompute() ? 1 : 0);
        return h * 3.0;
      },
      x));
  EXPECT_FALSE(nn::in_recompute());
  nn::grad(y, {x});
  EXPECT_EQ(seen, (std::vector<int>{0, 1}));
}

// The memory saving: a 100-layer MLP with ten checkpointed segments keeps
// the ten segment inputs plus one segment's activations while it backprops.
TEST(NnCheckpoint, SequentialSavesMemory) {
  const int          L = 100;
  const std::int64_t B = 512, W = 32;  // activations dominate the weights
  const auto         w   = mlp_weights(L, W);
  const Tensor       x   = nn::normal({B, W});
  const std::int64_t act = B * W * 4, wbytes = W * W * 4;
  std::vector<nn::CheckpointFn1> layers;
  for (const auto& wi : w) layers.push_back([wi](const Tensor& h) { return mlp_layer(h, wi); });

  auto measure = [&](std::int64_t segments, std::vector<Tensor>* grads) {
    nn::PeakMemoryScope scope;
    Tensor              loss;
    if (segments == 0) {
      Tensor h = x;
      for (const auto& f : layers) h = f(h);
      loss = nn::mean(nn::square(h));
    } else {
      loss = nn::mean(nn::square(nn::checkpoint_sequential(layers, segments, x)));
    }
    *grads = nn::grad(loss, w);
    return scope.peak_above_baseline();
  };
  std::vector<Tensor> g_plain, g_ckpt;
  const std::int64_t  plain = measure(0, &g_plain);
  const std::int64_t  ckpt  = measure(10, &g_ckpt);
  expect_close(g_ckpt, g_plain, 1e-6);
  EXPECT_GE(plain, L * act);
  // 10 segment inputs + the 10 activations of one recomputed segment + temporaries + weight gradients.
  EXPECT_LE(ckpt, (10 + 10 + 6) * act + L * wbytes);
  EXPECT_LT(ckpt * 3, plain) << "checkpointing should cut the activation memory several times";
  std::cout << "  100-layer MLP: activation peak " << plain / 1024 << " KiB plain, " << ckpt / 1024
            << " KiB with 10 checkpointed segments\n";
}

// ---- reversible sequences ----------------------------------------------------------------------

std::vector<nn::ReversibleBlock> coupling_blocks(int n, const std::vector<Tensor>& w) {
  std::vector<nn::ReversibleBlock> blocks;
  for (int i = 0; i < n; ++i) {
    Tensor wf = w[2 * i], wg = w[2 * i + 1];
    blocks.push_back({[wf](const Tensor& h) { return nn::tanh(nn::matmul(h, wf)); },
                      [wg](const Tensor& h) { return nn::sigmoid(nn::matmul(h, wg)) - 0.5; }});
  }
  return blocks;
}

std::vector<Tensor> coupling_plain(const std::vector<nn::ReversibleBlock>& blocks, Tensor a, Tensor b) {
  for (const auto& k : blocks) {
    a = a + k.f(b);
    b = b + k.g(a);
  }
  return {a, b};
}

TEST(NnReversible, MatchesPlainAutogradAndGradchecks) {
  const auto w      = mlp_weights(8, 4, DType::Float64, 21);
  const auto blocks = coupling_blocks(4, w);
  nn::Generator gen(22);
  Tensor        x1 = nn::normal({3, 4}, 0, 1, gen, DType::Float64).requires_grad_();
  Tensor        x2 = nn::normal({3, 4}, 0, 1, gen, DType::Float64).requires_grad_();
  auto          y  = nn::reversible_sequence(blocks, x1, x2);
  auto          yp = coupling_plain(blocks, x1, x2);
  EXPECT_EQ(max_abs_diff(y[0], yp[0]), 0);
  EXPECT_EQ(max_abs_diff(y[1], yp[1]), 0);
  EXPECT_EQ(y[0].grad_fn_name(), "ReversibleSequence");
  auto loss = [](const std::vector<Tensor>& o) { return nn::sum(nn::square(o[0])) + nn::sum(nn::sin(o[1])); };
  Inputs wrt{x1, x2};
  wrt.insert(wrt.end(), w.begin(), w.end());
  expect_close(nn::grad(loss(y), wrt), nn::grad(loss(yp), wrt), 1e-12);

  auto r = nn::gradcheck(
      [](const Inputs& v) {
        auto b = coupling_blocks(2, {v[2], v[3], v[4], v[5]});
        auto o = nn::reversible_sequence(b, v[0], v[1]);
        return nn::concat({o[0], o[1] * 2.0}, 1);
      },
      {x1, x2, w[0], w[1], w[2], w[3]});
  EXPECT_TRUE(r.ok) << r.message;

  nn::GradOptions o;
  o.create_graph = true;
  EXPECT_THROW(nn::grad(loss(nn::reversible_sequence(blocks, x1, x2)), {x1}, o), std::invalid_argument);
}

TEST(NnReversible, MemoryDoesNotGrowWithDepth) {
  const std::int64_t B = 512, W = 16, act = B * W * 4;
  auto               measure = [&](int n, bool reversible) {
    const auto    w = mlp_weights(2 * n, W, DType::Float32, 23);
    const auto    blocks = coupling_blocks(n, w);
    nn::Generator gen(24);
    Tensor        x1 = nn::normal({B, W}, 0, 1, gen).requires_grad_(), x2 = nn::normal({B, W}, 0, 1, gen).requires_grad_();
    nn::PeakMemoryScope scope;
    auto   y    = reversible ? nn::reversible_sequence(blocks, x1, x2) : coupling_plain(blocks, x1, x2);
    Tensor loss = nn::sum(nn::square(y[0])) + nn::sum(nn::square(y[1]));
    y.clear();
    auto g = nn::grad(loss, w);
    return scope.peak_above_baseline() - bytes_of(g);
  };
  const std::int64_t rev8 = measure(8, true), rev32 = measure(32, true), plain32 = measure(32, false);
  EXPECT_LE(rev32, rev8 + 8 * act) << "reversible memory should not grow with depth";
  EXPECT_LE(rev32, 24 * act);
  EXPECT_GT(plain32, 4 * rev32);
  std::cout << "  32 coupling blocks: activation peak " << plain32 / 1024 << " KiB plain, " << rev32 / 1024
            << " KiB reversible (" << rev8 / 1024 << " KiB with 8 blocks)\n";
}

}  // namespace
