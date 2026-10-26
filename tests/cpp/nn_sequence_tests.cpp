// Tests for datamunge::nn CTC loss / decoding and the linear-chain CRF.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <datamunge/nn/autograd.hpp>
#include <datamunge/nn/gradcheck.hpp>
#include <datamunge/nn/io.hpp>
#include <datamunge/nn/ops.hpp>
#include <datamunge/nn/random.hpp>
#include <datamunge/nn/sequence.hpp>
#include <filesystem>
#include <limits>
#include <vector>

namespace nn = datamunge::nn;
using nn::DType;
using nn::Tensor;

namespace {

using I64 = std::int64_t;

Tensor rnd(const nn::Shape& shape, std::uint64_t seed, DType dt = DType::Float64, double sd = 1.0) {
  nn::Generator gen(seed);
  return nn::normal(shape, 0.0, sd, gen, dt);
}

Tensor leaf(const Tensor& t) {
  Tensor l = t.detach();
  l.requires_grad_(true);
  return l;
}

Tensor ints(const std::vector<I64>& v, nn::Shape shape = {}) { return nn::from_vector(v, shape); }

/// Brute-force CTC over every frame path: loss and d loss / d logits for one
/// sequence; logits as [T][C] (already sliced to the valid frames).
struct Brute {
  double              loss;
  std::vector<double> grad;
};
Brute ctc_brute(const std::vector<std::vector<double>>& z, const std::vector<I64>& label, I64 blank) {
  const I64 T = static_cast<I64>(z.size()), C = static_cast<I64>(z[0].size());
  std::vector<std::vector<double>> p(T, std::vector<double>(C));
  for (I64 t = 0; t < T; ++t) {
    double s = 0;
    for (I64 c = 0; c < C; ++c) s += std::exp(z[t][c]);
    for (I64 c = 0; c < C; ++c) p[t][c] = std::exp(z[t][c]) / s;
  }
  I64 paths = 1;
  for (I64 t = 0; t < T; ++t) paths *= C;
  double              total = 0;
  std::vector<double> occ(T * C, 0.0);
  for (I64 k = 0; k < paths; ++k) {
    std::vector<I64> path(T);
    I64              n = k;
    double           pr = 1;
    for (I64 t = 0; t < T; ++t) path[t] = n % C, n /= C, pr *= p[t][path[t]];
    if (nn::ctc_collapse(path, blank) != label) continue;
    total += pr;
    for (I64 t = 0; t < T; ++t) occ[t * C + path[t]] += pr;
  }
  Brute b{-std::log(total), std::vector<double>(T * C)};
  for (I64 t = 0; t < T; ++t)
    for (I64 c = 0; c < C; ++c) b.grad[t * C + c] = p[t][c] - occ[t * C + c] / total;
  return b;
}

/// Path score with optional start/end, from flat arrays.
double crf_path_score(const std::vector<double>& U, I64 b, I64 T, I64 K, const std::vector<I64>& y,
                      const std::vector<double>& A, const std::vector<double>& st, const std::vector<double>& en) {
  if (y.empty()) return 0;
  double s = st[y[0]] + en[y.back()];
  for (std::size_t t = 0; t < y.size(); ++t) {
    s += U[(b * T + static_cast<I64>(t)) * K + y[t]];
    if (t) s += A[y[t - 1] * K + y[t]];
  }
  return s;
}

}  // namespace

// ---- CTC ----------------------------------------------------------------------------------------

TEST(NnSequenceCtc, CollapseMergesRepeatsBeforeDroppingBlanks) {
  EXPECT_EQ(nn::ctc_collapse({1, 1, 36, 1, 36, 1}, 36), (std::vector<I64>{1, 1, 1}));
  EXPECT_TRUE(nn::ctc_collapse({36, 36}, 36).empty());
  EXPECT_EQ(nn::ctc_collapse({0, 0, 1, 2, 2}, 2, false), (std::vector<I64>{0, 0, 1}));
}

// tflow crnn_ctc's fixture: repeated label "00" in 3 frames, blank = C - 1.
TEST(NnSequenceCtc, TflowRepeatedLabelFixture) {
  const Tensor z = nn::tensor({.2, -.3, .4, -.1, .5, .1, .7, -.2, .3}, {3, 1, 3}, DType::Float32);
  auto         r = nn::ctc_loss_with_gradient(z, ints({0, 0}, {1, 2}), ints({2}), ints({3}));
  const auto   b = ctc_brute({{.2, -.3, .4}, {-.1, .5, .1}, {.7, -.2, .3}}, {0, 0}, 2);
  EXPECT_NEAR(r.loss.item<double>(), b.loss, 5e-7);
  const auto g = r.gradient.to_vector<double>();
  for (std::size_t i = 0; i < g.size(); ++i) EXPECT_NEAR(g[i], b.grad[i], 1e-6);
}

TEST(NnSequenceCtc, MatchesBruteForceVariableLengthsBothLayoutsAndBlanks) {
  const I64 T = 5, B = 4, C = 4;
  const Tensor z = rnd({T, B, C}, 11);
  const auto   zv = z.to_vector<double>();
  const std::vector<std::vector<I64>> labels{{1, 2}, {1, 1}, {}, {2, 1, 2}};
  const std::vector<I64>              lt{5, 4, 3, 5};
  for (I64 blank : {I64{0}, I64{-1}}) {
    const I64 bl = blank < 0 ? C - 1 : 0;
    // remap labels so none equals the blank
    std::vector<I64> flat(B * 3, -7), ll;
    std::vector<std::vector<I64>> lab = labels;
    for (auto& s : lab)
      for (auto& c : s) c = bl == 0 ? c : c - 1;
    for (I64 b = 0; b < B; ++b) {
      for (std::size_t i = 0; i < lab[b].size(); ++i) flat[b * 3 + static_cast<I64>(i)] = lab[b][i];
      ll.push_back(static_cast<I64>(lab[b].size()));
    }
    for (bool time_major : {true, false}) {
      nn::CtcOptions o;
      o.blank      = blank;
      o.time_major = time_major;
      const Tensor x = time_major ? z : nn::transpose(z, {1, 0, 2}).contiguous();
      auto         r = nn::ctc_loss_with_gradient(x, ints(flat, {B, 3}), ints(ll), ints(lt), o);
      const Tensor g = time_major ? r.gradient : nn::transpose(r.gradient, {1, 0, 2}).contiguous();
      const auto   gv = g.to_vector<double>();
      for (I64 b = 0; b < B; ++b) {
        std::vector<std::vector<double>> zz(lt[b], std::vector<double>(C));
        for (I64 t = 0; t < lt[b]; ++t)
          for (I64 c = 0; c < C; ++c) zz[t][c] = zv[(t * B + b) * C + c];
        const auto br = ctc_brute(zz, lab[b], bl);
        EXPECT_NEAR(r.loss.to_vector<double>()[b], br.loss, 1e-10);
        for (I64 t = 0; t < T; ++t)
          for (I64 c = 0; c < C; ++c)
            EXPECT_NEAR(gv[(t * B + b) * C + c], t < lt[b] ? br.grad[t * C + c] : 0.0, 1e-10);
      }
    }
  }
}

TEST(NnSequenceCtc, Float32AgreesWithFloat64) {
  const Tensor z = rnd({6, 2, 5}, 3, DType::Float64, 2.0);
  const auto   a = nn::ctc_loss(z, ints({0, 1, 0, 2, 2, 3}, {2, 3}), ints({3, 2}), ints({6, 5}));
  const auto   f = nn::ctc_loss(nn::cast(z, DType::Float32), ints({0, 1, 0, 2, 2, 3}, {2, 3}), ints({3, 2}), ints({6, 5}));
  EXPECT_EQ(f.dtype(), DType::Float32);
  for (int i = 0; i < 2; ++i) EXPECT_NEAR(a.to_vector<double>()[i], f.to_vector<double>()[i], 1e-5);
}

TEST(NnSequenceCtc, Gradcheck) {
  const Tensor z = leaf(rnd({5, 3, 4}, 5));
  const Tensor labels = ints({0, 1, 0, 2, 2, -1, 1, -1, -1}, {3, 3});
  auto f = [&](const std::vector<Tensor>& in) {
    nn::CtcOptions o;
    o.blank = 3;
    // weight the batch so every example's upstream gradient differs
    return nn::sum(nn::mul(nn::ctc_loss(in[0], labels, ints({3, 2, 1}), ints({5, 4, 3}), o), nn::tensor({1.0, 2.0, -0.5}, {}, DType::Float64)));
  };
  const auto r = nn::gradcheck(f, {z});
  EXPECT_TRUE(r.ok) << r.message;
  // backward of the gradient w.r.t. the upstream weights is exact (first-order primitive)
  const Tensor w = leaf(nn::tensor({1.0, 2.0, -0.5}, {}, DType::Float64));
  auto h = [&](const std::vector<Tensor>& in) {
    const Tensor zz   = leaf(in[0]);
    const Tensor loss = nn::ctc_loss(zz, labels, ints({3, 2, 1}), ints({5, 4, 3}), {3, true, false});
    const auto   g    = nn::grad(nn::sum(nn::mul(loss, in[1])), {zz}, nn::GradOptions{true, true, false});
    return nn::sum(nn::square(g[0]));
  };
  const auto r2 = nn::gradcheck(h, {z.detach(), w});
  EXPECT_TRUE(r2.ok) << r2.message;
}

TEST(NnSequenceCtc, BatchMajorGradcheck) {
  const Tensor z = leaf(rnd({2, 4, 3}, 8));
  auto         f = [&](const std::vector<Tensor>& in) {
    nn::CtcOptions o;
    o.blank      = 0;
    o.time_major = false;
    return nn::mean(nn::ctc_loss(in[0], ints({1, 2, 2, 2}, {2, 2}), ints({2, 2}), ints({4, 3}), o));
  };
  const auto r = nn::gradcheck(f, {z});
  EXPECT_TRUE(r.ok) << r.message;
}

TEST(NnSequenceCtc, InfeasibleLabelThrowsOrIsIgnored) {
  const Tensor z = rnd({2, 1, 3}, 1);
  EXPECT_THROW(nn::ctc_loss(z, ints({0, 0}, {1, 2}), ints({2}), ints({2})), std::invalid_argument);
  nn::CtcOptions o;
  o.ignore_longer_outputs_than_inputs = true;
  auto r = nn::ctc_loss_with_gradient(z, ints({0, 0}, {1, 2}), ints({2}), ints({2}), o);
  EXPECT_EQ(r.loss.item<double>(), 0.0);
  EXPECT_EQ(nn::sum(nn::abs(r.gradient)).item<double>(), 0.0);
  EXPECT_THROW(nn::ctc_loss(z, ints({2}, {1, 1}), ints({1}), ints({2})), std::invalid_argument);  // blank as label
}

TEST(NnSequenceCtc, EmptyLabelIsAllBlankPath) {
  const Tensor z  = rnd({3, 1, 3}, 4);
  const Tensor ls = nn::log_softmax(z, -1);
  const double expect = -(nn::sum(nn::slice(ls, {0, 0, 2}, {3, 1, 1})).item<double>());
  EXPECT_NEAR(nn::ctc_loss(z, ints({-1}, {1, 1}), ints({0}), ints({3})).item<double>(), expect, 1e-12);
}

TEST(NnSequenceCtc, GreedyDecode) {
  // [T=6, B=2, C=3], blank = 2
  std::vector<double> v(36, 0.0);
  const std::vector<I64> p0{0, 0, 2, 0, 1, 1}, p1{2, 2, 1, 2, 1, 0};
  for (I64 t = 0; t < 6; ++t) {
    v[(t * 2 + 0) * 3 + p0[t]] = 2.0;
    v[(t * 2 + 1) * 3 + p1[t]] = 1.5;
  }
  const Tensor z = nn::from_vector(v, {6, 2, 3});
  const auto   d = nn::ctc_greedy_decode(z, ints({6, 4}));
  EXPECT_EQ(d.sequences[0], (std::vector<I64>{0, 0, 1}));
  EXPECT_EQ(d.sequences[1], (std::vector<I64>{1}));
  EXPECT_EQ(d.dense.shape(), (nn::Shape{2, 3}));
  EXPECT_EQ(d.dense.to_vector<I64>(), (std::vector<I64>{0, 0, 1, 1, -1, -1}));
  EXPECT_NEAR(d.neg_sum_logits.to_vector<double>()[0], -12.0, 1e-12);
  EXPECT_NEAR(d.neg_sum_logits.to_vector<double>()[1], -6.0, 1e-12);
  EXPECT_NEAR(d.path_log_prob.to_vector<double>()[0], 6 * (2.0 - std::log(std::exp(2.0) + 2.0)), 1e-12);
}

// ---- CRF ---------------------------------------------------------------------------------------

namespace {

struct CrfCase {
  I64    B = 5, T = 4, K = 3;
  Tensor U, A, st, en, len, tags;
  CrfCase() {
    U    = rnd({B, T, K}, 21);
    A    = rnd({K, K}, 22);
    st   = rnd({K}, 23);
    en   = rnd({K}, 24);
    len  = ints({0, 1, 2, 3, 4});
    tags = ints({0, 1, 2, 1, 2, -1, 0, 0, 1, 1, 0, 0, 2, 0, 1, 2, 0, 2, 2, 1}, {B, T});
  }
};

}  // namespace

TEST(NnSequenceCrf, LogNormScoreAndViterbiMatchBruteForce) {
  CrfCase    c;
  const auto U = c.U.to_vector<double>(), A = c.A.to_vector<double>();
  for (bool ends : {false, true}) {
    const Tensor st = ends ? c.st : Tensor(), en = ends ? c.en : Tensor();
    const auto   sv = ends ? c.st.to_vector<double>() : std::vector<double>(3, 0.0);
    const auto   ev = ends ? c.en.to_vector<double>() : std::vector<double>(3, 0.0);
    const auto   z     = nn::crf_log_norm(c.U, c.len, c.A, st, en).to_vector<double>();
    const auto   score = nn::crf_sequence_score(c.U, c.tags, c.len, c.A, st, en).to_vector<double>();
    const auto   ll    = nn::crf_log_likelihood(c.U, c.tags, c.len, c.A, st, en).to_vector<double>();
    const auto   dec   = nn::crf_decode(c.U, c.len, c.A, st, en, -1);
    const auto   tv    = c.tags.to_vector<I64>();
    const auto   dt    = dec.tags.to_vector<I64>();
    for (I64 b = 0; b < c.B; ++b) {
      I64 paths = 1;
      for (I64 t = 0; t < b; ++t) paths *= 3;
      double best = -INFINITY, mass = 0;
      std::vector<I64> arg;
      std::vector<double> scores;
      for (I64 k = 0; k < paths; ++k) {
        std::vector<I64> y(b);
        I64              n = k;
        for (I64 t = b - 1; t >= 0; --t) y[t] = n % 3, n /= 3;
        const double s = crf_path_score(U, b, c.T, c.K, y, A, sv, ev);
        scores.push_back(s);
        if (s > best) best = s, arg = y;
      }
      for (double s : scores) mass += std::exp(s - best);
      const double logz = b ? best + std::log(mass) : 0.0;
      EXPECT_NEAR(z[b], logz, 1e-12);
      std::vector<I64> gold(tv.begin() + b * c.T, tv.begin() + b * c.T + b);
      EXPECT_NEAR(score[b], crf_path_score(U, b, c.T, c.K, gold, A, sv, ev), 1e-12);
      EXPECT_NEAR(ll[b], score[b] - z[b], 1e-12);
      EXPECT_NEAR(dec.scores.to_vector<double>()[b], b ? best : 0.0, 1e-12);
      for (I64 t = 0; t < c.T; ++t) EXPECT_EQ(dt[b * c.T + t], t < b ? arg[t] : -1);
    }
  }
}

TEST(NnSequenceCrf, MarginalsSumToOneOnValidPositions) {
  CrfCase      c;
  const Tensor U = leaf(c.U);
  const auto   g = nn::grad(nn::sum(nn::crf_log_norm(U, c.len, c.A, c.st, c.en)), {U});
  const auto   m = nn::sum(g[0], {2}).to_vector<double>();
  for (I64 b = 0; b < c.B; ++b)
    for (I64 t = 0; t < c.T; ++t) EXPECT_NEAR(m[b * c.T + t], t < b ? 1.0 : 0.0, 1e-12);
}

TEST(NnSequenceCrf, Gradcheck) {
  CrfCase c;
  auto    f = [&](const std::vector<Tensor>& in) {
    return nn::sum(nn::mul(nn::crf_log_likelihood(in[0], c.tags, c.len, in[1], in[2], in[3]),
                           nn::tensor({1.0, -1.0, 0.5, 2.0, 1.5}, {}, DType::Float64)));
  };
  const std::vector<Tensor> in{leaf(c.U), leaf(c.A), leaf(c.st), leaf(c.en)};
  const auto r = nn::gradcheck(f, in);
  EXPECT_TRUE(r.ok) << r.message;
  const auto r2 = nn::gradgradcheck(f, in);
  EXPECT_TRUE(r2.ok) << r2.message;
}

TEST(NnSequenceCrf, GradcheckNoEndpointsAndForwardTrace) {
  CrfCase c;
  auto    f = [&](const std::vector<Tensor>& in) { return nn::mean(nn::crf_log_norm(in[0], c.len, in[1])); };
  const auto r = nn::gradcheck(f, {leaf(c.U), leaf(c.A)});
  EXPECT_TRUE(r.ok) << r.message;
  const auto alphas = nn::crf_forward(c.U, c.len, c.A, c.st);
  EXPECT_EQ(alphas.shape(), (nn::Shape{5, 4, 3}));
  // past the length the alpha is carried unchanged
  const auto a = alphas.to_vector<double>();
  for (I64 k = 0; k < 3; ++k) EXPECT_EQ(a[(2 * 4 + 3) * 3 + k], a[(2 * 4 + 1) * 3 + k]);
}

TEST(NnSequenceCrf, Float32AndPaddingInvariance) {
  CrfCase      c;
  const auto   z64 = nn::crf_log_norm(c.U, c.len, c.A, c.st, c.en).to_vector<double>();
  const auto   z32 = nn::crf_log_norm(nn::cast(c.U, DType::Float32), c.len, nn::cast(c.A, DType::Float32),
                                      nn::cast(c.st, DType::Float32), nn::cast(c.en, DType::Float32))
                       .to_vector<double>();
  for (I64 b = 0; b < c.B; ++b) EXPECT_NEAR(z32[b], z64[b], 1e-5);
  // changing potentials and tags past each length changes nothing
  auto u = c.U.to_vector<double>();
  auto t = c.tags.to_vector<I64>();
  for (I64 b = 0; b < c.B; ++b)
    for (I64 s = b; s < c.T; ++s) {
      for (I64 k = 0; k < 3; ++k) u[(b * 4 + s) * 3 + k] += 10.0 * (k + 1);
      t[b * 4 + s] = 2;
    }
  const Tensor U2 = nn::from_vector(u, {5, 4, 3}), T2 = ints(t, {5, 4});
  const auto   ll1 = nn::crf_log_likelihood(c.U, c.tags, c.len, c.A, c.st, c.en).to_vector<double>();
  const auto   ll2 = nn::crf_log_likelihood(U2, T2, c.len, c.A, c.st, c.en).to_vector<double>();
  for (I64 b = 0; b < c.B; ++b) EXPECT_NEAR(ll1[b], ll2[b], 1e-12);
}

TEST(NnSequenceCrf, ViterbiDiffersFromUnaryArgmaxWhenTransitionsDominate) {
  // Strong "stay" transitions override a single noisy unary.
  const Tensor U = nn::tensor({2, 0, 0, 2, 0, 1.5, 2, 0, 0}, {1, 3, 3}, DType::Float64);
  const Tensor A = nn::tensor({{3, -3, -3}, {-3, 3, -3}, {-3, -3, 3}}, DType::Float64);
  const auto   d = nn::crf_decode(U, ints({3}), A);
  EXPECT_EQ(d.tags.to_vector<I64>(), (std::vector<I64>{0, 0, 0}));
  EXPECT_EQ(d.tags.dtype(), DType::Int32);
}

// TF-native values saved by tflow's bilstm_crf (crf_fixture_outputs.npz), when present.
TEST(NnSequenceCrf, MatchesTflowNativeFixture) {
  const char*           home = std::getenv("HOME");
  std::filesystem::path dir  = std::filesystem::path(home ? home : ".") / "projects/tflow_examples/output/models/bilstm_crf";
  if (!std::filesystem::exists(dir / "crf_fixture_outputs.npz")) GTEST_SKIP() << "no tflow fixture";
  auto in = nn::load_npz((dir / "crf_fixture_inputs.npz").string());
  auto out = nn::load_npz((dir / "crf_fixture_outputs.npz").string());
  auto get = [](nn::NamedTensors& m, const std::string& k) {
    for (auto& [n, t] : m)
      if (n == k) return t;
    throw std::runtime_error("missing " + k);
  };
  const Tensor U = leaf(nn::cast(get(in, "unaries"), DType::Float64)), A = leaf(nn::cast(get(in, "transition"), DType::Float64));
  const Tensor st = leaf(nn::cast(get(in, "start"), DType::Float64)), en = leaf(nn::cast(get(in, "end"), DType::Float64));
  const Tensor len = get(in, "lengths"), lab = get(in, "labels");
  const Tensor z  = nn::crf_log_norm(U, len, A, st, en);
  const Tensor sc = nn::crf_sequence_score(U, lab, len, A, st, en);
  auto expect_close = [](const Tensor& a, const Tensor& b, double tol) {
    const auto x = a.to_vector<double>(), y = b.to_vector<double>();
    ASSERT_EQ(x.size(), y.size());
    for (std::size_t i = 0; i < x.size(); ++i) EXPECT_NEAR(x[i], y[i], tol) << i;
  };
  expect_close(z, get(out, "log_partition"), 2e-6);
  expect_close(sc, get(out, "gold_score"), 2e-6);
  expect_close(nn::stop_gradient(nn::crf_forward(U, len, A, st)), nn::slice_axis(get(out, "alphas"), 1, 1, 5), 2e-6);
  const auto pg = nn::grad(nn::sum(z), {U, A, st, en});
  expect_close(pg[0], get(out, "partition_gradient_unaries"), 2e-6);
  expect_close(pg[1], get(out, "partition_gradient_transition"), 2e-6);
  expect_close(pg[2], get(out, "partition_gradient_start"), 2e-6);
  expect_close(pg[3], get(out, "partition_gradient_end"), 2e-6);
  const auto lg = nn::grad(nn::mean(nn::sub(nn::crf_log_norm(U, len, A, st, en), nn::crf_sequence_score(U, lab, len, A, st, en))), {U, A, st, en});
  expect_close(lg[0], get(out, "gradient_unaries"), 2e-6);
  expect_close(lg[1], get(out, "gradient_transition"), 2e-6);
  const auto d = nn::crf_decode(get(in, "unaries"), len, get(in, "transition"), get(in, "start"), get(in, "end"), -1);
  EXPECT_EQ(d.tags.to_vector<I64>(), get(out, "viterbi").to_vector<I64>());
  expect_close(d.scores, get(out, "viterbi_scores"), 1e-9);
}
