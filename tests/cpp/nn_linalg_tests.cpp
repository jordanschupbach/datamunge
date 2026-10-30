// Tests for datamunge::nn linear algebra (linalg.hpp) and einsum (einsum.hpp).

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <datamunge/nn/einsum.hpp>
#include <datamunge/nn/linalg.hpp>
#include <datamunge/nn/nn.hpp>
#include <vector>

namespace nn = datamunge::nn;
using nn::DType;
using nn::Tensor;

namespace {

using Inputs = std::vector<Tensor>;

Tensor rnd(const nn::Shape& shape, std::uint64_t seed, DType dt = DType::Float64) {
  nn::Generator gen(seed);
  return nn::normal(shape, 0.0, 1.0, gen, dt);
}

Tensor leaf(const Tensor& t) {
  Tensor l = t.detach();
  l.requires_grad_(true);
  return l;
}

/// A well-conditioned SPD batch: X X^T / n + I.
Tensor spd(const nn::Shape& batch, std::int64_t n, std::uint64_t seed, DType dt = DType::Float64) {
  nn::Shape s = batch;
  s.push_back(n);
  s.push_back(n);
  const Tensor x = rnd(s, seed, dt);
  return nn::add(nn::div(nn::matmul(x, x, false, true), double(n)), nn::eye(n, dt));
}

double max_abs_diff(const Tensor& a, const Tensor& b) {
  return nn::max(nn::abs(nn::sub(nn::cast(a, DType::Float64), nn::cast(b, DType::Float64)))).item();
}

Tensor sym(const Tensor& a) { return nn::mul(nn::add(a, nn::matrix_transpose(a)), 0.5); }

#define EXPECT_GRADCHECK(f, inputs)                         \
  do {                                                      \
    const auto r_ = nn::gradcheck(f, inputs);               \
    EXPECT_TRUE(r_.ok) << r_.message;                       \
  } while (0)
#define EXPECT_GRADGRADCHECK(f, inputs)                     \
  do {                                                      \
    const auto r_ = nn::gradgradcheck(f, inputs);           \
    EXPECT_TRUE(r_.ok) << r_.message;                       \
  } while (0)

Tensor cat_flat(const std::vector<Tensor>& ts) {
  std::vector<Tensor> flat;
  for (const auto& t : ts) flat.push_back(nn::flatten(t));
  return nn::concat(flat, 0);
}

}  // namespace

// ---- structure ---------------------------------------------------------------------------------

TEST(NnLinalgStructure, DiagOps) {
  const Tensor d = nn::tensor({1, 2, 3}, {3}, DType::Float64);
  const Tensor m = nn::matrix_diag(d);
  EXPECT_EQ(m.shape(), (nn::Shape{3, 3}));
  EXPECT_EQ(m.to_vector<double>(), (std::vector<double>{1, 0, 0, 0, 2, 0, 0, 0, 3}));
  const Tensor a = nn::reshape(nn::arange(0, 24, 1, DType::Float64), {2, 3, 4});
  EXPECT_EQ(nn::matrix_diag_part(a).to_vector<double>(), (std::vector<double>{0, 5, 10, 12, 17, 22}));
  EXPECT_EQ(nn::trace(a).to_vector<double>(), (std::vector<double>{15, 51}));
  const Tensor s = nn::set_diag(a, nn::zeros({2, 3}, DType::Float64));
  EXPECT_EQ(nn::matrix_diag_part(s).to_vector<double>(), (std::vector<double>(6, 0.0)));
  EXPECT_EQ(s.at({0, 0, 1}), 1);
  EXPECT_EQ(s.at({1, 2, 3}), 23);
}

TEST(NnLinalgStructure, BandPart) {
  const Tensor a = nn::ones({4, 4}, DType::Float64);
  // causal mask (lower triangle)
  EXPECT_EQ(nn::band_part(a, -1, 0).to_vector<double>(),
            (std::vector<double>{1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1}));
  EXPECT_EQ(nn::band_part(a, 0, -1).to_vector<double>(),
            (std::vector<double>{1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1}));
  EXPECT_EQ(nn::band_part(a, 1, 1).to_vector<double>(),
            (std::vector<double>{1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1}));
  EXPECT_EQ(nn::band_part(a, 0, 0).to_vector<double>(), nn::eye(4, DType::Float64).to_vector<double>());
}

TEST(NnLinalgStructure, Gradients) {
  const Tensor a = leaf(rnd({2, 3, 4}, 1)), d = leaf(rnd({2, 3}, 2));
  EXPECT_GRADCHECK([](const Inputs& in) { return nn::matrix_diag(in[0]); }, {d});
  EXPECT_GRADCHECK([](const Inputs& in) { return nn::matrix_diag_part(in[0]); }, {a});
  EXPECT_GRADCHECK([](const Inputs& in) { return nn::set_diag(in[0], in[1]); }, Inputs({a, d}));
  EXPECT_GRADCHECK([](const Inputs& in) { return nn::band_part(in[0], 1, 0); }, {a});
  EXPECT_GRADCHECK([](const Inputs& in) { return nn::trace(in[0]); }, {a});
  EXPECT_GRADGRADCHECK([](const Inputs& in) { return nn::square(nn::matrix_diag_part(in[0])); }, {a});
}

// ---- cholesky ----------------------------------------------------------------------------------

TEST(NnLinalgCholesky, Reconstruction) {
  for (DType dt : {DType::Float64, DType::Float32}) {
    const Tensor a = spd({3, 2}, 5, 3, dt);
    const Tensor l = nn::cholesky(a);
    EXPECT_EQ(l.dtype(), dt);
    EXPECT_EQ(l.shape(), a.shape());
    EXPECT_LT(max_abs_diff(nn::matmul(l, l, false, true), a), dt == DType::Float64 ? 1e-12 : 1e-5);
    EXPECT_EQ(max_abs_diff(nn::band_part(l, -1, 0), l), 0.0);
  }
  // Only the lower triangle is read.
  const Tensor a  = spd({}, 4, 4);
  const Tensor big = nn::full({4, 4}, 100.0, DType::Float64);
  const Tensor up  = nn::add(a, nn::sub(nn::band_part(big, 0, -1), nn::band_part(big, 0, 0)));  // garbage above
  EXPECT_LT(max_abs_diff(nn::cholesky(up), nn::cholesky(a)), 1e-14);
  EXPECT_THROW(nn::cholesky(nn::tensor({{1, 2}, {2, 1}}, DType::Float64)), std::invalid_argument);
  EXPECT_THROW(nn::cholesky(nn::zeros({2, 3})), std::invalid_argument);
}

TEST(NnLinalgCholesky, Gradients) {
  const Tensor a = leaf(spd({2}, 4, 5));
  const auto   f = [](const Inputs& in) { return nn::cholesky(sym(in[0])); };
  EXPECT_GRADCHECK(f, {a});
  EXPECT_GRADGRADCHECK(f, {a});
}

// ---- triangular_solve / solve / inverse ------------------------------------------------------------

TEST(NnLinalgSolve, TriangularValues) {
  const Tensor full  = nn::add(rnd({2, 4, 4}, 6), nn::mul(nn::eye(4, DType::Float64), 4.0));
  const Tensor b     = rnd({2, 4, 3}, 7);
  for (bool lower : {true, false})
    for (bool adjoint : {false, true}) {
      const Tensor tri = lower ? nn::band_part(full, -1, 0) : nn::band_part(full, 0, -1);
      const Tensor x   = nn::triangular_solve(full, b, lower, adjoint);  // other triangle ignored
      const Tensor ax  = nn::matmul(tri, x, adjoint, false);
      EXPECT_LT(max_abs_diff(ax, b), 1e-12) << lower << adjoint;
    }
  // Broadcasting: one matrix, a batch of right-hand sides.
  const Tensor l  = nn::band_part(nn::select(full, 0, 0), -1, 0);
  const Tensor x  = nn::triangular_solve(l, b);
  EXPECT_EQ(x.shape(), (nn::Shape{2, 4, 3}));
  EXPECT_LT(max_abs_diff(nn::matmul(l, x), b), 1e-12);
}

TEST(NnLinalgSolve, TriangularGradients) {
  const Tensor a = leaf(nn::add(rnd({2, 3, 3}, 8), nn::mul(nn::eye(3, DType::Float64), 3.0)));
  const Tensor b = leaf(rnd({2, 3, 2}, 9));
  for (bool lower : {true, false})
    for (bool adjoint : {false, true}) {
      const auto f = [=](const Inputs& in) { return nn::triangular_solve(in[0], in[1], lower, adjoint); };
      EXPECT_GRADCHECK(f, Inputs({a, b}));
      EXPECT_GRADGRADCHECK(f, Inputs({a, b}));
    }
  const Tensor a1 = leaf(nn::add(rnd({3, 3}, 10), nn::mul(nn::eye(3, DType::Float64), 3.0)));
  EXPECT_GRADCHECK([](const Inputs& in) { return nn::triangular_solve(in[0], in[1]); }, Inputs({a1, b}));
}

TEST(NnLinalgSolve, GeneralSolveAndInverse) {
  const Tensor a = rnd({3, 5, 5}, 11), b = rnd({3, 5, 2}, 12);
  EXPECT_LT(max_abs_diff(nn::matmul(a, nn::solve(a, b)), b), 1e-10);
  EXPECT_LT(max_abs_diff(nn::matmul(a, nn::solve(a, b, true), true, false), b), 1e-10);
  const Tensor eye = nn::broadcast_to(nn::eye(5, DType::Float64), {3, 5, 5});
  EXPECT_LT(max_abs_diff(nn::matmul(a, nn::inverse(a)), eye), 1e-10);
  EXPECT_LT(max_abs_diff(nn::inverse(a, true), nn::matrix_transpose(nn::inverse(a))), 1e-10);
  const Tensor af = nn::cast(a, DType::Float32);
  EXPECT_EQ(nn::solve(af, nn::cast(b, DType::Float32)).dtype(), DType::Float32);
  EXPECT_THROW(nn::solve(nn::tensor({{1, 2}, {2, 4}}, DType::Float64), nn::ones({2, 1}, DType::Float64)),
               std::invalid_argument);
  // cholesky_solve
  const Tensor s = spd({2}, 4, 13), r = rnd({2, 4, 3}, 14);
  EXPECT_LT(max_abs_diff(nn::cholesky_solve(nn::cholesky(s), r), nn::solve(s, r)), 1e-12);
}

TEST(NnLinalgSolve, Gradients) {
  const Tensor a = leaf(nn::add(rnd({2, 3, 3}, 15), nn::mul(nn::eye(3, DType::Float64), 2.0)));
  const Tensor b = leaf(rnd({2, 3, 2}, 16));
  for (bool adjoint : {false, true}) {
    const auto f = [=](const Inputs& in) { return nn::solve(in[0], in[1], adjoint); };
    EXPECT_GRADCHECK(f, Inputs({a, b}));
    EXPECT_GRADGRADCHECK(f, Inputs({a, b}));
  }
  EXPECT_GRADCHECK([](const Inputs& in) { return nn::inverse(in[0]); }, {a});
  const Tensor s = leaf(spd({}, 3, 17));
  EXPECT_GRADCHECK([](const Inputs& in) { return nn::cholesky_solve(nn::cholesky(sym(in[0])), in[1]); },
                   Inputs({s, b}));
}

// ---- determinants ------------------------------------------------------------------------------------

TEST(NnLinalgDet, Values) {
  const Tensor a = nn::tensor({{1, 2}, {3, 4}}, DType::Float64);
  EXPECT_NEAR(nn::det(a).item(), -2.0, 1e-14);
  const auto sl = nn::slogdet(a);
  EXPECT_EQ(sl.sign.item(), -1.0);
  EXPECT_NEAR(sl.log_abs_det.item(), std::log(2.0), 1e-14);
  const auto sing = nn::slogdet(nn::tensor({{1, 2}, {2, 4}}, DType::Float64));
  EXPECT_EQ(sing.sign.item(), 0.0);
  EXPECT_TRUE(std::isinf(sing.log_abs_det.item()));
  const Tensor s  = spd({3}, 5, 18);
  const auto   ss = nn::slogdet(s);
  EXPECT_LT(max_abs_diff(nn::logdet(s), ss.log_abs_det), 1e-12);
  EXPECT_LT(max_abs_diff(nn::det(s), nn::exp(ss.log_abs_det)), 1e-10);
  EXPECT_EQ(nn::det(s).shape(), (nn::Shape{3}));
}

TEST(NnLinalgDet, Gradients) {
  const Tensor a = leaf(rnd({2, 3, 3}, 19));
  EXPECT_GRADCHECK([](const Inputs& in) { return nn::det(in[0]); }, {a});
  EXPECT_GRADCHECK([](const Inputs& in) { return nn::slogdet(in[0]).log_abs_det; }, {a});
  EXPECT_GRADGRADCHECK([](const Inputs& in) { return nn::slogdet(in[0]).log_abs_det; }, {a});
  const Tensor s = leaf(spd({2}, 3, 20));
  const auto   f = [](const Inputs& in) { return nn::logdet(sym(in[0])); };
  EXPECT_GRADCHECK(f, {s});
  EXPECT_GRADGRADCHECK(f, {s});
}

// ---- qr ----------------------------------------------------------------------------------------------

TEST(NnLinalgQr, Reconstruction) {
  for (nn::Shape shape : {nn::Shape{2, 6, 4}, nn::Shape{4, 4}, nn::Shape{3, 5}}) {
    const Tensor a = rnd(shape, 21);
    for (bool full : {false, true}) {
      const auto         qr = nn::qr(a, full);
      const std::int64_t m = a.size(-2), n = a.size(-1), k = std::min(m, n);
      EXPECT_EQ(qr.q.size(-1), full ? m : k);
      EXPECT_EQ(qr.r.size(-2), full ? m : k);
      EXPECT_LT(max_abs_diff(nn::matmul(qr.q, qr.r), a), 1e-12);
      EXPECT_LT(max_abs_diff(nn::matmul(qr.q, qr.q, true, false), nn::eye(qr.q.size(-1), DType::Float64)), 1e-12);
      EXPECT_EQ(max_abs_diff(nn::band_part(qr.r, 0, -1), qr.r), 0.0);
    }
  }
  const auto qf = nn::qr(rnd({5, 3}, 22, DType::Float32));
  EXPECT_EQ(qf.q.dtype(), DType::Float32);
}

TEST(NnLinalgQr, Gradients) {
  for (nn::Shape shape : {nn::Shape{2, 5, 3}, nn::Shape{3, 3}, nn::Shape{3, 5}}) {
    const Tensor a = leaf(rnd(shape, 23));
    EXPECT_GRADCHECK([](const Inputs& in) {
      const auto qr = nn::qr(in[0]);
      return cat_flat({qr.q, qr.r});
    }, {a});
  }
  const Tensor a = leaf(rnd({4, 3}, 24));
  EXPECT_GRADGRADCHECK([](const Inputs& in) { return nn::qr(in[0]).r; }, {a});
}

// ---- svd ---------------------------------------------------------------------------------------------

TEST(NnLinalgSvd, Reconstruction) {
  for (nn::Shape shape : {nn::Shape{2, 6, 4}, nn::Shape{4, 4}, nn::Shape{3, 5}, nn::Shape{40, 7}}) {
    const Tensor a = rnd(shape, 25);
    for (bool full : {false, true}) {
      const auto         r = nn::svd(a, true, full);
      const std::int64_t m = a.size(-2), n = a.size(-1), k = std::min(m, n);
      EXPECT_EQ(r.s.size(-1), k);
      EXPECT_EQ(r.u.size(-1), full ? m : k);
      EXPECT_EQ(r.v.size(-1), full ? n : k);
      const Tensor u1 = nn::slice_axis(r.u, -1, 0, k), v1 = nn::slice_axis(r.v, -1, 0, k);
      EXPECT_LT(max_abs_diff(nn::matmul(nn::mul(u1, nn::expand_dims(r.s, -2)), v1, false, true), a), 1e-12);
      EXPECT_LT(max_abs_diff(nn::matmul(r.u, r.u, true, false), nn::eye(r.u.size(-1), DType::Float64)), 1e-12);
      EXPECT_LT(max_abs_diff(nn::matmul(r.v, r.v, true, false), nn::eye(r.v.size(-1), DType::Float64)), 1e-12);
      const auto s = r.s.to_vector<double>();
      for (std::size_t i = 1; i < s.size(); ++i)
        if ((i % static_cast<std::size_t>(k)) != 0) EXPECT_GE(s[i - 1], s[i]);
    }
  }
  // Known singular values, and a rank-deficient matrix.
  const Tensor d = nn::tensor({{0, 3, 0}, {-5, 0, 0}, {0, 0, 1}, {0, 0, 0}}, DType::Float64);
  EXPECT_LT(max_abs_diff(nn::svd(d, false).s, nn::tensor({5, 3, 1}, {3}, DType::Float64)), 1e-14);
  const Tensor x  = rnd({5, 1}, 26);
  const Tensor r1 = nn::matmul(x, x, false, true);
  const auto   rd = nn::svd(r1);
  EXPECT_LT(max_abs_diff(nn::matmul(rd.u, rd.u, true, false), nn::eye(5, DType::Float64)), 1e-12);
  EXPECT_LT(max_abs_diff(nn::matmul(nn::mul(rd.u, nn::expand_dims(rd.s, -2)), rd.v, false, true), r1), 1e-12);
  EXPECT_NEAR(rd.s.to_vector<double>()[0], nn::sum(nn::square(x)).item(), 1e-12);
  EXPECT_EQ(nn::svd(rnd({3, 3}, 27, DType::Float32)).u.dtype(), DType::Float32);
}

TEST(NnLinalgSvd, Gradients) {
  // Sign-invariant functions of U and V, plus s.
  const auto f = [](bool full) {
    return [full](const Inputs& in) {
      const auto         r = nn::svd(in[0], true, full);
      const std::int64_t k = r.s.size(-1);
      const Tensor u1 = nn::slice_axis(r.u, -1, 0, k), v1 = nn::slice_axis(r.v, -1, 0, k);
      const Tensor w  = nn::arange(1, double(k) + 1, 1, DType::Float64);
      return cat_flat({r.s, nn::square(r.u), nn::square(r.v), nn::matmul(nn::mul(u1, w), v1, false, true)});
    };
  };
  for (nn::Shape shape : {nn::Shape{2, 5, 3}, nn::Shape{3, 3}, nn::Shape{3, 5}}) {
    const Tensor a = leaf(rnd(shape, 28));
    EXPECT_GRADCHECK(f(false), {a});
  }
  for (nn::Shape shape : {nn::Shape{4, 3}, nn::Shape{3, 4}}) EXPECT_GRADCHECK(f(true), {leaf(rnd(shape, 29))});
  const Tensor a = leaf(rnd({4, 3}, 30));
  EXPECT_GRADCHECK([](const Inputs& in) { return nn::svd(in[0], false).s; }, {a});
  EXPECT_GRADGRADCHECK([](const Inputs& in) { return nn::svd(in[0], false).s; }, {a});
}

// ---- eigh --------------------------------------------------------------------------------------------

TEST(NnLinalgEigh, Values) {
  for (std::int64_t n : {1, 2, 5, 12}) {
    const Tensor a = sym(rnd({3, n, n}, 31));
    const auto   r = nn::eigh(a);
    EXPECT_LT(max_abs_diff(nn::matmul(a, r.v), nn::mul(r.v, nn::expand_dims(r.e, -2))), 1e-12) << n;
    EXPECT_LT(max_abs_diff(nn::matmul(r.v, r.v, true, false), nn::eye(n, DType::Float64)), 1e-12);
    const auto e = r.e.to_vector<double>();
    for (std::size_t i = 1; i < e.size(); ++i)
      if (i % static_cast<std::size_t>(n)) EXPECT_LE(e[i - 1], e[i]);
  }
  const auto r = nn::eigh(nn::tensor({{2, 1}, {1, 2}}, DType::Float64));
  EXPECT_LT(max_abs_diff(r.e, nn::tensor({1, 3}, {2}, DType::Float64)), 1e-14);
  // Lower triangle only; repeated eigenvalues are fine for values.
  const Tensor lower = nn::tensor({{2, 99}, {1, 2}}, DType::Float64);
  EXPECT_LT(max_abs_diff(nn::eigvalsh(lower), nn::tensor({1, 3}, {2}, DType::Float64)), 1e-14);
  EXPECT_LT(max_abs_diff(nn::eigvalsh(nn::eye(4, DType::Float64)), nn::ones({4}, DType::Float64)), 1e-14);
  EXPECT_EQ(nn::self_adjoint_eig(nn::eye(3)).v.dtype(), DType::Float32);
}

TEST(NnLinalgEigh, Gradients) {
  const Tensor a = leaf(rnd({2, 4, 4}, 32));
  // Sign-invariant: eigenvalues, squared vectors and V diag(w) V^T. The forward reads the lower triangle.
  const auto f = [](const Inputs& in) {
    const auto   r = nn::eigh(in[0]);
    const Tensor w = nn::arange(1, 5, 1, DType::Float64);
    return cat_flat({r.e, nn::square(r.v), nn::matmul(nn::mul(r.v, w), r.v, false, true)});
  };
  EXPECT_GRADCHECK(f, {a});
  EXPECT_GRADCHECK([](const Inputs& in) { return nn::eigvalsh(in[0]); }, {a});
  EXPECT_GRADGRADCHECK([](const Inputs& in) { return nn::eigvalsh(sym(in[0])); }, {a});
}

// ---- einsum ------------------------------------------------------------------------------------------

TEST(NnEinsum, Patterns) {
  const Tensor a = rnd({3, 4}, 40), b = rnd({4, 5}, 41), v = rnd({4}, 42), w = rnd({4}, 43);
  EXPECT_LT(max_abs_diff(nn::einsum("ij,jk->ik", a, b), nn::matmul(a, b)), 1e-12);
  EXPECT_LT(max_abs_diff(nn::einsum("ij,jk", a, b), nn::matmul(a, b)), 1e-12);
  EXPECT_LT(max_abs_diff(nn::einsum("ij,kj->ik", a, nn::matrix_transpose(b)), nn::matmul(a, b)), 1e-12);
  EXPECT_LT(max_abs_diff(nn::einsum("ji,jk->ik", nn::matrix_transpose(a), b), nn::matmul(a, b)), 1e-12);
  EXPECT_LT(max_abs_diff(nn::einsum("ij,jk->ki", a, b), nn::matrix_transpose(nn::matmul(a, b))), 1e-12);
  const Tensor x = rnd({2, 3, 4}, 44), y = rnd({2, 4, 5}, 45);
  EXPECT_LT(max_abs_diff(nn::einsum("bij,bjk->bik", x, y), nn::matmul(x, y)), 1e-12);
  EXPECT_LT(max_abs_diff(nn::einsum("...ij,...jk->...ik", x, y), nn::matmul(x, y)), 1e-12);
  EXPECT_LT(max_abs_diff(nn::einsum("...ij,jk->...ik", x, b), nn::matmul(x, b)), 1e-12);
  const Tensor q = rnd({2, 3, 5, 4}, 46), k = rnd({2, 3, 6, 4}, 47);
  EXPECT_LT(max_abs_diff(nn::einsum("bhqd,bhkd->bhqk", q, k), nn::matmul(q, k, false, true)), 1e-12);
  EXPECT_NEAR(nn::einsum("i,i->", v, w).item(), nn::dot(v, w).item(), 1e-12);
  EXPECT_LT(max_abs_diff(nn::einsum("i,j->ij", v, w), nn::outer(v, w)), 1e-12);
  EXPECT_LT(max_abs_diff(nn::einsum("i,i->i", v, w), nn::mul(v, w)), 1e-12);
  EXPECT_LT(max_abs_diff(nn::einsum("ij,j->i", a, v), nn::squeeze(nn::matmul(a, nn::reshape(v, {4, 1})), {1})), 1e-12);
  EXPECT_LT(max_abs_diff(nn::einsum("bi,bi->b", a, a), nn::sum(nn::square(a), {1})), 1e-12);
  // one operand
  EXPECT_LT(max_abs_diff(nn::einsum("ij->ji", a), nn::matrix_transpose(a)), 0.0 + 1e-15);
  const Tensor s = rnd({4, 4}, 48);
  EXPECT_LT(max_abs_diff(nn::einsum("ii->i", s), nn::matrix_diag_part(s)), 1e-15);
  EXPECT_NEAR(nn::einsum("ii->", s).item(), nn::trace(s).item(), 1e-12);
  EXPECT_NEAR(nn::einsum("ii", s).item(), nn::trace(s).item(), 1e-12);
  EXPECT_NEAR(nn::einsum("ij->", a).item(), nn::sum(a).item(), 1e-12);
  EXPECT_LT(max_abs_diff(nn::einsum("ij->j", a), nn::sum(a, {0})), 1e-12);
  EXPECT_LT(max_abs_diff(nn::einsum("bij->bji", x), nn::swapaxes(x, 1, 2)), 1e-15);
  EXPECT_LT(max_abs_diff(nn::einsum("...ij->...ji", x), nn::swapaxes(x, 1, 2)), 1e-15);
  EXPECT_LT(max_abs_diff(nn::einsum("bii->b", rnd({2, 3, 3}, 49)), nn::trace(rnd({2, 3, 3}, 49))), 1e-12);
  // dynamic programming patterns: "sat,t->sa" and "sa,sat->st"
  const Tensor P = rnd({5, 2, 5}, 50), V = rnd({5}, 51), pi = rnd({5, 2}, 52);
  Tensor       ref = nn::sum(nn::mul(P, V), {2});
  EXPECT_LT(max_abs_diff(nn::einsum("sat,t->sa", P, V), ref), 1e-12);
  ref = nn::sum(nn::mul(nn::expand_dims(pi, 2), P), {1});
  EXPECT_LT(max_abs_diff(nn::einsum("sa,sat->st", pi, P), ref), 1e-12);
  // ellipsis broadcasting of batch axes
  const Tensor x1 = rnd({1, 3, 4}, 53), y3 = rnd({2, 1, 4, 5}, 54);
  EXPECT_LT(max_abs_diff(nn::einsum("...ij,...jk->...ik", x1, y3), nn::matmul(x1, y3)), 1e-12);
  // three operands, and a repeated index with a shared letter
  const Tensor c = rnd({5, 2}, 55);
  EXPECT_LT(max_abs_diff(nn::einsum("ij,jk,kl->il", {a, b, c}), nn::matmul(nn::matmul(a, b), c)), 1e-12);
  EXPECT_LT(max_abs_diff(nn::einsum("ii,i->i", s, v), nn::mul(nn::matrix_diag_part(s), v)), 1e-12);
  // float32 stays float32
  EXPECT_EQ(nn::einsum("ij,jk->ik", nn::cast(a, DType::Float32), nn::cast(b, DType::Float32)).dtype(), DType::Float32);
}

TEST(NnEinsum, Errors) {
  const Tensor a = nn::ones({2, 3}), b = nn::ones({4, 5});
  EXPECT_THROW(nn::einsum("ij,jk->ik", a, b), std::invalid_argument);
  EXPECT_THROW(nn::einsum("ijk->i", a), std::invalid_argument);
  EXPECT_THROW(nn::einsum("ij->ix", a), std::invalid_argument);
  EXPECT_THROW(nn::einsum("ij,jk->ik", {a}), std::invalid_argument);
  EXPECT_THROW(nn::einsum("i.j->i", a), std::invalid_argument);
}

TEST(NnEinsum, Gradients) {
  const Tensor a = leaf(rnd({2, 3}, 60)), b = leaf(rnd({3, 4}, 61)), v = leaf(rnd({3}, 62));
  const Tensor q = leaf(rnd({2, 2, 3, 2}, 63)), k = leaf(rnd({2, 2, 4, 2}, 64)), s = leaf(rnd({3, 3}, 65));
  const std::vector<std::pair<std::string, Inputs>> cases{
      {"ij,jk->ik", {a, b}},  {"ij,j->i", {a, v}},       {"bhqd,bhkd->bhqk", {q, k}}, {"i,i->", {v, v}},
      {"ij->ji", {a}},        {"ii->i", {s}},            {"ii->", {s}},               {"i,j->ij", {v, v}},
      {"...ij,...kj->...ik", {q, k}}, {"ij,ij->i", {a, a}}};
  for (const auto& [eq, inputs] : cases) {
    const auto f = [eq = eq](const Inputs& in) { return nn::square(nn::einsum(eq, in)); };
    const auto r1 = nn::gradcheck(f, inputs);
    EXPECT_TRUE(r1.ok) << eq << ": " << r1.message;
    const auto r2 = nn::gradgradcheck(f, inputs);
    EXPECT_TRUE(r2.ok) << eq << ": " << r2.message;
  }
}

TEST(NnLinalgBatch, LargeBatchesRunInParallel) {
  // Many small matrices go through the thread pool; results match a per-matrix loop.
  const Tensor a = spd({300}, 6, 70);
  const Tensor l = nn::cholesky(a);
  EXPECT_LT(max_abs_diff(nn::matmul(l, l, false, true), a), 1e-12);
  const Tensor l7 = nn::cholesky(nn::select(a, 0, 7));
  EXPECT_EQ(max_abs_diff(nn::select(l, 0, 7), l7), 0.0);
  const auto r = nn::eigh(a);
  EXPECT_LT(max_abs_diff(nn::matmul(a, r.v), nn::mul(r.v, nn::expand_dims(r.e, -2))), 1e-12);
  const auto s = nn::svd(nn::slice_axis(a, -1, 0, 4));
  EXPECT_EQ(s.s.shape(), (nn::Shape{300, 4}));
  // An error in one matrix of the batch propagates.
  Tensor bad = a.clone();
  {
    nn::NoGradGuard guard;
    bad.data_ptr<double>()[123 * 36] = -1;
  }
  EXPECT_THROW(nn::cholesky(bad), std::invalid_argument);
}

// ---- LAPACK backend ----------------------------------------------------------------------------------

namespace {

/// Sets a linalg backend for the scope of a test.
struct BackendScope {
  nn::LinalgBackend saved = nn::linalg_backend();
  explicit BackendScope(nn::LinalgBackend b) { nn::set_linalg_backend(b); }
  ~BackendScope() { nn::set_linalg_backend(saved); }
};

/// Runs f under the built-in and the LAPACK backend and returns both results.
template <typename F>
std::pair<Tensor, Tensor> both_backends(F&& f) {
  Tensor builtin, lapack;
  {
    BackendScope s(nn::LinalgBackend::Builtin);
    builtin = f();
  }
  {
    BackendScope s(nn::LinalgBackend::Lapack);
    lapack = f();
  }
  return {builtin, lapack};
}

/// Relative max difference of two results.
double rel_diff(const std::pair<Tensor, Tensor>& r) {
  const double scale = std::max(1.0, nn::max(nn::abs(nn::cast(r.first, DType::Float64))).item());
  return max_abs_diff(r.first, r.second) / scale;
}

}  // namespace

TEST(NnLinalgBackend, LapackMatchesBuiltin) {
  if (!nn::linalg_has_lapack()) GTEST_SKIP() << "built without LAPACK";
  // Sizes on both sides of the Auto thresholds; same values and sign conventions to rounding.
  for (std::int64_t n : {1, 3, 8, 40, 150}) {
    const Tensor a = rnd({2, n, n}, 80 + n), b = rnd({2, n, 3}, 81 + n), s = spd({2}, n, 82 + n);
    const Tensor well = nn::add(a, nn::mul(nn::eye(n, DType::Float64), 2.0 * std::sqrt(double(n))));
    EXPECT_LT(rel_diff(both_backends([&] { return nn::cholesky(s); })), 1e-12) << n;
    EXPECT_LT(rel_diff(both_backends([&] { return nn::eigh(s).e; })), 1e-12) << n;
    EXPECT_LT(rel_diff(both_backends([&] { return nn::eigh(sym(a)).v; })), 1e-9) << n;
    EXPECT_LT(rel_diff(both_backends([&] { return nn::eigvalsh(sym(a)); })), 1e-12) << n;
    EXPECT_LT(rel_diff(both_backends([&] { return nn::solve(well, b); })), 1e-11) << n;
    EXPECT_LT(rel_diff(both_backends([&] { return nn::solve(well, b, true); })), 1e-11) << n;
    EXPECT_LT(rel_diff(both_backends([&] { return nn::det(nn::div(well, std::sqrt(double(n)))); })), 1e-11) << n;
    EXPECT_LT(rel_diff(both_backends([&] { return nn::slogdet(a).log_abs_det; })), 1e-11) << n;
    EXPECT_LT(rel_diff(both_backends([&] { return nn::slogdet(a).sign; })), 0.5) << n;
    for (bool lower : {true, false})
      for (bool adj : {false, true})
        EXPECT_LT(rel_diff(both_backends([&] { return nn::triangular_solve(well, b, lower, adj); })), 1e-10) << n;
    const auto e = both_backends([&] { return cat_flat({nn::eig(well).e.real, nn::eig(well).e.imag}); });
    EXPECT_LT(rel_diff(e), 1e-9) << n;
  }
  for (nn::Shape shape : {nn::Shape{2, 6, 4}, nn::Shape{3, 5}, nn::Shape{70, 40}, nn::Shape{40, 70}, nn::Shape{64, 64}}) {
    const Tensor a = rnd(shape, 90);
    for (bool full : {false, true}) {
      const std::int64_t k = std::min(a.size(-2), a.size(-1));
      EXPECT_LT(rel_diff(both_backends([&] { return nn::svd(a, false).s; })), 1e-12);
      EXPECT_LT(rel_diff(both_backends([&] { return nn::svd(a, true, full).s; })), 1e-12);
      // Thin factors are unique up to the sign convention (distinct singular values).
      EXPECT_LT(rel_diff(both_backends([&] { return nn::slice_axis(nn::svd(a, true, full).u, -1, 0, k); })), 1e-9);
      EXPECT_LT(rel_diff(both_backends([&] { return nn::slice_axis(nn::svd(a, true, full).v, -1, 0, k); })), 1e-9);
      EXPECT_LT(rel_diff(both_backends([&] { return nn::slice_axis(nn::qr(a, full).q, -1, 0, k); })), 1e-11);
      EXPECT_LT(rel_diff(both_backends([&] { return nn::qr(a, full).r; })), 1e-11);
    }
    // Full factors are still orthonormal under LAPACK.
    BackendScope scope(nn::LinalgBackend::Lapack);
    const auto r = nn::svd(a, true, true);
    EXPECT_LT(max_abs_diff(nn::matmul(r.u, r.u, true, false), nn::eye(r.u.size(-1), DType::Float64)), 1e-12);
    EXPECT_LT(max_abs_diff(nn::matmul(r.v, r.v, true, false), nn::eye(r.v.size(-1), DType::Float64)), 1e-12);
    const auto q = nn::qr(a, true);
    EXPECT_LT(max_abs_diff(nn::matmul(q.q, q.r), a), 1e-12);
  }
  // Errors are the same.
  BackendScope scope(nn::LinalgBackend::Lapack);
  EXPECT_THROW(nn::cholesky(nn::tensor({{1, 2}, {2, 1}}, DType::Float64)), std::invalid_argument);
  EXPECT_THROW(nn::solve(nn::tensor({{1, 2}, {2, 4}}, DType::Float64), nn::ones({2, 1}, DType::Float64)),
               std::invalid_argument);
  EXPECT_EQ(nn::slogdet(nn::tensor({{1, 2}, {2, 4}}, DType::Float64)).sign.item(), 0.0);
}

TEST(NnLinalgBackend, GradientsUnderLapack) {
  // Same gradient code; check it on the LAPACK kernels too.
  BackendScope scope(nn::LinalgBackend::Lapack);
  const Tensor a = leaf(rnd({2, 5, 3}, 91));
  EXPECT_GRADCHECK([](const Inputs& in) {
    const auto r = nn::svd(in[0]);
    return cat_flat({r.s, nn::square(r.u), nn::square(r.v)});
  }, {a});
  EXPECT_GRADCHECK([](const Inputs& in) {
    const auto qr = nn::qr(in[0]);
    return cat_flat({qr.q, qr.r});
  }, {a});
  const Tensor s = leaf(rnd({2, 4, 4}, 92));
  EXPECT_GRADCHECK([](const Inputs& in) {
    const auto r = nn::eigh(in[0]);
    return cat_flat({r.e, nn::square(r.v)});
  }, {s});
  const Tensor p = leaf(spd({2}, 4, 93));
  EXPECT_GRADCHECK([](const Inputs& in) { return nn::cholesky(sym(in[0])); }, {p});
  EXPECT_GRADCHECK([](const Inputs& in) { return nn::slogdet(in[0]).log_abs_det; }, {p});
  const Tensor b = leaf(rnd({2, 4, 2}, 94));
  EXPECT_GRADCHECK([](const Inputs& in) { return nn::solve(in[0], in[1]); }, Inputs({p, b}));
  EXPECT_GRADCHECK([](const Inputs& in) { return nn::triangular_solve(in[0], in[1], false, true); }, Inputs({p, b}));
}

TEST(NnLinalgBackend, SignConventions) {
  for (auto backend : {nn::LinalgBackend::Builtin, nn::LinalgBackend::Lapack}) {
    BackendScope scope(backend);
    // QR of an upper triangular matrix: no reflections, Q = I, R = A (as LAPACK and TF/Eigen).
    const Tensor up = nn::tensor({{2, 1, 3}, {0, -1, 4}, {0, 0, 5}}, DType::Float64);
    const auto   qr = nn::qr(up);
    EXPECT_LT(max_abs_diff(qr.q, nn::eye(3, DType::Float64)), 1e-15);
    EXPECT_LT(max_abs_diff(qr.r, up), 1e-15);
    // QR: R's diagonal has the opposite sign of the reflected column's leading entry.
    const auto q2 = nn::qr(nn::tensor({{3, 1}, {4, 2}}, DType::Float64));
    EXPECT_NEAR(q2.r.to_vector<double>()[0], -5.0, 1e-14);
    // SVD / eigh: largest-magnitude entry of each U column / eigenvector is positive.
    const Tensor a  = rnd({7, 5}, 95);
    const auto   sv = nn::svd(a);
    const Tensor um = nn::max(nn::abs(sv.u), {-2}), up_ = nn::max(sv.u, {-2});
    EXPECT_LT(max_abs_diff(um, up_), 1e-15);
    const auto   eh = nn::eigh(sym(rnd({6, 6}, 96)));
    EXPECT_LT(max_abs_diff(nn::max(nn::abs(eh.v), {-2}), nn::max(eh.v, {-2})), 1e-15);
  }
}

// ---- lstsq / pinv ------------------------------------------------------------------------------------

TEST(NnLinalgLstsq, Values) {
  // Overdetermined: the normal equations' solution.
  const Tensor a = rnd({2, 7, 3}, 100), b = rnd({2, 7, 2}, 101);
  const Tensor x = nn::lstsq(a, b);
  EXPECT_EQ(x.shape(), (nn::Shape{2, 3, 2}));
  const Tensor ne = nn::solve(nn::matmul(a, a, true, false), nn::matmul(a, b, true, false));
  EXPECT_LT(max_abs_diff(x, ne), 1e-12);
  EXPECT_LT(max_abs_diff(nn::lstsq(a, b, 0.0, false), ne), 1e-12);
  // L2 regularised: (A^T A + l I)^-1 A^T b.
  const Tensor xr = nn::lstsq(a, b, 0.5);
  const Tensor rr = nn::solve(nn::add(nn::matmul(a, a, true, false), nn::mul(nn::eye(3, DType::Float64), 0.5)),
                              nn::matmul(a, b, true, false));
  EXPECT_LT(max_abs_diff(xr, rr), 1e-12);
  // Underdetermined: the minimum-norm solution A^T (A A^T)^-1 b, which solves A x = b exactly.
  const Tensor w = rnd({3, 6}, 102), c = rnd({3, 1}, 103);
  const Tensor xu = nn::lstsq(w, c);
  EXPECT_LT(max_abs_diff(nn::matmul(w, xu), c), 1e-12);
  EXPECT_LT(max_abs_diff(nn::lstsq(w, c, 0.0, false), xu), 1e-12);
  const Tensor xur = nn::lstsq(w, c, 0.3);
  EXPECT_LT(max_abs_diff(xur, nn::matmul(w, nn::solve(nn::add(nn::matmul(w, w, false, true),
                                                               nn::mul(nn::eye(3, DType::Float64), 0.3)),
                                                      c),
                                         true, false)),
            1e-12);
  // Rank deficient (duplicate column): fast=false gives the minimum-norm solution.
  const Tensor col = rnd({5, 1}, 104);
  const Tensor rd  = nn::concat({col, col, rnd({5, 1}, 105)}, -1);
  const Tensor y   = rnd({5, 1}, 106);
  const Tensor xm  = nn::lstsq(rd, y, 0.0, false);
  const auto   xv  = xm.to_vector<double>();
  EXPECT_NEAR(xv[0], xv[1], 1e-12);  // weight split evenly between the duplicates
  EXPECT_LT(max_abs_diff(nn::matmul(nn::matmul(rd, rd, true, false), xm), nn::matmul(rd, y, true, false)), 1e-12);
  EXPECT_EQ(nn::matrix_solve_ls(nn::cast(a, DType::Float32), b).dtype(), DType::Float64);
  EXPECT_THROW(nn::lstsq(a, rnd({2, 6, 2}, 107)), std::invalid_argument);
}

TEST(NnLinalgLstsq, Gradients) {
  for (nn::Shape shape : {nn::Shape{6, 3}, nn::Shape{3, 5}})
    for (double l2 : {0.0, 0.7}) {
      const Tensor a = leaf(rnd(shape, 110)), b = leaf(rnd({shape[0], 2}, 111));
      const auto   f = [l2](const Inputs& in) { return nn::lstsq(in[0], in[1], l2); };
      EXPECT_GRADCHECK(f, Inputs({a, b}));
      EXPECT_GRADGRADCHECK(f, Inputs({a, b}));
    }
  const Tensor a = leaf(rnd({2, 5, 3}, 112)), b = leaf(rnd({2, 5, 1}, 113));
  EXPECT_GRADCHECK([](const Inputs& in) { return nn::lstsq(in[0], in[1], 0.0, false); }, Inputs({a, b}));
}

TEST(NnLinalgPinv, Values) {
  for (nn::Shape shape : {nn::Shape{2, 5, 3}, nn::Shape{3, 6}, nn::Shape{4, 4}}) {
    const Tensor a = rnd(shape, 120);
    const Tensor p = nn::pinv(a);
    EXPECT_EQ(p.size(-2), a.size(-1));
    EXPECT_EQ(p.size(-1), a.size(-2));
    // Moore-Penrose conditions.
    EXPECT_LT(max_abs_diff(nn::matmul(nn::matmul(a, p), a), a), 1e-12);
    EXPECT_LT(max_abs_diff(nn::matmul(nn::matmul(p, a), p), p), 1e-12);
    const Tensor ap = nn::matmul(a, p), pa = nn::matmul(p, a);
    EXPECT_LT(max_abs_diff(ap, nn::matrix_transpose(ap)), 1e-12);
    EXPECT_LT(max_abs_diff(pa, nn::matrix_transpose(pa)), 1e-12);
  }
  EXPECT_LT(max_abs_diff(nn::pinv(rnd({4, 4}, 121)), nn::inverse(rnd({4, 4}, 121))), 1e-10);
  // Rank one: pinv(x y^T) = y x^T / (|x|^2 |y|^2).
  const Tensor x = rnd({4, 1}, 122), y = rnd({3, 1}, 123);
  const Tensor r1 = nn::matmul(x, y, false, true);
  const double nx = nn::sum(nn::square(x)).item(), ny = nn::sum(nn::square(y)).item();
  EXPECT_LT(max_abs_diff(nn::pinv(r1), nn::div(nn::matmul(y, x, false, true), nx * ny)), 1e-12);
  // rcond drops small singular values.
  const Tensor d = nn::tensor({{2, 0}, {0, 1e-3}}, DType::Float64);
  EXPECT_LT(max_abs_diff(nn::pinv(d), nn::tensor({{0.5, 0}, {0, 1e3}}, DType::Float64)), 1e-9);
  EXPECT_LT(max_abs_diff(nn::pinv(d, 1e-2), nn::tensor({{0.5, 0}, {0, 0}}, DType::Float64)), 1e-15);
  EXPECT_EQ(nn::pinv(nn::zeros({3, 2})).shape(), (nn::Shape{2, 3}));
  EXPECT_LT(nn::max(nn::abs(nn::pinv(nn::zeros({3, 2})))).item(), 1e-30);
}

TEST(NnLinalgPinv, Gradients) {
  for (nn::Shape shape : {nn::Shape{2, 4, 3}, nn::Shape{3, 4}}) {
    const Tensor a = leaf(rnd(shape, 124));
    EXPECT_GRADCHECK([](const Inputs& in) { return nn::pinv(in[0]); }, {a});
  }
}

// ---- eig ---------------------------------------------------------------------------------------------

namespace {

/// max |A v - v diag(e)| for complex eigenpairs of a real A.
double eig_residual(const Tensor& a, const nn::EigResult& r) {
  const Tensor av_re = nn::matmul(a, r.v.real), av_im = nn::matmul(a, r.v.imag);
  const Tensor er = nn::expand_dims(r.e.real, -2), ei = nn::expand_dims(r.e.imag, -2);
  const Tensor ve_re = nn::sub(nn::mul(r.v.real, er), nn::mul(r.v.imag, ei));
  const Tensor ve_im = nn::add(nn::mul(r.v.real, ei), nn::mul(r.v.imag, er));
  return std::max(max_abs_diff(av_re, ve_re), max_abs_diff(av_im, ve_im));
}

}  // namespace

TEST(NnLinalgEig, Values) {
  for (auto backend : {nn::LinalgBackend::Builtin, nn::LinalgBackend::Auto}) {
    BackendScope scope(backend);
    // Rotation: eigenvalues -i, +i (equal modulus: ordered by imaginary part).
    const auto rot = nn::eig(nn::tensor({{0, -1}, {1, 0}}, DType::Float64));
    EXPECT_LT(max_abs_diff(rot.e.real, nn::zeros({2}, DType::Float64)), 1e-15);
    EXPECT_LT(max_abs_diff(rot.e.imag, nn::tensor({-1, 1}, {2}, DType::Float64)), 1e-15);
    EXPECT_LT(eig_residual(nn::tensor({{0, -1}, {1, 0}}, DType::Float64), rot), 1e-15);
    // Triangular: the diagonal, sorted by modulus.
    const Tensor tri = nn::tensor({{3, 1, 2}, {0, -1, 5}, {0, 0, 2}}, DType::Float64);
    const auto   rt  = nn::eig(tri);
    EXPECT_LT(max_abs_diff(rt.e.real, nn::tensor({-1, 2, 3}, {3}, DType::Float64)), 1e-14);
    EXPECT_LT(max_abs_diff(rt.e.imag, nn::zeros({3}, DType::Float64)), 1e-15);
    EXPECT_LT(eig_residual(tri, rt), 1e-14);
    // Random matrices (complex pairs), batched; unit-norm vectors with the largest entry real positive.
    for (std::int64_t n : {1, 2, 5, 9, 30}) {
      const Tensor a = rnd({3, n, n}, 130 + n);
      const auto   r = nn::eig(a);
      EXPECT_LT(eig_residual(a, r), 1e-11) << n;
      const Tensor norms = nn::sum(nn::add(nn::square(r.v.real), nn::square(r.v.imag)), {-2});
      EXPECT_LT(max_abs_diff(norms, nn::ones({3, n}, DType::Float64)), 1e-12);
      const auto   mod = nn::complex_abs(r.v);
      const Tensor at  = nn::argmax(mod, -2);
      const auto   idx = at.to_vector<std::int64_t>();
      const auto   im  = r.v.imag.to_vector<double>();
      const auto   re  = r.v.real.to_vector<double>();
      for (std::int64_t p = 0; p < 3; ++p)
        for (std::int64_t j = 0; j < n; ++j) {
          const std::int64_t o = p * n * n + idx[p * n + j] * n + j;
          EXPECT_NEAR(im[o], 0.0, 1e-14);
          EXPECT_GT(re[o], 0.0);
        }
      // Sorted by modulus; sum = trace; eigvals agrees.
      const auto mods = nn::complex_abs(r.e).to_vector<double>();
      for (std::int64_t p = 0; p < 3; ++p)
        for (std::int64_t j = 1; j < n; ++j) EXPECT_LE(mods[p * n + j - 1], mods[p * n + j] + 1e-12);
      EXPECT_LT(max_abs_diff(nn::sum(r.e.real, {-1}), nn::trace(a)), 1e-11);
      EXPECT_LT(max_abs_diff(nn::sum(r.e.imag, {-1}), nn::zeros({3}, DType::Float64)), 1e-11);
      EXPECT_LT(max_abs_diff(nn::eigvals(a).real, r.e.real), 1e-12);
    }
    // Symmetric input: real eigenvalues matching eigh.
    const Tensor s = sym(rnd({6, 6}, 140));
    const auto   rs = nn::eig(s);
    EXPECT_LT(max_abs_diff(rs.e.imag, nn::zeros({6}, DType::Float64)), 1e-14);
    const auto ev = rs.e.real.to_vector<double>(), eh = nn::eigvalsh(s).to_vector<double>();
    std::vector<double> sorted_ev = ev;
    std::sort(sorted_ev.begin(), sorted_ev.end());
    for (std::size_t i = 0; i < eh.size(); ++i) EXPECT_NEAR(sorted_ev[i], eh[i], 1e-12);
    // Zero and float32 matrices.
    const auto z = nn::eig(nn::zeros({3, 3}, DType::Float64));
    EXPECT_LT(max_abs_diff(z.e.real, nn::zeros({3}, DType::Float64)), 0.0 + 1e-300);
    EXPECT_LT(eig_residual(nn::zeros({3, 3}, DType::Float64), z), 1e-300);
    const auto f32 = nn::eig(rnd({4, 4}, 141, DType::Float32));
    EXPECT_EQ(f32.e.dtype(), DType::Float32);
    EXPECT_EQ(f32.v.dtype(), DType::Float32);
    EXPECT_THROW(nn::eig(rnd({3, 4}, 142)), std::invalid_argument);
  }
}

TEST(NnLinalgEig, BuiltinMatchesLapack) {
  if (!nn::linalg_has_lapack()) GTEST_SKIP() << "built without LAPACK";
  for (std::int64_t n : {2, 4, 7, 20}) {
    const Tensor a = rnd({2, n, n}, 150 + n);
    EXPECT_LT(rel_diff(both_backends([&] { return cat_flat({nn::eig(a).e.real, nn::eig(a).e.imag}); })), 1e-11) << n;
    EXPECT_LT(rel_diff(both_backends([&] { return nn::eig(a).v.real; })), 1e-9) << n;
    EXPECT_LT(rel_diff(both_backends([&] { return nn::eig(a).v.imag; })), 1e-9) << n;
  }
}

TEST(NnLinalgEig, Gradients) {
  // A real matrix with one complex pair and well separated moduli (the order stays fixed).
  const Tensor base = nn::tensor({{4.0, 0.3, -0.2, 0.1}, {0.2, 1.0, -2.0, 0.3}, {0.1, 1.5, 1.2, -0.2}, {0.3, 0.1, 0.2, -6.0}},
                                 DType::Float64);
  const Tensor a = leaf(nn::stack({base, nn::add(base, nn::mul(rnd({4, 4}, 160), 0.05))}, 0));
  // Eigenvalues and phase-invariant functions of the eigenvectors.
  const auto f = [](const Inputs& in) {
    const auto r = nn::eig(in[0]);
    return cat_flat({r.e.real, r.e.imag, nn::complex_abs_squared(r.v)});
  };
  EXPECT_GRADCHECK(f, {a});
  const auto g = [](const Inputs& in) {
    const auto e = nn::eigvals(in[0]);
    return cat_flat({e.real, e.imag});
  };
  EXPECT_GRADCHECK(g, {a});
  EXPECT_GRADGRADCHECK(g, {a});
  // A symmetric input gives eigh's eigenvalue gradient (for an order-invariant loss).
  const Tensor s  = leaf(sym(rnd({4, 4}, 161)));
  const auto   cube_sum = [](const Tensor& e) { return nn::sum(nn::mul(e, nn::mul(e, e))); };
  const Tensor ge = nn::grad(cube_sum(nn::eigvals(s).real), {s})[0];
  const Tensor gh = nn::grad(cube_sum(nn::eigvalsh(sym(s))), {s})[0];
  EXPECT_LT(max_abs_diff(ge, gh), 1e-10);
}

TEST(NnLinalgSolve, SharedFactorsMatchIndependentBatches) {
  const auto previous = nn::linalg_backend();
  for (auto backend : {nn::LinalgBackend::Builtin, nn::LinalgBackend::Lapack}) {
    if (backend == nn::LinalgBackend::Lapack && !nn::linalg_has_lapack()) continue;
    nn::set_linalg_backend(backend);
    for (auto dt : {DType::Float32, DType::Float64}) {
      // Non-symmetric, offset/strided input; explicit batch copies take the
      // independent-factorization path and provide the reference.
      auto storage = nn::add(nn::mul(rnd({4, 4}, 813, dt), 0.1), nn::eye(4, dt));
      auto a = nn::matrix_transpose(storage);
      auto b = rnd({2, 3, 4, 2}, 814, dt);
      auto expanded = nn::broadcast_to(a, {2, 3, 4, 4}).clone();
      for (bool adjoint : {false, true}) {
        auto actual = nn::solve(a, b, adjoint);
        auto expected = nn::solve(expanded, b, adjoint);
        EXPECT_LT(max_abs_diff(actual, expected), dt == DType::Float64 ? 1e-12 : 1e-5);
        EXPECT_LT(max_abs_diff(nn::matmul(a, actual, adjoint, false), b), dt == DType::Float64 ? 1e-12 : 1e-5);
      }
    }
    EXPECT_THROW(nn::solve(nn::zeros({4, 4}), nn::ones({3, 4, 2})), std::invalid_argument);
  }
  nn::set_linalg_backend(previous);
}

TEST(NnLinalgSolve, SharedFactorsHigherDerivatives) {
  auto a = leaf(nn::add(nn::mul(rnd({2, 2}, 821), 0.1), nn::eye(2, DType::Float64)));
  auto b = leaf(rnd({3, 2, 1}, 822));
  for (bool adjoint : {false, true}) {
    nn::TensorFunction f = [adjoint](const Inputs& in) { return nn::solve(in[0], in[1], adjoint); };
    EXPECT_GRADCHECK(f, (Inputs{a, b}));
    EXPECT_GRADGRADCHECK(f, (Inputs{a, b}));
  }
}

TEST(NnLinalgSolve, SharedFactorsParallelAndEmpty) {
  const int previous = nn::get_num_threads();
  auto a = nn::add(nn::mul(rnd({32, 32}, 835), 0.01), nn::eye(32, DType::Float64));
  auto b = rnd({16, 32, 3}, 836);
  nn::set_num_threads(1);
  auto expected = nn::solve(a, b);
  nn::set_num_threads(4);
  EXPECT_LT(max_abs_diff(nn::solve(a, b), expected), 1e-12);
  EXPECT_EQ(nn::solve(nn::zeros({0, 0}), nn::zeros({3, 0, 2})).shape(), (nn::Shape{3, 0, 2}));
  nn::set_num_threads(previous);
}

TEST(NnMatmulCopies, PartialBatchBroadcastAndTransposeGradients) {
  for (auto dtype : {DType::Float32, DType::Float64}) {
    for (bool ta : {false, true}) for (bool tb : {false, true}) {
      auto a = rnd(ta ? nn::Shape{2, 1, 3, 2} : nn::Shape{2, 1, 2, 3}, 910, dtype);
      auto b = rnd(tb ? nn::Shape{1, 4, 5, 3} : nn::Shape{1, 4, 3, 5}, 911, dtype);
      a = nn::matrix_transpose(nn::matrix_transpose(a).clone());
      b = nn::matrix_transpose(nn::matrix_transpose(b).clone());
      auto sa = a.shape(), sb = b.shape(); sa[1] = 4; sb[0] = 2;
      auto expected = nn::matmul(nn::broadcast_to(a, sa).clone(), nn::broadcast_to(b, sb).clone(), ta, tb);
      EXPECT_LT(max_abs_diff(nn::matmul(a, b, ta, tb), expected), dtype == DType::Float64 ? 1e-12 : 1e-5);
    }
  }
  auto a = leaf(rnd({2, 1, 2, 3}, 912));
  auto b = leaf(rnd({1, 3, 3, 2}, 913));
  nn::TensorFunction f = [](const Inputs& values) { return nn::matmul(values[0], values[1]); };
  EXPECT_GRADCHECK(f, (Inputs{a, b}));
  EXPECT_GRADGRADCHECK(f, (Inputs{a, b}));
}

TEST(NnEinsum, ImplicitOutputOrderAndMultiOperandLabelLifetimes) {
  auto a = nn::reshape(nn::arange(0, 12, 1, DType::Float64), {3, 4});
  auto b = nn::reshape(nn::arange(0, 8, 1, DType::Float64), {4, 2});
  auto c = nn::reshape(nn::arange(0, 10, 1, DType::Float64), {2, 5});
  EXPECT_EQ(nn::einsum("jI,Ia", a, b).to_vector<double>(),
            nn::transpose(nn::matmul(a, b)).to_vector<double>());
  auto result = nn::einsum("ab,bc,cd", std::vector<Tensor>{a, b, c});
  EXPECT_EQ(result.shape(), (nn::Shape{3, 5}));
  EXPECT_EQ(result.to_vector<double>(), nn::matmul(nn::matmul(a, b), c).to_vector<double>());
  EXPECT_THROW(nn::einsum("jI,Ia->jj", a, b), std::invalid_argument);
}
