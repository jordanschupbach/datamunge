#include <gtest/gtest.h>

#include <datamunge/fda/hierarchical_bspline.hpp>
#include <datamunge/fda/pspline.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>
#include <vector>

namespace fda = datamunge::fda;
using fda::BSplineBasis;
using fda::HierarchicalBSpline;
using fda::TensorBSpline;
using fda::TruncatedHierarchicalBSpline;

namespace {

using Points = std::vector<std::vector<double>>;

std::vector<double> random_values(std::size_t n, unsigned seed, double lo = -2.0, double hi = 2.0) {
    std::mt19937                           rng(seed);
    std::uniform_real_distribution<double> u(lo, hi);
    std::vector<double>                    v(n);
    for (auto& x : v) x = u(rng);
    return v;
}

Points grid_points(int n, int dims, double lo = 0.0, double hi = 1.0) {
    Points           out;
    std::vector<int> idx(static_cast<std::size_t>(dims), 0);
    while (true) {
        std::vector<double> p;
        for (int k = 0; k < dims; ++k) p.push_back(lo + (hi - lo) * idx[static_cast<std::size_t>(k)] / (n - 1));
        out.push_back(p);
        int k = dims - 1;
        while (k >= 0 && ++idx[static_cast<std::size_t>(k)] == n) idx[static_cast<std::size_t>(k--)] = 0;
        if (k < 0) break;
    }
    return out;
}

TensorBSpline make_2d_base() {
    const auto b = BSplineBasis::open_uniform(2, 4);
    return TensorBSpline({b, b}, random_values(16, 11));
}

double surface(const std::vector<double>& p) { return std::sin(3.0 * p[0]) + std::cos(2.0 * p[1]) * p[0]; }

} // namespace

// ── Knot refinement ─────────────────────────────────────────────────────────

TEST(BSplineRefinement, DyadicRefineDoublesSpansAndKeepsDomain) {
    const auto coarse = BSplineBasis::open_uniform(2, 5); // 3 spans
    const auto r      = fda::dyadic_refine(coarse);
    EXPECT_EQ(r.basis.basis_size(), 8u); // 6 spans + degree 2
    EXPECT_EQ(r.basis.degree(), 2u);
    EXPECT_EQ(r.basis.domain(), coarse.domain());
    EXPECT_EQ(r.matrix.rows(), 8u);
    EXPECT_EQ(r.matrix.cols(), 5u);
}

TEST(BSplineRefinement, ReproducesRandomSplinesForManyDegreesAndKnotPatterns) {
    const std::vector<std::vector<double>> interiors = {{}, {0.3}, {0.2, 0.5, 0.55}, {0.4, 0.4}, {0.1, 0.7, 0.7, 0.7}};
    for (std::size_t p = 0; p <= 4; ++p) {
        for (std::size_t s = 0; s < interiors.size(); ++s) {
            std::vector<double> knots(p + 1, 0.0);
            std::size_t         run = 1;
            bool                ok  = true;
            for (std::size_t i = 0; i < interiors[s].size(); ++i) {
                run = (i > 0 && interiors[s][i] == interiors[s][i - 1]) ? run + 1 : 1;
                ok  = ok && run <= p + 1;
                knots.push_back(interiors[s][i]);
            }
            if (!ok) continue;
            knots.insert(knots.end(), p + 1, 1.0);
            const BSplineBasis coarse(p, knots);
            const TensorBSpline f(coarse, random_values(coarse.basis_size(), 7 + static_cast<unsigned>(p * 10 + s)));

            for (const auto& inserted : std::vector<std::vector<double>>{{0.15}, {0.6, 0.6}, {0.05, 0.33, 0.9}}) {
                bool valid = true;
                for (double x : inserted) {
                    std::size_t mult = std::count(knots.begin(), knots.end(), x) +
                                       std::count(inserted.begin(), inserted.end(), x);
                    valid = valid && mult <= p + 1;
                }
                if (!valid) continue;
                const auto          r = fda::refine_basis(coarse, inserted);
                const TensorBSpline g(r.basis, r.apply(f.coefficients()));
                for (int i = 0; i <= 200; ++i) {
                    const double x = i / 200.0;
                    EXPECT_NEAR(g.evaluate(x), f.evaluate(x), 1e-12) << "p=" << p << " s=" << s << " x=" << x;
                }
            }
        }
    }
}

TEST(BSplineRefinement, OperatorRowsArePartitionsOfUnity) {
    const auto r = fda::dyadic_refine(BSplineBasis(3, {0, 0, 0, 0, 0.2, 0.7, 1, 1, 1, 1}));
    const auto ones = r.apply(std::vector<double>(6, 1.0));
    for (double v : ones) EXPECT_NEAR(v, 1.0, 1e-14);
}

TEST(BSplineRefinement, ManyRepeatedRefinementsStayExactAndFast) {
    BSplineBasis              basis(2, {0, 0, 0, 1, 2, 3, 3, 3});
    std::vector<double>       coef = {0.0, 1.5, 3.0, 2.5, 1.0};
    const TensorBSpline       original(basis, coef);
    const auto                start = std::chrono::steady_clock::now();
    for (int i = 0; i < 12; ++i) {
        const auto r = fda::dyadic_refine(basis);
        coef         = r.apply(coef);
        basis        = r.basis;
    }
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    EXPECT_EQ(basis.basis_size(), 3u * 4096u + 2u);
    EXPECT_LT(elapsed, 5.0);
    const TensorBSpline refined(basis, coef);
    for (double t : {0.0, 0.4, 0.9, 1.5, 2.1, 2.7, 3.0}) EXPECT_NEAR(refined.evaluate(t), original.evaluate(t), 1e-10);
}

TEST(BSplineRefinement, RejectsInvalidInput) {
    const auto clamped = BSplineBasis::open_uniform(2, 4);
    EXPECT_THROW(fda::refine_basis(clamped, {0.0}), std::invalid_argument);
    EXPECT_THROW(fda::refine_basis(clamped, {1.5}), std::invalid_argument);
    EXPECT_THROW(fda::refine_basis(clamped, {NAN}), std::invalid_argument);
    EXPECT_THROW(fda::refine_basis(clamped, {0.3, 0.3, 0.3, 0.3}), std::invalid_argument);
    const BSplineBasis unclamped(2, {0, 1, 2, 3, 4, 5, 6});
    EXPECT_THROW(fda::dyadic_refine(unclamped), std::invalid_argument);
    const auto r = fda::dyadic_refine(clamped);
    EXPECT_THROW(r.apply({1.0}), std::invalid_argument);
}

// ── TensorBSpline ───────────────────────────────────────────────────────────

TEST(TensorBSpline, LinearCurveReproducesIdentity) {
    const TensorBSpline f(BSplineBasis(1, {0, 0, 1, 1}), {0.0, 1.0});
    for (double t : {0.0, 0.25, 0.5, 0.75, 1.0}) EXPECT_NEAR(f.evaluate(t), t, 1e-14);
}

TEST(TensorBSpline, BilinearSurface) {
    const auto          b = BSplineBasis::open_uniform(1, 2);
    const TensorBSpline f({b, b}, {0, 1, 1, 2}); // x + y
    for (double x : {0.0, 0.3, 1.0})
        for (double y : {0.0, 0.6, 1.0}) EXPECT_NEAR(f.evaluate({x, y}), x + y, 1e-14);
}

TEST(TensorBSpline, Constant3dVolumeAndShape) {
    const auto          b = BSplineBasis::open_uniform(2, 4);
    const TensorBSpline f({b, b, b}, std::vector<double>(64, 3.5));
    EXPECT_EQ(f.dimensions(), 3u);
    EXPECT_EQ(f.shape(), (std::vector<int>{4, 4, 4}));
    EXPECT_NEAR(f.evaluate({0.1, 0.5, 0.9}), 3.5, 1e-13);
}

TEST(TensorBSpline, ClampsOutsideTheDomain) {
    const TensorBSpline f(BSplineBasis::open_uniform(1, 2), {1.0, 3.0});
    EXPECT_DOUBLE_EQ(f.evaluate(-5.0), 1.0);
    EXPECT_DOUBLE_EQ(f.evaluate(5.0), 3.0);
}

TEST(TensorBSpline, RejectsBadConstruction) {
    const auto b = BSplineBasis::open_uniform(1, 2);
    EXPECT_THROW(TensorBSpline({b, b}, {1, 2, 3}), std::invalid_argument);
    EXPECT_THROW(TensorBSpline(std::vector<BSplineBasis>{}, {}), std::invalid_argument);
    const TensorBSpline f({b, b}, {0, 1, 1, 2});
    EXPECT_THROW(f.evaluate({0.5}), std::invalid_argument);
    EXPECT_THROW(f.evaluate(0.5), std::invalid_argument);
    EXPECT_THROW(f.evaluate({0.5, NAN}), std::invalid_argument);
}

TEST(TensorBSpline, CoefficientAccessByMultiIndex) {
    const auto    b = BSplineBasis::open_uniform(1, 3);
    TensorBSpline f({b, b}, std::vector<double>(9, 0.0));
    f.set_coefficient({2, 1}, 4.0);
    EXPECT_DOUBLE_EQ(f.coefficient({2, 1}), 4.0);
    EXPECT_DOUBLE_EQ(f.coefficients()[2 * 3 + 1], 4.0);
    EXPECT_THROW(f.coefficient({3, 0}), std::out_of_range);
    EXPECT_THROW(f.coefficient({-1, 0}), std::out_of_range);
    EXPECT_THROW(f.coefficient({0}), std::invalid_argument);
}

TEST(TensorBSpline, GridMatchesPointwiseIn2dAnd3d) {
    const auto          b2 = BSplineBasis::open_uniform(2, 5);
    const TensorBSpline f2({b2, BSplineBasis::open_uniform(3, 6)}, random_values(30, 3));
    const std::vector<std::vector<double>> axes2 = {{0.0, 0.3, 1.0}, {0.1, 0.5, 0.7, 0.9}};
    const auto                             g2    = f2.evaluate_grid(axes2);
    ASSERT_EQ(g2.size(), 12u);
    std::size_t i = 0;
    for (double x : axes2[0])
        for (double y : axes2[1]) EXPECT_DOUBLE_EQ(g2[i++], f2.evaluate({x, y}));

    const TensorBSpline f3({b2, b2, b2}, random_values(125, 4));
    const auto          g3 = f3.evaluate_grid({{0.2, 0.8}, {0.5}, {0.1, 0.9}});
    ASSERT_EQ(g3.size(), 4u);
    EXPECT_DOUBLE_EQ(g3[3], f3.evaluate({0.8, 0.5, 0.9}));
    EXPECT_THROW(f3.evaluate_grid({{0.2}, {0.5}}), std::invalid_argument);
    EXPECT_THROW(f3.evaluate_grid({{0.2}, {}, {0.1}}), std::invalid_argument);
}

TEST(TensorBSpline, DyadicRefineReproducesSurfaceAndVolume) {
    const auto          b = BSplineBasis::open_uniform(2, 5);
    const TensorBSpline f2({b, BSplineBasis::open_uniform(3, 4)}, random_values(20, 5));
    const auto          g2 = f2.dyadic_refine();
    EXPECT_EQ(g2.shape(), (std::vector<int>{8, 5}));
    for (const auto& p : grid_points(9, 2)) EXPECT_NEAR(g2.evaluate(p), f2.evaluate(p), 1e-12);

    const TensorBSpline f3({b, b, b}, random_values(125, 6));
    const auto          g3 = f3.dyadic_refine();
    for (const auto& p : grid_points(5, 3)) EXPECT_NEAR(g3.evaluate(p), f3.evaluate(p), 1e-12);
}

// ── Penalties and fits ──────────────────────────────────────────────────────

TEST(TensorDifferencePenalty, OneDimensionalMatchesPSplinePenalty) {
    for (int q = 1; q <= 3; ++q) {
        const auto sparse = fda::tensor_difference_penalty({7}, {1.0}, q);
        const auto dense  = fda::pspline_penalty_matrix(7, static_cast<std::size_t>(q));
        for (std::size_t i = 0; i < 7; ++i)
            for (std::size_t j = 0; j < 7; ++j) {
                double v = 0.0;
                for (std::size_t k = 0; k < sparse.nnz(); ++k)
                    if (sparse.row_indices()[k] == i && sparse.col_indices()[k] == j) v = sparse.values()[k];
                EXPECT_NEAR(v, dense(i, j), 1e-12) << "q=" << q;
            }
    }
}

TEST(TensorDifferencePenalty, TwoDimensionalIsKroneckerSum) {
    const auto p = fda::tensor_difference_penalty({3, 4}, {2.0, 0.5}, 1);
    const auto a = fda::pspline_penalty_matrix(3, 1);
    const auto b = fda::pspline_penalty_matrix(4, 1);
    for (std::size_t r = 0; r < 12; ++r)
        for (std::size_t c = 0; c < 12; ++c) {
            const std::size_t ri = r / 4, rj = r % 4, ci = c / 4, cj = c % 4;
            const double expected = 2.0 * a(ri, ci) * (rj == cj) + 0.5 * b(rj, cj) * (ri == ci);
            double       v        = 0.0;
            for (std::size_t k = 0; k < p.nnz(); ++k)
                if (p.row_indices()[k] == r && p.col_indices()[k] == c) v = p.values()[k];
            EXPECT_NEAR(v, expected, 1e-12);
        }
    const auto ridge = fda::tensor_difference_penalty({3}, {1.0}, 0);
    EXPECT_EQ(ridge.nnz(), 3u);
    EXPECT_THROW(fda::tensor_difference_penalty({2}, {1.0}, 2), std::invalid_argument);
    EXPECT_THROW(fda::tensor_difference_penalty({3}, {-1.0}, 1), std::invalid_argument);
    EXPECT_THROW(fda::tensor_difference_penalty({3, 3}, {1.0}, 1), std::invalid_argument);
}

TEST(FitTensorBSpline, RecoversKnownNetExactly) {
    const auto          b = BSplineBasis::open_uniform(2, 5);
    const TensorBSpline truth({b, b}, random_values(25, 21));
    const auto          pts = grid_points(9, 2);
    const auto          fit = fda::fit_tensor_bspline({b, b}, pts, truth.evaluate_points(pts));
    for (std::size_t i = 0; i < 25; ++i) EXPECT_NEAR(fit.coefficients()[i], truth.coefficients()[i], 1e-9);
}

TEST(FitTensorBSpline, ConjugateGradientAgreesWithCholesky) {
    const auto b   = BSplineBasis::open_uniform(3, 7);
    const auto pts = grid_points(15, 2);
    std::vector<double> y;
    for (const auto& p : pts) y.push_back(surface(p));
    fda::SplineSolveOptions direct{fda::SplineSolver::Cholesky};
    fda::SplineSolveOptions cg{fda::SplineSolver::ConjugateGradient};
    const auto              a = fda::fit_tensor_bspline({b, b}, pts, y, direct);
    const auto              c = fda::fit_tensor_bspline({b, b}, pts, y, cg);
    for (std::size_t i = 0; i < a.coefficient_count(); ++i) EXPECT_NEAR(a.coefficients()[i], c.coefficients()[i], 1e-7);
}

TEST(FitTensorBSpline, SmoothsNoiseTowardsTruth) {
    const auto                       b = BSplineBasis::open_uniform(3, 6);
    std::mt19937                     rng(5);
    std::normal_distribution<double> noise(0.0, 0.1);
    Points                           pts = grid_points(25, 2);
    std::vector<double>              y;
    for (const auto& p : pts) y.push_back(surface(p) + noise(rng));
    const auto fit = fda::fit_tensor_bspline({b, b}, pts, y);
    double     err_fit = 0.0, err_data = 0.0;
    for (std::size_t i = 0; i < pts.size(); ++i) {
        err_fit += std::pow(fit.evaluate(pts[i]) - surface(pts[i]), 2);
        err_data += std::pow(y[i] - surface(pts[i]), 2);
    }
    EXPECT_LT(err_fit, 0.25 * err_data);
}

TEST(FitTensorBSpline, RejectsBadInputAndSingularDesigns) {
    const auto b = BSplineBasis::open_uniform(1, 2);
    EXPECT_THROW(fda::fit_tensor_bspline({b}, {{0.0}, {1.0}}, {1.0}), std::invalid_argument);
    EXPECT_THROW(fda::fit_tensor_bspline({b}, {{0.0, 1.0}}, {1.0}), std::invalid_argument);
    EXPECT_THROW(fda::fit_tensor_bspline({b}, {}, {}), std::invalid_argument);
    EXPECT_THROW(fda::fit_tensor_bspline({}, {{0.0}}, {1.0}), std::invalid_argument);
    EXPECT_THROW(fda::fit_tensor_bspline({BSplineBasis::open_uniform(3, 10)}, {{0.0}, {1.0}}, {0.0, 1.0}),
                 std::runtime_error);
}

TEST(FitTensorPSpline, ZeroLambdaMatchesUnpenalized) {
    const auto b   = BSplineBasis::open_uniform(2, 5);
    const auto pts = grid_points(10, 2);
    std::vector<double> y;
    for (const auto& p : pts) y.push_back(surface(p));
    const auto a = fda::fit_tensor_bspline({b, b}, pts, y);
    const auto c = fda::fit_tensor_pspline({b, b}, pts, y, {0.0, 0.0});
    for (std::size_t i = 0; i < a.coefficient_count(); ++i) EXPECT_NEAR(a.coefficients()[i], c.coefficients()[i], 1e-10);
}

TEST(FitTensorPSpline, OneDimensionalAgreesWithFitPSpline) {
    const auto          basis = BSplineBasis::open_uniform(3, 12);
    std::vector<double> x, y;
    Points              pts;
    for (int i = 0; i < 60; ++i) {
        x.push_back(i / 59.0);
        y.push_back(std::sin(6.0 * x.back()) + 0.1 * std::cos(37.0 * i));
        pts.push_back({x.back()});
    }
    for (int q : {1, 2, 3}) {
        const auto ref = fda::fit_pspline(x, y, basis, {2.5, static_cast<std::size_t>(q)});
        const auto mine = fda::fit_tensor_pspline({basis}, pts, y, {2.5}, q);
        for (std::size_t i = 0; i < 12; ++i) EXPECT_NEAR(mine.coefficients()[i], ref.coefficients[i], 1e-9) << q;
    }
}

TEST(FitTensorPSpline, PenaltySmoothsAndIsAnisotropic) {
    const auto b   = BSplineBasis::open_uniform(3, 10);
    const auto pts = grid_points(20, 2);
    std::vector<double> y;
    for (const auto& p : pts) y.push_back(std::sin(12.0 * p[0]) + std::sin(12.0 * p[1]));
    auto roughness = [&](const TensorBSpline& f, int axis) {
        double r = 0.0;
        for (int i = 0; i < 10; ++i)
            for (int j = 1; j + 1 < 10; ++j) {
                std::vector<int> m{i, j}, l{i, j - 1}, h{i, j + 1};
                if (axis == 0) m = {j, i}, l = {j - 1, i}, h = {j + 1, i};
                r += std::pow(f.coefficient(h) - 2 * f.coefficient(m) + f.coefficient(l), 2);
            }
        return r;
    };
    const auto plain = fda::fit_tensor_bspline({b, b}, pts, y);
    const auto both  = fda::fit_tensor_pspline({b, b}, pts, y, {10.0, 10.0});
    const auto only0 = fda::fit_tensor_pspline({b, b}, pts, y, {100.0, 0.0});
    EXPECT_LT(roughness(both, 0), roughness(plain, 0));
    EXPECT_LT(roughness(both, 1), roughness(plain, 1));
    EXPECT_LT(roughness(only0, 0), 0.1 * roughness(only0, 1));
}

TEST(FitTensorPSpline, UnderdeterminedAndHigherDimensionalFitsAreWellPosed) {
    const auto b    = BSplineBasis::open_uniform(3, 8);
    const auto fit  = fda::fit_tensor_pspline({b, b}, {{0.1, 0.1}, {0.9, 0.2}, {0.5, 0.8}, {0.3, 0.6}}, {1, 2, 3, 4},
                                              {1.0, 1.0});
    EXPECT_TRUE(std::isfinite(fit.evaluate({0.5, 0.5})));
    const auto c3   = BSplineBasis::open_uniform(2, 4);
    const auto pts3 = grid_points(5, 3);
    std::vector<double> y3;
    for (std::size_t i = 0; i < pts3.size(); ++i) y3.push_back(2.5);
    const auto f3 = fda::fit_tensor_pspline({c3, c3, c3}, pts3, y3, {1.0, 1.0, 1.0}, 1);
    EXPECT_NEAR(f3.evaluate({0.3, 0.4, 0.5}), 2.5, 1e-10); // constants are in the order-1 null space
    EXPECT_THROW(fda::fit_tensor_pspline({b}, {{0.0}}, {1.0}, {1.0, 1.0}), std::invalid_argument);
    EXPECT_THROW(fda::fit_tensor_pspline({b}, {{0.0}}, {1.0}, {1.0}, -1), std::invalid_argument);
}

// ── HierarchicalBSpline (overlay) ───────────────────────────────────────────

TEST(HierarchicalBSpline, RootCellMatchesItsPatch) {
    const auto          b = BSplineBasis::open_uniform(2, 4);
    const TensorBSpline patch({b, b}, random_values(16, 31));
    HierarchicalBSpline h({0.0, 0.0}, {2.0, 4.0});
    h.set_spline(h.root(), patch);
    EXPECT_NEAR(h.evaluate({1.0, 1.0}), patch.evaluate({0.5, 0.25}), 1e-14);
    EXPECT_TRUE(h.has_spline(0));
    EXPECT_EQ(h.spline(0).coefficient_count(), 16u);
}

TEST(HierarchicalBSpline, RefinementAddsLocalContributionInLocalCoordinates) {
    const auto          c = BSplineBasis::open_uniform(0, 1);
    const auto          l = BSplineBasis::open_uniform(1, 2);
    HierarchicalBSpline h({0.0}, {1.0});
    h.set_spline(0, TensorBSpline(c, {1.0}));
    const int child = h.refine(0, 1, TensorBSpline(l, {0.0, 1.0}));
    const int grand = h.refine(child, 0, TensorBSpline(l, {0.0, 10.0}));
    EXPECT_EQ(h.depth(grand), 2);
    EXPECT_EQ(h.parent(grand), child);
    EXPECT_DOUBLE_EQ(h.evaluate({0.25}), 1.0);
    EXPECT_NEAR(h.evaluate({0.875}), 1.0 + 0.75, 1e-14);                // child only
    EXPECT_NEAR(h.evaluate({0.625}), 1.0 + 0.25 + 10.0 * 0.5, 1e-14);   // child + grandchild
    EXPECT_EQ(h.cell_min(grand), (std::vector<double>{0.5}));
    EXPECT_EQ(h.cell_max(grand), (std::vector<double>{0.75}));
}

TEST(HierarchicalBSpline, RejectsInvalidDomainsPatchesAndIds) {
    EXPECT_THROW(HierarchicalBSpline({0.0}, {0.0}), std::invalid_argument);
    EXPECT_THROW(HierarchicalBSpline({0.0, 0.0}, {1.0}), std::invalid_argument);
    EXPECT_THROW(HierarchicalBSpline({}, {}), std::invalid_argument);
    HierarchicalBSpline h({0.0, 0.0}, {1.0, 1.0});
    EXPECT_THROW(h.set_spline(0, TensorBSpline(BSplineBasis::open_uniform(1, 2), {0, 1})), std::invalid_argument);
    EXPECT_THROW(h.child(0, 4), std::out_of_range);
    EXPECT_THROW(h.depth(9), std::out_of_range);
    EXPECT_THROW(h.spline(0), std::invalid_argument);
    EXPECT_THROW(h.evaluate({2.0, 0.5}), std::invalid_argument);
    EXPECT_EQ(h.locate({2.0, 0.5}), -1);
}

TEST(HierarchicalBSpline, LocateRefineAndSubdivide) {
    HierarchicalBSpline h({0.0, 0.0}, {1.0, 1.0});
    const int           c = h.refine(0, 3); // upper-right
    EXPECT_EQ(h.cell_count(), 2);
    EXPECT_EQ(h.child(0, 0), -1);
    EXPECT_EQ(h.locate({0.9, 0.9}), c);
    EXPECT_EQ(h.locate({0.1, 0.9}), 0);
    EXPECT_EQ(h.locate({1.0, 1.0}), c);
    const auto kids = h.subdivide(0);
    EXPECT_EQ(kids.size(), 4u);
    EXPECT_EQ(kids[3], c);
    EXPECT_EQ(h.cell_count(), 5);
}

TEST(HierarchicalBSpline, RefineRegionTouchesOnlyOverlappingCellsIn2dAnd3d) {
    HierarchicalBSpline h({0.0, 0.0}, {1.0, 1.0});
    int                 calls = 0;
    h.refine_region({0.0, 0.0}, {0.2, 0.2}, 3, [&](const auto&, const auto&) {
        ++calls;
        return std::nullopt;
    });
    EXPECT_EQ(h.cell_count(), 7); // 1 + 1 + 1 + 4 cells overlapping [0, 0.2]^2
    EXPECT_EQ(calls, 6);
    EXPECT_EQ(h.depth(h.locate({0.05, 0.05})), 3);
    EXPECT_EQ(h.depth(h.locate({0.9, 0.9})), 0);

    HierarchicalBSpline v({0.0, 0.0, 0.0}, {1.0, 1.0, 1.0});
    v.refine_region({0.4, 0.4, 0.4}, {0.6, 0.6, 0.6}, 1);
    EXPECT_EQ(v.cell_count(), 9);
    EXPECT_THROW(v.refine_region({0.5, 0.5, 0.5}, {0.4, 0.6, 0.6}, 1), std::invalid_argument);
}

TEST(HierarchicalBSpline, GridMatchesPointwise) {
    const auto          b = BSplineBasis::open_uniform(2, 3);
    HierarchicalBSpline h({0.0, 0.0}, {1.0, 1.0});
    h.set_spline(0, TensorBSpline({b, b}, random_values(9, 41)));
    h.refine(0, 0, TensorBSpline({b, b}, random_values(9, 42)));
    const std::vector<std::vector<double>> axes = {{0.1, 0.4, 0.8}, {0.2, 0.45}};
    const auto                             g    = h.evaluate_grid(axes);
    std::size_t                            i    = 0;
    for (double x : axes[0])
        for (double y : axes[1]) EXPECT_DOUBLE_EQ(g[i++], h.evaluate({x, y}));
}

TEST(HierarchicalBSpline, FitCellRecoversFunctionAndExplainsResiduals) {
    const auto          b   = BSplineBasis::open_uniform(2, 4);
    HierarchicalBSpline h({0.0, 0.0}, {1.0, 1.0});
    const TensorBSpline truth({b, b}, random_values(16, 51));
    const auto          pts = grid_points(8, 2);
    h.fit_cell(0, {b, b}, pts, truth.evaluate_points(pts));
    for (const auto& p : pts) EXPECT_NEAR(h.evaluate(p), truth.evaluate(p), 1e-9);

    // A child fit with the raw targets only explains the residual left by the root.
    const int child = h.refine(0, 0);
    Points    local;
    std::vector<double> targets;
    for (const auto& p : grid_points(6, 2, 0.0, 0.5)) {
        local.push_back(p);
        targets.push_back(truth.evaluate(p) + 0.5 * p[0]);
    }
    h.fit_cell(child, {b, b}, local, targets);
    for (std::size_t i = 0; i < local.size(); ++i) EXPECT_NEAR(h.evaluate(local[i]), targets[i], 1e-9);
    EXPECT_NEAR(h.evaluate({0.9, 0.9}), truth.evaluate({0.9, 0.9}), 1e-9);

    EXPECT_THROW(h.fit_cell(child, {b, b}, local, targets), std::invalid_argument);
    EXPECT_THROW(h.fit_cell(h.refine(0, 3), {b, b}, {{0.1, 0.1}}, {1.0}), std::invalid_argument);
    EXPECT_THROW(h.fit_cell(h.refine(0, 1), {b, b}, {}, {}), std::invalid_argument);
}

TEST(HierarchicalBSpline, PenalizedCellFits) {
    const auto          b   = BSplineBasis::open_uniform(2, 4);
    const auto          pts = grid_points(7, 2);
    std::vector<double> y;
    for (const auto& p : pts) y.push_back(surface(p));
    HierarchicalBSpline a({0.0, 0.0}, {1.0, 1.0}), c({0.0, 0.0}, {1.0, 1.0});
    a.fit_cell(0, {b, b}, pts, y);
    c.fit_cell_penalized(0, {b, b}, pts, y, {0.0, 0.0});
    for (const auto& p : pts) EXPECT_NEAR(a.evaluate(p), c.evaluate(p), 1e-10);

    HierarchicalBSpline u({0.0, 0.0, 0.0}, {1.0, 1.0, 1.0});
    u.fit_cell_penalized(0, {b, b, b}, {{0.2, 0.3, 0.4}, {0.6, 0.5, 0.9}}, {1.0, 2.0}, {1.0, 1.0, 1.0}, 0);
    EXPECT_TRUE(std::isfinite(u.evaluate({0.5, 0.5, 0.5})));
    EXPECT_THROW(u.fit_cell_penalized(0, {b, b, b}, {{0.2, 0.3, 0.4}}, {1.0}, {1.0, 1.0, 1.0}), std::invalid_argument);
}

TEST(HierarchicalBSpline, DrillsDeepQuicklyAndStaysLocal) {
    HierarchicalBSpline h({0.0, 0.0}, {1.0, 1.0});
    const auto          b = BSplineBasis::open_uniform(2, 4);
    h.fit_cell_penalized(0, {b, b}, grid_points(5, 2), std::vector<double>(25, 10.0), {1e-4, 1e-4});
    int        cell  = 0;
    const auto start = std::chrono::steady_clock::now();
    for (int depth = 0; depth < 10; ++depth) {
        cell = h.subdivide(cell)[0];
        Points              pts;
        std::vector<double> y;
        const auto          lo = h.cell_min(cell), hi = h.cell_max(cell);
        for (double fu : {0.2, 0.4, 0.6, 0.8})
            for (double fv : {0.2, 0.4, 0.6, 0.8}) {
                pts.push_back({lo[0] + fu * (hi[0] - lo[0]), lo[1] + fv * (hi[1] - lo[1])});
                y.push_back(10.0 + depth);
            }
        h.fit_cell_penalized(cell, {b, b}, pts, y, {1e-4, 1e-4});
    }
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    EXPECT_LT(elapsed, 10.0);
    EXPECT_NEAR(h.evaluate({0.9, 0.9}), 10.0, 1e-6);
}

// ── TruncatedHierarchicalBSpline ────────────────────────────────────────────

TEST(TruncatedHierarchicalBSpline, ReproducesBaseBeforeAndAfterRefinement) {
    const auto                   base = make_2d_base();
    TruncatedHierarchicalBSpline thb(base);
    EXPECT_EQ(thb.level_count(), 1);
    EXPECT_EQ(thb.basis_size(), 16u);
    for (const auto& p : grid_points(7, 2)) EXPECT_NEAR(thb.evaluate(p), base.evaluate(p), 1e-13);
    thb.refine_region({0.0, 0.0}, {0.5, 0.5}, 2);
    thb.refine(thb.locate({0.9, 0.1}), 2);
    EXPECT_EQ(thb.level_count(), 3);
    for (const auto& p : grid_points(13, 2)) EXPECT_NEAR(thb.evaluate(p), base.evaluate(p), 1e-12);
}

TEST(TruncatedHierarchicalBSpline, TruncatedBasisIsAPartitionOfUnity) {
    const auto                   b = BSplineBasis::open_uniform(3, 6);
    TruncatedHierarchicalBSpline thb(TensorBSpline({b, b}, std::vector<double>(36, 0.0)));
    thb.refine_region({0.0, 0.0}, {0.6, 0.45}, 1);
    thb.refine_region({0.1, 0.1}, {0.3, 0.35}, 3);
    thb.refine_region({0.7, 0.7}, {0.9, 1.0}, 2);
    std::vector<std::vector<std::vector<int>>> active;
    for (int l = 0; l < thb.level_count(); ++l) active.push_back(thb.active_indices(l));
    for (const auto& p : grid_points(17, 2)) {
        double sum = 0.0;
        for (int l = 0; l < thb.level_count(); ++l)
            for (const auto& idx : active[static_cast<std::size_t>(l)]) {
                const double v = thb.truncated_basis(l, idx, p);
                EXPECT_GE(v, -1e-12);
                sum += v;
            }
        EXPECT_NEAR(sum, 1.0, 1e-12) << p[0] << "," << p[1];
    }
}

TEST(TruncatedHierarchicalBSpline, CoarseEditAfterRefinementStaysContinuous) {
    // Editing an active level-0 function whose support crosses into the refined region
    // must change the finer level too (truncation), so the function stays continuous.
    const auto                   b = BSplineBasis::open_uniform(2, 6);
    TruncatedHierarchicalBSpline thb(TensorBSpline(b, std::vector<double>(6, 0.0)));
    thb.refine_region({0.0}, {0.5}, 1);
    std::vector<int> straddling;
    for (const auto& idx : thb.active_indices(0)) {
        const double lo = thb.support_min(0, idx)[0], hi = thb.support_max(0, idx)[0];
        if (lo < 0.5 && hi > 0.5) straddling = idx;
    }
    ASSERT_FALSE(straddling.empty());
    thb.set_coefficient(0, straddling, 5.0);
    const double left  = thb.evaluate({0.5 - 1e-9});
    const double right = thb.evaluate({0.5 + 1e-9});
    EXPECT_GT(std::abs(right), 0.1);
    EXPECT_NEAR(left, right, 1e-6);
    // The hierarchy equals sum of coefficients times truncated basis functions.
    for (double x = 0.0; x <= 1.0; x += 0.05) {
        double sum = 0.0;
        for (int l = 0; l < thb.level_count(); ++l)
            for (const auto& idx : thb.active_indices(l)) sum += thb.coefficient(l, idx) * thb.truncated_basis(l, idx, {x});
        EXPECT_NEAR(thb.evaluate({x}), sum, 1e-12) << x;
    }
}

TEST(TruncatedHierarchicalBSpline, EditingActiveFineControlIsLocal) {
    const auto                   base = make_2d_base();
    TruncatedHierarchicalBSpline thb(base);
    thb.refine_region({0.0, 0.0}, {0.5, 0.5}, 2);
    std::vector<int> chosen;
    for (const auto& idx : thb.active_indices(2)) {
        const auto lo = thb.support_min(2, idx), hi = thb.support_max(2, idx);
        if (lo[0] <= 0.1 && 0.1 <= hi[0] && lo[1] <= 0.1 && 0.1 <= hi[1]) {
            chosen = idx;
            break;
        }
    }
    ASSERT_FALSE(chosen.empty());
    const double before = thb.evaluate({0.1, 0.1});
    thb.set_coefficient(2, chosen, thb.coefficient(2, chosen) + 100.0);
    EXPECT_NE(thb.evaluate({0.1, 0.1}), before);
    EXPECT_NEAR(thb.evaluate({0.9, 0.9}), base.evaluate({0.9, 0.9}), 1e-12);
}

TEST(TruncatedHierarchicalBSpline, FullyRefiningDeactivatesTheCoarserLevel) {
    TruncatedHierarchicalBSpline thb(make_2d_base());
    EXPECT_EQ(thb.active_indices(0).size(), 16u);
    thb.refine_region({0.0, 0.0}, {1.0, 1.0}, 1);
    EXPECT_TRUE(thb.active_indices(0).empty());
    EXPECT_EQ(thb.active_indices(1).size(), 36u);
    EXPECT_FALSE(thb.is_active(0, {0, 0}));
    EXPECT_THROW(thb.set_coefficient(0, {0, 0}, 42.0), std::invalid_argument);
}

TEST(TruncatedHierarchicalBSpline, RejectsInvalidUse) {
    TruncatedHierarchicalBSpline thb(make_2d_base());
    EXPECT_EQ(thb.locate({1.5, 0.5}), -1);
    EXPECT_THROW(thb.evaluate({1.5, 0.5}), std::invalid_argument);
    EXPECT_THROW(thb.level(1), std::out_of_range);
    EXPECT_THROW(thb.coefficient(0, {4, 0}), std::out_of_range);
    EXPECT_THROW(thb.is_active(0, {0}), std::invalid_argument);
    EXPECT_THROW(thb.refine(0, 4), std::out_of_range);
    const BSplineBasis unclamped(2, {0, 1, 2, 3, 4, 5, 6});
    EXPECT_THROW(TruncatedHierarchicalBSpline(TensorBSpline(unclamped, {0, 0, 0, 0})), std::invalid_argument);
}

TEST(TruncatedHierarchicalBSpline, GridMatchesPointwiseAndWorksIn3d) {
    TruncatedHierarchicalBSpline thb(make_2d_base());
    thb.refine_region({0.2, 0.2}, {0.6, 0.6}, 2);
    const std::vector<std::vector<double>> axes = {{0.1, 0.3, 0.7}, {0.25, 0.5}};
    const auto                             g    = thb.evaluate_grid(axes);
    std::size_t                            i    = 0;
    for (double x : axes[0])
        for (double y : axes[1]) EXPECT_DOUBLE_EQ(g[i++], thb.evaluate({x, y}));

    const auto          b = BSplineBasis::open_uniform(2, 3);
    const TensorBSpline base3({b, b, b}, random_values(27, 61));
    TruncatedHierarchicalBSpline t3(base3);
    t3.refine_region({0.0, 0.0, 0.0}, {0.5, 0.5, 0.5}, 2);
    for (const auto& p : grid_points(5, 3)) EXPECT_NEAR(t3.evaluate(p), base3.evaluate(p), 1e-12);
    const auto active = t3.active_indices(2);
    ASSERT_FALSE(active.empty());
    t3.set_coefficient(2, active.front(), 50.0);
    EXPECT_NEAR(t3.evaluate({0.9, 0.9, 0.9}), base3.evaluate({0.9, 0.9, 0.9}), 1e-12);
}

TEST(TruncatedHierarchicalBSpline, LevelZeroFitMatchesDirectTensorFit) {
    const auto          b   = BSplineBasis::open_uniform(2, 5);
    const auto          pts = grid_points(9, 2);
    std::vector<double> y;
    for (const auto& p : pts) y.push_back(surface(p));
    TruncatedHierarchicalBSpline thb(TensorBSpline({b, b}, std::vector<double>(25, 0.0)));
    thb.fit_level(0, pts, y);
    const auto direct = fda::fit_tensor_bspline({b, b}, pts, y);
    for (std::size_t i = 0; i < 25; ++i) EXPECT_NEAR(thb.level(0).coefficients()[i], direct.coefficients()[i], 1e-9);
}

TEST(TruncatedHierarchicalBSpline, GlobalFitRecoversAHierarchicalFunctionExactly) {
    const auto                   b = BSplineBasis::open_uniform(2, 4);
    TruncatedHierarchicalBSpline truth(TensorBSpline({b, b}, std::vector<double>(16, 0.0)));
    truth.refine_region({0.0, 0.0}, {0.5, 0.75}, 1);
    truth.refine_region({0.0, 0.0}, {0.25, 0.25}, 2);
    unsigned seed = 70;
    for (int l = 0; l < truth.level_count(); ++l)
        for (const auto& idx : truth.active_indices(l)) truth.set_coefficient(l, idx, random_values(1, seed++)[0]);

    TruncatedHierarchicalBSpline fit(TensorBSpline({b, b}, std::vector<double>(16, 0.0)));
    fit.refine_region({0.0, 0.0}, {0.5, 0.75}, 1);
    fit.refine_region({0.0, 0.0}, {0.25, 0.25}, 2);
    const auto pts = grid_points(33, 2);
    fit.fit(pts, truth.evaluate_points(pts));
    EXPECT_EQ(fit.basis_size(), truth.basis_size());
    for (int l = 0; l < truth.level_count(); ++l)
        for (const auto& idx : truth.active_indices(l)) EXPECT_NEAR(fit.coefficient(l, idx), truth.coefficient(l, idx), 1e-8);
    for (const auto& p : grid_points(21, 2)) EXPECT_NEAR(fit.evaluate(p), truth.evaluate(p), 1e-8);

    fda::SplineSolveOptions cg{fda::SplineSolver::ConjugateGradient};
    TruncatedHierarchicalBSpline fit_cg(fit);
    fit_cg.fit(pts, truth.evaluate_points(pts), cg);
    for (const auto& p : grid_points(11, 2)) EXPECT_NEAR(fit_cg.evaluate(p), truth.evaluate(p), 1e-6);
}

TEST(TruncatedHierarchicalBSpline, RefinedLevelFitIsLocalAndReducesResidual) {
    const auto          b = BSplineBasis::open_uniform(2, 4);
    auto                target = [](const std::vector<double>& p) {
        return std::exp(-60.0 * ((p[0] - 0.2) * (p[0] - 0.2) + (p[1] - 0.2) * (p[1] - 0.2)));
    };
    const auto          pts = grid_points(41, 2);
    std::vector<double> y;
    for (const auto& p : pts) y.push_back(target(p));
    TruncatedHierarchicalBSpline thb(TensorBSpline({b, b}, std::vector<double>(16, 0.0)));
    thb.fit_level(0, pts, y);
    auto sse = [&](double lo, double hi) {
        double s = 0.0;
        for (std::size_t i = 0; i < pts.size(); ++i)
            if (pts[i][0] >= lo && pts[i][0] <= hi && pts[i][1] >= lo && pts[i][1] <= hi)
                s += std::pow(thb.evaluate(pts[i]) - y[i], 2);
        return s;
    };
    const double coarse_err = sse(0.0, 0.5);
    thb.refine_region({0.0, 0.0}, {0.5, 0.5}, 1);
    thb.refine_region({0.0, 0.0}, {0.2, 0.2}, 2);
    const double far_before = thb.evaluate({0.95, 0.95});
    thb.fit_level(1, pts, y);
    const double err1 = sse(0.0, 0.5);
    thb.fit_level(2, pts, y);
    const double err2 = sse(0.0, 0.5);
    EXPECT_LT(err1, coarse_err);
    EXPECT_LE(err2, err1 + 1e-12);
    EXPECT_NEAR(thb.evaluate({0.95, 0.95}), far_before, 1e-12);

    // Least-squares optimality: the residual is orthogonal to every active truncated
    // level-2 function over the points the level-2 fit uses.
    for (const auto& idx : thb.active_indices(2)) {
        double dot = 0.0;
        for (std::size_t i = 0; i < pts.size(); ++i)
            if (thb.depth(thb.locate(pts[i])) >= 2)
                dot += thb.truncated_basis(2, idx, pts[i]) * (y[i] - thb.evaluate(pts[i]));
        EXPECT_NEAR(dot, 0.0, 1e-10);
    }
    // A joint fit of all levels can only do better.
    TruncatedHierarchicalBSpline joint(thb);
    joint.fit(pts, y);
    double joint_err = 0.0, seq_err = 0.0;
    for (std::size_t i = 0; i < pts.size(); ++i) {
        joint_err += std::pow(joint.evaluate(pts[i]) - y[i], 2);
        seq_err += std::pow(thb.evaluate(pts[i]) - y[i], 2);
    }
    EXPECT_LE(joint_err, seq_err + 1e-10);

    EXPECT_THROW(thb.fit_level(0, pts, std::vector<double>(3, 0.0)), std::invalid_argument);
    EXPECT_THROW(thb.fit_level(0, {}, {}), std::invalid_argument);
    EXPECT_THROW(thb.fit_level(1, {{0.9, 0.9}}, {1.0}), std::invalid_argument); // no point at level 1
    TruncatedHierarchicalBSpline full(TensorBSpline({b, b}, std::vector<double>(16, 0.0)));
    full.refine_region({0.0, 0.0}, {1.0, 1.0}, 1);
    EXPECT_THROW(full.fit_level(0, pts, y), std::invalid_argument); // no active functions
}

TEST(TruncatedHierarchicalBSpline, PenalizedFits) {
    const auto          b   = BSplineBasis::open_uniform(2, 5);
    const auto          pts = grid_points(12, 2);
    std::vector<double> y;
    for (const auto& p : pts) y.push_back(surface(p));
    TruncatedHierarchicalBSpline a(TensorBSpline({b, b}, std::vector<double>(25, 0.0)));
    a.refine_region({0.0, 0.0}, {0.5, 0.5}, 1);
    TruncatedHierarchicalBSpline c(a);
    a.fit_level(0, pts, y);
    c.fit_level_penalized(0, pts, y, {0.0, 0.0});
    for (const auto& p : pts) EXPECT_NEAR(a.evaluate(p), c.evaluate(p), 1e-9);

    // Solver override agrees, penalized fit stays local.
    TruncatedHierarchicalBSpline d(c), e(c);
    d.fit_level_penalized(1, pts, y, {0.1, 0.1}, 2, {fda::SplineSolver::Cholesky});
    e.fit_level_penalized(1, pts, y, {0.1, 0.1}, 2, {fda::SplineSolver::ConjugateGradient});
    for (const auto& p : pts) EXPECT_NEAR(d.evaluate(p), e.evaluate(p), 1e-7);
    EXPECT_NEAR(d.evaluate({0.95, 0.95}), c.evaluate({0.95, 0.95}), 1e-12);

    // Underdetermined but penalized: still well posed.
    TruncatedHierarchicalBSpline u(TensorBSpline({b, b}, std::vector<double>(25, 0.0)));
    u.refine_region({0.0, 0.0}, {0.5, 0.5}, 1);
    u.fit_level_penalized(1, {{0.1, 0.1}, {0.3, 0.2}}, {1.0, 2.0}, {1.0, 1.0}, 0);
    u.fit_penalized({{0.1, 0.1}, {0.9, 0.2}, {0.6, 0.6}}, {1.0, 2.0, 3.0}, {1.0, 1.0}, 0);
    EXPECT_TRUE(std::isfinite(u.evaluate({0.5, 0.5})));
    // The all-level penalty leaves constants unpenalized: constant data stay exact.
    TruncatedHierarchicalBSpline k(TensorBSpline({b, b}, std::vector<double>(25, 0.0)));
    k.refine_region({0.0, 0.0}, {0.5, 0.5}, 1);
    k.fit_penalized(pts, std::vector<double>(pts.size(), 3.0), {100.0, 100.0}, 1);
    for (const auto& p : grid_points(9, 2)) EXPECT_NEAR(k.evaluate(p), 3.0, 1e-10);
    EXPECT_THROW(u.fit_level_penalized(1, {{0.1, 0.1}}, {1.0}, {1.0}), std::invalid_argument);
    EXPECT_THROW(u.fit_penalized({{0.1, 0.1}}, {1.0}, {1.0, -1.0}), std::invalid_argument);
}

TEST(TruncatedHierarchicalBSpline, RefinesToHighDepthQuicklyAndStaysCorrect) {
    const auto                   base = make_2d_base();
    TruncatedHierarchicalBSpline thb(base);
    const std::vector<double>    target{0.02, 0.02};
    const auto                   start = std::chrono::steady_clock::now();
    for (int level = 1; level <= 7; ++level) {
        const int  cell = thb.locate(target);
        const auto lo = thb.cell_min(cell), hi = thb.cell_max(cell);
        thb.refine_region(lo, hi, level);
        ASSERT_FALSE(thb.active_indices(level).empty()) << level;
        Points              pts;
        std::vector<double> y;
        for (double fu : {0.2, 0.4, 0.6, 0.8})
            for (double fv : {0.2, 0.4, 0.6, 0.8}) {
                pts.push_back({lo[0] + fu * (hi[0] - lo[0]), lo[1] + fv * (hi[1] - lo[1])});
                y.push_back(base.evaluate(pts.back()) + (level >= 2 ? 0.01 * level : 0.0));
            }
        thb.fit_level_penalized(level, pts, y, {1e-4, 1e-4}, 2);
    }
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    EXPECT_LT(elapsed, 20.0);
    EXPECT_EQ(thb.level_count(), 8);
    EXPECT_NEAR(thb.evaluate({0.9, 0.9}), base.evaluate({0.9, 0.9}), 1e-9);
}
