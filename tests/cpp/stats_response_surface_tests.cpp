#include <datamunge/stats/response_surface.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace {
using namespace datamunge::stats;

// Reference values: rsm 2.10 in R 4.5.3. ccd(2, n0 = c(3, 2), alpha = "rotatable",
// oneblock = TRUE, randomize = FALSE); rsm(y ~ SO(x1, x2)); canonical().
TEST(ResponseSurface, CentralCompositeFitMatchesRsm) {
    const auto d = central_composite_design(2,{.factorial_center_points=3,.axial_center_points=2});
    ASSERT_EQ(d.runs.rows(),13U);
    EXPECT_NEAR(d.alpha,std::sqrt(2.0),1e-15);
    const std::vector<double> x1{-1,1,-1,1,0,0,0,-1.4142135623731,1.4142135623731,0,0,0,0};
    for (std::size_t r = 0; r < 13; ++r)
        EXPECT_NEAR(d.runs(r,0),x1[r],1e-12);
    const std::vector<double> y{76.5,77.0,78.0,79.5,79.9,80.3,80.0,75.6,78.4,77.0,78.5,79.7,80.2};
    const auto s = fit_quadratic_surface(d.runs,y);
    const std::vector<double> rsm{80.020000000000024,.744974746830580,.765165042944950,.250000000000011,
                                  -1.416249999999998,-1.041249999999998};
    for (std::size_t i = 0; i < rsm.size(); ++i)
        EXPECT_NEAR(s.coefficients[i],rsm[i],1e-10);
    EXPECT_NEAR(s.residual_sd,.452063628441529,1e-10);
    EXPECT_NEAR(s.stationary_point[0],.298602989927863,1e-10);
    EXPECT_NEAR(s.stationary_point[1],.403272888560346,1e-10);
    EXPECT_NEAR(s.eigenvalues[0],-1.0034030452835,1e-10);
    EXPECT_NEAR(s.eigenvalues[1],-1.4540969547165,1e-10);
    EXPECT_NEAR(std::abs(s.eigenvectors(0,0)),.28978414868844,1e-9);
    EXPECT_EQ(s.kind,StationaryKind::maximum);
    EXPECT_NEAR(s.stationary_response,s.intercept+.5*(s.linear[0]*s.stationary_point[0]+
                                                     s.linear[1]*s.stationary_point[1]),1e-12);
}

// Reference values: bbd(3, n0 = 3, randomize = FALSE) and rsm(y ~ SO(x1, x2, x3)), canonical(threshold = 0).
TEST(ResponseSurface, BoxBehnkenFitMatchesRsm) {
    const auto d = box_behnken_design(3,3);
    const std::vector<double> rsm_runs{-1,-1,0,1,-1,0,-1,1,0,1,1,0,-1,0,-1,1,0,-1,-1,0,1,1,0,1,
                                       0,-1,-1,0,1,-1,0,-1,1,0,1,1,0,0,0,0,0,0,0,0,0};
    ASSERT_EQ(d.rows(),15U);
    for (std::size_t r = 0; r < 15; ++r)
        for (std::size_t c = 0; c < 3; ++c)
            EXPECT_EQ(d(r,c),rsm_runs[r*3+c]);
    const std::vector<double> y{52,61,60,70,55,63,59,64,50,62,58,66,72,71,73};
    const auto s = fit_quadratic_surface(d,y);
    const std::vector<double> rsm{72,4,4.625,2.125,.25,-.75,-1,-5,-6.25,-6.75};
    for (std::size_t i = 0; i < rsm.size(); ++i)
        EXPECT_NEAR(s.coefficients[i],rsm[i],1e-10);
    EXPECT_NEAR(s.residual_sd,1.6583123951777,1e-10);
    EXPECT_NEAR(s.r_squared,.980568117580554,1e-12);
    EXPECT_NEAR(s.stationary_point[0],.401153212520593,1e-10);
    EXPECT_NEAR(s.stationary_point[1],.369402426239329,1e-10);
    EXPECT_NEAR(s.stationary_point[2],.107757975138535,1e-10);
    EXPECT_NEAR(s.eigenvalues[2],-7.09091907529433,1e-10);
    EXPECT_EQ(s.kind,StationaryKind::maximum);
}

// bbd(4, n0 = 3, block = FALSE) lists the same 27 runs in its blocked order.
TEST(ResponseSurface, FourFactorBoxBehnkenHasRsmRuns) {
    const auto d = box_behnken_design(4,3);
    ASSERT_EQ(d.rows(),27U);
    std::vector<std::vector<double>> rows;
    for (std::size_t r = 0; r < 27; ++r)
        rows.push_back({d(r,0),d(r,1),d(r,2),d(r,3)});
    std::ranges::sort(rows);
    std::vector<std::vector<double>> expected;
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = i+1; j < 4; ++j)
            for (const double a : {-1.0,1.0})
                for (const double b : {-1.0,1.0}) {
                    std::vector<double> v(4,0);
                    v[i] = a;
                    v[j] = b;
                    expected.push_back(v);
                }
    for (int c = 0; c < 3; ++c)
        expected.push_back({0,0,0,0});
    std::ranges::sort(expected);
    EXPECT_EQ(rows,expected);
}

TEST(ResponseSurface, ClassifiesSaddlesAndRidgesAndValidates) {
    const auto d = central_composite_design(2,{.axial=AxialDistance::spherical});
    std::vector<double> saddle, ridge;
    for (std::size_t r = 0; r < d.runs.rows(); ++r) {
        const double a = d.runs(r,0), b = d.runs(r,1);
        saddle.push_back(5+a*a-b*b);
        ridge.push_back(5+a-(a+b)*(a+b)); // B is singular along (1, -1)
    }
    EXPECT_EQ(fit_quadratic_surface(d.runs,saddle).kind,StationaryKind::saddle);
    EXPECT_EQ(fit_quadratic_surface(d.runs,ridge).kind,StationaryKind::ridge);
    const auto fractional = central_composite_design(5,{.generators={"ABCD"}});
    EXPECT_EQ(fractional.factorial_runs,16U);
    EXPECT_NEAR(fractional.alpha,2.0,1e-15);
    EXPECT_EQ(central_composite_design(3,{.axial=AxialDistance::face_centered}).alpha,1.0);
    EXPECT_THROW((void)central_composite_design(3,{.axial=AxialDistance::custom}),std::invalid_argument);
    EXPECT_THROW((void)box_behnken_design(2),std::invalid_argument);
    // Corners plus centers only: the three pure quadratic columns coincide.
    datamunge::linalg::DenseMatrix<double> cube(12,3);
    for (std::size_t r = 0; r < 8; ++r)
        for (std::size_t j = 0; j < 3; ++j)
            cube(r,j) = (r >> j & 1U) ? 1.0 : -1.0;
    std::vector<double> y(12);
    for (std::size_t r = 0; r < y.size(); ++r)
        y[r] = .1*static_cast<double>(r*r);
    EXPECT_THROW((void)fit_quadratic_surface(cube,y),std::runtime_error);
    EXPECT_THROW((void)fit_quadratic_surface(d.runs,std::vector<double>(3,1.0)),std::invalid_argument);
}

} // namespace
