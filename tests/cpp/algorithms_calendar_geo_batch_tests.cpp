#include <gtest/gtest.h>

#include <datamunge/algorithms/computus.hpp>
#include <datamunge/algorithms/doomsday.hpp>
#include <datamunge/algorithms/geohash.hpp>
#include <datamunge/algorithms/kabsch.hpp>
#include <datamunge/algorithms/vincenty.hpp>
#include <datamunge/algorithms/zeller.hpp>

#include <cmath>
#include <vector>

using namespace datamunge::algorithms;

// ---- Zeller's congruence ---------------------------------------------------

TEST(Zeller, KnownDates) {
    // 2000-01-01 was a Saturday.
    EXPECT_EQ(zeller_day_name(zeller_gregorian(2000, 1, 1)), "Saturday");
    // 2025-01-01 was a Wednesday.
    EXPECT_EQ(zeller_day_name(zeller_gregorian(2025, 1, 1)), "Wednesday");
    // 1969-07-20 (Apollo 11 Moon landing) was a Sunday.
    EXPECT_EQ(zeller_day_name(zeller_gregorian(1969, 7, 20)), "Sunday");
    // 1789-07-14 (storming of the Bastille) was a Tuesday.
    EXPECT_EQ(zeller_day_name(zeller_gregorian(1789, 7, 14)), "Tuesday");
}

// ---- Doomsday rule ---------------------------------------------------------

TEST(Doomsday, KnownDates) {
    EXPECT_EQ(doomsday_day_name(doomsday_weekday(2000, 1, 1)), "Saturday");
    EXPECT_EQ(doomsday_day_name(doomsday_weekday(2025, 1, 1)), "Wednesday");
    EXPECT_EQ(doomsday_day_name(doomsday_weekday(1969, 7, 20)), "Sunday");
}

TEST(Doomsday, AgreesWithZeller) {
    // The two independent algorithms must agree on every date.
    static const char* names[] = {"Sunday",   "Monday", "Tuesday",  "Wednesday",
                                  "Thursday", "Friday", "Saturday"};
    for (int y = 1901; y <= 2099; y += 37)
        for (int md = 0; md < 3; ++md) {
            int m = 1 + (md * 5), d = 1 + md * 9;
            std::string dd = names[doomsday_weekday(y, m, d)];
            // Map Zeller's h (0=Sat, 1=Sun, ...) to Sunday-based (0=Sun): subtract 1 mod 7.
            int         zsun = (zeller_gregorian(y, m, d) + 6) % 7;  // now 0=Sunday
            std::string zz   = names[zsun];
            EXPECT_EQ(dd, zz) << "date " << y << "-" << m << "-" << d;
        }
}

// ---- Computus (Easter) -----------------------------------------------------

TEST(Computus, KnownEasters) {
    auto e2024 = computus_gregorian(2024);
    EXPECT_EQ(e2024.month, 3);
    EXPECT_EQ(e2024.day, 31);  // 31 March 2024
    auto e2025 = computus_gregorian(2025);
    EXPECT_EQ(e2025.month, 4);
    EXPECT_EQ(e2025.day, 20);  // 20 April 2025
    auto e2000 = computus_gregorian(2000);
    EXPECT_EQ(e2000.month, 4);
    EXPECT_EQ(e2000.day, 23);  // 23 April 2000
}

// ---- Geohash ---------------------------------------------------------------

TEST(Geohash, CanonicalExample) {
    // The Wikipedia reference: (42.6, -5.6) -> "ezs42".
    EXPECT_EQ(geohash_encode(42.6, -5.6, 5), "ezs42");
}

TEST(Geohash, PrefixProperty) {
    std::string a = geohash_encode(42.6, -5.6, 8);
    std::string b = geohash_encode(42.6, -5.6, 5);
    EXPECT_EQ(a.substr(0, 5), b);  // shorter hash is a prefix of the longer
}

// ---- Vincenty / haversine --------------------------------------------------

TEST(Vincenty, FlindersToBuninyong) {
    // Vincenty's own test line (Flinders Peak -> Buninyong): distance 54972.271 m on WGS-84.
    auto r = vincenty_inverse(-37.95103341666667, 144.42486788888889,
                              -37.65282113888889, 143.92649552777778);
    EXPECT_TRUE(r.converged);
    EXPECT_NEAR(r.distance_m, 54972.271, 0.01);
}

TEST(Vincenty, CoincidentPointsZero) {
    auto r = vincenty_inverse(40.0, -75.0, 40.0, -75.0);
    EXPECT_NEAR(r.distance_m, 0.0, 1e-6);
}

TEST(Vincenty, EllipsoidDiffersFromSphere) {
    // Vincenty and haversine agree to ~0.3% but are not identical.
    double v = vincenty_inverse(51.5074, -0.1278, 40.7128, -74.0060).distance_m;  // London-NYC
    double h = haversine_distance(51.5074, -0.1278, 40.7128, -74.0060);
    EXPECT_NEAR(v, h, 0.01 * v);       // within 1%
    EXPECT_GT(std::fabs(v - h), 1.0);  // but genuinely different (ellipsoid vs sphere)
}

// ---- Kabsch ----------------------------------------------------------------

TEST(Kabsch, RecoversKnownRotation) {
    std::vector<Point3> src = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 1, 1}};
    // Rotate 90 deg about z and translate by (5, -2, 1).
    double th = M_PI / 2, c = std::cos(th), s = std::sin(th);
    std::vector<Point3> tgt;
    for (auto& p : src)
        tgt.push_back({c * p[0] - s * p[1] + 5.0, s * p[0] + c * p[1] - 2.0, p[2] + 1.0});

    auto r = kabsch(src, tgt);
    EXPECT_NEAR(r.rmsd, 0.0, 1e-6);            // exact alignment exists
    EXPECT_NEAR(r.rotation[0][1], -1.0, 1e-6); // recovered R equals the z-rotation
    EXPECT_NEAR(r.rotation[1][0], 1.0, 1e-6);
    EXPECT_NEAR(r.rotation[2][2], 1.0, 1e-6);
    EXPECT_NEAR(r.translation[0], 5.0, 1e-6);
    EXPECT_NEAR(r.translation[1], -2.0, 1e-6);
    EXPECT_NEAR(r.translation[2], 1.0, 1e-6);
}

TEST(Kabsch, NonzeroRmsdOnNoisyData) {
    std::vector<Point3> src = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    std::vector<Point3> tgt = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1.5}};  // stretched, not rigid
    auto                r   = kabsch(src, tgt);
    EXPECT_GT(r.rmsd, 0.0);  // no rigid transform fits exactly
}
