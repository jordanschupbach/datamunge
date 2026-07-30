#include <gtest/gtest.h>

#include <datamunge/algorithms/bernstein_vazirani.hpp>
#include <datamunge/algorithms/deutsch_jozsa.hpp>
#include <datamunge/algorithms/grover.hpp>
#include <datamunge/algorithms/phase_estimation.hpp>
#include <datamunge/algorithms/qft.hpp>
#include <datamunge/algorithms/quantum_sim.hpp>
#include <datamunge/algorithms/simon.hpp>

#include <cmath>
#include <complex>
#include <cstddef>

using namespace datamunge::algorithms;

// ---- Simulator basics ------------------------------------------------------

TEST(QuantumSim, BellState) {
    auto s = qstate_zero(2);
    h_gate(s, 0);
    cnot(s, 0, 1);
    // (|00> + |11>)/sqrt2
    EXPECT_NEAR(std::norm(s[0]), 0.5, 1e-12);
    EXPECT_NEAR(std::norm(s[3]), 0.5, 1e-12);
    EXPECT_NEAR(std::norm(s[1]), 0.0, 1e-12);
    EXPECT_NEAR(std::norm(s[2]), 0.0, 1e-12);
}

// ---- QFT vs DFT ------------------------------------------------------------

TEST(QFT, MatchesDFT) {
    int         n = 3;
    std::size_t N = std::size_t{1} << n;
    QuantumState in(N);
    double       nrm = 0;
    for (std::size_t i = 0; i < N; ++i) { in[i] = Complex(std::cos(0.7 * i + 1), std::sin(0.3 * i)); nrm += std::norm(in[i]); }
    for (auto& z : in) z /= std::sqrt(nrm);

    QuantumState s = in;
    qft(s, n);
    for (std::size_t k = 0; k < N; ++k) {
        Complex acc = 0;
        for (std::size_t j = 0; j < N; ++j) acc += in[j] * std::polar(1.0, 2 * M_PI * double(j * k) / double(N));
        acc /= std::sqrt(double(N));
        EXPECT_NEAR(std::abs(acc - s[k]), 0.0, 1e-12);
    }
    inverse_qft(s, n);
    for (std::size_t i = 0; i < N; ++i) EXPECT_NEAR(std::abs(s[i] - in[i]), 0.0, 1e-12);
}

// ---- Deutsch-Jozsa ---------------------------------------------------------

TEST(DeutschJozsa, ConstantAndBalanced) {
    int n = 4;
    auto constant0 = deutsch_jozsa(n, [](std::size_t) { return 0; });
    EXPECT_TRUE(constant0.constant);
    EXPECT_NEAR(constant0.prob_all_zero, 1.0, 1e-12);

    auto constant1 = deutsch_jozsa(n, [](std::size_t) { return 1; });
    EXPECT_TRUE(constant1.constant);

    // Balanced: parity of x (0 for half, 1 for the other half).
    auto balanced = deutsch_jozsa(n, [](std::size_t x) {
        int p = 0; while (x) { p ^= 1; x &= x - 1; } return p;
    });
    EXPECT_FALSE(balanced.constant);
    EXPECT_NEAR(balanced.prob_all_zero, 0.0, 1e-12);
}

// ---- Bernstein-Vazirani ----------------------------------------------------

TEST(BernsteinVazirani, RecoversSecret) {
    for (std::size_t secret : {std::size_t(0b1011), std::size_t(0b0110), std::size_t(0b1111), std::size_t(0)})
        EXPECT_EQ(bernstein_vazirani(5, secret), secret);
}

// ---- Grover ----------------------------------------------------------------

TEST(Grover, FindsMarkedItem) {
    int n = 6;  // N = 64
    for (std::size_t target : {std::size_t(0), std::size_t(42), std::size_t(63)}) {
        auto r = grover(n, target);
        EXPECT_EQ(r.measured, target);
        EXPECT_GT(r.success_prob, 0.99);
    }
}

TEST(Grover, SuccessTracePeaksThenFalls) {
    int  n     = 6;
    auto trace = grover_success_trace(n, 10, 20);
    // Probability rises to near 1 around pi/4 sqrt(N) ~ 6, then falls (over-rotation).
    int peak = 0;
    for (int i = 1; i < (int)trace.size(); ++i) if (trace[i] > trace[peak]) peak = i;
    EXPECT_GE(peak, 4);
    EXPECT_LE(peak, 8);
    EXPECT_GT(trace[peak], 0.99);
    EXPECT_LT(trace.back(), trace[peak]);  // over-iterating lowers success
}

// ---- Simon -----------------------------------------------------------------

TEST(Simon, RecoversPeriod) {
    for (std::size_t period : {std::size_t(0b101), std::size_t(0b011), std::size_t(0b110)}) {
        auto r = simon(3, period);
        EXPECT_EQ(r.recovered_period, period);
        // Every measured y must be orthogonal to the period.
        for (std::size_t y : r.observed_y) {
            std::size_t m = y & period;
            int         p = 0; while (m) { p ^= 1; m &= m - 1; }
            EXPECT_EQ(p, 0);
        }
    }
}

// ---- Quantum phase estimation ----------------------------------------------

TEST(PhaseEstimation, ExactDyadicPhase) {
    // phi = 3/8 is exactly representable with t=3 bits.
    auto r = phase_estimation(3, 0.375);
    EXPECT_EQ(r.measured, std::size_t(3));
    EXPECT_NEAR(r.estimate, 0.375, 1e-12);
    EXPECT_NEAR(r.probability, 1.0, 1e-12);
}

TEST(PhaseEstimation, ApproximatesNonDyadicPhase) {
    // phi = 0.2 -> nearest 5-bit value is 6/32 = 0.1875 (0.2*32 = 6.4).
    auto r = phase_estimation(5, 0.2);
    EXPECT_EQ(r.measured, std::size_t(6));
    EXPECT_NEAR(r.estimate, 0.1875, 1e-12);
    EXPECT_GT(r.probability, 0.5);
}
