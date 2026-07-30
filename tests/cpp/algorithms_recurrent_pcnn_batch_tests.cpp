#include <gtest/gtest.h>

#include <datamunge/algorithms/almeida_pineda.hpp>
#include <datamunge/algorithms/canopy_clustering.hpp>
#include <datamunge/algorithms/pcnn.hpp>

#include <cstddef>
#include <vector>

using datamunge::algorithms::almeida_pineda_predict;
using datamunge::algorithms::almeida_pineda_train;
using datamunge::algorithms::canopy_clustering;
using datamunge::algorithms::pcnn_run;
using datamunge::algorithms::PCNNParameters;

TEST(AlmeidaPineda, LearnsFixedPointMapping) {
    // Three input patterns, each with a distinct target output value the recurrent
    // network must produce at equilibrium. 2 inputs, 2 hidden, 1 output (unit 4).
    std::vector<std::vector<double>> inputs  = {{1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}};
    std::vector<std::vector<double>> targets = {{0.6}, {-0.6}, {0.3}};
    std::vector<std::size_t>         outputs = {4};

    auto res = almeida_pineda_train(inputs, targets, /*num_units=*/5, /*num_inputs=*/2, outputs,
                                    /*epochs=*/600, /*lr=*/0.2, /*seed=*/1, /*settle=*/80, /*relax=*/0.4);

    ASSERT_GE(res.error_history.size(), 2u);
    // Training error drops substantially and ends small.
    EXPECT_LT(res.error_history.back(), res.error_history.front());
    EXPECT_LT(res.error_history.back(), 0.02);
    // Predictions track the targets.
    for (std::size_t p = 0; p < inputs.size(); ++p) {
        auto out = almeida_pineda_predict(res.net, inputs[p], 80, 0.4);
        EXPECT_NEAR(out[0], targets[p][0], 0.15);
    }
}

TEST(PCNN, FiresBrighterRegionsEarlier) {
    // 10x10 image: left half bright (0.9), right half dim (0.3).
    std::vector<std::vector<double>> img(10, std::vector<double>(10, 0.0));
    for (std::size_t i = 0; i < 10; ++i)
        for (std::size_t j = 0; j < 10; ++j) img[i][j] = (j < 5) ? 0.9 : 0.3;

    PCNNParameters p;
    p.iterations = 20;
    auto r        = pcnn_run(img, p);

    ASSERT_EQ(r.first_fire.size(), 10u);
    // Average first-fire iteration: bright region fires strictly earlier than dim region.
    double bright = 0, dim = 0;
    int    nb = 0, nd = 0;
    for (std::size_t i = 0; i < 10; ++i)
        for (std::size_t j = 0; j < 10; ++j) {
            if (r.first_fire[i][j] == 0) continue;  // never fired
            if (j < 5) { bright += r.first_fire[i][j]; ++nb; }
            else { dim += r.first_fire[i][j]; ++nd; }
        }
    ASSERT_GT(nb, 0);
    ASSERT_GT(nd, 0);
    EXPECT_LT(bright / nb, dim / nd);
}

TEST(Canopy, CoversSeparatedBlobsWithFewCanopies) {
    // Three well-separated blobs of 20 points each (deterministic layout).
    std::vector<std::vector<double>> pts;
    const std::vector<std::vector<double>> centers = {{0, 0}, {10, 0}, {0, 10}};
    for (const auto& c : centers)
        for (int i = 0; i < 20; ++i) pts.push_back({c[0] + 0.05 * i, c[1] + 0.03 * i});

    // t2 larger than a blob's spread (so a whole blob is claimed by its seed) but far
    // below the ~10-unit inter-blob distance -> exactly one canopy per blob.
    auto canopies = canopy_clustering(pts, /*t1=*/3.0, /*t2=*/2.5);
    EXPECT_EQ(canopies.size(), 3u);
    for (const auto& c : canopies) EXPECT_EQ(c.members.size(), 20u);  // each canopy = one full blob
}
