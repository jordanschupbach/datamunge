#include <gtest/gtest.h>

#include <datamunge/cv/nn.hpp>

using namespace datamunge::cv;
using namespace datamunge::linalg;
using namespace datamunge::image;

TEST(Conv2D, IdentityKernelReproducesTheInputExactly) {
    const Tensor input = Tensor::from_values({1, 3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    const Tensor kernel = Tensor::from_values({1, 1, 3, 3}, {0, 0, 0, 0, 1, 0, 0, 0, 0});
    const Tensor bias = Tensor::from_values({1}, {0});
    const Tensor out = conv2d(input, kernel, bias, 1, 1);
    ASSERT_EQ(out.shape(), (std::vector<std::size_t>{1, 3, 3}));
    for (std::size_t i = 0; i < 9; ++i) EXPECT_DOUBLE_EQ(out.at_flat(i), input.at_flat(i));
}

TEST(Conv2D, BoxSumKernelWithNoPaddingProducesTheTotalSum) {
    const Tensor input = Tensor::from_values({1, 3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    const Tensor box_kernel = Tensor::from_values({1, 1, 3, 3}, {1, 1, 1, 1, 1, 1, 1, 1, 1});
    const Tensor bias = Tensor::from_values({1}, {0});
    const Tensor out = conv2d(input, box_kernel, bias, 1, 0);
    ASSERT_EQ(out.shape(), (std::vector<std::size_t>{1, 1, 1}));
    EXPECT_DOUBLE_EQ(out.at_flat(0), 45.0);
}

TEST(Conv2D, BiasIsAddedToEveryOutputElement) {
    const Tensor input = Tensor::from_values({1, 3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    const Tensor box_kernel = Tensor::from_values({1, 1, 3, 3}, {1, 1, 1, 1, 1, 1, 1, 1, 1});
    const Tensor bias = Tensor::from_values({1}, {10});
    const Tensor out = conv2d(input, box_kernel, bias, 1, 0);
    EXPECT_DOUBLE_EQ(out.at_flat(0), 55.0);
}

TEST(Conv2D, RejectsMismatchedChannelCounts) {
    const Tensor input = Tensor::zeros({2, 4, 4});
    const Tensor kernel = Tensor::zeros({1, 3, 3, 3});
    const Tensor bias = Tensor::zeros({1});
    EXPECT_THROW(conv2d(input, kernel, bias), std::invalid_argument);
}

TEST(MaxPool2D, MatchesHandComputedWindowMaxima) {
    const Tensor input = Tensor::from_values({1, 4, 4}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    const Tensor out = max_pool2d(input, 2, 2);
    ASSERT_EQ(out.shape(), (std::vector<std::size_t>{1, 2, 2}));
    EXPECT_DOUBLE_EQ(out.at({0, 0, 0}), 6.0);
    EXPECT_DOUBLE_EQ(out.at({0, 1, 1}), 16.0);
}

TEST(AvgPool2D, MatchesHandComputedWindowMeans) {
    const Tensor input = Tensor::from_values({1, 4, 4}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    const Tensor out = avg_pool2d(input, 2, 2);
    EXPECT_DOUBLE_EQ(out.at({0, 0, 0}), 3.5);
}

TEST(Relu, ZeroesNegativesAndPassesPositivesThrough) {
    const Tensor input = Tensor::from_values({4}, {-2.0, -0.5, 0.5, 2.0});
    const Tensor out = relu(input);
    EXPECT_DOUBLE_EQ(out.at_flat(0), 0.0);
    EXPECT_DOUBLE_EQ(out.at_flat(1), 0.0);
    EXPECT_DOUBLE_EQ(out.at_flat(2), 0.5);
    EXPECT_DOUBLE_EQ(out.at_flat(3), 2.0);
}

TEST(Sigmoid, MapsZeroToOneHalf) {
    const Tensor out = sigmoid(Tensor::from_values({1}, {0.0}));
    EXPECT_NEAR(out.at_flat(0), 0.5, 1e-9);
}

TEST(Softmax, OutputSumsToOneAndPreservesOrdering) {
    const Tensor logits = Tensor::from_values({3}, {1.0, 2.0, 3.0});
    const Tensor probs = softmax(logits);
    EXPECT_NEAR(probs.at_flat(0) + probs.at_flat(1) + probs.at_flat(2), 1.0, 1e-9);
    EXPECT_GT(probs.at_flat(2), probs.at_flat(1));
    EXPECT_GT(probs.at_flat(1), probs.at_flat(0));
}

TEST(Softmax, RejectsNonOneDimensionalInput) {
    const Tensor input = Tensor::zeros({2, 2});
    EXPECT_THROW(softmax(input), std::invalid_argument);
}

TEST(ImageToTensor, ProducesCorrectlyNormalizedChannelPlanes) {
    Image img(2, 2, ImageMode::RGB);
    img.set_pixel(0, 0, Pixel{255, 0, 0, 255});
    const Tensor t = image_to_tensor(img);
    ASSERT_EQ(t.shape(), (std::vector<std::size_t>{3, 2, 2}));
    EXPECT_NEAR(t.at({0, 0, 0}), 1.0, 1e-9);
    EXPECT_NEAR(t.at({1, 0, 0}), 0.0, 1e-9);
}
