// Tests for the binding-friendly nn facade (NnTensor, NnModel).

#include <gtest/gtest.h>

#include <cmath>
#include <datamunge/nn/api.hpp>
#include <filesystem>
#include <vector>

namespace nn = datamunge::nn;
using nn::NnModel;
using nn::NnTensor;

TEST(NnApi, TensorBasics) {
  const NnTensor a({1, 2, 3, 4, 5, 6}, {2, 3});
  EXPECT_EQ(a.shape(), (std::vector<int>{2, 3}));
  EXPECT_EQ(a.dtype(), "float32");
  EXPECT_EQ(a.numel(), 6);
  EXPECT_DOUBLE_EQ(a.at({1, 2}), 6.0);
  EXPECT_EQ(a.transpose().shape(), (std::vector<int>{3, 2}));
  EXPECT_EQ(a.sum().item(), 21.0);
  EXPECT_EQ(a.sum({0}).values(), (std::vector<double>{5, 7, 9}));
  EXPECT_EQ(a.matmul(a, false, true).values(), (std::vector<double>{14, 32, 32, 77}));
  EXPECT_EQ(a.argmax(1).values(), (std::vector<double>{2, 2}));
  EXPECT_EQ(a.astype("int64").dtype(), "int64");
  EXPECT_EQ(NnTensor::arange(0, 5).values(), (std::vector<double>{0, 1, 2, 3, 4}));
  EXPECT_EQ(NnTensor::randn({3}, 7).values(), NnTensor::randn({3}, 7).values());
  EXPECT_NE(a.str().find("shape=[2, 3]"), std::string::npos);
  EXPECT_THROW(NnTensor({1, 2, 3}, {2, 2}), std::invalid_argument);
  EXPECT_THROW(NnTensor::zeros({2}, "complex"), std::invalid_argument);
}

TEST(NnApi, TensorGradients) {
  NnTensor x({1, 2, 3}, {3}, "float64");
  x.set_requires_grad(true);
  NnTensor y = x.square().sum();  // d/dx sum(x^2) = 2x
  y.backward();
  EXPECT_EQ(x.grad().values(), (std::vector<double>{2, 4, 6}));
  x.zero_grad();
  EXPECT_FALSE(x.grad().defined());
}

TEST(NnApi, ModelTrainsAndRoundTrips) {
  // y = 2 x0 - x1 + 0.5, learnable by a linear layer.
  const NnTensor x = NnTensor::uniform({64, 2}, 3, -1, 1);
  const NnTensor w({2, -1}, {2, 1});
  const NnTensor y = x.matmul(w).add_scalar(0.5);
  NnModel        model(11);
  model.add_dense(1);
  model.set_optimizer("adam", 0.05);
  double first = 0, last = 0;
  for (int step = 0; step < 300; ++step) {
    last = model.train_step(x, y, "mse");
    if (step == 0) first = last;
  }
  EXPECT_LT(last, 1e-3);
  EXPECT_LT(last, first);
  EXPECT_EQ(model.parameter_names(), (std::vector<std::string>{"dense0/W", "dense0/b"}));
  EXPECT_NEAR(model.parameter("dense0/W").values()[0], 2.0, 0.05);
  EXPECT_EQ(model.parameter_count(), 3);

  const auto path = (std::filesystem::temp_directory_path() / "nn_api_model.npz").string();
  model.save(path);
  NnModel copy;
  copy.add_dense(1);
  copy.load(path);
  EXPECT_EQ(copy.predict(x).values(), model.predict(x).values());
}

TEST(NnApi, ConvModelGradientsAndBatchNorm) {
  NnModel model(5);
  model.add_reshape({6, 6, 1});
  model.add_conv2d(4, 3, 1, "same", "relu");
  model.add_batch_norm();
  model.add_max_pool2d(2, 2);
  model.add_flatten();
  model.add_dropout(0.1);
  model.add_dense(3);
  const NnTensor x = NnTensor::randn({8, 36}, 1);
  const NnTensor labels({0, 1, 2, 0, 1, 2, 0, 1}, {8}, "int64");
  EXPECT_EQ(model.predict(x).shape(), (std::vector<int>{8, 3}));
  const double loss = model.compute_gradients(x, labels, "softmax_cross_entropy");
  EXPECT_TRUE(std::isfinite(loss));
  EXPECT_EQ(model.gradient("conv1/W").shape(), (std::vector<int>{3, 3, 1, 4}));
  EXPECT_NE(model.summary().find("batchnorm2"), std::string::npos);
  model.set_optimizer("sgd", 0.1);
  for (int i = 0; i < 20; ++i) model.train_step(x, labels, "softmax_cross_entropy");
  EXPECT_LT(model.loss(x, labels, "softmax_cross_entropy"), loss);
  EXPECT_THROW(model.add_dense(2, "nonsense"), std::invalid_argument);
  EXPECT_FALSE(nn::nn_gemm_backend().empty());
}
