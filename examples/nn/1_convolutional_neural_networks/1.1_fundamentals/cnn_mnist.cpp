// CMD: just run-nn cnn_mnist
// A two-convolution, two-dense-layer network on the original MNIST dataset.
//
// Port of tflow's cnn_mnist. Preparation (in ~/projects/tflow_examples):
// python3 scripts/fetch_datasets.py mnist --split both.
// Reshape normalized pixels to NHWC; apply 3x3 Conv(16)-ReLU-pool, then
// 3x3 Conv(32)-ReLU-pool, flatten the 7x7 maps, Dense(64)-ReLU and ten logits.
// Fit all 55k training rows, select a checkpoint on the same 5k validation
// split as the linear/MLP baselines, and evaluate the official 10k test rows.
// Use Adam and L2 on convolution/dense kernels, without image augmentation.
// Default: six epochs, batch 256, lr=.001; require at least 98% test accuracy.
// Options: data=PATH epochs=6 batch=256 lr=.001 threads=N.
// Checkpoint and validation history: output/nn/models/cnn_mnist/.

#include "common/cnn_baselines.hpp"
#include "common/mnist.hpp"

int main(int argc, char** argv) {
  auto architecture = [](mnist::nn::ParameterStore& p, const mnist::nn::Tensor& x, bool) {
    return cnn_baselines::MnistCnn(p, x).logits;
  };
  return mnist::Run(argc, argv, "cnn_mnist", {}, 6, .001, .98, architecture);
}
