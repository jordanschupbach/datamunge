// CMD: just run-nn mlp_mnist
// A multilayer perceptron on the original MNIST handwritten digits.
//
// Port of tflow's mlp_mnist. Preparation (in ~/projects/tflow_examples):
// python3 scripts/fetch_datasets.py mnist --split both; the data are read from
// $DATAMUNGE_DATA_DIR/mnist (default ~/projects/tflow_examples/output/datasets).
// Architecture: flattened 784 pixels -> 128 ReLU -> 64 ReLU -> 10 logits,
// with 109,386 learned parameters, Adam and small L2 weight regularization.
// Reuse the softmax example's 55k/5k fit/validation split, keep the lowest
// validation cross-entropy checkpoint, and evaluate all 10k test images once.
// Print the confusion matrix and save the checkpoint (.npz) and validation
// history under output/nn/models/mlp_mnist. Require at least 96% test accuracy
// at the default training budget. No augmentation or access to test data
// during training. Options: data=PATH epochs=12 batch=256 lr=.001 threads=N
// (tflow's device= option is dropped; nn runs on the CPU).

#include "common/mnist.hpp"

int main(int argc, char** argv) { return mnist::Run(argc, argv, "mlp_mnist", {128, 64}, 12, .001, .96); }
