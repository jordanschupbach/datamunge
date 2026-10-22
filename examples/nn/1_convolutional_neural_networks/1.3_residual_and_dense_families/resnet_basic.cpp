// CMD: just run-nn resnet_basic
// ResNet-18 and ResNet-34 with post-activation basic residual blocks.
//
// Port of tflow's resnet_basic. Implement the 2/2/2/2 and 3/4/6/3 stage
// layouts from the original paper. Full 224-pixel/1000-class models have
// 11,689,512 and 21,797,672 learned parameters. Check stage dimensions,
// normalization state counts and finite nonzero derivatives through every
// parameter using a two-image training-mode batch. Verify identity/projection
// shortcuts and their input derivatives with a zero residual branch: the
// post-addition ReLU still clips negative inputs.
// https://arxiv.org/abs/1512.03385
//
// Separately train both depths on CIFAR-10 with quarter widths and a 3x3,
// stride-one stem without max pooling. Retain all four stages and blocks;
// use 12k fit / 5k validation rows, twelve epochs, random crop/flip, Adam(.001)
// and kernel L2(.0001). Restore the lowest validation-NLL checkpoint and
// evaluate all 10k test images. These are compact adaptations, not ImageNet
// benchmarks or the original three-stage CIFAR ResNet architecture.
// Options: data=PATH epochs=12 batch=128 lr=.001 fit=12000 augment=0|1 threads=N.

#include "common/resnet_demo.hpp"

int main(int argc, char** argv) {
  try {
    resnet_demo::ResNetBlockChecks(false);
    resnet_demo::FullResNetChecks(18, 11689512, 9600);
    resnet_demo::FullResNetChecks(34, 21797672, 17024);
    for (int depth : {18, 34}) {
      image_training::ImageTrainingConfig cfg;
      cfg.name                  = "resnet" + std::to_string(depth) + "_compact_cifar10";
      cfg.maximum_fit_rows      = 12000;
      cfg.epochs                = 12;
      cfg.minimum_test_accuracy = .4;
      if (image_training::RunCifar10(argc, argv, cfg, resnet_demo::CompactResNet(depth))) return 1;
    }
    std::cout << "All ResNet-18/34 checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
