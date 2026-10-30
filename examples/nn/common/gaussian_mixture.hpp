// Compatibility forwarding header; canonical implementation is in the library.
#pragma once
#include <datamunge/nn/gaussian_mixture.hpp>
namespace ml {
namespace nn = datamunge::nn;
using datamunge::nn::ml::GaussianMixtureState;
using datamunge::nn::ml::GaussianMixtureStep;
using datamunge::nn::ml::GaussianMixture;
}
