// Compatibility forwarding header; canonical implementation is in the library.
#pragma once
#include <datamunge/nn/polynomial.hpp>
namespace ml {
namespace nn = datamunge::nn;
using datamunge::nn::ml::PolynomialFeatures;
}
