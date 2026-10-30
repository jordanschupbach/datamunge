// Compatibility forwarding header; canonical implementation is in the library.
#pragma once
#include <datamunge/nn/linear.hpp>
namespace ml {
namespace nn = datamunge::nn;
using datamunge::nn::ml::LeastSquaresQr;
using datamunge::nn::ml::LeastSquaresCholesky;
using datamunge::nn::ml::ElasticNetStep;
}
