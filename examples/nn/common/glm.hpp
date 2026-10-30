// Compatibility forwarding header; canonical implementation is in the library.
#pragma once
#include <datamunge/nn/glm.hpp>
namespace ml {
namespace nn = datamunge::nn;
using datamunge::nn::ml::GlmFamily;
using datamunge::nn::ml::GlmNll;
}
