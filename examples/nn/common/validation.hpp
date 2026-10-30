// Compatibility forwarding header; canonical implementation is in the library.
#pragma once
#include <datamunge/nn/validation.hpp>
namespace ml {
namespace nn = datamunge::nn;
using datamunge::nn::ml::Fold;
using datamunge::nn::ml::CompleteFolds;
using datamunge::nn::ml::KFold;
using datamunge::nn::ml::StratifiedKFold;
using datamunge::nn::ml::CrossValidationScore;
using datamunge::nn::ml::CrossValidate;
}
