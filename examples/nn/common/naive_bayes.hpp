// Compatibility forwarding header; canonical implementation is in the library.
#pragma once
#include <datamunge/nn/naive_bayes.hpp>
namespace ml {
using datamunge::nn::ml::NaiveBayesKind;
using datamunge::nn::ml::NaiveBayes;
}
