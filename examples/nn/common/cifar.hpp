#include "common/binary.hpp"
// Compatibility forwarding header; canonical implementation is in the library.
#pragma once
#include <datamunge/nn/cifar.hpp>
namespace data {
using datamunge::nn::data::Cifar;
using datamunge::nn::data::CifarDataset;
using datamunge::nn::data::ParseCifar;
using datamunge::nn::data::LoadCifar;
}  // namespace data
