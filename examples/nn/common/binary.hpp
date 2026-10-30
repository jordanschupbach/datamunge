// Compatibility forwarding header; canonical implementation is in the library.
#pragma once
#include <datamunge/nn/binary.hpp>
namespace data {
using datamunge::nn::data::Bytes;
using datamunge::nn::data::ReadBinary;
using datamunge::nn::data::WriteBinary;
using datamunge::nn::data::Product;
using datamunge::nn::data::BigEndian32;
}  // namespace data
