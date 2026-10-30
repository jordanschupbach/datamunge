#include "common/binary.hpp"
// Compatibility forwarding header; canonical implementation is in the library.
#pragma once
#include <datamunge/nn/tfrecord.hpp>
namespace data {
using datamunge::nn::data::Crc32c;
using datamunge::nn::data::MaskedCrc;
using datamunge::nn::data::ValidateCompression;
using datamunge::nn::data::WriteTFRecords;
using datamunge::nn::data::ReadTFRecords;
using datamunge::nn::data::Feature;
using datamunge::nn::data::Example;
using datamunge::nn::data::SerializeExample;
using datamunge::nn::data::ParseFeature;
using datamunge::nn::data::ParseExample;
using datamunge::nn::data::DenseSpec;
using datamunge::nn::data::ParseSingleExample;
}  // namespace data
