#include "common/binary.hpp"
// Compatibility forwarding header; canonical implementation is in the library.
#pragma once
#include <datamunge/nn/csv.hpp>
namespace data {
using datamunge::nn::data::CsvTable;
using datamunge::nn::data::ParseCsv;
using datamunge::nn::data::LoadCsv;
using datamunge::nn::data::Numeric;
using datamunge::nn::data::FeatureKind;
using datamunge::nn::data::FeatureSpec;
using datamunge::nn::data::FittedFeature;
using datamunge::nn::data::FeatureColumns;
}  // namespace data
