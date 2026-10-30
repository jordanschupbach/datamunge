#pragma once
#include <datamunge/nn/character_cnn_models.hpp>
namespace charcnn {namespace nn=datamunge::nn;using nn::Tensor;
using datamunge::nn::models::CharacterAlphabet;
using datamunge::nn::models::CharacterEncode;
using datamunge::nn::models::CharacterConfig;
using datamunge::nn::models::CharacterFinalLength;
using datamunge::nn::models::CharacterOutput;
using datamunge::nn::models::CharacterCnn;
}
