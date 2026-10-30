#pragma once
#include <datamunge/nn/inpainting_models.hpp>
namespace inpaint { namespace nn=datamunge::nn;
using datamunge::nn::models::InpaintNorm;
using datamunge::nn::models::ContextConv;
using datamunge::nn::models::ContextUp;
using datamunge::nn::models::ContextConfig;
using datamunge::nn::models::ContextResult;
using datamunge::nn::models::ContextEncoder;
using datamunge::nn::models::ContextDiscriminatorResult;
using datamunge::nn::models::ContextDiscriminator;
using datamunge::nn::models::ContextBCE;
using datamunge::nn::models::ContextReconstruction;
using datamunge::nn::models::ContextComposite;
using datamunge::nn::models::PartialConvResult;
using datamunge::nn::models::PartialConv;
}
