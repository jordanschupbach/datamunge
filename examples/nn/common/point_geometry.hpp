#pragma once
#include <datamunge/nn/point_geometry.hpp>
namespace point_geometry {namespace nn=datamunge::nn;using nn::Tensor;
using datamunge::nn::models::PointGather;
using datamunge::nn::models::PointFps;
using datamunge::nn::models::PointNeighborhood;
using datamunge::nn::models::PointBall;
using datamunge::nn::models::PointGrouped;
using datamunge::nn::models::PointGroup;
}
