#pragma once

// Convenience umbrella header -- includes the whole computer-vision module: corner/feature
// detection (Harris, Shi-Tomasi, FAST, Difference-of-Gaussians blobs), Hough line/circle
// transforms, binary morphology + segmentation (erode/dilate/open/close, connected-component
// labeling, Otsu thresholding, k-means color segmentation), Gaussian/Laplacian image
// pyramids, sparse Lucas-Kanade optical flow, and basic (inference-only, no training) neural-
// network building blocks -- conv2d/pooling/activations -- on top of linalg::Tensor.

#include <datamunge/cv/blob.hpp>
#include <datamunge/cv/corners.hpp>
#include <datamunge/cv/hough.hpp>
#include <datamunge/cv/morphology.hpp>
#include <datamunge/cv/nn.hpp>
#include <datamunge/cv/optical_flow.hpp>
#include <datamunge/cv/pyramid.hpp>
#include <datamunge/cv/segmentation.hpp>
