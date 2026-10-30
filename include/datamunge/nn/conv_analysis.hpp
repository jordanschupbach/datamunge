#pragma once

/// @file conv_analysis.hpp
/// @brief Receptive fields of convolution stacks, a 2-D convolution written out as its matrix, and the singular
///        values of a circular convolution from the Fourier transform of its kernel.
///
/// A convolution is a linear map with a very particular matrix. conv2d_matrix() builds that matrix, so properties
/// can be checked with plain linear algebra: the transposed convolution is its transpose, and its singular values
/// bound how much the layer can stretch a signal. For circular (wrap-around) padding the matrix is block doubly
/// circulant, and conv2d_circular_singular_values() gets all its singular values from small per-frequency matrices
/// (Sedghi, Gupta and Long 2019). receptive_fields() applies the usual size-and-jump recursion to a stack of layers.

#include "datamunge/nn/conv.hpp"
#include "datamunge/nn/tensor.hpp"

#include <cstdint>
#include <vector>

namespace datamunge::nn {

/// @brief Geometry of one convolution or pooling layer along one axis.
struct ConvGeometry {
    /// Kernel (window) length.
    std::int64_t kernel{1};
    /// Stride.
    std::int64_t stride{1};
    /// Dilation (1 for none).
    std::int64_t dilation{1};
};

/// @brief Receptive field of one unit after some layers, along one axis, in input positions.
struct ReceptiveField {
    /// Number of input positions that can affect the unit.
    std::int64_t size{1};
    /// Distance in the input between neighbouring units of this layer (product of the strides).
    std::int64_t jump{1};
    /// Input position of the centre of the first unit's field with VALID padding (0-based).
    double start{0.0};
};

/// @brief Receptive field after each layer of a stack: \f$r_l=r_{l-1}+(k_l^{\mathrm{eff}}-1)j_{l-1}\f$ and
///        \f$j_l=j_{l-1}s_l\f$, with effective kernel \f$k^{\mathrm{eff}}=d(k-1)+1\f$, starting from \f$r_0=j_0=1\f$.
/// @param layers  Layers in order, from the input.
/// @return One entry per layer.
/// @throws std::invalid_argument  If a kernel, stride or dilation is less than 1.
[[nodiscard]] std::vector<ReceptiveField> receptive_fields(const std::vector<ConvGeometry>& layers);

/// @brief The matrix M of a single-example 2-D convolution: \f$\operatorname{vec}(y)=M\operatorname{vec}(x)\f$ for x [H, W, Cin] and
///        y [Ho, Wo, Cout], both flattened in row-major (NHWC) order.
///
/// Follows conv2d()'s conventions: cross-correlation with an HWIO kernel [kh, kw, Cin / groups, Cout], strides,
/// dilations, groups and Valid, Same (TensorFlow's split of the padding) or Explicit padding. With @p circular the
/// input wraps around instead of being zero-padded, with the output size and alignment of Same padding.
/// Every entry is a kernel weight or zero; the result is float64 with Ho·Wo·Cout rows and H·W·Cin columns.
///
/// @param height  Input height H.
/// @param width  Input width W.
/// @param kernel  HWIO kernel.
/// @param options  Strides, dilations, padding and groups.
/// @param circular  Wrap around instead of zero padding.
/// @return The convolution matrix.
/// @throws std::invalid_argument  If the kernel is not rank 4 or does not match the groups.
[[nodiscard]] Tensor conv2d_matrix(std::int64_t height, std::int64_t width, const Tensor& kernel, const Conv2DOptions& options = {},
                                   bool circular = false);

/// @brief All singular values of the stride-1 circular convolution of an H x W input with @p kernel, from its 2-D DFT.
///
/// For each frequency (u, v) the transformed kernel \f$\hat K(u,v)=\sum_{a,b}K_{ab}\,e^{-2\pi i(au/H+bv/W)}\f$ is a
/// Cin x Cout complex matrix, and the singular values of the convolution are the union of theirs (Sedghi et al. 2019),
/// H·W·min(Cin, Cout) values in all. Cost: H·W small SVDs instead of one of size H·W·Cin.
///
/// @param kernel  HWIO kernel [kh, kw, Cin, Cout] with kh <= H and kw <= W (groups 1, dilation 1).
/// @param height  Input height H.
/// @param width  Input width W.
/// @return The singular values, descending, float64.
/// @throws std::invalid_argument  If the kernel is not rank 4 or larger than the input.
[[nodiscard]] Tensor conv2d_circular_singular_values(const Tensor& kernel, std::int64_t height, std::int64_t width);

}  // namespace datamunge::nn
