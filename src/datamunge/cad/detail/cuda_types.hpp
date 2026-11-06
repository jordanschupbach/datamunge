#pragma once
/// @file cuda_types.hpp
/// @brief SDK-free, fixed-width host/device layout for the opt-in CAD CUDA ray tracer.

/// @brief Private binary layouts shared by the CAD kernel and host uploader.
namespace datamunge::cad::detail::cuda {
/// @brief Packed float32 vector without CUDA-specific alignment.
struct V3 {
  float x; ///< First vector component.
  float y; ///< Second vector component.
  float z; ///< Third vector component.
};
/// @brief Packed straight-alpha color, with the color space specified by its owner.
struct V4 {
  float x; ///< Red channel.
  float y; ///< Green channel.
  float z; ///< Blue channel.
  float w; ///< Alpha channel.
};
/// @brief World triangle with shading attributes in original flattened order.
struct Triangle {
  V3    origin;     ///< First world vertex.
  V3    edge1;      ///< Second minus first world vertex.
  V3    edge2;      ///< Third minus first world vertex.
  V3    normals[3]; ///< Three world vertex normals.
  float uv[6];      ///< Three interleaved UV pairs.
  V3    tangent;    ///< World tangent for the normal map.
  float epsilon;    ///< Original relative determinant threshold.
  int   object;     ///< Source object ID.
  int   material;   ///< Material index local to the base or actor scene.
};
/// @brief Balanced BVH node; zero count selects the two child indices.
struct Node {
  V3  lo;    ///< Conservatively rounded lower world bounds.
  V3  hi;    ///< Conservatively rounded upper world bounds.
  int start; ///< First leaf order index.
  int count; ///< Leaf triangle count, or zero for an internal node.
  int left;  ///< Left child index for an internal node.
  int right; ///< Right child index for an internal node.
};
/// @brief Material whose optional texture indices are local to its base or actor scene.
struct Material {
  V4    color;         ///< Straight-alpha sRGB tint.
  float roughness;     ///< Scalar roughness when unmapped.
  float metallic;      ///< Scalar metallic fraction when unmapped.
  int   color_map;     ///< RGBA texture index, or -1.
  int   normal_map;    ///< Tangent normal texture index, or -1.
  int   roughness_map; ///< Red-channel roughness texture index, or -1.
  int   metallic_map;  ///< Red-channel metallic texture index, or -1.
};
/// @brief Repeating bilinear RGBA8 view into the scene's packed texture byte array.
struct Texture {
  int width;  ///< Positive width in texels.
  int height; ///< Positive height in texels.
  int offset; ///< First RGBA byte offset.
};
/// @brief Visible light, preserving scene order and the first-light shadow rule.
struct Light {
  V3    position;  ///< World position for a point light.
  V3    direction; ///< Unit directional-light vector.
  V3    color;     ///< Linear light RGB.
  float intensity; ///< Nonnegative strength.
  float range;     ///< Positive point-light cutoff in world units.
  int   point;     ///< Nonzero selects a point light.
};
/// @brief Synchronous readback pixel with centre-ray geometry.
struct Pixel {
  unsigned char rgba[4]; ///< Straight-alpha display RGBA8.
  float         depth;   ///< Normalized-ray world distance, infinity for background.
  int           object;  ///< Source scene ID, or -1.
};
/// @brief By-value kernel controls; no process pointers or SDK state.
struct Parameters {
  V3    origin;                ///< Camera world position.
  V3    forward;               ///< Unit camera forward direction.
  V3    right;                 ///< Unit camera horizontal direction.
  V3    up;                    ///< Unit camera vertical direction.
  float scale;                 ///< Tangent of half the perspective vertical FOV.
  float aspect;                ///< Width divided by height.
  float ortho_height;          ///< Vertical parallel-projection span in world units.
  float near_clip;             ///< Inclusive minimum ray distance.
  float far_clip;              ///< Exclusive maximum ray distance.
  int   width;                 ///< Output pixel width.
  int   height;                ///< Output pixel height.
  int   projection;            ///< Projection enum value.
  int   samples;               ///< Square primary sample grid dimension.
  int   style;                 ///< RenderStyle enum value.
  int   smooth;                ///< Nonzero interpolates vertex normals.
  int   both;                  ///< Nonzero intersects both triangle orientations.
  int   layers;                ///< Maximum transparency layers.
  int   shadows;               ///< Nonzero enables key visibility rays.
  int   shadow_samples;        ///< Finite-disk key sample count.
  int   ao_samples;            ///< Cosine-hemisphere sample count.
  int   physical;              ///< Nonzero enables linear-light GGX/filmic output.
  float ao_distance;           ///< Local occlusion radius in world units.
  float exposure;              ///< Positive linear exposure.
  float wire_width;            ///< Positive approximate edge width in pixels.
  float angular_radius;        ///< Key-light disk radius in radians.
  V4    background;            ///< Straight-alpha sRGB background.
  V4    wire_color;            ///< Straight-alpha wire color.
  int   base_count;            ///< Triangle index separating base and actor arrays.
  int   node_count;            ///< Valid hierarchy node count.
  int   light_count;           ///< Visible light count.
  V3    shadow_directions[32]; ///< Pixel-independent directional-key rays.
  V3    ao[16];                ///< Hemisphere coefficients in original sample order.
  float studio_environment;    ///< Opt-in analytic studio radiance strength; zero retains legacy
                               ///< ambient.
};
static_assert(
    sizeof(V3) == 12 && sizeof(Triangle) == 120 && sizeof(Node) == 40 && sizeof(Pixel) == 12 &&
        sizeof(Material) == 40 && sizeof(Texture) == 12 && sizeof(Light) == 48 &&
        sizeof(Parameters) <= 4096,
    "CAD CUDA storage ABI must match the generated kernel");
} // namespace datamunge::cad::detail::cuda
