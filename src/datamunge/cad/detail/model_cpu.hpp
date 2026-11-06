#pragma once

// CPU ownership adapter for mep's importers and primitive builders. Texture
// pixels stay in shared values; no context, GPU upload, or global cache exists.
#include <vector>

#include "types.hpp"

namespace datamunge::cad::detail::gfx {
Model                      LoadObjModel(const char* path);
Model                      LoadGltfModel(const char* path);
Model                      LoadIqmModel(const char* path);
Model                      LoadVoxModel(const char* path);
Model                      LoadM3dModel(const char* path);
Model                      LoadModel(const char* path);
void                       UnloadModel(Model model);
void                       UnloadMesh(Mesh mesh);
inline void                UploadMesh(Mesh*, bool) {} // CPU arrays already own the complete mesh.
Material                   LoadMaterialDefault();
Mesh                       GenMeshCube(float width, float height, float length);
Mesh                       GenMeshSphere(float radius, int rings, int slices);
Mesh                       GenMeshCylinder(float radius, float height, int slices);
Mesh                       GenMeshCone(float radius, float height, int slices);
Mesh                       GenMeshPlane(float width, float length, int res_x, int res_z);
Mesh                       GenMeshTorus(float radius, float size, int rad_seg, int sides);
Image                      LoadImage(const char* path);
Image                      GenImageColor(int width, int height, Color color);
void                       UnloadImage(Image image);
void                       ImageFormat(Image* image, int format);
Texture2D                  LoadTextureFromImage(Image image);
Image                      LoadImageFromTexture(Texture2D texture);
std::vector<unsigned char> ExportImageToMemory(Image image, const char* extension);
bool                       ExportImage(Image image, const char* path);
} // namespace datamunge::cad::detail::gfx
