#include "detail/model_cpu.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>

#include "detail/image_codec.hpp"
#include "detail/jpeg_codec.hpp"
#include "detail/png_codec.hpp"
#include "detail/vecmath.hpp"

namespace datamunge::cad::detail::gfx {
Model LoadModel(const char* path) {
  std::string name(path);
  std::transform(
      name.begin(), name.end(), name.begin(), [](unsigned char c) { return std::tolower(c); });
  if (name.ends_with(".obj"))
    return LoadObjModel(path);
  if (name.ends_with(".gltf") || name.ends_with(".glb"))
    return LoadGltfModel(path);
  if (name.ends_with(".iqm"))
    return LoadIqmModel(path);
  if (name.ends_with(".vox"))
    return LoadVoxModel(path);
  if (name.ends_with(".m3d"))
    return LoadM3dModel(path);
  return {};
}
void UnloadMesh(Mesh m) {
  delete[] m.vertices;
  delete[] m.normals;
  delete[] m.texcoords;
  delete[] m.colors;
  delete[] m.indices;
  delete[] m.animVertices;
  delete[] m.animNormals;
  delete[] m.boneIds;
  delete[] m.boneWeights;
  delete[] m.boneMatrices;
  delete[] m.vboId;
}
void UnloadModel(Model m) {
  for (int i = 0; i < m.meshCount; ++i)
    UnloadMesh(m.meshes[i]);
  for (int i = 0; i < m.materialCount; ++i)
    delete[] m.materials[i].maps;
  delete[] m.meshes;
  delete[] m.materials;
  delete[] m.meshMaterial;
}
Material LoadMaterialDefault() {
  Material m;
  m.maps                              = new MaterialMap[kMaxMaterialMaps]();
  m.maps[kMaterialMapAlbedo].color    = White;
  m.maps[kMaterialMapRoughness].value = .5f;
  return m;
}
Image LoadImage(const char* path) {
  int         w = 0, h = 0;
  std::string error;
  auto*       pixels = image_codec::DecodeFile(path, &w, &h, &error);
  return {pixels, w, h, 1, kPixelFormatR8G8B8A8};
}
Image GenImageColor(int w, int h, Color c) {
  if (w <= 0 || h <= 0)
    return {};
  const auto bytes = static_cast<std::size_t>(w) * h * 4;
  auto*      p     = static_cast<unsigned char*>(std::malloc(bytes));
  if (!p)
    throw std::bad_alloc();
  for (std::size_t i = 0; i < bytes; i += 4) {
    p[i]     = c.r;
    p[i + 1] = c.g;
    p[i + 2] = c.b;
    p[i + 3] = c.a;
  }
  return {p, w, h, 1, kPixelFormatR8G8B8A8};
}
void UnloadImage(Image image) {
  std::free(image.data);
}
void ImageFormat(Image* image, int format) {
  if (!image || image->format != kPixelFormatR8G8B8A8 || format != kPixelFormatR8G8B8A8)
    throw std::invalid_argument("CAD CPU adapter requires RGBA8 images");
}
Texture2D LoadTextureFromImage(Image image) {
  Texture2D t;
  if (!image.data || image.width <= 0 || image.height <= 0)
    return t;
  const auto* p = static_cast<const unsigned char*>(image.data);
  t.pixels      = std::make_shared<std::vector<unsigned char>>(
      p, p + static_cast<std::size_t>(image.width) * image.height * 4);
  t.id     = 1;
  t.width  = image.width;
  t.height = image.height;
  t.format = kPixelFormatR8G8B8A8;
  return t;
}
Image LoadImageFromTexture(Texture2D t) {
  if (!t.pixels)
    return {};
  Image i = GenImageColor(t.width, t.height, {0, 0, 0, 0});
  std::memcpy(i.data, t.pixels->data(), t.pixels->size());
  return i;
}
std::vector<unsigned char> ExportImageToMemory(Image i, const char* ext) {
  if (!i.data || i.width <= 0 || i.height <= 0)
    return {};
  const std::string suffix(ext);
  const auto        bytes =
      (suffix == ".jpg" || suffix == ".jpeg")
                 ? jpeg::Encode(i.width, i.height, 4, static_cast<unsigned char*>(i.data), i.width * 4, 90)
                 : png::Encode(i.width, i.height, 4, static_cast<unsigned char*>(i.data), i.width * 4);
  return {bytes.begin(), bytes.end()};
}
bool ExportImage(Image i, const char* path) {
  auto bytes = ExportImageToMemory(i, std::string(path).ends_with(".jpg") ? ".jpg" : ".png");
  std::ofstream f(path, std::ios::binary);
  f.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  return !bytes.empty() && bool(f);
}
} // namespace datamunge::cad::detail::gfx
