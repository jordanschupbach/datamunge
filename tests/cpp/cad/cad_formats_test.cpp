#include <algorithm>
#include <bit>
#include <cstdint>
#include <datamunge/cad/cad.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

// The public port must remain usable beside the source application's headers.
#define MEP_MODEL3D_DOC_H
#define MEP_MODEL3D_BLEND_IMPORT_H
#include "detail/json.hpp"
using namespace datamunge::cad;
namespace {
using Bytes = std::vector<unsigned char>;
void u32(Bytes& b, std::uint32_t v) {
  for (int i = 0; i < 4; ++i)
    b.push_back(static_cast<unsigned char>(v >> (8 * i)));
}
void set(Bytes& b, std::size_t offset, std::uint32_t v) {
  for (int i = 0; i < 4; ++i)
    b.at(offset + i) = static_cast<unsigned char>(v >> (8 * i));
}
void f32(Bytes& b, float v) {
  u32(b, std::bit_cast<std::uint32_t>(v));
}
void tag(Bytes& b, const std::string& t) {
  b.insert(b.end(), t.begin(), t.end());
}
void write(const std::string& p, const Bytes& b) {
  std::ofstream f(p, std::ios::binary);
  f.write(reinterpret_cast<const char*>(b.data()), b.size());
}
void check(bool b) {
  if (!b)
    throw std::runtime_error("Model format oracle failed");
}
Bytes triangle_iqm() {
  Bytes      b(124);
  const char magic[] = "INTERQUAKEMODEL";
  std::copy(magic, magic + 16, b.begin());
  set(b, 16, 2);
  set(b, 20, 216);
  set(b, 36, 1);
  set(b, 40, 124);
  set(b, 44, 1);
  set(b, 48, 3);
  set(b, 52, 148);
  set(b, 56, 1);
  set(b, 60, 168);
  for (auto n : {0u, 0u, 0u, 3u, 0u, 1u})
    u32(b, n);
  for (auto n : {0u, 0u, 7u, 3u, 180u})
    u32(b, n);
  for (auto n : {0u, 1u, 2u})
    u32(b, n);
  for (float x : {-1.f, -1.f, 0.f, 1.f, -1.f, 0.f, 0.f, 1.f, 0.f})
    f32(b, x);
  return b;
}
Bytes voxel() {
  Bytes b;
  tag(b, "VOX ");
  u32(b, 150);
  tag(b, "MAIN");
  u32(b, 0);
  u32(b, 44);
  tag(b, "SIZE");
  u32(b, 12);
  u32(b, 0);
  u32(b, 1);
  u32(b, 1);
  u32(b, 1);
  tag(b, "XYZI");
  u32(b, 8);
  u32(b, 0);
  u32(b, 1);
  b.insert(b.end(), {0, 0, 0, 1});
  return b;
}
Bytes triangle_m3d() {
  Bytes b;
  tag(b, "3DMO");
  u32(b, 100);
  tag(b, "HEAD");
  u32(b, 16);
  f32(b, 1);
  u32(b,
      2u | (3u << 6) | (3u << 8) |
          (3u << 14)); // Float coordinates; byte vertex indices; no color/UV/skin fields.
  tag(b, "VRTS");
  u32(b, 56);
  for (auto p : {Vec3d{-1, -1, 0}, Vec3d{1, -1, 0}, Vec3d{0, 1, 0}}) {
    f32(b, p.x);
    f32(b, p.y);
    f32(b, p.z);
    f32(b, 1);
  }
  tag(b, "MESH");
  u32(b, 12);
  b.insert(b.end(), {0x30, 0, 1, 2});
  tag(b, "OMD3");
  u32(b, 8);
  return b;
}
} // namespace
int main() {
  std::filesystem::create_directories("cad-format-artifacts");
  write("cad-format-artifacts/triangle.iqm", triangle_iqm());
  write("cad-format-artifacts/cube.vox", voxel());
  write("cad-format-artifacts/triangle.m3d", triangle_m3d());
  for (auto name : {"triangle.iqm", "triangle.m3d", "cube.vox"}) {
    auto s = load_scene("cad-format-artifacts/" + std::string(name));
    check(s.objects.size() == 1);
    check(s.TotalTriangleCount() == (std::string(name) == "cube.vox" ? 12 : 1));
    RenderOptions options;
    options.width  = 48;
    options.height = 48;
    auto image     = render(s, fit_camera(s), options);
    check(std::any_of(image.object_ids.begin(), image.object_ids.end(), [](int id) {
      return id != -1;
    }));
  }
  auto bad = triangle_iqm();
  bad.resize(124);
  write("cad-format-artifacts/truncated.iqm", bad);
  bool rejected = false;
  try {
    load_scene("cad-format-artifacts/truncated.iqm");
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  check(rejected);
  auto box = Solid::box(1, 1, 1);
  box.save("cad-format-artifacts/box.gltf");
  std::ifstream input("cad-format-artifacts/box.gltf");
  std::string   json{std::istreambuf_iterator<char>(input), {}};
  while (json.size() % 4)
    json += ' ';
  Bytes glb;
  u32(glb, 0x46546c67);
  u32(glb, 2);
  u32(glb, static_cast<uint32_t>(20 + json.size()));
  u32(glb, static_cast<uint32_t>(json.size()));
  u32(glb, 0x4e4f534a);
  tag(glb, json);
  write("cad-format-artifacts/box.glb", glb);
  check(load_scene("cad-format-artifacts/box.glb").TotalTriangleCount() == 12);
  // A glTF column-major matrix translates the whole unit cube. This
  // oracle is independent of the importer's internal Matrix memory layout.
  detail::Json matrix_document;
  check(detail::Json::Parse(json, &matrix_document));
  auto placement = detail::Json::Array();
  for (int value : {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 5, 2, 1, 1})
    placement.push_back(value);
  auto node      = detail::Json::Object();
  node["mesh"]   = 0;
  node["matrix"] = placement;
  auto nodes     = detail::Json::Array();
  nodes.push_back(node);
  matrix_document["nodes"] = nodes;
  const auto matrix_text   = matrix_document.dump();
  write("cad-format-artifacts/matrix.gltf", Bytes(matrix_text.begin(), matrix_text.end()));
  auto translated = load_scene("cad-format-artifacts/matrix.gltf");
  check(translated.objects.size() == 1);
  Vec3d       minimum{1e30, 1e30, 1e30}, maximum{-1e30, -1e30, -1e30};
  const auto& positions = translated.meshes.front().positions;
  for (std::size_t i = 0; i < positions.size(); i += 3)
    for (int axis = 0; axis < 3; ++axis) {
      minimum[axis] = std::min(minimum[axis], double(positions[i + axis]));
      maximum[axis] = std::max(maximum[axis], double(positions[i + axis]));
    }
  check((minimum - Vec3d{5, 2, 1}).Length() < 1e-6);
  check((maximum - Vec3d{6, 3, 2}).Length() < 1e-6);
  std::cout << "IQM, VOX, M3D and GLB import/render oracles passed\n";
}
