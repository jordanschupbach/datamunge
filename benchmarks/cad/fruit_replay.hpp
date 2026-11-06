#pragma once
/// @file fruit_replay.hpp
/// @brief Shared recorded-pose reconstruction and HUD helpers for bounded workcell benchmarks.
#include <dm/dm.hpp>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace {
namespace c = dm::cad;
namespace s = dm::sim;
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
using Row = std::map<std::string, std::string>;

/// @brief Read the unquoted numeric CSV files emitted by the native playground.
std::vector<Row> csv(const fs::path& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("Missing replay input: " + path.string());
  std::string line, field;
  std::getline(in, line);
  std::vector<std::string> keys;
  std::istringstream header(line);
  while (std::getline(header, field, ',')) keys.push_back(field);
  std::vector<Row> rows;
  while (std::getline(in, line)) {
    std::istringstream values(line);
    Row row;
    for (const auto& key : keys) {
      if (!std::getline(values, field, ',')) throw std::runtime_error("Malformed replay CSV");
      row[key] = field;
    }
    rows.push_back(std::move(row));
  }
  return rows;
}
/// @brief Time synchronous owned work in milliseconds, including its normal allocations.
template<class F> double timed(F&& f) {
  auto begin = Clock::now();
  f();
  return std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
}
/// @brief Append world-space objects using the report's exact mesh/material remapping.
void append(c::Scene& scene, c::Scene extra) {
  int base = scene.meshes.size(), tex = scene.textures.size();
  for (auto& m : extra.meshes) scene.meshes.push_back(std::move(m));
  for (auto& t : extra.textures) scene.textures.push_back(std::move(t));
  for (auto o : extra.objects) {
    o.mesh_index += base; o.parent = -1;
    for (int* p : {&o.texture_index,&o.normal_map_index,&o.roughness_map_index,&o.metallic_map_index})
      if (*p >= 0) *p += tex;
    scene.AddObject(o);
  }
}
/// @brief Reconstruct actual recorded frames; geometry/appearance comes from each fruit's seed.
std::vector<s::StreamFrame> replay(const fs::path& dir) {
  auto poses = csv(dir / "trajectory.csv");
  if (poses.size() != 900) throw std::runtime_error("Expected 900 complete recorded frames");
  std::vector<s::StreamFrame> frames(poses.size());
  for (std::size_t i = 0; i < poses.size(); ++i) {
    const auto& r = poses[i]; auto& f = frames[i];
    if (std::stoul(r.at("frame")) != i) throw std::runtime_error("Nonsequential replay");
    f.time = std::stod(r.at("time")); f.target = std::stoull(r.at("target"));
    f.arm.grasped = std::stoi(r.at("grasped")); f.arm.finger = std::stod(r.at("finger"));
    for (int k = 0; k < 6; ++k) f.arm.joints[k] = std::stod(r.at("q" + std::to_string(k + 1)));
  }
  s::ConveyorEnvironment sampler(s::FruitVariation::Diverse);
  std::map<std::uint64_t, s::ConveyorObservation> appearance;
  for (const auto& r : csv(dir / "fruit_trajectory.csv")) {
    auto& f = frames.at(std::stoul(r.at("frame")));
    s::StreamFruit p; p.id = std::stoull(r.at("id")); p.seed = std::stoull(r.at("seed"));
    if (!appearance.contains(p.seed)) { sampler.reset(p.seed); appearance[p.seed] = sampler.observation(); }
    p.body = appearance.at(p.seed);
    p.body.position = {std::stod(r.at("x")),std::stod(r.at("z")),std::stod(r.at("y"))};
    p.body.rotation = {std::stod(r.at("qw")),std::stod(r.at("qx")),std::stod(r.at("qy")),std::stod(r.at("qz"))};
    p.arrival = std::stod(r.at("scheduled_arrival"));
    p.status = s::StreamFruitStatus(std::stoi(r.at("status")));
    p.prediction = std::stoi(r.at("prediction")); p.confidence = std::stod(r.at("confidence"));
    f.fruit.push_back(p);
  }
  return frames;
}
/// @brief Emit raw per-call samples; cases are deliberately serialized to avoid oversubscription.
void emit(std::ostream& out, const std::string& suite, int frame, int rep,
          const std::string& stage, double ms) {
  out << suite << ',' << frame << ',' << rep << ',' << stage << ',' << ms << '\n'; out.flush();
}
/// @brief Draw the same inset-sized blit and representative HUD text as the report.
void overlay(c::RenderResult& image, const c::RenderResult& sensor) {
  for (int y = 0; y < 256; ++y)
    std::copy_n(sensor.rgba.begin() + y * 256 * 4, 256 * 4,
                image.rgba.begin() + ((y + 20) * image.width + 20) * 4);
  c::draw_text(image,"T 40.7 S  POISSON 0.12/S  ARRIVED 2",25,578,2);
  c::draw_text(image,"SORTED 0  WRONG 0  MISSED 1  ACTIVE 1",25,602,2);
}
} // namespace
