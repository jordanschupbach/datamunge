/// @file fruit_wrist.cpp
/// @brief Bounded exact-buffer wrist benchmarks with immediate-predecessor cache priming.
#include <iomanip>

#include "fruit_replay.hpp"

int main(int argc, char** argv) try {
  if (argc < 3 || argc > 4)
    throw std::invalid_argument("Usage: fruit_wrist_bench ROOT OUTPUT [REPETITIONS=3]");
  auto root = fs::absolute(argv[1]), output = fs::absolute(argv[2]);
  int  repeats = argc == 4 ? std::stoi(argv[3]) : 3;
  if (repeats < 1 || repeats > 3)
    throw std::invalid_argument("Repetitions must be 1..3");
  fs::create_directories(output);
  auto                  frames = replay(root / "docs/mepml/examples/build/fruit_stream_realistic");
  s::FruitStreamOptions config;
  config.visual_style = s::WorkcellVisualStyle::Realistic;
  s::FruitStreamEnvironment env(
      (root / "build/cad-assets/rebot/Rebot_Arm_description/DM/urdf/ReBot_Arm_DM.urdf").string(),
      config);
  std::ofstream raw(output / "samples.csv");
  raw << "frame,repetition,mode,pose_changed,milliseconds\n";
  raw.precision(12);
  std::ofstream boxes(output / "boxes.csv");
  boxes << "frame,mode,x,y,width,height,foreground\n";
  /// @brief Persist exact native buffers; RGB byte comparisons do not hide depth/ID differences.
  auto save = [&](const std::string& name, const c::RenderResult& image) {
    std::ofstream rgba(output / (name + ".rgba"), std::ios::binary);
    rgba.write(reinterpret_cast<const char*>(image.rgba.data()), image.rgba.size());
    std::ofstream depth(output / (name + ".depth"), std::ios::binary);
    depth.write(
        reinterpret_cast<const char*>(image.depth.data()), image.depth.size() * sizeof(double));
    std::ofstream ids(output / (name + ".ids"), std::ios::binary);
    ids.write(
        reinterpret_cast<const char*>(image.object_ids.data()),
        image.object_ids.size() * sizeof(int));
  };
  for (int index : {37, 611, 637, 712}) {
    s::DevArmWristCamera camera(env.arm_model(), 256, 95);
    camera.set_render_threads(16);
    const auto& current  = frames[index];
    const auto& previous = frames[index - 1];
    auto        actors = env.fruit_scene(current), prior = env.fruit_scene(previous);
    bool        changed =
        current.arm.joints != previous.arm.joints || current.arm.finger != previous.arm.finger;
    /// @brief Reproduce the report's capture/inspection choice, including reference-cache state.
    auto view = [&](const s::StreamFrame& frame, const c::Scene& scene, bool offscreen = false) {
      s::WristInspection result;
      auto focus = std::find_if(frame.fruit.begin(), frame.fruit.end(), [&](const auto& p) {
        return p.id == frame.target;
      });
      if (focus == frame.fruit.end())
        result.rgb = camera.capture_actors(frame.arm, scene);
      else {
        auto p     = focus->body.position;
        auto point = c::Vec3d{p[0], p[2], p[1]};
        if (offscreen) {
          auto optical = camera.pose(frame.arm);
          point        = optical.position - (optical.target - optical.position).Normalized();
        }
        result = camera.inspect_actors(frame.arm, scene, point);
      }
      return result;
    };
    for (int rep = 0; rep < repeats; ++rep) {
      view(previous, prior);
      s::WristInspection result;
      auto               ms = timed([&] { result = view(current, actors); });
      raw << index << ',' << rep << ",video," << changed << ',' << ms << '\n';
      if (rep == 0) {
        save(std::to_string(index) + "_rgb", result.rgb);
        save(std::to_string(index) + "_crop", result.crop);
        boxes << index << ",video," << result.box.x << ',' << result.box.y << ','
              << result.box.width << ',' << result.box.height << ',' << result.foreground_pixels
              << '\n';
      }
      std::cout << index << " video " << ms << " ms\n";
    }
    if (current.target) {
      view(previous, prior);
      s::WristInspection result;
      auto               ms = timed([&] { result = view(current, actors, true); });
      raw << index << ",0,offscreen," << changed << ',' << ms << '\n';
      save(std::to_string(index) + "_offscreen", result.rgb);
      boxes << index << ",offscreen," << result.box.x << ',' << result.box.y << ','
            << result.box.width << ',' << result.box.height << ',' << result.foreground_pixels
            << '\n';
    }
  }
} catch (const std::exception& e) {
  std::cerr << e.what() << '\n';
  return 1;
}
