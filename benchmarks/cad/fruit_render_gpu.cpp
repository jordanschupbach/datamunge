/// @file fruit_render_gpu.cpp
/// @brief Four-pose CPU/CUDA comparisons; no video generation, training or full replay.
#include <cmath>
#include <iomanip>

#include "fruit_replay.hpp"

int main(int argc, char** argv) try {
  if (argc < 3 || argc > 4)
    throw std::invalid_argument("Usage: fruit_render_gpu_bench ROOT OUTPUT [REPETITIONS=3]");
  auto root = fs::absolute(argv[1]), output = fs::absolute(argv[2]);
  int  repeats = argc == 4 ? std::stoi(argv[3]) : 3;
  if (repeats < 1 || repeats > 5)
    throw std::invalid_argument("Repetitions must be 1..5; use short bounded comparisons");
  fs::create_directories(output);
  auto data   = root / "docs/mepml/examples/build/fruit_stream_realistic";
  auto frames = replay(data);
  auto info   = dm::compute::probe({dm::compute::Backend::CUDA, 0});
  if (!info.available)
    throw std::runtime_error(info.diagnostic);
  std::unique_ptr<c::CudaRenderer> gpu;
  double                initialize = timed([&] { gpu = std::make_unique<c::CudaRenderer>(); });
  s::FruitStreamOptions config;
  config.visual_style = s::WorkcellVisualStyle::Realistic;
  s::FruitStreamEnvironment env(
      (root / "build/cad-assets/rebot/Rebot_Arm_description/DM/urdf/ReBot_Arm_DM.urdf").string(),
      config);
  s::DevArmWristCamera wrist(env.arm_model(), 256, 95);
  auto                 scenery = env.arm_model().static_scene();
  scenery.lights               = s::workcell_lights(config.visual_style);
  c::Camera camera;
  camera.position     = {-1.5, -1.7, 2.4};
  camera.target       = {-.16, 0, .5};
  camera.vertical_fov = 43;
  auto options        = s::workcell_render_options(config.visual_style, 960, 640);
  options.threads     = 16;
  std::ofstream samples(output / "samples.csv");
  samples << "frame,repetition,backend,milliseconds\n";
  samples.precision(12);
  std::ofstream quality(output / "quality.json");
  quality << std::setprecision(12) << "{\"device\":" << std::quoted(info.name)
          << ",\"renderer_initialization_ms\":" << initialize << ",\"repetitions\":" << repeats
          << ",\"cpu_workers\":16,\"width\":960,\"height\":640,\"frames\":[";
  bool comma = false;
  for (int index : {37, 611, 712, 862}) {
    const auto& frame = frames.at(index);
    auto        scene = scenery;
    append(scene, env.arm_model().robot_scene(frame.arm));
    append(scene, wrist.housing(frame.arm));
    std::unique_ptr<c::PreparedScene> prepared;
    double                            prepare =
        timed([&] { prepared = std::make_unique<c::PreparedScene>(std::move(scene)); });
    auto            actors    = env.fruit_scene(frame);
    auto            reference = prepared->render(actors, camera, options);
    c::RenderResult device;
    double          cold = timed([&] { device = gpu->render(*prepared, actors, camera, options); });
    double          square = 0, absolute = 0, depth_square = 0, depth_max = 0;
    std::size_t     mismatch = 0, matching = 0, intersection = 0, union_count = 0;
    int             max_color = 0;
    for (std::size_t i = 0; i < reference.depth.size(); ++i) {
      bool a = reference.object_ids[i] >= 0, b = device.object_ids[i] >= 0;
      intersection += a && b;
      union_count += a || b;
      mismatch += reference.object_ids[i] != device.object_ids[i];
      if (a && b && reference.object_ids[i] == device.object_ids[i]) {
        double d = std::abs(reference.depth[i] - device.depth[i]);
        depth_square += d * d;
        depth_max = std::max(depth_max, d);
        ++matching;
      }
      for (int k = 0; k < 3; ++k) {
        int d = std::abs(int(reference.rgba[4 * i + k]) - int(device.rgba[4 * i + k]));
        square += double(d) * d;
        absolute += d;
        max_color = std::max(max_color, d);
      }
    }
    double rmse        = std::sqrt(square / (reference.depth.size() * 3)),
           id_fraction = double(mismatch) / reference.depth.size();
    if (comma)
      quality << ',';
    comma = true;
    quality << "{\"frame\":" << index << ",\"cpu_prepare_ms\":" << prepare
            << ",\"first_gpu_render_ms\":" << cold << ",\"rgb_rmse_255\":" << rmse
            << ",\"rgb_mae_255\":" << absolute / (reference.depth.size() * 3)
            << ",\"max_color_error\":" << max_color << ",\"id_mismatch_fraction\":" << id_fraction
            << ",\"foreground_iou\":" << (union_count ? double(intersection) / union_count : 1)
            << ",\"depth_rmse_matching_ids\":"
            << std::sqrt(depth_square / std::max(matching, std::size_t(1)))
            << ",\"max_depth_error_matching_ids\":" << depth_max << '}';
    reference.save((output / ("cpu_" + std::to_string(index) + ".png")).string());
    device.save((output / ("cuda_" + std::to_string(index) + ".png")).string());
    std::cout << "frame " << index << " RGB RMSE " << rmse << " ID mismatch " << id_fraction
              << " cold CUDA " << cold << " ms\n";
    if (rmse > 3 || id_fraction > .001)
      throw std::runtime_error("CUDA workcell comparison failed the explicit visual/geometry gate");
    for (int rep = 0; rep < repeats; ++rep)
      for (int k = 0; k < 2; ++k) {
        bool   cuda = (k + rep) % 2;
        double ms   = timed([&] {
          device = cuda ? gpu->render(*prepared, actors, camera, options)
                          : prepared->render(actors, camera, options);
        });
        samples << index << ',' << rep << ',' << (cuda ? "cuda" : "cpu") << ',' << ms << '\n';
        std::cout << (cuda ? "cuda" : "cpu") << " " << ms << " ms\n";
      }
  }
  quality << "]}\n";
} catch (const std::exception& e) {
  std::cerr << e.what() << '\n';
  return 1;
}
