/// @file cad_studio_light_test.cpp
/// @brief Opt-in studio lighting, legacy defaults and unchanged geometry buffers.
#include <datamunge/cad/render.hpp>
#include <iostream>
#include <limits>
#include <stdexcept>
using namespace datamunge::cad;
int main() {
  try {
    Scene scene;
    auto  id                        = AddPrimitiveToScene(&scene, PrimitiveKind::Sphere);
    scene.FindObject(id)->metallic  = .8f;
    scene.FindObject(id)->roughness = .25f;
    Light off;
    off.visible = false;
    scene.AddLight(off);
    Camera        camera = fit_camera(scene);
    RenderOptions o;
    o.width              = 64;
    o.height             = 48;
    o.physically_based   = true;
    o.shadows            = false;
    o.threads            = 1;
    auto legacy          = render(scene, camera, o);
    o.studio_environment = .7;
    auto lit             = render(scene, camera, o);
    if (legacy.rgba == lit.rgba || legacy.depth != lit.depth || legacy.object_ids != lit.object_ids)
      throw std::runtime_error("studio radiance failed to change only color");
    o.threads     = 4;
    auto parallel = render(scene, camera, o);
    if (lit.rgba != parallel.rgba)
      throw std::runtime_error("studio lighting depends on worker count");
    o.style              = RenderStyle::Unlit;
    auto a               = render(scene, camera, o);
    o.studio_environment = 0;
    auto b               = render(scene, camera, o);
    if (a.rgba != b.rgba)
      throw std::runtime_error("unlit surfaces unexpectedly use environment lighting");
    for (double bad : {-1., 5., std::numeric_limits<double>::quiet_NaN()}) {
      o.studio_environment = bad;
      bool failed          = false;
      try {
        render(scene, camera, o);
      } catch (const std::invalid_argument&) {
        failed = true;
      }
      if (!failed)
        throw std::runtime_error("invalid studio strength accepted");
    }
    std::cout
        << "Studio lighting is opt-in, geometry-preserving, worker-independent and validated\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
