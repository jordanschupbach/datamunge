#include <dm/dm.hpp>
#include <filesystem>
#include <iostream>

int main(int argc, char** argv) {
  const std::filesystem::path assets = argc > 1 ? argv[1] : "build/cad-assets/rebot";
  const std::filesystem::path output = argc > 2 ? argv[2] : "build/cad-demo";
  std::filesystem::create_directories(output);
  const auto urdf = assets / "Rebot_Arm_description/DM/urdf/ReBot_Arm_DM.urdf";
  auto       arm  = dm::cad::View::from_urdf(urdf.string());
  arm.set_joint("joint2", -.8);
  arm.set_joint("joint3", -.7);
  arm.set_joint("finger_left", .015); // finger_right follows the URDF mimic relation.
  arm.size(1100, 850);
  arm.samples(2);
  arm.light(.5, -1, 2, "#ffffff", 1.3);
  arm.light(-1, -.5, .5, "#dce8ff", .6);
  arm.render((output / "rebot_arm.png").string());
  std::cout << "| Model | Objects | Triangles | Independent joints |\n"
            << "| --- | ---: | ---: | ---: |\n"
            << "| Seeed reBot B601-DM | " << arm.object_count() << " | " << arm.triangle_count()
            << " | " << arm.joint_names().size() << " |\n";

  // Exact B-rep import of a separate manufactured arm link; units are mm
  // in this STEP file, while the URDF and its STL meshes use metres.
  const auto step = assets / "hardware/reBot_B601_DM/Metal_Parts/03_Link1.step";
  auto       part = dm::cad::View::from_file(step.string());
  part.size(900, 650);
  part.samples(2);
  part.light(.5, -1, 2, "#ffffff", 1.2);
  part.light(-1, .5, .5, "#dce8ff", .5);
  part.render((output / "rebot_link1.png").string());
  part.save_gltf((output / "rebot_link1.gltf").string());
  std::cout << "\nSTEP link: " << part.object_count() << " body, " << part.triangle_count()
            << " triangles.\n";
}
