#pragma once
/// @file rebot_sources.hpp
/// @brief Pinned Seeed DevArm source assets; mirrors examples/cad/rebot-assets.json.
#include <cstdint>

namespace datamunge::datasets::detail {
/// @brief One immutable source fingerprint from the Seeed CAD asset manifest.
struct RebotSource {
  const char*   path;   ///< Repository-relative path, also the local cache layout.
  std::uint64_t size;   ///< Exact byte length before SHA-256 verification.
  const char*   sha256; ///< Expected lowercase SHA-256 digest.
};
/// @brief Pinned upstream revision for the DevArm CAD and URDF example.
inline constexpr const char* rebot_revision = "922c5fa3d8c8919db1847b49166f02fc26c8fdad";
/// @brief Source license, URDF, STEP geometry and every visual/collision STL dependency.
inline constexpr RebotSource rebot_sources[] = {
    {"LICENSE", 14392, "e258a9c6833bfd17fb90252df8c33e29d13fabc2768508248927824c17878db7"},
    {"Rebot_Arm_description/DM/urdf/ReBot_Arm_DM.urdf", 16263, "bce0d4740ec97a27e30d521e890416231c6d0117093fa997d1ae7ac95204ffb6"},
    {"hardware/reBot_B601_DM/Metal_Parts/03_Link1.step", 158334, "a3edd4ae72ce32c7b9e5517b19a950305bfff9b25d73f39ea848d2bdd3dee2f2"},
    {"hardware/reBot_B601_DM/reBot_B601_DM_v1.1_20260625.step", 34840241, "a1e301ee03d46589e26d3fc2272d19ba53c1878f0104274695e5c9a88aa512d2"},
    {"Rebot_Arm_description/DM/meshes/shared/colored_left_finger_carriage_grey.stl", 20084, "f650b231f1d366e0413db8bd4f6e9925442fb576841914aad57dde5a9bae5c46"},
    {"Rebot_Arm_description/DM/meshes/shared/colored_left_finger_travel_stop_yellow.stl", 54084, "2fdf7c97d0388fe3c8a5fe2408a1d97201243be729cb1f7a3b56c3916c92da53"},
    {"Rebot_Arm_description/DM/meshes/shared/colored_right_finger_carriage_grey.stl", 20084, "efa43d88c9bed0a66ac84a9cf26ac1c5326178dc93b4a01e0e1dbfa5b4430d22"},
    {"Rebot_Arm_description/DM/meshes/shared/colored_right_finger_travel_stop_yellow.stl", 54084, "1d28609e4761077beac32d127aaefa23ffb1a615de766ce0b71256344f735e0b"},
    {"Rebot_Arm_description/DM/meshes/visual/colored_base_link_hardware_black.stl", 1698884, "12deea0cd6f358ba23c53d27ff63e6ed4c5dfdc4014dd4dbef0e04966e0ec184"},
    {"Rebot_Arm_description/DM/meshes/visual/colored_base_link_matte_black.stl", 1993184, "a692eb4c625b6fa5be2f7d7831c8fb358aa4c7aa1cefa2ff1c133e22779cb33e"},
    {"Rebot_Arm_description/DM/meshes/visual/colored_base_link_silver_trim.stl", 387884, "da6fefa2ff02dd661b74a5454e7b090d91a3059249790af7241c4364edaf7283"},
    {"Rebot_Arm_description/DM/meshes/visual/colored_gripper_base_hardware_black.stl", 124284, "9e89b784b7db552d5fea7c5f57a8317f4d23ace19e594bb59e18ada37471569e"},
    {"Rebot_Arm_description/DM/meshes/visual/colored_gripper_base_metal.stl", 131484, "ac9032a968febcca070696e3f75f10d98727eb74548f1e502428a9b89ffdb187"},
    {"Rebot_Arm_description/DM/meshes/visual/colored_gripper_base_seeed_yellow.stl", 336784, "5632650c7506d5ea9699fd7fee674c937ec764f98ce2123d80a5094d94c2e6f1"},
    {"Rebot_Arm_description/DM/meshes/visual/colored_left_finger_finger_black.stl", 57184, "5473c38c7de9d4c8297e0a5134c14cfc8f9d6376b5d8257931ba4ae1b1e3dcda"},
    {"Rebot_Arm_description/DM/meshes/visual/colored_left_finger_rack_metal.stl", 107484, "938e863507227a7dd96bd4bb2196a7b3361b2718c44cd39bc2ffc39482ab5455"},
    {"Rebot_Arm_description/DM/meshes/visual/colored_link1_anodized_grey.stl", 123184, "8ba2520c9321fc5532f4ce52b2cf39a8f42f5f6eaef86524b313ea2a3942e5fa"},
    {"Rebot_Arm_description/DM/meshes/visual/colored_link1_hardware_black.stl", 17884, "4825fcb6b569439a9dc6cce09547cb143316597a7fe0f63bd727e7bb42390ea9"},
    {"Rebot_Arm_description/DM/meshes/visual/colored_link1_matte_black.stl", 196284, "e4f57546af71031f144fc67b59c4c64a93764ac25f6ed079fc5c2c4ee54fd768"},
    {"Rebot_Arm_description/DM/meshes/visual/colored_link2_anodized_grey.stl", 477684, "a0ceb3fddf2f65db57f7e2518febc124d6847d8ce199ae4073e9902432628c94"},
    {"Rebot_Arm_description/DM/meshes/visual/colored_link2_hardware_black.stl", 731984, "f50c02f1568b636edffd1a5eb5aab1abe4add9f3731c51c1f061525a43daacf2"},
    {"Rebot_Arm_description/DM/meshes/visual/colored_link2_matte_black.stl", 3785384, "f12463e6f38763bd96b3907d58bdbab4a93e297de72e891c28df02d701af984b"},
    {"Rebot_Arm_description/DM/meshes/visual/colored_link2_seeed_yellow.stl", 97084, "56a5cd3550a46626f528a67de323e4835757140ed3d784bb5455a7db6442be39"},
    {"Rebot_Arm_description/DM/meshes/visual/colored_link3_anodized_grey.stl", 608584, "b8687a7fdd6aef7d764a7364ae42d7637dc4f6c38e0a258e9d8b23c453f567bd"},
    {"Rebot_Arm_description/DM/meshes/visual/colored_link3_hardware_black.stl", 990684, "e983eb537d06b7427e4391a7cbe343d8e03667033f32d2b20a98ab9ff55540ee"},
    {"Rebot_Arm_description/DM/meshes/visual/colored_link3_matte_black.stl", 1997834, "7dd5ecba918916ca4f7322731ea69a7b5012fa9a18ebd3d6017e6950610035e1"},
    {"Rebot_Arm_description/DM/meshes/visual/colored_link3_seeed_yellow.stl", 258234, "39ab75600959b4ef82a10ff9e9e1c6fdd74a51ed5fb5ee192df5b8da69d2ae36"},
    {"Rebot_Arm_description/DM/meshes/visual/colored_link4_anodized_grey.stl", 294084, "4ce8aff6c90e347c3bdf72506a681777e70b061d2df62fb9441bf350743093db"},
    {"Rebot_Arm_description/DM/meshes/visual/colored_link4_hardware_black.stl", 605884, "9e26076a2565f53ccd9bd167947dabc7dd9c1186866094016d270b118a5c1234"},
    {"Rebot_Arm_description/DM/meshes/visual/colored_link4_matte_black.stl", 1891684, "faeffab64bf462b2c07a889a0fa70a5f326ab069e61f166f634f2729b45d4679"},
    {"Rebot_Arm_description/DM/meshes/visual/colored_link4_seeed_yellow.stl", 67884, "380299025b1e0ebb312b8f91155cc69120b6b16756b8dbb34172ac425f252640"},
    {"Rebot_Arm_description/DM/meshes/visual/colored_link5_anodized_grey.stl", 179984, "b1b4caeba17d30bb7999ccd54bfbaa8227d8a76a37cd0e6c66414455383fc24f"},
    {"Rebot_Arm_description/DM/meshes/visual/colored_link5_hardware_black.stl", 352484, "75adf40d6c770d6d391134aa6fd380c1e18ff91ddda99c6472465ba3d48b8717"},
    {"Rebot_Arm_description/DM/meshes/visual/colored_link5_matte_black.stl", 1726884, "c4555b72ccfb8cd0af6876e823bd8e2f19ba42e555289f6e851073988cb19cae"},
    {"Rebot_Arm_description/DM/meshes/visual/colored_link6_hardware_black.stl", 146384, "e4dedf8865e4cbf055701ccd3b3bf220b9d16b2fbc1cafabfb57fd9001aceb3d"},
    {"Rebot_Arm_description/DM/meshes/visual/colored_link6_matte_black.stl", 1848184, "4d122a7978946ba06d67f7dd03dbd81436a328b8856bcd60fa67fbf95dc1954d"},
    {"Rebot_Arm_description/DM/meshes/visual/colored_right_finger_finger_black.stl", 57184, "caf83c51fb2d98cba27414ba903297a76f0bdbc22513b3e97b3cff44ecb80693"},
    {"Rebot_Arm_description/DM/meshes/visual/colored_right_finger_rack_metal.stl", 107484, "0ae753bbf754c48ad918195c310735a397ae4d0d8250297af1ef3afe65d6d9f5"},
    {"Rebot_Arm_description/DM/meshes/collision/base_link.STL", 4079784, "14d98e291f5eb20ce50886fcbc99427606a46ea5cc60fe3e89326e7c45b3d73f"},
    {"Rebot_Arm_description/DM/meshes/collision/link1.STL", 337184, "33fc9ebe2654cc27d47edd97fb90b13d2611ed2eba734332b9110d93c24cc25d"},
    {"Rebot_Arm_description/DM/meshes/collision/link2.STL", 5091884, "3dc33c484f02966f43c37b38d848669d31018140a56e3ac5c9366add36cc3fa0"},
    {"Rebot_Arm_description/DM/meshes/collision/link3.STL", 3855084, "783383016dbe0a258bd94e7fffb66c7056c9aa9dd42b61e5abb24cb228a99035"},
    {"Rebot_Arm_description/DM/meshes/collision/link4.STL", 2859284, "95364d6a7008b62fe173f447219c2ce2cd2e6f0882997374cb1d8f60652c5848"},
    {"Rebot_Arm_description/DM/meshes/collision/link5.STL", 2259184, "19ea62cabf41394d4de0ce4437298484227f37fe5e0f99c20cea2d260ede45e8"},
    {"Rebot_Arm_description/DM/meshes/collision/link6.STL", 1994484, "6315a8ae1c03a2806f9e3ade2ca7e8740f34dfe05898fc02b620b0a2e459546a"},
    {"Rebot_Arm_description/DM/meshes/collision/gripper_base.stl", 592384, "5e7b0473acaf31fc5c511a3591311e6561ad6066af8e135e21e18c096e1e7da5"},
    {"Rebot_Arm_description/DM/meshes/collision/left_finger.stl", 238584, "cf8a3acbbfbcaf15e3de0663e80956a1601d3595ea451107174620e8bfcbeb5c"},
    {"Rebot_Arm_description/DM/meshes/collision/right_finger.stl", 238584, "e4b5ebef66e0b2b3e5d6e0ab59342716cca585db94c5bd4fc95b79f21c6657ef"},
};
} // namespace datamunge::datasets::detail
