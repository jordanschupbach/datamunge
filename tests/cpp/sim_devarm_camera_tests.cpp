#include <gtest/gtest.h>

#include <datamunge/sim/devarm_camera.hpp>
#include <filesystem>
namespace s = datamunge::sim;
namespace {
std::string fixture() {
  return (std::filesystem::path(__FILE__).parent_path().parent_path() /
          "fixtures/devarm_pick/robot.urdf")
      .string();
}
} // namespace
TEST(DevArmCamera, ExtrinsicsFollowRealWristAndNeverFruitMetadata) {
  s::DevArmPickEnvironment env(fixture());
  s::DevArmWristCamera     camera(env, 32);
  auto                     a  = env.reset(0);
  auto                     ca = camera.pose(a);
  auto                     b  = env.reset(3);
  auto                     cb = camera.pose(b);
  EXPECT_DOUBLE_EQ(ca.position.x, cb.position.x);
  EXPECT_DOUBLE_EQ(ca.target.z, cb.target.z);
  b.fruit       = {20, -40, 80};
  b.kind        = s::FruitKind::Apple;
  b.destination = s::PickBin::Yellow;
  auto same     = camera.pose(b);
  EXPECT_DOUBLE_EQ(same.position.y, cb.position.y);
  EXPECT_DOUBLE_EQ(same.target.x, cb.target.x);
  b.joints[0] += .2;
  auto moved = camera.pose(b);
  EXPECT_GT((moved.position - cb.position).Length(), .02);
  EXPECT_NEAR((moved.target - moved.position).Length(), (cb.target - cb.position).Length(), 1e-12);
  EXPECT_THROW(s::DevArmWristCamera(env, 0), std::invalid_argument);
  auto housing = camera.housing(a);
  EXPECT_EQ(housing.objects.size(), 3);
}
TEST(DevArmCamera, CachedBackgroundStillRendersPayloadAndReturnsOwnedPixels) {
  s::DevArmPickEnvironment env(fixture());
  s::DevArmWristCamera     camera(env, 32);
  auto                     f     = env.reset(0);
  auto                     first = camera.capture(f);
  ASSERT_EQ(first.rgba.size(), 32 * 32 * 4);
  auto copy   = first.rgba;
  auto repeat = camera.capture(f);
  EXPECT_EQ(first.rgba, repeat.rgba);
  f.fruit.x += .04;
  auto shifted = camera.capture(f);
  EXPECT_NE(copy, shifted.rgba);
  EXPECT_EQ(first.rgba, copy);
  f.kind = s::FruitKind::Banana;
  EXPECT_THROW(camera.capture(f), std::invalid_argument);
}
TEST(DevArmPick, GoalComesFromCallerAndPersistsWithoutFollowingTrueFruitClass) {
  s::DevArmPickEnvironment env(fixture());
  env.reset(0);
  env.set_destination(s::PickBin::Yellow);
  EXPECT_EQ(env.observation().destination, s::PickBin::Yellow);
  EXPECT_EQ(env.reset(3).destination, s::PickBin::Yellow);
  EXPECT_THROW(env.set_destination(static_cast<s::PickBin>(99)), std::invalid_argument);
  env.step(s::PickAction::Wait);
  EXPECT_THROW(env.set_destination(s::PickBin::Red), std::logic_error);
  auto trained = s::train_red_bin(env, 1, 42);
  EXPECT_EQ(trained.destination, s::PickBin::Red);
  EXPECT_EQ(env.observation().destination, s::PickBin::Red);
  auto yellow = s::train_pick_bin(env, s::PickBin::Yellow, 1, 42);
  EXPECT_EQ(yellow.destination, s::PickBin::Yellow);
  EXPECT_THROW(s::train_pick_bin(env, static_cast<s::PickBin>(99), 1), std::invalid_argument);
}
TEST(DevArmPick, YellowGoalRequiresPhysicalYellowPlacementForEitherTrueClass) {
  s::DevArmPickEnvironment env(fixture());
  // A fixed geometric regression route, never used by training or the deployed controller.
  const std::vector<int> route{5, 5, 5, 5, 6, 2, 2, 4, 2, 4, 4, 4, 2, 1,
                               4, 1, 2, 4, 2, 2, 2, 2, 2, 2, 0, 5, 7};
  for (unsigned seed : {0, 3}) {
    env.reset(seed);
    env.set_destination(s::PickBin::Yellow);
    s::PickObservation last;
    for (int a : route)
      last = env.step(static_cast<s::PickAction>(a));
    EXPECT_TRUE(last.success);
    EXPECT_TRUE(last.released);
    EXPECT_FALSE(last.grasped);
    EXPECT_GT(last.fruit.y, .30);
    EXPECT_LT(last.fruit.z, .56);
  }
  env.reset(3);
  env.set_destination(s::PickBin::Red);
  s::PickObservation wrong;
  for (int a : route)
    wrong = env.step(static_cast<s::PickAction>(a));
  EXPECT_FALSE(wrong.success);
  EXPECT_TRUE(wrong.terminated);
  EXPECT_GT(wrong.fruit.y, .30);
}
TEST(DevArmCamera, RgbLocalizationFollowsVisibleFruitAndNeverAnnotatesModelPixels) {
  s::DevArmPickEnvironment env(fixture(), 80, s::FruitVariation::Diverse);
  s::DevArmWristCamera     camera(env, 128, 95);
  auto                     frame     = env.reset(7);
  auto                     raw       = camera.capture(frame);
  auto                     inspected = camera.inspect(frame);
  ASSERT_EQ(raw.rgba, inspected.rgb.rgba);
  ASSERT_GT(inspected.box.width, 0);
  EXPECT_GT(inspected.foreground_pixels, 9);
  EXPECT_EQ(inspected.crop.rgba.size(), 64 * 64 * 4);
  auto shifted = frame;
  shifted.fruit.y += .04;
  EXPECT_NE(camera.inspect(shifted).box.x, inspected.box.x);
  // RGB localizer has no semantic class input: appearance and pose are held constant.
  frame.fruit = {100, 100, 100};
  auto empty  = camera.inspect(frame);
  EXPECT_EQ(empty.box.width, 0);
  EXPECT_TRUE(empty.crop.rgba.empty());
  EXPECT_EQ(camera.capture(shifted).rgba, camera.inspect(shifted).rgb.rgba);
  EXPECT_THROW(camera.inspect(shifted, 0), std::invalid_argument);
  EXPECT_THROW(s::DevArmWristCamera(env, 64, 125), std::invalid_argument);
}
TEST(DevArmPick, DiverseFeedUsesPhysicalBeltAndObservedLateralState) {
  s::DevArmPickEnvironment env(fixture(), 80, s::FruitVariation::Diverse);
  EXPECT_EQ(env.state_space_size(), 8911);
  auto       a     = env.reset_feed(7);
  const auto start = a.fruit;
  EXPECT_EQ(a.feed_stage, s::ConveyorStage::Travelling);
  EXPECT_THROW(env.step(s::PickAction::Close), std::logic_error);
  int frames = 0;
  env.record([&](const auto&) { ++frames; });
  for (int i = 0; i < 900 && a.feed_stage == s::ConveyorStage::Travelling; ++i)
    a = env.advance_feed(1. / 120);
  EXPECT_EQ(a.feed_stage, s::ConveyorStage::Waiting);
  EXPECT_GT(frames, 50);
  EXPECT_GT(a.fruit.x - start.x, .7);
  EXPECT_NEAR(a.fruit.y, start.y, .002);
  EXPECT_LT(a.state, env.state_space_size());
  EXPECT_EQ(a.decisions, 0);
  env.step(s::PickAction::Wait);
  EXPECT_THROW(env.advance_feed(1. / 120), std::logic_error);
}
TEST(DevArmCamera, RealisticCameraUsesTheSameFullSceneForRgbAndEmptyReference) {
  s::DevArmPickEnvironment env(fixture(), 80, s::FruitVariation::Diverse);
  env.set_visual_style(s::WorkcellVisualStyle::Realistic);
  s::DevArmWristCamera camera(env, 64, 95);
  auto                 frame = env.reset(7);
  auto                 input = camera.inspect_actors(frame, env.fruit_scene(frame), frame.fruit);
  EXPECT_EQ(input.rgb.rgba, camera.capture_actors(frame, env.fruit_scene(frame)).rgba);
  ASSERT_GT(input.foreground_pixels, 9);
  EXPECT_GT(input.box.width, 0);
  EXPECT_LT(input.box.width, 64);
  EXPECT_EQ(input.crop.rgba.size(), 64 * 64 * 4);
  frame.fruit = {100, 100, 100};
  auto empty  = camera.inspect_actors(frame, env.fruit_scene(frame), frame.fruit);
  EXPECT_EQ(empty.box.width, 0);
  EXPECT_TRUE(empty.crop.rgba.empty());
}
TEST(DevArmCamera, WorkerBudgetAndPreparedPoseCachePreservePixels) {
  s::DevArmPickEnvironment env(fixture(), 80, s::FruitVariation::Diverse);
  env.set_visual_style(s::WorkcellVisualStyle::Realistic);
  s::DevArmWristCamera camera(env, 48, 95);
  auto                 frame  = env.reset(7);
  auto                 actors = env.fruit_scene(frame);
  camera.set_render_threads(1);
  auto first = camera.capture_actors(frame, actors);
  camera.set_render_threads(4);
  auto second = camera.capture_actors(frame, actors);
  EXPECT_EQ(first.rgba, second.rgba);
  EXPECT_EQ(first.depth, second.depth);
  EXPECT_EQ(first.object_ids, second.object_ids);
  EXPECT_THROW(camera.set_render_threads(-1), std::invalid_argument);
  EXPECT_THROW(camera.set_render_threads(257), std::invalid_argument);
  frame.joints[0] += .1;
  auto moved = camera.capture_actors(frame, actors);
  EXPECT_NE(first.rgba, moved.rgba);
  auto restored = env.observation();
  EXPECT_EQ(first.rgba, camera.capture_actors(restored, actors).rgba);
  env.set_visual_style(s::WorkcellVisualStyle::Simple);
  camera.capture_actors(restored, actors);
  env.set_visual_style(s::WorkcellVisualStyle::Realistic);
  EXPECT_EQ(first.rgba, camera.capture_actors(restored, actors).rgba);
}
TEST(DevArmCamera, RealisticRendererIsRepeatableAndValidatesSamplingControls) {
  s::ConveyorEnvironment env(s::FruitVariation::Diverse);
  env.set_visual_style(s::WorkcellVisualStyle::Realistic);
  auto scene   = env.static_scene();
  auto options = s::workcell_render_options(s::WorkcellVisualStyle::Realistic, 32, 24);
  scene.lights = s::workcell_lights(s::WorkcellVisualStyle::Realistic);
  datamunge::cad::Camera camera;
  camera.position     = {-1.5, -1.7, 2.4};
  camera.target       = {-.16, 0, .5};
  camera.vertical_fov = 43;
  auto a              = datamunge::cad::render(scene, camera, options);
  auto b              = datamunge::cad::render(scene, camera, options);
  EXPECT_EQ(a.rgba, b.rgba);
  options.shadow_samples = 0;
  EXPECT_THROW(datamunge::cad::render(scene, camera, options), std::invalid_argument);
  options.shadow_samples             = 4;
  options.ambient_occlusion_distance = -1;
  EXPECT_THROW(datamunge::cad::render(scene, camera, options), std::invalid_argument);
  EXPECT_THROW(
      s::workcell_render_options(s::WorkcellVisualStyle::Realistic, 0, 32), std::invalid_argument);
}
TEST(DevArmCamera, AddedVisualControlsPreserveLegacyAggregateInitialization) {
  datamunge::cad::RenderOptions legacy{
      32,
      24,
      1,
      1,
      datamunge::cad::RenderStyle::Shaded,
      {.9f, .9f, .9f, 1},
      {0, 0, 0, 1},
      1.,
      true,
      true,
      true,
      32};
  EXPECT_FALSE(legacy.physically_based);
  EXPECT_DOUBLE_EQ(legacy.studio_environment,0);
  EXPECT_EQ(legacy.shadow_samples, 1);
  EXPECT_EQ(legacy.transparency_layers, 32);
}

TEST(DevArmCamera, RealisticBaseImagesAreOwnedAndFollowExactJointAndFingerPose) {
  s::DevArmPickEnvironment env(fixture(), 80, s::FruitVariation::Diverse);
  env.set_visual_style(s::WorkcellVisualStyle::Realistic);
  s::DevArmWristCamera  camera(env, 48, 95);
  auto                  frame = env.reset(7);
  datamunge::cad::Scene empty;
  auto                  first = camera.capture_actors(frame, empty);
  auto                  saved = first;
  first.rgba[0] ^= 255;
  first.depth[0]      = 123;
  first.object_ids[0] = 999;
  auto second         = camera.capture_actors(frame, empty);
  EXPECT_EQ(second.rgba, saved.rgba);
  EXPECT_EQ(second.depth, saved.depth);
  EXPECT_EQ(second.object_ids, saved.object_ids);
  for (int step = 0; step < 3; ++step) {
    if (step == 0)
      frame.joints[0] += .1;
    else if (step == 1)
      frame.finger = .02;
    else
      frame = env.observation();
    s::DevArmWristCamera fresh(env, 48, 95);
    auto                 expected = fresh.capture_actors(frame, empty),
         actual                   = camera.capture_actors(frame, empty);
    EXPECT_EQ(actual.rgba, expected.rgba);
    EXPECT_EQ(actual.depth, expected.depth);
    EXPECT_EQ(actual.object_ids, expected.object_ids);
  }
  camera.set_render_threads(1);
  auto single = camera.capture_actors(frame, empty);
  camera.set_render_threads(4);
  EXPECT_EQ(single.rgba, camera.capture_actors(frame, empty).rgba);
  auto invalid = empty;
  invalid.textures.push_back({"bad", 1, 1, {}});
  EXPECT_THROW(camera.capture_actors(frame, invalid), std::invalid_argument);
  auto optical = camera.pose(frame);
  auto behind  = optical.position - (optical.target - optical.position).Normalized();
  EXPECT_THROW(camera.inspect_actors(frame, invalid, behind), std::invalid_argument);
}

TEST(DevArmCamera, RegisteredReferenceDoesNotCacheMovingNonFruitActors) {
  namespace c = datamunge::cad;
  s::DevArmPickEnvironment env(fixture(), 80, s::FruitVariation::Diverse);
  env.set_visual_style(s::WorkcellVisualStyle::Realistic);
  s::DevArmWristCamera camera(env, 64, 95);
  auto                 frame  = env.reset(7);
  auto                 actors = env.fruit_scene(frame);
  camera.inspect_actors(frame, actors, frame.fruit); // Prime the base-only reference.
  auto  id    = c::AddPrimitiveToScene(&actors, c::PrimitiveKind::Cube);
  auto* mark  = actors.FindObject(id);
  mark->name  = "moving calibration paddle";
  mark->scale = {.03f, .04f, .05f};
  for (int step = 0; step < 3; ++step) {
    mark->position = {
        float(frame.fruit.x + .04 * step), float(frame.fruit.y), float(frame.fruit.z)};
    auto                 actual = camera.inspect_actors(frame, actors, frame.fruit);
    s::DevArmWristCamera fresh(env, 64, 95);
    auto                 expected = fresh.inspect_actors(frame, actors, frame.fruit);
    EXPECT_EQ(actual.rgb.rgba, expected.rgb.rgba);
    EXPECT_EQ(actual.rgb.depth, expected.rgb.depth);
    EXPECT_EQ(actual.rgb.object_ids, expected.rgb.object_ids);
    EXPECT_EQ(actual.crop.rgba, expected.crop.rgba);
    EXPECT_EQ(actual.foreground_pixels, expected.foreground_pixels);
    EXPECT_EQ(actual.box.x, expected.box.x);
    EXPECT_EQ(actual.box.y, expected.box.y);
    EXPECT_EQ(actual.box.width, expected.box.width);
    EXPECT_EQ(actual.box.height, expected.box.height);
  }
  actors.objects.pop_back();
  auto                 actual = camera.inspect_actors(frame, actors, frame.fruit);
  s::DevArmWristCamera fresh(env, 64, 95);
  auto                 expected = fresh.inspect_actors(frame, actors, frame.fruit);
  EXPECT_EQ(actual.crop.rgba, expected.crop.rgba);
  EXPECT_EQ(actual.foreground_pixels, expected.foreground_pixels);
}

TEST(DevArmCamera, OffscreenFocusReturnsUnchangedRawCaptureAndEmptyDetection) {
  s::DevArmPickEnvironment env(fixture(), 80, s::FruitVariation::Diverse);
  for (auto style : {s::WorkcellVisualStyle::Simple, s::WorkcellVisualStyle::Realistic}) {
    env.set_visual_style(style);
    s::DevArmWristCamera camera(env, 48, 95);
    auto                 frame  = env.reset(7);
    auto                 actors = env.fruit_scene(frame);
    camera.inspect_actors(frame, actors, frame.fruit);
    auto optical  = camera.pose(frame);
    auto behind   = optical.position - (optical.target - optical.position).Normalized();
    auto expected = camera.capture_actors(frame, actors);
    auto actual   = camera.inspect_actors(frame, actors, behind);
    EXPECT_EQ(actual.rgb.rgba, expected.rgba);
    EXPECT_EQ(actual.rgb.depth, expected.depth);
    EXPECT_EQ(actual.rgb.object_ids, expected.object_ids);
    EXPECT_TRUE(actual.crop.rgba.empty());
    EXPECT_EQ(actual.foreground_pixels, 0);
    EXPECT_EQ(actual.box.width, 0);
    EXPECT_EQ(actual.box.height, 0);
  }
}

TEST(DevArmCamera, ParallelInspectionsMatchSingleWorkerAndRecoverAfterInvalidActors) {
  s::DevArmPickEnvironment env(fixture(), 80, s::FruitVariation::Diverse);
  env.set_visual_style(s::WorkcellVisualStyle::Realistic);
  s::DevArmWristCamera sequential(env, 48, 95), parallel(env, 48, 95);
  sequential.set_render_threads(1);
  parallel.set_render_threads(8);
  auto frame  = env.reset(7);
  auto actors = env.fruit_scene(frame);
  for (int step = 0; step < 3; ++step) {
    if (step == 1) {
      frame.joints[0] += .08;
      frame.finger = .025;
    } else if (step == 2)
      frame = env.observation();
    auto a = sequential.inspect_actors(frame, actors, frame.fruit),
         b = parallel.inspect_actors(frame, actors, frame.fruit);
    EXPECT_EQ(a.rgb.rgba, b.rgb.rgba);
    EXPECT_EQ(a.rgb.depth, b.rgb.depth);
    EXPECT_EQ(a.rgb.object_ids, b.rgb.object_ids);
    EXPECT_EQ(a.crop.rgba, b.crop.rgba);
    EXPECT_EQ(a.box.x, b.box.x);
    EXPECT_EQ(a.box.y, b.box.y);
    EXPECT_EQ(a.box.width, b.box.width);
    EXPECT_EQ(a.box.height, b.box.height);
    EXPECT_EQ(a.foreground_pixels, b.foreground_pixels);
  }
  auto invalid = actors;
  invalid.textures.push_back({"bad", 1, 1, {}});
  EXPECT_THROW(parallel.inspect_actors(frame, invalid, frame.fruit), std::invalid_argument);
  EXPECT_EQ(
      sequential.inspect_actors(frame, actors, frame.fruit).crop.rgba,
      parallel.inspect_actors(frame, actors, frame.fruit).crop.rgba);
}

TEST(WorkcellPresentation, OwnedDeterministicFinishesPreserveSensorAssetsAndPhysics) {
  namespace c = datamunge::cad;
  s::ConveyorEnvironment env(s::FruitVariation::Diverse);
  env.set_visual_style(s::WorkcellVisualStyle::Realistic);
  const auto before    = env.observation();
  auto       source    = env.static_scene();
  auto       colors    = source.objects;
  auto       texture   = source.textures.front().pixels;
  auto       decorated = s::workcell_presentation(source, true),
       again           = s::workcell_presentation(source, true);
  EXPECT_EQ(source.textures.front().pixels, texture);
  EXPECT_EQ(source.objects.size(), colors.size());
  EXPECT_GT(decorated.objects.size(), source.objects.size());
  EXPECT_EQ(decorated.objects.size(), again.objects.size());
  EXPECT_EQ(decorated.textures.back().pixels, again.textures.back().pixels);
  for (std::size_t i = 0; i < source.objects.size(); ++i) {
    EXPECT_EQ(source.objects[i].color.r, colors[i].color.r);
    EXPECT_EQ(source.objects[i].position.x, decorated.objects[i].position.x);
    EXPECT_EQ(source.objects[i].mesh_index, decorated.objects[i].mesh_index);
    EXPECT_EQ(
        source.meshes[source.objects[i].mesh_index].positions,
        decorated.meshes[decorated.objects[i].mesh_index].positions);
  }
  EXPECT_EQ(before.position, env.observation().position);
  EXPECT_EQ(before.rotation, env.observation().rotation);
  EXPECT_EQ(before.seed, env.observation().seed);
  auto options = s::workcell_presentation_options(80, 60);
  EXPECT_EQ(options.samples, 2);
  EXPECT_GT(options.studio_environment, 0);
  EXPECT_DOUBLE_EQ(
      s::workcell_render_options(s::WorkcellVisualStyle::Realistic, 80, 60).studio_environment, 0);
  EXPECT_THROW(s::workcell_presentation_options(0, 10), std::invalid_argument);
  EXPECT_THROW(s::workcell_presentation(source, false, {{0, 0, 0}}), std::invalid_argument);
  auto invalid = source;
  invalid.textures[0].pixels.clear();
  EXPECT_THROW(s::workcell_presentation(invalid), std::invalid_argument);
}

TEST(WorkcellPresentation, DetailedFruitAndCableMeshesRemainFiniteAndSourceOwned) {
  s::ConveyorEnvironment env(s::FruitVariation::Diverse);
  env.set_visual_style(s::WorkcellVisualStyle::Realistic);
  auto source = env.dynamic_scene();
  auto image  = source.textures.back().pixels;
  auto improved =
      s::workcell_presentation(source, false, {{0, 0, .8}, {.15, .1, 1.0}, {.25, .1, .9}});
  EXPECT_EQ(source.textures.back().pixels, image);
  EXPECT_GT(improved.objects.size(), source.objects.size());
  bool botanical = false, cable = false;
  for (const auto& t : improved.textures) {
    if (t.name == "presentation botanical pigment") {
      botanical = true;
      EXPECT_EQ(t.width, 512);
      EXPECT_EQ(t.height, 256);
    }
    EXPECT_EQ(t.pixels.size(), std::size_t(t.width) * t.height * 4);
  }
  for (const auto& o : improved.objects) {
    cable |= o.name == "presentation flexible servo harness";
    ASSERT_GE(o.mesh_index, 0);
    for (float v : improved.meshes[o.mesh_index].positions)
      EXPECT_TRUE(std::isfinite(v));
  }
  EXPECT_TRUE(botanical);
  EXPECT_TRUE(cable);
}
