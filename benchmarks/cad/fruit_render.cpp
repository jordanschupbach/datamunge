/// @file fruit_render.cpp
/// @brief Stratified CPU workcell benchmark with the original full-minute protocol.
#include "fruit_replay.hpp"

/// @brief Run replay samples; arguments are root, output, repetition count and optional worker budget.
int main(int argc, char** argv) try {
  if (argc != 4 && argc != 5) throw std::runtime_error("Usage: fruit_render_bench ROOT OUTPUT REPETITIONS [WORKERS]");
  fs::path root = fs::absolute(argv[1]), output = fs::absolute(argv[2]);
  int repeats = std::stoi(argv[3]);
  if (repeats < 1 || repeats > 20) throw std::runtime_error("Repetitions must be 1..20");
  int budget=argc==5?std::stoi(argv[4]):0;
  if(budget<0||budget>256)throw std::runtime_error("Worker budget must be 0..256");
  fs::create_directories(output);
  auto dir = root / "docs/mepml/examples/build/fruit_stream_realistic";
  dm::nn::set_num_threads(1); // Match the report; renderer threads are controlled separately.
  auto frames = replay(dir);
  auto urdf = root / "build/cad-assets/rebot/Rebot_Arm_description/DM/urdf/ReBot_Arm_DM.urdf";
  s::FruitStreamOptions config; config.visual_style = s::WorkcellVisualStyle::Realistic;
  s::FruitStreamEnvironment env(urdf.string(), config);
  s::DevArmWristCamera wrist(env.arm_model(),256,95);
  wrist.set_render_threads(budget);
  auto scenery = env.arm_model().static_scene(); scenery.lights = s::workcell_lights(config.visual_style);
  c::Camera overview; overview.position={-1.5,-1.7,2.4}; overview.target={-.16,0,.5}; overview.vertical_fov=43;
  auto options = s::workcell_render_options(config.visual_style,960,640);
  options.threads=budget;
  std::ofstream raw(output / "samples.csv"); raw.precision(12);
  raw << "suite,frame,repetition,stage,milliseconds\n";
  // Equal-width temporal strata cover the whole minute, including idle time and bursts.
  std::vector<int> selected;
  for (int i=0;i<12;++i) selected.push_back(37 + 75*i);
  auto active_it = std::find_if(frames.begin(),frames.end(),[](const auto& f){return f.target && !f.fruit.empty() && std::any_of(f.fruit.begin(),f.fruit.end(),[&](const auto& p){return p.id==f.target && p.prediction>=0;});});
  if(active_it==frames.end()) throw std::runtime_error("Replay contains no classified target");
  const auto& active = *active_it;
  int active_index = &active - frames.data();
  // A full warmup is excluded from all samples (filesystem/page caches are warm).
  auto warm_scene=scenery; append(warm_scene,env.arm_model().robot_scene(active.arm));
  auto warm_actors=env.fruit_scene(active); append(warm_scene,warm_actors);
  auto warm=c::render(warm_scene,overview,options); wrist.capture_actors(active.arm,warm_actors);
  std::unique_ptr<c::PreparedScene> overview_cache;
  std::array<double,6> cached_joints{};double cached_finger=0;
  for(int rep=0;rep<repeats;++rep) for(int index:selected) {
    // Sample the actual immediate predecessor's cache state, not the previous
    // sampled stratum (which would turn seven of twelve samples into false
    // cache misses; the recorded minute has only 68 pose changes / 900 frames).
    const auto& previous=frames[index-1];
    if(!overview_cache||cached_joints!=previous.arm.joints||cached_finger!=previous.arm.finger){
      auto fixed=scenery;append(fixed,env.arm_model().robot_scene(previous.arm));append(fixed,wrist.housing(previous.arm));
      overview_cache=std::make_unique<c::PreparedScene>(std::move(fixed));cached_joints=previous.arm.joints;cached_finger=previous.arm.finger;
    }
    wrist.capture_actors(previous.arm,env.fruit_scene(previous));
    const auto& f=frames[index]; c::Scene actors; c::RenderResult image,sensor;
    double sum=0;
    auto stage=[&](const std::string& name,auto&& fn){double ms=timed(fn);emit(raw,"pipeline",index,rep,name,ms);sum+=ms;};
    stage("actors",[&]{actors=env.fruit_scene(f);});
    stage("overview_prepare",[&]{
      if(!overview_cache||cached_joints!=f.arm.joints||cached_finger!=f.arm.finger){
        auto scene=scenery;append(scene,env.arm_model().robot_scene(f.arm));append(scene,wrist.housing(f.arm));
        overview_cache=std::make_unique<c::PreparedScene>(std::move(scene));cached_joints=f.arm.joints;cached_finger=f.arm.finger;
      }
    });
    stage("overview_render",[&]{image=overview_cache->render(actors,overview,options);});
    auto focus=std::find_if(f.fruit.begin(),f.fruit.end(),[&](const auto& p){return p.id==f.target;});
    stage(focus==f.fruit.end()?"wrist_capture":"wrist_inspect",[&]{
      if(focus==f.fruit.end())sensor=wrist.capture_actors(f.arm,actors);
      else { const auto& p=focus->body.position; sensor=wrist.inspect_actors(f.arm,actors,{p[0],p[2],p[1]}).rgb; }
    });
    stage("overlay",[&]{overlay(image,sensor);});
    stage("png_save",[&]{image.save((output/("sample_"+std::to_string(index)+".png")).string());});
    emit(raw,"pipeline",index,rep,"total",sum);
    std::cerr<<"pipeline "<<rep+1<<'/'<<repeats<<" frame "<<index<<" "<<sum<<" ms\n";
  }
  auto scene=scenery;append(scene,env.arm_model().robot_scene(active.arm));append(scene,wrist.housing(active.arm));append(scene,warm_actors);
  // Same prepared scene: render-only controls exclude asset generation and PNG output.
  std::vector<std::pair<std::string,c::RenderOptions>> variants;
  for(int threads:{1,2,4,8,16}) {auto o=options;o.threads=threads;variants.emplace_back("threads_"+std::to_string(threads),o);}
  auto o=options;o.ambient_occlusion_samples=0;variants.emplace_back("no_ao",o);
  o=options;o.shadow_samples=1;variants.emplace_back("one_shadow",o);
  o=options;o.shadows=false;o.ambient_occlusion_samples=0;variants.emplace_back("no_secondary_rays",o);
  o=options;o.width=640;o.height=426;variants.emplace_back("resolution_640x426",o);
  o=options;o.width=o.height=1;o.threads=1;o.shadows=false;o.ambient_occlusion_samples=0;variants.emplace_back("build_plus_one_pixel",o);
  std::vector<unsigned char> reference_pixels;
  for(int rep=0;rep<repeats;++rep) {
    // Rotate order to reduce a consistent warmup/temperature advantage.
    for(std::size_t k=0;k<variants.size();++k){const auto& [name,v]=variants[(k+rep)%variants.size()];
      auto ms=timed([&]{warm=c::render(scene,overview,v);});emit(raw,"render_control",active_index,rep,name,ms);
      if(name.starts_with("threads_")){
        if(reference_pixels.empty())reference_pixels=warm.rgba;
        else if(warm.rgba!=reference_pixels)throw std::runtime_error("Thread-count control changed pixels");
      }
      if(rep==0)warm.save((output/(name+".png")).string());
      std::cerr<<name<<" "<<ms<<" ms\n";
    }
    emit(raw,"component",active_index,rep,"static_scene_copy",timed([&]{auto x=env.arm_model().static_scene();if(x.objects.empty())throw std::runtime_error("Empty static scene");}));
    emit(raw,"component",active_index,rep,"robot_scene",timed([&]{auto x=env.arm_model().robot_scene(active.arm);if(x.objects.empty())throw std::runtime_error("Empty robot");}));
    emit(raw,"component",active_index,rep,"wrist_capture",timed([&]{warm=wrist.capture_actors(active.arm,warm_actors);}));
    const auto& p=*std::find_if(active.fruit.begin(),active.fruit.end(),[&](const auto& p){return p.id==active.target;});
    emit(raw,"component",active_index,rep,"wrist_inspect",timed([&]{warm=wrist.inspect_actors(active.arm,warm_actors,{p.body.position[0],p.body.position[2],p.body.position[1]}).rgb;}));
  }
  auto classifier=dm::nn::MobileNetClassifier::load((dir/"mobile.npz").string());
  const auto& p=*std::find_if(active.fruit.begin(),active.fruit.end(),[&](const auto& p){return p.id==active.target;});
  auto crop=wrist.inspect_actors(active.arm,warm_actors,{p.body.position[0],p.body.position[2],p.body.position[1]}).crop;
  if(crop.rgba.empty())throw std::runtime_error("Active replay crop is empty");
  auto tensor=[](const c::RenderResult& image){std::vector<float> rgb;rgb.reserve(image.width*image.height*3);for(std::size_t i=0;i<image.rgba.size();i+=4)for(int k=0;k<3;++k)rgb.push_back(image.rgba[i+k]/255.f);return dm::nn::from_vector(rgb,{1,image.height,image.width,3});};
  auto input=tensor(crop);classifier.predict(input);
  for(int rep=0;rep<repeats;++rep)
    emit(raw,"component",active_index,rep,"classifier_predict",timed([&]{for(int j=0;j<50;++j)classifier.predict(input).to_vector<float>();})/50);
  auto policy=s::StreamPolicy::load((dir/"policy.txt").string());
  std::ofstream outcomes(output/"simulation.csv");outcomes<<"repetition,seconds,arrivals,correct,wrong,missed,pending,frames\n";
  for(int rep=0;rep<repeats;++rep){
    env.set_perception([&](const s::StreamFrame& f,std::uint64_t id){
      const auto& p=*std::find_if(f.fruit.begin(),f.fruit.end(),[&](const auto& p){return p.id==id;});
      auto inspected=wrist.inspect_actors(f.arm,env.fruit_scene(f),{p.body.position[0],p.body.position[2],p.body.position[1]});
      if(inspected.crop.rgba.empty())return s::StreamPrediction{};
      auto prob=classifier.predict(tensor(inspected.crop)).to_vector<float>();int label=prob[1]>prob[0]?1:0;return s::StreamPrediction{label,prob[label]};
    });
    std::vector<s::StreamFrame> recorded;
    env.record([&](const auto& f){if(f.time<=60+1e-9)recorded.push_back(f);},15);
    auto ms=timed([&]{auto f=env.reset(2026113001ULL,.12);while(f.time<60)f=env.step(policy.action(f.state)).frame;});
    env.record({});
    if(recorded.size()!=901||std::abs(recorded.back().time-60)>1e-8)throw std::runtime_error("Simulation horizon mismatch");
    const auto& f=recorded.back();const auto& x=f.counts;
    outcomes<<rep<<','<<f.time<<','<<x.arrivals<<','<<x.correct<<','<<x.wrong<<','<<x.missed<<','<<x.arrivals-x.correct-x.wrong-x.missed<<','<<recorded.size()<<'\n';
    emit(raw,"simulation",-1,rep,"full_physics_actual_rgb_60s",ms);
    std::cerr<<"simulation "<<rep<<" "<<ms<<" ms\n";
  }
  env.set_perception({});
  std::size_t triangles=0,storage=0,texbytes=0;
  for(const auto& obj:scene.objects)if(obj.visible&&obj.mesh_index>=0)triangles+=scene.meshes.at(obj.mesh_index).indices.size()/3;
  for(const auto& m:scene.meshes)storage+=4*(m.positions.size()+m.normals.size()+m.texcoords.size()+m.indices.size());
  for(const auto& t:scene.textures)texbytes+=t.pixels.size();
  std::ofstream info(output/"scene.json");info<<"{\"objects\":"<<scene.objects.size()<<",\"stored_meshes\":"<<scene.meshes.size()<<",\"instanced_triangles\":"<<triangles<<",\"mesh_bytes\":"<<storage<<",\"texture_bytes\":"<<texbytes<<",\"active_frame\":"<<active_index<<",\"stratified_frames\":12,\"repetitions\":"<<repeats<<",\"worker_budget\":"<<budget<<",\"prepared_overview\":true,\"cache_primed_with_immediate_predecessor\":true}\n";
  return 0;
} catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
