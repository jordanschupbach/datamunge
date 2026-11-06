/// @file cad_prepared_render_test.cpp
/// @brief Independent prepared-scene ownership, actor/material, shadow and buffer regressions.
#include <datamunge/cad/render.hpp>
#include <cmath>
#include <future>
#include <iostream>
#include <limits>
#include <stdexcept>
using namespace datamunge::cad;
namespace {
/// @brief Reject failed invariants in Release builds as well as Debug builds.
void check(bool ok){if(!ok)throw std::runtime_error("prepared render invariant failed");}
/// @brief Require exact image, primary depth and source object-ID equality.
void same(const RenderResult& a,const RenderResult& b){
  check(a.width==b.width&&a.height==b.height&&a.rgba==b.rgba&&a.depth==b.depth&&a.object_ids==b.object_ids);
}
/// @brief Combine scenes with the same owned ID/texture remapping used by production callers.
Scene combine(Scene base,const Scene& actors){
  int mesh=base.meshes.size(),tex=base.textures.size();
  base.meshes.insert(base.meshes.end(),actors.meshes.begin(),actors.meshes.end());
  base.textures.insert(base.textures.end(),actors.textures.begin(),actors.textures.end());
  for(auto o:actors.objects){o.mesh_index+=mesh;for(int* p:{&o.texture_index,&o.normal_map_index,&o.roughness_map_index,&o.metallic_map_index})if(*p>=0)*p+=tex;base.AddObject(o);}
  return base;
}
}
int main(){
  Scene base;
  auto floor=AddPrimitiveToScene(&base,PrimitiveKind::Cube);
  base.FindObject(floor)->position={0,0,-.7f};base.FindObject(floor)->scale={6,6,.2f};
  auto box=AddPrimitiveToScene(&base,PrimitiveKind::Cube);
  base.FindObject(box)->position={-1,0,.1f};base.FindObject(box)->scale={-.6f,.8f,1.2f};
  base.AddLight(Light{}); // Actor IDs must include the base's light allocation.
  Camera camera;camera.position={3,-5,3};camera.target={0,0,0};
  RenderOptions options;options.width=96;options.height=72;options.threads=1;
  options.physically_based=true;options.shadow_samples=3;options.light_angular_radius=.07;
  options.ambient_occlusion_samples=2;
  PreparedScene prepared(base);
  auto snapshot=render(base,camera,options);
  same(prepared.render(camera,options),snapshot);
  auto owned=prepared;
  base.objects.clear(); // Snapshot/copies must outlive and ignore caller changes.
  same(prepared.render(camera,options),snapshot);
  same(prepared.render(camera,options),owned.render(camera,options));
  Scene actors;
  auto sphere=AddPrimitiveToScene(&actors,PrimitiveKind::Sphere);
  actors.FindObject(sphere)->position={1,0,0};actors.FindObject(sphere)->scale={.6f,.6f,.6f};
  actors.textures.push_back({"green",1,1,{30,210,80,255}});
  actors.FindObject(sphere)->texture_index=0;
  Scene source;
  auto id=AddPrimitiveToScene(&source,PrimitiveKind::Cube);
  source.FindObject(id)->position={-1.5f,0,0};
  source.AddLight(Light{});
  PreparedScene with_base(source);
  {
    PreparedScene fresh(source);
    auto plain=std::async(std::launch::async,[&]{return fresh.render(camera,options);});
    auto moving=std::async(std::launch::async,[&]{return fresh.render(actors,camera,options);});
    same(plain.get(),render(source,camera,options));
    same(moving.get(),render(combine(source,actors),camera,options));
  }
  for(auto style:{RenderStyle::Shaded,RenderStyle::Unlit,RenderStyle::Wireframe,RenderStyle::ShadedWireframe})
    for(int samples:{1,2}){
      options.style=style;options.samples=samples;
      same(with_base.render(actors,camera,options),render(combine(source,actors),camera,options));
    }
  options.style=RenderStyle::Shaded;options.samples=1;
  auto before=with_base.render(actors,camera,options);
  actors.FindObject(sphere)->position={.6f,.5f,.1f};
  actors.FindObject(sphere)->color.a=.35f;
  same(with_base.render(actors,camera,options),render(combine(source,actors),camera,options));
  check(before.rgba!=with_base.render(actors,camera,options).rgba);
  actors.objects.clear();same(with_base.render(actors,camera,options),with_base.render(camera,options));
  auto asynchronous=std::async(std::launch::async,[&]{return with_base.render(camera,options);});
  same(asynchronous.get(),with_base.render(camera,options));
  auto invalid=source;invalid.objects[0].scale.x=0;bool rejected=false;
  try{PreparedScene bad(invalid);}catch(const std::invalid_argument&){rejected=true;}check(rejected);
  options.exposure=std::numeric_limits<double>::quiet_NaN();rejected=false;
  try{with_base.render(camera,options);}catch(const std::invalid_argument&){rejected=true;}check(rejected);
  std::cout<<"Prepared scenes: exact color/depth/IDs, ownership, actors, maps, styles, samples, transparency, concurrency and validation passed\n";
}
