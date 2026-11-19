#pragma once

/// @file vsim_tasks.hpp
/// @brief Optional Gymnasium-style articulated environments backed by vsim and Jolt.

#include <datamunge/rl/environments/interface.hpp>
#include <vsim/io.hpp>
#include <vsim/physics.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <memory>
#include <numbers>

namespace datamunge::rl::environments {

/// @brief Select the observation, actuator and reward rules for an authored scene.
/// @par Example
/// @code
/// auto task = dm::rl::environments::VsimTask::Hopper;
/// @endcode
enum class VsimTask {
  InvertedPendulum, ///< Slider force and pole-angle termination.
  Hopper ///< Three hinge torques and torso health termination.
};

/// @brief Own a vsim scene and expose its articulated task as a Datamunge environment.
/// Enable DATAMUNGE_ENABLE_VSIM_TASKS before including dm/dm.hpp and link
/// against the vsim SDK. Each reset restores the authored state before
/// perturbing the bodies and joints with the environment RNG.
/// @par Example
/// @code
/// namespace e=dm::rl::environments;
/// e::VsimTaskEnv task(e::VsimTask::Hopper,"hopper.json");
/// const auto obs=task.reset(3);
/// const auto step=task.step(e::Action({0.f,0.f,0.f}));
/// @endcode
/// @see EnvBase
class VsimTaskEnv : public EnvBase {
public:
  /// @brief Construct a task from its authored vsim JSON scene.
  /// @param task Selects which joint names and task geometry are required.
  /// @param file Path to the corresponding JSON scene.
  /// @param horizon Maximum agent steps per episode, default 1000.
  /// @throws std::invalid_argument For unsupported tasks, invalid scene entities or horizon.
  /// @see VsimTaskEnv
  VsimTaskEnv(VsimTask task,const std::filesystem::path& file,int horizon=1000)
      : EnvBase(horizon), task_(task), initial_(vsim::load_scene(file)), scene_(initial_),
        world_(std::make_unique<vsim::PhysicsWorld>(scene_)), start_(world_->save_state()) {
    if (horizon<1 || (task!=VsimTask::InvertedPendulum && task!=VsimTask::Hopper))
      throw std::invalid_argument("invalid vsim task or horizon");
    root_=find(pendulum()?"cart":"torso");
    if (pendulum()) {
      joints_[0]=root_;
      joints_[1]=find("pole");
    } else {
      joints_={find("thigh"),find("leg"),find("foot")};
    }
    for (int i=0;i<(pendulum()?2:3);++i)
      if (!scene_.get(joints_[i]).joint) throw std::invalid_argument("vsim task needs the authored joints");
    if (!scene_.get(root_).rigid_body)
      throw std::invalid_argument("vsim task requires a root body");
  }

  /// @brief Keep the scene at the address borrowed by the physics world.
  /// @param other Environment whose ownership cannot be transferred.
  VsimTaskEnv(VsimTaskEnv&& other) = delete;

  /// @brief Return the task's canonical name.
  /// @return InvertedPendulum or Hopper.
  /// @see VsimTaskEnv
  std::string name() const override { return pendulum()?"InvertedPendulum":"Hopper"; }
  /// @brief Return the observation width.
  /// @return Four for the pendulum and eleven for Hopper.
  /// @see VsimTaskEnv
  int obs_size() const override { return pendulum()?4:11; }
  /// @brief Return action limits in the task's control units.
  /// @return One coordinate in [-3,3] or three coordinates in [-1,1].
  /// @see VsimTaskEnv
  ActionSpace action_space() const override {
    return pendulum()?ActionSpace::Continuous({-3.f},{3.f})
                     :ActionSpace::Continuous({-1.f,-1.f,-1.f},{1.f,1.f,1.f});
  }
  /// @brief Return actuator entity names in action order.
  /// @return Names from the scene.
  /// @see VsimTaskEnv
  std::vector<std::string> action_names() const override {
    if (pendulum()) return {scene_.get(root_).name};
    return {scene_.get(joints_[0]).name,scene_.get(joints_[1]).name,scene_.get(joints_[2]).name};
  }
  /// @brief Borrow the live scene for diagnostics and rendering.
  /// @return Scene owned by this environment.
  /// @see VsimTaskEnv
  const vsim::Scene& scene() const { return scene_; }
  /// @brief Read the current position of the body tracked by the report camera.
  /// @return Root body position in meters.
  /// @see scene()
  glm::vec3 root_position() const { return scene_.get(root_).transform.position; }

  /// @brief Advance one action and calculate reward and termination.
  /// The pendulum advances two 20 ms substeps. Hopper advances four 2 ms substeps.
  /// @param action Continuous action in the task's control units.
  /// @return The new observation, reward, termination and truncation flags.
  /// @throws std::invalid_argument For an invalid action width or nonfinite value.
  /// @see action_space()
  StepResult step(const Action& action) override {
    const auto& control=action.continuous;
    const int width=pendulum()?1:3;
    if (control.size()!=static_cast<std::size_t>(width) ||
        !std::all_of(control.begin(),control.end(),[](float u){return std::isfinite(u);}))
      throw std::invalid_argument("invalid vsim control");
    const float before=root_position().x;
    const float substep=pendulum()?.02f:.002f;
    const int repeats=pendulum()?2:4;
    if (!failed_) {
      try {
        for (int k=0;k<repeats;++k) {
          if (pendulum()) {
            world_->apply_joint_force(root_,100.f*std::clamp(control[0]/3.f,-1.f,1.f));
          } else {
            constexpr std::array gear{80.f,60.f,20.f};
            for (int i=0;i<3;++i)
              world_->apply_joint_force(joints_[i],gear[i]*std::clamp(control[i],-1.f,1.f));
          }
          world_->step(substep);
        }
      } catch (const std::exception&) {
        failed_=true;
      }
    }
    StepResult result;
    result.obs=observe();
    result.reward=1.f;
    if (pendulum()) {
      result.terminated=failed_ || std::abs(result.obs[1])>.2f;
    } else {
      float cost=0;
      for (float u:control) cost+=1e-3f*u*u;
      result.reward=(root_position().x-before)/(substep*repeats)+1.f-cost;
      result.terminated=failed_ || result.obs[0]<=.7f || std::abs(result.obs[1])>=.2f || !healthy_rates();
    }
    result.truncated=!result.terminated && CountStep();
    return result;
  }

protected:
  /// @brief Restore the authored poses and sample the task's reset perturbations.
  /// @return Initial observation after perturbation.
  Obs Reset() override {
    BeginEpisode();
    try {
      world_->restore_state(start_);
    } catch (const std::exception&) {
      world_.reset();
      scene_=initial_;
      world_=std::make_unique<vsim::PhysicsWorld>(scene_);
      start_=world_->save_state();
    }
    failed_=false;
    if (pendulum()) {
      perturb(joints_[0],.01f);
      perturb(joints_[1],.01f);
    } else {
      move_hopper();
      for (auto id:joints_) perturb(id,.005f);
    }
    return observe();
  }

private:
  /// @brief Choose the slider-and-hinge task geometry when true.
  bool pendulum() const { return task_==VsimTask::InvertedPendulum; }
  /// @brief Resolve a required entity by its authored name.
  vsim::EntityId find(const char* name) const {
    for (auto id:scene_.entities()) if (scene_.get(id).name==name) return id;
    throw std::invalid_argument(std::string("vsim scene missing entity ")+name);
  }
  /// @brief Add independent position and rate noise within joint position limits.
  void perturb(vsim::EntityId id,float range) {
    const auto state=world_->joint_state(id);
    const auto& joint=*scene_.get(id).joint;
    float position=state.position+Uniform(-range,range);
    if (joint.limited) position=std::clamp(position,joint.limit_min,joint.limit_max);
    world_->set_joint_state(id,position,state.velocity+Uniform(-range,range));
  }
  /// @brief Perturb the full Hopper tree rigidly before moving individual joints.
  void move_hopper() {
    constexpr float n=.005f;
    const float dx=Uniform(-n,n),dy=Uniform(-n,n),da=Uniform(-n,n);
    const float vx=Uniform(-n,n),vy=Uniform(-n,n),wz=Uniform(-n,n);
    const auto pivot=root_position();
    const auto rotation=glm::angleAxis(da,glm::vec3(0,0,1));
    for (auto id:scene_.entities()) {
      auto body=scene_.get(id);
      if (!body.rigid_body || body.rigid_body->motion!="dynamic") continue;
      body.transform.position=pivot+rotation*(body.transform.position-pivot)+glm::vec3(dx,dy,0);
      body.transform.rotation=glm::normalize(rotation*body.transform.rotation);
      body.rigid_body->linear_velocity+=glm::vec3(vx,vy,0)+
          glm::cross(glm::vec3(0,0,wz),body.transform.position-pivot);
      body.rigid_body->angular_velocity+=glm::vec3(0,0,wz);
      scene_.set(id,body);
    }
  }
  /// @brief Check component-wise linear and angular rate bounds of dynamic bodies.
  bool healthy_rates() const {
    for (auto id:scene_.entities()) {
      const auto& body=scene_.get(id).rigid_body;
      if (!body || body->motion!="dynamic") continue;
      for (int k=0;k<3;++k)
        if (std::abs(body->linear_velocity[k])>=100 || std::abs(body->angular_velocity[k])>=100)
          return false;
    }
    return true;
  }
  /// @brief Gather live coordinates and clipped Hopper rates.
  Obs observe() const {
    if (failed_) return Obs(obs_size(),0.f);
    if (pendulum()) {
      const auto cart=world_->joint_state(joints_[0]), pole=world_->joint_state(joints_[1]);
      return {cart.position,pole.position,cart.velocity,pole.velocity};
    }
    const auto& torso=scene_.get(root_);
    const auto& q=torso.transform.rotation;
    float pitch=2.f*std::atan2(q.z,q.w);
    constexpr float pi=std::numbers::pi_v<float>;
    if (pitch>pi) pitch-=2*pi;
    if (pitch<=-pi) pitch+=2*pi;
    Obs obs{torso.transform.position.y,pitch};
    Obs rates{torso.rigid_body->linear_velocity.x,torso.rigid_body->linear_velocity.y,
              torso.rigid_body->angular_velocity.z};
    for (auto id:joints_) {
      const auto joint=world_->joint_state(id);
      obs.push_back(joint.position);
      rates.push_back(joint.velocity);
    }
    for (auto rate:rates) obs.push_back(std::clamp(rate,-10.f,10.f));
    return obs;
  }

  VsimTask task_; ///< Task-specific control and observation rules.
  vsim::Scene initial_; ///< Original scene used to recover after a failed reset.
  vsim::Scene scene_; ///< Current simulated scene.
  std::unique_ptr<vsim::PhysicsWorld> world_; ///< Jolt world borrowing the current scene.
  vsim::PhysicsState start_; ///< Physics snapshot before reset noise.
  vsim::EntityId root_{}; ///< Cart or torso entity ID.
  std::array<vsim::EntityId,3> joints_{}; ///< Observed joints in task order.
  bool failed_=false; ///< Physics failure terminal flag.
};

} // namespace datamunge::rl::environments
