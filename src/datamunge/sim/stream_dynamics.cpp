/// @file stream_dynamics.cpp
/// @brief Explicit native/reference backend dispatch with no silent substitution.
#include "stream_dynamics.hpp"

#include <datamunge/sim/fruit_stream.hpp>
#include <stdexcept>

#include "mujoco_stream.hpp"
#include "native_stream.hpp"
namespace datamunge::sim::detail {
/// @brief Exclusive explicitly selected native or reference solver.
struct StreamDynamics::Impl {
  std::unique_ptr<NativeStreamDynamics> native;    ///< Independent in-house solver.
  std::unique_ptr<MujocoStreamDynamics> reference; ///< Explicit optional reference engine.
};
StreamDynamics::StreamDynamics(
    const std::string& path,
    const World&       w,
    double             dt,
    std::size_t        n,
    ArticulatedBackend backend)
    : impl_(std::make_unique<Impl>()) {
  if (backend == ArticulatedBackend::Native)
    impl_->native = std::make_unique<NativeStreamDynamics>(path, w, dt, n);
  else if (backend == ArticulatedBackend::MuJoCo)
    impl_->reference = std::make_unique<MujocoStreamDynamics>(path, w, dt, n);
  else
    throw std::invalid_argument("unknown articulated backend");
}
StreamDynamics::~StreamDynamics() = default;
void StreamDynamics::reset(const std::array<double, 6>& q, double f) {
  if (impl_->native)
    return impl_->native->reset(q, f);
  return impl_->reference->reset(q, f);
}
void StreamDynamics::spawn(BodyId id, const ConveyorObservation& o) {
  if (impl_->native)
    return impl_->native->spawn(id, o);
  return impl_->reference->spawn(id, o);
}
void StreamDynamics::remove(BodyId id) {
  if (impl_->native)
    return impl_->native->remove(id);
  return impl_->reference->remove(id);
}
void StreamDynamics::step(const std::array<double, 6>& q, double f) {
  if (impl_->native)
    return impl_->native->step(q, f);
  return impl_->reference->step(q, f);
}
void StreamDynamics::observe(PickObservation& o) const {
  if (impl_->native)
    return impl_->native->observe(o);
  return impl_->reference->observe(o);
}
cad::Vec3d StreamDynamics::tool_point() const {
  if (impl_->native)
    return impl_->native->tool_point();
  return impl_->reference->tool_point();
}
Body StreamDynamics::body(BodyId id, Body b) const {
  if (impl_->native)
    return impl_->native->body(id, b);
  return impl_->reference->body(id, b);
}
bool StreamDynamics::gripped(BodyId id) const {
  if (impl_->native)
    return impl_->native->gripped(id);
  return impl_->reference->gripped(id);
}
double StreamDynamics::grip_force(BodyId id) const {
  if (impl_->native)
    return impl_->native->grip_force(id);
  return impl_->reference->grip_force(id);
}
bool StreamDynamics::bin_supported(BodyId id) const {
  if (impl_->native)
    return impl_->native->bin_supported(id);
  return impl_->reference->bin_supported(id);
}
double StreamDynamics::effort_ratio() const {
  if (impl_->native)
    return impl_->native->effort_ratio();
  return impl_->reference->effort_ratio();
}
std::array<double, 4> StreamDynamics::diagnostics() const {
  if (impl_->native)
    return impl_->native->diagnostics();
  return impl_->reference->diagnostics();
}
} // namespace datamunge::sim::detail
namespace datamunge::sim {
std::string articulated_physics_version(ArticulatedBackend b) {
  if (b == ArticulatedBackend::Native)
    return "datamunge-vsim-1";
  if (b == ArticulatedBackend::MuJoCo)
    return articulated_physics_version();
  throw std::invalid_argument("unknown articulated backend");
}
} // namespace datamunge::sim
