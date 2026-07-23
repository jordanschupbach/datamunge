#include <datamunge/datamunge_c.h>

#include <datamunge/optim/function_types.hpp>
#include <datamunge/optim/proximal_gradient.hpp>
#include <datamunge/optim/fista.hpp>
#include <datamunge/optim/levenberg_marquardt.hpp>
#include <datamunge/optim/newton.hpp>
#include <datamunge/optim/trust_region_newton.hpp>
#include <datamunge/optim/augmented_lagrangian.hpp>
#include <datamunge/optim/sqp.hpp>
#include <datamunge/optim/interior_point.hpp>
#include <datamunge/optim/bayesian_optimization.hpp>
#include <datamunge/optim/rbf_gaussian_process_surrogate.hpp>

#include <cstdint>
#include <exception>
#include <string>
#include <vector>

namespace {

struct JsCb {
  napi_env env;
  napi_ref fn_ref;
};

static double datamunge_js_trampoline(double x, void* userdata) {
  if (!userdata) {
    return x;
  }
  auto* cb = static_cast<JsCb*>(userdata);
  Napi::Env env(cb->env);
  Napi::HandleScope scope(env);

  napi_value fn_value = nullptr;
  if (napi_get_reference_value(cb->env, cb->fn_ref, &fn_value) != napi_ok) {
    return x;
  }
  auto fn = Napi::Function(env, fn_value);
  Napi::Value result = fn.Call({ Napi::Number::New(env, x) });
  if (!result.IsNumber()) {
    Napi::TypeError::New(env, "callback must return a number").ThrowAsJavaScriptException();
    return x;
  }
  return result.As<Napi::Number>().DoubleValue();
}

static Napi::Value datamunge_js_call_with_function(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 2 || !info[0].IsNumber() || !info[1].IsFunction()) {
    Napi::TypeError::New(env, "expected (number x, function cb)").ThrowAsJavaScriptException();
    return env.Null();
  }

  const double x = info[0].As<Napi::Number>().DoubleValue();
  auto fn = info[1].As<Napi::Function>();

  JsCb cb{};
  cb.env = env;
  cb.fn_ref = nullptr;
  if (napi_create_reference(env, fn, 1, &cb.fn_ref) != napi_ok) {
    Napi::Error::New(env, "failed to create callback reference").ThrowAsJavaScriptException();
    return env.Null();
  }

  double out = datamunge_call_double_cb(x, datamunge_js_trampoline, &cb);
  napi_delete_reference(env, cb.fn_ref);
  return Napi::Number::New(env, out);
}

static Napi::Value datamunge_js_map_array_with_function(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 2 || !info[0].IsArray() || !info[1].IsFunction()) {
    Napi::TypeError::New(env, "expected (number[] values, function cb)").ThrowAsJavaScriptException();
    return env.Null();
  }

  auto arr = info[0].As<Napi::Array>();
  const uint32_t len = arr.Length();
  std::vector<double> in;
  in.reserve(len);
  for (uint32_t i = 0; i < len; i++) {
    Napi::Value v = arr.Get(i);
    if (!v.IsNumber()) {
      Napi::TypeError::New(env, "values must be numbers").ThrowAsJavaScriptException();
      return env.Null();
    }
    in.push_back(v.As<Napi::Number>().DoubleValue());
  }

  auto fn = info[1].As<Napi::Function>();
  JsCb cb{};
  cb.env = env;
  cb.fn_ref = nullptr;
  if (napi_create_reference(env, fn, 1, &cb.fn_ref) != napi_ok) {
    Napi::Error::New(env, "failed to create callback reference").ThrowAsJavaScriptException();
    return env.Null();
  }

  std::vector<double> out(len, 0.0);
  datamunge_map_dvector_cb(in.data(), static_cast<size_t>(len), out.data(), datamunge_js_trampoline, &cb);
  napi_delete_reference(env, cb.fn_ref);

  Napi::Array outArr = Napi::Array::New(env, len);
  for (uint32_t i = 0; i < len; i++) {
    outArr.Set(i, Napi::Number::New(env, out[i]));
  }
  return outArr;
}

// ===========================================================================
// datamunge::optim bridge -- SWIG's Node/N-API backend does not generate any
// director code at all (confirmed: `grep -c -i director` on the generated
// wrapper is 0, even though `%feature("director")` lines are present in
// datamungejs.i for parity with other languages' .i files). That means a JS
// class can never really "subclass" datamunge::optim::ArbitraryFunction (or
// any of the newer HessianFunction / ResidualFunction / ProximalFunction /
// EqualityConstrainedFunction / InequalityConstrainedFunction /
// BayesianSurrogate interfaces) and have C++ call back into overridden
// methods the way it does in Python/Ruby/Perl.
//
// Instead of trying to make SWIG's own pointer-wrapping machinery work for
// director-less abstract classes (its type-info tables/constructors aren't
// even declared yet at the point in the generated file where this .inl gets
// injected -- this file is spliced in right after the *first*
// `#include <napi.h>`, long before the SWIGTYPE_p_* macros exist), each new
// optimizer gets one small hand-written "run_*" entry point below. It:
//   1. takes a plain JS object with the interface's required methods
//      (evaluate/gradient/hessian/proximal/constraints/.../residuals/...),
//   2. wraps that object in a small adapter class implementing the matching
//      datamunge::optim::* interface by looking up and calling those methods
//      by name (duck typing -- the closest JS analogue of "subclassing" this
//      backend can offer),
//   3. runs the real C++ optimizer against the adapter synchronously on the
//      current thread (no persistent handles/thread-safe functions needed --
//      the JS object never needs to outlive this single call), and
//   4. returns the optimized coordinates and final objective value as plain
//      JS values.
//
// This mirrors the existing call_with_function/map_array_with_function
// bridge's spirit (JS function/object in, C++ calls back into it, plain JS
// values out) but generalizes it from a single double->double callback to
// the richer multi-method, vector/matrix-valued interfaces the 6 new
// abstract types require.
// ===========================================================================

static Napi::Array DVectorToJsArray(Napi::Env env, const std::vector<double>& v) {
  Napi::Array arr = Napi::Array::New(env, v.size());
  for (uint32_t i = 0; i < v.size(); i++) {
    arr.Set(i, Napi::Number::New(env, v[i]));
  }
  return arr;
}

static Napi::Array DMatrixToJsArray(Napi::Env env, const std::vector<std::vector<double>>& m) {
  Napi::Array arr = Napi::Array::New(env, m.size());
  for (uint32_t i = 0; i < m.size(); i++) {
    arr.Set(i, DVectorToJsArray(env, m[i]));
  }
  return arr;
}

static std::vector<double> JsArrayToDVector(Napi::Env env, Napi::Value val, const char* what) {
  if (!val.IsArray()) {
    Napi::TypeError::New(env, std::string(what) + " must be a number[]").ThrowAsJavaScriptException();
    return {};
  }
  auto arr = val.As<Napi::Array>();
  const uint32_t len = arr.Length();
  std::vector<double> out;
  out.reserve(len);
  for (uint32_t i = 0; i < len; i++) {
    Napi::Value v = arr.Get(i);
    if (!v.IsNumber()) {
      Napi::TypeError::New(env, std::string(what) + " must contain only numbers").ThrowAsJavaScriptException();
      return {};
    }
    out.push_back(v.As<Napi::Number>().DoubleValue());
  }
  return out;
}

static std::vector<std::vector<double>> JsArrayToDMatrix(Napi::Env env, Napi::Value val, const char* what) {
  if (!val.IsArray()) {
    Napi::TypeError::New(env, std::string(what) + " must be a number[][]").ThrowAsJavaScriptException();
    return {};
  }
  auto arr = val.As<Napi::Array>();
  const uint32_t len = arr.Length();
  std::vector<std::vector<double>> out;
  out.reserve(len);
  for (uint32_t i = 0; i < len; i++) {
    out.push_back(JsArrayToDVector(env, arr.Get(i), what));
  }
  return out;
}

// Small option-object readers: every *Options struct field is optional and
// falls back to the C++ default already baked into `def`.
static double OptDouble(Napi::Object opts, const char* key, double def) {
  if (!opts.Has(key)) return def;
  Napi::Value v = opts.Get(key);
  if (v.IsUndefined() || v.IsNull()) return def;
  return v.As<Napi::Number>().DoubleValue();
}

static std::size_t OptSizeT(Napi::Object opts, const char* key, std::size_t def) {
  if (!opts.Has(key)) return def;
  Napi::Value v = opts.Get(key);
  if (v.IsUndefined() || v.IsNull()) return def;
  return static_cast<std::size_t>(v.As<Napi::Number>().Int64Value());
}

static std::uint64_t OptU64(Napi::Object opts, const char* key, std::uint64_t def) {
  if (!opts.Has(key)) return def;
  Napi::Value v = opts.Get(key);
  if (v.IsUndefined() || v.IsNull()) return def;
  return static_cast<std::uint64_t>(v.As<Napi::Number>().Int64Value());
}

static Napi::Object OptionsObjectArg(Napi::Env env, const Napi::CallbackInfo& info, size_t idx) {
  if (info.Length() > idx && info[idx].IsObject()) {
    return info[idx].As<Napi::Object>();
  }
  return Napi::Object::New(env);
}

// Duck-typed adapter base: wraps a plain JS object and calls its named
// methods with a `this` binding of that same object (so JS-side methods can
// use `this.foo` freely, matching ordinary JS class-instance ergonomics even
// though there is no real C++ inheritance underneath).
struct JsObjectAdapterBase {
  Napi::Env env;
  Napi::Object obj;

  JsObjectAdapterBase(Napi::Env e, Napi::Object o) : env(e), obj(o) {}

  // NOTE: deliberately does NOT open its own Napi::HandleScope. The napi_env's ambient
  // callback scope (opened by the runtime for the entire duration of the outer
  // datamunge_js_run_*() call) is already live here and stays live until that outer call
  // returns to JS -- every Value this method creates or receives remains valid for the
  // whole optimize() run. A HandleScope opened and closed *inside* this method would
  // invalidate its own return value the moment the method returns (the handle is only
  // guaranteed alive while its creating scope is open), which is exactly wrong for a
  // function whose entire purpose is to hand a Value back to its caller. An earlier
  // version of this file did open one here; it "worked" for a single call (e.g. the
  // first gradient() of an optimizer run, by luck of memory reuse) and then corrupted
  // state on the very next nested call (e.g. hessian()), surfacing as a generic
  // "TypeError: Cannot convert undefined or null to object" with no relation to the
  // actual bug. Loops here run at most a few hundred iterations (optimizer
  // max_iterations), nowhere near enough to need periodic scope flushing.
  Napi::Value CallMethod(const char* name, const std::initializer_list<napi_value>& args) const {
    Napi::Value fnVal = obj.Get(name);
    if (!fnVal.IsFunction()) {
      Napi::TypeError::New(
          env, std::string("object passed to optim adapter is missing required method '") + name + "'")
          .ThrowAsJavaScriptException();
      return env.Undefined();
    }
    return fnVal.As<Napi::Function>().Call(obj, args);
  }

  double CallDouble(const char* name, const std::vector<double>& coordinates) const {
    Napi::Value r = CallMethod(name, { DVectorToJsArray(env, coordinates) });
    if (!r.IsNumber()) {
      Napi::TypeError::New(env, std::string(name) + "() must return a number").ThrowAsJavaScriptException();
      return 0.0;
    }
    return r.As<Napi::Number>().DoubleValue();
  }

  std::vector<double> CallVector(const char* name, const std::vector<double>& coordinates) const {
    Napi::Value r = CallMethod(name, { DVectorToJsArray(env, coordinates) });
    return JsArrayToDVector(env, r, name);
  }

  std::vector<std::vector<double>> CallMatrix(const char* name, const std::vector<double>& coordinates) const {
    Napi::Value r = CallMethod(name, { DVectorToJsArray(env, coordinates) });
    return JsArrayToDMatrix(env, r, name);
  }
};

struct JsArbitraryFunction : public datamunge::optim::ArbitraryFunction, private JsObjectAdapterBase {
  JsArbitraryFunction(Napi::Env e, Napi::Object o) : JsObjectAdapterBase(e, o) {}
  double evaluate(const std::vector<double>& coordinates) override { return CallDouble("evaluate", coordinates); }
};

struct JsProximalFunction : public datamunge::optim::ProximalFunction, private JsObjectAdapterBase {
  JsProximalFunction(Napi::Env e, Napi::Object o) : JsObjectAdapterBase(e, o) {}
  double evaluate(const std::vector<double>& coordinates) override { return CallDouble("evaluate", coordinates); }
  std::vector<double> gradient(const std::vector<double>& coordinates) override {
    return CallVector("gradient", coordinates);
  }
  std::vector<double> proximal(const std::vector<double>& point, double step) override {
    Napi::Value r = CallMethod("proximal", { DVectorToJsArray(env, point), Napi::Number::New(env, step) });
    return JsArrayToDVector(env, r, "proximal");
  }
};

struct JsHessianFunction : public datamunge::optim::HessianFunction, private JsObjectAdapterBase {
  JsHessianFunction(Napi::Env e, Napi::Object o) : JsObjectAdapterBase(e, o) {}
  double evaluate(const std::vector<double>& coordinates) override { return CallDouble("evaluate", coordinates); }
  std::vector<double> gradient(const std::vector<double>& coordinates) override {
    return CallVector("gradient", coordinates);
  }
  std::vector<std::vector<double>> hessian(const std::vector<double>& coordinates) override {
    return CallMatrix("hessian", coordinates);
  }
};

struct JsEqualityConstrainedFunction : public datamunge::optim::EqualityConstrainedFunction,
                                        private JsObjectAdapterBase {
  JsEqualityConstrainedFunction(Napi::Env e, Napi::Object o) : JsObjectAdapterBase(e, o) {}
  double evaluate(const std::vector<double>& coordinates) override { return CallDouble("evaluate", coordinates); }
  std::vector<double> gradient(const std::vector<double>& coordinates) override {
    return CallVector("gradient", coordinates);
  }
  std::vector<double> constraints(const std::vector<double>& coordinates) override {
    return CallVector("constraints", coordinates);
  }
  std::vector<std::vector<double>> constraint_jacobian(const std::vector<double>& coordinates) override {
    return CallMatrix("constraint_jacobian", coordinates);
  }
};

struct JsInequalityConstrainedFunction : public datamunge::optim::InequalityConstrainedFunction,
                                          private JsObjectAdapterBase {
  JsInequalityConstrainedFunction(Napi::Env e, Napi::Object o) : JsObjectAdapterBase(e, o) {}
  double evaluate(const std::vector<double>& coordinates) override { return CallDouble("evaluate", coordinates); }
  std::vector<double> gradient(const std::vector<double>& coordinates) override {
    return CallVector("gradient", coordinates);
  }
  std::vector<double> inequalities(const std::vector<double>& coordinates) override {
    return CallVector("inequalities", coordinates);
  }
  std::vector<std::vector<double>> inequality_jacobian(const std::vector<double>& coordinates) override {
    return CallMatrix("inequality_jacobian", coordinates);
  }
};

struct JsResidualFunction : public datamunge::optim::ResidualFunction, private JsObjectAdapterBase {
  JsResidualFunction(Napi::Env e, Napi::Object o) : JsObjectAdapterBase(e, o) {}
  std::vector<double> residuals(const std::vector<double>& coordinates) override {
    return CallVector("residuals", coordinates);
  }
  std::vector<std::vector<double>> jacobian(const std::vector<double>& coordinates) override {
    return CallMatrix("jacobian", coordinates);
  }
};

struct JsBayesianSurrogate : public datamunge::optim::BayesianSurrogate, private JsObjectAdapterBase {
  JsBayesianSurrogate(Napi::Env e, Napi::Object o) : JsObjectAdapterBase(e, o) {}
  void fit(const std::vector<std::vector<double>>& points, const std::vector<double>& values) override {
    CallMethod("fit", { DMatrixToJsArray(env, points), DVectorToJsArray(env, values) });
  }
  double acquisition(const std::vector<double>& point, double incumbent) override {
    Napi::Value r = CallMethod("acquisition", { DVectorToJsArray(env, point), Napi::Number::New(env, incumbent) });
    if (!r.IsNumber()) {
      Napi::TypeError::New(env, "acquisition() must return a number").ThrowAsJavaScriptException();
      return 0.0;
    }
    return r.As<Napi::Number>().DoubleValue();
  }
};

static Napi::Object MakeResult(Napi::Env env, const std::vector<double>& coordinates, double value) {
  Napi::Object result = Napi::Object::New(env);
  result.Set("coordinates", DVectorToJsArray(env, coordinates));
  result.Set("value", Napi::Number::New(env, value));
  return result;
}

// The optimizer algorithms themselves can throw plain std::exceptions (e.g. Newton on a
// singular Hessian, InteriorPoint when it can't find a feasible step) -- and the
// SWIG-generated wrapper for this backend has no exception translation of its own
// (`grep -c "catch (std::exception" src/datamunge_js_wrap.cpp` is 0 project-wide), so an
// uncaught one propagates straight through the N-API boundary and calls std::terminate,
// crashing the whole Node process instead of surfacing as a catchable JS error. Every
// run_*() entry point below funnels its actual optimizer call through this helper so a
// bad problem (e.g. an infeasible InteriorPoint starting point) is a normal try/catch-able
// JS exception instead of a hard process crash.
template <typename Fn>
static Napi::Value SafeOptimize(Napi::Env env, Fn&& body) {
  try {
    return body();
  } catch (const std::exception& e) {
    Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
    return env.Null();
  } catch (...) {
    Napi::Error::New(env, "unknown C++ exception thrown from datamunge optim bridge").ThrowAsJavaScriptException();
    return env.Null();
  }
}

// -- ProximalFunction: ProximalGradient / FISTA --------------------------

static Napi::Value datamunge_js_run_proximal_gradient(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 2 || !info[0].IsObject()) {
    Napi::TypeError::New(env, "expected (object proximalFn, number[] initialCoords, object? options)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  JsProximalFunction fn(env, info[0].As<Napi::Object>());
  std::vector<double> coords = JsArrayToDVector(env, info[1], "initialCoords");
  Napi::Object opts = OptionsObjectArg(env, info, 2);
  datamunge::optim::ProximalGradientOptions options{};
  options.step_size = OptDouble(opts, "step_size", options.step_size);
  options.max_iterations = OptSizeT(opts, "max_iterations", options.max_iterations);
  options.tolerance = OptDouble(opts, "tolerance", options.tolerance);
  datamunge::optim::ProximalGradient optimizer(options);
  return SafeOptimize(env, [&]() -> Napi::Value {
    double value = optimizer.optimize(fn, coords);
    return MakeResult(env, coords, value);
  });
}

static Napi::Value datamunge_js_run_fista(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 2 || !info[0].IsObject()) {
    Napi::TypeError::New(env, "expected (object proximalFn, number[] initialCoords, object? options)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  JsProximalFunction fn(env, info[0].As<Napi::Object>());
  std::vector<double> coords = JsArrayToDVector(env, info[1], "initialCoords");
  Napi::Object opts = OptionsObjectArg(env, info, 2);
  datamunge::optim::FISTAOptions options{};
  options.step_size = OptDouble(opts, "step_size", options.step_size);
  options.max_iterations = OptSizeT(opts, "max_iterations", options.max_iterations);
  options.tolerance = OptDouble(opts, "tolerance", options.tolerance);
  datamunge::optim::FISTA optimizer(options);
  return SafeOptimize(env, [&]() -> Napi::Value {
    double value = optimizer.optimize(fn, coords);
    return MakeResult(env, coords, value);
  });
}

// -- HessianFunction: Newton / TrustRegionNewton --------------------------

static Napi::Value datamunge_js_run_newton(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 2 || !info[0].IsObject()) {
    Napi::TypeError::New(env, "expected (object hessianFn, number[] initialCoords, object? options)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  JsHessianFunction fn(env, info[0].As<Napi::Object>());
  std::vector<double> coords = JsArrayToDVector(env, info[1], "initialCoords");
  Napi::Object opts = OptionsObjectArg(env, info, 2);
  datamunge::optim::NewtonOptions options{};
  options.max_iterations = OptSizeT(opts, "max_iterations", options.max_iterations);
  options.tolerance = OptDouble(opts, "tolerance", options.tolerance);
  options.damping = OptDouble(opts, "damping", options.damping);
  datamunge::optim::Newton optimizer(options);
  return SafeOptimize(env, [&]() -> Napi::Value {
    double value = optimizer.optimize(fn, coords);
    return MakeResult(env, coords, value);
  });
}

static Napi::Value datamunge_js_run_trust_region_newton(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 2 || !info[0].IsObject()) {
    Napi::TypeError::New(env, "expected (object hessianFn, number[] initialCoords, object? options)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  JsHessianFunction fn(env, info[0].As<Napi::Object>());
  std::vector<double> coords = JsArrayToDVector(env, info[1], "initialCoords");
  Napi::Object opts = OptionsObjectArg(env, info, 2);
  datamunge::optim::TrustRegionNewtonOptions options{};
  options.initial_radius = OptDouble(opts, "initial_radius", options.initial_radius);
  options.max_radius = OptDouble(opts, "max_radius", options.max_radius);
  options.max_iterations = OptSizeT(opts, "max_iterations", options.max_iterations);
  options.tolerance = OptDouble(opts, "tolerance", options.tolerance);
  datamunge::optim::TrustRegionNewton optimizer(options);
  return SafeOptimize(env, [&]() -> Napi::Value {
    double value = optimizer.optimize(fn, coords);
    return MakeResult(env, coords, value);
  });
}

// -- EqualityConstrainedFunction: AugmentedLagrangian / SQP ---------------

static Napi::Value datamunge_js_run_augmented_lagrangian(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 2 || !info[0].IsObject()) {
    Napi::TypeError::New(env, "expected (object equalityFn, number[] initialCoords, object? options)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  JsEqualityConstrainedFunction fn(env, info[0].As<Napi::Object>());
  std::vector<double> coords = JsArrayToDVector(env, info[1], "initialCoords");
  Napi::Object opts = OptionsObjectArg(env, info, 2);
  datamunge::optim::AugmentedLagrangianOptions options{};
  options.step_size = OptDouble(opts, "step_size", options.step_size);
  options.initial_penalty = OptDouble(opts, "initial_penalty", options.initial_penalty);
  options.max_outer_iterations = OptSizeT(opts, "max_outer_iterations", options.max_outer_iterations);
  options.inner_iterations = OptSizeT(opts, "inner_iterations", options.inner_iterations);
  options.tolerance = OptDouble(opts, "tolerance", options.tolerance);
  datamunge::optim::AugmentedLagrangian optimizer(options);
  return SafeOptimize(env, [&]() -> Napi::Value {
    double value = optimizer.optimize(fn, coords);
    return MakeResult(env, coords, value);
  });
}

static Napi::Value datamunge_js_run_sqp(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 2 || !info[0].IsObject()) {
    Napi::TypeError::New(env, "expected (object equalityFn, number[] initialCoords, object? options)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  JsEqualityConstrainedFunction fn(env, info[0].As<Napi::Object>());
  std::vector<double> coords = JsArrayToDVector(env, info[1], "initialCoords");
  Napi::Object opts = OptionsObjectArg(env, info, 2);
  datamunge::optim::SQPOptions options{};
  options.step_size = OptDouble(opts, "step_size", options.step_size);
  options.regularization = OptDouble(opts, "regularization", options.regularization);
  options.max_iterations = OptSizeT(opts, "max_iterations", options.max_iterations);
  options.tolerance = OptDouble(opts, "tolerance", options.tolerance);
  datamunge::optim::SQP optimizer(options);
  return SafeOptimize(env, [&]() -> Napi::Value {
    double value = optimizer.optimize(fn, coords);
    return MakeResult(env, coords, value);
  });
}

// -- InequalityConstrainedFunction: InteriorPoint --------------------------

static Napi::Value datamunge_js_run_interior_point(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 2 || !info[0].IsObject()) {
    Napi::TypeError::New(env, "expected (object inequalityFn, number[] initialCoords, object? options)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  JsInequalityConstrainedFunction fn(env, info[0].As<Napi::Object>());
  std::vector<double> coords = JsArrayToDVector(env, info[1], "initialCoords");
  Napi::Object opts = OptionsObjectArg(env, info, 2);
  datamunge::optim::InteriorPointOptions options{};
  options.step_size = OptDouble(opts, "step_size", options.step_size);
  options.initial_barrier = OptDouble(opts, "initial_barrier", options.initial_barrier);
  options.barrier_decay = OptDouble(opts, "barrier_decay", options.barrier_decay);
  options.max_outer_iterations = OptSizeT(opts, "max_outer_iterations", options.max_outer_iterations);
  options.inner_iterations = OptSizeT(opts, "inner_iterations", options.inner_iterations);
  datamunge::optim::InteriorPoint optimizer(options);
  return SafeOptimize(env, [&]() -> Napi::Value {
    double value = optimizer.optimize(fn, coords);
    return MakeResult(env, coords, value);
  });
}

// -- ResidualFunction: LevenbergMarquardt ----------------------------------

static Napi::Value datamunge_js_run_levenberg_marquardt(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 2 || !info[0].IsObject()) {
    Napi::TypeError::New(env, "expected (object residualFn, number[] initialCoords, object? options)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  JsResidualFunction fn(env, info[0].As<Napi::Object>());
  std::vector<double> coords = JsArrayToDVector(env, info[1], "initialCoords");
  Napi::Object opts = OptionsObjectArg(env, info, 2);
  datamunge::optim::LevenbergMarquardtOptions options{};
  options.initial_damping = OptDouble(opts, "initial_damping", options.initial_damping);
  options.max_iterations = OptSizeT(opts, "max_iterations", options.max_iterations);
  options.tolerance = OptDouble(opts, "tolerance", options.tolerance);
  datamunge::optim::LevenbergMarquardt optimizer(options);
  return SafeOptimize(env, [&]() -> Napi::Value {
    double value = optimizer.optimize(fn, coords);
    return MakeResult(env, coords, value);
  });
}

// -- BayesianOptimization: user-supplied BayesianSurrogate, or the
//    ready-made RBFGaussianProcessSurrogate (no adapter needed) -----------

static datamunge::optim::BayesianOptimizationOptions ReadBayesOptions(Napi::Object opts) {
  datamunge::optim::BayesianOptimizationOptions options{};
  options.initial_samples = OptSizeT(opts, "initial_samples", options.initial_samples);
  options.max_iterations = OptSizeT(opts, "max_iterations", options.max_iterations);
  options.seed = OptU64(opts, "seed", options.seed);
  return options;
}

static Napi::Value datamunge_js_run_bayesian_optimization(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 5 || !info[0].IsObject() || !info[1].IsObject()) {
    Napi::TypeError::New(
        env,
        "expected (object objectiveFn, object surrogate, number[] initialCoords, number[] lower, number[] upper, "
        "object? options)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  JsArbitraryFunction objective(env, info[0].As<Napi::Object>());
  JsBayesianSurrogate surrogate(env, info[1].As<Napi::Object>());
  std::vector<double> coords = JsArrayToDVector(env, info[2], "initialCoords");
  std::vector<double> lower = JsArrayToDVector(env, info[3], "lower");
  std::vector<double> upper = JsArrayToDVector(env, info[4], "upper");
  Napi::Object opts = OptionsObjectArg(env, info, 5);
  datamunge::optim::BayesianOptimization optimizer(ReadBayesOptions(opts));
  return SafeOptimize(env, [&]() -> Napi::Value {
    double value = optimizer.optimize(objective, coords, lower, upper, surrogate);
    return MakeResult(env, coords, value);
  });
}

static Napi::Value datamunge_js_run_bayesian_optimization_rbf(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 4 || !info[0].IsObject()) {
    Napi::TypeError::New(
        env,
        "expected (object objectiveFn, number[] initialCoords, number[] lower, number[] upper, object? "
        "surrogateOptions, object? options)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  JsArbitraryFunction objective(env, info[0].As<Napi::Object>());
  std::vector<double> coords = JsArrayToDVector(env, info[1], "initialCoords");
  std::vector<double> lower = JsArrayToDVector(env, info[2], "lower");
  std::vector<double> upper = JsArrayToDVector(env, info[3], "upper");
  Napi::Object surrogateOpts = OptionsObjectArg(env, info, 4);
  const double length_scale = OptDouble(surrogateOpts, "length_scale", 1.0);
  const double noise = OptDouble(surrogateOpts, "noise", 1e-6);
  datamunge::optim::RBFGaussianProcessSurrogate surrogate(length_scale, noise);
  Napi::Object opts = OptionsObjectArg(env, info, 5);
  datamunge::optim::BayesianOptimization optimizer(ReadBayesOptions(opts));
  return SafeOptimize(env, [&]() -> Napi::Value {
    double value = optimizer.optimize(objective, coords, lower, upper, surrogate);
    return MakeResult(env, coords, value);
  });
}

static void DatamungeJS_RegisterCallbackBridge(Napi::Env env, Napi::Object exports) {
  exports.Set("call_with_function", Napi::Function::New(env, datamunge_js_call_with_function));
  exports.Set("map_array_with_function", Napi::Function::New(env, datamunge_js_map_array_with_function));

  exports.Set("run_proximal_gradient", Napi::Function::New(env, datamunge_js_run_proximal_gradient));
  exports.Set("run_fista", Napi::Function::New(env, datamunge_js_run_fista));
  exports.Set("run_newton", Napi::Function::New(env, datamunge_js_run_newton));
  exports.Set("run_trust_region_newton", Napi::Function::New(env, datamunge_js_run_trust_region_newton));
  exports.Set("run_augmented_lagrangian", Napi::Function::New(env, datamunge_js_run_augmented_lagrangian));
  exports.Set("run_sqp", Napi::Function::New(env, datamunge_js_run_sqp));
  exports.Set("run_interior_point", Napi::Function::New(env, datamunge_js_run_interior_point));
  exports.Set("run_levenberg_marquardt", Napi::Function::New(env, datamunge_js_run_levenberg_marquardt));
  exports.Set("run_bayesian_optimization", Napi::Function::New(env, datamunge_js_run_bayesian_optimization));
  exports.Set("run_bayesian_optimization_rbf", Napi::Function::New(env, datamunge_js_run_bayesian_optimization_rbf));
}

} // namespace
