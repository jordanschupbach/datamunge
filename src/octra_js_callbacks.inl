#include <octra/octra_c.h>

namespace {

struct JsCb {
  napi_env env;
  napi_ref fn_ref;
};

static double octra_js_trampoline(double x, void* userdata) {
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

static Napi::Value octra_js_call_with_function(const Napi::CallbackInfo& info) {
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

  double out = octra_call_double_cb(x, octra_js_trampoline, &cb);
  napi_delete_reference(env, cb.fn_ref);
  return Napi::Number::New(env, out);
}

static Napi::Value octra_js_map_array_with_function(const Napi::CallbackInfo& info) {
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
  octra_map_dvector_cb(in.data(), static_cast<size_t>(len), out.data(), octra_js_trampoline, &cb);
  napi_delete_reference(env, cb.fn_ref);

  Napi::Array outArr = Napi::Array::New(env, len);
  for (uint32_t i = 0; i < len; i++) {
    outArr.Set(i, Napi::Number::New(env, out[i]));
  }
  return outArr;
}

static void OctraJS_RegisterCallbackBridge(Napi::Env env, Napi::Object exports) {
  exports.Set("call_with_function", Napi::Function::New(env, octra_js_call_with_function));
  exports.Set("map_array_with_function", Napi::Function::New(env, octra_js_map_array_with_function));
}

} // namespace

