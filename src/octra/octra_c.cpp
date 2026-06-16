#include <octra/octra_c.h>

#include <octra/octra.hpp>

extern "C" {

void octra_hello(void) {
  octra::hello();
}

void octra_make_dvector(double a, double b, double c, double* out3) {
  const auto v = octra::make_dvector(a, b, c);
  out3[0]      = v[0];
  out3[1]      = v[1];
  out3[2]      = v[2];
}

double octra_sum_dvector(const double* values, size_t len) {
  std::vector<double> v;
  v.reserve(len);
  for (size_t i = 0; i < len; i++) {
    v.push_back(values[i]);
  }
  return octra::sum_dvector(v);
}

octra_dpair octra_make_dpair(double a, double b) {
  const auto p = octra::make_dpair(a, b);
  return octra_dpair{p.first, p.second};
}

double octra_sum_dpair(octra_dpair values) {
  return octra::sum_dpair({values.first, values.second});
}

double octra_call_double_cb(double x, octra_double_cb cb, void* userdata) {
  return cb ? cb(x, userdata) : x;
}

void octra_map_dvector_cb(
    const double*   values,
    size_t          len,
    double*         out,
    octra_double_cb cb,
    void*           userdata) {
  if (!values || !out) {
    return;
  }
  for (size_t i = 0; i < len; i++) {
    out[i] = cb ? cb(values[i], userdata) : values[i];
  }
}

} // extern "C"
