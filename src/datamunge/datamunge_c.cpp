#include <datamunge/datamunge_c.h>

#include <datamunge/datamunge.hpp>

extern "C" {

void datamunge_hello(void) {
  datamunge::hello();
}

void datamunge_make_dvector(double a, double b, double c, double* out3) {
  const auto v = datamunge::make_dvector(a, b, c);
  out3[0]      = v[0];
  out3[1]      = v[1];
  out3[2]      = v[2];
}

double datamunge_sum_dvector(const double* values, size_t len) {
  std::vector<double> v;
  v.reserve(len);
  for (size_t i = 0; i < len; i++) {
    v.push_back(values[i]);
  }
  return datamunge::sum_dvector(v);
}

datamunge_dpair datamunge_make_dpair(double a, double b) {
  const auto p = datamunge::make_dpair(a, b);
  return datamunge_dpair{p.first, p.second};
}

double datamunge_sum_dpair(datamunge_dpair values) {
  return datamunge::sum_dpair({values.first, values.second});
}

double datamunge_call_double_cb(double x, datamunge_double_cb cb, void* userdata) {
  return cb ? cb(x, userdata) : x;
}

void datamunge_map_dvector_cb(
    const double*   values,
    size_t          len,
    double*         out,
    datamunge_double_cb cb,
    void*           userdata) {
  if (!values || !out) {
    return;
  }
  for (size_t i = 0; i < len; i++) {
    out[i] = cb ? cb(values[i], userdata) : values[i];
  }
}

} // extern "C"
