#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct datamunge_dpair {
  double first;
  double second;
} datamunge_dpair;

typedef double (*datamunge_double_cb)(double x, void* userdata);

void datamunge_hello(void);

/* Writes 3 doubles to `out3` (must point to at least 3 elements). */
void datamunge_make_dvector(double a, double b, double c, double* out3);

double datamunge_sum_dvector(const double* values, size_t len);

datamunge_dpair datamunge_make_dpair(double a, double b);

double datamunge_sum_dpair(datamunge_dpair values);

/* Calls `cb(x, userdata)` and returns the result. */
double datamunge_call_double_cb(double x, datamunge_double_cb cb, void* userdata);

/*
 * Maps `values[0..len)` through `cb` into `out` (must point to at least `len`
 * elements).
 */
void datamunge_map_dvector_cb(
    const double*   values,
    size_t          len,
    double*         out,
    datamunge_double_cb cb,
    void*           userdata);

#ifdef __cplusplus
} /* extern "C" */
#endif
