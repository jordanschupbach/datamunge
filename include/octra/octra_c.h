#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct octra_dpair {
  double first;
  double second;
} octra_dpair;

typedef double (*octra_double_cb)(double x, void* userdata);

void octra_hello(void);

/* Writes 3 doubles to `out3` (must point to at least 3 elements). */
void octra_make_dvector(double a, double b, double c, double* out3);

double octra_sum_dvector(const double* values, size_t len);

octra_dpair octra_make_dpair(double a, double b);

double octra_sum_dpair(octra_dpair values);

/* Calls `cb(x, userdata)` and returns the result. */
double octra_call_double_cb(double x, octra_double_cb cb, void* userdata);

/*
 * Maps `values[0..len)` through `cb` into `out` (must point to at least `len`
 * elements).
 */
void octra_map_dvector_cb(
    const double*   values,
    size_t          len,
    double*         out,
    octra_double_cb cb,
    void*           userdata);

#ifdef __cplusplus
} /* extern "C" */
#endif
