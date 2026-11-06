// Native CUDA forward kernels for dm::nn GPU inference.
//
// Compiled to PTX by scripts/prebuild_nn_cuda_kernels.py (`just prebuild-nn-cuda-kernels`)
// and embedded in src/datamunge/nn/device_kernels_ptx.hpp; the library never needs nvcc.
// Every entry takes its buffers first, then one by-value parameter block, matching
// compute::KernelSource. Launches use 256-thread blocks over a 2-D grid of blocks.
// Floating-point contraction is disabled (-fmad=false) to stay close to the CPU kernels.

typedef unsigned long long u64;
typedef long long          i64;

__device__ __forceinline__ u64 global_index() {
  return (u64(blockIdx.y) * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
}

/// out[i] = a[i] + b[i % period]: equal shapes, or b repeating over a's last axes.
struct Elementwise {
  u64 count;
  u64 period;
};
template <class T>
__device__ void add(const T* a, const T* b, T* out, Elementwise p) {
  const u64 i = global_index();
  if (i < p.count)
    out[i] = a[i] + b[i % p.period];
}
/// out[i] = x[i] > 0 ? x[i] : 0, so NaN maps to zero exactly like the CPU kernel.
template <class T>
__device__ void relu(const T* x, T* out, Elementwise p) {
  const u64 i = global_index();
  if (i < p.count)
    out[i] = x[i] > T(0) ? x[i] : T(0);
}

/// NHWC patches: row (n, oy, ox), column (ky, kx, c); padded taps are zero.
struct Patches {
  u64 count;
  i64 h, w, c, kh, kw, oh, ow, sh, sw, dh, dw, pt, pl;
};
template <class T>
__device__ void im2col(const T* x, T* out, Patches p) {
  const u64 i = global_index();
  if (i >= p.count)
    return;
  i64       t  = i64(i);
  const i64 c  = t % p.c;
  t /= p.c;
  const i64 kx = t % p.kw;
  t /= p.kw;
  const i64 ky = t % p.kh;
  t /= p.kh;
  const i64 ox = t % p.ow;
  t /= p.ow;
  const i64 oy = t % p.oh;
  const i64 n  = t / p.oh;
  const i64 y = oy * p.sh - p.pt + ky * p.dh, xx = ox * p.sw - p.pl + kx * p.dw;
  out[i] = y >= 0 && y < p.h && xx >= 0 && xx < p.w ? x[((n * p.h + y) * p.w + xx) * p.c + c]
                                                     : T(0);
}

/// Inference batch norm over the last axis, in the CPU order x*scale + (beta - mean*scale).
struct Normalize {
  u64    count;
  u64    channels;
  double eps;
};
template <class T>
__device__ void batch_norm(
    const T* x, const T* gamma, const T* beta, const T* mean, const T* variance, T* out,
    Normalize p) {
  const u64 i = global_index();
  if (i >= p.count)
    return;
  const u64 c     = i % p.channels;
  const T   scale = gamma[c] * (T(1) / sqrt(variance[c] + T(p.eps)));
  out[i]          = x[i] * scale + (beta[c] - mean[c] * scale);
}

/// Mean over the H*W pixels of each (image, channel), accumulated in double.
struct Pool {
  u64 count;
  u64 pixels;
  u64 channels;
};
template <class T>
__device__ void global_average(const T* x, T* out, Pool p) {
  const u64 i = global_index();
  if (i >= p.count)
    return;
  const u64 n = i / p.channels, c = i % p.channels;
  double    s = 0;
  for (u64 k = 0; k < p.pixels; ++k)
    s += double(x[(n * p.pixels + k) * p.channels + c]);
  out[i] = T(s / double(p.pixels));
}

/// Copy one concat input: row o of `rows` contiguous values lands at out[o*out_row + offset].
struct Block {
  u64 count;
  u64 row;
  u64 out_row;
  u64 offset;
};
template <class T>
__device__ void copy_block(const T* x, T* out, Block p) {
  const u64 i = global_index();
  if (i < p.count)
    out[(i / p.row) * p.out_row + p.offset + i % p.row] = x[i];
}

/// Constant padding of a rank <= 4 tensor (leading axes of length one fill unused ranks).
struct Padding {
  u64    count;
  i64    out[4];
  i64    in[4];
  i64    before[4];
  double value;
};
template <class T>
__device__ void pad_constant(const T* x, T* out, Padding p) {
  const u64 i = global_index();
  if (i >= p.count)
    return;
  i64  t = i64(i), source = 0, stride = 1;
  bool inside = true;
  for (int d = 3; d >= 0; --d) {
    const i64 k = t % p.out[d] - p.before[d];
    t /= p.out[d];
    inside = inside && k >= 0 && k < p.in[d];
    source += k * stride;
    stride *= p.in[d];
  }
  out[i] = inside ? x[source] : T(p.value);
}

/// NHWC max pooling over in-bounds taps: the first tap is taken, later ones replace it only
/// when strictly greater (the CPU rule, so NaN handling matches); an empty window gives -inf.
struct MaxPool {
  u64 count;
  i64 h, w, c, kh, kw, oh, ow, sh, sw, pt, pl;
};
template <class T>
__device__ void max_pool(const T* x, T* out, MaxPool p) {
  const u64 i = global_index();
  if (i >= p.count)
    return;
  i64       t  = i64(i);
  const i64 c  = t % p.c;
  t /= p.c;
  const i64 ox = t % p.ow;
  t /= p.ow;
  const i64 oy = t % p.oh;
  const i64 n  = t / p.oh;
  const i64 y0 = oy * p.sh - p.pt, x0 = ox * p.sw - p.pl;
  T         best  = -T(1) / T(0);
  bool      found = false;
  for (i64 y = y0 > 0 ? y0 : 0; y < y0 + p.kh && y < p.h; ++y)
    for (i64 xx = x0 > 0 ? x0 : 0; xx < x0 + p.kw && xx < p.w; ++xx) {
      const T v = x[((n * p.h + y) * p.w + xx) * p.c + c];
      if (v > best || !found) {
        best  = v;
        found = true;
      }
    }
  out[i] = best;
}

#define DM_ENTRIES(T, S)                                                                       \
  extern "C" __global__ void add_##S(const T* a, const T* b, T* out, Elementwise p) {          \
    add(a, b, out, p);                                                                         \
  }                                                                                            \
  extern "C" __global__ void relu_##S(const T* x, T* out, Elementwise p) { relu(x, out, p); }  \
  extern "C" __global__ void im2col_##S(const T* x, T* out, Patches p) { im2col(x, out, p); }  \
  extern "C" __global__ void batch_norm_##S(                                                   \
      const T* x, const T* g, const T* b, const T* m, const T* v, T* out, Normalize p) {        \
    batch_norm(x, g, b, m, v, out, p);                                                         \
  }                                                                                            \
  extern "C" __global__ void global_average_##S(const T* x, T* out, Pool p) {                  \
    global_average(x, out, p);                                                                 \
  }                                                                                            \
  extern "C" __global__ void copy_block_##S(const T* x, T* out, Block p) {                     \
    copy_block(x, out, p);                                                                     \
  }                                                                                            \
  extern "C" __global__ void pad_constant_##S(const T* x, T* out, Padding p) {                 \
    pad_constant(x, out, p);                                                                   \
  }                                                                                            \
  extern "C" __global__ void max_pool_##S(const T* x, T* out, MaxPool p) { max_pool(x, out, p); }

DM_ENTRIES(float, f32)
DM_ENTRIES(double, f64)
