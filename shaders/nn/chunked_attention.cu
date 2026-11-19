// Each thread owns one output row, so tile traversal needs no atomics or score matrix.
struct AttentionShape {
  unsigned queries, keys, width, value_width, query_tile, key_tile;
};

__device__ unsigned row_index(unsigned rows, unsigned tile) {
  tile = min(rows, tile);
  const unsigned blocks_per_tile = 1 + (tile - 1) / blockDim.x;
  const unsigned within_tile = (blockIdx.x % blocks_per_tile) * blockDim.x + threadIdx.x;
  const unsigned start = (blockIdx.x / blocks_per_tile) * tile;
  if (within_tile >= tile || within_tile >= rows - start) return rows;
  return start + within_tile;
}

__device__ double dot_rows(const double* a, const double* b, unsigned width) {
  double result = 0;
  for (unsigned i = 0; i < width; ++i) result += a[i] * b[i];
  return result;
}

extern "C" __global__ void chunked_forward(
    const double* q, const double* k, const double* v,
    double* output, double* maximum, double* normalizer, AttentionShape shape) {
  const auto i = row_index(shape.queries, shape.query_tile);
  if (i >= shape.queries) return;
  auto* row = output + i * shape.value_width;
  for (unsigned d = 0; d < shape.value_width; ++d) row[d] = 0;
  double largest = -__builtin_huge_val();
  double total = 0;
  const double scale = 1 / sqrt(double(shape.width));
  for (unsigned start = 0; start < shape.keys;) {
    const unsigned end = start + min(shape.key_tile, shape.keys - start);
    for (unsigned j = start; j < end; ++j) {
      const double score = scale * dot_rows(q + i * shape.width, k + j * shape.width, shape.width);
      const double next = fmax(largest, score);
      const double carry = exp(largest - next);
      const double weight = exp(score - next);
      for (unsigned d = 0; d < shape.value_width; ++d)
        row[d] = carry * row[d] + weight * v[j * shape.value_width + d];
      total = carry * total + weight;
      largest = next;
    }
    start = end;
  }
  for (unsigned d = 0; d < shape.value_width; ++d) row[d] /= total;
  maximum[i] = largest;
  normalizer[i] = total;
}

extern "C" __global__ void chunked_query_gradient(
    const double* q, const double* k, const double* v, const double* dy,
    const double* output, const double* maximum, const double* normalizer,
    double* dq, double* correction, AttentionShape shape) {
  const auto i = row_index(shape.queries, shape.query_tile);
  if (i >= shape.queries) return;
  const double scale = 1 / sqrt(double(shape.width));
  const auto* upstream = dy + i * shape.value_width;
  const double delta = dot_rows(upstream, output + i * shape.value_width, shape.value_width);
  correction[i] = delta;
  auto* gradient = dq + i * shape.width;
  for (unsigned d = 0; d < shape.width; ++d) gradient[d] = 0;
  for (unsigned start = 0; start < shape.keys;) {
    const unsigned end = start + min(shape.key_tile, shape.keys - start);
    for (unsigned j = start; j < end; ++j) {
      const auto* key = k + j * shape.width;
      const double score = scale * dot_rows(q + i * shape.width, key, shape.width);
      const double p = exp(score - maximum[i]) / normalizer[i];
      const double ds = scale * p * (dot_rows(upstream, v + j * shape.value_width, shape.value_width) - delta);
      for (unsigned d = 0; d < shape.width; ++d) gradient[d] += ds * key[d];
    }
    start = end;
  }
}

extern "C" __global__ void chunked_key_value_gradient(
    const double* q, const double* k, const double* v, const double* dy,
    const double* maximum, const double* normalizer, const double* correction,
    double* dk, double* dv, AttentionShape shape) {
  const auto j = row_index(shape.keys, shape.key_tile);
  if (j >= shape.keys) return;
  const double scale = 1 / sqrt(double(shape.width));
  const auto* key = k + j * shape.width;
  const auto* value = v + j * shape.value_width;
  auto* key_gradient = dk + j * shape.width;
  auto* value_gradient = dv + j * shape.value_width;
  for (unsigned d = 0; d < shape.width; ++d) key_gradient[d] = 0;
  for (unsigned d = 0; d < shape.value_width; ++d) value_gradient[d] = 0;
  for (unsigned start = 0; start < shape.queries;) {
    const unsigned end = start + min(shape.query_tile, shape.queries - start);
    for (unsigned i = start; i < end; ++i) {
      const auto* query = q + i * shape.width;
      const auto* upstream = dy + i * shape.value_width;
      const double p = exp(scale * dot_rows(query, key, shape.width) - maximum[i]) / normalizer[i];
      const double ds = scale * p * (dot_rows(upstream, value, shape.value_width) - correction[i]);
      for (unsigned d = 0; d < shape.width; ++d) key_gradient[d] += ds * query[d];
      for (unsigned d = 0; d < shape.value_width; ++d) value_gradient[d] += p * upstream[d];
    }
    start = end;
  }
}
