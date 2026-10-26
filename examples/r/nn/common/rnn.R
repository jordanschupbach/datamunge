# Masking and head helpers shared by the explicit-unroll recurrent cells
# (examples/nn/common/rnn.hpp; tflow's common/nn/rnn.h).

# Float [B, 1] mask: 1 where lengths > t.
rnn_step_mask <- function(lengths, t) nn$expand_dims(nn$cast(nn$greater(lengths, as.numeric(t)), nn$DType$Float32), 1L)

# Bool [B, 1] mask for where()-style blends.
rnn_step_valid <- function(lengths, t) nn$expand_dims(nn$greater(lengths, as.numeric(t)), 1L)

# mask * next + (1 - mask) * previous (tflow's arithmetic masking).
rnn_blend <- function(mask, next_value, previous) nn$add(nn$mul(mask, next_value), nn$mul(nn$sub(1, mask), previous))

# Dense head "<name>/W" ~ N(0, 0.1^2), "<name>/b" = 0.
rnn_head <- function(store, name, x, incoming, outgoing) {
    weight <- store$get(paste0(name, "/W"), c(incoming, outgoing), nn$init$normal(0.1))
    bias <- store$get(paste0(name, "/b"), outgoing, nn$init$zeros())
    nn$bias_add(nn$matmul(x, weight), bias)
}

# Zeros where the [B, 1] bool mask is false.
rnn_masked_diagnostics <- function(valid, x) nn$where(valid, x, nn$zeros_like(x))

# The parameter `name`, created from the tensor make() on first use (a C++
# lambda initializer that ignores the generator, e.g. a gate bias).
rnn_fixed_parameter <- function(store, name, make) {
    if (!store$contains(name)) store$set(name, make())
    store$at(name)
}

# Tensor-vector helpers.
rnn_steps <- function(x) nn$tensor_list(nn$time_steps(x))
rnn_stack_time <- function(steps) nn$stack_time(nn$tensor_vector(steps))
rnn_concat <- function(tensors, axis) nn$concat(nn$tensor_vector(tensors), axis)
rnn_split <- function(x, parts, axis) nn$tensor_list(nn$split(x, parts, axis))

# x rounded to float32 (C++ float literals and float arithmetic such as std::sqrt(6.f / 14)).
rnn_float32 <- function(x) readBin(writeBin(as.numeric(x), raw(), size = 4L), "double", n = length(x), size = 4L)

# std::sqrt(float(numerator) / float(denominator)) evaluated in float32.
rnn_float32_sqrt_ratio <- function(numerator, denominator)
    rnn_float32(sqrt(rnn_float32(rnn_float32(numerator) / rnn_float32(denominator))))
