# Compose depthwise + pointwise filters into an equivalent dense HWIO kernel
# (examples/nn/common/separable.hpp): W[h,w,i,o] = sum_m D[h,w,i,m] P[0,0,i*M+m,o].
# Built from differentiable ops, so gradients reach both factors.

separable_compose_separable_kernel <- function(depthwise, pointwise, kernel, inputs, multiplier, outputs) {
    if (kernel <= 0 || inputs <= 0 || multiplier <= 0 || outputs <= 0)
        stop("positive separable dimensions required")
    spatial <- nn$transpose(nn$reshape(depthwise, c(kernel * kernel, inputs, multiplier)), c(1, 0, 2))
    mixing <- nn$reshape(pointwise, c(inputs, multiplier, outputs))
    product <- nn$matmul(spatial, mixing)
    nn$reshape(nn$transpose(product, c(1, 0, 2)), c(kernel, kernel, inputs, outputs))
}
