# Native R NN example. Explicit weight/spectral reparameterization algorithms.
.nn_common <- file.path(Sys.getenv("DATAMUNGE_EXAMPLES_ROOT",getwd()),"examples/r/nn/common.R")
if (!file.exists(.nn_common)) .nn_common <- system.file("examples/nn/common.R",package="datamungerNN",mustWork=TRUE)
source(.nn_common,local=TRUE)
rm(.nn_common)

random_tensor <- function(shape,rng) {rng$reset_normal();nn$from_vector(vapply(seq_len(prod(shape)),function(i) rng$normal32(),numeric(1)),shape)}

weight_norm_dense <- function(parameters,name,x,input_width,output_width) {
    fresh <- !parameters$contains(paste0(name,"/v"))
    v <- parameters$get(paste0(name,"/v"),c(input_width,output_width),nn$init$glorot_uniform())
    if (fresh) nn$no_grad(function() parameters$set(paste0(name,"/g"),nn$sqrt(nn$sum(nn$square(v),0L))))
    g <- parameters$at(paste0(name,"/g"));bias <- parameters$get(paste0(name,"/b"),output_width,nn$init$zeros())
    norm <- nn$sqrt(nn$sum(nn$square(v),0L));weights <- nn$mul(v,nn$div(g,norm))
    list(y=nn$bias_add(nn$matmul(x,weights),bias),weights=weights)
}

spectral_norm_dense <- function(parameters,stats,name,x,input_width,output_width,iterations=1L) {
    limit <- nn$sqrt(nn$div(nn$scalar(6),input_width+output_width))$item()
    weights <- parameters$get(paste0(name,"/W"),c(input_width,output_width),nn$init$uniform(limit))
    bias <- parameters$get(paste0(name,"/b"),output_width,nn$init$zeros())
    u0 <- stats$get(paste0(name,"/u"),c(1L,output_width),nn$init$normal(1))
    l2 <- function(tensor) nn$mul(tensor,nn$rsqrt(nn$add(nn$sum(nn$square(tensor),1L,TRUE),1e-12)))
    u <- u0$detach()
    for (i in seq_len(iterations)) {v <- l2(nn$matmul(u,weights,FALSE,TRUE));u <- l2(nn$matmul(v,weights))}
    u <- nn$stop_gradient(u);v <- nn$stop_gradient(v)
    sigma <- nn$squeeze(nn$matmul(nn$matmul(v,weights),u,FALSE,TRUE))
    normalized <- nn$div(weights,sigma)
    list(y=nn$bias_add(nn$matmul(x,normalized),bias),weights=normalized,sigma=sigma,u0=u0,u=u)
}

weight_norm <- function() {
    p <- nn$ParameterStore(1L);rng <- nn$Mt19937_64(1L)
    x <- random_tensor(c(4L,5L),rng);r <- random_tensor(c(4L,3L),rng)
    result <- weight_norm_dense(p,"d",x,5L,3L);y <- result$y;weights <- result$weights
    v <- p$at("d/v");g <- p$at("d/g")
    grads <- nn$grad(nn$sum(nn$mul(y,r),c(0L,1L)),list(v,g,weights))
    dv <- grads[[1L]];dg <- grads[[2L]];dw <- grads[[3L]]
    error_g <- error_v <- error_dot <- error_norm <- 0
    for (j in 0:2) {
        vv <- vapply(0:4,function(i) v$at(c(i,j)),numeric(1));wv <- vapply(0:4,function(i) weights$at(c(i,j)),numeric(1))
        gradient <- vapply(0:4,function(i) dw$at(c(i,j)),numeric(1));direction <- vapply(0:4,function(i) dv$at(c(i,j)),numeric(1))
        nv <- sqrt(sum(vv*vv));nw <- sqrt(sum(wv*wv));expected_g <- sum(gradient*vv)/nv
        expected_v <- g$at(j)/nv*(gradient-expected_g*vv/nv);dot <- sum(vv*direction)
        error_g <- max(error_g,abs(expected_g-dg$at(j)));error_v <- max(error_v,max(abs(expected_v-direction)))
        error_dot <- max(error_dot,abs(dot));error_norm <- max(error_norm,abs(nw-g$at(j)))
        cat("Column",j,"v norm / gain / W norm / max W-v / v dot gradient:",nv,g$at(j),nw,max(abs(wv-vv)),dot,"\n")
    }
    stopifnot(max(error_g,error_v,error_dot,error_norm)<1e-5)
    cat("Weight-norm errors gain/direction/orthogonality/norm:",error_g,error_v,error_dot,error_norm,"\n")
}

spectral_norm <- function() {
    none <- nn$zeros(c(1L,64L));sigma_true <- 0;last_error <- 1
    for (iterations in c(1L,2L,3L,5L,10L,20L)) {
        p <- nn$ParameterStore(2L);stats <- nn$ParameterStore(3L)
        result <- spectral_norm_dense(p,stats,"d",none,64L,32L,iterations)
        if (iterations==1L) sigma_true <- nn$svd(p$at("d/W"),FALSE)$s$values()[1L]
        estimate <- result$sigma$item();last_error <- abs(estimate-sigma_true)/sigma_true
        stopifnot(estimate<=sigma_true*(1+1e-5))
        cat("Power iterations:",iterations,"sigma:",estimate,"relative error:",last_error,"\n")
    }
    stopifnot(last_error<1e-2);cat("SVD largest singular value:",sigma_true,"\n")
    p <- nn$ParameterStore(2L);stats <- nn$ParameterStore(3L)
    for (step in 1:20) {
        result <- spectral_norm_dense(p,stats,"d",none,64L,32L)
        nn$no_grad(function() result$u0$copy_(result$u))
        if (step %in% c(1L,2L,3L,5L,10L,20L)) cat("Persistent-vector step:",step,"sigma:",result$sigma$item(),"relative error:",abs(result$sigma$item()-sigma_true)/sigma_true,"\n")
    }
    x <- random_tensor(c(1000L,64L),nn$Mt19937_64(4L))
    result <- spectral_norm_dense(p,stats,"d",x,64L,32L)
    normalized_sigma <- nn$svd(result$weights,FALSE)$s$values()[1L]
    xv <- matrix(x$values(),ncol=64L,byrow=TRUE);yv <- matrix(result$y$values(),ncol=32L,byrow=TRUE)
    maximum <- max(sqrt(rowSums(yv*yv)/rowSums(xv*xv)))
    stopifnot(abs(normalized_sigma-1)<1e-2,maximum<=normalized_sigma*(1+1e-5))
    cat("Normalized spectral norm:",normalized_sigma,"max output/input norm ratio over 1000 vectors:",maximum,"\n")
}

main <- function(arguments=commandArgs(TRUE)) {
    nn$set_num_threads(2L)
    weight_norm();spectral_norm()
    cat("All weight/spectral normalization checks passed.\n")
    0L
}
if (sys.nframe()==0L) quit(status=main())
