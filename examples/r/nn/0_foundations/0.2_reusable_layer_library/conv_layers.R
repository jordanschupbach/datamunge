# Native R NN example. Convolution variants checked against explicit reference loops.
# CMD: nix develop .#r --command Rscript examples/r/nn/0_foundations/0.2_reusable_layer_library/conv_layers.R
.nn_common <- file.path(Sys.getenv("DATAMUNGE_EXAMPLES_ROOT",getwd()),"examples/r/nn/common.R")
if (!file.exists(.nn_common)) .nn_common <- system.file("examples/nn/common.R",package="datamungerNN",mustWork=TRUE)
source(.nn_common,local=TRUE)
rm(.nn_common)

random_tensor <- function(shape,rng) {rng$reset_normal();nn$from_vector(vapply(seq_len(prod(shape)),function(i) rng$normal32(),numeric(1)),shape)}
configuration <- function(stride=1L,dilation=1L,padding=nn$Padding$Same) {
    option <- nn$ConvLayerOptions();option$stride <- stride;option$dilation <- dilation;option$padding <- padding;option$bias <- FALSE;option
}
difference <- function(a,b) {stopifnot(identical(a$shape(),b$shape()));max(abs(a$values()-b$values()))}
coordinates <- function(shape) as.matrix(expand.grid(lapply(shape,function(size) seq_len(size)-1L)))

reference <- function(x,weights,stride=1L,dilation=1L,padding=nn$Padding$Same,depthwise=FALSE) {
    shape <- as.numeric(x$shape());kernel <- as.numeric(weights$shape());dims <- length(shape)-2L
    sizes <- shape[2:(length(shape)-1L)];ks <- head(kernel,-2L);cin <- tail(shape,1L);mult <- tail(kernel,1L)
    stopifnot(length(ks)==dims)
    same <- padding==nn$Padding$Same
    out <- if(same) ceiling(sizes/stride) else floor((sizes-dilation*(ks-1)-1)/stride)+1
    before <- if(same) floor(pmax(0,(out-1)*stride+dilation*(ks-1)+1-sizes)/2) else rep(0,dims)
    cout <- if(depthwise) cin*mult else mult
    output_shape <- c(shape[1L],out,cout);values <- numeric(prod(output_shape))
    positions <- coordinates(out);offsets <- coordinates(ks)
    output_index <- function(coords) {
        result <- 0
        for (i in seq_along(coords)) result <- result*output_shape[i]+coords[i]
        result+1L
    }
    for (n in 0:(shape[1L]-1L)) for (pi in seq_len(nrow(positions))) for (channel in 0:(cout-1L)) {
        position <- positions[pi,];total <- 0
        for (ki in seq_len(nrow(offsets))) {
            offset <- offsets[ki,];source <- position*stride+offset*dilation-before
            if (any(source<0|source>=sizes)) next
            if (depthwise) {
                ci <- channel%/%mult;m <- channel%%mult
                total <- total+x$at(c(n,source,ci))*weights$at(c(offset,ci,m))
            } else for (ci in 0:(cin-1L)) total <- total+x$at(c(n,source,ci))*weights$at(c(offset,ci,channel))
        }
        values[output_index(c(n,position,channel))] <- total
    }
    nn$from_vector(values,output_shape)
}

output_sizes <- function() {
    expected <- c(5L,3L,3L,2L,7L,7L,4L,3L);row <- 1L
    for (padding in c(nn$Padding$Valid,nn$Padding$Same)) for (stride in 1:3) for (dilation in 1:2) {
        if (stride>1L&&dilation>1L) next
        p <- nn$ParameterStore(1L);option <- configuration(stride,dilation,padding)
        y <- nn$conv1d(p,"conv",nn$zeros(c(1L,7L,1L)),1L,3L,option)
        formula <- nn$conv_output_size(7L,3L,stride,dilation,padding)
        stopifnot(y$size(1L)==formula,formula==expected[row]);row <- row+1L
        cat("Output size padding/stride/dilation:",padding,stride,dilation,"formula/op:",formula,y$size(1L),"\n")
    }
}

conv2d_checks <- function() {
    x <- random_tensor(c(2L,6L,5L,3L),nn$Mt19937_64(1L))
    for (padding in c(nn$Padding$Valid,nn$Padding$Same)) for (stride in 1:2) for (dilation in 1:2) {
        if (stride>1L&&dilation>1L) next
        p <- nn$ParameterStore(2L)
        y <- nn$conv2d(p,"conv",x,4L,3L,configuration(stride,dilation,padding))
        error <- difference(y,reference(x,p$at("conv/W"),stride,dilation,padding));stopifnot(error<1e-5)
        cat("Conv2D padding/stride/dilation:",padding,stride,dilation,"shape:",y$shape(),"reference error:",error,"\n")
    }
    p <- nn$ParameterStore(2L);nn$conv2d(p,"conv",x,4L,3L)
    stopifnot(p$count()==112L);cat("3x3 Conv2D 3->4 with bias parameters:",p$count(),"\n")
}

conv1d_and_3d <- function() {
    rng <- nn$Mt19937_64(3L);p <- nn$ParameterStore(3L);option <- configuration(padding=nn$Padding$Valid)
    x1 <- random_tensor(c(2L,9L,2L),rng);x3 <- random_tensor(c(1L,4L,4L,4L,2L),rng)
    y1 <- nn$conv1d(p,"c1",x1,3L,3L,option);y3 <- nn$conv3d(p,"c3",x3,3L,2L,option)
    w1 <- nn$reshape(p$at("c1/W"),c(3L,2L,3L));w3 <- p$at("c3/W")
    d1 <- difference(y1,reference(x1,w1,padding=nn$Padding$Valid));d3 <- difference(y3,reference(x3,w3,padding=nn$Padding$Valid))
    stopifnot(identical(as.numeric(y1$shape()),c(2,7,3)),identical(as.numeric(y3$shape()),c(1,3,3,3,3)),max(d1,d3)<1e-5,w3$numel()==48L)
    cat("Conv1D/3D shapes:",y1$shape(),y3$shape(),"reference errors:",d1,d3,"\n")
}

depthwise_separable <- function() {
    x <- random_tensor(c(1L,6L,6L,3L),nn$Mt19937_64(4L));p <- nn$ParameterStore(4L);option <- configuration()
    dw <- nn$depthwise_conv2d(p,"dw",x,2L,3L,option);sep <- nn$separable_conv2d(p,"sep",x,8L,3L,option)
    dd <- difference(dw,reference(x,p$at("dw/W"),depthwise=TRUE))
    h <- reference(x,p$at("sep/depthwise/W"),depthwise=TRUE)
    ds <- difference(sep,reference(h,p$at("sep/pointwise/W"),padding=nn$Padding$Valid))
    stopifnot(identical(as.numeric(dw$shape()),c(1,6,6,6)),identical(as.numeric(sep$shape()),c(1,6,6,8)),max(dd,ds)<1e-5)
    full <- nn$ParameterStore(0L);separable <- nn$ParameterStore(0L);z <- nn$zeros(c(1L,4L,4L,64L))
    nn$conv2d(full,"f",z,128L,3L,option);nn$separable_conv2d(separable,"s",z,128L,3L,option)
    stopifnot(full$count()==73728L,separable$count()==8768L)
    cat("Depthwise/separable reference errors:",dd,ds,"full/separable parameters:",full$count(),separable$count(),"\n")
}

transposed <- function() {
    rng <- nn$Mt19937_64(5L);p <- nn$ParameterStore(5L);option <- configuration(stride=2L)
    x <- random_tensor(c(1L,6L,6L,2L),rng)$requires_grad_();r <- random_tensor(c(1L,3L,3L,3L),rng)
    y <- nn$conv2d(p,"conv",x,3L,3L,option);gx <- nn$grad(nn$sum(nn$mul(y,r)),list(x))[[1L]]
    weights <- p$at("conv/W")$detach()
    back <- nn$conv2d_backprop_input(c(1L,6L,6L,2L),weights,r,nn$conv_options(2L))
    transpose <- nn$conv2d_transpose(r,weights,c(1L,6L,6L,2L),nn$conv_options(2L))
    db <- difference(gx,back);dt <- difference(gx,transpose);stopifnot(max(db,dt)<1e-6)
    cat("Transposed convolution as input gradient errors:",db,dt,"\n")
    p <- nn$ParameterStore(6L);x <- nn$ones(c(1L,4L,4L,1L));same <- configuration(stride=2L);valid <- configuration(stride=2L,padding=nn$Padding$Valid)
    nn$conv2d_transpose(p,"same",x,1L,3L,same);p$set("same/W",nn$ones(c(3L,3L,1L,1L)))
    up <- nn$conv2d_transpose(p,"same",x,1L,3L,same);up_valid <- nn$conv2d_transpose(p,"valid",x,1L,3L,valid)
    stopifnot(identical(as.numeric(up$shape()),c(1,8,8,1)),identical(as.numeric(up_valid$shape()),c(1,9,9,1)))
    cat("Ones-kernel checkerboard overlap counts:\n")
    for (i in 0:7) {
        row <- numeric(8L)
        for (j in 0:7) {
            expected <- (if(i>=2L&&i%%2L==0L) 2 else 1)*(if(j>=2L&&j%%2L==0L) 2 else 1)
            stopifnot(up$at(c(0L,i,j,0L))==expected);row[j+1L] <- expected
        }
        cat(row,"\n")
    }
}

dilated <- function() {
    p <- nn$ParameterStore(7L);x <- random_tensor(c(1L,8L,8L,1L),nn$Mt19937_64(7L))
    y <- nn$conv2d(p,"d",x,1L,3L,configuration(dilation=2L))
    w3 <- p$at("d/W")$values();w5 <- numeric(25L)
    for (a in 0:2) for (b in 0:2) w5[2L*a*5L+2L*b+1L] <- w3[a*3L+b+1L]
    ordinary <- nn$conv2d(x,nn$from_vector(w5,c(5L,5L,1L,1L)),nn$conv_options(1L,nn$Padding$Same))
    error <- difference(y,ordinary);stopifnot(error<1e-5)
    cat("5x5 kernel with holes:",w5,"dilated comparison error:",error,"\n")
    cat("Receptive fields for dilations 1,1,1 / 1,2,4 / 1,2,4,8:",1+2*3,1+2*7,1+2*15,"\n")
}

main <- function(arguments=commandArgs(TRUE)) {
    nn$set_num_threads(2L)
    output_sizes();conv2d_checks();conv1d_and_3d();depthwise_separable();transposed();dilated()
    cat("All convolution layer checks passed.\n")
    0L
}
if (sys.nframe()==0L) quit(status=main())
