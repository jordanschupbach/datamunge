# Native R NN example. PCA and kernel projection are implemented here.
# CMD: nix develop .#r --command Rscript examples/r/nn/0_foundations/0.6_classical_machine_learning/principal_components.R
.nn_common <- file.path(Sys.getenv("DATAMUNGE_EXAMPLES_ROOT", getwd()), "examples/r/nn/common.R")
if (!file.exists(.nn_common)) .nn_common <- system.file("examples/nn/common.R", package="datamungerNN", mustWork=TRUE)
source(.nn_common, local=TRUE)
rm(.nn_common)

maximum_error <- function(tensor) max(abs(tensor$values()))

linear <- function() {
    n <- 128L; a <- .6; b <- .4
    rotation <- rbind(c(cos(a),-sin(a)*cos(b),sin(a)*sin(b)),
                      c(sin(a),cos(a)*cos(b),-cos(a)*sin(b)), c(0,sin(b),cos(b)))
    offset <- c(1,-2,.5)
    data <- function(shift) {
        values <- numeric(n*3L)
        for (i in 0:(n-1L)) {
            t <- 2*pi*(i+shift)/n
            latent <- c(sqrt(18)*cos(t),sqrt(2)*sin(t),sqrt(.08)*cos(2*t))
            for (d in 1:3) {
                value <- offset[d]
                for (j in 1:3) value <- value+rotation[d,j]*latent[j]
                values[i*3L+d] <- value
            }
        }
        nn$from_vector_double(values,c(n,3L))
    }
    training <- data(0); held_out <- data(.5)
    mean <- nn$mean(training,0L,TRUE)
    factor <- nn$svd(nn$sub(training,mean))
    components <- nn$slice(factor$v,c(0L,0L),c(3L,2L))
    variances <- nn$div(nn$square(nn$slice(factor$s,0L,2L)),n-1L)
    transform <- function(x) nn$matmul(nn$sub(x,mean),components)
    reconstruct <- function(scores) nn$add(mean,nn$matmul(scores,components,FALSE,TRUE))
    scores <- transform(training)
    whitened <- nn$div(scores,nn$sqrt(variances))
    covariance <- nn$div(nn$matmul(whitened,whitened,TRUE,FALSE),n-1L)
    stopifnot(max(abs(mean$values()-offset))<1e-12)
    expected <- c(9*n/(n-1L),n/(n-1L))
    for (c in 0:1) {
        dot <- sum(vapply(0:2,function(d) components$at(c(d,c))*rotation[d+1L,c+1L],numeric(1)))
        stopifnot(abs(abs(dot)-1)<1e-12,abs(variances$at(c)-expected[c+1L])<1e-11)
    }
    stopifnot(maximum_error(nn$sub(covariance,nn$eye(2L,nn$DType$Float64)))<1e-12)
    errors <- vapply(list(training,held_out),function(x) {
        error <- nn$mean(nn$square(nn$sub(x,reconstruct(transform(x)))))$item()
        stopifnot(abs(error-.04/3)<1e-12); error
    },numeric(1))
    cat("PCA variances:",variances$values(),"explained fraction:",10/10.04,"train / held-out MSE:",errors,"\n")
}

kernel <- function() {
    n <- 80L; retained <- 3L
    points <- numeric(n*2L); queries <- numeric(12L)
    for (i in 0:(n-1L)) {
        angle <- 2*pi*(i%/%2L)/(n%/%2L); radius <- if(i%%2L) 2 else .5
        points[i*2L+1:2] <- radius*c(cos(angle),sin(angle))
    }
    for (i in 0:5) {
        angle <- .13+.7*i; radius <- if(i%%2L) 2 else .5
        queries[i*2L+1:2] <- radius*c(cos(angle),sin(angle))
    }
    x <- nn$from_vector(points,c(n,2L)); query <- nn$from_vector(queries,c(6L,2L))
    rbf <- function(a,b) {
        delta <- nn$sub(nn$expand_dims(a,1L),nn$expand_dims(b,0L))
        nn$exp(nn$mul(nn$sum(nn$square(delta),2L),-.5))
    }
    gram <- rbf(x,x)
    column_mean <- nn$mean(gram,0L,TRUE); grand_mean <- nn$mean(gram,c(0L,1L))
    center <- function(cross) nn$add(nn$sub(nn$sub(cross,column_mean),nn$mean(cross,1L,TRUE)),grand_mean)
    centered <- center(gram)
    eigen <- nn$eigh(centered)
    values <- nn$reverse(nn$slice(eigen$e,n-retained,retained),0L)
    vectors <- nn$reverse(nn$slice(eigen$v,c(0L,n-retained),c(n,retained)),1L)
    root <- nn$expand_dims(nn$sqrt(values),0L)
    scaled <- nn$div(vectors,root); training_scores <- nn$mul(vectors,root)
    project <- function(q) nn$matmul(center(rbf(q,x)),scaled)
    stopifnot(maximum_error(nn$sum(centered,1L))<2e-5,eigen$e$at(0L)>-1e-5,values$at(retained-1L)>.1)
    residual <- nn$sub(nn$matmul(centered,vectors),nn$mul(vectors,nn$expand_dims(values,0L)))
    stopifnot(maximum_error(residual)<2e-5,maximum_error(nn$sub(training_scores,project(x)))<2e-5)
    projected <- project(query)
    for (i in 0:5) {
        single <- project(nn$slice_axis(query,0L,i,i+1L))
        stopifnot(maximum_error(nn$sub(single,nn$slice_axis(projected,0L,i,i+1L)))<2e-6)
    }
    cat("Kernel PCA retained eigenvalues:",values$values(),"\n")
    cat("Centered Gram, eigenproblem, training reprojection and six independent query projections passed.\n")
}

main <- function(arguments=commandArgs(TRUE)) {
    linear(); kernel(); cat("All PCA and kernel-PCA checks passed.\n"); 0L
}
if (sys.nframe()==0L) quit(status=main())
