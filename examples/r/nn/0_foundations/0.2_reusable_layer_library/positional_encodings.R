# Native R NN example. Sinusoidal, learned, rotary and ALiBi checks.
# CMD: nix develop .#r --command Rscript examples/r/nn/0_foundations/0.2_reusable_layer_library/positional_encodings.R
.nn_common <- file.path(Sys.getenv("DATAMUNGE_EXAMPLES_ROOT",getwd()),"examples/r/nn/common.R")
if (!file.exists(.nn_common)) .nn_common <- system.file("examples/nn/common.R",package="datamungerNN",mustWork=TRUE)
source(.nn_common,local=TRUE)
rm(.nn_common)

random_tensor <- function(shape,rng) {rng$reset_normal();nn$from_vector(rng$normal32_values(prod(shape)),shape)}

sinusoidal <- function() {
    length <- 64L;width <- 16L;table <- nn$sinusoidal_encoding(length,width)
    cat("Sinusoidal wavelengths:",2*pi*10000^(2*(0:7)/width),"\n")
    for (position in c(0L,1L,2L,3L,10L,50L)) cat("Position",position,vapply(0:(width-1L),function(j) table$at(c(position,j)),numeric(1)),"\n")
    norms <- vapply(0:(length-1L),function(t) sqrt(sum(vapply(0:(width-1L),function(j) table$at(c(t,j))^2,numeric(1)))),numeric(1))
    stopifnot(max(abs(norms-sqrt(width/2)))<1e-5);error <- 0
    for (offset in c(1L,5L,20L)) for (t in 0:(length-offset-1L)) for (i in 0:(width%/%2L-1L)) {
        theta <- offset/10000^(2*i/width);sn <- table$at(c(t,2L*i));cs <- table$at(c(t,2L*i+1L))
        error <- max(error,abs(sn*cos(theta)+cs*sin(theta)-table$at(c(t+offset,2L*i))),abs(-sn*sin(theta)+cs*cos(theta)-table$at(c(t+offset,2L*i+1L))))
    }
    stopifnot(error<1e-6);cat("Row norm range:",range(norms),"fixed-rotation shift error:",error,"\n")
    for (t in c(0L,7L,30L)) {
        dots <- numeric(9L)
        for (offset in 0:8) {
            dot <- sum(vapply(0:(width-1L),function(j) table$at(c(t,j))*table$at(c(t+offset,j)),numeric(1)))
            reference <- sum(vapply(0:(width-1L),function(j) table$at(c(0L,j))*table$at(c(offset,j)),numeric(1)))
            stopifnot(abs(dot-reference)<1e-5);dots[offset+1L] <- dot
        }
        cat("Offset dot products at position",t,dots,"\n")
    }
}

learned <- function() {
    p <- nn$ParameterStore(1L);table <- nn$learned_encoding(p,"enc",10L,4L)
    used <- nn$slice(table,c(0L,0L),c(6L,4L));rng <- nn$Mt19937_64(2L)
    x <- random_tensor(c(3L,6L,4L),rng);r <- random_tensor(c(3L,6L,4L),rng)
    y <- nn$add(x,used);gradient <- nn$grad(nn$sum(nn$mul(y,r)),list(p$at("enc/pos")))[[1L]]
    stopifnot(identical(as.numeric(y$shape()),c(3,6,4)));norms <- numeric(10L)
    for (position in 0:9) {
        square <- sum(vapply(0:3,function(j) gradient$at(c(position,j))^2,numeric(1)))
        if (position<6L) stopifnot(square>0) else stopifnot(square==0)
        norms[position+1L] <- sqrt(square)
    }
    cat("Learned positional gradient row norms:",norms,"\n")
}

rotary <- function() {
    length <- 12L;width <- 8L;rng <- nn$Mt19937_64(3L)
    q <- random_tensor(c(1L,length,width),rng);k <- random_tensor(c(1L,length,width),rng)
    rotated <- nn$apply_rotary(q);error <- norm_error <- 0
    for (t in 0:(length-1L)) {
        before <- after <- 0
        for (j in 0:(width%/%2L-1L)) {
            theta <- t/10000^(2*j/width);a <- q$at(c(0L,t,2L*j));b <- q$at(c(0L,t,2L*j+1L))
            ra <- a*cos(theta)-b*sin(theta);rb <- a*sin(theta)+b*cos(theta)
            oa <- rotated$at(c(0L,t,2L*j));ob <- rotated$at(c(0L,t,2L*j+1L))
            error <- max(error,abs(ra-oa),abs(rb-ob));before <- before+a*a+b*b;after <- after+oa*oa+ob*ob
        }
        norm_error <- max(norm_error,abs(before-after))
    }
    stopifnot(error<1e-5,norm_error<1e-4)
    q_same <- nn$broadcast_to(nn$slice(q,c(0L,0L,0L),c(1L,1L,width)),c(1L,length,width))
    k_same <- nn$broadcast_to(nn$slice(k,c(0L,0L,0L),c(1L,1L,width)),c(1L,length,width))
    scores <- nn$matmul(nn$apply_rotary(q_same),nn$apply_rotary(k_same),FALSE,TRUE)
    cat("RoPE hand-rotation error / squared-norm error:",error,norm_error,"\n")
    for (offset in c(0L,1L,3L)) {
        values <- numeric()
        for (n in seq.int(0L,length-offset-1L,2L)) {
            value <- scores$at(c(0L,n+offset,n));stopifnot(abs(value-scores$at(c(0L,offset,0L)))<1e-4);values <- c(values,value)
        }
        cat("Relative-position score offset",offset,values,"\n")
    }
}

alibi <- function() {
    heads <- 4L;length <- 8L;slopes <- nn$alibi_slopes(heads);bias <- nn$alibi_bias(heads,length)
    weights <- nn$softmax(nn$add(nn$zeros(c(heads,length,length)),bias))
    cat("ALiBi slopes:",slopes,"head-0 bias:\n")
    for (m in 0:(length-1L)) cat(vapply(0:(length-1L),function(n) bias$at(c(0L,m,n)),numeric(1)),"\n")
    for (head in 0:(heads-1L)) {
        z <- sum(exp(-slopes[head+1L]*(length-1L-(0:(length-1L)))));row <- numeric(length)
        for (n in 0:(length-1L)) {
            actual <- weights$at(c(head,length-1L,n));expected <- exp(-slopes[head+1L]*(length-1L-n))/z
            stopifnot(abs(actual-expected)<1e-6);row[n+1L] <- actual
        }
        for (m in 0:(length-1L)) {
            stopifnot(abs(sum(vapply(0:(length-1L),function(n) weights$at(c(head,m,n)),numeric(1)))-1)<1e-6)
            if (m<length-1L) stopifnot(all(vapply((m+1L):(length-1L),function(n) weights$at(c(head,m,n))==0,logical(1))))
        }
        cat("Head",head,"last-query attention:",row,"\n")
    }
}

main <- function(arguments=commandArgs(TRUE)) {
    nn$set_num_threads(2L)
    sinusoidal();learned();rotary();alibi()
    cat("All positional encoding checks passed.\n")
    0L
}
if (sys.nframe()==0L) quit(status=main())
