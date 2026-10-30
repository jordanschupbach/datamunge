# Native R NN example. Pooling windows, gradients, global/3-D ops and translation.
# CMD: nix develop .#r --command Rscript examples/r/nn/0_foundations/0.2_reusable_layer_library/pooling.R
.nn_common <- file.path(Sys.getenv("DATAMUNGE_EXAMPLES_ROOT",getwd()),"examples/r/nn/common.R")
if (!file.exists(.nn_common)) .nn_common <- system.file("examples/nn/common.R",package="datamungerNN",mustWork=TRUE)
source(.nn_common,local=TRUE)
rm(.nn_common)

require_values <- function(tensor,expected) stopifnot(length(tensor$values())==length(expected),max(abs(tensor$values()-expected))<1e-6)
image <- function(height,width,values) nn$from_vector(values,c(1L,height,width,1L))

by_hand <- function() {
    x <- image(5L,5L,1:25)
    cases <- list(list("max VALID",nn$max_pool2d,nn$Padding$Valid,c(7,9,17,19)),
        list("avg VALID",nn$avg_pool2d,nn$Padding$Valid,c(4,6,14,16)),
        list("max SAME",nn$max_pool2d,nn$Padding$Same,c(7,9,10,17,19,20,22,24,25)),
        list("avg SAME",nn$avg_pool2d,nn$Padding$Same,c(4,6,7.5,14,16,17.5,21.5,23.5,25)))
    for (entry in cases) {
        output <- entry[[2L]](x,2L,2L,entry[[3L]]);require_values(output,entry[[4L]])
        cat(entry[[1L]],output$str(),"\n")
    }
    cat("SAME border averages exclude padding; bottom-right is 25/1.\n")
}

gradients <- function() {
    x <- image(4L,4L,c(9,1,1,1,1,1,1,8,1,1,1,1,7,1,6,1))$requires_grad_()
    gm <- nn$grad(nn$sum(nn$max_pool2d(x,2L,2L)),list(x))[[1L]]
    ga <- nn$grad(nn$sum(nn$avg_pool2d(x,2L,2L)),list(x))[[1L]]
    require_values(gm,c(1,0,0,0,0,0,0,1,0,0,0,0,1,0,1,0));require_values(ga,rep(.25,16L))
    cat("Max-pool gradient:",gm$str(),"Average-pool gradient:",ga$str(),"\n")
}

global_and_3d <- function() {
    rng <- nn$Mt19937_64(1L)
    x <- nn$from_vector(vapply(seq_len(2L*7L*5L*3L),function(i) rng$normal32(),numeric(1)),c(2L,7L,5L,3L))
    volume <- nn$from_vector(vapply(seq_len(8L*8L*8L*2L),function(i) rng$normal32(),numeric(1)),c(1L,8L,8L,8L,2L))
    average <- nn$global_avg_pool2d(x);maximum <- nn$global_max_pool2d(x)
    m3 <- nn$max_pool3d(volume,2L,2L);a3 <- nn$avg_pool3d(volume,2L,2L)
    error_avg <- 0;error_max <- 0
    for (n in 0:1) for (c in 0:2) {
        values <- numeric()
        for (i in 0:6) for (j in 0:4) values <- c(values,x$at(c(n,i,j,c)))
        error_avg <- max(error_avg,abs(sum(values)/35-average$at(c(n,c))))
        error_max <- max(error_max,abs(max(values)-maximum$at(c(n,c))))
    }
    window <- numeric()
    for (a in 0:1) for (b in 0:1) for (c in 0:1) window <- c(window,volume$at(c(0L,2L+a,4L+b,6L+c,1L)))
    op_m <- m3$at(c(0L,1L,2L,3L,1L));op_a <- a3$at(c(0L,1L,2L,3L,1L))
    stopifnot(identical(as.numeric(average$shape()),c(2,3)),identical(as.numeric(m3$shape()),c(1,4,4,4,2)),identical(m3$shape(),a3$shape()))
    stopifnot(error_avg<1e-6,error_max==0,abs(op_m-max(window))<1e-6,abs(op_a-sum(window)/8)<1e-6)
    cat("Global average / max errors:",error_avg,error_max,"3-D max / avg cell:",op_m,op_a,"\n")
}

translation <- function() {
    rng <- nn$Mt19937_64(2L)
    original <- vapply(seq_len(256L),function(i) rng$uniform32(),numeric(1))
    shifted <- numeric(256L)
    for (i in 0:15) for (j in 1:15) shifted[i*16L+j+1L] <- original[i*16L+j]
    a <- image(16L,16L,original);b <- image(16L,16L,shifted)
    pools <- function(x) {
        p <- nn$max_pool2d(x,2L,2L)
        list(p,nn$max_pool2d(p,2L,2L),nn$global_max_pool2d(x))
    }
    out <- pools(a);shift <- pools(b)
    agreement <- function(u,v) 100*mean(abs(u$values()-v$values())<=1e-6)
    raw <- agreement(a,b);one <- agreement(out[[1L]],shift[[1L]]);two <- agreement(out[[2L]],shift[[2L]])
    global_same <- abs(out[[3L]]$item()-shift[[3L]]$item())<1e-6
    stopifnot(round(raw)==0,round(one)==28,round(two)==31,global_same)
    cat("Translation agreement percent, raw / one / two pools:",raw,one,two,"global max unchanged:",global_same,"\n")
}

main <- function(arguments=commandArgs(TRUE)) {
    nn$set_num_threads(2L)
    by_hand();gradients();global_and_3d();translation()
    cat("All pooling checks passed.\n")
    0L
}
if (sys.nframe()==0L) quit(status=main())
