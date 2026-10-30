# Native R NN example. Layer, RMS, group and instance normalization checks.
# CMD: nix develop .#r --command Rscript examples/r/nn/0_foundations/0.2_reusable_layer_library/normalization_layers.R
.nn_common <- file.path(Sys.getenv("DATAMUNGE_EXAMPLES_ROOT",getwd()),"examples/r/nn/common.R")
if (!file.exists(.nn_common)) .nn_common <- system.file("examples/nn/common.R",package="datamungerNN",mustWork=TRUE)
source(.nn_common,local=TRUE)
rm(.nn_common)

EPS <- 1e-5
standardize <- function(values,groups,group,center=TRUE) {
    sums <- numeric(groups);squares <- numeric(groups);counts <- numeric(groups)
    for (i in seq_along(values)) {
        g <- group(i-1L)+1L;value <- values[i]
        sums[g] <- sums[g]+value;squares[g] <- squares[g]+value*value;counts[g] <- counts[g]+1
    }
    vapply(seq_along(values),function(i) {
        g <- group(i-1L)+1L;average <- if(center) sums[g]/counts[g] else 0
        variance <- squares[g]/counts[g]-average*average
        (values[i]-average)/sqrt(variance+EPS)
    },numeric(1))
}
max_error <- function(tensor,expected) {stopifnot(length(tensor$values())==length(expected));max(abs(tensor$values()-expected))}
difference <- function(a,b) max_error(a,b$values())

layer_and_rms <- function() {
    p <- nn$ParameterStore();rng <- nn$Mt19937_64(1L)
    values <- vapply(1:18,function(i) rng$normal32(2,3),numeric(1));x <- nn$from_vector(values,c(3L,6L))
    ln <- nn$layer_norm(p,"ln",x,EPS);rms <- nn$rms_norm(p,"rms",x,EPS)
    error_ln <- max_error(ln,standardize(values,3L,function(i) i%/%6L))
    error_rms <- max_error(rms,standardize(values,3L,function(i) i%/%6L,FALSE))
    stopifnot(error_ln<1e-5,error_rms<1e-5)
    cat("Layer/RMS hand errors:",error_ln,error_rms,"\n")
    cat("Row 0 input:",values[1:6],"LN:",ln$values()[1:6],"RMS:",rms$values()[1:6],"\n")
    rng$reset_normal();sequence <- vapply(1:48,function(i) rng$normal32(-1,2),numeric(1))
    output <- nn$layer_norm(p,"ln",nn$from_vector(sequence,c(2L,4L,6L)),EPS)
    error <- max_error(output,standardize(sequence,8L,function(i) i%/%6L))
    stopifnot(identical(as.numeric(output$shape()),c(2,4,6)),error<1e-5)
    changed <- values;changed[7:12] <- 100*((0:5)-2L)
    altered <- nn$layer_norm(p,"ln",nn$from_vector(changed,c(3L,6L)),EPS)
    batch_difference <- difference(nn$select(ln,0L,0L),nn$select(altered,0L,0L))
    stopifnot(batch_difference==0)
    centered <- nn$sub(x,nn$mean(x,1L,TRUE))
    zero_difference <- difference(nn$layer_norm(p,"ln",centered,EPS),nn$rms_norm(p,"rms",centered,EPS))
    stopifnot(zero_difference<1e-5)
    cat("Sequence hand error:",error,"other-row change:",batch_difference,"zero-mean LN/RMS difference:",zero_difference,"\n")
}

group_norms <- function() {
    p <- nn$ParameterStore();rng <- nn$Mt19937_64(2L)
    raw <- nn$from_vector(vapply(1:256,function(i) rng$normal32(),numeric(1)),c(2L,4L,4L,8L))
    channels <- nn$arange(0,8)
    x <- nn$add(nn$mul(raw,nn$add(channels,1)),nn$mul(channels,3))
    values <- x$values();outputs <- list()
    for (groups in c(1L,2L,4L,8L)) {
        output <- nn$group_norm(p,paste0("gn",groups),x,groups,EPS);outputs[[length(outputs)+1L]] <- output
        per <- 8L%/%groups
        expected <- standardize(values,2L*groups,function(i) (i%/%128L)*groups+(i%%8L)%/%per)
        error <- max_error(output,expected);stopifnot(error<1e-5)
        cat("GroupNorm G=",groups,"hand error:",error,"\n")
    }
    instance <- nn$instance_norm(p,"in",x,EPS)
    stopifnot(difference(outputs[[4L]],instance)==0)
    mean1 <- numeric(8L);mean8 <- numeric(8L)
    for (c in 0:7) for (h in 0:3) for (w in 0:3) {
        mean1[c+1L] <- mean1[c+1L]+outputs[[1L]]$at(c(0L,h,w,c))/16
        mean8[c+1L] <- mean8[c+1L]+outputs[[4L]]$at(c(0L,h,w,c))/16
    }
    stopifnot(max(abs(mean8))<1e-5)
    cat("G=1 example-0 channel means:",mean1,"\n")
    cat("G=8 example-0 channel means:",mean8,"InstanceNorm equals G=C exactly.\n")
}

invariances <- function() {
    p <- nn$ParameterStore();rng <- nn$Mt19937_64(3L)
    x <- nn$from_vector(vapply(1:64,function(i) rng$normal32(.5,2),numeric(1)),c(4L,16L))
    baseline_ln <- nn$layer_norm(p,"ln",x,EPS);baseline_rms <- nn$rms_norm(p,"rms",x,EPS)
    cases <- list(list("shift",1,5,TRUE,FALSE),list("scale",3,0,TRUE,TRUE),
        list("negative scale",-2,0,FALSE,FALSE),list("scale and shift",3,5,TRUE,FALSE))
    for (entry in cases) {
        transformed <- nn$add(nn$mul(x,entry[[2L]]),entry[[3L]])
        dl <- difference(nn$layer_norm(p,"ln",transformed,EPS),baseline_ln)
        dr <- difference(nn$rms_norm(p,"rms",transformed,EPS),baseline_rms)
        stopifnot((dl<1e-4)==entry[[4L]],(dr<1e-4)==entry[[5L]])
        cat(entry[[1L]],"LN/RMS differences:",dl,dr,"\n")
    }
    row <- nn$select(baseline_ln,0L,0L)$values();average <- mean(row);variance <- mean(row*row)-average*average
    stopifnot(abs(average)<1e-6,abs(variance-1)<1e-4)
    cat("LN output row-0 mean / variance:",average,variance,"\n")
}

main <- function(arguments=commandArgs(TRUE)) {
    nn$set_num_threads(2L)
    layer_and_rms();group_norms();invariances()
    cat("All normalization layer checks passed.\n")
    0L
}
if (sys.nframe()==0L) quit(status=main())
