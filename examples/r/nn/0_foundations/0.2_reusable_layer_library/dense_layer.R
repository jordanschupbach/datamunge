# Native R NN example. Dense forward passes, sharing, initializer scales and gradients.
# CMD: nix develop .#r --command Rscript examples/r/nn/0_foundations/0.2_reusable_layer_library/dense_layer.R
.nn_common <- file.path(Sys.getenv("DATAMUNGE_EXAMPLES_ROOT",getwd()),"examples/r/nn/common.R")
if (!file.exists(.nn_common)) .nn_common <- system.file("examples/nn/common.R",package="datamungerNN",mustWork=TRUE)
source(.nn_common,local=TRUE)
rm(.nn_common)

forward_pass <- function() {
    p <- nn$ParameterStore(1L)
    linear <- nn$DenseOptions(); no_bias <- nn$DenseOptions(); no_bias$bias <- FALSE
    relu <- nn$DenseOptions(); relu$activation <- nn$Activation$Relu; relu$bias_value <- .1
    x <- nn$from_vector(c(1,2,3,0,-1,1,2,2,2,-1,0,1),c(4L,3L))
    outputs <- list(linear=nn$dense(p,"lin",x,2L,linear),no_bias=nn$dense(p,"nob",x,2L,no_bias),relu=nn$dense(p,"relu",x,2L,relu))
    cat("x",x$str(),"W (lin)",p$at("lin/W")$str(),"\n")
    for (name in names(outputs)) cat(name,outputs[[name]]$str(),"\n")
    w <- p$at("relu/W"); b <- p$at("relu/b"); error <- 0
    for (i in 0:3) for (j in 0:1) {
        expected <- b$at(j)
        for (k in 0:2) expected <- expected+x$at(c(i,k))*w$at(c(k,j))
        error <- max(error,abs(max(0,expected)-outputs$relu$at(c(i,j))))
    }
    stopifnot(error<1e-5,p$count()==22L,abs(b$at(0L)-.1)<1e-7)
    cat("Dense forward max hand-computation error:",error,"parameters:",p$count(),"\n")
}

sharing_and_sequences <- function() {
    p <- nn$ParameterStore(2L)
    same <- nn$from_vector(c(1,2,3,0,-1,1),c(2L,3L))
    values <- nn$sub(nn$mul(nn$from_vector((0:29)%%7L),.1),.3)
    sequence <- nn$reshape(values,c(2L,5L,3L))
    a <- nn$dense(p,"shared",same,2L); b <- nn$dense(p,"shared",same$clone(),2L)
    output <- nn$dense(p,"shared",sequence,2L)
    stopifnot(identical(a$values(),b$values()),p$count()==8L,p$size()==2L,identical(as.numeric(output$shape()),c(2,5,2)))
    row <- nn$reshape(nn$select(nn$select(sequence,0L,1L),0L,3L),c(1L,3L))
    expected <- nn$select(nn$select(output,0L,1L),0L,3L)
    error <- max(abs(nn$sub(nn$dense(p,"shared",row,2L),expected)$values()))
    stopifnot(error<1e-6)
    cat("Three shared applications use eight parameters; sequence shape:",output$shape(),"row error:",error,"\n")
}

initializer_scales <- function() {
    cases <- list(list("glorot_uniform + tanh",nn$init$glorot_uniform(),nn$Activation$Tanh),
        list("glorot_uniform + relu",nn$init$glorot_uniform(),nn$Activation$Relu),
        list("he_normal + relu",nn$init$he_normal(),nn$Activation$Relu),
        list("lecun_normal + selu",nn$init$lecun_normal(),nn$Activation$Selu),
        list("lecun_normal + relu",nn$init$lecun_normal(),nn$Activation$Relu))
    depth <- 24L; width <- 256L; batch <- 512L
    rng <- nn$Mt19937_64(0L)
    inputs <- nn$from_vector(vapply(seq_len(batch*width),function(i) rng$normal32(),numeric(1)),c(batch,width))
    deviations <- nn$no_grad(function() {
        result <- list()
        for (entry in cases) {
            p <- nn$ParameterStore(3L); h <- inputs; row <- numeric(depth)
            for (layer in 0:(depth-1L)) {
                option <- nn$DenseOptions(); option$init <- entry[[2L]]; option$bias <- FALSE
                z <- nn$dense(p,paste0("l",layer),h,width,option)
                h <- nn$activate(z,entry[[3L]])
                row[layer+1L] <- sqrt(nn$mean(nn$square(nn$cast(z,nn$DType$Float64)))$item())
            }
            result[[length(result)+1L]] <- row
            cat(entry[[1L]],"pre-activation RMS at layers 1,4,8,12,16,20,24:",row[c(1L,4L,8L,12L,16L,20L,24L)],"\n")
        }
        result
    })
    stopifnot(abs(deviations[[1L]][1L]-1)<.1,deviations[[1L]][24L]>.08,deviations[[1L]][24L]<.3,
        deviations[[2L]][24L]<.01,deviations[[3L]][24L]>.6,deviations[[3L]][24L]<2.5,
        all(deviations[[4L]]>.85&deviations[[4L]]<1.15),deviations[[5L]][24L]<.01)
}

gradients <- function() {
    p <- nn$ParameterStore(4L)
    x <- nn$from_vector(c(1,2,3,0,-1,1,2,2,2,-1,0,1),c(4L,3L))
    target <- nn$from_vector(c(1,0,0,1,1,1,0,0),c(4L,2L))
    loss <- function() {
        y <- nn$dense(p,"d",x,2L)
        list(value=nn$mean(nn$square(nn$sub(y,target)),c(0L,1L)),y=y)
    }
    initial <- loss(); value <- initial$value; y <- initial$y
    w <- p$at("d/W"); b <- p$at("d/b")
    grads <- nn$grad(value,list(w,b)); gw <- grads[[1L]]; gb <- grads[[2L]]; error <- 0
    for (k in 0:2) for (j in 0:1) {
        expected <- 0
        for (i in 0:3) expected <- expected+.25*x$at(c(i,k))*(y$at(c(i,j))-target$at(c(i,j)))
        error <- max(error,abs(expected-gw$at(c(k,j))))
    }
    for (j in 0:1) {
        expected <- 0
        for (i in 0:3) expected <- expected+.25*(y$at(c(i,j))-target$at(c(i,j)))
        error <- max(error,abs(expected-gb$at(j)))
    }
    eps <- nn$scalar(1e-3)$item()
    finite_error <- nn$no_grad(function() {
        worst <- 0
        for (entry in list(list(w,gw),list(b,gb))) {
            parameter <- entry[[1L]]; gradient <- entry[[2L]]
            base <- parameter$clone(); original <- base$values(); expected <- gradient$values()
            flat <- nn$reshape(parameter,-1L)
            for (i in seq_len(parameter$numel())-1L) {
                cell <- nn$select(flat,0L,i)
                cell$fill_(original[i+1L]+eps); plus <- loss()$value$item(); parameter$copy_(base)
                cell$fill_(original[i+1L]-eps); minus <- loss()$value$item(); parameter$copy_(base)
                worst <- max(worst,abs((plus-minus)/(2*eps)-expected[i+1L]))
            }
        }
        worst
    })
    gmax <- max(abs(gw$values()))
    stopifnot(error<1e-5,finite_error/gmax<5e-3)
    cat("Loss:",value$item(),"dL/dW:",gw$str(),"dL/db:",gb$str(),"\n")
    cat("Gradient max closed-form error:",error,"finite-difference error:",finite_error,"relative:",finite_error/gmax,"\n")
}

main <- function(arguments=commandArgs(TRUE)) {
    nn$set_num_threads(2L)
    forward_pass(); sharing_and_sequences(); initializer_scales(); gradients()
    cat("All dense layer checks passed.\n")
    0L
}
if (sys.nframe()==0L) quit(status=main())
