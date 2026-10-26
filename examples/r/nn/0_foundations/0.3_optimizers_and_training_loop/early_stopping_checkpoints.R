# Native R NN example. Validation monitoring and complete model/optimizer checkpoints.
.nn_common <- file.path(Sys.getenv("DATAMUNGE_EXAMPLES_ROOT",getwd()),"examples/r/nn/common.R")
if (!file.exists(.nn_common)) .nn_common <- system.file("examples/nn/common.R",package="datamungerNN",mustWork=TRUE)
source(.nn_common,local=TRUE)
rm(.nn_common)

Model <- function(initial=0) {
    store <- nn$ParameterStore(); w <- store$get("w",integer(),nn$init$constant(initial))
    cfg <- nn$OptimizerConfig(); cfg$kind <- nn$OptimizerKind$Adam
    optimizer <- nn$Optimizer(store,.1,cfg)
    train <- function(count) {
        for (i in seq_len(count)) {
            optimizer$set_learning_rate(nn$schedule$step_decay(.1,.5,25L,optimizer$iterations()))
            optimizer$minimize(nn$square(nn$sub(w,4)))
        }
    }
    validation <- function() nn$no_grad(function() nn$square(nn$sub(w,2))$item())
    state <- function() {
        result <- list()
        for (name in store$names()) result[[paste0("params/",name)]] <- store$at(name)$detach()$clone()
        slots <- optimizer$state()
        for (i in seq_len(slots$size())-1L) {
            pair <- slots$`__getitem__`(i)
            result[[paste0("optimizer/",pair$first)]] <- pair$second$clone()
        }
        result
    }
    save <- function(path) nn$save_npz(path,state())
    restore <- function(path,weights_only=FALSE) {
        archive <- nn$load_npz(path); params <- list(); slots <- nn$NamedTensors()
        for (name in names(archive)) {
            if (startsWith(name,"params/")) params[[substring(name,8L)]] <- archive[[name]]
            else if (startsWith(name,"optimizer/")) slots$push_back(nn$NamedTensor(substring(name,11L),archive[[name]]))
            else stop("unexpected checkpoint entry ",name)
        }
        if (!setequal(names(params),store$names())) stop("checkpoint parameter names do not match")
        if (!weights_only&&slots$size()==0L) stop("checkpoint has no optimizer state")
        if (!weights_only) {
            for (i in seq_len(slots$size())-1L) {
                pair <- slots$`__getitem__`(i)
                if (!pair$first %in% c("step","nadam_product","m/0","v/0") || pair$second$dim()!=0L)
                    stop("invalid scalar Adam checkpoint slot ",pair$first)
            }
        }
        probe <- Model()
        for (name in names(params)) probe$store$set(name,params[[name]])
        if (!weights_only) probe$optimizer$load_state(slots)
        for (name in names(params)) store$set(name,params[[name]])
        if (!weights_only) optimizer$load_state(slots)
    }
    list(store=store,w=w,optimizer=optimizer,train=train,validation=validation,state=state,save=save,restore=restore)
}

monitor_checks <- function() {
    stop <- nn$EarlyStopping(2L,.1)
    stopifnot(stop$observe(1,0L)$improved,!stop$observe(.95,1L)$improved)
    stopifnot(stop$observe(.8,2L)$improved,stop$misses()==0L)
    stopifnot(!stop$observe(.8,3L)$stop,stop$observe(.85,4L)$stop)
    maximize <- nn$EarlyStopping(0L,0,TRUE)
    stopifnot(!maximize$observe(.4,0L)$stop,maximize$observe(.5,1L)$improved,maximize$observe(.5,2L)$stop)
    stopifnot(tryCatch({maximize$observe(.6,2L);FALSE},error=function(e) TRUE))
    cat("Patience, min_delta, tie, maximize and epoch-order checks passed.\n")
}

best_checkpoint <- function(path,epochs) {
    model <- Model(); monitor <- nn$EarlyStopping(6L,1e-6); stopped <- FALSE
    for (epoch in 0:(epochs-1L)) {
        model$train(1L); last_loss <- model$validation()
        decision <- monitor$observe(last_loss,epoch)
        if (decision$improved) model$save(path)
        if (decision$stop) {stopped <- TRUE; break}
    }
    if (epochs>=200L) stopifnot(stopped)
    model$restore(path); restored <- model$validation()
    stopifnot(abs(restored-monitor$best())<1e-7,restored<=last_loss)
    if (stopped) stopifnot(restored<.01,restored<last_loss)
    stopifnot(model$optimizer$iterations()==monitor$best_epoch()+1L)
    cat(sprintf("%s after epoch %d; best epoch %d; validation %.9g -> restored %.9g\n",if(stopped) "Stopped" else "Budget exhausted",epoch,monitor$best_epoch(),last_loss,restored))
    cat(sprintf("Restored w=%.9g, optimizer step=%d\n",model$w$item(),model$optimizer$iterations()))
}

resume <- function(path,prefix,suffix) {
    original <- Model(); original$train(prefix); original$save(path); original$train(suffix)
    expected <- original$state()
    restored <- Model(NaN); restored$restore(path); restored$train(suffix)
    actual <- restored$state(); stopifnot(identical(names(actual),names(expected))); worst <- 0
    for (name in names(actual)) {
        a <- actual[[name]]; e <- expected[[name]]
        stopifnot(identical(a$shape(),e$shape()))
        if (name=="optimizer/step") stopifnot(a$item()==prefix+suffix,e$item()==prefix+suffix)
        error <- max(abs(a$values()-e$values())); stopifnot(error<1e-7); worst <- max(worst,error)
    }
    cat(sprintf("%d + save/restore + %d updates = %d uninterrupted updates; max state error %.9g\n",prefix,suffix,prefix+suffix,worst))
    weights_only <- Model(); weights_only$restore(path,TRUE); weights_only$train(suffix)
    correct <- restored$w$item(); different <- weights_only$w$item()
    stopifnot(abs(different-correct)>1e-7)
    cat(sprintf("Weights-only resume w=%.9g; full-state resume w=%.9g\n",different,correct))
    stopifnot(tryCatch({restored$restore(paste0(path,"_missing.npz"));FALSE},error=function(e) TRUE),restored$w$item()==correct)
    broken <- file.path(dirname(path),"broken.npz"); arrays <- original$state()
    arrays[['params/w']] <- nn$scalar(99); arrays[['optimizer/m/0']] <- nn$zeros(2L)
    nn$save_npz(broken,arrays); before <- restored$state()
    stopifnot(tryCatch({restored$restore(broken);FALSE},error=function(e) TRUE))
    after <- restored$state()
    for (name in names(before)) stopifnot(identical(after[[name]]$values(),before[[name]]$values()))
}

main <- function(arguments=commandArgs(TRUE)) {
    args <- nn_options(arguments,list(epochs=5L,full=FALSE))
    if (is.na(args$epochs)||args$epochs<2L) stop("at least two epochs required for checkpoint resume")
    directory <- file.path(nn_output_root(args$full),"checkpoints/early_stopping")
    dir.create(directory,recursive=TRUE,showWarnings=FALSE)
    monitor_checks(); best_checkpoint(file.path(directory,"best.npz"),args$epochs)
    prefix <- max(1L,args$epochs%/%2L)
    resume(file.path(directory,"resume.npz"),prefix,args$epochs-prefix)
    cat("All early stopping and checkpoint checks passed.\n")
    0L
}
if (sys.nframe()==0L) quit(status=main())
