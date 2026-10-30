#!/usr/bin/env Rscript
# Prepare a standard R package from the functional SWIG API and R examples.
# No Python tooling or compiled experiment entry points are used.
arguments <- commandArgs(TRUE)
if (length(arguments)) stop("This package preparation command takes no arguments")
if (!file.exists("prebindings/nn/src/nn.i")) stop("Run from the repository root")

write_changed <- function(path, text) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  bytes <- charToRaw(paste0(paste(text, collapse = "\n"), "\n"))
  if (!file.exists(path) || !identical(readBin(path, "raw", file.info(path)$size), bytes))
    writeBin(bytes, path)
}
copy_changed <- function(source, destination) {
  dir.create(dirname(destination), recursive = TRUE, showWarnings = FALSE)
  if (!file.exists(destination) ||
      !identical(unname(tools::md5sum(source)), unname(tools::md5sum(destination)))) {
    if (!file.copy(source, destination, overwrite = TRUE, copy.date = TRUE))
      stop("Cannot copy ", source, " to ", destination)
  }
}
copy_tree <- function(source, destination, pattern = NULL) {
  files <- list.files(source, recursive = TRUE, full.names = TRUE, all.files = TRUE,
                      no.. = TRUE, pattern = pattern)
  files <- files[!dir.exists(files)]
  files <- files[!grepl("[.](o|so|dll|dylib)$", files)]
  for (file in files) copy_changed(file, file.path(destination, substring(file, nchar(source) + 2L)))
}

stage <- "build/nn-r-native/source/datamungerNN"
dir.create(file.path(stage, "src"), recursive = TRUE, showWarnings = FALSE)
copy_tree("prebindings/nn/r", stage)
copy_tree("include", file.path(stage, "inst/include"))
# Header-only example components bound by examples/bindings/nn_components.i.
copy_tree("examples/nn/common", file.path(stage, "inst/include/common"), "[.]hpp$")
copy_tree("src/datamunge", file.path(stage, "src/datamunge"), "[.](cpp|h|hpp)$")
for (file in list.files(pattern = "^LICENSE")) if (!dir.exists(file)) copy_changed(file, file.path(stage, file))

# Ship the actual R implementations and their small utility module.
example_root <- "examples/r/nn"
examples <- list.files(example_root, "[.]R$", recursive = TRUE, full.names = TRUE)
native <- examples[vapply(examples, function(file)
  startsWith(readLines(file, n = 1L, warn = FALSE), "# Native R NN example"), logical(1))]
installed_examples <- file.path(stage, "inst/examples/nn")
# Flat installed names avoid non-portable >100-byte source-archive paths.
# The checkout retains its numbered tutorial layout.
if (dir.exists(installed_examples)) {
  stale_directories <- list.dirs(installed_examples, recursive = FALSE, full.names = TRUE)
  if (length(stale_directories)) unlink(stale_directories, recursive = TRUE)
  stale_files <- setdiff(list.files(installed_examples, full.names = TRUE),
                        file.path(installed_examples, c("common.R", basename(native))))
  if (length(stale_files)) unlink(stale_files)
}
for (file in c(file.path(example_root, "common.R"), native))
  copy_changed(file, file.path(installed_examples, basename(file)))
# Shared helpers (model pieces/data readers) sourced by examples through nn_helper().
helpers <- list.files(file.path(example_root, "common"), "[.]R$", full.names = TRUE)
if (length(helpers)) dir.create(file.path(installed_examples, "common"), recursive = TRUE, showWarnings = FALSE)
for (file in helpers) copy_changed(file, file.path(installed_examples, "common", basename(file)))

# Stage SWIG output atomically and normalize its temporary invocation path.
temporary <- tempfile(".swig-", tmpdir = "build/nn-r-native")
dir.create(temporary, recursive = TRUE)
flags <- c("-c++", "-r", "-Iinclude", "-Iexamples/bindings", "-Iexamples/nn", "-dll", "datamungerNN", "-package", "datamungerNN")
if (identical(Sys.info()[["sysname"]], "Linux")) flags <- c(flags, "-DSWIGWORDSIZE64")
flags <- c(flags, "-outdir", temporary, "-o", file.path(temporary, "nn_r_wrap.cpp"), "prebindings/nn/src/nn.i")
status <- system2("swig", vapply(flags, shQuote, character(1)))
if (status != 0L) stop("SWIG generation failed with status ", status)
for (file in c("nn_r_wrap.cpp", "datamungerNN.R")) {
  text <- readLines(file.path(temporary, file), warn = FALSE)
  text <- gsub(temporary, "src", text, fixed = TRUE)
  if (file == "datamungerNN.R") {
    # The R backend emits original C++ field names in S4 representations,
    # including ignored/renamed fields. Quote reserved slot names so that
    # components with an `in` channel count produce valid R source.
    text <- sub("^([[:space:]]*)(in)([[:space:]]*=)", "\\1`\\2`\\3", text)
    text <- gsub("([@$])in\\b", "\\1`in`", text, perl = TRUE)
  }
  destination <- if (file == "nn_r_wrap.cpp") file.path(stage, "src", file) else file.path(stage, "R/nn_native.R")
  write_changed(destination, text)
}
unlink(temporary, recursive = TRUE)

sources <- list.files(file.path(stage, "src/datamunge"), "[.]cpp$", recursive = TRUE)
objects <- paste0("datamunge/", sub("[.]cpp$", ".o", sources))
self_contained <- c(paste("OBJECTS =", paste(c("nn_r_wrap.o", objects), collapse = " ")),
  "PKG_CPPFLAGS = -I../inst/include", "PKG_LIBS = -pthread")
# Changing the immutable Nix library prefix must also relink an existing wrapper.
core_flags <- "self-contained"
if (nzchar(Sys.which("pkg-config")) && system2("pkg-config", c("--exists", "datamunge")) == 0L)
  core_flags <- system2("pkg-config", c("--cflags", "--libs", "--variable=libdir", "datamunge"), stdout=TRUE)
write_changed(file.path(stage,"src/system-core.txt"),core_flags)
system_core <- c("OBJECTS = nn_r_wrap.o", "PKG_CPPFLAGS = -I../inst/include $(shell pkg-config --cflags datamunge)",
  "PKG_LIBS = $(shell pkg-config --libs datamunge)")
makevars <- c("CXX_STD = CXX23", "ifeq ($(DATAMUNGE_USE_SYSTEM_LIB),1)", system_core,
  "PKG_LIBS += -Wl,-rpath,$(shell pkg-config --variable=libdir datamunge)", "else", self_contained,
  "ifeq ($(shell uname -s),Darwin)",
  "PKG_CPPFLAGS += -DDATAMUNGE_HAVE_CBLAS=1 -DDATAMUNGE_HAVE_LAPACK=1 -DACCELERATE_NEW_LAPACK=1",
  "PKG_LIBS += -framework Accelerate", "endif", "endif", "all: $(SHLIB)",
  "HEADERS := $(shell find ../inst/include -name '*.hpp')", "$(OBJECTS): $(HEADERS)",
  "ifeq ($(DATAMUNGE_USE_SYSTEM_LIB),1)", "$(SHLIB): system-core.txt", "endif")
write_changed(file.path(stage, "src/Makevars"), makevars)
write_changed(file.path(stage, "src/Makevars.win"), c("CXX_STD = CXX23", self_contained, "all: $(SHLIB)"))
write_changed(file.path(stage, ".Rbuildignore"), "[.](o|so|dll|dylib)$")
cat(sprintf("Prepared datamungerNN with %d native R examples in %s\n", length(native), stage))
