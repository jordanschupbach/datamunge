# CMD: nix develop .#r --command env R_LIBS_USER=build/r/library Rscript examples/r/stl_ex.r
# REPL: nix develop .#r --command env R_LIBS_USER=build/r/library R --no-save --quiet
# STL containers cross the binding as plain R values: std::vector as an atomic vector (or a
# list for class elements), std::pair as list(first =, second =).
library(datamunger)

v <- make_dvector(1.5, 2.5, 3.5)
print(v)
print(sum_dvector(c(v, 10)))

p <- make_dpair(1.1, 2.2)
p$first
p$second
print(sum_dpair(p))

# The DVector/DPair/IPair names of the old SWIG bindings build the same plain values.
v <- datamunger::DVector(c(0.0, 0.0, 0.0,
    1.0, 0.0, 0.0,
    1.0, 1.0, 0.0,
    0.0, 1.0, 0.0))
print(length(v))
p <- datamunger::IPair(1.1, 2.2) # converts to integer
p$first
p$second
