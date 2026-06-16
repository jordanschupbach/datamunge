library(octrar)

octrar::hello()

# Can only coerce (can't use methods directly without a wrapper)
v <- octrar::DVector(c(0.0, 0.0, 0.0,
    1.0, 0.0, 0.0,
    1.0, 1.0, 0.0,
    0.0, 1.0, 0.0))

# This works as expected
p <- octrar::DPair(1.1, 2.2)
p$first
p$second

# This works as expected
p <- octrar::IPair(1.1, 2.2) # converts to integer
p$first
p$second


