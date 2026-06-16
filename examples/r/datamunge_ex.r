library(datamunger)

datamunger::hello()

# Can only coerce (can't use methods directly without a wrapper)
v <- datamunger::DVector(c(0.0, 0.0, 0.0,
    1.0, 0.0, 0.0,
    1.0, 1.0, 0.0,
    0.0, 1.0, 0.0))

# This works as expected
p <- datamunger::DPair(1.1, 2.2)
p$first
p$second

# This works as expected
p <- datamunger::IPair(1.1, 2.2) # converts to integer
p$first
p$second


