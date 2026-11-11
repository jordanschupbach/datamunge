# CMD: nix develop .#r --command Rscript examples/r/bspline_ex.r
# Fits a two-dimensional B-spline surface with the bs(x, y) formula term.
# The formula uses cubic B-splines with six basis functions per input axis.
library(datamunger)

x <- numeric()
y <- numeric()
z <- numeric()
for (iy in 0:7) {
  for (ix in 0:7) {
    xv <- ix / 7
    yv <- iy / 7
    x <- c(x, xv)
    y <- c(y, yv)
    z <- c(z, sin(xv) + yv ^ 2)
  }
}

data <- DataFrame$empty()
data$add_numeric_column("x", x)
data$add_numeric_column("y", y)
data$add_numeric_column("z", z)

surface <- LM$new(data, "z ~ bs(x, y)")
cat("Fitted z ~ bs(x, y)\n")
surface$print_summary()

new_points <- DataFrame$empty()
new_points$add_numeric_column("x", c(0.25, 0.75))
new_points$add_numeric_column("y", c(0.50, 0.25))
cat("Predictions:\n")
print(surface$predict(new_points))
