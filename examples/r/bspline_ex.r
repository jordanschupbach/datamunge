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

data <- DataFrame_empty()
DataFrame_add_numeric_column(data, "x", x)
DataFrame_add_numeric_column(data, "y", y)
DataFrame_add_numeric_column(data, "z", z)

surface <- LM(data, "z ~ bs(x, y)")
cat("Fitted z ~ bs(x, y)\n")
LM_print_summary(surface)

new_points <- DataFrame_empty()
DataFrame_add_numeric_column(new_points, "x", c(0.25, 0.75))
DataFrame_add_numeric_column(new_points, "y", c(0.50, 0.25))
cat("Predictions:\n")
print(LM_predict(surface, new_points))
