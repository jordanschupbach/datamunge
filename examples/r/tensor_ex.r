# CMD: nix develop .#r --command env R_LIBS_USER=build/r/library Rscript examples/r/tensor_ex.r
# REPL: nix develop .#r --command env R_LIBS_USER=build/r/library R --no-save --quiet
# Dimensions are reported via ndim()/individual known values where the shape matters for the
# narrative.
library(datamunger)

cat("=================== Construction ===================\n")
z <- Tensor$zeros(c(2, 3))
cat("zeros([2,3]):", z$to_string(), "\n")

eye <- Tensor$eye(3)
cat("eye(3):", eye$to_string(), "\n")

r <- Tensor$arange(0.0, 12.0, 1.0)$reshape(c(3, 4))
cat("arange(0,12).reshape([3,4]):", r$to_string(), "\n")

cat("\n=================== Shape ops ===================\n")
rt <- r$transpose()
cat("transpose -> ndim", rt$ndim(), "\n")
sliced <- r$slice(1, 1, 3)
cat("slice(axis=1, start=1, stop=3):", sliced$to_string(), "\n")

cat("\n=================== Broadcasting arithmetic ===================\n")
col <- Tensor$from_values(c(3, 1), c(1, 2, 3))
row <- Tensor$from_values(c(1, 4), c(10, 20, 30, 40))
broadcast_sum <- col$add(row)
cat("(3,1) + (1,4) ->", broadcast_sum$to_string(), "\n")

cat("\n=================== Reductions ===================\n")
cat("r.sum() =", r$sum(), ", r.mean() =", r$mean(), "\n")
col_means <- r$mean_axis(0)
cat("column means (axis=0):", col_means$to_string(), "\n")

cat("\n=================== Linear algebra ===================\n")
a <- Tensor$from_values(c(2, 3), c(1, 2, 3, 4, 5, 6))
b <- Tensor$from_values(c(3, 2), c(7, 8, 9, 10, 11, 12))
cat("matmul(2x3, 3x2) ->", a$matmul(b)$to_string(), "\n")

v1 <- Tensor$from_values(c(3), c(1, 2, 3))
v2 <- Tensor$from_values(c(3), c(4, 5, 6))
cat("dot([1,2,3], [4,5,6]) =", v1$dot(v2), "\n")
cat("outer(v1, v2) ->", v1$outer(v2)$to_string(), "\n")

cat("\n=================== Comparisons & masks ===================\n")
mask <- r$greater_equal(Tensor$full(c(3, 4), 6.0))
cat("r >= 6 ->", mask$to_string(), "\n")
cat("count(r >= 6) =", mask$sum(), "\n")

cat("\n=================== A real dataset as a Tensor ===================\n")
iris <- data$iris()
n <- iris$nrows()
flat <- c()
for (i in 0:(n - 1)) {
  flat <- c(flat, iris$numeric_at("Sepal.Length", i), iris$numeric_at("Sepal.Width", i),
            iris$numeric_at("Petal.Length", i), iris$numeric_at("Petal.Width", i))
}
X <- Tensor$from_values(c(n, 4), flat)
cat("iris feature tensor ndim:", X$ndim(), "(150 rows x 4 features)\n")

feature_means <- X$mean_axis(0)
centered <- X$subtract(feature_means$reshape(c(1, 4)))
scatter <- centered$transpose()$matmul(centered)
cat("feature means (Sepal.Length, Sepal.Width, Petal.Length, Petal.Width):", feature_means$to_string(), "\n")
cat("(X-mean)^T (X-mean) [4x4 scatter matrix]:", scatter$to_string(16), "\n")

species <- c()
for (i in 0:(n - 1)) species <- c(species, iris$string_at("Species", i))
species_tensor <- Tensor$from_string_values(c(n), species)
setosa_mask <- species_tensor$equal(Tensor$from_string_values(c(1), c("setosa")))
cat("setosa count =", setosa_mask$sum(), "(of", n, "rows)\n")
