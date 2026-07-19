# See examples/r/lm_ex.r for notes on the flat ClassName_method(obj, ...) call convention.
# Note: Tensor_shape()'s return value (a vector<size_t>) can't be marshaled back to R in this
# package's bindings (a distinct bug from the vector<string> issues -- see
# datamunge_r_dollar_dispatch_bug.md), so this avoids printing it directly and instead reports
# dimensions via ndim()/individual known values where the shape matters for the narrative.
library(datamunger)

cat("=================== Construction ===================\n")
z <- Tensor_zeros(c(2, 3))
cat("zeros([2,3]):", Tensor_to_string(z), "\n")

eye <- Tensor_eye(3)
cat("eye(3):", Tensor_to_string(eye), "\n")

r <- Tensor_reshape(Tensor_arange(0.0, 12.0, 1.0), c(3, 4))
cat("arange(0,12).reshape([3,4]):", Tensor_to_string(r), "\n")

cat("\n=================== Shape ops ===================\n")
rt <- Tensor_transpose(r)
cat("transpose -> ndim", Tensor_ndim(rt), "\n")
sliced <- Tensor_slice(r, 1, 1, 3)
cat("slice(axis=1, start=1, stop=3):", Tensor_to_string(sliced), "\n")

cat("\n=================== Broadcasting arithmetic ===================\n")
col <- Tensor_from_values(c(3, 1), c(1, 2, 3))
row <- Tensor_from_values(c(1, 4), c(10, 20, 30, 40))
broadcast_sum <- Tensor_add(col, row)
cat("(3,1) + (1,4) ->", Tensor_to_string(broadcast_sum), "\n")

cat("\n=================== Reductions ===================\n")
cat("r.sum() =", Tensor_sum(r), ", r.mean() =", Tensor_mean(r), "\n")
col_means <- Tensor_mean_axis(r, 0)
cat("column means (axis=0):", Tensor_to_string(col_means), "\n")

cat("\n=================== Linear algebra ===================\n")
a <- Tensor_from_values(c(2, 3), c(1, 2, 3, 4, 5, 6))
b <- Tensor_from_values(c(3, 2), c(7, 8, 9, 10, 11, 12))
cat("matmul(2x3, 3x2) ->", Tensor_to_string(Tensor_matmul(a, b)), "\n")

v1 <- Tensor_from_values(c(3), c(1, 2, 3))
v2 <- Tensor_from_values(c(3), c(4, 5, 6))
cat("dot([1,2,3], [4,5,6]) =", Tensor_dot(v1, v2), "\n")
cat("outer(v1, v2) ->", Tensor_to_string(Tensor_outer(v1, v2)), "\n")

cat("\n=================== Comparisons & masks ===================\n")
mask <- Tensor_greater_equal(r, Tensor_full(c(3, 4), 6.0))
cat("r >= 6 ->", Tensor_to_string(mask), "\n")
cat("count(r >= 6) =", Tensor_sum(mask), "\n")

cat("\n=================== A real dataset as a Tensor ===================\n")
iris <- DataFrame_iris()
n <- DataFrame_nrows(iris)
flat <- c()
for (i in 0:(n - 1)) {
  flat <- c(flat, DataFrame_numeric_at(iris, "Sepal.Length", i), DataFrame_numeric_at(iris, "Sepal.Width", i),
            DataFrame_numeric_at(iris, "Petal.Length", i), DataFrame_numeric_at(iris, "Petal.Width", i))
}
X <- Tensor_from_values(c(n, 4), flat)
cat("iris feature tensor ndim:", Tensor_ndim(X), "(150 rows x 4 features)\n")

feature_means <- Tensor_mean_axis(X, 0)
centered <- Tensor_subtract(X, Tensor_reshape(feature_means, c(1, 4)))
scatter <- Tensor_matmul(Tensor_transpose(centered), centered)
cat("feature means (Sepal.Length, Sepal.Width, Petal.Length, Petal.Width):", Tensor_to_string(feature_means), "\n")
cat("(X-mean)^T (X-mean) [4x4 scatter matrix]:", Tensor_to_string(scatter, 16), "\n")

species <- c()
for (i in 0:(n - 1)) species <- c(species, DataFrame_string_at(iris, "Species", i))
species_tensor <- Tensor_from_string_values(c(n), species)
setosa_mask <- Tensor_equal(species_tensor, Tensor_from_string_values(c(1), c("setosa")))
cat("setosa count =", Tensor_sum(setosa_mask), "(of", n, "rows)\n")
