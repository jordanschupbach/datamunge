# CMD: just TARGET=tensor_ex run-julia

# Tensors: construction, shape ops, broadcasting, reductions, linear algebra, masks, iris.
using Datamunge

shape_string(tensor) = "[$(join(shape(tensor), ", "))]"

println("=================== Construction ===================")
println("zeros([2,3]): $(to_string(zeros(Tensor, [2, 3])))")
println("eye(3): $(to_string(eye(Tensor, 3)))")

const r = reshape(arange(Tensor, 0.0, 12.0, 1.0), [3, 4])
println("arange(0,12).reshape([3,4]): $(to_string(r))")

println("\n=================== Shape ops ===================")
println("transpose -> shape $(shape_string(transpose(r)))")
println("slice(axis=1, start=1, stop=3): $(to_string(slice(r, 1, 1, 3)))")

println("\n=================== Broadcasting arithmetic ===================")
let col = from_values(Tensor, [3, 1], [1.0, 2.0, 3.0]), row = from_values(Tensor, [1, 4], [10.0, 20.0, 30.0, 40.0])
    println("(3,1) + (1,4) -> $(to_string(add(col, row)))")
end

println("\n=================== Reductions ===================")
println("r.sum() = $(sum(r)), r.mean() = $(mean(r))")
println("column means (axis=0): $(to_string(mean_axis(r, 0)))")

println("\n=================== Linear algebra ===================")
let a = from_values(Tensor, [2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    b = from_values(Tensor, [3, 2], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    println("matmul(2x3, 3x2) -> $(to_string(matmul(a, b)))")
end
let v1 = from_values(Tensor, [3], [1.0, 2.0, 3.0]), v2 = from_values(Tensor, [3], [4.0, 5.0, 6.0])
    println("dot([1,2,3], [4,5,6]) = $(dot(v1, v2))")
    println("outer(v1, v2) -> $(to_string(outer(v1, v2)))")
end

println("\n=================== Comparisons & masks ===================")
let mask = greater_equal(r, full(Tensor, [3, 4], 6.0))
    println("r >= 6 -> $(to_string(mask))")
    println("count(r >= 6) = $(sum(mask))")
end

println("\n=================== A real dataset as a Tensor ===================")
let iris = Datamunge.Data.iris(), n = nrows(iris)
    columns = ("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width")
    flat = [numeric_at(iris, column, i) for i in 0:n-1 for column in columns]
    x = from_values(Tensor, [n, 4], flat)
    println("iris feature tensor shape: $(shape_string(x))")

    feature_means = mean_axis(x, 0)
    centered = subtract(x, reshape(feature_means, [1, 4]))
    scatter = matmul(transpose(centered), centered)
    println("feature means (Sepal.Length, Sepal.Width, Petal.Length, Petal.Width): $(to_string(feature_means))")
    println("(X-mean)^T (X-mean) [4x4 scatter matrix]: $(to_string(scatter, 16))")

    species = [string_at(iris, "Species", i) for i in 0:n-1]
    species_tensor = from_string_values(Tensor, [n], species)
    setosa_mask = equal(species_tensor, from_string_values(Tensor, [1], ["setosa"]))
    println("setosa count = $(sum(setosa_mask)) (of $n rows)")
end
