from pydatamunge import datamunge as dm


def sv(*dims):
    """Shape/index-vector arguments (std::vector<size_t>) need an explicit SizeVector
    wrapper -- unlike double/string vectors, plain Python lists aren't accepted directly."""
    return dm.SizeVector(list(dims))


print("=================== Construction ===================")
z = dm.Tensor.zeros(sv(2, 3))
print("zeros([2,3]):", z.to_string())

eye = dm.Tensor.eye(3)
print("eye(3):", eye.to_string())

r = dm.Tensor.arange(0.0, 12.0, 1.0).reshape(sv(3, 4))
print("arange(0,12).reshape([3,4]):", r.to_string())

print("\n=================== Shape ops ===================")
rt = r.transpose()
print("transpose -> shape", rt.shape())
sliced = r.slice(1, 1, 3)
print("slice(axis=1, start=1, stop=3):", sliced.to_string())

print("\n=================== Broadcasting arithmetic ===================")
col = dm.Tensor.from_values(sv(3, 1), [1, 2, 3])
row = dm.Tensor.from_values(sv(1, 4), [10, 20, 30, 40])
broadcast_sum = col.add(row)
print("(3,1) + (1,4) ->", broadcast_sum.to_string())

print("\n=================== Reductions ===================")
print("r.sum() =", r.sum(), ", r.mean() =", r.mean())
col_means = r.mean_axis(0)
print("column means (axis=0):", col_means.to_string())

print("\n=================== Linear algebra ===================")
a = dm.Tensor.from_values(sv(2, 3), [1, 2, 3, 4, 5, 6])
b = dm.Tensor.from_values(sv(3, 2), [7, 8, 9, 10, 11, 12])
print("matmul(2x3, 3x2) ->", a.matmul(b).to_string())

v1 = dm.Tensor.from_values(sv(3), [1, 2, 3])
v2 = dm.Tensor.from_values(sv(3), [4, 5, 6])
print("dot([1,2,3], [4,5,6]) =", v1.dot(v2))
print("outer(v1, v2) ->", v1.outer(v2).to_string())

print("\n=================== Comparisons & masks ===================")
mask = r.greater_equal(dm.Tensor.full(sv(3, 4), 6.0))
print("r >= 6 ->", mask.to_string())
print("count(r >= 6) =", mask.sum())

print("\n=================== A real dataset as a Tensor ===================")
iris = dm.DataFrame.iris()
flat = []
for i in range(iris.nrows()):
    flat.append(iris.numeric_at("Sepal.Length", i))
    flat.append(iris.numeric_at("Sepal.Width", i))
    flat.append(iris.numeric_at("Petal.Length", i))
    flat.append(iris.numeric_at("Petal.Width", i))
X = dm.Tensor.from_values(sv(iris.nrows(), 4), flat)
print("iris feature tensor shape:", X.shape())

feature_means = X.mean_axis(0)
centered = X.subtract(feature_means.reshape(sv(1, 4)))
scatter = centered.transpose().matmul(centered)
print("feature means (Sepal.Length, Sepal.Width, Petal.Length, Petal.Width):", feature_means.to_string())
print("(X-mean)^T (X-mean) [4x4 scatter matrix]:", scatter.to_string(16))

species = [iris.string_at("Species", i) for i in range(iris.nrows())]
species_tensor = dm.Tensor.from_string_values(sv(iris.nrows()), species)
setosa_mask = species_tensor.equal(dm.Tensor.from_string_values(sv(1), ["setosa"]))
print("setosa count =", setosa_mask.sum(), "(of", iris.nrows(), "rows)")
