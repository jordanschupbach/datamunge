local dm = require("datamunge")

local function sv(t)
  local v = dm.SizeVector(#t)
  for i, x in ipairs(t) do v[i - 1] = x end
  return v
end

local function dv(t)
  local v = dm.DVector(#t)
  for i, x in ipairs(t) do v[i - 1] = x end
  return v
end

local function to_table(vec)
  local out = {}
  for i = 0, vec:size() - 1 do out[i + 1] = vec[i] end
  return out
end

print("=================== Construction ===================")
local z = dm.Tensor.zeros(sv({2, 3}))
print("zeros([2,3]): " .. z:to_string())

local eye = dm.Tensor.eye(3)
print("eye(3): " .. eye:to_string())

local r = dm.Tensor.arange(0.0, 12.0, 1.0):reshape(sv({3, 4}))
print("arange(0,12).reshape([3,4]): " .. r:to_string())

print("\n=================== Shape ops ===================")
local rt = r:transpose()
print("transpose -> shape [" .. table.concat(to_table(rt:shape()), ", ") .. "]")
local sliced = r:slice(1, 1, 3)
print("slice(axis=1, start=1, stop=3): " .. sliced:to_string())

print("\n=================== Broadcasting arithmetic ===================")
local col = dm.Tensor.from_values(sv({3, 1}), dv({1, 2, 3}))
local row = dm.Tensor.from_values(sv({1, 4}), dv({10, 20, 30, 40}))
local broadcast_sum = col:add(row)
print("(3,1) + (1,4) -> " .. broadcast_sum:to_string())

print("\n=================== Reductions ===================")
print("r.sum() = " .. r:sum() .. ", r.mean() = " .. r:mean())
local col_means = r:mean_axis(0)
print("column means (axis=0): " .. col_means:to_string())

print("\n=================== Linear algebra ===================")
local a = dm.Tensor.from_values(sv({2, 3}), dv({1, 2, 3, 4, 5, 6}))
local b = dm.Tensor.from_values(sv({3, 2}), dv({7, 8, 9, 10, 11, 12}))
print("matmul(2x3, 3x2) -> " .. a:matmul(b):to_string())

local v1 = dm.Tensor.from_values(sv({3}), dv({1, 2, 3}))
local v2 = dm.Tensor.from_values(sv({3}), dv({4, 5, 6}))
print("dot([1,2,3], [4,5,6]) = " .. v1:dot(v2))
print("outer(v1, v2) -> " .. v1:outer(v2):to_string())

print("\n=================== Comparisons & masks ===================")
local mask = r:greater_equal(dm.Tensor.full(sv({3, 4}), 6.0))
print("r >= 6 -> " .. mask:to_string())
print("count(r >= 6) = " .. mask:sum())

print("\n=================== A real dataset as a Tensor ===================")
local iris = dm.DataFrame.iris()
local flat = {}
for i = 0, iris:nrows() - 1 do
  table.insert(flat, iris:numeric_at("Sepal.Length", i))
  table.insert(flat, iris:numeric_at("Sepal.Width", i))
  table.insert(flat, iris:numeric_at("Petal.Length", i))
  table.insert(flat, iris:numeric_at("Petal.Width", i))
end
local x = dm.Tensor.from_values(sv({iris:nrows(), 4}), dv(flat))
print("iris feature tensor shape: [" .. table.concat(to_table(x:shape()), ", ") .. "]")

local feature_means = x:mean_axis(0)
local centered = x:subtract(feature_means:reshape(sv({1, 4})))
local scatter = centered:transpose():matmul(centered)
print("feature means (Sepal.Length, Sepal.Width, Petal.Length, Petal.Width): " .. feature_means:to_string())
print("(X-mean)^T (X-mean) [4x4 scatter matrix]: " .. scatter:to_string(16))

local species = dm.SVector(iris:nrows())
for i = 0, iris:nrows() - 1 do species[i] = iris:string_at("Species", i) end
local species_tensor = dm.Tensor.from_string_values(sv({iris:nrows()}), species)
local setosa_only = dm.SVector(1)
setosa_only[0] = "setosa"
local setosa_mask = species_tensor:equal(dm.Tensor.from_string_values(sv({1}), setosa_only))
print("setosa count = " .. setosa_mask:sum() .. " (of " .. iris:nrows() .. " rows)")
