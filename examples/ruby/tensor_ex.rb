# CMD: just TARGET=tensor_ex run-ruby
# Tensors: construction, shape ops, broadcasting, reductions, linear algebra, masks, iris.
require "datamunge"

Tensor = Datamunge::Tensor

def shape_string(tensor) = "[#{tensor.shape.join(", ")}]"

puts "=================== Construction ==================="
puts "zeros([2,3]): #{Tensor.zeros([2, 3])}"
puts "eye(3): #{Tensor.eye(3)}"

r = Tensor.arange(0.0, 12.0, 1.0).reshape([3, 4])
puts "arange(0,12).reshape([3,4]): #{r}"

puts "\n=================== Shape ops ==================="
puts "transpose -> shape #{shape_string(r.transpose)}"
puts "slice(axis=1, start=1, stop=3): #{r.slice(1, 1, 3)}"

puts "\n=================== Broadcasting arithmetic ==================="
col = Tensor.from_values([3, 1], [1.0, 2.0, 3.0])
row = Tensor.from_values([1, 4], [10.0, 20.0, 30.0, 40.0])
puts "(3,1) + (1,4) -> #{col.add(row)}"

puts "\n=================== Reductions ==================="
puts "r.sum() = #{r.sum}, r.mean() = #{r.mean}"
puts "column means (axis=0): #{r.mean_axis(0)}"

puts "\n=================== Linear algebra ==================="
a = Tensor.from_values([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
b = Tensor.from_values([3, 2], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
puts "matmul(2x3, 3x2) -> #{a.matmul(b)}"
v1 = Tensor.from_values([3], [1.0, 2.0, 3.0])
v2 = Tensor.from_values([3], [4.0, 5.0, 6.0])
puts "dot([1,2,3], [4,5,6]) = #{v1.dot(v2)}"
puts "outer(v1, v2) -> #{v1.outer(v2)}"

puts "\n=================== Comparisons & masks ==================="
mask = r.greater_equal(Tensor.full([3, 4], 6.0))
puts "r >= 6 -> #{mask}"
puts "count(r >= 6) = #{mask.sum}"

puts "\n=================== A real dataset as a Tensor ==================="
iris = Datamunge::Data.iris
n = iris.nrows
columns = ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"]
flat = (0...n).flat_map { |i| columns.map { |column| iris.numeric_at(column, i) } }
x = Tensor.from_values([n, 4], flat)
puts "iris feature tensor shape: #{shape_string(x)}"

feature_means = x.mean_axis(0)
centered = x.subtract(feature_means.reshape([1, 4]))
scatter = centered.transpose.matmul(centered)
puts "feature means (Sepal.Length, Sepal.Width, Petal.Length, Petal.Width): #{feature_means}"
puts "(X-mean)^T (X-mean) [4x4 scatter matrix]: #{scatter.to_string(16)}"

species = Array.new(n) { |i| iris.string_at("Species", i) }
species_tensor = Tensor.from_string_values([n], species)
setosa_mask = species_tensor.equal(Tensor.from_string_values([1], ["setosa"]))
puts "setosa count = #{setosa_mask.sum} (of #{n} rows)"
