require "octruby"

# Unlike Python, plain Ruby arrays are accepted directly for std::vector<size_t> (shape/index)
# parameters -- no SizeVector wrapper needed.

puts "=================== Construction ==================="
z = Datamunge::Tensor.zeros([2, 3])
puts "zeros([2,3]): #{z.to_string}"

eye = Datamunge::Tensor.eye(3)
puts "eye(3): #{eye.to_string}"

r = Datamunge::Tensor.arange(0.0, 12.0, 1.0).reshape([3, 4])
puts "arange(0,12).reshape([3,4]): #{r.to_string}"

puts "\n=================== Shape ops ==================="
rt = r.transpose
puts "transpose -> shape #{rt.shape}"
sliced = r.slice(1, 1, 3)
puts "slice(axis=1, start=1, stop=3): #{sliced.to_string}"

puts "\n=================== Broadcasting arithmetic ==================="
col = Datamunge::Tensor.from_values([3, 1], [1, 2, 3])
row = Datamunge::Tensor.from_values([1, 4], [10, 20, 30, 40])
broadcast_sum = col.add(row)
puts "(3,1) + (1,4) -> #{broadcast_sum.to_string}"

puts "\n=================== Reductions ==================="
puts "r.sum() = #{r.sum}, r.mean() = #{r.mean}"
col_means = r.mean_axis(0)
puts "column means (axis=0): #{col_means.to_string}"

puts "\n=================== Linear algebra ==================="
a = Datamunge::Tensor.from_values([2, 3], [1, 2, 3, 4, 5, 6])
b = Datamunge::Tensor.from_values([3, 2], [7, 8, 9, 10, 11, 12])
puts "matmul(2x3, 3x2) -> #{a.matmul(b).to_string}"

v1 = Datamunge::Tensor.from_values([3], [1, 2, 3])
v2 = Datamunge::Tensor.from_values([3], [4, 5, 6])
puts "dot([1,2,3], [4,5,6]) = #{v1.dot(v2)}"
puts "outer(v1, v2) -> #{v1.outer(v2).to_string}"

puts "\n=================== Comparisons & masks ==================="
mask = r.greater_equal(Datamunge::Tensor.full([3, 4], 6.0))
puts "r >= 6 -> #{mask.to_string}"
puts "count(r >= 6) = #{mask.sum}"

puts "\n=================== A real dataset as a Tensor ==================="
iris = Datamunge::DataFrame.iris
flat = []
iris.nrows.times do |i|
  flat << iris.numeric_at("Sepal.Length", i)
  flat << iris.numeric_at("Sepal.Width", i)
  flat << iris.numeric_at("Petal.Length", i)
  flat << iris.numeric_at("Petal.Width", i)
end
x = Datamunge::Tensor.from_values([iris.nrows, 4], flat)
puts "iris feature tensor shape: #{x.shape}"

feature_means = x.mean_axis(0)
centered = x.subtract(feature_means.reshape([1, 4]))
scatter = centered.transpose.matmul(centered)
puts "feature means (Sepal.Length, Sepal.Width, Petal.Length, Petal.Width): #{feature_means.to_string}"
puts "(X-mean)^T (X-mean) [4x4 scatter matrix]: #{scatter.to_string(16)}"

species = (0...iris.nrows).map { |i| iris.string_at("Species", i) }
species_tensor = Datamunge::Tensor.from_string_values([iris.nrows], species)
setosa_mask = species_tensor.equal(Datamunge::Tensor.from_string_values([1], ["setosa"]))
puts "setosa count = #{setosa_mask.sum} (of #{iris.nrows} rows)"
