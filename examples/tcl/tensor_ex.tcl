# CMD: just TARGET=tensor_ex run-tcl
# Tensors: construction, shape ops, broadcasting, reductions, linear algebra, masks, iris.
package require datamunge

namespace path ::datamunge

proc shape_string {tensor} {
    return "\[[join [$tensor shape] {, }]\]"
}

puts "=================== Construction ==================="
puts "zeros(\[2,3\]): [[Tensor zeros {2 3}] to_string]"
puts "eye(3): [[Tensor eye 3] to_string]"

set r [[Tensor arange 0.0 12.0 1.0] reshape {3 4}]
puts "arange(0,12).reshape(\[3,4\]): [$r to_string]"

puts "\n=================== Shape ops ==================="
puts "transpose -> shape [shape_string [$r transpose]]"
puts "slice(axis=1, start=1, stop=3): [[$r slice 1 1 3] to_string]"

puts "\n=================== Broadcasting arithmetic ==================="
set col [Tensor from_values {3 1} {1.0 2.0 3.0}]
set row [Tensor from_values {1 4} {10.0 20.0 30.0 40.0}]
puts "(3,1) + (1,4) -> [[$col add $row] to_string]"

puts "\n=================== Reductions ==================="
puts "r.sum() = [$r sum], r.mean() = [$r mean]"
puts "column means (axis=0): [[$r mean_axis 0] to_string]"

puts "\n=================== Linear algebra ==================="
set a [Tensor from_values {2 3} {1.0 2.0 3.0 4.0 5.0 6.0}]
set b [Tensor from_values {3 2} {7.0 8.0 9.0 10.0 11.0 12.0}]
puts "matmul(2x3, 3x2) -> [[$a matmul $b] to_string]"
set v1 [Tensor from_values {3} {1.0 2.0 3.0}]
set v2 [Tensor from_values {3} {4.0 5.0 6.0}]
puts "dot(\[1,2,3\], \[4,5,6\]) = [$v1 dot $v2]"
puts "outer(v1, v2) -> [[$v1 outer $v2] to_string]"

puts "\n=================== Comparisons & masks ==================="
set mask [$r greater_equal [Tensor full {3 4} 6.0]]
puts "r >= 6 -> [$mask to_string]"
puts "count(r >= 6) = [$mask sum]"

puts "\n=================== A real dataset as a Tensor ==================="
set iris [Data::iris]
set n [$iris nrows]
set columns {Sepal.Length Sepal.Width Petal.Length Petal.Width}
set flat {}
for {set i 0} {$i < $n} {incr i} {
    foreach column $columns {
        lappend flat [$iris numeric_at $column $i]
    }
}
set x [Tensor from_values [list $n 4] $flat]
puts "iris feature tensor shape: [shape_string $x]"

set feature_means [$x mean_axis 0]
set centered [$x subtract [$feature_means reshape {1 4}]]
set scatter [[$centered transpose] matmul $centered]
puts "feature means (Sepal.Length, Sepal.Width, Petal.Length, Petal.Width): [$feature_means to_string]"
puts "(X-mean)^T (X-mean) \[4x4 scatter matrix\]: [$scatter to_string 16]"

set species {}
for {set i 0} {$i < $n} {incr i} {
    lappend species [$iris string_at Species $i]
}
set species_tensor [Tensor from_string_values [list $n] $species]
set setosa_mask [$species_tensor equal [Tensor from_string_values {1} {setosa}]]
puts "setosa count = [$setosa_mask sum] (of $n rows)"
