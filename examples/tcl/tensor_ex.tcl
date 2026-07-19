package require Datamunge 0.0.1

# Tcl's specialize_std_vector scalar fast-path covers double/string/int both ways, so plain
# Tcl lists auto-convert directly to DVector/SVector/IVector parameters -- but NOT to
# SizeVector (std::size_t doesn't textually match the fast-path's "unsigned long" spelling),
# so shape vectors still need explicit construction via new_SizeVector + push.
proc szv {values} {
  set v [datamunge::new_SizeVector]
  foreach x $values { datamunge::SizeVector_push $v $x }
  return $v
}

proc shape_str {shp} {
  return [join $shp ", "]
}

puts "=================== Construction ==================="
set z [datamunge::Tensor_zeros [szv {2 3}]]
puts "zeros(\[2,3\]): [datamunge::Tensor_to_string $z]"

set eyet [datamunge::Tensor_eye 3]
puts "eye(3): [datamunge::Tensor_to_string $eyet]"

set r [datamunge::Tensor_reshape [datamunge::Tensor_arange 0.0 12.0 1.0] [szv {3 4}]]
puts "arange(0,12).reshape(\[3,4\]): [datamunge::Tensor_to_string $r]"

puts "\n=================== Shape ops ==================="
set rt [datamunge::Tensor_transpose $r]
puts "transpose -> shape \[[shape_str [datamunge::Tensor_shape $rt]]\]"
set sliced [datamunge::Tensor_slice $r 1 1 3]
puts "slice(axis=1, start=1, stop=3): [datamunge::Tensor_to_string $sliced]"

puts "\n=================== Broadcasting arithmetic ==================="
set col [datamunge::Tensor_from_values [szv {3 1}] {1.0 2.0 3.0}]
set row [datamunge::Tensor_from_values [szv {1 4}] {10.0 20.0 30.0 40.0}]
set broadcast_sum [datamunge::Tensor_add $col $row]
puts "(3,1) + (1,4) -> [datamunge::Tensor_to_string $broadcast_sum]"

puts "\n=================== Reductions ==================="
puts "r.sum() = [datamunge::Tensor_sum $r], r.mean() = [datamunge::Tensor_mean $r]"
set col_means [datamunge::Tensor_mean_axis $r 0]
puts "column means (axis=0): [datamunge::Tensor_to_string $col_means]"

puts "\n=================== Linear algebra ==================="
set a [datamunge::Tensor_from_values [szv {2 3}] {1.0 2.0 3.0 4.0 5.0 6.0}]
set b [datamunge::Tensor_from_values [szv {3 2}] {7.0 8.0 9.0 10.0 11.0 12.0}]
puts "matmul(2x3, 3x2) -> [datamunge::Tensor_to_string [datamunge::Tensor_matmul $a $b]]"

set v1 [datamunge::Tensor_from_values [szv {3}] {1.0 2.0 3.0}]
set v2 [datamunge::Tensor_from_values [szv {3}] {4.0 5.0 6.0}]
puts "dot(\[1,2,3\], \[4,5,6\]) = [datamunge::Tensor_dot $v1 $v2]"
puts "outer(v1, v2) -> [datamunge::Tensor_to_string [datamunge::Tensor_outer $v1 $v2]]"

puts "\n=================== Comparisons & masks ==================="
set mask [datamunge::Tensor_greater_equal $r [datamunge::Tensor_full [szv {3 4}] 6.0]]
puts "r >= 6 -> [datamunge::Tensor_to_string $mask]"
puts "count(r >= 6) = [datamunge::Tensor_sum $mask]"

puts "\n=================== A real dataset as a Tensor ==================="
set iris [datamunge::DataFrame_iris]
set n [datamunge::DataFrame_nrows $iris]
set flat {}
for {set i 0} {$i < $n} {incr i} {
  lappend flat [datamunge::DataFrame_numeric_at $iris "Sepal.Length" $i]
  lappend flat [datamunge::DataFrame_numeric_at $iris "Sepal.Width" $i]
  lappend flat [datamunge::DataFrame_numeric_at $iris "Petal.Length" $i]
  lappend flat [datamunge::DataFrame_numeric_at $iris "Petal.Width" $i]
}
set x [datamunge::Tensor_from_values [szv [list $n 4]] $flat]
puts "iris feature tensor shape: \[[shape_str [datamunge::Tensor_shape $x]]\]"

set feature_means [datamunge::Tensor_mean_axis $x 0]
set centered [datamunge::Tensor_subtract $x [datamunge::Tensor_reshape $feature_means [szv {1 4}]]]
set scatter [datamunge::Tensor_matmul [datamunge::Tensor_transpose $centered] $centered]
puts "feature means (Sepal.Length, Sepal.Width, Petal.Length, Petal.Width): [datamunge::Tensor_to_string $feature_means]"
puts "(X-mean)^T (X-mean) \[4x4 scatter matrix\]: [datamunge::Tensor_to_string $scatter 16]"

set species {}
for {set i 0} {$i < $n} {incr i} {
  lappend species [datamunge::DataFrame_string_at $iris "Species" $i]
}
set species_tensor [datamunge::Tensor_from_string_values [szv [list $n]] $species]
set setosa_mask [datamunge::Tensor_equal $species_tensor [datamunge::Tensor_from_string_values [szv {1}] {setosa}]]
puts "setosa count = [datamunge::Tensor_sum $setosa_mask] (of $n rows)"
