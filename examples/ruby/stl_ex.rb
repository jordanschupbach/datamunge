# CMD: just TARGET=stl_ex run-ruby
# C++ std::vector and std::pair values in Ruby: vectors are Arrays, pairs are two-element Arrays.
require "datamunge"

n = 100
v = Array.new(n) { |i| i * 1.5 }
v.each { |x| puts x }
v2 = Array.new(n) { |i| (i * 1.5).floor }
v2.each { |x| puts x }

p = [3, 4]
puts "p: (#{p[0]}, #{p[1]})"

p2 = [10, 20]
puts "p2: (#{p2[0]}, #{p2[1]})"

# A C++ std::pair<double,double> comes back as a two-element Array.
p3 = Datamunge.make_dpair(1.25, 2.75)
puts "make_dpair(1.25, 2.75): (#{p3.join(", ")}), sum_dpair = #{Datamunge.sum_dpair(p3)}"
