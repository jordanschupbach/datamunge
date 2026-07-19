require "octruby"

n = 100
v = Datamunge::DVector.new(n)

n.times { |i| v[i] = i * 1.5 }
n.times { |i| puts v[i] }

v2 = Datamunge::IVector.new(n)
n.times { |i| v2[i] = (i * 1.5).to_i }
n.times { |i| puts v2[i] }

p = Datamunge::IPair.new(3, 4)
puts "p: (#{p.first}, #{p.second})"

p2 = Datamunge::DPair.new(10, 20)
puts "p2: (#{p2.first}, #{p2.second})"
