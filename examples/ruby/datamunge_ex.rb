# CMD: just TARGET=datamunge_ex run-ruby
# Smoke test of the basic bindings: vectors, pairs and a native Callback object.
require "datamunge"

Datamunge.hello

v = Datamunge.make_dvector(1.0, 2.0, 3.0)
puts "sum_dvector:\t#{Datamunge.sum_dvector(v)}"

p = Datamunge.make_dpair(1.25, 2.75)
puts "sum_dpair:\t#{Datamunge.sum_dpair(p)}"

cb = Datamunge::Callback.new
puts "call_with_callback(3.0):\t#{Datamunge.call_with_callback(3.0, cb)}"
v2 = Datamunge.map_dvector_with_callback(Datamunge.make_dvector(1.0, 2.0, 3.0), cb)
puts "sum_dvector(map_dvector_with_callback(1,2,3)):\t#{Datamunge.sum_dvector(v2)}"
cb.close
