require "octruby"

Octra.hello

dp = Octra::DPair.new(1.0, 2.0)
dp.first
dp.second
puts "DPair created: #{dp.inspect}"
dp.delete if dp.respond_to?(:delete)

class TimesTwo < Octra::Callback
  def call(x)
    x * 2.0
  end
end

cb = TimesTwo.new
puts "call_with_callback(3.0) = #{Octra.call_with_callback(3.0, cb)}"
v = Octra.make_dvector(1.0, 2.0, 3.0)
v2 = Octra.map_dvector_with_callback(v, cb)
puts "sum_dvector(map_dvector_with_callback(1,2,3)) = #{Octra.sum_dvector(v2)}"
