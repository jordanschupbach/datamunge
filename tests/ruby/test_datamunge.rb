require "minitest/autorun"

require "octruby"

class TestDatamunge < Minitest::Test
  def test_loads_and_calls_hello
    assert Datamunge.respond_to?(:hello)
    Datamunge.hello
  end

  def test_can_allocate_and_free_pair
    pair = Datamunge::DPair.new(1.0, 2.0)
    assert pair
    pair.delete if pair.respond_to?(:delete)
  end

  class TimesTwo < Datamunge::Callback
    def call(x)
      x * 2.0
    end
  end

  def test_can_pass_callback_into_cpp
    cb = TimesTwo.new
    assert_in_delta 6.0, Datamunge.call_with_callback(3.0, cb), 1e-9

    v = Datamunge.make_dvector(1.0, 2.0, 3.0)
    out = Datamunge.map_dvector_with_callback(v, cb)
    assert_in_delta 12.0, Datamunge.sum_dvector(out), 1e-9
  end
end
