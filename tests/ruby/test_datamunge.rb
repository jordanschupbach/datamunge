# Smoke tests of the Datamunge Ruby binding (just test-ruby). Argument: the generated Ruby file
# (default src/datamungeruby/lib/datamunge.rb), for the YARD documentation checks.
require "datamunge"

$checks = 0
def check(ok, message)
  $checks += 1
  raise "check failed: #{message}" unless ok
end

def caught
  yield
  nil
rescue Exception => e # rubocop:disable Lint/RescueException
  e
end

# A Ruby subclass overriding the C++ Callback's virtual call() (a director).
class Doubler < Datamunge::Callback
  def call(x) = 2 * x
end

class FromRuby < StandardError; end

# An override that raises.
class Failing < Datamunge::Callback
  def call(_x) = raise(FromRuby, "from Ruby")
end

# Smoke test: call into the native library.
Datamunge.hello

# std::vector<double> and std::pair<double, double> round trips.
v = Datamunge.make_dvector(1.0, 2.0, 3.0)
check(v == [1.0, 2.0, 3.0], "std::vector<double> is an Array of Float")
check(Datamunge.sum_dvector(v) == 6.0, "sum_dvector")
check(Datamunge.sum_dvector([0.5, 1.5]) == 2.0, "Arrays convert to std::vector")
p = Datamunge.make_dpair(1.25, 2.75)
check(p == [1.25, 2.75], "std::pair is a two-element Array")
check(Datamunge.sum_dpair(p) == 4.0, "sum_dpair")

# The C++ Callback class, and a Ruby subclass overriding its virtual call().
cb = Datamunge::Callback.new
check(Datamunge.call_with_callback(3.0, cb) == 3.0, "native Callback")
check(Datamunge.sum_dvector(Datamunge.map_dvector_with_callback(Datamunge.make_dvector(1.0, 2.0, 3.0), cb)) == 6.0,
      "native map")
cb.close
doubler = Doubler.new
check(Datamunge.call_with_callback(3.0, doubler) == 6.0, "the Ruby override runs")
check(Datamunge.sum_dvector(Datamunge.map_dvector_with_callback([1.0, 2.0, 3.0], doubler)) == 12.0,
      "override from C++ loops")

# A Ruby exception inside an override reaches the caller unchanged.
check(caught { Datamunge.call_with_callback(1.0, Failing.new) }.is_a?(FromRuby), "the Ruby exception propagates")

# C++ exceptions raise classes that include Datamunge::Error.
error = caught { Datamunge.runif(1.0, 0.0, 3) }
check(error.is_a?(Datamunge::Error) && error.is_a?(StandardError), "C++ exceptions raise Datamunge::Error")

# Classes and methods.
frame = Datamunge::DataFrame.new
frame.add_numeric_column("x", [1.0, 2.0, 3.0])
check(frame.nrows == 3, "DataFrame")
check(frame.columns == ["x"], "std::vector<std::string> is an Array of String")
frame.close
check(frame.closed?, "close destroys the C++ object")

# YARD comments carry the C++ documentation.
path = ARGV[0] || File.expand_path("../../src/datamungeruby/lib/datamunge.rb", __dir__)
source = File.read(path, encoding: "UTF-8")
doc = source[/((?:^ *#.*\n)+) *def self\.seql\(/, 1].to_s
check(doc.include?("evenly spaced"), "YARD documentation")
check(doc.include?("@param n [Integer]"), "parameter documentation")

puts "test_datamunge.rb: all #{$checks} checks passed"
