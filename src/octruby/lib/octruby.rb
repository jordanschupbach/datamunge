dir = File.join(__dir__, "octruby")
Dir[File.join(dir, "*.rb")].sort.each { |p| require p }
require File.join(dir, "octruby")

# Keep the public namespace consistent with other bindings.
Octra = Octruby unless defined?(Octra)
