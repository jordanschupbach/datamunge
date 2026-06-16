Gem::Specification.new do |spec|
  spec.name = "octruby"
  spec.version = "0.0.1"
  spec.summary = "Ruby bindings to the octra library (SWIG)."
  spec.license = "Unlicense"
  spec.files = Dir[
    "lib/**/*.rb",
    "ext/**/*.{rb,c,cc,cpp,cxx,h,hpp}",
    "README.md",
    "LICENSE",
  ]
  spec.require_paths = ["lib"]
  spec.extensions = ["ext/octruby/extconf.rb"]
end

