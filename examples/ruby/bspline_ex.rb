# CMD: just TARGET=bspline_ex run-ruby
# Tensor-product B-spline regression: fit z ~ bs(x, y) to a smooth surface and predict.
require "datamunge"

# Sample a smooth surface z = sin(x) + y^2 on an 8x8 grid over [0, 1]^2.
x, y, z = [], [], []
8.times do |iy|
  8.times do |ix|
    xv, yv = ix / 7.0, iy / 7.0
    x << xv
    y << yv
    z << Math.sin(xv) + yv * yv
  end
end
data = Datamunge::DataFrame.new
data.add_numeric_column("x", x)
data.add_numeric_column("y", y)
data.add_numeric_column("z", z)

surface = Datamunge::LM.new(data, "z ~ bs(x, y)")
puts "Fitted z ~ bs(x, y)"
surface.print_summary

new_points = Datamunge::DataFrame.new
new_points.add_numeric_column("x", [0.25, 0.75])
new_points.add_numeric_column("y", [0.50, 0.25])
pred = surface.predict(new_points)
printf("Predictions: [%.6f, %.6f]\n", pred[0], pred[1])
