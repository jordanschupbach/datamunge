require "octruby"

hp = [110.0, 110.0, 93.0, 110.0, 175.0, 105.0, 245.0, 62.0, 95.0, 123.0]
wt = [2.62, 2.875, 2.32, 3.215, 3.44, 3.46, 3.57, 3.19, 3.15, 3.44]
transmission = %w[manual manual manual automatic automatic
                  automatic automatic automatic automatic automatic]
mpg = [21.0, 21.0, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2]

cars = Datamunge::DataFrame.new
cars.add_numeric_column("hp", hp)
cars.add_numeric_column("wt", wt)
cars.add_string_column("transmission", transmission)
cars.add_numeric_column("mpg", mpg)

puts "Fitting: mpg ~ hp + wt + transmission\n\n"
model = Datamunge::LM.new(cars, "mpg ~ hp + wt + transmission")
model.print_summary

puts "\nSequential ANOVA:"
puts model.anova.to_string

newcars = Datamunge::DataFrame.new
newcars.add_numeric_column("hp", [150.0, 90.0])
newcars.add_numeric_column("wt", [3.0, 2.5])
newcars.add_string_column("transmission", %w[manual automatic])

frame = model.predict_frame(newcars, "confidence")
puts "\nPredictions with 95% confidence intervals:"
puts frame.to_string

model.save_diagnostic_plots("lm_ex_diagnostics")
puts "\nSaved diagnostic plots as lm_ex_diagnostics_*.svg"
