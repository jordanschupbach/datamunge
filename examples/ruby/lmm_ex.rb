# CMD: just TARGET=lmm_ex run-ruby
# Linear mixed models: random intercept (penguins) and random intercept + slope (simulated schools).
require "datamunge"

rng = Random.new(2024)
# Box-Muller on (0, 1] uniforms (avoids log(0)).
gauss = -> { Math.sqrt(-2 * Math.log(1 - rng.rand)) * Math.cos(2 * Math::PI * (1 - rng.rand)) }

puts "=================== Random intercept on a real dataset (penguins) ==================="
Datamunge::LMM.new(Datamunge::Data.penguins, "body_mass_g ~ flipper_length_mm + bill_length_mm + (1 | species)").print_summary

puts "\n=================== Random intercept + slope on a simulated multi-school dataset ==================="
# 30 schools, ~25 students each: score depends on study_hours, but both the baseline score
# and the return on an extra hour of study vary by school -- the textbook case for a
# random-intercept-and-slope model.
n_schools = 30
school_intercept = Array.new(n_schools) { gauss.() * 6 }
school_slope = Array.new(n_schools) { gauss.() * 1.2 }
true_intercept = 60.0
true_slope = 3.0

school = []
study_hours = []
score = []
n_schools.times do |s|
  (15 + rng.rand(0...21)).times do
    hours = rng.rand * 10
    school << s.to_f
    study_hours << hours
    score << true_intercept + school_intercept[s] + (true_slope + school_slope[s]) * hours + gauss.() * 4
  end
end
df = Datamunge::DataFrame.new
df.add_numeric_column("school", school)
df.add_numeric_column("study_hours", study_hours)
df.add_numeric_column("score", score)

model = Datamunge::LMM.new(df, "score ~ study_hours + (1 + study_hours | school)")
model.print_summary

puts "\nTrue generating values: intercept=#{true_intercept}, slope=#{true_slope}, random-intercept SD=6.0, random-slope SD=1.2, residual SD=4.0"

puts "\n--- Best Linear Unbiased Predictors (BLUPs) for a few schools ---"
labels = model.group_labels
3.times do |idx|
  intercept_shift, slope_shift = model.random_effects_for_group(idx)
  puts "school #{labels[idx]}: intercept shift=#{intercept_shift}, slope shift=#{slope_shift}"
end

puts "\n--- Prediction: population-level vs. school-adjusted ---"
population = Datamunge::DataFrame.new
school0 = Datamunge::DataFrame.new
population.add_numeric_column("study_hours", [5.0])
school0.add_numeric_column("study_hours", [5.0])
school0.add_numeric_column("school", [0.0])
puts "5 study hours, unseen school:      #{model.predict(population)[0]} (fixed effects only)"
puts "5 study hours, school 0 (known):    #{model.predict(school0)[0]} (fixed effects + school 0's BLUP)"
