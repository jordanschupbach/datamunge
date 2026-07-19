require "octruby"

puts "=================== Random intercept on a real dataset (penguins) ==================="
penguins = Datamunge::DataFrame.penguins
species_model = Datamunge::LMM.new(penguins, "body_mass_g ~ flipper_length_mm + bill_length_mm + (1 | species)")
species_model.print_summary

puts "\n=================== Random intercept + slope on a simulated multi-school dataset ===================\n"
# 30 schools, ~25 students each: score depends on study_hours, but both the baseline score
# and the return on an extra hour of study vary by school -- the textbook case for a
# random-intercept-and-slope model.
srand(2024)
n_schools = 30
school_intercept = Array.new(n_schools) { Math.sqrt(-2.0 * Math.log(rand)) * Math.cos(2.0 * Math::PI * rand) * 6.0 }
school_slope = Array.new(n_schools) { Math.sqrt(-2.0 * Math.log(rand)) * Math.cos(2.0 * Math::PI * rand) * 1.2 }

true_intercept = 60.0
true_slope = 3.0
school = []
study_hours = []
score = []
(0...n_schools).each do |s|
  n_students = rand(15..35)
  n_students.times do
    hours = rand * 10.0
    noise = Math.sqrt(-2.0 * Math.log(rand)) * Math.cos(2.0 * Math::PI * rand) * 4.0
    s_val = true_intercept + school_intercept[s] + (true_slope + school_slope[s]) * hours + noise
    school << s.to_f
    study_hours << hours
    score << s_val
  end
end

df = Datamunge::DataFrame.new
df.add_numeric_column("school", school)
df.add_numeric_column("study_hours", study_hours)
df.add_numeric_column("score", score)

model = Datamunge::LMM.new(df, "score ~ study_hours + (1 + study_hours | school)")
model.print_summary

puts "\nTrue generating values: intercept=#{true_intercept}, slope=#{true_slope}, " \
     "random-intercept SD=6.0, random-slope SD=1.2, residual SD=4.0"

puts "\n--- Best Linear Unbiased Predictors (BLUPs) for a few schools ---"
group_labels = model.group_labels
[0, 1, 2].each do |idx|
  re = model.random_effects_for_group(idx)
  puts "school #{group_labels[idx]}: intercept shift=#{re[0]}, slope shift=#{re[1]}"
end

puts "\n--- Prediction: population-level vs. school-adjusted ---"
newdata_population = Datamunge::DataFrame.new
newdata_population.add_numeric_column("study_hours", [5.0])
newdata_school0 = Datamunge::DataFrame.new
newdata_school0.add_numeric_column("study_hours", [5.0])
newdata_school0.add_numeric_column("school", [0.0])
puts "5 study hours, unseen school:      #{model.predict(newdata_population)[0]} (fixed effects only)"
puts "5 study hours, school 0 (known):    #{model.predict(newdata_school0)[0]} (fixed effects + school 0's BLUP)"
