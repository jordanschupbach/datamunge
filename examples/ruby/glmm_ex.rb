require "octruby"

puts "=================== Binomial (logistic) mixed model on a real dataset (penguins) ===================\n"
# Predicting sex from body mass, grouped by island: unlike species (which is almost perfectly
# confounded with island in this dataset -- Gentoo penguins occur on Biscoe island only), sex
# is not tied to location, so this is a well-behaved fit.
penguins = Datamunge::DataFrame.penguins
is_male = []
body_mass = []
island = []
penguins.nrows.times do |i|
  next if penguins.is_null("sex", i) || penguins.is_null("body_mass_g", i) || penguins.is_null("island", i)
  is_male << (penguins.string_at("sex", i) == "male" ? 1.0 : 0.0)
  body_mass << penguins.numeric_at("body_mass_g", i)
  island << penguins.string_at("island", i)
end

sex_df = Datamunge::DataFrame.new
sex_df.add_numeric_column("is_male", is_male)
sex_df.add_numeric_column("body_mass_g", body_mass)
sex_df.add_string_column("island", island)

sex_model = Datamunge::GLMM.new(sex_df, "is_male ~ body_mass_g + (1 | island)", "binomial")
sex_model.print_summary

puts "\n=================== Poisson mixed model on simulated multi-site count data ===================\n"
# 25 stores, ~20 days each: daily visit counts depend on a promo intensity score, but both
# the baseline traffic and the promo's effectiveness vary by store.
srand(4242)
n_stores = 25
store_effect = Array.new(n_stores) { Math.sqrt(-2.0 * Math.log(rand)) * Math.cos(2.0 * Math::PI * rand) * 0.4 }

true_intercept = 2.0
true_slope = 0.3
store = []
promo = []
visits = []
(0...n_stores).each do |s|
  n_days = rand(15..25)
  n_days.times do
    promo_intensity = rand * 3.0
    lam = Math.exp(true_intercept + store_effect[s] + true_slope * promo_intensity)
    # Knuth's Poisson sampler.
    l_thresh = Math.exp(-lam)
    k = 0
    p = 1.0
    loop do
      k += 1
      p *= rand
      break if p <= l_thresh
    end
    store << s.to_f
    promo << promo_intensity
    visits << (k - 1).to_f
  end
end

df = Datamunge::DataFrame.new
df.add_numeric_column("store", store)
df.add_numeric_column("promo", promo)
df.add_numeric_column("visits", visits)

store_model = Datamunge::GLMM.new(df, "visits ~ promo + (1 | store)", "poisson")
store_model.print_summary

puts "\nTrue generating values: intercept=#{true_intercept}, slope=#{true_slope}, " \
     "random-intercept SD (log scale)=0.4"

puts "\n--- BLUPs for a few stores ---"
group_labels = store_model.group_labels
[0, 1, 2].each do |idx|
  puts "store #{group_labels[idx]}: intercept shift=#{store_model.random_effects_for_group(idx)[0]}"
end

puts "\n--- Prediction: population-level vs. store-adjusted ---"
newdata_population = Datamunge::DataFrame.new
newdata_population.add_numeric_column("promo", [1.5])
newdata_store0 = Datamunge::DataFrame.new
newdata_store0.add_numeric_column("promo", [1.5])
newdata_store0.add_numeric_column("store", [0.0])
puts "promo=1.5, unseen store:   #{store_model.predict(newdata_population)[0]} expected visits (fixed effects only)"
puts "promo=1.5, store 0 (known): #{store_model.predict(newdata_store0)[0]} expected visits " \
     "(fixed effects + store 0's BLUP)"
