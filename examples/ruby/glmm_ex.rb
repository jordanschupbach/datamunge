# CMD: just TARGET=glmm_ex run-ruby
# Generalized linear mixed models: a binomial GLMM on penguins and a Poisson GLMM on simulated store data.
require "datamunge"

rng = Random.new(4242)
gauss = -> { Math.sqrt(-2 * Math.log(1 - rng.rand)) * Math.cos(2 * Math::PI * rng.rand) }  # 1 - u keeps log() away from 0

puts "=================== Binomial (logistic) mixed model on a real dataset (penguins) ==================="
# Predicting sex from body mass, grouped by island: unlike species (which is almost perfectly
# confounded with island in this dataset -- Gentoo penguins occur on Biscoe island only), sex
# is not tied to location, so this is a well-behaved fit.
penguins = Datamunge::Data.penguins
rows = (0...penguins.nrows).reject { |i| %w[sex body_mass_g island].any? { |c| penguins.is_null(c, i) } }
sex_df = Datamunge::DataFrame.new
sex_df.add_numeric_column("is_male", rows.map { |i| penguins.string_at("sex", i) == "male" ? 1.0 : 0.0 })
sex_df.add_numeric_column("body_mass_g", rows.map { |i| penguins.numeric_at("body_mass_g", i) })
sex_df.add_string_column("island", rows.map { |i| penguins.string_at("island", i) })
Datamunge::GLMM.new(sex_df, "is_male ~ body_mass_g + (1 | island)", "binomial").print_summary

puts "\n=================== Poisson mixed model on simulated multi-site count data ==================="
# 25 stores, ~20 days each: daily visit counts depend on a promo intensity score, but both
# the baseline traffic and the promo's effectiveness vary by store.
n_stores = 25
store_effect = Array.new(n_stores) { gauss.() * 0.4 }
true_intercept = 2.0
true_slope = 0.3

store = []
promo = []
visits = []
n_stores.times do |s|
  (15 + rng.rand(0...11)).times do
    promo_intensity = rng.rand * 3
    lam = Math.exp(true_intercept + store_effect[s] + true_slope * promo_intensity)
    # Knuth's Poisson sampler.
    threshold = Math.exp(-lam)
    k = 0
    p = 1.0
    loop do
      k += 1
      p *= rng.rand
      break unless p > threshold
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

puts "\nTrue generating values: intercept=#{true_intercept}, slope=#{true_slope}, random-intercept SD (log scale)=0.4"

puts "\n--- BLUPs for a few stores ---"
labels = store_model.group_labels
3.times { |idx| puts "store #{labels[idx]}: intercept shift=#{store_model.random_effects_for_group(idx)[0]}" }

puts "\n--- Prediction: population-level vs. store-adjusted ---"
population = Datamunge::DataFrame.new
store0 = Datamunge::DataFrame.new
population.add_numeric_column("promo", [1.5])
store0.add_numeric_column("promo", [1.5])
store0.add_numeric_column("store", [0.0])
puts "promo=1.5, unseen store:   #{store_model.predict(population)[0]} expected visits (fixed effects only)"
puts "promo=1.5, store 0 (known): #{store_model.predict(store0)[0]} expected visits (fixed effects + store 0's BLUP)"
