# CMD: just TARGET=glmm_ex run-julia

# Generalized linear mixed models: a binomial GLMM on penguins and a Poisson GLMM on simulated store data.
using Datamunge
using Random

const rng = Xoshiro(4242)
uniform() = rand(rng)
function gauss()
    u1, u2 = 1 - uniform(), uniform()  # 1 - u keeps log() away from 0
    sqrt(-2 * log(u1)) * cos(2π * u2)
end

println("=================== Binomial (logistic) mixed model on a real dataset (penguins) ===================")
# Predicting sex from body mass, grouped by island: unlike species (which is almost perfectly
# confounded with island in this dataset -- Gentoo penguins occur on Biscoe island only), sex
# is not tied to location, so this is a well-behaved fit.
let penguins = Datamunge.Data.penguins()
    is_male, body_mass, island = Float64[], Float64[], String[]
    for i in 0:nrows(penguins)-1
        any(c -> is_null(penguins, c, i), ("sex", "body_mass_g", "island")) && continue
        push!(is_male, string_at(penguins, "sex", i) == "male" ? 1.0 : 0.0)
        push!(body_mass, numeric_at(penguins, "body_mass_g", i))
        push!(island, string_at(penguins, "island", i))
    end
    sex_df = DataFrame()
    add_numeric_column(sex_df, "is_male", is_male)
    add_numeric_column(sex_df, "body_mass_g", body_mass)
    add_string_column(sex_df, "island", island)
    print_summary(GLMM(sex_df, "is_male ~ body_mass_g + (1 | island)", "binomial"))
end

println("\n=================== Poisson mixed model on simulated multi-site count data ===================")
# 25 stores, ~20 days each: daily visit counts depend on a promo intensity score, but both
# the baseline traffic and the promo's effectiveness vary by store.
const n_stores = 25
const store_effect = [gauss() * 0.4 for _ in 1:n_stores]
const true_intercept = 2.0
const true_slope = 0.3

const store_model = let store = Float64[], promo = Float64[], visits = Float64[]
    for s in 0:n_stores-1
        n_days = 15 + rand(rng, 0:10)
        for _ in 1:n_days
            promo_intensity = uniform() * 3
            lam = exp(true_intercept + store_effect[s + 1] + true_slope * promo_intensity)
            # Knuth's Poisson sampler.
            threshold, k, p = exp(-lam), 0, 1.0
            while true
                k += 1
                p *= uniform()
                p <= threshold && break
            end
            push!(store, s)
            push!(promo, promo_intensity)
            push!(visits, k - 1)
        end
    end
    df = DataFrame()
    add_numeric_column(df, "store", store)
    add_numeric_column(df, "promo", promo)
    add_numeric_column(df, "visits", visits)
    GLMM(df, "visits ~ promo + (1 | store)", "poisson")
end

print_summary(store_model)

println("\nTrue generating values: intercept=$true_intercept, slope=$true_slope, random-intercept SD (log scale)=0.4")

println("\n--- BLUPs for a few stores ---")
let labels = group_labels(store_model)
    for idx in 0:2
        println("store $(labels[idx + 1]): intercept shift=$(random_effects_for_group(store_model, idx)[1])")
    end
end

println("\n--- Prediction: population-level vs. store-adjusted ---")
let population = DataFrame(), store0 = DataFrame()
    add_numeric_column(population, "promo", [1.5])
    add_numeric_column(store0, "promo", [1.5])
    add_numeric_column(store0, "store", [0.0])
    println("promo=1.5, unseen store:   $(predict(store_model, population)[1]) expected visits (fixed effects only)")
    println("promo=1.5, store 0 (known): $(predict(store_model, store0)[1]) expected visits (fixed effects + store 0's BLUP)")
end
