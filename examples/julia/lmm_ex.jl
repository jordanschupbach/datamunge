# CMD: just TARGET=lmm_ex run-julia

# Linear mixed models: random intercept (penguins) and random intercept + slope (simulated schools).
using Datamunge
using Random

const rng = Xoshiro(2024)
uniform() = 1 - rand(rng)  # (0, 1]: avoids log(0) in the Box-Muller transform.
gauss() = sqrt(-2 * log(uniform())) * cos(2π * uniform())

println("=================== Random intercept on a real dataset (penguins) ===================")
print_summary(LMM(Datamunge.Data.penguins(), "body_mass_g ~ flipper_length_mm + bill_length_mm + (1 | species)"))

println("\n=================== Random intercept + slope on a simulated multi-school dataset ===================")
# 30 schools, ~25 students each: score depends on study_hours, but both the baseline score
# and the return on an extra hour of study vary by school -- the textbook case for a
# random-intercept-and-slope model.
const n_schools = 30
const school_intercept = [gauss() * 6 for _ in 1:n_schools]
const school_slope = [gauss() * 1.2 for _ in 1:n_schools]
const true_intercept = 60.0
const true_slope = 3.0

const df = let school = Float64[], study_hours = Float64[], score = Float64[]
    for s in 0:n_schools-1, _ in 1:15 + rand(rng, 0:20)
        hours = rand(rng) * 10
        push!(school, s)
        push!(study_hours, hours)
        push!(score, true_intercept + school_intercept[s + 1] + (true_slope + school_slope[s + 1]) * hours + gauss() * 4)
    end
    frame = DataFrame()
    add_numeric_column(frame, "school", school)
    add_numeric_column(frame, "study_hours", study_hours)
    add_numeric_column(frame, "score", score)
    frame
end

const model = LMM(df, "score ~ study_hours + (1 + study_hours | school)")
print_summary(model)

println("\nTrue generating values: intercept=$true_intercept, slope=$true_slope, random-intercept SD=6.0, random-slope SD=1.2, residual SD=4.0")

println("\n--- Best Linear Unbiased Predictors (BLUPs) for a few schools ---")
let labels = group_labels(model)
    for idx in 0:2
        re = random_effects_for_group(model, idx)
        println("school $(labels[idx + 1]): intercept shift=$(re[1]), slope shift=$(re[2])")
    end
end

println("\n--- Prediction: population-level vs. school-adjusted ---")
let population = DataFrame(), school0 = DataFrame()
    add_numeric_column(population, "study_hours", [5.0])
    add_numeric_column(school0, "study_hours", [5.0])
    add_numeric_column(school0, "school", [0.0])
    println("5 study hours, unseen school:      $(predict(model, population)[1]) (fixed effects only)")
    println("5 study hours, school 0 (known):    $(predict(model, school0)[1]) (fixed effects + school 0's BLUP)")
end
