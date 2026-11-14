# CMD: just TARGET=lm_ex run-julia

# Linear model: mpg ~ hp + wt + transmission, its ANOVA table and predictions.
using Datamunge

const cars = let frame = DataFrame()
    add_numeric_column(frame, "hp", [110.0, 110.0, 93.0, 110.0, 175.0, 105.0, 245.0, 62.0, 95.0, 123.0])
    add_numeric_column(frame, "wt", [2.62, 2.875, 2.32, 3.215, 3.44, 3.46, 3.57, 3.19, 3.15, 3.44])
    add_string_column(frame, "transmission", ["manual", "manual", "manual", "automatic", "automatic",
                                              "automatic", "automatic", "automatic", "automatic", "automatic"])
    add_numeric_column(frame, "mpg", [21.0, 21.0, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2])
    frame
end

println("Fitting: mpg ~ hp + wt + transmission\n")
const model = LM(cars, "mpg ~ hp + wt + transmission")
print_summary(model)

println("\nSequential ANOVA:\n$(to_string(anova(model)))")

let newcars = DataFrame()
    add_numeric_column(newcars, "hp", [150.0, 90.0])
    add_numeric_column(newcars, "wt", [3.0, 2.5])
    add_string_column(newcars, "transmission", ["manual", "automatic"])
    println("\nPredictions with 95% confidence intervals:\n$(to_string(predict_frame(model, newcars, "confidence")))")
end

save_diagnostic_plots(model, "lm_ex_diagnostics")
println("\nSaved diagnostic plots as lm_ex_diagnostics_*.svg")
