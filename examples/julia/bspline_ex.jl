# CMD: just TARGET=bspline_ex run-julia

# Tensor-product B-spline regression: fit z ~ bs(x, y) to a smooth surface and predict.
using Datamunge
using Printf

# Sample a smooth surface z = sin(x) + y^2 on an 8x8 grid over [0, 1]^2.
const data = let x = Float64[], y = Float64[], z = Float64[]
    for iy in 0:7, ix in 0:7
        xv, yv = ix / 7, iy / 7
        push!(x, xv); push!(y, yv); push!(z, sin(xv) + yv^2)
    end
    frame = DataFrame()
    add_numeric_column(frame, "x", x)
    add_numeric_column(frame, "y", y)
    add_numeric_column(frame, "z", z)
    frame
end

const surface = LM(data, "z ~ bs(x, y)")
println("Fitted z ~ bs(x, y)")
print_summary(surface)

let new_points = DataFrame()
    add_numeric_column(new_points, "x", [0.25, 0.75])
    add_numeric_column(new_points, "y", [0.50, 0.25])
    pred = predict(surface, new_points)
    @printf("Predictions: [%.6f, %.6f]\n", pred[1], pred[2])
end
