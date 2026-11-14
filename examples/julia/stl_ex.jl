# CMD: just TARGET=stl_ex run-julia

# C++ std::vector and std::pair values in Julia: vectors are Vectors, pairs are tuples.
using Datamunge

let n = 100
    v = [i * 1.5 for i in 0:n-1]
    foreach(println, v)
    v2 = [floor(Int, i * 1.5) for i in 0:n-1]
    foreach(println, v2)
end

let p = (3, 4)
    println("p: ($(p[1]), $(p[2]))")
end

let p2 = (10, 20)
    println("p2: ($(p2[1]), $(p2[2]))")
end

# A C++ std::pair<double,double> comes back as a tuple.
let p3 = make_dpair(1.25, 2.75)
    println("make_dpair(1.25, 2.75): $p3, sum_dpair = $(sum_dpair(p3))")
end
