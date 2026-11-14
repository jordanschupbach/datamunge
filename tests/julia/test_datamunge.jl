# Smoke tests of the Datamunge Julia package (just test-julia).
using Datamunge
import REPL # Docs.doc lives there
const dm = Datamunge

checks = 0
function check(ok, message)
    global checks += 1
    ok === true || error("check failed: $message")
end
doc(name::Symbol) = string(Base.Docs.doc(Base.Docs.Binding(Datamunge, name)))

# Smoke test: call into the native library.
hello()

# std::vector<double> and std::pair<double,double> round trips.
let v = make_dvector(1.0, 2.0, 3.0)
    check(v isa Vector{Float64}, "std::vector<double> is a Vector{Float64}")
    check(v == [1.0, 2.0, 3.0], "vector contents")
    check(sum_dvector(v) == 6.0, "sum_dvector")
    check(sum_dvector((0.5, 1.5)) == 2.0, "tuples convert to std::vector")
end
let p = make_dpair(1.25, 2.75)
    check(p == (1.25, 2.75), "std::pair is a tuple")
    check(sum_dpair(p) == 4.0, "sum_dpair")
end

# The C++ Callback class, and a Julia subclass overriding its virtual call() (a director).
let cb = Callback()
    check(call_with_callback(3.0, cb) == 3.0, "native Callback")
    check(sum_dvector(map_dvector_with_callback(make_dvector(1.0, 2.0, 3.0), cb)) == 6.0, "native map")
end

@subclass mutable struct Doubler <: Callback end
dm.call(self::Doubler, x) = 2x
let doubler = Doubler()
    check(call_with_callback(3.0, doubler) == 6.0, "the Julia override runs")
    check(sum_dvector(map_dvector_with_callback([1.0, 2.0, 3.0], doubler)) == 12.0, "override from C++ loops")
end

# A Julia exception inside an override reaches the caller as the original exception.
struct FromJulia <: Exception end
@subclass mutable struct Failing <: Callback end
dm.call(self::Failing, x) = throw(FromJulia())
check((try call_with_callback(1.0, Failing()) catch e; e end) isa FromJulia, "the Julia exception propagates")

# C++ exceptions become NativeError.
check((try seql(0.0, 1.0, -1) catch e; e end) isa NativeError, "C++ exceptions throw NativeError")

# Classes, fields as properties and docstrings carrying the C++ documentation.
let frame = DataFrame()
    add_numeric_column(frame, "x", [1.0, 2.0, 3.0])
    check(nrows(frame) == 3, "DataFrame")
end
check(occursin("evenly spaced", doc(:seql)), "docstrings")
check(occursin("# Arguments", doc(:seql)), "argument sections")

println("test_datamunge.jl: all $checks checks passed")
