# CMD: just TARGET=datamunge_ex run-julia

# Smoke test of the basic bindings: vectors, pairs and a native Callback object.
using Datamunge

hello()

let v = make_dvector(1.0, 2.0, 3.0)
    println("sum_dvector:\t$(sum_dvector(v))")
end

let p = make_dpair(1.25, 2.75)
    println("sum_dpair:\t$(sum_dpair(p))")
end

let cb = Callback()
    println("call_with_callback(3.0):\t$(call_with_callback(3.0, cb))")
    v2 = map_dvector_with_callback(make_dvector(1.0, 2.0, 3.0), cb)
    println("sum_dvector(map_dvector_with_callback(1,2,3)):\t$(sum_dvector(v2))")
end
