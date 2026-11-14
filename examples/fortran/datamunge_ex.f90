! CMD: just TARGET=datamunge_ex run-fortran

! Smoke test of the basic bindings: vectors, pairs and a native Callback object.
program datamunge_ex
use datamunge
implicit none
real(c_double), allocatable :: v(:), v2(:)
type(pair_real_real) :: p
type(Callback) :: cb

call hello()

v = make_dvector(1d0, 2d0, 3d0)
print '(a, f0.1)', 'sum_dvector:' // achar(9), sum_dvector(v)

p = make_dpair(1.25d0, 2.75d0)
print '(a, f0.1)', 'sum_dpair:' // achar(9), sum_dpair(p)

cb = Callback()
print '(a, f0.1)', 'call_with_callback(3.0):' // achar(9), call_with_callback(3d0, cb)
v2 = map_dvector_with_callback(make_dvector(1d0, 2d0, 3d0), cb)
print '(a, f0.1)', 'sum_dvector(map_dvector_with_callback(1,2,3)):' // achar(9), sum_dvector(v2)
end program
