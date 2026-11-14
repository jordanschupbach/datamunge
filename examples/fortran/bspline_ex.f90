! CMD: just TARGET=bspline_ex run-fortran

! Tensor-product B-spline regression: fit z ~ bs(x, y) to a smooth surface and predict.
program bspline_ex
use datamunge
implicit none
real(c_double) :: x(64), y(64), z(64), xv, yv
real(c_double), allocatable :: pred(:)
type(DataFrame) :: frame, new_points
type(LM) :: surface
integer :: ix, iy, k

! Sample a smooth surface z = sin(x) + y^2 on an 8x8 grid over [0, 1]^2.
k = 0
do iy = 0, 7
    do ix = 0, 7
        xv = ix / 7d0
        yv = iy / 7d0
        k = k + 1
        x(k) = xv
        y(k) = yv
        z(k) = sin(xv) + yv * yv
    end do
end do
frame = DataFrame()
call frame%add_numeric_column('x', x)
call frame%add_numeric_column('y', y)
call frame%add_numeric_column('z', z)

surface = LM(frame, 'z ~ bs(x, y)')
print '(a)', 'Fitted z ~ bs(x, y)'
call surface%print_summary()

new_points = DataFrame()
call new_points%add_numeric_column('x', [0.25d0, 0.75d0])
call new_points%add_numeric_column('y', [0.50d0, 0.25d0])
pred = surface%predict(new_points)
print '(a)', 'Predictions: [' // fix6(pred(1)) // ', ' // fix6(pred(2)) // ']'
contains
! x with 6 decimals and a leading zero (like C# {x:F6} / %.6f).
function fix6(x) result(s)
    real(c_double), intent(in) :: x
    character(len=:), allocatable :: s
    character(len=32) :: buf
    write(buf, '(f32.6)') x
    s = trim(adjustl(buf))
end function
end program
