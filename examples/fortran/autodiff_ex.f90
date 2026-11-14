! CMD: just TARGET=autodiff_ex run-fortran

! Automatic differentiation: forward mode (Dual), reverse mode (Tape/Var), Hessians (HyperDual).
program autodiff_ex
use datamunge
implicit none

! Note: the C++ autodiff_ex.cpp additionally demonstrates the generic template drivers
! (derivative/gradient_forward/gradient_reverse/jacobian_forward/hessian), which can't be
! bound. This port demonstrates the same ideas with the bound named-method API.
! v%index() is the Var's 0-based position in the adjoint vector t%backward returns.
! Fortran can't call a method on a function result, so every intermediate gets its own variable.

print '(a)', '=================== Forward mode: scalar derivative ==================='
block
    type(Dual) :: x0, x2, x3, twox, f
    x0 = Dual(2d0, 1d0)  ! seed derivative = 1 to read df/dx directly
    x2 = x0%multiply(x0)
    x3 = x2%multiply(x0)
    twox = x0%multiply_scalar(2d0)
    f = x3%subtract(twox)  ! x^3 - 2x
    print '(a)', "f(x) = x^3 - 2x, f'(2) = " // num(f%derivative()) // ' (exact: 10)'
end block

print '(a)', new_line('a') // '=================== Reverse mode: build a graph by hand ==================='
block
    type(Tape) :: t
    type(Var) :: a, b, ab, sina, y
    real(c_double), allocatable :: adjoint(:)
    t = Tape()
    a = Var(t, 2d0)
    b = Var(t, 3d0)
    ab = a%multiply(b)
    sina = a%sin()
    y = ab%add(sina)
    print '(a)', 'y = a*b + sin(a) at a=2, b=3 -> y = ' // num(y%value())
    adjoint = t%backward(y)
    print '(a)', 'dy/da = ' // num(adjoint(a%index() + 1)) // ' (exact: b + cos(a))'
    print '(a)', 'dy/db = ' // num(adjoint(b%index() + 1)) // ' (exact: a)'
end block

print '(a)', new_line('a') // '=================== Forward vs reverse mode agree on the Rosenbrock function ==================='
block
    real(c_double), parameter :: px = 0d0, py = 0d0
    real(c_double) :: gx, gy
    type(Dual) :: fd
    type(Tape) :: t
    type(Var) :: vx, vy, vf
    real(c_double), allocatable :: grad(:)
    ! Forward mode: one Dual pass per partial derivative, seeding the direction of interest.
    fd = rosenbrock_dual(Dual(px, 1d0), Dual(py, 0d0))
    gx = fd%derivative()
    fd = rosenbrock_dual(Dual(px, 0d0), Dual(py, 1d0))
    gy = fd%derivative()
    ! Reverse mode: one Tape/Var pass computes every partial at once.
    t = Tape()
    vx = Var(t, px)
    vy = Var(t, py)
    vf = rosenbrock_var(t, vx, vy)
    grad = t%backward(vf)
    print '(a)', 'f(0,0) = ' // num(rosenbrock_value(px, py))
    print '(a)', 'gradient (forward mode): [' // num(gx) // ', ' // num(gy) // ']'
    print '(a)', 'gradient (reverse mode): [' // num(grad(vx%index() + 1)) // ', ' // num(grad(vy%index() + 1)) // ']'
end block

print '(a)', new_line('a') // '=================== Jacobian of a vector-valued function ==================='
! f(x,y) = [x^2, xy, y^3] at (2,3) -- one Dual pass per (output, input) pair.
block
    real(c_double) :: jac(3, 2), seeds(2, 2)
    type(Dual) :: dx, dy, outputs(3)
    integer :: row, col
    seeds = reshape([1d0, 0d0, 0d0, 1d0], [2, 2])  ! column col = (sx, sy)
    do col = 1, 2
        dx = Dual(2d0, seeds(1, col))
        dy = Dual(3d0, seeds(2, col))
        call vector_fn(dx, dy, outputs)
        do row = 1, 3
            jac(row, col) = outputs(row)%derivative()
        end do
    end do
    print '(a)', 'f(x,y) = [x^2, xy, y^3] at (2,3), Jacobian:'
    do row = 1, 3
        print '(a)', '  [' // num(jac(row, 1)) // ', ' // num(jac(row, 2)) // ']'
    end do
end block

print '(a)', new_line('a') // '=================== Hessian via second-order forward mode (HyperDual) ==================='
! One HyperDual pass per (i, j) pair reads off d^2f/dxi dxj directly from eps1eps2.
block
    real(c_double), parameter :: mx = 1d0, my = 1d0  ! the Rosenbrock function's minimum
    real(c_double) :: seeds(2, 2), h(2, 2)
    type(HyperDual) :: fh
    integer :: i, j
    seeds = reshape([1d0, 0d0, 0d0, 1d0], [2, 2])  ! column i = (x, y) seed
    do i = 1, 2
        do j = 1, 2
            fh = rosenbrock_hyperdual(HyperDual(mx, seeds(1, i), seeds(1, j), 0d0), &
                                      HyperDual(my, seeds(2, i), seeds(2, j), 0d0))
            h(i, j) = fh%eps1eps2()
        end do
    end do
    print '(a)', 'Hessian of the Rosenbrock function at its minimum (1,1):'
    do i = 1, 2
        print '(a)', '  [' // num(h(i, 1)) // ', ' // num(h(i, 2)) // ']'
    end do
end block

print '(a)', new_line('a') // '=================== Gradient descent driven by reverse-mode gradients ==================='
block
    real(c_double) :: point(2), loss
    real(c_double), parameter :: learning_rate = 0.001d0
    integer, parameter :: n_steps = 2000
    integer :: step
    type(Tape) :: t
    type(Var) :: vx, vy, vf
    real(c_double), allocatable :: grad(:)
    point = [-1.2d0, 1d0]  ! the classic Rosenbrock start
    do step = 0, n_steps - 1
        t = Tape()
        vx = Var(t, point(1))
        vy = Var(t, point(2))
        vf = rosenbrock_var(t, vx, vy)
        grad = t%backward(vf)
        loss = vf%value()
        point(1) = point(1) - learning_rate * grad(vx%index() + 1)
        point(2) = point(2) - learning_rate * grad(vy%index() + 1)
        if (step == 0 .or. step == n_steps - 1) then
            print '(a, i0, a)', 'step ', step, ': loss = ' // num(loss) // ', x = [' // num(point(1)) // ', ' &
                // num(point(2)) // ']'
        end if
    end do
    print '(a)', '(true minimum is at [1, 1] with loss 0)'
end block

contains

function rosenbrock_dual(x0, x1) result(res)
    type(Dual), intent(in) :: x0, x1
    type(Dual) :: res
    type(Dual) :: one, a, sq, b, aa, bb, bb100
    one = Dual(1d0, 0d0)
    a = one%subtract(x0)
    sq = x0%multiply(x0)
    b = x1%subtract(sq)
    aa = a%multiply(a)
    bb = b%multiply(b)
    bb100 = bb%multiply_scalar(100d0)
    res = aa%add(bb100)
end function

function rosenbrock_value(x0, x1) result(res)
    real(c_double), intent(in) :: x0, x1
    real(c_double) :: res
    res = (1d0 - x0)**2 + 100d0 * (x1 - x0 * x0)**2
end function

! Build the Rosenbrock graph on the tape; returns the output Var.
function rosenbrock_var(t, x0, x1) result(res)
    type(Tape), intent(in) :: t
    type(Var), intent(in) :: x0, x1
    type(Var) :: res
    type(Var) :: one, a, sq, b, aa, bb, bb100
    one = Var(t, 1d0)
    a = one%subtract(x0)
    sq = x0%multiply(x0)
    b = x1%subtract(sq)
    aa = a%multiply(a)
    bb = b%multiply(b)
    bb100 = bb%multiply_scalar(100d0)
    res = aa%add(bb100)
end function

function rosenbrock_hyperdual(x0, x1) result(res)
    type(HyperDual), intent(in) :: x0, x1
    type(HyperDual) :: res
    type(HyperDual) :: one, a, sq, b, aa, bb, bb100
    one = HyperDual(1d0, 0d0, 0d0, 0d0)
    a = one%subtract(x0)
    sq = x0%multiply(x0)
    b = x1%subtract(sq)
    aa = a%multiply(a)
    bb = b%multiply(b)
    bb100 = bb%multiply_scalar(100d0)
    res = aa%add(bb100)
end function

subroutine vector_fn(x, y, outputs)
    type(Dual), intent(in) :: x, y
    type(Dual), intent(inout) :: outputs(3)
    type(Dual) :: yy
    outputs(1) = x%multiply(x)
    outputs(2) = x%multiply(y)
    yy = y%multiply(y)
    outputs(3) = yy%multiply(y)
end subroutine

function num(x) result(s)
    ! Shortest round-trip decimal form of x (like Julia/C# default double printing).
    real(c_double), intent(in) :: x
    character(len=:), allocatable :: s
    character(len=40) :: buf, fmt
    real(c_double) :: y
    integer :: k, e
    if (x == 0d0) then
        s = '0.0'
        return
    end if
    do k = 1, 17
        write(fmt, '(a, i0, a)') '(es30.', k, 'e3)'
        write(buf, fmt) x
        read(buf, *) y
        if (y == x) exit
    end do
    read(buf(index(buf, 'E') + 1:), *) e
    if (e >= -5 .and. e < 16) then
        write(fmt, '(a, i0, a)') '(f40.', max(1, k - e), ')'
        write(buf, fmt) x
    end if
    s = trim(adjustl(buf))
    if (index(s, 'E') > 0) then
        write(buf, '(a, i0)') s(:index(s, 'E') - 1) // 'e', e
        s = trim(buf)
    end if
    if (scan(s, 'Ee') == 0) then
        do while (s(len(s):len(s)) == '0' .and. s(len(s) - 1:len(s) - 1) /= '.')
            s = s(:len(s) - 1)
        end do
    end if
end function
end program
