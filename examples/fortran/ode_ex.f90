! CMD: just TARGET=ode_ex run-fortran

! ODEs: built-in named systems, harmonic-oscillator energy check and a Lorenz phase plot.
program ode_ex
use datamunge
implicit none

! NOTE: like the Lua version this example uses the built-in named systems via solve_builtin.

print '(a)', '=================== Every named built-in system (no director needed) ==================='
block
    type(ODESolver) :: solver
    solver = ODESolver()
    call run_builtin(solver, 'exponential_decay', [1d0], [1d0])
    call run_builtin(solver, 'logistic_growth', [1d0, 1d0], [0.5d0])
    call run_builtin(solver, 'harmonic_oscillator', [1d0], [1d0, 0d0])
    call run_builtin(solver, 'van_der_pol', [1d0], [2d0, 0d0])
    call run_builtin(solver, 'lorenz', [10d0, 28d0, 8d0 / 3], [1d0, 1d0, 1d0])
end block

print '(/, a)', '=================== Harmonic oscillator (energy conservation) ==================='
block
    type(ODESolution) :: sol
    type(ODESolver) :: solver
    type(RPlot) :: p
    real(c_double), allocatable :: last(:), ts(:), xs(:), vs(:)
    real(c_double) :: x, v

    solver = rk4_solver(0.01d0)
    sol = solver%solve_builtin('harmonic_oscillator', [1d0], [1d0, 0d0], 0d0, 20d0)
    last = last_state(sol)
    x = last(1)
    v = last(2)
    print '(a, a, a, a, a)', 'x(20) = ', f6(x), ' (cos(20) = ', f6(cos(20d0)), ')'
    print '(a, a, a)', 'energy x^2+v^2 = ', f6(x * x + v * v), ' (should stay near 1.0)'
    ts = solution_times(sol)
    xs = solution_component(sol, 1)
    vs = solution_component(sol, 2)
    p = rplot_plot(ts, xs, 'l', 'x(t)')
    p = p%lines(ts, vs, 'v(t)')
    p = p%title_rplot('Harmonic Oscillator')
    p = p%x_label_rplot('t')
    p = p%y_label_rplot('state')
    call p%save_svg('ode_harmonic_oscillator_fortran.svg')
    print '(a)', 'wrote ode_harmonic_oscillator_fortran.svg'
end block

print '(/, a)', '=================== Lorenz attractor (phase plane) ==================='
block
    type(ODESolution) :: sol
    type(ODESolver) :: solver
    type(RPlot) :: p
    real(c_double), allocatable :: xs(:), zs(:)

    solver = rk4_solver(0.005d0)
    sol = solver%solve_builtin('lorenz', [10d0, 28d0, 8d0 / 3], [1d0, 1d0, 1d0], 0d0, 25d0)
    print '(a, i0)', 'steps_taken = ', sol%get_steps_taken()
    xs = solution_component(sol, 1)
    zs = solution_component(sol, 3)
    p = rplot_plot(xs, zs, 'l', 'trajectory')
    p = p%title_rplot('Lorenz Attractor (x-z phase plane)')
    p = p%x_label_rplot('x')
    p = p%y_label_rplot('z')
    call p%save_svg('ode_lorenz_phase_plane_fortran.svg')
    print '(a)', 'wrote ode_lorenz_phase_plane_fortran.svg'
end block

contains

subroutine run_builtin(solver, name, params, y0)
    type(ODESolver), intent(in) :: solver
    character(len=*), intent(in) :: name
    real(c_double), intent(in) :: params(:), y0(:)
    type(ODESolution) :: sol
    real(c_double), allocatable :: last(:)
    character(len=22) :: name_col
    character(len=6) :: steps_col
    character(len=:), allocatable :: state
    integer :: i

    sol = solver%solve_builtin(name, params, y0, 0d0, 1d0)
    last = last_state(sol)
    state = ''
    do i = 1, size(last)
        if (i > 1) state = state // ', '
        state = state // fmt_g(last(i))
    end do
    name_col = name
    write (steps_col, '(i0)') sol%get_steps_taken()
    print '(a, a, a, a, a, a)', name_col, ' steps=', steps_col, ' final state=[', state, ']'
end subroutine

! Like C's %.6f (f0.6 would drop the leading zero).
function f6(x) result(s)
    real(c_double), intent(in) :: x
    character(len=:), allocatable :: s
    character(len=32) :: buf

    write (buf, '(f32.6)') x
    s = trim(adjustl(buf))
end function

! Approximate C's %g: six significant digits, no padding.
function fmt_g(x) result(s)
    real(c_double), intent(in) :: x
    character(len=:), allocatable :: s
    character(len=32) :: buf
    integer :: e

    if (x == 0) then
        s = '0'
        return
    end if
    write (buf, '(es14.5e3)') x
    read (buf(index(buf, 'E') + 1:), *) e
    if (e < -4 .or. e >= 6) then
        s = strip_zeros(trim(adjustl(buf(:index(buf, 'E') - 1))))
        write (buf, '(i0)') abs(e)
        if (abs(e) < 10) buf = '0' // trim(buf)
        s = s // 'e' // merge('-', '+', e < 0) // trim(buf)
    else
        write (buf, '(f0.' // char(ichar('0') + 5 - e) // ')') x
        s = strip_zeros(trim(adjustl(buf)))
        if (s(1:1) == '.') s = '0' // s
        if (s(1:2) == '-.') s = '-0' // s(2:)
    end if
end function

function strip_zeros(t) result(s)
    character(len=*), intent(in) :: t
    character(len=:), allocatable :: s

    s = t
    if (index(s, '.') == 0) return
    do while (s(len(s):len(s)) == '0')
        s = s(:len(s) - 1)
    end do
    if (s(len(s):len(s)) == '.') s = s(:len(s) - 1)
end function

function last_state(sol) result(state)
    type(ODESolution), intent(in) :: sol
    real(c_double), allocatable :: state(:)

    state = sol%state_at(sol%size() - 1)
end function

! The time vector, one value per step.
function solution_times(sol) result(ts)
    type(ODESolution), intent(in) :: sol
    real(c_double), allocatable :: ts(:)
    integer :: i

    allocate(ts(sol%size()))
    do i = 0, sol%size() - 1
        ts(i + 1) = sol%time_at(i)
    end do
end function

! One state component (1-based), one value per step.
function solution_component(sol, component) result(values)
    type(ODESolution), intent(in) :: sol
    integer, intent(in) :: component
    real(c_double), allocatable :: values(:), state(:)
    integer :: i

    allocate(values(sol%size()))
    do i = 0, sol%size() - 1
        state = sol%state_at(i)
        values(i + 1) = state(component)
    end do
end function

function rk4_solver(step) result(solver)
    real(c_double), intent(in) :: step
    type(ODESolver) :: solver
    type(ODEOptions) :: options

    options = ODEOptions()
    call options%set_method(StepMethod_RK4)
    call options%set_step_size(step)
    solver = ODESolver(options)
end function
end program
