! CMD: just TARGET=optim_advanced_ex run-fortran

! Advanced datamunge optimizers (ProximalGradient, FISTA, LevenbergMarquardt, Newton, TrustRegionNewton,
! AugmentedLagrangian, SQP, InteriorPoint, BayesianOptimization, RBFGaussianProcessSurrogate) from Fortran.
!
! Every one of these optimizers takes an interface (ProximalFunction / HessianFunction /
! EqualityConstrainedFunction / InequalityConstrainedFunction / ResidualFunction, or
! ArbitraryFunction for BayesianOptimization) that the C++ library leaves abstract: callers
! implement the objective. From Fortran, extend the interface type and override its type-bound
! procedures (then call <type>_init on a target variable); the optimizer then calls the Fortran
! overrides. Coordinates are updated in place through an intent(inout) allocatable array (the C++
! parameter is a non-const std::vector<double>&).
module optim_advanced_objectives
use datamunge
implicit none
private
public :: Quadratic, Exponential, Incomplete

! f(x, y) = (x - 1)^2 + 10 (y + 2)^2 with exact gradient and Hessian.
type, extends(HessianFunction) :: Quadratic
contains
    procedure :: evaluate => quadratic_evaluate
    procedure :: gradient => quadratic_gradient
    procedure :: hessian => quadratic_hessian
end type

! Residuals of y = a * exp(b t) against exact data from a = 2, b = 0.5.
type, extends(ResidualFunction) :: Exponential
    real(c_double), allocatable :: ts(:)
contains
    procedure :: residuals => exponential_residuals
    procedure :: jacobian => exponential_jacobian
end type

! Leaves gradient() and hessian() out.
type, extends(HessianFunction) :: Incomplete
contains
    procedure :: evaluate => incomplete_evaluate
end type

contains

function quadratic_evaluate(self, coordinates, status) result(r)
    class(Quadratic) :: self
    real(c_double), intent(in) :: coordinates(:)
    type(rosetta_error), intent(out), optional :: status
    real(c_double) :: r
    if (present(status)) status%code = 0
    r = (coordinates(1) - 1)**2 + 10 * (coordinates(2) + 2)**2
end function

function quadratic_gradient(self, coordinates, status) result(r)
    class(Quadratic) :: self
    real(c_double), intent(in) :: coordinates(:)
    type(rosetta_error), intent(out), optional :: status
    real(c_double), allocatable :: r(:)
    if (present(status)) status%code = 0
    r = [2 * (coordinates(1) - 1), 20 * (coordinates(2) + 2)]
end function

function quadratic_hessian(self, coordinates, status) result(r)
    class(Quadratic) :: self
    real(c_double), intent(in) :: coordinates(:)
    type(rosetta_error), intent(out), optional :: status
    type(vector_real), allocatable :: r(:)
    if (present(status)) status%code = 0
    allocate(r(size(coordinates)))
    r(1)%v = [2d0, 0d0]
    r(2)%v = [0d0, 20d0]
end function

function exponential_residuals(self, coordinates, status) result(r)
    class(Exponential) :: self
    real(c_double), intent(in) :: coordinates(:)
    type(rosetta_error), intent(out), optional :: status
    real(c_double), allocatable :: r(:)
    if (present(status)) status%code = 0
    r = coordinates(1) * exp(coordinates(2) * self%ts) - 2 * exp(0.5d0 * self%ts)
end function

function exponential_jacobian(self, coordinates, status) result(r)
    class(Exponential) :: self
    real(c_double), intent(in) :: coordinates(:)
    type(rosetta_error), intent(out), optional :: status
    type(vector_real), allocatable :: r(:)
    integer :: i
    real(c_double) :: t
    if (present(status)) status%code = 0
    allocate(r(size(self%ts)))
    do i = 1, size(self%ts)
        t = self%ts(i)
        r(i)%v = [exp(coordinates(2) * t), coordinates(1) * t * exp(coordinates(2) * t)]
    end do
end function

function incomplete_evaluate(self, coordinates, status) result(r)
    class(Incomplete) :: self
    real(c_double), intent(in) :: coordinates(:)
    type(rosetta_error), intent(out), optional :: status
    real(c_double) :: r
    if (present(status)) status%code = 0
    r = 0 * size(coordinates)
end function
end module

program optim_advanced_ex
use datamunge
use optim_advanced_objectives
implicit none
real(c_double), parameter :: xs(6) = [0.0d0, 0.5d0, 1.0d0, 1.5d0, 2.0d0, 3.0d0]
real(c_double), parameter :: probes(3) = [0.25d0, 1.0d0, 2.5d0]
real(c_double) :: ys(6), incumbent, value, sse
type(vector_real) :: points(6)
type(RBFGaussianProcessSurrogate) :: surrogate
type(ProximalGradientOptions) :: pg_options
type(LevenbergMarquardtOptions) :: lm_options
type(BayesianOptimizationOptions) :: bo_options
type(ProximalGradient) :: pg
type(FISTA) :: fista1
type(LevenbergMarquardt) :: lm1
type(Newton) :: newton1
type(TrustRegionNewton) :: trn
type(AugmentedLagrangian) :: al
type(SQP) :: sqp1
type(InteriorPoint) :: ip
type(BayesianOptimization) :: bo
type(Quadratic), target :: quad
type(Exponential), target :: expo
type(Incomplete), target :: incomp
type(rosetta_error) :: err
real(c_double), allocatable :: start(:), params(:), origin(:)
integer :: i

print '(a)', '=================== RBFGaussianProcessSurrogate (concrete BayesianSurrogate) ==================='
! Fit a 1D surrogate to a simple bowl y = (x-1)^2 sampled at a few points,
! then check the acquisition function favors an unexplored region.
ys = (xs - 1)**2
do i = 1, size(xs)
    points(i)%v = [xs(i)]
end do
surrogate = RBFGaussianProcessSurrogate(1d0, 1d-6)
call surrogate%fit(points, ys)
incumbent = min(0d0, minval(ys))  ! best observed value so far
print '(a)', 'incumbent (best observed y): ' // num(incumbent)
do i = 1, size(probes)
    print '(a, a, a, a)', 'acquisition(x=', fixed(probes(i), 2), ') = ', &
        fixed(surrogate%acquisition([probes(i)], incumbent), 6)
end do

print '(/, a)', '=================== Options structs + optimizer construction (all 10 new classes) ==================='
! Every new Options struct and optimizer class constructs cleanly, proving
! the binding for the whole batch is wired correctly.
pg_options = ProximalGradientOptions()
call pg_options%set_step_size(0.05d0)
pg = ProximalGradient(pg_options)
print '(a)', 'ProximalGradient: ' // pg%class_name() // '()'
fista1 = FISTA(FISTAOptions())
print '(a)', 'FISTA: ' // fista1%class_name() // '()'
lm_options = LevenbergMarquardtOptions()
call lm_options%set_max_iterations(200)
lm1 = LevenbergMarquardt(lm_options)
print '(a)', 'LevenbergMarquardt: ' // lm1%class_name() // '()'
newton1 = Newton(NewtonOptions())
print '(a)', 'Newton: ' // newton1%class_name() // '()'
trn = TrustRegionNewton(TrustRegionNewtonOptions())
print '(a)', 'TrustRegionNewton: ' // trn%class_name() // '()'
al = AugmentedLagrangian(AugmentedLagrangianOptions())
print '(a)', 'AugmentedLagrangian: ' // al%class_name() // '()'
sqp1 = SQP(SQPOptions())
print '(a)', 'SQP: ' // sqp1%class_name() // '()'
ip = InteriorPoint(InteriorPointOptions())
print '(a)', 'InteriorPoint: ' // ip%class_name() // '()'
bo_options = BayesianOptimizationOptions()
call bo_options%set_initial_samples(8)
bo = BayesianOptimization(bo_options)
print '(a)', 'BayesianOptimization: ' // bo%class_name() // '()'

print '(/, a)', '=================== Fortran objectives (extending the C++ interfaces) ==================='
! Newton's method on f(x, y) = (x - 1)^2 + 10 (y + 2)^2 with exact gradient and Hessian.
call hessianfunction_init(quad)
start = [5d0, 5d0]
value = newton1%optimize(quad, start)
print '(a, a, a, a, a, a, a)', 'Newton: f = ', general(value, 6), ' at (', fixed(start(1), 6), ', ', &
    fixed(start(2), 6), ') (exact minimum 0 at (1, -2))'

! Levenberg-Marquardt fitting y = a * exp(b t) to exact data from a = 2, b = 0.5.
expo%ts = [0.0d0, 0.5d0, 1.0d0, 1.5d0, 2.0d0]
call residualfunction_init(expo)
params = [1d0, 0.1d0]
sse = lm1%optimize(expo, params)
print '(a, a, a, a, a, a, a)', 'LevenbergMarquardt: residual = ', general(sse, 3), ', a = ', fixed(params(1), 6), &
    ', b = ', fixed(params(2), 6), ' (exact: 2, 0.5)'

! An interface method a subclass leaves out throws when the optimizer calls it.
call hessianfunction_init(incomp)
origin = [0d0, 0d0]
value = newton1%optimize(incomp, origin, status=err)
print '(a)', 'missing gradient(): ok=' // yes_no(err%code == 0) // ', error mentions it: ' // &
    yes_no(err%code /= 0 .and. index(err%message, 'gradient') > 0)

contains

function yes_no(flag) result(text)
    logical, intent(in) :: flag
    character(len=:), allocatable :: text
    text = merge('true ', 'false', flag)
    text = trim(text)
end function

! Fixed-point notation with `digits` decimals and a leading zero (like %.Nf).
function fixed(x, digits) result(text)
    real(c_double), intent(in) :: x
    integer, intent(in) :: digits
    character(len=:), allocatable :: text
    character(len=40) :: buffer, fmt
    write (fmt, '(a, i0, a)') '(f40.', digits, ')'
    write (buffer, fmt) x
    text = trim(adjustl(buffer))
end function

! `digits` significant digits, trailing zeros dropped (close to %.Ng).
function general(x, digits) result(text)
    real(c_double), intent(in) :: x
    integer, intent(in) :: digits
    character(len=:), allocatable :: text
    character(len=40) :: buffer, fmt
    integer :: e
    if (x == 0) then
        text = '0'
        return
    end if
    e = floor(log10(abs(x)))
    if (e < -4 .or. e >= digits) then
        write (fmt, '(a, i0, a)') '(es40.', digits - 1, 'e2)'
    else
        write (fmt, '(a, i0, a)') '(f40.', max(digits - 1 - e, 0), ')'
    end if
    write (buffer, fmt) x
    text = trim(adjustl(buffer))
end function
! Shortest round-trip decimal text for x (like C#'s/Julia's default double printing).
function num(x) result(s)
    real(c_double), intent(in) :: x
    character(len=:), allocatable :: s
    character(len=48) :: buf, fmt
    real(c_double) :: y
    integer :: d, e, k
    if (x == aint(x) .and. abs(x) < 1d15) then
        write(buf, '(i0)') int(x, c_int64_t)
        s = trim(buf)
        return
    end if
    do d = 2, 17
        write(fmt, '(a, i0, a, i0, a)') '(es', d + 10, '.', d - 1, 'e3)'
        write(buf, fmt) x
        read(buf, *) y
        if (y == x) exit
    end do
    read(buf(index(buf, 'E') + 1:), *) e
    if (e < -5 .or. e >= 15) then
        s = trim(adjustl(buf))
        write(buf, '(a, i0)') s(:index(s, 'E') - 1) // 'e', e
        s = trim(buf)
        return
    end if
    write(fmt, '(a, i0, a)') '(f40.', max(d - 1 - e, 1), ')'
    write(buf, fmt) x
    s = trim(adjustl(buf))
    k = len(s)
    do while (s(k:k) == '0' .and. s(k - 1:k - 1) /= '.')
        k = k - 1
    end do
    s = s(1:k)
end function
end program
