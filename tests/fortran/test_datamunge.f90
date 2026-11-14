! Smoke tests of the datamunge Fortran binding (just test-fortran). Argument: the generated module
! source (default src/datamungefortran/datamunge.f90), for the documentation checks.
module test_datamunge_support
use datamunge
implicit none
private
public :: check, Doubler, Failing, checks, doc_of

integer :: checks = 0

! A Fortran extension overriding the C++ Callback's virtual call() (a director).
type, extends(Callback) :: Doubler
contains
    procedure :: call => doubler_call
end type

! An override that reports an error.
type, extends(Callback) :: Failing
contains
    procedure :: call => failing_call
end type

contains

subroutine check(ok, what)
    logical, intent(in) :: ok
    character(len=*), intent(in) :: what
    checks = checks + 1
    if (.not. ok) error stop 'check failed: ' // what
end subroutine

function doubler_call(self, x, status) result(r)
    class(Doubler) :: self
    real(c_double), intent(in) :: x
    type(rosetta_error), intent(out), optional :: status
    real(c_double) :: r
    if (present(status)) status%code = 0
    r = 2 * x
end function

function failing_call(self, x, status) result(r)
    class(Failing) :: self
    real(c_double), intent(in) :: x
    type(rosetta_error), intent(out), optional :: status
    real(c_double) :: r
    r = x
    if (present(status)) then
        status%code = 2
        status%message = 'from Fortran'
    end if
end function

! The FORD (!>) block right above the first line of `path` containing `declaration`.
function doc_of(path, declaration) result(block)
    character(len=*), intent(in) :: path, declaration
    character(len=:), allocatable :: block
    character(len=4096) :: buffer
    character(len=:), allocatable :: line
    integer :: unit, ios
    block = ''
    open(newunit=unit, file=path, status='old', action='read', iostat=ios)
    if (ios /= 0) error stop 'cannot open ' // path
    do
        read(unit, '(a)', iostat=ios) buffer
        if (ios /= 0) exit
        line = trim(adjustl(buffer))
        if (index(line, '!>') == 1) then
            block = block // trim(line(3:)) // new_line('a')
        else if (index(buffer, declaration) > 0) then
            close(unit)
            return
        else
            block = ''
        end if
    end do
    close(unit)
    error stop 'not declared: ' // declaration
end function
end module

program test_datamunge
use datamunge
use test_datamunge_support
implicit none
type(rosetta_error) :: st
type(Callback) :: cb
type(Doubler), target :: twice
type(Failing), target :: bad
type(DataFrame) :: frame
type(pair_real_real) :: p
type(rosetta_string), allocatable :: cols(:)
real(c_double), allocatable :: v(:), draws(:)
real(c_double) :: r
character(len=:), allocatable :: source, d

source = 'src/datamungefortran/datamunge.f90'
if (command_argument_count() > 0) then
    deallocate(source)
    allocate(character(len=4096) :: source)
    call get_command_argument(1, source)
    source = trim(source)
end if

! Smoke test: call into the native library.
call hello()

! std::vector<double> and std::pair<double,double> round trips.
v = make_dvector(1d0, 2d0, 3d0)
call check(size(v) == 3 .and. all(v == [1d0, 2d0, 3d0]), 'std::vector<double> is a real array')
call check(sum_dvector(v) == 6, 'sum_dvector')
call check(sum_dvector([0.5d0, 1.5d0]) == 2, 'array constructors convert to std::vector')
p = make_dpair(1.25d0, 2.75d0)
call check(p%first == 1.25d0 .and. p%second == 2.75d0, 'std::pair is a derived type')
call check(sum_dpair(p) == 4, 'sum_dpair')

! The C++ Callback class, and a Fortran extension overriding its virtual call() (a director).
cb = Callback()
call check(call_with_callback(3d0, cb) == 3, 'native Callback')
call check(sum_dvector(map_dvector_with_callback(make_dvector(1d0, 2d0, 3d0), cb)) == 6, 'native map')
call callback_init(twice)
call check(call_with_callback(3d0, twice) == 6, 'the Fortran override runs')
call check(sum_dvector(map_dvector_with_callback([1d0, 2d0, 3d0], twice)) == 12, 'override from C++ loops')
r = call_with_callback(2d0, twice, status=st)
call check(st%code == 0 .and. r == 4, 'status reports success')

! An error reported by an override reaches the caller's status.
call callback_init(bad)
r = call_with_callback(1d0, bad, status=st)
call check(st%code /= 0 .and. index(st%message, 'from Fortran') > 0, 'the Fortran error propagates')

! C++ exceptions set status (without status=, they stop the program).
draws = runif(1d0, 0d0, 3, status=st)
call check(st%code /= 0 .and. len(st%message) > 0, 'C++ exceptions set status')

! Classes and type-bound procedures.
frame = DataFrame()
call frame%add_numeric_column('x', [1d0, 2d0, 3d0])
call check(frame%nrows() == 3, 'DataFrame')
cols = frame%columns()
call check(size(cols) == 1 .and. cols(1)%s == 'x', 'std::vector<std::string> is a rosetta_string array')

! FORD comments carry the C++ documentation.
d = doc_of(source, 'function seql_1(')
call check(index(d, 'evenly spaced') > 0, 'FORD documentation')
call check(index(d, '@param n') > 0, 'parameter documentation')

print '(a, i0, a)', 'test_datamunge.f90: all ', checks, ' checks passed'
end program
