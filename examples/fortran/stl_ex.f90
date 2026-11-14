! CMD: just TARGET=stl_ex run-fortran

! C++ std::vector and std::pair values in Fortran: vectors are allocatable arrays, pairs are derived types.
program stl_ex
use datamunge
implicit none
integer, parameter :: n = 100
integer :: i
real(c_double), allocatable :: v(:)
integer, allocatable :: v2(:)
integer :: p(2), p2(2)
type(pair_real_real) :: p3

v = [(i * 1.5d0, i = 0, n - 1)]
do i = 1, n
    print '(a)', num(v(i))
end do
v2 = [(floor(i * 1.5d0), i = 0, n - 1)]
do i = 1, n
    print '(i0)', v2(i)
end do

p = [3, 4]
print '(a, i0, a, i0, a)', 'p: (', p(1), ', ', p(2), ')'

p2 = [10, 20]
print '(a, i0, a, i0, a)', 'p2: (', p2(1), ', ', p2(2), ')'

! A C++ std::pair<double,double> comes back as a pair_real_real (%first, %second).
p3 = make_dpair(1.25d0, 2.75d0)
print '(a)', 'make_dpair(1.25, 2.75): (' // num(p3%first) // ', ' // num(p3%second) // '), sum_dpair = ' &
    // num(sum_dpair(p3))
contains
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
