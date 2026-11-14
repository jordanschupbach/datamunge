! CMD: just TARGET=lmm_ex run-fortran

! Linear mixed models: random intercept (penguins) and random intercept + slope (simulated schools).
program lmm_ex
use datamunge
use datamunge_data, only: penguins
implicit none
integer, parameter :: n_schools = 30
real(c_double), parameter :: true_intercept = 60d0, true_slope = 3d0
real(c_double), parameter :: pi = 3.14159265358979323846d0
real(c_double) :: school_intercept(n_schools), school_slope(n_schools), hours
real(c_double), allocatable :: school(:), study_hours(:), score(:), predicted(:), re(:)
type(rosetta_string), allocatable :: labels(:)
type(DataFrame) :: frame, df, population, school0
type(LMM) :: model
integer :: s, student, n_students, idx

call seed_rng(2024)

print '(a)', '=================== Random intercept on a real dataset (penguins) ==================='
frame = penguins()
model = LMM(frame, 'body_mass_g ~ flipper_length_mm + bill_length_mm + (1 | species)')
call model%print_summary()

print '(/, a)', '=================== Random intercept + slope on a simulated multi-school dataset ==================='
! 30 schools, ~25 students each: score depends on study_hours, but both the baseline score
! and the return on an extra hour of study vary by school -- the textbook case for a
! random-intercept-and-slope model.
do s = 1, n_schools
    school_intercept(s) = gauss() * 6
end do
do s = 1, n_schools
    school_slope(s) = gauss() * 1.2d0
end do

allocate(school(0), study_hours(0), score(0))
do s = 0, n_schools - 1
    n_students = 15 + int(random_unit() * 21)
    do student = 1, n_students
        hours = random_unit() * 10
        school = [school, real(s, c_double)]
        study_hours = [study_hours, hours]
        score = [score, true_intercept + school_intercept(s + 1) + (true_slope + school_slope(s + 1)) * hours &
            + gauss() * 4]
    end do
end do
df = DataFrame()
call df%add_numeric_column('school', school)
call df%add_numeric_column('study_hours', study_hours)
call df%add_numeric_column('score', score)

model = LMM(df, 'score ~ study_hours + (1 + study_hours | school)')
call model%print_summary()

print '(/, a, f0.1, a, f0.1, a)', 'True generating values: intercept=', true_intercept, ', slope=', true_slope, &
    ', random-intercept SD=6.0, random-slope SD=1.2, residual SD=4.0'

print '(/, a)', '--- Best Linear Unbiased Predictors (BLUPs) for a few schools ---'
labels = model%group_labels()
do idx = 0, 2
    re = model%random_effects_for_group(idx)
    print '(a, a, a, a, a, a)', 'school ', labels(idx + 1)%s, ': intercept shift=', num(re(1)), ', slope shift=', num(re(2))
end do

print '(/, a)', '--- Prediction: population-level vs. school-adjusted ---'
population = DataFrame()
school0 = DataFrame()
call population%add_numeric_column('study_hours', [5d0])
call school0%add_numeric_column('study_hours', [5d0])
call school0%add_numeric_column('school', [0d0])
predicted = model%predict(population)
print '(a, a, a)', '5 study hours, unseen school:      ', num(predicted(1)), ' (fixed effects only)'
predicted = model%predict(school0)
print '(a, a, a)', '5 study hours, school 0 (known):    ', num(predicted(1)), ' (fixed effects + school 0''s BLUP)'

contains

subroutine seed_rng(seed)
    integer, intent(in) :: seed
    integer :: n
    integer, allocatable :: values(:)
    call random_seed(size=n)
    allocate(values(n))
    values = seed
    call random_seed(put=values)
end subroutine

! [0, 1)
real(c_double) function random_unit()
    call random_number(random_unit)
end function

! (0, 1]: avoids log(0) in the Box-Muller transform.
real(c_double) function uniform()
    uniform = 1 - random_unit()
end function

real(c_double) function gauss()
    real(c_double) :: u1, u2
    u1 = uniform()
    u2 = uniform()
    gauss = sqrt(-2 * log(u1)) * cos(2 * pi * u2)
end function

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
