! CMD: just TARGET=glmm_ex run-fortran

! Generalized linear mixed models: a binomial GLMM on penguins and a Poisson GLMM on simulated store data.
program glmm_ex
use datamunge
use datamunge_data, only: penguins
implicit none
integer, parameter :: n_stores = 25
real(c_double), parameter :: true_intercept = 2d0, true_slope = 0.3d0
real(c_double), parameter :: pi = 3.14159265358979323846d0
real(c_double) :: store_effect(n_stores)
real(c_double), allocatable :: store(:), promo(:), visits(:), predicted(:), re(:)
type(rosetta_string), allocatable :: labels(:)
type(DataFrame) :: df, population, store0
type(GLMM) :: store_model
integer :: s, day, n_days, k, idx
real(c_double) :: promo_intensity, lam, limit_p, p

call seed_rng(4242)

print '(a)', '=================== Binomial (logistic) mixed model on a real dataset (penguins) ==================='
! Predicting sex from body mass, grouped by island: unlike species (which is almost perfectly
! confounded with island in this dataset -- Gentoo penguins occur on Biscoe island only), sex
! is not tied to location, so this is a well-behaved fit.
call binomial_fit()

print '(/, a)', '=================== Poisson mixed model on simulated multi-site count data ==================='
! 25 stores, ~20 days each: daily visit counts depend on a promo intensity score, but both
! the baseline traffic and the promo's effectiveness vary by store.
do s = 1, n_stores
    store_effect(s) = gauss() * 0.4d0
end do

allocate(store(0), promo(0), visits(0))
do s = 0, n_stores - 1
    n_days = 15 + int(uniform() * 11)
    do day = 1, n_days
        promo_intensity = uniform() * 3
        lam = exp(true_intercept + store_effect(s + 1) + true_slope * promo_intensity)
        ! Knuth's Poisson sampler.
        limit_p = exp(-lam)
        k = 0
        p = 1d0
        do
            k = k + 1
            p = p * uniform()
            if (p <= limit_p) exit
        end do
        store = [store, real(s, c_double)]
        promo = [promo, promo_intensity]
        visits = [visits, real(k - 1, c_double)]
    end do
end do
df = DataFrame()
call df%add_numeric_column('store', store)
call df%add_numeric_column('promo', promo)
call df%add_numeric_column('visits', visits)
store_model = GLMM(df, 'visits ~ promo + (1 | store)', 'poisson')

call store_model%print_summary()

print '(/, a, f0.1, a, f0.1, a)', 'True generating values: intercept=', true_intercept, ', slope=0', true_slope, &
    ', random-intercept SD (log scale)=0.4'

print '(/, a)', '--- BLUPs for a few stores ---'
labels = store_model%group_labels()
do idx = 0, 2
    re = store_model%random_effects_for_group(idx)
    print '(a, a, a, a)', 'store ', labels(idx + 1)%s, ': intercept shift=', num(re(1))
end do

print '(/, a)', '--- Prediction: population-level vs. store-adjusted ---'
population = DataFrame()
store0 = DataFrame()
call population%add_numeric_column('promo', [1.5d0])
call store0%add_numeric_column('promo', [1.5d0])
call store0%add_numeric_column('store', [0d0])
predicted = store_model%predict(population)
print '(a, a, a)', 'promo=1.5, unseen store:   ', num(predicted(1)), ' expected visits (fixed effects only)'
predicted = store_model%predict(store0)
print '(a, a, a)', 'promo=1.5, store 0 (known): ', num(predicted(1)), ' expected visits (fixed effects + store 0''s BLUP)'

contains

subroutine binomial_fit()
    type(DataFrame) :: frame, sex_df
    type(GLMM) :: sex_model
    real(c_double), allocatable :: is_male(:), body_mass(:)
    character(len=16), allocatable :: island(:)
    integer :: i, n, m

    frame = penguins()
    n = frame%nrows()
    allocate(is_male(n), body_mass(n), island(n))
    m = 0
    do i = 0, n - 1
        if (frame%is_null('sex', i) .or. frame%is_null('body_mass_g', i) .or. frame%is_null('island', i)) cycle
        m = m + 1
        is_male(m) = merge(1d0, 0d0, frame%string_at('sex', i) == 'male')
        body_mass(m) = frame%numeric_at('body_mass_g', i)
        island(m) = frame%string_at('island', i)
    end do
    sex_df = DataFrame()
    call sex_df%add_numeric_column('is_male', is_male(1:m))
    call sex_df%add_numeric_column('body_mass_g', body_mass(1:m))
    call sex_df%add_string_column('island', island(1:m))
    sex_model = GLMM(sex_df, 'is_male ~ body_mass_g + (1 | island)', 'binomial')
    call sex_model%print_summary()
end subroutine

subroutine seed_rng(seed)
    integer, intent(in) :: seed
    integer :: n
    integer, allocatable :: values(:)
    call random_seed(size=n)
    allocate(values(n))
    values = seed
    call random_seed(put=values)
end subroutine

real(c_double) function uniform()
    call random_number(uniform)
end function

real(c_double) function gauss()
    real(c_double) :: u1, u2
    u1 = 1 - uniform()  ! 1 - u keeps log() away from 0
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
