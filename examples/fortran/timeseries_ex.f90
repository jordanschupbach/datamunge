! CMD: just TARGET=timeseries_ex run-fortran

! Time series: ARIMA, seasonal ARIMA and exponential smoothing (SES, Holt, Holt-Winters).
program timeseries_ex
use datamunge
implicit none
real(c_double), parameter :: pi = acos(-1d0)
real(c_double), parameter :: seasonal_shape(12) = [0.8d0, 0.75d0, 0.9d0, 0.95d0, 1.0d0, 1.05d0, 1.1d0, 1.05d0, &
    1.0d0, 1.1d0, 1.3d0, 1.6d0]
real(c_double), parameter :: flat(10) = [50.2d0, 49.8d0, 50.5d0, 49.6d0, 50.1d0, 50.3d0, 49.9d0, 50.0d0, &
    50.4d0, 49.7d0]
real(c_double) :: y(150), s(120), y2(48), level, prev_shock, shock, prev
real(c_double), allocatable :: ar(:), ma(:), sar(:)
type(ARIMAOptions) :: options, options2
type(ARIMA) :: model, model2
type(pair_vector_real_vector_real) :: intervals
type(ExponentialSmoothingOptions) :: es_options, ses_options, holt_options
type(ExponentialSmoothing) :: es_model, ses, holt
integer :: i

print '(a)', '=================== ARIMA(1,1,1) on a simulated random walk with drift ==================='
call seed(42)
level = 100
prev_shock = 0
do i = 1, 150
    shock = gauss()
    level = level + 0.3d0 + 0.4d0 * prev_shock + shock  ! integrated ARMA(0,1) increments with drift
    y(i) = level
    prev_shock = shock
end do

options = ARIMAOptions()
call options%set_p(1)
call options%set_d(1)
call options%set_q(1)
call options%set_de_population_size(80)
call options%set_de_max_generations(400)
model = ARIMA(y, options)

ar = model%ar_coefficients()
ma = model%ma_coefficients()
print '(a)', 'AR coefficient: ' // num(ar(1))
print '(a)', 'MA coefficient: ' // num(ma(1))
print '(a)', 'sigma^2: ' // num(model%sigma2()) // ', AIC: ' // num(model%aic()) // ', BIC: ' // num(model%bic())

intervals = model%forecast_with_intervals(6)
print '(a)', '6-step forecast: ' // joined(intervals%first)
print '(a)', 'forecast std. errors: ' // joined(intervals%second)

print '(/, a)', '=================== SARIMA(1,0,0)(1,1,0)_12 on a seasonal series ==================='
call seed(7)
prev = 0
do i = 0, 119
    prev = 0.5d0 * prev + gauss()
    s(i + 1) = 20 + 0.2d0 * i + 5 * sin(2 * pi * i / 12) + prev
end do

options2 = ARIMAOptions()
call options2%set_p(1)
call options2%set_seasonal_p(1)
call options2%set_seasonal_d(1)
call options2%set_seasonal_period(12)
call options2%set_de_population_size(100)
call options2%set_de_max_generations(500)
model2 = ARIMA(s, options2)
ar = model2%ar_coefficients()
sar = model2%seasonal_ar_coefficients()
print '(a)', 'AR coefficient: ' // num(ar(1)) // ', seasonal AR coefficient: ' // num(sar(1))
print '(a)', '12-step forecast: ' // joined(model2%forecast(12))

print '(/, a)', '=================== Exponential smoothing: Holt-Winters on retail-style seasonal data ==================='
call seed(11)
do i = 0, 47
    y2(i + 1) = (100 + 2 * i) * seasonal_shape(mod(i, 12) + 1) + gauss() * 3
end do

es_options = ExponentialSmoothingOptions()
call es_options%set_trend(TrendType_Additive)
call es_options%set_seasonal(SeasonalType_Multiplicative)
call es_options%set_seasonal_period(12)
call es_options%set_de_population_size(60)
call es_options%set_de_max_generations(300)
es_model = ExponentialSmoothing(y2, es_options)
print '(a)', 'alpha=' // num(es_model%alpha()) // ' beta=' // num(es_model%beta()) // ' gamma=' // num(es_model%gamma())
print '(a)', 'sigma^2: ' // num(es_model%sigma2()) // ', AIC: ' // num(es_model%aic())
print '(a)', '12-month forecast: ' // joined(es_model%forecast(12))

print '(/, a)', "=================== Simple exponential smoothing vs. Holt's linear trend ==================="
ses_options = ExponentialSmoothingOptions()
ses = ExponentialSmoothing(flat, ses_options)
print '(a)', 'SES alpha= ' // num(ses%alpha())
print '(a)', 'SES 5-step forecast: ' // joined(ses%forecast(5))

holt_options = ExponentialSmoothingOptions()
call holt_options%set_trend(TrendType_Additive)
holt = ExponentialSmoothing(flat, holt_options)
print '(a)', 'Holt alpha=' // num(holt%alpha()) // ' beta=' // num(holt%beta())
print '(a)', 'Holt 5-step forecast: ' // joined(holt%forecast(5))

contains

! Restart the (gfortran) generator from a fixed seed.
subroutine seed(n)
    integer, intent(in) :: n
    integer :: k, j
    call random_seed(size=k)
    call random_seed(put=[(n + 37 * j, j = 1, k)])
end subroutine

! (0, 1]: avoids log(0) in the Box-Muller transform.
function uniform() result(u)
    real(c_double) :: u
    call random_number(u)
    u = 1 - u
end function

function gauss() result(g)
    real(c_double) :: g, u1, u2
    u1 = uniform()
    u2 = uniform()
    g = sqrt(-2 * log(u1)) * cos(2 * pi * u2)
end function

function joined(v) result(text)
    real(c_double), intent(in) :: v(:)
    character(len=:), allocatable :: text
    integer :: j
    text = ''
    do j = 1, size(v)
        if (j > 1) text = text // ', '
        text = text // num(v(j))
    end do
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
