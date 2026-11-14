! CMD: just TARGET=hypothesis_tests_ex run-fortran

! Classical hypothesis tests (t, Wilcoxon, ANOVA, correlation, F, normality, chi-squared, binomial) on iris.
program hypothesis_tests_ex
use datamunge
use datamunge_data, only: iris
implicit none
type(DataFrame) :: frame
real(c_double), allocatable :: setosa_petal(:), versicolor_petal(:), virginica_petal(:), all_petal(:)
real(c_double), allocatable :: sepal_length(:), petal_length(:), setosa_sepal(:), sorted(:)
real(c_double) :: median_all
integer :: setosa_long, setosa_short, versicolor_long, versicolor_short, long_count, n

frame = iris()

setosa_petal = column_for_species(frame, 'Petal.Length', 'setosa')
versicolor_petal = column_for_species(frame, 'Petal.Length', 'versicolor')
virginica_petal = column_for_species(frame, 'Petal.Length', 'virginica')

print '(a)', '=================== t-tests: petal length, setosa vs. versicolor ==================='
call print_result('Welch two-sample t-test', t_test_two_sample(setosa_petal, versicolor_petal))
call print_result('Wilcoxon rank-sum test', wilcoxon_rank_sum_test(setosa_petal, versicolor_petal))

print '(/, a)', '=================== One-way ANOVA / Kruskal-Wallis across all 3 species ==================='
all_petal = [setosa_petal, versicolor_petal, virginica_petal]
call print_result('One-way ANOVA', one_way_anova(all_petal, &
    [size(setosa_petal), size(versicolor_petal), size(virginica_petal)]))
call print_result('Kruskal-Wallis', kruskal_wallis_test(all_petal, &
    [size(setosa_petal), size(versicolor_petal), size(virginica_petal)]))

print '(/, a)', '=================== Correlation: sepal length vs. petal length ==================='
sepal_length = column(frame, 'Sepal.Length')
petal_length = column(frame, 'Petal.Length')
call print_result('Pearson correlation', pearson_correlation_test(sepal_length, petal_length))
call print_result('Spearman correlation', spearman_correlation_test(sepal_length, petal_length))

print '(/, a)', '=================== F-test: petal length variance, setosa vs. virginica ==================='
call print_result('F test', f_test_variance(setosa_petal, virginica_petal))

print '(/, a)', '=================== Normality: is sepal length normally distributed within setosa? ==================='
setosa_sepal = column_for_species(frame, 'Sepal.Length', 'setosa')
call print_result('Shapiro-Francia', shapiro_francia_test(setosa_sepal))
call print_result('KS vs. fitted normal', ks_test_one_sample_normal(setosa_sepal, 5.006d0, 0.3525d0))

print '(/, a)', '=================== Chi-squared / Fisher: is petal length "long" independent of species? ==================='
sorted = sort_values(all_petal)
median_all = sorted(size(sorted) / 2 + 1)
setosa_long = count(setosa_petal > median_all)
setosa_short = size(setosa_petal) - setosa_long
versicolor_long = count(versicolor_petal > median_all)
versicolor_short = size(versicolor_petal) - versicolor_long
print '(a, i0, a, i0, a, i0, a, i0, a)', 'table: setosa=[', setosa_long, ',', setosa_short, '] versicolor=[', &
    versicolor_long, ',', versicolor_short, ']'
call print_result('Chi-squared independence', chi_squared_test_independence( &
    real([setosa_long, setosa_short, versicolor_long, versicolor_short], c_double), 2, 2))
call print_result('Fisher''s exact test', fisher_exact_test_2x2(setosa_long, setosa_short, versicolor_long, &
    versicolor_short))

print '(/, a)', '=================== Proportion / binomial: fraction of "long" petals overall ==================='
long_count = count(all_petal > median_all)
n = size(all_petal)
call print_result('One-sample proportion test (vs 0.5)', proportion_test_one_sample(long_count, n, 0.5d0))
call print_result('Exact binomial test (vs 0.5)', binomial_test(long_count, n, 0.5d0))

contains

subroutine print_result(label, r)
    character(len=*), intent(in) :: label
    type(HypothesisTestResult), intent(in) :: r
    character(len=:), allocatable :: alternative_name

    select case (r%get_alternative())
    case (Alternative_Less)
        alternative_name = 'less'
    case (Alternative_Greater)
        alternative_name = 'greater'
    case default
        alternative_name = 'two.sided'
    end select
    write (*, '(a, a, a)', advance='no') label, ': statistic=', num(r%get_statistic())
    if (r%get_parameter1() /= 0) write (*, '(a, a)', advance='no') ', df1=', num(r%get_parameter1())
    if (r%get_parameter2() /= 0) write (*, '(a, a)', advance='no') ', df2=', num(r%get_parameter2())
    write (*, '(a, a, a, a, a)', advance='no') ', p=', num(r%get_p_value()), ' (', alternative_name, ')'
    if (r%get_has_conf_int()) write (*, '(a, a, a, a, a)', advance='no') ', CI=[', num(r%get_conf_int_lower()), &
        ', ', num(r%get_conf_int_upper()), ']'
    write (*, '(a, a)') ' -- ', r%get_method()
end subroutine

function column_for_species(df, column_name, species) result(values)
    type(DataFrame), intent(in) :: df
    character(len=*), intent(in) :: column_name, species
    real(c_double), allocatable :: values(:)
    integer :: i

    allocate(values(0))
    do i = 0, df%nrows() - 1
        if (df%string_at('Species', i) == species) values = [values, df%numeric_at(column_name, i)]
    end do
end function

function column(df, column_name) result(values)
    type(DataFrame), intent(in) :: df
    character(len=*), intent(in) :: column_name
    real(c_double), allocatable :: values(:)
    integer :: i

    allocate(values(df%nrows()))
    do i = 0, df%nrows() - 1
        values(i + 1) = df%numeric_at(column_name, i)
    end do
end function

function sort_values(x) result(y)
    real(c_double), intent(in) :: x(:)
    real(c_double), allocatable :: y(:)
    real(c_double) :: key
    integer :: i, j

    y = x
    do i = 2, size(y)
        key = y(i)
        j = i - 1
        do while (j >= 1)
            if (y(j) <= key) exit
            y(j + 1) = y(j)
            j = j - 1
        end do
        y(j + 1) = key
    end do
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
