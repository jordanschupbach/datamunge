! CMD: just TARGET=tensor_ex run-fortran

! Tensors: construction, shape ops, broadcasting, reductions, linear algebra, masks, iris.
program tensor_ex
use datamunge
use datamunge_data, only: iris
implicit none
type(Tensor) :: t, r

print '(a)', '=================== Construction ==================='
t = tensor_zeros([2, 3])
print '(a, a)', 'zeros([2,3]): ', t%to_string()
t = tensor_eye(3)
print '(a, a)', 'eye(3): ', t%to_string()

r = tensor_arange(0d0, 12d0, 1d0)
r = r%reshape([3, 4])
print '(a, a)', 'arange(0,12).reshape([3,4]): ', r%to_string()

print '(/, a)', '=================== Shape ops ==================='
t = r%transpose()
print '(a, a)', 'transpose -> shape ', shape_string(t)
t = r%slice(1, 1, 3)
print '(a, a)', 'slice(axis=1, start=1, stop=3): ', t%to_string()

print '(/, a)', '=================== Broadcasting arithmetic ==================='
block
    type(Tensor) :: col, row, total
    col = tensor_from_values([3, 1], [1d0, 2d0, 3d0])
    row = tensor_from_values([1, 4], [10d0, 20d0, 30d0, 40d0])
    total = col%add(row)
    print '(a, a)', '(3,1) + (1,4) -> ', total%to_string()
end block

print '(/, a)', '=================== Reductions ==================='
print '(a, a, a, a)', 'r.sum() = ', num(r%sum()), ', r.mean() = ', num(r%mean())
t = r%mean_axis(0)
print '(a, a)', 'column means (axis=0): ', t%to_string()

print '(/, a)', '=================== Linear algebra ==================='
block
    type(Tensor) :: a, b, product
    a = tensor_from_values([2, 3], [1d0, 2d0, 3d0, 4d0, 5d0, 6d0])
    b = tensor_from_values([3, 2], [7d0, 8d0, 9d0, 10d0, 11d0, 12d0])
    product = a%matmul(b)
    print '(a, a)', 'matmul(2x3, 3x2) -> ', product%to_string()
end block
block
    type(Tensor) :: v1, v2, outer_product
    v1 = tensor_from_values([3], [1d0, 2d0, 3d0])
    v2 = tensor_from_values([3], [4d0, 5d0, 6d0])
    print '(a, a)', 'dot([1,2,3], [4,5,6]) = ', num(v1%dot(v2))
    outer_product = v1%outer(v2)
    print '(a, a)', 'outer(v1, v2) -> ', outer_product%to_string()
end block

print '(/, a)', '=================== Comparisons & masks ==================='
block
    type(Tensor) :: six, mask
    six = tensor_full([3, 4], 6d0)
    mask = r%greater_equal(six)
    print '(a, a)', 'r >= 6 -> ', mask%to_string()
    print '(a, a)', 'count(r >= 6) = ', num(mask%sum())
end block

print '(/, a)', '=================== A real dataset as a Tensor ==================='
block
    character(len=12), parameter :: columns(4) = [character(len=12) :: 'Sepal.Length', 'Sepal.Width', &
        'Petal.Length', 'Petal.Width']
    type(DataFrame) :: frame
    type(Tensor) :: x, feature_means, means_row, centered, centered_t, scatter, species_tensor, setosa, setosa_mask
    real(c_double), allocatable :: flat(:)
    character(len=16), allocatable :: species(:)
    integer :: n, i, j

    frame = iris()
    n = frame%nrows()
    allocate(flat(4 * n), species(n))
    do i = 0, n - 1
        do j = 1, 4
            flat(4 * i + j) = frame%numeric_at(trim(columns(j)), i)
        end do
    end do
    x = tensor_from_values([n, 4], flat)
    print '(a, a)', 'iris feature tensor shape: ', shape_string(x)

    feature_means = x%mean_axis(0)
    means_row = feature_means%reshape([1, 4])
    centered = x%subtract(means_row)
    centered_t = centered%transpose()
    scatter = centered_t%matmul(centered)
    print '(a, a)', 'feature means (Sepal.Length, Sepal.Width, Petal.Length, Petal.Width): ', feature_means%to_string()
    print '(a, a)', '(X-mean)^T (X-mean) [4x4 scatter matrix]: ', scatter%to_string(16)

    do i = 0, n - 1
        species(i + 1) = frame%string_at('Species', i)
    end do
    species_tensor = tensor_from_string_values([n], species)
    setosa = tensor_from_string_values([1], ['setosa'])
    setosa_mask = species_tensor%equal(setosa)
    print '(a, a, a, i0, a)', 'setosa count = ', num(setosa_mask%sum()), ' (of ', n, ' rows)'
end block

contains

function shape_string(x) result(s)
    type(Tensor), intent(in) :: x
    character(len=:), allocatable :: s
    integer, allocatable :: dims(:)
    character(len=24) :: buf
    integer :: i

    dims = x%shape()
    s = '['
    do i = 1, size(dims)
        write (buf, '(i0)') dims(i)
        if (i > 1) s = s // ', '
        s = s // trim(buf)
    end do
    s = s // ']'
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
