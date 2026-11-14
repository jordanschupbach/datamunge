! CMD: just TARGET=mds_ex run-fortran

! Classical multidimensional scaling of iris: euclidean embedding, plot, and a manhattan comparison.
program mds_ex
use datamunge
use datamunge_data
implicit none
character(len=12), parameter :: feature_columns(4) = &
    [character(len=12) :: 'Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']
type(DataFrame) :: frame, embedding
type(MDS) :: mds_euclidean, mds_manhattan
type(RPlot) :: plt
character(len=32), allocatable :: species(:)
integer :: i

frame = iris()
allocate(species(frame%nrows()))
do i = 1, size(species)
    species(i) = frame%string_at('Species', i - 1)
end do

print '(a)', '=================== Classical MDS on iris (euclidean) ==================='
mds_euclidean = MDS(frame, feature_columns, 2, 'euclidean')
call mds_euclidean%print_summary()

embedding = mds_euclidean%embedding_frame()
print '(a)', new_line('a') // 'Embedding as a DataFrame:' // new_line('a') // embedding%to_string()

plt = mds_euclidean%plot_embedding_grouped(species)
call plt%save_svg('mds_iris_embedding.svg')
print '(a)', new_line('a') // 'Embedding scatter (colored by species) saved as mds_iris_embedding.svg'

print '(a)', new_line('a') // '=================== Classical MDS on iris (manhattan) ==================='
mds_manhattan = MDS(frame, feature_columns, 2, 'manhattan')
print '(a, a, a, a)', 'Goodness of fit: euclidean=', fix4(mds_euclidean%goodness_of_fit()), &
    ', manhattan=', fix4(mds_manhattan%goodness_of_fit())
contains
! x with 4 decimals and a leading zero (like C# {x:F4} / %.4f).
function fix4(x) result(s)
    real(c_double), intent(in) :: x
    character(len=:), allocatable :: s
    character(len=32) :: buf
    write(buf, '(f32.4)') x
    s = trim(adjustl(buf))
end function
end program
