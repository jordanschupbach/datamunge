! CMD: just TARGET=clustering_ex run-fortran

! Clustering iris: k-means, agglomerative (several linkages, dendrogram re-cut) and DBSCAN.
program clustering_ex
use datamunge
use datamunge_data, only: iris
implicit none
character(len=12), parameter :: feature_columns(4) = [character(len=12) :: 'Sepal.Length', 'Sepal.Width', &
    'Petal.Length', 'Petal.Width']
character(len=8), parameter :: linkages(3) = [character(len=8) :: 'ward', 'complete', 'average']
type(DataFrame) :: frame
type(KMeans) :: kmeans_model
type(AgglomerativeClustering) :: model, ward_model
type(DBSCAN) :: dbscan_model
type(rosetta_string), allocatable :: species(:)
integer :: i

frame = iris()
allocate(species(frame%nrows()))
do i = 1, size(species)
    species(i)%s = frame%string_at('Species', i - 1)
end do

print '(a)', '=================== K-means (k=3) on iris ==================='
kmeans_model = KMeans(frame, feature_columns, 3)
print '(a)', 'Inertia: ' // num(kmeans_model%inertia()) // ', iterations (best run): ' // &
    itoa(kmeans_model%iterations_used())
call report_purity(kmeans_model%labels(), 3)

print '(/, a)', '=================== Agglomerative clustering on iris ==================='
do i = 1, size(linkages)
    model = AgglomerativeClustering(frame, feature_columns, 3, trim(linkages(i)))
    print '(a)', '--- linkage=' // trim(linkages(i)) // ' ---'
    call report_purity(model%labels(), 3)
end do

print '(/, a)', '--- Re-cutting the ward dendrogram at k=2 without refitting ---'
ward_model = AgglomerativeClustering(frame, feature_columns, 3, 'ward')
call report_purity(ward_model%cut(2), 2)

print '(/, a)', '=================== DBSCAN on iris ==================='
dbscan_model = DBSCAN(frame, feature_columns, 0.6d0, 5)
print '(a)', 'Clusters found: ' // itoa(dbscan_model%n_clusters()) // ', noise points: ' // &
    itoa(dbscan_model%n_noise()) // ' (of ' // itoa(dbscan_model%observations()) // ')'
call report_purity(dbscan_model%labels(), dbscan_model%n_clusters())

contains

! For each cluster, print its size and its majority species (negative labels are noise).
subroutine report_purity(cluster_labels, n_clusters)
    integer, intent(in) :: cluster_labels(:)
    integer, intent(in) :: n_clusters
    type(rosetta_string), allocatable :: names(:)
    integer, allocatable :: votes(:, :)
    integer :: c, k, j, total, best
    ! The distinct species, in order of first appearance.
    allocate(names(0))
    do j = 1, size(species)
        if (.not. any([(names(k)%s == species(j)%s, k = 1, size(names))])) names = [names, species(j)]
    end do
    allocate(votes(size(names), 0:n_clusters - 1), source=0)
    do j = 1, size(cluster_labels)
        if (cluster_labels(j) < 0) cycle
        do k = 1, size(names)
            if (names(k)%s == species(j)%s) votes(k, cluster_labels(j)) = votes(k, cluster_labels(j)) + 1
        end do
    end do
    do c = 0, n_clusters - 1
        total = sum(votes(:, c))
        if (total == 0) cycle
        best = maxloc(votes(:, c), 1)
        print '(a)', '  cluster ' // itoa(c) // ': ' // itoa(total) // ' points, majority ' // names(best)%s // &
            ' (' // itoa(votes(best, c)) // '/' // itoa(total) // ')'
    end do
end subroutine

function itoa(i) result(s)
    integer, intent(in) :: i
    character(len=:), allocatable :: s
    character(len=24) :: buf
    write(buf, '(i0)') i
    s = trim(buf)
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
