! CMD: just TARGET=pca_ex run-fortran

! PCA on iris: summary, PC1 loadings, scores frame, score/scree plots, scaled vs unscaled.
program pca_ex
use datamunge
use datamunge_data, only: iris
implicit none
character(len=12), parameter :: feature_columns(4) = &
    [character(len=12) :: 'Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']
type(DataFrame) :: frame, scores
type(PCA) :: scaled, unscaled
type(RPlot) :: fig
character(len=32), allocatable :: species(:)
type(rosetta_string), allocatable :: names(:)
real(c_double), allocatable :: loadings(:), ratio_scaled(:), ratio_unscaled(:)
character(len=12) :: buf
integer :: i

frame = iris()

! species labels, for the grouped (colored-by-species) scores plot
allocate(species(frame%nrows()))
do i = 1, size(species)
    species(i) = frame%string_at('Species', i - 1)
end do

print '(a)', '=================== PCA on iris (scaled) ==================='
scaled = PCA(frame, feature_columns, .true., .true.)
call scaled%print_summary()

print '(a)', new_line('a') // 'PC1 loadings (which original features drive it):'
names = scaled%feature_names()
loadings = scaled%component_loadings(0)
do i = 1, min(size(names), size(loadings))
    write(buf, '(f12.4)') loadings(i)
    print '(a)', '  ' // names(i)%s // ': ' // trim(adjustl(buf))
end do

scores = scaled%scores_frame()
print '(a)', new_line('a') // 'Training scores as a DataFrame:' // new_line('a') // scores%to_string()

fig = scaled%plot_scores_grouped(species)
call fig%save_svg('pca_iris_scores.svg')
print '(a)', new_line('a') // 'Scores scatter (colored by species) saved as pca_iris_scores.svg'

fig = scaled%plot_scree()
call fig%save_svg('pca_iris_scree.svg')
print '(a)', 'Scree plot saved as pca_iris_scree.svg'

print '(a)', new_line('a') // '=================== PCA on iris (unscaled) ==================='
unscaled = PCA(frame, feature_columns, .true., .false.)
ratio_unscaled = unscaled%explained_variance_ratio()
ratio_scaled = scaled%explained_variance_ratio()
print '(a, f0.2, a, f0.2, a)', 'PC1 explains ', 100 * ratio_unscaled(1), '% of variance (vs. ', &
    100 * ratio_scaled(1), '% scaled) -- Sepal.Length''s larger raw variance dominates the unscaled covariance matrix.'
end program
