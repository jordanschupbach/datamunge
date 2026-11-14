! CMD: just TARGET=dataframe_ex run-fortran

! DataFrame basics: nulls, drop_duplicates, fill_null, select/sort, group-by, join and summaries.
program dataframe_ex
use datamunge
implicit none
type(DataFrame) :: sales, cleaned, selected, grouped, targets, joined
integer, allocatable :: dims(:)

sales = DataFrame()
call sales%add_string_column('region', [character(len=6) :: 'west', 'west', 'east', 'south', 'south', 'south'])
call sales%add_string_column('product', [character(len=6) :: 'widget', 'widget', 'widget', 'gizmo', 'gizmo', 'gizmo'])
call sales%add_numeric_column('sales', [10d0, 10d0, 14d0, 8d0, 0d0, 11d0], [1, 1, 1, 1, 0, 1])
call sales%add_string_column('quarter', [character(len=2) :: 'Q1', 'Q1', 'Q1', 'Q2', 'Q2', ''], [1, 1, 1, 1, 1, 0])

print '(a)', 'raw data' // new_line('a') // sales%to_string() // new_line('a')

cleaned = sales%drop_duplicates([character(len=7) :: 'region', 'product', 'sales', 'quarter'])
call cleaned%fill_null_string('quarter', 'unknown')
call cleaned%fill_null_numeric('sales', 0d0)
print '(a)', 'after drop_duplicates + fill_null' // new_line('a') // cleaned%to_string() // new_line('a')

! Fortran cannot call a type-bound procedure on a function result: one variable per step.
selected = cleaned%select([character(len=7) :: 'region', 'sales', 'quarter'])
selected = selected%sort_by('sales', .false.)
print '(a)', 'selected + sorted' // new_line('a') // selected%to_string() // new_line('a')

grouped = cleaned%group_by_sum(['region'], ['sales'])
grouped = grouped%sort_by('sales', .false.)
print '(a)', 'group_by_sum(region)' // new_line('a') // grouped%to_string() // new_line('a')

targets = DataFrame()
call targets%add_string_column('region', [character(len=5) :: 'west', 'east', 'south'])
call targets%add_numeric_column('target', [18d0, 12d0, 25d0])
joined = grouped%join(targets, 'region', 'region', 'left')
print '(a)', 'joined with targets' // new_line('a') // joined%to_string() // new_line('a')

dims = cleaned%shape()
print '(a, i0, a, i0, a)', 'shape = (', dims(1), ', ', dims(2), ')'
print '(a, i0)', 'sales count = ', cleaned%numeric_count('sales')
print '(a, i0)', 'sales nulls = ', cleaned%numeric_null_count('sales')
print '(a, f0.1)', 'sales sum = ', cleaned%numeric_sum('sales')
print '(a, f0.1)', 'sales mean = ', cleaned%numeric_mean('sales')
end program
