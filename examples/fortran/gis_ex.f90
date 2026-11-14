! CMD: just TARGET=gis_ex run-fortran

! ShapeLayer: write a tiny synthetic shapefile (.shp + .dbf) by hand, read it back and draw it as a map.
! Real .shp/.dbf files are large binary bundles that don't belong in this repo, so this example
! writes two "counties" (one plain square, one with a lake-shaped hole) using the same ESRI byte
! layout ShapeLayer.read() expects, then reads them back through the public API.
program gis_ex
use datamunge
use, intrinsic :: iso_fortran_env, only: int16, int32
implicit none
character(len=*), parameter :: base_path = 'datamunge_gis_ex_counties_fortran'
character(len=*), parameter :: svg_path = 'datamunge_gis_ex_map_fortran.svg'
! Per the ESRI winding convention, outer rings are clockwise, holes counterclockwise.
! Each shape is its points (all rings concatenated) plus the 0-based start index of every ring.
real(c_double), parameter :: lake_x(10) = [0d0, 0d0, 10d0, 10d0, 0d0, 3d0, 4d0, 4d0, 3d0, 3d0]
real(c_double), parameter :: lake_y(10) = [0d0, 10d0, 10d0, 0d0, 0d0, 3d0, 3d0, 4d0, 4d0, 3d0]
integer, parameter :: lake_parts(2) = [0, 5]
real(c_double), parameter :: plain_x(5) = [10d0, 10d0, 20d0, 20d0, 10d0]
real(c_double), parameter :: plain_y(5) = [0d0, 10d0, 10d0, 0d0, 0d0]
integer, parameter :: plain_parts(1) = [0]
type(ShapeLayer) :: counties
type(DataFrame) :: attrs
type(RPlot) :: fig
real(c_double), allocatable :: bb(:)
character(len=:), allocatable :: line
integer :: i

call write_counties_shp(base_path // '.shp')
call write_counties_dbf(base_path // '.dbf', [character(len=9) :: 'Lakeside', 'Plainview'], &
    [character(len=5) :: '48231', '19876'])

counties = shapelayer_read(base_path)

print '(a, i0, a)', 'shapes = ', counties%size(), ', shape_type = ' // counties%shape_type()
! Roughly C's %g: integral doubles print without a fractional part.
bb = counties%bounds()
line = 'bounds = ['
do i = 1, size(bb)
    if (i > 1) line = line // ', '
    line = line // fmt_g(bb(i))
end do
print '(a)', line // ']' // new_line('a')

attrs = counties%attributes()
print '(a)', 'attributes' // new_line('a') // attrs%to_string() // new_line('a')

do i = 0, counties%size() - 1
    print '(a, i0, a)', attrs%string_at('NAME', i) // ': ', counties%num_parts(i), ' ring(s)'
end do

fig = counties%plot()
call fig%save_svg(svg_path)
print '(a)', new_line('a') // 'map saved to ' // svg_path

contains

function fmt_g(x) result(s)
    real(c_double), intent(in) :: x
    character(len=:), allocatable :: s
    character(len=32) :: buf
    if (x == aint(x)) then
        write(buf, '(i0)') nint(x)
    else
        write(buf, '(f0.6)') x
    end if
    s = trim(buf)
end function

! Native byte order is checked once so the packers work on either endianness.
logical function little_endian()
    integer(int32), parameter :: one = 1
    character(len=4) :: bytes
    bytes = transfer(one, bytes)
    little_endian = ichar(bytes(1:1)) == 1
end function

function reversed(s) result(r)
    character(len=*), intent(in) :: s
    character(len=len(s)) :: r
    integer :: i
    do i = 1, len(s)
        r(i:i) = s(len(s) - i + 1:len(s) - i + 1)
    end do
end function

! Little- and big-endian packers that append to a byte buffer (a character string).
subroutine pack_i4_le(buf, value)
    character(len=:), allocatable, intent(inout) :: buf
    integer, intent(in) :: value
    character(len=4) :: bytes
    bytes = transfer(int(value, int32), bytes)
    if (.not. little_endian()) bytes = reversed(bytes)
    buf = buf // bytes
end subroutine

subroutine pack_i4_be(buf, value)
    character(len=:), allocatable, intent(inout) :: buf
    integer, intent(in) :: value
    character(len=4) :: bytes
    bytes = transfer(int(value, int32), bytes)
    if (little_endian()) bytes = reversed(bytes)
    buf = buf // bytes
end subroutine

subroutine pack_i2_le(buf, value)
    character(len=:), allocatable, intent(inout) :: buf
    integer, intent(in) :: value
    character(len=2) :: bytes
    bytes = transfer(int(value, int16), bytes)
    if (.not. little_endian()) bytes = reversed(bytes)
    buf = buf // bytes
end subroutine

subroutine pack_double_le(buf, x)
    character(len=:), allocatable, intent(inout) :: buf
    real(c_double), intent(in) :: x
    character(len=8) :: bytes
    bytes = transfer(x, bytes)
    if (.not. little_endian()) bytes = reversed(bytes)
    buf = buf // bytes
end subroutine

subroutine pack_zeros(buf, n)
    character(len=:), allocatable, intent(inout) :: buf
    integer, intent(in) :: n
    buf = buf // repeat(achar(0), n)
end subroutine

! Bounding box (minx, miny, maxx, maxy) of every point.
function ring_bounds(xs, ys) result(b)
    real(c_double), intent(in) :: xs(:), ys(:)
    real(c_double) :: b(4)
    b = [minval(xs), minval(ys), maxval(xs), maxval(ys)]
end function

! Bytes of one Polygon record's contents (without the record header).
function polygon_record(xs, ys, parts) result(out)
    real(c_double), intent(in) :: xs(:), ys(:)
    integer, intent(in) :: parts(:)
    character(len=:), allocatable :: out
    real(c_double) :: b(4)
    integer :: i
    out = ''
    call pack_i4_le(out, 5)  ! shape type: Polygon
    b = ring_bounds(xs, ys)
    do i = 1, 4
        call pack_double_le(out, b(i))
    end do
    call pack_i4_le(out, size(parts))
    call pack_i4_le(out, size(xs))
    do i = 1, size(parts)
        call pack_i4_le(out, parts(i))
    end do
    do i = 1, size(xs)
        call pack_double_le(out, xs(i))
        call pack_double_le(out, ys(i))
    end do
end function

subroutine write_bytes(path, bytes)
    character(len=*), intent(in) :: path, bytes
    integer :: unit
    open(newunit=unit, file=path, access='stream', form='unformatted', status='replace', action='write')
    write(unit) bytes
    close(unit)
end subroutine

subroutine write_counties_shp(path)
    character(len=*), intent(in) :: path
    character(len=:), allocatable :: f, lake, plain
    real(c_double) :: b(4)
    integer :: i, total_words
    lake = polygon_record(lake_x, lake_y, lake_parts)
    plain = polygon_record(plain_x, plain_y, plain_parts)
    total_words = (4 + len(lake) / 2) + (4 + len(plain) / 2)
    f = ''
    call pack_i4_be(f, 9994)
    do i = 1, 5
        call pack_i4_be(f, 0)
    end do
    call pack_i4_be(f, 50 + total_words)
    call pack_i4_le(f, 1000)
    call pack_i4_le(f, 5)  ! Polygon
    b = ring_bounds([lake_x, plain_x], [lake_y, plain_y])
    do i = 1, 4
        call pack_double_le(f, b(i))
    end do
    do i = 1, 4
        call pack_double_le(f, 0d0)
    end do
    call pack_i4_be(f, 1)
    call pack_i4_be(f, len(lake) / 2)
    f = f // lake
    call pack_i4_be(f, 2)
    call pack_i4_be(f, len(plain) / 2)
    f = f // plain
    call write_bytes(path, f)
end subroutine

function ljust(s, n, ch) result(r)
    character(len=*), intent(in) :: s, ch
    integer, intent(in) :: n
    character(len=n) :: r
    if (len(s) >= n) then
        r = s(1:n)
    else
        r = s // repeat(ch, n - len(s))
    end if
end function

subroutine write_counties_dbf(path, names, populations)
    character(len=*), intent(in) :: path, names(:), populations(:)
    integer, parameter :: header_size = 32 + 2 * 32 + 1
    integer, parameter :: record_size = 1 + 12 + 8
    character(len=:), allocatable :: f
    integer :: i
    f = achar(3) // repeat(achar(0), 3)
    call pack_i4_le(f, size(names))
    call pack_i2_le(f, header_size)
    call pack_i2_le(f, record_size)
    call pack_zeros(f, 20)
    call write_field(f, 'NAME', 'C', 12)
    call write_field(f, 'POP', 'N', 8)
    f = f // achar(13)
    do i = 1, size(names)
        f = f // ' ' // ljust(trim(names(i)), 12, ' ') // ljust(trim(populations(i)), 8, ' ')
    end do
    call write_bytes(path, f)
end subroutine

! One 32-byte DBF field descriptor.
subroutine write_field(f, name, field_type, length)
    character(len=:), allocatable, intent(inout) :: f
    character(len=*), intent(in) :: name, field_type
    integer, intent(in) :: length
    f = f // ljust(name, 11, achar(0)) // field_type
    call pack_zeros(f, 4)
    f = f // achar(length)
    call pack_zeros(f, 15)
end subroutine
end program
