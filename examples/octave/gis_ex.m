% CMD: just TARGET=gis_ex run-octave
% ShapeLayer: write a tiny synthetic shapefile (.shp + .dbf) by hand, read it back and draw it as a map.
% Real .shp/.dbf files are large binary bundles that don't belong in this repo, so this example
% writes two "counties" (one plain square, one with a lake-shaped hole) using the same ESRI byte
% layout ShapeLayer.read() expects, then reads them back through the public API.
1;

% Little- and big-endian packers returning uint8 row vectors.
function b = pack_i4_le (v)
  b = typecast (int32 (v), 'uint8');
  if (strcmp (computer_endian (), 'big'))
    b = fliplr (b);
  end
end

function b = pack_i4_be (v)
  b = fliplr (pack_i4_le (v));
end

function b = pack_i2_le (v)
  b = typecast (int16 (v), 'uint8');
  if (strcmp (computer_endian (), 'big'))
    b = fliplr (b);
  end
end

function b = pack_double_le (x)
  b = [];
  for v = x(:)'
    bytes = typecast (double (v), 'uint8');
    if (strcmp (computer_endian (), 'big'))
      bytes = fliplr (bytes);
    end
    b = [b, bytes];
  end
end

function e = computer_endian ()
  [~, ~, e] = computer ();
  if (e == 'L')
    e = 'little';
  else
    e = 'big';
  end
end

% Pad or truncate s to n characters with ch, as uint8 bytes.
function b = ljust (s, n, ch)
  b = uint8 (repmat (ch, 1, n));
  m = min (numel (s), n);
  b(1:m) = uint8 (s(1:m));
end

% Bounding box (minx, miny, maxx, maxy) of every point in a cell array of N x 2 rings.
function b = ring_bounds (rings)
  points = vertcat (rings{:});
  b = [min(points(:, 1)), min(points(:, 2)), max(points(:, 1)), max(points(:, 2))];
end

% Bytes of one Polygon record's contents (without the record header).
function out = polygon_record (rings)
  out = [pack_i4_le(5), pack_double_le(ring_bounds (rings))];  % shape type: Polygon
  lengths = cellfun (@rows, rings);
  out = [out, pack_i4_le(numel (rings)), pack_i4_le(sum (lengths))];
  starts = [0, cumsum(lengths(1:end - 1))];
  for s = starts
    out = [out, pack_i4_le(s)];
  end
  points = vertcat (rings{:})';
  out = [out, pack_double_le(points(:))];
end

function write_bytes (path, bytes)
  fid = fopen (path, 'wb');
  fwrite (fid, bytes, 'uint8');
  fclose (fid);
end

function write_counties_shp (path, shapes)
  contents = cellfun (@polygon_record, shapes, 'UniformOutput', false);
  total_words = sum (cellfun (@(c) 4 + numel (c) / 2, contents));
  f = [pack_i4_be(9994), repmat(pack_i4_be(0), 1, 5), pack_i4_be(50 + total_words)];
  f = [f, pack_i4_le(1000), pack_i4_le(5)];  % Polygon
  all_rings = [shapes{:}];
  f = [f, pack_double_le(ring_bounds (all_rings)), pack_double_le(zeros (1, 4))];
  for i = 1:numel (contents)
    f = [f, pack_i4_be(i), pack_i4_be(numel (contents{i}) / 2), contents{i}];
  end
  write_bytes (path, f);
end

function write_counties_dbf (path, names, populations)
  header_size = 32 + 2 * 32 + 1;
  record_size = 1 + 12 + 8;
  f = [uint8([3 0 0 0]), pack_i4_le(numel (names)), pack_i2_le(header_size), pack_i2_le(record_size), zeros(1, 20, 'uint8')];
  field = @(name, type, len) [ljust(name, 11, char (0)), uint8(type), zeros(1, 4, 'uint8'), uint8(len), zeros(1, 15, 'uint8')];
  f = [f, field('NAME', 'C', 12), field('POP', 'N', 8), uint8(13)];
  for i = 1:numel (names)
    f = [f, uint8(' '), ljust(names{i}, 12, ' '), ljust(populations{i}, 8, ' ')];
  end
  write_bytes (path, f);
end

base_path = 'datamunge_gis_ex_counties_octave';

% Per the ESRI winding convention, outer rings are clockwise, holes counterclockwise.
plain = {[10 0; 10 10; 20 10; 20 0; 10 0]};
with_lake = {[0 0; 0 10; 10 10; 10 0; 0 0], [3 3; 4 3; 4 4; 3 4; 3 3]};

write_counties_shp ([base_path '.shp'], {with_lake, plain});
write_counties_dbf ([base_path '.dbf'], {'Lakeside', 'Plainview'}, {'48231', '19876'});

counties = datamunge.ShapeLayer.read (base_path);

printf ('shapes = %d, shape_type = %s\n', counties.size_ (), counties.shape_type ());
% %g: integral doubles print without a fractional part.
printf ('bounds = [%s]\n\n', strjoin (arrayfun (@(x) sprintf ('%g', x), counties.bounds (), 'UniformOutput', false), ', '));

attrs = counties.attributes ();
printf ('attributes\n%s\n\n', attrs.to_string ());

for i = 0:counties.size_ () - 1
  printf ('%s: %d ring(s)\n', attrs.string_at ('NAME', i), counties.num_parts (i));
end

svg_path = 'datamunge_gis_ex_map_octave.svg';
counties.plot ().save_svg (svg_path);
printf ('\nmap saved to %s\n', svg_path);
