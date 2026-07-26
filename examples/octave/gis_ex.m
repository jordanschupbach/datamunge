1;

% Demonstrates datamunge.ShapeLayer: reading a shapefile (.shp geometry + .dbf attributes) and
% drawing it as a map. Real .shp/.dbf files are large binary bundles that don't belong in this
% repo, so this example first writes a tiny synthetic shapefile by hand (two "counties": one
% plain square, one with a lake-shaped hole) using the same ESRI byte layout ShapeLayer.read()
% expects, then reads it back through the public API. Octave's fwrite writes a raw uint8 byte
% vector we assemble ourselves; the pack helpers below mirror Python's struct.pack. (This host,
% aarch64-darwin, is little-endian, so typecast(x, 'uint8') already yields little-endian bytes;
% big-endian fields are produced by reversing them.)
datamunge;

function b = i32le(v)
  b = typecast(int32(v), 'uint8');
endfunction

function b = i32be(v)
  b = fliplr(typecast(int32(v), 'uint8'));
endfunction

function b = u32le(v)
  b = typecast(uint32(v), 'uint8');
endfunction

function b = u16le(v)
  b = typecast(uint16(v), 'uint8');
endfunction

function b = dle(v)
  b = typecast(double(v), 'uint8');
endfunction

function b = ljust_bytes(s, n, ch)
  raw = uint8(s);
  if numel(raw) >= n
    b = raw(1:n);
  else
    b = [raw, repmat(uint8(ch), 1, n - numel(raw))];
  end
endfunction

% rings: cell array of Nx2 [x y] point matrices (first ring outer, rest holes).
function out = polygon_record(rings)
  all_points = [];
  for i = 1:numel(rings)
    all_points = [all_points; rings{i}];
  end
  minx = min(all_points(:, 1)); maxx = max(all_points(:, 1));
  miny = min(all_points(:, 2)); maxy = max(all_points(:, 2));
  out = i32le(5); % shape type: Polygon
  out = [out, dle(minx), dle(miny), dle(maxx), dle(maxy)];
  out = [out, i32le(numel(rings))];
  out = [out, i32le(rows(all_points))];
  start = 0;
  for i = 1:numel(rings)
    out = [out, i32le(start)];
    start = start + rows(rings{i});
  end
  for i = 1:numel(rings)
    ring = rings{i};
    for p = 1:rows(ring)
      out = [out, dle(ring(p, 1)), dle(ring(p, 2))];
    end
  end
endfunction

% shapes: cell array, each element a rings-cell (as above).
function write_counties_shp(path, shapes)
  contents = {};
  total_words = 0;
  all_points = [];
  for i = 1:numel(shapes)
    contents{i} = polygon_record(shapes{i});
    total_words = total_words + 4 + numel(contents{i}) / 2;
    rings = shapes{i};
    for j = 1:numel(rings)
      all_points = [all_points; rings{j}];
    end
  end
  minx = min(all_points(:, 1)); maxx = max(all_points(:, 1));
  miny = min(all_points(:, 2)); maxy = max(all_points(:, 2));

  bytes = i32be(9994);
  for k = 1:5
    bytes = [bytes, i32be(0)];
  end
  bytes = [bytes, i32be(50 + total_words)];
  bytes = [bytes, i32le(1000)];
  bytes = [bytes, i32le(5)]; % Polygon
  bytes = [bytes, dle(minx), dle(miny), dle(maxx), dle(maxy)];
  bytes = [bytes, dle(0.0), dle(0.0), dle(0.0), dle(0.0)];
  for i = 1:numel(contents)
    content = contents{i};
    bytes = [bytes, i32be(i)];
    bytes = [bytes, i32be(numel(content) / 2)];
    bytes = [bytes, content];
  end

  fid = fopen(path, "wb");
  fwrite(fid, bytes, "uint8");
  fclose(fid);
endfunction

function write_counties_dbf(path, names, populations)
  header_size = 32 + 2 * 32 + 1;
  record_size = 1 + 12 + 8;

  bytes = uint8([3, 0, 0, 0]);
  bytes = [bytes, u32le(numel(names))];
  bytes = [bytes, u16le(header_size)];
  bytes = [bytes, u16le(record_size)];
  bytes = [bytes, zeros(1, 20, "uint8")];

  % field descriptor: name(11) + type(1) + reserved(4) + length(1) + reserved(15)
  fields = {{"NAME", "C", 12}, {"POP", "N", 8}};
  for i = 1:numel(fields)
    fld = fields{i};
    bytes = [bytes, ljust_bytes(fld{1}, 11, 0)];
    bytes = [bytes, uint8(fld{2})];
    bytes = [bytes, zeros(1, 4, "uint8")];
    bytes = [bytes, uint8(fld{3})];
    bytes = [bytes, zeros(1, 15, "uint8")];
  end
  bytes = [bytes, uint8(13)]; % 0x0D header terminator

  for i = 1:numel(names)
    bytes = [bytes, uint8(32)]; % deletion flag (space = active)
    bytes = [bytes, ljust_bytes(names{i}, 12, 32)];
    bytes = [bytes, ljust_bytes(populations{i}, 8, 32)];
  end

  fid = fopen(path, "wb");
  fwrite(fid, bytes, "uint8");
  fclose(fid);
endfunction

base = "datamunge_gis_ex_counties_octave";

% Per the ESRI winding convention, outer rings are clockwise, holes counterclockwise.
plain = {[10, 0; 10, 10; 20, 10; 20, 0; 10, 0]};
with_lake = { ...
  [0, 0; 0, 10; 10, 10; 10, 0; 0, 0], ...
  [3, 3; 4, 3; 4, 4; 3, 4; 3, 3] ...
};

write_counties_shp([base, ".shp"], {with_lake, plain});
write_counties_dbf([base, ".dbf"], {"Lakeside", "Plainview"}, {"48231", "19876"});

counties = ShapeLayer_read(base);

printf("shapes = %d, shape_type = %s\n", ShapeLayer_size(counties), ShapeLayer_shape_type(counties));
bounds = ShapeLayer_bounds(counties);  % std::vector<double> -> native Octave cell array
bt = {};
for i = 1:numel(bounds)
  bt{end + 1} = sprintf("%g", bounds{i});
end
printf("bounds = [%s]\n\n", strjoin(bt, ", "));

attributes = ShapeLayer_attributes(counties);
printf("attributes\n");
printf("%s\n\n", DataFrame_to_string(attributes));

for i = 0:(ShapeLayer_size(counties) - 1)
  name = DataFrame_string_at(attributes, "NAME", i);
  printf("%s: %d ring(s)\n", name, ShapeLayer_num_parts(counties, i));
end

svg_path = "datamunge_gis_ex_map_octave.svg";
Plot_save_svg(ShapeLayer_plot(counties), svg_path);
printf("\nmap saved to %s\n", svg_path);
