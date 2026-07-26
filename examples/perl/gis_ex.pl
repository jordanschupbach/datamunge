use strict;
use warnings;

use List::Util qw(min max);

use Datamunge;

# Demonstrates Datamunge::ShapeLayer: reading a shapefile (.shp geometry + .dbf attributes)
# and drawing it as a map. Real .shp/.dbf files are large binary bundles that don't belong in
# this repo, so this example first writes a tiny synthetic shapefile by hand (two "counties":
# one plain square, one with a lake-shaped hole) using the same ESRI byte layout
# ShapeLayer.read() expects, then reads it back through the public API. Perl's pack() mirrors
# Python's struct.pack for the binary layout ("l"/"L" = 32-bit int, "S" = 16-bit, "d" = double;
# "<"/">" select little-/big-endian).

sub polygon_record {
  my ($rings) = @_;
  my @all_points = map { @$_ } @$rings;
  my @xs = map { $_->[0] } @all_points;
  my @ys = map { $_->[1] } @all_points;

  my $out = pack("l<", 5);    # shape type: Polygon
  $out .= pack("d<4", min(@xs), min(@ys), max(@xs), max(@ys));
  $out .= pack("l<", scalar(@$rings));
  $out .= pack("l<", scalar(@all_points));
  my $start = 0;
  for my $ring (@$rings) {
    $out .= pack("l<", $start);
    $start += scalar(@$ring);
  }
  for my $ring (@$rings) {
    for my $p (@$ring) {
      $out .= pack("d<2", $p->[0], $p->[1]);
    }
  }
  return $out;
}

sub write_counties_shp {
  my ($path, $shapes) = @_;
  my @contents = map { polygon_record($_) } @$shapes;
  my $total_words = 0;
  $total_words += 4 + int(length($_) / 2) for @contents;

  my @all_points = map { @$_ } map { @$_ } @$shapes;
  my @xs = map { $_->[0] } @all_points;
  my @ys = map { $_->[1] } @all_points;

  open(my $fh, ">:raw", $path) or die "cannot open $path: $!";
  print $fh pack("l>", 9994);
  print $fh pack("l>5", 0, 0, 0, 0, 0);
  print $fh pack("l>", 50 + $total_words);
  print $fh pack("l<", 1000);
  print $fh pack("l<", 5);    # Polygon
  print $fh pack("d<4", min(@xs), min(@ys), max(@xs), max(@ys));
  print $fh pack("d<4", 0.0, 0.0, 0.0, 0.0);
  my $i = 1;
  for my $content (@contents) {
    print $fh pack("l>", $i);
    print $fh pack("l>", int(length($content) / 2));
    print $fh $content;
    $i++;
  }
  close($fh);
}

sub write_counties_dbf {
  my ($path, $names, $populations) = @_;
  my $header_size = 32 + 2 * 32 + 1;
  my $record_size = 1 + 12 + 8;

  open(my $fh, ">:raw", $path) or die "cannot open $path: $!";
  print $fh pack("C4", 0x03, 0, 0, 0);
  print $fh pack("L<", scalar(@$names));
  print $fh pack("S<", $header_size);
  print $fh pack("S<", $record_size);
  print $fh "\0" x 20;

  my $write_field = sub {
    my ($name, $field_type, $length) = @_;
    print $fh pack("a11", $name);    # ljust(11, "\0")
    print $fh $field_type;
    print $fh "\0" x 4;
    print $fh chr($length);
    print $fh "\0" x 15;
  };

  $write_field->("NAME", "C", 12);
  $write_field->("POP", "N", 8);
  print $fh chr(0x0D);

  for (my $k = 0; $k < scalar(@$names); $k++) {
    print $fh " ";
    print $fh pack("A12", substr($names->[$k], 0, 12));         # ljust(12, " ")
    print $fh pack("A8", substr($populations->[$k], 0, 8));     # ljust(8, " ")
  }
  close($fh);
}

my $base = "datamunge_gis_ex_counties_perl";

# Per the ESRI winding convention, outer rings are clockwise, holes counterclockwise.
my $plain = [[[10, 0], [10, 10], [20, 10], [20, 0], [10, 0]]];
my $with_lake = [
  [[0, 0], [0, 10], [10, 10], [10, 0], [0, 0]],
  [[3, 3], [4, 3], [4, 4], [3, 4], [3, 3]],
];

write_counties_shp($base . ".shp", [$with_lake, $plain]);
write_counties_dbf($base . ".dbf", ["Lakeside", "Plainview"], ["48231", "19876"]);

my $counties = Datamunge::ShapeLayer::read($base);

print "shapes = " . $counties->size() . ", shape_type = " . $counties->shape_type() . "\n";
my $bounds = $counties->bounds();
print "bounds = [" . join(", ", map { sprintf("%g", $_) } @$bounds) . "]\n";
print "\n";

my $attributes = $counties->attributes();
print "attributes\n";
print $attributes->to_string(), "\n";
print "\n";

for (my $i = 0; $i < $counties->size(); $i++) {
  my $name = $attributes->string_at("NAME", $i);
  print "$name: " . $counties->num_parts($i) . " ring(s)\n";
}

my $svg_path = "datamunge_gis_ex_map_perl.svg";
$counties->plot()->save_svg($svg_path);
print "\nmap saved to $svg_path\n";
