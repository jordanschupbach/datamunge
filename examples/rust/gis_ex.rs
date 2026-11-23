// CMD: just TARGET=gis_ex run-rust
// ShapeLayer: write a tiny synthetic shapefile (.shp + .dbf) by hand, read it back and draw it as a map.
// Real .shp/.dbf files are large binary bundles that don't belong in this repo, so this example
// writes two "counties" (one plain square, one with a lake-shaped hole) using the same ESRI byte
// layout ShapeLayer::read() expects, then reads them back through the public API.
use datamunge::*;

const BASE_PATH: &str = "datamunge_gis_ex_counties_rust";

type Ring = Vec<(f64, f64)>;

fn main() -> datamunge::Result<()> {
    // Per the ESRI winding convention, outer rings are clockwise, holes counterclockwise.
    let plain: Vec<Ring> = vec![vec![(10.0, 0.0), (10.0, 10.0), (20.0, 10.0), (20.0, 0.0), (10.0, 0.0)]];
    let with_lake: Vec<Ring> = vec![
        vec![(0.0, 0.0), (0.0, 10.0), (10.0, 10.0), (10.0, 0.0), (0.0, 0.0)],
        vec![(3.0, 3.0), (4.0, 3.0), (4.0, 4.0), (3.0, 4.0), (3.0, 3.0)],
    ];

    write_counties_shp(&format!("{BASE_PATH}.shp"), &[with_lake, plain]).expect("write .shp");
    write_counties_dbf(&format!("{BASE_PATH}.dbf"), &["Lakeside", "Plainview"], &["48231", "19876"]).expect("write .dbf");

    let counties = ShapeLayer::read(BASE_PATH)?.expect("shape layer");

    println!("shapes = {}, shape_type = {}", counties.size()?, counties.shape_type()?);
    // Roughly C's %g: integral doubles print without a fractional part.
    let fmt_g = |x: &f64| if *x == x.floor() { format!("{}", *x as i64) } else { format!("{x:.6}") };
    println!("bounds = [{}]\n", counties.bounds()?.iter().map(fmt_g).collect::<Vec<_>>().join(", "));

    let attrs = counties.attributes()?.expect("attributes");
    println!("attributes\n{attrs}\n");

    for i in 0..counties.size()? {
        println!("{}: {} ring(s)", attrs.string_at("NAME", i)?, counties.num_parts(i)?);
    }

    const SVG_PATH: &str = "datamunge_gis_ex_map_rust.svg";
    counties.plot()?.save_svg(SVG_PATH)?;
    println!("\nmap saved to {SVG_PATH}");
    Ok(())
}

/// Bounding box (minx, miny, maxx, maxy) of every point in `rings`.
fn ring_bounds<'a>(rings: impl IntoIterator<Item = &'a Ring>) -> [f64; 4] {
    rings.into_iter().flatten().fold(
        [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY],
        |[x0, y0, x1, y1], &(x, y)| [x0.min(x), y0.min(y), x1.max(x), y1.max(y)],
    )
}

/// Bytes of one Polygon record's contents (without the record header).
fn polygon_record(rings: &[Ring]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend(5i32.to_le_bytes()); // shape type: Polygon
    ring_bounds(rings).iter().for_each(|b| out.extend(b.to_le_bytes()));
    out.extend((rings.len() as i32).to_le_bytes());
    out.extend((rings.iter().map(Vec::len).sum::<usize>() as i32).to_le_bytes());
    let mut start = 0;
    for ring in rings {
        out.extend((start as i32).to_le_bytes());
        start += ring.len();
    }
    for &(x, y) in rings.iter().flatten() {
        out.extend(x.to_le_bytes());
        out.extend(y.to_le_bytes());
    }
    out
}

fn write_counties_shp(path: &str, shapes: &[Vec<Ring>]) -> std::io::Result<()> {
    let contents: Vec<Vec<u8>> = shapes.iter().map(|s| polygon_record(s)).collect();
    let total_words: usize = contents.iter().map(|c| 4 + c.len() / 2).sum();
    let mut f = Vec::new();
    f.extend(9994i32.to_be_bytes());
    (0..5).for_each(|_| f.extend(0i32.to_be_bytes()));
    f.extend((50 + total_words as i32).to_be_bytes());
    f.extend(1000i32.to_le_bytes());
    f.extend(5i32.to_le_bytes()); // Polygon
    ring_bounds(shapes.iter().flatten()).iter().for_each(|b| f.extend(b.to_le_bytes()));
    (0..4).for_each(|_| f.extend(0.0f64.to_le_bytes()));
    for (i, content) in contents.iter().enumerate() {
        f.extend((i as i32 + 1).to_be_bytes());
        f.extend((content.len() as i32 / 2).to_be_bytes());
        f.extend(content);
    }
    std::fs::write(path, f)
}

/// `s` cut or padded with `ch` to exactly `n` bytes.
fn ljust(s: &str, n: usize, ch: u8) -> Vec<u8> {
    let mut bytes = s.as_bytes().to_vec();
    bytes.resize(n, ch);
    bytes
}

fn write_counties_dbf(path: &str, names: &[&str], populations: &[&str]) -> std::io::Result<()> {
    const HEADER_SIZE: i16 = 32 + 2 * 32 + 1;
    const RECORD_SIZE: i16 = 1 + 12 + 8;
    let mut f = vec![0x03, 0, 0, 0];
    f.extend((names.len() as i32).to_le_bytes());
    f.extend(HEADER_SIZE.to_le_bytes());
    f.extend(RECORD_SIZE.to_le_bytes());
    f.extend([0; 20]);
    let write_field = |f: &mut Vec<u8>, name: &str, field_type: &str, length: u8| {
        f.extend(ljust(name, 11, 0));
        f.extend(field_type.as_bytes());
        f.extend([0; 4]);
        f.push(length);
        f.extend([0; 15]);
    };
    write_field(&mut f, "NAME", "C", 12);
    write_field(&mut f, "POP", "N", 8);
    f.push(0x0D);
    for (name, pop) in names.iter().zip(populations) {
        f.push(b' ');
        f.extend(ljust(name, 12, b' '));
        f.extend(ljust(pop, 8, b' '));
    }
    std::fs::write(path, f)
}
