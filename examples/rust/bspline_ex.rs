// CMD: just TARGET=bspline_ex run-rust
// Tensor-product B-spline regression: fit z ~ bs(x, y) to a smooth surface and predict.
use datamunge::*;

fn main() -> datamunge::Result<()> {
    // Sample a smooth surface z = sin(x) + y^2 on an 8x8 grid over [0, 1]^2.
    let (mut x, mut y, mut z) = (Vec::new(), Vec::new(), Vec::new());
    for iy in 0..8 {
        for ix in 0..8 {
            let (xv, yv) = (ix as f64 / 7.0, iy as f64 / 7.0);
            x.push(xv);
            y.push(yv);
            z.push(xv.sin() + yv * yv);
        }
    }
    let data = DataFrame::new()?;
    data.add_numeric_column("x", &x)?;
    data.add_numeric_column("y", &y)?;
    data.add_numeric_column("z", &z)?;

    let surface = LM::new(&data, "z ~ bs(x, y)")?;
    println!("Fitted z ~ bs(x, y)");
    surface.print_summary()?;

    let new_points = DataFrame::new()?;
    new_points.add_numeric_column("x", &[0.25, 0.75])?;
    new_points.add_numeric_column("y", &[0.50, 0.25])?;
    let pred = surface.predict(&new_points)?;
    println!("Predictions: [{:.6}, {:.6}]", pred[0], pred[1]);
    Ok(())
}
