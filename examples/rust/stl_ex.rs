// CMD: just TARGET=stl_ex run-rust
// C++ std::vector and std::pair values in Rust: vectors are Vec<T>, pairs are tuples.
use datamunge::*;
use datamunge_rust_examples::N;

fn main() -> datamunge::Result<()> {
    const N_VALUES: usize = 100;
    let v: Vec<f64> = (0..N_VALUES).map(|i| i as f64 * 1.5).collect();
    v.iter().for_each(|&x| println!("{}", N(x)));
    let v2: Vec<i32> = (0..N_VALUES).map(|i| (i as f64 * 1.5).floor() as i32).collect();
    v2.iter().for_each(|x| println!("{x}"));

    let p = (3, 4);
    println!("p: ({}, {})", p.0, p.1);

    let p2 = (10, 20);
    println!("p2: ({}, {})", p2.0, p2.1);

    // A C++ std::pair<double,double> comes back as a tuple.
    let p3 = make_dpair(1.25, 2.75)?;
    println!("make_dpair(1.25, 2.75): ({}, {}), sum_dpair = {}", N(p3.0), N(p3.1), N(sum_dpair(p3)?));
    Ok(())
}
