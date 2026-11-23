// CMD: just TARGET=datamunge_ex run-rust
// Smoke test of the basic bindings: vectors, pairs and a native Callback object.
use datamunge::*;
use datamunge_rust_examples::N;

fn main() -> datamunge::Result<()> {
    hello()?;

    let v = make_dvector(1.0, 2.0, 3.0)?;
    println!("sum_dvector:\t{}", N(sum_dvector(&v)?));

    let p = make_dpair(1.25, 2.75)?;
    println!("sum_dpair:\t{}", N(sum_dpair(p)?));

    let cb = Callback::new()?;
    println!("call_with_callback(3.0):\t{}", N(call_with_callback(3.0, Some(&cb))?));
    let v2 = map_dvector_with_callback(&make_dvector(1.0, 2.0, 3.0)?, Some(&cb))?;
    println!("sum_dvector(map_dvector_with_callback(1,2,3)):\t{}", N(sum_dvector(&v2)?));
    Ok(())
}
