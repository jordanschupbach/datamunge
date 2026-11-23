//! Smoke tests of the Rosetta-generated Rust binding (just test-rust).
use datamunge::*;

#[test]
fn loads_and_calls() -> Result<()> {
    hello()?;
    let v = make_dvector(1.0, 2.0, 3.0)?;
    assert_eq!(v, [1.0, 2.0, 3.0]);
    assert_eq!(sum_dvector(&v)?, 6.0);
    assert_eq!(sum_dvector(&[0.5, 1.5])?, 2.0);
    let p = make_dpair(4.0, 5.0)?;
    assert_eq!(p, (4.0, 5.0));
    assert_eq!(sum_dpair(p)?, 9.0);
    Ok(())
}

struct Doubler;
impl CallbackImpl for Doubler {
    fn call(&self, _this: &Callback, x: f64) -> Result<f64> {
        Ok(2.0 * x)
    }
}

#[derive(Debug)]
struct FromRust;
impl std::fmt::Display for FromRust {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("from Rust")
    }
}
impl std::error::Error for FromRust {}

struct Failing;
impl CallbackImpl for Failing {
    fn call(&self, _this: &Callback, _x: f64) -> Result<f64> {
        Err(Error::from_source(FromRust))
    }
}

#[test]
fn callbacks_both_ways() -> Result<()> {
    // The C++ Callback class, and a Rust implementation of its virtual call() (a director).
    let native = Callback::new()?;
    assert_eq!(call_with_callback(3.0, Some(&native))?, 3.0);
    assert_eq!(sum_dvector(&map_dvector_with_callback(&[1.0, 2.0, 3.0], Some(&native))?)?, 6.0);
    let doubler = Callback::from_impl(Doubler)?;
    assert_eq!(call_with_callback(3.0, Some(&doubler))?, 6.0);
    assert_eq!(map_dvector_with_callback(&[1.0, 2.0, 3.0], Some(&doubler))?, [2.0, 4.0, 6.0]);
    // A Rust error inside the implementation comes back as a Callback error with the original source.
    let failing = Callback::from_impl(Failing)?;
    let error = call_with_callback(1.0, Some(&failing)).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Callback);
    assert!(std::error::Error::source(&error).is_some_and(|s| s.is::<FromRust>()));
    Ok(())
}

#[test]
fn errors_and_classes() -> Result<()> {
    // C++ exceptions become errors with their kind.
    let error = runif(1.0, 0.0, 3).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::InvalidArgument);
    let frame = DataFrame::new()?;
    frame.add_numeric_column("x", &[1.0, 2.0, 3.0])?;
    assert_eq!(frame.nrows()?, 3);
    let seq = seql(0.0, 1.0, 5)?;
    assert_eq!(seq, [0.0, 0.25, 0.5, 0.75, 1.0]);
    // The policy module datamunge::data.
    let iris = data::iris()?.expect("iris");
    assert_eq!(iris.nrows()?, 150);
    Ok(())
}
