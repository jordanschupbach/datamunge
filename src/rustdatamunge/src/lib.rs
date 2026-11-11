mod bindings;

pub fn hello() {
    unsafe { bindings::datamunge_hello() }
}

pub fn make_dvector(a: f64, b: f64, c: f64) -> [f64; 3] {
    let mut out = [0.0_f64; 3];
    unsafe { bindings::datamunge_make_dvector(a, b, c, out.as_mut_ptr()) };
    out
}

pub fn sum_dvector(values: &[f64]) -> f64 {
    unsafe { bindings::datamunge_sum_dvector(values.as_ptr(), values.len()) }
}

pub fn make_dpair(a: f64, b: f64) -> (f64, f64) {
    let p = unsafe { bindings::datamunge_make_dpair(a, b) };
    (p.first, p.second)
}

pub fn sum_dpair(values: (f64, f64)) -> f64 {
    let p = bindings::datamunge_dpair {
        first: values.0,
        second: values.1,
    };
    unsafe { bindings::datamunge_sum_dpair(p) }
}

pub fn call_with_callback(
    x: f64,
    cb: bindings::datamunge_double_cb,
    userdata: *mut std::ffi::c_void,
) -> f64 {
    unsafe { bindings::datamunge_call_double_cb(x, cb, userdata) }
}

pub fn map_dvector_with_callback(
    values: &[f64],
    cb: bindings::datamunge_double_cb,
    userdata: *mut std::ffi::c_void,
) -> Vec<f64> {
    let mut out = vec![0.0_f64; values.len()];
    unsafe {
        bindings::datamunge_map_dvector_cb(
            values.as_ptr(),
            values.len(),
            out.as_mut_ptr(),
            cb,
            userdata,
        )
    };
    out
}

extern "C" {
    fn datamunge_rust_bspline_predict(
        x: *const f64,
        y: *const f64,
        len: usize,
        basis: usize,
        query: *const f64,
        query_len: usize,
        out: *mut f64,
    ) -> i32;
}

/// Fit a cubic B-spline regression using datamunge's C++ core and predict at `query`.
///
/// Inputs are borrowed only during the call; predictions own their storage. Coordinates
/// and responses must be finite. `x` and `y` must be nonempty with equal lengths, and
/// `basis` must be at least four. Insufficient data/rank, invalid inputs and native
/// allocation failures return `Err`. An empty query returns an empty vector after
/// validating the fit. Predictions outside the fitted domain use the nearest boundary.
/// Calls are independent and do not change datamunge's random seed.
///
/// ```
/// let x = [0.0, 0.25, 0.5, 0.75, 1.0];
/// let prediction = rustdatamunge::bspline_predict(&x, &x, 4, &[0.5]).unwrap();
/// assert!((prediction[0] - 0.5).abs() < 1e-10);
/// ```
pub fn bspline_predict(
    x: &[f64],
    y: &[f64],
    basis: usize,
    query: &[f64],
) -> Result<Vec<f64>, &'static str> {
    if x.is_empty() || x.len() != y.len() || basis < 4 {
        return Err("expected equally sized nonempty x/y and at least four basis functions");
    }
    if x.iter().chain(y).chain(query).any(|v| !v.is_finite()) {
        return Err("coordinates and responses must be finite");
    }
    let mut predictions = vec![0.0; query.len()];
    // All buffers remain alive for the call and the owned output cannot alias inputs.
    let status = unsafe {
        datamunge_rust_bspline_predict(
            x.as_ptr(),
            y.as_ptr(),
            x.len(),
            basis,
            query.as_ptr(),
            query.len(),
            predictions.as_mut_ptr(),
        )
    };
    if status == 0 {
        Ok(predictions)
    } else {
        Err("native B-spline fit or prediction failed")
    }
}

#[cfg(test)]
mod bspline_tests {
    use super::bspline_predict;

    #[test]
    fn fits_linear_data_and_clamps_extrapolation() {
        let x = [0.0, 0.25, 0.5, 0.75, 1.0];
        let y: Vec<_> = x.iter().map(|x| 2.0 * x + 1.0).collect();
        let fitted = bspline_predict(&x, &y, 4, &[-1.0, 0.5, 2.0]).unwrap();
        for (actual, expected) in fitted.iter().zip([1.0, 2.0, 3.0]) {
            assert!((actual - expected).abs() < 1e-10);
        }
        assert!(bspline_predict(&x, &y, 4, &[]).unwrap().is_empty());
    }

    #[test]
    fn rejects_invalid_inputs_and_catches_native_fit_errors() {
        assert!(bspline_predict(&[], &[], 4, &[]).is_err());
        assert!(bspline_predict(&[0.0], &[], 4, &[]).is_err());
        assert!(bspline_predict(&[0.0], &[1.0], 3, &[]).is_err());
        assert!(bspline_predict(&[f64::NAN], &[1.0], 4, &[]).is_err());
        assert!(bspline_predict(&[0.0], &[f64::INFINITY], 4, &[]).is_err());
        assert!(bspline_predict(&[0.0], &[1.0], 4, &[f64::INFINITY]).is_err());
        // Native rank-deficient designs report failure without unwinding into Rust.
        assert!(bspline_predict(&[0.0; 5], &[1.0; 5], 4, &[0.0]).is_err());
    }
}
