mod bindings;

pub fn hello() {
    unsafe { bindings::octra_hello() }
}

pub fn make_dvector(a: f64, b: f64, c: f64) -> [f64; 3] {
    let mut out = [0.0_f64; 3];
    unsafe { bindings::octra_make_dvector(a, b, c, out.as_mut_ptr()) };
    out
}

pub fn sum_dvector(values: &[f64]) -> f64 {
    unsafe { bindings::octra_sum_dvector(values.as_ptr(), values.len()) }
}

pub fn make_dpair(a: f64, b: f64) -> (f64, f64) {
    let p = unsafe { bindings::octra_make_dpair(a, b) };
    (p.first, p.second)
}

pub fn sum_dpair(values: (f64, f64)) -> f64 {
    let p = bindings::octra_dpair {
        first: values.0,
        second: values.1,
    };
    unsafe { bindings::octra_sum_dpair(p) }
}

pub fn call_with_callback(
    x: f64,
    cb: bindings::octra_double_cb,
    userdata: *mut std::ffi::c_void,
) -> f64 {
    unsafe { bindings::octra_call_double_cb(x, cb, userdata) }
}

pub fn map_dvector_with_callback(
    values: &[f64],
    cb: bindings::octra_double_cb,
    userdata: *mut std::ffi::c_void,
) -> Vec<f64> {
    let mut out = vec![0.0_f64; values.len()];
    unsafe {
        bindings::octra_map_dvector_cb(
            values.as_ptr(),
            values.len(),
            out.as_mut_ptr(),
            cb,
            userdata,
        )
    };
    out
}
