#[test]
fn loads_and_calls() {
    rustdatamunge::hello();

    let v = rustdatamunge::make_dvector(1.0, 2.0, 3.0);
    assert_eq!(v, [1.0, 2.0, 3.0]);
    assert_eq!(rustdatamunge::sum_dvector(&v), 6.0);

    let p = rustdatamunge::make_dpair(4.0, 5.0);
    assert_eq!(p, (4.0, 5.0));
    assert_eq!(rustdatamunge::sum_dpair(p), 9.0);
}

#[test]
fn can_pass_callback_into_c() {
    unsafe extern "C" fn times_two(x: f64, _userdata: *mut std::ffi::c_void) -> f64 {
        x * 2.0
    }

    let y = rustdatamunge::call_with_callback(3.0, Some(times_two), std::ptr::null_mut());
    assert_eq!(y, 6.0);

    let out = rustdatamunge::map_dvector_with_callback(
        &[1.0, 2.0, 3.0],
        Some(times_two),
        std::ptr::null_mut(),
    );
    assert_eq!(out, vec![2.0, 4.0, 6.0]);
}
