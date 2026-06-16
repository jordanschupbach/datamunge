fn main() {
    rustoctra::hello();

    let v = rustoctra::make_dvector(1.0, 2.0, 3.0);
    println!("make_dvector -> {:?}", v);
    println!("sum_dvector -> {}", rustoctra::sum_dvector(&v));

    let p = rustoctra::make_dpair(4.0, 5.0);
    println!("make_dpair -> {:?}", p);
    println!("sum_dpair -> {}", rustoctra::sum_dpair(p));

    unsafe extern "C" fn times_two(x: f64, _userdata: *mut std::ffi::c_void) -> f64 {
        x * 2.0
    }

    println!(
        "call_with_callback(3.0) -> {}",
        rustoctra::call_with_callback(3.0, Some(times_two), std::ptr::null_mut())
    );
    let out = rustoctra::map_dvector_with_callback(
        &[1.0, 2.0, 3.0],
        Some(times_two),
        std::ptr::null_mut(),
    );
    println!("map_dvector_with_callback([1,2,3]) -> {:?}", out);
}
