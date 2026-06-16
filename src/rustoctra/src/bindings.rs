/* Auto-generated via `just prebuild-rust` (bindgen). */

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

use std::os::raw::c_double;
use std::os::raw::c_void;

pub type size_t = usize;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct octra_dpair {
    pub first: c_double,
    pub second: c_double,
}

pub type octra_double_cb =
    ::std::option::Option<unsafe extern "C" fn(x: c_double, userdata: *mut c_void) -> c_double>;

extern "C" {
    pub fn octra_hello();
    pub fn octra_make_dvector(a: c_double, b: c_double, c: c_double, out3: *mut c_double);
    pub fn octra_sum_dvector(values: *const c_double, len: size_t) -> c_double;
    pub fn octra_make_dpair(a: c_double, b: c_double) -> octra_dpair;
    pub fn octra_sum_dpair(values: octra_dpair) -> c_double;
    pub fn octra_call_double_cb(
        x: c_double,
        cb: octra_double_cb,
        userdata: *mut c_void,
    ) -> c_double;
    pub fn octra_map_dvector_cb(
        values: *const c_double,
        len: size_t,
        out: *mut c_double,
        cb: octra_double_cb,
        userdata: *mut c_void,
    );
}

// Keep bindgen output layout stable if re-generated.
#[allow(dead_code)]
const _BINDGEN_DUMMY: *const c_void = std::ptr::null();
