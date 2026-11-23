//! The datamunge Rust examples: each `examples/rust/<name>.rs` is a Cargo example
//! (`cargo run --example <name>`, or `just TARGET=<name> run-rust`).
//!
//! This library holds the small helpers the examples share, chiefly number formatting that
//! matches the other language ports (.NET's default `double.ToString()` and its `G<n>` format).

use std::fmt;

/// Splits a finite, nonzero `x` formatted as `{:.*e}` (or shortest `{:e}`) into its sign,
/// significant digits (no dot) and decimal exponent.
fn sci_parts(text: &str) -> (bool, String, i32) {
    let (mantissa, exponent) = text.split_once('e').expect("scientific notation");
    let negative = mantissa.starts_with('-');
    let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    (negative, digits, exponent.parse().expect("exponent"))
}

/// Lays out significant `digits` (trailing zeros already trimmed) with decimal exponent `exp`,
/// in scientific notation (`1.5E+20`, `1E-05`) when `scientific`, else positionally.
fn layout(negative: bool, digits: &str, exp: i32, scientific: bool) -> String {
    let mut out = String::new();
    if negative {
        out.push('-');
    }
    if scientific {
        out.push_str(&digits[..1]);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        out.push_str(&format!("E{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs()));
    } else if exp < 0 {
        out.push_str("0.");
        out.push_str(&"0".repeat((-exp - 1) as usize));
        out.push_str(digits);
    } else {
        let int_len = exp as usize + 1;
        if digits.len() <= int_len {
            out.push_str(digits);
            out.push_str(&"0".repeat(int_len - digits.len()));
        } else {
            out.push_str(&digits[..int_len]);
            out.push('.');
            out.push_str(&digits[int_len..]);
        }
    }
    out
}

fn special(x: f64) -> Option<String> {
    if x.is_nan() {
        Some("NaN".into())
    } else if x.is_infinite() {
        Some(if x > 0.0 { "∞" } else { "-∞" }.into())
    } else if x == 0.0 {
        Some(if x.is_sign_negative() { "-0" } else { "0" }.into())
    } else {
        None
    }
}

/// Formats `x` like .NET's default `double.ToString()`: the shortest round-trip digits,
/// positional unless the exponent is below -4 or at least 15 (`6`, `0.1`, `1E-05`).
pub fn net(x: f64) -> String {
    if let Some(s) = special(x) {
        return s;
    }
    let (negative, digits, exp) = sci_parts(&format!("{:e}", x));
    let digits = digits.trim_end_matches('0');
    let digits = if digits.is_empty() { "0" } else { digits };
    layout(negative, digits, exp, !(-4..15).contains(&exp))
}

/// Formats `x` like .NET's `ToString("G<precision>")` (C's `%g` with .NET's exponent style).
pub fn net_g(x: f64, precision: usize) -> String {
    if let Some(s) = special(x) {
        return s;
    }
    let (negative, digits, exp) = sci_parts(&format!("{:.*e}", precision.max(1) - 1, x));
    let digits = digits.trim_end_matches('0');
    let digits = if digits.is_empty() { "0" } else { digits };
    layout(negative, digits, exp, exp < -4 || exp >= precision as i32)
}

/// Joins the .NET-formatted values with `sep`.
pub fn join(values: &[f64], sep: &str) -> String {
    values.iter().map(|&v| net(v)).collect::<Vec<_>>().join(sep)
}

/// A `Display` wrapper for [`net`], so `format!("{}", N(x))` prints like .NET.
#[derive(Clone, Copy, Debug)]
pub struct N(pub f64);

impl fmt::Display for N {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad(&net(self.0))
    }
}

/// Formats a bool like .NET (`True`, `False`).
pub fn net_bool(b: bool) -> &'static str {
    if b {
        "True"
    } else {
        "False"
    }
}

/// .NET's seeded `System.Random` (Knuth's subtractive generator, the algorithm `new Random(seed)`
/// still uses), so the examples that simulate data reproduce the other ports' numbers exactly.
pub struct DotNetRandom {
    seed_array: [i32; 56],
    inext: usize,
    inextp: usize,
}

impl DotNetRandom {
    const MBIG: i32 = i32::MAX;
    const MSEED: i32 = 161_803_398;

    /// `new Random(seed)`.
    pub fn new(seed: i32) -> Self {
        let mut seed_array = [0i32; 56];
        let subtraction = if seed == i32::MIN { i32::MAX } else { seed.abs() };
        let mut mj = Self::MSEED - subtraction;
        seed_array[55] = mj;
        let mut mk = 1i32;
        for i in 1..55 {
            let ii = (21 * i) % 55;
            seed_array[ii] = mk;
            mk = mj - mk;
            if mk < 0 {
                mk += Self::MBIG;
            }
            mj = seed_array[ii];
        }
        for _ in 1..5 {
            for i in 1..56 {
                let n = (i + 30) % 55;
                seed_array[i] = seed_array[i].wrapping_sub(seed_array[1 + n]);
                if seed_array[i] < 0 {
                    seed_array[i] += Self::MBIG;
                }
            }
        }
        DotNetRandom { seed_array, inext: 0, inextp: 21 }
    }

    fn internal_sample(&mut self) -> i32 {
        let next = if self.inext + 1 >= 56 { 1 } else { self.inext + 1 };
        let nextp = if self.inextp + 1 >= 56 { 1 } else { self.inextp + 1 };
        let mut ret = self.seed_array[next].wrapping_sub(self.seed_array[nextp]);
        if ret == Self::MBIG {
            ret -= 1;
        }
        if ret < 0 {
            ret += Self::MBIG;
        }
        self.seed_array[next] = ret;
        self.inext = next;
        self.inextp = nextp;
        ret
    }

    /// `NextDouble()`: uniform on [0, 1).
    pub fn next_double(&mut self) -> f64 {
        self.internal_sample() as f64 * (1.0 / Self::MBIG as f64)
    }

    /// `Next(min, max)`: uniform integer on [min, max) (for ranges below 2^31).
    pub fn next_range(&mut self, min: i32, max: i32) -> i32 {
        let range = max as i64 - min as i64;
        assert!(range <= i32::MAX as i64, "large ranges are not supported");
        (self.next_double() * range as f64) as i32 + min
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_like_dotnet() {
        assert_eq!(net(6.0), "6");
        assert_eq!(net(0.1), "0.1");
        assert_eq!(net(1e-5), "1E-05");
        assert_eq!(net(0.0001), "0.0001");
        assert_eq!(net(1e15), "1E+15");
        assert_eq!(net(123456789012345.0), "123456789012345");
        assert_eq!(net(-2.5), "-2.5");
        assert_eq!(net(1.5e-7), "1.5E-07");
        assert_eq!(net(f64::MAX), "1.7976931348623157E+308");
        assert_eq!(net_g(1234567.0, 6), "1.23457E+06");
        assert_eq!(net_g(0.5, 6), "0.5");
        assert_eq!(net_g(3.0e-7, 3), "3E-07");
        assert_eq!(net_g(-0.000123456789, 6), "-0.000123457");
    }

    #[test]
    fn dotnet_random_sequence() {
        let mut rng = DotNetRandom::new(42);
        assert_eq!(rng.next_double(), 0.6681064659115423);
    }
}
