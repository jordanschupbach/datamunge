package datamunge;

import java.lang.Math;
import java.lang.Override;
import java.lang.String;

/**
 * A C++ {@code std::complex<double>}.
 *
 * @param real the real part
 * @param imag the imaginary part
 */
public record NativeComplex(double real, double imag) {
    /**
     * The absolute value.
     *
     * @return {@code |z|}
     */
    public double abs() { return Math.hypot(real, imag); }

    /**
     * Shows the number.
     *
     * @return {@code (re,im)}, as C++ prints it
     */
    @Override
    public String toString() { return "(" + real + "," + imag + ")"; }
}
