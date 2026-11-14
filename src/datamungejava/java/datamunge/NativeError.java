package datamunge;

import java.lang.ArithmeticException;
import java.lang.IllegalArgumentException;
import java.lang.IndexOutOfBoundsException;
import java.lang.Override;
import java.lang.String;
import java.lang.UnsupportedOperationException;

/**
 * An error raised by native module {@code datamunge}: every exception the binding throws implements it.
 *
 * <p>C++ {@code std::invalid_argument} becomes {@link InvalidArgument} (an
 * {@link IllegalArgumentException}), {@code std::out_of_range} {@link OutOfRange} (an
 * {@link IndexOutOfBoundsException}), {@code std::domain_error}, {@code std::overflow_error},
 * {@code std::underflow_error} and {@code std::range_error} {@link Arithmetic} (an
 * {@link ArithmeticException}), a pure virtual method a Java subclass does not implement
 * {@link NotImplemented} (an {@link UnsupportedOperationException}), and anything else
 * {@link NativeException}.
 */
public interface NativeError {
    /**
     * What failed: the C++ exception's kind ({@code invalid_argument}, {@code out_of_range},
     * {@code domain_error}, {@code length_error}, {@code arithmetic_error}, {@code allocation},
     * {@code native}), or {@code binding} (conversion, overload or closed-object errors).
     *
     * @return the kind
     */
    String kind();

    /** A C++ {@code std::invalid_argument}. */
    final class InvalidArgument extends IllegalArgumentException implements NativeError {
        private static final long serialVersionUID = 1L;
        /** What failed. */
        private final String kind;
        /**
         * The error.
         *
         * @param kind the kind
         * @param message the C++ message
         */
        public InvalidArgument(String kind, String message) { super(message); this.kind = kind; }
        @Override
        public String kind() { return kind; }
    }

    /** A C++ {@code std::out_of_range}. */
    final class OutOfRange extends IndexOutOfBoundsException implements NativeError {
        private static final long serialVersionUID = 1L;
        /** What failed. */
        private final String kind;
        /**
         * The error.
         *
         * @param kind the kind
         * @param message the C++ message
         */
        public OutOfRange(String kind, String message) { super(message); this.kind = kind; }
        @Override
        public String kind() { return kind; }
    }

    /** A C++ arithmetic error ({@code std::domain_error}, {@code std::overflow_error}, ...). */
    final class Arithmetic extends ArithmeticException implements NativeError {
        private static final long serialVersionUID = 1L;
        /** What failed. */
        private final String kind;
        /**
         * The error.
         *
         * @param kind the kind
         * @param message the C++ message
         */
        public Arithmetic(String kind, String message) { super(message); this.kind = kind; }
        @Override
        public String kind() { return kind; }
    }

    /** A pure virtual C++ method that a Java subclass does not implement. */
    final class NotImplemented extends UnsupportedOperationException implements NativeError {
        private static final long serialVersionUID = 1L;
        /** What failed. */
        private final String kind;
        /**
         * The error.
         *
         * @param kind the kind
         * @param message the message
         */
        public NotImplemented(String kind, String message) { super(message); this.kind = kind; }
        @Override
        public String kind() { return kind; }
    }
}
