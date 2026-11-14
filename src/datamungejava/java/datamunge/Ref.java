package datamunge;

import java.lang.Object;
import java.lang.Override;
import java.lang.String;

/**
 * A mutable reference argument: C++ reads {@link #value} and writes its new value back.
 *
 * @param <T> the type of the value
 */
public final class Ref<T> {
    /** The value, updated by the call. */
    public T value;
    Object rosettaPending;
    boolean rosettaWritten;

    /**
     * A reference holding {@code value}.
     *
     * @param value the initial value
     */
    public Ref(T value) { this.value = value; }

    /**
     * Shows the value.
     *
     * @return the value as text
     */
    @Override
    public String toString() { return "Ref(" + value + ")"; }
}
