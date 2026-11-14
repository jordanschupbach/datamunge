package datamunge;

import java.lang.AutoCloseable;
import java.lang.Object;
import java.lang.Override;
import java.lang.Runnable;
import java.lang.String;

/**
 * Base class of the classes of native module {@code datamunge}: each instance owns (or borrows) a
 * C++ object.
 *
 * <p>{@link #close()} releases the C++ object now (use try-with-resources); otherwise a
 * {@link java.lang.ref.Cleaner} releases it after the Java object becomes unreachable. A Java
 * class extending a native class overrides its virtual methods, also when C++ calls them.
 */
public abstract class NativeObject implements AutoCloseable {
    /** Marks the constructors that wrap an existing C++ object (used by the binding). */
    public static final class Raw {
        private Raw() {}
    }
    /** The marker that generated constructors pass to their superclass. */
    protected static final Raw RAW = new Raw();

    // The C++ holder: also the Cleaner's action, so it must not refer back to the object.
    static final class Handle implements Runnable {
        volatile long address;
        boolean owns;
        @Override
        public void run() {
            final long a = address;
            address = 0;
            if (a != 0 && owns) Rosetta.releaseLater(a);
        }
    }
    final Handle rosettaHandle = new Handle();
    Object rosettaParent;
    long rosettaId;

    /**
     * An object without a C++ object yet (generated constructors create it).
     *
     * @param raw the marker {@link #RAW}
     */
    protected NativeObject(Raw raw) {}

    /**
     * Whether the C++ object is released (or was never created, or was moved to C++).
     *
     * @return true once {@link #close()} ran
     */
    public final boolean isClosed() { return rosettaHandle.address == 0; }

    /** Releases the C++ object now; later calls do nothing. */
    @Override
    public final void close() { Rosetta.close(this); }

    /**
     * The class name, as {@code Name()} or {@code Name(closed)}.
     *
     * @return a description of the object
     */
    @Override
    public String toString() { return getClass().getSimpleName() + (isClosed() ? "(closed)" : "()"); }
}
