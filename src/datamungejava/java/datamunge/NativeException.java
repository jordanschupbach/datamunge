package datamunge;

import java.lang.Override;
import java.lang.RuntimeException;
import java.lang.String;

/**
 * A C++ exception (other than the kinds {@link NativeError} maps to standard Java exceptions), or a
 * failed conversion, in a call to native module {@code datamunge}.
 */
public class NativeException extends RuntimeException implements NativeError {
    private static final long serialVersionUID = 1L;
    /** What failed. */
    private final String kind;

    /**
     * A native error.
     *
     * @param kind what failed (see {@link NativeError#kind()})
     * @param message the message
     */
    public NativeException(String kind, String message) {
        super(message);
        this.kind = kind;
    }

    @Override
    public String kind() { return kind; }
}
