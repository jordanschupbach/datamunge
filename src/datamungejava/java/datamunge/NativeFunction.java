package datamunge;

import java.lang.Object;
import java.lang.System;

/**
 * A C++ {@code std::function} returned to Java. Generated methods wrap it in the declared
 * functional interface; {@link #invoke(Object...)} calls it directly.
 */
public final class NativeFunction extends NativeObject {
    NativeFunction() { super(RAW); }

    /**
     * Calls the C++ function.
     *
     * @param arguments its arguments
     * @return its result, as a plain value
     */
    public Object invoke(Object... arguments) {
        final Object[] all = new Object[arguments.length + 1];
        all[0] = new Rosetta.Callee(this);
        System.arraycopy(arguments, 0, all, 1, arguments.length);
        return Rosetta.plain(Rosetta.call(0, all));
    }
}
