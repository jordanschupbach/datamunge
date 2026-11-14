package datamunge;

import java.lang.String;

/** Implemented by the enums of native module {@code datamunge}. */
public interface NativeEnum {
    /**
     * The C++ value of the enumerator.
     *
     * @return the value
     */
    long value();

    /**
     * The C++ name of the enumerator.
     *
     * @return the name
     */
    String nativeName();
}
