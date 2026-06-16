plugins {
  id("buildlogic.java-application-conventions")
}

import org.gradle.api.plugins.JavaApplication

application {
    // Define the main class for the application.
    mainClass.set("js.octra.joctra.examples.ExampleRunner")
}

dependencies {
    implementation("org.apache.commons:commons-text")
}

val nativeLibDir = file("$rootDir/src/joctra-octra/build/cmake").absolutePath
val nativeDepsDir = file("$rootDir/src/joctra-octra/build/cmake/_deps/octra-build").absolutePath

tasks.named<JavaExec>("run") {
    jvmArgs = listOf("-Djava.library.path=$nativeLibDir:$nativeDepsDir")
}

tasks.named<Test>("test") {
    systemProperty("java.library.path", "$nativeLibDir:$nativeDepsDir")
}
