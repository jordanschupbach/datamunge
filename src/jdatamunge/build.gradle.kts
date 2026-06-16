plugins {
  id("buildlogic.java-application-conventions")
}

import org.gradle.api.plugins.JavaApplication

application {
    // Define the main class for the application.
    mainClass.set("js.datamunge.jdatamunge.examples.ExampleRunner")
}

dependencies {
    implementation("org.apache.commons:commons-text")
}

val nativeLibDir = file("$rootDir/src/jdatamunge-datamunge/build/cmake").absolutePath
val nativeDepsDir = file("$rootDir/src/jdatamunge-datamunge/build/cmake/_deps/datamunge-build").absolutePath

tasks.named<JavaExec>("run") {
    jvmArgs = listOf("-Djava.library.path=$nativeLibDir:$nativeDepsDir")
}

tasks.named<Test>("test") {
    systemProperty("java.library.path", "$nativeLibDir:$nativeDepsDir")
}
