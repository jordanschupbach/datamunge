rootProject.name = "joctra"
include("joctra-octra")
include("joctra")
// include("app")

val bindingsDir = file("src")
project(":joctra-octra").projectDir = file("$bindingsDir/joctra-octra")
project(":joctra").projectDir = file("$bindingsDir/joctra")
