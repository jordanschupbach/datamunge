rootProject.name = "jdatamunge"
include("jdatamunge-datamunge")
include("jdatamunge")
// include("app")

val bindingsDir = file("src")
project(":jdatamunge-datamunge").projectDir = file("$bindingsDir/jdatamunge-datamunge")
project(":jdatamunge").projectDir = file("$bindingsDir/jdatamunge")
