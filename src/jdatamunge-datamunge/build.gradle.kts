import java.io.BufferedReader
import java.io.InputStreamReader

plugins {
  id("org.xbib.gradle.plugin.cmake") version "3.1.0"
}

cmake {
  // TODO: need to set CMAKE_PATH env variable... Is there a better way?
  executable = System.getenv("CMAKE_PATH") ?: "/usr/bin/env cmake"  // Default to /usr/bin/cmake if not set
  // executable = findCMakePath() ?: error("CMake not found. Please ensure it's installed and available in your PATH.")
  // executable='/my/path/to/cmake'
  // workingFolder=file("$buildDir/cmake")
  // sourceFolder=file("$projectDir/src")
  sourceFolder=file("./")
  // installPrefix="${System.properties['user.home']}"
  // generator='Visual Studio 15 2017'
  // platform="x64"
  // toolset='v141'
  // does it do two passes to build both static and shared libs?
  // optionally set to build static libs
  // buildStaticLibs=true
  // optionally set to build shared libs
  buildSharedLibs=true
  // define arbitrary CMake parameters. The below adds -Dtest=hello to cmake command line.
  // defs.test='hello'
  // cmakeBuild parameters
  // optional configuration to build
  // buildConfig='Release'
  // optional build target
  // buildTarget='install'
  // optional build clean. if set to true, calls cmake --build with --clean-first
  // buildClean=false
}
