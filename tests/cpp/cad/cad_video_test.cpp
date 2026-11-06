/// @file cad_video_test.cpp
/// @brief Real ffmpeg streaming, invalid-frame, filename and failed-child regressions.
#include <datamunge/cad/video.hpp>
#include <cstdlib>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#ifndef _WIN32
#include <pthread.h>
#include <signal.h>
#endif
using namespace datamunge::cad;
namespace fs=std::filesystem;
namespace {
/// @brief Require a runtime invariant in Release builds.
void check(bool ok){if(!ok)throw std::runtime_error("video invariant failed");}
/// @brief Read all owned file bytes for publication/failure checks.
std::string read(const fs::path& path){std::ifstream in(path,std::ios::binary);return {std::istreambuf_iterator<char>(in),{}};}
}
int main(){
#ifndef _WIN32
  struct sigaction original_handler{};sigaction(SIGPIPE,nullptr,&original_handler);
  sigset_t original_mask;pthread_sigmask(SIG_BLOCK,nullptr,&original_mask);
#endif
  auto dir=fs::temp_directory_path()/("datamunge-video-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));fs::create_directories(dir);
  auto path=dir/"video space'$().mp4";
  {std::ofstream old(path);old<<"original";}
  RenderResult frame;frame.width=64;frame.height=48;frame.rgba.assign(64*48*4,255);
  {
    VideoWriter video(path.string(),64,48,15);
    auto invalid=frame;invalid.rgba.pop_back();bool rejected=false;
    try{video.append(invalid);}catch(const std::invalid_argument&){rejected=true;}check(rejected&&video.frames_written()==0);
    video.append(frame);check(read(path)=="original");
    video.append(frame);video.finish();video.finish();check(video.frames_written()==2);
    auto data=read(path);check(data.size()>100&&data.substr(4,4)=="ftyp");
    rejected=false;try{video.append(frame);}catch(const std::logic_error&){rejected=true;}check(rejected);
  }
  auto completed=read(path);
  {VideoWriter abandoned(path.string(),64,48,15);abandoned.append(frame);}
  check(read(path)==completed);
  bool rejected=false;try{VideoWriter invalid(path.string(),63,48);}catch(const std::invalid_argument&){rejected=true;}check(rejected);
#ifndef _WIN32
  auto fake=dir/"fake";fs::create_directories(fake);
  {std::ofstream script(fake/"ffmpeg");script<<"#!/bin/sh\nexit 7\n";}
  fs::permissions(fake/"ffmpeg",fs::perms::owner_exec|fs::perms::owner_read|fs::perms::owner_write);
  const char* prior=std::getenv("PATH");const bool had_path=prior;std::string saved=prior?prior:"";
  setenv("PATH",fake.c_str(),1);bool failed=false;
  try{VideoWriter broken(path.string(),64,48,15);broken.append(frame);broken.finish();}catch(const std::exception&){failed=true;}
  if(had_path)setenv("PATH",saved.c_str(),1);else unsetenv("PATH");
  check(failed&&read(path)==completed); // No SIGPIPE exit, and no replacement on child failure.
  struct sigaction after_handler{};sigaction(SIGPIPE,nullptr,&after_handler);
  sigset_t after_mask;pthread_sigmask(SIG_BLOCK,nullptr,&after_mask);
  check(original_handler.sa_handler==after_handler.sa_handler);
  check(sigismember(&original_mask,SIGPIPE)==sigismember(&after_mask,SIGPIPE));
#endif
  for(const auto& file:fs::directory_iterator(dir))check(file.path().filename().string().find(".part.")==std::string::npos);
  fs::remove_all(dir);
  std::cout<<"Video: real encoding, exact frame validation, literal filenames, explicit publication, abandonment and failed child passed\n";
}
