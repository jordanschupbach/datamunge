/// @file video.cpp
/// @brief Native process-owned, bounded and shell-free RGBA-to-H.264 streaming.
#include <datamunge/cad/video.hpp>
#include <algorithm>
#include <cerrno>
#include <condition_variable>
#include <filesystem>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <spawn.h>
#include <pthread.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
/// @brief Process environment forwarded unchanged to the native ffmpeg child.
extern char** environ;
#endif
namespace datamunge::cad {
/// @brief Encoder state; construction cleanup and destruction both preserve an old destination.
struct VideoWriter::Impl {
  std::filesystem::path destination; ///< Final path, replaced only after successful encoding.
  std::filesystem::path temporary; ///< Private output on the same filesystem as destination.
  int width; ///< Checked even image width.
  int height; ///< Checked even image height.
  int fps; ///< Checked positive integer frame rate.
  std::size_t count{0}; ///< Successfully sent complete frames.
  bool failed{false}; ///< Whether a complete-frame write failed.
  bool published{false}; ///< Whether finish() successfully replaced the destination.
  bool closed{false}; ///< Whether input has closed and the child was reaped.
  int exit_status{-1}; ///< Cached child exit status after closing input.
#ifdef _WIN32
  HANDLE input{nullptr}; ///< Owned parent pipe writer.
  HANDLE process{nullptr}; ///< Owned encoder process.

  /// @brief Quote one argv item for Windows' native command-line parser, without invoking a shell.
  /// @param value Owned argument bytes converted to the native wide path representation.
  /// @return Quoted argument preserving embedded quotes and trailing backslashes.
  static std::wstring quote(const std::wstring& value){
    std::wstring out=L"\"";std::size_t slashes=0;
    for(wchar_t ch:value){if(ch==L'\\'){++slashes;continue;}out.append(ch==L'"'?2*slashes+1:slashes,L'\\');out+=ch;slashes=0;}
    out.append(2*slashes,L'\\');out+=L'"';return out;
  }
#else
  int input{-1}; ///< Owned parent pipe writer; only the private feeder writes this descriptor.
  pid_t process{-1}; ///< Owned encoder process, never waited by another writer.
  std::thread feeder; ///< Synchronous handoff worker with SIGPIPE blocked only on this thread.
  std::mutex mutex; ///< Protects the one-frame borrowed handoff and worker state.
  std::condition_variable changed; ///< Wakes sender/feeder without polling or an unbounded queue.
  bool ready{false}; ///< Whether thread-local signal setup completed, under mutex.
  bool stop{false}; ///< Whether the feeder should exit, under mutex.
  bool request{false}; ///< Whether borrowed bytes await a pipe write, under mutex.
  bool done{false}; ///< Whether the outstanding frame write completed, under mutex.
  int feeder_error{0}; ///< Startup/write errno; zero means a complete successful handoff.
  const unsigned char* data{nullptr}; ///< Borrowed until append() receives completion.
  std::size_t bytes{0}; ///< Number of borrowed bytes requested for this handoff.

  /// @brief Feed one frame at a time with thread-local SIGPIPE blocking; pending signals die with this worker.
  void feed(){
    sigset_t blocked;sigemptyset(&blocked);sigaddset(&blocked,SIGPIPE);
    int error=pthread_sigmask(SIG_BLOCK,&blocked,nullptr);
    std::unique_lock lock(mutex);feeder_error=error;ready=true;changed.notify_all();if(error)return;
    for(;;){
      changed.wait(lock,[&]{return stop||request;});if(stop)return;
      auto buffer=data;auto size=bytes;lock.unlock();std::size_t sent=0;error=0;
      while(sent<size){auto n=write(input,buffer+sent,size-sent);if(n<0&&errno==EINTR)continue;if(n<=0){error=n<0?errno:EIO;break;}sent+=std::size_t(n);}
      lock.lock();feeder_error=error;request=false;done=true;changed.notify_all();if(error)return;
    }
  }
  /// @brief Borrow a complete frame until the bounded pipe accepts it, then release its caller.
  /// @param buffer Borrowed bytes kept alive by append() until this returns.
  /// @param size Exact configured RGBA byte count.
  /// @throws std::runtime_error Child termination or pipe write failure.
  void send_frame(const unsigned char* buffer,std::size_t size){
    std::unique_lock lock(mutex);if(feeder_error)throw std::runtime_error("Video feeder failed");
    data=buffer;bytes=size;done=false;request=true;changed.notify_all();changed.wait(lock,[&]{return done||feeder_error;});
    if(feeder_error)throw std::runtime_error("Video encoder stopped accepting frames (errno "+std::to_string(feeder_error)+")");
  }
#endif
  /// @brief Create the private file and spawn the encoder, cleaning all partial state on failure.
  /// @param path Nonempty validated destination.
  /// @param w Checked even width.
  /// @param h Checked even height.
  /// @param rate Checked positive integer rate.
  /// @throws std::runtime_error Temporary-file, pipe or child-start failure.
  Impl(const std::string& path,int w,int h,int rate):destination(std::filesystem::absolute(path)),width(w),height(h),fps(rate){
    try{
#ifdef _WIN32
      wchar_t file[MAX_PATH];
      if(!GetTempFileNameW(destination.parent_path().c_str(),L"dmv",0,file))throw std::runtime_error("Cannot create video temporary file");
      temporary=file;
#else
      auto pattern=destination.string()+".part.XXXXXX";std::vector<char> file(pattern.begin(),pattern.end());file.push_back(0);
      int descriptor=mkstemp(file.data());
      if(descriptor<0)throw std::runtime_error("Cannot create video temporary file");
      close(descriptor);temporary=file.data();
#endif
      std::vector<std::string> args={"ffmpeg","-hide_banner","-loglevel","error","-y","-nostdin","-f","rawvideo","-pixel_format","rgba","-video_size",std::to_string(width)+"x"+std::to_string(height),"-framerate",std::to_string(fps),"-i","pipe:0","-an","-c:v","libx264","-profile:v","baseline"};
      auto macroblocks=((width+15)/16)*((height+15)/16);
      if(macroblocks<=3600&&macroblocks*fps<=108000){args.push_back("-level");args.push_back("3.1");}
      for(const auto& arg:std::vector<std::string>{"-bf","0","-g",std::to_string(2*fps),"-preset","fast","-crf","19","-pix_fmt","yuv420p","-movflags","+faststart","-f","mp4",temporary.string()})args.push_back(arg);
#ifdef _WIN32
      SECURITY_ATTRIBUTES security{sizeof(SECURITY_ATTRIBUTES),nullptr,TRUE};HANDLE reader=nullptr;
      if(!CreatePipe(&reader,&input,&security,0))throw std::runtime_error("Cannot create video input pipe");
      if(!SetHandleInformation(input,HANDLE_FLAG_INHERIT,0)){CloseHandle(reader);throw std::runtime_error("Cannot configure video input pipe");}
      std::wstring command;
      for(const auto& arg:args){if(!command.empty())command+=L' ';command+=quote(std::filesystem::path(arg).wstring());}
      STARTUPINFOW startup{};startup.cb=sizeof(startup);startup.dwFlags=STARTF_USESTDHANDLES;startup.hStdInput=reader;startup.hStdOutput=GetStdHandle(STD_OUTPUT_HANDLE);startup.hStdError=GetStdHandle(STD_ERROR_HANDLE);
      PROCESS_INFORMATION child{};BOOL ok=CreateProcessW(nullptr,command.data(),nullptr,nullptr,TRUE,CREATE_NO_WINDOW,nullptr,nullptr,&startup,&child);CloseHandle(reader);
      if(!ok)throw std::runtime_error("Cannot start ffmpeg; install ffmpeg with libx264 on PATH");
      process=child.hProcess;CloseHandle(child.hThread);
#else
      int descriptors[2];if(pipe(descriptors)!=0)throw std::runtime_error("Cannot create video input pipe");
      int reader=descriptors[0];input=descriptors[1];
      // Avoid accidentally closing STDIN in spawn actions if the caller had it closed.
      if(input<3){int fd=fcntl(input,F_DUPFD,3);if(fd<0){close(reader);throw std::runtime_error("Cannot duplicate video input");}close(input);input=fd;}
      if(reader<3){int fd=fcntl(reader,F_DUPFD,3);if(fd<0){close(reader);throw std::runtime_error("Cannot duplicate video reader");}close(reader);reader=fd;}
      if(fcntl(input,F_SETFD,FD_CLOEXEC)<0||fcntl(reader,F_SETFD,FD_CLOEXEC)<0){close(reader);throw std::runtime_error("Cannot configure video descriptors");}
      posix_spawn_file_actions_t actions;
      int error=posix_spawn_file_actions_init(&actions);
      if(error){close(reader);throw std::runtime_error("Cannot initialize video child actions");}
      error=posix_spawn_file_actions_adddup2(&actions,reader,STDIN_FILENO);
      if(!error)error=posix_spawn_file_actions_addclose(&actions,reader);
      if(!error)error=posix_spawn_file_actions_addclose(&actions,input);
      std::vector<char*> argv;for(auto& arg:args)argv.push_back(arg.data());argv.push_back(nullptr);
      pid_t pid=-1;if(!error)error=posix_spawnp(&pid,"ffmpeg",&actions,nullptr,argv.data(),environ);
      posix_spawn_file_actions_destroy(&actions);close(reader);
      if(error)throw std::runtime_error("Cannot start ffmpeg; install ffmpeg with libx264 on PATH");
      process=pid;
      feeder=std::thread([this]{feed();});
      std::unique_lock lock(mutex);changed.wait(lock,[&]{return ready;});
      if(feeder_error)throw std::runtime_error("Cannot initialize video feeder signal mask");
#endif
    }catch(...){cleanup();throw;}
  }
  /// @brief Close bounded input and reap only this writer's child; safe during failure cleanup.
  void close_wait()noexcept{
    if(closed)return;closed=true;
#ifdef _WIN32
    if(input){CloseHandle(input);input=nullptr;}
    if(process){DWORD code=1;if(WaitForSingleObject(process,INFINITE)==WAIT_OBJECT_0&&GetExitCodeProcess(process,&code))exit_status=int(code);CloseHandle(process);process=nullptr;}
#else
    if(feeder.joinable()){{std::lock_guard lock(mutex);stop=true;changed.notify_all();}feeder.join();}
    if(input>=0){close(input);input=-1;}
    if(process>0){int status=0;pid_t result;do{result=waitpid(process,&status,0);}while(result<0&&errno==EINTR);if(result==process)exit_status=WIFEXITED(status)?WEXITSTATUS(status):128+WTERMSIG(status);process=-1;}
#endif
  }
  /// @brief Reap the encoder and remove uncommitted output without propagating exceptions.
  void cleanup()noexcept{close_wait();if(!published&&!temporary.empty()){std::error_code error;std::filesystem::remove(temporary,error);}}
  /// @brief Release all process/file resources; publishing requires an explicit finish().
  ~Impl(){cleanup();}
};
VideoWriter::VideoWriter(const std::string& path,int width,int height,int fps){
  if(path.empty()||path.find('\0')!=std::string::npos||width<2||height<2||width>8192||height>8192||width%2||height%2||fps<1||fps>240)throw std::invalid_argument("Video needs a valid path, even dimensions 2..8192 and fps 1..240");
  impl_=std::make_unique<Impl>(path,width,height,fps);
}
VideoWriter::~VideoWriter()=default;
void VideoWriter::append(const RenderResult& frame){
  auto& p=*impl_;if(p.closed)throw std::logic_error("Video encoder is closed");
  if(p.failed)throw std::runtime_error("Video encoder previously failed");
  if(frame.width!=p.width||frame.height!=p.height||frame.rgba.size()!=std::size_t(p.width)*p.height*4)throw std::invalid_argument("Video frame dimensions or RGBA bytes mismatch");
#ifdef _WIN32
  std::size_t sent=0;
  while(sent<frame.rgba.size()){
    DWORD bytes=0;if(!WriteFile(p.input,frame.rgba.data()+sent,DWORD(frame.rgba.size()-sent),&bytes,nullptr)||!bytes){p.failed=true;throw std::runtime_error("Video encoder stopped accepting frames");}sent+=bytes;
  }
#else
  try{p.send_frame(frame.rgba.data(),frame.rgba.size());}catch(...){p.failed=true;throw;}
#endif
  ++p.count;
}
void VideoWriter::finish(){
  auto& p=*impl_;if(p.published)return;if(!p.count)throw std::logic_error("Video has no complete frames");
  p.close_wait();if(p.failed||p.exit_status!=0)throw std::runtime_error("Video encoding failed (exit "+std::to_string(p.exit_status)+")");
#ifdef _WIN32
  if(!MoveFileExW(p.temporary.c_str(),p.destination.c_str(),MOVEFILE_REPLACE_EXISTING|MOVEFILE_WRITE_THROUGH))throw std::runtime_error("Cannot publish completed video");
#else
  std::error_code error;std::filesystem::rename(p.temporary,p.destination,error);if(error)throw std::runtime_error("Cannot publish completed video: "+error.message());
#endif
  p.published=true;
}
std::size_t VideoWriter::frames_written()const{return impl_->count;}
} // namespace datamunge::cad
