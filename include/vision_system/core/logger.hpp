#ifndef VISION_SYSTEM_CORE_LOGGER_HPP
#define VISION_SYSTEM_CORE_LOGGER_HPP

#include <string>

enum class LogLevel {
    INFO,
    WARNING,
    ERROR
};

enum class ErrorCode {
    SUCCESS = 0,
    CAMERA_ENUM_FAILED,
    CAMERA_INIT_FAILED,
    CAMERA_PLAY_FAILED,
    CAMERA_DISCONNECTED,
    MODEL_LOAD_FAILED,
    CONFIG_LOAD_FAILED,
    FRAME_GET_FAILED
};

class Logger {
public:
    static void log(LogLevel level, ErrorCode code, const std::string& msg = "");
    static std::string getErrorMsg(ErrorCode code);
};

#endif
