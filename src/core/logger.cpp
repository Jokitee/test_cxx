/**
 * @brief 全局日志与报错系统实现
 * @author jokit
 * @date 2026-07-15
 */
#include "vision_system/core/logger.hpp"
#include <iostream>

std::string Logger::getErrorMsg(ErrorCode code) {
    switch (code) {
        case ErrorCode::SUCCESS: return "Success";
        case ErrorCode::CAMERA_ENUM_FAILED: return "Camera enumeration failed.";
        case ErrorCode::CAMERA_INIT_FAILED: return "Camera initialization failed.";
        case ErrorCode::CAMERA_PLAY_FAILED: return "Camera play failed.";
        case ErrorCode::CAMERA_DISCONNECTED: return "Camera disconnected.";
        case ErrorCode::MODEL_LOAD_FAILED: return "Model loading failed.";
        case ErrorCode::CONFIG_LOAD_FAILED: return "Configuration loading failed.";
        case ErrorCode::FRAME_GET_FAILED: return "Failed to get image frame.";
        default: return "Unknown Error";
    }
}

void Logger::log(LogLevel level, ErrorCode code, const std::string& msg) {
    std::string prefix;
    switch (level) {
        case LogLevel::INFO: prefix = "[INFO] "; break;
        case LogLevel::WARNING: prefix = "[WARNING] "; break;
        case LogLevel::ERROR: prefix = "[ERROR] "; break;
    }
    
    std::cerr << prefix << getErrorMsg(code);
    if (!msg.empty()) {
        std::cerr << " - " << msg;
    }
    std::cerr << std::endl;
}
