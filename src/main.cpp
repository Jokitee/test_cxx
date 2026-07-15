/**
 * @brief 自动瞄准系统主入口
 * @author jokit
 * @date 2026-07-15
 */
#include "vision_system/core/pipeline.hpp"
#include "vision_system/core/logger.hpp"
#include <iostream>
#include <thread>
#include <chrono>

int main() {
    Logger::log(LogLevel::INFO, ErrorCode::SUCCESS, "AutoAim Application Starting...");

    // 实例化主流水线引擎，传入配置文件
    Pipeline vision_pipeline("config/vision_system.yaml");
    
    // 启动多线程工作流（含硬件探测、相机捕获、解算等）
    vision_pipeline.start();

    // 阻塞主线程以保持后台双工作线程运行
    while (vision_pipeline.isRunning()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    
    vision_pipeline.stop();
    Logger::log(LogLevel::INFO, ErrorCode::SUCCESS, "Application Exited gracefully.");
    
    return 0;
}
