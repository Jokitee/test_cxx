#ifndef VISION_SYSTEM_INPUT_CAMERA_WRAPPER_HPP
#define VISION_SYSTEM_INPUT_CAMERA_WRAPPER_HPP

#include "vision_system/core/config_manager.hpp"
#include <opencv2/opencv.hpp>
#include <mutex>
#include <thread>
#include <chrono>

class MVCamera {
public:
    MVCamera();
    ~MVCamera();
    bool init(const CameraConfig& cfg);
    bool read(cv::Mat& outImage);
    bool checkConnection();
    bool reconnect();

private:
    int hCamera_ = -1;
    unsigned char* pRgbBuffer_ = nullptr;
    std::mutex cam_mutex_;
    CameraConfig cfg_;
    
    // 视频流支持
    bool is_video_mode_ = false;
    cv::VideoCapture cap_;
};

#endif
