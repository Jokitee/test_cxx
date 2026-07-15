/**
 * @brief 迈德威视相机SDK硬件封装层
 * @author jokit
 * @date 2026-07-15
 */
#include "vision_system/input/camera_wrapper.hpp"
#include "vision_system/core/logger.hpp"
#include "CameraApi.h"

MVCamera::MVCamera() {}

MVCamera::~MVCamera() {
    std::lock_guard<std::mutex> lock(cam_mutex_);
    if (hCamera_ != -1) {
        CameraUnInit(hCamera_);
    }
    if (pRgbBuffer_) {
        free(pRgbBuffer_);
    }
}

bool MVCamera::init(const CameraConfig& cfg) {
    std::lock_guard<std::mutex> lock(cam_mutex_);
    cfg_ = cfg;
    
    if (cfg_.mode == "video") {
        is_video_mode_ = true;
        if (!cap_.open(cfg_.video_path)) {
            Logger::log(LogLevel::ERROR, ErrorCode::CAMERA_INIT_FAILED, "Cannot open video file: " + cfg_.video_path);
            return false;
        }
        Logger::log(LogLevel::INFO, ErrorCode::SUCCESS, "Loaded video file for testing: " + cfg_.video_path);
        return true;
    }

    is_video_mode_ = false;
    CameraSdkInit(1);

    int iCameraCounts = 1;
    tSdkCameraDevInfo tCameraEnumList;
    if (CameraEnumerateDevice(&tCameraEnumList, &iCameraCounts) != CAMERA_STATUS_SUCCESS || iCameraCounts == 0) {
        Logger::log(LogLevel::ERROR, ErrorCode::CAMERA_ENUM_FAILED);
        return false;
    }

    if (CameraInit(&tCameraEnumList, -1, -1, &hCamera_) != CAMERA_STATUS_SUCCESS) {
        Logger::log(LogLevel::ERROR, ErrorCode::CAMERA_INIT_FAILED);
        return false;
    }

    tSdkCameraCapbility tCapability;
    CameraGetCapability(hCamera_, &tCapability);

    pRgbBuffer_ = (unsigned char*)malloc(tCapability.sResolutionRange.iHeightMax * tCapability.sResolutionRange.iWidthMax * 3);

    if (CameraPlay(hCamera_) != CAMERA_STATUS_SUCCESS) {
        Logger::log(LogLevel::ERROR, ErrorCode::CAMERA_PLAY_FAILED);
        return false;
    }

    CameraSetAeState(hCamera_, false);
    CameraSetExposureTime(hCamera_, cfg_.exposure_time);
    CameraSetGain(hCamera_, cfg_.gain_r, cfg_.gain_g, cfg_.gain_b);

    if (tCapability.sIspCapacity.bMonoSensor) {
        CameraSetIspOutFormat(hCamera_, CAMERA_MEDIA_TYPE_MONO8);
    } else {
        CameraSetIspOutFormat(hCamera_, CAMERA_MEDIA_TYPE_BGR8);
    }
    
    return true;
}

bool MVCamera::read(cv::Mat& outImage) {
    std::lock_guard<std::mutex> lock(cam_mutex_);
    if (is_video_mode_) {
        if (!cap_.read(outImage)) {
            // 如果视频播完，循环播放
            cap_.set(cv::CAP_PROP_POS_FRAMES, 0);
            cap_.read(outImage);
        }
        if (outImage.empty()) return false;
        
        // 简单延时模拟相机帧率，防止读取过快占用100%CPU
        std::this_thread::sleep_for(std::chrono::milliseconds(30)); 
        return true;
    }

    if (hCamera_ == -1 || !pRgbBuffer_) return false;

    tSdkFrameHead sFrameInfo;
    BYTE* pbyBuffer;

    if (CameraGetImageBuffer(hCamera_, &sFrameInfo, &pbyBuffer, 1000) == CAMERA_STATUS_SUCCESS) {
        CameraImageProcess(hCamera_, pbyBuffer, pRgbBuffer_, &sFrameInfo);

        outImage = cv::Mat(
            cv::Size(sFrameInfo.iWidth, sFrameInfo.iHeight),
            sFrameInfo.uiMediaType == CAMERA_MEDIA_TYPE_MONO8 ? CV_8UC1 : CV_8UC3,
            pRgbBuffer_
        ).clone(); // MUST clone since we will release the buffer

        CameraReleaseImageBuffer(hCamera_, pbyBuffer);
        return true;
    }
    return false;
}

bool MVCamera::checkConnection() {
    std::lock_guard<std::mutex> lock(cam_mutex_);
    if (hCamera_ == -1) return false;
    return CameraConnectTest(hCamera_) == CAMERA_STATUS_SUCCESS;
}

bool MVCamera::reconnect() {
    std::lock_guard<std::mutex> lock(cam_mutex_);
    if (hCamera_ == -1) return false;
    if (CameraReConnect(hCamera_) == CAMERA_STATUS_SUCCESS) {
        CameraSetAeState(hCamera_, false);
        CameraSetExposureTime(hCamera_, cfg_.exposure_time);
        CameraSetGain(hCamera_, cfg_.gain_r, cfg_.gain_g, cfg_.gain_b);
        return true;
    }
    return false;
}
