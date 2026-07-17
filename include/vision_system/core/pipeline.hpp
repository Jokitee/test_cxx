#ifndef VISION_SYSTEM_CORE_PIPELINE_HPP
#define VISION_SYSTEM_CORE_PIPELINE_HPP

#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <opencv2/opencv.hpp>

#include "vision_system/core/config_manager.hpp"
#include "vision_system/input/camera_wrapper.hpp"
#include "vision_system/decision/armor_detector.hpp"
#include "vision_system/decision/classifier.hpp"
#include "vision_system/decision/pnp_estimator.hpp"
#include "vision_system/decision/armor_model.hpp"
#include "vision_system/decision/fire_control_planner.hpp"
#include "vision_system/output/serial_port.hpp"

struct FrameData {
    cv::Mat image;
    int64_t timestamp;
};

class Pipeline {
public:
    Pipeline(const std::string& config_path);
    ~Pipeline();

    void start();
    void stop();
    bool isRunning() const { return is_running_; }

private:
    void cameraLoop();
    void processLoop();
    void monitorLoop();

    ConfigManager config_;
    MVCamera camera_;
    ArmorDetector detector_;
    FeatureDetectorONNX classifier_;
    PoseEstimator estimator_;

    // 输出与通信控制
    SerialPort serial_;
    ProtocolSender* sender_ = nullptr;

    // 线程与同步控制
    std::atomic<bool> is_running_{false};
    std::thread camera_thread_;
    std::thread process_thread_;
    std::thread monitor_thread_;

    std::mutex frame_mutex_;
    std::condition_variable frame_cv_;
    FrameData latest_frame_;
    bool new_frame_ = false;

    // 预分配模型对象以避免重复分配
    armor_model::Camera am_cam_;
    armor_model::VehicleModel am_model_;
    
    // 决策规划器
    rma::FireControlPlanner planner_;
    
    // 系统资源探测
    unsigned int num_cores_;
};

#endif // VISION_SYSTEM_CORE_PIPELINE_HPP
