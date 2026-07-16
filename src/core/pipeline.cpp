/**
 * @brief 视觉系统多线程调度管线
 * @author jokit
 * @date 2026-07-15
 */
#include "vision_system/core/pipeline.hpp"
#include "vision_system/core/logger.hpp"
#include "vision_system/decision/tracker.hpp"
#include "vision_system/decision/vehicle_manager.hpp"
#include "vision_system/processing/image_processor.hpp"
#include <iostream>
#include <chrono>

Pipeline::Pipeline(const std::string& config_path) 
    : config_(config_path), 
      detector_(config_.getDetectorConfig()),
      estimator_("asset/camera_calibration.yml"),
      am_cam_(estimator_.cameraMatrix(), estimator_.distCoeffs(), 1280, 720),
      am_model_(0.300, 0.150, 15.0)
{
    // ==========================================
    // 1. 系统核心探测与多线程资源分配调度
    // ==========================================
    num_cores_ = std::thread::hardware_concurrency();
    Logger::log(LogLevel::INFO, ErrorCode::SUCCESS, 
                "Hardware Concurrency (Cores/Threads) detected: " + std::to_string(num_cores_));

    // 根据硬件核心数合理分配 OpenCV 的底层运算线程（比如保留 2 个核心给操作系统和相机取帧）
    int cv_threads = std::max(1, static_cast<int>(num_cores_) - 2); 
    cv::setNumThreads(cv_threads);
    Logger::log(LogLevel::INFO, ErrorCode::SUCCESS, 
                "Allocated CPU threads for OpenCV processing: " + std::to_string(cv_threads));

    // 设置初始位姿的世界坐标（此部分可扩展为对接 IMU 或云台姿态）
    cv::Mat R_wc = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat t_wc = cv::Mat::zeros(3, 1, CV_64F);
    estimator_.setWorldPose(R_wc, t_wc);
}

Pipeline::~Pipeline() {
    stop();
}

void Pipeline::start() {
    if (!config_.isLoaded()) {
        Logger::log(LogLevel::ERROR, ErrorCode::CONFIG_LOAD_FAILED, "Pipeline setup aborted due to config error.");
        return;
    }

    if (!camera_.init(config_.getCameraConfig())) {
        Logger::log(LogLevel::ERROR, ErrorCode::CAMERA_INIT_FAILED, "Pipeline setup aborted due to camera error.");
        return;
    }

    // 尝试初始化串口通信 (非致命错误，即使失败也可无头/单机运行)
    if (serial_.init(config_.getSerialConfig().port_name, config_.getSerialConfig().baud_rate)) {
        sender_ = new ProtocolSender(serial_);
    } else {
        Logger::log(LogLevel::WARNING, ErrorCode::SUCCESS, "Serial port not connected, running without serial output.");
    }

    is_running_ = true;
    // 启动经典的“生产者-消费者”双线程流水线模型以及相机监控线程
    camera_thread_ = std::thread(&Pipeline::cameraLoop, this);
    process_thread_ = std::thread(&Pipeline::processLoop, this);
    monitor_thread_ = std::thread(&Pipeline::monitorLoop, this);
    
    Logger::log(LogLevel::INFO, ErrorCode::SUCCESS, "Vision Pipeline Threads started successfully.");
}

void Pipeline::stop() {
    if (!is_running_) return;
    
    is_running_ = false;
    frame_cv_.notify_all(); // 唤醒所有可能在等待的线程
    
    if (camera_thread_.joinable()) camera_thread_.join();
    if (process_thread_.joinable()) process_thread_.join();
    if (monitor_thread_.joinable()) monitor_thread_.join();
    
    if (sender_) {
        delete sender_;
        sender_ = nullptr;
    }
    serial_.close();
    
    Logger::log(LogLevel::INFO, ErrorCode::SUCCESS, "Vision Pipeline fully stopped.");
}

void Pipeline::monitorLoop() {
    // 监视器线程：定期检测相机物理连接状态并自动重连
    while (is_running_) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        
        if (!camera_.checkConnection()) {
            Logger::log(LogLevel::WARNING, ErrorCode::CAMERA_DISCONNECTED, "Camera connection lost. Attempting to reconnect...");
            
            if (camera_.reconnect()) {
                Logger::log(LogLevel::INFO, ErrorCode::SUCCESS, "Camera successfully reconnected.");
            } else {
                Logger::log(LogLevel::ERROR, ErrorCode::CAMERA_INIT_FAILED, "Camera reconnect failed.");
            }
        }
    }
}

void Pipeline::cameraLoop() {
    // 生产者线程：负责毫无延迟地从相机硬件死循环取图
    while (is_running_) {
        cv::Mat frame;
        if (camera_.read(frame)) {
            std::lock_guard<std::mutex> lock(frame_mutex_);
            latest_frame_.image = frame;
            auto now = std::chrono::steady_clock::now();
            latest_frame_.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
            new_frame_ = true;
            frame_cv_.notify_one(); // 通知处理线程
        }
    }
}

void Pipeline::processLoop() {
    // 初始化车辆管理订阅器 (Vehicle Manager)
    VehicleManager vehicle_manager(am_cam_, estimator_);

    while (is_running_) {
        FrameData current_frame;
        {
            // 线程同步，取走最新的一帧
            std::unique_lock<std::mutex> lock(frame_mutex_);
            frame_cv_.wait(lock, [this] { return new_frame_ || !is_running_; });
            if (!is_running_) break;
            current_frame = latest_frame_;
            new_frame_ = false; // 清除标志位，避免重复处理
        }

        if (current_frame.image.empty()) continue;

        // ============================================
        // 核心流程：调度各解耦模块完成一条完整的识别管线
        // ============================================

        std::vector<lightbors> armors;

        // 1. ArmorFinder: 传统视觉提取装甲板 ROI
        if (config_.getModulesConfig().enable_detector) {
            armors = detector_.detect(current_frame.image);
        }

        // 2. Classifier: 神经网络字符识别
        if (config_.getModulesConfig().enable_classifier && !armors.empty()) {
            for (auto& a : armors) {
                cv::Mat ROI;
                CropAndResize(a, current_frame.image, ROI);
                a.ID = classifier_.detect(ROI, a);
            }
        } else {
            // 如果禁用分类器，为测试方便，将所有灯条强行认为是 ID "1"
            for (auto& a : armors) {
                a.ID = "1";
            }
        }

        // 3. 消息发布: 丢给车辆管理器进行跟踪、聚合与内部模型迭代
        if (config_.getModulesConfig().enable_pnp) {
            vehicle_manager.update(armors, current_frame.timestamp);
        }

        // 4. 获取最新的车辆追踪状态并渲染三维框
        if (config_.getUIConfig().draw_3d_box || sender_ != nullptr) {
            for (auto& pair : vehicle_manager.getNodes()) {
                VehicleNode& node = pair.second;
                // 只有当这个车辆在当前帧依然被跟踪且成功解算，才进行操作
                if (node.is_tracking && !node.latest_obs.armors.empty()) {
                    
                    // --- 串口发送控制指令与坐标系变换 ---
                    if (sender_ != nullptr && config_.getModulesConfig().enable_serial) {
                        // 1. 目标在相机坐标系下的 3D 位置
                        Eigen::Vector3d pos_cam = node.current_pose.translation();
                        
                        // 2. 测试阶段：固定相机在载车的上方 40cm，前 10cm。无偏航俯仰。
                        // 遵循 OpenCV 坐标系：X 右，Y 下，Z 前。
                        // 上方 40cm -> Y = -0.40m
                        // 前方 10cm -> Z = 0.10m
                        Eigen::Vector3d cam_offset(0.0, -0.40, 0.10);
                        
                        // 3. 实时计算当前相机到载车坐标系的变换矩阵 T_cam_world
                        armor_model::Pose T_cam_world = CoordinateTransformer::calcCameraToWorld(0.0f, 0.0f, cam_offset);
                        
                        // 4. 将目标坐标点转换到载车底盘(世界)坐标系
                        Eigen::Vector3d pos_world = T_cam_world * pos_cam;
                        
                        // 5. 根据车身坐标系提取真实解算的偏航角和俯仰角
                        double x = pos_world.x();
                        double y = pos_world.y();
                        double z = pos_world.z();
                        
                        float yaw = static_cast<float>(atan2(x, z) * 180.0 / CV_PI);
                        float pitch = static_cast<float>(atan2(-y, z) * 180.0 / CV_PI);
                        
                        sender_->sendTarget(yaw, pitch, 0, 0);
                    }

                    if (config_.getUIConfig().draw_3d_box) {
                        current_frame.image = armor_model::ModelVisualizer::render3DView(
                            node.optimizer.getModel(), node.current_pose, am_cam_, current_frame.image, &node.latest_obs);
                    }
                }
            }
        }

        // 5. 渲染基础 2D 文本 ID 标签、装甲板 2D 边框以及检测到的所有灯条
        if (config_.getUIConfig().draw_lightbars) {
            for (const auto& rect : node.detector.getLightbars()) {
                cv::Point2f pts[4];
                rect.points(pts);
                for (int i = 0; i < 4; i++) {
                    cv::line(current_frame.image, pts[i], pts[(i + 1) % 4], cv::Scalar(255, 0, 255), 2);
                }
            }
        }

        if (config_.getUIConfig().draw_2d_id) {
            for (auto& a : armors) {
                // 画出所有画面中检测配对成功的 armor
                for (int i = 0; i < 4; i++) {
                    cv::line(current_frame.image, a.armor_point[i], a.armor_point[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);
                }
                
                if (a.righting) {
                    cv::putText(current_frame.image, a.ID, a.armor_point[0], cv::FONT_HERSHEY_SIMPLEX, 3.0, cv::Scalar(255,0,0), 3);
                }
            }
        }

        // 6. UI 可视化输出
        if (config_.getUIConfig().show_window) {
            cv::imshow("Tracking (Consumer Thread)", current_frame.image);
            if (cv::waitKey(1) == 27) { // 监测到按 ESC 退出
                is_running_ = false; 
            }
        } else {
            // 无头模式下，避免线程空转过快占用 100% CPU，由于使用了条件变量通知，这里可以直接忽略
            // waitKey(1) 仅在有 UI 时需要用于刷新 GUI 事件循环
        }
    }
}
