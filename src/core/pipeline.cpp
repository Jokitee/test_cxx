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
    // 初始化规划器配置
    auto p_cfg = config_.getPlannerConfig();
    rma::ScoringWeights sw;
    sw.w_dist   = p_cfg.w_dist;
    sw.w_angle  = p_cfg.w_angle;
    sw.w_conf   = p_cfg.w_conf;
    sw.w_threat = p_cfg.w_threat;
    rma::ArmorSelectConfig ac;
    ac.switch_threshold    = p_cfg.switch_threshold;
    ac.min_effective_angle = p_cfg.min_effective_angle;
    rma::FilterConfig fc;
    fc.max_lost_frames  = p_cfg.max_lost_frames;
    fc.max_cov_trace    = p_cfg.max_cov_trace;
    fc.max_range        = p_cfg.max_range;
    fc.max_normal_angle = p_cfg.max_normal_angle;
    planner_.setScoringWeights(sw);
    planner_.setArmorSelectConfig(ac);
    planner_.setFilterConfig(fc);
    planner_.setSwitchPenaltyLambda(p_cfg.lambda);
    planner_.setBaseAngleThreshold(p_cfg.base_thresh);
    planner_.setMinLockStableFrames(p_cfg.min_lock_frames);
    planner_.setSmoothingAlpha(p_cfg.alpha);
    planner_.setMaxAngularRate(p_cfg.max_rate);

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

        // ============================================
        // 4. 构建 PlannerInput 并调用 FireControlPlanner
        // ============================================
        rma::PlannerInput planner_input;
        planner_input.gimbal_position = Eigen::Vector3d(0.0, 0.0, 0.0); 
        planner_input.gimbal_yaw = 0.0;
        planner_input.gimbal_pitch = 0.0;
        
        static int64_t last_time = 0;
        if (last_time > 0) {
            planner_input.dt = (current_frame.timestamp - last_time) / 1000.0;
        } else {
            planner_input.dt = 1.0 / 120.0; 
        }
        last_time = current_frame.timestamp;
        
        planner_input.ballistic_params.bullet_speed = 17.0;
        planner_input.ballistic_params.drag_k = 0.0;

        Eigen::Vector3d cam_offset(0.0, -0.40, 0.10);
        armor_model::Pose T_cam_world = CoordinateTransformer::calcCameraToWorld(0.0f, 0.0f, cam_offset);

        for (auto& pair : vehicle_manager.getNodes()) {
            VehicleNode& node = pair.second;
            if (!node.is_tracking || node.latest_obs.armors.empty() || !node.tracker) continue;

            rma::TargetState ts;
            try {
                ts.vehicle_id = std::stoi(node.vehicle_id);
            } catch(...) {
                ts.vehicle_id = 1; // 默认 fallback
            }
            ts.frames_since_update = 0; 
            ts.total_observed_frames = node.tracker->update_count_;
            ts.model_state = rma::ModelState::FULL; 

            Eigen::VectorXd x = node.tracker->getEKFState();
            Eigen::MatrixXd P = node.tracker->getEKFCovariance();
            ts.covariance = P;

            Eigen::Vector3d xyz_pseudo(x(0), x(2), x(4)); 
            Eigen::Vector3d v_pseudo(x(1), x(3), x(5));
            
            Eigen::Vector3d xyz_cam(-xyz_pseudo.y(), -xyz_pseudo.z(), xyz_pseudo.x());
            Eigen::Vector3d v_cam(-v_pseudo.y(), -v_pseudo.z(), v_pseudo.x());

            Eigen::Vector3d xyz_world_cv = T_cam_world * xyz_cam;
            Eigen::Vector3d v_world_cv = T_cam_world.linear() * v_cam;

            // Planner Coordinate System: X Right, Y Up, Z Forward
            ts.position = Eigen::Vector3d(xyz_world_cv.x(), -xyz_world_cv.y(), xyz_world_cv.z());
            ts.velocity = Eigen::Vector3d(v_world_cv.x(), -v_world_cv.y(), v_world_cv.z());

            ts.angular_velocity = Eigen::Vector3d(0, -x(7), 0); 
            ts.rotation_radius = x(8);

            int armor_num = 4;
            for (int i = 0; i < armor_num; ++i) {
                rma::ArmorPlate ap;
                ap.id = i;
                
                Eigen::Vector3d a_pseudo = node.tracker->getArmorXYZ(x, i);
                Eigen::Vector3d a_cam(-a_pseudo.y(), -a_pseudo.z(), a_pseudo.x());
                Eigen::Vector3d a_world_cv = T_cam_world * a_cam;
                ap.position = Eigen::Vector3d(a_world_cv.x(), -a_world_cv.y(), a_world_cv.z());

                double armor_yaw = x(6) + i * M_PI / 2.0;
                Eigen::Vector3d n_pseudo(std::cos(armor_yaw), std::sin(armor_yaw), 0);
                Eigen::Vector3d n_cam(-n_pseudo.y(), -n_pseudo.z(), n_pseudo.x());
                Eigen::Vector3d n_world_cv = T_cam_world.linear() * n_cam;
                ap.normal = Eigen::Vector3d(n_world_cv.x(), -n_world_cv.y(), n_world_cv.z());

                Eigen::Vector3d los = (planner_input.gimbal_position - ap.position).normalized();
                double cos_theta = ap.normal.dot(los);
                cos_theta = std::max(-1.0, std::min(1.0, cos_theta));
                ap.relative_angle = std::acos(cos_theta);
                
                ap.angle_rate = x(7); 
                ts.armors.push_back(ap);
            }
            planner_input.targets.push_back(ts);
        }

        rma::FireControlOutput planner_out = planner_.update(planner_input);

        if (sender_ != nullptr && config_.getModulesConfig().enable_serial) {
            if (planner_out.state != rma::TrackingState::SEARCHING) {
                float yaw = static_cast<float>(planner_out.target_yaw * 180.0 / M_PI);
                float pitch = static_cast<float>(planner_out.target_pitch * 180.0 / M_PI);
                sender_->sendTarget(yaw, pitch, 0, 0);
            }
        }

        // 5. 渲染三维框 (剥离了原有的解算代码)
        if (config_.getUIConfig().draw_3d_box || sender_ != nullptr) {
            for (auto& pair : vehicle_manager.getNodes()) {
                VehicleNode& node = pair.second;
                if (node.is_tracking && !node.latest_obs.armors.empty()) {


                    if (config_.getUIConfig().draw_3d_box) {
                        current_frame.image = armor_model::ModelVisualizer::render3DView(
                            node.getVehicleModel(), node.current_pose, am_cam_, current_frame.image, &node.latest_obs);
                    }
                }
            }
        }

        // 5. 渲染基础 2D 文本 ID 标签、装甲板 2D 边框以及检测到的所有灯条
        if (config_.getUIConfig().draw_lightbars) {
            for (const auto& rect : detector_.getLightbars()) {
                cv::Point2f pts[4];
                rect.points(pts);
                for (int i = 0; i < 4; i++) {
                    cv::line(current_frame.image, pts[i], pts[(i + 1) % 4], cv::Scalar(255, 0, 255), 2);
                }
            }
        }

        if (config_.getUIConfig().draw_2d_id) {
            for (auto& a : armors) {
                // 前置条件：灯条配对成功 且 数字识别有效（非 unknown、非空）
                bool is_valid_armor = a.righting &&
                                      !a.ID.empty() &&
                                      a.ID != "unknown";
                if (!is_valid_armor) continue;

                // 画出有效 armor 的 2D 边框
                for (int i = 0; i < 4; i++) {
                    cv::line(current_frame.image, a.armor_point[i], a.armor_point[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);
                }
                // 绘制 ID 标签
                cv::putText(current_frame.image, a.ID, a.armor_point[0], cv::FONT_HERSHEY_SIMPLEX, 3.0, cv::Scalar(255, 0, 0), 3);
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
