#ifndef VISION_SYSTEM_DECISION_VEHICLE_MANAGER_HPP
#define VISION_SYSTEM_DECISION_VEHICLE_MANAGER_HPP

#include "vision_system/core/types.hpp"
#include "vision_system/decision/tracker.hpp"
#include "vision_system/decision/armor_model.hpp"
#include "vision_system/decision/pnp_estimator.hpp"
#include "vision_system/decision/ekf.hpp"
#include "vision_system/decision/vehicle_tracker.hpp"
#include <map>
#include <string>
#include <vector>
#include <memory>

// 代表战场上单个车辆实体的状态节点
class VehicleNode {
public:
    VehicleNode(const std::string& id, const armor_model::Camera& cam);
    
    // 订阅当前帧的装甲板消息
    void addArmor(const lightbors& armor);
    
    // 执行当前帧的内部模型迭代与位姿更新
    void processFrame(int64_t timestamp, PoseEstimator& estimator, vision_system::SavedVehicleParams& saved_params);

    std::string vehicle_id;
    VehicleArmors armors_buffer;
    
    // 缓存相机参数，供 3D 到 2D 投影关联使用
    armor_model::Camera cam_; 
    
    // 全新的 11维 EKF 目标跟踪器，取代原先的单帧 LM 优化器和 6D 滤波器
    std::unique_ptr<vision_system::VehicleTracker> tracker;
    
    armor_model::Pose current_pose;
    bool is_tracking = false;
    
    armor_model::FrameObservation latest_obs;

    int64_t last_timestamp = 0;
    bool tracker_initialized = false;
    
    // 缓存上一次滤波后的角点，key为 plate_id (0-3)
    std::map<int, std::array<cv::Point2f, 4>> filtered_corners_;
    // 记录每个 plate_id 上一次更新的时间戳，防止很久没更新时发生突变
    std::map<int, int64_t> last_corner_update_time_;

    // 为了兼容旧代码提供一个 vehicle model
    armor_model::VehicleModel getVehicleModel() const;
};

// 全局车辆管理器
class VehicleManager {
public:
    VehicleManager(const armor_model::Camera& cam, PoseEstimator& estimator);
    
    // 统一订阅检测模块输出，处理并分发当前帧的所有装甲板
    void update(const std::vector<lightbors>& armors, int64_t timestamp);
    
    std::map<std::string, VehicleNode>& getNodes() { return nodes_; }

private:
    std::map<std::string, VehicleNode> nodes_;
    std::map<std::string, vision_system::SavedVehicleParams> saved_params_;
    armor_model::Camera am_cam_;
    PoseEstimator& estimator_;
};

#endif // VISION_SYSTEM_DECISION_VEHICLE_MANAGER_HPP
