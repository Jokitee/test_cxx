#ifndef VISION_SYSTEM_DECISION_VEHICLE_MANAGER_HPP
#define VISION_SYSTEM_DECISION_VEHICLE_MANAGER_HPP

#include "vision_system/core/types.hpp"
#include "vision_system/decision/tracker.hpp"
#include "vision_system/decision/armor_model.hpp"
#include "vision_system/decision/pnp_estimator.hpp"
#include "vision_system/decision/ekf.hpp"
#include <map>
#include <string>
#include <vector>

// 代表战场上单个车辆实体的状态节点
class VehicleNode {
public:
    VehicleNode(const std::string& id, const armor_model::Camera& cam);
    
    // 订阅当前帧的装甲板消息
    void addArmor(const lightbors& armor);
    
    // 执行当前帧的内部模型迭代与位姿更新
    void processFrame(int64_t timestamp, PoseEstimator& estimator);

    std::string vehicle_id;
    VehicleArmors armors_buffer;
    
    // 该车辆持有的独立模型与优化器实例，保留历史参数状态，实现模型迭代合理化
    armor_model::ModelOptimizer optimizer;
    
    armor_model::Pose current_pose;
    bool is_tracking = false;
    
    armor_model::FrameObservation latest_obs;
    armor_model::OptimizationResult latest_opt_result;

    // EKF 状态估计器
    vision_system::ExtendedKalmanFilter ekf;
    int64_t last_timestamp = 0;
    bool ekf_initialized = false;
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
    armor_model::Camera am_cam_;
    PoseEstimator& estimator_;
};

#endif // VISION_SYSTEM_DECISION_VEHICLE_MANAGER_HPP
