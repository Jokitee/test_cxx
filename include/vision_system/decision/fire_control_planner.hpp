// ============================================================
// fire_control_planner.hpp
// RoboMaster 自瞄决策规划器 — 头文件
// C++11 | 坐标系: x-右 y-上 z-前
// ============================================================
#pragma once

#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <cstdint>
#include <limits>
#include <algorithm>

namespace rma {

// ======================== 枚举 ========================

enum class ModelState : uint8_t {
    NOT_BUILT = 0,   // 观测帧数不足，无可用估计
    ROUGH     = 1,   // 粗略 EKF 已收敛（单板 CV/CA 模型）
    FULL      = 2    // 完整小陀螺模型已收敛
};

enum class TrackingState : uint8_t {
    SEARCHING = 0,   // 无可用目标
    TRACKING  = 1,   // 有目标但置信度不足
    LOCKED    = 2    // 稳定锁定
};

// ======================== 数据结构 ========================

/// 单块装甲板观测
struct ArmorPlate {
    int           id             = 0;
    Eigen::Vector3d position     = Eigen::Vector3d::Zero();   // 世界坐标
    Eigen::Vector3d normal       = Eigen::Vector3d::UnitX();  // 向外法向量
    double        relative_angle = 0.0;  // 法向量与视线夹角 (rad)，0 = 正对
    double        angle_rate     = 0.0;  // d(relative_angle)/dt (rad/s)
};

/// 单个目标的 EKF 状态
struct TargetState {
    int        vehicle_id   = -1;
    ModelState model_state  = ModelState::NOT_BUILT;

    // ROUGH / FULL 均可用
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    Eigen::Vector3d velocity = Eigen::Vector3d::Zero();

    // 仅 FULL 模型
    Eigen::Vector3d angular_velocity = Eigen::Vector3d::Zero();
    double          rotation_radius  = 0.0;
    Eigen::MatrixXd covariance;       // 动态尺寸

    std::vector<ArmorPlate> armors;

    int frames_since_update   = 999;
    int total_observed_frames = 0;
};

/// 弹道参数
struct BallisticParams {
    double bullet_speed = 25.0;  // m/s
    double gravity      = 9.81;  // m/s^2
    double drag_k       = 0.0;   // 线性空气阻力系数 (0 = 真空)
};

/// 评分权重
struct ScoringWeights {
    double w_dist   = 0.40;
    double w_angle  = 0.35;
    double w_conf   = 0.20;
    double w_threat = 0.05;
};

/// 装甲板选择参数
struct ArmorSelectConfig {
    double switch_threshold    = 0.35;   // rad — 超过此值开始寻找下一块板
    double min_effective_angle = 0.70;   // rad — 硬性上限
};

/// 筛选阈值
struct FilterConfig {
    int    max_lost_frames  = 10;
    double max_cov_trace    = 100.0;
    double max_range        = 15.0;      // m
    double max_normal_angle = M_PI / 2.0;
};

/// 弹道解算结果
struct BallisticSolution {
    double yaw      = 0.0;
    double pitch    = 0.0;
    double fly_time = 0.0;
    bool   valid    = false;
};

/// 规划器输出（每帧）
struct FireControlOutput {
    TrackingState     state       = TrackingState::SEARCHING;
    double            target_yaw   = 0.0;
    double            target_pitch = 0.0;
    double            confidence   = 0.0;
    int               vehicle_id   = -1;
    int               armor_id     = -1;
    BallisticSolution solution;
};

/// 规划器输入
struct PlannerInput {
    std::vector<TargetState> targets;
    Eigen::Vector3d  gimbal_position;        // 世界坐标
    double           gimbal_yaw   = 0.0;     // rad
    double           gimbal_pitch = 0.0;     // rad
    BallisticParams  ballistic_params;
    double           remaining_heat = 0.0;
    double           remaining_ammo = 0.0;
    double           dt = 1.0 / 120.0;       // 帧周期 (s)
};

// ======================== 内部评分详情 ========================

struct TargetScoreDetail {
    int    target_idx;   // PlannerInput::targets 下标
    int    vehicle_id;
    double total;
    double s_dist;
    double s_angle;
    double s_conf;
    double s_threat;
};

// ======================== 规划器主类 ========================

class FireControlPlanner {
public:
    FireControlPlanner();

    /// 每帧调用，返回云台指令 + 元数据
    FireControlOutput update(const PlannerInput& input);

    // ---- 调参接口 ----
    void setScoringWeights(const ScoringWeights& w);
    void setArmorSelectConfig(const ArmorSelectConfig& c);
    void setFilterConfig(const FilterConfig& c);
    void setSwitchPenaltyLambda(double lambda);
    void setBaseAngleThreshold(double rad);
    void setMinLockStableFrames(int n);
    void setSmoothingAlpha(double alpha);        // 0 = 无平滑, 1 = 完全保持
    void setMaxAngularRate(double rad_per_sec);   // 角速度变化率限幅

private:
    // ---- §2 目标筛选 ----
    std::vector<int> filterTargets(const PlannerInput& input) const;

    // ---- §3 目标评分 ----
    std::vector<TargetScoreDetail> scoreTargets(
        const PlannerInput& input,
        const std::vector<int>& indices) const;
    int pickBestTarget(
        const std::vector<TargetScoreDetail>& scores,
        int current_vehicle_id) const;

    // ---- §4 装甲板选择 ----
    int selectArmorStatic(const TargetState& target) const;       // 4.1
    int selectArmorHandoff(const TargetState& target) const;      // 4.2

    // ---- §5 弹道解算 ----
    BallisticSolution solveBallistics(
        const Eigen::Vector3d& target_pos,
        const Eigen::Vector3d& target_vel,
        const Eigen::Vector3d& gimbal_pos,
        const BallisticParams& params) const;
    static bool solvePitch(
        double dist_xy, double height,
        const BallisticParams& params,
        double& out_pitch, double& out_time);

    // ---- §6 持续瞄准正确性保证 ----
    TrackingState decideState(
        const TargetState& target, int armor_idx,
        double plate_angle, double distance) const;
    double angleThreshold(double distance) const;
    double computeConfidence(
        TrackingState state, double plate_angle,
        double distance, ModelState ms) const;

    // ---- §6.5 平滑 ----
    double filterAngle(double raw, double prev) const;
    double rateLimitAngle(double raw, double prev, double dt) const;

    // ---- 工具函数 ----
    static double wrapAngle(double a);
    static double angleDiff(double a, double b);

    // ---- 内部状态 ----
    TrackingState state_;
    int    last_target_;      // 上一帧目标 vehicle_id
    int    last_armor_;       // 上一帧装甲板 idx
    double last_yaw_;
    double last_pitch_;
    int    stable_frames_;    // LOCKED 稳定帧计数

    // ---- 配置 ----
    ScoringWeights    sw_;
    ArmorSelectConfig ac_;
    FilterConfig      fc_;
    double lambda_;           // 目标切换惩罚 λ
    double base_thresh_;      // 基础角度容忍阈值 (rad)
    int    min_lock_frames_;  // 进入 LOCKED 所需最小稳定帧数
    double alpha_;            // 低通滤波系数
    double max_rate_;         // 最大角速度变化率 (rad/s)
};

} // namespace rma