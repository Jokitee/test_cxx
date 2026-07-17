// ============================================================
// fire_control_planner.cpp
// RoboMaster 自瞄决策规划器 — 实现
// ============================================================
#include "vision_system\decision\fire_control_planner.hpp"

#include <cmath>
#include <algorithm>
#include <cassert>

namespace rma {

// ============================================================
// 构造 / 调参
// ============================================================

FireControlPlanner::FireControlPlanner()
    : state_(TrackingState::SEARCHING),
      last_target_(-1),
      last_armor_(-1),
      last_yaw_(0.0),
      last_pitch_(0.0),
      stable_frames_(0),
      lambda_(0.20),
      base_thresh_(0.05),       // ≈2.9°
      min_lock_frames_(5),
      alpha_(0.30),
      max_rate_(10.0)           // rad/s
{}

void FireControlPlanner::setScoringWeights(const ScoringWeights& w)       { sw_ = w; }
void FireControlPlanner::setArmorSelectConfig(const ArmorSelectConfig& c) { ac_ = c; }
void FireControlPlanner::setFilterConfig(const FilterConfig& c)           { fc_ = c; }
void FireControlPlanner::setSwitchPenaltyLambda(double l)                 { lambda_ = l; }
void FireControlPlanner::setBaseAngleThreshold(double r)                  { base_thresh_ = r; }
void FireControlPlanner::setMinLockStableFrames(int n)                    { min_lock_frames_ = n; }
void FireControlPlanner::setSmoothingAlpha(double a)                      { alpha_ = a; }
void FireControlPlanner::setMaxAngularRate(double r)                      { max_rate_ = r; }

// ============================================================
// 工具函数
// ============================================================

double FireControlPlanner::wrapAngle(double a)
{
    // 将角度归一化到 (-π, π]
    while (a >  M_PI) a -= 2.0 * M_PI;
    while (a <= -M_PI) a += 2.0 * M_PI;
    return a;
}

double FireControlPlanner::angleDiff(double a, double b)
{
    // 最短角差，结果 ∈ (-π, π]
    return wrapAngle(a - b);
}

// ============================================================
// 主循环 (§1 + 整体调度)
// ============================================================

FireControlOutput FireControlPlanner::update(const PlannerInput& input)
{
    const double dt = input.dt;
    FireControlOutput out;

    // ---- §2 筛选 ----
    auto candidates = filterTargets(input);

    if (candidates.empty()) {
        out.state       = TrackingState::SEARCHING;
        out.target_yaw   = last_yaw_;
        out.target_pitch = last_pitch_;
        out.confidence   = 0.0;
        out.vehicle_id   = -1;
        out.armor_id     = -1;
        // 长时间丢失 → 角度平滑衰减到最后已知方向 (§6.4)
        last_target_    = -1;
        last_armor_     = -1;
        stable_frames_  = 0;
        state_          = TrackingState::SEARCHING;
        return out;
    }

    // ---- §3 评分 + 选择 ----
    auto scores    = scoreTargets(input, candidates);
    int  tgt_idx   = pickBestTarget(scores, last_target_);

    if (tgt_idx < 0 ||
        tgt_idx >= static_cast<int>(input.targets.size())) {
        out.state       = TrackingState::SEARCHING;
        out.target_yaw   = last_yaw_;
        out.target_pitch = last_pitch_;
        out.confidence   = 0.0;
        state_          = TrackingState::SEARCHING;
        return out;
    }

    const TargetState& target = input.targets[tgt_idx];

    // ---- §4 装甲板选择 ----
    int armor_idx = -1;
    if (target.model_state == ModelState::FULL) {
        armor_idx = selectArmorHandoff(target);  // 4.2
    } else {
        // MODEL_ROUGH / MODEL_NOT_BUILT → 退化为静止/慢速逻辑 (4.1)
        armor_idx = selectArmorStatic(target);
    }

    if (armor_idx < 0 ||
        armor_idx >= static_cast<int>(target.armors.size())) {
        // 无有效装甲板 → TRACKING 低置信度
        out.state       = TrackingState::TRACKING;
        out.target_yaw   = last_yaw_;
        out.target_pitch = last_pitch_;
        out.confidence   = 0.1;
        out.vehicle_id   = target.vehicle_id;
        out.armor_id     = -1;
        last_target_     = target.vehicle_id;
        stable_frames_   = 0;
        state_           = TrackingState::TRACKING;
        return out;
    }

    // ---- §5 弹道解算 (含飞行时间迭代) ----
    const Eigen::Vector3d aim_pos = target.armors[armor_idx].position;
    const double distance = (aim_pos - input.gimbal_position).norm();
    const double plate_angle = target.armors[armor_idx].relative_angle;

    BallisticSolution sol = solveBallistics(
        aim_pos, target.velocity,
        input.gimbal_position, input.ballistic_params);

    if (!sol.valid) {
        out.state       = TrackingState::TRACKING;
        out.target_yaw   = last_yaw_;
        out.target_pitch = last_pitch_;
        out.confidence   = 0.1;
        out.vehicle_id   = target.vehicle_id;
        out.armor_id     = armor_idx;
        last_target_     = target.vehicle_id;
        state_           = TrackingState::TRACKING;
        return out;
    }

    // ---- §6 状态机 + 置信度 ----
    TrackingState new_state =
        decideState(target, armor_idx, plate_angle, distance);

    // 装甲板刚发生衔接切换 → 不立刻跳到 LOCKED (§4.2 最后一条)
    if (target.vehicle_id == last_target_ && armor_idx != last_armor_) {
        if (new_state == TrackingState::LOCKED) {
            new_state    = TrackingState::TRACKING;
            stable_frames_ = 0;
        }
    }

    if (new_state == TrackingState::LOCKED) {
        ++stable_frames_;
    } else {
        stable_frames_ = 0;
    }
    // 连续稳定帧数不足 → 保持 TRACKING
    if (new_state == TrackingState::LOCKED &&
        stable_frames_ < min_lock_frames_) {
        new_state = TrackingState::TRACKING;
    }

    double conf = computeConfidence(
        new_state, plate_angle, distance, target.model_state);

    // ---- §6.5 平滑 ----
    double out_yaw   = rateLimitAngle(sol.yaw,   last_yaw_,   dt);
    out_yaw          = filterAngle(out_yaw, last_yaw_);

    double out_pitch = rateLimitAngle(sol.pitch, last_pitch_, dt);
    out_pitch        = filterAngle(out_pitch, last_pitch_);

    // ---- 输出 ----
    out.state        = new_state;
    out.target_yaw   = out_yaw;
    out.target_pitch = out_pitch;
    out.confidence   = conf;
    out.vehicle_id   = target.vehicle_id;
    out.armor_id     = armor_idx;
    out.solution     = sol;
    out.solution.yaw   = out_yaw;    // 平滑后的最终角度
    out.solution.pitch = out_pitch;

    // ---- 更新内部状态 ----
    state_       = new_state;
    last_target_ = target.vehicle_id;
    last_armor_  = armor_idx;
    last_yaw_    = out_yaw;
    last_pitch_  = out_pitch;

    return out;
}

// ============================================================
// §2 目标筛选
// ============================================================

std::vector<int> FireControlPlanner::filterTargets(
    const PlannerInput& input) const
{
    std::vector<int> result;
    for (int i = 0; i < static_cast<int>(input.targets.size()); ++i) {
        const TargetState& t = input.targets[i];

        // MODEL_NOT_BUILT：不参与评分，但继续跟踪（文档 §2 注释）
        if (t.model_state == ModelState::NOT_BUILT) continue;

        // 跟踪丢失超时
        if (t.frames_since_update > fc_.max_lost_frames) continue;

        // 协方差过大
        if (t.covariance.size() > 0 &&
            t.covariance.trace() > fc_.max_cov_trace)
            continue;

        // 超出有效射程
        double dist = (t.position - input.gimbal_position).norm();
        if (dist > fc_.max_range) continue;

        // 所有装甲板法向量均背对己方
        bool any_face_us = false;
        for (const auto& a : t.armors) {
            if (std::abs(a.relative_angle) < fc_.max_normal_angle) {
                any_face_us = true;
                break;
            }
        }
        if (!any_face_us && !t.armors.empty()) continue;

        // TODO: 己方射界机械限位 / 友军遮挡排除

        result.push_back(i);
    }
    return result;
}

// ============================================================
// §3 目标评分
// ============================================================

std::vector<TargetScoreDetail> FireControlPlanner::scoreTargets(
    const PlannerInput& input,
    const std::vector<int>& indices) const
{
    const size_t n = indices.size();
    if (n == 0) return {};

    std::vector<TargetScoreDetail> scores(n);
    std::vector<double> raw_dist(n), raw_conf(n);
    double max_raw_dist = 0.0, max_raw_conf = 0.0;

    // ---- 计算原始值 ----
    for (size_t k = 0; k < n; ++k) {
        const TargetState& t = input.targets[indices[k]];
        scores[k].target_idx = indices[k];
        scores[k].vehicle_id = t.vehicle_id;

        // S_dist 原始: 1/(d+ε)
        double d = (t.position - input.gimbal_position).norm();
        raw_dist[k] = 1.0 / (d + 1e-6);
        max_raw_dist = std::max(max_raw_dist, raw_dist[k]);

        // S_conf 原始: 1/(trace(P)+ε)
        double tr = t.covariance.size() > 0 ? t.covariance.trace() : 1e6;
        raw_conf[k] = 1.0 / (tr + 1e-6);
        max_raw_conf = std::max(max_raw_conf, raw_conf[k]);
    }

    // ---- 归一化 + 加权求和 ----
    for (size_t k = 0; k < n; ++k) {
        const TargetState& t = input.targets[indices[k]];

        // S_dist ∈ [0,1]
        scores[k].s_dist =
            (max_raw_dist > 0.0) ? raw_dist[k] / max_raw_dist : 0.0;

        // S_angle = 1 − min_armor_angle / (π/2)
        double min_ang = M_PI;
        for (const auto& a : t.armors) {
            min_ang = std::min(min_ang, std::abs(a.relative_angle));
        }
        scores[k].s_angle = std::max(0.0,
            std::min(1.0, 1.0 - min_ang / (M_PI / 2.0)));

        // S_conf ∈ [0,1]
        scores[k].s_conf =
            (max_raw_conf > 0.0) ? raw_conf[k] / max_raw_conf : 0.0;

        // S_threat: 默认 0，可由外部逻辑覆盖
        scores[k].s_threat = 0.0;

        // 综合得分
        scores[k].total = sw_.w_dist   * scores[k].s_dist
                        + sw_.w_angle  * scores[k].s_angle
                        + sw_.w_conf   * scores[k].s_conf
                        + sw_.w_threat * scores[k].s_threat;
    }
    return scores;
}

int FireControlPlanner::pickBestTarget(
    const std::vector<TargetScoreDetail>& scores,
    int current_vehicle_id) const
{
    if (scores.empty()) return -1;

    // 找到当前目标的得分
    double current_score = -1.0;
    for (const auto& s : scores) {
        if (s.vehicle_id == current_vehicle_id) {
            current_score = s.total;
            break;
        }
    }

    // 应用切换惩罚后选最优
    int    best_idx = -1;
    double best_adj = -std::numeric_limits<double>::max();

    for (const auto& s : scores) {
        double adjusted = s.total;
        // §3.2: 若当前目标仍可用，切换到新目标需扣 λ
        if (s.vehicle_id != current_vehicle_id && current_score > 0.0) {
            adjusted -= lambda_;
        }
        if (adjusted > best_adj) {
            best_adj = adjusted;
            best_idx = s.target_idx;
        }
    }
    return best_idx;
}

// ============================================================
// §4 装甲板选择
// ============================================================

/// §4.1 静止/慢速目标：选 relative_angle 最小（最正对）的板
int FireControlPlanner::selectArmorStatic(const TargetState& target) const
{
    if (target.armors.empty()) return -1;
    int best = 0;
    double best_ang = std::abs(target.armors[0].relative_angle);
    for (int i = 1; i < static_cast<int>(target.armors.size()); ++i) {
        double a = std::abs(target.armors[i].relative_angle);
        if (a < best_ang) {
            best_ang = a;
            best = i;
        }
    }
    return best;
}

/// §4.2 小陀螺目标：锁定 + 阈值触发的衔接式选择
int FireControlPlanner::selectArmorHandoff(const TargetState& target) const
{
    const auto& armors = target.armors;
    const int n = static_cast<int>(armors.size());
    if (n == 0) return -1;

    // 当前锁定的板（同车才有效）
    int cur = (target.vehicle_id == last_target_) ? last_armor_ : -1;
    if (cur < 0 || cur >= n) cur = -1;

    // 1. 当前板仍在稳定区间 → 继续锁定
    if (cur >= 0) {
        if (std::abs(armors[cur].relative_angle) < ac_.switch_threshold) {
            return cur;
        }
    }

    // 2. 当前板已超过切换阈值（或尚未锁定）：
    //    寻找"正在转向正对方向"且已进入有效角度范围内的候选板
    int    best = -1;
    double best_ang = std::numeric_limits<double>::max();

    for (int i = 0; i < n; ++i) {
        const ArmorPlate& a = armors[i];
        double abs_angle = std::abs(a.relative_angle);
        // 正在转入正对方向: angle 与 angle_rate 异号
        bool approaching = (a.relative_angle * a.angle_rate) < 0.0;
        if (approaching && abs_angle <= ac_.min_effective_angle) {
            if (abs_angle < best_ang) {
                best_ang = abs_angle;
                best = i;
            }
        }
    }
    if (best >= 0) return best;

    // 3. 无板满足衔接条件 → 保留当前板（只要未超过硬上限）
    if (cur >= 0 &&
        std::abs(armors[cur].relative_angle) <= ac_.min_effective_angle) {
        return cur;
    }

    return -1; // 确实无可用板
}

// ============================================================
// §5 弹道解算
// ============================================================

/// 预测位置（匀速模型，用于弹道-预测迭代耦合）
static Eigen::Vector3d predictCV(
    const Eigen::Vector3d& pos,
    const Eigen::Vector3d& vel,
    double t)
{
    return pos + vel * t;
}

BallisticSolution FireControlPlanner::solveBallistics(
    const Eigen::Vector3d& target_pos,
    const Eigen::Vector3d& target_vel,
    const Eigen::Vector3d& gimbal_pos,
    const BallisticParams& params) const
{
    BallisticSolution sol;

    // 附录 A: 飞行时间迭代耦合
    Eigen::Vector3d predicted = target_pos;
    double fly_time = 0.15; // 初始猜测

    for (int iter = 0; iter < 3; ++iter) {
        Eigen::Vector3d delta = predicted - gimbal_pos;
        double dist_xy = std::sqrt(delta.x() * delta.x() +
                                   delta.z() * delta.z());
        double height  = delta.y();

        double pitch = 0.0, t = 0.0;
        if (!solvePitch(dist_xy, height, params, pitch, t)) {
            return sol; // valid = false
        }
        fly_time = t;
        // §6.3: 假设立即开火，用飞行时间预测目标位置
        predicted = predictCV(target_pos, target_vel, fly_time);
    }

    // 最终解算
    Eigen::Vector3d final_delta = predicted - gimbal_pos;
    double final_dist_xy = std::sqrt(final_delta.x() * final_delta.x() +
                                      final_delta.z() * final_delta.z());
    // 简化：yaw 直接用 atan2
    sol.yaw      = std::atan2(final_delta.x(), final_delta.z());
    sol.fly_time = fly_time;

    // 重新计算最终 pitch（保持一致性）
    double p = 0.0, ft = 0.0;
    if (solvePitch(final_dist_xy, final_delta.y(), params, p, ft)) {
        sol.pitch = p;
        sol.fly_time = ft;
        sol.valid  = true;
    } else {
        // 若最终迭代失败，退回倒数第二次结果
        // (不太可能发生，但防御性编程)
        sol.valid = false;
    }
    return sol;
}

/// §5 + 附录 A: 给定水平距离和高度差，解算 pitch 和飞行时间
bool FireControlPlanner::solvePitch(
    double dist_xy, double height,
    const BallisticParams& params,
    double& out_pitch, double& out_time)
{
    const double v0 = params.bullet_speed;
    const double g  = params.gravity;
    const double k  = params.drag_k;

    // ---- 极近距离 ----
    if (dist_xy < 0.01) {
        out_pitch = std::atan2(height, 0.01);
        out_time  = 0.0;
        return true;
    }

    // ---- 无空气阻力：解析解 ----
    if (k < 1e-6) {
        // 令 u = tan(θ)，a = g·d²/(2·v0²)
        // 二次方程: a·u² − d·u + (h + a) = 0
        double a = g * dist_xy * dist_xy / (2.0 * v0 * v0);
        double B = dist_xy;                       // −B 项的系数
        double C = height + a;

        double disc = B * B - 4.0 * a * C;
        if (disc < 0.0) return false;             // 不可达

        double sd = std::sqrt(disc);
        double u1 = (B + sd) / (2.0 * a);
        double u2 = (B - sd) / (2.0 * a);

        double p1 = std::atan(u1);
        double p2 = std::atan(u2);
        // 选平射解（绝对值更小的 pitch）
        out_pitch = (std::abs(p1) < std::abs(p2)) ? p1 : p2;

        double cp = std::cos(out_pitch);
        if (std::abs(cp) < 1e-9) return false;
        out_time = dist_xy / (v0 * cp);
        return true;
    }

    // ---- 有空气阻力：数值二分法 ----
    // x(t) = (vx0/k)·(1 − e^{−kt})
    // y(t) = (vy0 + g/k)/k · (1 − e^{−kt}) − g·t/k
    auto height_at_pitch = [&](double pitch) -> double {
        double vx0 = v0 * std::cos(pitch);
        double vy0 = v0 * std::sin(pitch);
        double ratio = dist_xy * k / vx0;
        if (ratio >= 1.0 || ratio < 0.0) return -1e9; // 弹丸到不了
        double t   = -std::log(1.0 - ratio) / k;
        double ekt = std::exp(-k * t);
        return (vy0 + g / k) / k * (1.0 - ekt) - g * t / k;
    };

    double lo = -M_PI / 4.0;
    double hi =  M_PI / 4.0;
    double y_lo = height_at_pitch(lo);
    double y_hi = height_at_pitch(hi);

    // 解不在搜索区间内
    if ((y_lo - height) * (y_hi - height) > 0.0) return false;

    for (int i = 0; i < 64; ++i) {
        double mid = (lo + hi) * 0.5;
        double y_m = height_at_pitch(mid);
        if (y_m < height) lo = mid;
        else              hi = mid;
    }

    out_pitch = (lo + hi) * 0.5;

    // 反算飞行时间
    double vx0  = v0 * std::cos(out_pitch);
    double ratio = dist_xy * k / vx0;
    if (ratio >= 1.0) return false;
    out_time = -std::log(1.0 - ratio) / k;
    return true;
}

// ============================================================
// §6 持续瞄准正确性保证
// ============================================================

TrackingState FireControlPlanner::decideState(
    const TargetState& target, int armor_idx,
    double plate_angle, double distance) const
{
    if (armor_idx < 0) return TrackingState::TRACKING;

    // §1.1: MODEL_ROUGH 不允许进入 LOCKED
    if (target.model_state == ModelState::ROUGH) {
        return TrackingState::TRACKING;
    }

    // 协方差收敛
    double tr = target.covariance.size() > 0
                    ? target.covariance.trace()
                    : 1e6;
    bool cov_ok = (tr < fc_.max_cov_trace);

    // §6.2: 角度在动态阈值内
    double thresh  = angleThreshold(distance);
    bool   ang_ok  = (std::abs(plate_angle) < thresh);

    if (cov_ok && ang_ok) return TrackingState::LOCKED;
    return TrackingState::TRACKING;
}

/// §6.2 角度容忍阈值随距离衰减
/// threshold(d) = base × (min_ref / d)
double FireControlPlanner::angleThreshold(double distance) const
{
    const double min_ref = 1.0;  // 1 m 参考距离
    double raw = base_thresh_ * (min_ref / std::max(distance, 0.1));
    return std::min(raw, M_PI / 4.0);  // 上限 45°
}

/// 锁定置信度
double FireControlPlanner::computeConfidence(
    TrackingState state, double plate_angle,
    double distance, ModelState ms) const
{
    if (state == TrackingState::SEARCHING) return 0.0;

    // 基础值
    double base = (state == TrackingState::LOCKED) ? 0.90 : 0.40;

    // §1.1: 粗略模型打折
    if (ms == ModelState::ROUGH) base *= 0.60;

    // 装甲板倾斜度越大，置信度越低
    double angle_factor =
        1.0 - std::min(1.0, std::abs(plate_angle) / ac_.min_effective_angle);

    // 距离衰减
    double dist_factor = 1.0 / (1.0 + distance * 0.10);

    return std::max(0.0,
        std::min(1.0, base * angle_factor * (0.5 + 0.5 * dist_factor)));
}

// ============================================================
// §6.5 平滑约束
// ============================================================

/// 一阶低通滤波（带角度环绕处理）
double FireControlPlanner::filterAngle(double raw, double prev) const
{
    double delta = angleDiff(raw, prev);
    return wrapAngle(prev + alpha_ * delta);
}

/// 角速度变化率限幅
double FireControlPlanner::rateLimitAngle(
    double raw, double prev, double dt) const
{
    double delta    = angleDiff(raw, prev);
    double max_d    = max_rate_ * dt;
    if (delta >  max_d) delta =  max_d;
    if (delta < -max_d) delta = -max_d;
    return wrapAngle(prev + delta);
}

} // namespace rma