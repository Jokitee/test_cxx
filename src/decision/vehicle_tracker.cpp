#include "vision_system/decision/vehicle_tracker.hpp"
#include <cmath>

namespace vision_system {

double VehicleTracker::limit_rad(double angle) {
    while (angle > CV_PI) angle -= 2.0 * CV_PI;
    while (angle < -CV_PI) angle += 2.0 * CV_PI;
    return angle;
}

Eigen::Vector3d VehicleTracker::xyz2ypd(const Eigen::Vector3d& xyz) {
    double yaw = std::atan2(xyz[1], xyz[0]);
    double dist = std::sqrt(xyz[0] * xyz[0] + xyz[1] * xyz[1]);
    double pitch = std::atan2(xyz[2], dist);
    return Eigen::Vector3d(yaw, pitch, std::sqrt(dist * dist + xyz[2] * xyz[2])); // [yaw, pitch, distance]
}

Eigen::Matrix3d VehicleTracker::xyz2ypd_jacobian(const Eigen::Vector3d& xyz) {
    double x = xyz[0], y = xyz[1], z = xyz[2];
    double x2y2 = x * x + y * y;
    double d2 = x2y2;
    double d = std::sqrt(d2);
    double dist2 = x2y2 + z * z;
    double dist = std::sqrt(dist2);

    Eigen::Matrix3d J;
    // Yaw partial derivatives
    J(0, 0) = -y / d2;
    J(0, 1) =  x / d2;
    J(0, 2) =  0.0;

    // Pitch partial derivatives
    J(1, 0) = -(x * z) / (dist2 * d);
    J(1, 1) = -(y * z) / (dist2 * d);
    J(1, 2) = d / dist2;

    // Distance partial derivatives
    J(2, 0) = x / dist;
    J(2, 1) = y / dist;
    J(2, 2) = z / dist;
    
    return J;
}

VehicleTracker::VehicleTracker(const armor_model::ArmorObservation& init_obs, int64_t t, const armor_model::Pose& T_init, const SavedVehicleParams& params) 
    : t_(t) {
    // 根据板的ID推断是几块板模型
    armor_num_ = (init_obs.plate_id == 0 || init_obs.plate_id == 2) ? 4 : 4; 
    
    Eigen::Vector3d t_cam_armor = T_init.translation();
    Eigen::Matrix3d R_cam_armor = T_init.linear();
    Eigen::Vector3d xyz_cam = t_cam_armor;
    
    // 映射到伪世界坐标系
    double cx = xyz_cam.z();
    double cy = -xyz_cam.x();
    double cz = -xyz_cam.y();
    
    // 获取初始装甲板的 yaw
    Eigen::Vector3d N_cam = T_init.linear().col(2);
    Eigen::Vector3d N_world(N_cam.z(), -N_cam.x(), -N_cam.y());
    double face_yaw = std::atan2(N_world.y(), N_world.x());
    
    // 从缓存参数或默认值初始化半径和高度偏移
    double r_init = params.valid ? params.r : 0.25;
    double dz_init = params.valid ? params.dz : 0.0;
    double h_init = params.valid ? params.h : 0.1;

    cx += r_init * std::cos(face_yaw);
    cy += r_init * std::sin(face_yaw);
    
    // 初始化 11D 状态 [x, vx, y, vy, z, vz, yaw, vyaw, r, dz, h]
    Eigen::VectorXd x0(11);
    x0 << cx, 0.0, cy, 0.0, cz, 0.0, face_yaw, 0.0, r_init, dz_init, h_init;
    
    Eigen::MatrixXd P0 = Eigen::MatrixXd::Identity(11, 11) * 1.0;
    
    // 如果使用了历史收敛参数，则直接赋予较小的初始协方差
    if (params.valid) {
        P0(8, 8) = 1e-4;
        P0(9, 9) = 1e-4;
        P0(10, 10) = 1e-4;
    }
    
    auto x_add = [](const Eigen::VectorXd& a, const Eigen::VectorXd& b) -> Eigen::VectorXd {
        Eigen::VectorXd c = a + b;
        c(6) = limit_rad(c(6));
        return c;
    };
    
    ekf_ = ExtendedKalmanFilter(x0, P0, x_add);
}

void VehicleTracker::predict(int64_t t) {
    double dt = (t - t_) / 1000.0;
    t_ = t;
    if (dt <= 0) return;

    Eigen::MatrixXd F = Eigen::MatrixXd::Identity(11, 11);
    F(0, 1) = dt;
    F(2, 3) = dt;
    F(4, 5) = dt;
    F(6, 7) = dt;

    double v1 = 100.0; // xyz acceleration variance
    double v2 = 400.0; // yaw angular acceleration variance
    
    double a = dt * dt * dt * dt / 4.0;
    double b = dt * dt * dt / 2.0;
    double c = dt * dt;
    
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(11, 11);
    Q(0,0) = a*v1; Q(0,1) = b*v1;
    Q(1,0) = b*v1; Q(1,1) = c*v1;
    
    Q(2,2) = a*v1; Q(2,3) = b*v1;
    Q(3,2) = b*v1; Q(3,3) = c*v1;
    
    Q(4,4) = a*v1; Q(4,5) = b*v1;
    Q(5,4) = b*v1; Q(5,5) = c*v1;
    
    Q(6,6) = a*v2; Q(6,7) = b*v2;
    Q(7,6) = b*v2; Q(7,7) = c*v2;
    
    // parameters r, l, h have minimal process noise
    Q(8,8) = 1e-4;
    Q(9,9) = 1e-4;
    Q(10,10) = 1e-4;

    auto f = [&F](const Eigen::VectorXd& x) -> Eigen::VectorXd {
        Eigen::VectorXd x_prior = F * x;
        x_prior(6) = limit_rad(x_prior(6));
        return x_prior;
    };

    // 防止转速 vyaw 爆炸导致模型疯狂旋转 (限制最大转速约为 4 rad/s)
    if (std::abs(ekf_.x(7)) > 4.0) {
        ekf_.x(7) = (ekf_.x(7) > 0) ? 4.0 : -4.0;
    }

    ekf_.predict(F, Q, f);
}

Eigen::Vector3d VehicleTracker::getArmorXYZ(const Eigen::VectorXd& x, int id) const {
    double angle = limit_rad(x(6) + id * 2.0 * CV_PI / armor_num_);
    bool use_l_h = (armor_num_ == 4) && (id == 1 || id == 3);
    
    double r = use_l_h ? x(8) + x(9) : x(8);
    double cx = x(0) - r * std::cos(angle);
    double cy = x(2) - r * std::sin(angle);
    double cz = use_l_h ? x(4) + x(10) : x(4);
    
    return {cx, cy, cz};
}

Eigen::MatrixXd VehicleTracker::getArmorJacobian(const Eigen::VectorXd& x, int id) const {
    double angle = limit_rad(x(6) + id * 2.0 * CV_PI / armor_num_);
    bool use_l_h = (armor_num_ == 4) && (id == 1 || id == 3);
    
    double r = use_l_h ? x(8) + x(9) : x(8);
    double dx_da = r * std::sin(angle);
    double dy_da = -r * std::cos(angle);
    
    double dx_dr = -std::cos(angle);
    double dy_dr = -std::sin(angle);
    double dx_dl = use_l_h ? -std::cos(angle) : 0.0;
    double dy_dl = use_l_h ? -std::sin(angle) : 0.0;
    
    double dz_dh = use_l_h ? 1.0 : 0.0;
    
    Eigen::MatrixXd H_armor_xyza = Eigen::MatrixXd::Zero(4, 11);
    H_armor_xyza(0, 0) = 1.0; H_armor_xyza(0, 6) = dx_da; H_armor_xyza(0, 8) = dx_dr; H_armor_xyza(0, 9) = dx_dl;
    H_armor_xyza(1, 2) = 1.0; H_armor_xyza(1, 6) = dy_da; H_armor_xyza(1, 8) = dy_dr; H_armor_xyza(1, 9) = dy_dl;
    H_armor_xyza(2, 4) = 1.0; H_armor_xyza(2, 10) = dz_dh;
    H_armor_xyza(3, 6) = 1.0;

    Eigen::Vector3d armor_xyz = getArmorXYZ(x, id);
    Eigen::Matrix3d H_armor_ypd = xyz2ypd_jacobian(armor_xyz);
    
    Eigen::MatrixXd H_armor_ypda = Eigen::MatrixXd::Zero(4, 4);
    H_armor_ypda.block<3,3>(0, 0) = H_armor_ypd;
    H_armor_ypda(3, 3) = 1.0;
    
    return H_armor_ypda * H_armor_xyza;
}

int VehicleTracker::matchArmor(const Eigen::Vector3d& t_cam_armor, const Eigen::Matrix3d& R_cam_armor) const {
    Eigen::Vector3d t_world_armor(t_cam_armor.z(), -t_cam_armor.x(), -t_cam_armor.y());
    Eigen::Vector3d ypd_obs = xyz2ypd(t_world_armor);

    // 提取观测法向量（指向车体内部）在世界坐标系下的朝向角
    Eigen::Vector3d N_cam = R_cam_armor.col(2);
    Eigen::Vector3d N_world(N_cam.z(), -N_cam.x(), -N_cam.y());
    double face_yaw_obs = std::atan2(N_world.y(), N_world.x());

    // 当前估计角速度，用于方向性惩罚
    double vyaw = ekf_.x(7);

    int best_id = -1;
    double min_error = 1e9;

    // ── 第一遍：带法向量硬约束的加权匹配 ──────────────────────────────────
    // 遍历所有装甲板（不再只取最近3个），用法向量硬截止过滤不可能的候选，
    // 防止 EKF 把转走的板预测到新板位置后被错误接受。
    for (int id = 0; id < armor_num_; ++id) {
        double face_yaw_pred = limit_rad(ekf_.x(6) + id * 2.0 * CV_PI / armor_num_);
        double face_diff = std::abs(limit_rad(face_yaw_obs - face_yaw_pred));

        // 硬约束：法向量偏差超过 90° 说明该板朝向与观测严重不符，直接跳过
        if (face_diff > CV_PI / 2.0) continue;

        Eigen::Vector3d ypd_pred = xyz2ypd(getArmorXYZ(ekf_.x, id));

        // 位置误差：水平角(yaw)权重更高，因为装甲板在水平方向区分度更好
        double pos_error = std::abs(limit_rad(ypd_obs[0] - ypd_pred[0])) * 2.0
                         + std::abs(ypd_obs[1] - ypd_pred[1]);

        // 法向量误差
        double face_error = face_diff;

        // vyaw 方向性惩罚：
        // 车辆旋转时，EKF 可能把"已转走的板"预测到"新板"的位置，
        // 对法向量偏差大的候选按角速度大小追加惩罚，使其分数劣于真正朝向一致的候选。
        double vyaw_penalty = face_diff * std::min(std::abs(vyaw), 4.0) * 0.25;

        double total_error = pos_error + face_error + vyaw_penalty;
        if (total_error < min_error) {
            min_error = total_error;
            best_id = id;
        }
    }

    // ── 第二遍（兜底）：若所有候选都被硬约束过滤，回退到最小综合误差 ───────
    // 此情况通常发生在 EKF 刚初始化或状态大幅漂移时
    if (best_id == -1) {
        min_error = 1e9;
        for (int id = 0; id < armor_num_; ++id) {
            double face_yaw_pred = limit_rad(ekf_.x(6) + id * 2.0 * CV_PI / armor_num_);
            double face_diff = std::abs(limit_rad(face_yaw_obs - face_yaw_pred));
            Eigen::Vector3d ypd_pred = xyz2ypd(getArmorXYZ(ekf_.x, id));
            double pos_error = std::abs(limit_rad(ypd_obs[0] - ypd_pred[0])) * 2.0;
            double total_error = pos_error + face_diff;
            if (total_error < min_error) {
                min_error = total_error;
                best_id = id;
            }
        }
    }

    return best_id;
}

void VehicleTracker::update(int plate_id, const Eigen::Vector3d& t_cam_armor, const Eigen::Matrix3d& R_cam_armor) {
    Eigen::MatrixXd H = getArmorJacobian(ekf_.x, plate_id);
    
    // 计算 YPD 和 Face Yaw
    Eigen::Vector3d t_world_armor(t_cam_armor.z(), -t_cam_armor.x(), -t_cam_armor.y());
    Eigen::Vector3d ypd_in_world = xyz2ypd(t_world_armor);
    
    // 保持和 matchArmor 一致，使用指向车体内部的法向量
    Eigen::Vector3d N_cam = R_cam_armor.col(2);
    Eigen::Vector3d N_world(N_cam.z(), -N_cam.x(), -N_cam.y());
    double face_yaw = std::atan2(N_world.y(), N_world.x());
    
    Eigen::VectorXd R_dig(4);
    R_dig << 4e-3, 4e-3, 0.1, 9e-2;
    Eigen::MatrixXd R = R_dig.asDiagonal();
    
    auto h = [&](const Eigen::VectorXd& x) -> Eigen::VectorXd {
        Eigen::Vector3d xyz = getArmorXYZ(x, plate_id);
        Eigen::Vector3d ypd = xyz2ypd(xyz);
        double angle = limit_rad(x(6) + plate_id * 2.0 * CV_PI / armor_num_);
        Eigen::VectorXd res(4);
        res << ypd[0], ypd[1], ypd[2], angle;
        return res;
    };
    
    auto z_subtract = [](const Eigen::VectorXd& a, const Eigen::VectorXd& b) -> Eigen::VectorXd {
        Eigen::VectorXd c = a - b;
        c(0) = limit_rad(c(0));
        c(1) = limit_rad(c(1));
        c(3) = limit_rad(c(3));
        return c;
    };
    
    Eigen::VectorXd z(4);
    z << ypd_in_world[0], ypd_in_world[1], ypd_in_world[2], face_yaw;
    
    ekf_.update(z, H, R, h, z_subtract);
    update_count_++;
}

armor_model::Pose VehicleTracker::getCurrentPose() const {
    Eigen::VectorXd x = ekf_.x;
    // 伪世界坐标系下的中心
    Eigen::Vector3d xyz_world(x(0), x(2), x(4));
    
    // 映射回 OpenCV 相机坐标系 (X为右, Y为下, Z为前)
    // 根据之前的映射: X_w=Z_c, Y_w=-X_c, Z_w=-Y_c
    // 所以: X_c = -Y_w, Y_c = -Z_w, Z_c = X_w
    Eigen::Vector3d xyz_cam(-xyz_world.y(), -xyz_world.z(), xyz_world.x());
    
    // 恢复位姿朝向
    double yaw_world = x(6);
    
    // 伪世界坐标系到相机的旋转矩阵
    Eigen::Matrix3d R_cam_world;
    R_cam_world <<  0, -1,  0,
                    0,  0, -1,
                    1,  0,  0;
                    
    Eigen::Matrix3d R_world_obj;
    R_world_obj << std::cos(yaw_world), -std::sin(yaw_world), 0,
                   std::sin(yaw_world),  std::cos(yaw_world), 0,
                                     0,                    0, 1;
                               
    armor_model::Pose T;
    T.linear() = R_cam_world * R_world_obj;
    T.translation() = xyz_cam;
    return T;
}

void VehicleTracker::updateVehicleModel(armor_model::VehicleModel& model) const {
    model.updateParameters(ekf_.x(8), ekf_.x(10), 15.0); // d = r
}

} // namespace vision_system
