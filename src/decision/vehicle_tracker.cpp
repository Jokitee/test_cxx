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

VehicleTracker::VehicleTracker(const armor_model::ArmorObservation& init_obs, int64_t t, const armor_model::Pose& T_init)
    : t_(t) {
    
    Eigen::Vector3d xyz_cam = T_init.translation();
    
    // 映射到伪世界坐标系
    double cx = xyz_cam.z();
    double cy = -xyz_cam.x();
    double cz = -xyz_cam.y();
    
    // 获取初始装甲板的 yaw
    Eigen::Vector3d N_cam = T_init.linear().col(2);
    Eigen::Vector3d N_world(N_cam.z(), -N_cam.x(), -N_cam.y());
    double face_yaw = std::atan2(N_world.y(), N_world.x());
    
    // 假设初始半径 0.25, dz=0.1
    double r = 0.25;
    cx += r * std::cos(face_yaw);
    cy += r * std::sin(face_yaw);
    
    // 初始化 11D 状态 [x, vx, y, vy, z, vz, yaw, vyaw, r, l, h]
    Eigen::VectorXd x0(11);
    x0 << cx, 0.0, cy, 0.0, cz, 0.0, face_yaw, 0.0, r, 0.0, 0.1;
    
    Eigen::MatrixXd P0 = Eigen::MatrixXd::Identity(11, 11) * 1.0;
    
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
    Eigen::Vector3d ypd_in_world = xyz2ypd(t_world_armor);
    
    // 提取在相机坐标系下的法向量 (指向车体内部，使得正面装甲板的yaw在世界系为0)
    Eigen::Vector3d N_cam = R_cam_armor.col(2);
    Eigen::Vector3d N_world(N_cam.z(), -N_cam.x(), -N_cam.y());
    double face_yaw = std::atan2(N_world.y(), N_world.x());
    
    // 获取四个预测装甲板的状态，并按距离(ypd[2])排序，只取最近的 3 个，剔除背面的装甲板
    std::vector<std::pair<int, Eigen::Vector3d>> predicted_armors;
    for (int id = 0; id < armor_num_; ++id) {
        predicted_armors.push_back({id, xyz2ypd(getArmorXYZ(ekf_.x, id))});
    }
    std::sort(predicted_armors.begin(), predicted_armors.end(),
              [](const std::pair<int, Eigen::Vector3d>& a, const std::pair<int, Eigen::Vector3d>& b) {
                  return a.second[2] < b.second[2];
              });
              
    int best_id = 0;
    double min_error = 1e9;
    
    for (int i = 0; i < 3 && i < armor_num_; ++i) {
        int id = predicted_armors[i].first;
        Eigen::Vector3d ypd_pred = predicted_armors[i].second;
        double face_yaw_pred = limit_rad(ekf_.x(6) + id * 2.0 * CV_PI / armor_num_);
        
        // 角度误差 = 位置的偏航角误差 + 装甲板自身朝向角误差
        double error = std::abs(limit_rad(ypd_in_world[0] - ypd_pred[0])) + 
                       std::abs(limit_rad(face_yaw - face_yaw_pred));
                       
        if (error < min_error) {
            min_error = error;
            best_id = id;
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
    model.updateParameters(ekf_.x(8) * 2.0, ekf_.x(10), 15.0); // r*2 = d
}

} // namespace vision_system
