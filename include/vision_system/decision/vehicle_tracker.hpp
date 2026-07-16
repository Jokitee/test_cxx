#ifndef VISION_SYSTEM_DECISION_VEHICLE_TRACKER_HPP
#define VISION_SYSTEM_DECISION_VEHICLE_TRACKER_HPP

#include <Eigen/Dense>
#include <chrono>
#include "vision_system/decision/ekf.hpp"
#include "vision_system/core/types.hpp"
#include "vision_system/decision/armor_model.hpp"

namespace vision_system {

class VehicleTracker {
public:
    // T_init is the initial camera-to-object pose
    VehicleTracker(const armor_model::ArmorObservation& init_obs, int64_t t, const armor_model::Pose& T_init);

    void predict(int64_t t);
    
    // t_cam_armor, R_cam_armor: PnP解算出来的装甲板在相机坐标系下的位姿
    int matchArmor(const Eigen::Vector3d& t_cam_armor, const Eigen::Matrix3d& R_cam_armor) const;
    void update(int plate_id, const Eigen::Vector3d& t_cam_armor, const Eigen::Matrix3d& R_cam_armor);

    Eigen::VectorXd getEKFState() const { return ekf_.getState(); }
    
    // x vx y vy z vz yaw vyaw r l h
    // 0  1  2  3  4  5   6    7   8 9 10

    armor_model::Pose getCurrentPose() const;
    void updateVehicleModel(armor_model::VehicleModel& model) const;

    bool is_tracking = true;
    int update_count_ = 0;

private:
    ExtendedKalmanFilter ekf_;
    int64_t t_; // milliseconds
    int armor_num_ = 4;

    Eigen::Vector3d getArmorXYZ(const Eigen::VectorXd& x, int id) const;
    Eigen::MatrixXd getArmorJacobian(const Eigen::VectorXd& x, int id) const;

    static double limit_rad(double angle);
    static Eigen::Vector3d xyz2ypd(const Eigen::Vector3d& xyz);
    static Eigen::Matrix3d xyz2ypd_jacobian(const Eigen::Vector3d& xyz);
};

} // namespace vision_system

#endif // VISION_SYSTEM_DECISION_VEHICLE_TRACKER_HPP
