#include "vision_system/decision/ekf.hpp"
#include <iostream>

namespace vision_system {

ExtendedKalmanFilter::ExtendedKalmanFilter() {
    // 默认我们追踪 6 维状态: [x, y, z, vx, vy, vz]
    // 观测 3 维状态: [x, y, z]
    n_ = 6;
    m_ = 3;

    x_ = Eigen::VectorXd::Zero(n_);
    P_ = Eigen::MatrixXd::Identity(n_, n_);
    
    Q_ = Eigen::MatrixXd::Identity(n_, n_) * 0.1; // 过程噪声预设
    R_ = Eigen::MatrixXd::Identity(m_, m_) * 0.01; // 观测噪声预设
}

void ExtendedKalmanFilter::init(const Eigen::VectorXd& x0, double t0) {
    x_ = x0;
    P_ = Eigen::MatrixXd::Identity(n_, n_) * 1.0;
}

void ExtendedKalmanFilter::reset(const Eigen::VectorXd& x0) {
    x_ = x0;
    // 重置协方差矩阵，表示对当前状态不太确定
    P_ = Eigen::MatrixXd::Identity(n_, n_) * 10.0;
}

Eigen::VectorXd ExtendedKalmanFilter::f(const Eigen::VectorXd& x, double dt) {
    // 匀速直线运动 (Constant Velocity) 模型
    Eigen::VectorXd x_pred = x;
    x_pred(0) += x(3) * dt; // x = x + vx * dt
    x_pred(1) += x(4) * dt; // y = y + vy * dt
    x_pred(2) += x(5) * dt; // z = z + vz * dt
    return x_pred;
}

Eigen::MatrixXd ExtendedKalmanFilter::calcF(const Eigen::VectorXd& x, double dt) {
    Eigen::MatrixXd F = Eigen::MatrixXd::Identity(n_, n_);
    F(0, 3) = dt;
    F(1, 4) = dt;
    F(2, 5) = dt;
    return F;
}

Eigen::VectorXd ExtendedKalmanFilter::h(const Eigen::VectorXd& x) {
    // 纯观测位置 [x, y, z]
    Eigen::VectorXd z_pred = Eigen::VectorXd::Zero(m_);
    z_pred(0) = x(0);
    z_pred(1) = x(1);
    z_pred(2) = x(2);
    return z_pred;
}

Eigen::MatrixXd ExtendedKalmanFilter::calcH(const Eigen::VectorXd& x) {
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(m_, n_);
    H(0, 0) = 1.0;
    H(1, 1) = 1.0;
    H(2, 2) = 1.0;
    return H;
}

void ExtendedKalmanFilter::predict(double dt) {
    Eigen::MatrixXd F = calcF(x_, dt);
    
    // 状态预测: x = f(x)
    x_ = f(x_, dt);
    
    // 协方差预测: P = F * P * F^T + Q
    P_ = F * P_ * F.transpose() + Q_;
}

double ExtendedKalmanFilter::update(const Eigen::VectorXd& z) {
    Eigen::MatrixXd H = calcH(x_);
    Eigen::VectorXd z_pred = h(x_);
    
    // 观测残差 (Innovation)
    Eigen::VectorXd y = z - z_pred;
    
    // 残差协方差 S = H * P * H^T + R
    Eigen::MatrixXd S = H * P_ * H.transpose() + R_;
    
    // 卡方检验 (Chi-square test): 评估实际观测与预测的马氏距离
    // chi2 = y^T * S^-1 * y
    double chi2 = y.transpose() * S.inverse() * y;
    
    // 卡尔曼增益 K = P * H^T * S^-1
    Eigen::MatrixXd K = P_ * H.transpose() * S.inverse();
    
    // 状态更新
    x_ = x_ + K * y;
    
    // 协方差更新 P = (I - K * H) * P
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n_, n_);
    P_ = (I - K * H) * P_;
    
    return chi2;
}

} // namespace vision_system
