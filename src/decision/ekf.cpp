#include "vision_system/decision/ekf.hpp"
#include <iostream>

namespace vision_system {

ExtendedKalmanFilter::ExtendedKalmanFilter(
    const Eigen::VectorXd& x0, const Eigen::MatrixXd& P0,
    std::function<Eigen::VectorXd(const Eigen::VectorXd&, const Eigen::VectorXd&)> x_add)
    : x(x0), P(P0), x_add_(x_add) {
    I = Eigen::MatrixXd::Identity(P.rows(), P.cols());
}

void ExtendedKalmanFilter::init(const Eigen::VectorXd& x0, const Eigen::MatrixXd& P0) {
    x = x0;
    P = P0;
    I = Eigen::MatrixXd::Identity(P.rows(), P.cols());
}

void ExtendedKalmanFilter::reset(const Eigen::VectorXd& x0) {
    x = x0;
    // 重置协方差矩阵，表示对当前状态不太确定
    P = Eigen::MatrixXd::Identity(P.rows(), P.cols()) * 10.0;
}

void ExtendedKalmanFilter::predict(
    const Eigen::MatrixXd& F, const Eigen::MatrixXd& Q,
    std::function<Eigen::VectorXd(const Eigen::VectorXd&)> f) {
    
    // 状态预测: x = f(x)
    x = f(x);
    
    // 协方差预测: P = F * P * F^T + Q
    P = F * P * F.transpose() + Q;
}

void ExtendedKalmanFilter::update(
    const Eigen::VectorXd& z, const Eigen::MatrixXd& H, const Eigen::MatrixXd& R,
    std::function<Eigen::VectorXd(const Eigen::VectorXd&)> h,
    std::function<Eigen::VectorXd(const Eigen::VectorXd&, const Eigen::VectorXd&)> z_subtract) {
    
    // 观测残差 (Innovation)
    Eigen::VectorXd y = z_subtract(z, h(x));
    
    // 残差协方差 S = H * P * H^T + R
    Eigen::MatrixXd S = H * P * H.transpose() + R;
    
    // 卡尔曼增益 K = P * H^T * S^-1
    Eigen::MatrixXd K = P * H.transpose() * S.inverse();
    
    // 状态更新
    x = x_add_ ? x_add_(x, K * y) : (x + K * y);
    
    // 协方差更新 P = (I - K * H) * P
    P = (I - K * H) * P;
}

} // namespace vision_system
