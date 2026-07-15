#ifndef VISION_SYSTEM_DECISION_EKF_HPP
#define VISION_SYSTEM_DECISION_EKF_HPP

#include <Eigen/Dense>

namespace vision_system {

class ExtendedKalmanFilter {
public:
    ExtendedKalmanFilter();

    // 初始化滤波器
    void init(const Eigen::VectorXd& x0, double t0);

    // 预测步骤
    // dt: 距离上一次预测或更新的时间差(秒)
    void predict(double dt);

    // 更新步骤
    // z: 当前观测值
    // 返回 卡方值 (chi-square)，可用于判断观测是否属于突变或异常
    double update(const Eigen::VectorXd& z);

    // 强制重置状态 (用于卡方检验失败、目标丢失后重新追踪)
    void reset(const Eigen::VectorXd& x0);

    // 获取当前最优状态估计
    Eigen::VectorXd getState() const { return x_; }
    
    // 设置过程噪声 Q 和 观测噪声 R
    void setQ(const Eigen::MatrixXd& Q) { Q_ = Q; }
    void setR(const Eigen::MatrixXd& R) { R_ = R; }

private:
    // --- EKF 核心函数，用户可根据具体模型进行继承或修改 ---
    // 状态方程推演 f(x, dt)
    Eigen::VectorXd f(const Eigen::VectorXd& x, double dt);
    // 状态方程雅可比矩阵 F
    Eigen::MatrixXd calcF(const Eigen::VectorXd& x, double dt);
    
    // 观测方程 h(x)
    Eigen::VectorXd h(const Eigen::VectorXd& x);
    // 观测方程雅可比矩阵 H
    Eigen::MatrixXd calcH(const Eigen::VectorXd& x);

private:
    int n_; // 状态维度
    int m_; // 观测维度

    Eigen::VectorXd x_; // 状态向量 [x, y, z, vx, vy, vz]
    Eigen::MatrixXd P_; // 状态协方差矩阵

    Eigen::MatrixXd Q_; // 过程噪声协方差矩阵
    Eigen::MatrixXd R_; // 观测噪声协方差矩阵
};

} // namespace vision_system

#endif // VISION_SYSTEM_DECISION_EKF_HPP
