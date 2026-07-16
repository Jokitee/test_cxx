#ifndef VISION_SYSTEM_DECISION_EKF_HPP
#define VISION_SYSTEM_DECISION_EKF_HPP

#include <Eigen/Dense>
#include <functional>

namespace vision_system {

class ExtendedKalmanFilter {
public:
    Eigen::VectorXd x; // public for direct read/write if needed
    Eigen::MatrixXd P;

    ExtendedKalmanFilter() = default;
    
    // 初始化滤波器
    ExtendedKalmanFilter(
        const Eigen::VectorXd& x0, const Eigen::MatrixXd& P0,
        std::function<Eigen::VectorXd(const Eigen::VectorXd&, const Eigen::VectorXd&)> x_add =
            [](const Eigen::VectorXd& a, const Eigen::VectorXd& b) { return a + b; });

    void init(const Eigen::VectorXd& x0, const Eigen::MatrixXd& P0);
    
    void reset(const Eigen::VectorXd& x0);

    // 预测步骤
    void predict(
        const Eigen::MatrixXd& F, const Eigen::MatrixXd& Q,
        std::function<Eigen::VectorXd(const Eigen::VectorXd&)> f);

    // 更新步骤
    void update(
        const Eigen::VectorXd& z, const Eigen::MatrixXd& H, const Eigen::MatrixXd& R,
        std::function<Eigen::VectorXd(const Eigen::VectorXd&)> h,
        std::function<Eigen::VectorXd(const Eigen::VectorXd&, const Eigen::VectorXd&)> z_subtract =
            [](const Eigen::VectorXd& a, const Eigen::VectorXd& b) { return a - b; });

    Eigen::VectorXd getState() const { return x; }

private:
    Eigen::MatrixXd I;
    std::function<Eigen::VectorXd(const Eigen::VectorXd&, const Eigen::VectorXd&)> x_add_;
};

} // namespace vision_system

#endif // VISION_SYSTEM_DECISION_EKF_HPP
