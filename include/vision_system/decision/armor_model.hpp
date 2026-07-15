// armor_model.hpp
#ifndef ARMOR_MODEL_HPP
#define ARMOR_MODEL_HPP

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <vector>
#include <array>
#include <memory>

namespace armor_model {

using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix3d;
using Mat4 = Eigen::Matrix4d;
using Pose = Eigen::Isometry3d;
using Vector6d = Eigen::Matrix<double, 6, 1>;

// ==================== 基础几何定义 ====================

struct ArmorPlate {
    int id;                    // 0:前 1:右 2:后 3:左
    std::string name;
    
    // 模型参数（待优化）
    double d;                  // 水平投影距离
    double h;                  // Z轴投影高度
    double tilt_angle;         // 与Z轴夹角（度），默认15°
    
    // 装甲板物理尺寸（常量）
    double width;              // 板宽（水平方向）
    double height;             // 板高（垂直方向）
    
    // 3D角点（物体坐标系，8个点，但装甲板是矩形用4个角点）
    std::array<Vec3, 4> corners_object;   // 物体坐标系下4角点
    std::array<Vec3, 4> corners_world;    // 世界坐标系下4角点（应用位姿后）
    
    // 法向量（指向中心）
    Vec3 normal_object;
    Vec3 normal_world;
    
    // 中心点
    Vec3 center_object;
    Vec3 center_world;
    
    ArmorPlate(int _id, double _d, double _h, double _tilt = 15.0,
               double _w = 0.230, double _he = 0.127);  // RM标准装甲板尺寸
    
    // 根据d,h,tilt重新计算物体坐标系下的几何
    void updateGeometry();
    
    // 应用位姿变换到世界/相机坐标系
    void applyPose(const Pose& T_world_object);
};

// ==================== 车辆模型（4个装甲板） ====================

class VehicleModel {
public:
    // 构造：初始参数
    VehicleModel(double d_init = 0.300, double h_init = 0.150, 
                 double tilt = 15.0);
    
    // 获取四个装甲板（const/non-const）
    const std::vector<ArmorPlate>& getPlates() const { return plates_; }
    std::vector<ArmorPlate>& getPlates() { return plates_; }
    
    // 更新模型参数（优化后调用）
    void updateParameters(double d, double h, double tilt = 15.0);
    
    // 获取当前参数
    double getD() const { return d_; }
    double getH() const { return h_; }
    double getTilt() const { return tilt_; }
    
    // 应用位姿到所有装甲板
    void applyPose(const Pose& T_world_object);
    
    // 获取所有角点（用于投影）
    std::vector<Vec3> getAllCorners() const;
    
    // 根据ID获取装甲板
    const ArmorPlate* getPlateById(int id) const;
    
private:
    double d_, h_, tilt_;
    std::vector<ArmorPlate> plates_;
};

// ==================== 投影与观测 ====================

struct Camera {
    cv::Mat K;           // 3x3 内参矩阵
    cv::Mat D;           // 畸变系数
    int img_width, img_height;
    
    Camera() = default;
    Camera(const cv::Mat& _K, const cv::Mat& _D, int w, int h);
    
    // 3D点投影到图像
    cv::Point2f project(const Vec3& pt_world, const Pose& T_cam_world) const;
    
    // 批量投影
    std::vector<cv::Point2f> projectPoints(
        const std::vector<Vec3>& pts_world, 
        const Pose& T_cam_world) const;
    
    // 重投影误差计算
    double reprojectionError(
        const std::vector<Vec3>& pts_3d,
        const std::vector<cv::Point2f>& pts_2d_obs,
        const Pose& T_cam_world) const;
};

// ==================== 观测数据结构 ====================

struct ArmorObservation {
    int plate_id;                    // 装甲板编号
    std::array<cv::Point2f, 4> corners_img;  // 图像中检测到的4角点
    double confidence;               // 检测置信度
    int64_t timestamp;               // 时间戳
};

struct FrameObservation {
    int64_t timestamp;
    std::vector<ArmorObservation> armors;
    cv::Mat image;                   // 可选：原始图像
};

// ==================== 模型优化器 ====================

class ModelOptimizer {
public:
    struct OptimizationResult {
        double d_opt;           // 优化后的d
        double h_opt;           // 优化后的h
        double tilt_opt;        // 优化后的tilt（通常固定）
        Pose T_cam_object;      // 优化后的位姿
        double final_error;     // 最终重投影误差
        int iterations;         // 迭代次数
        bool converged;         // 是否收敛
    };
    
    ModelOptimizer(const Camera& cam, const VehicleModel& model_init);
    
    // 单帧优化：同时优化模型参数和位姿
    OptimizationResult optimizeSingleFrame(
        const FrameObservation& obs,
        const Pose& T_init,              // 位姿初值（PnP结果）
        bool optimize_d = true,          // 是否优化d
        bool optimize_h = true,          // 是否优化h
        bool optimize_tilt = false       // 通常固定15°
    );
    
    // 多帧联合优化（更稳定）
    OptimizationResult optimizeMultiFrame(
        const std::vector<FrameObservation>& obs_list,
        const std::vector<Pose>& T_init_list,
        bool optimize_d = true,
        bool optimize_h = true
    );
    
    // 获取当前模型
    const VehicleModel& getModel() const { return model_; }
    
private:
    Camera cam_;
    VehicleModel model_;
    
    // 构建优化问题
    struct Residual {
        int plate_id;
        int corner_idx;
        cv::Point2f obs_pt;
        double weight;
    };
    
    // 计算残差和雅可比（用于LM算法）
    void computeResidualsAndJacobian(
        const std::vector<Residual>& residuals,
        const Pose& T_cam_obj,
        double d, double h, double tilt,
        Eigen::VectorXd& r,           // 残差向量 (2N)
        Eigen::MatrixXd& J            // 雅可比 (2N x 6+3) [位姿6DOF + d,h,tilt]
    );
    
    // 位姿参数化 <-> 李代数
    static Pose paramsToPose(const Vector6d& xi);      
    static Vector6d poseToParams(const Pose& T);      
};

// ==================== 可视化 ====================

class ModelVisualizer {
public:
    // 3D视图渲染（OpenCV 3D投影）
    static cv::Mat render3DView(
        const VehicleModel& model,
        const Pose& T_cam_object,
        const Camera& cam,
        const cv::Mat& bg_image,        // 背景图像（可选）
        const FrameObservation* obs = nullptr  // 观测叠加（可选）
    );
    
    // 纯3D线框视图（不依赖相机图像）
    static cv::Mat renderWireframe(
        const VehicleModel& model,
        const Pose& T_cam_object,
        const Camera& cam,
        int width = 800,
        int height = 600
    );
    
    // 绘制坐标轴
    static void drawAxis(cv::Mat& img, const Camera& cam, 
                         const Pose& T_cam_world, double length = 0.1);
    
    // 绘制装甲板（填充+边框）
    static void drawArmorPlate(cv::Mat& img, const Camera& cam,
                               const ArmorPlate& plate,
                               const Pose& T_cam_object,
                               const cv::Scalar& color,
                               int thickness = 2);
};

// ==================== 工具函数 ====================

// Eigen <-> OpenCV 转换
cv::Mat eigenToCvMat(const Mat3& R);
cv::Mat eigenToCvMat(const Vec3& t);
Mat3 cvMatToEigen3d(const cv::Mat& R);
Vec3 cvMatToEigenVec(const cv::Mat& t);
Pose cvToEigenPose(const cv::Mat& R, const cv::Mat& t);

// 四元数 <-> 旋转矩阵
Eigen::Quaterniond rotmatToQuat(const Mat3& R);
Mat3 quatToRotmat(const Eigen::Quaterniond& q);

} // namespace armor_model

#endif // ARMOR_MODEL_HPP