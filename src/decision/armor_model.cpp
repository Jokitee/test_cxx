/**
 * @brief 车辆模型优化与LM非线性位姿解算
 * @author jokit
 * @date 2026-07-15
 */
// armor_model.cpp
#include "vision_system/decision/armor_model.hpp"
#include <cmath>
#include <iostream>

namespace armor_model {

// ==================== ArmorPlate实现 ====================

ArmorPlate::ArmorPlate(int _id, double _d, double _h, double _tilt,
                       double _w, double _he)
    : id(_id), d(_d), h(_h), tilt_angle(_tilt), width(_w), height(_he) {
    
    static const std::array<std::string, 4> names = {"Front", "Right", "Back", "Left"};
    name = names[id];
    updateGeometry();
}

void ArmorPlate::updateGeometry() {
    double tilt_rad = tilt_angle * M_PI / 180.0;
    
    // 侧装甲板的高低差（对应 EKF 中的 x(10)）
    double current_h = (id == 1 || id == 3) ? h : 0.0;
    
    // 装甲板中心在物体坐标系的位置
    // 前(0,d,0), 右(d,0,h), 后(0,-d,0), 左(-d,0,h)
    switch (id) {
        case 0: center_object = Vec3(-d,  0, current_h); break;  // 前
        case 1: center_object = Vec3( 0, -d, current_h); break;  // 右
        case 2: center_object = Vec3( d,  0, current_h); break;  // 后
        case 3: center_object = Vec3( 0,  d, current_h); break;  // 左
    }
    
    Vec3 to_center = -center_object;
    to_center.z() = 0;  // 只取水平分量
    to_center.normalize();
    
    // 法向量：指向外侧，水平分量背向中心，同时有向上的Z分量（与垂直方向夹角15°）
    // 即法向量与水平面（XY平面）的夹角为 tilt_angle = 15°
    // 这意味着法向的水平分量大小为 cos(15°)，垂直分量大小为 sin(15°)
    // 注意到 to_center 是指向中心的，所以如果法向量要指向车外，必须取反
    // 为了和之前的系统约定一致，如果原来的逻辑是把 normal 取作为 to_center 的方向...
    // 但物理上，法向量应该指向外，也就是 -to_center。
    // 我们查看后续坐标系建立：
    // X_local = Vec3(-to_center.y(), to_center.x(), 0).normalized();
    // Y_local = Z_local.cross(X_local).normalized();
    // 假设 Z_local 是物理法向（向外）。则 Z_local 的水平分量应该是 -to_center
    normal_object = Vec3(
        -to_center.x() * std::cos(tilt_rad),
        -to_center.y() * std::cos(tilt_rad),
        std::sin(tilt_rad)
    );
    normal_object.normalize();
    
    // 构建装甲板局部坐标系
    // Z_local = normal（指向外/上）
    // X_local = 水平切向（垂直于到中心的向量）
    // Y_local = Z_local × X_local（板子的垂直方向）
    
    Vec3 X_local, Y_local, Z_local;
    Z_local = normal_object;
    
    // X_local：水平方向，垂直于到中心的连线
    if (std::abs(to_center.x()) < 1e-6 && std::abs(to_center.y()) < 1e-6) {
        X_local = Vec3(1, 0, 0);  // 退化情况
    } else {
        X_local = Vec3(-to_center.y(), to_center.x(), 0).normalized();
    }
    
    Y_local = Z_local.cross(X_local).normalized();
    // 重新正交化
    X_local = Y_local.cross(Z_local).normalized();
    
    // 计算4个角点（中心 ± width/2 * X ± height/2 * Y）
    double w2 = width / 2.0;
    double h2 = height / 2.0;
    
    corners_object[0] = center_object - w2 * X_local - h2 * Y_local;  // 左下
    corners_object[1] = center_object + w2 * X_local - h2 * Y_local;  // 右下
    corners_object[2] = center_object + w2 * X_local + h2 * Y_local;  // 右上
    corners_object[3] = center_object - w2 * X_local + h2 * Y_local;  // 左上
}

void ArmorPlate::applyPose(const Pose& T_world_object) {
    center_world = T_world_object * center_object;
    normal_world = T_world_object.rotation() * normal_object;
    
    for (int i = 0; i < 4; ++i) {
        corners_world[i] = T_world_object * corners_object[i];
    }
}

// ==================== VehicleModel实现 ====================

VehicleModel::VehicleModel(double d_init, double h_init, double tilt) 
    : d_(d_init), h_(h_init), tilt_(tilt) {
    
    plates_.reserve(4);
    for (int i = 0; i < 4; ++i) {
        plates_.emplace_back(i, d_, h_, tilt_, 0.135, 0.055);
    }
}

void VehicleModel::updateParameters(double d, double h, double tilt) {
    d_ = d; h_ = h; tilt_ = tilt;
    for (auto& plate : plates_) {
        plate.d = d;
        plate.h = h;
        plate.tilt_angle = tilt;
        plate.updateGeometry();
    }
}

void VehicleModel::applyPose(const Pose& T_world_object) {
    for (auto& plate : plates_) {
        plate.applyPose(T_world_object);
    }
}

std::vector<Vec3> VehicleModel::getAllCorners() const {
    std::vector<Vec3> all;
    all.reserve(16);
    for (const auto& p : plates_) {
        for (const auto& c : p.corners_object) {
            all.push_back(c);
        }
    }
    return all;
}

const ArmorPlate* VehicleModel::getPlateById(int id) const {
    for (const auto& p : plates_) {
        if (p.id == id) return &p;
    }
    return nullptr;
}

// ==================== Camera实现 ====================

Camera::Camera(const cv::Mat& _K, const cv::Mat& _D, int w, int h)
    : K(_K.clone()), D(_D.clone()), img_width(w), img_height(h) {}

cv::Point2f Camera::project(const Vec3& pt_world, const Pose& T_cam_world) const {
    Vec3 pt_cam = T_cam_world * pt_world;
    
    if (pt_cam.z() <= 1e-6) {
        return cv::Point2f(-1, -1);  // 无效投影
    }
    
    // 透视投影
    double x = pt_cam.x() / pt_cam.z();
    double y = pt_cam.y() / pt_cam.z();
    
    // 畸变（简化：假设已去畸变，或K已包含）
    cv::Point2f pt_img;
    pt_img.x = static_cast<float>(K.at<double>(0,0) * x + K.at<double>(0,2));
    pt_img.y = static_cast<float>(K.at<double>(1,1) * y + K.at<double>(1,2));
    
    return pt_img;
}

std::vector<cv::Point2f> Camera::projectPoints(
    const std::vector<Vec3>& pts_world, 
    const Pose& T_cam_world) const {
    
    std::vector<cv::Point2f> result;
    result.reserve(pts_world.size());
    
    for (const auto& pt : pts_world) {
        result.push_back(project(pt, T_cam_world));
    }
    return result;
}

double Camera::reprojectionError(
    const std::vector<Vec3>& pts_3d,
    const std::vector<cv::Point2f>& pts_2d_obs,
    const Pose& T_cam_world) const {
    
    if (pts_3d.size() != pts_2d_obs.size()) return -1.0;
    
    double error = 0.0;
    for (size_t i = 0; i < pts_3d.size(); ++i) {
        cv::Point2f proj = project(pts_3d[i], T_cam_world);
        double dx = proj.x - pts_2d_obs[i].x;
        double dy = proj.y - pts_2d_obs[i].y;
        error += dx * dx + dy * dy;
    }
    return std::sqrt(error / pts_3d.size());
}

// ==================== ModelOptimizer实现 ====================

ModelOptimizer::ModelOptimizer(const Camera& cam, const VehicleModel& model_init)
    : cam_(cam), model_(model_init) {}

Pose ModelOptimizer::paramsToPose(const Vector6d& xi) {
    Pose T = Pose::Identity();
    Eigen::Vector3d omega = xi.head<3>();
    double theta = omega.norm();
    
    Mat3 R;
    if (theta < 1e-10) {
        R = Mat3::Identity();
    } else {
        Eigen::AngleAxisd aa(theta, omega.normalized());
        R = aa.toRotationMatrix();
    }
    
    T.linear() = R;
    T.translation() = xi.tail<3>();
    return T;
}

Vector6d ModelOptimizer::poseToParams(const Pose& T) {
    Vector6d xi;
    Eigen::AngleAxisd aa(T.rotation());
    xi.head<3>() = aa.angle() * aa.axis();
    xi.tail<3>() = T.translation();
    return xi;
}

void ModelOptimizer::computeResidualsAndJacobian(
    const std::vector<Residual>& residuals,
    const Pose& T_cam_obj,
    double d, double h, double tilt,
    Eigen::VectorXd& r,
    Eigen::MatrixXd& J) {
    
    int n = residuals.size();
    r.resize(2 * n);
    J.resize(2 * n, 9);  // 6(位姿) + 3(d,h,tilt)
    
    // 临时模型用于计算导数
    VehicleModel temp_model(d, h, tilt);
    temp_model.applyPose(T_cam_obj);
    
    double eps = 1e-6;
    
    for (int i = 0; i < n; ++i) {
        const auto& res = residuals[i];
        const auto* plate = temp_model.getPlateById(res.plate_id);
        if (!plate) continue;
        
        Vec3 pt_world = plate->corners_world[res.corner_idx];
        cv::Point2f proj = cam_.project(pt_world, Pose::Identity());
        
        // 残差
        r(2*i)   = (proj.x - res.obs_pt.x) * res.weight;
        r(2*i+1) = (proj.y - res.obs_pt.y) * res.weight;
        
        // 数值差分计算雅可比（简化版，可替换为解析导数）
        // 位姿扰动
        for (int j = 0; j < 6; ++j) {
            Vector6d xi = poseToParams(T_cam_obj);
            xi(j) += eps;
            Pose T_perturb = paramsToPose(xi);
            
            VehicleModel m_perturb(d, h, tilt);
            m_perturb.applyPose(T_perturb);
            const auto* p_perturb = m_perturb.getPlateById(res.plate_id);
            cv::Point2f proj_p = cam_.project(
                p_perturb->corners_world[res.corner_idx], Pose::Identity());
            
            J(2*i, j)   = (proj_p.x - proj.x) / eps * res.weight;
            J(2*i+1, j) = (proj_p.y - proj.y) / eps * res.weight;
        }
        
        // d扰动
        {
            VehicleModel m_perturb(d + eps, h, tilt);
            m_perturb.applyPose(T_cam_obj);
            const auto* p_perturb = m_perturb.getPlateById(res.plate_id);
            cv::Point2f proj_p = cam_.project(
                p_perturb->corners_world[res.corner_idx], Pose::Identity());
            
            J(2*i, 6)   = (proj_p.x - proj.x) / eps * res.weight;
            J(2*i+1, 6) = (proj_p.y - proj.y) / eps * res.weight;
        }
        
        // h扰动
        {
            VehicleModel m_perturb(d, h + eps, tilt);
            m_perturb.applyPose(T_cam_obj);
            const auto* p_perturb = m_perturb.getPlateById(res.plate_id);
            cv::Point2f proj_p = cam_.project(
                p_perturb->corners_world[res.corner_idx], Pose::Identity());
            
            J(2*i, 7)   = (proj_p.x - proj.x) / eps * res.weight;
            J(2*i+1, 7) = (proj_p.y - proj.y) / eps * res.weight;
        }
        
        // tilt扰动
        {
            VehicleModel m_perturb(d, h, tilt + eps);
            m_perturb.applyPose(T_cam_obj);
            const auto* p_perturb = m_perturb.getPlateById(res.plate_id);
            cv::Point2f proj_p = cam_.project(
                p_perturb->corners_world[res.corner_idx], Pose::Identity());
            
            J(2*i, 8)   = (proj_p.x - proj.x) / eps * res.weight;
            J(2*i+1, 8) = (proj_p.y - proj.y) / eps * res.weight;
        }
    }
}

ModelOptimizer::OptimizationResult ModelOptimizer::optimizeSingleFrame(
    const FrameObservation& obs,
    const Pose& T_init,
    bool optimize_d,
    bool optimize_h,
    bool optimize_tilt) {
    
    OptimizationResult result;
    result.d_opt = model_.getD();
    result.h_opt = model_.getH();
    result.tilt_opt = model_.getTilt();
    result.T_cam_object = T_init;
    
    // 构建残差列表
    std::vector<Residual> residuals;
    for (const auto& armor : obs.armors) {
        for (int i = 0; i < 4; ++i) {
            Residual r;
            r.plate_id = armor.plate_id;
            r.corner_idx = i;
            r.obs_pt = armor.corners_img[i];
            r.weight = armor.confidence;
            residuals.push_back(r);
        }
    }
    
    // LM算法参数
    const int max_iter = 50;
    const double lambda_init = 0.01;
    double lambda = lambda_init;
    
    Vector6d xi = poseToParams(T_init);
    double d = model_.getD();
    double h = model_.getH();
    double tilt = model_.getTilt();
    
    // 确定优化变量
    std::vector<int> opt_vars;  // 0-5:位姿, 6:d, 7:h, 8:tilt
    for (int i = 0; i < 6; ++i) opt_vars.push_back(i);
    if (optimize_d) opt_vars.push_back(6);
    if (optimize_h) opt_vars.push_back(7);
    if (optimize_tilt) opt_vars.push_back(8);
    
    int var_dim = opt_vars.size();
    
    for (int iter = 0; iter < max_iter; ++iter) {
        Pose T_cur = paramsToPose(xi);
        
        Eigen::VectorXd r;
        Eigen::MatrixXd J_full;
        computeResidualsAndJacobian(residuals, T_cur, d, h, tilt, r, J_full);
        
        // 提取相关列
        Eigen::MatrixXd J(r.size(), var_dim);
        for (int i = 0; i < var_dim; ++i) {
            J.col(i) = J_full.col(opt_vars[i]);
        }
        
        double error = r.squaredNorm();
        
        // 正规方程
        Eigen::MatrixXd H = J.transpose() * J;
        Eigen::VectorXd b = -J.transpose() * r;
        
        // LM阻尼
        for (int i = 0; i < var_dim; ++i) {
            H(i, i) += lambda * H(i, i);
        }
        
        Eigen::VectorXd delta = H.ldlt().solve(b);
        
        // 更新参数
        Vector6d xi_new = xi;
        double d_new = d, h_new = h, tilt_new = tilt;
        
        for (int i = 0; i < var_dim; ++i) {
            int var_idx = opt_vars[i];
            if (var_idx < 6) xi_new(var_idx) += delta(i);
            else if (var_idx == 6) d_new += delta(i);
            else if (var_idx == 7) h_new += delta(i);
            else if (var_idx == 8) tilt_new += delta(i);
        }
        
        // 评估新误差
        Pose T_new = paramsToPose(xi_new);
        VehicleModel model_new(d_new, h_new, tilt_new);
        model_new.applyPose(T_new);
        
        double new_error = 0.0;
        for (const auto& res : residuals) {
            const auto* plate = model_new.getPlateById(res.plate_id);
            if (!plate) continue;
            cv::Point2f proj = cam_.project(
                plate->corners_world[res.corner_idx], Pose::Identity());
            double dx = (proj.x - res.obs_pt.x) * res.weight;
            double dy = (proj.y - res.obs_pt.y) * res.weight;
            new_error += dx*dx + dy*dy;
        }
        
        // 接受或拒绝更新
        if (new_error < error) {
            xi = xi_new;
            d = d_new; h = h_new; tilt = tilt_new;
            lambda *= 0.1;
            
            if (std::sqrt(error / residuals.size()) < 1.0) {  // 像素级收敛
                result.converged = true;
                break;
            }
        } else {
            lambda *= 10.0;
        }
    }
    
    result.d_opt = d;
    result.h_opt = h;
    result.tilt_opt = tilt;
    result.T_cam_object = paramsToPose(xi);
    
    // 计算最终误差
    VehicleModel final_model(d, h, tilt);
    final_model.applyPose(result.T_cam_object);
    double final_err = 0.0;
    for (const auto& res : residuals) {
        const auto* plate = final_model.getPlateById(res.plate_id);
        if (!plate) continue;
        cv::Point2f proj = cam_.project(
            plate->corners_world[res.corner_idx], Pose::Identity());
        double dx = proj.x - res.obs_pt.x;
        double dy = proj.y - res.obs_pt.y;
        final_err += dx*dx + dy*dy;
    }
    result.final_error = std::sqrt(final_err / residuals.size());
    result.iterations = max_iter;
    
    // 更新内部模型
    model_.updateParameters(d, h, tilt);
    
    return result;
}

// ==================== ModelVisualizer实现 ====================

cv::Mat ModelVisualizer::render3DView(
    const VehicleModel& model,
    const Pose& T_cam_object,
    const Camera& cam,
    const cv::Mat& bg_image,
    const FrameObservation* obs) {
    
    cv::Mat img = bg_image.empty() ? 
        cv::Mat(cam.img_height, cam.img_width, CV_8UC3, cv::Scalar(0,0,0)) : 
        bg_image.clone();
    
    // 绘制每个装甲板
    const std::array<cv::Scalar, 4> colors = {
        cv::Scalar(0, 255, 0),    // 前 - 绿
        cv::Scalar(0, 0, 255),    // 右 - 红
        cv::Scalar(255, 0, 0),    // 后 - 蓝
        cv::Scalar(0, 255, 255)   // 左 - 黄
    };
    
    for (const auto& plate : model.getPlates()) {
        drawArmorPlate(img, cam, plate, T_cam_object, colors[plate.id], 2);
        
        // 绘制法向量
        Vec3 center_cam = T_cam_object * plate.center_object;
        Vec3 normal_end = center_cam + 0.1 * (T_cam_object.rotation() * plate.normal_object);
        
        cv::Point2f p1 = cam.project(plate.center_object, T_cam_object);
        cv::Point2f p2 = cam.project(
            T_cam_object.inverse() * normal_end, T_cam_object);
        
        if (p1.x > 0 && p2.x > 0) {
            cv::arrowedLine(img, p1, p2, cv::Scalar(255,255,255), 1);
        }
    }
    
    // 绘制物体坐标轴
    drawAxis(img, cam, T_cam_object, 0.2);
    
    // 叠加观测（如果提供）
    if (obs) {
        for (const auto& armor : obs->armors) {
            for (int i = 0; i < 4; ++i) {
                cv::circle(img, armor.corners_img[i], 3, 
                          cv::Scalar(255, 255, 255), -1);
            }
            cv::polylines(img, std::vector<std::vector<cv::Point>>{
                {armor.corners_img[0], armor.corners_img[1], 
                 armor.corners_img[2], armor.corners_img[3]}
            }, true, cv::Scalar(255, 255, 255), 1);
        }
    }
    
    return img;
}

cv::Mat ModelVisualizer::renderWireframe(
    const VehicleModel& model,
    const Pose& T_cam_object,
    const Camera& cam,
    int width, int height) {
    
    cv::Mat img(height, width, CV_8UC3, cv::Scalar(30, 30, 30));
    
    // 使用虚拟相机进行3D渲染
    // 这里简化：使用正交投影或简单的透视投影
    
    // 计算包围盒确定视角
    double min_x = 1e9, max_x = -1e9;
    double min_y = 1e9, max_y = -1e9;
    
    for (const auto& plate : model.getPlates()) {
        for (const auto& c : plate.corners_world) {
            Vec3 pt_cam = T_cam_object * c;
            min_x = std::min(min_x, pt_cam.x());
            max_x = std::max(max_x, pt_cam.x());
            min_y = std::min(min_y, pt_cam.y());
            max_y = std::max(max_y, pt_cam.y());
        }
    }
    
    double scale = std::min(width, height) / std::max(max_x - min_x, max_y - min_y) * 0.4;
    double cx = width / 2.0;
    double cy = height / 2.0;
    
    auto project2d = [&](const Vec3& pt_world) -> cv::Point2f {
        Vec3 pt_cam = T_cam_object * pt_world;
        return cv::Point2f(
            static_cast<float>(cx + pt_cam.x() * scale),
            static_cast<float>(cy - pt_cam.y() * scale)  // Y轴翻转
        );
    };
    
    const std::array<cv::Scalar, 4> colors = {
        cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255),
        cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 255)
    };
    
    // 绘制装甲板
    for (const auto& plate : model.getPlates()) {
        std::vector<cv::Point> pts;
        for (const auto& c : plate.corners_object) {
            pts.push_back(project2d(c));
        }
        
        cv::polylines(img, std::vector<std::vector<cv::Point>>{pts}, 
                     true, colors[plate.id], 2);
        
        // 填充半透明
        cv::fillConvexPoly(img, pts, colors[plate.id] * 0.3);
        
        // 中心点
        cv::Point2f center = project2d(plate.center_object);
        cv::circle(img, center, 4, cv::Scalar(255, 255, 255), -1);
        
        // 法向量
        Vec3 normal_end = plate.center_object + 0.15 * plate.normal_object;
        cv::Point2f n_end = project2d(normal_end);
        cv::arrowedLine(img, center, n_end, cv::Scalar(255, 255, 255), 2);
    }
    
    // 绘制中心
    cv::circle(img, cv::Point(static_cast<int>(cx), static_cast<int>(cy)), 
              5, cv::Scalar(255, 255, 255), -1);
    
    // 坐标轴
    Vec3 origin(0,0,0);
    cv::Point2f o = project2d(origin);
    cv::Point2f x_end = project2d(Vec3(0.2, 0, 0));
    cv::Point2f y_end = project2d(Vec3(0, 0.2, 0));
    cv::Point2f z_end = project2d(Vec3(0, 0, 0.2));
    
    cv::arrowedLine(img, o, x_end, cv::Scalar(0, 0, 255), 2);  // X-红
    cv::arrowedLine(img, o, y_end, cv::Scalar(0, 255, 0), 2);  // Y-绿
    cv::arrowedLine(img, o, z_end, cv::Scalar(255, 0, 0), 2);  // Z-蓝
    
    return img;
}

void ModelVisualizer::drawAxis(cv::Mat& img, const Camera& cam,
                                const Pose& T_cam_world, double length) {
    Vec3 origin(0, 0, 0);
    Vec3 x_end(length, 0, 0);
    Vec3 y_end(0, length, 0);
    Vec3 z_end(0, 0, length);
    
    cv::Point2f o = cam.project(origin, T_cam_world);
    cv::Point2f x = cam.project(x_end, T_cam_world);
    cv::Point2f y = cam.project(y_end, T_cam_world);
    cv::Point2f z = cam.project(z_end, T_cam_world);
    
    if (o.x > 0) {
        cv::arrowedLine(img, o, x, cv::Scalar(0, 0, 255), 2);
        cv::arrowedLine(img, o, y, cv::Scalar(0, 255, 0), 2);
        cv::arrowedLine(img, o, z, cv::Scalar(255, 0, 0), 2);
    }
}

void ModelVisualizer::drawArmorPlate(cv::Mat& img, const Camera& cam,
                                      const ArmorPlate& plate,
                                      const Pose& T_cam_object,
                                      const cv::Scalar& color,
                                      int thickness) {
    std::vector<cv::Point> pts;
    for (const auto& c : plate.corners_object) {
        cv::Point2f p = cam.project(c, T_cam_object);
        pts.emplace_back(static_cast<int>(p.x), static_cast<int>(p.y));
    }
    
    if (pts.size() == 4) {
        cv::polylines(img, std::vector<std::vector<cv::Point>>{pts}, 
                     true, color, thickness);
        
        // 填充半透明
        cv::Mat overlay;
        img.copyTo(overlay);
        cv::fillConvexPoly(overlay, pts, color);
        cv::addWeighted(img, 0.7, overlay, 0.3, 0, img);
        
        // 角点编号
        for (int i = 0; i < 4; ++i) {
            cv::putText(img, std::to_string(i), pts[i], 
                       cv::FONT_HERSHEY_SIMPLEX, 0.4, 
                       cv::Scalar(255, 255, 255), 1);
        }
    }
}

// ==================== 工具函数实现 ====================

cv::Mat eigenToCvMat(const Mat3& R) {
    cv::Mat mat(3, 3, CV_64F);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            mat.at<double>(i, j) = R(i, j);
    return mat;
}

cv::Mat eigenToCvMat(const Vec3& t) {
    cv::Mat mat(3, 1, CV_64F);
    for (int i = 0; i < 3; ++i)
        mat.at<double>(i, 0) = t(i);
    return mat;
}

Mat3 cvMatToEigen3d(const cv::Mat& R) {
    Mat3 mat;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            mat(i, j) = R.at<double>(i, j);
    return mat;
}

Vec3 cvMatToEigenVec(const cv::Mat& t) {
    return Vec3(t.at<double>(0,0), t.at<double>(1,0), t.at<double>(2,0));
}

Pose cvToEigenPose(const cv::Mat& R, const cv::Mat& t) {
    Pose T = Pose::Identity();
    T.linear() = cvMatToEigen3d(R);
    T.translation() = cvMatToEigenVec(t);
    return T;
}

Eigen::Quaterniond rotmatToQuat(const Mat3& R) {
    return Eigen::Quaterniond(R);
}

Mat3 quatToRotmat(const Eigen::Quaterniond& q) {
    return q.toRotationMatrix();
}

} // namespace armor_model