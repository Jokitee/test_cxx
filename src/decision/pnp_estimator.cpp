/**
 * @brief PnP(Perspective-n-Point)位姿解算器
 * @author jokit
 * @date 2026-07-15
 */
#include "vision_system/decision/pnp_estimator.hpp"
#include <iostream>

PoseEstimator::PoseEstimator(const std::string& yamlPath) {
    loadCameraYAML(yamlPath);
}

bool PoseEstimator::loadCameraYAML(const std::string& path) {
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "无法打开 " << path << std::endl;
        return false;
    }
    fs["camera_matrix"]            >> K_;
    fs["distortion_coefficients"]  >> dist_;
    fs["image_width"]              >> width_;
    fs["image_height"]             >> height_;
    fs.release();

    if (K_.rows != 3 || K_.cols != 3 || dist_.cols != 5) {
        std::cerr << "参数尺寸错误" << std::endl;
        return false;
    }
    intrinsicsLoaded_ = true;
    return true;
}

void PoseEstimator::setWorldPose(const cv::Mat& R_wc, const cv::Mat& t_wc) {
    CV_Assert(R_wc.type() == CV_64F && R_wc.rows == 3 && R_wc.cols == 3);
    CV_Assert(t_wc.type() == CV_64F);
    R_wc_ = R_wc.clone();
    t_wc_ = t_wc.clone();
    extrinsicsSet_ = true;
}

bool PoseEstimator::estimatePose(cv::InputArray objPts,
                                 cv::InputArray imgPts,
                                 cv::Vec4d&    q_wo,
                                 cv::Mat&      t_wo,
                                 cv::Mat&      R_wo) const
{
    if (!intrinsicsLoaded_ || !extrinsicsSet_) {
        std::cerr << "请先加载内参并设置外参" << std::endl;
        return false;
    }

    cv::Mat rvec_co, tvec_co;
    bool ok = cv::solvePnP(objPts, imgPts, K_, dist_,
                           rvec_co, tvec_co,
                           false, cv::SOLVEPNP_IPPE);
    if (!ok) return false;

    cv::Mat R_co;
    cv::Rodrigues(rvec_co, R_co);

    R_wo = R_wc_ * R_co;
    t_wo = t_wc_ + R_wc_ * tvec_co;

    q_wo = rotMatToQuat(R_wo);
    return true;
}

cv::Vec4d PoseEstimator::rotMatToQuat(const cv::Mat& R) {
    CV_Assert(R.type() == CV_64F && R.rows == 3 && R.cols == 3);

    double m00 = R.at<double>(0,0), m01 = R.at<double>(0,1), m02 = R.at<double>(0,2);
    double m10 = R.at<double>(1,0), m11 = R.at<double>(1,1), m12 = R.at<double>(1,2);
    double m20 = R.at<double>(2,0), m21 = R.at<double>(2,1), m22 = R.at<double>(2,2);

    double tr = m00 + m11 + m22;
    double qx, qy, qz, qw;

    if (tr > 0) {
        double s = std::sqrt(tr + 1.0) * 2;
        qw = 0.25 * s;
        qx = (m21 - m12) / s;
        qy = (m02 - m20) / s;
        qz = (m10 - m01) / s;
    } else if (m00 > m11 && m00 > m22) {
        double s = std::sqrt(1.0 + m00 - m11 - m22) * 2;
        qw = (m21 - m12) / s;
        qx = 0.25 * s;
        qy = (m01 + m10) / s;
        qz = (m02 + m20) / s;
    } else if (m11 > m22) {
        double s = std::sqrt(1.0 + m11 - m00 - m22) * 2;
        qw = (m02 - m20) / s;
        qx = (m01 + m10) / s;
        qy = 0.25 * s;
        qz = (m12 + m21) / s;
    } else {
        double s = std::sqrt(1.0 + m22 - m00 - m11) * 2;
        qw = (m10 - m01) / s;
        qx = (m02 + m20) / s;
        qy = (m12 + m21) / s;
        qz = 0.25 * s;
    }
    return cv::Vec4d(qx, qy, qz, qw);
}
