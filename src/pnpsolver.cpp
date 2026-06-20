#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

// ---------- 旋转矩阵 → 四元数 (x, y, z, w) ----------
cv::Vec4d rotMatrixToQuat(const cv::Mat& R) {
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
    } else if ((m00 > m11) && (m00 > m22)) {
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

// ---------- 核心估计函数（带畸变系数）----------
bool estimateObjectPoseWorld(
    const cv::Mat& K,           // 内参 3x3
    const cv::Mat& distCoeffs,  // 畸变 1x5
    const cv::Mat& R_wc,        // 相机在世界系的旋转 3x3
    const cv::Mat& t_wc,        // 相机在世界系的平移 3x1
    const cv::Mat& objPts,      // 物体坐标系下的三维点 (Nx3)
    const cv::Mat& imgPts,      // 对应的图像点 (Nx2)
    cv::Vec4d& q_wo,            // 输出四元数 (x,y,z,w)
    cv::Mat& t_wo               // 输出平移 (3x1)
) {
    cv::Mat rvec_co, tvec_co;
    bool ok = cv::solvePnP(objPts, imgPts, K, distCoeffs,
                           rvec_co, tvec_co, false, cv::SOLVEPNP_EPNP);
    if (!ok) return false;

    cv::Mat R_co;
    cv::Rodrigues(rvec_co, R_co);
    cv::Mat R_wo = R_wc * R_co;
    t_wo = t_wc + R_wc * tvec_co;
    q_wo = rotMatrixToQuat(R_wo);
    return true;
}

// ---------- 从 YAML 加载相机参数 ----------
bool loadCameraYAML(const std::string& path,
                    cv::Mat& K,
                    cv::Mat& dist,
                    int& width,
                    int& height) {
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "无法打开 " << path << std::endl;
        return false;
    }
    fs["camera_matrix"] >> K;
    fs["distortion_coefficients"] >> dist;
    fs["image_width"] >> width;
    fs["image_height"] >> height;
    fs.release();
    if (K.rows != 3 || K.cols != 3 || dist.cols != 5) {
        std::cerr << "参数尺寸错误" << std::endl;
        return false;
    }
    return true;
}

// ================= 演示 =================
int main() {
    // 1. 加载你的标定参数
    cv::Mat K, dist;
    int imgW, imgH;
    if (!loadCameraYAML("asset/camera_calibration.yml", K, dist, imgW, imgH))
        return -1;
    std::cout << "相机内参: \n" << K << "\n畸变: " << dist << "\n\n";

    // 2. 这里填写你已知的相机外参（世界坐标系下）
    //    ↓↓↓ 下面是示例数据，请换成你自己的相机真实位姿
    cv::Mat R_wc = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat t_wc = (cv::Mat_<double>(3,1) << 0.0, 0.0, 5.0);

    // 3. 物体上四个角点的三维坐标（物体自身坐标系，已知）
    //    ↓↓↓ 请换成你自己的物体点坐标
    double height = 125.0; // 物体边长
    double width = 135.0; //
    std::vector<cv::Point3d> objList = {
        cv::Point3d(-width/2, -height/2, 0),
        cv::Point3d( width/2, -height/2, 0),
        cv::Point3d( width/2,  height/2, 0),
        cv::Point3d(-width/2,  height/2, 0)
    };
    cv::Mat objPts(objList, false);  // 4x3

    // 4. 对应图像点（从图像上提取的像素坐标，通常已经包含畸变）
    //    ↓↓↓ 请换成你自己观测到的图像点
    //    这里为了演示，用理想投影+畸变模拟生成一组点
    cv::Mat t_wo_true = (cv::Mat_<double>(3,1) << 1.0, 0.0, 8.0);
    cv::Mat rvec_true = (cv::Mat_<double>(3,1) << 0.0, CV_PI/6.0, 0.0); // 绕Y轴30度
    cv::Mat R_wo_true;
    cv::Rodrigues(rvec_true, R_wo_true);

    cv::Mat R_cw = R_wc.t();
    cv::Mat t_cw = -R_cw * t_wc;
    std::vector<cv::Point2d> imgList;
    for (size_t i = 0; i < objList.size(); ++i) {
        cv::Mat pObj = (cv::Mat_<double>(3,1) << objList[i].x, objList[i].y, objList[i].z);
        cv::Mat pWorld = R_wo_true * pObj + t_wo_true;
        cv::Mat pCam = R_cw * (pWorld - t_cw);
        // 理想投影
        double u = K.at<double>(0,0) * pCam.at<double>(0) / pCam.at<double>(2) + K.at<double>(0,2);
        double v = K.at<double>(1,1) * pCam.at<double>(1) / pCam.at<double>(2) + K.at<double>(1,2);
        imgList.push_back(cv::Point2d(u, v));
    }
    cv::Mat imgPts(imgList, false);

    // 5. 执行位姿估计
    cv::Vec4d q_wo_est;
    cv::Mat t_wo_est;
    bool ok = estimateObjectPoseWorld(K, dist, R_wc, t_wc, objPts, imgPts, q_wo_est, t_wo_est);
    if (!ok) {
        std::cerr << "PnP 失败" << std::endl;
        return -1;
    }

    // 6. 输出结果
    cv::Vec4d q_wo_true = rotMatrixToQuat(R_wo_true);
    std::cout << "真实四元数 (x,y,z,w): " << q_wo_true << std::endl;
    std::cout << "估计四元数 (x,y,z,w): " << q_wo_est << std::endl;
    std::cout << "真实位置: " << t_wo_true.t() << std::endl;
    std::cout << "估计位置: " << t_wo_est.t() << std::endl;

    return 0;
}