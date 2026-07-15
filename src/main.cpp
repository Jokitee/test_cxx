#include <iostream>
#include "CameraApi.h"
#include "armor_model.hpp"
#include "opencv2/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/dnn.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdio.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <map>
#include <string>

unsigned char           * g_pRgbBuffer;

struct lightbors
{
    cv::RotatedRect left_lightbors;
    cv::RotatedRect right_lightbors;
    cv::RotatedRect left_armor;
    cv::RotatedRect right_armor;
    cv::Point2f armor_center;
    cv::Point2f armor_point[4];     // 次序为 左上 -> 右上 -> 右下 -> 左下  基于Opencv右正下正的坐标系
    std::string ID;         // 装甲板数字
    bool righting = 0;

    // 反馈当前检测各参数
    std::string test_ID;
};

// 车辆装甲板类：将同一车辆检测到的装甲板聚合在一起
class VehicleArmors {
public:
    std::string vehicle_id;          // 车辆装甲板数字/名称（如 "1", "2" 等）
    std::vector<lightbors> plates;   // 属于该车辆的装甲板集合

    VehicleArmors() = default;
    explicit VehicleArmors(const std::string& id) : vehicle_id(id) {}

    // 录入装甲板
    void addPlate(const lightbors& armor) {
        if (armor.ID == vehicle_id || vehicle_id.empty()) {
            if (vehicle_id.empty()) vehicle_id = armor.ID;
            plates.push_back(armor);
        }
    }

    // 根据 armor_model 的命名规则生成观测数据，匹配装甲板的 plate_id
    armor_model::FrameObservation getObservation(int64_t timestamp = 0) const {
        armor_model::FrameObservation obs;
        obs.timestamp = timestamp;

        // 根据装甲板在图像中的 x 坐标从左到右排序
        std::vector<lightbors> sorted_plates = plates;
        std::sort(sorted_plates.begin(), sorted_plates.end(), [](const lightbors& a, const lightbors& b) {
            return a.armor_center.x < b.armor_center.x;
        });

        for (size_t i = 0; i < sorted_plates.size(); ++i) {
            armor_model::ArmorObservation a_obs;
            
            // 配合 armor_model 命名规则: 0:前 1:右 2:后 3:左
            // 根据图像中的装甲板相对位置判断 ID 
            if (sorted_plates.size() == 1) {
                a_obs.plate_id = 0; // 视野中只有一个，默认作为前装甲板(0)
            } else if (sorted_plates.size() == 2) {
                // 如果有两个装甲板，假设看到的是车辆偏侧面
                // 左侧的是左装甲板(3)，右侧的是前装甲板(0)
                if (i == 0) a_obs.plate_id = 3; 
                if (i == 1) a_obs.plate_id = 0; 
            } else {
                a_obs.plate_id = i % 4; // 兜底逻辑
            }

            a_obs.confidence = 1.0;
            for (int j = 0; j < 4; j++) {
                a_obs.corners_img[j] = sorted_plates[i].armor_point[j];
            }
            obs.armors.push_back(a_obs);
        }
        return obs;
    }
};

// 角度矫正函数(针对于Opencv矩形拟合的狗屎算法)
float getright_angle(const cv::RotatedRect& rect) {
    float angle = rect.angle;          // 总是 width 边的方向，∈[-90°, 0°]
    if (rect.size.width < rect.size.height)
        angle += 90.f;                // 现在 angle 是 height 边（长边）的方向

    // 归一化到 [0, 180)
    if (angle < 0.f)   angle += 180.f;   // 例如 -10° → 170°
    if (angle >= 180.f) angle -= 180.f;  // 理论上不会超过 180，但可防万一
    return angle;
}

// 长边拉长函数
cv::RotatedRect stretchLongSide(const cv::RotatedRect& rect, float scale) {
    cv::RotatedRect result = rect;
    
    // 比较 width 和 height，长边是较大的那个
    if (rect.size.width >= rect.size.height) {
        result.size.width *= scale;  // width 是长边，拉长它
    } else {
        result.size.height *= scale; // height 是长边，拉长它
    }
    
    return result;
}

// 角点交换函数
void points_exchange(cv::Point2f points[4]) {
    std::vector<cv::Point2f> pts(points, points + 4);
    
    // Step 1: 按 y 坐标升序排序（y 小在上，y 大在下）
    // y 相同则按 x 排序（x 小在左）
    std::sort(pts.begin(), pts.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
        if (std::abs(a.y - b.y) > 1e-5) {
            return a.y < b.y;  // y 小的排在前面（上方）
        }
        return a.x < b.x;      // y 相同，x 小的排在前面
    });
    
    // Step 2: 分组 —— 上方两点（y 较小）和下方两点（y 较大）
    cv::Point2f top1 = pts[0];   // y 最小
    cv::Point2f top2 = pts[1];   // y 次小
    cv::Point2f bottom1 = pts[2]; // y 次大
    cv::Point2f bottom2 = pts[3]; // y 最大
    
    // Step 3: 上方两点按 x 排序，确定左上和右上
    cv::Point2f top_left, top_right;
    if (top1.x < top2.x) {
        top_left  = top1;   // x 小 → 左上
        top_right = top2;   // x 大 → 右上
    } else {
        top_left  = top2;
        top_right = top1;
    }
    
    // Step 4: 下方两点按 x 排序，确定左下和右下
    cv::Point2f bottom_left, bottom_right;
    if (bottom1.x < bottom2.x) {
        bottom_left  = bottom1;  // x 小 → 左下
        bottom_right = bottom2;  // x 大 → 右下
    } else {
        bottom_left  = bottom2;
        bottom_right = bottom1;
    }
    
    // Step 5: 按顺时针重新赋值：左上 → 右上 → 右下 → 左下
    points[0] = top_left;      // 左上
    points[1] = top_right;     // 右上
    points[2] = bottom_right;  // 右下
    points[3] = bottom_left;   // 左下
}

// 装甲板计算函数
void set_light(lightbors& armor_light){

    armor_light.left_lightbors = stretchLongSide(armor_light.left_lightbors, 2.4f);
    armor_light.right_lightbors = stretchLongSide(armor_light.right_lightbors, 2.4f);
    // 获取角点并放进数组内部
    cv::Point2f left_points[4];
    cv::Point2f right_points[4];
    armor_light.left_lightbors.points(left_points);
    armor_light.right_lightbors.points(right_points);
    points_exchange(left_points);
    points_exchange(right_points);

    for(size_t i = 0 ; i < 4; i++){
        if (i == 0 || i == 3)
        {
            armor_light.armor_point[i] = left_points[i];
        }else{
            armor_light.armor_point[i] = right_points[i];
        }
        
    }
    armor_light.armor_center = {(armor_light.left_armor.center.x + armor_light.right_armor.center.x) / 2 , (armor_light.left_armor.center.y + armor_light.right_armor.center.y) / 2};

}

// 获取待检测的ROI (已修改为适配 ONNX 训练集的直接矩形裁切方案)
void CropAndResize(const lightbors& armor_light, cv::Mat& endB, cv::Mat& outImage) {
    // 在前面的 set_light 函数中，灯条长度已经被 stretchLongSide 拉长了 2.4 倍。
    // 这与 auto_aim 源码中 (上下各自延伸 1.125) 非常接近。
    // 因此我们直接用这4个延展后的角点计算包围盒 Bounding Box。
    
    float min_x = std::min({armor_light.armor_point[0].x, armor_light.armor_point[1].x, armor_light.armor_point[2].x, armor_light.armor_point[3].x});
    float max_x = std::max({armor_light.armor_point[0].x, armor_light.armor_point[1].x, armor_light.armor_point[2].x, armor_light.armor_point[3].x});
    float min_y = std::min({armor_light.armor_point[0].y, armor_light.armor_point[1].y, armor_light.armor_point[2].y, armor_light.armor_point[3].y});
    float max_y = std::max({armor_light.armor_point[0].y, armor_light.armor_point[1].y, armor_light.armor_point[2].y, armor_light.armor_point[3].y});

    // 边界安全校验，防止截取越界
    int roi_left = std::max(static_cast<int>(min_x), 0);
    int roi_top = std::max(static_cast<int>(min_y), 0);
    int roi_right = std::min(static_cast<int>(max_x), endB.cols);
    int roi_bottom = std::min(static_cast<int>(max_y), endB.rows);

    cv::Rect roi(roi_left, roi_top, roi_right - roi_left, roi_bottom - roi_top);

    // 进行裁切
    if (roi.area() > 0) {
        outImage = endB(roi).clone();
    } else {
        outImage = cv::Mat();
    }
}

// 自研分类算法 (原 SVM 模型 - 已注释化)
/*
class FeatureDetector8Classes {
private:
    cv::Ptr<cv::ml::SVM> svm;
    cv::HOGDescriptor hog;
    cv::Size imgSize = cv::Size(48, 36); // 归一化尺寸，需与训练时一致
    
    // 映射表：SVM输出的索引 0~7 对应您的8个特征名称
    std::vector<std::string> class_names = {"1", "2", "3", "4", "5", "6", "7", "sentinel"};
 
public:
    FeatureDetector8Classes() {
        // 初始化HOG参数（低算力优选参数）
        hog.winSize = imgSize;
        hog.blockSize = cv::Size(12, 12);
        hog.blockStride = cv::Size(6, 6);
        hog.cellSize = cv::Size(6, 6);
        hog.nbins = 9;
        
        // 加载训练好的SVM模型
        try {
            svm = cv::ml::SVM::load("asset/svm_model.xml");
            if (svm.empty()) {
                std::cerr << "模型加载失败" << std::endl;
            }
        } catch (const cv::Exception& e) {
            std::cerr << "模型加载异常: " << e.what() << std::endl;
            svm.release();
        }
    }
 
    std::string detect(const cv::Mat& inputImg, lightbors& armor_light) {
        if (inputImg.empty()) return "unknown";
 
        // ================= 1. 图像归一化处理 =================
        cv::Mat normalizedImg;
        // 尺寸归一化
        resize(inputImg, normalizedImg, imgSize, 0, 0, cv::INTER_LINEAR);

        if (normalizedImg.channels() == 3) {
        cvtColor(normalizedImg, normalizedImg, cv::COLOR_BGR2GRAY);
        }

        // 光照归一化（抗光照干扰）
        equalizeHist(normalizedImg, normalizedImg);
        // ====================================================
 
        // ================= 2. 特征提取 =================
        std::vector<float> descriptors;
        hog.compute(normalizedImg, descriptors);
        cv::Mat featureMat(1, descriptors.size(), CV_32FC1, descriptors.data());
 
        // ================= 3. 分类预测 =================
        cv::Mat response;
        svm->predict(featureMat, response);
        
        // 获取SVM预测的整数索引 (0~7)
        int predicted_label = static_cast<int>(response.at<float>(0, 0));
        
        // 将索引映射为指定的名称并返回
        if (predicted_label >= 0 && predicted_label < 8) {
            armor_light.righting = 1;
            return class_names[predicted_label];
        }
        armor_light.righting = 0;
        return "unknown";
    }
};
*/

// 新的 ONNX 分类模型算法
class FeatureDetectorONNX {
private:
    cv::dnn::Net net_;
    // 对应 auto_aim 中 9 个分类：0-one, 1-two, 2-three, 3-four, 4-five, 5-sentry, 6-outpost, 7-base, 8-not_armor
    std::vector<std::string> class_names = {"1", "2", "3", "4", "5", "sentry", "outpost", "base", "not_armor"};
    
public:
    FeatureDetectorONNX() {
        try {
            // 加载 ONNX 模型
            net_ = cv::dnn::readNetFromONNX("asset/tiny_resnet.onnx");
            if (net_.empty()) {
                std::cerr << "ONNX 模型加载失败" << std::endl;
            }
        } catch (const cv::Exception& e) {
            std::cerr << "ONNX 模型加载异常: " << e.what() << std::endl;
        }
    }

    std::string detect(const cv::Mat& inputImg, lightbors& armor_light) {
        if (inputImg.empty()) return "unknown";

        // 1. 转灰度图
        cv::Mat gray;
        if (inputImg.channels() == 3) {
            cv::cvtColor(inputImg, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = inputImg.clone();
        }

        // 2. 构建 32x32 的全黑底图，进行等比例缩放
        cv::Mat input = cv::Mat(32, 32, CV_8UC1, cv::Scalar(0));
        double scale = std::min(32.0 / gray.cols, 32.0 / gray.rows);
        int h = static_cast<int>(gray.rows * scale);
        int w = static_cast<int>(gray.cols * scale);
        
        if (h == 0 || w == 0) {
            armor_light.righting = 0;
            return "unknown";
        }
        
        cv::Rect roi(0, 0, w, h);
        cv::resize(gray, input(roi), cv::Size(w, h));

        // 3. 构建 Blob 并送入网络 (归一化 1.0/255.0)
        cv::Mat blob = cv::dnn::blobFromImage(input, 1.0 / 255.0, cv::Size(), cv::Scalar());
        net_.setInput(blob);
        cv::Mat outputs = net_.forward();

        // 4. 后处理 (Softmax)
        float max_val = *std::max_element(outputs.begin<float>(), outputs.end<float>());
        cv::exp(outputs - max_val, outputs);
        float sum = cv::sum(outputs)[0];
        outputs /= sum;

        // 5. 提取置信度与类别 ID
        double confidence;
        cv::Point label_point;
        cv::minMaxLoc(outputs.reshape(1, 1), nullptr, &confidence, nullptr, &label_point);
        int label_id = label_point.x;

        // 设置阈值并映射名称
        // 阈值设为 0.5，且不采用 8(not_armor)
        if (confidence > 0.5 && label_id >= 0 && label_id < 8) {
            armor_light.righting = 1;
            return class_names[label_id];
        }

        armor_light.righting = 0;
        return "unknown";
    }
};

// Pnp解算类
class PoseEstimator {
public:
    // ========== 构造函数 ==========

    // 空构造，之后手动设置参数
    PoseEstimator() = default;

    // 从 YAML 文件加载内参和畸变
    explicit PoseEstimator(const std::string& yamlPath) {
        loadCameraYAML(yamlPath);
    }

    // ========== 公有接口 ==========

    // 从 YAML 加载相机内参和畸变系数
    bool loadCameraYAML(const std::string& path) {
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

    // 设置相机在世界坐标系下的外参
    void setWorldPose(const cv::Mat& R_wc, const cv::Mat& t_wc) {
        CV_Assert(R_wc.type() == CV_64F && R_wc.rows == 3 && R_wc.cols == 3);
        CV_Assert(t_wc.type() == CV_64F);
        R_wc_ = R_wc.clone();
        t_wc_ = t_wc.clone();
        extrinsicsSet_ = true;
    }

    // 核心：估计物体在世界坐标系下的位姿
    //   objPts  - 物体坐标系下的三维点 (N×3)
    //   imgPts  - 对应的图像二维点 (N×2)
    //   q_wo    - 输出四元数 (x, y, z, w)
    //   t_wo    - 输出平移向量 (3×1)
    bool estimatePose(cv::InputArray objPts,
                      cv::InputArray imgPts,
                      cv::Vec4d&    q_wo,
                      cv::Mat&      t_wo,
                      cv::Mat&      R_wo) const
    {
        if (!intrinsicsLoaded_ || !extrinsicsSet_) {
            std::cerr << "请先加载内参并设置外参" << std::endl;
            return false;
        }

        // 1) solvePnP → 相机系下的物体位姿
        cv::Mat rvec_co, tvec_co;
        bool ok = cv::solvePnP(objPts, imgPts, K_, dist_,
                               rvec_co, tvec_co,
                               false, cv::SOLVEPNP_IPPE);
        if (!ok) return false;

        // 2) 旋转向量 → 旋转矩阵
        cv::Mat R_co;
        cv::Rodrigues(rvec_co, R_co);

        // 3) 相机系 → 世界系
        R_wo = R_wc_ * R_co;
        t_wo = t_wc_ + R_wc_ * tvec_co;

        // 4) 旋转矩阵 → 四元数
        q_wo = rotMatToQuat(R_wo);
        return true;
    }

    // ========== 静态工具函数 ==========

    // 旋转矩阵 → 四元数 (x, y, z, w)
    static cv::Vec4d rotMatToQuat(const cv::Mat& R) {
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

    // ========== Getter ==========
    const cv::Mat& cameraMatrix()    const { return K_; }
    const cv::Mat& distCoeffs()      const { return dist_; }
    const cv::Mat& worldRotation()   const { return R_wc_; }
    const cv::Mat& worldTranslation()const { return t_wc_; }
    int imageWidth()                 const { return width_; }
    int imageHeight()                const { return height_; }

private:
    // 内参
    cv::Mat K_;
    cv::Mat dist_;
    int     width_  = 0;
    int     height_ = 0;

    // 外参（相机在世界系的位姿）
    cv::Mat R_wc_;
    cv::Mat t_wc_;

    // 状态标志
    bool intrinsicsLoaded_ = false;
    bool extrinsicsSet_    = false;
};

// 画三维立体框
void drawCube(
    cv::Mat& image,
    const cv::Mat& R_wo,
    const cv::Mat& t_wo,
    const cv::Mat& K,
    const cv::Mat& dist)
{
    // 静态缓存，避免重复分配
    static std::vector<cv::Point3f> cube3d;
    std::vector<cv::Point3f> pts_cam;
    std::vector<cv::Point2f> pts2d;
    
    if (cube3d.empty()) {
        float h = 125.0f, w = 135.0f, l = 30.0f;
        cube3d = {
        {-w/2,-h/2,0}, {w/2,-h/2,0}, {w/2,h/2,0}, {-w/2,h/2,0},
        {-w/2,-h/2,l}, {w/2,-h/2,l}, {w/2,h/2,l}, {-w/2,h/2,l}
        };
    }
    
    pts_cam.resize(8);
    pts2d.resize(8);
    
    // 使用原始指针/数组，避免 cv::Mat 堆分配
    const double* R = R_wo.ptr<double>();
    const double* t = t_wo.ptr<double>();
    
    for (int i = 0; i < 8; ++i) {
        const auto& p = cube3d[i];
        double x = R[0]*p.x + R[1]*p.y + R[2]*p.z + t[0];
        double y = R[3]*p.x + R[4]*p.y + R[5]*p.z + t[1];
        double z = R[6]*p.x + R[7]*p.y + R[8]*p.z + t[2];
        pts_cam[i] = cv::Point3f(static_cast<float>(x),
                                  static_cast<float>(y),
                                  static_cast<float>(z));
    }
    
    cv::projectPoints(pts_cam, cv::Vec3d::zeros(), cv::Vec3d::zeros(),
                      K, dist, pts2d);

    // 画 12 条边
    int edges[][2] = {
        {0,1},{1,2},{2,3},{3,0},   // 底面
        {4,5},{5,6},{6,7},{7,4},   // 顶面
        {0,4},{1,5},{2,6},{3,7}    // 竖边
    };
    for (auto& e : edges) {
        cv::line(image, pts2d[e[0]], pts2d[e[1]],
                 cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
    }
}

int main(){
    int iCameraCounts = 1;
    int iStatues = -1;
    int setStatues = -1;
    tSdkCameraDevInfo   tCameraEnumList;
    double*     pfExposureTime;
    int     hCamera;
    tSdkCameraCapbility     tCapability;
    tSdkFrameHead   sFrameInfo;
    BYTE*       pbyBuffer;
    int     iDisplatFrames = 10000;
    int     channel = 3;

    /////////////////////////////////////////////////////////
    std::vector<lightbors> armor;
    FeatureDetectorONNX onnx_detector; // 在循环外部实例化一次，防止每帧重复加载模型
    /////////////////////////////////////////////////////////

    CameraSdkInit(1);

    iStatues = CameraEnumerateDevice(&tCameraEnumList, &iCameraCounts);
    printf("statue = %d\n", iStatues);

    printf("count = %d\n", iCameraCounts);
    if(iCameraCounts == 0){
        return -1;
    }

    iStatues = CameraInit(&tCameraEnumList, -1, -1, &hCamera);

    printf("state = %d\n", iStatues);
    if(iStatues != CAMERA_STATUS_SUCCESS){
        return -1;
    }

    CameraGetCapability(hCamera, &tCapability);

    g_pRgbBuffer = (unsigned char*)malloc(tCapability.sResolutionRange.iHeightMax*tCapability.sResolutionRange.iWidthMax*3);

    CameraPlay(hCamera);
    CameraSetAeState(hCamera, false);
    setStatues = CameraSetExposureTime(hCamera, 5000);
    CameraSetGain(hCamera, 100, 70, 50);
    printf("statue = %d\n", setStatues);

    if(tCapability.sIspCapacity.bMonoSensor){
        channel = 1;
        CameraSetIspOutFormat(hCamera, CAMERA_MEDIA_TYPE_MONO8);
    }else{
        channel = 3;
        CameraSetIspOutFormat(hCamera, CAMERA_MEDIA_TYPE_BGR8);
    }

    ///////////////     Pnp解算验证
    PoseEstimator estimator("asset/camera_calibration.yml");
    // 2 设置相机在世界系的外参（目前设定为世界坐标系和相机坐标系一致）
    cv::Mat R_wc = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat t_wc = cv::Mat::zeros(3, 1, CV_64F);

    //////////////////
    const auto& K = estimator.cameraMatrix();
    const auto& dist = estimator.distCoeffs();
    armor_model::Camera am_cam(K, dist, 1280, 720); // 这里的 1280,720 请替换为你相机的真实分辨率

     // 假设半宽 d=0.3m(300mm), 距地高度 h=0.15m(150mm), 装甲板倾角 15 度
    armor_model::VehicleModel am_model(0.300, 0.150, 15.0); 
    armor_model::ModelOptimizer optimizer(am_cam, am_model);
    //////////////////
    
    estimator.setWorldPose(R_wc, t_wc);
    float height = 125.0f; // 物体边长
    float width = 135.0f; //
    std::vector<cv::Point3f> objPts = {
        {-width/2, -height/2, 0}, 
        {width/2, -height/2, 0},
        {width/2,  height/2, 0},
        {-width/2,  height/2, 0},
    };
    ///////////////

    while(iDisplatFrames){

        if(CameraGetImageBuffer(hCamera, &sFrameInfo, &pbyBuffer, 1000) == CAMERA_STATUS_SUCCESS){

            CameraImageProcess(hCamera, pbyBuffer, g_pRgbBuffer, &sFrameInfo);

            cv::Mat matImage(
					cv::Size(sFrameInfo.iWidth,sFrameInfo.iHeight), 
					sFrameInfo.uiMediaType == CAMERA_MEDIA_TYPE_MONO8 ? CV_8UC1 : CV_8UC3,
					g_pRgbBuffer
					);


            /*      此内容为灯条检测部分
                    作用为测试灯条检测如何编写
            */
            std::vector<cv::Mat> channels;
            cv::split(matImage, channels);
            cv::Mat r = channels[2];
            cv::Mat mask;
            cv::threshold(r, mask, 150, 255, cv::THRESH_BINARY);
            // cv::Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
            // dilate(mask, r, kernel);
            std::vector<std::vector<cv::Point>> counters;
            cv::findContours(mask, counters, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            std::vector<cv::RotatedRect> end_rects;

            // 灯条检测逻辑
            for(auto& cnt : counters){
                cv::RotatedRect rotRect = cv::minAreaRect(cnt);

                // 面积检测筛选
                float area = rotRect.size.width * rotRect.size.height;
                if (area < 50 || area > 9000) continue;

                // 灯条比例筛选
                float width = std::min(rotRect.size.width, rotRect.size.height);
                float height = std::max(rotRect.size.width, rotRect.size.height);
                float ratio = height / width;
                if (ratio < 4 || ratio > 15) continue;

                // 灯条合理角度筛选
                if(getright_angle(rotRect) < 5 || getright_angle(rotRect) > 175)continue;

                end_rects.push_back(rotRect);

                cv::Point2f pts[4];
                rotRect.points(pts);
                for (int i = 0; i < 4; i++)
                {
                    cv::line(matImage, pts[i], pts[(i+1)%4], cv::Scalar(0,255,0), 2);
                }
            }

            armor.clear();
            if(end_rects.size() >= 2){
                // 灯条配对逻辑
                lightbors armor_light;
                float angle_TF, first_max, second_max, getheight, getlight, distance_TF, height_TF;
                for(size_t i = 0; i < end_rects.size()-1; i++){
                    for(size_t j = i + 1; j < end_rects.size(); j++){
 
                        /*
                        *****    灯条匹配逻辑      ******
                        */
                        // 倾斜角度偏差检测
                        angle_TF = getright_angle(end_rects[i]) - getright_angle(end_rects[j]);
                        if (fabs(angle_TF) > 6.5)continue;
                        
                        // 灯条距离与灯条长度比值检测
                        first_max = std::max(end_rects[i].size.width, end_rects[i].size.height);
                        second_max = std::max(end_rects[j].size.width, end_rects[j].size.height);
                        getheight = sqrt(pow(end_rects[i].center.x - end_rects[j].center.x, 2)+pow(end_rects[i].center.y - end_rects[j].center.y, 2));
                        getlight = (first_max + second_max) / 2;
                        distance_TF = getheight / getlight;
                        if (distance_TF > 3.0 || distance_TF < 2.1)continue;

                        /*
                        *****   灯条归位并定点    *****
                        */
                        if(end_rects[i].center.x < end_rects[j].center.x){
                            armor_light.left_lightbors = end_rects[i];
                            armor_light.right_lightbors = end_rects[j];
                        }else{
                            armor_light.left_lightbors = end_rects[j];
                            armor_light.right_lightbors = end_rects[i];
                        }


                        /* 测试 */
                        /////////
                        std::ostringstream oss;
                        oss << std::fixed << std::setprecision(2);
                        oss << "angle_TF:" << fabs(angle_TF) << "," << "distance_TF" << distance_TF;
                        armor_light.test_ID = oss.str();
                        /////////

                        // 计算每对灯条的装甲板
                        set_light(armor_light);

                        // 放入装甲板总数组
                        armor.push_back(armor_light);

                    }
                }

                /*
                const auto& cameraMatrix = estimator.cameraMatrix();
                const auto& distCoeffs = estimator.distCoeffs();
                std::vector<cv::Point2f>imgPts;
                cv::Mat R_wo, t_wo;
                cv::Vec4d   q_wo;
                imgPts.reserve(4);  // 预分配内存

                ////////////////////////////////
                // 【关键修复 1】：必须先运行 ONNX 识别模型！
                // 因为如果不运行它，cnt.righting 永远不会变成 1，下面的 if(cnt.righting) 永远进不去，就不会画任何东西！
                for(auto& cnt_string : armor){
                    cv::Mat ROI;
                    CropAndResize(cnt_string, matImage, ROI);
                    // 只有这里跑了识别，真正的装甲板才会被打上 righting = 1 的标签
                    cnt_string.ID = onnx_detector.detect(ROI, cnt_string);
                }

                // 将识别到的有效装甲板按车辆 ID 分组
                std::map<std::string, VehicleArmors> vehicles;
                for(auto& cnt : armor){
                    if(cnt.righting){ // righting == 1 表示模型识别到了有效数字（不是 unknown）
                        vehicles[cnt.ID].vehicle_id = cnt.ID;
                        vehicles[cnt.ID].addPlate(cnt);
                    }
                }

                // 对于每个车辆分别进行位姿解算和优化
                for(auto& pair : vehicles){
                    const auto& vehicle = pair.second;
                    armor_model::FrameObservation obs = vehicle.getObservation(0);
                    
                    armor_model::Pose T_init = armor_model::Pose::Identity();
                    bool has_init_pose = false;

                    // 借用老 PnP 提供一个基础初值 (T_init)
                    // 为了初始位姿能更好收敛，用被判断为前装甲板(0号)或排在最后(最右边)的一块作初值计算
                    lightbors init_plate = vehicle.plates[0];
                    if (vehicle.plates.size() > 1) {
                        for (const auto& p : vehicle.plates) {
                            if (p.armor_center.x > init_plate.armor_center.x) {
                                init_plate = p;
                            }
                        }
                    }

                    cv::Vec4d q_wo_tmp; cv::Mat t_wo_tmp, R_wo_tmp;
                    std::vector<cv::Point2f> imgPts = {
                        init_plate.armor_point[0], init_plate.armor_point[1],
                        init_plate.armor_point[2], init_plate.armor_point[3]
                    };
                    
                    if (estimator.estimatePose(objPts, imgPts, q_wo_tmp, t_wo_tmp, R_wo_tmp)) {
                        Eigen::Matrix3d R_eigen = armor_model::cvMatToEigen3d(R_wo_tmp);
                        Eigen::Vector3d t_eigen = armor_model::cvMatToEigenVec(t_wo_tmp);
                        
                        T_init.linear() = R_eigen;
                        // 注意：这里用装甲板位姿近似整车位姿，有 30cm 的初始偏差
                        T_init.translation() = t_eigen / 1000.0; 
                        has_init_pose = true;
                    }

                    // 如果有观测数据，则执行 LM 非线性优化
                    if (!obs.armors.empty() && has_init_pose) {
                        auto opt_result = optimizer.optimizeSingleFrame(obs, T_init, false, false, false);
                        
                        // 去掉严格的 converged 判断强制渲染！
                        matImage = armor_model::ModelVisualizer::render3DView(
                            optimizer.getModel(), opt_result.T_cam_object, am_cam, matImage, &obs);
                    }
                }
                ////////////////
                */

                // for(auto& pnppose : armor){
                //     if (pnppose.armor_point == nullptr) continue;
                //     std::vector<cv::Point2f> imgPts = {
                //     pnppose.armor_point[0], pnppose.armor_point[1],
                //     pnppose.armor_point[2], pnppose.armor_point[3]
                //     };
                //     if (estimator.estimatePose(objPts, imgPts, q_wo, t_wo, R_wo)) {
                //         drawCube(matImage, R_wo, t_wo,
                //                 cameraMatrix,
                //                 distCoeffs);
                //     }else{continue;}
                // }

                
                // // 此处为字符识别部分，采用新集成的 ONNX 识别算法
                // for(auto& cnt_string : armor){
                //     // FeatureDetector8Classes detector;  // 淘汰的 SVM 算法
                //     cv::Mat ROI;
                //     CropAndResize(cnt_string, matImage, ROI);
                //     cnt_string.ID = onnx_detector.detect(ROI, cnt_string);
                // }

                
                // 将疑似装甲板全部绘制出来       并配上识别字符
                for(auto& cnt : armor){
                    if(/*cnt.righting*/1){

                        ////////////////  测试
                        // std::vector<cv::Point2f> imgPts = {
                        // cnt.armor_point[0], cnt.armor_point[1],
                        // cnt.armor_point[2], cnt.armor_point[3]
                        // };
                        // if (estimator.estimatePose(objPts, imgPts, q_wo, t_wo, R_wo)) {
                        //     drawCube(matImage, R_wo, t_wo,
                        //             cameraMatrix,
                        //             distCoeffs);
                        // }else{continue;}
                        // ////////////////  测试

                        // cv::putText(matImage, cnt.ID, cnt.armor_point[0], cv::FONT_HERSHEY_SIMPLEX, 3.0, cv::Scalar(255,0,0), 3);
                        for (int i = 0; i < 4; i++)
                        {
                            cv::line(matImage, cnt.armor_point[i], cnt.armor_point[(i+1)%4], cv::Scalar(0,0,254), 2);
                        }
                    }
                }
            
            }

            imshow("Tracking", matImage);
            // imshow("tae", grayImage);

            cv::waitKey(5);

            CameraReleaseImageBuffer(hCamera, pbyBuffer);
        }
    }

    CameraUnInit(hCamera);

    free(g_pRgbBuffer);


    return 0;
}

