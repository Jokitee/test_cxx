#include <iostream>
#include "CameraApi.h"
#include "opencv2/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdio.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <map>
#include <string>

using namespace cv;

unsigned char           * g_pRgbBuffer;

struct lightbors
{
    cv::RotatedRect left_lightbors;
    cv::RotatedRect right_lightbors;
    cv::RotatedRect left_armor;
    cv::RotatedRect right_armor;
    cv::Point2f armor_center;
    cv::Point2f armor_point[4];     // 次序为 左上 -> 右上 -> 右下 -> 左下
    std::string ID;         // 装甲板数字
    bool righting = 0;

    // 反馈当前检测各参数
    std::string test_ID;
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
void points_exchange(Point2f points[4]) {
    std::vector<Point2f> pts(points, points + 4);
    
    // Step 1: 按 y 坐标升序排序（y 小在上，y 大在下）
    // y 相同则按 x 排序（x 小在左）
    std::sort(pts.begin(), pts.end(), [](const Point2f& a, const Point2f& b) {
        if (std::abs(a.y - b.y) > 1e-5) {
            return a.y < b.y;  // y 小的排在前面（上方）
        }
        return a.x < b.x;      // y 相同，x 小的排在前面
    });
    
    // Step 2: 分组 —— 上方两点（y 较小）和下方两点（y 较大）
    Point2f top1 = pts[0];   // y 最小
    Point2f top2 = pts[1];   // y 次小
    Point2f bottom1 = pts[2]; // y 次大
    Point2f bottom2 = pts[3]; // y 最大
    
    // Step 3: 上方两点按 x 排序，确定左上和右上
    Point2f top_left, top_right;
    if (top1.x < top2.x) {
        top_left  = top1;   // x 小 → 左上
        top_right = top2;   // x 大 → 右上
    } else {
        top_left  = top2;
        top_right = top1;
    }
    
    // Step 4: 下方两点按 x 排序，确定左下和右下
    Point2f bottom_left, bottom_right;
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
    Point2f left_points[4];
    Point2f right_points[4];
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

// 获取待检测的ROI
void CropAndResize(const lightbors& armor_light, cv::Mat& endB, cv::Mat& outImage) {
    
    int dst_w = 64;
    int dst_h = 64;

    // armor_point[0~3] 四个角点（假设顺序：左上、右上、右下、左下，顺时针或逆时针均可）
    cv::Point2f srcPts[4] = {
        armor_light.armor_point[0],
        armor_light.armor_point[1],
        armor_light.armor_point[2],
        armor_light.armor_point[3]
    };

    // ========== 计算各边中点 ==========
    // 上边中点（0和1之间）
    cv::Point2f mid_top    = (srcPts[0] + srcPts[1]) * 0.5f;
    // 下边中点（3和2之间）  
    cv::Point2f mid_bottom = (srcPts[3] + srcPts[2]) * 0.5f;
    // 左边中点（0和3之间）
    cv::Point2f mid_left   = (srcPts[0] + srcPts[3]) * 0.5f;
    // 右边中点（1和2之间）
    cv::Point2f mid_right  = (srcPts[1] + srcPts[2]) * 0.5f;

    // ========== 构建新的四个角点（截取中间一半长度）==========
    // 
    // 原理：以竖直方向（左右边中点连线）为基准
    // 上边从中心往两边各取一半（即原长的1/4处）
    // 这样新上边长度 = 原上边长度 / 2
    //
    // 新左上 = 上边中点 + (原左上 - 上边中点) * 0.5
    //       = 上边中点往左上方向走一半距离
    cv::Point2f new_top_left  = mid_top + (srcPts[0] - mid_top) * 0.5f;
    cv::Point2f new_top_right = mid_top + (srcPts[1] - mid_top) * 0.5f;
    
    // 下边同理
    cv::Point2f new_bottom_right = mid_bottom + (srcPts[2] - mid_bottom) * 0.5f;
    cv::Point2f new_bottom_left  = mid_bottom + (srcPts[3] - mid_bottom) * 0.5f;

    // 源四边形（截取后的区域）
    cv::Point2f srcQuad[4] = {
        new_top_left,      // 新左上
        new_top_right,     // 新右上
        new_bottom_right,  // 新右下
        new_bottom_left    // 新左下
    };

    // 目标矩形（正放，64x64）
    cv::Point2f dstQuad[4] = {
        cv::Point2f(0, 0),              // 左上
        cv::Point2f(dst_w - 1, 0),     // 右上
        cv::Point2f(dst_w - 1, dst_h - 1), // 右下
        cv::Point2f(0, dst_h - 1)      // 左下
    };

    // ========== 透视变换 ==========
    cv::Mat transform = cv::getPerspectiveTransform(srcQuad, dstQuad);
    cv::warpPerspective(endB, outImage, transform, cv::Size(dst_w, dst_h));
}

// 自研分类算法
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
 
    /**
     * 检测特征并返回特征名称
     * @param inputImg 输入的局部特征图像 (BGR格式)
     * @return 返回特征名称，失败返回 "unknown"
     */
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
    setStatues = CameraSetExposureTime(hCamera, 3000);
    CameraSetGain(hCamera, 100, 70, 50);
    printf("statue = %d\n", setStatues);

    if(tCapability.sIspCapacity.bMonoSensor){
        channel = 1;
        CameraSetIspOutFormat(hCamera, CAMERA_MEDIA_TYPE_MONO8);
    }else{
        channel = 3;
        CameraSetIspOutFormat(hCamera, CAMERA_MEDIA_TYPE_BGR8);
    }

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
                for(size_t i = 0; i < end_rects.size()-1; i++){
                    for(size_t j = i + 1; j < end_rects.size(); j++){
 
                        /*
                        *****    灯条匹配逻辑      ******
                        */
                        // 倾斜角度偏差检测
                        float angle_TF = getright_angle(end_rects[i]) - getright_angle(end_rects[j]);
                        if (fabs(angle_TF) > 6.5)continue;
                        
                        // 灯条距离与灯条长度比值检测
                        float first_max = std::max(end_rects[i].size.width, end_rects[i].size.height);
                        float second_max = std::max(end_rects[j].size.width, end_rects[j].size.height);
                        float getheight = sqrt(pow(end_rects[i].center.x - end_rects[j].center.x, 2)+pow(end_rects[i].center.y - end_rects[j].center.y, 2));
                        float getlight = (first_max + second_max) / 2;
                        float distance_TF = getheight / getlight;
                        if (distance_TF > 3.0 || distance_TF < 2.3)continue;

                        /*
                        *****   灯条归位并定点    *****
                        */
                        lightbors armor_light;

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

                // 此处为字符识别部分，为了降低算力，采用自研的识别算法
                for(auto& cnt_string : armor){
                    FeatureDetector8Classes detector;
                    cv::Mat ROI;
                    CropAndResize(cnt_string, matImage, ROI);
                    cnt_string.ID = detector.detect(ROI, cnt_string);
                }

                // 将疑似装甲板全部绘制出来       并配上识别字符
                for(auto& cnt : armor){

                    // if(cnt.righting){
                        //  cv::putText(matImage, cnt.ID, cnt.armor_point[0], cv::FONT_HERSHEY_SIMPLEX, 3.0, cv::Scalar(255,0,0), 3);
                        for (int i = 0; i < 4; i++)
                        {
                            cv::line(matImage, cnt.armor_point[i], cnt.armor_point[(i+1)%4], cv::Scalar(0,0,254), 2);
                        }
                        cv::putText(matImage, cnt.test_ID, cnt.armor_point[0], cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255,0,0), 3);
                    // }

                    // if(cnt.righting){
                    //     cv::putText(matImage, cnt.ID, cnt.armor_point[0], cv::FONT_HERSHEY_SIMPLEX, 3.0, cv::Scalar(255,0,0), 3);
                    // }

                    // cv::putText(matImage, cnt.test_ID, cnt.armor_point[0], cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255,0,0), 3);
                    // for (int i = 0; i < 4; i++)
                    // {
                    //     cv::line(matImage, cnt.armor_point[i], cnt.armor_point[(i+1)%4], cv::Scalar(0,0,254), 2);
                    // }
                }
            }

            imshow("Tracking", matImage);
            // imshow("tae", grayImage);

            waitKey(5);

            CameraReleaseImageBuffer(hCamera, pbyBuffer);
        }
    }

    CameraUnInit(hCamera);

    free(g_pRgbBuffer);


    return 0;
}

